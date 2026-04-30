"""
RoutineRunner — Manages registration, scheduling, and execution of routines.

The runner is the central coordinator.  It:
    1. Registers routine definitions.
    2. Arms schedule triggers in a background scheduler thread.
    3. Dispatches event triggers when ``emit_event()`` is called.
    4. Exposes ``fire()`` for manual / webhook invocation.
    5. Tracks execution history per routine.

Thread safety:
    All public methods are protected by a ``threading.Lock``.  Routine
    executions run in a ``ThreadPoolExecutor`` so the scheduler thread
    is never blocked.

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

from .routine import (
    Routine,
    RoutineConfig,
    RoutineResult,
    RoutineRunRecord,
    RoutineStatus,
    ResultStatus,
)
from .triggers import (
    EventTrigger,
    ManualTrigger,
    RoutineTrigger,
    ScheduleTrigger,
    TriggerType,
    WebhookTrigger,
)

logger = logging.getLogger("Nono.Routines")

# Default scheduler tick interval (seconds)
_DEFAULT_TICK_SECONDS: float = 30.0

# Maximum concurrent routine executions
_DEFAULT_MAX_WORKERS: int = 4


# ── Callbacks ─────────────────────────────────────────────────────────────────

OnRoutineStart = Callable[[str, dict[str, Any]], None]
"""``(routine_name, context) -> None``.  Fired before execution."""

OnRoutineComplete = Callable[[str, RoutineResult], None]
"""``(routine_name, result) -> None``.  Fired after execution."""

OnRoutineError = Callable[[str, Exception], None]
"""``(routine_name, exception) -> None``.  Fired on unhandled error."""


# ── Runner ────────────────────────────────────────────────────────────────────

class RoutineRunner:
    """Central coordinator for routine lifecycle.

    Args:
        tick_seconds: Scheduler polling interval.
        max_workers: Thread pool size for concurrent executions.

    Example:
        >>> runner = RoutineRunner()
        >>> runner.register(my_routine)
        >>> runner.start()
        >>> result = runner.fire("my_routine", context={"key": "value"})
        >>> runner.stop()
    """

    def __init__(
        self,
        tick_seconds: float = _DEFAULT_TICK_SECONDS,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        self._routines: dict[str, Routine] = {}
        self._history: dict[str, list[RoutineRunRecord]] = defaultdict(list)
        self._lock = threading.Lock()
        self._tick_seconds = tick_seconds
        self._max_workers = max_workers
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._scheduler_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Callbacks
        self._on_start: list[OnRoutineStart] = []
        self._on_complete: list[OnRoutineComplete] = []
        self._on_error: list[OnRoutineError] = []

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, routine: Routine) -> None:
        """Register a routine definition.

        Args:
            routine: The routine to register.

        Raises:
            ValueError: If a routine with the same name already exists.
        """
        with self._lock:
            if routine.name in self._routines:
                raise ValueError(
                    f"Routine {routine.name!r} is already registered. "
                    f"Use update() to modify it."
                )
            self._routines[routine.name] = routine
            routine.status = RoutineStatus.IDLE
            logger.info("Registered routine %r", routine.name)

    def unregister(self, name: str) -> Routine:
        """Remove a routine by name.

        Args:
            name: Routine name.

        Returns:
            The removed routine.

        Raises:
            KeyError: If the routine is not found.
        """
        with self._lock:
            if name not in self._routines:
                raise KeyError(f"Routine {name!r} not found")
            routine = self._routines.pop(name)
            routine.status = RoutineStatus.DISABLED
            logger.info("Unregistered routine %r", name)
            return routine

    def update(self, routine: Routine) -> None:
        """Replace an existing routine definition.

        Args:
            routine: Updated routine (matched by name).

        Raises:
            KeyError: If no routine with this name exists.
        """
        with self._lock:
            if routine.name not in self._routines:
                raise KeyError(f"Routine {routine.name!r} not found")
            routine.updated_at = datetime.now(timezone.utc)
            self._routines[routine.name] = routine
            logger.info("Updated routine %r", routine.name)

    def get(self, name: str) -> Routine:
        """Retrieve a routine by name.

        Args:
            name: Routine name.

        Returns:
            The routine definition.

        Raises:
            KeyError: If not found.
        """
        with self._lock:
            if name not in self._routines:
                raise KeyError(f"Routine {name!r} not found")
            return self._routines[name]

    def list_routines(self) -> list[Routine]:
        """Return all registered routines.

        Returns:
            List of routine definitions.
        """
        with self._lock:
            return list(self._routines.values())

    # ── Lifecycle callbacks ───────────────────────────────────────────────

    def on_start(self, callback: OnRoutineStart) -> None:
        """Register a callback fired before routine execution.

        Args:
            callback: ``(routine_name, context) -> None``.
        """
        self._on_start.append(callback)

    def on_complete(self, callback: OnRoutineComplete) -> None:
        """Register a callback fired after routine execution.

        Args:
            callback: ``(routine_name, result) -> None``.
        """
        self._on_complete.append(callback)

    def on_error(self, callback: OnRoutineError) -> None:
        """Register a callback fired on execution error.

        Args:
            callback: ``(routine_name, exception) -> None``.
        """
        self._on_error.append(callback)

    # ── Scheduler ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler background thread.

        Arms all registered schedule triggers and begins the tick loop.
        Call ``stop()`` to shut down cleanly.
        """
        with self._lock:
            if self._running:
                logger.warning("RoutineRunner is already running")
                return

            self._stop_event.clear()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="routine-worker",
            )
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="routine-scheduler",
                daemon=True,
            )
            self._running = True

            # Activate all routines with schedule triggers
            for routine in self._routines.values():
                if routine.status in (RoutineStatus.IDLE, RoutineStatus.ACTIVE):
                    routine.status = RoutineStatus.ACTIVE

            self._scheduler_thread.start()
            logger.info(
                "RoutineRunner started (tick=%.1fs, workers=%d)",
                self._tick_seconds,
                self._max_workers,
            )

    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler and shut down the executor.

        Args:
            wait: If ``True``, wait for running executions to complete.
        """
        with self._lock:
            if not self._running:
                return

            self._stop_event.set()
            self._running = False

        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=self._tick_seconds + 5)
            self._scheduler_thread = None

        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

        with self._lock:
            for routine in self._routines.values():
                if routine.status == RoutineStatus.ACTIVE:
                    routine.status = RoutineStatus.IDLE

        logger.info("RoutineRunner stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is active."""
        return self._running

    def _scheduler_loop(self) -> None:
        """Background loop that checks schedule triggers each tick."""
        logger.debug("Scheduler loop started")
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            with self._lock:
                routines_snapshot = list(self._routines.values())

            for routine in routines_snapshot:
                if routine.status != RoutineStatus.ACTIVE:
                    continue

                for trigger in routine.triggers:
                    if isinstance(trigger, ScheduleTrigger) and trigger.should_fire(now):
                        logger.info(
                            "Schedule trigger fired for %r",
                            routine.name,
                        )
                        trigger.mark_fired(now)
                        self._submit_execution(
                            routine,
                            context={},
                            trigger_type="schedule",
                        )
                        break  # one fire per tick per routine

            self._stop_event.wait(self._tick_seconds)

        logger.debug("Scheduler loop exited")

    # ── Execution ─────────────────────────────────────────────────────────

    def fire(
        self,
        name: str,
        context: dict[str, Any] | None = None,
        trigger_type: str = "manual",
        wait: bool = True,
        timeout: float | None = None,
    ) -> RoutineResult:
        """Manually fire a routine.

        Args:
            name: Routine name.
            context: Input data passed to the executable.
            trigger_type: Label for the trigger source.
            wait: If ``True``, block until completion and return the result.
            timeout: Maximum seconds to wait (only if ``wait=True``).

        Returns:
            The execution result.

        Raises:
            KeyError: If the routine is not found.
        """
        with self._lock:
            if name not in self._routines:
                raise KeyError(f"Routine {name!r} not found")
            routine = self._routines[name]

        if routine.status == RoutineStatus.PAUSED:
            return RoutineResult(
                routine_name=name,
                status=ResultStatus.CANCELLED,
                error="Routine is paused",
                trigger_type=trigger_type,
            )

        if routine.status == RoutineStatus.DISABLED:
            return RoutineResult(
                routine_name=name,
                status=ResultStatus.CANCELLED,
                error="Routine is disabled",
                trigger_type=trigger_type,
            )

        future = self._submit_execution(
            routine,
            context=context or {},
            trigger_type=trigger_type,
        )

        if wait and future is not None:
            effective_timeout = timeout or routine.config.timeout_seconds or None
            try:
                return future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError:
                return RoutineResult(
                    routine_name=name,
                    status=ResultStatus.TIMEOUT,
                    error=f"Execution timed out after {effective_timeout}s",
                    trigger_type=trigger_type,
                )

        return RoutineResult(
            routine_name=name,
            status=ResultStatus.SUCCESS,
            output="Execution submitted (async)",
            trigger_type=trigger_type,
        )

    def _submit_execution(
        self,
        routine: Routine,
        context: dict[str, Any],
        trigger_type: str,
    ) -> concurrent.futures.Future | None:
        """Submit a routine execution to the thread pool.

        Args:
            routine: The routine to execute.
            context: Input data.
            trigger_type: Trigger label.

        Returns:
            Future for the execution, or ``None`` if no executor.
        """
        if self._executor is None:
            # No scheduler running — execute synchronously
            result = self._execute_routine(routine, context, trigger_type)
            # Return a resolved future
            f: concurrent.futures.Future[RoutineResult] = concurrent.futures.Future()
            f.set_result(result)
            return f

        return self._executor.submit(
            self._execute_routine,
            routine,
            context,
            trigger_type,
        )

    def _execute_routine(
        self,
        routine: Routine,
        context: dict[str, Any],
        trigger_type: str,
    ) -> RoutineResult:
        """Execute a routine's executable and capture the result.

        Handles retries, timeout tracking, lifecycle callbacks, and
        history recording.

        Args:
            routine: The routine to execute.
            context: Input data.
            trigger_type: Trigger label.

        Returns:
            The execution result.
        """
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)

        # Merge environment into context
        effective_context = {**routine.config.environment, **context}

        # Apply input template
        user_message = routine.input_template
        if user_message:
            try:
                user_message = user_message.format(**effective_context)
            except KeyError:
                pass  # leave unresolved placeholders

        # Fire on_start callbacks
        for cb in self._on_start:
            try:
                cb(routine.name, effective_context)
            except Exception as exc:
                logger.warning("on_start callback error: %s", exc)

        with self._lock:
            routine.status = RoutineStatus.RUNNING

        result = RoutineResult(
            routine_name=routine.name,
            run_id=run_id,
            started_at=started_at,
            trigger_type=trigger_type,
            context=effective_context,
        )

        max_attempts = max(1, routine.config.max_retries + 1)

        for attempt in range(max_attempts):
            try:
                output = self._invoke_executable(
                    routine,
                    user_message=user_message or str(effective_context),
                    context=effective_context,
                )
                result.status = ResultStatus.SUCCESS
                result.output = str(output) if output else ""

                if isinstance(output, dict):
                    result.data = output
                break

            except Exception as exc:
                logger.error(
                    "Routine %r execution failed (attempt %d/%d): %s",
                    routine.name,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                result.status = ResultStatus.FAILURE
                result.error = str(exc)

                if attempt < max_attempts - 1:
                    time.sleep(routine.config.retry_delay_seconds)

        finished_at = datetime.now(timezone.utc)
        result.finished_at = finished_at
        result.duration_seconds = (finished_at - started_at).total_seconds()

        # Update routine status
        with self._lock:
            if result.status == ResultStatus.SUCCESS:
                routine.status = RoutineStatus.ACTIVE if self._running else RoutineStatus.IDLE
            else:
                routine.status = RoutineStatus.ERROR

        # Record history
        record = RoutineRunRecord(
            run_id=run_id,
            routine_name=routine.name,
            trigger_type=trigger_type,
            status=result.status.value,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=result.duration_seconds,
            output_preview=result.output[:200] if result.output else "",
            error=result.error,
        )
        self._add_history(routine.name, record)

        # Fire on_complete / on_error callbacks
        if result.status == ResultStatus.SUCCESS:
            for cb in self._on_complete:
                try:
                    cb(routine.name, result)
                except Exception as exc:
                    logger.warning("on_complete callback error: %s", exc)
        else:
            for cb in self._on_error:
                try:
                    cb(routine.name, Exception(result.error))
                except Exception as exc:
                    logger.warning("on_error callback error: %s", exc)

        logger.info(
            "Routine %r completed: status=%s duration=%.1fs",
            routine.name,
            result.status.value,
            result.duration_seconds,
        )

        return result

    def _invoke_executable(
        self,
        routine: Routine,
        user_message: str,
        context: dict[str, Any],
    ) -> Any:
        """Dispatch execution to the appropriate executable type.

        Supports:
            - **Agent** (has ``run`` + ``name`` attributes): uses ``Runner``.
            - **Workflow** (has ``run`` + ``steps`` attributes): calls ``run()``.
            - **Callable**: calls with ``context`` as keyword arguments.

        Args:
            routine: The routine definition.
            user_message: Resolved user message.
            context: Execution context.

        Returns:
            Output from the executable.

        Raises:
            ValueError: If no executable is configured.
            TypeError: If the executable type is not recognized.
        """
        exe = routine.executable

        if exe is None:
            raise ValueError(
                f"Routine {routine.name!r} has no executable configured"
            )

        # Agent — use Runner
        if hasattr(exe, "run") and hasattr(exe, "name") and hasattr(exe, "instruction"):
            return self._run_agent(exe, user_message, context)

        # Workflow — call run() with context as initial state
        if hasattr(exe, "run") and hasattr(exe, "steps"):
            return exe.run(**context)

        # Plain callable
        if callable(exe):
            return exe(**context)

        raise TypeError(
            f"Executable of type {type(exe).__name__} is not supported. "
            f"Expected an Agent, Workflow, or callable."
        )

    def _run_agent(
        self,
        agent: Any,
        user_message: str,
        context: dict[str, Any],
    ) -> str:
        """Execute an agent via Runner.

        Args:
            agent: The agent instance.
            user_message: User message.
            context: State updates.

        Returns:
            Agent text response.
        """
        # Late import to avoid circular dependency
        from ..agent import Runner, Session

        session = Session()
        runner = Runner(agent, session=session)
        response = runner.run(user_message, **context)
        return response

    # ── Event Dispatch ────────────────────────────────────────────────────

    def emit_event(
        self,
        event_name: str,
        event_data: dict[str, Any] | None = None,
    ) -> list[RoutineResult]:
        """Emit an application event and fire matching routines.

        Checks all registered routines for ``EventTrigger``s that match
        the event.

        Args:
            event_name: Event name (e.g., ``"pr.opened"``).
            event_data: Event payload.

        Returns:
            List of execution results for all matched routines.
        """
        results: list[RoutineResult] = []
        data = event_data or {}

        with self._lock:
            routines_snapshot = list(self._routines.values())

        for routine in routines_snapshot:
            if routine.status in (RoutineStatus.PAUSED, RoutineStatus.DISABLED):
                continue

            for trigger in routine.triggers:
                if isinstance(trigger, EventTrigger) and trigger.matches_event(event_name, data):
                    logger.info(
                        "Event %r matched trigger for routine %r",
                        event_name,
                        routine.name,
                    )
                    future = self._submit_execution(
                        routine,
                        context={"event_name": event_name, **data},
                        trigger_type=f"event:{event_name}",
                    )
                    if future is not None:
                        try:
                            result = future.result(timeout=routine.config.timeout_seconds or None)
                            results.append(result)
                        except concurrent.futures.TimeoutError:
                            results.append(RoutineResult(
                                routine_name=routine.name,
                                status=ResultStatus.TIMEOUT,
                                error="Event-triggered execution timed out",
                                trigger_type=f"event:{event_name}",
                            ))
                    break  # one fire per routine per event

        return results

    # ── Control ───────────────────────────────────────────────────────────

    def pause(self, name: str) -> None:
        """Pause a routine — triggers are ignored until resumed.

        Args:
            name: Routine name.

        Raises:
            KeyError: If not found.
        """
        with self._lock:
            if name not in self._routines:
                raise KeyError(f"Routine {name!r} not found")
            self._routines[name].status = RoutineStatus.PAUSED
            logger.info("Paused routine %r", name)

    def resume(self, name: str) -> None:
        """Resume a paused routine.

        Args:
            name: Routine name.

        Raises:
            KeyError: If not found.
        """
        with self._lock:
            if name not in self._routines:
                raise KeyError(f"Routine {name!r} not found")
            routine = self._routines[name]
            if routine.status == RoutineStatus.PAUSED:
                routine.status = RoutineStatus.ACTIVE if self._running else RoutineStatus.IDLE
                logger.info("Resumed routine %r", name)

    # ── History ───────────────────────────────────────────────────────────

    def _add_history(self, name: str, record: RoutineRunRecord) -> None:
        """Append a run record, trimming to max_history.

        Args:
            name: Routine name.
            record: Run record.
        """
        with self._lock:
            history = self._history[name]
            history.append(record)
            routine = self._routines.get(name)
            max_h = routine.config.max_history if routine else 100
            if len(history) > max_h:
                self._history[name] = history[-max_h:]

    def get_history(self, name: str) -> list[RoutineRunRecord]:
        """Retrieve execution history for a routine.

        Args:
            name: Routine name.

        Returns:
            List of run records (most recent last).
        """
        with self._lock:
            return list(self._history.get(name, []))

    def clear_history(self, name: str) -> None:
        """Clear execution history for a routine.

        Args:
            name: Routine name.
        """
        with self._lock:
            self._history[name] = []

    # ── Introspection ─────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a summary of the runner state.

        Returns:
            Dictionary with runner status and routine summaries.
        """
        with self._lock:
            routines_info = []
            for r in self._routines.values():
                history = self._history.get(r.name, [])
                last_run = history[-1] if history else None
                routines_info.append({
                    "name": r.name,
                    "status": r.status.value,
                    "triggers": [t.trigger_type.value for t in r.triggers],
                    "total_runs": len(history),
                    "last_run": last_run.to_dict() if last_run else None,
                })

            return {
                "running": self._running,
                "tick_seconds": self._tick_seconds,
                "max_workers": self._max_workers,
                "total_routines": len(self._routines),
                "routines": routines_info,
            }

    def __repr__(self) -> str:
        return (
            f"RoutineRunner(routines={len(self._routines)}, "
            f"running={self._running})"
        )

    def __enter__(self) -> RoutineRunner:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
