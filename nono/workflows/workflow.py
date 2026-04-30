"""
Workflow - Multi-step execution pipeline with conditional branching.

Provides a fluent API for building directed graphs of operations, where each
step receives a shared state dict and returns updates.  Steps can be plain
functions, coroutines, or AI-powered callables that use Nono's connector.

Inspired by LangFrame's Workflow API, but fully standalone — no LangGraph
dependency required.

Usage:
    from nono.workflows import Workflow

    def research(state):
        return {"notes": gather_info(state["topic"])}

    def write(state):
        return {"draft": compose(state["notes"])}

    flow = Workflow()
    flow.step("research", research)
    flow.step("write", write)
    flow.connect("research", "write")
    result = flow.run(topic="AI trends 2026")

    # Conditional branching (callable):
    flow.branch("review", lambda s: "publish" if s["score"] > 0.8 else "rewrite")

    # Conditional branching (predicate with then/otherwise):
    flow.branch_if("review", lambda s: s["score"] > 0.8, then="publish", otherwise="rewrite")

    # Dynamic manipulation:
    flow.insert_before("write", "outline", outline_fn)
    flow.insert_after("write", "fact_check", fact_check_fn)
    flow.remove_step("fact_check")
    flow.replace_step("write", new_write_fn)
    flow.swap_steps("research", "write")

    # Streaming execution (yields state after each step):
    for step_name, state_update in flow.stream(topic="AI trends"):
        print(f"[{step_name}] -> {state_update}")

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import logging
import operator
import pathlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

from ..hitl import (
    AsyncHumanInputHandler,
    HumanInputHandler,
    HumanInputResponse,
    HumanRejectError,
    format_state_for_review,
)

logger = logging.getLogger("Nono.Workflow")


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from synchronous code, handling missing event loops.

    Uses ``asyncio.run()`` which creates a fresh loop — safe in worker
    threads and Python 3.10+, unlike the deprecated
    ``asyncio.get_event_loop().run_until_complete()``.
    """
    return asyncio.run(coro)


def _ensure_executed_steps(state: dict) -> None:
    """Ensure ``__executed_steps__`` is a ``set`` in *state*.

    Handles deserialized JSON where the value may be a ``list`` or
    missing entirely.

    Args:
        state: Workflow state dict (mutated in place).
    """
    raw = state.get("__executed_steps__")

    if raw is None:
        state["__executed_steps__"] = set()
    elif isinstance(raw, set):
        pass
    elif isinstance(raw, (list, tuple, frozenset)):
        state["__executed_steps__"] = set(raw)
    else:
        state["__executed_steps__"] = set()


# ── Callback type aliases ────────────────────────────────────────────────────

BeforeStepCallback = Callable[[str, dict], Optional[dict]]
"""``(step_name, state) -> Optional[dict]``.  Return a dict to short-circuit
the step with that result (skipping execution).  Return ``None`` to proceed."""

AfterStepCallback = Callable[[str, dict, Optional[dict]], Optional[dict]]
"""``(step_name, state, result) -> Optional[dict]``.  Return a dict to
replace the step's result.  Return ``None`` to keep the original result."""

BetweenStepsCallback = Callable[[str, Optional[str], dict], Optional[bool]]
"""``(completed_step, next_step, state) -> Optional[bool]``.

Fired **between** two steps — after one completes and before the next begins.
``next_step`` is ``None`` when the workflow is about to end.  Return ``False``
to halt execution early; any other value (including ``None``) continues."""

StepExecutingCallback = Callable[[str, dict, int], None]
"""``(step_name, state, attempt) -> None``.

Fired right before each execution attempt inside the retry loop.
``attempt`` is 1-based (1 = first try, 2 = first retry, etc.).
Useful for monitoring, progress reporting, or per-attempt telemetry."""

StepExecutedCallback = Callable[[str, dict, int, Optional[str]], None]
"""``(step_name, state, attempt, error) -> None``.

Fired right after each execution attempt inside the retry loop.
``attempt`` is 1-based.  ``error`` is ``None`` on success or the
error message string on failure.  Useful for per-attempt telemetry
and measuring individual retry durations."""

OnStartCallback = Callable[[str, dict], None]
"""``(workflow_name, state) -> None``.

Fired once when the workflow begins execution, after initial
validation and before the first step runs."""

OnEndCallback = Callable[[str, dict, int], None]
"""``(workflow_name, state, steps_executed) -> None``.

Fired once when the workflow finishes execution, whether it
completed normally, was halted by ``on_between_steps``, or
encountered a cycle.  ``steps_executed`` is the number of
steps that ran."""

# ── Sentinel for "end of workflow" ───────────────────────────────────────────

END = "__end__"

# ── Per-step error handling defaults ────────────────────────────────────────

DEFAULT_STEP_RETRIES: int = 0
"""Default retry count for steps without explicit ``retry`` setting."""

_MAX_ERROR_RECOVERIES: int = 20
"""Maximum consecutive error-recovery hops before the workflow raises.

Prevents infinite loops when error-recovery routes form a cycle
(e.g. step A → on_error → B → on_error → A).
"""

_MAX_TRANSITIONS: int = 10_000
"""Maximum audit-trail entries kept per workflow run.

When exceeded, oldest entries are discarded.  Prevents unbounded memory
growth in long-running workflows with loops.
"""

# ── State transition audit trail ────────────────────────────────────────────


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a single step execution for the audit trail.

    Attributes:
        step: Step name that produced this transition.
        keys_changed: Set of state keys that were added or modified.
        branch_taken: Name of the next step chosen by a branch, or ``None``.
        duration_ms: Wall-clock execution time in milliseconds.
        retries: Number of retry attempts before success (0 = first try).
        error: Exception message if the step failed, or ``None``.
        state_snapshot: Deep copy of the full state **after** this step
            executed.  Used by time-travel APIs (:meth:`Workflow.get_state_at`,
            :meth:`Workflow.get_history`, :meth:`Workflow.replay_from`).
    """

    step: str
    keys_changed: frozenset[str] = field(default_factory=frozenset)
    branch_taken: str | None = None
    duration_ms: float = 0.0
    retries: int = 0
    error: str | None = None
    state_snapshot: dict = field(default_factory=dict, repr=False)


# ── Optional state schema ───────────────────────────────────────────────────

ReducerFn = Callable[[Any, Any], Any]
"""``(existing_value, new_value) -> merged_value``.  Used by
:class:`StateSchema` to control how a key is updated (e.g. append vs replace)."""


class StateSchema:
    """Optional declarative schema for workflow state.

    Declares expected keys with Python types and optional reducer functions.
    Validation runs at workflow boundaries (init, after each step) when a
    schema is attached.

    Args:
        fields: Mapping ``{key: type}`` for expected state keys.
        reducers: Optional mapping ``{key: reducer_fn}`` where
            ``reducer_fn(old, new) -> merged``.  Keys without a reducer
            use simple replacement (``state[key] = new_value``).

    Example:
        >>> schema = StateSchema(
        ...     fields={"topic": str, "score": float, "notes": list},
        ...     reducers={"notes": lambda old, new: (old or []) + new},
        ... )
        >>> flow = Workflow("pipeline", schema=schema)
    """

    def __init__(
        self,
        fields: dict[str, type],
        *,
        reducers: dict[str, ReducerFn] | None = None,
    ) -> None:
        self._fields = dict(fields)
        self._reducers: dict[str, ReducerFn] = dict(reducers or {})

    @property
    def fields(self) -> dict[str, type]:
        """Declared field names and their expected types."""
        return dict(self._fields)

    @property
    def reducers(self) -> dict[str, ReducerFn]:
        """Declared reducer functions keyed by field name."""
        return dict(self._reducers)

    def validate(self, state: dict, *, step_name: str = "") -> list[str]:
        """Validate *state* against the schema.

        Only checks keys that are **present** in *state* — missing keys
        are not errors (they may be produced by later steps).

        Args:
            state: Current workflow state dict.
            step_name: Optional step name for error context.

        Returns:
            List of human-readable validation error strings (empty = valid).
        """
        errors: list[str] = []
        ctx = f" (after {step_name!r})" if step_name else ""

        for key, expected_type in self._fields.items():
            if key not in state:
                continue

            value = state[key]

            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"State key {key!r}{ctx}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

        return errors

    def apply_reducers(self, state: dict, update: dict) -> dict:
        """Merge *update* into *state* using registered reducers.

        Keys with a reducer call ``reducer(state[key], update[key])``.
        Keys without a reducer use simple replacement.

        Args:
            state: Current state (mutated in place).
            update: Dict of new values to merge.

        Returns:
            The mutated *state* dict.
        """
        for key, value in update.items():
            if key in self._reducers and key in state:
                state[key] = self._reducers[key](state[key], value)
            else:
                state[key] = value

        return state

# ── Helpers for branch_if ────────────────────────────────────────────────────

_CMP_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
    "is": operator.is_,
    "is not": operator.is_not,
}


def _resolve_key(state: dict, key: str) -> Any:
    """Resolve a dotted key like ``result.score`` from a nested dict.

    Args:
        state: State dictionary.
        key: Dotted key path (e.g. ``"result.score"``).

    Returns:
        Resolved value, or ``None`` if the path doesn't exist.
    """
    obj: Any = state
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


class WorkflowError(Exception):
    """Base exception for workflow errors."""


class StepNotFoundError(WorkflowError):
    """Raised when a referenced step does not exist."""


class DuplicateStepError(WorkflowError):
    """Raised when adding a step with a name that already exists."""


class StepTimeoutError(WorkflowError):
    """Raised when a step exceeds its configured timeout."""


class JoinPredecessorError(WorkflowError):
    """Raised when a strict join node's required predecessors have not executed."""


class _LoopDef:
    """Internal definition for a loop step."""

    __slots__ = ("condition", "max_iterations", "timeout_seconds")

    def __init__(
        self,
        condition: Callable[[dict], bool],
        max_iterations: int,
        timeout_seconds: float | None = None,
    ) -> None:
        self.condition = condition
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds


class _JoinDef:
    """Internal definition for a join step."""

    __slots__ = ("wait_for", "reducer", "strict")

    def __init__(
        self,
        wait_for: list[str],
        reducer: Callable[[dict], dict] | None,
        strict: bool = False,
    ) -> None:
        self.wait_for = wait_for
        self.reducer = reducer
        self.strict = strict


class Workflow:
    """Multi-step execution pipeline with conditional branching.

    Steps are maintained in an ordered list.  You can manipulate the
    pipeline dynamically before calling :meth:`run`.

    Each step function receives the full state ``dict`` and returns a
    ``dict`` with the keys to update.  Steps can be sync functions or
    ``async`` coroutines.

    Args:
        name: Human-readable name for the workflow (used in logs).

    Example:
        >>> flow = Workflow("my_pipeline")
        >>> flow.step("greet", lambda s: {"msg": "hello"})
        >>> result = flow.run()
        {'msg': 'hello'}
    """

    def __init__(
        self,
        name: str = "workflow",
        *,
        schema: StateSchema | None = None,
        record_snapshots: bool = True,
    ) -> None:
        self._name = name
        self._step_order: list[str] = []
        self._step_fns: dict[str, Callable] = {}
        self._edges: list[tuple[str, str]] = []
        self._branches: dict[str, Callable[[dict], str]] = {}
        self._entry: str | None = None
        self._before_step: BeforeStepCallback | None = None
        self._after_step: AfterStepCallback | None = None
        self._between_steps: BetweenStepsCallback | None = None
        self._step_executing: StepExecutingCallback | None = None
        self._step_executed: StepExecutedCallback | None = None
        self._on_start: OnStartCallback | None = None
        self._on_end: OnEndCallback | None = None
        self._parallel_groups: dict[str, list[str]] = {}
        self._loop_defs: dict[str, _LoopDef] = {}
        self._join_defs: dict[str, _JoinDef] = {}
        self._checkpoint_dir: pathlib.Path | None = None
        self._checkpoint_step_index: int = 0
        # Per-step error recovery
        self._step_on_error: dict[str, str] = {}
        self._step_retry: dict[str, int] = {}
        # State schema (optional)
        self._schema: StateSchema | None = schema
        # Audit trail (populated during execution)
        self._transitions: deque[StateTransition] = deque(maxlen=_MAX_TRANSITIONS)
        # When False, state_snapshot in transitions is empty — saves
        # memory for workflows with large state dicts.  Time-travel
        # APIs (get_state_at, get_history, replay_from) will not have
        # snapshot data when disabled.
        self._record_snapshots: bool = record_snapshots

    def _make_snapshot(self, state: dict) -> dict:
        """Return a deep copy of *state* if snapshots are enabled.

        Args:
            state: Current workflow state.

        Returns:
            Deep copy of *state*, or an empty dict when
            ``record_snapshots`` is ``False``.
        """
        if not self._record_snapshots:
            return {}
        try:
            return copy.deepcopy(state)
        except Exception:
            # Non-serializable objects in state — fall back to shallow copy
            logger.warning("deepcopy failed for state snapshot; using shallow copy.")
            return dict(state)

    # ── Step execution helpers ────────────────────────────────────────────

    def _retry_step_sync(
        self,
        step_name: str,
        fn: Callable,
        state: dict,
        max_attempts: int,
    ) -> tuple[Any, int]:
        """Execute *fn* with retries, callbacks, and coroutine handling.

        Args:
            step_name: Name of the step (for logging/callbacks).
            fn: Step function ``(state) -> dict``.
            state: Current workflow state.
            max_attempts: Maximum execution attempts.

        Returns:
            Tuple of ``(result, attempt_count)``.

        Raises:
            The last exception if all attempts fail.
        """
        result: Any = None
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self._step_executing is not None:
                    self._step_executing(step_name, state, attempt)

                result = fn(state)

                if asyncio.iscoroutine(result):
                    result = _run_coroutine_sync(result)

                if self._step_executed is not None:
                    self._step_executed(step_name, state, attempt, None)

                return result, attempt
            except Exception as exc:
                if self._step_executed is not None:
                    self._step_executed(step_name, state, attempt, str(exc))

                last_exc = exc

                if attempt < max_attempts:
                    logger.warning(
                        "[%s] Step %r failed (attempt %d/%d): %s",
                        self._name, step_name, attempt, max_attempts, exc,
                    )

        raise last_exc  # type: ignore[misc]

    async def _retry_step_async(
        self,
        step_name: str,
        fn: Callable,
        state: dict,
        max_attempts: int,
    ) -> tuple[Any, int]:
        """Async version of :meth:`_retry_step_sync`.

        Awaits coroutine results instead of using ``run_until_complete``.
        """
        result: Any = None
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self._step_executing is not None:
                    self._step_executing(step_name, state, attempt)

                result = fn(state)

                if asyncio.iscoroutine(result):
                    result = await result

                if self._step_executed is not None:
                    self._step_executed(step_name, state, attempt, None)

                return result, attempt
            except Exception as exc:
                if self._step_executed is not None:
                    self._step_executed(step_name, state, attempt, str(exc))

                last_exc = exc

                if attempt < max_attempts:
                    logger.warning(
                        "[%s] Step %r failed (attempt %d/%d): %s",
                        self._name, step_name, attempt, max_attempts, exc,
                    )

        raise last_exc  # type: ignore[misc]

    def _apply_result(self, step_name: str, state: dict, result: Any) -> Any:
        """Post-process step result: after_step callback, schema, checkpoint.

        Args:
            step_name: Name of the step.
            state: Current workflow state (mutated in-place).
            result: Raw result from the step function.

        Returns:
            The (potentially replaced) result.
        """
        if self._after_step is not None:
            replacement = self._after_step(step_name, state, result)
            if replacement is not None:
                result = replacement

        if isinstance(result, dict):
            if self._schema is not None:
                self._schema.apply_reducers(state, result)
            else:
                state.update(result)

        if self._schema is not None:
            errors = self._schema.validate(state, step_name=step_name)
            for err in errors:
                logger.warning("[%s] Schema: %s", self._name, err)

        state["__executed_steps__"].add(step_name)
        self.checkpoint(state, step_name)
        return result

    def _record_step_transition(
        self,
        step_name: str,
        state: dict,
        *,
        result: Any = None,
        keys_before: set[str],
        attempt: int,
        _t0: int,
        error: str | None = None,
    ) -> None:
        """Append a :class:`StateTransition` to the audit trail.

        Args:
            step_name: Name of the step.
            state: Current workflow state.
            result: Step result (for computing changed keys).
            keys_before: State keys before execution.
            attempt: Attempt number (1-based).
            _t0: ``time.perf_counter_ns()`` timestamp from step start.
            error: Error message, if the step failed.
        """
        elapsed_ms = (time.perf_counter_ns() - _t0) / 1_000_000

        if error is not None:
            self._transitions.append(StateTransition(
                step=step_name,
                keys_changed=frozenset(),
                duration_ms=elapsed_ms,
                retries=max(0, attempt - 1),
                error=error,
                state_snapshot=self._make_snapshot(state),
            ))
        else:
            next_step = self._next_step(step_name, state)
            branch_taken = next_step if step_name in self._branches else None
            keys_after = set(state.keys())
            changed = frozenset(
                k for k in (keys_after | keys_before)
                if k not in keys_before or (isinstance(result, dict) and k in result)
            )
            self._transitions.append(StateTransition(
                step=step_name,
                keys_changed=changed,
                branch_taken=branch_taken,
                duration_ms=elapsed_ms,
                retries=max(0, attempt - 1),
                state_snapshot=self._make_snapshot(state),
            ))

    def _handle_error_recovery(
        self,
        step_name: str,
        exc: Exception,
        state: dict,
        error_recoveries: int,
    ) -> tuple[str, int] | None:
        """Check for error-recovery routing.

        Args:
            step_name: Name of the failed step.
            exc: The exception that was raised.
            state: Current workflow state (``__error__`` key may be set).
            error_recoveries: Current consecutive recovery count.

        Returns:
            ``(recovery_step, updated_recovery_count)`` if recovery is
            available, or ``None`` to let the exception propagate.

        Raises:
            WorkflowError: If recovery count exceeds ``_MAX_ERROR_RECOVERIES``.
        """
        recovery_step = self._step_on_error.get(step_name)

        if recovery_step is None:
            return None

        error_recoveries += 1
        if error_recoveries > _MAX_ERROR_RECOVERIES:
            raise WorkflowError(
                f"Exceeded {_MAX_ERROR_RECOVERIES} consecutive "
                f"error-recovery hops (last: {step_name!r} → "
                f"{recovery_step!r}).  Possible recovery cycle."
            )

        logger.warning(
            "[%s] Step %r failed, routing to recovery step %r: %s",
            self._name, step_name, recovery_step, exc,
        )
        state["__error__"] = {
            "step": step_name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return recovery_step, error_recoveries

    # ── Callback setters ──────────────────────────────────────────────────

    def on_before_step(self, callback: BeforeStepCallback) -> Workflow:
        """Register a callback invoked **before** each step executes.

        The callback receives ``(step_name, state)`` and may return a
        ``dict`` to skip the step entirely (the dict is used as the
        step result), or ``None`` to let the step run normally.

        Args:
            callback: ``(step_name, state) -> Optional[dict]``.

        Returns:
            Self for fluent chaining.
        """
        self._before_step = callback
        return self

    def on_after_step(self, callback: AfterStepCallback) -> Workflow:
        """Register a callback invoked **after** each step executes.

        The callback receives ``(step_name, state, result)`` and may
        return a ``dict`` to replace the step's result, or ``None`` to
        keep it unchanged.

        Args:
            callback: ``(step_name, state, result) -> Optional[dict]``.

        Returns:
            Self for fluent chaining.
        """
        self._after_step = callback
        return self

    def on_between_steps(self, callback: BetweenStepsCallback) -> Workflow:
        """Register a callback invoked **between** consecutive steps.

        Fired after step N completes (including audit trail and
        checkpoint) and before step N+1's ``on_before_step``.

        The callback receives ``(completed_step, next_step, state)``.
        ``next_step`` is ``None`` when the workflow is about to end.

        Return ``False`` to halt the workflow early.  Any other return
        value (including ``None``) lets execution continue.

        Args:
            callback: ``(completed_step, next_step, state) -> Optional[bool]``.

        Returns:
            Self for fluent chaining.
        """
        self._between_steps = callback
        return self

    def on_step_executing(self, callback: StepExecutingCallback) -> Workflow:
        """Register a callback invoked right before each execution attempt.

        Called inside the retry loop, just before calling ``fn(state)``.
        ``attempt`` is 1-based (1 = first try, 2 = first retry, etc.).

        Use this for per-attempt monitoring, progress reporting, or
        injecting telemetry.

        Args:
            callback: ``(step_name, state, attempt) -> None``.

        Returns:
            Self for fluent chaining.
        """
        self._step_executing = callback
        return self

    def on_step_executed(self, callback: StepExecutedCallback) -> Workflow:
        """Register a callback invoked right after each execution attempt.

        Called inside the retry loop, just after ``fn(state)`` returns or
        raises.  ``attempt`` is 1-based.  ``error`` is ``None`` on success
        or the error message string on failure.

        This is the counterpart of :meth:`on_step_executing`.

        Args:
            callback: ``(step_name, state, attempt, error) -> None``.

        Returns:
            Self for fluent chaining.
        """
        self._step_executed = callback
        return self

    def on_start(self, callback: OnStartCallback) -> Workflow:
        """Register a callback invoked when the workflow starts.

        Fired once at the beginning of ``run()``, ``run_async()``,
        ``stream()``, ``astream()``, or ``replay_from()``, after
        validation and before the first step executes.

        Args:
            callback: ``(workflow_name, state) -> None``.

        Returns:
            Self for fluent chaining.
        """
        self._on_start = callback
        return self

    def on_end(self, callback: OnEndCallback) -> Workflow:
        """Register a callback invoked when the workflow finishes.

        Fired once at the end of execution, regardless of how it ended
        (normal completion, halt via ``on_between_steps``, or cycle
        detection).  Receives the workflow name, final state, and the
        number of steps executed.

        Args:
            callback: ``(workflow_name, state, steps_executed) -> None``.

        Returns:
            Self for fluent chaining.
        """
        self._on_end = callback
        return self

    # ── Read-only helpers ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Returns the workflow name."""
        return self._name

    @property
    def steps(self) -> list[str]:
        """Returns an ordered list of step names."""
        return list(self._step_order)

    @property
    def transitions(self) -> list[StateTransition]:
        """Return the audit trail recorded during the last execution.

        Each entry is a :class:`StateTransition` with the step name,
        changed keys, branch taken, duration, retry count, and error.
        The list is cleared at the start of every ``run()`` /
        ``run_async()`` call.
        """
        return list(self._transitions)

    @property
    def schema(self) -> StateSchema | None:
        """Return the optional :class:`StateSchema` attached to this workflow."""
        return self._schema

    # ── Time-travel API ───────────────────────────────────────────────────

    def get_state_at(self, step_name: str) -> dict | None:
        """Return the state snapshot recorded after *step_name* in the last run.

        This is an **in-memory** lookup — no disk access.  The snapshots
        are stored inside the :class:`StateTransition` records produced
        during the most recent ``run()``, ``run_async()``, ``stream()``,
        or ``astream()`` call.

        If the step executed more than once (e.g. inside a loop or error
        recovery), the **last** occurrence is returned.

        Args:
            step_name: Name of the step to look up.

        Returns:
            Deep-copied state dict immediately after that step, or
            ``None`` if the step was not recorded.
        """
        for t in reversed(self._transitions):
            if t.step == step_name and t.state_snapshot:
                return copy.deepcopy(t.state_snapshot)

        return None

    def get_history(self) -> list[tuple[str, dict]]:
        """Return the complete execution history from the last run.

        Each entry is a ``(step_name, state_snapshot)`` tuple in
        execution order — analogous to collecting ``stream()`` output,
        but available retroactively after ``run()`` or ``run_async()``.

        Returns:
            List of ``(step_name, state_snapshot_copy)`` tuples.
        """
        return [
            (t.step, copy.deepcopy(t.state_snapshot))
            for t in self._transitions
            if t.state_snapshot
        ]

    def replay_from(self, step_name: str, **overrides: Any) -> dict:
        """Re-execute the workflow starting from the step **after** *step_name*.

        The state is restored from the in-memory snapshot (or from a
        persisted step-indexed checkpoint if no in-memory snapshot is
        available) and the remaining steps are executed via the shared
        :meth:`_step_loop_sync`.

        Use *overrides* to patch the restored state before replaying —
        this enables "what-if" scenarios.

        Args:
            step_name: Step to rewind to.  Execution resumes from the
                **next** step in the resolved execution order.
            **overrides: Key-value pairs to patch into the restored
                state before replaying.

        Returns:
            The final state dict after replay completes.

        Raises:
            StepNotFoundError: If *step_name* is not a registered step.
            WorkflowError: If no snapshot is available for *step_name*.
        """
        if step_name not in self._step_fns:
            raise StepNotFoundError(f"Step {step_name!r} not found.")

        # Try in-memory snapshot first
        restored = self.get_state_at(step_name)

        # Fall back to disk checkpoint
        if restored is None:
            restored = self.get_checkpoint_at(step_name)

        if restored is None:
            raise WorkflowError(
                f"No state snapshot available for step {step_name!r}. "
                f"Run the workflow first or enable checkpointing."
            )

        restored.update(overrides)

        # Determine the step to start from.
        # Use _next_step() so that branches are re-evaluated with the
        # (possibly overridden) restored state.
        start_step = self._next_step(step_name, restored)

        if start_step is None or start_step == END:
            return restored  # step_name was terminal

        logger.info(
            "[%s] Replaying from step %r (state from %r + %d overrides).",
            self._name, start_step, step_name, len(overrides),
        )

        # Reset audit trail
        self._transitions = deque(maxlen=_MAX_TRANSITIONS)

        # Validate restored state
        if self._schema is not None:
            errors = self._schema.validate(restored, step_name="__replay__")

            for err in errors:
                logger.warning("[%s] Schema: %s", self._name, err)

        state = restored

        # on_start callback
        if self._on_start is not None:
            self._on_start(self._name, state)

        step_count = 0

        for _, _ in self._step_loop_sync(state, start_step):
            step_count += 1

        # on_end callback
        if self._on_end is not None:
            self._on_end(self._name, state, step_count)

        logger.info(
            "Completed: Workflow '%s' replay (%d steps executed)",
            self._name, step_count,
        )
        return state

    def __len__(self) -> int:
        return len(self._step_order)

    def __contains__(self, name: str) -> bool:
        return name in self._step_fns

    def __repr__(self) -> str:
        return f"Workflow({self._name!r}, steps={self._step_order})"

    # ── Adding steps ──────────────────────────────────────────────────────

    def step(
        self,
        name: str,
        fn: Callable,
        *,
        on_error: str | None = None,
        retry: int = DEFAULT_STEP_RETRIES,
    ) -> Workflow:
        """Register a named step at the end of the pipeline.

        Args:
            name: Unique step name.
            fn: Callable that receives ``state: dict`` and returns a ``dict`` update.
            on_error: Optional step name to route to when this step raises
                an exception (after exhausting retries).  The error details
                are stored in ``state["__error__"]``.  When ``None`` the
                exception propagates normally.
            retry: Number of retry attempts before giving up (0 = no retries).

        Returns:
            Self for fluent chaining.

        Raises:
            DuplicateStepError: If a step with the same name exists.
        """
        if name in self._step_fns:
            raise DuplicateStepError(
                f"Step {name!r} already exists. Use replace_step() to update it."
            )
        self._step_fns[name] = fn
        self._step_order.append(name)

        if on_error is not None:
            self._step_on_error[name] = on_error

        if retry > 0:
            self._step_retry[name] = retry

        if self._entry is None:
            self._entry = name
        return self

    def human_step(
        self,
        name: str,
        *,
        handler: HumanInputHandler,
        prompt: str = "Awaiting human input...",
        on_reject: str | None = None,
        state_key: str = "human_input",
        display_keys: list[str] | None = None,
    ) -> Workflow:
        """Register a human-in-the-loop checkpoint step.

        The step pauses execution, calls *handler* to collect human input,
        and merges the response into the workflow state.  A ``before_step``
        callback fires before the handler is called and an ``after_step``
        callback fires after the handler returns (standard Workflow
        lifecycle).

        The human response is stored in ``state[state_key]`` as a dict
        with keys ``approved``, ``message``, plus any extra ``data``.

        When *display_keys* is given, the values of those state keys are
        appended to the *prompt* before calling the handler, so the human
        can see the content that needs approval.

        Args:
            name: Unique step name.
            handler: Sync callback ``(step_name, state, prompt) ->
                HumanInputResponse``.  Blocks until the human responds.
            prompt: Message shown to the human.
            on_reject: Step to branch to when the human rejects.
                ``None`` raises ``HumanRejectError``.
            state_key: Key in workflow state where the response is stored.
            display_keys: State keys whose values are appended to the
                prompt for human review.  ``None`` skips enrichment
                (the handler still receives the full state).

        Returns:
            Self for fluent chaining.

        Raises:
            DuplicateStepError: If a step with the same name exists.

        Example:
            >>> flow = Workflow("review")
            >>> flow.step("draft", write_draft)
            >>> flow.human_step("review", handler=console_handler,
            ...                 prompt="Approve the draft?",
            ...                 display_keys=["draft"])
            >>> flow.step("publish", publish_fn)
            >>> flow.connect("draft", "review")
            >>> flow.connect("review", "publish")
        """

        def _human_fn(state: dict) -> dict:
            logger.info("[%s] Requesting human input: %s", self._name, prompt)

            # Enrich prompt with review content when display_keys is set
            effective_prompt = prompt

            if display_keys:
                review_block = format_state_for_review(state, display_keys)
                effective_prompt = f"{prompt}\n\nContent to review:\n{review_block}"

            response = handler(name, dict(state), effective_prompt)

            logger.info(
                "[%s] Human responded: approved=%s message=%r",
                self._name, response.approved, response.message,
            )

            result: dict[str, Any] = {
                state_key: {
                    "approved": response.approved,
                    "message": response.message,
                    **response.data,
                },
            }

            if response.data:
                result.update(response.data)

            if not response.approved:
                result["human_rejected"] = True

                if on_reject is None:
                    raise HumanRejectError(name, response.message)

            return result

        self.step(name, _human_fn)

        # Auto-register branch to on_reject step when configured
        if on_reject is not None:

            def _reject_branch(state: dict, *, _name: str = name) -> str:
                hi = state.get(state_key, {})

                if isinstance(hi, dict) and not hi.get("approved", True):
                    return on_reject

                # Approved — follow normal edge resolution
                for src, dst in self._edges:
                    if src == _name:
                        return dst

                # Fall back to next in registration order
                try:
                    idx = self._step_order.index(_name)

                    if idx + 1 < len(self._step_order):
                        return self._step_order[idx + 1]
                except ValueError:
                    pass

                return END

            if name not in self._branches:
                self._branches[name] = _reject_branch

        return self

    def insert_at(self, index: int, name: str, fn: Callable) -> Workflow:
        """Insert a step at a specific position (0-based).

        Args:
            index: Position to insert at.
            name: Unique step name.
            fn: Step callable.

        Returns:
            Self for fluent chaining.
        """
        if name in self._step_fns:
            raise DuplicateStepError(f"Step {name!r} already exists.")
        self._step_fns[name] = fn
        index = max(0, min(index, len(self._step_order)))
        self._step_order.insert(index, name)
        if index == 0:
            self._entry = name
        return self

    def insert_before(self, ref: str, name: str, fn: Callable) -> Workflow:
        """Insert *name* immediately before *ref*.

        Args:
            ref: Existing step to insert before.
            name: New step name.
            fn: Step callable.

        Returns:
            Self for fluent chaining.
        """
        idx = self._index_of(ref)
        return self.insert_at(idx, name, fn)

    def insert_after(self, ref: str, name: str, fn: Callable) -> Workflow:
        """Insert *name* immediately after *ref*.

        Args:
            ref: Existing step to insert after.
            name: New step name.
            fn: Step callable.

        Returns:
            Self for fluent chaining.
        """
        idx = self._index_of(ref)
        return self.insert_at(idx + 1, name, fn)

    # ── Removing / replacing steps ────────────────────────────────────────

    def remove_step(self, name: str) -> Workflow:
        """Remove a step and any edges/branches referencing it.

        Args:
            name: Step to remove.

        Returns:
            Self for fluent chaining.

        Raises:
            StepNotFoundError: If the step does not exist.
        """
        self._index_of(name)
        self._step_order.remove(name)
        del self._step_fns[name]
        self._edges = [(s, d) for s, d in self._edges if s != name and d != name]
        self._branches.pop(name, None)
        if self._entry == name:
            self._entry = self._step_order[0] if self._step_order else None
        return self

    def replace_step(self, name: str, fn: Callable) -> Workflow:
        """Replace the function of an existing step, keeping its position.

        Args:
            name: Step whose function to replace.
            fn: New callable.

        Returns:
            Self for fluent chaining.
        """
        self._index_of(name)
        self._step_fns[name] = fn
        return self

    # ── Reordering ────────────────────────────────────────────────────────

    def move_to(self, name: str, index: int) -> Workflow:
        """Move an existing step to *index* (0-based).

        Args:
            name: Step to move.
            index: Target position.

        Returns:
            Self for fluent chaining.
        """
        self._index_of(name)
        self._step_order.remove(name)
        index = max(0, min(index, len(self._step_order)))
        self._step_order.insert(index, name)
        self._entry = self._step_order[0]
        return self

    def move_before(self, ref: str, name: str) -> Workflow:
        """Move *name* immediately before *ref*.

        Args:
            ref: Reference step.
            name: Step to move.

        Returns:
            Self for fluent chaining.
        """
        self._index_of(name)
        self._step_order.remove(name)
        idx = self._index_of(ref)
        self._step_order.insert(idx, name)
        self._entry = self._step_order[0]
        return self

    def move_after(self, ref: str, name: str) -> Workflow:
        """Move *name* immediately after *ref*.

        Args:
            ref: Reference step.
            name: Step to move.

        Returns:
            Self for fluent chaining.
        """
        self._index_of(name)
        self._step_order.remove(name)
        idx = self._index_of(ref)
        self._step_order.insert(idx + 1, name)
        self._entry = self._step_order[0]
        return self

    def swap_steps(self, a: str, b: str) -> Workflow:
        """Swap the positions of two steps.

        Args:
            a: First step name.
            b: Second step name.

        Returns:
            Self for fluent chaining.
        """
        idx_a = self._index_of(a)
        idx_b = self._index_of(b)
        self._step_order[idx_a], self._step_order[idx_b] = (
            self._step_order[idx_b],
            self._step_order[idx_a],
        )
        self._entry = self._step_order[0]
        return self

    # ── Connecting / branching ────────────────────────────────────────────

    def connect(self, *names: str) -> Workflow:
        """Connect steps in order: ``flow.connect("a", "b", "c")`` creates a->b->c.

        Args:
            *names: Ordered step names to connect.

        Returns:
            Self for fluent chaining.
        """
        for i in range(len(names) - 1):
            self._edges.append((names[i], names[i + 1]))
        return self

    def branch(self, from_step: str, condition: Callable[[dict], str]) -> Workflow:
        """Add conditional routing from *from_step*.

        ``condition`` receives the state dict and returns the name of the
        next step, or ``END`` to finish.

        Args:
            from_step: Step that triggers the branch.
            condition: Callable ``(state) -> next_step_name``.

        Returns:
            Self for fluent chaining.

        Example:
            >>> flow.branch("review", lambda s: "publish" if s["score"] > 0.8 else "rewrite")
        """
        self._branches[from_step] = condition
        return self

    def branch_if(
        self,
        from_step: str,
        condition: Callable[[dict], bool],
        *,
        then: str,
        otherwise: str,
    ) -> Workflow:
        """Add conditional routing using a predicate callable.

        Args:
            from_step: Step that triggers the branch.
            condition: Callable ``(state) -> bool``.
            then: Step to go to when the condition returns ``True``.
            otherwise: Step to go to when the condition returns ``False``.

        Returns:
            Self for fluent chaining.

        Example:
            >>> flow.branch_if("review", lambda s: s["score"] > 0.8,
            ...                then="publish", otherwise="rewrite")
        """
        def _route(state: dict) -> str:
            try:
                return then if condition(state) else otherwise
            except (TypeError, KeyError):
                return otherwise

        self._branches[from_step] = _route
        return self

    def score_gate(
        self,
        from_step: str,
        key: str,
        threshold: float | int,
        *,
        then: str,
        otherwise: str,
        op: str = ">=",
    ) -> Workflow:
        """Convenience branch for numeric score thresholds.

        A common pattern is to route based on a score stored in the state
        dict (e.g. ``state["quality_score"] >= 80``).  ``score_gate`` wraps
        ``branch_if`` with readable, declarative semantics so you don't
        have to write a lambda.

        Args:
            from_step: Step that triggers the branch.
            key: State key containing the numeric value.
            threshold: Value to compare against.
            then: Step to go to when the comparison is ``True``.
            otherwise: Step to go to when the comparison is ``False``.
            op: Comparison operator as string.  One of
                ``>=``, ``>``, ``<=``, ``<``, ``==``, ``!=``.

        Returns:
            Self for fluent chaining.

        Example:
            >>> flow.score_gate("evaluate", "quality_score", 80,
            ...                 then="publish", otherwise="revise")
        """
        ops = {
            ">=": operator.ge,
            ">": operator.gt,
            "<=": operator.le,
            "<": operator.lt,
            "==": operator.eq,
            "!=": operator.ne,
        }
        cmp = ops.get(op)
        if cmp is None:
            raise ValueError(
                f"Unsupported operator {op!r}. Choose from: {list(ops)}"
            )

        return self.branch_if(
            from_step,
            lambda s, _c=cmp, _k=key, _t=threshold: _c(s.get(_k, 0), _t),
            then=then,
            otherwise=otherwise,
        )

    def parallel_step(
        self,
        name: str,
        fns: dict[str, Callable],
        *,
        max_workers: int | None = None,
        timeout_seconds: float | None = None,
    ) -> Workflow:
        """Register a parallel execution step that fans-out to multiple functions.

        All *fns* run concurrently on the **same** state snapshot and their
        result dicts are merged into the state when they all finish (join).

        The step is atomic from the graph's perspective — a single node
        that internally fans-out, then joins.

        Args:
            name: Unique step name (appears as one node in the graph).
            fns: Mapping ``{sub_name: callable}`` — each callable receives
                ``state: dict`` and returns ``dict``.
            max_workers: Thread pool size (``None`` = one thread per fn).
            timeout_seconds: Maximum wall-clock seconds to wait for all
                sub-steps.  ``None`` = no limit.  Raises
                ``StepTimeoutError`` on expiry.

        Returns:
            Self for fluent chaining.

        Raises:
            DuplicateStepError: If *name* already exists.
            StepTimeoutError: If *timeout_seconds* is exceeded.

        Example:
            >>> flow.parallel_step("fetch", {
            ...     "api":  lambda s: {"api_data": call_api(s["query"])},
            ...     "db":   lambda s: {"db_data": query_db(s["query"])},
            ... }, timeout_seconds=30)
        """
        sub_names = list(fns.keys())

        def _parallel_fn(state: dict) -> dict:
            snapshot = dict(state)
            merged: dict[str, Any] = {}
            workers = max_workers or len(fns)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(fn, dict(snapshot)): sub
                    for sub, fn in fns.items()
                }

                done, not_done = concurrent.futures.wait(
                    futures, timeout=timeout_seconds,
                )

                if not_done:
                    for f in not_done:
                        f.cancel()
                    timed_out_subs = [futures[f] for f in not_done]
                    raise StepTimeoutError(
                        f"Parallel step {name!r}: sub-steps {timed_out_subs} "
                        f"exceeded {timeout_seconds}s timeout."
                    )

                for future in done:
                    sub = futures[future]
                    result = future.result()

                    if isinstance(result, dict):
                        merged.update(result)

                    logger.debug(
                        "[%s] Parallel sub-step %r completed.", self._name, sub,
                    )

            return merged

        self.step(name, _parallel_fn)
        self._parallel_groups[name] = sub_names
        return self

    def loop_step(
        self,
        name: str,
        fn: Callable,
        *,
        condition: Callable[[dict], bool],
        max_iterations: int = 10,
        timeout_seconds: float | None = None,
    ) -> Workflow:
        """Register a deterministic loop step.

        The step executes *fn* repeatedly while *condition(state)* returns
        ``True``, up to *max_iterations*.  An optional *timeout_seconds*
        limits the total wall-clock time for the entire loop.

        Args:
            name: Unique step name.
            fn: Callable that receives ``state: dict`` and returns ``dict``.
            condition: Predicate ``(state) -> bool``.  The loop continues
                while this returns ``True``.
            max_iterations: Safety cap to prevent infinite loops.
            timeout_seconds: Maximum wall-clock seconds for all iterations
                combined.  ``None`` = no limit.  Raises
                ``StepTimeoutError`` when exceeded.

        Returns:
            Self for fluent chaining.

        Raises:
            DuplicateStepError: If *name* already exists.
            StepTimeoutError: If *timeout_seconds* is exceeded.

        Example:
            >>> flow.loop_step(
            ...     "refine",
            ...     refine_fn,
            ...     condition=lambda s: s.get("quality", 0) < 0.9,
            ...     max_iterations=5,
            ...     timeout_seconds=60,
            ... )
        """
        loop_def = _LoopDef(
            condition=condition,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )

        def _loop_fn(state: dict) -> dict:
            iteration = 0
            merged: dict[str, Any] = {}
            deadline = (
                time.monotonic() + loop_def.timeout_seconds
                if loop_def.timeout_seconds is not None
                else None
            )

            while iteration < loop_def.max_iterations:
                if not loop_def.condition(state):
                    break

                if deadline is not None and time.monotonic() > deadline:
                    raise StepTimeoutError(
                        f"Loop step {name!r} exceeded {loop_def.timeout_seconds}s "
                        f"timeout after {iteration} iterations."
                    )

                logger.debug(
                    "[%s] Loop %r iteration %d", self._name, name, iteration + 1,
                )
                result = fn(state)

                if asyncio.iscoroutine(result):
                    result = _run_coroutine_sync(result)

                if isinstance(result, dict):
                    state.update(result)
                    merged.update(result)

                iteration += 1

            merged["__loop_iterations__"] = iteration
            logger.info(
                "[%s] Loop %r completed after %d iterations.",
                self._name, name, iteration,
            )
            return merged

        self.step(name, _loop_fn)
        self._loop_defs[name] = loop_def
        return self

    def join(
        self,
        name: str,
        wait_for: list[str],
        *,
        reducer: Callable[[dict], dict] | None = None,
        strict: bool = False,
    ) -> Workflow:
        """Register an explicit join (wait-for-all) node.

        In a graph with multiple branches, *join* blocks until all steps
        listed in *wait_for* have executed.  An optional *reducer* can
        post-process the merged state.

        In the current synchronous engine the join validates that all
        required predecessors have run.  In streaming / async modes it
        serves as a synchronisation barrier marker.

        Args:
            name: Unique step name for the join node.
            wait_for: List of step names that must complete before
                this node executes.
            reducer: Optional ``(state) -> dict`` to transform the
                merged state after the barrier.
            strict: When ``True``, raise :class:`JoinPredecessorError`
                if any predecessor has not executed.  When ``False``
                (default), only log a warning.

        Returns:
            Self for fluent chaining.

        Raises:
            JoinPredecessorError: If *strict* is ``True`` and
                predecessors are missing.

        Example:
            >>> flow.join("merge", wait_for=["branch_a", "branch_b"],
            ...           reducer=lambda s: {"summary": s["a"] + s["b"]})
        """
        join_def = _JoinDef(wait_for=list(wait_for), reducer=reducer, strict=strict)

        def _join_fn(state: dict) -> dict:
            executed = state.get("__executed_steps__", set())
            missing = [s for s in join_def.wait_for if s not in executed]

            if missing:
                if join_def.strict:
                    raise JoinPredecessorError(
                        f"Join {name!r}: required predecessors not yet "
                        f"executed: {missing}"
                    )

                logger.warning(
                    "[%s] Join %r: predecessors not yet executed: %s",
                    self._name, name, missing,
                )

            if join_def.reducer is not None:
                return join_def.reducer(state)

            return {}

        self.step(name, _join_fn)
        self._join_defs[name] = join_def
        return self

    # ── Checkpointing ─────────────────────────────────────────────────────

    def enable_checkpoints(self, directory: str | pathlib.Path) -> Workflow:
        """Enable automatic state checkpointing to disk.

        After each step, the workflow state is persisted as a JSON file
        inside *directory*.  Use :meth:`resume` to restart from the last
        checkpoint.

        Args:
            directory: Folder where checkpoint files are written.

        Returns:
            Self for fluent chaining.
        """
        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = path
        return self

    def checkpoint(self, state: dict, step_name: str) -> None:
        """Persist *state* to disk after *step_name* completes.

        Two files are written:

        1. **Latest checkpoint** — ``{name}_checkpoint.json`` (overwritten
           each step, used by :meth:`resume` for crash recovery).
        2. **Step-indexed checkpoint** — ``{name}_step_{index:03d}_{step_name}.json``
           (append-only, used by :meth:`replay_from` for time-travel).

        Both files are written atomically (write-then-rename) so a crash
        mid-write never leaves a corrupt checkpoint.

        Args:
            state: Current workflow state dict.
            step_name: Name of the step that just finished.
        """
        if self._checkpoint_dir is None:
            return

        step_index = self._checkpoint_step_index
        self._checkpoint_step_index += 1

        # Convert sets to lists for JSON serialization
        serializable_state = dict(state)
        raw_exec = serializable_state.get("__executed_steps__")

        if isinstance(raw_exec, (set, frozenset)):
            serializable_state["__executed_steps__"] = sorted(raw_exec)

        data = {
            "workflow": self._name,
            "step": step_name,
            "step_index": step_index,
            "state": serializable_state,
        }

        # Latest checkpoint (overwrites)
        target = self._checkpoint_dir / f"{self._name}_checkpoint.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)

        # Step-indexed checkpoint (append-only for time-travel)
        step_file = self._checkpoint_dir / f"{self._name}_step_{step_index:03d}_{step_name}.json"
        step_tmp = step_file.with_suffix(".tmp")
        step_tmp.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
        step_tmp.replace(step_file)

        logger.debug("[%s] Checkpoint saved after step %r (index %d).", self._name, step_name, step_index)

    def resume(self) -> tuple[dict, str | None]:
        """Load the last checkpoint and return (state, last_step).

        Returns:
            Tuple of ``(state_dict, last_completed_step)`` or
            ``({}, None)`` if no checkpoint exists.
        """
        if self._checkpoint_dir is None:
            return {}, None

        target = self._checkpoint_dir / f"{self._name}_checkpoint.json"

        if not target.exists():
            return {}, None

        data = json.loads(target.read_text(encoding="utf-8"))
        # Sync in-memory counter with existing step files on disk so
        # subsequent checkpoint() calls don't overwrite old step files.
        existing = sum(
            1 for f in self._checkpoint_dir.iterdir()
            if f.name.startswith(f"{self._name}_step_") and f.suffix == ".json"
        )
        self._checkpoint_step_index = existing
        state = data.get("state", {})
        _ensure_executed_steps(state)
        return state, data.get("step")

    def get_checkpoint_at(self, step_name: str) -> dict | None:
        """Load a step-indexed checkpoint by step name.

        Scans the checkpoint directory for a file matching the step name.
        If multiple executions of the same step exist, the **last** one
        is returned.

        Args:
            step_name: Name of the step whose checkpoint to load.

        Returns:
            State dict at that step, or ``None`` if not found.
        """
        if self._checkpoint_dir is None:
            return None

        candidates = sorted(
            f for f in self._checkpoint_dir.iterdir()
            if f.name.startswith(f"{self._name}_step_")
            and f.name.endswith(f"_{step_name}.json")
        )

        if not candidates:
            return None

        data = json.loads(candidates[-1].read_text(encoding="utf-8"))
        state = data.get("state")

        if state is not None:
            _ensure_executed_steps(state)

        return state

    def list_checkpoints(self) -> list[tuple[int, str]]:
        """List all step-indexed checkpoints on disk.

        Returns:
            Sorted list of ``(step_index, step_name)`` tuples.
        """
        if self._checkpoint_dir is None:
            return []

        results: list[tuple[int, str]] = []

        for f in sorted(self._checkpoint_dir.iterdir()):
            if not (f.name.startswith(f"{self._name}_step_") and f.suffix == ".json"):
                continue

            # Parse: {name}_step_{index:03d}_{step_name}.json
            remainder = f.stem[len(f"{self._name}_step_"):]
            parts = remainder.split("_", 1)

            if len(parts) == 2:
                try:
                    idx = int(parts[0])
                    results.append((idx, parts[1]))
                except ValueError:
                    pass

        return results

    def cleanup_checkpoints(self, keep_last_n: int = 10) -> int:
        """Remove old step-indexed checkpoint files, keeping the newest.

        The latest ``{name}_checkpoint.json`` file is never removed.

        Args:
            keep_last_n: Number of most recent step files to retain.

        Returns:
            Number of files deleted.
        """
        if self._checkpoint_dir is None:
            return 0

        step_files = sorted(
            f for f in self._checkpoint_dir.iterdir()
            if f.name.startswith(f"{self._name}_step_") and f.suffix == ".json"
        )

        to_delete = step_files[:-keep_last_n] if keep_last_n > 0 else step_files
        for f in to_delete:
            f.unlink(missing_ok=True)

        return len(to_delete)

    # ── Core execution loops ─────────────────────────────────────────────

    def _step_loop_sync(
        self,
        state: dict,
        start_step: str | None,
        trace_collector: Any | None = None,
    ) -> Iterator[tuple[str, str | None]]:
        """Core synchronous execution loop.

        Walks the step graph from *start_step*, executing each step,
        recording audit-trail transitions, handling retries, error
        recovery, and ``on_between_steps`` callbacks.

        State is mutated **in place**.

        Args:
            state: Mutable workflow state dict.
            start_step: First step to execute (or ``None`` / ``END``
                to skip).
            trace_collector: Optional trace collector for observability.

        Yields:
            ``(step_name, next_step)`` after each successful step
            execution.
        """
        error_recoveries = 0
        visited: set[str] = set()
        current = start_step

        _ensure_executed_steps(state)

        while current and current != END:
            if current in visited and current not in self._branches:
                logger.warning("Cycle detected at step %r, stopping.", current)
                break
            visited.add(current)

            fn = self._step_fns.get(current)

            if fn is None:
                raise StepNotFoundError(
                    f"Step {current!r} not found during execution."
                )

            logger.info("[%s] Executing step: %s", self._name, current)

            skipped_result: dict | None = None

            if self._before_step is not None:
                skipped_result = self._before_step(current, state)

            if trace_collector is not None:
                trace_collector.start_trace(
                    agent_name=current,
                    agent_type="WorkflowStep",
                    input_message=str(list(state.keys()))[:200],
                )

            _t0 = time.perf_counter_ns()
            max_attempts = self._step_retry.get(current, DEFAULT_STEP_RETRIES) + 1
            attempt = 0
            keys_before = set(state.keys())

            try:
                if skipped_result is not None:
                    result: Any = skipped_result
                else:
                    try:
                        result, attempt = self._retry_step_sync(
                            current, fn, state, max_attempts,
                        )
                    except Exception:
                        attempt = max_attempts
                        raise

                result = self._apply_result(current, state, result)

                if trace_collector is not None:
                    trace_collector.end_trace(
                        output=str(
                            list(result.keys())
                            if isinstance(result, dict) else result
                        )[:200],
                    )

                next_step = self._next_step(current, state)
                self._record_step_transition(
                    current, state, result=result,
                    keys_before=keys_before,
                    attempt=attempt, _t0=_t0,
                )

            except Exception as exc:
                self._record_step_transition(
                    current, state, keys_before=keys_before,
                    attempt=attempt, _t0=_t0, error=str(exc),
                )

                if trace_collector is not None:
                    trace_collector.end_trace(error=str(exc))

                recovery = self._handle_error_recovery(
                    current, exc, state, error_recoveries,
                )

                if recovery is not None:
                    current, error_recoveries = recovery
                    continue

                raise

            error_recoveries = 0
            yield current, next_step

            if self._between_steps is not None:
                if self._between_steps(current, next_step, state) is False:
                    logger.info(
                        "[%s] on_between_steps halted execution after %r.",
                        self._name, current,
                    )
                    break

            current = next_step

    async def _step_loop_async(
        self,
        state: dict,
        start_step: str | None,
        trace_collector: Any | None = None,
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Core asynchronous execution loop.

        Async counterpart of :meth:`_step_loop_sync`.  Awaits coroutine
        step functions instead of using ``asyncio.run()``.

        Args:
            state: Mutable workflow state dict.
            start_step: First step to execute (or ``None`` / ``END``
                to skip).
            trace_collector: Optional trace collector for observability.

        Yields:
            ``(step_name, next_step)`` after each successful step
            execution.
        """
        error_recoveries = 0
        visited: set[str] = set()
        current = start_step

        _ensure_executed_steps(state)

        while current and current != END:
            if current in visited and current not in self._branches:
                logger.warning("Cycle detected at step %r, stopping.", current)
                break
            visited.add(current)

            fn = self._step_fns.get(current)

            if fn is None:
                raise StepNotFoundError(
                    f"Step {current!r} not found during execution."
                )

            logger.info("[%s] Executing step: %s", self._name, current)

            skipped_result: dict | None = None

            if self._before_step is not None:
                skipped_result = self._before_step(current, state)

            if trace_collector is not None:
                trace_collector.start_trace(
                    agent_name=current,
                    agent_type="WorkflowStep",
                    input_message=str(list(state.keys()))[:200],
                )

            _t0 = time.perf_counter_ns()
            max_attempts = self._step_retry.get(current, DEFAULT_STEP_RETRIES) + 1
            attempt = 0
            keys_before = set(state.keys())

            try:
                if skipped_result is not None:
                    result: Any = skipped_result
                else:
                    try:
                        result, attempt = await self._retry_step_async(
                            current, fn, state, max_attempts,
                        )
                    except Exception:
                        attempt = max_attempts
                        raise

                result = self._apply_result(current, state, result)

                if trace_collector is not None:
                    trace_collector.end_trace(
                        output=str(
                            list(result.keys())
                            if isinstance(result, dict) else result
                        )[:200],
                    )

                next_step = self._next_step(current, state)
                self._record_step_transition(
                    current, state, result=result,
                    keys_before=keys_before,
                    attempt=attempt, _t0=_t0,
                )

            except Exception as exc:
                self._record_step_transition(
                    current, state, keys_before=keys_before,
                    attempt=attempt, _t0=_t0, error=str(exc),
                )

                if trace_collector is not None:
                    trace_collector.end_trace(error=str(exc))

                recovery = self._handle_error_recovery(
                    current, exc, state, error_recoveries,
                )

                if recovery is not None:
                    current, error_recoveries = recovery
                    continue

                raise

            error_recoveries = 0
            yield current, next_step

            if self._between_steps is not None:
                if self._between_steps(current, next_step, state) is False:
                    logger.info(
                        "[%s] on_between_steps halted execution after %r.",
                        self._name, current,
                    )
                    break

            current = next_step

    # ── Declarative loading ──────────────────────────────────────────────

    def run(self, *, trace_collector: Any | None = None, resume: bool = False, **initial_state: Any) -> dict:
        """Execute the workflow and return the final state.

        If no explicit edges are defined, steps execute in registration order.
        When *resume* is ``True`` and a checkpoint exists, execution restarts
        from the step after the last completed checkpoint.

        Features integrated into the execution loop:

        - **Per-step retry**: steps registered with ``retry=N`` are retried
          up to *N* times before the error propagates or recovery kicks in.
        - **Error recovery routing**: steps registered with ``on_error``
          route to a designated fallback step instead of raising.  The
          exception info is available in ``state["__error__"]``.
        - **State transition audit trail**: every step records a
          :class:`StateTransition` accessible via :attr:`transitions`.
        - **Schema validation**: when a :class:`StateSchema` is attached,
          state is validated after each step and warnings are logged.

        Args:
            trace_collector: Optional ``TraceCollector`` for structured observability.
            resume: If ``True``, resume from the last checkpoint.
            **initial_state: Key-value pairs for the initial state.

        Returns:
            The final state dict after all steps have executed.

        Raises:
            WorkflowError: If the workflow has no steps.
        """
        if not self._step_order:
            raise WorkflowError("Cannot run an empty workflow. Add steps first.")

        logger.info("Starting: Workflow '%s' (%d steps)", self._name, len(self._step_order))

        # Reset audit trail
        self._transitions = deque(maxlen=_MAX_TRANSITIONS)

        # Validate initial state against schema
        if self._schema is not None:
            errors = self._schema.validate(initial_state, step_name="__init__")

            for err in errors:
                logger.warning("[%s] Schema: %s", self._name, err)

        # Start workflow-level trace
        if trace_collector is not None:
            trace_collector.start_trace(
                agent_name=self._name,
                agent_type="Workflow",
                input_message=str(initial_state)[:200],
            )

        state = dict(initial_state)
        execution_order = self._resolve_execution_order()
        current = execution_order[0] if execution_order else self._entry

        # on_start callback
        if self._on_start is not None:
            self._on_start(self._name, state)

        # Resume from checkpoint if requested
        if resume:
            ckpt_state, last_step = self.resume()

            if last_step is not None:
                state.update(ckpt_state)

                try:
                    idx = execution_order.index(last_step)

                    if idx + 1 < len(execution_order):
                        current = execution_order[idx + 1]
                    else:
                        current = None
                except ValueError:
                    pass

                logger.info("[%s] Resumed after step %r.", self._name, last_step)

        step_count = 0

        for _, _ in self._step_loop_sync(state, current, trace_collector):
            step_count += 1

        if trace_collector is not None:
            trace_collector.end_trace(output=f"Completed: {step_count} steps")

        # on_end callback
        if self._on_end is not None:
            self._on_end(self._name, state, step_count)

        logger.info("Completed: Workflow '%s' (%d steps executed)", self._name, step_count)
        return state

    async def run_async(self, *, trace_collector: Any | None = None, **initial_state: Any) -> dict:
        """Async version of :meth:`run`.

        Includes the same per-step retry, error recovery, audit trail,
        and schema validation as the synchronous :meth:`run`.

        Args:
            trace_collector: Optional ``TraceCollector`` for structured observability.
            **initial_state: Key-value pairs for the initial state.

        Returns:
            The final state dict after all steps have executed.
        """
        if not self._step_order:
            raise WorkflowError("Cannot run an empty workflow. Add steps first.")

        logger.info("Starting: Workflow '%s' async (%d steps)", self._name, len(self._step_order))

        # Reset audit trail
        self._transitions = deque(maxlen=_MAX_TRANSITIONS)

        if self._schema is not None:
            errors = self._schema.validate(initial_state, step_name="__init__")

            for err in errors:
                logger.warning("[%s] Schema: %s", self._name, err)

        if trace_collector is not None:
            trace_collector.start_trace(
                agent_name=self._name,
                agent_type="Workflow",
                input_message=str(initial_state)[:200],
            )

        state = dict(initial_state)
        execution_order = self._resolve_execution_order()
        current = execution_order[0] if execution_order else self._entry

        # on_start callback
        if self._on_start is not None:
            self._on_start(self._name, state)

        step_count = 0

        async for _, _ in self._step_loop_async(state, current, trace_collector):
            step_count += 1

        if trace_collector is not None:
            trace_collector.end_trace(output=f"Completed: {step_count} steps")

        # on_end callback
        if self._on_end is not None:
            self._on_end(self._name, state, step_count)

        logger.info("Completed: Workflow '%s' async (%d steps executed)", self._name, step_count)
        return state

    def stream(self, *, trace_collector: Any | None = None, **initial_state: Any) -> Iterator[tuple[str, dict]]:
        """Execute and yield ``(step_name, state_snapshot)`` after each step.

        Includes per-step retry, error recovery, schema validation and
        audit trail — same as :meth:`run`.

        Args:
            trace_collector: Optional ``TraceCollector`` for structured observability.
            **initial_state: Key-value pairs for the initial state.

        Yields:
            Tuple of ``(step_name, state_copy)`` after each step executes.
        """
        if not self._step_order:
            raise WorkflowError("Cannot stream an empty workflow. Add steps first.")

        self._transitions = deque(maxlen=_MAX_TRANSITIONS)

        if trace_collector is not None:
            trace_collector.start_trace(
                agent_name=self._name,
                agent_type="Workflow",
                input_message=str(initial_state)[:200],
            )

        state = dict(initial_state)
        execution_order = self._resolve_execution_order()
        current = execution_order[0] if execution_order else self._entry

        # on_start callback
        if self._on_start is not None:
            self._on_start(self._name, state)

        step_count = 0

        for step_name, _ in self._step_loop_sync(state, current, trace_collector):
            step_count += 1
            yield step_name, copy.deepcopy(state)

        if trace_collector is not None:
            trace_collector.end_trace(output=f"Completed: {step_count} steps")

        # on_end callback
        if self._on_end is not None:
            self._on_end(self._name, state, step_count)

    async def astream(self, *, trace_collector: Any | None = None, **initial_state: Any) -> AsyncIterator[tuple[str, dict]]:
        """Async version of :meth:`stream`.

        Includes per-step retry, error recovery, schema validation and
        audit trail — same as :meth:`run_async`.

        Args:
            trace_collector: Optional ``TraceCollector`` for structured observability.
            **initial_state: Key-value pairs for the initial state.

        Yields:
            Tuple of ``(step_name, state_copy)`` after each step executes.
        """
        if not self._step_order:
            raise WorkflowError("Cannot stream an empty workflow. Add steps first.")

        self._transitions = deque(maxlen=_MAX_TRANSITIONS)

        if trace_collector is not None:
            trace_collector.start_trace(
                agent_name=self._name,
                agent_type="Workflow",
                input_message=str(initial_state)[:200],
            )

        state = dict(initial_state)
        execution_order = self._resolve_execution_order()
        current = execution_order[0] if execution_order else self._entry

        # on_start callback
        if self._on_start is not None:
            self._on_start(self._name, state)

        step_count = 0

        async for step_name, _ in self._step_loop_async(state, current, trace_collector):
            step_count += 1
            yield step_name, copy.deepcopy(state)

        if trace_collector is not None:
            trace_collector.end_trace(output=f"Completed: {step_count} steps")

        # on_end callback
        if self._on_end is not None:
            self._on_end(self._name, state, step_count)

    # ── Visualization ─────────────────────────────────────────────────────

    def describe(self) -> str:
        """Return a human-readable summary of the workflow graph.

        Returns:
            Multi-line string describing steps, edges, and branches.
        """
        lines = [f"Workflow: {self._name}", f"Steps ({len(self._step_order)}):"]

        for i, name in enumerate(self._step_order):
            marker = " (entry)" if name == self._entry else ""
            lines.append(f"  {i}. {name}{marker}")

        if self._edges:
            lines.append("Edges:")
            for src, dst in self._edges:
                lines.append(f"  {src} -> {dst}")

        if self._branches:
            lines.append("Branches:")
            for src in self._branches:
                lines.append(f"  {src} -> (conditional)")

        return "\n".join(lines)

    def draw(self, *, title: bool = True) -> str:
        """Render the workflow as an ASCII tree diagram.

        Args:
            title: Whether to include the workflow root label.

        Returns:
            Multi-line ASCII string showing the flow.

        Example:
            >>> flow = Workflow("demo")
            >>> flow.step("a", lambda s: s)
            >>> flow.step("b", lambda s: s)
            >>> flow.connect("a", "b")
            >>> print(flow.draw())
        """
        from ..visualize import draw_workflow
        return draw_workflow(self, title=title)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _index_of(self, name: str) -> int:
        """Return the index of *name* or raise ``StepNotFoundError``.

        Args:
            name: Step name to find.

        Returns:
            Index in the step order list.

        Raises:
            StepNotFoundError: If the step is not registered.
        """
        try:
            return self._step_order.index(name)
        except ValueError:
            raise StepNotFoundError(
                f"Step {name!r} not found. Current steps: {self._step_order}"
            )

    def _resolve_execution_order(self) -> list[str]:
        """Determine execution order from edges, or fall back to registration order.

        If explicit edges form a chain starting from the entry node, that
        chain is used.  Otherwise, registration order is returned.

        Returns:
            Ordered list of step names.
        """
        if not self._edges:
            return list(self._step_order)

        # Build adjacency from edges (non-branching only)
        adj: dict[str, str] = {}
        for src, dst in self._edges:
            if src not in self._branches:
                adj[src] = dst

        # Walk from entry following the adjacency
        order: list[str] = []
        current = self._entry
        seen: set[str] = set()
        while current and current not in seen and current != END:
            seen.add(current)
            order.append(current)
            current = adj.get(current)

        # Include any steps not reachable from the chain but present in step_order
        for name in self._step_order:
            if name not in seen:
                order.append(name)

        return order

    def _next_step(self, current: str, state: dict) -> str | None:
        """Determine the next step after *current*.

        Checks branches first, then explicit edges.  Falls back to
        registration order only when no edges or branches are defined
        at all (pure linear pipeline).

        Args:
            current: Current step name.
            state: Current workflow state.

        Returns:
            Next step name, ``END``, or ``None`` if no next step.
        """
        # 1. Check conditional branches
        if current in self._branches:
            next_name = self._branches[current](state)
            logger.debug("[%s] Branch from %s -> %s", self._name, current, next_name)
            if next_name == END:
                return END
            if next_name in self._step_fns:
                return next_name
            logger.warning(
                "Branch from %r returned unknown step %r, ending workflow.",
                current,
                next_name,
            )
            return None

        # 2. Check explicit edges
        for src, dst in self._edges:
            if src == current:
                return dst

        # 3. Fall back to next step in order ONLY if no edges or branches
        #    are defined (pure linear mode).  When the graph has routing
        #    definitions, steps without outgoing connections are terminal.
        if not self._edges and not self._branches:
            try:
                idx = self._step_order.index(current)
                if idx + 1 < len(self._step_order):
                    return self._step_order[idx + 1]
            except ValueError:
                pass

        return None


# ── Declarative workflow builders ────────────────────────────────────────


def _build_step_fn(step_def: dict[str, Any]) -> Callable[[dict], dict]:
    """Build a step callable from a declarative definition.

    Args:
        step_def: Step definition dict with ``type`` and optional parameters.

    Returns:
        A callable suitable for ``Workflow.step()``.
    """
    step_type = step_def.get("type", "passthrough")

    if step_type == "tasker":
        return tasker_node(
            provider=step_def.get("provider", "google"),
            model=step_def.get("model", "gemini-3-flash-preview"),
            system_prompt=step_def.get("system_prompt"),
            input_key=step_def.get("input_key", "input"),
            output_key=step_def.get("output_key", "output"),
            temperature=step_def.get("temperature", 0.7),
            max_tokens=step_def.get("max_tokens", 2048),
            task_file=step_def.get("task_file"),
        )

    if step_type == "passthrough":
        return lambda state: {}

    raise WorkflowError(f"Unknown step type {step_type!r} in declarative definition.")


def load_workflow(
    path: str | pathlib.Path,
    *,
    step_registry: dict[str, Callable] | None = None,
) -> Workflow:
    """Load a workflow from a YAML or JSON file.

    Supports ``.json``, ``.yaml``, and ``.yml`` extensions.

    The declarative format is::

        name: my_pipeline
        steps:
          - name: fetch
            type: passthrough          # or "tasker"
          - name: process
            type: tasker
            provider: google
            system_prompt: "Summarise."
            input_key: data
            output_key: summary
        edges:
          - [fetch, process]
        branches:
          - from: process
            condition: score > 0.8
            then: publish
            otherwise: revise

    Steps with ``type: passthrough`` (the default) require a matching
    entry in *step_registry* or they become no-ops.

    Args:
        path: Path to the YAML/JSON file.
        step_registry: Optional mapping ``{step_name: callable}`` for
            steps whose logic cannot be expressed declaratively.

    Returns:
        A fully-wired ``Workflow`` instance ready to ``run()``.

    Raises:
        WorkflowError: On parse or validation errors.
        FileNotFoundError: If the file does not exist.
    """
    filepath = pathlib.Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"Workflow file not found: {filepath}")

    raw = filepath.read_text(encoding="utf-8")

    if filepath.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise WorkflowError(
                "PyYAML is required for YAML workflow files. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(raw)
    elif filepath.suffix == ".json":
        data = json.loads(raw)
    else:
        raise WorkflowError(
            f"Unsupported file extension {filepath.suffix!r}. "
            "Use .json, .yaml, or .yml."
        )

    if not isinstance(data, dict):
        raise WorkflowError("Workflow file must contain a mapping at the top level.")

    reg = step_registry or {}
    wf_name = data.get("name", filepath.stem)
    flow = Workflow(wf_name)

    # Register steps
    for step_def in data.get("steps", []):
        sname = step_def.get("name")

        if not sname:
            raise WorkflowError("Each step must have a 'name' field.")

        if sname in reg:
            flow.step(sname, reg[sname])
        else:
            flow.step(sname, _build_step_fn(step_def))

    # Register edges
    for edge in data.get("edges", []):
        if isinstance(edge, list) and len(edge) >= 2:
            flow.connect(*edge)

    # Register branches
    for br in data.get("branches", []):
        from_step = br.get("from")
        then_step = br.get("then")
        otherwise_step = br.get("otherwise")
        cond_expr = br.get("condition", "")

        if from_step and then_step and otherwise_step and cond_expr:
            # Parse simple conditions like "key > value" or "key == value"
            predicate = _parse_condition(cond_expr)
            flow.branch_if(from_step, predicate, then=then_step, otherwise=otherwise_step)

    # Checkpoint directory
    checkpoint_dir = data.get("checkpoint_dir")

    if checkpoint_dir:
        flow.enable_checkpoints(checkpoint_dir)

    return flow


def _parse_condition(expr: str) -> Callable[[dict], bool]:
    """Parse a simple condition expression into a predicate.

    Supports expressions like ``"score > 0.8"`` or ``"status == done"``.

    Args:
        expr: Condition string.

    Returns:
        Predicate ``(state) -> bool``.
    """
    for op_str, op_fn in sorted(_CMP_OPS.items(), key=lambda x: -len(x[0])):
        if f" {op_str} " in expr:
            key, _, value_str = expr.partition(f" {op_str} ")
            key = key.strip()
            value_str = value_str.strip()

            # Try to coerce the value
            try:
                value: Any = json.loads(value_str)
            except (json.JSONDecodeError, ValueError):
                value = value_str

            def _pred(state: dict, *, _k: str = key, _op: Callable = op_fn, _v: Any = value) -> bool:
                resolved = _resolve_key(state, _k)

                try:
                    return _op(resolved, _v)
                except (TypeError, ValueError):
                    return False

            return _pred

    raise WorkflowError(f"Cannot parse condition: {expr!r}")


# ── Node factories — produce step callables for Workflow.step() ──────────


def tasker_node(
    *,
    provider: str = "google",
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    temperature: Union[float, str] = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
    output_schema: Optional[dict[str, Any]] = None,
    input_key: str = "input",
    output_key: str = "output",
    task_file: Optional[str] = None,
) -> Callable[[dict], dict]:
    """Create a step function that delegates to ``TaskExecutor``.

    Returns a standard ``Callable[[dict], dict]`` that reads from
    ``state[input_key]``, runs a ``TaskExecutor``, and writes to
    ``state[output_key]``.  Use it with ``Workflow.step()`` — no special
    method required.

    Two modes:

    - **Inline**: set ``provider``, ``model``, ``system_prompt``, etc.
      Calls ``TaskExecutor.execute()``.
    - **JSON task file**: set ``task_file``.  Calls
      ``TaskExecutor.run_json_task()``.

    Args:
        provider: AI provider.
        model: Model name.
        api_key: Optional API key (auto-resolved if ``None``).
        temperature: Sampling temperature (float or preset name).
        max_tokens: Maximum response tokens.
        system_prompt: Optional system instruction.
        output_schema: Optional JSON schema for structured output.
        input_key: State key to read the input from.
        output_key: State key to write the result to.
        task_file: Path to a JSON task definition file.

    Returns:
        A callable suitable for ``Workflow.step(name, fn)``.

    Example:
        >>> from nono.workflows import Workflow, tasker_node
        >>> flow = Workflow("pipeline")
        >>> flow.step("classify", tasker_node(
        ...     system_prompt="Classify as positive or negative.",
        ...     input_key="text", output_key="sentiment",
        ... ))
        >>> result = flow.run(text="I love this product!")
    """

    def _fn(state: dict) -> dict:
        from ..tasker.genai_tasker import TaskExecutor

        executor = TaskExecutor(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        prompt = str(state.get(input_key, ""))

        if task_file:
            result = executor.run_json_task(task_file, prompt)
        else:
            input_data: Union[str, list[dict[str, str]]]
            if system_prompt:
                input_data = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                input_data = prompt
            result = executor.execute(input_data, output_schema=output_schema)

        return {output_key: result}

    return _fn


def agent_node(
    agent: Any,
    *,
    input_key: str = "input",
    output_key: str = "output",
    state_keys: Optional[dict[str, str]] = None,
) -> Callable[[dict], dict]:
    """Create a step function that runs a Nono ``Agent``.

    Returns a standard ``Callable[[dict], dict]`` that reads from
    ``state[input_key]``, invokes ``Runner(agent).run()``, and writes to
    ``state[output_key]``.  Use it with ``Workflow.step()``.

    Args:
        agent: A ``BaseAgent`` instance (``Agent``, ``LlmAgent``, etc.).
        input_key: State key to read the user message from.
        output_key: State key to write the agent response to.
        state_keys: Optional mapping ``{runner_state_key: workflow_state_key}``
            to forward workflow state entries to the Runner session.

    Returns:
        A callable suitable for ``Workflow.step(name, fn)``.

    Example:
        >>> from nono.agent import Agent
        >>> from nono.workflows import Workflow, agent_node
        >>> writer = Agent(name="writer", instruction="Write a blog post.")
        >>> flow = Workflow("blog")
        >>> flow.step("write", agent_node(writer, input_key="topic", output_key="draft"))
        >>> result = flow.run(topic="AI trends 2026")
    """

    def _fn(state: dict) -> dict:
        from ..agent.runner import Runner

        runner_inst = Runner(agent=agent)
        extra: dict[str, Any] = {}
        if state_keys:
            extra = {
                sk: state.get(wk, "")
                for sk, wk in state_keys.items()
            }
        message = str(state.get(input_key, ""))
        response = runner_inst.run(message, **extra)
        return {output_key: response}

    return _fn


def human_node(
    *,
    handler: HumanInputHandler,
    prompt: str = "Awaiting human input...",
    state_key: str = "human_input",
    on_reject: str = "error",
) -> Callable[[dict], dict]:
    """Create a step function that pauses for human input.

    Returns a standard ``Callable[[dict], dict]`` that blocks until the
    human responds.  Use it with ``Workflow.step()`` when you prefer the
    factory-function style over ``Workflow.human_step()``.

    Unlike ``human_step()``, this factory does **not** auto-register a
    reject branch — use ``Workflow.branch()`` / ``branch_if()`` manually
    if you need conditional routing on rejection.

    Args:
        handler: Sync callback ``(step_name, state, prompt) ->
            HumanInputResponse``.
        prompt: Message shown to the human.
        state_key: Key in workflow state where the response is stored.
        on_reject: ``"error"`` to raise ``HumanRejectError``,
            ``"continue"`` to proceed with the rejection in state.

    Returns:
        A callable suitable for ``Workflow.step(name, fn)``.

    Example:
        >>> from nono.workflows import Workflow, human_node
        >>> from nono.hitl import HumanInputResponse
        >>>
        >>> def cli_handler(step, state, prompt):
        ...     answer = input(f"[{step}] {prompt}: ")
        ...     return HumanInputResponse(approved=answer.lower() != "no", message=answer)
        >>>
        >>> flow = Workflow("review")
        >>> flow.step("draft", draft_fn)
        >>> flow.step("review", human_node(handler=cli_handler, prompt="Approve?"))
        >>> flow.step("publish", publish_fn)
    """

    def _fn(state: dict) -> dict:
        # The step name is not available here; use a generic label
        step_name = state.get("__current_step__", "human_input")
        logger.info("Requesting human input: %s", prompt)

        response = handler(step_name, dict(state), prompt)

        logger.info(
            "Human responded: approved=%s message=%r",
            response.approved, response.message,
        )

        result: dict[str, Any] = {
            state_key: {
                "approved": response.approved,
                "message": response.message,
                **response.data,
            },
        }

        if response.data:
            result.update(response.data)

        if not response.approved:
            result["human_rejected"] = True

            if on_reject == "error":
                raise HumanRejectError(step_name, response.message)

        return result

    return _fn
