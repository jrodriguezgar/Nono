"""
DurableSandboxAgent — fault-tolerant sandbox execution with snapshotting.

Extends :class:`~nono.sandbox.sandbox_agent.SandboxAgent` so that sandbox
failures trigger automatic **rehydration**: the harness snapshots the
accumulated state, provisions a fresh container, and retries from the last
checkpoint.

The harness (this process) and the compute (remote sandbox) are formally
separated — credentials, session state, and event history never leave the
harness; only the code and manifest are sent to the sandbox.

Part of the Nono Agent Architecture (NAA).

Example:
    >>> from nono.sandbox.durable_agent import DurableSandboxAgent
    >>> from nono.sandbox import SandboxRunConfig
    >>> from nono.sandbox.base import SandboxProvider
    >>>
    >>> agent = DurableSandboxAgent(
    ...     name="ResilientWorker",
    ...     sandbox_config=SandboxRunConfig(
    ...         provider=SandboxProvider.E2B,
    ...         snapshot=True,
    ...     ),
    ...     max_retries=3,
    ... )
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Iterator

from ..agent.base import Event, EventType, InvocationContext
from .base import BaseSandboxClient, SandboxRunConfig, SandboxStatus
from .harness import HarnessRuntime, SnapshotStore
from .sandbox_agent import SandboxAgent

logger = logging.getLogger("Nono.Sandbox.DurableAgent")


class DurableSandboxAgent(SandboxAgent):
    """Sandbox agent with automatic snapshotting and rehydration.

    On sandbox failure the harness:

    1. Takes a snapshot of accumulated output and session state.
    2. (Optionally) takes a provider-native snapshot if supported.
    3. Provisions a fresh sandbox or restores from the provider snapshot.
    4. Resumes execution from the checkpoint.

    All retry logic is delegated to :class:`~nono.sandbox.harness.HarnessRuntime`
    so that the agent only sees a single ``SandboxResult``.

    Args:
        name: Agent name.
        sandbox_config: Execution configuration.
        max_retries: Maximum retry attempts.
        retry_delay: Seconds between retries.
        snapshot_store: Where to persist snapshots (default: in-memory).
        description: Human-readable description.
        instructions: System instructions (informational — no LLM).
    """

    def __init__(
        self,
        *,
        name: str = "DurableSandboxAgent",
        sandbox_config: SandboxRunConfig | None = None,
        client: BaseSandboxClient | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        snapshot_store: SnapshotStore | None = None,
        description: str = (
            "Executes code in an external sandbox with automatic "
            "snapshotting and retry on failure."
        ),
        instructions: str = "",
    ) -> None:
        super().__init__(
            name=name,
            sandbox_config=sandbox_config,
            client=client,
            description=description,
            instructions=instructions,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._snapshot_store = snapshot_store

    @property
    def runtime(self) -> HarnessRuntime:
        """Build or reuse a :class:`HarnessRuntime` backed by this agent's client."""
        return HarnessRuntime(
            client=self.client,
            store=self._snapshot_store or SnapshotStore.in_memory(),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

    # ── BaseAgent contract ────────────────────────────────────────────

    def _run_impl(self, context: InvocationContext) -> Iterator[Event]:
        """Execute the user message as code with durable retry.

        The entire execution is wrapped by :class:`HarnessRuntime` so
        that transient sandbox failures are handled transparently.

        Args:
            context: Invocation context with the code to execute.

        Yields:
            Events reporting sandbox execution progress and results.
        """
        code = context.user_message
        logger.info(
            "DurableSandboxAgent %r: executing %d chars in %s sandbox "
            "(max_retries=%d).",
            self.name,
            len(code),
            self.sandbox_config.provider.value,
            self.max_retries,
        )

        yield Event(
            event_type=EventType.STATE_UPDATE,
            author=self.name,
            content=(
                f"Running code in {self.sandbox_config.provider.value} "
                f"sandbox (durable, max retries={self.max_retries})..."
            ),
            data={
                "sandbox_provider": self.sandbox_config.provider.value,
                "durable": True,
                "max_retries": self.max_retries,
            },
        )

        # Capture session state for the harness
        session_state: dict = {}

        if context.session is not None:
            session_state = dict(context.session.state)

        # Delegate to HarnessRuntime
        rt = self.runtime

        try:
            result = rt.execute(
                code=code,
                config=self.sandbox_config,
                session_state=session_state,
            )
        except Exception as exc:
            logger.error(
                "DurableSandboxAgent %r: harness error: %s",
                self.name,
                exc,
            )
            yield Event(
                event_type=EventType.ERROR,
                author=self.name,
                content=f"Durable sandbox execution failed: {exc}",
            )
            return

        # Emit harness events as state updates
        for evt in rt.events:
            yield Event(
                event_type=EventType.STATE_UPDATE,
                author=self.name,
                content=f"[harness] {evt.kind}: {evt.message}",
                data={
                    "harness_event": evt.kind,
                    "snapshot_id": evt.snapshot_id,
                    "attempt": evt.attempt,
                },
            )

        # Emit the combined result
        yield Event(
            event_type=EventType.TOOL_RESULT,
            author=self.name,
            content=result.stdout or result.stderr,
            data={
                "status": result.status.value,
                "exit_code": result.exit_code,
                "sandbox_id": result.sandbox_id,
                "snapshot_id": result.snapshot_id,
                "duration_seconds": result.duration_seconds,
                "output_files": list(result.output_files.keys()),
                "attempts": result.metadata.get("attempts", 1),
            },
        )

        # Store output files in shared content
        if result.output_files and context.session is not None:
            for filename, content_bytes in result.output_files.items():
                context.session.shared_content.save(
                    name=f"sandbox:{filename}",
                    data=content_bytes,
                    content_type="application/octet-stream",
                    metadata={"sandbox_id": result.sandbox_id},
                    created_by=self.name,
                )

        # Write back session state from harness
        if context.session is not None and session_state:
            context.session.state_update(session_state)

        # Final agent response
        attempts = result.metadata.get("attempts", 1)

        if result.success:
            summary = result.stdout or "(no output)"

            if attempts > 1:
                summary += f"\n(completed after {attempts} attempt(s))"
        else:
            summary = (
                f"Sandbox execution failed after {attempts} attempt(s) "
                f"(exit code {result.exit_code}).\n"
                f"stderr: {result.stderr}"
            )

        yield Event(
            event_type=EventType.AGENT_MESSAGE,
            author=self.name,
            content=summary,
        )

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncIterator[Event]:
        """Async execution — delegates to the sync implementation.

        Args:
            context: Invocation context.

        Yields:
            Events from durable sandbox execution.
        """
        for event in self._run_impl(context):
            yield event
