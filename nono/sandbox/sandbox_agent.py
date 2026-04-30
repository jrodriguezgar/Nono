"""
SandboxAgent — an agent that executes code inside external sandboxes.

Wraps an ``LlmAgent`` so that any code the model generates is executed in
a controlled sandbox environment (E2B, Modal, Daytona, etc.) instead of the
local machine.  The sandbox workspace is described by a ``Manifest``.

Part of the Nono Agent Architecture (NAA).

Example:
    >>> from nono.sandbox import SandboxAgent, SandboxRunConfig, Manifest, LocalDir
    >>> from nono.sandbox.base import SandboxProvider
    >>>
    >>> agent = SandboxAgent(
    ...     name="Analyst",
    ...     model="gemini-3-flash-preview",
    ...     instructions="Analyse data files and produce a summary.",
    ...     sandbox_config=SandboxRunConfig(
    ...         provider=SandboxProvider.E2B,
    ...         manifest=Manifest(entries={"data": LocalDir(src="/tmp/reports")}),
    ...     ),
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from ..agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from .base import (
    BaseSandboxClient,
    SandboxProvider,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)

logger = logging.getLogger("Nono.Sandbox.Agent")


# ── Provider → Client registry ────────────────────────────────────────────────

_CLIENT_REGISTRY: dict[SandboxProvider, type[BaseSandboxClient]] = {}


def _ensure_registry() -> None:
    """Lazily populate the provider → client mapping."""
    if _CLIENT_REGISTRY:
        return

    from .clients.e2b import E2BSandboxClient
    from .clients.modal import ModalSandboxClient
    from .clients.daytona import DaytonaSandboxClient
    from .clients.blaxel import BlaxelSandboxClient
    from .clients.cloudflare import CloudflareSandboxClient
    from .clients.runloop import RunloopSandboxClient
    from .clients.vercel import VercelSandboxClient

    _CLIENT_REGISTRY.update({
        SandboxProvider.E2B: E2BSandboxClient,
        SandboxProvider.MODAL: ModalSandboxClient,
        SandboxProvider.DAYTONA: DaytonaSandboxClient,
        SandboxProvider.BLAXEL: BlaxelSandboxClient,
        SandboxProvider.CLOUDFLARE: CloudflareSandboxClient,
        SandboxProvider.RUNLOOP: RunloopSandboxClient,
        SandboxProvider.VERCEL: VercelSandboxClient,
    })


def get_sandbox_client(
    provider: SandboxProvider,
    **kwargs: Any,
) -> BaseSandboxClient:
    """Instantiate a sandbox client for the given provider.

    Args:
        provider: Target sandbox provider.
        **kwargs: Extra keyword arguments forwarded to the client constructor.

    Returns:
        A ``BaseSandboxClient`` instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if not isinstance(provider, SandboxProvider):
        raise ValueError(
            f"Unsupported sandbox provider: {provider!r}. "
            f"Supported: {[p.value for p in SandboxProvider]}"
        )

    _ensure_registry()
    cls = _CLIENT_REGISTRY.get(provider)

    if cls is None:
        raise ValueError(
            f"Unsupported sandbox provider: {provider.value!r}. "
            f"Supported: {[p.value for p in SandboxProvider]}"
        )

    return cls(**kwargs)


# ── SandboxAgent ──────────────────────────────────────────────────────────────

class SandboxAgent(BaseAgent):
    """Agent that delegates code execution to an external sandbox.

    The agent itself does **not** call an LLM — it receives code (typically
    from a parent ``LlmAgent``) and runs it inside the configured sandbox.
    Results are returned as events so the orchestrating agent can continue.

    For a fully autonomous sandbox-aware agent, pair this with an ``LlmAgent``
    that uses ``SandboxAgent`` as a sub-agent (tool-based handoff).

    Args:
        name: Agent name.
        sandbox_config: Sandbox execution configuration.
        client: Pre-built sandbox client (overrides ``sandbox_config.provider``).
        description: Human-readable description of the agent.
        instructions: System-level instructions (informational only — no LLM).
    """

    def __init__(
        self,
        *,
        name: str = "SandboxAgent",
        sandbox_config: SandboxRunConfig | None = None,
        client: BaseSandboxClient | None = None,
        description: str = "Executes code in an external sandbox environment.",
        instructions: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.instructions = instructions
        self.sandbox_config = sandbox_config or SandboxRunConfig()
        self._client = client

    @property
    def client(self) -> BaseSandboxClient:
        """Return (or lazily create) the sandbox client."""
        if self._client is None:
            self._client = get_sandbox_client(self.sandbox_config.provider)

        return self._client

    # ── BaseAgent contract ────────────────────────────────────────────

    def _run_impl(self, context: InvocationContext) -> Iterator[Event]:
        """Execute the user message as code inside the sandbox.

        The content of ``context.user_message`` is treated as code to run.

        Args:
            context: Invocation context with the code to execute.

        Yields:
            Events reporting sandbox execution progress and results.
        """
        code = context.user_message
        logger.info(
            "SandboxAgent %r: executing %d chars in %s sandbox.",
            self.name,
            len(code),
            self.sandbox_config.provider.value,
        )

        yield Event(
            event_type=EventType.STATE_UPDATE,
            author=self.name,
            content=f"Running code in {self.sandbox_config.provider.value} sandbox...",
            data={"sandbox_provider": self.sandbox_config.provider.value},
        )

        try:
            result = self.client.execute(code, self.sandbox_config)
        except Exception as exc:
            logger.error("SandboxAgent %r: execution error: %s", self.name, exc)
            yield Event(
                event_type=EventType.ERROR,
                author=self.name,
                content=f"Sandbox execution failed: {exc}",
            )
            return

        # Emit the result
        yield Event(
            event_type=EventType.TOOL_RESULT,
            author=self.name,
            content=result.stdout or result.stderr,
            data={
                "status": result.status.value,
                "exit_code": result.exit_code,
                "sandbox_id": result.sandbox_id,
                "duration_seconds": result.duration_seconds,
                "output_files": list(result.output_files.keys()),
            },
        )

        # Store output files in shared content if available
        if result.output_files and context.session is not None:
            for filename, content_bytes in result.output_files.items():
                context.session.shared_content.save(
                    name=f"sandbox:{filename}",
                    data=content_bytes,
                    content_type="application/octet-stream",
                    metadata={"sandbox_id": result.sandbox_id},
                    created_by=self.name,
                )

        # Final agent response
        if result.success:
            summary = result.stdout or "(no output)"
        else:
            summary = (
                f"Sandbox execution failed (exit code {result.exit_code}).\n"
                f"stderr: {result.stderr}"
            )

        yield Event(
            event_type=EventType.AGENT_MESSAGE,
            author=self.name,
            content=summary,
        )

    async def _run_async_impl(self, context: InvocationContext) -> AsyncIterator[Event]:
        """Async execution — delegates to the sync implementation.

        Args:
            context: Invocation context.

        Yields:
            Events from sandbox execution.
        """
        for event in self._run_impl(context):
            yield event
