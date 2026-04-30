"""
Base sandbox client and result types.

Defines the ``BaseSandboxClient`` abstract class that all sandbox providers
implement, plus ``SandboxResult`` and ``SandboxRunConfig`` data containers.

Part of the Nono Agent Architecture (NAA).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .manifest import Manifest

logger = logging.getLogger("Nono.Sandbox")


# ── Enums ─────────────────────────────────────────────────────────────────────

class SandboxProvider(Enum):
    """Supported external sandbox providers."""

    E2B = "e2b"
    MODAL = "modal"
    DAYTONA = "daytona"
    BLAXEL = "blaxel"
    CLOUDFLARE = "cloudflare"
    RUNLOOP = "runloop"
    VERCEL = "vercel"


class SandboxStatus(Enum):
    """Lifecycle status of a sandbox instance."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ── Config & Result ───────────────────────────────────────────────────────────

@dataclass
class SandboxRunConfig:
    """Configuration for a single sandbox execution.

    Args:
        provider: Which sandbox provider to use.
        timeout: Maximum execution time in seconds (0 = provider default).
        environment: Environment variables passed into the sandbox.
        packages: Python packages to install before execution.
        working_dir: Working directory inside the sandbox.
        manifest: Workspace manifest describing mounted files.
        keep_alive: Keep the sandbox alive after execution for inspection.
        snapshot: Enable state snapshots for durability / rehydration.
        metadata: Extra provider-specific options.
    """

    provider: SandboxProvider = SandboxProvider.E2B
    timeout: int = 300
    environment: dict[str, str] = field(default_factory=dict)
    packages: list[str] = field(default_factory=list)
    working_dir: str = "/home/user"
    manifest: Manifest | None = None
    keep_alive: bool = False
    snapshot: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result returned after sandbox execution completes.

    Args:
        status: Final status of the execution.
        stdout: Captured standard output.
        stderr: Captured standard error.
        exit_code: Process exit code (``0`` = success).
        output_files: Mapping of relative path → bytes for files
            collected from the ``OutputDir``.
        sandbox_id: Provider-assigned identifier for the sandbox instance.
        snapshot_id: Snapshot identifier (if ``snapshot=True``).
        duration_seconds: Wall-clock duration of the execution.
        metadata: Extra provider-specific data.
    """

    status: SandboxStatus = SandboxStatus.COMPLETED
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    output_files: dict[str, bytes] = field(default_factory=dict)
    sandbox_id: str = ""
    snapshot_id: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return ``True`` when the sandbox exited cleanly."""
        return self.status == SandboxStatus.COMPLETED and self.exit_code == 0


# ── Abstract client ───────────────────────────────────────────────────────────

class BaseSandboxClient(ABC):
    """Abstract base class for sandbox provider clients.

    Every provider subclass must implement :meth:`execute`, and optionally
    :meth:`execute_async`, :meth:`snapshot`, and :meth:`restore`.

    The ``execute`` contract:

    1. Provision or reuse a sandbox environment.
    2. Materialise the ``Manifest`` entries into the sandbox filesystem.
    3. Run the supplied *command* (or code string).
    4. Collect outputs and return a ``SandboxResult``.

    Args:
        api_key_env: Name of the env-var holding the provider API key.
    """

    def __init__(self, *, api_key_env: str = "") -> None:
        self._api_key_env = api_key_env

    # ── required ──────────────────────────────────────────────────────

    @abstractmethod
    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Run *code* inside the sandbox and return the result.

        Args:
            code: Python code (or shell command) to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult`` with captured output and status.
        """

    # ── optional ──────────────────────────────────────────────────────

    async def execute_async(
        self,
        code: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Async version of :meth:`execute`.

        Default implementation delegates to the synchronous method.

        Args:
            code: Code to execute.
            config: Execution configuration.

        Returns:
            A ``SandboxResult``.
        """
        return self.execute(code, config)

    def snapshot(self, sandbox_id: str) -> str:
        """Take a state snapshot of a running sandbox.

        Args:
            sandbox_id: Provider sandbox identifier.

        Returns:
            A snapshot ID that can be passed to :meth:`restore`.

        Raises:
            NotImplementedError: Provider does not support snapshots.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support snapshots."
        )

    def restore(self, snapshot_id: str, config: SandboxRunConfig) -> str:
        """Restore a sandbox from a snapshot.

        Args:
            snapshot_id: ID returned by a previous :meth:`snapshot` call.
            config: Configuration for the restored sandbox.

        Returns:
            The new sandbox ID.

        Raises:
            NotImplementedError: Provider does not support snapshots.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support restore."
        )

    def terminate(self, sandbox_id: str) -> None:
        """Terminate and clean up a sandbox instance.

        Args:
            sandbox_id: Provider sandbox identifier.
        """
        logger.warning(
            "%s.terminate() not implemented — sandbox %s may leak.",
            type(self).__name__,
            sandbox_id,
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _get_api_key(self) -> str:
        """Read the API key from the configured environment variable.

        Returns:
            The API key string.

        Raises:
            EnvironmentError: If the env-var is not set.
        """
        import os

        if not self._api_key_env:
            return ""

        key = os.environ.get(self._api_key_env, "")

        if not key:
            raise EnvironmentError(
                f"Sandbox API key not found. Set the {self._api_key_env!r} "
                "environment variable."
            )

        return key
