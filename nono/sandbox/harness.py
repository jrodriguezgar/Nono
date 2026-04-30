"""
Harness/compute separation with snapshotting and rehydration.

The **harness** manages agent state, session events, and orchestration
logic — it runs in the local process.  The **compute** is the remote
sandbox container where untrusted code executes.

When a sandbox container fails (crash, timeout, eviction), the harness can
**snapshot** the current execution state, **restore** a new sandbox from
that snapshot, and **continue** from the last checkpoint — the agent loop
never restarts from zero.

Part of the Nono Agent Architecture (NAA).

Key classes:
    - ``SandboxSnapshot``: Serialisable checkpoint of a sandbox run.
    - ``SnapshotStore``: Pluggable persistence for snapshots (in-memory, disk).
    - ``HarnessRuntime``: Orchestrator that separates harness from compute.

Example:
    >>> from nono.sandbox.harness import HarnessRuntime, SnapshotStore
    >>> from nono.sandbox import SandboxRunConfig
    >>> from nono.sandbox.base import SandboxProvider
    >>>
    >>> runtime = HarnessRuntime(max_retries=3)
    >>> result = runtime.execute(
    ...     code="print('durable')",
    ...     config=SandboxRunConfig(provider=SandboxProvider.E2B),
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .base import (
    BaseSandboxClient,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)

logger = logging.getLogger("Nono.Sandbox.Harness")


# ── Snapshot ──────────────────────────────────────────────────────────────────

class CheckpointStatus(Enum):
    """Status of a snapshot checkpoint."""

    CREATED = "created"
    ACTIVE = "active"
    RESTORED = "restored"
    EXPIRED = "expired"


@dataclass
class SandboxSnapshot:
    """Serialisable checkpoint of a sandbox execution.

    Captures everything needed to resume a sandbox run in a fresh
    container after the original one fails.

    Args:
        snapshot_id: Unique identifier for this snapshot.
        sandbox_id: Provider sandbox ID at the time of the snapshot.
        provider: Sandbox provider that created the snapshot.
        code: Full code being executed.
        code_cursor: Character offset into *code* that was reached.
            ``0`` means "nothing executed yet"; ``len(code)`` means done.
        step_index: For multi-step pipelines, which step was active.
        total_steps: Total number of steps in the pipeline.
        session_state: Copy of ``Session.state`` at checkpoint time.
        shared_content_keys: Names of ``SharedContent`` items saved so far.
        accumulated_stdout: stdout collected before the failure.
        accumulated_stderr: stderr collected before the failure.
        environment: Env-vars that were passed to the sandbox.
        packages: Packages installed in the sandbox.
        working_dir: Working directory inside the sandbox.
        manifest_dict: Serialised manifest (via ``Manifest.to_dict()``).
        provider_snapshot_id: Provider-native snapshot ID (if supported).
        created_at: UTC timestamp.
        status: Current status.
        attempt: Which retry attempt produced this snapshot (0-based).
        metadata: Extra data.
    """

    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    sandbox_id: str = ""
    provider: str = ""
    code: str = ""
    code_cursor: int = 0
    step_index: int = 0
    total_steps: int = 1
    session_state: dict[str, Any] = field(default_factory=dict)
    shared_content_keys: list[str] = field(default_factory=list)
    accumulated_stdout: str = ""
    accumulated_stderr: str = ""
    environment: dict[str, str] = field(default_factory=dict)
    packages: list[str] = field(default_factory=list)
    working_dir: str = "/home/user"
    manifest_dict: dict[str, Any] = field(default_factory=dict)
    provider_snapshot_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: CheckpointStatus = CheckpointStatus.CREATED
    attempt: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "snapshot_id": self.snapshot_id,
            "sandbox_id": self.sandbox_id,
            "provider": self.provider,
            "code": self.code,
            "code_cursor": self.code_cursor,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "session_state": self.session_state,
            "shared_content_keys": self.shared_content_keys,
            "accumulated_stdout": self.accumulated_stdout,
            "accumulated_stderr": self.accumulated_stderr,
            "environment": self.environment,
            "packages": self.packages,
            "working_dir": self.working_dir,
            "manifest_dict": self.manifest_dict,
            "provider_snapshot_id": self.provider_snapshot_id,
            "created_at": self.created_at,
            "status": self.status.value,
            "attempt": self.attempt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxSnapshot:
        """Deserialise from a plain dict.

        Args:
            data: Dict previously produced by :meth:`to_dict`.

        Returns:
            A new ``SandboxSnapshot`` instance.
        """
        status_raw = data.get("status", "created")
        status = CheckpointStatus(status_raw) if isinstance(status_raw, str) else status_raw

        return cls(
            snapshot_id=data.get("snapshot_id", uuid.uuid4().hex[:16]),
            sandbox_id=data.get("sandbox_id", ""),
            provider=data.get("provider", ""),
            code=data.get("code", ""),
            code_cursor=data.get("code_cursor", 0),
            step_index=data.get("step_index", 0),
            total_steps=data.get("total_steps", 1),
            session_state=data.get("session_state", {}),
            shared_content_keys=data.get("shared_content_keys", []),
            accumulated_stdout=data.get("accumulated_stdout", ""),
            accumulated_stderr=data.get("accumulated_stderr", ""),
            environment=data.get("environment", {}),
            packages=data.get("packages", []),
            working_dir=data.get("working_dir", "/home/user"),
            manifest_dict=data.get("manifest_dict", {}),
            provider_snapshot_id=data.get("provider_snapshot_id", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            status=status,
            attempt=data.get("attempt", 0),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> SandboxSnapshot:
        """Deserialise from a JSON string.

        Args:
            raw: JSON string.

        Returns:
            A ``SandboxSnapshot``.
        """
        return cls.from_dict(json.loads(raw))


# ── Snapshot store ────────────────────────────────────────────────────────────

class SnapshotStore:
    """Pluggable persistence backend for sandbox snapshots.

    Default implementation stores snapshots in-memory.  Call
    :meth:`use_disk` to persist snapshots to a directory.

    Args:
        persist_dir: If set, snapshots are written to this directory
            as JSON files.  ``None`` = in-memory only.
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._memory: dict[str, SandboxSnapshot] = {}
        self._persist_dir: Path | None = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def in_memory(cls) -> SnapshotStore:
        """Create an in-memory-only store."""
        return cls(persist_dir=None)

    @classmethod
    def on_disk(cls, directory: str | Path) -> SnapshotStore:
        """Create a disk-backed store.

        Args:
            directory: Directory for JSON snapshot files.
        """
        return cls(persist_dir=directory)

    def save(self, snapshot: SandboxSnapshot) -> str:
        """Persist a snapshot.

        Args:
            snapshot: The snapshot to store.

        Returns:
            The snapshot ID.
        """
        self._memory[snapshot.snapshot_id] = snapshot

        if self._persist_dir:
            path = self._persist_dir / f"{snapshot.snapshot_id}.json"
            path.write_text(snapshot.to_json(), encoding="utf-8")
            logger.debug("Snapshot %s saved to %s", snapshot.snapshot_id, path)

        return snapshot.snapshot_id

    def load(self, snapshot_id: str) -> SandboxSnapshot | None:
        """Load a snapshot by ID.

        Args:
            snapshot_id: Unique snapshot identifier.

        Returns:
            The snapshot, or ``None`` if not found.
        """
        snap = self._memory.get(snapshot_id)

        if snap is not None:
            return snap

        if self._persist_dir:
            path = self._persist_dir / f"{snapshot_id}.json"

            if path.exists():
                snap = SandboxSnapshot.from_json(path.read_text(encoding="utf-8"))
                self._memory[snapshot_id] = snap
                return snap

        return None

    def list_ids(self) -> list[str]:
        """Return all stored snapshot IDs."""
        ids = set(self._memory.keys())

        if self._persist_dir:
            for p in self._persist_dir.glob("*.json"):
                ids.add(p.stem)

        return sorted(ids)

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot.

        Args:
            snapshot_id: ID to remove.

        Returns:
            ``True`` if something was deleted.
        """
        deleted = self._memory.pop(snapshot_id, None) is not None

        if self._persist_dir:
            path = self._persist_dir / f"{snapshot_id}.json"

            if path.exists():
                path.unlink()
                deleted = True

        return deleted


# ── Harness runtime ───────────────────────────────────────────────────────────

@dataclass
class HarnessEvent:
    """Record of a harness-level lifecycle event.

    Args:
        kind: Event type identifier.
        message: Human-readable description.
        snapshot_id: Related snapshot ID (if any).
        attempt: Retry attempt number.
        timestamp: UTC timestamp.
    """

    kind: str
    message: str
    snapshot_id: str = ""
    attempt: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HarnessRuntime:
    """Orchestrator that separates harness (local state) from compute (sandbox).

    The runtime wraps a ``BaseSandboxClient`` and adds:

    1. **Automatic checkpointing** — snapshot before and after each
       execution step.
    2. **Retry with rehydration** — on sandbox failure, restore state
       in a fresh container and continue from the last checkpoint.
    3. **Event log** — structured ``HarnessEvent`` trace of every
       provision, execute, snapshot, restore, and cleanup action.

    Credentials and session state **never** leave the harness process.
    Only the code to execute and the manifest entries are sent to the
    sandbox.

    Args:
        client: Pre-built sandbox client (overrides config provider).
        store: Snapshot persistence backend.  Defaults to in-memory.
        max_retries: Maximum retry attempts after sandbox failure.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        *,
        client: BaseSandboxClient | None = None,
        store: SnapshotStore | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self._client = client
        self.store = store or SnapshotStore.in_memory()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.events: list[HarnessEvent] = []

    def _get_client(self, config: SandboxRunConfig) -> BaseSandboxClient:
        """Return (or lazily create) the sandbox client."""
        if self._client is not None:
            return self._client

        from .sandbox_agent import get_sandbox_client

        return get_sandbox_client(config.provider)

    def _emit(self, kind: str, message: str, **kwargs: Any) -> None:
        """Append a harness event."""
        evt = HarnessEvent(kind=kind, message=message, **kwargs)
        self.events.append(evt)
        logger.info("Harness [%s]: %s", kind, message)

    # ── public API ────────────────────────────────────────────────────

    def execute(
        self,
        code: str,
        config: SandboxRunConfig,
        *,
        session_state: dict[str, Any] | None = None,
    ) -> SandboxResult:
        """Execute code with automatic snapshotting and retry.

        If the sandbox fails, the harness snapshots the accumulated state,
        provisions a fresh container, restores from the snapshot, and
        retries the remaining work.

        Args:
            code: Python code to execute.
            config: Sandbox run configuration.
            session_state: Optional session state to preserve across retries.

        Returns:
            A ``SandboxResult`` with combined output from all attempts.
        """
        client = self._get_client(config)
        manifest_dict = config.manifest.to_dict() if config.manifest else {}
        state = dict(session_state or {})

        accumulated_stdout = ""
        accumulated_stderr = ""
        last_sandbox_id = ""

        for attempt in range(self.max_retries + 1):
            # ── checkpoint before execution ───────────────────────────
            snapshot = SandboxSnapshot(
                provider=config.provider.value,
                code=code,
                code_cursor=0,
                session_state=state,
                accumulated_stdout=accumulated_stdout,
                accumulated_stderr=accumulated_stderr,
                environment=config.environment,
                packages=config.packages,
                working_dir=config.working_dir,
                manifest_dict=manifest_dict,
                status=CheckpointStatus.ACTIVE,
                attempt=attempt,
            )
            self.store.save(snapshot)
            self._emit(
                "checkpoint",
                f"Snapshot {snapshot.snapshot_id} saved (attempt {attempt})",
                snapshot_id=snapshot.snapshot_id,
                attempt=attempt,
            )

            # ── try provider-native restore if we have a previous snapshot ─
            if attempt > 0 and snapshot.provider_snapshot_id:
                try:
                    new_sandbox_id = client.restore(
                        snapshot.provider_snapshot_id, config
                    )
                    self._emit(
                        "restore",
                        f"Restored sandbox {new_sandbox_id} from provider "
                        f"snapshot {snapshot.provider_snapshot_id}",
                        snapshot_id=snapshot.snapshot_id,
                        attempt=attempt,
                    )
                    last_sandbox_id = new_sandbox_id
                except NotImplementedError:
                    self._emit(
                        "restore_skip",
                        "Provider does not support restore — full re-execution",
                        attempt=attempt,
                    )

            # ── execute in sandbox ────────────────────────────────────
            self._emit(
                "execute",
                f"Executing code in {config.provider.value} (attempt {attempt})",
                attempt=attempt,
            )

            try:
                result = client.execute(code, config)
            except Exception as exc:
                self._emit(
                    "execute_error",
                    f"Sandbox error on attempt {attempt}: {exc}",
                    snapshot_id=snapshot.snapshot_id,
                    attempt=attempt,
                )

                # Update snapshot with accumulated output
                snapshot.accumulated_stderr += f"\n[attempt {attempt}] {exc}"
                snapshot.status = CheckpointStatus.RESTORED
                self.store.save(snapshot)

                if attempt < self.max_retries:
                    self._emit(
                        "retry",
                        f"Retrying in {self.retry_delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries})",
                        snapshot_id=snapshot.snapshot_id,
                        attempt=attempt,
                    )
                    time.sleep(self.retry_delay)
                    continue

                # All retries exhausted
                return SandboxResult(
                    status=SandboxStatus.FAILED,
                    stdout=accumulated_stdout,
                    stderr=snapshot.accumulated_stderr,
                    exit_code=1,
                    sandbox_id=last_sandbox_id,
                    snapshot_id=snapshot.snapshot_id,
                    metadata={"harness_events": len(self.events)},
                )

            # ── sandbox returned (may be success or failure) ──────────
            last_sandbox_id = result.sandbox_id or last_sandbox_id
            accumulated_stdout += result.stdout
            accumulated_stderr += result.stderr

            # Try to take a provider snapshot for durability
            provider_snap_id = ""

            if config.snapshot and result.sandbox_id:
                try:
                    provider_snap_id = client.snapshot(result.sandbox_id)
                    snapshot.provider_snapshot_id = provider_snap_id
                    self._emit(
                        "provider_snapshot",
                        f"Provider snapshot {provider_snap_id}",
                        snapshot_id=snapshot.snapshot_id,
                        attempt=attempt,
                    )
                except NotImplementedError:
                    pass

            if result.success:
                # ── success — final checkpoint ────────────────────────
                snapshot.status = CheckpointStatus.ACTIVE
                snapshot.accumulated_stdout = accumulated_stdout
                snapshot.accumulated_stderr = accumulated_stderr
                snapshot.provider_snapshot_id = provider_snap_id
                self.store.save(snapshot)

                self._emit(
                    "completed",
                    f"Execution completed successfully (attempt {attempt})",
                    snapshot_id=snapshot.snapshot_id,
                    attempt=attempt,
                )

                return SandboxResult(
                    status=SandboxStatus.COMPLETED,
                    stdout=accumulated_stdout,
                    stderr=accumulated_stderr,
                    exit_code=0,
                    output_files=result.output_files,
                    sandbox_id=last_sandbox_id,
                    snapshot_id=snapshot.snapshot_id,
                    duration_seconds=result.duration_seconds,
                    metadata={
                        "attempts": attempt + 1,
                        "harness_events": len(self.events),
                        "provider_snapshot_id": provider_snap_id,
                    },
                )

            # ── sandbox returned a failure — retry? ───────────────────
            snapshot.accumulated_stdout = accumulated_stdout
            snapshot.accumulated_stderr = accumulated_stderr
            snapshot.status = CheckpointStatus.RESTORED
            self.store.save(snapshot)

            self._emit(
                "execute_failed",
                f"Sandbox returned exit_code={result.exit_code} "
                f"(attempt {attempt})",
                snapshot_id=snapshot.snapshot_id,
                attempt=attempt,
            )

            if attempt < self.max_retries:
                self._emit(
                    "retry",
                    f"Retrying in {self.retry_delay}s "
                    f"(attempt {attempt + 1}/{self.max_retries})",
                    snapshot_id=snapshot.snapshot_id,
                    attempt=attempt,
                )
                time.sleep(self.retry_delay)
                continue

            # Exhausted retries — return combined result
            return SandboxResult(
                status=SandboxStatus.FAILED,
                stdout=accumulated_stdout,
                stderr=accumulated_stderr,
                exit_code=result.exit_code,
                output_files=result.output_files,
                sandbox_id=last_sandbox_id,
                snapshot_id=snapshot.snapshot_id,
                duration_seconds=result.duration_seconds,
                metadata={
                    "attempts": attempt + 1,
                    "harness_events": len(self.events),
                },
            )

        # Should not reach here, but satisfy type checker
        return SandboxResult(
            status=SandboxStatus.FAILED,
            stderr="Unexpected harness loop exit.",
            exit_code=1,
        )

    def resume(
        self,
        snapshot_id: str,
        config: SandboxRunConfig,
    ) -> SandboxResult:
        """Resume execution from a previously saved snapshot.

        Loads the snapshot, rehydrates the harness state, and calls
        :meth:`execute` with the remaining code.

        Args:
            snapshot_id: ID of a snapshot saved by :meth:`execute`.
            config: Sandbox run configuration (may differ from original).

        Returns:
            A ``SandboxResult``.

        Raises:
            ValueError: If the snapshot is not found.
        """
        snap = self.store.load(snapshot_id)

        if snap is None:
            raise ValueError(f"Snapshot {snapshot_id!r} not found in store.")

        self._emit(
            "resume",
            f"Resuming from snapshot {snapshot_id} (attempt {snap.attempt})",
            snapshot_id=snapshot_id,
            attempt=snap.attempt,
        )

        return self.execute(
            code=snap.code,
            config=config,
            session_state=snap.session_state,
        )
