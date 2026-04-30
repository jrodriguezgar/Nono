"""
KeepInMind — Optional persistent conversation memory for Nono agents.

Provides a pluggable memory system that stores and retrieves conversation
history across sessions.  When attached to a ``Session``, the memory is
automatically loaded into the LLM context so the agent *remembers*
previous interactions.

Architecture:
    - **MemoryStore** (ABC): Backend contract — implement ``save``, ``load``,
      ``list_sessions``, ``delete``, ``clear``.
    - **FileMemoryStore**: Default backend that persists each session as a
      JSON Lines file on disk (zero external dependencies).
    - **KeepInMind**: High-level façade that wraps a store, attaches to a
      ``Session``, and exposes ``remember`` / ``commit`` helpers.

Usage:
    from nono.agent import Agent, Runner, Session, KeepInMind

    # Persistent memory in a local directory
    memory = KeepInMind(path="./memory")

    session = Session(memory=memory)
    runner = Runner(agent, session=session)

    runner.run("My name is Carlos")     # saved automatically
    runner.run("What is my name?")      # agent sees prior turns

    # Next process — same memory directory → remembers Carlos
    memory2 = KeepInMind(path="./memory")
    session2 = Session(session_id=session.session_id, memory=memory2)
    runner2 = Runner(agent, session=session2)
    runner2.run("Do you remember my name?")   # "Carlos"
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("Nono.Agent.KeepInMind")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryEntry:
    """A single remembered message.

    Args:
        role: Message role (``"user"``, ``"assistant"``, ``"system"``).
        content: Message text.
        agent_name: Name of the agent that produced or received this message.
        timestamp: UTC timestamp when the message was recorded.
        metadata: Arbitrary extra data.
    """

    role: str
    content: str
    agent_name: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe).

        Returns:
            Dictionary representation of this entry.
        """
        return {
            "role": self.role,
            "content": self.content,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialise from a plain dict.

        Args:
            data: Dictionary with entry fields.

        Returns:
            A ``MemoryEntry`` instance.
        """
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            agent_name=data.get("agent_name", ""),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )

    def to_message(self) -> dict[str, str]:
        """Convert to LLM message format.

        Returns:
            ``{"role": ..., "content": ...}`` dict.
        """
        return {"role": self.role, "content": self.content}


# ── Abstract store ────────────────────────────────────────────────────────────

class MemoryStore(ABC):
    """Backend contract for persistent memory storage.

    Implementations must handle serialisation and I/O.  All methods
    are synchronous — async wrappers can be added externally.
    """

    @abstractmethod
    def save(self, session_id: str, entries: list[MemoryEntry]) -> None:
        """Persist entries for a session (append or overwrite).

        Args:
            session_id: Unique session identifier.
            entries: Memory entries to store.
        """

    @abstractmethod
    def load(self, session_id: str) -> list[MemoryEntry]:
        """Load all entries for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of previously stored entries (empty if none).
        """

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """List all session IDs that have stored memory.

        Returns:
            List of session ID strings.
        """

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete all memory for a session.

        Args:
            session_id: Session to delete.

        Returns:
            ``True`` if the session existed and was deleted.
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored memory across all sessions."""


# ── File-based store ──────────────────────────────────────────────────────────

class FileMemoryStore(MemoryStore):
    """JSON Lines file-based memory store (zero dependencies).

    Each session is stored as ``<session_id>.jsonl`` inside ``path``.
    One line per ``MemoryEntry``.

    Args:
        path: Directory to store memory files.
        max_entries: Maximum entries per session file. ``0`` means no limit.
            When exceeded, the oldest entries are trimmed on ``save()``.

    Example:
        >>> store = FileMemoryStore("./memory")
        >>> store.save("abc123", [MemoryEntry("user", "hello")])
        >>> store.load("abc123")
        [MemoryEntry(role='user', content='hello', ...)]
    """

    def __init__(self, path: str | Path, *, max_entries: int = 0) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self.max_entries: int = max_entries

    @property
    def path(self) -> Path:
        """Root directory for memory files."""
        return self._path

    def _session_file(self, session_id: str) -> Path:
        """Return the file path for a given session.

        Args:
            session_id: Session identifier.

        Returns:
            Path to the JSONL file.

        Raises:
            ValueError: If session_id contains path-traversal characters.
        """
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")

        if safe_id != session_id:
            raise ValueError(f"Invalid session_id: {session_id!r}")

        return self._path / f"{safe_id}.jsonl"

    def save(self, session_id: str, entries: list[MemoryEntry]) -> None:
        """Persist entries as JSON Lines (overwrites previous content).

        When ``max_entries`` is set, only the most recent entries are kept.

        Args:
            session_id: Unique session identifier.
            entries: Memory entries to store.
        """
        if self.max_entries > 0 and len(entries) > self.max_entries:
            entries = entries[-self.max_entries:]
        file_path = self._session_file(session_id)
        with open(file_path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        logger.debug("KeepInMind: saved %d entries for session %r", len(entries), session_id)

    def append_entry(self, session_id: str, entry: MemoryEntry) -> None:
        """Append a single entry using O(1) file-append mode.

        Args:
            session_id: Unique session identifier.
            entry: Entry to append.
        """
        file_path = self._session_file(session_id)
        with open(file_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def load(self, session_id: str) -> list[MemoryEntry]:
        """Load entries from a JSON Lines file.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of ``MemoryEntry`` instances (empty if file doesn't exist).
        """
        file_path = self._session_file(session_id)

        if not file_path.exists():
            return []

        entries: list[MemoryEntry] = []

        with open(file_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    data = json.loads(stripped)
                    entries.append(MemoryEntry.from_dict(data))
                except json.JSONDecodeError:
                    logger.warning(
                        "KeepInMind: skipping malformed line %d in %s",
                        line_num, file_path.name,
                    )

        logger.debug("KeepInMind: loaded %d entries for session %r", len(entries), session_id)
        return entries

    def list_sessions(self) -> list[str]:
        """List session IDs by scanning ``*.jsonl`` files.

        Returns:
            Sorted list of session ID strings.
        """
        return sorted(
            f.stem for f in self._path.glob("*.jsonl") if f.is_file()
        )

    def delete(self, session_id: str) -> bool:
        """Delete the memory file for a session.

        Args:
            session_id: Session to delete.

        Returns:
            ``True`` if the file existed and was deleted.
        """
        file_path = self._session_file(session_id)

        if file_path.exists():
            file_path.unlink()
            logger.debug("KeepInMind: deleted memory for session %r", session_id)
            return True

        return False

    def clear(self) -> None:
        """Delete all ``*.jsonl`` files in the memory directory."""
        for f in self._path.glob("*.jsonl"):
            f.unlink()
        logger.debug("KeepInMind: cleared all memory in %s", self._path)


# ── KeepInMind façade ─────────────────────────────────────────────────────────

class KeepInMind:
    """Persistent conversation memory for Nono agents.

    High-level façade that wraps a ``MemoryStore`` and provides a clean API
    for loading prior context, recording new turns, and committing to disk.

    This is **optional** — agents work perfectly without it.  Attach a
    ``KeepInMind`` instance to a ``Session`` to enable persistence.

    Args:
        path: Directory for ``FileMemoryStore`` (convenience shortcut).
        store: Custom ``MemoryStore`` backend (overrides ``path``).
        max_turns: Maximum number of past turns to inject into context.
            ``None`` means no limit.

    Example:
        >>> memory = KeepInMind(path="./memory")
        >>> session = Session(memory=memory)
        >>> # … agent runs and memory auto-commits …
        >>> memory.entries("session-id-here")
        [MemoryEntry(...), ...]
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        store: MemoryStore | None = None,
        max_turns: int | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        elif path is not None:
            self._store = FileMemoryStore(path)
        else:
            raise ValueError("Provide either 'path' or 'store'.")

        self.max_turns = max_turns
        self._lock = threading.Lock()

    @property
    def store(self) -> MemoryStore:
        """The underlying memory store."""
        return self._store

    # ── Read ──────────────────────────────────────────────────────────────

    def recall(self, session_id: str) -> list[MemoryEntry]:
        """Load remembered entries for a session.

        Applies ``max_turns`` truncation if configured.

        Args:
            session_id: Session to recall.

        Returns:
            List of ``MemoryEntry`` instances.
        """
        entries = self._store.load(session_id)

        if self.max_turns is not None and len(entries) > self.max_turns:
            entries = entries[-self.max_turns:]

        return entries

    def recall_messages(self, session_id: str) -> list[dict[str, str]]:
        """Load remembered entries as LLM message dicts.

        Args:
            session_id: Session to recall.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        return [e.to_message() for e in self.recall(session_id)]

    def entries(self, session_id: str) -> list[MemoryEntry]:
        """Alias for ``recall`` — returns all stored entries (untruncated).

        Args:
            session_id: Session to query.

        Returns:
            List of ``MemoryEntry`` instances.
        """
        return self._store.load(session_id)

    def sessions(self) -> list[str]:
        """List all session IDs with stored memory.

        Returns:
            Sorted list of session ID strings.
        """
        return self._store.list_sessions()

    # ── Write ─────────────────────────────────────────────────────────────

    def commit(self, session_id: str, entries: list[MemoryEntry]) -> None:
        """Save entries to the store (overwrites previous data).

        Args:
            session_id: Session to save.
            entries: Full list of entries to persist.
        """
        self._store.save(session_id, entries)

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        agent_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Append a single entry and persist immediately.

        Args:
            session_id: Session to append to.
            role: Message role.
            content: Message text.
            agent_name: Name of the producing agent.
            metadata: Extra data.

        Returns:
            The newly created ``MemoryEntry``.
        """
        entry = MemoryEntry(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata or {},
        )
        with self._lock:
            if hasattr(self._store, "append_entry"):
                self._store.append_entry(session_id, entry)
            else:
                existing = self._store.load(session_id)
                existing.append(entry)
                self._store.save(session_id, existing)
        return entry

    # ── Delete ────────────────────────────────────────────────────────────

    def forget(self, session_id: str) -> bool:
        """Delete all memory for a session.

        Args:
            session_id: Session to forget.

        Returns:
            ``True`` if memory existed and was deleted.
        """
        return self._store.delete(session_id)

    def forget_all(self) -> None:
        """Delete all stored memory across all sessions."""
        self._store.clear()

    def __repr__(self) -> str:
        store_type = type(self._store).__name__
        return f"KeepInMind(store={store_type}, max_turns={self.max_turns})"
