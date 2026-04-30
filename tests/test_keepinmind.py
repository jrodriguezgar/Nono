"""Tests for nono.agent.keepinmind — Persistent conversation memory.

Covers:
- MemoryEntry: serialisation round-trip, to_message
- FileMemoryStore: save, load, list_sessions, delete, clear, path-traversal
- KeepInMind: recall, recall_messages, append, commit, forget, forget_all
- Session integration: auto-load, auto-commit, no-memory fallback

Run:
    uv run pytest tests/test_keepinmind.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nono.agent.base import Event, EventType, Session
from nono.agent.keepinmind import (
    FileMemoryStore,
    KeepInMind,
    MemoryEntry,
    MemoryStore,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    """Temporary directory for memory files."""
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def store(mem_dir: Path) -> FileMemoryStore:
    """FileMemoryStore backed by ``mem_dir``."""
    return FileMemoryStore(mem_dir)


@pytest.fixture
def kim(mem_dir: Path) -> KeepInMind:
    """KeepInMind façade backed by a temp directory."""
    return KeepInMind(path=mem_dir)


# ── MemoryEntry ───────────────────────────────────────────────────────────────


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_round_trip(self) -> None:
        entry = MemoryEntry(role="user", content="hello", agent_name="bot")
        rebuilt = MemoryEntry.from_dict(entry.to_dict())
        assert rebuilt.role == "user"
        assert rebuilt.content == "hello"
        assert rebuilt.agent_name == "bot"

    def test_to_message(self) -> None:
        entry = MemoryEntry(role="assistant", content="hi there")
        msg = entry.to_message()
        assert msg == {"role": "assistant", "content": "hi there"}

    def test_defaults(self) -> None:
        entry = MemoryEntry(role="user", content="x")
        assert entry.agent_name == ""
        assert entry.metadata == {}
        assert entry.timestamp  # non-empty


# ── FileMemoryStore ───────────────────────────────────────────────────────────


class TestFileMemoryStore:
    """Tests for the JSONL-based file store."""

    def test_save_and_load(self, store: FileMemoryStore) -> None:
        entries = [
            MemoryEntry(role="user", content="ping"),
            MemoryEntry(role="assistant", content="pong"),
        ]
        store.save("s1", entries)
        loaded = store.load("s1")
        assert len(loaded) == 2
        assert loaded[0].content == "ping"
        assert loaded[1].content == "pong"

    def test_load_empty(self, store: FileMemoryStore) -> None:
        assert store.load("nonexistent") == []

    def test_list_sessions(self, store: FileMemoryStore) -> None:
        store.save("alpha", [MemoryEntry("user", "a")])
        store.save("beta", [MemoryEntry("user", "b")])
        sessions = store.list_sessions()
        assert sessions == ["alpha", "beta"]

    def test_delete(self, store: FileMemoryStore) -> None:
        store.save("s1", [MemoryEntry("user", "x")])
        assert store.delete("s1") is True
        assert store.load("s1") == []
        assert store.delete("s1") is False

    def test_clear(self, store: FileMemoryStore) -> None:
        store.save("a", [MemoryEntry("user", "1")])
        store.save("b", [MemoryEntry("user", "2")])
        store.clear()
        assert store.list_sessions() == []

    def test_overwrite(self, store: FileMemoryStore) -> None:
        store.save("s1", [MemoryEntry("user", "first")])
        store.save("s1", [MemoryEntry("user", "second")])
        loaded = store.load("s1")
        assert len(loaded) == 1
        assert loaded[0].content == "second"

    def test_path_traversal_rejected(self, store: FileMemoryStore) -> None:
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.save("../evil", [MemoryEntry("user", "bad")])

    def test_slash_rejected(self, store: FileMemoryStore) -> None:
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.save("a/b", [MemoryEntry("user", "bad")])

    def test_malformed_line_skipped(self, store: FileMemoryStore, mem_dir: Path) -> None:
        file_path = mem_dir / "bad.jsonl"
        file_path.write_text('{"role":"user","content":"ok"}\nNOT JSON\n', encoding="utf-8")
        loaded = store.load("bad")
        assert len(loaded) == 1
        assert loaded[0].content == "ok"


# ── KeepInMind ────────────────────────────────────────────────────────────────


class TestKeepInMind:
    """Tests for the KeepInMind façade."""

    def test_recall_empty(self, kim: KeepInMind) -> None:
        assert kim.recall("new-session") == []

    def test_append_and_recall(self, kim: KeepInMind) -> None:
        kim.append("s1", "user", "hello")
        kim.append("s1", "assistant", "hi", agent_name="bot")
        entries = kim.recall("s1")
        assert len(entries) == 2
        assert entries[0].role == "user"
        assert entries[1].agent_name == "bot"

    def test_recall_messages(self, kim: KeepInMind) -> None:
        kim.append("s1", "user", "hello")
        kim.append("s1", "assistant", "world")
        msgs = kim.recall_messages("s1")
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]

    def test_max_turns(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir, max_turns=2)
        kim.append("s1", "user", "a")
        kim.append("s1", "assistant", "b")
        kim.append("s1", "user", "c")
        kim.append("s1", "assistant", "d")
        recalled = kim.recall("s1")
        assert len(recalled) == 2
        assert recalled[0].content == "c"
        assert recalled[1].content == "d"

    def test_entries_returns_full(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir, max_turns=1)
        kim.append("s1", "user", "a")
        kim.append("s1", "assistant", "b")
        assert len(kim.entries("s1")) == 2
        assert len(kim.recall("s1")) == 1

    def test_commit_overwrites(self, kim: KeepInMind) -> None:
        kim.append("s1", "user", "old")
        kim.commit("s1", [MemoryEntry("user", "new")])
        assert len(kim.entries("s1")) == 1
        assert kim.entries("s1")[0].content == "new"

    def test_forget(self, kim: KeepInMind) -> None:
        kim.append("s1", "user", "x")
        assert kim.forget("s1") is True
        assert kim.recall("s1") == []
        assert kim.forget("s1") is False

    def test_forget_all(self, kim: KeepInMind) -> None:
        kim.append("s1", "user", "a")
        kim.append("s2", "user", "b")
        kim.forget_all()
        assert kim.sessions() == []

    def test_sessions(self, kim: KeepInMind) -> None:
        kim.append("beta", "user", "x")
        kim.append("alpha", "user", "y")
        assert kim.sessions() == ["alpha", "beta"]

    def test_custom_store(self, store: FileMemoryStore) -> None:
        kim = KeepInMind(store=store)
        kim.append("s1", "user", "test")
        assert kim.recall("s1")[0].content == "test"

    def test_no_path_no_store_raises(self) -> None:
        with pytest.raises(ValueError, match="path.*store"):
            KeepInMind()

    def test_repr(self, kim: KeepInMind) -> None:
        r = repr(kim)
        assert "FileMemoryStore" in r
        assert "max_turns" in r


# ── Session integration ───────────────────────────────────────────────────────


class TestSessionMemory:
    """Tests for Session ↔ KeepInMind integration."""

    def test_session_without_memory(self) -> None:
        session = Session()
        session.add_event(Event(EventType.USER_MESSAGE, "user", "hi"))
        assert len(session) == 1

    def test_session_auto_commit(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir)
        session = Session(session_id="test-sess", memory=kim)
        session.add_event(Event(EventType.USER_MESSAGE, "user", "hello"))
        session.add_event(Event(EventType.AGENT_MESSAGE, "bot", "world"))
        # Tool calls should NOT be committed
        session.add_event(Event(EventType.TOOL_CALL, "bot", "calling tool"))

        stored = kim.entries("test-sess")
        assert len(stored) == 2
        assert stored[0].role == "user"
        assert stored[1].role == "assistant"

    def test_session_auto_load(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir)
        kim.append("persist", "user", "remember me")
        kim.append("persist", "assistant", "I will")

        session = Session(session_id="persist", memory=kim)
        assert len(session) == 2
        msgs = session.get_messages()
        assert msgs[0] == {"role": "user", "content": "remember me"}
        assert msgs[1] == {"role": "assistant", "content": "I will"}

    def test_session_persist_and_restore(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir)

        # First session
        s1 = Session(session_id="conv1", memory=kim)
        s1.add_event(Event(EventType.USER_MESSAGE, "user", "My name is Ada"))
        s1.add_event(Event(EventType.AGENT_MESSAGE, "bot", "Hello Ada!"))

        # Second session — same ID → loads history
        s2 = Session(session_id="conv1", memory=kim)
        assert len(s2) == 2
        s2.add_event(Event(EventType.USER_MESSAGE, "user", "What is my name?"))
        assert len(s2) == 3

        # Memory has all 3
        assert len(kim.entries("conv1")) == 3

    def test_max_turns_on_load(self, mem_dir: Path) -> None:
        kim = KeepInMind(path=mem_dir, max_turns=1)
        kim.append("s1", "user", "old")
        kim.append("s1", "assistant", "ancient")
        kim.append("s1", "user", "recent")

        session = Session(session_id="s1", memory=kim)
        # Only the last turn is loaded due to max_turns=1
        assert len(session) == 1
        assert session.events[0].content == "recent"
