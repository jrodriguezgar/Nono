"""Tests for nono.sandbox.harness and nono.sandbox.durable_agent modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── SandboxSnapshot tests ────────────────────────────────────────────────────


class TestSandboxSnapshot:
    """Test SandboxSnapshot dataclass and serialisation."""

    def test_default_construction(self):
        from nono.sandbox.harness import SandboxSnapshot, CheckpointStatus

        snap = SandboxSnapshot()

        assert len(snap.snapshot_id) == 16
        assert snap.sandbox_id == ""
        assert snap.provider == ""
        assert snap.code == ""
        assert snap.code_cursor == 0
        assert snap.step_index == 0
        assert snap.total_steps == 1
        assert snap.session_state == {}
        assert snap.shared_content_keys == []
        assert snap.accumulated_stdout == ""
        assert snap.accumulated_stderr == ""
        assert snap.working_dir == "/home/user"
        assert snap.status == CheckpointStatus.CREATED
        assert snap.attempt == 0

    def test_to_dict(self):
        from nono.sandbox.harness import SandboxSnapshot, CheckpointStatus

        snap = SandboxSnapshot(
            snapshot_id="abc123",
            sandbox_id="sbx-001",
            provider="e2b",
            code="print('hi')",
            code_cursor=10,
            session_state={"key": "value"},
            status=CheckpointStatus.ACTIVE,
            attempt=2,
        )
        d = snap.to_dict()

        assert d["snapshot_id"] == "abc123"
        assert d["sandbox_id"] == "sbx-001"
        assert d["provider"] == "e2b"
        assert d["code"] == "print('hi')"
        assert d["code_cursor"] == 10
        assert d["session_state"] == {"key": "value"}
        assert d["status"] == "active"
        assert d["attempt"] == 2

    def test_from_dict(self):
        from nono.sandbox.harness import SandboxSnapshot, CheckpointStatus

        data = {
            "snapshot_id": "xyz789",
            "provider": "modal",
            "code": "x = 1",
            "status": "restored",
            "attempt": 1,
            "session_state": {"a": 1},
        }
        snap = SandboxSnapshot.from_dict(data)

        assert snap.snapshot_id == "xyz789"
        assert snap.provider == "modal"
        assert snap.code == "x = 1"
        assert snap.status == CheckpointStatus.RESTORED
        assert snap.attempt == 1
        assert snap.session_state == {"a": 1}

    def test_from_dict_defaults(self):
        from nono.sandbox.harness import SandboxSnapshot, CheckpointStatus

        snap = SandboxSnapshot.from_dict({})

        assert snap.code == ""
        assert snap.status == CheckpointStatus.CREATED
        assert snap.working_dir == "/home/user"

    def test_roundtrip_dict(self):
        from nono.sandbox.harness import SandboxSnapshot, CheckpointStatus

        original = SandboxSnapshot(
            snapshot_id="roundtrip",
            provider="e2b",
            code="print(42)",
            session_state={"count": 42},
            accumulated_stdout="42\n",
            status=CheckpointStatus.ACTIVE,
        )
        restored = SandboxSnapshot.from_dict(original.to_dict())

        assert restored.snapshot_id == original.snapshot_id
        assert restored.provider == original.provider
        assert restored.code == original.code
        assert restored.session_state == original.session_state
        assert restored.accumulated_stdout == original.accumulated_stdout
        assert restored.status == original.status

    def test_to_json(self):
        from nono.sandbox.harness import SandboxSnapshot

        snap = SandboxSnapshot(snapshot_id="json-test", code="1+1")
        raw = snap.to_json()
        parsed = json.loads(raw)

        assert parsed["snapshot_id"] == "json-test"
        assert parsed["code"] == "1+1"

    def test_from_json(self):
        from nono.sandbox.harness import SandboxSnapshot

        raw = json.dumps({"snapshot_id": "from-json", "code": "pass"})
        snap = SandboxSnapshot.from_json(raw)

        assert snap.snapshot_id == "from-json"
        assert snap.code == "pass"

    def test_roundtrip_json(self):
        from nono.sandbox.harness import SandboxSnapshot

        original = SandboxSnapshot(
            snapshot_id="json-round",
            code="import os",
            packages=["numpy"],
        )
        restored = SandboxSnapshot.from_json(original.to_json())

        assert restored.snapshot_id == original.snapshot_id
        assert restored.packages == ["numpy"]


class TestCheckpointStatus:
    """Test CheckpointStatus enum."""

    def test_values(self):
        from nono.sandbox.harness import CheckpointStatus

        assert CheckpointStatus.CREATED.value == "created"
        assert CheckpointStatus.ACTIVE.value == "active"
        assert CheckpointStatus.RESTORED.value == "restored"
        assert CheckpointStatus.EXPIRED.value == "expired"


# ── SnapshotStore tests ───────────────────────────────────────────────────────


class TestSnapshotStore:
    """Test in-memory and disk-backed snapshot stores."""

    def test_in_memory_save_load(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        store = SnapshotStore.in_memory()
        snap = SandboxSnapshot(snapshot_id="mem-1", code="x=1")

        sid = store.save(snap)
        loaded = store.load(sid)

        assert loaded is not None
        assert loaded.snapshot_id == "mem-1"
        assert loaded.code == "x=1"

    def test_in_memory_load_missing(self):
        from nono.sandbox.harness import SnapshotStore

        store = SnapshotStore.in_memory()

        assert store.load("nonexistent") is None

    def test_in_memory_list_ids(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        store = SnapshotStore.in_memory()
        store.save(SandboxSnapshot(snapshot_id="a"))
        store.save(SandboxSnapshot(snapshot_id="b"))

        ids = store.list_ids()

        assert "a" in ids
        assert "b" in ids

    def test_in_memory_delete(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        store = SnapshotStore.in_memory()
        store.save(SandboxSnapshot(snapshot_id="del-me"))

        assert store.delete("del-me") is True
        assert store.load("del-me") is None
        assert store.delete("del-me") is False

    def test_disk_save_load(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore.on_disk(tmpdir)
            snap = SandboxSnapshot(snapshot_id="disk-1", code="y=2")

            store.save(snap)

            # Verify file exists
            path = Path(tmpdir) / "disk-1.json"
            assert path.exists()

            # Load from a fresh store (only disk)
            store2 = SnapshotStore.on_disk(tmpdir)
            loaded = store2.load("disk-1")

            assert loaded is not None
            assert loaded.code == "y=2"

    def test_disk_list_ids(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore.on_disk(tmpdir)
            store.save(SandboxSnapshot(snapshot_id="d1"))
            store.save(SandboxSnapshot(snapshot_id="d2"))

            ids = store.list_ids()

            assert "d1" in ids
            assert "d2" in ids

    def test_disk_delete(self):
        from nono.sandbox.harness import SnapshotStore, SandboxSnapshot

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore.on_disk(tmpdir)
            store.save(SandboxSnapshot(snapshot_id="del-disk"))

            assert store.delete("del-disk") is True
            assert not (Path(tmpdir) / "del-disk.json").exists()
            assert store.delete("del-disk") is False


# ── HarnessEvent tests ────────────────────────────────────────────────────────


class TestHarnessEvent:
    """Test HarnessEvent data container."""

    def test_construction(self):
        from nono.sandbox.harness import HarnessEvent

        evt = HarnessEvent(
            kind="checkpoint",
            message="saved snapshot",
            snapshot_id="snap-1",
            attempt=0,
        )

        assert evt.kind == "checkpoint"
        assert evt.message == "saved snapshot"
        assert evt.snapshot_id == "snap-1"
        assert evt.attempt == 0
        assert evt.timestamp  # auto-generated


# ── HarnessRuntime tests ─────────────────────────────────────────────────────


class TestHarnessRuntime:
    """Test HarnessRuntime orchestrator."""

    def _make_client(
        self,
        *,
        results: list | None = None,
        exception: Exception | None = None,
    ) -> MagicMock:
        """Build a mock BaseSandboxClient."""
        from nono.sandbox.base import SandboxResult, SandboxStatus

        client = MagicMock()

        if exception is not None:
            client.execute.side_effect = exception
            return client

        if results is not None:
            client.execute.side_effect = results
            return client

        # Default: success
        client.execute.return_value = SandboxResult(
            status=SandboxStatus.COMPLETED,
            stdout="ok\n",
            exit_code=0,
            sandbox_id="sbx-100",
        )
        return client

    def test_success_first_attempt(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import SandboxRunConfig, SandboxStatus

        client = self._make_client()
        rt = HarnessRuntime(client=client, max_retries=2, retry_delay=0)

        result = rt.execute(code="print(1)", config=SandboxRunConfig())

        assert result.success
        assert result.stdout == "ok\n"
        assert result.metadata["attempts"] == 1
        assert client.execute.call_count == 1

    def test_retry_on_exception(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import (
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
        )

        client = self._make_client(results=[
            RuntimeError("container crashed"),
            SandboxResult(
                status=SandboxStatus.COMPLETED,
                stdout="recovered\n",
                exit_code=0,
                sandbox_id="sbx-201",
            ),
        ])
        rt = HarnessRuntime(client=client, max_retries=2, retry_delay=0)

        result = rt.execute(code="print(1)", config=SandboxRunConfig())

        assert result.success
        assert "recovered" in result.stdout
        assert client.execute.call_count == 2

    def test_retry_on_failed_exit_code(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import (
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
        )

        client = self._make_client(results=[
            SandboxResult(
                status=SandboxStatus.FAILED,
                stderr="segfault",
                exit_code=139,
                sandbox_id="sbx-300",
            ),
            SandboxResult(
                status=SandboxStatus.COMPLETED,
                stdout="fixed\n",
                exit_code=0,
                sandbox_id="sbx-301",
            ),
        ])
        rt = HarnessRuntime(client=client, max_retries=2, retry_delay=0)

        result = rt.execute(code="run()", config=SandboxRunConfig())

        assert result.success
        assert "fixed" in result.stdout
        assert client.execute.call_count == 2

    def test_exhausted_retries_exception(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import SandboxRunConfig, SandboxStatus

        client = self._make_client(exception=RuntimeError("always fails"))
        rt = HarnessRuntime(client=client, max_retries=2, retry_delay=0)

        result = rt.execute(code="boom()", config=SandboxRunConfig())

        assert not result.success
        assert result.status == SandboxStatus.FAILED
        assert client.execute.call_count == 3  # 1 initial + 2 retries

    def test_exhausted_retries_exit_code(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import (
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
        )

        client = self._make_client(results=[
            SandboxResult(status=SandboxStatus.FAILED, exit_code=1),
            SandboxResult(status=SandboxStatus.FAILED, exit_code=1),
            SandboxResult(status=SandboxStatus.FAILED, exit_code=1),
        ])
        rt = HarnessRuntime(client=client, max_retries=2, retry_delay=0)

        result = rt.execute(code="fail()", config=SandboxRunConfig())

        assert not result.success
        assert result.exit_code == 1
        assert result.metadata["attempts"] == 3

    def test_snapshot_stored(self):
        from nono.sandbox.harness import HarnessRuntime, SnapshotStore

        from nono.sandbox.base import SandboxRunConfig

        client = self._make_client()
        store = SnapshotStore.in_memory()
        rt = HarnessRuntime(
            client=client, store=store, max_retries=0, retry_delay=0
        )

        result = rt.execute(code="x=1", config=SandboxRunConfig())

        assert result.success
        ids = store.list_ids()
        assert len(ids) >= 1

        snap = store.load(ids[0])
        assert snap is not None
        assert snap.code == "x=1"

    def test_session_state_preserved(self):
        from nono.sandbox.harness import HarnessRuntime, SnapshotStore
        from nono.sandbox.base import SandboxRunConfig

        client = self._make_client()
        store = SnapshotStore.in_memory()
        rt = HarnessRuntime(
            client=client, store=store, max_retries=0, retry_delay=0
        )

        result = rt.execute(
            code="pass",
            config=SandboxRunConfig(),
            session_state={"count": 42},
        )

        assert result.success
        snap = store.load(store.list_ids()[0])
        assert snap is not None
        assert snap.session_state == {"count": 42}

    def test_harness_events_recorded(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import SandboxRunConfig

        client = self._make_client()
        rt = HarnessRuntime(client=client, max_retries=0, retry_delay=0)

        rt.execute(code="pass", config=SandboxRunConfig())

        assert len(rt.events) >= 2  # checkpoint + execute + completed
        kinds = [e.kind for e in rt.events]
        assert "checkpoint" in kinds
        assert "execute" in kinds

    def test_resume_from_snapshot(self):
        from nono.sandbox.harness import (
            HarnessRuntime,
            SandboxSnapshot,
            SnapshotStore,
        )
        from nono.sandbox.base import SandboxRunConfig

        store = SnapshotStore.in_memory()
        snap = SandboxSnapshot(
            snapshot_id="resume-1",
            code="print('resumed')",
            session_state={"step": 5},
        )
        store.save(snap)

        client = self._make_client()
        rt = HarnessRuntime(
            client=client, store=store, max_retries=0, retry_delay=0
        )

        result = rt.resume(snapshot_id="resume-1", config=SandboxRunConfig())

        assert result.success
        assert client.execute.call_count == 1

    def test_resume_missing_snapshot_raises(self):
        from nono.sandbox.harness import HarnessRuntime, SnapshotStore
        from nono.sandbox.base import SandboxRunConfig

        rt = HarnessRuntime(
            client=self._make_client(),
            store=SnapshotStore.in_memory(),
            max_retries=0,
            retry_delay=0,
        )

        with pytest.raises(ValueError, match="not found"):
            rt.resume(snapshot_id="missing", config=SandboxRunConfig())

    def test_provider_snapshot_attempted_when_enabled(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import (
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
        )

        client = self._make_client()
        client.snapshot.return_value = "psnap-001"

        rt = HarnessRuntime(client=client, max_retries=0, retry_delay=0)
        config = SandboxRunConfig(snapshot=True)

        result = rt.execute(code="pass", config=config)

        assert result.success
        client.snapshot.assert_called_once_with("sbx-100")
        assert result.metadata.get("provider_snapshot_id") == "psnap-001"

    def test_provider_snapshot_not_supported(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import SandboxRunConfig

        client = self._make_client()
        client.snapshot.side_effect = NotImplementedError("nope")

        rt = HarnessRuntime(client=client, max_retries=0, retry_delay=0)
        config = SandboxRunConfig(snapshot=True)

        result = rt.execute(code="pass", config=config)

        assert result.success  # Should not fail due to missing snapshot support

    def test_accumulated_output_across_retries(self):
        from nono.sandbox.harness import HarnessRuntime
        from nono.sandbox.base import (
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
        )

        client = self._make_client(results=[
            SandboxResult(
                status=SandboxStatus.FAILED,
                stdout="partial\n",
                stderr="err1",
                exit_code=1,
            ),
            SandboxResult(
                status=SandboxStatus.COMPLETED,
                stdout="done\n",
                exit_code=0,
            ),
        ])
        rt = HarnessRuntime(client=client, max_retries=1, retry_delay=0)

        result = rt.execute(code="work()", config=SandboxRunConfig())

        assert result.success
        assert "partial" in result.stdout
        assert "done" in result.stdout


# ── DurableSandboxAgent tests ─────────────────────────────────────────────────


class TestDurableSandboxAgent:
    """Test DurableSandboxAgent with mocked sandbox client."""

    def _make_mock_client(
        self,
        *,
        stdout: str = "output",
        exit_code: int = 0,
        status: str = "completed",
    ) -> MagicMock:
        from nono.sandbox.base import SandboxResult, SandboxStatus

        client = MagicMock()
        client.execute.return_value = SandboxResult(
            status=SandboxStatus(status),
            stdout=stdout,
            exit_code=exit_code,
            sandbox_id="sbx-durable",
        )
        return client

    def test_creation(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent

        agent = DurableSandboxAgent(name="Durable1", max_retries=5)

        assert agent.name == "Durable1"
        assert agent.max_retries == 5

    def test_success_execution(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxRunConfig
        from nono.agent.base import InvocationContext, Session

        client = self._make_mock_client(stdout="hello world")
        agent = DurableSandboxAgent(
            name="TestDurable",
            client=client,
            max_retries=1,
            retry_delay=0,
        )

        session = Session()
        ctx = InvocationContext(
            session=session,
            user_message="print('hello world')",
        )

        events = list(agent._run_impl(ctx))

        # Expect: STATE_UPDATE, harness events (STATE_UPDATEs), TOOL_RESULT, AGENT_MESSAGE
        event_types = [e.event_type.value for e in events]
        assert "state_update" in event_types
        assert "tool_result" in event_types
        assert "agent_message" in event_types

        # Agent message should contain output
        agent_msg = [e for e in events if e.event_type.value == "agent_message"][-1]
        assert "hello world" in agent_msg.content

    def test_retry_on_failure(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxResult, SandboxRunConfig, SandboxStatus
        from nono.agent.base import InvocationContext, Session

        client = MagicMock()
        client.execute.side_effect = [
            RuntimeError("boom"),
            SandboxResult(
                status=SandboxStatus.COMPLETED,
                stdout="recovered\n",
                exit_code=0,
                sandbox_id="sbx-r",
            ),
        ]

        agent = DurableSandboxAgent(
            name="RetryAgent",
            client=client,
            max_retries=2,
            retry_delay=0,
        )

        ctx = InvocationContext(
            session=Session(),
            user_message="risky_code()",
        )

        events = list(agent._run_impl(ctx))
        agent_msg = [e for e in events if e.event_type.value == "agent_message"][-1]

        assert "recovered" in agent_msg.content

    def test_all_retries_exhausted(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxRunConfig
        from nono.agent.base import InvocationContext, Session

        client = MagicMock()
        client.execute.side_effect = RuntimeError("always fails")
        client.snapshot.side_effect = NotImplementedError

        agent = DurableSandboxAgent(
            name="FailAgent",
            client=client,
            max_retries=1,
            retry_delay=0,
        )

        ctx = InvocationContext(
            session=Session(),
            user_message="crash()",
        )

        events = list(agent._run_impl(ctx))
        agent_msg = [e for e in events if e.event_type.value == "agent_message"][-1]

        assert "failed" in agent_msg.content.lower()

    def test_output_files_stored_in_shared_content(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxResult, SandboxRunConfig, SandboxStatus
        from nono.agent.base import InvocationContext, Session

        client = MagicMock()
        client.execute.return_value = SandboxResult(
            status=SandboxStatus.COMPLETED,
            stdout="done",
            exit_code=0,
            sandbox_id="sbx-files",
            output_files={"report.csv": b"a,b\n1,2"},
        )

        agent = DurableSandboxAgent(
            name="FileAgent",
            client=client,
            max_retries=0,
            retry_delay=0,
        )

        session = Session()
        ctx = InvocationContext(session=session, user_message="make_report()")

        list(agent._run_impl(ctx))

        assert session.shared_content.load("sandbox:report.csv") is not None

    def test_no_session_does_not_crash(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxRunConfig
        from nono.agent.base import InvocationContext

        client = self._make_mock_client()
        agent = DurableSandboxAgent(
            name="NoSessionAgent",
            client=client,
            max_retries=0,
            retry_delay=0,
        )

        ctx = InvocationContext(session=None, user_message="pass")

        events = list(agent._run_impl(ctx))
        assert len(events) >= 1

    def test_with_disk_backed_store(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.harness import SnapshotStore
        from nono.sandbox.base import SandboxRunConfig
        from nono.agent.base import InvocationContext, Session

        with tempfile.TemporaryDirectory() as tmpdir:
            store = SnapshotStore.on_disk(tmpdir)
            client = self._make_mock_client()

            agent = DurableSandboxAgent(
                name="DiskAgent",
                client=client,
                max_retries=0,
                retry_delay=0,
                snapshot_store=store,
            )

            ctx = InvocationContext(
                session=Session(),
                user_message="print(1)",
            )

            events = list(agent._run_impl(ctx))

            # Verify snapshots were persisted
            assert len(store.list_ids()) >= 1

    def test_harness_events_emitted(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent
        from nono.sandbox.base import SandboxRunConfig
        from nono.agent.base import InvocationContext, Session

        client = self._make_mock_client()
        agent = DurableSandboxAgent(
            name="EventAgent",
            client=client,
            max_retries=0,
            retry_delay=0,
        )

        ctx = InvocationContext(
            session=Session(),
            user_message="pass",
        )

        events = list(agent._run_impl(ctx))
        harness_events = [
            e for e in events
            if e.event_type.value == "state_update" and "[harness]" in e.content
        ]

        assert len(harness_events) >= 1


# ── Module import tests ───────────────────────────────────────────────────────


class TestHarnessModuleImports:
    """Verify public symbols are importable from the package."""

    def test_import_from_harness(self):
        from nono.sandbox.harness import (
            CheckpointStatus,
            HarnessEvent,
            HarnessRuntime,
            SandboxSnapshot,
            SnapshotStore,
        )

        assert CheckpointStatus is not None
        assert HarnessEvent is not None
        assert HarnessRuntime is not None
        assert SandboxSnapshot is not None
        assert SnapshotStore is not None

    def test_import_from_durable_agent(self):
        from nono.sandbox.durable_agent import DurableSandboxAgent

        assert DurableSandboxAgent is not None

    def test_import_from_package(self):
        from nono.sandbox import (
            CheckpointStatus,
            DurableSandboxAgent,
            HarnessEvent,
            HarnessRuntime,
            SandboxSnapshot,
            SnapshotStore,
        )

        assert DurableSandboxAgent is not None
        assert HarnessRuntime is not None
        assert SandboxSnapshot is not None
        assert SnapshotStore is not None
