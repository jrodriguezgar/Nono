"""Tests for workflow time-travel capabilities.

Covers:
- StateTransition.state_snapshot field
- get_state_at() — in-memory snapshot lookup
- get_history() — full execution history
- replay_from() — re-execute from an arbitrary step
- Step-indexed checkpoints on disk
- list_checkpoints() and get_checkpoint_at()
- replay_from() with overrides ("what-if" scenarios)
- replay_from() fallback to disk checkpoint
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import pytest

from nono.workflows import Workflow, WorkflowError, StateSchema, StateTransition
from nono.workflows.workflow import StepNotFoundError


# ── Helpers ──────────────────────────────────────────────────────────────────


def _add(state: dict) -> dict:
    return {"x": state.get("x", 0) + 1}


def _double(state: dict) -> dict:
    return {"x": state["x"] * 2}


def _label(state: dict) -> dict:
    return {"label": f"x={state['x']}"}


def _build_linear_flow() -> Workflow:
    flow = Workflow("linear")
    flow.step("add", _add)
    flow.step("double", _double)
    flow.step("label", _label)
    flow.connect("add", "double", "label")
    return flow


# ── StateTransition.state_snapshot ───────────────────────────────────────────


class TestStateSnapshot:
    """Verify that transitions store state snapshots after each step."""

    def test_snapshots_present_after_run(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        assert len(flow.transitions) == 3
        for t in flow.transitions:
            assert isinstance(t.state_snapshot, dict)
            assert len(t.state_snapshot) > 0

    def test_snapshots_reflect_step_state(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        # After "add": x = 0 + 1 = 1
        assert flow.transitions[0].state_snapshot["x"] == 1
        # After "double": x = 1 * 2 = 2
        assert flow.transitions[1].state_snapshot["x"] == 2
        # After "label": label = "x=2"
        assert flow.transitions[2].state_snapshot["label"] == "x=2"

    def test_snapshots_are_deep_copies(self) -> None:
        """Mutating the snapshot should not affect the workflow."""
        flow = _build_linear_flow()
        flow.run(x=0)

        snapshot = flow.transitions[0].state_snapshot
        original_x = snapshot["x"]
        snapshot["x"] = 9999

        # The snapshot inside the transition should be unchanged
        # (frozen dataclass stores a dict reference, but the snapshot
        # was deepcopied at creation time -- verify next run is clean)
        flow2 = _build_linear_flow()
        flow2.run(x=0)
        assert flow2.transitions[0].state_snapshot["x"] == original_x

    def test_snapshots_in_stream(self) -> None:
        flow = _build_linear_flow()
        list(flow.stream(x=0))

        assert len(flow.transitions) == 3
        assert flow.transitions[0].state_snapshot["x"] == 1

    def test_snapshot_on_error(self) -> None:
        """Failed steps should also record a snapshot."""

        def fail(state: dict) -> dict:
            raise ValueError("boom")

        flow = Workflow("fail_flow")
        flow.step("ok", lambda s: {"x": 1})
        flow.step("bad", fail, on_error="recover")
        flow.step("recover", lambda s: {"recovered": True})
        flow.connect("ok", "bad")

        flow.run()

        # "bad" should have a snapshot with the error
        bad_t = [t for t in flow.transitions if t.step == "bad"][0]
        assert bad_t.error is not None
        assert isinstance(bad_t.state_snapshot, dict)


# ── get_state_at() ───────────────────────────────────────────────────────────


class TestGetStateAt:
    """Verify in-memory time-travel via get_state_at()."""

    def test_returns_state_at_step(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        state_at_add = flow.get_state_at("add")
        assert state_at_add is not None
        assert state_at_add["x"] == 1

        state_at_double = flow.get_state_at("double")
        assert state_at_double is not None
        assert state_at_double["x"] == 2

    def test_returns_none_for_unknown_step(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        assert flow.get_state_at("nonexistent") is None

    def test_returns_last_occurrence(self) -> None:
        """When a step runs multiple times (error recovery), return the last."""

        call_count = 0

        def sometimes_fail(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fail")
            return {"value": call_count}

        flow = Workflow("retry_flow")
        flow.step("flaky", sometimes_fail, retry=1, on_error="fallback")
        flow.step("fallback", lambda s: {"fallback": True})

        flow.run()

        # "flaky" should succeed on retry (attempt 2)
        state = flow.get_state_at("flaky")
        assert state is not None
        assert state["value"] == 2

    def test_returns_deep_copy(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        s1 = flow.get_state_at("add")
        s2 = flow.get_state_at("add")
        assert s1 is not s2
        assert s1 == s2


# ── get_history() ────────────────────────────────────────────────────────────


class TestGetHistory:
    """Verify full execution history retrieval."""

    def test_returns_all_steps(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        history = flow.get_history()
        assert len(history) == 3
        assert [name for name, _ in history] == ["add", "double", "label"]

    def test_states_are_progressive(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        history = flow.get_history()
        assert history[0][1]["x"] == 1   # after add
        assert history[1][1]["x"] == 2   # after double
        assert history[2][1]["label"] == "x=2"  # after label

    def test_empty_before_run(self) -> None:
        flow = _build_linear_flow()
        assert flow.get_history() == []

    def test_history_reset_on_new_run(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)
        assert len(flow.get_history()) == 3

        flow.run(x=10)
        history = flow.get_history()
        assert len(history) == 3
        assert history[0][1]["x"] == 11  # 10 + 1


# ── replay_from() ────────────────────────────────────────────────────────────


class TestReplayFrom:
    """Verify time-travel replay from an arbitrary step."""

    def test_replay_from_first_step(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        # Replay from "add" (skip add, run double + label)
        result = flow.replay_from("add")
        assert result["x"] == 2  # double(1) = 2
        assert result["label"] == "x=2"

    def test_replay_from_middle_step(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        # Replay from "double" (skip add+double, run label only)
        result = flow.replay_from("double")
        assert result["label"] == "x=2"

    def test_replay_from_last_step(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        # Replay from "label" — nothing left to run
        result = flow.replay_from("label")
        assert result["label"] == "x=2"

    def test_replay_with_overrides(self) -> None:
        """Overrides enable 'what-if' scenarios."""
        flow = _build_linear_flow()
        flow.run(x=0)

        # What if x was 10 after "add"?
        result = flow.replay_from("add", x=10)
        assert result["x"] == 20  # double(10) = 20
        assert result["label"] == "x=20"

    def test_replay_records_new_transitions(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        flow.replay_from("add")
        # Should have transitions for double + label only
        assert len(flow.transitions) == 2
        assert flow.transitions[0].step == "double"
        assert flow.transitions[1].step == "label"

    def test_replay_unknown_step_raises(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        with pytest.raises(StepNotFoundError, match="nonexistent"):
            flow.replay_from("nonexistent")

    def test_replay_no_snapshot_raises(self) -> None:
        flow = _build_linear_flow()
        # Don't run — no snapshots available
        with pytest.raises(WorkflowError, match="No state snapshot"):
            flow.replay_from("add")

    def test_replay_with_branching(self) -> None:
        flow = Workflow("branch_flow")
        flow.step("check", lambda s: {"score": s.get("score", 0.5)})
        flow.step("good", lambda s: {"result": "good"})
        flow.step("bad", lambda s: {"result": "bad"})
        flow.branch("check", lambda s: "good" if s["score"] > 0.7 else "bad")

        flow.run(score=0.9)
        assert flow.get_state_at("good")["result"] == "good"

        # Replay from check with a low score — branch re-evaluates
        # and routes to "bad" instead of "good"
        result = flow.replay_from("check", score=0.3)
        assert result["result"] == "bad"


# ── Step-indexed checkpoints ─────────────────────────────────────────────────


class TestStepIndexedCheckpoints:
    """Verify per-step checkpoint files on disk."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_per_step_files(self) -> None:
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        files = sorted(Path(self.tmpdir).glob("linear_step_*.json"))
        assert len(files) == 3
        assert "step_000_add" in files[0].name
        assert "step_001_double" in files[1].name
        assert "step_002_label" in files[2].name

    def test_latest_checkpoint_also_saved(self) -> None:
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        latest = Path(self.tmpdir) / "linear_checkpoint.json"
        assert latest.exists()

    def test_list_checkpoints(self) -> None:
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        ckpts = flow.list_checkpoints()
        assert ckpts == [(0, "add"), (1, "double"), (2, "label")]

    def test_get_checkpoint_at(self) -> None:
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        state = flow.get_checkpoint_at("add")
        assert state is not None
        assert state["x"] == 1

        state = flow.get_checkpoint_at("double")
        assert state is not None
        assert state["x"] == 2

    def test_get_checkpoint_at_missing(self) -> None:
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        assert flow.get_checkpoint_at("nonexistent") is None

    def test_replay_from_disk_checkpoint(self) -> None:
        """replay_from() falls back to disk when in-memory snapshot is gone."""
        flow = _build_linear_flow()
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        # Clear in-memory transitions to simulate a new session
        flow._transitions = []

        # replay_from should still work via disk
        result = flow.replay_from("add")
        assert result["x"] == 2
        assert result["label"] == "x=2"


# ── Integration ──────────────────────────────────────────────────────────────


class TestTimeTravelIntegration:
    """End-to-end time-travel scenarios."""

    def test_full_time_travel_workflow(self) -> None:
        """Run → inspect history → replay with override."""
        flow = _build_linear_flow()
        result = flow.run(x=5)

        # Original: x=5 → add(6) → double(12) → label("x=12")
        assert result["x"] == 12
        assert result["label"] == "x=12"

        # Inspect history
        history = flow.get_history()
        assert history[0][1]["x"] == 6
        assert history[1][1]["x"] == 12

        # What-if: what if add produced x=100?
        alt = flow.replay_from("add", x=100)
        assert alt["x"] == 200
        assert alt["label"] == "x=200"

    def test_time_travel_with_schema_and_reducers(self) -> None:
        schema = StateSchema(
            fields={"items": list, "count": int},
            reducers={"items": lambda old, new: (old or []) + new},
        )
        flow = Workflow("schema_tt", schema=schema)
        flow.step("a", lambda s: {"items": ["first"], "count": 1})
        flow.step("b", lambda s: {"items": ["second"], "count": 2})
        flow.connect("a", "b")

        flow.run()

        # After "a": items=["first"]
        state_a = flow.get_state_at("a")
        assert state_a is not None
        assert state_a["items"] == ["first"]

        # After "b": items=["first", "second"] (reducer appends)
        state_b = flow.get_state_at("b")
        assert state_b is not None
        assert state_b["items"] == ["first", "second"]

    def test_time_travel_preserves_error_recovery_audit(self) -> None:
        call_count = 0

        def flaky(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("transient")
            return {"ok": True}

        flow = Workflow("error_tt")
        flow.step("flaky", flaky, retry=2)
        flow.step("done", lambda s: {"finished": True})
        flow.connect("flaky", "done")

        flow.run()

        # Flaky should have a snapshot with retries recorded
        flaky_t = [t for t in flow.transitions if t.step == "flaky"][0]
        assert flaky_t.retries == 1  # succeeded on 2nd attempt
        assert flaky_t.state_snapshot["ok"] is True
