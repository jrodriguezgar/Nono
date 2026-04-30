"""Tests for state management fixes.

Covers:
- __executed_steps__ serialization (set ↔ list in JSON checkpoints)
- join(strict=True) raising JoinPredecessorError
- _ensure_executed_steps helper
- Execution loop consistency (replay_from uses shared _step_loop_sync)

Run:
    uv run pytest tests/test_state_fixes.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nono.workflows import (
    END,
    Workflow,
    WorkflowError,
)
from nono.workflows.workflow import (
    JoinPredecessorError,
    StateSchema,
    _ensure_executed_steps,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _add(state: dict) -> dict:
    return {"x": state.get("x", 0) + 1}


def _double(state: dict) -> dict:
    return {"x": state["x"] * 2}


def _noop(state: dict) -> dict:
    return {}


def _fail_once_then_ok(counter: dict) -> callable:
    """Factory: returns a step that fails the first call, succeeds after."""

    def _fn(state: dict) -> dict:
        counter["n"] = counter.get("n", 0) + 1

        if counter["n"] < 2:
            raise RuntimeError("transient")

        return {"recovered": True}

    return _fn


# ──────────────────────────────────────────────────────────────────────────────
# _ensure_executed_steps
# ──────────────────────────────────────────────────────────────────────────────


class TestEnsureExecutedSteps:
    """Verify the helper normalizes __executed_steps__ to a set."""

    def test_missing_key_creates_empty_set(self) -> None:
        state: dict = {}
        _ensure_executed_steps(state)
        assert state["__executed_steps__"] == set()
        assert isinstance(state["__executed_steps__"], set)

    def test_list_converted_to_set(self) -> None:
        state = {"__executed_steps__": ["a", "b", "c"]}
        _ensure_executed_steps(state)
        assert state["__executed_steps__"] == {"a", "b", "c"}
        assert isinstance(state["__executed_steps__"], set)

    def test_set_unchanged(self) -> None:
        original = {"a", "b"}
        state = {"__executed_steps__": original}
        _ensure_executed_steps(state)
        assert state["__executed_steps__"] is original

    def test_tuple_converted(self) -> None:
        state = {"__executed_steps__": ("x", "y")}
        _ensure_executed_steps(state)
        assert state["__executed_steps__"] == {"x", "y"}
        assert isinstance(state["__executed_steps__"], set)

    def test_invalid_type_replaced_with_empty_set(self) -> None:
        state = {"__executed_steps__": "not_a_collection"}
        _ensure_executed_steps(state)
        assert state["__executed_steps__"] == set()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint serialization: __executed_steps__ survives JSON round-trip
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckpointExecutedSteps:
    """Verify __executed_steps__ is properly serialized and deserialized."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def test_checkpoint_serializes_set_as_list(self) -> None:
        flow = Workflow("ckpt_test")
        flow.step("a", _add)
        flow.step("b", _double)
        flow.connect("a", "b")
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        # Read the latest checkpoint and verify __executed_steps__ is a list
        ckpt_file = Path(self.tmpdir) / "ckpt_test_checkpoint.json"
        data = json.loads(ckpt_file.read_text(encoding="utf-8"))
        raw = data["state"]["__executed_steps__"]

        assert isinstance(raw, list)
        assert set(raw) == {"a", "b"}

    def test_resume_restores_set(self) -> None:
        flow = Workflow("resume_test")
        flow.step("a", _add)
        flow.step("b", _double)
        flow.step("c", lambda s: {"label": f"x={s['x']}"})
        flow.connect("a", "b", "c")
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        state, last_step = flow.resume()

        assert isinstance(state["__executed_steps__"], set)
        assert "a" in state["__executed_steps__"]
        assert "b" in state["__executed_steps__"]

    def test_get_checkpoint_at_restores_set(self) -> None:
        flow = Workflow("getckpt_test")
        flow.step("a", _add)
        flow.step("b", _double)
        flow.connect("a", "b")
        flow.enable_checkpoints(self.tmpdir)
        flow.run(x=0)

        state = flow.get_checkpoint_at("a")

        assert state is not None
        assert isinstance(state["__executed_steps__"], set)
        assert "a" in state["__executed_steps__"]

    def test_run_with_resume_maintains_set(self) -> None:
        """run(resume=True) should have __executed_steps__ as a set."""
        flow = Workflow("run_resume")
        flow.step("a", _add)
        flow.step("b", _double)
        flow.step("c", lambda s: {"done": True})
        flow.connect("a", "b", "c")
        flow.enable_checkpoints(self.tmpdir)

        # First run stops after b
        flow.run(x=0)

        # Second run resumes — __executed_steps__ from JSON must be a set
        flow2 = Workflow("run_resume")
        flow2.step("a", _add)
        flow2.step("b", _double)
        flow2.step("c", lambda s: {"done": True})
        flow2.connect("a", "b", "c")
        flow2.enable_checkpoints(self.tmpdir)

        result = flow2.run(x=0, resume=True)
        assert isinstance(result["__executed_steps__"], set)


# ──────────────────────────────────────────────────────────────────────────────
# join(strict=True)
# ──────────────────────────────────────────────────────────────────────────────


class TestJoinStrict:
    """Verify join(strict=True) raises when predecessors are missing."""

    def test_strict_join_raises_when_predecessors_missing(self) -> None:
        flow = Workflow("strict_join")
        flow.step("a", _add)
        flow.join("merge", wait_for=["a", "b"], strict=True)
        flow.connect("a", "merge")

        with pytest.raises(JoinPredecessorError, match="predecessors not yet executed"):
            flow.run(x=0)

    def test_strict_join_passes_when_all_predecessors_executed(self) -> None:
        flow = Workflow("strict_ok")
        flow.step("a", _add)
        flow.step("b", lambda s: {"y": 10})
        flow.join("merge", wait_for=["a", "b"], strict=True)
        flow.connect("a", "b", "merge")

        result = flow.run(x=0)

        assert result["x"] == 1  # add: 0+1=1, b doesn't change x
        assert result["y"] == 10

    def test_non_strict_join_warns_but_continues(self) -> None:
        """Default strict=False: join warns but does not raise."""
        flow = Workflow("soft_join")
        flow.step("a", _add)
        flow.join("merge", wait_for=["a", "nonexistent"])
        flow.connect("a", "merge")

        # Should not raise
        result = flow.run(x=0)
        assert result["x"] == 1

    def test_strict_join_with_reducer(self) -> None:
        flow = Workflow("strict_reducer")
        flow.step("a", lambda s: {"val_a": 10})
        flow.step("b", lambda s: {"val_b": 20})
        flow.join(
            "merge",
            wait_for=["a", "b"],
            strict=True,
            reducer=lambda s: {"total": s["val_a"] + s["val_b"]},
        )
        flow.connect("a", "b", "merge")

        result = flow.run()
        assert result["total"] == 30

    def test_strict_join_in_stream(self) -> None:
        flow = Workflow("strict_stream")
        flow.step("a", _add)
        flow.join("merge", wait_for=["a", "missing"], strict=True)
        flow.connect("a", "merge")

        with pytest.raises(JoinPredecessorError):
            list(flow.stream(x=0))


# ──────────────────────────────────────────────────────────────────────────────
# replay_from uses shared _step_loop_sync (no inline retry duplication)
# ──────────────────────────────────────────────────────────────────────────────


class TestReplayFromConsistency:
    """Verify replay_from uses the shared loop (retry, callbacks, etc.)."""

    def test_replay_from_uses_retry(self) -> None:
        """replay_from should retry steps, just like run()."""
        counter: dict = {"n": 0}

        flow = Workflow("replay_retry")
        flow.step("a", _add)
        flow.step("b", _fail_once_then_ok(counter), retry=2)
        flow.connect("a", "b")
        flow.run(x=0)

        # Reset counter and replay from "a"
        counter["n"] = 0
        result = flow.replay_from("a")

        assert result["recovered"] is True

    def test_replay_from_fires_step_executing_callback(self) -> None:
        """replay_from should invoke on_step_executing, like run()."""
        log: list[str] = []

        flow = Workflow("replay_cb")
        flow.step("a", _add)
        flow.step("b", _double)
        flow.connect("a", "b")
        flow.on_step_executing(lambda name, state, attempt: log.append(name))
        flow.run(x=0)

        log.clear()
        flow.replay_from("a")

        assert "b" in log

    def test_replay_from_error_recovery(self) -> None:
        """replay_from should use on_error routing from _step_loop_sync."""
        flow = Workflow("replay_recovery")
        flow.step("a", _add)
        flow.step("b", lambda s: (_ for _ in ()).throw(RuntimeError("fail")),
                   on_error="fallback")
        flow.step("fallback", lambda s: {"rescued": True})
        flow.connect("a", "b")
        flow.run(x=0)

        result = flow.replay_from("a")
        assert result["rescued"] is True
