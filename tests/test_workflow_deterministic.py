"""Tests for deterministic workflow orchestration: parallel, loop, join,
checkpointing, declarative loading, and visualization.

Run:
    uv run pytest tests/test_workflow_deterministic.py -v
"""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nono.workflows import Workflow, END, WorkflowError, load_workflow
from nono.visualize import draw_workflow


# ── Helpers ─────────────────────────────────────────────────────────────────


def _double(state: dict) -> dict:
    return {"value": state.get("value", 0) * 2}


def _increment(state: dict) -> dict:
    return {"value": state.get("value", 0) + 1}


def _noop(state: dict) -> dict:
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# parallel_step
# ──────────────────────────────────────────────────────────────────────────────


class TestParallelStep:
    """parallel_step: fan-out, concurrent execution, merge."""

    def test_parallel_step_merges_results(self) -> None:
        flow = Workflow("par")
        flow.parallel_step("fetch", {
            "api": lambda s: {"api_data": "from_api"},
            "db": lambda s: {"db_data": "from_db"},
        })
        result = flow.run()

        assert result["api_data"] == "from_api"
        assert result["db_data"] == "from_db"

    def test_parallel_step_receives_state_snapshot(self) -> None:
        flow = Workflow("par_snap")
        flow.step("init", lambda s: {"x": 10})
        flow.parallel_step("read", {
            "a": lambda s: {"a_val": s.get("x", 0) + 1},
            "b": lambda s: {"b_val": s.get("x", 0) + 2},
        })
        flow.connect("init", "read")
        result = flow.run()

        assert result["a_val"] == 11
        assert result["b_val"] == 12

    def test_parallel_step_with_max_workers(self) -> None:
        flow = Workflow("par_workers")
        flow.parallel_step("compute", {
            "a": lambda s: {"a": 1},
            "b": lambda s: {"b": 2},
            "c": lambda s: {"c": 3},
        }, max_workers=1)
        result = flow.run()

        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3

    def test_parallel_step_in_graph(self) -> None:
        flow = Workflow("par_graph")
        flow.step("start", lambda s: {"query": "test"})
        flow.parallel_step("fetch", {
            "fast": lambda s: {"fast": True},
            "slow": lambda s: {"slow": True},
        })
        flow.step("merge", lambda s: {"done": s.get("fast") and s.get("slow")})
        flow.connect("start", "fetch", "merge")
        result = flow.run()

        assert result["done"] is True


# ──────────────────────────────────────────────────────────────────────────────
# loop_step
# ──────────────────────────────────────────────────────────────────────────────


class TestLoopStep:
    """loop_step: repeat while condition holds, up to max_iterations."""

    def test_loop_runs_until_condition_false(self) -> None:
        flow = Workflow("loop")
        flow.step("init", lambda s: {"counter": 0})
        flow.loop_step(
            "inc",
            lambda s: {"counter": s["counter"] + 1},
            condition=lambda s: s["counter"] < 5,
            max_iterations=20,
        )
        flow.connect("init", "inc")
        result = flow.run()

        assert result["counter"] == 5
        assert result["__loop_iterations__"] == 5

    def test_loop_respects_max_iterations(self) -> None:
        flow = Workflow("loop_max")
        flow.step("init", lambda s: {"counter": 0})
        flow.loop_step(
            "inc",
            lambda s: {"counter": s["counter"] + 1},
            condition=lambda s: True,  # never stops
            max_iterations=3,
        )
        flow.connect("init", "inc")
        result = flow.run()

        assert result["counter"] == 3
        assert result["__loop_iterations__"] == 3

    def test_loop_zero_iterations_when_condition_false(self) -> None:
        flow = Workflow("loop_zero")
        flow.loop_step(
            "noop",
            _noop,
            condition=lambda s: False,
            max_iterations=10,
        )
        result = flow.run()

        assert result["__loop_iterations__"] == 0

    def test_loop_in_pipeline(self) -> None:
        flow = Workflow("pipe_loop")
        flow.step("start", lambda s: {"quality": 0.5})
        flow.loop_step(
            "improve",
            lambda s: {"quality": min(s["quality"] + 0.2, 1.0)},
            condition=lambda s: s["quality"] < 0.9,
            max_iterations=10,
        )
        flow.step("done", lambda s: {"status": "published"})
        flow.connect("start", "improve", "done")
        result = flow.run()

        assert result["quality"] >= 0.9
        assert result["status"] == "published"


# ──────────────────────────────────────────────────────────────────────────────
# join
# ──────────────────────────────────────────────────────────────────────────────


class TestJoin:
    """join: explicit wait-for-all barrier node."""

    def test_join_passes_when_predecessors_executed(self) -> None:
        flow = Workflow("join_ok")
        flow.step("a", lambda s: {"a": 1})
        flow.step("b", lambda s: {"b": 2})
        flow.join("merge", wait_for=["a", "b"])
        flow.connect("a", "b", "merge")
        result = flow.run()

        assert result["a"] == 1
        assert result["b"] == 2

    def test_join_with_reducer(self) -> None:
        flow = Workflow("join_reduce")
        flow.step("a", lambda s: {"x": 10})
        flow.step("b", lambda s: {"y": 20})
        flow.join(
            "sum",
            wait_for=["a", "b"],
            reducer=lambda s: {"total": s.get("x", 0) + s.get("y", 0)},
        )
        flow.connect("a", "b", "sum")
        result = flow.run()

        assert result["total"] == 30

    def test_join_tracks_executed_steps(self) -> None:
        flow = Workflow("join_track")
        flow.step("a", _noop)
        flow.step("b", _noop)
        flow.join("barrier", wait_for=["a", "b"])
        flow.connect("a", "b", "barrier")
        result = flow.run()
        executed = result.get("__executed_steps__", set())

        assert "a" in executed
        assert "b" in executed
        assert "barrier" in executed


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckpointing:
    """enable_checkpoints / checkpoint / resume."""

    def test_checkpoint_saves_state(self, tmp_path) -> None:
        flow = Workflow("ckpt")
        flow.enable_checkpoints(tmp_path)
        flow.step("a", lambda s: {"x": 42})
        flow.run()

        ckpt = tmp_path / "ckpt_checkpoint.json"
        assert ckpt.exists()

        data = json.loads(ckpt.read_text(encoding="utf-8"))
        assert data["step"] == "a"
        assert data["state"]["x"] == 42

    def test_resume_from_checkpoint(self, tmp_path) -> None:
        flow = Workflow("ckpt2")
        flow.enable_checkpoints(tmp_path)
        flow.step("a", lambda s: {"x": 1})
        flow.step("b", lambda s: {"y": s.get("x", 0) + 10})
        flow.connect("a", "b")

        # First run
        flow.run()

        # Modify checkpoint to simulate partial run (only "a" done)
        ckpt = tmp_path / "ckpt2_checkpoint.json"
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        data["step"] = "a"
        data["state"] = {"x": 100}
        ckpt.write_text(json.dumps(data), encoding="utf-8")

        # Resume should pick up from after "a"
        result = flow.run(resume=True)

        assert result["y"] == 110

    def test_resume_no_checkpoint_runs_normally(self, tmp_path) -> None:
        flow = Workflow("ckpt3")
        flow.enable_checkpoints(tmp_path)
        flow.step("a", lambda s: {"x": 5})
        result = flow.run(resume=True)

        assert result["x"] == 5

    def test_resume_without_enable_returns_empty(self) -> None:
        flow = Workflow("no_ckpt")
        state, last = flow.resume()

        assert state == {}
        assert last is None


# ──────────────────────────────────────────────────────────────────────────────
# Declarative loading (JSON)
# ──────────────────────────────────────────────────────────────────────────────


class TestDeclarativeLoading:
    """load_workflow: build Workflow from JSON/YAML files."""

    def test_load_json_workflow(self, tmp_path) -> None:
        definition = {
            "name": "test_flow",
            "steps": [
                {"name": "start", "type": "passthrough"},
                {"name": "end", "type": "passthrough"},
            ],
            "edges": [["start", "end"]],
        }
        path = tmp_path / "flow.json"
        path.write_text(json.dumps(definition), encoding="utf-8")

        flow = load_workflow(path, step_registry={
            "start": lambda s: {"begun": True},
            "end": lambda s: {"finished": True},
        })
        result = flow.run()

        assert result["begun"] is True
        assert result["finished"] is True

    def test_load_json_with_branches(self, tmp_path) -> None:
        definition = {
            "name": "branched",
            "steps": [
                {"name": "check"},
                {"name": "pass"},
                {"name": "fail"},
            ],
            "edges": [["check", "pass"], ["check", "fail"]],
            "branches": [
                {
                    "from": "check",
                    "condition": "score > 0.5",
                    "then": "pass",
                    "otherwise": "fail",
                }
            ],
        }
        path = tmp_path / "branched.json"
        path.write_text(json.dumps(definition), encoding="utf-8")

        flow = load_workflow(path, step_registry={
            "check": lambda s: {"score": 0.9},
            "pass": lambda s: {"result": "passed"},
            "fail": lambda s: {"result": "failed"},
        })
        result = flow.run()

        assert result["result"] == "passed"

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_workflow("/nonexistent/path.json")

    def test_load_unsupported_extension_raises(self, tmp_path) -> None:
        path = tmp_path / "flow.xml"
        path.write_text("<workflow/>", encoding="utf-8")

        with pytest.raises(WorkflowError, match="Unsupported file extension"):
            load_workflow(path)

    def test_load_json_with_checkpoint(self, tmp_path) -> None:
        ckpt_dir = tmp_path / "ckpt"
        definition = {
            "name": "ckpt_flow",
            "steps": [{"name": "a"}],
            "checkpoint_dir": str(ckpt_dir),
        }
        path = tmp_path / "flow.json"
        path.write_text(json.dumps(definition), encoding="utf-8")

        flow = load_workflow(path, step_registry={"a": lambda s: {"done": True}})

        assert flow._checkpoint_dir is not None


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────


class TestVisualization:
    """draw_workflow renders new node types correctly."""

    def test_parallel_step_icon(self) -> None:
        flow = Workflow("vis_par")
        flow.parallel_step("fetch", {
            "a": _noop,
            "b": _noop,
        })
        output = draw_workflow(flow)

        assert "⏸" in output
        assert "parallel" in output

    def test_loop_step_icon(self) -> None:
        flow = Workflow("vis_loop")
        flow.loop_step("refine", _noop, condition=lambda s: False, max_iterations=5)
        output = draw_workflow(flow)

        assert "🔁" in output
        assert "loop max 5x" in output

    def test_join_icon(self) -> None:
        flow = Workflow("vis_join")
        flow.step("a", _noop)
        flow.step("b", _noop)
        flow.join("merge", wait_for=["a", "b"])
        flow.connect("a", "b", "merge")
        output = draw_workflow(flow)

        assert "⏩" in output
        assert "join" in output

    def test_combined_visualization(self) -> None:
        flow = Workflow("full")
        flow.step("start", _noop)
        flow.parallel_step("fetch", {"x": _noop, "y": _noop})
        flow.join("merge", wait_for=["fetch"])
        flow.loop_step("refine", _noop, condition=lambda s: False, max_iterations=3)
        flow.step("end", _noop)
        flow.connect("start", "fetch", "merge", "refine", "end")
        output = draw_workflow(flow)

        assert "📋 full" in output
        assert "⏸ fetch" in output
        assert "⏩ merge" in output
        assert "🔁 refine" in output
        assert "○ start" in output
        assert "○ end" in output


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full deterministic pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """End-to-end deterministic orchestration."""

    def test_full_pipeline_with_all_node_types(self) -> None:
        flow = Workflow("full_pipeline")

        # Start
        flow.step("init", lambda s: {"data": [1, 2, 3], "quality": 0.3})

        # Parallel fetch
        flow.parallel_step("fetch", {
            "enrich": lambda s: {"enriched": [x * 10 for x in s["data"]]},
            "validate": lambda s: {"valid": len(s["data"]) > 0},
        })

        # Join
        flow.join("sync", wait_for=["fetch"])

        # Loop to improve quality
        flow.loop_step(
            "improve",
            lambda s: {"quality": s["quality"] + 0.25},
            condition=lambda s: s["quality"] < 0.9,
            max_iterations=5,
        )

        # Branch on quality
        flow.step("publish", lambda s: {"status": "published"})
        flow.step("reject", lambda s: {"status": "rejected"})

        flow.connect("init", "fetch", "sync", "improve")
        flow.branch_if(
            "improve",
            lambda s: s["quality"] >= 0.9,
            then="publish",
            otherwise="reject",
        )

        result = flow.run()

        assert result["enriched"] == [10, 20, 30]
        assert result["valid"] is True
        assert result["quality"] >= 0.9
        assert result["status"] == "published"

    def test_full_pipeline_with_checkpointing(self, tmp_path) -> None:
        flow = Workflow("ckpt_pipe")
        flow.enable_checkpoints(tmp_path)
        flow.step("a", lambda s: {"x": 1})
        flow.parallel_step("par", {
            "p1": lambda s: {"p1": s["x"] + 1},
            "p2": lambda s: {"p2": s["x"] + 2},
        })
        flow.join("j", wait_for=["par"], reducer=lambda s: {"sum": s["p1"] + s["p2"]})
        flow.connect("a", "par", "j")
        result = flow.run()

        assert result["sum"] == 5

        ckpt = tmp_path / "ckpt_pipe_checkpoint.json"
        assert ckpt.exists()
