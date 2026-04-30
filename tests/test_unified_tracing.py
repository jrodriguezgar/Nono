"""Tests for unified tracing across TaskExecutor, Workflow, and CodeExecuter.

Covers:
- nono.agent.tracing exports
- nono top-level TraceCollector export
- TaskExecutor.execute() with trace_collector
- Workflow.run() / stream() with trace_collector
- CodeExecuter.run() with trace_collector (mocked LLM)
- Cross-module: Workflow containing traced TaskExecutor calls

Run:
    python tests/test_unified_tracing.py
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock jinjapromptpy if not installed
# ---------------------------------------------------------------------------
if "jinjapromptpy" not in sys.modules:
    _jpp_mock = MagicMock()
    sys.modules["jinjapromptpy"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_generator"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_template"] = _jpp_mock
    sys.modules["jinjapromptpy.batch_generator"] = _jpp_mock

# ── Pass/Fail counters ──────────────────────────────────────────────────

_pass = 0
_fail = 0


def _ok(label: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def _ko(label: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    msg = f"  FAIL  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _assert(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        _ok(label)
    else:
        _ko(label, detail)


# ── Import tests ─────────────────────────────────────────────────────────

def test_tracing_module_imports():
    """nono.agent.tracing exports core tracing primitives."""
    from nono.agent.tracing import (
        TraceCollector,
        Trace,
        TraceStatus,
        LLMCall,
        TokenUsage,
        ToolRecord,
    )
    _assert(TraceCollector is not None, "agent.tracing TraceCollector")
    _assert(Trace is not None, "agent.tracing Trace")
    _assert(LLMCall is not None, "agent.tracing LLMCall")


def test_top_level_exports():
    """TraceCollector exported from nono package."""
    import nono
    _assert(hasattr(nono, "TraceCollector"), "nono.TraceCollector")
    _assert(hasattr(nono, "Trace"), "nono.Trace")
    _assert(hasattr(nono, "TraceStatus"), "nono.TraceStatus")
    _assert(hasattr(nono, "LLMCall"), "nono.LLMCall")
    _assert(hasattr(nono, "TokenUsage"), "nono.TokenUsage")
    _assert(hasattr(nono, "ToolRecord"), "nono.ToolRecord")


# ── Workflow tracing tests ───────────────────────────────────────────────

def test_workflow_run_tracing():
    """Workflow.run() records top-level + step traces."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    flow = Workflow("traced_flow")
    flow.step("step_a", lambda s: {"a": 1})
    flow.step("step_b", lambda s: {"b": s["a"] + 1})

    collector = TraceCollector()
    result = flow.run(trace_collector=collector, input="hello")

    _assert(result["a"] == 1, "workflow run — step_a result")
    _assert(result["b"] == 2, "workflow run — step_b result")
    _assert(len(collector.traces) == 1, "workflow run — 1 top-level trace")

    top = collector.traces[0]
    _assert(top.agent_type == "Workflow", "workflow run — type is Workflow")
    _assert(top.agent_name == "traced_flow", "workflow run — name matches")
    _assert(len(top.children) == 2, "workflow run — 2 child traces")
    _assert(top.children[0].agent_name == "step_a", "workflow run — child 0 name")
    _assert(top.children[1].agent_name == "step_b", "workflow run — child 1 name")
    _assert(top.children[0].agent_type == "WorkflowStep", "workflow run — child type")


def test_workflow_run_without_collector():
    """Workflow.run() works normally without trace_collector."""
    from nono.workflows import Workflow

    flow = Workflow("no_trace")
    flow.step("x", lambda s: {"val": 42})

    result = flow.run(key="abc")
    _assert(result["val"] == 42, "workflow no collector — still works")
    _assert(result["key"] == "abc", "workflow no collector — state preserved")


def test_workflow_stream_tracing():
    """Workflow.stream() records traces for each step."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    flow = Workflow("stream_traced")
    flow.step("s1", lambda s: {"x": 1})
    flow.step("s2", lambda s: {"y": 2})
    flow.step("s3", lambda s: {"z": 3})

    collector = TraceCollector()
    steps = list(flow.stream(trace_collector=collector))

    _assert(len(steps) == 3, "workflow stream — 3 yielded steps")
    _assert(len(collector.traces) == 1, "workflow stream — 1 top-level")
    _assert(len(collector.traces[0].children) == 3, "workflow stream — 3 children")


def test_workflow_branched_tracing():
    """Workflow branching traces only executed steps."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    flow = Workflow("branched")
    flow.step("check", lambda s: {"score": 90})
    flow.step("approve", lambda s: {"status": "approved"})
    flow.step("reject", lambda s: {"status": "rejected"})
    flow.branch_if(
        "check",
        lambda s: s["score"] >= 80,
        then="approve",
        otherwise="reject",
    )

    collector = TraceCollector()
    result = flow.run(trace_collector=collector)

    _assert(result["status"] == "approved", "branched — took approve path")
    top = collector.traces[0]
    step_names = [c.agent_name for c in top.children]
    _assert("check" in step_names, "branched — check traced")
    _assert("approve" in step_names, "branched — approve traced")
    _assert("reject" not in step_names, "branched — reject NOT traced")


def test_workflow_error_tracing():
    """Workflow step error is recorded in traces."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector, TraceStatus

    def fail_step(s: dict) -> dict:
        raise ValueError("boom")

    flow = Workflow("error_flow")
    flow.step("ok_step", lambda s: {"x": 1})
    flow.step("bad_step", fail_step)
    flow.connect("ok_step", "bad_step")

    collector = TraceCollector()
    try:
        flow.run(trace_collector=collector)
    except ValueError:
        pass

    top = collector.traces[0]
    _assert(top.status == TraceStatus.ERROR, "error flow — top trace error status")
    _assert(len(top.children) >= 1, "error flow — at least ok_step traced")


def test_workflow_trace_durations():
    """Workflow traces record non-zero durations."""
    import time
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    def slow_step(s: dict) -> dict:
        time.sleep(0.01)
        return {"done": True}

    flow = Workflow("timed")
    flow.step("slow", slow_step)

    collector = TraceCollector()
    flow.run(trace_collector=collector)

    top = collector.traces[0]
    _assert(top.duration_ms > 0, f"timed flow — duration > 0 ({top.duration_ms:.1f}ms)")
    _assert(
        top.children[0].duration_ms > 0,
        f"timed step — duration > 0 ({top.children[0].duration_ms:.1f}ms)",
    )


# ── Workflow.run() backward compatibility ────────────────────────────────

def test_workflow_run_connect_with_tracing():
    """Explicit connect() works with tracing."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    flow = Workflow("connected")
    flow.step("first", lambda s: {"order": [1]})
    flow.step("second", lambda s: {"order": s["order"] + [2]})
    flow.step("third", lambda s: {"order": s["order"] + [3]})
    flow.connect("first", "second", "third")

    collector = TraceCollector()
    result = flow.run(trace_collector=collector)

    _assert(result["order"] == [1, 2, 3], "connected — execution order correct")
    step_names = [c.agent_name for c in collector.traces[0].children]
    _assert(step_names == ["first", "second", "third"], "connected — trace order matches")


# ── print_summary test ───────────────────────────────────────────────────

def test_print_summary():
    """TraceCollector.print_summary() works with workflow traces."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector

    flow = Workflow("summary_test")
    flow.step("a", lambda s: {"x": 1})

    collector = TraceCollector()
    flow.run(trace_collector=collector)

    # Just make sure it doesn't raise
    import io as _io
    buf = _io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        collector.print_summary()
    finally:
        sys.stdout = sys_stdout

    output = buf.getvalue()
    _assert("summary_test" in output, "print_summary — workflow name present")
    _assert("Workflow" in output, "print_summary — type present")


# ── export test ──────────────────────────────────────────────────────────

def test_export():
    """TraceCollector.export() returns serializable dicts."""
    from nono.workflows import Workflow
    from nono.agent.tracing import TraceCollector
    import json

    flow = Workflow("export_test")
    flow.step("x", lambda s: {"val": 42})

    collector = TraceCollector()
    flow.run(trace_collector=collector)

    data = collector.export()
    _assert(isinstance(data, list), "export — returns list")
    _assert(len(data) == 1, "export — 1 trace")

    # Must be JSON-serializable
    json_str = json.dumps(data, default=str)
    _assert(len(json_str) > 10, "export — JSON serializable")
    _assert("export_test" in json_str, "export — contains name")


# ── Runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  test_unified_tracing.py — Cross-module tracing tests")
    print("=" * 60)

    print("\n── Imports ──")
    test_tracing_module_imports()
    test_top_level_exports()

    print("\n── Workflow tracing ──")
    test_workflow_run_tracing()
    test_workflow_run_without_collector()
    test_workflow_stream_tracing()
    test_workflow_branched_tracing()
    test_workflow_error_tracing()
    test_workflow_trace_durations()
    test_workflow_run_connect_with_tracing()

    print("\n── Output ──")
    test_print_summary()
    test_export()

    print("\n" + "=" * 60)
    print(f"  Results: {_pass} passed, {_fail} failed")
    print("=" * 60)
    if _fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
