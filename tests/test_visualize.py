"""Tests for nono.visualize — ASCII visualization of Workflows and Agent trees.

Covers:
- draw_workflow: linear, branched, auto-linear, empty
- draw_agent: single, with tools, sequential, parallel, loop, nested
- draw(): auto-detection and error handling
- Convenience methods: Workflow.draw(), BaseAgent.draw()

Run:
    python tests/test_visualize.py
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Iterator, AsyncIterator
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows so emoji / box-drawing chars don't crash
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Path setup — allow running as ``python tests/test_visualize.py``
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

from nono.agent.base import BaseAgent, Event, EventType, InvocationContext
from nono.agent.tool import FunctionTool
from nono.agent.workflow_agents import (
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)
from nono.workflows import Workflow
from nono.visualize import draw, draw_agent, draw_workflow

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


# ── Mock Agent ───────────────────────────────────────────────────────────


class MockAgent(BaseAgent):
    """Minimal concrete BaseAgent for testing."""

    def __init__(self, **kw: Any):
        super().__init__(**kw)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "ok")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "ok")


# ── Tool helpers ─────────────────────────────────────────────────────────


def _add(a: int, b: int) -> str:
    return str(a + b)


def _multiply(a: int, b: int) -> str:
    return str(a * b)


_tool_add = FunctionTool(_add, name="add", description="Add numbers")
_tool_mul = FunctionTool(_multiply, name="multiply", description="Multiply")


# ── Workflow drawing tests ───────────────────────────────────────────────

def test_draw_workflow_empty():
    """Empty workflow returns sentinel text."""
    flow = Workflow("empty")
    result = draw_workflow(flow)
    _assert(result == "(empty workflow)", "draw_workflow empty")


def test_draw_workflow_single_step():
    """Single step draws one tree node."""
    flow = Workflow("single")
    flow.step("alpha", lambda s: s)
    result = draw_workflow(flow)
    _assert("alpha" in result, "draw_workflow single — contains step name")
    _assert("○" in result, "draw_workflow single — contains step icon")
    _assert("└──" in result, "draw_workflow single — contains tree connector")
    _assert("📋" in result, "draw_workflow single — contains workflow icon")


def test_draw_workflow_linear():
    """Linear pipeline with connect() draws tree nodes."""
    flow = Workflow("linear")
    flow.step("fetch", lambda s: s)
    flow.step("process", lambda s: s)
    flow.step("store", lambda s: s)
    flow.connect("fetch", "process")
    flow.connect("process", "store")
    result = draw_workflow(flow)

    _assert("fetch" in result, "draw_workflow linear — contains fetch")
    _assert("process" in result, "draw_workflow linear — contains process")
    _assert("store" in result, "draw_workflow linear — contains store")
    _assert("├──" in result, "draw_workflow linear — has tree connectors")
    _assert("📋 linear" in result, "draw_workflow linear — title")


def test_draw_workflow_no_title():
    """Workflow drawn without title when title=False."""
    flow = Workflow("notitled")
    flow.step("a", lambda s: s)
    result = draw_workflow(flow, title=False)
    _assert("Workflow" not in result, "draw_workflow title=False")


def test_draw_workflow_auto_linear():
    """Steps without explicit edges render in registration order."""
    flow = Workflow("auto")
    flow.step("first", lambda s: s)
    flow.step("second", lambda s: s)
    flow.step("third", lambda s: s)
    result = draw_workflow(flow)
    lines = result.split("\n")
    # first should appear before second, second before third
    first_idx = next(i for i, l in enumerate(lines) if "first" in l)
    second_idx = next(i for i, l in enumerate(lines) if "second" in l)
    third_idx = next(i for i, l in enumerate(lines) if "third" in l)
    _assert(first_idx < second_idx < third_idx,
            "draw_workflow auto-linear — order preserved")
    _assert("├──" in result, "draw_workflow auto-linear — has tree connectors")


def test_draw_workflow_branched():
    """Branched workflow shows branch icon with nested targets."""
    flow = Workflow("branched")
    flow.step("classify", lambda s: s)
    flow.step("approve", lambda s: s)
    flow.step("reject", lambda s: s)
    flow.connect("classify", "approve")
    flow.connect("classify", "reject")
    flow.branch("classify", lambda s: "approve" if s.get("ok") else "reject")
    result = draw_workflow(flow)
    _assert("◆" in result, "draw_workflow branched — has branch symbol")
    _assert("◆ classify" in result, "draw_workflow branched — branch node")
    _assert("approve" in result, "draw_workflow branched — approve target")
    _assert("reject" in result, "draw_workflow branched — reject target")
    # Targets are nested under the branch step
    lines = result.split("\n")
    classify_idx = next(i for i, l in enumerate(lines) if "classify" in l)
    approve_idx = next(i for i, l in enumerate(lines) if "approve" in l)
    _assert(approve_idx > classify_idx, "draw_workflow branched — targets nested under branch")


def test_draw_workflow_method():
    """Workflow.draw() convenience method works."""
    flow = Workflow("method")
    flow.step("x", lambda s: s)
    result = flow.draw()
    _assert("x" in result, "Workflow.draw() method — contains step")
    _assert(isinstance(result, str), "Workflow.draw() returns str")


# ── Agent tree drawing tests ─────────────────────────────────────────────

def test_draw_agent_single():
    """Single agent with no tools or children."""
    agent = MockAgent(name="solo")
    result = draw_agent(agent)
    _assert("solo" in result, "draw_agent single — contains name")
    _assert("MockAgent" in result, "draw_agent single — contains class")
    _assert("├" not in result, "draw_agent single — no children markers")


def test_draw_agent_with_tools():
    """Agent with tools shows them as children."""
    agent = MockAgent(name="calc")
    agent.tools = [_tool_add, _tool_mul]
    result = draw_agent(agent)
    lines = result.split("\n")
    _assert("🔧 add" in result, "draw_agent tools — shows add")
    _assert("🔧 multiply" in result, "draw_agent tools — shows multiply")
    _assert("├── 🔧 add" in result, "draw_agent tools — first not last")
    _assert("└── 🔧 multiply" in result, "draw_agent tools — last uses └")


def test_draw_agent_sequential():
    """SequentialAgent shows sub-agents."""
    a = MockAgent(name="step_a")
    b = MockAgent(name="step_b")
    c = MockAgent(name="step_c")
    seq = SequentialAgent(name="pipeline", sub_agents=[a, b, c])
    result = draw_agent(seq)
    _assert("⏩" in result, "draw_agent sequential — icon")
    _assert("├── ○ step_a" in result, "draw_agent sequential — first")
    _assert("├── ○ step_b" in result, "draw_agent sequential — middle")
    _assert("└── ○ step_c" in result, "draw_agent sequential — last")


def test_draw_agent_parallel():
    """ParallelAgent shows workers count."""
    a = MockAgent(name="worker_a")
    b = MockAgent(name="worker_b")
    par = ParallelAgent(name="fanout", sub_agents=[a, b], max_workers=4)
    result = draw_agent(par)
    _assert("⏸" in result, "draw_agent parallel — icon")
    _assert("4 workers" in result, "draw_agent parallel — worker count")


def test_draw_agent_loop():
    """LoopAgent shows max iterations."""
    a = MockAgent(name="refiner")
    loop = LoopAgent(name="improve", sub_agents=[a], max_iterations=3)
    result = draw_agent(loop)
    _assert("🔁" in result, "draw_agent loop — icon")
    _assert("max 3x" in result, "draw_agent loop — iteration count")
    _assert("└── ○ refiner" in result, "draw_agent loop — child")


def test_draw_agent_nested():
    """Deeply nested hierarchy with correct tree connectors."""
    tool = FunctionTool(_add, name="add", description="Add")
    researcher = MockAgent(name="researcher")
    researcher.tools = [tool]
    writer = MockAgent(name="writer")
    reviewer = MockAgent(name="reviewer")

    par = ParallelAgent(name="gather", sub_agents=[researcher, writer],
                        max_workers=2)
    loop = LoopAgent(name="refine", sub_agents=[reviewer], max_iterations=5)
    pipeline = SequentialAgent(name="full", sub_agents=[par, loop])

    result = draw_agent(pipeline)
    lines = result.split("\n")

    # Root level
    _assert("⏩ full (SequentialAgent)" in lines[0],
            "nested — root label")

    # par is not last child → uses ├──
    _assert("├── ⏸ gather" in result, "nested — par uses ├──")

    # loop is last child → uses └──
    _assert("└── 🔁 refine" in result, "nested — loop uses └──")

    # researcher under par (not last) uses │ for continuation
    _assert("│   ├── ○ researcher" in result,
            "nested — researcher under par")

    # tool under researcher
    _assert("│   │   └── 🔧 add" in result, "nested — tool under researcher")

    # writer under par (last child of par)
    _assert("│   └── ○ writer" in result, "nested — writer under par")

    # reviewer under loop — loop uses └ so prefix is spaces
    _assert("    └── ○ reviewer" in result,
            "nested — reviewer under loop (spaces)")


def test_draw_agent_method():
    """BaseAgent.draw() convenience method works."""
    agent = MockAgent(name="test_method")
    result = agent.draw()
    _assert("test_method" in result, "BaseAgent.draw() — contains name")
    _assert(isinstance(result, str), "BaseAgent.draw() returns str")


# ── draw() auto-detect tests ────────────────────────────────────────────

def test_draw_auto_workflow():
    """draw() auto-detects Workflow."""
    flow = Workflow("autodetect")
    flow.step("x", lambda s: s)
    result = draw(flow)
    _assert("📋 autodetect" in result, "draw() auto — workflow")


def test_draw_auto_agent():
    """draw() auto-detects BaseAgent."""
    agent = MockAgent(name="autodetect")
    result = draw(agent)
    _assert("autodetect" in result, "draw() auto — agent")


def test_draw_invalid_type():
    """draw() raises TypeError for unknown types."""
    try:
        draw("not an agent or workflow")
        _ko("draw() invalid type — no error raised")
    except TypeError as exc:
        _assert("str" in str(exc), "draw() TypeError mentions type")


# ── Exports test ─────────────────────────────────────────────────────────

def test_exports():
    """Visualization functions are exported from nono package."""
    import nono
    _assert(hasattr(nono, "draw"), "nono.draw exported")
    _assert(hasattr(nono, "draw_agent"), "nono.draw_agent exported")
    _assert(hasattr(nono, "draw_workflow"), "nono.draw_workflow exported")


# ── Runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  test_visualize.py — ASCII visualization tests")
    print("=" * 60)

    # Workflow tests
    print("\n── Workflow drawing ──")
    test_draw_workflow_empty()
    test_draw_workflow_single_step()
    test_draw_workflow_linear()
    test_draw_workflow_no_title()
    test_draw_workflow_auto_linear()
    test_draw_workflow_branched()
    test_draw_workflow_method()

    # Agent tests
    print("\n── Agent tree drawing ──")
    test_draw_agent_single()
    test_draw_agent_with_tools()
    test_draw_agent_sequential()
    test_draw_agent_parallel()
    test_draw_agent_loop()
    test_draw_agent_nested()
    test_draw_agent_method()

    # Auto-detect tests
    print("\n── draw() auto-detect ──")
    test_draw_auto_workflow()
    test_draw_auto_agent()
    test_draw_invalid_type()

    # Exports
    print("\n── Exports ──")
    test_exports()

    print("\n" + "=" * 60)
    print(f"  Results: {_pass} passed, {_fail} failed")
    print("=" * 60)
    if _fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
