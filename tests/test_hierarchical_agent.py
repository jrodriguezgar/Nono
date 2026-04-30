"""Tests for HierarchicalAgent.

Verifies:
- HierarchicalAgent delegates to departments and synthesises results.
- Multi-level hierarchy (departments with their own sub-agents) works.
- Manager re-delegates to different departments across rounds.
- Async path mirrors sync behaviour.
- Edge cases: no departments, unknown department, immediate synthesise.
- result_key stores synthesis in session state.
- Org-chart includes nested agent descriptions.

Run:
    python -m pytest tests/test_hierarchical_agent.py -v
"""

import asyncio
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import AsyncIterator, Iterator

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.workflow_agents import (
    HierarchicalAgent,
    SequentialAgent,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


class StubAgent(BaseAgent):
    """Returns a fixed response."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description=f"stub-{name}")
        self._response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


def _ctx(message: str = "test task") -> InvocationContext:
    """Build a minimal InvocationContext."""
    return InvocationContext(session=Session(), user_message=message)


# ── Sync tests ────────────────────────────────────────────────────────────────


def test_hierarchical_sync_delegates_then_synthesises():
    """Manager delegates to one department, then synthesises."""
    backend = StubAgent(name="backend_team", response="API built")
    qa = StubAgent(name="qa_team", response="Tests passed")

    manager = HierarchicalAgent(
        name="cto",
        sub_agents=[backend, qa],
        max_iterations=3,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        # Round 1: delegate to backend_team
        '{"action": "delegate", "agent": "backend_team", '
        '"message": "build API", "reason": "need implementation"}',
        # Round 2: delegate to qa_team
        '{"action": "delegate", "agent": "qa_team", '
        '"message": "test API", "reason": "need testing"}',
        # Round 3: synthesise
        '{"action": "synthesise", "reason": "all done"}',
        # Synthesis call
        "The API was built and tested successfully.",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx("build and test an API")))

    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 2
    assert transfer_events[0].data["department"] == "backend_team"
    assert transfer_events[1].data["department"] == "qa_team"

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # Sub-agent messages + final synthesis
    assert any("API built" in e.content for e in agent_msgs)
    assert any("Tests passed" in e.content for e in agent_msgs)
    assert any("successfully" in e.content for e in agent_msgs)


def test_hierarchical_sync_single_delegation():
    """Manager delegates once, then synthesises (max_iterations reached)."""
    worker = StubAgent(name="devops", response="deployed")

    manager = HierarchicalAgent(
        name="lead",
        sub_agents=[worker],
        max_iterations=2,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        # Round 1: delegate
        '{"action": "delegate", "agent": "devops", "reason": "deploy"}',
        # Round 2: synthesise
        '{"action": "synthesise", "reason": "done"}',
        # Synthesis call
        "Deployment complete.",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx("deploy")))

    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 1

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Deployment complete" in e.content for e in agent_msgs)


def test_hierarchical_sync_no_departments_yields_error():
    """HierarchicalAgent with no departments yields an error event."""
    manager = HierarchicalAgent(name="empty_cto", sub_agents=[])

    events = list(manager._run_impl(_ctx()))

    error_events = [e for e in events if e.event_type == EventType.ERROR]
    assert len(error_events) == 1
    assert "No departments" in error_events[0].content


def test_hierarchical_sync_unknown_department_falls_back():
    """Manager picks unknown department, falls back to first."""
    alpha = StubAgent(name="alpha_dept", response="alpha-result")
    beta = StubAgent(name="beta_dept", response="beta-result")

    manager = HierarchicalAgent(
        name="fallback_cto",
        sub_agents=[alpha, beta],
        max_iterations=1,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        # Picks a non-existent department
        '{"action": "delegate", "agent": "nonexistent", "reason": "oops"}',
        # Synthesis call
        "alpha-result summarised",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx()))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("alpha-result" in e.content for e in agent_msgs)


def test_hierarchical_sync_immediate_synthesise():
    """Manager synthesises immediately without delegating."""
    worker = StubAgent(name="dept", response="unused")

    manager = HierarchicalAgent(
        name="quick_cto",
        sub_agents=[worker],
        max_iterations=3,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "synthesise", "reason": "already know the answer"}',
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx()))

    # No transfers, no synthesis (no collected outputs)
    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 0


def test_hierarchical_sync_result_key():
    """result_key stores synthesis in session state."""
    worker = StubAgent(name="team", response="output")

    manager = HierarchicalAgent(
        name="keyed_cto",
        sub_agents=[worker],
        max_iterations=2,
        result_key="final_answer",
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "team", "reason": "go"}',
        '{"action": "synthesise", "reason": "done"}',
        "The final synthesis.",
    ]
    manager._service = mock_service

    ctx = _ctx("task")
    list(manager._run_impl(ctx))

    assert ctx.session.state["final_answer"] == "The final synthesis."


def test_hierarchical_sync_nested_hierarchy():
    """Departments that are themselves orchestration agents work correctly."""
    dev = StubAgent(name="developer", response="code written")
    tester = StubAgent(name="tester", response="tests pass")

    backend_team = SequentialAgent(
        name="backend_team",
        description="Backend development team",
        sub_agents=[dev, tester],
    )
    qa = StubAgent(name="qa", response="reviewed")

    cto = HierarchicalAgent(
        name="cto",
        sub_agents=[backend_team, qa],
        max_iterations=2,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "backend_team", '
        '"message": "implement feature", "reason": "need code"}',
        '{"action": "synthesise", "reason": "code is done"}',
        "Feature implemented with code and tests.",
    ]
    cto._service = mock_service

    events = list(cto._run_impl(_ctx("build feature")))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # Should see messages from both dev and tester (SequentialAgent ran both)
    assert any("code written" in e.content for e in agent_msgs)
    assert any("tests pass" in e.content for e in agent_msgs)
    # And the synthesis
    assert any("implemented" in e.content for e in agent_msgs)


def test_hierarchical_sync_parse_failure_triggers_synthesise():
    """Unparseable manager response falls back to synthesise action."""
    worker = StubAgent(name="dept", response="result")

    manager = HierarchicalAgent(
        name="broken_cto",
        sub_agents=[worker],
        max_iterations=2,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        # First: delegate correctly
        '{"action": "delegate", "agent": "dept", "reason": "go"}',
        # Second: unparseable response → falls back to synthesise
        "this is not json at all",
        # Synthesis call
        "Synthesised from dept output.",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx()))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Synthesised" in e.content for e in agent_msgs)


def test_hierarchical_sync_max_iterations_exhausted():
    """Manager keeps delegating until max_iterations, then synthesises."""
    w1 = StubAgent(name="dept_a", response="output_a")
    w2 = StubAgent(name="dept_b", response="output_b")

    manager = HierarchicalAgent(
        name="busy_cto",
        sub_agents=[w1, w2],
        max_iterations=2,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "dept_a", "reason": "first"}',
        '{"action": "delegate", "agent": "dept_b", "reason": "second"}',
        # Loop ends, synthesis follows
        "Combined output from both departments.",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx()))

    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 2
    assert transfer_events[0].data["round"] == 1
    assert transfer_events[1].data["round"] == 2


# ── Async tests ───────────────────────────────────────────────────────────────


def test_hierarchical_async_delegates_then_synthesises():
    """Async path: delegates to departments and synthesises."""
    backend = StubAgent(name="backend", response="API ready")
    frontend = StubAgent(name="frontend", response="UI ready")

    manager = HierarchicalAgent(
        name="async_cto",
        sub_agents=[backend, frontend],
        max_iterations=3,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "backend", "reason": "API first"}',
        '{"action": "delegate", "agent": "frontend", "reason": "UI next"}',
        '{"action": "synthesise", "reason": "all done"}',
        "Full stack application ready.",
    ]
    manager._service = mock_service

    async def _run():
        return [
            event async for event in manager._run_async_impl(
                _ctx("full stack"),
            )
        ]

    events = asyncio.run(_run())

    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 2

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Full stack" in e.content for e in agent_msgs)


def test_hierarchical_async_no_departments_yields_error():
    """Async path: no departments yields an error event."""
    manager = HierarchicalAgent(name="async_empty", sub_agents=[])

    async def _run():
        return [
            event async for event in manager._run_async_impl(_ctx())
        ]

    events = asyncio.run(_run())

    error_events = [e for e in events if e.event_type == EventType.ERROR]
    assert len(error_events) == 1
    assert "No departments" in error_events[0].content


def test_hierarchical_async_result_key():
    """Async path: result_key stores synthesis in session state."""
    worker = StubAgent(name="team", response="output")

    manager = HierarchicalAgent(
        name="async_keyed",
        sub_agents=[worker],
        max_iterations=2,
        result_key="async_result",
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "team", "reason": "go"}',
        '{"action": "synthesise", "reason": "done"}',
        "Async synthesis result.",
    ]
    manager._service = mock_service

    ctx = _ctx("task")

    async def _run():
        return [event async for event in manager._run_async_impl(ctx)]

    asyncio.run(_run())

    assert ctx.session.state["async_result"] == "Async synthesis result."


# ── Org-chart utility ─────────────────────────────────────────────────────────


def test_describe_org_chart_flat():
    """Org-chart for flat agents includes name and type."""
    agents = [
        StubAgent(name="writer", response=""),
        StubAgent(name="editor", response=""),
    ]
    chart = HierarchicalAgent._describe_org_chart(agents)

    assert '"writer"' in chart
    assert '"editor"' in chart
    assert "StubAgent" in chart


def test_describe_org_chart_nested():
    """Org-chart for nested agents shows hierarchy with indentation."""
    child = StubAgent(name="developer", response="")
    parent = SequentialAgent(
        name="backend",
        description="Backend team",
        sub_agents=[child],
    )
    chart = HierarchicalAgent._describe_org_chart([parent])

    assert '"backend"' in chart
    assert '"developer"' in chart
    assert "SequentialAgent" in chart
    # Child should be indented
    lines = chart.split("\n")
    backend_line = [l for l in lines if "backend" in l][0]
    dev_line = [l for l in lines if "developer" in l][0]
    assert len(dev_line) - len(dev_line.lstrip()) > len(backend_line) - len(backend_line.lstrip())


def test_hierarchical_custom_synthesis_prompt():
    """Custom synthesis_prompt is used in the synthesis call."""
    worker = StubAgent(name="dept", response="data")

    manager = HierarchicalAgent(
        name="custom_cto",
        sub_agents=[worker],
        max_iterations=2,
        synthesis_prompt="Summarise in bullet points.",
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "dept", "reason": "go"}',
        '{"action": "synthesise", "reason": "done"}',
        "- Data collected.",
    ]
    manager._service = mock_service

    events = list(manager._run_impl(_ctx()))

    # Verify synthesis prompt was passed in the synthesis call
    synthesis_call = mock_service.generate_completion.call_args_list[-1]
    messages_arg = synthesis_call.kwargs.get(
        "messages", synthesis_call.args[0] if synthesis_call.args else [],
    )
    assert any(
        "bullet points" in m.get("content", "")
        for m in messages_arg
    )


def test_hierarchical_manager_instruction():
    """manager_instruction is included in the system prompt."""
    worker = StubAgent(name="dept", response="ok")

    manager = HierarchicalAgent(
        name="instructed_cto",
        sub_agents=[worker],
        max_iterations=1,
        manager_instruction="Always delegate to dept first.",
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "dept", "reason": "as instructed"}',
        "Synthesised.",
    ]
    manager._service = mock_service

    list(manager._run_impl(_ctx()))

    # Check first LLM call includes the custom instruction
    first_call = mock_service.generate_completion.call_args_list[0]
    messages_arg = first_call.kwargs.get(
        "messages", first_call.args[0] if first_call.args else [],
    )
    system_content = messages_arg[0].get("content", "")
    assert "Always delegate to dept first" in system_content
