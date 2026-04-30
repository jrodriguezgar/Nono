"""Tests for RouterAgent — LLM-powered orchestrator with execution modes."""

import asyncio
import json
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock, patch

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.workflow_agents import RouterAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

class StubAgent(BaseAgent):
    """Minimal agent that yields a fixed response."""

    def __init__(self, *, name: str, description: str = "", response: str = "ok") -> None:
        super().__init__(name=name, description=description)
        self._response = response

    def _run_impl(self, ctx):
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(self, ctx):
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


def _make_router(sub_agents, **kwargs):
    """Build a RouterAgent with a mocked LLM service."""
    router = RouterAgent(
        name="test_router",
        sub_agents=sub_agents,
        provider="google",
        **kwargs,
    )
    router._service = MagicMock()
    return router


# ── Tests: single mode (backward-compatible) ─────────────────────────────────

def test_router_single_mode():
    """LLM picks 'writer' in single mode and router delegates to it."""
    coder = StubAgent(name="coder", description="Writes code", response="code!")
    writer = StubAgent(name="writer", description="Writes prose", response="prose!")

    router = _make_router([coder, writer])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "single", "agents": ["writer"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Write an essay")
    result = router.run(ctx)

    assert result == "prose!"
    router._service.generate_completion.assert_called_once()
    print("PASS: test_router_single_mode")


def test_router_legacy_agent_name():
    """Legacy format with agent_name (no mode) still works."""
    agent = StubAgent(name="helper", response="done")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"agent_name": "helper"}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="help")
    result = router.run(ctx)

    assert result == "done"
    print("PASS: test_router_legacy_agent_name")


def test_router_fallback_on_unknown_agent():
    """When LLM returns an unknown agent name, fallback to the first sub-agent."""
    agent_a = StubAgent(name="alpha", response="alpha_response")
    agent_b = StubAgent(name="beta", response="beta_response")

    router = _make_router([agent_a, agent_b])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "single", "agents": ["nonexistent"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="hello")
    result = router.run(ctx)

    assert result == "alpha_response"
    print("PASS: test_router_fallback_on_unknown_agent")


def test_router_fallback_on_malformed_json():
    """When LLM returns garbage, fallback to the first sub-agent."""
    agent_a = StubAgent(name="alpha", response="fallback")

    router = _make_router([agent_a])
    router._service.generate_completion.return_value = "not json at all"

    session = Session()
    ctx = InvocationContext(session=session, user_message="test")
    result = router.run(ctx)

    assert result == "fallback"
    print("PASS: test_router_fallback_on_malformed_json")


def test_router_refined_message():
    """Router passes the refined message from LLM to the sub-agent."""
    agent = StubAgent(name="helper", response="done")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "single", "agents": ["helper"], "message": "refined request"}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="original")
    router.run(ctx)

    transfer_events = [
        e for e in session.events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 1
    assert transfer_events[0].data["message"] == "refined request"
    print("PASS: test_router_refined_message")


def test_router_emits_transfer_event_with_mode():
    """Router yields an AGENT_TRANSFER event with mode info."""
    agent = StubAgent(name="worker", response="result")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "single", "agents": ["worker"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="do work")
    router.run(ctx)

    transfer = [e for e in session.events if e.event_type == EventType.AGENT_TRANSFER]
    assert len(transfer) == 1
    assert transfer[0].data["mode"] == "single"
    print("PASS: test_router_emits_transfer_event_with_mode")


def test_router_no_sub_agents():
    """Router with no sub-agents yields an ERROR event."""
    router = _make_router([])

    session = Session()
    ctx = InvocationContext(session=session, user_message="hello")
    result = router.run(ctx)

    assert result == ""
    error_events = [e for e in session.events if e.event_type == EventType.ERROR]
    assert len(error_events) == 1
    print("PASS: test_router_no_sub_agents")


def test_router_strips_markdown_fences():
    """Router handles LLM responses wrapped in ```json fences."""
    agent = StubAgent(name="dev", response="coded")

    router = _make_router([agent])
    router._service.generate_completion.return_value = (
        '```json\n{"mode": "single", "agents": ["dev"]}\n```'
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="code something")
    result = router.run(ctx)

    assert result == "coded"
    print("PASS: test_router_strips_markdown_fences")


def test_router_routing_instruction():
    """Custom routing_instruction is included in the system prompt."""
    agent = StubAgent(name="a", response="ok")
    router = _make_router([agent], routing_instruction="Prefer agent 'a' always.")
    messages = router._build_routing_messages("test")

    assert any("Prefer agent 'a' always." in m["content"] for m in messages)
    print("PASS: test_router_routing_instruction")


# ── Tests: sequential mode ────────────────────────────────────────────────────

def test_router_sequential_mode():
    """LLM picks sequential mode — agents run in order."""
    a = StubAgent(name="step1", response="first")
    b = StubAgent(name="step2", response="second")
    c = StubAgent(name="step3", response="third")

    router = _make_router([a, b, c])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "sequential", "agents": ["step1", "step2", "step3"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="pipeline task")
    result = router.run(ctx)

    # The last agent's response is the final result
    assert result == "third"
    # All three produce AGENT_MESSAGE events
    agent_msgs = [e for e in session.events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(agent_msgs) == 3
    assert [e.author for e in agent_msgs] == ["step1", "step2", "step3"]

    # Transfer event has mode=sequential
    transfer = [e for e in session.events if e.event_type == EventType.AGENT_TRANSFER]
    assert transfer[0].data["mode"] == "sequential"
    print("PASS: test_router_sequential_mode")


# ── Tests: parallel mode ──────────────────────────────────────────────────────

def test_router_parallel_mode():
    """LLM picks parallel mode — agents run concurrently."""
    a = StubAgent(name="src1", response="data1")
    b = StubAgent(name="src2", response="data2")

    router = _make_router([a, b])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "parallel", "agents": ["src1", "src2"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="gather data")
    router.run(ctx)

    agent_msgs = [e for e in session.events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(agent_msgs) == 2
    # Both agents responded (order may vary)
    authors = {e.author for e in agent_msgs}
    assert authors == {"src1", "src2"}

    transfer = [e for e in session.events if e.event_type == EventType.AGENT_TRANSFER]
    assert transfer[0].data["mode"] == "parallel"
    print("PASS: test_router_parallel_mode")


# ── Tests: loop mode ─────────────────────────────────────────────────────────

def test_router_loop_mode():
    """LLM picks loop mode — agent runs multiple iterations."""
    agent = StubAgent(name="refiner", response="refined")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "loop", "agents": ["refiner"], "max_iterations": 2}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="improve this")
    router.run(ctx)

    agent_msgs = [e for e in session.events if e.event_type == EventType.AGENT_MESSAGE]
    # LoopAgent runs the sub-agent max_iterations times
    assert len(agent_msgs) == 2

    transfer = [e for e in session.events if e.event_type == EventType.AGENT_TRANSFER]
    assert transfer[0].data["mode"] == "loop"
    print("PASS: test_router_loop_mode")


def test_router_loop_default_max_iterations():
    """Loop mode uses router's max_iterations default when not specified."""
    agent = StubAgent(name="looper", response="looped")

    router = _make_router([agent], max_iterations=2)
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "loop", "agents": ["looper"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="refine")
    router.run(ctx)

    agent_msgs = [e for e in session.events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(agent_msgs) == 2
    print("PASS: test_router_loop_default_max_iterations")


# ── Tests: async ──────────────────────────────────────────────────────────────

def test_router_async_sequential():
    """RouterAgent works with run_async in sequential mode."""
    a = StubAgent(name="a", response="first")
    b = StubAgent(name="b", response="second")

    router = _make_router([a, b])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "sequential", "agents": ["a", "b"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="async seq")
    result = asyncio.run(router.run_async(ctx))

    assert result == "second"
    print("PASS: test_router_async_sequential")


# ── Tests: edge cases ─────────────────────────────────────────────────────────

def test_router_sequential_single_agent_runs_directly():
    """Sequential mode with one agent runs it directly (no wrapper)."""
    agent = StubAgent(name="solo", response="solo_result")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "sequential", "agents": ["solo"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="test")
    result = router.run(ctx)

    assert result == "solo_result"
    print("PASS: test_router_sequential_single_agent_runs_directly")


def test_router_invalid_mode_falls_back_to_single():
    """Unknown mode string is treated as single."""
    agent = StubAgent(name="x", response="ok")

    router = _make_router([agent])
    router._service.generate_completion.return_value = json.dumps(
        {"mode": "invalid_mode", "agents": ["x"]}
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="test")
    result = router.run(ctx)

    assert result == "ok"
    print("PASS: test_router_invalid_mode_falls_back_to_single")


if __name__ == "__main__":
    test_router_single_mode()
    test_router_legacy_agent_name()
    test_router_fallback_on_unknown_agent()
    test_router_fallback_on_malformed_json()
    test_router_refined_message()
    test_router_emits_transfer_event_with_mode()
    test_router_no_sub_agents()
    test_router_strips_markdown_fences()
    test_router_routing_instruction()
    test_router_sequential_mode()
    test_router_parallel_mode()
    test_router_loop_mode()
    test_router_loop_default_max_iterations()
    test_router_async_sequential()
    test_router_sequential_single_agent_runs_directly()
    test_router_invalid_mode_falls_back_to_single()
    print()
    print("=" * 60)
    print("  All 16 RouterAgent tests PASSED")
    print("=" * 60)
