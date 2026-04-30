"""Tests for TreeOfThoughtsAgent, PlannerAgent, SubQuestionAgent,
ContextFilterAgent, and ReflexionAgent.

Run:
    python -m pytest tests/test_tot_planner_subq_ctxfilter_reflexion.py -v
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.workflow_agents import (
    ContextFilterAgent,
    PlannerAgent,
    ReflexionAgent,
    SubQuestionAgent,
    TreeOfThoughtsAgent,
)


# ── Stub agents ───────────────────────────────────────────────────────────────


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


class EchoAgent(BaseAgent):
    """Echoes the user message."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description=f"echo-{name}")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)


class CountingAgent(BaseAgent):
    """Returns a different response each call."""

    def __init__(self, *, name: str, responses: list[str]) -> None:
        super().__init__(name=name, description=f"counting-{name}")
        self._responses = responses
        self._call = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        resp = self._responses[self._call % len(self._responses)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        resp = self._responses[self._call % len(self._responses)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)


def _ctx(message: str = "test") -> InvocationContext:
    """Build a minimal InvocationContext."""
    return InvocationContext(session=Session(), user_message=message)


# ══════════════════════════════════════════════════════════════════════════════
# TreeOfThoughtsAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_tot_sync_threshold_met_at_depth_1():
    """A high-scoring branch at depth 1 stops the tree early."""
    agent = StubAgent(name="thinker", response="CORRECT: 42")

    tot = TreeOfThoughtsAgent(
        name="tot",
        agent=agent,
        evaluate_fn=lambda r: 1.0 if "CORRECT" in r else 0.3,
        n_branches=2,
        top_k=1,
        max_depth=3,
        threshold=0.8,
    )

    events = list(tot._run_impl(_ctx("What is 6*7?")))

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert any(e.data.get("met_threshold") for e in updates)

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("CORRECT" in e.content for e in msgs)


def test_tot_sync_no_threshold_returns_best():
    """When no branch meets threshold, the best result is still returned."""
    agent = CountingAgent(
        name="thinker",
        responses=["short", "medium length", "very long detailed answer"],
    )

    tot = TreeOfThoughtsAgent(
        name="tot_exhaust",
        agent=agent,
        evaluate_fn=lambda r: float(len(r)),
        n_branches=3,
        top_k=2,
        max_depth=2,
        threshold=999.0,  # never met
    )

    ctx = _ctx()
    events = list(tot._run_impl(ctx))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_tot_sync_result_key():
    """result_key stores tree info."""
    agent = StubAgent(name="thinker", response="ok")

    tot = TreeOfThoughtsAgent(
        name="tot_key",
        agent=agent,
        evaluate_fn=lambda r: 0.5,
        n_branches=2,
        top_k=1,
        max_depth=2,
        threshold=0.9,
        result_key="tree_info",
    )

    ctx = _ctx()
    list(tot._run_impl(ctx))

    info = ctx.session.state["tree_info"]
    assert "depth" in info
    assert "score" in info
    assert "path" in info


def test_tot_sync_state_updates_per_depth():
    """STATE_UPDATE events are emitted per depth level."""
    agent = StubAgent(name="thinker", response="thought")

    tot = TreeOfThoughtsAgent(
        name="tot_updates",
        agent=agent,
        evaluate_fn=lambda r: 0.5,
        n_branches=2,
        top_k=1,
        max_depth=3,
        threshold=0.9,
    )

    events = list(tot._run_impl(_ctx()))

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert len(updates) >= 2  # At least depth 1 and 2


def test_tot_async_threshold_met():
    """Async path: threshold met at depth 1."""
    agent = StubAgent(name="thinker", response="CORRECT: yes")

    tot = TreeOfThoughtsAgent(
        name="async_tot",
        agent=agent,
        evaluate_fn=lambda r: 1.0 if "CORRECT" in r else 0.1,
        n_branches=2,
        top_k=1,
        max_depth=3,
        threshold=0.8,
    )

    async def _run():
        return [event async for event in tot._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert any(e.data.get("met_threshold") for e in updates)


# ══════════════════════════════════════════════════════════════════════════════
# PlannerAgent tests (mock LLM calls)
# ══════════════════════════════════════════════════════════════════════════════


def _mock_planner_service(plan_json: str, synthesis: str = "Final answer"):
    """Create a mock service that returns plan then synthesis."""
    call_count = 0

    def generate_completion(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return plan_json
        return synthesis

    svc = MagicMock()
    svc.generate_completion = generate_completion
    return svc


def test_planner_sync_sequential_plan():
    """PlannerAgent executes a sequential plan."""
    researcher = StubAgent(name="researcher", response="Found: X")
    writer = StubAgent(name="writer", response="Article about X")

    plan = json.dumps([
        {"step": 1, "agent": "researcher", "task": "Research X", "depends_on": []},
        {"step": 2, "agent": "writer", "task": "Write about X", "depends_on": [1]},
    ])

    planner = PlannerAgent(
        name="pm",
        sub_agents=[researcher, writer],
        result_key="plan_info",
    )

    svc = _mock_planner_service(plan, "Synthesised article about X")
    planner._service = svc

    ctx = _ctx("Write about X")
    events = list(planner._run_impl(ctx))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Synthesised" in e.content for e in msgs)

    info = ctx.session.state["plan_info"]
    assert len(info["plan"]) == 2


def test_planner_sync_parallel_steps():
    """Steps without dependencies execute in parallel."""
    a = StubAgent(name="a", response="A result")
    b = StubAgent(name="b", response="B result")

    plan = json.dumps([
        {"step": 1, "agent": "a", "task": "Do A", "depends_on": []},
        {"step": 2, "agent": "b", "task": "Do B", "depends_on": []},
    ])

    planner = PlannerAgent(name="pm", sub_agents=[a, b])
    planner._service = _mock_planner_service(plan, "Combined A+B")

    events = list(planner._run_impl(_ctx()))

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert any("Completed" in e.content for e in updates)


def test_planner_sync_empty_plan():
    """Invalid plan yields ERROR."""
    planner = PlannerAgent(
        name="pm",
        sub_agents=[StubAgent(name="x", response="ok")],
    )
    planner._service = _mock_planner_service("invalid json!!!")

    events = list(planner._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1
    assert "Failed to generate" in errors[0].content


def test_planner_sync_no_agents():
    """No sub-agents yields ERROR."""
    planner = PlannerAgent(name="pm")
    events = list(planner._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1


def test_planner_async_executes():
    """Async path: planner works end-to-end."""
    agent = StubAgent(name="worker", response="done")

    plan = json.dumps([
        {"step": 1, "agent": "worker", "task": "Do work", "depends_on": []},
    ])

    planner = PlannerAgent(name="pm", sub_agents=[agent], result_key="info")
    planner._service = _mock_planner_service(plan, "Async synthesis")

    ctx = _ctx()

    async def _run():
        return [event async for event in planner._run_async_impl(ctx)]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Async synthesis" in e.content for e in msgs)


# ══════════════════════════════════════════════════════════════════════════════
# SubQuestionAgent tests (mock LLM calls)
# ══════════════════════════════════════════════════════════════════════════════


def _mock_subq_service(sub_questions_json: str, synthesis: str = "Final"):
    """Create a mock service for SubQuestionAgent."""
    call_count = 0

    def generate_completion(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return sub_questions_json
        return synthesis

    svc = MagicMock()
    svc.generate_completion = generate_completion
    return svc


def test_subq_sync_decomposes_and_answers():
    """SubQuestionAgent decomposes, dispatches, and synthesises."""
    market = StubAgent(name="market", response="Market is growing")
    tech = StubAgent(name="tech", response="AI is advancing")

    sub_qs = json.dumps([
        {"question": "What is the market trend?", "agent": "market"},
        {"question": "What tech advances?", "agent": "tech"},
    ])

    sqd = SubQuestionAgent(
        name="analyst",
        sub_agents=[market, tech],
        result_key="analysis",
    )
    sqd._service = _mock_subq_service(sub_qs, "Market grows with AI advances")

    ctx = _ctx("How does AI affect the market?")
    events = list(sqd._run_impl(ctx))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Market grows" in e.content for e in msgs)

    info = ctx.session.state["analysis"]
    assert len(info["sub_questions"]) == 2


def test_subq_sync_bad_decomposition():
    """Invalid decomposition yields ERROR."""
    sqd = SubQuestionAgent(
        name="analyst",
        sub_agents=[StubAgent(name="x", response="ok")],
    )
    sqd._service = _mock_subq_service("not json!")

    events = list(sqd._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1


def test_subq_sync_no_agents():
    """No sub-agents yields ERROR."""
    sqd = SubQuestionAgent(name="analyst")
    events = list(sqd._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1


def test_subq_async_decomposes():
    """Async path works end-to-end."""
    agent = StubAgent(name="expert", response="Expert answer")

    sub_qs = json.dumps([
        {"question": "Q1?", "agent": "expert"},
    ])

    sqd = SubQuestionAgent(
        name="analyst",
        sub_agents=[agent],
        result_key="async_info",
    )
    sqd._service = _mock_subq_service(sub_qs, "Async synthesis")

    ctx = _ctx()

    async def _run():
        return [event async for event in sqd._run_async_impl(ctx)]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Async synthesis" in e.content for e in msgs)


# ══════════════════════════════════════════════════════════════════════════════
# ContextFilterAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_ctxfilter_sync_max_history():
    """max_history limits events each agent sees."""
    agent = EchoAgent(name="focused")

    cf = ContextFilterAgent(
        name="filter",
        sub_agents=[agent],
        max_history=2,
        result_key="stats",
    )

    ctx = _ctx("new request")
    # Add some events to session history
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "old1", "old message 1"),
    )
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "old2", "old message 2"),
    )
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "old3", "old message 3"),
    )

    events = list(cf._run_impl(ctx))

    stats = ctx.session.state["stats"]
    assert stats["filter_stats"]["focused"] == 2  # max_history=2
    assert stats["original_events"] == 3


def test_ctxfilter_sync_include_sources():
    """include_sources whitelist works."""
    agent = EchoAgent(name="reader")

    cf = ContextFilterAgent(
        name="filter",
        sub_agents=[agent],
        include_sources=["important"],
        result_key="stats",
    )

    ctx = _ctx("read this")
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "important", "key info"),
    )
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "noise", "ignore me"),
    )

    list(cf._run_impl(ctx))

    stats = ctx.session.state["stats"]
    assert stats["filter_stats"]["reader"] == 1  # only "important"


def test_ctxfilter_sync_exclude_sources():
    """exclude_sources blacklist works."""
    agent = EchoAgent(name="reader")

    cf = ContextFilterAgent(
        name="filter",
        sub_agents=[agent],
        exclude_sources=["debug"],
        result_key="stats",
    )

    ctx = _ctx("go")
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "main", "useful"),
    )
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "debug", "noisy"),
    )

    list(cf._run_impl(ctx))

    stats = ctx.session.state["stats"]
    assert stats["filter_stats"]["reader"] == 1  # "debug" excluded


def test_ctxfilter_sync_custom_filter_fn():
    """Custom filter_fn is used when provided."""
    agent = EchoAgent(name="custom")

    def my_filter(ag: Any, events: list) -> list:
        return [e for e in events if "VIP" in e.content]

    cf = ContextFilterAgent(
        name="filter",
        sub_agents=[agent],
        filter_fn=my_filter,
        result_key="stats",
    )

    ctx = _ctx("go")
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "a", "VIP message"),
    )
    ctx.session.add_event(
        Event(EventType.AGENT_MESSAGE, "b", "normal"),
    )

    list(cf._run_impl(ctx))

    stats = ctx.session.state["stats"]
    # Custom function is used, stats computed from it
    assert stats["original_events"] == 2


def test_ctxfilter_sync_parallel_mode():
    """Parallel mode runs all agents concurrently."""
    a = StubAgent(name="a", response="A out")
    b = StubAgent(name="b", response="B out")

    cf = ContextFilterAgent(
        name="par_filter",
        sub_agents=[a, b],
        mode="parallel",
        result_key="stats",
    )

    ctx = _ctx("test")
    events = list(cf._run_impl(ctx))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) == 2


def test_ctxfilter_sync_no_agents():
    """No sub-agents yields ERROR."""
    cf = ContextFilterAgent(name="empty")
    events = list(cf._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1


def test_ctxfilter_async_sequential():
    """Async sequential path works."""
    agent = StubAgent(name="a", response="async out")

    cf = ContextFilterAgent(
        name="async_filter",
        sub_agents=[agent],
        max_history=5,
    )

    async def _run():
        return [event async for event in cf._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) == 1


def test_ctxfilter_async_parallel():
    """Async parallel path works."""
    a = StubAgent(name="a", response="A")
    b = StubAgent(name="b", response="B")

    cf = ContextFilterAgent(
        name="async_par",
        sub_agents=[a, b],
        mode="parallel",
        result_key="stats",
    )

    ctx = _ctx()

    async def _run():
        return [event async for event in cf._run_async_impl(ctx)]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) == 2


# ══════════════════════════════════════════════════════════════════════════════
# ReflexionAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_reflexion_sync_passes_first_attempt():
    """Agent passes evaluation on first try."""
    agent = StubAgent(name="coder", response="def solve(): return 42")
    evaluator = StubAgent(name="reviewer", response="PASS: correct")

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
        threshold=0.8,
        result_key="reflexion_info",
    )

    ctx = _ctx("Write solve()")
    events = list(reflex._run_impl(ctx))

    info = ctx.session.state["reflexion_info"]
    assert info["accepted_attempt"] == 1
    assert info["score"] == 1.0


def test_reflexion_sync_improves_on_retry():
    """Agent fails first, evaluator provides feedback, second attempt passes."""
    agent = CountingAgent(
        name="coder",
        responses=["bad code", "good code"],
    )
    evaluator = CountingAgent(
        name="reviewer",
        responses=["FAIL: missing edge case", "PASS: all tests pass"],
    )

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
        threshold=0.8,
        max_attempts=2,
        result_key="info",
    )

    ctx = _ctx("Write code")
    events = list(reflex._run_impl(ctx))

    info = ctx.session.state["info"]
    assert info["accepted_attempt"] == 2
    assert len(info["attempts"]) == 2


def test_reflexion_sync_all_attempts_fail():
    """All attempts fail — last output is yielded."""
    agent = StubAgent(name="coder", response="bad code")
    evaluator = StubAgent(name="reviewer", response="FAIL: always wrong")

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 0.2,
        threshold=0.8,
        max_attempts=2,
        result_key="info",
    )

    ctx = _ctx("Write code")
    events = list(reflex._run_impl(ctx))

    info = ctx.session.state["info"]
    assert info["accepted_attempt"] is None
    assert len(info["attempts"]) == 2

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert any("exhausted" in str(e.data) for e in updates)


def test_reflexion_sync_memories_accumulate():
    """Memory accumulates across attempts and is stored in session.state."""
    agent = StubAgent(name="coder", response="code")
    evaluator = StubAgent(name="reviewer", response="FAIL: needs work")

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 0.3,
        threshold=0.8,
        max_attempts=3,
        memory_key="my_memory",
    )

    ctx = _ctx()
    list(reflex._run_impl(ctx))

    memories = ctx.session.state["my_memory"]
    assert len(memories) == 3  # One lesson per failed attempt


def test_reflexion_sync_previous_memories_used():
    """Pre-existing memories are included in prompts."""
    agent = EchoAgent(name="coder")
    evaluator = StubAgent(name="reviewer", response="PASS: good")

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
        threshold=0.8,
        memory_key="my_memory",
    )

    ctx = _ctx("Write code")
    ctx.session.state["my_memory"] = [
        {"attempt": "1", "feedback": "FAIL", "lesson": "Always add tests"},
    ]
    events = list(reflex._run_impl(ctx))

    # The echo agent should include the lessons in its output
    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("LESSONS" in e.content for e in msgs)


def test_reflexion_async_passes():
    """Async path: passes on first attempt."""
    agent = StubAgent(name="coder", response="good code")
    evaluator = StubAgent(name="reviewer", response="PASS")

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
        threshold=0.8,
        result_key="info",
    )

    ctx = _ctx()

    async def _run():
        return [event async for event in reflex._run_async_impl(ctx)]

    asyncio.run(_run())

    info = ctx.session.state["info"]
    assert info["accepted_attempt"] == 1


def test_reflexion_async_retries():
    """Async path: retries and eventually passes."""
    agent = CountingAgent(
        name="coder", responses=["bad", "good"],
    )
    evaluator = CountingAgent(
        name="reviewer", responses=["FAIL", "PASS"],
    )

    reflex = ReflexionAgent(
        name="learner",
        agent=agent,
        evaluator=evaluator,
        score_fn=lambda r: 1.0 if "PASS" in r else 0.2,
        threshold=0.8,
        max_attempts=2,
        result_key="info",
    )

    ctx = _ctx()

    async def _run():
        return [event async for event in reflex._run_async_impl(ctx)]

    asyncio.run(_run())

    info = ctx.session.state["info"]
    assert info["accepted_attempt"] == 2
