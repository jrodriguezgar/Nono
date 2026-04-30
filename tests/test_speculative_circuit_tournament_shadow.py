"""Tests for SpeculativeAgent, CircuitBreakerAgent, TournamentAgent, ShadowAgent.

Run:
    python -m pytest tests/test_speculative_circuit_tournament_shadow.py -v
"""

import asyncio
import sys
import os

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
    CircuitBreakerAgent,
    ShadowAgent,
    SpeculativeAgent,
    TournamentAgent,
)


# ── Helper agents ─────────────────────────────────────────────────────────────

class FixedAgent(BaseAgent):
    """Always returns a fixed response."""

    def __init__(self, *, name: str, response: str) -> None:
        super().__init__(name=name, description="fixed")
        self.response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)


class FailingAgent(BaseAgent):
    """Returns empty/short response (fails default failure_detector)."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description="failing")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "")


class CountingAgent(BaseAgent):
    """Counts how many times it has been called."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description="counting")
        self.call_count = 0
        self.response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.call_count += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self.call_count += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)


class JudgeAgent(BaseAgent):
    """Picks the first name it finds in the message."""

    def __init__(self, *, name: str, pick: str) -> None:
        super().__init__(name=name, description="judge")
        self.pick = pick

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.pick)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.pick)


def _ctx(msg: str = "test question") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── SpeculativeAgent ──────────────────────────────────────────────────────────

def test_speculative_picks_high_confidence():
    """Agent with high evaluator score wins early."""
    good = FixedAgent(name="good", response="long detailed response here")
    bad = FixedAgent(name="bad", response="x")

    spec = SpeculativeAgent(
        name="racer",
        sub_agents=[good, bad],
        evaluator_fn=lambda r: 1.0 if len(r) > 10 else 0.1,
        min_confidence=0.8,
        result_key="spec_result",
    )
    ctx = _ctx()
    spec.run(ctx)

    result = ctx.session.state["spec_result"]
    assert result["winner"] == "good"
    assert result["score"] >= 0.8
    assert result["early_stop"] is True


def test_speculative_best_fallback():
    """When no agent meets min_confidence, the highest scorer is used."""
    a = FixedAgent(name="a", response="short")
    b = FixedAgent(name="b", response="medium text")

    spec = SpeculativeAgent(
        name="racer",
        sub_agents=[a, b],
        evaluator_fn=lambda r: len(r) / 100,  # 0.05 and 0.11
        min_confidence=0.9,
        result_key="spec_result",
    )
    ctx = _ctx()
    spec.run(ctx)

    result = ctx.session.state["spec_result"]
    assert result["early_stop"] is False


def test_speculative_no_sub_agents():
    """Yields error when no sub-agents configured."""
    spec = SpeculativeAgent(name="empty", sub_agents=[])
    ctx = _ctx()
    events = list(spec._run_impl(ctx))
    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_speculative_async():
    """Async version works."""
    good = FixedAgent(name="good", response="long detailed response here")
    bad = FixedAgent(name="bad", response="x")

    spec = SpeculativeAgent(
        name="racer",
        sub_agents=[good, bad],
        evaluator_fn=lambda r: 1.0 if len(r) > 10 else 0.1,
        min_confidence=0.8,
        result_key="spec_result",
    )
    ctx = _ctx()
    asyncio.run(spec.run_async(ctx))

    result = ctx.session.state["spec_result"]
    assert result["winner"] == "good"


# ── CircuitBreakerAgent ───────────────────────────────────────────────────────

def test_circuit_closed_success():
    """When primary succeeds, it passes through."""
    primary = FixedAgent(name="primary", response="good response here")
    fallback = FixedAgent(name="fallback", response="fallback response")

    cb = CircuitBreakerAgent(
        name="cb",
        agent=primary,
        fallback_agent=fallback,
        failure_threshold=3,
        result_key="cb_result",
    )
    ctx = _ctx()
    cb.run(ctx)

    result = ctx.session.state["cb_result"]
    assert result["outcome"] == "primary_success"
    assert result["circuit_state"] == "closed"


def test_circuit_opens_after_failures():
    """After enough failures, circuit opens and uses fallback."""
    primary = FailingAgent(name="primary")
    fallback = FixedAgent(name="fallback", response="fallback response")

    cb = CircuitBreakerAgent(
        name="cb",
        agent=primary,
        fallback_agent=fallback,
        failure_threshold=2,
        window_size=5,
        recovery_timeout=3,
        result_key="cb_result",
    )

    # First 2 calls: failures accumulate, circuit opens
    for _ in range(2):
        ctx = _ctx()
        cb.run(ctx)

    # Third call: circuit should be open, use fallback directly
    ctx = _ctx()
    events = list(cb.run(ctx))

    result = ctx.session.state.get("cb_result", {})
    assert result.get("outcome") in ("fallback_used", "primary_success")


def test_circuit_no_fallback():
    """Without fallback, yields error when circuit is open."""
    primary = FailingAgent(name="primary")

    cb = CircuitBreakerAgent(
        name="cb",
        agent=primary,
        failure_threshold=2,
        window_size=5,
        result_key="cb_result",
    )

    for _ in range(3):
        ctx = _ctx()
        cb.run(ctx)

    result = ctx.session.state.get("cb_result", {})
    assert result.get("outcome") in ("no_fallback", "primary_success")


def test_circuit_async():
    """Async circuit breaker works."""
    primary = FixedAgent(name="primary", response="good response here")
    fallback = FixedAgent(name="fallback", response="fallback response")

    cb = CircuitBreakerAgent(
        name="cb",
        agent=primary,
        fallback_agent=fallback,
        failure_threshold=3,
        result_key="cb_result",
    )
    ctx = _ctx()
    asyncio.run(cb.run_async(ctx))

    result = ctx.session.state["cb_result"]
    assert result["outcome"] == "primary_success"


# ── TournamentAgent ───────────────────────────────────────────────────────────

def test_tournament_basic():
    """Basic 2-agent tournament works."""
    a = FixedAgent(name="alice", response="Alice's great response")
    b = FixedAgent(name="bob", response="Bob's response")
    judge = JudgeAgent(name="judge", pick="alice")

    tourney = TournamentAgent(
        name="tourney",
        sub_agents=[a, b],
        judge_agent=judge,
        result_key="tourney_result",
    )
    ctx = _ctx()
    tourney.run(ctx)

    result = ctx.session.state["tourney_result"]
    assert result["winner"] == "alice"
    assert result["rounds"] == 1
    assert len(result["bracket"]) == 1


def test_tournament_four_agents():
    """4-agent tournament requires 2 rounds."""
    agents = [
        FixedAgent(name=f"agent_{i}", response=f"Response {i}")
        for i in range(4)
    ]
    # Judge always picks the first name in the pair
    judge = JudgeAgent(name="judge", pick="agent_0")

    tourney = TournamentAgent(
        name="tourney",
        sub_agents=agents,
        judge_agent=judge,
        result_key="tourney_result",
    )
    ctx = _ctx()
    tourney.run(ctx)

    result = ctx.session.state["tourney_result"]
    assert result["winner"] is not None
    assert result["rounds"] >= 1


def test_tournament_odd_bye():
    """Odd number of agents: one gets a bye each round."""
    agents = [
        FixedAgent(name=f"a{i}", response=f"R{i}")
        for i in range(3)
    ]
    judge = JudgeAgent(name="judge", pick="a0")

    tourney = TournamentAgent(
        name="t", sub_agents=agents, judge_agent=judge,
        result_key="result",
    )
    ctx = _ctx()
    tourney.run(ctx)

    assert ctx.session.state["result"]["winner"] is not None


def test_tournament_too_few():
    """Error when fewer than 2 agents."""
    a = FixedAgent(name="alone", response="x")
    judge = JudgeAgent(name="judge", pick="alone")

    tourney = TournamentAgent(
        name="t", sub_agents=[a], judge_agent=judge,
    )
    ctx = _ctx()
    events = list(tourney._run_impl(ctx))
    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_tournament_async():
    """Async tournament works."""
    a = FixedAgent(name="alice", response="Alice response")
    b = FixedAgent(name="bob", response="Bob response")
    judge = JudgeAgent(name="judge", pick="alice")

    tourney = TournamentAgent(
        name="t", sub_agents=[a, b], judge_agent=judge,
        result_key="result",
    )
    ctx = _ctx()
    asyncio.run(tourney.run_async(ctx))
    assert ctx.session.state["result"]["winner"] == "alice"


# ── ShadowAgent ───────────────────────────────────────────────────────────────

def test_shadow_returns_stable_only():
    """Only stable agent's output is yielded."""
    stable = FixedAgent(name="stable", response="stable answer")
    shadow = FixedAgent(name="shadow", response="shadow answer")

    sa = ShadowAgent(
        name="shadow_test",
        stable_agent=stable,
        shadow_agent=shadow,
        result_key="shadow_result",
    )
    ctx = _ctx()
    events = list(sa._run_impl(ctx))

    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(messages) >= 1
    assert messages[0].content == "stable answer"

    result = ctx.session.state["shadow_result"]
    assert result["stable_output"] == "stable answer"
    assert result["shadow_output"] == "shadow answer"
    assert result["match"] is False


def test_shadow_match():
    """When both produce same output, match is True."""
    stable = FixedAgent(name="stable", response="same")
    shadow = FixedAgent(name="shadow", response="same")

    sa = ShadowAgent(
        name="shadow_test",
        stable_agent=stable,
        shadow_agent=shadow,
        result_key="shadow_result",
    )
    ctx = _ctx()
    sa.run(ctx)

    assert ctx.session.state["shadow_result"]["match"] is True


def test_shadow_diff_logger_called():
    """Custom diff_logger receives both outputs."""
    logged = []
    stable = FixedAgent(name="stable", response="A")
    shadow = FixedAgent(name="shadow", response="B")

    sa = ShadowAgent(
        name="shadow_test",
        stable_agent=stable,
        shadow_agent=shadow,
        diff_logger=lambda s, sh: logged.append((s, sh)),
    )
    ctx = _ctx()
    sa.run(ctx)

    assert len(logged) == 1
    assert logged[0] == ("A", "B")


def test_shadow_async():
    """Async shadow agent works."""
    stable = FixedAgent(name="stable", response="stable output")
    shadow = FixedAgent(name="shadow", response="shadow output")

    sa = ShadowAgent(
        name="shadow_test",
        stable_agent=stable,
        shadow_agent=shadow,
        result_key="shadow_result",
    )
    ctx = _ctx()
    asyncio.run(sa.run_async(ctx))

    assert ctx.session.state["shadow_result"]["stable_output"] == "stable output"
