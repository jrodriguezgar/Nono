"""Tests for MonteCarloAgent, GraphOfThoughtsAgent, BlackboardAgent,
MixtureOfExpertsAgent, and CoVeAgent.

Verifies:
- MonteCarloAgent runs MCTS simulations with UCT scoring.
- GraphOfThoughtsAgent generates, aggregates, and scores thoughts.
- BlackboardAgent activates experts on a shared board.
- MixtureOfExpertsAgent gates and blends expert outputs.
- CoVeAgent runs the 4-phase verification pipeline.
- Both sync and async paths work correctly.
- Error handling when agents are misconfigured.

Run:
    python -m pytest tests/test_montecarlo_got_blackboard_moe_cove.py -v
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, AsyncIterator, Iterator

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.workflow_agents import (
    BlackboardAgent,
    CoVeAgent,
    GraphOfThoughtsAgent,
    MixtureOfExpertsAgent,
    MonteCarloAgent,
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


class CounterAgent(BaseAgent):
    """Returns an incrementing counter, useful for distinct outputs."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description=f"counter-{name}")
        self._counter = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self._counter += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name, f"response-{self._counter}",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        self._counter += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name, f"response-{self._counter}",
        )


class QuestionListAgent(BaseAgent):
    """Returns a list of verification questions (one per line)."""

    def __init__(self, *, name: str, questions: list[str]) -> None:
        super().__init__(name=name, description=f"qlist-{name}")
        self._questions = questions

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            "\n".join(self._questions),
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            "\n".join(self._questions),
        )


def _make_ctx(message: str = "test") -> InvocationContext:
    """Create a basic InvocationContext."""
    return InvocationContext(session=Session(), user_message=message)


# ═══════════════════════════════════════════════════════════════════════
# MonteCarloAgent
# ═══════════════════════════════════════════════════════════════════════


class TestMonteCarloAgent:
    """Test MCTS orchestration."""

    def test_basic_mcts(self) -> None:
        """Run MCTS with a simple counter agent and scoring."""
        agent = CounterAgent(name="thinker")
        scored_values: list[str] = []

        def score_fn(response: str) -> float:
            scored_values.append(response)
            return 0.8

        mcts = MonteCarloAgent(
            name="mcts",
            agent=agent,
            evaluate_fn=score_fn,
            n_simulations=5,
            max_depth=2,
            result_key="mcts_result",
        )
        ctx = _make_ctx("solve this")
        events = list(mcts._run_impl(ctx))

        msg_events = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg_events) >= 1
        assert ctx.session.state["mcts_result"]["simulations"] == 5
        assert ctx.session.state["mcts_result"]["best_score"] > 0

    def test_mcts_no_agent_error(self) -> None:
        """Error when no agent configured."""
        mcts = MonteCarloAgent(name="empty")
        ctx = _make_ctx()
        events = list(mcts._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_mcts_custom_exploration(self) -> None:
        """Custom exploration weight is respected."""
        agent = StubAgent(name="stub", response="fixed")
        mcts = MonteCarloAgent(
            name="mcts2",
            agent=agent,
            n_simulations=3,
            exploration_weight=0.0,  # pure exploitation
        )
        assert mcts.exploration_weight == 0.0
        ctx = _make_ctx()
        events = list(mcts._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1

    def test_mcts_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            agent = CounterAgent(name="async_thinker")
            mcts = MonteCarloAgent(
                name="async_mcts",
                agent=agent,
                n_simulations=3,
                result_key="async_res",
            )
            ctx = _make_ctx("async test")
            return [e async for e in mcts._run_async_impl(ctx)]

        events = asyncio.run(_run())
        assert any(e.event_type == EventType.AGENT_MESSAGE for e in events)

    def test_mcts_result_key_has_best_path(self) -> None:
        """result_key stores the best_path list."""
        agent = StubAgent(name="path_thinker", response="good answer")
        mcts = MonteCarloAgent(
            name="mcts_path",
            agent=agent,
            evaluate_fn=lambda _: 0.9,
            n_simulations=4,
            result_key="path_res",
        )
        ctx = _make_ctx()
        list(mcts._run_impl(ctx))
        assert "best_path" in ctx.session.state["path_res"]
        assert len(ctx.session.state["path_res"]["best_path"]) >= 1


# ═══════════════════════════════════════════════════════════════════════
# GraphOfThoughtsAgent
# ═══════════════════════════════════════════════════════════════════════


class TestGraphOfThoughtsAgent:
    """Test DAG-based thought orchestration."""

    def test_basic_got(self) -> None:
        """Generate + aggregate + score pipeline."""
        thinker = CounterAgent(name="gen")
        merger = StubAgent(name="merge", response="merged thought")

        got = GraphOfThoughtsAgent(
            name="got",
            agent=thinker,
            aggregate_agent=merger,
            score_fn=lambda r: 1.0 if "merged" in r else 0.3,
            n_branches=2,
            n_rounds=2,
            result_key="got_result",
        )
        ctx = _make_ctx("think about this")
        events = list(got._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1
        assert "merged" in msg[-1].content
        assert ctx.session.state["got_result"]["total_thoughts"] > 0

    def test_got_no_aggregate(self) -> None:
        """Works without an aggregate agent."""
        thinker = StubAgent(name="t", response="idea")
        got = GraphOfThoughtsAgent(
            name="got_simple",
            agent=thinker,
            n_branches=3,
            n_rounds=1,
            result_key="simple_got",
        )
        ctx = _make_ctx()
        events = list(got._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1

    def test_got_no_agent_error(self) -> None:
        """Error when no agent configured."""
        got = GraphOfThoughtsAgent(name="empty_got")
        ctx = _make_ctx()
        events = list(got._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_got_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            thinker = CounterAgent(name="async_gen")
            got = GraphOfThoughtsAgent(
                name="async_got",
                agent=thinker,
                n_branches=2,
                n_rounds=1,
                result_key="ares",
            )
            ctx = _make_ctx()
            return [e async for e in got._run_async_impl(ctx)]

        events = asyncio.run(_run())
        assert any(e.event_type == EventType.AGENT_MESSAGE for e in events)


# ═══════════════════════════════════════════════════════════════════════
# BlackboardAgent
# ═══════════════════════════════════════════════════════════════════════


class TestBlackboardAgent:
    """Test shared blackboard architecture."""

    def test_basic_blackboard(self) -> None:
        """Experts write to shared board, iterations tracked."""
        expert_a = StubAgent(name="analyst", response="analysis done")
        expert_b = StubAgent(name="designer", response="design done")

        bb = BlackboardAgent(
            name="board",
            sub_agents=[expert_a, expert_b],
            max_iterations=3,
            result_key="bb_res",
        )
        ctx = _make_ctx("build a system")
        events = list(bb._run_impl(ctx))

        state_events = [
            e for e in events if e.event_type == EventType.STATE_UPDATE
        ]
        assert len(state_events) >= 1
        assert "bb_res" in ctx.session.state

    def test_blackboard_termination(self) -> None:
        """Termination function stops early."""
        expert = StubAgent(name="e", response="done")

        bb = BlackboardAgent(
            name="term_board",
            sub_agents=[expert],
            termination_fn=lambda board: len(board) >= 1,
            max_iterations=10,
        )
        ctx = _make_ctx()
        events = list(bb._run_impl(ctx))

        # Should stop after 1 iteration since termination fires after board gets 1 entry
        state_updates = [
            e for e in events if e.event_type == EventType.STATE_UPDATE
        ]
        assert len(state_updates) <= 2  # max 1 iteration + maybe a final

    def test_blackboard_custom_controller(self) -> None:
        """Custom controller picks specific expert."""
        a = StubAgent(name="alpha", response="A")
        b = StubAgent(name="beta", response="B")

        bb = BlackboardAgent(
            name="ctrl_board",
            sub_agents=[a, b],
            controller_fn=lambda board, agents: agents[1],  # always beta
            max_iterations=2,
        )
        ctx = _make_ctx()
        events = list(bb._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        # Should have output from beta
        assert any("B" in e.content for e in msg)

    def test_blackboard_no_experts_error(self) -> None:
        """Error when no experts configured."""
        bb = BlackboardAgent(name="empty_bb")
        ctx = _make_ctx()
        events = list(bb._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_blackboard_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            e = StubAgent(name="async_expert", response="async done")
            bb = BlackboardAgent(
                name="async_bb",
                sub_agents=[e],
                max_iterations=2,
            )
            ctx = _make_ctx()
            return [ev async for ev in bb._run_async_impl(ctx)]

        events = asyncio.run(_run())
        assert any(e.event_type == EventType.AGENT_MESSAGE for e in events)


# ═══════════════════════════════════════════════════════════════════════
# MixtureOfExpertsAgent
# ═══════════════════════════════════════════════════════════════════════


class TestMixtureOfExpertsAgent:
    """Test MoE gating and weighted blending."""

    def test_basic_moe(self) -> None:
        """Uniform gating, top-2 experts activated."""
        a = StubAgent(name="math", response="42")
        b = StubAgent(name="code", response="print('hi')")
        c = StubAgent(name="write", response="essay")

        moe = MixtureOfExpertsAgent(
            name="moe",
            sub_agents=[a, b, c],
            top_k=2,
            result_key="moe_res",
        )
        ctx = _make_ctx("solve x+2=5")
        events = list(moe._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1
        assert ctx.session.state["moe_res"]["top_k"] == 2

    def test_custom_gating(self) -> None:
        """Custom gating assigns unequal weights."""
        a = StubAgent(name="fast", response="quick answer")
        b = StubAgent(name="slow", response="thorough answer")

        def gate(msg: str, agents: list[BaseAgent]) -> dict[str, float]:
            return {"fast": 0.9, "slow": 0.1}

        moe = MixtureOfExpertsAgent(
            name="moe_custom",
            sub_agents=[a, b],
            gating_fn=gate,
            top_k=1,
            result_key="gate_res",
        )
        ctx = _make_ctx()
        events = list(moe._run_impl(ctx))
        assert ctx.session.state["gate_res"]["selected"] == ["fast"]

    def test_custom_combine(self) -> None:
        """Custom combine function used."""
        a = StubAgent(name="x", response="foo")
        b = StubAgent(name="y", response="bar")

        moe = MixtureOfExpertsAgent(
            name="moe_comb",
            sub_agents=[a, b],
            top_k=2,
            combine_fn=lambda outputs: "|".join(o[1] for o in outputs),
        )
        ctx = _make_ctx()
        events = list(moe._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        combined = msg[-1].content
        assert "foo" in combined and "bar" in combined

    def test_moe_no_experts_error(self) -> None:
        """Error when no experts."""
        moe = MixtureOfExpertsAgent(name="empty_moe")
        ctx = _make_ctx()
        events = list(moe._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_moe_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            a = StubAgent(name="a_moe", response="async_a")
            b = StubAgent(name="b_moe", response="async_b")
            moe = MixtureOfExpertsAgent(
                name="async_moe",
                sub_agents=[a, b],
                top_k=2,
                result_key="async_moe_res",
            )
            ctx = _make_ctx()
            return [e async for e in moe._run_async_impl(ctx)]

        events = asyncio.run(_run())
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1


# ═══════════════════════════════════════════════════════════════════════
# CoVeAgent
# ═══════════════════════════════════════════════════════════════════════


class TestCoVeAgent:
    """Test Chain-of-Verification pipeline."""

    def test_basic_cove(self) -> None:
        """4-phase pipeline: draft, plan, verify, revise."""
        drafter = StubAgent(name="drafter", response="initial draft")
        planner = QuestionListAgent(
            name="planner",
            questions=["Is fact A correct?", "Is fact B correct?"],
        )
        verifier = StubAgent(name="verifier", response="Yes, confirmed")
        reviser = StubAgent(name="reviser", response="final verified answer")

        cove = CoVeAgent(
            name="cove",
            drafter=drafter,
            planner=planner,
            verifier=verifier,
            reviser=reviser,
            result_key="cove_res",
        )
        ctx = _make_ctx("What is X?")
        events = list(cove._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert msg[-1].content == "final verified answer"
        r = ctx.session.state["cove_res"]
        assert r["draft"] == "initial draft"
        assert len(r["questions"]) == 2
        assert len(r["verifications"]) == 2
        assert r["final"] == "final verified answer"

    def test_cove_max_questions(self) -> None:
        """max_questions limits verification count."""
        drafter = StubAgent(name="d", response="draft")
        planner = QuestionListAgent(
            name="p",
            questions=["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
        )
        verifier = StubAgent(name="v", response="ok")
        reviser = StubAgent(name="r", response="done")

        cove = CoVeAgent(
            name="cove_max",
            drafter=drafter,
            planner=planner,
            verifier=verifier,
            reviser=reviser,
            max_questions=2,
            result_key="max_res",
        )
        ctx = _make_ctx()
        list(cove._run_impl(ctx))
        assert len(ctx.session.state["max_res"]["verifications"]) == 2

    def test_cove_missing_agents_error(self) -> None:
        """Error when agents are missing."""
        cove = CoVeAgent(
            name="cove_err",
            drafter=StubAgent(name="d", response="x"),
            planner=None,
        )
        ctx = _make_ctx()
        events = list(cove._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_cove_async(self) -> None:
        """Async path works with parallel verification."""

        async def _run() -> list[Event]:
            drafter = StubAgent(name="ad", response="async draft")
            planner = QuestionListAgent(
                name="ap", questions=["Q1?", "Q2?"],
            )
            verifier = StubAgent(name="av", response="confirmed")
            reviser = StubAgent(name="ar", response="async final")

            cove = CoVeAgent(
                name="async_cove",
                drafter=drafter,
                planner=planner,
                verifier=verifier,
                reviser=reviser,
                result_key="async_cove_res",
            )
            ctx = _make_ctx()
            events = [e async for e in cove._run_async_impl(ctx)]
            # Verify via context
            assert ctx.session.state["async_cove_res"]["final"] == "async final"
            return events

        events = asyncio.run(_run())
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert msg[-1].content == "async final"
