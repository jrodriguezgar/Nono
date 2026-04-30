"""Tests for SkeletonOfThoughtAgent, LeastToMostAgent, SelfDiscoverAgent,
GeneticAlgorithmAgent, and MultiArmedBanditAgent."""

from __future__ import annotations

import pytest

from nono.agent import (
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.base import BaseAgent
from nono.agent.workflow_agents import (
    GeneticAlgorithmAgent,
    LeastToMostAgent,
    MultiArmedBanditAgent,
    SelfDiscoverAgent,
    SkeletonOfThoughtAgent,
)

# ── helpers ──────────────────────────────────────────────────────────


class _Stub(BaseAgent):
    """Deterministic stub that returns a fixed response."""

    def __init__(self, name: str, response: str = "stub") -> None:
        super().__init__(name=name)
        self._response = response

    def _run_impl(self, ctx):
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(self, ctx):
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class _DynamicStub(BaseAgent):
    """Returns different responses on successive calls."""

    def __init__(self, name: str, responses: list[str]) -> None:
        super().__init__(name=name)
        self._responses = list(responses)
        self._idx = 0

    def _run_impl(self, ctx):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)

    async def _run_async_impl(self, ctx):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)


def _collect(agent: BaseAgent, msg: str = "test") -> tuple[list[Event], Session]:
    session = Session()
    ctx = InvocationContext(session=session, user_message=msg)
    events = list(agent._run_impl_traced(ctx))
    return events, session


# ── SkeletonOfThoughtAgent ───────────────────────────────────────────


class TestSkeletonOfThoughtAgent:
    def test_basic_pipeline(self):
        skeleton = _Stub("skeleton", "1. Point A\n2. Point B\n3. Point C")
        worker = _Stub("worker", "Elaboration text.")
        assembler = _Stub("assembler", "Final assembled document.")

        sot = SkeletonOfThoughtAgent(
            name="sot",
            skeleton_agent=skeleton,
            worker_agent=worker,
            assembler_agent=assembler,
            result_key="sot_result",
        )

        events, session = _collect(sot)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert any("Final assembled" in m for m in messages)
        assert "sot_result" in session.state
        assert session.state["sot_result"]["skeleton"] == ["Point A", "Point B", "Point C"]

    def test_empty_skeleton_error(self):
        skeleton = _Stub("skeleton", "")
        worker = _Stub("worker", "text")
        assembler = _Stub("assembler", "final")

        sot = SkeletonOfThoughtAgent(
            name="sot",
            skeleton_agent=skeleton,
            worker_agent=worker,
            assembler_agent=assembler,
        )
        events, _ = _collect(sot)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_max_points_limit(self):
        skeleton = _Stub(
            "skeleton",
            "1. A\n2. B\n3. C\n4. D\n5. E\n6. F\n7. G\n8. H",
        )
        worker = _Stub("worker", "text")
        assembler = _Stub("assembler", "final")

        sot = SkeletonOfThoughtAgent(
            name="sot",
            skeleton_agent=skeleton,
            worker_agent=worker,
            assembler_agent=assembler,
            max_points=3,
            result_key="res",
        )
        events, session = _collect(sot)
        assert len(session.state["res"]["skeleton"]) == 3

    def test_missing_agents_error(self):
        sot = SkeletonOfThoughtAgent(name="sot")
        events, _ = _collect(sot)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_parse_skeleton_fallback(self):
        """When no numbered lines found, fallback to sentence split."""
        points = SkeletonOfThoughtAgent._parse_skeleton(
            "First idea. Second idea. Third idea.", 5,
        )
        assert len(points) >= 2


# ── LeastToMostAgent ────────────────────────────────────────────────


class TestLeastToMostAgent:
    def test_basic_pipeline(self):
        decomposer = _Stub("dec", "1. Easy\n2. Medium\n3. Hard")
        solver = _Stub("solver", "Solved!")
        synth = _Stub("synth", "Combined solution.")

        l2m = LeastToMostAgent(
            name="l2m",
            decomposer_agent=decomposer,
            solver_agent=solver,
            synthesizer_agent=synth,
            result_key="l2m_result",
        )

        events, session = _collect(l2m)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert any("Combined" in m for m in messages)
        assert len(session.state["l2m_result"]["solutions"]) == 3

    def test_max_subproblems(self):
        decomposer = _Stub("dec", "1. A\n2. B\n3. C\n4. D\n5. E")
        solver = _Stub("solver", "Done")
        synth = _Stub("synth", "Final")

        l2m = LeastToMostAgent(
            name="l2m",
            decomposer_agent=decomposer,
            solver_agent=solver,
            synthesizer_agent=synth,
            max_subproblems=2,
            result_key="res",
        )
        events, session = _collect(l2m)
        assert len(session.state["res"]["solutions"]) == 2

    def test_missing_agents_error(self):
        l2m = LeastToMostAgent(name="l2m")
        events, _ = _collect(l2m)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_empty_decomposition_error(self):
        decomposer = _Stub("dec", "")
        solver = _Stub("solver", "Done")
        synth = _Stub("synth", "Final")

        l2m = LeastToMostAgent(
            name="l2m",
            decomposer_agent=decomposer,
            solver_agent=solver,
            synthesizer_agent=synth,
        )
        events, _ = _collect(l2m)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors


# ── SelfDiscoverAgent ───────────────────────────────────────────────


class TestSelfDiscoverAgent:
    def test_four_phases(self):
        agent = _DynamicStub(
            "thinker",
            [
                "Selected: Step-by-Step, Decomposition",
                "Adapted modules for the task",
                "Step 1: decompose. Step 2: analyse.",
                "The final answer is 42.",
            ],
        )

        sd = SelfDiscoverAgent(
            name="sd", agent=agent, result_key="sd_result",
        )
        events, session = _collect(sd)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert any("42" in m for m in messages)
        res = session.state["sd_result"]
        assert "selected" in res
        assert "adapted" in res
        assert "structure" in res
        assert "final" in res

    def test_custom_modules(self):
        agent = _DynamicStub("t", ["a", "b", "c", "d"])
        sd = SelfDiscoverAgent(
            name="sd",
            agent=agent,
            reasoning_modules=["Logic", "Math"],
        )
        assert sd.reasoning_modules == ["Logic", "Math"]

    def test_no_agent_error(self):
        sd = SelfDiscoverAgent(name="sd")
        events, _ = _collect(sd)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors


# ── GeneticAlgorithmAgent ───────────────────────────────────────────


class TestGeneticAlgorithmAgent:
    def test_basic_evolution(self):
        gen = _Stub("gen", "initial solution")
        cross = _Stub("cross", "crossed solution")
        mut = _Stub("mut", "mutated solution")

        ga = GeneticAlgorithmAgent(
            name="ga",
            agent=gen,
            crossover_agent=cross,
            mutation_agent=mut,
            fitness_fn=lambda r: len(r) / 100,
            population_size=4,
            n_generations=2,
            result_key="ga_result",
        )

        events, session = _collect(ga)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        res = session.state["ga_result"]
        assert res["best_score"] > 0
        assert len(res["generations"]) == 2

    def test_elite_preserved(self):
        gen = _DynamicStub("gen", ["short", "a much longer response here"])
        cross = _Stub("cross", "child")
        mut = _Stub("mut", "mutant")

        ga = GeneticAlgorithmAgent(
            name="ga",
            agent=gen,
            crossover_agent=cross,
            mutation_agent=mut,
            fitness_fn=lambda r: len(r),
            population_size=3,
            n_generations=1,
            elite_count=1,
            result_key="res",
        )
        events, session = _collect(ga)
        assert session.state["res"]["best_score"] > 0

    def test_missing_agents_error(self):
        ga = GeneticAlgorithmAgent(name="ga")
        events, _ = _collect(ga)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_zero_mutation_rate(self):
        gen = _Stub("gen", "solution")
        cross = _Stub("cross", "crossed")
        mut = _Stub("mut", "should not run")

        ga = GeneticAlgorithmAgent(
            name="ga",
            agent=gen,
            crossover_agent=cross,
            mutation_agent=mut,
            mutation_rate=0.0,
            population_size=3,
            n_generations=1,
        )
        events, _ = _collect(ga)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages


# ── MultiArmedBanditAgent ───────────────────────────────────────────


class TestMultiArmedBanditAgent:
    def test_epsilon_greedy(self):
        a = _Stub("a", "response_a")
        b = _Stub("b", "response_b")

        bandit = MultiArmedBanditAgent(
            name="bandit",
            sub_agents=[a, b],
            reward_fn=lambda r: 1.0 if "a" in r else 0.0,
            strategy="epsilon_greedy",
            epsilon=0.0,  # always exploit
            result_key="bandit_result",
        )

        session = Session()
        # Run multiple times to build stats
        for _ in range(5):
            ctx = InvocationContext(session=session, user_message="test")
            list(bandit.run(ctx))

        stats = session.state["_bandit_stats"]
        assert stats["a"]["total"] + stats["b"]["total"] == 5

    def test_ucb1_strategy(self):
        a = _Stub("a", "good")
        b = _Stub("b", "bad")

        bandit = MultiArmedBanditAgent(
            name="bandit",
            sub_agents=[a, b],
            reward_fn=lambda r: 1.0 if r == "good" else 0.0,
            strategy="ucb1",
            result_key="res",
        )
        events, session = _collect(bandit)
        assert session.state["res"]["selected"] in {"a", "b"}

    def test_thompson_strategy(self):
        a = _Stub("a", "ok")
        b = _Stub("b", "ok")

        bandit = MultiArmedBanditAgent(
            name="bandit",
            sub_agents=[a, b],
            strategy="thompson",
        )
        events, _ = _collect(bandit)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages

    def test_stats_persist_across_calls(self):
        a = _Stub("a", "good")
        b = _Stub("b", "bad")

        bandit = MultiArmedBanditAgent(
            name="bandit",
            sub_agents=[a, b],
            reward_fn=lambda r: 1.0 if r == "good" else 0.0,
            strategy="epsilon_greedy",
            epsilon=1.0,  # always explore
        )

        session = Session()
        for _ in range(10):
            ctx = InvocationContext(session=session, user_message="test")
            list(bandit.run(ctx))

        stats = session.state["_bandit_stats"]
        total = stats["a"]["total"] + stats["b"]["total"]
        assert total == 10

    def test_no_agents_error(self):
        bandit = MultiArmedBanditAgent(name="bandit")
        events, _ = _collect(bandit)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors
