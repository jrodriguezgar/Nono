"""Tests for BacktrackingAgent, ChainOfDensityAgent, MediatorAgent,
DivideAndConquerAgent, and BeamSearchAgent."""

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
    BacktrackingAgent,
    BeamSearchAgent,
    ChainOfDensityAgent,
    DivideAndConquerAgent,
    MediatorAgent,
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


# ── BacktrackingAgent ────────────────────────────────────────────────


class TestBacktrackingAgent:
    def test_all_pass(self):
        a1 = _Stub("a1", "step1_ok")
        a2 = _Stub("a2", "step2_ok")

        bt = BacktrackingAgent(
            name="bt",
            sub_agents=[a1, a2],
            validate_fn=lambda s: True,
            result_key="bt_res",
        )

        events, session = _collect(bt)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "step2_ok" in messages[-1]
        assert session.state["bt_res"]["total_retries"] == 0

    def test_backtrack_and_retry(self):
        # First call to a1 produces "BAD", second produces "GOOD"
        a1 = _DynamicStub("a1", ["BAD", "GOOD"])
        a2 = _Stub("a2", "final")

        call_count = {"n": 0}

        def validate(s):
            call_count["n"] += 1
            return s != "BAD"

        bt = BacktrackingAgent(
            name="bt",
            sub_agents=[a1, a2],
            validate_fn=validate,
            max_retries=3,
            result_key="res",
        )

        events, session = _collect(bt)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages[-1] == "final"
        assert session.state["res"]["total_retries"] == 1

    def test_max_retries_exhausted(self):
        a = _Stub("a", "always_bad")
        bt = BacktrackingAgent(
            name="bt",
            sub_agents=[a],
            validate_fn=lambda s: False,
            max_retries=2,
            result_key="res",
        )
        events, session = _collect(bt)
        # Should still produce output even after exhaustion
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["res"]["total_retries"] == 3  # initial + 2 retries

    def test_no_agents_error(self):
        bt = BacktrackingAgent(name="bt")
        events, _ = _collect(bt)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_custom_backtrack_to(self):
        a1 = _Stub("a1", "ok")
        a2 = _DynamicStub("a2", ["BAD", "GOOD"])
        a3 = _Stub("a3", "final")

        bt = BacktrackingAgent(
            name="bt",
            sub_agents=[a1, a2, a3],
            validate_fn=lambda s: s != "BAD",
            backtrack_to=1,  # restart from a2
            max_retries=2,
        )

        events, _ = _collect(bt)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages[-1] == "final"


# ── ChainOfDensityAgent ─────────────────────────────────────────────


class TestChainOfDensityAgent:
    def test_basic_densification(self):
        # Round 1: return denser text, Round 2: return FULLY_DENSE
        agent = _DynamicStub("d", [
            "Initial sparse text",   # initial
            "Denser text v1",        # round 1
            "FULLY_DENSE",           # round 2 → stop
        ])

        cod = ChainOfDensityAgent(
            name="cod",
            agent=agent,
            n_rounds=5,
            result_key="cod_res",
        )

        events, session = _collect(cod)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["cod_res"]["completed"]

    def test_all_rounds_used(self):
        agent = _DynamicStub("d", [
            "sparse",     # initial
            "denser",     # round 1
            "densest",    # round 2
        ])

        cod = ChainOfDensityAgent(
            name="cod", agent=agent, n_rounds=2, result_key="res",
        )
        events, session = _collect(cod)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["res"]["completed"]

    def test_no_agent_error(self):
        cod = ChainOfDensityAgent(name="cod")
        events, _ = _collect(cod)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors


# ── MediatorAgent ────────────────────────────────────────────────────


class TestMediatorAgent:
    def test_basic_mediation(self):
        p1 = _Stub("optimist", "We should invest")
        p2 = _Stub("pessimist", "We should save")
        mediator = _Stub("mediator", "Compromise: invest 50%, save 50%")

        med = MediatorAgent(
            name="med",
            sub_agents=[p1, p2],
            mediator_agent=mediator,
            result_key="med_res",
        )

        events, session = _collect(med)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "Compromise" in messages[-1]
        assert "optimist" in session.state["med_res"]["proposals"]
        assert "pessimist" in session.state["med_res"]["proposals"]

    def test_no_proposal_agents_error(self):
        med = MediatorAgent(name="med", mediator_agent=_Stub("m", "x"))
        events, _ = _collect(med)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_no_mediator_error(self):
        med = MediatorAgent(name="med", sub_agents=[_Stub("p", "x")])
        events, _ = _collect(med)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_three_proposals(self):
        p1 = _Stub("a", "Plan A")
        p2 = _Stub("b", "Plan B")
        p3 = _Stub("c", "Plan C")
        mediator = _Stub("med", "Combined plan")

        m = MediatorAgent(
            name="m",
            sub_agents=[p1, p2, p3],
            mediator_agent=mediator,
            result_key="res",
        )
        events, session = _collect(m)
        assert len(session.state["res"]["proposals"]) == 3


# ── DivideAndConquerAgent ────────────────────────────────────────────


class TestDivideAndConquerAgent:
    def test_base_case_direct_solve(self):
        splitter = _Stub("split", "1. sub1\n2. sub2")
        solver = _Stub("solve", "solution")
        merger = _Stub("merge", "merged result")

        dc = DivideAndConquerAgent(
            name="dc",
            splitter_agent=splitter,
            solver_agent=solver,
            merger_agent=merger,
            is_base_case=lambda p: True,  # everything is base case
            result_key="dc_res",
        )

        events, session = _collect(dc)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        # Since is_base_case always True, it should solve directly
        assert "solution" in messages[-1]

    def test_recursive_split_merge(self):
        splitter = _Stub("split", "1. Part A\n2. Part B")
        solver = _Stub("solve", "solved")
        merger = _Stub("merge", "merged")

        dc = DivideAndConquerAgent(
            name="dc",
            splitter_agent=splitter,
            solver_agent=solver,
            merger_agent=merger,
            is_base_case=lambda p: len(p) < 10,
            max_depth=2,
            result_key="dc_res",
        )

        events, session = _collect(dc)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["dc_res"]["completed"]

    def test_missing_agents_error(self):
        dc = DivideAndConquerAgent(name="dc")
        events, _ = _collect(dc)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_max_depth_respected(self):
        splitter = _Stub("split", "1. A\n2. B")
        solver = _Stub("solve", "leaf")
        merger = _Stub("merge", "merged")

        dc = DivideAndConquerAgent(
            name="dc",
            splitter_agent=splitter,
            solver_agent=solver,
            merger_agent=merger,
            is_base_case=lambda p: False,  # never base case
            max_depth=1,
        )

        events, _ = _collect(dc)
        # Should still complete (max_depth forces base case)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages

    def test_parse_subproblems(self):
        text = "1. First sub\n2. Second sub\n3. Third sub"
        subs = DivideAndConquerAgent._parse_subproblems(text)
        assert len(subs) == 3
        assert subs[0] == "First sub"


# ── BeamSearchAgent ──────────────────────────────────────────────────


class TestBeamSearchAgent:
    def test_basic_beam_search(self):
        agent = _Stub("expander", "expanded reasoning")

        bs = BeamSearchAgent(
            name="bs",
            agent=agent,
            score_fn=lambda s: len(s) / 100.0,
            beam_width=2,
            n_expansions=2,
            n_steps=2,
            result_key="bs_res",
        )

        events, session = _collect(bs)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["bs_res"]["completed"]
        assert session.state["bs_res"]["beam_width"] == 2

    def test_single_beam(self):
        agent = _Stub("e", "result")
        bs = BeamSearchAgent(
            name="bs", agent=agent, beam_width=1, n_steps=1, n_expansions=1,
        )
        events, _ = _collect(bs)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages

    def test_score_fn_selects_best(self):
        # Each expansion returns different text; score_fn prefers longer
        agent = _DynamicStub("e", ["short", "a longer response keeps winning"])
        bs = BeamSearchAgent(
            name="bs",
            agent=agent,
            score_fn=lambda s: len(s),
            beam_width=1,
            n_expansions=2,
            n_steps=2,
            result_key="res",
        )
        events, session = _collect(bs)
        assert session.state["res"]["best_score"] > 0

    def test_no_agent_error(self):
        bs = BeamSearchAgent(name="bs")
        events, _ = _collect(bs)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_step_updates(self):
        agent = _Stub("e", "text")
        bs = BeamSearchAgent(
            name="bs", agent=agent, n_steps=3, beam_width=2, n_expansions=1,
        )
        events, _ = _collect(bs)
        state_updates = [e.content for e in events if e.event_type == EventType.STATE_UPDATE]
        assert any("Step 1" in s for s in state_updates)
        assert any("Step 3" in s for s in state_updates)
