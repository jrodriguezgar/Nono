"""Tests for SelfConsistencyAgent, MixtureOfAgentsAgent, StepBackAgent,
OrchestratorWorkerAgent, and SelfRefineAgent."""

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
    MixtureOfAgentsAgent,
    OrchestratorWorkerAgent,
    SelfConsistencyAgent,
    SelfRefineAgent,
    StepBackAgent,
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


# ── SelfConsistencyAgent ─────────────────────────────────────────────


class TestSelfConsistencyAgent:
    def test_majority_vote(self):
        # 3 out of 5 say "Paris" → Paris wins
        agent = _DynamicStub("sampler", ["Paris", "London", "Paris", "Berlin", "Paris"])

        sc = SelfConsistencyAgent(
            name="sc",
            agent=agent,
            n_samples=5,
            extract_fn=lambda s: s.strip().lower(),
            result_key="sc_result",
        )

        events, session = _collect(sc)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert any("Paris" in m for m in messages)
        assert session.state["sc_result"]["winner"] == "paris"
        assert session.state["sc_result"]["n_samples"] == 5

    def test_single_sample(self):
        agent = _Stub("sampler", "42")
        sc = SelfConsistencyAgent(name="sc", agent=agent, n_samples=1)
        events, _ = _collect(sc)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages[-1] == "42"

    def test_no_agent_error(self):
        sc = SelfConsistencyAgent(name="sc")
        events, _ = _collect(sc)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_custom_extract_fn(self):
        agent = _DynamicStub("s", ["answer: YES", "answer: yes", "answer: NO"])
        sc = SelfConsistencyAgent(
            name="sc",
            agent=agent,
            n_samples=3,
            extract_fn=lambda s: s.split(":")[-1].strip().upper(),
            result_key="res",
        )
        events, session = _collect(sc)
        assert session.state["res"]["winner"] == "YES"


# ── MixtureOfAgentsAgent ────────────────────────────────────────────


class TestMixtureOfAgentsAgent:
    def test_basic_moa_pipeline(self):
        p1 = _Stub("p1", "Proposal A")
        p2 = _Stub("p2", "Proposal B")
        final = _Stub("final", "Synthesised result")

        moa = MixtureOfAgentsAgent(
            name="moa",
            proposer_agents=[p1, p2],
            final_agent=final,
            n_layers=1,
            result_key="moa_res",
        )

        events, session = _collect(moa)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "Synthesised result" in messages[-1]
        assert session.state["moa_res"]["completed"]

    def test_multi_layer(self):
        p1 = _Stub("p1", "layer1")
        p2 = _Stub("p2", "layer1")
        agg = _Stub("agg", "refined")
        final = _Stub("final", "done")

        moa = MixtureOfAgentsAgent(
            name="moa",
            proposer_agents=[p1, p2],
            aggregator_agents=[agg],
            final_agent=final,
            n_layers=3,
            result_key="res",
        )

        events, session = _collect(moa)
        # Should have state updates for each layer
        state_updates = [e.content for e in events if e.event_type == EventType.STATE_UPDATE]
        assert any("Layer 1" in s for s in state_updates)
        assert any("Layer 2" in s for s in state_updates)

    def test_no_proposers_error(self):
        moa = MixtureOfAgentsAgent(name="moa")
        events, _ = _collect(moa)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_no_final_agent(self):
        p1 = _Stub("p1", "result directly")
        moa = MixtureOfAgentsAgent(
            name="moa", proposer_agents=[p1], n_layers=1,
        )
        events, _ = _collect(moa)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages


# ── StepBackAgent ────────────────────────────────────────────────────


class TestStepBackAgent:
    def test_basic_two_phase(self):
        abstractor = _Stub("abs", "General principle: thermodynamics")
        reasoner = _Stub("reason", "The answer is entropy.")

        sb = StepBackAgent(
            name="sb",
            abstractor_agent=abstractor,
            reasoner_agent=reasoner,
            result_key="sb_res",
        )

        events, session = _collect(sb)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "entropy" in messages[-1]
        assert session.state["sb_res"]["abstraction"] == "General principle: thermodynamics"

    def test_missing_agents_error(self):
        sb = StepBackAgent(name="sb")
        events, _ = _collect(sb)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_partial_agents_error(self):
        sb = StepBackAgent(name="sb", abstractor_agent=_Stub("a", "x"))
        events, _ = _collect(sb)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors


# ── OrchestratorWorkerAgent ──────────────────────────────────────────


class TestOrchestratorWorkerAgent:
    def test_no_agents_error(self):
        ow = OrchestratorWorkerAgent(name="ow")
        events, _ = _collect(ow)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_completion_keyword_detection(self):
        # The OrchestratorWorker needs an LLM, but we test the parse logic
        ow = OrchestratorWorkerAgent(
            name="ow",
            sub_agents=[_Stub("w1", "done")],
            completion_keyword="DONE",
        )
        # Test _parse_delegation directly
        assert ow._parse_delegation("DONE") is None
        assert ow._parse_delegation("all DONE now") is None

    def test_parse_delegation_json(self):
        ow = OrchestratorWorkerAgent(
            name="ow",
            sub_agents=[_Stub("writer", "result")],
        )
        result = ow._parse_delegation('{"worker": "writer", "task": "Write docs"}')
        assert result == ("writer", "Write docs")

    def test_parse_delegation_fallback(self):
        ow = OrchestratorWorkerAgent(
            name="ow",
            sub_agents=[_Stub("writer", "result")],
        )
        result = ow._parse_delegation("Please delegate to writer for this")
        assert result is not None
        assert result[0] == "writer"


# ── SelfRefineAgent ──────────────────────────────────────────────────


class TestSelfRefineAgent:
    def test_basic_refinement(self):
        # Cycle: generate → critique (no stop) → refine → critique (stop)
        agent = _DynamicStub("refiner", [
            "Draft v1",                          # initial
            "Issues: grammar",                   # critique 1
            "Draft v2 improved",                 # refine 1
            "NO_ISSUES_FOUND",                   # critique 2 → stop
        ])

        sr = SelfRefineAgent(
            name="sr",
            agent=agent,
            max_iterations=3,
            result_key="sr_res",
        )

        events, session = _collect(sr)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages
        assert session.state["sr_res"]["completed"]

    def test_max_iterations_reached(self):
        agent = _DynamicStub("a", [
            "draft",        # initial
            "issue found",  # critique
            "refined",      # refine
            "still issue",  # critique
            "refined v2",   # refine
        ])

        sr = SelfRefineAgent(name="sr", agent=agent, max_iterations=2)
        events, _ = _collect(sr)
        messages = [e.content for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert messages

    def test_no_agent_error(self):
        sr = SelfRefineAgent(name="sr")
        events, _ = _collect(sr)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert errors

    def test_custom_stop_phrase(self):
        agent = _DynamicStub("a", ["draft", "ALL_GOOD"])
        sr = SelfRefineAgent(
            name="sr", agent=agent, stop_phrase="ALL_GOOD",
            max_iterations=5, result_key="res",
        )
        events, session = _collect(sr)
        # Should stop after first critique
        assert any("stopped" in str(i) for i in session.state["res"]["iterations"])
