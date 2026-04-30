"""Tests for Batch 4 orchestration agents (P1-P37, agents 64-100).

Covers: RephraseAndRespond, CumulativeReasoning, MultiPersona, AntColony,
PipelineParallel, ContractNet, RedTeam, FeedbackLoop, Winnowing,
MixtureOfThoughts, SimulatedAnnealing, TabuSearch, ParticleSwarm,
DifferentialEvolution, BayesianOptimization, AnalogicalReasoning,
ThreadOfThought, ExpertPrompting, BufferOfThoughts, ChainOfAbstraction,
Verifier, ProgOfThought, InnerMonologue, RolePlaying, GossipProtocol,
Auction, DelphiMethod, NominalGroup, ActiveRetrieval, IterativeRetrieval,
PromptChain, HypothesisTesting, SkillLibrary, RecursiveCritic,
DemonstrateSearchPredict, DoubleLoopLearning, Agenda.

Run:
    python -m pytest tests/test_batch4_agents.py -v
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
    ActiveRetrievalAgent,
    AgendaAgent,
    AnalogicalReasoningAgent,
    AntColonyAgent,
    AuctionAgent,
    BayesianOptimizationAgent,
    BufferOfThoughtsAgent,
    ChainOfAbstractionAgent,
    ContractNetAgent,
    CumulativeReasoningAgent,
    DelphiMethodAgent,
    DemonstrateSearchPredictAgent,
    DifferentialEvolutionAgent,
    DoubleLoopLearningAgent,
    ExpertPromptingAgent,
    FeedbackLoopAgent,
    GossipProtocolAgent,
    HypothesisTestingAgent,
    InnerMonologueAgent,
    IterativeRetrievalAgent,
    MixtureOfThoughtsAgent,
    MultiPersonaAgent,
    NominalGroupAgent,
    ParticleSwarmAgent,
    PipelineParallelAgent,
    ProgOfThoughtAgent,
    PromptChainAgent,
    RecursiveCriticAgent,
    RedTeamAgent,
    RephraseAndRespondAgent,
    RolePlayingAgent,
    SimulatedAnnealingAgent,
    SkillLibraryAgent,
    TabuSearchAgent,
    ThreadOfThoughtAgent,
    VerifierAgent,
    WinnowingAgent,
)


# ── Stub agents ───────────────────────────────────────────────────────────────


class _Stub(BaseAgent):
    """Returns a fixed response."""

    def __init__(self, *, name: str = "stub", response: str = "ok") -> None:
        super().__init__(name=name, description=f"stub-{name}")
        self._response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class _DynamicStub(BaseAgent):
    """Returns a response based on call count."""

    def __init__(
        self, *, name: str = "dyn", responses: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, description=f"dyn-{name}")
        self._responses = responses or ["response"]
        self._call = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        r = self._responses[min(self._call, len(self._responses) - 1)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, r)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        r = self._responses[min(self._call, len(self._responses) - 1)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, r)


class _NumberStub(BaseAgent):
    """Returns a number string for score/rank parsing."""

    def __init__(self, *, name: str = "num", value: str = "5") -> None:
        super().__init__(name=name, description=f"num-{name}")
        self._value = value

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._value)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._value)


class _CodeStub(BaseAgent):
    """Returns code wrapped in markdown fences."""

    def __init__(
        self, *, name: str = "code", code: str = "print(42)",
    ) -> None:
        super().__init__(name=name, description=f"code-{name}")
        self._code = code

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"```python\n{self._code}\n```",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"```python\n{self._code}\n```",
        )


class _DoneStub(BaseAgent):
    """Returns DONE after N calls."""

    def __init__(self, *, name: str = "done", done_after: int = 2) -> None:
        super().__init__(name=name, description=f"done-{name}")
        self._done_after = done_after
        self._call = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self._call += 1
        if self._call >= self._done_after:
            yield Event(EventType.AGENT_MESSAGE, self.name, "DONE: final answer")
        else:
            yield Event(EventType.AGENT_MESSAGE, self.name, "action step")

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        self._call += 1
        if self._call >= self._done_after:
            yield Event(EventType.AGENT_MESSAGE, self.name, "DONE: final answer")
        else:
            yield Event(EventType.AGENT_MESSAGE, self.name, "action step")


def _ctx(msg: str = "hello") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


def _collect(events: Iterator[Event]) -> list[Event]:
    return list(events)


async def _acollect(events: AsyncIterator[Event]) -> list[Event]:
    return [ev async for ev in events]


# ── RephraseAndRespondAgent ───────────────────────────────────────────────────


def test_rephrase_and_respond_sync():
    agent = RephraseAndRespondAgent(
        name="rar",
        rephrase_agent=_Stub(name="rep", response="rephrased question"),
        solver_agent=_Stub(name="sol", response="solved!"),
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert msgs[-1].content == "solved!"


def test_rephrase_and_respond_async():
    agent = RephraseAndRespondAgent(
        name="rar",
        rephrase_agent=_Stub(name="rep", response="rephrased"),
        solver_agent=_Stub(name="sol", response="answer"),
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert msgs[-1].content == "answer"


# ── CumulativeReasoningAgent ─────────────────────────────────────────────────


def test_cumulative_reasoning_sync():
    agent = CumulativeReasoningAgent(
        name="cr",
        proposer_agent=_Stub(name="p", response="fact A"),
        verifier_agent=_Stub(name="v", response="ACCEPT"),
        reporter_agent=_Stub(name="r", response="final report"),
        n_rounds=2,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_cumulative_reasoning_async():
    agent = CumulativeReasoningAgent(
        name="cr",
        proposer_agent=_Stub(name="p", response="fact"),
        verifier_agent=_Stub(name="v", response="REJECT"),
        reporter_agent=_Stub(name="r", response="report"),
        n_rounds=1,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── MultiPersonaAgent ────────────────────────────────────────────────────────


def test_multi_persona_sync():
    agent = MultiPersonaAgent(
        name="mp",
        agent=_Stub(name="a", response="perspective"),
        personas=["scientist", "artist"],
        result_key="mp_result",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "mp_result" in ctx.session.state


# ── AntColonyAgent ───────────────────────────────────────────────────────────


def test_ant_colony_sync():
    agent = AntColonyAgent(
        name="ant",
        agent=_Stub(name="a", response="solution"),
        score_fn=lambda r: 0.7,
        n_ants=2,
        n_rounds=2,
        result_key="ant_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "ant_res" in ctx.session.state


# ── PipelineParallelAgent ────────────────────────────────────────────────────


def test_pipeline_parallel_sync():
    agent = PipelineParallelAgent(
        name="pp",
        stages=[_Stub(name="s1", response="step1"), _Stub(name="s2", response="step2")],
        items_key="items",
    )
    ctx = _ctx()
    ctx.session.state["items"] = ["a", "b"]
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_pipeline_parallel_async():
    agent = PipelineParallelAgent(
        name="pp",
        stages=[_Stub(name="s1", response="done")],
        items_key="data",
    )
    ctx = _ctx()
    ctx.session.state["data"] = ["x"]
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(ctx)))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ContractNetAgent ─────────────────────────────────────────────────────────


def test_contract_net_sync():
    agent = ContractNetAgent(
        name="cn",
        sub_agents=[_Stub(name="a1", response="bid1"), _Stub(name="a2", response="bid2")],
        bid_fn=lambda _n, r: len(r),
        result_key="cn_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── RedTeamAgent ─────────────────────────────────────────────────────────────


def test_red_team_sync():
    agent = RedTeamAgent(
        name="rt",
        defender_agent=_Stub(name="def", response="secure output"),
        attacker_agent=_Stub(name="atk", response="attack probe"),
        n_rounds=2,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_red_team_async():
    agent = RedTeamAgent(
        name="rt",
        defender_agent=_Stub(name="def", response="safe"),
        attacker_agent=_Stub(name="atk", response="hack?"),
        n_rounds=1,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── FeedbackLoopAgent ────────────────────────────────────────────────────────


def test_feedback_loop_sync():
    agent = FeedbackLoopAgent(
        name="fl",
        sub_agents=[_Stub(name="a", response="same answer")],
        max_iterations=3,
        threshold=0.5,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── WinnowingAgent ──────────────────────────────────────────────────────────


def test_winnowing_sync():
    agent = WinnowingAgent(
        name="win",
        agent=_Stub(name="gen", response="candidate"),
        evaluator_agent=_NumberStub(name="eval", value="8"),
        n_candidates=3,
        cull_fraction=0.5,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── MixtureOfThoughtsAgent ──────────────────────────────────────────────────


def test_mixture_of_thoughts_sync():
    agent = MixtureOfThoughtsAgent(
        name="mot",
        agent=_Stub(name="a", response="thought"),
        selector_agent=_Stub(name="sel", response="best thought"),
        result_key="mot_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "mot_res" in ctx.session.state


# ── SimulatedAnnealingAgent ─────────────────────────────────────────────────


def test_simulated_annealing_sync():
    agent = SimulatedAnnealingAgent(
        name="sa",
        agent=_Stub(name="a", response="solution"),
        score_fn=lambda r: 0.5,
        n_iterations=3,
        result_key="sa_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "sa_res" in ctx.session.state


def test_simulated_annealing_async():
    agent = SimulatedAnnealingAgent(
        name="sa",
        agent=_Stub(name="a", response="result"),
        score_fn=lambda r: 0.6,
        n_iterations=2,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── TabuSearchAgent ─────────────────────────────────────────────────────────


def test_tabu_search_sync():
    agent = TabuSearchAgent(
        name="ts",
        agent=_DynamicStub(name="a", responses=["sol1", "sol2", "sol3"]),
        score_fn=lambda r: len(r),
        n_iterations=3,
        tabu_size=2,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ParticleSwarmAgent ──────────────────────────────────────────────────────


def test_particle_swarm_sync():
    agent = ParticleSwarmAgent(
        name="pso",
        agent=_Stub(name="a", response="particle_pos"),
        score_fn=lambda r: 0.5,
        n_particles=3,
        n_iterations=2,
        result_key="pso_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── DifferentialEvolutionAgent ──────────────────────────────────────────────


def test_differential_evolution_sync():
    agent = DifferentialEvolutionAgent(
        name="de",
        agent=_Stub(name="a", response="individual"),
        score_fn=lambda r: 0.5,
        population_size=4,
        n_generations=2,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── BayesianOptimizationAgent ───────────────────────────────────────────────


def test_bayesian_optimization_sync():
    agent = BayesianOptimizationAgent(
        name="bo",
        agent=_Stub(name="a", response="candidate"),
        score_fn=lambda r: 0.7,
        n_iterations=3,
        result_key="bo_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "bo_res" in ctx.session.state


def test_bayesian_optimization_async():
    agent = BayesianOptimizationAgent(
        name="bo",
        agent=_Stub(name="a", response="x"),
        score_fn=lambda r: 0.3,
        n_iterations=2,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── AnalogicalReasoningAgent ────────────────────────────────────────────────


def test_analogical_reasoning_sync():
    agent = AnalogicalReasoningAgent(
        name="ar",
        agent=_Stub(name="a", response="analogy answer"),
        n_analogies=2,
        result_key="ar_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "ar_res" in ctx.session.state


# ── ThreadOfThoughtAgent ────────────────────────────────────────────────────


def test_thread_of_thought_sync():
    agent = ThreadOfThoughtAgent(
        name="thot",
        agent=_Stub(name="a", response="walkthrough answer"),
        result_key="thot_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_thread_of_thought_async():
    agent = ThreadOfThoughtAgent(
        name="thot",
        agent=_Stub(name="a", response="done"),
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ExpertPromptingAgent ────────────────────────────────────────────────────


def test_expert_prompting_sync():
    agent = ExpertPromptingAgent(
        name="ep",
        agent=_Stub(name="a", response="expert answer"),
        result_key="ep_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "ep_res" in ctx.session.state


# ── BufferOfThoughtsAgent ───────────────────────────────────────────────────


def test_buffer_of_thoughts_sync():
    agent = BufferOfThoughtsAgent(
        name="bot",
        agent=_DynamicStub(name="a", responses=["1", "answer", "template"]),
        initial_buffer=["decompose step by step"],
        result_key="bot_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_buffer_of_thoughts_empty_buffer():
    agent = BufferOfThoughtsAgent(
        name="bot",
        agent=_Stub(name="a", response="answer"),
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ChainOfAbstractionAgent ─────────────────────────────────────────────────


def test_chain_of_abstraction_sync():
    agent = ChainOfAbstractionAgent(
        name="coa",
        agent=_Stub(name="a", response="grounded chain"),
        result_key="coa_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "coa_res" in ctx.session.state


# ── VerifierAgent ────────────────────────────────────────────────────────────


def test_verifier_sync():
    agent = VerifierAgent(
        name="ver",
        generator=_Stub(name="gen", response="solution"),
        verifier=_NumberStub(name="v", value="8"),
        n_solutions=3,
        result_key="ver_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert ctx.session.state["ver_res"]["best_score"] == 8.0


def test_verifier_async():
    agent = VerifierAgent(
        name="ver",
        generator=_Stub(name="gen", response="ans"),
        verifier=_NumberStub(name="v", value="7"),
        n_solutions=2,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ProgOfThoughtAgent ──────────────────────────────────────────────────────


def test_prog_of_thought_sync():
    agent = ProgOfThoughtAgent(
        name="pot",
        agent=_CodeStub(name="c", code="print(42)"),
        result_key="pot_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "42" in msgs[-1].content


def test_prog_of_thought_error():
    agent = ProgOfThoughtAgent(
        name="pot",
        agent=_CodeStub(name="c", code="raise ValueError('bad')"),
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert "error" in msgs[-1].content.lower() or "Error" in msgs[-1].content


# ── InnerMonologueAgent ─────────────────────────────────────────────────────


def test_inner_monologue_sync():
    agent = InnerMonologueAgent(
        name="im",
        agent=_DoneStub(name="a", done_after=2),
        max_steps=5,
        result_key="im_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "DONE" in msgs[-1].content


def test_inner_monologue_async():
    agent = InnerMonologueAgent(
        name="im",
        agent=_DoneStub(name="a", done_after=1),
        max_steps=3,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── RolePlayingAgent ────────────────────────────────────────────────────────


def test_role_playing_sync():
    agent = RolePlayingAgent(
        name="rp",
        instructor_agent=_Stub(name="inst", response="do X"),
        assistant_agent=_Stub(name="asst", response="done X"),
        n_turns=2,
        result_key="rp_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_role_playing_error_no_agents():
    agent = RolePlayingAgent(name="rp")
    evs = _collect(agent._run_impl_traced(_ctx()))
    errs = [e for e in evs if e.event_type == EventType.ERROR]
    assert len(errs) >= 1


# ── GossipProtocolAgent ─────────────────────────────────────────────────────


def test_gossip_protocol_sync():
    agent = GossipProtocolAgent(
        name="gp",
        sub_agents=[_Stub(name="a", response="info A"), _Stub(name="b", response="info B")],
        n_rounds=2,
        result_key="gp_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_gossip_protocol_error_too_few():
    agent = GossipProtocolAgent(
        name="gp",
        sub_agents=[_Stub(name="a", response="solo")],
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    errs = [e for e in evs if e.event_type == EventType.ERROR]
    assert len(errs) >= 1


# ── AuctionAgent ─────────────────────────────────────────────────────────────


def test_auction_sync():
    agent = AuctionAgent(
        name="auc",
        sub_agents=[
            _Stub(name="short", response="hi"),
            _Stub(name="long", response="hello world long response"),
        ],
        bid_fn=lambda _n, r: float(len(r)),
        result_key="auc_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── DelphiMethodAgent ───────────────────────────────────────────────────────


def test_delphi_method_sync():
    agent = DelphiMethodAgent(
        name="dm",
        experts=[_Stub(name="e1", response="opinion1"), _Stub(name="e2", response="opinion2")],
        facilitator=_Stub(name="fac", response="consensus"),
        n_rounds=2,
        result_key="dm_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_delphi_method_async():
    agent = DelphiMethodAgent(
        name="dm",
        experts=[_Stub(name="e1", response="a"), _Stub(name="e2", response="b")],
        facilitator=_Stub(name="f", response="ok"),
        n_rounds=1,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── NominalGroupAgent ───────────────────────────────────────────────────────


def test_nominal_group_sync():
    agent = NominalGroupAgent(
        name="ng",
        sub_agents=[
            _Stub(name="a", response="idea A"),
            _Stub(name="b", response="idea B"),
            _NumberStub(name="c", value="1, 2"),
        ],
        n_top=2,
        result_key="ng_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── ActiveRetrievalAgent ────────────────────────────────────────────────────


def test_active_retrieval_sync():
    agent = ActiveRetrievalAgent(
        name="ar",
        agent=_Stub(name="a", response="answer"),
        retriever=_Stub(name="r", response="extra context"),
        confidence_fn=lambda _: 0.9,
        threshold=0.5,
        result_key="ar_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_active_retrieval_low_confidence():
    call_count = {"n": 0}

    def conf_fn(_):
        call_count["n"] += 1
        return 0.3 if call_count["n"] <= 1 else 0.9

    agent = ActiveRetrievalAgent(
        name="ar",
        agent=_Stub(name="a", response="improved"),
        retriever=_Stub(name="r", response="more info"),
        confidence_fn=conf_fn,
        threshold=0.5,
        max_retrievals=2,
    )
    evs = _collect(agent._run_impl_traced(_ctx()))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── IterativeRetrievalAgent ─────────────────────────────────────────────────


def test_iterative_retrieval_sync():
    agent = IterativeRetrievalAgent(
        name="ir",
        agent=_Stub(name="a", response="step result"),
        retriever=_Stub(name="r", response="evidence"),
        n_steps=2,
        result_key="ir_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── PromptChainAgent ────────────────────────────────────────────────────────


def test_prompt_chain_sync():
    agent = PromptChainAgent(
        name="pc",
        agent=_Stub(name="a", response="chained output"),
        prompts=[
            "Summarise: {input}",
            "Refine: {previous}",
        ],
        result_key="pc_res",
    )
    ctx = _ctx("some text")
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_prompt_chain_async():
    agent = PromptChainAgent(
        name="pc",
        agent=_Stub(name="a", response="ok"),
        prompts=["{input}"],
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── HypothesisTestingAgent ──────────────────────────────────────────────────


def test_hypothesis_testing_sync():
    agent = HypothesisTestingAgent(
        name="ht",
        agent=_Stub(name="a", response="hypothesis + refine"),
        tester=_Stub(name="t", response="1: survives, 2: falsified"),
        n_hypotheses=2,
        result_key="ht_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── SkillLibraryAgent ───────────────────────────────────────────────────────


def test_skill_library_fresh_solve():
    agent = SkillLibraryAgent(
        name="sl",
        agent=_Stub(name="a", response="new skill answer"),
        result_key="sl_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert len(agent.skills) >= 1


def test_skill_library_reuse():
    agent = SkillLibraryAgent(
        name="sl",
        agent=_DynamicStub(name="a", responses=["1", "adapted"]),
        initial_skills=[{"name": "greet", "description": "hello task", "answer": "hi there"}],
    )
    evs = _collect(agent._run_impl_traced(_ctx("say hello")))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── RecursiveCriticAgent ────────────────────────────────────────────────────


def test_recursive_critic_sync():
    agent = RecursiveCriticAgent(
        name="rc",
        agent=_Stub(name="a", response="critiqued answer"),
        depth=2,
        result_key="rc_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert ctx.session.state["rc_res"]["depth"] == 2


def test_recursive_critic_async():
    agent = RecursiveCriticAgent(
        name="rc",
        agent=_Stub(name="a", response="deep critique"),
        depth=1,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── DemonstrateSearchPredictAgent ───────────────────────────────────────────


def test_dsp_sync():
    agent = DemonstrateSearchPredictAgent(
        name="dsp",
        agent=_Stub(name="a", response="predicted"),
        retriever=_Stub(name="r", response="passages"),
        n_demos=2,
        result_key="dsp_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


def test_dsp_async():
    agent = DemonstrateSearchPredictAgent(
        name="dsp",
        agent=_Stub(name="a", response="ans"),
        retriever=_Stub(name="r", response="info"),
        n_demos=1,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── DoubleLoopLearningAgent ─────────────────────────────────────────────────


def test_double_loop_learning_sync_good():
    agent = DoubleLoopLearningAgent(
        name="dll",
        agent=_Stub(name="a", response="good answer"),
        quality_fn=lambda _: 0.9,
        threshold=0.7,
        result_key="dll_res",
    )
    ctx = _ctx()
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert ctx.session.state["dll_res"]["loops"] == 1


def test_double_loop_learning_async():
    agent = DoubleLoopLearningAgent(
        name="dll",
        agent=_Stub(name="a", response="ok"),
        quality_fn=lambda _: 0.8,
        threshold=0.5,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── AgendaAgent ─────────────────────────────────────────────────────────────


def test_agenda_sync():
    agent = AgendaAgent(
        name="ag",
        agent=_Stub(name="a", response="sub-goal resolved"),
        max_steps=5,
        result_key="ag_res",
    )
    ctx = _ctx("build a house")
    evs = _collect(agent._run_impl_traced(ctx))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    assert "ag_res" in ctx.session.state


def test_agenda_async():
    agent = AgendaAgent(
        name="ag",
        agent=_Stub(name="a", response="done"),
        max_steps=3,
    )
    evs = asyncio.run(_acollect(agent._run_async_impl_traced(_ctx())))
    msgs = [e for e in evs if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1


# ── Error handling: no-agent cases ──────────────────────────────────────────


def test_no_agent_errors():
    """Agents that require an agent/sub_agents return ERROR when None."""
    agents_to_test = [
        SimulatedAnnealingAgent(name="sa"),
        TabuSearchAgent(name="ts"),
        AnalogicalReasoningAgent(name="ar"),
        ThreadOfThoughtAgent(name="thot"),
        ExpertPromptingAgent(name="ep"),
        BufferOfThoughtsAgent(name="bot"),
        ChainOfAbstractionAgent(name="coa"),
        ProgOfThoughtAgent(name="pot"),
        InnerMonologueAgent(name="im"),
        PromptChainAgent(name="pc"),
        RecursiveCriticAgent(name="rc"),
        DoubleLoopLearningAgent(name="dll"),
        AgendaAgent(name="ag"),
    ]
    for agent in agents_to_test:
        evs = _collect(agent._run_impl_traced(_ctx()))
        errs = [e for e in evs if e.event_type == EventType.ERROR]
        assert len(errs) >= 1, f"{agent.name} should error without agent"
