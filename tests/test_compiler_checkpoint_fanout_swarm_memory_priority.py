"""Tests for CompilerAgent, CheckpointableAgent, DynamicFanOutAgent,
SwarmAgent, MemoryConsolidationAgent, PriorityQueueAgent.

Run:
    python -m pytest tests/test_compiler_checkpoint_fanout_swarm_memory_priority.py -v
"""

import asyncio
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
    CheckpointableAgent,
    CompilerAgent,
    DynamicFanOutAgent,
    MemoryConsolidationAgent,
    PriorityQueueAgent,
    SwarmAgent,
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


class EchoAgent(BaseAgent):
    """Echoes user_message."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description="echo")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo:{ctx.user_message}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo:{ctx.user_message}")


class AppendAgent(BaseAgent):
    """Appends a suffix to user_message."""

    def __init__(self, *, name: str, suffix: str) -> None:
        super().__init__(name=name, description="append")
        self.suffix = suffix

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{ctx.user_message}{self.suffix}",
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{ctx.user_message}{self.suffix}",
        )


class CountingAgent(BaseAgent):
    """Counts calls and tracks order."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description="counting")
        self.call_count = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.call_count += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{self.name}:{self.call_count}",
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self.call_count += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{self.name}:{self.call_count}",
        )


class HandoffAgent(BaseAgent):
    """Sets __next_agent__ in session.state to trigger swarm handoff."""

    def __init__(self, *, name: str, next_agent: str = "",
                 done: bool = False, response: str = "handled") -> None:
        super().__init__(name=name, description="handoff")
        self.next_agent = next_agent
        self.done = done
        self.response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if self.next_agent:
            ctx.session.state["__next_agent__"] = self.next_agent
        if self.done:
            ctx.session.state["__done__"] = True
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        if self.next_agent:
            ctx.session.state["__next_agent__"] = self.next_agent
        if self.done:
            ctx.session.state["__done__"] = True
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)


class InstructionAgent(BaseAgent):
    """Agent with modifiable instruction attribute."""

    def __init__(self, *, name: str, instruction: str = "") -> None:
        super().__init__(name=name, description="instruction")
        self.instruction = instruction

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        # Return the instruction as the response (for testing CompilerAgent)
        yield Event(EventType.AGENT_MESSAGE, self.name, self.instruction)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.instruction)


def _ctx(msg: str = "test question") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── CheckpointableAgent ──────────────────────────────────────────────────────

def test_checkpoint_basic():
    """Runs all sub-agents and checkpoints each step."""
    a = AppendAgent(name="a", suffix="+A")
    b = AppendAgent(name="b", suffix="+B")

    ckpt = CheckpointableAgent(
        name="ckpt",
        sub_agents=[a, b],
        checkpoint_key="ckpt_data",
        result_key="ckpt_result",
    )
    ctx = _ctx("start")
    ckpt.run(ctx)

    result = ctx.session.state["ckpt_result"]
    assert result["completed"] is True
    assert result["total_steps"] == 2
    assert len(result["outputs"]) == 2


def test_checkpoint_resume():
    """Resume from a checkpoint skips completed steps."""
    a = CountingAgent(name="a")
    b = CountingAgent(name="b")
    c = CountingAgent(name="c")

    ckpt = CheckpointableAgent(
        name="ckpt",
        sub_agents=[a, b, c],
        checkpoint_key="ckpt_data",
    )

    # Simulate: first 2 steps already complete
    session = Session()
    session.state["ckpt_data"] = {
        "completed_step": 2,
        "last_output": "previous_output",
        "outputs": ["out1", "out2"],
        "total_steps": 3,
    }
    ctx = InvocationContext(session=session, user_message="question")
    ckpt.run(ctx)

    # Only step 3 (agent 'c') should have run
    assert a.call_count == 0
    assert b.call_count == 0
    assert c.call_count == 1


def test_checkpoint_empty():
    """Error when no sub-agents configured."""
    ckpt = CheckpointableAgent(name="empty", sub_agents=[])
    ctx = _ctx()
    events = list(ckpt._run_impl(ctx))
    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_checkpoint_async():
    """Async checkpoint works."""
    a = AppendAgent(name="a", suffix="+A")
    b = AppendAgent(name="b", suffix="+B")

    ckpt = CheckpointableAgent(
        name="ckpt",
        sub_agents=[a, b],
        checkpoint_key="ckpt_data",
        result_key="ckpt_result",
    )
    ctx = _ctx("start")
    asyncio.run(ckpt.run_async(ctx))

    result = ctx.session.state["ckpt_result"]
    assert result["completed"] is True


# ── SwarmAgent ────────────────────────────────────────────────────────────────

def test_swarm_basic_handoff():
    """Agent A hands off to agent B, B finishes."""
    triage = HandoffAgent(name="triage", next_agent="billing", response="triaged")
    billing = HandoffAgent(name="billing", done=True, response="billed")

    swarm = SwarmAgent(
        name="swarm",
        sub_agents=[triage, billing],
        initial_agent="triage",
        result_key="swarm_result",
    )
    ctx = _ctx("help me")
    swarm.run(ctx)

    result = ctx.session.state["swarm_result"]
    assert result["chain"] == ["triage", "billing"]
    assert result["handoffs"] == 1


def test_swarm_no_handoff():
    """Single agent that doesn't hand off — chain has one entry."""
    solo = HandoffAgent(name="solo", response="done")

    swarm = SwarmAgent(
        name="swarm",
        sub_agents=[solo],
        initial_agent="solo",
        result_key="swarm_result",
    )
    ctx = _ctx("hello")
    swarm.run(ctx)

    result = ctx.session.state["swarm_result"]
    assert result["chain"] == ["solo"]
    assert result["handoffs"] == 0


def test_swarm_context_variables():
    """Context variables are included in the message."""
    solo = HandoffAgent(name="solo", response="done")

    swarm = SwarmAgent(
        name="swarm",
        sub_agents=[solo],
        initial_agent="solo",
        context_variables={"tier": "premium", "lang": "en"},
    )
    ctx = _ctx("help")
    swarm.run(ctx)
    # Context should be stored back
    assert ctx.session.state["swarm_context"]["tier"] == "premium"


def test_swarm_max_handoffs():
    """Stops after max_handoffs."""
    # Create agents that chain forever
    a = HandoffAgent(name="a", next_agent="b", response="a")
    b = HandoffAgent(name="b", next_agent="a", response="b")

    swarm = SwarmAgent(
        name="swarm",
        sub_agents=[a, b],
        initial_agent="a",
        max_handoffs=4,
        result_key="swarm_result",
    )
    ctx = _ctx("loop")
    swarm.run(ctx)

    result = ctx.session.state["swarm_result"]
    assert len(result["chain"]) <= 4


def test_swarm_async():
    """Async swarm works."""
    triage = HandoffAgent(name="triage", next_agent="billing", response="triaged")
    billing = HandoffAgent(name="billing", done=True, response="billed")

    swarm = SwarmAgent(
        name="swarm",
        sub_agents=[triage, billing],
        initial_agent="triage",
        result_key="swarm_result",
    )
    ctx = _ctx("help")
    asyncio.run(swarm.run_async(ctx))

    result = ctx.session.state["swarm_result"]
    assert result["chain"] == ["triage", "billing"]


# ── MemoryConsolidationAgent ─────────────────────────────────────────────────

def test_memory_no_consolidation():
    """Below threshold, no consolidation happens."""
    main = FixedAgent(name="main", response="output")
    summarizer = FixedAgent(name="summarizer", response="summary")

    mc = MemoryConsolidationAgent(
        name="mc",
        main_agent=main,
        summarizer_agent=summarizer,
        event_threshold=100,
        result_key="mc_result",
    )
    ctx = _ctx("hello")
    mc.run(ctx)

    result = ctx.session.state["mc_result"]
    assert result["consolidated"] is False


def test_memory_consolidation_triggered():
    """Above threshold, consolidation fires."""
    main = FixedAgent(name="main", response="output")
    summarizer = FixedAgent(name="summarizer", response="summary of history")

    mc = MemoryConsolidationAgent(
        name="mc",
        main_agent=main,
        summarizer_agent=summarizer,
        event_threshold=3,
        keep_recent=1,
        memory_key="mem",
        result_key="mc_result",
    )

    session = Session()
    # Add enough events to exceed threshold
    for i in range(5):
        session.add_event(Event(EventType.AGENT_MESSAGE, "old", f"msg {i}"))

    ctx = InvocationContext(session=session, user_message="current")
    mc.run(ctx)

    assert ctx.session.state.get("mem") is not None


def test_memory_async():
    """Async consolidation works."""
    main = FixedAgent(name="main", response="output")
    summarizer = FixedAgent(name="summarizer", response="summary")

    mc = MemoryConsolidationAgent(
        name="mc",
        main_agent=main,
        summarizer_agent=summarizer,
        event_threshold=100,
        result_key="mc_result",
    )
    ctx = _ctx("hello")
    asyncio.run(mc.run_async(ctx))

    assert ctx.session.state["mc_result"]["consolidated"] is False


# ── PriorityQueueAgent ────────────────────────────────────────────────────────

def test_priority_order():
    """Agents execute in priority order."""
    urgent = CountingAgent(name="urgent")
    normal = CountingAgent(name="normal")
    background = CountingAgent(name="background")

    pq = PriorityQueueAgent(
        name="pq",
        sub_agents=[background, urgent, normal],  # unsorted
        priority_map={"urgent": 0, "normal": 1, "background": 2},
        result_key="pq_result",
    )
    ctx = _ctx("work")
    pq.run(ctx)

    result = ctx.session.state["pq_result"]
    order = result["execution_order"]
    assert order.index("urgent") < order.index("normal")
    assert order.index("normal") < order.index("background")


def test_priority_stop_condition():
    """Stop condition halts processing before lower priorities."""
    urgent = FixedAgent(name="urgent", response="done")
    normal = CountingAgent(name="normal")

    pq = PriorityQueueAgent(
        name="pq",
        sub_agents=[urgent, normal],
        priority_map={"urgent": 0, "normal": 1},
        stop_condition=lambda state: True,  # always stop
        result_key="pq_result",
    )
    ctx = _ctx("work")
    pq.run(ctx)

    # Stop fires before even the first priority
    result = ctx.session.state["pq_result"]
    assert result["total"] == 0


def test_priority_same_level_parallel():
    """Agents at the same priority level run concurrently."""
    a = FixedAgent(name="a", response="A")
    b = FixedAgent(name="b", response="B")
    c = FixedAgent(name="c", response="C")

    pq = PriorityQueueAgent(
        name="pq",
        sub_agents=[a, b, c],
        priority_map={"a": 0, "b": 0, "c": 1},
        result_key="pq_result",
    )
    ctx = _ctx("work")
    pq.run(ctx)

    result = ctx.session.state["pq_result"]
    # a and b should both run before c
    assert "c" in result["execution_order"]
    assert result["total"] == 3


def test_priority_empty():
    """Error when no sub-agents."""
    pq = PriorityQueueAgent(name="empty", sub_agents=[])
    ctx = _ctx()
    events = list(pq._run_impl(ctx))
    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_priority_async():
    """Async priority queue works."""
    urgent = CountingAgent(name="urgent")
    normal = CountingAgent(name="normal")

    pq = PriorityQueueAgent(
        name="pq",
        sub_agents=[normal, urgent],
        priority_map={"urgent": 0, "normal": 1},
        result_key="pq_result",
    )
    ctx = _ctx("work")
    asyncio.run(pq.run_async(ctx))

    result = ctx.session.state["pq_result"]
    assert result["total"] == 2


# ── CompilerAgent ─────────────────────────────────────────────────────────────

def test_compiler_no_examples():
    """Error when no examples provided."""
    target = InstructionAgent(name="target", instruction="be helpful")

    compiler = CompilerAgent(
        name="compiler",
        target_agent=target,
        examples=[],
    )
    ctx = _ctx()
    events = list(compiler._run_impl(ctx))
    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_compiler_perfect_score_no_iteration():
    """If first iteration is perfect, no LLM call needed."""
    target = FixedAgent(name="target", response="4")
    target.instruction = "solve math"

    compiler = CompilerAgent(
        name="compiler",
        target_agent=target,
        examples=[{"input": "2+2", "expected": "4"}],
        metric_fn=lambda o, e: 1.0 if e in o else 0.0,
        max_iterations=5,
        result_key="compiler_result",
    )
    ctx = _ctx()
    compiler.run(ctx)

    result = ctx.session.state["compiler_result"]
    assert result["best_score"] == 1.0
    # Should stop after 1 iteration
    assert len(result["iterations"]) == 1


# ── DynamicFanOutAgent (mock LLM) ────────────────────────────────────────────

def test_dynamic_fanout_with_mock_service():
    """DynamicFanOutAgent with mocked service decomposes and reduces."""

    class MockService:
        """Returns a fixed JSON array for decomposition."""
        def generate_completion(self, **kwargs: Any) -> str:
            return '["task A", "task B", "task C"]'

    worker = EchoAgent(name="worker")
    reducer = FixedAgent(name="reducer", response="combined result")

    fanout = DynamicFanOutAgent(
        name="fanout",
        worker_agent=worker,
        reducer_agent=reducer,
        max_items=5,
        result_key="fanout_result",
    )
    # Inject mock service
    fanout._service = MockService()

    ctx = _ctx("do research")
    events = list(fanout._run_impl(ctx))

    result = ctx.session.state["fanout_result"]
    assert result["count"] == 3
    assert len(result["items"]) == 3
    # Check worker processed items
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(messages) >= 1  # At least the reducer output


def test_dynamic_fanout_empty_decomposition():
    """Error when decomposition returns empty list."""

    class MockService:
        def generate_completion(self, **kwargs: Any) -> str:
            return "not valid json"

    worker = EchoAgent(name="worker")
    reducer = FixedAgent(name="reducer", response="combined")

    fanout = DynamicFanOutAgent(
        name="fanout",
        worker_agent=worker,
        reducer_agent=reducer,
    )
    fanout._service = MockService()

    ctx = _ctx("do research")
    events = list(fanout._run_impl(ctx))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) >= 1


def test_dynamic_fanout_async():
    """Async DynamicFanOutAgent with mock service."""

    class MockService:
        def generate_completion(self, **kwargs: Any) -> str:
            return '["item1", "item2"]'

    worker = EchoAgent(name="worker")
    reducer = FixedAgent(name="reducer", response="combined")

    fanout = DynamicFanOutAgent(
        name="fanout",
        worker_agent=worker,
        reducer_agent=reducer,
        result_key="fanout_result",
    )
    fanout._service = MockService()

    ctx = _ctx("research topic")
    asyncio.run(fanout.run_async(ctx))

    result = ctx.session.state["fanout_result"]
    assert result["count"] == 2
