"""Tests for MapReduceAgent, ConsensusAgent, and ProducerReviewerAgent.

Verifies:
- MapReduceAgent fans out to mappers, then reduces.
- ConsensusAgent collects votes and passes them to a judge.
- ProducerReviewerAgent iterates until approval or max iterations.
- Both sync and async paths work correctly.

Run:
    python -m pytest tests/test_mapreduce_consensus_producer_reviewer.py -v
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
    ConsensusAgent,
    MapReduceAgent,
    ProducerReviewerAgent,
)


# ── Stub agents ───────────────────────────────────────────────────────────────

class StubAgent(BaseAgent):
    """Returns a fixed response, optionally echoing the input."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description=f"stub-{name}")
        self._response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class EchoAgent(BaseAgent):
    """Echoes back the user message with a prefix."""

    def __init__(self, *, name: str, prefix: str = "") -> None:
        super().__init__(name=name, description=f"echo-{name}")
        self._prefix = prefix

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"{self._prefix}{ctx.user_message}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"{self._prefix}{ctx.user_message}")


class CountingReviewerAgent(BaseAgent):
    """Approves on the Nth call (1-indexed)."""

    def __init__(self, *, name: str, approve_on: int = 2, keyword: str = "APPROVED") -> None:
        super().__init__(name=name, description="reviewing agent")
        self._approve_on = approve_on
        self._keyword = keyword
        self._call_count = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self._call_count += 1
        if self._call_count >= self._approve_on:
            yield Event(EventType.AGENT_MESSAGE, self.name, f"{self._keyword}: looks good")
        else:
            yield Event(EventType.AGENT_MESSAGE, self.name, "Needs more detail")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self._call_count += 1
        if self._call_count >= self._approve_on:
            yield Event(EventType.AGENT_MESSAGE, self.name, f"{self._keyword}: looks good")
        else:
            yield Event(EventType.AGENT_MESSAGE, self.name, "Needs more detail")


def _ctx(msg: str = "hello") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── MapReduceAgent tests ─────────────────────────────────────────────────────

def test_mapreduce_sync_runs_mappers_and_reducer():
    """MapReduceAgent runs all mappers then the reducer."""
    m1 = StubAgent(name="mapper1", response="result-1")
    m2 = StubAgent(name="mapper2", response="result-2")
    reducer = EchoAgent(name="reducer", prefix="reduced: ")

    agent = MapReduceAgent(
        name="mr",
        sub_agents=[m1, m2],
        reduce_agent=reducer,
    )

    ctx = _ctx("query")
    events = list(agent._run_impl(ctx))
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

    assert len(messages) == 3  # 2 mappers + 1 reducer
    authors = {e.author for e in messages}
    assert "mapper1" in authors
    assert "mapper2" in authors
    assert "reducer" in authors

    reducer_msg = next(e for e in messages if e.author == "reducer")
    assert "result-1" in reducer_msg.content
    assert "result-2" in reducer_msg.content


def test_mapreduce_stores_result_key():
    """MapReduceAgent stores mapper results in session.state when result_key is set."""
    m1 = StubAgent(name="m1", response="r1")
    m2 = StubAgent(name="m2", response="r2")
    reducer = StubAgent(name="red", response="done")

    ctx = _ctx("go")
    agent = MapReduceAgent(
        name="mr",
        sub_agents=[m1, m2],
        reduce_agent=reducer,
        result_key="map_results",
    )

    list(agent._run_impl(ctx))
    assert "map_results" in ctx.session.state
    assert ctx.session.state["map_results"]["m1"] == "r1"
    assert ctx.session.state["map_results"]["m2"] == "r2"


def test_mapreduce_async():
    """MapReduceAgent works in async mode."""
    m1 = StubAgent(name="a1", response="async-1")
    m2 = StubAgent(name="a2", response="async-2")
    reducer = EchoAgent(name="red", prefix="merged: ")

    agent = MapReduceAgent(
        name="mr_async",
        sub_agents=[m1, m2],
        reduce_agent=reducer,
    )

    async def _collect():
        return [e async for e in agent._run_async_impl(_ctx("async_query"))]

    events = asyncio.run(_collect())
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

    assert len(messages) == 3
    reducer_msg = next(e for e in messages if e.author == "red")
    assert "async-1" in reducer_msg.content
    assert "async-2" in reducer_msg.content


# ── ConsensusAgent tests ──────────────────────────────────────────────────────

def test_consensus_sync_collects_votes_and_judges():
    """ConsensusAgent runs voters in parallel then passes to judge."""
    v1 = StubAgent(name="voter1", response="answer-A")
    v2 = StubAgent(name="voter2", response="answer-B")
    judge = EchoAgent(name="judge", prefix="verdict: ")

    agent = ConsensusAgent(
        name="cons",
        sub_agents=[v1, v2],
        judge_agent=judge,
    )

    events = list(agent._run_impl(_ctx("what is 2+2?")))
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

    assert len(messages) == 3  # 2 voters + 1 judge
    judge_msg = next(e for e in messages if e.author == "judge")
    assert "answer-A" in judge_msg.content
    assert "answer-B" in judge_msg.content


def test_consensus_stores_result_key():
    """ConsensusAgent stores voter answers in session.state."""
    v1 = StubAgent(name="v1", response="yes")
    v2 = StubAgent(name="v2", response="no")
    judge = StubAgent(name="j", response="maybe")

    ctx = _ctx("question")
    agent = ConsensusAgent(
        name="cons",
        sub_agents=[v1, v2],
        judge_agent=judge,
        result_key="votes",
    )

    list(agent._run_impl(ctx))
    assert ctx.session.state["votes"]["v1"] == "yes"
    assert ctx.session.state["votes"]["v2"] == "no"


def test_consensus_async():
    """ConsensusAgent works in async mode."""
    v1 = StubAgent(name="va1", response="opt-1")
    v2 = StubAgent(name="va2", response="opt-2")
    judge = EchoAgent(name="judge", prefix="final: ")

    agent = ConsensusAgent(
        name="cons_async",
        sub_agents=[v1, v2],
        judge_agent=judge,
    )

    async def _collect():
        return [e async for e in agent._run_async_impl(_ctx("async question"))]

    events = asyncio.run(_collect())
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

    assert len(messages) == 3
    judge_msg = next(e for e in messages if e.author == "judge")
    assert "opt-1" in judge_msg.content


# ── ProducerReviewerAgent tests ──────────────────────────────────────────────

def test_producer_reviewer_stops_on_approval():
    """ProducerReviewerAgent stops when reviewer includes approval keyword."""
    producer = StubAgent(name="writer", response="draft content")
    reviewer = CountingReviewerAgent(name="editor", approve_on=2)

    agent = ProducerReviewerAgent(
        name="pr",
        producer=producer,
        reviewer=reviewer,
        max_iterations=5,
    )

    events = list(agent._run_impl(_ctx("write a blog")))
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    states = [e for e in events if e.event_type == EventType.STATE_UPDATE]

    # Should have run 2 iterations: iter1 rejected, iter2 approved
    assert len(states) == 2
    assert len(messages) == 4  # produce + review + produce + review
    assert "APPROVED" in messages[-1].content


def test_producer_reviewer_respects_max_iterations():
    """ProducerReviewerAgent stops at max_iterations even without approval."""
    producer = StubAgent(name="writer", response="draft")
    reviewer = CountingReviewerAgent(name="editor", approve_on=99)

    agent = ProducerReviewerAgent(
        name="pr",
        producer=producer,
        reviewer=reviewer,
        max_iterations=2,
    )

    events = list(agent._run_impl(_ctx("write")))
    states = [e for e in events if e.event_type == EventType.STATE_UPDATE]

    assert len(states) == 2


def test_producer_reviewer_async_stops_on_approval():
    """Async ProducerReviewerAgent stops when reviewer approves."""
    producer = StubAgent(name="writer", response="async draft")
    reviewer = CountingReviewerAgent(name="editor", approve_on=1)

    agent = ProducerReviewerAgent(
        name="pr_async",
        producer=producer,
        reviewer=reviewer,
        max_iterations=5,
    )

    async def _collect():
        return [e async for e in agent._run_async_impl(_ctx("write async"))]

    events = asyncio.run(_collect())
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    states = [e for e in events if e.event_type == EventType.STATE_UPDATE]

    # Approved on first iteration
    assert len(states) == 1
    assert "APPROVED" in messages[-1].content


def test_producer_reviewer_custom_keyword():
    """ProducerReviewerAgent uses a custom approval keyword."""
    producer = StubAgent(name="writer", response="content")
    reviewer = CountingReviewerAgent(name="editor", approve_on=1, keyword="LGTM")

    agent = ProducerReviewerAgent(
        name="pr_custom",
        producer=producer,
        reviewer=reviewer,
        max_iterations=3,
        approval_keyword="LGTM",
    )

    events = list(agent._run_impl(_ctx("write")))
    messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

    assert "LGTM" in messages[-1].content


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_mapreduce_sync_runs_mappers_and_reducer()
    print("PASS: test_mapreduce_sync_runs_mappers_and_reducer")

    test_mapreduce_stores_result_key()
    print("PASS: test_mapreduce_stores_result_key")

    test_mapreduce_async()
    print("PASS: test_mapreduce_async")

    test_consensus_sync_collects_votes_and_judges()
    print("PASS: test_consensus_sync_collects_votes_and_judges")

    test_consensus_stores_result_key()
    print("PASS: test_consensus_stores_result_key")

    test_consensus_async()
    print("PASS: test_consensus_async")

    test_producer_reviewer_stops_on_approval()
    print("PASS: test_producer_reviewer_stops_on_approval")

    test_producer_reviewer_respects_max_iterations()
    print("PASS: test_producer_reviewer_respects_max_iterations")

    test_producer_reviewer_async_stops_on_approval()
    print("PASS: test_producer_reviewer_async_stops_on_approval")

    test_producer_reviewer_custom_keyword()
    print("PASS: test_producer_reviewer_custom_keyword")

    print("\nAll tests passed!")
