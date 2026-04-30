"""Tests for ParallelAgent.result_key — automatic result collection.

Verifies that:
- Without result_key, no state entry is written.
- With result_key, session.state[key] contains {agent_name: response} for
  every sub-agent that produced an AGENT_MESSAGE.
- Works in both sync (run) and async (run_async) modes.
- Combines correctly with message_map.
- In a SequentialAgent pipeline, the next agent can read the collected dict.
- When a sub-agent emits multiple AGENT_MESSAGE events, only the last is kept.

Run:
    python tests/test_parallel_result_key.py
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
from nono.agent.workflow_agents import ParallelAgent, SequentialAgent


# ── Helper agents ─────────────────────────────────────────────────────────────

class EchoAgent(BaseAgent):
    """Echoes the user_message as an AGENT_MESSAGE."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description="echo")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo:{ctx.user_message}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo:{ctx.user_message}")


class FixedAgent(BaseAgent):
    """Always returns a fixed response."""

    def __init__(self, *, name: str, response: str) -> None:
        super().__init__(name=name, description="fixed")
        self.response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.response)


class MultiMessageAgent(BaseAgent):
    """Yields multiple AGENT_MESSAGE events."""

    def __init__(self, *, name: str, messages: list[str]) -> None:
        super().__init__(name=name, description="multi")
        self.messages = messages

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        for m in self.messages:
            yield Event(EventType.AGENT_MESSAGE, self.name, m)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        for m in self.messages:
            yield Event(EventType.AGENT_MESSAGE, self.name, m)


class StateReaderAgent(BaseAgent):
    """Reads a key from session.state and stores it locally."""

    def __init__(self, *, name: str, state_key: str) -> None:
        super().__init__(name=name, description="reader")
        self.state_key = state_key
        self.captured_value: object = None

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.captured_value = ctx.session.state.get(self.state_key)
        yield Event(EventType.AGENT_MESSAGE, self.name, str(self.captured_value))

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self.captured_value = ctx.session.state.get(self.state_key)
        yield Event(EventType.AGENT_MESSAGE, self.name, str(self.captured_value))


def _ctx(msg: str = "default") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── Tests: no result_key (backward compatibility) ────────────────────────────

def test_no_result_key_no_state():
    """Without result_key, session.state is not modified by ParallelAgent."""
    a = FixedAgent(name="a", response="hello")
    b = FixedAgent(name="b", response="world")

    par = ParallelAgent(name="par", sub_agents=[a, b])
    ctx = _ctx()
    par.run(ctx)

    assert "parallel_results" not in ctx.session.state
    print("PASS: test_no_result_key_no_state")


# ── Tests: result_key collects responses ─────────────────────────────────────

def test_result_key_collects_all():
    """result_key writes {agent_name: response} for every sub-agent."""
    a = FixedAgent(name="a", response="hello")
    b = FixedAgent(name="b", response="world")

    par = ParallelAgent(
        name="par", sub_agents=[a, b], result_key="results",
    )
    ctx = _ctx()
    par.run(ctx)

    collected = ctx.session.state["results"]
    assert collected["a"] == "hello", f"a → {collected.get('a')!r}"
    assert collected["b"] == "world", f"b → {collected.get('b')!r}"
    assert len(collected) == 2
    print("PASS: test_result_key_collects_all")


def test_result_key_async():
    """Async: result_key works the same way."""
    a = FixedAgent(name="a", response="async-a")
    b = FixedAgent(name="b", response="async-b")

    par = ParallelAgent(
        name="par", sub_agents=[a, b], result_key="async_results",
    )
    ctx = _ctx()
    asyncio.run(par.run_async(ctx))

    collected = ctx.session.state["async_results"]
    assert collected["a"] == "async-a"
    assert collected["b"] == "async-b"
    print("PASS: test_result_key_async")


def test_result_key_with_message_map():
    """result_key and message_map work together."""
    a = EchoAgent(name="a")
    b = EchoAgent(name="b")

    par = ParallelAgent(
        name="par",
        sub_agents=[a, b],
        message_map={"a": "topic A", "b": "topic B"},
        result_key="combo",
    )
    ctx = _ctx("ignored")
    par.run(ctx)

    collected = ctx.session.state["combo"]
    assert collected["a"] == "echo:topic A"
    assert collected["b"] == "echo:topic B"
    print("PASS: test_result_key_with_message_map")


def test_result_key_multi_message_keeps_last():
    """When a sub-agent emits multiple AGENT_MESSAGE events, only the last
    content is stored in the collected dict."""
    m = MultiMessageAgent(name="m", messages=["first", "second", "third"])

    par = ParallelAgent(name="par", sub_agents=[m], result_key="multi")
    ctx = _ctx()
    par.run(ctx)

    assert ctx.session.state["multi"]["m"] == "third"
    print("PASS: test_result_key_multi_message_keeps_last")


def test_result_key_multi_message_async():
    """Async: multiple AGENT_MESSAGE events — last wins."""
    m = MultiMessageAgent(name="m", messages=["alpha", "beta"])

    par = ParallelAgent(name="par", sub_agents=[m], result_key="multi_async")
    ctx = _ctx()
    asyncio.run(par.run_async(ctx))

    assert ctx.session.state["multi_async"]["m"] == "beta"
    print("PASS: test_result_key_multi_message_async")


# ── Tests: nested in SequentialAgent ─────────────────────────────────────────

def test_result_key_readable_by_next_agent():
    """In a Sequential pipeline, the next agent can read the collected dict
    from session.state."""
    a = FixedAgent(name="a", response="data-a")
    b = FixedAgent(name="b", response="data-b")

    par = ParallelAgent(
        name="par", sub_agents=[a, b], result_key="par_out",
    )
    reader = StateReaderAgent(name="reader", state_key="par_out")

    seq = SequentialAgent(name="seq", sub_agents=[par, reader])
    ctx = _ctx("start")
    seq.run(ctx)

    assert reader.captured_value == {"a": "data-a", "b": "data-b"}, (
        f"reader got {reader.captured_value!r}"
    )
    print("PASS: test_result_key_readable_by_next_agent")


def test_result_key_readable_by_next_agent_async():
    """Async: same pipeline — next agent reads from session.state."""
    a = FixedAgent(name="a", response="async-data-a")
    b = FixedAgent(name="b", response="async-data-b")

    par = ParallelAgent(
        name="par", sub_agents=[a, b], result_key="par_out",
    )
    reader = StateReaderAgent(name="reader", state_key="par_out")

    seq = SequentialAgent(name="seq", sub_agents=[par, reader])
    ctx = _ctx("start")
    asyncio.run(seq.run_async(ctx))

    assert reader.captured_value == {"a": "async-data-a", "b": "async-data-b"}
    print("PASS: test_result_key_readable_by_next_agent_async")


def test_result_key_single_agent():
    """result_key works with a single sub-agent."""
    a = FixedAgent(name="solo", response="only one")

    par = ParallelAgent(name="par", sub_agents=[a], result_key="single")
    ctx = _ctx()
    par.run(ctx)

    assert ctx.session.state["single"] == {"solo": "only one"}
    print("PASS: test_result_key_single_agent")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_no_result_key_no_state()
    test_result_key_collects_all()
    test_result_key_async()
    test_result_key_with_message_map()
    test_result_key_multi_message_keeps_last()
    test_result_key_multi_message_async()
    test_result_key_readable_by_next_agent()
    test_result_key_readable_by_next_agent_async()
    test_result_key_single_agent()
    print()
    print("All test_parallel_result_key tests passed!")
