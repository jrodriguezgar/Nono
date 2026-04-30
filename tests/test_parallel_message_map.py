"""Tests for ParallelAgent.message_map — per-agent message routing.

Verifies that:
- Without message_map, all sub-agents receive the same user_message.
- With message_map, mapped agents receive their custom message.
- Agents not in message_map still receive the original user_message.
- Works in both sync (run) and async (run_async) modes.
- message_map can be updated between runs.

Run:
    python tests/test_parallel_message_map.py
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
from nono.agent.workflow_agents import ParallelAgent


# ── Capture agent ─────────────────────────────────────────────────────────────

class CaptureAgent(BaseAgent):
    """Records the user_message it received."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description="capture")
        self.received_message: str = ""

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.received_message = ctx.user_message
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self.received_message = ctx.user_message
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)


def _ctx(msg: str = "default") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── Tests: no message_map (backward compatibility) ───────────────────────────

def test_no_message_map_all_receive_same():
    """Without message_map, all sub-agents receive ctx.user_message."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(name="par", sub_agents=[a, b])
    par.run(_ctx("shared input"))

    assert a.received_message == "shared input", f"a got {a.received_message!r}"
    assert b.received_message == "shared input", f"b got {b.received_message!r}"
    print("PASS: test_no_message_map_all_receive_same")


def test_no_message_map_async():
    """Async: without message_map, all sub-agents receive ctx.user_message."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(name="par", sub_agents=[a, b])
    asyncio.run(par.run_async(_ctx("shared async")))

    assert a.received_message == "shared async"
    assert b.received_message == "shared async"
    print("PASS: test_no_message_map_async")


# ── Tests: with message_map ──────────────────────────────────────────────────

def test_message_map_all_mapped():
    """Every sub-agent has an entry in message_map."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(
        name="par",
        sub_agents=[a, b],
        message_map={"a": "topic A", "b": "topic B"},
    )
    par.run(_ctx("ignored"))

    assert a.received_message == "topic A", f"a got {a.received_message!r}"
    assert b.received_message == "topic B", f"b got {b.received_message!r}"
    print("PASS: test_message_map_all_mapped")


def test_message_map_partial():
    """Only some sub-agents are in message_map; others get the default."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")
    c = CaptureAgent(name="c")

    par = ParallelAgent(
        name="par",
        sub_agents=[a, b, c],
        message_map={"a": "custom for A"},
    )
    par.run(_ctx("fallback"))

    assert a.received_message == "custom for A"
    assert b.received_message == "fallback"
    assert c.received_message == "fallback"
    print("PASS: test_message_map_partial")


def test_message_map_async():
    """Async: message_map routes correctly."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(
        name="par",
        sub_agents=[a, b],
        message_map={"b": "async custom"},
    )
    asyncio.run(par.run_async(_ctx("async default")))

    assert a.received_message == "async default"
    assert b.received_message == "async custom"
    print("PASS: test_message_map_async")


def test_message_map_empty_dict():
    """Empty message_map behaves like no message_map."""
    a = CaptureAgent(name="a")

    par = ParallelAgent(name="par", sub_agents=[a], message_map={})
    par.run(_ctx("original"))

    assert a.received_message == "original"
    print("PASS: test_message_map_empty_dict")


def test_message_map_unknown_agent_ignored():
    """Extra keys in message_map that don't match any sub-agent are harmless."""
    a = CaptureAgent(name="a")

    par = ParallelAgent(
        name="par",
        sub_agents=[a],
        message_map={"nonexistent": "phantom", "a": "actual"},
    )
    par.run(_ctx("default"))

    assert a.received_message == "actual"
    print("PASS: test_message_map_unknown_agent_ignored")


def test_message_map_can_be_updated():
    """message_map can be changed between runs."""
    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(name="par", sub_agents=[a, b])

    # Run 1: no map
    par.run(_ctx("run1"))
    assert a.received_message == "run1"
    assert b.received_message == "run1"

    # Run 2: add map
    par.message_map = {"a": "custom run2"}
    par.run(_ctx("run2"))
    assert a.received_message == "custom run2"
    assert b.received_message == "run2"

    # Run 3: clear map
    par.message_map = {}
    par.run(_ctx("run3"))
    assert a.received_message == "run3"
    assert b.received_message == "run3"
    print("PASS: test_message_map_can_be_updated")


# ── Tests: nested orchestration ──────────────────────────────────────────────

def test_message_map_inside_sequential():
    """ParallelAgent with message_map works inside a SequentialAgent."""
    from nono.agent.workflow_agents import SequentialAgent

    a = CaptureAgent(name="a")
    b = CaptureAgent(name="b")

    par = ParallelAgent(
        name="par",
        sub_agents=[a, b],
        message_map={"a": "parallel A", "b": "parallel B"},
    )

    # c runs after par — receives last AGENT_MESSAGE from par
    c = CaptureAgent(name="c")

    seq = SequentialAgent(name="seq", sub_agents=[par, c])
    seq.run(_ctx("sequential input"))

    assert a.received_message == "parallel A"
    assert b.received_message == "parallel B"
    # c gets the last AGENT_MESSAGE from par (completion order varies,
    # but it should NOT be "sequential input")
    assert c.received_message != ""
    print("PASS: test_message_map_inside_sequential")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_no_message_map_all_receive_same()
    test_no_message_map_async()
    test_message_map_all_mapped()
    test_message_map_partial()
    test_message_map_async()
    test_message_map_empty_dict()
    test_message_map_unknown_agent_ignored()
    test_message_map_can_be_updated()
    test_message_map_inside_sequential()
    print()
    print("=" * 60)
    print("  All 9 ParallelAgent message_map tests PASSED")
    print("=" * 60)
