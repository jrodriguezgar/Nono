"""Tests proving sync/async mode propagation in orchestration agents.

Verifies that:
- Sync orchestration (run) ONLY calls _run_impl on every sub-agent.
- Async orchestration (run_async) ONLY calls _run_async_impl on every sub-agent.
- The two paths never cross (sync never triggers async, async never triggers sync).

Run:
    python tests/test_sync_async_orchestration.py
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
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
)


# ── Spy agent ─────────────────────────────────────────────────────────────────

class SpyAgent(BaseAgent):
    """Records which execution path was called: sync, async, or both."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description="spy")
        self._response = response
        self.sync_called: bool = False
        self.async_called: bool = False

    def reset(self) -> None:
        self.sync_called = False
        self.async_called = False

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.sync_called = True
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        self.async_called = True
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


def _ctx(msg: str = "hello") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── SequentialAgent ───────────────────────────────────────────────────────────

def test_sequential_sync_only_calls_run_impl():
    """SequentialAgent.run() must call _run_impl, never _run_async_impl."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")

    seq = SequentialAgent(name="seq", sub_agents=[a, b])
    seq.run(_ctx())

    assert a.sync_called is True, "a._run_impl was NOT called"
    assert a.async_called is False, "a._run_async_impl was called during sync"
    assert b.sync_called is True, "b._run_impl was NOT called"
    assert b.async_called is False, "b._run_async_impl was called during sync"
    print("PASS: test_sequential_sync_only_calls_run_impl")


def test_sequential_async_only_calls_run_async_impl():
    """SequentialAgent.run_async() must call _run_async_impl, never _run_impl."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")

    seq = SequentialAgent(name="seq", sub_agents=[a, b])
    asyncio.run(seq.run_async(_ctx()))

    assert a.async_called is True, "a._run_async_impl was NOT called"
    assert a.sync_called is False, "a._run_impl was called during async"
    assert b.async_called is True, "b._run_async_impl was NOT called"
    assert b.sync_called is False, "b._run_impl was called during async"
    print("PASS: test_sequential_async_only_calls_run_async_impl")


# ── ParallelAgent ─────────────────────────────────────────────────────────────

def test_parallel_sync_only_calls_run_impl():
    """ParallelAgent.run() must call _run_impl on all sub-agents."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")
    c = SpyAgent(name="c")

    par = ParallelAgent(name="par", sub_agents=[a, b, c])
    par.run(_ctx())

    for spy in (a, b, c):
        assert spy.sync_called is True, f"{spy.name}._run_impl was NOT called"
        assert spy.async_called is False, f"{spy.name}._run_async_impl was called during sync"
    print("PASS: test_parallel_sync_only_calls_run_impl")


def test_parallel_async_only_calls_run_async_impl():
    """ParallelAgent.run_async() must call _run_async_impl on all sub-agents."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")
    c = SpyAgent(name="c")

    par = ParallelAgent(name="par", sub_agents=[a, b, c])
    asyncio.run(par.run_async(_ctx()))

    for spy in (a, b, c):
        assert spy.async_called is True, f"{spy.name}._run_async_impl was NOT called"
        assert spy.sync_called is False, f"{spy.name}._run_impl was called during async"
    print("PASS: test_parallel_async_only_calls_run_async_impl")


# ── LoopAgent ─────────────────────────────────────────────────────────────────

def test_loop_sync_only_calls_run_impl():
    """LoopAgent.run() must call _run_impl each iteration, never _run_async_impl."""
    a = SpyAgent(name="a")

    loop = LoopAgent(name="loop", sub_agents=[a], max_iterations=2)
    loop.run(_ctx())

    assert a.sync_called is True, "a._run_impl was NOT called"
    assert a.async_called is False, "a._run_async_impl was called during sync"
    print("PASS: test_loop_sync_only_calls_run_impl")


def test_loop_async_only_calls_run_async_impl():
    """LoopAgent.run_async() must call _run_async_impl, never _run_impl."""
    a = SpyAgent(name="a")

    loop = LoopAgent(name="loop", sub_agents=[a], max_iterations=2)
    asyncio.run(loop.run_async(_ctx()))

    assert a.async_called is True, "a._run_async_impl was NOT called"
    assert a.sync_called is False, "a._run_impl was called during async"
    print("PASS: test_loop_async_only_calls_run_async_impl")


# ── Nested: Sequential(Parallel(...)) ────────────────────────────────────────

def test_nested_sync_propagates_through_tree():
    """Sync mode propagates: Sequential -> Parallel -> SpyAgents."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")
    c = SpyAgent(name="c")

    inner = ParallelAgent(name="inner_par", sub_agents=[a, b])
    outer = SequentialAgent(name="outer_seq", sub_agents=[inner, c])
    outer.run(_ctx())

    for spy in (a, b, c):
        assert spy.sync_called is True, f"{spy.name}._run_impl was NOT called"
        assert spy.async_called is False, f"{spy.name}._run_async_impl was called during sync"
    print("PASS: test_nested_sync_propagates_through_tree")


def test_nested_async_propagates_through_tree():
    """Async mode propagates: Sequential -> Parallel -> SpyAgents."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")
    c = SpyAgent(name="c")

    inner = ParallelAgent(name="inner_par", sub_agents=[a, b])
    outer = SequentialAgent(name="outer_seq", sub_agents=[inner, c])
    asyncio.run(outer.run_async(_ctx()))

    for spy in (a, b, c):
        assert spy.async_called is True, f"{spy.name}._run_async_impl was NOT called"
        assert spy.sync_called is False, f"{spy.name}._run_impl was called during async"
    print("PASS: test_nested_async_propagates_through_tree")


# ── Deep nesting: Sequential(Loop(Parallel(...))) ────────────────────────────

def test_deep_nesting_sync():
    """Sync propagates through 3 levels: Sequential -> Loop -> Parallel."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")

    par = ParallelAgent(name="par", sub_agents=[a, b])
    loop = LoopAgent(name="loop", sub_agents=[par], max_iterations=2)
    seq = SequentialAgent(name="seq", sub_agents=[loop])
    seq.run(_ctx())

    for spy in (a, b):
        assert spy.sync_called is True, f"{spy.name}._run_impl was NOT called"
        assert spy.async_called is False, f"{spy.name}._run_async_impl was called during sync"
    print("PASS: test_deep_nesting_sync")


def test_deep_nesting_async():
    """Async propagates through 3 levels: Sequential -> Loop -> Parallel."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")

    par = ParallelAgent(name="par", sub_agents=[a, b])
    loop = LoopAgent(name="loop", sub_agents=[par], max_iterations=2)
    seq = SequentialAgent(name="seq", sub_agents=[loop])
    asyncio.run(seq.run_async(_ctx()))

    for spy in (a, b):
        assert spy.async_called is True, f"{spy.name}._run_async_impl was NOT called"
        assert spy.sync_called is False, f"{spy.name}._run_impl was called during async"
    print("PASS: test_deep_nesting_async")


# ── No cross-contamination after switching modes ─────────────────────────────

def test_sync_then_async_no_cross_contamination():
    """Running sync first and async second never mixes the call paths."""
    a = SpyAgent(name="a")
    b = SpyAgent(name="b")
    seq = SequentialAgent(name="seq", sub_agents=[a, b])

    # Step 1: sync
    seq.run(_ctx())
    assert a.sync_called is True
    assert a.async_called is False
    assert b.sync_called is True
    assert b.async_called is False

    # Reset
    a.reset()
    b.reset()

    # Step 2: async
    asyncio.run(seq.run_async(_ctx()))
    assert a.async_called is True
    assert a.sync_called is False
    assert b.async_called is True
    assert b.sync_called is False
    print("PASS: test_sync_then_async_no_cross_contamination")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_sequential_sync_only_calls_run_impl()
    test_sequential_async_only_calls_run_async_impl()
    test_parallel_sync_only_calls_run_impl()
    test_parallel_async_only_calls_run_async_impl()
    test_loop_sync_only_calls_run_impl()
    test_loop_async_only_calls_run_async_impl()
    test_nested_sync_propagates_through_tree()
    test_nested_async_propagates_through_tree()
    test_deep_nesting_sync()
    test_deep_nesting_async()
    test_sync_then_async_no_cross_contamination()
    print()
    print("=" * 60)
    print("  All 11 sync/async orchestration tests PASSED")
    print("=" * 60)
