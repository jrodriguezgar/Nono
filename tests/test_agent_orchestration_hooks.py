"""Tests for agent orchestration lifecycle hooks.

Covers the five hooks that mirror Workflow lifecycle hooks:
- on_start / on_end: orchestrator-level start/end events
- on_between_agents: fires between sequential sub-agents, halts on False
- on_agent_start: fires before each sub-agent executes
- on_agent_end: fires after each sub-agent executes, receives error info

Tested on: SequentialAgent, ParallelAgent, LoopAgent (sync + async).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

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


# ── Helpers ──────────────────────────────────────────────────────────────────


class EchoAgent(BaseAgent):
    """Simple agent that emits one AGENT_MESSAGE with a fixed reply."""

    def __init__(self, name: str, reply: str = "ok") -> None:
        super().__init__(name=name)
        self._reply = reply

    def _run_impl(self, ctx: InvocationContext):  # type: ignore[override]
        yield Event(EventType.AGENT_MESSAGE, self.name, self._reply)

    async def _run_async_impl(self, ctx: InvocationContext):  # type: ignore[override]
        yield Event(EventType.AGENT_MESSAGE, self.name, self._reply)


class FailAgent(BaseAgent):
    """Agent that always raises."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def _run_impl(self, ctx: InvocationContext):  # type: ignore[override]
        raise RuntimeError("boom")
        yield  # pragma: no cover

    async def _run_async_impl(self, ctx: InvocationContext):  # type: ignore[override]
        raise RuntimeError("boom")
        yield  # pragma: no cover


def _make_ctx(session: Session | None = None, msg: str = "test") -> InvocationContext:
    return InvocationContext(session=session or Session(), user_message=msg)


# ══════════════════════════════════════════════════════════════════════════════
# SequentialAgent
# ══════════════════════════════════════════════════════════════════════════════


class TestSequentialOnStart:
    """on_start fires once before the first sub-agent."""

    def test_fluent_returns_self(self) -> None:
        seq = SequentialAgent(name="s", sub_agents=[EchoAgent("a")])
        assert seq.on_start(lambda n, s: None) is seq

    def test_fires_once(self) -> None:
        calls: list[str] = []
        seq = SequentialAgent(name="seq", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        seq.on_start(lambda name, session: calls.append(name))
        seq.run(_make_ctx())

        assert calls == ["seq"]

    def test_fires_before_agent_start(self) -> None:
        events: list[str] = []
        seq = SequentialAgent(name="seq", sub_agents=[EchoAgent("a")])
        seq.on_start(lambda n, s: events.append("start"))
        seq.on_agent_start(lambda n, s: events.append(f"agent:{n}"))
        seq.run(_make_ctx())

        assert events[0] == "start"
        assert events[1] == "agent:a"


class TestSequentialOnEnd:
    """on_end fires once after all sub-agents complete."""

    def test_fluent_returns_self(self) -> None:
        seq = SequentialAgent(name="s", sub_agents=[EchoAgent("a")])
        assert seq.on_end(lambda n, s, c: None) is seq

    def test_fires_with_correct_count(self) -> None:
        calls: list[tuple[str, int]] = []
        seq = SequentialAgent(name="seq", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        seq.on_end(lambda name, session, n: calls.append((name, n)))
        seq.run(_make_ctx())

        assert calls == [("seq", 2)]

    def test_fires_after_halt(self) -> None:
        calls: list[int] = []
        seq = SequentialAgent(name="seq", sub_agents=[EchoAgent("a"), EchoAgent("b"), EchoAgent("c")])
        seq.on_between_agents(lambda p, n, s: False if p == "a" else None)
        seq.on_end(lambda name, session, n: calls.append(n))
        seq.run(_make_ctx())

        assert calls == [1]  # only 'a' ran before halt


class TestSequentialOnBetweenAgents:
    """on_between_agents fires between sequential sub-agents."""

    def test_fluent_returns_self(self) -> None:
        seq = SequentialAgent(name="s", sub_agents=[EchoAgent("a")])
        assert seq.on_between_agents(lambda p, n, s: None) is seq

    def test_fires_between_agents(self) -> None:
        calls: list[tuple[str, str | None]] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b"), EchoAgent("c")],
        )
        seq.on_between_agents(lambda prev, nxt, session: calls.append((prev, nxt)))
        seq.run(_make_ctx())

        assert calls == [("a", "b"), ("b", "c"), ("c", None)]

    def test_halts_on_false(self) -> None:
        executing: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b"), EchoAgent("c")],
        )
        seq.on_agent_start(lambda n, s: executing.append(n))
        seq.on_between_agents(lambda p, n, s: False if p == "a" else None)
        seq.run(_make_ctx())

        assert executing == ["a"]  # b and c never ran

    def test_none_return_continues(self) -> None:
        executing: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
        )
        seq.on_between_agents(lambda p, n, s: None)
        seq.on_agent_start(lambda n, s: executing.append(n))
        seq.run(_make_ctx())

        assert executing == ["a", "b"]


class TestSequentialOnAgentStart:
    """on_agent_start fires before each sub-agent."""

    def test_fluent_returns_self(self) -> None:
        seq = SequentialAgent(name="s", sub_agents=[EchoAgent("a")])
        assert seq.on_agent_start(lambda n, s: None) is seq

    def test_fires_per_agent(self) -> None:
        calls: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b"), EchoAgent("c")],
        )
        seq.on_agent_start(lambda name, session: calls.append(name))
        seq.run(_make_ctx())

        assert calls == ["a", "b", "c"]


class TestSequentialOnAgentEnd:
    """on_agent_end fires after each sub-agent with error info."""

    def test_fluent_returns_self(self) -> None:
        seq = SequentialAgent(name="s", sub_agents=[EchoAgent("a")])
        assert seq.on_agent_end(lambda n, s, e: None) is seq

    def test_fires_per_agent_no_error(self) -> None:
        calls: list[tuple[str, str | None]] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
        )
        seq.on_agent_end(lambda name, session, err: calls.append((name, err)))
        seq.run(_make_ctx())

        assert calls == [("a", None), ("b", None)]

    def test_captures_error(self) -> None:
        calls: list[tuple[str, str | None]] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), FailAgent("bad"), EchoAgent("c")],
        )
        seq.on_agent_end(lambda name, session, err: calls.append((name, err)))
        seq.run(_make_ctx())

        assert calls[0] == ("a", None)
        assert calls[1][0] == "bad"
        assert calls[1][1] is not None  # error message
        assert calls[2] == ("c", None)


class TestSequentialCombinedHooks:
    """All hooks fire in correct order for SequentialAgent."""

    def test_full_order(self) -> None:
        events: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
        )
        seq.on_start(lambda n, s: events.append("start"))
        seq.on_agent_start(lambda n, s: events.append(f"exec:{n}"))
        seq.on_agent_end(lambda n, s, e: events.append(f"done:{n}"))
        seq.on_between_agents(lambda p, nx, s: events.append(f"between:{p}->{nx}"))
        seq.on_end(lambda n, s, c: events.append(f"end:{c}"))
        seq.run(_make_ctx())

        assert events == [
            "start",
            "exec:a",
            "done:a",
            "between:a->b",
            "exec:b",
            "done:b",
            "between:b->None",
            "end:2",
        ]

    def test_no_hooks_works(self) -> None:
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
        )
        result = seq.run(_make_ctx())
        assert result == "ok"


class TestSequentialAsync:
    """Hooks fire in async mode too."""

    def test_all_hooks_fire_async(self) -> None:
        events: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
        )
        seq.on_start(lambda n, s: events.append("start"))
        seq.on_agent_start(lambda n, s: events.append(f"exec:{n}"))
        seq.on_agent_end(lambda n, s, e: events.append(f"done:{n}"))
        seq.on_between_agents(lambda p, nx, s: events.append(f"between:{p}->{nx}"))
        seq.on_end(lambda n, s, c: events.append(f"end:{c}"))
        asyncio.run(seq.run_async(_make_ctx()))

        assert events[0] == "start"
        assert "exec:a" in events
        assert "done:b" in events
        assert events[-1] == "end:2"

    def test_halts_on_false_async(self) -> None:
        executing: list[str] = []
        seq = SequentialAgent(
            name="seq",
            sub_agents=[EchoAgent("a"), EchoAgent("b"), EchoAgent("c")],
        )
        seq.on_agent_start(lambda n, s: executing.append(n))
        seq.on_between_agents(lambda p, n, s: False if p == "a" else None)
        asyncio.run(seq.run_async(_make_ctx()))

        assert executing == ["a"]


# ══════════════════════════════════════════════════════════════════════════════
# ParallelAgent
# ══════════════════════════════════════════════════════════════════════════════


class TestParallelOnStart:
    """on_start fires once before parallel execution begins."""

    def test_fires_once(self) -> None:
        calls: list[str] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        par.on_start(lambda name, session: calls.append(name))
        par.run(_make_ctx())

        assert calls == ["par"]


class TestParallelOnEnd:
    """on_end fires once after all parallel agents complete."""

    def test_fires_with_count(self) -> None:
        calls: list[tuple[str, int]] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        par.on_end(lambda name, session, n: calls.append((name, n)))
        par.run(_make_ctx())

        assert len(calls) == 1
        assert calls[0][0] == "par"
        assert calls[0][1] == 2


class TestParallelOnAgentStartEnd:
    """on_agent_start and on_agent_end fire per parallel sub-agent."""

    def test_fires_per_agent(self) -> None:
        starts: list[str] = []
        ends: list[tuple[str, str | None]] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        par.on_agent_start(lambda n, s: starts.append(n))
        par.on_agent_end(lambda n, s, e: ends.append((n, e)))
        par.run(_make_ctx())

        assert sorted(starts) == ["a", "b"]
        assert all(e is None for _, e in ends)

    def test_captures_error(self) -> None:
        ends: list[tuple[str, str | None]] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), FailAgent("bad")])
        par.on_agent_end(lambda n, s, e: ends.append((n, e)))
        par.run(_make_ctx())

        errors = {n: e for n, e in ends}
        assert errors["a"] is None
        assert errors["bad"] is not None


class TestParallelNoBetweenAgents:
    """on_between_agents has no effect on ParallelAgent (agents run concurrently)."""

    def test_no_between_calls(self) -> None:
        calls: list[str] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        par.on_between_agents(lambda p, n, s: calls.append(p))
        par.run(_make_ctx())

        assert calls == []  # never fired for parallel


class TestParallelAsync:
    """Hooks fire in async mode."""

    def test_all_hooks_fire_async(self) -> None:
        starts: list[str] = []
        ends: list[str] = []
        started: list[str] = []
        par = ParallelAgent(name="par", sub_agents=[EchoAgent("a"), EchoAgent("b")])
        par.on_start(lambda n, s: started.append(n))
        par.on_agent_start(lambda n, s: starts.append(n))
        par.on_agent_end(lambda n, s, e: ends.append(n))
        par.on_end(lambda n, s, c: started.append(f"end:{c}"))
        asyncio.run(par.run_async(_make_ctx()))

        assert started[0] == "par"
        assert sorted(starts) == ["a", "b"]
        assert sorted(ends) == ["a", "b"]


# ══════════════════════════════════════════════════════════════════════════════
# LoopAgent
# ══════════════════════════════════════════════════════════════════════════════


class TestLoopOnStart:
    """on_start fires once before the first loop iteration."""

    def test_fires_once(self) -> None:
        calls: list[str] = []
        loop = LoopAgent(name="loop", sub_agents=[EchoAgent("a")], max_iterations=3)
        loop.on_start(lambda name, session: calls.append(name))
        loop.run(_make_ctx())

        assert calls == ["loop"]


class TestLoopOnEnd:
    """on_end fires once with total agent invocations across all iterations."""

    def test_fires_with_total_count(self) -> None:
        calls: list[tuple[str, int]] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=2,
        )
        loop.on_end(lambda name, session, n: calls.append((name, n)))
        loop.run(_make_ctx())

        # 2 iterations × 2 agents = 4
        assert calls == [("loop", 4)]


class TestLoopOnBetweenAgents:
    """on_between_agents fires between agents within and across iterations."""

    def test_fires_between_agents_within_iteration(self) -> None:
        calls: list[tuple[str, str | None]] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=1,
        )
        loop.on_between_agents(lambda p, n, s: calls.append((p, n)))
        loop.run(_make_ctx())

        assert calls == [("a", "b"), ("b", None)]

    def test_halts_loop(self) -> None:
        executing: list[str] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=3,
        )
        loop.on_agent_start(lambda n, s: executing.append(n))
        # Halt after first "a"
        loop.on_between_agents(lambda p, n, s: False if p == "a" else None)
        loop.run(_make_ctx())

        assert executing == ["a"]  # b never ran, and no second iteration


class TestLoopOnAgentStartEnd:
    """on_agent_start and on_agent_end fire per sub-agent per iteration."""

    def test_fires_per_agent_per_iteration(self) -> None:
        starts: list[str] = []
        ends: list[tuple[str, str | None]] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=2,
        )
        loop.on_agent_start(lambda n, s: starts.append(n))
        loop.on_agent_end(lambda n, s, e: ends.append((n, e)))
        loop.run(_make_ctx())

        assert starts == ["a", "b", "a", "b"]
        assert all(e is None for _, e in ends)


class TestLoopCombinedHooks:
    """Full lifecycle ordering in a LoopAgent."""

    def test_full_order_single_iteration(self) -> None:
        events: list[str] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=1,
        )
        loop.on_start(lambda n, s: events.append("start"))
        loop.on_agent_start(lambda n, s: events.append(f"exec:{n}"))
        loop.on_agent_end(lambda n, s, e: events.append(f"done:{n}"))
        loop.on_between_agents(lambda p, nx, s: events.append(f"between:{p}->{nx}"))
        loop.on_end(lambda n, s, c: events.append(f"end:{c}"))
        loop.run(_make_ctx())

        assert events == [
            "start",
            "exec:a",
            "done:a",
            "between:a->b",
            "exec:b",
            "done:b",
            "between:b->None",
            "end:2",
        ]


class TestLoopAsync:
    """Hooks fire in async mode."""

    def test_all_hooks_fire_async(self) -> None:
        events: list[str] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a")],
            max_iterations=2,
        )
        loop.on_start(lambda n, s: events.append("start"))
        loop.on_agent_start(lambda n, s: events.append(f"exec:{n}"))
        loop.on_agent_end(lambda n, s, e: events.append(f"done:{n}"))
        loop.on_end(lambda n, s, c: events.append(f"end:{c}"))
        asyncio.run(loop.run_async(_make_ctx()))

        assert events[0] == "start"
        assert events.count("exec:a") == 2
        assert events[-1] == "end:2"

    def test_halts_on_false_async(self) -> None:
        executing: list[str] = []
        loop = LoopAgent(
            name="loop",
            sub_agents=[EchoAgent("a"), EchoAgent("b")],
            max_iterations=3,
        )
        loop.on_agent_start(lambda n, s: executing.append(n))
        loop.on_between_agents(lambda p, n, s: False if p == "a" else None)
        asyncio.run(loop.run_async(_make_ctx()))

        assert executing == ["a"]
