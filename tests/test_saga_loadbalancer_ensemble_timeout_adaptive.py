"""Tests for SagaAgent, LoadBalancerAgent, EnsembleAgent, TimeoutAgent,
and AdaptivePlannerAgent.

Verifies:
- SagaAgent runs steps with compensating rollback on failure.
- LoadBalancerAgent distributes requests via various strategies.
- EnsembleAgent aggregates outputs from multiple agents.
- TimeoutAgent enforces deadlines with fallback.
- AdaptivePlannerAgent re-plans after each step (LLM mocked).
- Both sync and async paths work correctly.
- Error handling when agents are misconfigured.

Run:
    python -m pytest tests/test_saga_loadbalancer_ensemble_timeout_adaptive.py -v
"""

import asyncio
import json
import sys
import os
import time
from unittest.mock import MagicMock, patch

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
    AdaptivePlannerAgent,
    EnsembleAgent,
    LoadBalancerAgent,
    SagaAgent,
    TimeoutAgent,
)


# ── Stub agents ───────────────────────────────────────────────────────────────


class StubAgent(BaseAgent):
    """Returns a fixed response."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description=f"stub-{name}")
        self._response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class FailAgent(BaseAgent):
    """Returns an ERROR response to simulate failure."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description=f"fail-{name}")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "ERROR: step failed")

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, "ERROR: step failed")


class SlowAgent(BaseAgent):
    """Simulates a slow agent that blocks for a given duration."""

    def __init__(self, *, name: str, delay: float = 5.0) -> None:
        super().__init__(name=name, description=f"slow-{name}")
        self._delay = delay

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        time.sleep(self._delay)
        yield Event(EventType.AGENT_MESSAGE, self.name, "slow done")

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        await asyncio.sleep(self._delay)
        yield Event(EventType.AGENT_MESSAGE, self.name, "slow done")


class CounterAgent(BaseAgent):
    """Tracks how many times it was called."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description=f"counter-{name}")
        self.call_count = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.call_count += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"call-{self.call_count}",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        self.call_count += 1
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"call-{self.call_count}",
        )


def _make_ctx(message: str = "test") -> InvocationContext:
    """Create a basic InvocationContext."""
    return InvocationContext(session=Session(), user_message=message)


# ═══════════════════════════════════════════════════════════════════════
# SagaAgent
# ═══════════════════════════════════════════════════════════════════════


class TestSagaAgent:
    """Test distributed transactions with compensating rollback."""

    def test_all_steps_succeed(self) -> None:
        """All saga steps complete, no compensation."""
        s1 = StubAgent(name="reserve", response="reserved")
        s2 = StubAgent(name="charge", response="charged")
        s3 = StubAgent(name="ship", response="shipped")

        saga = SagaAgent(
            name="order",
            steps=[
                {"action": s1, "compensate": StubAgent(name="unreserve", response="unreserved")},
                {"action": s2, "compensate": StubAgent(name="refund", response="refunded")},
                {"action": s3},
            ],
            failure_detector=lambda o: "ERROR" in o.upper(),
            result_key="saga_res",
        )
        ctx = _make_ctx("process order")
        events = list(saga._run_impl(ctx))

        r = ctx.session.state["saga_res"]
        assert r["success"] is True
        assert r["failed_step"] is None
        assert len(r["completed"]) == 3
        assert r["compensated"] == []

    def test_step_failure_triggers_rollback(self) -> None:
        """Failure at step 2 triggers compensators for step 1, 0."""
        s1 = StubAgent(name="step1", response="ok1")
        comp1 = StubAgent(name="comp1", response="compensated1")
        s2 = FailAgent(name="step2")
        comp2 = StubAgent(name="comp2", response="compensated2")  # won't run

        saga = SagaAgent(
            name="fail_saga",
            steps=[
                {"action": s1, "compensate": comp1},
                {"action": s2, "compensate": comp2},
            ],
            failure_detector=lambda o: "ERROR" in o.upper(),
            result_key="fail_res",
        )
        ctx = _make_ctx()
        events = list(saga._run_impl(ctx))

        r = ctx.session.state["fail_res"]
        assert r["success"] is False
        assert r["failed_step"] == 1
        assert 0 in r["compensated"]

    def test_no_steps_error(self) -> None:
        """Error when no steps configured."""
        saga = SagaAgent(name="empty_saga")
        events = list(saga._run_impl(_make_ctx()))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_saga_async(self) -> None:
        """Async path works."""

        async def _run() -> dict[str, Any]:
            s1 = StubAgent(name="a1", response="ok")
            s2 = StubAgent(name="a2", response="ok")
            saga = SagaAgent(
                name="async_saga",
                steps=[
                    {"action": s1},
                    {"action": s2},
                ],
                result_key="async_saga_res",
            )
            ctx = _make_ctx()
            events = [e async for e in saga._run_async_impl(ctx)]
            return ctx.session.state["async_saga_res"]

        r = asyncio.run(_run())
        assert r["success"] is True

    def test_saga_first_step_fails(self) -> None:
        """Failure at step 0 means no compensation needed."""
        fail = FailAgent(name="f0")
        saga = SagaAgent(
            name="first_fail",
            steps=[{"action": fail}],
            failure_detector=lambda o: "ERROR" in o.upper(),
            result_key="ff_res",
        )
        ctx = _make_ctx()
        list(saga._run_impl(ctx))
        r = ctx.session.state["ff_res"]
        assert r["failed_step"] == 0
        assert r["compensated"] == []


# ═══════════════════════════════════════════════════════════════════════
# LoadBalancerAgent
# ═══════════════════════════════════════════════════════════════════════


class TestLoadBalancerAgent:
    """Test load distribution strategies."""

    def test_round_robin(self) -> None:
        """Round-robin distributes evenly."""
        a = CounterAgent(name="a")
        b = CounterAgent(name="b")

        lb = LoadBalancerAgent(
            name="lb",
            sub_agents=[a, b],
            strategy="round_robin",
        )
        for _ in range(4):
            list(lb._run_impl(_make_ctx()))

        assert a.call_count == 2
        assert b.call_count == 2

    def test_random_strategy(self) -> None:
        """Random strategy picks from pool."""
        a = CounterAgent(name="r1")
        b = CounterAgent(name="r2")

        lb = LoadBalancerAgent(
            name="lb_rand",
            sub_agents=[a, b],
            strategy="random",
        )
        for _ in range(10):
            list(lb._run_impl(_make_ctx()))

        assert a.call_count + b.call_count == 10

    def test_least_used(self) -> None:
        """Least-used strategy picks agent with fewest calls."""
        a = CounterAgent(name="lu1")
        b = CounterAgent(name="lu2")

        lb = LoadBalancerAgent(
            name="lb_lu",
            sub_agents=[a, b],
            strategy="least_used",
            result_key="lb_res",
        )
        # First call — both at 0, picks first
        list(lb._run_impl(_make_ctx()))
        # Second call — lu1 has 1, lu2 has 0, should pick lu2
        list(lb._run_impl(_make_ctx()))

        assert a.call_count == 1
        assert b.call_count == 1

    def test_no_pool_error(self) -> None:
        """Error when no agents in pool."""
        lb = LoadBalancerAgent(name="empty_lb")
        events = list(lb._run_impl(_make_ctx()))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_custom_strategy(self) -> None:
        """Custom strategy function is used."""
        a = StubAgent(name="always", response="picked")
        b = StubAgent(name="never", response="not picked")

        lb = LoadBalancerAgent(
            name="lb_custom",
            sub_agents=[a, b],
            strategy=lambda agents, usage: agents[0],  # always first
        )
        events = list(lb._run_impl(_make_ctx()))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert any("picked" in e.content for e in msg)

    def test_lb_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            a = StubAgent(name="async_a", response="a_out")
            lb = LoadBalancerAgent(
                name="async_lb",
                sub_agents=[a],
            )
            ctx = _make_ctx()
            return [e async for e in lb._run_async_impl(ctx)]

        events = asyncio.run(_run())
        assert any(e.event_type == EventType.AGENT_MESSAGE for e in events)


# ═══════════════════════════════════════════════════════════════════════
# EnsembleAgent
# ═══════════════════════════════════════════════════════════════════════


class TestEnsembleAgent:
    """Test output aggregation from multiple agents."""

    def test_concat_strategy(self) -> None:
        """Default concat strategy combines all outputs."""
        a = StubAgent(name="m1", response="alpha")
        b = StubAgent(name="m2", response="beta")

        ens = EnsembleAgent(
            name="ens",
            sub_agents=[a, b],
            aggregate_fn="concat",
            result_key="ens_res",
        )
        ctx = _make_ctx()
        events = list(ens._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        combined = msg[-1].content
        assert "alpha" in combined
        assert "beta" in combined
        assert ctx.session.state["ens_res"]["count"] == 2

    def test_weighted_strategy(self) -> None:
        """Weighted strategy includes weight labels."""
        a = StubAgent(name="w1", response="foo")
        b = StubAgent(name="w2", response="bar")

        ens = EnsembleAgent(
            name="ens_w",
            sub_agents=[a, b],
            aggregate_fn="weighted",
            weights={"w1": 0.7, "w2": 0.3},
        )
        ctx = _make_ctx()
        events = list(ens._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "w=0.70" in msg[-1].content

    def test_custom_aggregate(self) -> None:
        """Custom aggregate function is used."""
        a = StubAgent(name="c1", response="X")
        b = StubAgent(name="c2", response="Y")

        ens = EnsembleAgent(
            name="ens_c",
            sub_agents=[a, b],
            aggregate_fn=lambda outputs: "+".join(o[1] for o in outputs),
        )
        ctx = _make_ctx()
        events = list(ens._run_impl(ctx))
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert "X" in msg[-1].content and "Y" in msg[-1].content

    def test_no_agents_error(self) -> None:
        """Error when no agents."""
        ens = EnsembleAgent(name="empty_ens")
        events = list(ens._run_impl(_make_ctx()))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_ensemble_async(self) -> None:
        """Async path works."""

        async def _run() -> list[Event]:
            a = StubAgent(name="ae1", response="async1")
            b = StubAgent(name="ae2", response="async2")
            ens = EnsembleAgent(
                name="async_ens",
                sub_agents=[a, b],
                result_key="aens_res",
            )
            ctx = _make_ctx()
            return [e async for e in ens._run_async_impl(ctx)]

        events = asyncio.run(_run())
        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1


# ═══════════════════════════════════════════════════════════════════════
# TimeoutAgent
# ═══════════════════════════════════════════════════════════════════════


class TestTimeoutAgent:
    """Test deadline wrapper."""

    def test_fast_agent_completes(self) -> None:
        """Agent that finishes in time succeeds."""
        fast = StubAgent(name="fast", response="quick result")

        guarded = TimeoutAgent(
            name="guarded",
            agent=fast,
            timeout_seconds=10,
            result_key="timeout_res",
        )
        ctx = _make_ctx()
        events = list(guarded._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert msg[0].content == "quick result"
        r = ctx.session.state["timeout_res"]
        assert r["timed_out"] is False

    def test_slow_agent_times_out(self) -> None:
        """Slow agent triggers fallback."""
        slow = SlowAgent(name="slow", delay=10.0)

        guarded = TimeoutAgent(
            name="guarded_slow",
            agent=slow,
            timeout_seconds=0.1,
            fallback_message="timed out!",
            result_key="to_res",
        )
        ctx = _make_ctx()
        events = list(guarded._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert msg[0].content == "timed out!"
        assert ctx.session.state["to_res"]["timed_out"] is True

    def test_no_agent_error(self) -> None:
        """Error when no agent configured."""
        guarded = TimeoutAgent(name="empty_to")
        events = list(guarded._run_impl(_make_ctx()))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_timeout_async_completes(self) -> None:
        """Async fast agent completes."""

        async def _run() -> dict[str, Any]:
            fast = StubAgent(name="afast", response="async quick")
            guarded = TimeoutAgent(
                name="async_guard",
                agent=fast,
                timeout_seconds=10,
                result_key="ato_res",
            )
            ctx = _make_ctx()
            events = [e async for e in guarded._run_async_impl(ctx)]
            return ctx.session.state["ato_res"]

        r = asyncio.run(_run())
        assert r["timed_out"] is False

    def test_timeout_async_fires(self) -> None:
        """Async slow agent triggers timeout."""

        async def _run() -> dict[str, Any]:
            slow = SlowAgent(name="aslow", delay=10.0)
            guarded = TimeoutAgent(
                name="async_guard_slow",
                agent=slow,
                timeout_seconds=0.1,
                fallback_message="async timeout",
                result_key="ato_fire",
            )
            ctx = _make_ctx()
            events = [e async for e in guarded._run_async_impl(ctx)]
            return ctx.session.state["ato_fire"]

        r = asyncio.run(_run())
        assert r["timed_out"] is True


# ═══════════════════════════════════════════════════════════════════════
# AdaptivePlannerAgent
# ═══════════════════════════════════════════════════════════════════════


class TestAdaptivePlannerAgent:
    """Test adaptive re-planning with mocked LLM."""

    def _mock_planner(
        self,
        steps: list[list[dict[str, str]]],
    ) -> AdaptivePlannerAgent:
        """Create an AdaptivePlannerAgent with calls to _plan mocked."""
        worker = StubAgent(name="worker", response="step done")
        planner = AdaptivePlannerAgent(
            name="planner",
            sub_agents=[worker],
            max_steps=5,
            result_key="plan_res",
        )

        call_idx = 0

        def fake_plan(question: str, history: list) -> list[dict[str, str]]:
            nonlocal call_idx
            if call_idx < len(steps):
                result = steps[call_idx]
                call_idx += 1
                return result
            return []

        planner._plan = fake_plan  # type: ignore[assignment]
        return planner

    def test_single_step_plan(self) -> None:
        """Plan with one step then done."""
        planner = self._mock_planner([
            [{"agent": "worker", "message": "do task"}],
            [],  # done
        ])
        ctx = _make_ctx("solve it")
        events = list(planner._run_impl(ctx))

        msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(msg) >= 1
        r = ctx.session.state["plan_res"]
        assert r["total_steps"] == 1
        assert r["completed"] is True

    def test_multi_step_plan(self) -> None:
        """Multi-step plan executed sequentially."""
        planner = self._mock_planner([
            [{"agent": "worker", "message": "step 1"}],
            [{"agent": "worker", "message": "step 2"}],
            [{"agent": "worker", "message": "step 3"}],
            [],  # done
        ])
        ctx = _make_ctx()
        events = list(planner._run_impl(ctx))
        r = ctx.session.state["plan_res"]
        assert r["total_steps"] == 3

    def test_agent_not_found(self) -> None:
        """Error when plan names a non-existent agent."""
        planner = self._mock_planner([
            [{"agent": "nonexistent", "message": "fail"}],
        ])
        ctx = _make_ctx()
        events = list(planner._run_impl(ctx))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_no_agents_error(self) -> None:
        """Error when no sub-agents."""
        planner = AdaptivePlannerAgent(name="empty_planner")
        events = list(planner._run_impl(_make_ctx()))
        assert any(e.event_type == EventType.ERROR for e in events)

    def test_adaptive_async(self) -> None:
        """Async path works with mocked plan."""

        async def _run() -> dict[str, Any]:
            worker = StubAgent(name="aworker", response="async done")
            planner = AdaptivePlannerAgent(
                name="async_planner",
                sub_agents=[worker],
                max_steps=5,
                result_key="aplan_res",
            )
            call_idx = 0
            steps = [
                [{"agent": "aworker", "message": "async step"}],
                [],
            ]

            def fake_plan(q: str, h: list) -> list:
                nonlocal call_idx
                if call_idx < len(steps):
                    r = steps[call_idx]
                    call_idx += 1
                    return r
                return []

            planner._plan = fake_plan  # type: ignore[assignment]
            ctx = _make_ctx()
            events = [e async for e in planner._run_async_impl(ctx)]
            return ctx.session.state["aplan_res"]

        r = asyncio.run(_run())
        assert r["total_steps"] == 1

    def test_max_steps_limit(self) -> None:
        """max_steps prevents infinite loop."""
        # Always return a step — never done
        worker = StubAgent(name="inf_worker", response="again")
        planner = AdaptivePlannerAgent(
            name="inf_planner",
            sub_agents=[worker],
            max_steps=3,
            result_key="inf_res",
        )
        planner._plan = lambda q, h: [{"agent": "inf_worker", "message": "go"}]  # type: ignore[assignment]
        ctx = _make_ctx()
        list(planner._run_impl(ctx))
        assert ctx.session.state["inf_res"]["total_steps"] == 3
