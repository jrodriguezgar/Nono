"""Tests for workflow lifecycle hooks.

Covers:
- on_start / on_end: workflow-level start/end events
- on_between_steps: fires between steps, receives correct args, halts on False
- on_step_executing: fires per-attempt before execution
- on_step_executed: fires per-attempt after execution with error info
- All hooks work with run(), run_async(), stream(), astream(), replay_from()
- Fluent API returns self
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nono.workflows import (
    Workflow,
    BetweenStepsCallback,
    OnEndCallback,
    OnStartCallback,
    StepExecutedCallback,
    StepExecutingCallback,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _add(state: dict) -> dict:
    return {"x": state.get("x", 0) + 1}


def _double(state: dict) -> dict:
    return {"x": state["x"] * 2}


def _label(state: dict) -> dict:
    return {"label": f"x={state['x']}"}


_fail_counter: int = 0


def _flaky(state: dict) -> dict:
    """Fails on first call, succeeds on second."""
    global _fail_counter
    _fail_counter += 1
    if _fail_counter % 2 == 1:
        raise ValueError("transient failure")
    return {"v": "ok"}


def _build_linear_flow() -> Workflow:
    """Create a simple add → double → label workflow."""
    flow = Workflow("hooks_test")
    flow.step("add", _add)
    flow.step("double", _double)
    flow.step("label", _label)
    flow.connect("add", "double", "label")
    return flow


# ── on_between_steps ─────────────────────────────────────────────────────────


class TestOnBetweenSteps:
    """Tests for the on_between_steps hook."""

    def test_fluent_returns_self(self) -> None:
        flow = _build_linear_flow()
        result = flow.on_between_steps(lambda p, n, s: None)
        assert result is flow

    def test_fires_between_steps_run(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, str | None, int]] = []

        def cb(prev: str, nxt: str | None, state: dict) -> None:
            calls.append((prev, nxt, state.get("x", 0)))

        flow.on_between_steps(cb)
        flow.run(x=0)

        # add → double, double → label, label → None (END)
        assert len(calls) == 3
        assert calls[0] == ("add", "double", 1)
        assert calls[1] == ("double", "label", 2)
        assert calls[2][0] == "label"

    def test_halts_on_false_run(self) -> None:
        flow = _build_linear_flow()

        def halt_after_double(prev: str, nxt: str | None, state: dict) -> bool | None:
            if prev == "double":
                return False
            return None

        flow.on_between_steps(halt_after_double)
        result = flow.run(x=0)

        # Execution should stop after double; label never executed
        assert "label" not in result
        assert result["x"] == 2

    def test_fires_between_steps_run_async(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_between_steps(lambda p, n, s: calls.append(p))
        asyncio.run(flow.run_async(x=0))

        assert calls == ["add", "double", "label"]

    def test_halts_on_false_run_async(self) -> None:
        flow = _build_linear_flow()
        flow.on_between_steps(lambda p, n, s: False if p == "add" else None)
        result = asyncio.run(flow.run_async(x=0))

        assert result["x"] == 1  # only add executed

    def test_fires_between_steps_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_between_steps(lambda p, n, s: calls.append(p))
        chunks = list(flow.stream(x=0))

        assert "add" in calls
        assert "double" in calls

    def test_halts_on_false_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []
        flow.on_between_steps(lambda p, n, s: (calls.append(p), False)[1] if p == "add" else calls.append(p))
        chunks = list(flow.stream(x=0))

        # Only add executed, so on_between_steps fires once (after add)
        assert calls == ["add"]

    def test_fires_between_steps_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_between_steps(lambda p, n, s: calls.append(p))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert "add" in calls

    def test_halts_on_false_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []
        flow.on_between_steps(lambda p, n, s: (calls.append(p), False)[1] if p == "add" else calls.append(p))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert calls == ["add"]

    def test_fires_between_steps_replay_from(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.run(x=0)  # populate transitions
        flow.on_between_steps(lambda p, n, s: calls.append(p))
        flow.replay_from("add", x=10)

        assert "double" in calls

    def test_halts_on_false_replay_from(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)  # populate transitions
        flow.on_between_steps(lambda p, n, s: False if p == "double" else None)
        result = flow.replay_from("add", x=10)

        assert result["x"] == 20  # replay starts after add with x=10, double→20, label skipped

    def test_none_return_continues(self) -> None:
        flow = _build_linear_flow()
        flow.on_between_steps(lambda p, n, s: None)
        result = flow.run(x=0)

        assert "label" in result  # all steps executed

    def test_true_return_continues(self) -> None:
        flow = _build_linear_flow()
        flow.on_between_steps(lambda p, n, s: True)
        result = flow.run(x=0)

        assert "label" in result


# ── on_step_executing ────────────────────────────────────────────────────────


class TestOnStepExecuting:
    """Tests for the on_step_executing hook."""

    def test_fluent_returns_self(self) -> None:
        flow = _build_linear_flow()
        result = flow.on_step_executing(lambda n, s, a: None)
        assert result is flow

    def test_fires_per_step_run(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, int]] = []

        flow.on_step_executing(lambda name, state, attempt: calls.append((name, attempt)))
        flow.run(x=0)

        assert ("add", 1) in calls
        assert ("double", 1) in calls
        assert ("label", 1) in calls
        assert len(calls) == 3

    def test_fires_per_attempt_on_retry(self) -> None:
        global _fail_counter
        _fail_counter = 0

        flow = Workflow("retry_hooks")
        flow.step("flaky", _flaky, retry=2)
        calls: list[tuple[str, int]] = []

        flow.on_step_executing(lambda name, state, attempt: calls.append((name, attempt)))
        flow.run()

        # Should fire twice: attempt 1 (fails), attempt 2 (succeeds)
        assert ("flaky", 1) in calls
        assert ("flaky", 2) in calls

    def test_fires_per_step_run_async(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, int]] = []

        flow.on_step_executing(lambda name, state, attempt: calls.append((name, attempt)))
        asyncio.run(flow.run_async(x=0))

        assert len(calls) == 3
        assert all(a == 1 for _, a in calls)

    def test_fires_per_step_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_step_executing(lambda name, state, attempt: calls.append(name))
        for _ in flow.stream(x=0):
            pass

        assert "add" in calls
        assert "double" in calls

    def test_fires_per_step_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_step_executing(lambda name, state, attempt: calls.append(name))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert "add" in calls

    def test_fires_per_step_replay_from(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)  # populate transitions

        calls: list[tuple[str, int]] = []
        flow.on_step_executing(lambda name, state, attempt: calls.append((name, attempt)))
        flow.replay_from("add", x=10)

        assert ("double", 1) in calls
        assert ("label", 1) in calls

    def test_receives_correct_state(self) -> None:
        flow = _build_linear_flow()
        states: list[dict] = []

        flow.on_step_executing(lambda name, state, attempt: states.append(dict(state)))
        flow.run(x=0)

        # Before add executes, x=0
        assert states[0]["x"] == 0
        # Before double executes, x=1
        assert states[1]["x"] == 1
        # Before label executes, x=2
        assert states[2]["x"] == 2


# ── Combined hooks ───────────────────────────────────────────────────────────


class TestCombinedHooks:
    """Tests for both hooks working together."""

    def test_all_hooks_fire_in_order(self) -> None:
        """Verify all seven hooks fire in the correct order."""
        flow = _build_linear_flow()
        events: list[str] = []

        flow.on_start(lambda name, state: events.append("start"))
        flow.on_step_executing(
            lambda name, state, attempt: events.append(f"executing:{name}")
        )
        flow.on_step_executed(
            lambda name, state, attempt, error: events.append(f"executed:{name}")
        )
        flow.on_between_steps(
            lambda prev, nxt, state: events.append(f"between:{prev}->{nxt}")
        )
        flow.on_end(lambda name, state, n: events.append(f"end:{n}"))
        flow.run(x=0)

        # Full order: start, executing, executed, between, ..., end
        assert events[0] == "start"
        assert events[1] == "executing:add"
        assert events[2] == "executed:add"
        assert events[3] == "between:add->double"
        assert events[4] == "executing:double"
        assert events[5] == "executed:double"
        assert events[6] == "between:double->label"
        assert events[7] == "executing:label"
        assert events[8] == "executed:label"
        assert events[-1] == "end:3"

    def test_between_halt_prevents_executing(self) -> None:
        """If on_between_steps halts after add, double's on_step_executing never fires."""
        flow = _build_linear_flow()
        executing_calls: list[str] = []

        flow.on_step_executing(
            lambda name, state, attempt: executing_calls.append(name)
        )
        flow.on_between_steps(
            lambda prev, nxt, state: False if prev == "add" else None
        )
        flow.run(x=0)

        assert executing_calls == ["add"]

    def test_existing_hooks_still_work(self) -> None:
        """Verify all seven hook types fire together."""
        flow = _build_linear_flow()
        events: list[str] = []

        flow.on_start(lambda name, state: events.append("start"))
        flow.on_before_step(lambda name, state: events.append(f"before:{name}"))
        flow.on_after_step(lambda name, state, result: events.append(f"after:{name}"))
        flow.on_step_executing(
            lambda name, state, attempt: events.append(f"executing:{name}")
        )
        flow.on_step_executed(
            lambda name, state, attempt, error: events.append(f"executed:{name}")
        )
        flow.on_between_steps(
            lambda prev, nxt, state: events.append(f"between:{prev}")
        )
        flow.on_end(lambda name, state, n: events.append("end"))
        flow.run(x=0)

        assert any(e == "start" for e in events)
        assert any(e.startswith("before:") for e in events)
        assert any(e.startswith("after:") for e in events)
        assert any(e.startswith("executing:") for e in events)
        assert any(e.startswith("executed:") for e in events)
        assert any(e.startswith("between:") for e in events)
        assert any(e == "end" for e in events)

    def test_no_hooks_works(self) -> None:
        """Workflow with no hooks set still runs normally."""
        flow = _build_linear_flow()
        result = flow.run(x=0)

        assert result["x"] == 2
        assert result["label"] == "x=2"


# ── on_start / on_end ────────────────────────────────────────────────────────


class TestOnStart:
    """Tests for the on_start hook."""

    def test_fluent_returns_self(self) -> None:
        flow = _build_linear_flow()
        result = flow.on_start(lambda name, state: None)
        assert result is flow

    def test_fires_once_at_start_run(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, int]] = []

        flow.on_start(lambda name, state: calls.append((name, state.get("x", -1))))
        flow.run(x=0)

        assert len(calls) == 1
        assert calls[0] == ("hooks_test", 0)

    def test_fires_before_first_step(self) -> None:
        flow = _build_linear_flow()
        events: list[str] = []

        flow.on_start(lambda name, state: events.append("start"))
        flow.on_before_step(lambda name, state: events.append(f"before:{name}"))
        flow.run(x=0)

        assert events[0] == "start"
        assert events[1] == "before:add"

    def test_fires_in_run_async(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_start(lambda name, state: calls.append(name))
        asyncio.run(flow.run_async(x=0))

        assert calls == ["hooks_test"]

    def test_fires_in_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_start(lambda name, state: calls.append(name))
        for _ in flow.stream(x=0):
            pass

        assert calls == ["hooks_test"]

    def test_fires_in_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_start(lambda name, state: calls.append(name))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert calls == ["hooks_test"]

    def test_fires_in_replay_from(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)  # populate transitions

        calls: list[str] = []
        flow.on_start(lambda name, state: calls.append(name))
        flow.replay_from("add", x=10)

        assert calls == ["hooks_test"]


class TestOnEnd:
    """Tests for the on_end hook."""

    def test_fluent_returns_self(self) -> None:
        flow = _build_linear_flow()
        result = flow.on_end(lambda name, state, n: None)
        assert result is flow

    def test_fires_once_at_end_run(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, int]] = []

        flow.on_end(lambda name, state, n: calls.append((name, n)))
        flow.run(x=0)

        assert len(calls) == 1
        assert calls[0] == ("hooks_test", 3)  # 3 steps executed

    def test_fires_after_last_step(self) -> None:
        flow = _build_linear_flow()
        events: list[str] = []

        flow.on_between_steps(lambda prev, nxt, state: events.append(f"between:{prev}"))
        flow.on_end(lambda name, state, n: events.append("end"))
        flow.run(x=0)

        assert events[-1] == "end"

    def test_fires_after_halt(self) -> None:
        flow = _build_linear_flow()
        calls: list[int] = []

        flow.on_between_steps(lambda p, n, s: False if p == "add" else None)
        flow.on_end(lambda name, state, n: calls.append(n))
        flow.run(x=0)

        assert calls == [1]  # only add executed before halt

    def test_receives_final_state(self) -> None:
        flow = _build_linear_flow()
        states: list[dict] = []

        flow.on_end(lambda name, state, n: states.append(dict(state)))
        flow.run(x=0)

        assert states[0]["x"] == 2
        assert states[0]["label"] == "x=2"

    def test_fires_in_run_async(self) -> None:
        flow = _build_linear_flow()
        calls: list[int] = []

        flow.on_end(lambda name, state, n: calls.append(n))
        asyncio.run(flow.run_async(x=0))

        assert calls == [3]

    def test_fires_in_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[int] = []

        flow.on_end(lambda name, state, n: calls.append(n))
        for _ in flow.stream(x=0):
            pass

        assert calls == [3]

    def test_fires_in_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[int] = []

        flow.on_end(lambda name, state, n: calls.append(n))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert calls == [3]

    def test_fires_in_replay_from(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        calls: list[int] = []
        flow.on_end(lambda name, state, n: calls.append(n))
        flow.replay_from("add", x=10)

        assert calls[0] == 2  # double + label


# ── on_step_executed ─────────────────────────────────────────────────────────


class TestOnStepExecuted:
    """Tests for the on_step_executed hook."""

    def test_fluent_returns_self(self) -> None:
        flow = _build_linear_flow()
        result = flow.on_step_executed(lambda n, s, a, e: None)
        assert result is flow

    def test_fires_per_step_run(self) -> None:
        flow = _build_linear_flow()
        calls: list[tuple[str, int, str | None]] = []

        flow.on_step_executed(
            lambda name, state, attempt, error: calls.append((name, attempt, error))
        )
        flow.run(x=0)

        assert ("add", 1, None) in calls
        assert ("double", 1, None) in calls
        assert ("label", 1, None) in calls

    def test_receives_error_on_failure(self) -> None:
        global _fail_counter
        _fail_counter = 0

        flow = Workflow("executed_retry")
        flow.step("flaky", _flaky, retry=2)
        calls: list[tuple[str, int, str | None]] = []

        flow.on_step_executed(
            lambda name, state, attempt, error: calls.append((name, attempt, error))
        )
        flow.run()

        # attempt 1 fails, attempt 2 succeeds
        assert calls[0] == ("flaky", 1, "transient failure")
        assert calls[1] == ("flaky", 2, None)

    def test_fires_in_run_async(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_step_executed(lambda n, s, a, e: calls.append(n))
        asyncio.run(flow.run_async(x=0))

        assert calls == ["add", "double", "label"]

    def test_fires_in_stream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_step_executed(lambda n, s, a, e: calls.append(n))
        for _ in flow.stream(x=0):
            pass

        assert "add" in calls

    def test_fires_in_astream(self) -> None:
        flow = _build_linear_flow()
        calls: list[str] = []

        flow.on_step_executed(lambda n, s, a, e: calls.append(n))

        async def _run():
            async for _ in flow.astream(x=0):
                pass

        asyncio.run(_run())
        assert "add" in calls

    def test_fires_in_replay_from(self) -> None:
        flow = _build_linear_flow()
        flow.run(x=0)

        calls: list[str] = []
        flow.on_step_executed(lambda n, s, a, e: calls.append(n))
        flow.replay_from("add", x=10)

        assert "double" in calls
        assert "label" in calls

    def test_executing_and_executed_pair(self) -> None:
        """Verify on_step_executing fires before fn, on_step_executed fires after."""
        flow = _build_linear_flow()
        events: list[str] = []

        flow.on_step_executing(lambda n, s, a: events.append(f"pre:{n}"))
        flow.on_step_executed(lambda n, s, a, e: events.append(f"post:{n}"))
        flow.run(x=0)

        # Pattern: pre, post, pre, post, ...
        assert events[0] == "pre:add"
        assert events[1] == "post:add"
        assert events[2] == "pre:double"
        assert events[3] == "post:double"
