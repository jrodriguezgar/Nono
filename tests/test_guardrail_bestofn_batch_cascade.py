"""Tests for GuardrailAgent, BestOfNAgent, BatchAgent, and CascadeAgent.

Verifies:
- GuardrailAgent pre/post validation, retry on rejection, error propagation.
- BestOfNAgent runs N times, picks best by score, stores results.
- BatchAgent processes item lists with concurrency, template formatting.
- CascadeAgent progressive stages with scoring threshold.
- Both sync and async paths work correctly.

Run:
    python -m pytest tests/test_guardrail_bestofn_batch_cascade.py -v
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
    BatchAgent,
    BestOfNAgent,
    CascadeAgent,
    GuardrailAgent,
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


class EchoAgent(BaseAgent):
    """Echoes the user message."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name=name, description=f"echo-{name}")

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, ctx.user_message)


class ErrorAgent(BaseAgent):
    """Yields an ERROR event."""

    def __init__(self, *, name: str, message: str = "blocked") -> None:
        super().__init__(name=name, description=f"error-{name}")
        self._message = message

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.ERROR, self.name, self._message)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.ERROR, self.name, self._message)


class CountingAgent(BaseAgent):
    """Returns a different response each call (for BestOfN variance)."""

    def __init__(self, *, name: str, responses: list[str]) -> None:
        super().__init__(name=name, description=f"counting-{name}")
        self._responses = responses
        self._call = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        resp = self._responses[self._call % len(self._responses)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        resp = self._responses[self._call % len(self._responses)]
        self._call += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, resp)


def _ctx(message: str = "test") -> InvocationContext:
    """Build a minimal InvocationContext."""
    return InvocationContext(session=Session(), user_message=message)


# ══════════════════════════════════════════════════════════════════════════════
# GuardrailAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_guardrail_sync_no_validators():
    """No validators — main agent runs once and output passes through."""
    main = StubAgent(name="writer", response="Hello world")

    guardrail = GuardrailAgent(
        name="simple_guard",
        main_agent=main,
    )

    events = list(guardrail._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Hello world" in e.content for e in msgs)


def test_guardrail_sync_pre_validator_transforms():
    """Pre-validator transforms the input message."""
    pre = StubAgent(name="sanitizer", response="sanitized input")
    main = EchoAgent(name="echo")

    guardrail = GuardrailAgent(
        name="transform_guard",
        main_agent=main,
        pre_validator=pre,
    )

    events = list(guardrail._run_impl(_ctx("raw input")))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("sanitized input" in e.content for e in msgs)


def test_guardrail_sync_pre_validator_blocks():
    """Pre-validator yields ERROR — main agent is skipped."""
    pre = ErrorAgent(name="blocker", message="PII detected")
    main = StubAgent(name="writer", response="should not appear")

    guardrail = GuardrailAgent(
        name="block_guard",
        main_agent=main,
        pre_validator=pre,
    )

    events = list(guardrail._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1
    assert "PII detected" in errors[0].content

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert not any("should not appear" in e.content for e in msgs)


def test_guardrail_sync_post_validator_passes():
    """Post-validator passes — output accepted on first try."""
    main = StubAgent(name="writer", response="clean output")
    post = StubAgent(name="checker", response="APPROVED: looks good")

    guardrail = GuardrailAgent(
        name="pass_guard",
        main_agent=main,
        post_validator=post,
        rejection_keyword="REJECTED",
    )

    events = list(guardrail._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 0


def test_guardrail_sync_post_validator_rejects_then_passes():
    """Post-validator rejects first, then passes on retry."""
    main = StubAgent(name="writer", response="output")
    post = CountingAgent(
        name="checker",
        responses=["REJECTED: toxic", "APPROVED: clean"],
    )

    guardrail = GuardrailAgent(
        name="retry_guard",
        main_agent=main,
        post_validator=post,
        rejection_keyword="REJECTED",
        max_retries=1,
    )

    events = list(guardrail._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 0


def test_guardrail_sync_all_retries_fail():
    """All retries fail → ERROR event."""
    main = StubAgent(name="writer", response="toxic content")
    post = StubAgent(name="checker", response="REJECTED: always bad")

    guardrail = GuardrailAgent(
        name="fail_guard",
        main_agent=main,
        post_validator=post,
        rejection_keyword="REJECTED",
        max_retries=1,
    )

    events = list(guardrail._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1
    assert "Post-validation failed" in errors[0].content


def test_guardrail_sync_result_key():
    """result_key stores the validated output."""
    main = StubAgent(name="writer", response="valid output")

    guardrail = GuardrailAgent(
        name="key_guard",
        main_agent=main,
        result_key="validated",
    )

    ctx = _ctx()
    list(guardrail._run_impl(ctx))

    assert ctx.session.state["validated"] == "valid output"


def test_guardrail_async_post_validates():
    """Async path: post-validation works."""
    main = StubAgent(name="writer", response="async output")
    post = StubAgent(name="checker", response="looks fine")

    guardrail = GuardrailAgent(
        name="async_guard",
        main_agent=main,
        post_validator=post,
        rejection_keyword="REJECTED",
    )

    async def _run():
        return [event async for event in guardrail._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 0


# ══════════════════════════════════════════════════════════════════════════════
# BestOfNAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_bestofn_sync_picks_highest_score():
    """Best-of-3 picks the response with the highest score."""
    agent = CountingAgent(
        name="writer",
        responses=["short", "medium length text", "a very long and detailed response"],
    )

    best = BestOfNAgent(
        name="best_writer",
        agent=agent,
        n=3,
        score_fn=lambda r: float(len(r)),
    )

    events = list(best._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(msgs) >= 1
    # The longest response should win
    assert any("very long" in e.content for e in msgs)


def test_bestofn_sync_default_score():
    """Default score_fn uses response length."""
    agent = CountingAgent(
        name="gen",
        responses=["x", "xxxx", "xx"],
    )

    best = BestOfNAgent(name="default_score", agent=agent, n=3)

    events = list(best._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # "xxxx" (len=4) should win
    assert any("xxxx" in e.content for e in msgs)


def test_bestofn_sync_result_key():
    """result_key stores scoring details."""
    agent = CountingAgent(
        name="gen",
        responses=["a", "bb", "c"],
    )

    best = BestOfNAgent(
        name="keyed",
        agent=agent,
        n=3,
        result_key="scores",
    )

    ctx = _ctx()
    list(best._run_impl(ctx))

    result = ctx.session.state["scores"]
    assert "best_index" in result
    assert "best_score" in result
    assert "all_scores" in result
    assert len(result["all_scores"]) == 3


def test_bestofn_async_picks_best():
    """Async path picks the best response."""
    agent = CountingAgent(
        name="writer",
        responses=["bad", "good response here", "meh"],
    )

    best = BestOfNAgent(
        name="async_best",
        agent=agent,
        n=3,
        score_fn=lambda r: float(len(r)),
    )

    async def _run():
        return [event async for event in best._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("good response" in e.content for e in msgs)


# ══════════════════════════════════════════════════════════════════════════════
# BatchAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_batch_sync_static_items():
    """Processes a static list of items."""
    agent = EchoAgent(name="classifier")

    batch = BatchAgent(
        name="batch_classify",
        agent=agent,
        items=["item_a", "item_b", "item_c"],
        result_key="results",
    )

    ctx = _ctx()
    events = list(batch._run_impl(ctx))

    results = ctx.session.state["results"]
    assert len(results) == 3
    assert all(isinstance(v, str) for v in results.values())


def test_batch_sync_items_from_state():
    """Reads items from session.state via items_key."""
    agent = EchoAgent(name="processor")

    batch = BatchAgent(
        name="batch_state",
        agent=agent,
        items_key="my_items",
        result_key="output",
    )

    ctx = _ctx()
    ctx.session.state["my_items"] = ["x", "y"]
    events = list(batch._run_impl(ctx))

    results = ctx.session.state["output"]
    assert len(results) == 2


def test_batch_sync_template():
    """Template formats items before sending to agent."""
    agent = EchoAgent(name="translator")

    batch = BatchAgent(
        name="batch_template",
        agent=agent,
        items=["hello", "world"],
        template="Translate: {item}",
        result_key="translations",
    )

    ctx = _ctx()
    list(batch._run_impl(ctx))

    results = ctx.session.state["translations"]
    assert "Translate: hello" in results[0] or "Translate: hello" in results[1]


def test_batch_sync_empty_items():
    """Empty items list yields ERROR."""
    agent = StubAgent(name="proc", response="ok")

    batch = BatchAgent(
        name="batch_empty",
        agent=agent,
        items=[],
        result_key="res",
    )

    events = list(batch._run_impl(_ctx()))

    errors = [e for e in events if e.event_type == EventType.ERROR]
    assert len(errors) == 1
    assert "No items" in errors[0].content


def test_batch_sync_summary_event():
    """Batch produces a summary AGENT_MESSAGE at the end."""
    agent = StubAgent(name="proc", response="done")

    batch = BatchAgent(
        name="batch_summary",
        agent=agent,
        items=["a", "b"],
        result_key="res",
    )

    events = list(batch._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Processed 2 items" in e.content for e in msgs)


def test_batch_async_processes_items():
    """Async path processes items concurrently."""
    agent = EchoAgent(name="proc")

    batch = BatchAgent(
        name="async_batch",
        agent=agent,
        items=["one", "two", "three"],
        result_key="async_res",
        max_workers=2,
    )

    ctx = _ctx()

    async def _run():
        return [event async for event in batch._run_async_impl(ctx)]

    asyncio.run(_run())

    results = ctx.session.state["async_res"]
    assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════════════
# CascadeAgent tests
# ══════════════════════════════════════════════════════════════════════════════


def test_cascade_sync_first_stage_passes():
    """First stage meets threshold — cascade stops."""
    fast = StubAgent(name="flash", response="CONFIDENT: the answer is 42")
    slow = StubAgent(name="pro", response="should not run")

    cascade = CascadeAgent(
        name="smart",
        sub_agents=[fast, slow],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.3,
        threshold=0.8,
    )

    events = list(cascade._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("42" in e.content for e in msgs)
    assert not any("should not run" in e.content for e in msgs)


def test_cascade_sync_falls_through():
    """First stage fails, second stage meets threshold."""
    fast = StubAgent(name="flash", response="unsure answer")
    slow = StubAgent(name="pro", response="CONFIDENT: correct answer")

    cascade = CascadeAgent(
        name="fallthrough",
        sub_agents=[fast, slow],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.3,
        threshold=0.8,
    )

    events = list(cascade._run_impl(_ctx()))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("correct answer" in e.content for e in msgs)


def test_cascade_sync_no_stage_meets_threshold():
    """No stage meets threshold — last stage output is returned."""
    a = StubAgent(name="tier1", response="meh")
    b = StubAgent(name="tier2", response="still meh")

    cascade = CascadeAgent(
        name="exhausted",
        sub_agents=[a, b],
        score_fn=lambda r: 0.3,
        threshold=0.8,
        result_key="cascade_info",
    )

    ctx = _ctx()
    events = list(cascade._run_impl(ctx))

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("still meh" in e.content for e in msgs)

    info = ctx.session.state["cascade_info"]
    assert info["met_threshold"] is False
    assert info["stage"] == 2


def test_cascade_sync_result_key_on_success():
    """result_key stores stage info on success."""
    agent = StubAgent(name="good", response="CONFIDENT: yes")

    cascade = CascadeAgent(
        name="keyed",
        sub_agents=[agent],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.0,
        threshold=0.5,
        result_key="info",
    )

    ctx = _ctx()
    list(cascade._run_impl(ctx))

    info = ctx.session.state["info"]
    assert info["stage"] == 1
    assert info["agent"] == "good"
    assert info["score"] == 1.0


def test_cascade_sync_state_update_events():
    """Cascade emits STATE_UPDATE events for each stage."""
    a = StubAgent(name="t1", response="low")
    b = StubAgent(name="t2", response="CONFIDENT: ok")

    cascade = CascadeAgent(
        name="stages",
        sub_agents=[a, b],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.2,
        threshold=0.8,
    )

    events = list(cascade._run_impl(_ctx()))

    updates = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert len(updates) == 2
    assert updates[0].data["stage"] == 1
    assert updates[1].data["stage"] == 2


def test_cascade_async_stops_on_threshold():
    """Async path: stops at first stage meeting threshold."""
    fast = StubAgent(name="flash", response="CONFIDENT: fast answer")

    cascade = CascadeAgent(
        name="async_cascade",
        sub_agents=[fast],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.0,
        threshold=0.5,
    )

    async def _run():
        return [event async for event in cascade._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("fast answer" in e.content for e in msgs)


def test_cascade_async_falls_through():
    """Async path: cascades through multiple stages."""
    a = StubAgent(name="t1", response="weak")
    b = StubAgent(name="t2", response="CONFIDENT: strong")

    cascade = CascadeAgent(
        name="async_fall",
        sub_agents=[a, b],
        score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.1,
        threshold=0.8,
        result_key="async_info",
    )

    ctx = _ctx()

    async def _run():
        return [event async for event in cascade._run_async_impl(ctx)]

    asyncio.run(_run())

    info = ctx.session.state["async_info"]
    assert info["agent"] == "t2"
    assert info["score"] == 1.0
