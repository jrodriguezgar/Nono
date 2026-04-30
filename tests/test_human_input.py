"""Tests for Human-in-the-Loop (HITL) — Workflow and Agent integration.

Covers:
- HumanInputResponse dataclass
- HumanRejectError exception
- Workflow.human_step() — approve, reject, reject-with-branch
- human_node() factory — approve, reject
- HumanInputAgent — approve, reject, before/after callbacks, async
- HumanInputAgent in SequentialAgent pipeline
- EventType.HUMAN_INPUT_REQUEST / HUMAN_INPUT_RESPONSE

Run:
    python tests/test_human_input.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "jinjapromptpy" not in sys.modules:
    _jpp_mock = MagicMock()
    sys.modules["jinjapromptpy"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_generator"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_template"] = _jpp_mock
    sys.modules["jinjapromptpy.batch_generator"] = _jpp_mock

from nono.hitl import (
    AsyncHumanInputHandler,
    HumanInputHandler,
    HumanInputResponse,
    HumanRejectError,
    format_state_for_review,
)
from nono.workflows import Workflow, WorkflowError, END, human_node
from nono.agent.base import Event, EventType, InvocationContext, Session
from nono.agent.human_input import HumanInputAgent
from nono.agent.workflow_agents import SequentialAgent

# ── Pass/Fail counters ──────────────────────────────────────────────────

_pass = 0
_fail = 0


def _ok(label: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def _err(label: str, exc: Exception) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL  {label}: {exc}")


# ── Helpers ──────────────────────────────────────────────────────────────

def _approve_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Always approves."""
    return HumanInputResponse(approved=True, message="Looks good!")


def _reject_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Always rejects."""
    return HumanInputResponse(approved=False, message="Needs work")


def _approve_with_data_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Approves and injects extra data."""
    return HumanInputResponse(
        approved=True,
        message="Approved with edits",
        data={"revised_topic": "AI safety"},
    )


def _conditional_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Approves if state has 'quality' > 0.8, rejects otherwise."""
    quality = state.get("quality", 0)
    if quality > 0.8:
        return HumanInputResponse(approved=True, message="Quality OK")
    return HumanInputResponse(approved=False, message=f"Quality too low: {quality}")


# =====================================================================
# 1. HumanInputResponse dataclass
# =====================================================================

def test_response_defaults() -> None:
    r = HumanInputResponse()
    assert r.approved is True
    assert r.message == ""
    assert r.data == {}
    _ok("HumanInputResponse defaults")


def test_response_with_data() -> None:
    r = HumanInputResponse(approved=False, message="No", data={"k": "v"})
    assert r.approved is False
    assert r.message == "No"
    assert r.data == {"k": "v"}
    _ok("HumanInputResponse with data")


# =====================================================================
# 2. HumanRejectError
# =====================================================================

def test_reject_error() -> None:
    err = HumanRejectError("review", "Needs more detail")
    assert err.step_name == "review"
    assert err.human_message == "Needs more detail"
    assert "review" in str(err)
    _ok("HumanRejectError attributes")


# =====================================================================
# 3. Workflow.human_step() — approve
# =====================================================================

def test_workflow_human_step_approve() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello world"})
    flow.human_step("review", handler=_approve_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"published": True})

    result = flow.run(topic="test")
    assert result["published"] is True
    assert result["human_input"]["approved"] is True
    assert result["human_input"]["message"] == "Looks good!"
    assert "human_rejected" not in result
    _ok("Workflow.human_step approve")


# =====================================================================
# 4. Workflow.human_step() — reject raises error
# =====================================================================

def test_workflow_human_step_reject_error() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=_reject_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"published": True})

    try:
        flow.run(topic="test")
        _err("Workflow.human_step reject raises HumanRejectError", Exception("No exception"))
    except HumanRejectError as e:
        assert e.step_name == "review"
        _ok("Workflow.human_step reject raises HumanRejectError")


# =====================================================================
# 5. Workflow.human_step() — reject with branch
# =====================================================================

def test_workflow_human_step_reject_branch() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step(
        "review",
        handler=_reject_handler,
        prompt="Approve?",
        on_reject="revise",
    )
    flow.step("publish", lambda s: {"published": True})
    flow.step("revise", lambda s: {"revised": True})
    flow.connect("draft", "review")
    flow.connect("review", "publish")

    result = flow.run(topic="test")
    assert result.get("revised") is True
    assert result.get("published") is not True  # should not reach publish
    assert result["human_rejected"] is True
    _ok("Workflow.human_step reject with branch")


# =====================================================================
# 6. Workflow.human_step() — approve follows normal edge
# =====================================================================

def test_workflow_human_step_approve_follows_edge() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step(
        "review",
        handler=_approve_handler,
        prompt="Approve?",
        on_reject="revise",
    )
    flow.step("publish", lambda s: {"published": True})
    flow.step("revise", lambda s: {"revised": True})
    flow.connect("draft", "review")
    flow.connect("review", "publish")

    result = flow.run(topic="test")
    assert result.get("published") is True
    assert result.get("revised") is not True
    _ok("Workflow.human_step approve follows normal edge")


# =====================================================================
# 7. Workflow.human_step() — data injection
# =====================================================================

def test_workflow_human_step_data_injection() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=_approve_with_data_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"final_topic": s.get("revised_topic", "")})

    result = flow.run(topic="test")
    assert result["revised_topic"] == "AI safety"
    assert result["final_topic"] == "AI safety"
    _ok("Workflow.human_step data injection")


# =====================================================================
# 8. human_node() factory — approve
# =====================================================================

def test_human_node_approve() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.step("review", human_node(handler=_approve_handler, prompt="Approve?"))
    flow.step("publish", lambda s: {"published": True})

    result = flow.run(topic="test")
    assert result["published"] is True
    assert result["human_input"]["approved"] is True
    _ok("human_node factory approve")


# =====================================================================
# 9. human_node() factory — reject raises error
# =====================================================================

def test_human_node_reject() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.step("review", human_node(handler=_reject_handler, prompt="Approve?"))

    try:
        flow.run(topic="test")
        _err("human_node reject raises error", Exception("No exception"))
    except HumanRejectError:
        _ok("human_node reject raises error")


# =====================================================================
# 10. human_node() factory — reject continue
# =====================================================================

def test_human_node_reject_continue() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.step("review", human_node(
        handler=_reject_handler, prompt="Approve?", on_reject="continue",
    ))
    flow.step("fallback", lambda s: {"fallback": True})

    result = flow.run(topic="test")
    assert result["human_rejected"] is True
    assert result.get("fallback") is True
    _ok("human_node reject continue")


# =====================================================================
# 11. HumanInputAgent — approve
# =====================================================================

def test_agent_hitl_approve() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Approve the plan?",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    events = list(agent._run_impl(ctx))

    event_types = [e.event_type for e in events]
    assert EventType.HUMAN_INPUT_REQUEST in event_types
    assert EventType.HUMAN_INPUT_RESPONSE in event_types
    assert EventType.AGENT_MESSAGE in event_types

    # State updated
    assert session.state["human_input"]["approved"] is True
    assert "human_rejected" not in session.state
    _ok("HumanInputAgent approve")


# =====================================================================
# 12. HumanInputAgent — reject raises error
# =====================================================================

def test_agent_hitl_reject_error() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_reject_handler,
        prompt="Approve?",
        on_reject="error",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    try:
        list(agent._run_impl(ctx))
        _err("HumanInputAgent reject raises error", Exception("No exception"))
    except HumanRejectError as e:
        assert e.step_name == "review"
        _ok("HumanInputAgent reject raises error")


# =====================================================================
# 13. HumanInputAgent — reject continue
# =====================================================================

def test_agent_hitl_reject_continue() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_reject_handler,
        prompt="Approve?",
        on_reject="continue",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    events = list(agent._run_impl(ctx))

    assert session.state["human_rejected"] is True
    last_msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE][-1]
    assert last_msg.content == "Needs work"
    _ok("HumanInputAgent reject continue")


# =====================================================================
# 14. HumanInputAgent — before_human callback skips
# =====================================================================

def test_agent_hitl_before_human_skip() -> None:
    def skip_callback(agent, ctx):
        return "Auto-approved by policy"

    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Approve?",
        before_human=skip_callback,
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    events = list(agent._run_impl(ctx))

    # Should have only AGENT_MESSAGE, no HUMAN_INPUT_REQUEST
    event_types = [e.event_type for e in events]
    assert EventType.HUMAN_INPUT_REQUEST not in event_types
    assert EventType.AGENT_MESSAGE in event_types
    assert events[0].content == "Auto-approved by policy"
    _ok("HumanInputAgent before_human skip")


# =====================================================================
# 15. HumanInputAgent — after_human callback overrides message
# =====================================================================

def test_agent_hitl_after_human_override() -> None:
    def after_cb(agent, ctx, response):
        return f"OVERRIDE: {response.message}"

    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Approve?",
        after_human=after_cb,
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    events = list(agent._run_impl(ctx))

    last_msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE][-1]
    assert last_msg.content == "OVERRIDE: Looks good!"
    _ok("HumanInputAgent after_human override")


# =====================================================================
# 16. HumanInputAgent — data injection into session state
# =====================================================================

def test_agent_hitl_data_injection() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_with_data_handler,
        prompt="Approve?",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    list(agent._run_impl(ctx))

    assert session.state["revised_topic"] == "AI safety"
    _ok("HumanInputAgent data injection")


# =====================================================================
# 17. HumanInputAgent — async execution
# =====================================================================

def test_agent_hitl_async() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Approve?",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    async def _run():
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        return events

    events = asyncio.run(_run())

    event_types = [e.event_type for e in events]
    assert EventType.HUMAN_INPUT_REQUEST in event_types
    assert EventType.HUMAN_INPUT_RESPONSE in event_types
    assert EventType.AGENT_MESSAGE in event_types
    assert session.state["human_input"]["approved"] is True
    _ok("HumanInputAgent async")


# =====================================================================
# 18. HumanInputAgent — async with async_handler
# =====================================================================

def test_agent_hitl_async_handler() -> None:
    async def async_handler(step_name, state, prompt):
        return HumanInputResponse(approved=True, message="Async approved")

    agent = HumanInputAgent(
        name="review",
        async_handler=async_handler,
        prompt="Approve?",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Check this")

    async def _run():
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        return events

    events = asyncio.run(_run())

    last_msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE][-1]
    assert last_msg.content == "Async approved"
    _ok("HumanInputAgent async_handler")


# =====================================================================
# 19. HumanInputAgent in SequentialAgent pipeline
# =====================================================================

def test_agent_hitl_in_sequential() -> None:
    class DummyAgent:
        """Minimal agent that yields one AGENT_MESSAGE."""
        def __init__(self, name, response_text):
            self.name = name
            self.description = ""
            self.sub_agents = []
            self.local_content = None
            self.before_agent_callback = None
            self.after_agent_callback = None
            self.before_tool_callback = None
            self.after_tool_callback = None

        def _run_impl_traced(self, ctx):
            yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

        def _run_impl(self, ctx):
            yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    # Use real simple agents instead
    from nono.agent.base import BaseAgent

    class SimpleAgent(BaseAgent):
        def __init__(self, name, response):
            super().__init__(name=name)
            self._response = response

        def _run_impl(self, ctx):
            yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

        async def _run_async_impl(self, ctx):
            yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    research = SimpleAgent("researcher", "Research results")
    review = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Approve research?",
    )
    writer = SimpleAgent("writer", "Final article")

    pipeline = SequentialAgent(
        name="pipeline",
        sub_agents=[research, review, writer],
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Write about AI")
    events = list(pipeline._run_impl(ctx))

    authors = [e.author for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert "researcher" in authors
    assert "review" in authors
    assert "writer" in authors
    assert session.state["human_input"]["approved"] is True
    _ok("HumanInputAgent in SequentialAgent pipeline")


# =====================================================================
# 20. HumanInputAgent — requires at least one handler
# =====================================================================

def test_agent_hitl_no_handler() -> None:
    try:
        HumanInputAgent(name="review", prompt="Approve?")
        _err("HumanInputAgent requires handler", Exception("No exception"))
    except ValueError:
        _ok("HumanInputAgent requires handler")


# =====================================================================
# 21. Workflow.human_step() with streaming
# =====================================================================

def test_workflow_human_step_stream() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=_approve_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"published": True})

    steps_seen = []
    for step_name, state in flow.stream(topic="test"):
        steps_seen.append(step_name)

    assert "review" in steps_seen
    assert "publish" in steps_seen
    _ok("Workflow.human_step streaming")


# =====================================================================
# 22. Workflow.human_step() with on_before_step / on_after_step callbacks
# =====================================================================

def test_workflow_human_step_callbacks() -> None:
    before_called = []
    after_called = []

    def before_cb(step_name, state):
        before_called.append(step_name)
        return None

    def after_cb(step_name, state, result):
        after_called.append(step_name)
        return None

    flow = Workflow("test")
    flow.on_before_step(before_cb)
    flow.on_after_step(after_cb)
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=_approve_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"published": True})

    flow.run(topic="test")

    assert "review" in before_called
    assert "review" in after_called
    _ok("Workflow.human_step triggers before/after callbacks")


# =====================================================================
# 23. Workflow.human_step() async
# =====================================================================

def test_workflow_human_step_async() -> None:
    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=_approve_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"published": True})

    result = asyncio.run(flow.run_async(topic="test"))
    assert result["published"] is True
    assert result["human_input"]["approved"] is True
    _ok("Workflow.human_step async")


# =====================================================================
# 24. HumanInputAgent — HUMAN_INPUT_REQUEST event has state keys
# =====================================================================

def test_agent_hitl_request_event_data() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Check this",
    )
    session = Session(state={"draft": "Hello", "score": 0.9})
    ctx = InvocationContext(session=session, user_message="Review")

    events = list(agent._run_impl(ctx))

    request_events = [e for e in events if e.event_type == EventType.HUMAN_INPUT_REQUEST]
    assert len(request_events) == 1
    assert "state_keys" in request_events[0].data
    assert "draft" in request_events[0].data["state_keys"]
    _ok("HUMAN_INPUT_REQUEST event contains state keys")


# =====================================================================
# 25. HumanInputAgent — HUMAN_INPUT_RESPONSE event has approved/data
# =====================================================================

def test_agent_hitl_response_event_data() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_with_data_handler,
        prompt="Check",
    )
    session = Session()
    ctx = InvocationContext(session=session, user_message="Review")

    events = list(agent._run_impl(ctx))

    response_events = [e for e in events if e.event_type == EventType.HUMAN_INPUT_RESPONSE]
    assert len(response_events) == 1
    assert response_events[0].data["approved"] is True
    assert response_events[0].data["data"]["revised_topic"] == "AI safety"
    _ok("HUMAN_INPUT_RESPONSE event contains approved + data")


# =====================================================================
# 26. format_state_for_review — all keys
# =====================================================================

def test_format_state_for_review_all() -> None:
    state = {"topic": "AI safety", "draft": "Short text"}
    result = format_state_for_review(state)
    assert "topic" in result
    assert "AI safety" in result
    assert "draft" in result
    assert "Short text" in result
    _ok("format_state_for_review shows all values")


# =====================================================================
# 27. format_state_for_review — specific display_keys
# =====================================================================

def test_format_state_for_review_display_keys() -> None:
    state = {"topic": "AI safety", "draft": "Draft text", "internal": "hidden"}
    result = format_state_for_review(state, display_keys=["draft"])
    assert "Draft text" in result
    assert "hidden" not in result
    assert "topic" not in result
    _ok("format_state_for_review filters by display_keys")


# =====================================================================
# 28. format_state_for_review — truncation
# =====================================================================

def test_format_state_for_review_truncation() -> None:
    state = {"long_text": "a" * 1000}
    result = format_state_for_review(state, max_value_len=50)
    assert len(result) < 1000
    assert "…" in result
    _ok("format_state_for_review truncates long values")


# =====================================================================
# 29. format_state_for_review — empty state
# =====================================================================

def test_format_state_for_review_empty() -> None:
    result = format_state_for_review({})
    assert "empty" in result.lower()
    _ok("format_state_for_review handles empty state")


# =====================================================================
# 30. Workflow.human_step() with display_keys
# =====================================================================

def test_workflow_human_step_display_keys() -> None:
    """display_keys enriches the prompt passed to the handler."""
    received_prompts: list[str] = []

    def capturing_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        received_prompts.append(prompt)
        return HumanInputResponse(approved=True, message="OK")

    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "The AI revolution continues."})
    flow.human_step(
        "review",
        handler=capturing_handler,
        prompt="Approve the draft?",
        display_keys=["draft"],
    )
    flow.step("publish", lambda s: {"published": True})

    result = flow.run(topic="AI")
    assert result["published"] is True
    assert len(received_prompts) == 1
    assert "The AI revolution continues." in received_prompts[0]
    assert "Content to review" in received_prompts[0]
    _ok("Workflow.human_step display_keys enriches prompt")


# =====================================================================
# 31. Workflow.human_step() without display_keys — prompt unchanged
# =====================================================================

def test_workflow_human_step_no_display_keys() -> None:
    received_prompts: list[str] = []

    def capturing_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        received_prompts.append(prompt)
        return HumanInputResponse(approved=True, message="OK")

    flow = Workflow("test")
    flow.step("draft", lambda s: {"draft": "Hello"})
    flow.human_step("review", handler=capturing_handler, prompt="Approve?")
    flow.step("publish", lambda s: {"ok": True})

    flow.run(topic="test")
    assert received_prompts[0] == "Approve?"
    _ok("Workflow.human_step without display_keys keeps prompt unchanged")


# =====================================================================
# 32. HumanInputAgent with display_keys
# =====================================================================

def test_agent_hitl_display_keys() -> None:
    received_prompts: list[str] = []

    def capturing_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        received_prompts.append(prompt)
        return HumanInputResponse(approved=True, message="OK")

    agent = HumanInputAgent(
        name="review",
        handler=capturing_handler,
        prompt="Approve the plan?",
        display_keys=["plan"],
    )
    session = Session(state={"plan": "Step 1: Research. Step 2: Write."})
    ctx = InvocationContext(session=session, user_message="Start")

    list(agent._run_impl(ctx))

    assert len(received_prompts) == 1
    assert "Step 1: Research" in received_prompts[0]
    assert "Content to review" in received_prompts[0]
    _ok("HumanInputAgent display_keys enriches prompt")


# =====================================================================
# 33. HumanInputAgent display_keys in event data
# =====================================================================

def test_agent_hitl_display_keys_event() -> None:
    agent = HumanInputAgent(
        name="review",
        handler=_approve_handler,
        prompt="Check",
        display_keys=["draft"],
    )
    session = Session(state={"draft": "Hello"})
    ctx = InvocationContext(session=session, user_message="Review")

    events = list(agent._run_impl(ctx))
    request_events = [e for e in events if e.event_type == EventType.HUMAN_INPUT_REQUEST]
    assert len(request_events) == 1
    assert request_events[0].data.get("display_keys") == ["draft"]
    _ok("HUMAN_INPUT_REQUEST event contains display_keys")


# =====================================================================
# Runner
# =====================================================================

def main() -> None:
    tests = [
        test_response_defaults,
        test_response_with_data,
        test_reject_error,
        test_workflow_human_step_approve,
        test_workflow_human_step_reject_error,
        test_workflow_human_step_reject_branch,
        test_workflow_human_step_approve_follows_edge,
        test_workflow_human_step_data_injection,
        test_human_node_approve,
        test_human_node_reject,
        test_human_node_reject_continue,
        test_agent_hitl_approve,
        test_agent_hitl_reject_error,
        test_agent_hitl_reject_continue,
        test_agent_hitl_before_human_skip,
        test_agent_hitl_after_human_override,
        test_agent_hitl_data_injection,
        test_agent_hitl_async,
        test_agent_hitl_async_handler,
        test_agent_hitl_in_sequential,
        test_agent_hitl_no_handler,
        test_workflow_human_step_stream,
        test_workflow_human_step_callbacks,
        test_workflow_human_step_async,
        test_agent_hitl_request_event_data,
        test_agent_hitl_response_event_data,
        test_format_state_for_review_all,
        test_format_state_for_review_display_keys,
        test_format_state_for_review_truncation,
        test_format_state_for_review_empty,
        test_workflow_human_step_display_keys,
        test_workflow_human_step_no_display_keys,
        test_agent_hitl_display_keys,
        test_agent_hitl_display_keys_event,
    ]

    print(f"\n{'=' * 60}")
    print("  Human-in-the-Loop (HITL) Tests")
    print(f"{'=' * 60}\n")

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            _err(test_fn.__name__, exc)

    print(f"\n{'─' * 60}")
    print(f"  Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
    print(f"{'─' * 60}\n")

    if _fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
