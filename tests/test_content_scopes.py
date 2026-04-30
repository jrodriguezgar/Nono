"""
Tests for dual-scope content: shared (session) and local (agent).

Verifies that:
- Each agent has its own ``local_content`` (private).
- Session ``shared_content`` is visible to all agents.
- ToolContext exposes both scopes with ``scope`` parameter.
- Scopes do not leak between agents.
"""

import asyncio
import json
import sys
import os
from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
    SharedContent,
)
from nono.agent.tool import FunctionTool, ToolContext, tool
from nono.agent.llm_agent import LlmAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

class MockAgent(BaseAgent):
    """Mock agent that returns a fixed reply."""

    def __init__(self, *, name: str, reply: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.reply = reply

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_agent_has_local_content():
    """Every agent gets its own local_content instance."""
    a = MockAgent(name="a", reply="ok")
    b = MockAgent(name="b", reply="ok")

    assert isinstance(a.local_content, SharedContent)
    assert isinstance(b.local_content, SharedContent)
    assert a.local_content is not b.local_content
    print("PASS: test_agent_has_local_content")


def test_local_content_is_private():
    """Content saved in local_content is not visible to other agents."""
    a = MockAgent(name="a", reply="ok")
    b = MockAgent(name="b", reply="ok")

    a.local_content.save("secret", "agent_a_data")
    assert a.local_content.load("secret") is not None
    assert b.local_content.load("secret") is None
    print("PASS: test_local_content_is_private")


def test_shared_content_is_session_wide():
    """shared_content on session is visible to all agents through ctx."""
    session = Session()
    session.shared_content.save("report", "shared data")

    # Both agents share the same session
    ctx_a = InvocationContext(session=session, user_message="hi")
    ctx_b = InvocationContext(session=session, user_message="hi")

    assert ctx_a.session.shared_content.load("report") is not None
    assert ctx_b.session.shared_content.load("report") is not None
    assert ctx_a.session.shared_content is ctx_b.session.shared_content
    print("PASS: test_shared_content_is_session_wide")


def test_tool_context_has_both_scopes():
    """ToolContext exposes shared_content and local_content."""
    session_content = SharedContent()
    agent_content = SharedContent()

    tc = ToolContext(
        shared_content=session_content,
        local_content=agent_content,
        agent_name="test",
    )

    assert tc.shared_content is session_content
    assert tc.local_content is agent_content
    print("PASS: test_tool_context_has_both_scopes")


def test_tool_context_save_load_shared():
    """save_content/load_content default to shared scope."""
    tc = ToolContext(agent_name="agent_x")

    tc.save_content("doc", "hello world")
    item = tc.load_content("doc")

    assert item is not None
    assert item.data == "hello world"
    assert item.created_by == "agent_x"
    # Should be in shared, not local
    assert "doc" in tc.shared_content
    assert "doc" not in tc.local_content
    print("PASS: test_tool_context_save_load_shared")


def test_tool_context_save_load_local():
    """save_content/load_content with scope='local' uses agent-private store."""
    tc = ToolContext(agent_name="agent_x")

    tc.save_content("scratch", "private data", scope="local")
    item = tc.load_content("scratch", scope="local")

    assert item is not None
    assert item.data == "private data"
    # Should be in local, not shared
    assert "scratch" in tc.local_content
    assert "scratch" not in tc.shared_content
    print("PASS: test_tool_context_save_load_local")


def test_tool_with_both_scopes():
    """A tool function can write to both shared and local content."""
    @tool(description="Process data.")
    def process(text: str, tool_context: ToolContext) -> str:
        tool_context.save_content("result", text)  # shared
        tool_context.save_content("draft", f"WIP: {text}", scope="local")  # local
        return "done"

    session_content = SharedContent()
    agent_content = SharedContent()
    tc = ToolContext(
        shared_content=session_content,
        local_content=agent_content,
        agent_name="processor",
    )

    result = process.invoke({"text": "hello"}, tool_context=tc)
    assert result == "done"
    assert session_content.load("result").data == "hello"
    assert agent_content.load("draft").data == "WIP: hello"
    assert session_content.load("draft") is None  # not in shared
    assert agent_content.load("result") is None   # not in local
    print("PASS: test_tool_with_both_scopes")


def test_llm_agent_passes_local_content():
    """LlmAgent passes its local_content to ToolContext during tool calls."""
    captured_contexts: list[ToolContext] = []

    @tool(description="Capture context.")
    def capture(tool_context: ToolContext) -> str:
        captured_contexts.append(tool_context)
        return "captured"

    agent = LlmAgent(
        name="test_agent",
        instruction="You are helpful.",
        tools=[capture],
    )
    agent._service = MagicMock()

    # Simulate LLM returning a tool call, then a final response
    tool_call = json.dumps({"name": "capture", "arguments": {}})
    agent._service.generate_completion = MagicMock(
        side_effect=[tool_call, "done"]
    )

    # Pre-populate agent's local content
    agent.local_content.save("private_key", "private_value")

    session = Session()
    session.shared_content.save("shared_key", "shared_value")
    ctx = InvocationContext(session=session, user_message="test")
    agent.run(ctx)

    assert len(captured_contexts) == 1
    tc = captured_contexts[0]
    # Shared content from session
    assert tc.shared_content is session.shared_content
    assert tc.shared_content.load("shared_key").data == "shared_value"
    # Local content from agent
    assert tc.local_content is agent.local_content
    assert tc.local_content.load("private_key").data == "private_value"
    print("PASS: test_llm_agent_passes_local_content")


def test_two_agents_different_local_same_shared():
    """Two agents on the same session have isolated local but shared content."""
    a = MockAgent(name="a", reply="ok")
    b = MockAgent(name="b", reply="ok")

    session = Session()

    # Save to shared via session
    session.shared_content.save("global", "visible to all")

    # Save to local per agent
    a.local_content.save("scratch", "a_private")
    b.local_content.save("scratch", "b_private")

    # Shared is the same
    assert session.shared_content.load("global").data == "visible to all"

    # Local is isolated
    assert a.local_content.load("scratch").data == "a_private"
    assert b.local_content.load("scratch").data == "b_private"

    # Cross-check: local doesn't leak
    assert "global" not in a.local_content
    assert "global" not in b.local_content
    print("PASS: test_two_agents_different_local_same_shared")


def test_tool_context_defaults():
    """ToolContext with no arguments gets empty SharedContent for both."""
    tc = ToolContext()
    assert isinstance(tc.shared_content, SharedContent)
    assert isinstance(tc.local_content, SharedContent)
    assert tc.shared_content is not tc.local_content
    assert len(tc.shared_content) == 0
    assert len(tc.local_content) == 0
    print("PASS: test_tool_context_defaults")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_agent_has_local_content()
    test_local_content_is_private()
    test_shared_content_is_session_wide()
    test_tool_context_has_both_scopes()
    test_tool_context_save_load_shared()
    test_tool_context_save_load_local()
    test_tool_with_both_scopes()
    test_llm_agent_passes_local_content()
    test_two_agents_different_local_same_shared()
    test_tool_context_defaults()
    print(f"\n{'='*60}")
    print("  All 10 tests PASSED")
    print(f"{'='*60}")
