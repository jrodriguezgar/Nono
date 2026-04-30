"""
Tests for the transfer_to_agent automatic tool.

Verifies that when an LlmAgent has sub_agents, the transfer_to_agent tool is
auto-registered and delegation works through the standard tool-calling mechanism.
"""

import asyncio
import json
import sys
import os
from typing import Any, AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nono.agent.base import BaseAgent, Event, EventType, InvocationContext, Session
from nono.agent.tool import FunctionTool, tool
from nono.agent.llm_agent import LlmAgent, _TRANSFER_TOOL_NAME


# ── Helpers ──────────────────────────────────────────────────────────────────

class MockSubAgent(BaseAgent):
    """Simple mock agent that returns a fixed reply."""

    def __init__(self, *, name: str, reply: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.reply = reply

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)


def _make_agent_with_sub_agents(
    sub_agents: list[BaseAgent],
    user_tools: list[FunctionTool] | None = None,
) -> LlmAgent:
    """Create a parent LlmAgent with sub-agents and a mock service."""
    agent = LlmAgent(
        name="coordinator",
        instruction="You are a coordinator.",
        tools=user_tools or [],
        sub_agents=sub_agents,
    )
    # Mock the service so we never hit a real API
    agent._service = MagicMock()
    return agent


# ── Tests ────────────────────────────────────────────────────────────────────

def test_transfer_tool_auto_registered():
    """transfer_to_agent tool appears in _all_tools when sub_agents exist."""
    math_agent = MockSubAgent(name="math_agent", reply="42", description="Math expert")
    agent = _make_agent_with_sub_agents([math_agent])

    all_tools = agent._all_tools
    names = [t.name for t in all_tools]
    assert _TRANSFER_TOOL_NAME in names, f"Expected {_TRANSFER_TOOL_NAME} in {names}"
    print("PASS: test_transfer_tool_auto_registered")


def test_transfer_tool_not_registered_without_sub_agents():
    """No transfer tool when there are no sub_agents."""
    agent = LlmAgent(name="solo", instruction="You work alone.")
    agent._service = MagicMock()

    all_tools = agent._all_tools
    names = [t.name for t in all_tools]
    assert _TRANSFER_TOOL_NAME not in names
    print("PASS: test_transfer_tool_not_registered_without_sub_agents")


def test_transfer_tool_schema():
    """The auto-generated tool has agent_name and message parameters."""
    sub = MockSubAgent(name="helper", reply="ok", description="A helper")
    agent = _make_agent_with_sub_agents([sub])

    transfer_tool = [t for t in agent._all_tools if t.name == _TRANSFER_TOOL_NAME][0]
    decl = transfer_tool.to_function_declaration()
    params = decl["function"]["parameters"]

    assert "agent_name" in params["properties"], "Missing agent_name param"
    assert "message" in params["properties"], "Missing message param"
    assert "agent_name" in params.get("required", [])
    assert "message" in params.get("required", [])
    print("PASS: test_transfer_tool_schema")


def test_transfer_tool_description_includes_agents():
    """The tool description lists available sub-agent names."""
    sub1 = MockSubAgent(name="writer", reply="draft", description="Writes content")
    sub2 = MockSubAgent(name="reviewer", reply="ok", description="Reviews content")
    agent = _make_agent_with_sub_agents([sub1, sub2])

    transfer_tool = [t for t in agent._all_tools if t.name == _TRANSFER_TOOL_NAME][0]
    desc = transfer_tool.description
    assert "writer" in desc
    assert "reviewer" in desc
    print("PASS: test_transfer_tool_description_includes_agents")


def test_transfer_delegates_to_sub_agent():
    """When LLM calls transfer_to_agent, the sub-agent runs and result feeds back."""
    math_agent = MockSubAgent(
        name="math_agent", reply="The answer is 42.", description="Math expert"
    )
    agent = _make_agent_with_sub_agents([math_agent])

    # First LLM call returns a transfer_to_agent tool call
    transfer_response = json.dumps({
        "name": "transfer_to_agent",
        "arguments": {"agent_name": "math_agent", "message": "What is 6 * 7?"},
    })
    # Second LLM call returns the final text answer
    final_response = "The math agent says the answer is 42."
    agent._service.generate_completion = MagicMock(
        side_effect=[transfer_response, final_response]
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Calculate 6 * 7")
    result = agent.run(ctx)

    assert result == final_response, f"Expected final response, got: {result}"

    # Verify events include AGENT_TRANSFER
    event_types = [e.event_type for e in session.events]
    assert EventType.AGENT_TRANSFER in event_types, (
        f"Expected AGENT_TRANSFER event, got: {event_types}"
    )

    # Verify the sub-agent's response is in the events
    tool_results = [e for e in session.events if e.event_type == EventType.TOOL_RESULT]
    assert any("42" in e.content for e in tool_results), (
        f"Expected sub-agent result in tool results"
    )
    print("PASS: test_transfer_delegates_to_sub_agent")


def test_transfer_unknown_agent():
    """transfer_to_agent with unknown agent_name returns an error."""
    sub = MockSubAgent(name="helper", reply="ok")
    agent = _make_agent_with_sub_agents([sub])

    transfer_response = json.dumps({
        "name": "transfer_to_agent",
        "arguments": {"agent_name": "nonexistent", "message": "hello"},
    })
    final_response = "Sorry, I could not transfer."
    agent._service.generate_completion = MagicMock(
        side_effect=[transfer_response, final_response]
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Help me")
    result = agent.run(ctx)

    # Should still get a final answer (LLM processes the error)
    assert result == final_response

    # The error message should be in a TOOL_RESULT event
    tool_results = [e for e in session.events if e.event_type == EventType.TOOL_RESULT]
    assert any("not found" in e.content for e in tool_results)
    print("PASS: test_transfer_unknown_agent")


def test_transfer_mixed_with_regular_tools():
    """transfer_to_agent works alongside regular user-defined tools."""
    @tool(description="Add two numbers.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    sub = MockSubAgent(name="explainer", reply="42 is the answer to everything.")
    agent = _make_agent_with_sub_agents([sub], user_tools=[add])

    all_tools = agent._all_tools
    names = [t.name for t in all_tools]
    assert "add" in names
    assert _TRANSFER_TOOL_NAME in names
    assert len(names) == 2
    print("PASS: test_transfer_mixed_with_regular_tools")


def test_build_messages_includes_transfer_note():
    """System prompt mentions transfer_to_agent when sub_agents exist."""
    sub = MockSubAgent(name="specialist", reply="ok", description="A specialist")
    agent = _make_agent_with_sub_agents([sub])

    session = Session()
    ctx = InvocationContext(session=session, user_message="Hello")
    messages = agent._build_messages(ctx)

    system_msg = messages[0]["content"]
    assert "transfer_to_agent" in system_msg
    assert "specialist" in system_msg
    print("PASS: test_build_messages_includes_transfer_note")


def test_transfer_async():
    """Async transfer_to_agent delegates correctly."""
    math_agent = MockSubAgent(
        name="math_agent", reply="42", description="Math expert"
    )
    agent = _make_agent_with_sub_agents([math_agent])

    transfer_response = json.dumps({
        "name": "transfer_to_agent",
        "arguments": {"agent_name": "math_agent", "message": "compute"},
    })
    final_response = "Done: 42"
    agent._service.generate_completion = MagicMock(
        side_effect=[transfer_response, final_response]
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Compute something")
    result = asyncio.run(agent.run_async(ctx))

    assert result == final_response
    event_types = [e.event_type for e in session.events]
    assert EventType.AGENT_TRANSFER in event_types
    print("PASS: test_transfer_async")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_transfer_tool_auto_registered()
    test_transfer_tool_not_registered_without_sub_agents()
    test_transfer_tool_schema()
    test_transfer_tool_description_includes_agents()
    test_transfer_delegates_to_sub_agent()
    test_transfer_unknown_agent()
    test_transfer_mixed_with_regular_tools()
    test_build_messages_includes_transfer_note()
    test_transfer_async()
    print(f"\n{'='*60}")
    print("  All 9 tests PASSED")
    print(f"{'='*60}")
