"""
Example: Agent - AI agent framework with tool calling and orchestration.
Usage: python agent_example.py

This example demonstrates:
1. Creating an LLM-powered agent (requires API key)
2. Defining tools with the @tool decorator
3. SequentialAgent: running sub-agents in order
4. ParallelAgent: running sub-agents concurrently
5. LoopAgent: iterating until a condition is met
6. Using Runner for convenient session management

NOTE: Examples 1 and 2 require a valid API key for the chosen provider.
      Examples 3-5 use mock agents that do not call an LLM.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.agent import (
    Agent,
    Runner,
    Session,
    FunctionTool,
    tool,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
    Event,
    EventType,
    InvocationContext,
    BaseAgent,
)
from typing import Any, AsyncIterator, Iterator


# ── Mock agent for offline examples ──────────────────────────────────────────

class MockAgent(BaseAgent):
    """A fake agent that returns a fixed response (no LLM call)."""

    def __init__(self, *, name: str, reply: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.reply = reply

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)


# ── Example 1: Simple agent with Runner (requires API key) ──────────────────

def example_basic_agent():
    """Basic LlmAgent usage with Runner.

    Requires a valid Google API key configured in config.toml or environment.
    """
    print(f"\n{'='*60}")
    print("  Example 1: Basic LLM Agent")
    print(f"{'='*60}")

    agent = Agent(
        name="assistant",
        model="gemini-3-flash-preview",
        provider="google",
        instruction="You are a helpful and concise assistant.",
    )

    runner = Runner(agent=agent)
    response = runner.run("What is the capital of France? Reply in one sentence.")
    print(f"Response: {response}")


# ── Example 2: Agent with tools (requires API key) ──────────────────────────

def example_agent_with_tools():
    """Agent with custom tools for function calling.

    Requires a valid Google API key configured in config.toml or environment.
    """
    print(f"\n{'='*60}")
    print("  Example 2: Agent with Tools")
    print(f"{'='*60}")

    @tool(description="Add two numbers together.")
    def add(a: int, b: int) -> str:
        return str(a + b)

    @tool(description="Multiply two numbers together.")
    def multiply(a: int, b: int) -> str:
        return str(a * b)

    agent = Agent(
        name="calculator",
        model="gemini-3-flash-preview",
        provider="google",
        instruction="You are a calculator assistant. Use tools for all math operations.",
        tools=[add, multiply],
    )

    runner = Runner(agent=agent)
    response = runner.run("What is 7 + 3, and then multiply the result by 5?")
    print(f"Response: {response}")

    # Inspect the session events
    print(f"\nSession events ({len(runner.session.events)}):")
    for event in runner.session.events:
        print(f"  [{event.event_type.value}] {event.author}: {event.content[:80]}")


# ── Example 3: SequentialAgent (offline) ─────────────────────────────────────

def example_sequential_agent():
    """SequentialAgent runs sub-agents one after another."""
    print(f"\n{'='*60}")
    print("  Example 3: Sequential Agent")
    print(f"{'='*60}")

    researcher = MockAgent(name="researcher", reply="Research: AI is transforming software.")
    writer = MockAgent(name="writer", reply="Draft: AI-powered apps are the future.")
    reviewer = MockAgent(name="reviewer", reply="Review: Article is clear and well-structured.")

    pipeline = SequentialAgent(
        name="article_pipeline",
        description="Write and review an article",
        sub_agents=[researcher, writer, reviewer],
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Write an article about AI")
    response = pipeline.run(ctx)
    print(f"Final response: {response}")

    print(f"\nAll events ({len(session.events)}):")
    for event in session.events:
        print(f"  [{event.author}] {event.content}")


# ── Example 4: ParallelAgent (offline) ──────────────────────────────────────

def example_parallel_agent():
    """ParallelAgent runs sub-agents concurrently."""
    print(f"\n{'='*60}")
    print("  Example 4: Parallel Agent")
    print(f"{'='*60}")

    web_search = MockAgent(name="web_search", reply="Web: Found 5 articles about quantum computing.")
    db_search = MockAgent(name="db_search", reply="DB: Found 3 internal reports on quantum research.")
    news_feed = MockAgent(name="news_feed", reply="News: 2 recent papers published this week.")

    gather = ParallelAgent(
        name="gather_info",
        description="Gather information from multiple sources",
        sub_agents=[web_search, db_search, news_feed],
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Find info about quantum computing")
    response = gather.run(ctx)
    print(f"Final response: {response}")

    print(f"\nAll events ({len(session.events)}):")
    for event in session.events:
        print(f"  [{event.author}] {event.content}")


# ── Example 5: LoopAgent (offline) ──────────────────────────────────────────

def example_loop_agent():
    """LoopAgent repeats sub-agents until a condition is met."""
    print(f"\n{'='*60}")
    print("  Example 5: Loop Agent")
    print(f"{'='*60}")

    iteration_counter = {"count": 0}

    class ImprovingAgent(BaseAgent):
        """Simulates an agent that improves quality each iteration."""
        def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
            iteration_counter["count"] += 1
            quality = iteration_counter["count"] * 0.35
            ctx.session.state["quality"] = quality

            yield Event(
                EventType.STATE_UPDATE, self.name, "",
                data={"quality": quality},
            )
            yield Event(
                EventType.AGENT_MESSAGE, self.name,
                f"Improved draft (quality={quality:.2f})",
            )

        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
            for event in self._run_impl(ctx):
                yield event

    improver = ImprovingAgent(name="improver", description="Improves the draft")

    loop = LoopAgent(
        name="refine_loop",
        description="Refine until quality > 0.9",
        sub_agents=[improver],
        max_iterations=5,
        stop_condition=lambda state: state.get("quality", 0) > 0.9,
    )

    session = Session()
    ctx = InvocationContext(session=session, user_message="Refine my article")
    response = loop.run(ctx)
    print(f"Final response: {response}")
    print(f"Final quality: {session.state.get('quality', 0):.2f}")
    print(f"Iterations: {iteration_counter['count']}")


# ── Example 6: Tool decorator and FunctionTool API ──────────────────────────

def example_tool_api():
    """Demonstrates the tool decorator and FunctionTool API."""
    print(f"\n{'='*60}")
    print("  Example 6: Tool API")
    print(f"{'='*60}")

    # Using the @tool decorator
    @tool(description="Convert Celsius to Fahrenheit.")
    def celsius_to_fahrenheit(celsius: float) -> str:
        return f"{celsius * 9/5 + 32:.1f}°F"

    print(f"Tool name: {celsius_to_fahrenheit.name}")
    print(f"Description: {celsius_to_fahrenheit.description}")
    print(f"Schema: {celsius_to_fahrenheit.parameters_schema}")
    print(f"Declaration: {celsius_to_fahrenheit.to_function_declaration()}")
    print(f"Result: {celsius_to_fahrenheit.invoke({'celsius': 100})}")

    # Creating a FunctionTool manually
    def search(query: str, max_results: int = 5) -> str:
        return f"Found {max_results} results for '{query}'"

    search_tool = FunctionTool(search, description="Search for information.")
    print(f"\nManual tool: {search_tool.name}")
    print(f"Schema: {search_tool.parameters_schema}")
    print(f"Result: {search_tool.invoke({'query': 'Python', 'max_results': 3})}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Offline examples (no API key needed)
    example_tool_api()
    example_sequential_agent()
    example_parallel_agent()
    example_loop_agent()

    # Online examples (uncomment if you have an API key configured)
    # example_basic_agent()
    # example_agent_with_tools()
