"""
Runner - Executes agents with session lifecycle management.

Provides a convenient API to run an agent with automatic session creation,
user-message recording, and event collection.

Part of the Nono Agent Architecture (NAA).

Usage:
    from nono.agent import Agent, Runner

    agent = Agent(name="helper", provider="google", instruction="Be helpful.")
    runner = Runner(agent)

    # Single turn
    response = runner.run("What's the capital of Spain?")

    # Multi-turn (same session)
    response = runner.run("And its population?")

    # Access session
    print(runner.session.events)
    print(runner.session.state)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator

from .base import BaseAgent, Event, EventType, InvocationContext, Session
from .tracing import TraceCollector

logger = logging.getLogger("Nono.Agent.Runner")


class Runner:
    """Execute agents with automatic session management.

    Args:
        agent: The root agent to execute.
        session: Existing session to reuse (auto-created if omitted).

    Example:
        >>> runner = Runner(my_agent)
        >>> response = runner.run("Hello!")
        >>> print(response)
        >>> print(runner.session.events)
    """

    def __init__(
        self,
        agent: BaseAgent,
        session: Session | None = None,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self.agent = agent
        self.session = session if session is not None else Session()
        self.trace_collector = trace_collector

    def run(self, user_message: str, **state_updates: Any) -> str:
        """Execute a single turn: send a user message and get the agent response.

        Args:
            user_message: The user's input message.
            **state_updates: Key-value pairs to merge into session state
                before execution.

        Returns:
            The agent's text response.
        """
        if state_updates:
            self.session.state_update(state_updates)

        ctx = InvocationContext(
            session=self.session,
            user_message=user_message,
            trace_collector=self.trace_collector,
        )

        response = self.agent.run(ctx)
        return response

    def stream(self, user_message: str, **state_updates: Any) -> Iterator[Event]:
        """Execute a turn and yield events as they are produced.

        Args:
            user_message: The user's input message.
            **state_updates: Key-value pairs to merge into session state.

        Yields:
            Event objects produced during execution.
        """
        if state_updates:
            self.session.state_update(state_updates)

        ctx = InvocationContext(
            session=self.session,
            user_message=user_message,
            trace_collector=self.trace_collector,
        )

        # before callback
        if self.agent.before_agent_callback:
            early = self.agent.before_agent_callback(self.agent, ctx)
            if early is not None:
                event = Event(EventType.AGENT_MESSAGE, self.agent.name, early)
                ctx.session.add_event(event)
                yield event
                return

        # core execution (traced)
        for event in self.agent._run_impl_traced(ctx):
            ctx.session.add_event(event)
            yield event

    def stream_text(self, user_message: str, **state_updates: Any) -> Iterator[Event]:
        """Token-level streaming — yields ``TEXT_CHUNK`` events.

        Like ``stream()``, but uses ``_run_stream_impl`` so that
        the final LLM response is streamed token-by-token.  Each
        chunk is a ``TEXT_CHUNK`` event followed by a final
        ``AGENT_MESSAGE`` with the full text.

        Args:
            user_message: The user's input message.
            **state_updates: Key-value pairs to merge into session state.

        Yields:
            Event objects — including ``TEXT_CHUNK`` for each token delta.
        """
        if state_updates:
            self.session.state_update(state_updates)

        ctx = InvocationContext(
            session=self.session,
            user_message=user_message,
            trace_collector=self.trace_collector,
        )

        if self.agent.before_agent_callback:
            early = self.agent.before_agent_callback(self.agent, ctx)
            if early is not None:
                event = Event(EventType.AGENT_MESSAGE, self.agent.name, early)
                ctx.session.add_event(event)
                yield event
                return

        from nono.agent.llm_agent import LlmAgent

        if isinstance(self.agent, LlmAgent):
            for event in self.agent._run_stream_impl(ctx):
                ctx.session.add_event(event)
                yield event
        else:
            # Non-LlmAgent: fall back to regular stream
            for event in self.agent._run_impl_traced(ctx):
                ctx.session.add_event(event)
                yield event

    async def run_async(self, user_message: str, **state_updates: Any) -> str:
        """Async version of ``run()``.

        Args:
            user_message: The user's input message.
            **state_updates: Key-value pairs to merge into session state.

        Returns:
            The agent's text response.
        """
        if state_updates:
            self.session.state_update(state_updates)

        ctx = InvocationContext(
            session=self.session,
            user_message=user_message,
            trace_collector=self.trace_collector,
        )

        return await self.agent.run_async(ctx)

    async def astream(self, user_message: str, **state_updates: Any) -> AsyncIterator[Event]:
        """Async version of ``stream()``.

        Args:
            user_message: The user's input message.
            **state_updates: Key-value pairs to merge into session state.

        Yields:
            Event objects produced during execution.
        """
        if state_updates:
            self.session.state_update(state_updates)

        ctx = InvocationContext(
            session=self.session,
            user_message=user_message,
            trace_collector=self.trace_collector,
        )

        if self.agent.before_agent_callback:
            early = self.agent.before_agent_callback(self.agent, ctx)
            if early is not None:
                event = Event(EventType.AGENT_MESSAGE, self.agent.name, early)
                ctx.session.add_event(event)
                yield event
                return

        async for event in self.agent._run_async_impl_traced(ctx):
            ctx.session.add_event(event)
            yield event

    def reset(self) -> None:
        """Create a fresh session, discarding the current one."""
        self.session = Session()
        logger.info("Session reset for agent %r.", self.agent.name)

    @property
    def history(self) -> list[Event]:
        """Returns the session event history."""
        return self.session.events

    def __repr__(self) -> str:
        return (
            f"Runner(agent={self.agent.name!r}, "
            f"session={self.session.session_id!r}, "
            f"events={len(self.session)})"
        )
