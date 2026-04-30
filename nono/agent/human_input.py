"""
HumanInputAgent - Pauses agentic workflow execution for human input.

A lightweight agent that emits a ``HUMAN_INPUT_REQUEST`` event, blocks
until a human responds via the configured handler, and emits a
``HUMAN_INPUT_RESPONSE`` event with the result.

Works as a sub-agent inside ``SequentialAgent``, ``LoopAgent``, or any
other workflow agent.

Usage:
    from nono.agent import SequentialAgent, Runner
    from nono.agent.human_input import HumanInputAgent
    from nono.hitl import HumanInputResponse

    def console_handler(step_name, state, prompt):
        answer = input(f"[{step_name}] {prompt}: ")
        return HumanInputResponse(approved=answer.lower() != "reject", message=answer)

    review = HumanInputAgent(
        name="human_review",
        handler=console_handler,
        prompt="Please review the draft and approve or suggest changes.",
    )

    pipeline = SequentialAgent(
        name="pipeline",
        sub_agents=[research_agent, review, writer_agent],
    )

    runner = Runner(pipeline)
    response = runner.run("Write about AI trends")

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, AsyncIterator, Callable, Iterator, Optional

from ..hitl import (
    AsyncHumanInputHandler,
    HumanInputHandler,
    HumanInputResponse,
    HumanRejectError,
    format_state_for_review,
)
from .base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
)

__all__ = [
    "HumanInputAgent",
]

logger = logging.getLogger("Nono.Agent.HumanInput")

# ── Callback type aliases for before/after human events ──────────────────────

BeforeHumanCallback = Callable[["HumanInputAgent", InvocationContext], Optional[str]]
"""``(agent, ctx) -> Optional[str]``.  Return a string to skip the human
interaction entirely and use that string as the agent's response."""

AfterHumanCallback = Callable[["HumanInputAgent", InvocationContext, HumanInputResponse], Optional[str]]
"""``(agent, ctx, response) -> Optional[str]``.  Return a string to
override the agent's produced message after the human responds."""


class HumanInputAgent(BaseAgent):
    """Agent that pauses execution and waits for human input.

    Emits ``HUMAN_INPUT_REQUEST`` before blocking and
    ``HUMAN_INPUT_RESPONSE`` after the human responds.  The human can
    approve, reject, or provide a custom message/prompt.

    When the human rejects and ``on_reject`` is ``"error"`` (default),
    ``HumanRejectError`` is raised.  When ``on_reject`` is ``"continue"``,
    the agent stores the rejection in ``session.state["human_rejected"]``
    and yields the human's message (or a default) as its response.

    Args:
        name: Agent name.
        description: Human-readable description.
        handler: Sync callback ``(step_name, state, prompt) -> HumanInputResponse``.
        async_handler: Async callback, used by ``_run_async_impl`` when set.
            Falls back to running ``handler`` in a thread if not provided.
        prompt: Message shown to the human when requesting input.
        on_reject: Behaviour on rejection — ``"error"`` to raise
            ``HumanRejectError``, ``"continue"`` to proceed with the
            rejection recorded in state.
        before_human: Called before the human interaction.
        after_human: Called after the human responds.
        state_key: Key in ``session.state`` where the ``HumanInputResponse``
            is stored (default ``"human_input"``).
        display_keys: State keys whose values are appended to the prompt
            so the human can review the content before approving.
            ``None`` skips enrichment (the handler still receives the
            full state).

    Example:
        >>> agent = HumanInputAgent(
        ...     name="approval",
        ...     handler=my_handler,
        ...     prompt="Approve the generated plan?",
        ...     display_keys=["draft"],
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        handler: HumanInputHandler | None = None,
        async_handler: AsyncHumanInputHandler | None = None,
        prompt: str = "Awaiting human input...",
        on_reject: str = "error",
        before_human: BeforeHumanCallback | None = None,
        after_human: AfterHumanCallback | None = None,
        state_key: str = "human_input",
        display_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description or "Human-in-the-loop checkpoint", **kwargs)
        if handler is None and async_handler is None:
            raise ValueError("At least one of 'handler' or 'async_handler' must be provided.")
        self._handler = handler
        self._async_handler = async_handler
        self.prompt = prompt
        self.on_reject = on_reject
        self.before_human = before_human
        self.after_human = after_human
        self.state_key = state_key
        self.display_keys = display_keys

    # ── Sync execution ────────────────────────────────────────────────────

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Pause for human input (sync).

        Args:
            ctx: The invocation context.

        Yields:
            HUMAN_INPUT_REQUEST, HUMAN_INPUT_RESPONSE, and AGENT_MESSAGE events.

        Raises:
            HumanRejectError: When the human rejects and ``on_reject == "error"``.
        """
        state = ctx.session.state

        # — before_human callback — may short-circuit
        if self.before_human is not None:
            early = self.before_human(self, ctx)

            if early is not None:
                logger.info("[%s] Skipped by before_human callback.", self.name)
                yield Event(EventType.AGENT_MESSAGE, self.name, early)
                return

        # — Request event (before) —
        logger.info("[%s] Requesting human input: %s", self.name, self.prompt)

        yield Event(
            EventType.HUMAN_INPUT_REQUEST,
            self.name,
            self.prompt,
            data={
                "state_keys": list(state.keys()),
                "display_keys": self.display_keys,
            },
        )

        # — Enrich prompt with review content —
        effective_prompt = self.prompt

        if self.display_keys:
            review_block = format_state_for_review(state, self.display_keys)
            effective_prompt = f"{self.prompt}\n\nContent to review:\n{review_block}"

        # — Block and wait for human —
        if self._handler is None:
            raise ValueError("No sync handler configured for HumanInputAgent.")

        response = self._handler(self.name, dict(state), effective_prompt)

        # — Response event (after) —
        logger.info(
            "[%s] Human responded: approved=%s message=%r",
            self.name, response.approved, response.message,
        )

        yield Event(
            EventType.HUMAN_INPUT_RESPONSE,
            self.name,
            response.message,
            data={
                "approved": response.approved,
                "data": response.data,
            },
        )

        # — Store in session state —
        ctx.session.state_set(self.state_key, {
            "approved": response.approved,
            "message": response.message,
            **response.data,
        })

        if response.data:
            ctx.session.state_update(response.data)

        # — after_human callback — may override message
        message = response.message or ("Approved" if response.approved else "Rejected")

        if self.after_human is not None:
            override = self.after_human(self, ctx, response)

            if override is not None:
                message = override

        # — Handle rejection —
        if not response.approved:
            ctx.session.state_set("human_rejected", True)

            if self.on_reject == "error":
                raise HumanRejectError(self.name, response.message)

        # — Final agent message —
        yield Event(EventType.AGENT_MESSAGE, self.name, message)

    # ── Async execution ───────────────────────────────────────────────────

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Pause for human input (async).

        Args:
            ctx: The invocation context.

        Yields:
            HUMAN_INPUT_REQUEST, HUMAN_INPUT_RESPONSE, and AGENT_MESSAGE events.

        Raises:
            HumanRejectError: When the human rejects and ``on_reject == "error"``.
        """
        state = ctx.session.state

        # — before_human callback — may short-circuit
        if self.before_human is not None:
            early = self.before_human(self, ctx)

            if early is not None:
                logger.info("[%s] Skipped by before_human callback (async).", self.name)
                yield Event(EventType.AGENT_MESSAGE, self.name, early)
                return

        # — Request event (before) —
        logger.info("[%s] Requesting human input (async): %s", self.name, self.prompt)

        yield Event(
            EventType.HUMAN_INPUT_REQUEST,
            self.name,
            self.prompt,
            data={
                "state_keys": list(state.keys()),
                "display_keys": self.display_keys,
            },
        )

        # — Enrich prompt with review content —
        effective_prompt = self.prompt

        if self.display_keys:
            review_block = format_state_for_review(state, self.display_keys)
            effective_prompt = f"{self.prompt}\n\nContent to review:\n{review_block}"

        # — Await human response —
        if self._async_handler is not None:
            response = await self._async_handler(self.name, dict(state), effective_prompt)
        elif self._handler is not None:
            response = await asyncio.to_thread(
                self._handler, self.name, dict(state), effective_prompt,
            )
        else:
            raise ValueError("No handler configured for HumanInputAgent.")

        # — Response event (after) —
        logger.info(
            "[%s] Human responded (async): approved=%s message=%r",
            self.name, response.approved, response.message,
        )

        yield Event(
            EventType.HUMAN_INPUT_RESPONSE,
            self.name,
            response.message,
            data={
                "approved": response.approved,
                "data": response.data,
            },
        )

        # — Store in session state —
        ctx.session.state_set(self.state_key, {
            "approved": response.approved,
            "message": response.message,
            **response.data,
        })

        if response.data:
            ctx.session.state_update(response.data)

        # — after_human callback — may override message
        message = response.message or ("Approved" if response.approved else "Rejected")

        if self.after_human is not None:
            override = self.after_human(self, ctx, response)

            if override is not None:
                message = override

        # — Handle rejection —
        if not response.approved:
            ctx.session.state_set("human_rejected", True)

            if self.on_reject == "error":
                raise HumanRejectError(self.name, response.message)

        # — Final agent message —
        yield Event(EventType.AGENT_MESSAGE, self.name, message)
