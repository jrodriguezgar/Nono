"""
Workflow agents: 100 orchestration patterns for deterministic, hybrid,
and intelligent multi-agent pipelines.

Includes flow control (Sequential, Parallel, Loop, Router), collective
reasoning (MapReduce, Consensus, Voting, Debate), quality assurance
(ProducerReviewer, Guardrail, BestOfN, SelfRefine, Verifier), planning
(Planner, AdaptivePlanner, SubQuestion, Agenda), optimisation (Genetic,
SimulatedAnnealing, TabuSearch, ParticleSwarm, DifferentialEvolution,
BayesianOptimization, AntColony), advanced reasoning (TreeOfThoughts,
GraphOfThoughts, MonteCarlo, BeamSearch, SelfDiscover, Reflexion,
AnalogicalReasoning, ThreadOfThought, BufferOfThoughts), multi-agent
communication (RolePlaying, GossipProtocol, Auction, DelphiMethod,
NominalGroup, ContractNet), retrieval-augmented (ActiveRetrieval,
IterativeRetrieval, DemonstrateSearchPredict), and more.

Deterministic orchestration agents that control execution flow of sub-agents
without using an LLM for routing decisions — plus ``RouterAgent`` which uses
a lightweight LLM call to dynamically pick the best sub-agent.

Usage:
    from nono.agent import Agent, SequentialAgent, ParallelAgent, LoopAgent, RouterAgent

    research = Agent(name="researcher", ...)
    writer   = Agent(name="writer", ...)
    reviewer = Agent(name="reviewer", ...)

    # Sequential: run agents one after another
    pipeline = SequentialAgent(
        name="pipeline",
        sub_agents=[research, writer, reviewer],
    )

    # Parallel: run agents concurrently
    parallel = ParallelAgent(
        name="gather",
        sub_agents=[research, writer],
    )

    # Loop: repeat until a condition is met
    loop = LoopAgent(
        name="refine",
        sub_agents=[writer, reviewer],
        max_iterations=3,
    )

    # Router: LLM picks the best agent per request
    router = RouterAgent(
        name="dispatcher",
        model="gemini-3-flash-preview",
        provider="google",
        sub_agents=[research, writer, reviewer],
    )
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import hashlib
import io
import json
import logging
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, AsyncIterator, Callable, Iterator, Optional

# Pre-compiled regex patterns for LLM output parsing
_LIST_ITEM_RE = re.compile(r"^\s*\d+[\.)\-]\s*(.+)", re.MULTILINE)
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DIGITS_RE = re.compile(r"\d+")

from .base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)

logger = logging.getLogger("Nono.Agent.Workflow")

_MAX_DEFAULT_PARALLEL_WORKERS: int = 32
"""Hard cap on the default thread pool size for :class:`ParallelAgent`.

When ``max_workers`` is not explicitly set, the agent uses
``len(sub_agents)`` (one thread per agent).  This constant prevents
accidentally spawning hundreds of OS threads if a very large list of
sub-agents is passed without an explicit limit.
"""

_MAX_HISTORY_ENTRY_CHARS: int = 4_000
"""Truncation limit for worker/agent outputs stored in delegation history.

Prevents unbounded memory growth when sub-agents produce very large
responses that are accumulated in the supervisor/manager/group-chat
history lists.
"""


def _truncate(text: str, limit: int = _MAX_HISTORY_ENTRY_CHARS) -> str:
    """Return *text* truncated to *limit* characters, with an indicator."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


_SERVICE_INIT_LOCK = threading.Lock()
"""Guards lazy service initialization in workflow agents (double-checked)."""

_FUTURE_TIMEOUT: int = 300
"""Timeout in seconds for ``future.result()`` inside ThreadPoolExecutor loops.

Prevents indefinite hangs when a sub-agent blocks forever.
"""


class SequentialAgent(BaseAgent):
    """Run sub-agents one after another in order.

    Each sub-agent receives the same session (with accumulated state and
    events).  The output of agent N is visible to agent N+1 through the
    session history.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ordered list of agents to execute sequentially.

    Example:
        >>> pipeline = SequentialAgent(
        ...     name="pipeline",
        ...     sub_agents=[research_agent, writer_agent, review_agent],
        ... )
        >>> result = pipeline.run(ctx)
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents, **kwargs)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute each sub-agent sequentially.

        If a sub-agent raises an exception, an error event is emitted and
        execution continues with the next sub-agent.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by each sub-agent.
        """
        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        agents_run = 0
        halted = False

        for i, agent in enumerate(self.sub_agents):
            logger.info("[%s] Running sub-agent: %s", self.name, agent.name)

            # on_agent_start hook
            if self._agent_executing is not None:
                self._agent_executing(agent.name, ctx.session)

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=ctx.user_message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            error_msg: str | None = None
            try:
                for event in agent._run_impl_traced(sub_ctx):
                    yield event

                    # Update user_message for next agent with last agent response
                    if event.event_type == EventType.AGENT_MESSAGE:
                        ctx.user_message = event.content
            except Exception as exc:
                error_msg = str(exc)
                logger.exception(
                    "[%s] Sub-agent %r failed", self.name, agent.name,
                )
                yield Event(
                    EventType.ERROR, self.name,
                    f"Sub-agent {agent.name!r} failed: {agent.name}",
                    data={"failed_agent": agent.name},
                )

            agents_run += 1

            # on_agent_end hook
            if self._agent_executed is not None:
                self._agent_executed(agent.name, ctx.session, error_msg)

            # on_between_agents hook — may halt execution
            if self._between_agents is not None:
                next_name = self.sub_agents[i + 1].name if i + 1 < len(self.sub_agents) else None
                if self._between_agents(agent.name, next_name, ctx.session) is False:
                    logger.info(
                        "[%s] on_between_agents halted execution after %r.",
                        self.name, agent.name,
                    )
                    halted = True
                    break

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: execute each sub-agent sequentially.

        If a sub-agent raises an exception, an error event is emitted and
        execution continues with the next sub-agent.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by each sub-agent.
        """
        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        agents_run = 0

        for i, agent in enumerate(self.sub_agents):
            logger.info("[%s] Running sub-agent (async): %s", self.name, agent.name)

            # on_agent_start hook
            if self._agent_executing is not None:
                self._agent_executing(agent.name, ctx.session)

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=ctx.user_message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            error_msg: str | None = None
            try:
                async for event in agent._run_async_impl_traced(sub_ctx):
                    yield event

                    if event.event_type == EventType.AGENT_MESSAGE:
                        ctx.user_message = event.content
            except Exception as exc:
                error_msg = str(exc)
                logger.exception(
                    "[%s] Sub-agent %r failed (async)", self.name, agent.name,
                )
                yield Event(
                    EventType.ERROR, self.name,
                    f"Sub-agent {agent.name!r} failed: {agent.name}",
                    data={"failed_agent": agent.name},
                )

            agents_run += 1

            # on_agent_end hook
            if self._agent_executed is not None:
                self._agent_executed(agent.name, ctx.session, error_msg)

            # on_between_agents hook — may halt execution
            if self._between_agents is not None:
                next_name = self.sub_agents[i + 1].name if i + 1 < len(self.sub_agents) else None
                if self._between_agents(agent.name, next_name, ctx.session) is False:
                    logger.info(
                        "[%s] on_between_agents halted execution after %r (async).",
                        self.name, agent.name,
                    )
                    break

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)


class ParallelAgent(BaseAgent):
    """Run sub-agents concurrently and collect their results.

    By default all sub-agents receive the same ``user_message``.  Use
    ``message_map`` to send a different message to specific agents.
    All sub-agents share the same session for event recording.  Results
    are collected in completion order.

    When ``result_key`` is provided, ``ParallelAgent`` automatically
    collects the last ``AGENT_MESSAGE`` content from every sub-agent and
    writes the mapping ``{agent_name: response}`` into
    ``session.state[result_key]``.  This makes all parallel responses
    available to subsequent agents in a ``SequentialAgent`` pipeline.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: List of agents to execute in parallel.
        max_workers: Maximum number of concurrent threads.
        message_map: Optional mapping ``{agent_name: message}``.
            Sub-agents whose name appears in the map receive that message
            instead of ``ctx.user_message``.  Agents not in the map still
            receive the original ``user_message``.
        result_key: Optional key for ``session.state``.  When set, the
            agent writes ``{agent_name: last_agent_message}`` into
            ``session.state[result_key]`` after all sub-agents finish.

    Example:
        >>> parallel = ParallelAgent(
        ...     name="gather",
        ...     sub_agents=[web_search_agent, db_search_agent],
        ... )

        >>> # Per-agent messages:
        >>> parallel = ParallelAgent(
        ...     name="gather",
        ...     sub_agents=[web_search_agent, db_search_agent],
        ...     message_map={"web_search": "AI trends", "db_search": "Q1 sales"},
        ... )

        >>> # Collect all results into session.state:
        >>> parallel = ParallelAgent(
        ...     name="gather",
        ...     sub_agents=[web_search_agent, db_search_agent],
        ...     result_key="parallel_results",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        max_workers: int | None = None,
        message_map: dict[str, str] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents, **kwargs)
        self.max_workers = max_workers
        self.message_map: dict[str, str] = message_map or {}
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute all sub-agents in parallel using a thread pool.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by sub-agents (in completion order).
        """
        if not self.sub_agents:
            return

        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        workers = self.max_workers or min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)

        def _run_agent(agent: BaseAgent) -> tuple[list[Event], str | None]:
            # on_agent_start hook
            if self._agent_executing is not None:
                self._agent_executing(agent.name, ctx.session)

            msg = self.message_map.get(agent.name, ctx.user_message)
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=msg,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            try:
                events = list(agent._run_impl_traced(sub_ctx))
                return events, None
            except Exception as exc:
                return [], str(exc)

        collected: dict[str, str] = {}
        agents_run = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_agent, agent): agent
                for agent in self.sub_agents
            }

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    events, error_msg = future.result(timeout=_FUTURE_TIMEOUT)
                    agents_run += 1
                    if error_msg is not None:
                        logger.error(
                            "[%s] Sub-agent %r failed: %s", self.name, agent.name, error_msg,
                        )
                        yield Event(
                            EventType.ERROR, agent.name,
                            f"Agent {agent.name} failed: {error_msg}",
                        )
                    else:
                        for event in events:
                            if (
                                self.result_key
                                and event.event_type == EventType.AGENT_MESSAGE
                            ):
                                collected[event.author] = event.content
                            yield event
                    # on_agent_end hook
                    if self._agent_executed is not None:
                        self._agent_executed(agent.name, ctx.session, error_msg)
                except Exception as e:
                    agents_run += 1
                    logger.error(
                        "[%s] Sub-agent %r failed: %s", self.name, agent.name, e,
                    )
                    error_event = Event(
                        EventType.ERROR, agent.name,
                        f"Agent {agent.name} failed: {e}",
                    )
                    yield error_event
                    if self._agent_executed is not None:
                        self._agent_executed(agent.name, ctx.session, str(e))

        if self.result_key:
            ctx.session.state_set(self.result_key, collected)

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: execute all sub-agents concurrently with ``asyncio.gather``.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by sub-agents (in completion order).
        """
        if not self.sub_agents:
            return

        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        async def _collect(agent: BaseAgent) -> tuple[BaseAgent, list[Event], str | None]:
            # on_agent_start hook
            if self._agent_executing is not None:
                self._agent_executing(agent.name, ctx.session)

            msg = self.message_map.get(agent.name, ctx.user_message)
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=msg,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            events = [event async for event in agent._run_async_impl_traced(sub_ctx)]
            return agent, events, None

        tasks = [_collect(agent) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected: dict[str, str] = {}
        agents_run = 0

        for i, result in enumerate(results):
            agent = self.sub_agents[i]
            agents_run += 1
            if isinstance(result, BaseException):
                logger.error(
                    "[%s] Sub-agent %r failed: %s", self.name, agent.name, result,
                )
                yield Event(
                    EventType.ERROR, agent.name,
                    f"Agent {agent.name} failed: {result}",
                )
                if self._agent_executed is not None:
                    self._agent_executed(agent.name, ctx.session, str(result))
            else:
                _, events, _ = result
                for event in events:
                    if (
                        self.result_key
                        and event.event_type == EventType.AGENT_MESSAGE
                    ):
                        collected[event.author] = event.content
                    yield event
                if self._agent_executed is not None:
                    self._agent_executed(agent.name, ctx.session, None)

        if self.result_key:
            ctx.session.state_set(self.result_key, collected)

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)


class LoopAgent(BaseAgent):
    """Repeat sub-agents in a loop until a condition is met or max iterations.

    In each iteration, all sub-agents run sequentially.  After each
    iteration, the ``stop_condition`` is evaluated on the session state.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Agents to execute each iteration (in order).
        max_iterations: Maximum number of loop iterations.
        stop_condition: Optional callable ``(session_state) -> bool``.
            Return ``True`` to stop the loop early.

    Example:
        >>> loop = LoopAgent(
        ...     name="refine",
        ...     sub_agents=[writer, reviewer],
        ...     max_iterations=3,
        ...     stop_condition=lambda state: state.get("quality", 0) > 0.9,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        max_iterations: int = 3,
        stop_condition: Callable[[dict[str, Any]], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents, **kwargs)
        self.max_iterations = max_iterations
        self.stop_condition = stop_condition

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute sub-agents in a loop.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced during each iteration.
        """
        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        agents_run = 0
        halted = False

        for iteration in range(1, self.max_iterations + 1):
            logger.info("[%s] Loop iteration %d/%d", self.name, iteration, self.max_iterations)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Loop iteration {iteration}/{self.max_iterations}",
                data={"iteration": iteration, "max_iterations": self.max_iterations},
            )

            for j, agent in enumerate(self.sub_agents):
                # on_agent_start hook
                if self._agent_executing is not None:
                    self._agent_executing(agent.name, ctx.session)

                sub_ctx = InvocationContext(
                    session=ctx.session,
                    user_message=ctx.user_message,
                    parent_agent=self,
                    trace_collector=ctx.trace_collector,
                )

                error_msg: str | None = None
                try:
                    for event in agent._run_impl_traced(sub_ctx):
                        yield event

                        if event.event_type == EventType.AGENT_MESSAGE:
                            ctx.user_message = event.content
                except Exception as exc:
                    error_msg = str(exc)
                    logger.exception(
                        "[%s] Sub-agent %r failed in iteration %d",
                        self.name, agent.name, iteration,
                    )
                    yield Event(
                        EventType.ERROR, self.name,
                        f"Sub-agent {agent.name!r} failed: {exc}",
                        data={"failed_agent": agent.name, "iteration": iteration},
                    )

                agents_run += 1

                # on_agent_end hook
                if self._agent_executed is not None:
                    self._agent_executed(agent.name, ctx.session, error_msg)

                # on_between_agents hook (within iteration)
                if self._between_agents is not None:
                    # Determine next: next agent in iteration, or first agent of next iteration, or None
                    if j + 1 < len(self.sub_agents):
                        next_name = self.sub_agents[j + 1].name
                    elif iteration < self.max_iterations:
                        next_name = self.sub_agents[0].name if self.sub_agents else None
                    else:
                        next_name = None
                    if self._between_agents(agent.name, next_name, ctx.session) is False:
                        logger.info(
                            "[%s] on_between_agents halted loop at iteration %d after %r.",
                            self.name, iteration, agent.name,
                        )
                        halted = True
                        break

            if halted:
                break

            # Check stop condition
            if self.stop_condition and self.stop_condition(ctx.session.state):
                logger.info(
                    "[%s] Stop condition met at iteration %d.", self.name, iteration,
                )
                break

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: execute sub-agents in a loop.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced during each iteration.
        """
        # on_start hook
        if self._on_start is not None:
            self._on_start(self.name, ctx.session)

        agents_run = 0
        halted = False

        for iteration in range(1, self.max_iterations + 1):
            logger.info("[%s] Async loop iteration %d/%d", self.name, iteration, self.max_iterations)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Loop iteration {iteration}/{self.max_iterations}",
                data={"iteration": iteration, "max_iterations": self.max_iterations},
            )

            for j, agent in enumerate(self.sub_agents):
                # on_agent_start hook
                if self._agent_executing is not None:
                    self._agent_executing(agent.name, ctx.session)

                sub_ctx = InvocationContext(
                    session=ctx.session,
                    user_message=ctx.user_message,
                    parent_agent=self,
                    trace_collector=ctx.trace_collector,
                )

                error_msg: str | None = None
                try:
                    async for event in agent._run_async_impl_traced(sub_ctx):
                        yield event

                        if event.event_type == EventType.AGENT_MESSAGE:
                            ctx.user_message = event.content
                except Exception as exc:
                    error_msg = str(exc)
                    logger.exception(
                        "[%s] Sub-agent %r failed in iteration %d (async)",
                        self.name, agent.name, iteration,
                    )
                    yield Event(
                        EventType.ERROR, self.name,
                        f"Sub-agent {agent.name!r} failed: {exc}",
                        data={"failed_agent": agent.name, "iteration": iteration},
                    )

                agents_run += 1

                # on_agent_end hook
                if self._agent_executed is not None:
                    self._agent_executed(agent.name, ctx.session, error_msg)

                # on_between_agents hook (within iteration)
                if self._between_agents is not None:
                    if j + 1 < len(self.sub_agents):
                        next_name = self.sub_agents[j + 1].name
                    elif iteration < self.max_iterations:
                        next_name = self.sub_agents[0].name if self.sub_agents else None
                    else:
                        next_name = None
                    if self._between_agents(agent.name, next_name, ctx.session) is False:
                        logger.info(
                            "[%s] on_between_agents halted loop at iteration %d after %r (async).",
                            self.name, iteration, agent.name,
                        )
                        halted = True
                        break

            if halted:
                break

            if self.stop_condition and self.stop_condition(ctx.session.state):
                logger.info(
                    "[%s] Stop condition met at iteration %d.", self.name, iteration,
                )
                break

        # on_end hook
        if self._on_end is not None:
            self._on_end(self.name, ctx.session, agents_run)


# ── RouterAgent ───────────────────────────────────────────────────────────────

class RouterAgent(BaseAgent):
    """LLM-powered orchestrator that dynamically picks agents AND execution mode.

    The LLM receives the list of available sub-agents and returns a JSON
    object specifying **which** agents to use and **how** to run them:

    - ``single``     — delegate to one agent (default).
    - ``sequential`` — run selected agents one after another.
    - ``parallel``   — run selected agents concurrently.
    - ``loop``       — repeat one agent up to *max_iterations* times.

    This makes ``RouterAgent`` a true orchestrator that composes
    ``SequentialAgent``, ``ParallelAgent``, and ``LoopAgent`` at runtime.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Candidate agents the orchestrator can use.
        model: LLM model for routing (default: provider's default).
        provider: LLM provider for the routing call.
        api_key: API key override.
        routing_instruction: Extra instruction appended to the routing prompt.
        temperature: LLM temperature for the routing call.
        max_iterations: Default max iterations when LLM chooses ``loop`` mode.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> router = RouterAgent(
        ...     name="orchestrator",
        ...     provider="google",
        ...     sub_agents=[researcher, writer, reviewer],
        ... )
        >>> # LLM may return: {"mode": "sequential", "agents": ["researcher", "writer"]}
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        routing_instruction: str = "",
        temperature: float = 0.0,
        max_iterations: int = 3,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.routing_instruction = routing_instruction
        self.temperature = temperature
        self.max_iterations = max_iterations

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service for routing."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    # ── Routing prompt ────────────────────────────────────────────────────

    def _build_routing_messages(self, user_message: str) -> list[dict[str, str]]:
        """Build the system + user messages for the routing LLM call.

        Args:
            user_message: The user's original message.

        Returns:
            List of message dicts for the LLM.
        """
        agent_list = "\n".join(
            f'  - "{a.name}": {a.description or "No description"}'
            for a in self.sub_agents
        )
        system = (
            "You are an orchestration assistant. Your ONLY job is to decide "
            "which agents to use and how to execute them.\n\n"
            f"Available agents:\n{agent_list}\n\n"
            "Execution modes:\n"
            '  - "single": delegate to one agent.\n'
            '  - "sequential": run agents one after another (output of each feeds the next).\n'
            '  - "parallel": run agents concurrently and collect all results.\n'
            '  - "loop": repeat one agent multiple iterations for refinement.\n\n'
            "Respond with a JSON object (no markdown fences):\n"
            "{\n"
            '  "mode": "single|sequential|parallel|loop",\n'
            '  "agents": ["agent_name", ...],\n'
            '  "message": "optionally refined message",\n'
            '  "max_iterations": 3  // only for loop mode\n'
            "}\n\n"
            "Rules:\n"
            "- mode MUST be one of: single, sequential, parallel, loop.\n"
            "- agents MUST be a list of one or more names from the list above.\n"
            "- For single mode, agents must have exactly one entry.\n"
            "- For loop mode, agents must have exactly one entry.\n"
            "- message is optional; omit it to forward the original request.\n"
            "- max_iterations is optional (default 3), only used in loop mode.\n"
            "- Do NOT answer the user's question — only decide the orchestration."
        )
        if self.routing_instruction:
            system += f"\n\n{self.routing_instruction}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _parse_routing_response(
        self, response: str,
    ) -> tuple[str, list[str], str | None, int]:
        """Parse the routing LLM's JSON response.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (mode, agent_names, optional message, max_iterations).
        """
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[%s] Failed to parse routing response: %s", self.name, text,
            )
            return "single", [], None, self.max_iterations

        mode = data.get("mode", "single")
        if mode not in ("single", "sequential", "parallel", "loop"):
            mode = "single"

        # Accept both "agents" (list) and legacy "agent_name" (str)
        agents: list[str] = data.get("agents") or []
        if not agents:
            agent_name = data.get("agent_name", "")
            if agent_name:
                agents = [agent_name]

        message = data.get("message") or None
        max_iter = int(data.get("max_iterations", self.max_iterations))

        return mode, agents, message, max_iter

    def _resolve_agents(self, names: list[str]) -> list[BaseAgent]:
        """Resolve agent names to BaseAgent instances, skipping unknowns.

        Args:
            names: List of agent name strings.

        Returns:
            List of resolved BaseAgent instances.
        """
        resolved: list[BaseAgent] = []
        for name in names:
            agent = self.find_sub_agent(name)
            if agent is None:
                logger.warning("[%s] Unknown agent %r — skipped", self.name, name)
            else:
                resolved.append(agent)
        return resolved

    def _build_ephemeral_agent(
        self, mode: str, agents: list[BaseAgent], max_iter: int,
    ) -> BaseAgent:
        """Build a temporary workflow agent for the chosen mode.

        Args:
            mode: Execution mode (single, sequential, parallel, loop).
            agents: Resolved sub-agents.
            max_iter: Max iterations for loop mode.

        Returns:
            The agent to execute.
        """
        if mode == "sequential" and len(agents) > 1:
            return SequentialAgent(
                name=f"{self.name}::sequential",
                sub_agents=agents,
            )
        if mode == "parallel" and len(agents) > 1:
            return ParallelAgent(
                name=f"{self.name}::parallel",
                sub_agents=agents,
            )
        if mode == "loop" and agents:
            return LoopAgent(
                name=f"{self.name}::loop",
                sub_agents=[agents[0]],
                max_iterations=max_iter,
            )
        # single or fallback
        return agents[0]

    # ── Core execution ────────────────────────────────────────────────────

    def _route(self, user_message: str) -> tuple[BaseAgent, str, str]:
        """Ask the LLM how to orchestrate the request.

        Args:
            user_message: The user's message.

        Returns:
            Tuple of (agent to execute, message to forward, mode label).
        """
        messages = self._build_routing_messages(user_message)
        response = self.service.generate_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
        )

        mode, agent_names, refined_message, max_iter = self._parse_routing_response(
            str(response),
        )
        agents = self._resolve_agents(agent_names)

        if not agents:
            logger.warning(
                "[%s] No valid agents resolved — using fallback %r",
                self.name, self.sub_agents[0].name,
            )
            return self.sub_agents[0], refined_message or user_message, "single"

        target = self._build_ephemeral_agent(mode, agents, max_iter)
        return target, refined_message or user_message, mode

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Orchestrate based on LLM routing decision.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by the orchestrated agents.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        target, message, mode = self._route(ctx.user_message)

        logger.info("[%s] Mode=%s, target=%r", self.name, mode, target.name)

        yield Event(
            EventType.AGENT_TRANSFER,
            self.name,
            f"Orchestrating ({mode}): {target.name}",
            data={
                "target_agent": target.name,
                "mode": mode,
                "message": message,
            },
        )

        sub_ctx = InvocationContext(
            session=ctx.session,
            user_message=message,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        for event in target._run_impl_traced(sub_ctx):
            yield event

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: orchestrate based on LLM routing decision.

        Args:
            ctx: The invocation context.

        Yields:
            Events produced by the orchestrated agents.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        messages = self._build_routing_messages(ctx.user_message)
        response = await asyncio.to_thread(
            self.service.generate_completion,
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
        )

        mode, agent_names, refined_message, max_iter = self._parse_routing_response(
            str(response),
        )
        agents = self._resolve_agents(agent_names)

        if not agents:
            logger.warning(
                "[%s] No valid agents resolved — using fallback %r",
                self.name, self.sub_agents[0].name,
            )
            agents = [self.sub_agents[0]]
            mode = "single"

        target = self._build_ephemeral_agent(mode, agents, max_iter)
        message = refined_message or ctx.user_message

        logger.info("[%s] Async mode=%s, target=%r", self.name, mode, target.name)

        yield Event(
            EventType.AGENT_TRANSFER,
            self.name,
            f"Orchestrating ({mode}): {target.name}",
            data={
                "target_agent": target.name,
                "mode": mode,
                "message": message,
            },
        )

        sub_ctx = InvocationContext(
            session=ctx.session,
            user_message=message,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        async for event in target._run_async_impl_traced(sub_ctx):
            yield event


# ── MapReduceAgent ────────────────────────────────────────────────────────────

class MapReduceAgent(BaseAgent):
    """Map-Reduce orchestration: fan-out to mappers, then reduce results.

    The **map** phase runs all ``sub_agents`` in parallel (each receives the
    same ``user_message`` or a per-agent message via ``message_map``).  The
    **reduce** phase feeds the collected results into a single
    ``reduce_agent`` that combines them into a final answer.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Mapper agents executed in parallel during the map phase.
        reduce_agent: Agent that receives all mapper outputs and produces
            the final combined result.
        max_workers: Maximum concurrent threads for the map phase.
        message_map: Optional ``{agent_name: message}`` overrides for
            individual mappers.
        result_key: Optional key to store the intermediate mapper results
            in ``session.state``.

    Example:
        >>> mapreduce = MapReduceAgent(
        ...     name="summarise_all",
        ...     sub_agents=[search_web, search_db, search_docs],
        ...     reduce_agent=summariser,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        reduce_agent: BaseAgent,
        max_workers: int | None = None,
        message_map: dict[str, str] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents, **kwargs)
        self.reduce_agent = reduce_agent
        self.max_workers = max_workers
        self.message_map: dict[str, str] = message_map or {}
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Map phase (parallel) then reduce phase (single agent).

        Args:
            ctx: The invocation context.

        Yields:
            Events from both mapper and reducer agents.
        """
        # ── Map phase ─────────────────────────────────────────────────
        if not self.sub_agents:
            return

        workers = self.max_workers or min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)

        def _run_mapper(agent: BaseAgent) -> list[Event]:
            msg = self.message_map.get(agent.name, ctx.user_message)
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=msg,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            return list(agent._run_impl_traced(sub_ctx))

        collected: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_mapper, agent): agent
                for agent in self.sub_agents
            }

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    events = future.result(timeout=_FUTURE_TIMEOUT)
                    for event in events:
                        if event.event_type == EventType.AGENT_MESSAGE:
                            collected[event.author] = event.content
                        yield event
                except Exception as e:
                    logger.error(
                        "[%s] Mapper %r failed: %s", self.name, agent.name, e,
                    )
                    yield Event(
                        EventType.ERROR, agent.name,
                        f"Mapper {agent.name} failed: {e}",
                    )

        if self.result_key:
            ctx.session.state_set(self.result_key, collected)

        # ── Reduce phase ──────────────────────────────────────────────
        reduce_input = "\n\n".join(
            f"[{name}]:\n{_truncate(content)}" for name, content in collected.items()
        )

        logger.info("[%s] Reduce phase with %r", self.name, self.reduce_agent.name)

        reduce_ctx = InvocationContext(
            session=ctx.session,
            user_message=reduce_input,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        for event in self.reduce_agent._run_impl_traced(reduce_ctx):
            yield event

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: map phase (concurrent) then reduce phase.

        Args:
            ctx: The invocation context.

        Yields:
            Events from both mapper and reducer agents.
        """
        if not self.sub_agents:
            return

        async def _collect_mapper(agent: BaseAgent) -> list[Event]:
            msg = self.message_map.get(agent.name, ctx.user_message)
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=msg,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            return [event async for event in agent._run_async_impl_traced(sub_ctx)]

        tasks = [_collect_mapper(agent) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected: dict[str, str] = {}

        for agent, result in zip(self.sub_agents, results):
            if isinstance(result, BaseException):
                logger.error(
                    "[%s] Mapper %r failed: %s", self.name, agent.name, result,
                )
                yield Event(
                    EventType.ERROR, agent.name,
                    f"Mapper {agent.name} failed: {result}",
                )
            else:
                for event in result:
                    if event.event_type == EventType.AGENT_MESSAGE:
                        collected[event.author] = event.content
                    yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, collected)

        # ── Reduce phase ──────────────────────────────────────────────
        reduce_input = "\n\n".join(
            f"[{name}]:\n{_truncate(content)}" for name, content in collected.items()
        )

        logger.info("[%s] Async reduce phase with %r", self.name, self.reduce_agent.name)

        reduce_ctx = InvocationContext(
            session=ctx.session,
            user_message=reduce_input,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        async for event in self.reduce_agent._run_async_impl_traced(reduce_ctx):
            yield event


# ── ConsensusAgent ────────────────────────────────────────────────────────────

class ConsensusAgent(BaseAgent):
    """Run multiple agents on the same input and synthesise a consensus.

    All ``sub_agents`` receive the **same** message and run in parallel.
    A dedicated ``judge_agent`` then reviews every response and produces
    a single consensus answer.

    This is useful when you want diverse perspectives (e.g. different models
    or prompts) and need to reconcile them into one trustworthy result.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Voter agents that each produce an independent answer.
        judge_agent: Agent that receives all voter answers and synthesises
            the final consensus.
        max_workers: Maximum concurrent threads for the voting phase.
        result_key: Optional key to store voter answers in ``session.state``.

    Example:
        >>> consensus = ConsensusAgent(
        ...     name="fact_check",
        ...     sub_agents=[model_a, model_b, model_c],
        ...     judge_agent=judge,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        judge_agent: BaseAgent,
        max_workers: int | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents, **kwargs)
        self.judge_agent = judge_agent
        self.max_workers = max_workers
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Vote phase (parallel) then judge phase.

        Args:
            ctx: The invocation context.

        Yields:
            Events from voters and the judge.
        """
        if not self.sub_agents:
            return

        workers = self.max_workers or min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)

        def _run_voter(agent: BaseAgent) -> list[Event]:
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=ctx.user_message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            return list(agent._run_impl_traced(sub_ctx))

        votes: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_voter, agent): agent
                for agent in self.sub_agents
            }

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    events = future.result(timeout=_FUTURE_TIMEOUT)
                    for event in events:
                        if event.event_type == EventType.AGENT_MESSAGE:
                            votes[event.author] = event.content
                        yield event
                except Exception as e:
                    logger.error(
                        "[%s] Voter %r failed: %s", self.name, agent.name, e,
                    )
                    yield Event(
                        EventType.ERROR, agent.name,
                        f"Voter {agent.name} failed: {e}",
                    )

        if self.result_key:
            ctx.session.state_set(self.result_key, votes)

        # ── Judge phase ───────────────────────────────────────────────
        judge_prompt = (
            "You are a consensus judge. Multiple agents answered the same question.\n"
            "Review all answers and produce a single authoritative consensus response.\n\n"
            f"Original question: {ctx.user_message}\n\n"
        )
        for voter_name, answer in votes.items():
            judge_prompt += f"[{voter_name}]:\n{_truncate(answer)}\n\n"

        logger.info("[%s] Judge phase with %r", self.name, self.judge_agent.name)

        judge_ctx = InvocationContext(
            session=ctx.session,
            user_message=judge_prompt,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        for event in self.judge_agent._run_impl_traced(judge_ctx):
            yield event

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Async: vote phase (concurrent) then judge phase.

        Args:
            ctx: The invocation context.

        Yields:
            Events from voters and the judge.
        """
        if not self.sub_agents:
            return

        async def _collect_voter(agent: BaseAgent) -> list[Event]:
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=ctx.user_message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            return [event async for event in agent._run_async_impl_traced(sub_ctx)]

        tasks = [_collect_voter(agent) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: dict[str, str] = {}

        for agent, result in zip(self.sub_agents, results):
            if isinstance(result, BaseException):
                logger.error(
                    "[%s] Voter %r failed: %s", self.name, agent.name, result,
                )
                yield Event(
                    EventType.ERROR, agent.name,
                    f"Voter {agent.name} failed: {result}",
                )
            else:
                for event in result:
                    if event.event_type == EventType.AGENT_MESSAGE:
                        votes[event.author] = event.content
                    yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, votes)

        # ── Judge phase ───────────────────────────────────────────────
        judge_prompt = (
            "You are a consensus judge. Multiple agents answered the same question.\n"
            "Review all answers and produce a single authoritative consensus response.\n\n"
            f"Original question: {ctx.user_message}\n\n"
        )
        for voter_name, answer in votes.items():
            judge_prompt += f"[{voter_name}]:\n{_truncate(answer)}\n\n"

        logger.info("[%s] Async judge phase with %r", self.name, self.judge_agent.name)

        judge_ctx = InvocationContext(
            session=ctx.session,
            user_message=judge_prompt,
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )

        async for event in self.judge_agent._run_async_impl_traced(judge_ctx):
            yield event


# ── ProducerReviewerAgent ─────────────────────────────────────────────────────

class ProducerReviewerAgent(BaseAgent):
    """Iterative produce-then-review loop with two specialised agents.

    The ``producer`` generates content, then the ``reviewer`` evaluates it
    and provides feedback.  The loop repeats until the reviewer approves
    (detected via ``approval_keyword``) or ``max_iterations`` is reached.

    On each iteration after the first, the producer receives the reviewer's
    feedback as its input so it can refine the output.

    Args:
        name: Agent name.
        description: Human-readable description.
        producer: Agent that generates or refines content.
        reviewer: Agent that evaluates the producer's output.
        max_iterations: Maximum produce-review cycles.
        approval_keyword: Substring in the reviewer's response that signals
            approval and stops the loop (case-insensitive).

    Example:
        >>> pr = ProducerReviewerAgent(
        ...     name="blog_pipeline",
        ...     producer=writer_agent,
        ...     reviewer=editor_agent,
        ...     max_iterations=3,
        ...     approval_keyword="APPROVED",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        producer: BaseAgent,
        reviewer: BaseAgent,
        max_iterations: int = 3,
        approval_keyword: str = "APPROVED",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[producer, reviewer], **kwargs,
        )
        self.producer = producer
        self.reviewer = reviewer
        self.max_iterations = max_iterations
        self.approval_keyword = approval_keyword

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Iterative produce-review loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each produce and review step.
        """
        current_input = ctx.user_message

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Iteration %d/%d", self.name, iteration, self.max_iterations,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Produce-review iteration {iteration}/{self.max_iterations}",
                data={"iteration": iteration, "max_iterations": self.max_iterations},
            )

            # ── Produce ───────────────────────────────────────────────
            producer_output = ""
            produce_ctx = InvocationContext(
                session=ctx.session,
                user_message=current_input,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            for event in self.producer._run_impl_traced(produce_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    producer_output = event.content
                yield event

            if not producer_output:
                logger.warning("[%s] Producer returned empty output.", self.name)
                break

            # ── Review ────────────────────────────────────────────────
            review_prompt = (
                f"Review the following content and provide feedback.\n"
                f"If the content is satisfactory, include '{self.approval_keyword}' "
                f"in your response.\n\n{_truncate(producer_output)}"
            )
            reviewer_output = ""
            review_ctx = InvocationContext(
                session=ctx.session,
                user_message=review_prompt,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            for event in self.reviewer._run_impl_traced(review_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    reviewer_output = event.content
                yield event

            # Check approval
            if self.approval_keyword.lower() in reviewer_output.lower():
                logger.info(
                    "[%s] Approved at iteration %d.", self.name, iteration,
                )
                break

            # Feed reviewer feedback to producer for next iteration
            current_input = (
                f"Previous output:\n{_truncate(producer_output)}\n\n"
                f"Reviewer feedback:\n{_truncate(reviewer_output)}\n\n"
                f"Please revise based on the feedback above."
            )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Iterative produce-review loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each produce and review step.
        """
        current_input = ctx.user_message

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Async iteration %d/%d", self.name, iteration, self.max_iterations,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Produce-review iteration {iteration}/{self.max_iterations}",
                data={"iteration": iteration, "max_iterations": self.max_iterations},
            )

            # ── Produce ───────────────────────────────────────────────
            producer_output = ""
            produce_ctx = InvocationContext(
                session=ctx.session,
                user_message=current_input,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            async for event in self.producer._run_async_impl_traced(produce_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    producer_output = event.content
                yield event

            if not producer_output:
                logger.warning("[%s] Producer returned empty output.", self.name)
                break

            # ── Review ────────────────────────────────────────────────
            review_prompt = (
                f"Review the following content and provide feedback.\n"
                f"If the content is satisfactory, include '{self.approval_keyword}' "
                f"in your response.\n\n{_truncate(producer_output)}"
            )
            reviewer_output = ""
            review_ctx = InvocationContext(
                session=ctx.session,
                user_message=review_prompt,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            async for event in self.reviewer._run_async_impl_traced(review_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    reviewer_output = event.content
                yield event

            # Check approval
            if self.approval_keyword.lower() in reviewer_output.lower():
                logger.info(
                    "[%s] Approved at iteration %d.", self.name, iteration,
                )
                break

            # Feed reviewer feedback to producer for next iteration
            current_input = (
                f"Previous output:\n{_truncate(producer_output)}\n\n"
                f"Reviewer feedback:\n{_truncate(reviewer_output)}\n\n"
                f"Please revise based on the feedback above."
            )


# ── DebateAgent ───────────────────────────────────────────────────────────────

class DebateAgent(BaseAgent):
    """Adversarial debate: two agents argue in rounds, a judge decides.

    In each round, ``agent_a`` argues first, then ``agent_b`` counter-argues
    seeing the opponent's position.  After ``max_rounds``, the ``judge``
    reviews the full debate transcript and produces a final verdict.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent_a: First debater.
        agent_b: Second debater.
        judge: Agent that renders the final verdict.
        max_rounds: Maximum number of debate rounds.
        resolution_keyword: Informational keyword in the verdict indicating
            the debate is resolved.

    Example:
        >>> debate = DebateAgent(
        ...     name="ai_ethics_debate",
        ...     agent_a=optimist,
        ...     agent_b=pessimist,
        ...     judge=moderator,
        ...     max_rounds=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        judge: BaseAgent,
        max_rounds: int = 3,
        resolution_keyword: str = "RESOLVED",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[agent_a, agent_b, judge], **kwargs,
        )
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.judge = judge
        self.max_rounds = max_rounds
        self.resolution_keyword = resolution_keyword

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run debate rounds then judge (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each debate round and the final verdict.
        """
        transcript: list[str] = []
        last_a = ""
        last_b = ""

        for round_num in range(1, self.max_rounds + 1):
            logger.info(
                "[%s] Debate round %d/%d", self.name, round_num, self.max_rounds,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Debate round {round_num}/{self.max_rounds}",
                data={"round": round_num, "max_rounds": self.max_rounds},
            )

            # ── Agent A argues ────────────────────────────────────────
            if round_num == 1:
                a_input = f"Debate topic: {ctx.user_message}\n\nPresent your argument."
            else:
                a_input = (
                    f"Debate topic: {ctx.user_message}\n\n"
                    f"Opponent's last argument:\n{_truncate(last_b)}\n\n"
                    f"Counter-argue and strengthen your position."
                )

            a_ctx = InvocationContext(
                session=ctx.session, user_message=a_input,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            for event in self.agent_a._run_impl_traced(a_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_a = event.content
                yield event

            transcript.append(f"[Round {round_num} - {self.agent_a.name}]:\n{_truncate(last_a)}")

            # ── Agent B counter-argues ────────────────────────────────
            b_input = (
                f"Debate topic: {ctx.user_message}\n\n"
                f"Opponent's argument:\n{_truncate(last_a)}\n\n"
                f"Counter-argue and present your position."
            )

            b_ctx = InvocationContext(
                session=ctx.session, user_message=b_input,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            for event in self.agent_b._run_impl_traced(b_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_b = event.content
                yield event

            transcript.append(f"[Round {round_num} - {self.agent_b.name}]:\n{_truncate(last_b)}")

        # ── Judge renders verdict ─────────────────────────────────────
        debate_text = "\n\n".join(transcript)
        judge_input = (
            f"You are the judge of a debate on: {ctx.user_message}\n\n"
            f"Full debate transcript:\n{debate_text}\n\n"
            f"Render your final verdict, synthesising the strongest arguments."
        )

        logger.info("[%s] Judge phase with %r", self.name, self.judge.name)

        judge_ctx = InvocationContext(
            session=ctx.session, user_message=judge_input,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        for event in self.judge._run_impl_traced(judge_ctx):
            yield event

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Run debate rounds then judge (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each debate round and the final verdict.
        """
        transcript: list[str] = []
        last_a = ""
        last_b = ""

        for round_num in range(1, self.max_rounds + 1):
            logger.info(
                "[%s] Async debate round %d/%d",
                self.name, round_num, self.max_rounds,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Debate round {round_num}/{self.max_rounds}",
                data={"round": round_num, "max_rounds": self.max_rounds},
            )

            # ── Agent A argues ────────────────────────────────────────
            if round_num == 1:
                a_input = f"Debate topic: {ctx.user_message}\n\nPresent your argument."
            else:
                a_input = (
                    f"Debate topic: {ctx.user_message}\n\n"
                    f"Opponent's last argument:\n{_truncate(last_b)}\n\n"
                    f"Counter-argue and strengthen your position."
                )

            a_ctx = InvocationContext(
                session=ctx.session, user_message=a_input,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            async for event in self.agent_a._run_async_impl_traced(a_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_a = event.content
                yield event

            transcript.append(f"[Round {round_num} - {self.agent_a.name}]:\n{_truncate(last_a)}")

            # ── Agent B counter-argues ────────────────────────────────
            b_input = (
                f"Debate topic: {ctx.user_message}\n\n"
                f"Opponent's argument:\n{_truncate(last_a)}\n\n"
                f"Counter-argue and present your position."
            )

            b_ctx = InvocationContext(
                session=ctx.session, user_message=b_input,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            async for event in self.agent_b._run_async_impl_traced(b_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_b = event.content
                yield event

            transcript.append(f"[Round {round_num} - {self.agent_b.name}]:\n{_truncate(last_b)}")

        # ── Judge renders verdict ─────────────────────────────────────
        debate_text = "\n\n".join(transcript)
        judge_input = (
            f"You are the judge of a debate on: {ctx.user_message}\n\n"
            f"Full debate transcript:\n{debate_text}\n\n"
            f"Render your final verdict, synthesising the strongest arguments."
        )

        logger.info("[%s] Async judge phase with %r", self.name, self.judge.name)

        judge_ctx = InvocationContext(
            session=ctx.session, user_message=judge_input,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        async for event in self.judge._run_async_impl_traced(judge_ctx):
            yield event


# ── EscalationAgent ───────────────────────────────────────────────────────────

class EscalationAgent(BaseAgent):
    """Try agents in order; stop at the first success.

    Each sub-agent is tried sequentially.  If a sub-agent succeeds (its
    response does **not** contain ``failure_keyword``), execution stops and
    that response is returned.  If all sub-agents fail, the last response
    is returned.

    Typical use: try a cheap/fast model first; escalate to a more powerful
    (and expensive) model only if the first one fails or is uncertain.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ordered list of agents to try (cheapest/fastest first).
        failure_keyword: Substring in a response that indicates failure
            (case-insensitive).  Triggers escalation to the next agent.
        on_escalation: Optional callback ``(agent_name, response) -> None``
            invoked each time an escalation occurs.

    Example:
        >>> escalation = EscalationAgent(
        ...     name="smart_fallback",
        ...     sub_agents=[fast_model, medium_model, powerful_model],
        ...     failure_keyword="I don't know",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        failure_keyword: str = "I don't know",
        on_escalation: Callable[[str, str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents, **kwargs,
        )
        self.failure_keyword = failure_keyword
        self.on_escalation = on_escalation

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Try each agent until one succeeds (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the agent that handles the request.
        """
        events: list[Event] = []

        for i, agent in enumerate(self.sub_agents):
            logger.info(
                "[%s] Trying agent %r (%d/%d)",
                self.name, agent.name, i + 1, len(self.sub_agents),
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Trying {agent.name} ({i + 1}/{len(self.sub_agents)})",
                data={"agent": agent.name, "level": i + 1},
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            last_content = ""
            events = []
            for event in agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content

            # Check if this agent succeeded
            if self.failure_keyword.lower() not in last_content.lower():
                yield from events
                logger.info(
                    "[%s] Resolved by %r at level %d",
                    self.name, agent.name, i + 1,
                )
                return

            # Failed — escalate
            logger.info("[%s] %r failed, escalating...", self.name, agent.name)
            if self.on_escalation:
                self.on_escalation(agent.name, last_content)

        # All agents failed — yield events from the last attempt
        yield from events

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Try each agent until one succeeds (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the agent that handles the request.
        """
        events: list[Event] = []

        for i, agent in enumerate(self.sub_agents):
            logger.info(
                "[%s] Async trying agent %r (%d/%d)",
                self.name, agent.name, i + 1, len(self.sub_agents),
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Trying {agent.name} ({i + 1}/{len(self.sub_agents)})",
                data={"agent": agent.name, "level": i + 1},
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            last_content = ""
            events = []
            async for event in agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content

            if self.failure_keyword.lower() not in last_content.lower():
                for event in events:
                    yield event
                logger.info(
                    "[%s] Resolved by %r at level %d",
                    self.name, agent.name, i + 1,
                )
                return

            logger.info("[%s] %r failed, escalating...", self.name, agent.name)
            if self.on_escalation:
                self.on_escalation(agent.name, last_content)

        for event in events:
            yield event


# ── SupervisorAgent ───────────────────────────────────────────────────────────

class SupervisorAgent(BaseAgent):
    """LLM-powered supervisor that delegates, evaluates, and re-delegates.

    The supervisor uses an LLM to:

    1. Pick a worker from ``sub_agents`` for the task.
    2. Evaluate the worker's output.
    3. Accept the result, or re-delegate to the same or different worker.

    This continues until the supervisor is satisfied or ``max_iterations``
    is reached.  Unlike ``RouterAgent`` (which routes once), the supervisor
    actively monitors and can redirect mid-execution.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Worker agents the supervisor can delegate to.
        model: LLM model for supervisor decisions.
        provider: LLM provider for supervisor decisions.
        api_key: API key override.
        supervisor_instruction: Extra instructions for the supervisor LLM.
        temperature: LLM temperature for supervisor decisions.
        max_iterations: Maximum delegation rounds.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> supervisor = SupervisorAgent(
        ...     name="manager",
        ...     provider="google",
        ...     sub_agents=[coder, writer, researcher],
        ...     max_iterations=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        supervisor_instruction: str = "",
        temperature: float = 0.0,
        max_iterations: int = 3,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents,
        )
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.supervisor_instruction = supervisor_instruction
        self.temperature = temperature
        self.max_iterations = max_iterations

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _build_delegate_messages(
        self, user_message: str, history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build messages for the supervisor's delegation/evaluation call.

        Args:
            user_message: The original user request.
            history: Previous delegation attempts and results.

        Returns:
            List of message dicts for the LLM.
        """
        agent_list = "\n".join(
            f'  - "{a.name}": {a.description or "No description"}'
            for a in self.sub_agents
        )
        system = (
            "You are a supervisor managing a team of specialist agents.\n\n"
            f"Available workers:\n{agent_list}\n\n"
            "Your job:\n"
            "1. Pick the best worker for the task (or re-delegate).\n"
            "2. After seeing a worker's output, decide: ACCEPT or "
            "RE-DELEGATE.\n\n"
            "Respond with JSON (no markdown fences):\n"
            "{\n"
            '  "action": "delegate|accept",\n'
            '  "agent": "worker_name",       // required for delegate\n'
            '  "message": "task for worker",  // optional: refine task\n'
            '  "reason": "why this decision"  // brief explanation\n'
            "}\n\n"
            "Rules:\n"
            "- First call: always delegate to a worker.\n"
            "- After seeing output: accept if good, else delegate again.\n"
            "- agent must be a name from the list above.\n"
        )
        if self.supervisor_instruction:
            system += f"\n{self.supervisor_instruction}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {user_message}"},
        ]
        messages.extend(history)
        return messages

    def _parse_supervisor_response(self, response: str) -> dict[str, str]:
        """Parse the supervisor's JSON response.

        Args:
            response: Raw LLM response text.

        Returns:
            Dict with keys: action, agent, message, reason.
        """
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[%s] Failed to parse supervisor response: %s",
                self.name, text,
            )
            return {
                "action": "accept", "agent": "",
                "message": "", "reason": "parse failure",
            }

        return {
            "action": data.get("action", "accept"),
            "agent": data.get("agent", ""),
            "message": data.get("message", ""),
            "reason": data.get("reason", ""),
        }

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Supervisor delegation loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from delegated workers and supervisor decisions.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No workers configured.")
            return

        history: list[dict[str, str]] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Supervisor round %d/%d",
                self.name, iteration, self.max_iterations,
            )

            messages = self._build_delegate_messages(ctx.user_message, history)
            response = self.service.generate_completion(
                messages=messages, temperature=self.temperature,
                max_tokens=256,
            )
            decision = self._parse_supervisor_response(str(response))

            if decision["action"] == "accept" and iteration > 1:
                logger.info(
                    "[%s] Supervisor accepted at round %d.",
                    self.name, iteration,
                )
                break

            worker = self.find_sub_agent(decision["agent"])
            if worker is None:
                logger.warning(
                    "[%s] Unknown worker %r, using first.",
                    self.name, decision["agent"],
                )
                worker = self.sub_agents[0]

            task_message = decision.get("message") or ctx.user_message

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Delegating to {worker.name}: {decision.get('reason', '')}",
                data={
                    "worker": worker.name,
                    "round": iteration,
                    "reason": decision.get("reason", ""),
                },
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=task_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            worker_output = ""
            for event in worker._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    worker_output = event.content
                yield event

            history.append({
                "role": "assistant",
                "content": json.dumps(decision),
            })
            history.append({
                "role": "user",
                "content": (
                    f"Worker '{worker.name}' responded:\n"
                    f"{_truncate(worker_output)}\n\n"
                    f"Evaluate: accept or re-delegate?"
                ),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Supervisor delegation loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from delegated workers and supervisor decisions.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No workers configured.")
            return

        history: list[dict[str, str]] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Async supervisor round %d/%d",
                self.name, iteration, self.max_iterations,
            )

            messages = self._build_delegate_messages(ctx.user_message, history)
            response = await asyncio.to_thread(
                self.service.generate_completion,
                messages=messages, temperature=self.temperature,
                max_tokens=256,
            )
            decision = self._parse_supervisor_response(str(response))

            if decision["action"] == "accept" and iteration > 1:
                logger.info(
                    "[%s] Supervisor accepted at round %d.",
                    self.name, iteration,
                )
                break

            worker = self.find_sub_agent(decision["agent"])
            if worker is None:
                logger.warning(
                    "[%s] Unknown worker %r, using first.",
                    self.name, decision["agent"],
                )
                worker = self.sub_agents[0]

            task_message = decision.get("message") or ctx.user_message

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Delegating to {worker.name}: {decision.get('reason', '')}",
                data={
                    "worker": worker.name,
                    "round": iteration,
                    "reason": decision.get("reason", ""),
                },
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=task_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            worker_output = ""
            async for event in worker._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    worker_output = event.content
                yield event

            history.append({
                "role": "assistant",
                "content": json.dumps(decision),
            })
            history.append({
                "role": "user",
                "content": (
                    f"Worker '{worker.name}' responded:\n"
                    f"{_truncate(worker_output)}\n\n"
                    f"Evaluate: accept or re-delegate?"
                ),
            })


# ── VotingAgent ───────────────────────────────────────────────────────────────

class VotingAgent(BaseAgent):
    """Majority-vote orchestration: N agents answer, most frequent wins.

    All ``sub_agents`` receive the same message and run in parallel.
    Their responses are normalised and the most frequent answer is
    selected — **no LLM judge required**.

    When there is a tie, the response from the agent that appears first
    in ``sub_agents`` wins.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Voter agents that each produce an answer.
        max_workers: Maximum concurrent threads for the voting phase.
        normalize: Optional callable to normalize responses before counting.
            Default: ``str.strip().lower()``.
        result_key: Optional key to store vote details in ``session.state``.

    Example:
        >>> voting = VotingAgent(
        ...     name="majority_vote",
        ...     sub_agents=[model_a, model_b, model_c],
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        max_workers: int | None = None,
        normalize: Callable[[str], str] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents, **kwargs,
        )
        self.max_workers = max_workers
        self.normalize = normalize or (lambda s: s.strip().lower())
        self.result_key = result_key

    def _pick_winner(self, votes: dict[str, str]) -> tuple[str, str]:
        """Select the most frequent answer.

        Args:
            votes: Mapping of ``{agent_name: raw_response}``.

        Returns:
            Tuple of (winning_agent_name, raw_response).
        """
        normalized: dict[str, str] = {
            name: self.normalize(resp) for name, resp in votes.items()
        }

        counter: collections.Counter[str] = collections.Counter(
            normalized.values(),
        )
        winning_norm = counter.most_common(1)[0][0]

        for name, norm in normalized.items():
            if norm == winning_norm:
                return name, votes[name]

        first = next(iter(votes))
        return first, votes[first]

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Vote phase (parallel) then pick majority (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from voters, then the winning response.
        """
        if not self.sub_agents:
            return

        workers = self.max_workers or min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)

        def _run_voter(agent: BaseAgent) -> list[Event]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            return list(agent._run_impl_traced(sub_ctx))

        votes: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_voter, agent): agent
                for agent in self.sub_agents
            }

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    events = future.result(timeout=_FUTURE_TIMEOUT)
                    for event in events:
                        if event.event_type == EventType.AGENT_MESSAGE:
                            votes[event.author] = event.content
                        yield event
                except Exception as exc:
                    logger.error(
                        "[%s] Voter %r failed: %s",
                        self.name, agent.name, exc,
                    )
                    yield Event(
                        EventType.ERROR, agent.name,
                        f"Voter {agent.name} failed: {exc}",
                    )

        if not votes:
            return

        winner_name, winner_response = self._pick_winner(votes)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "votes": {k: _truncate(v) for k, v in votes.items()},
                "winner": winner_name,
                "response": winner_response,
            })

        logger.info("[%s] Winner: %r", self.name, winner_name)

        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            winner_response,
            data={"winner": winner_name, "total_votes": len(votes)},
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Vote phase (concurrent) then pick majority (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from voters, then the winning response.
        """
        if not self.sub_agents:
            return

        async def _collect_voter(agent: BaseAgent) -> list[Event]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            return [
                event async for event in agent._run_async_impl_traced(sub_ctx)
            ]

        tasks = [_collect_voter(agent) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: dict[str, str] = {}

        for agent, result in zip(self.sub_agents, results):
            if isinstance(result, BaseException):
                logger.error(
                    "[%s] Voter %r failed: %s",
                    self.name, agent.name, result,
                )
                yield Event(
                    EventType.ERROR, agent.name,
                    f"Voter {agent.name} failed: {result}",
                )
            else:
                for event in result:
                    if event.event_type == EventType.AGENT_MESSAGE:
                        votes[event.author] = event.content
                    yield event

        if not votes:
            return

        winner_name, winner_response = self._pick_winner(votes)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "votes": {k: _truncate(v) for k, v in votes.items()},
                "winner": winner_name,
                "response": winner_response,
            })

        logger.info("[%s] Async winner: %r", self.name, winner_name)

        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            winner_response,
            data={"winner": winner_name, "total_votes": len(votes)},
        )


# ── HandoffAgent ──────────────────────────────────────────────────────────────


class HandoffAgent(BaseAgent):
    """Peer-to-peer handoff: agents transfer full control to each other.

    Unlike ``transfer_to_agent`` (agent-as-tools, where the caller retains
    control), handoff transfers **full ownership** of the conversation to
    the receiving agent.  The original agent is done once it hands off.

    Topology is a **mesh**: handoff rules define which agents can hand off
    to which others.  An ``entry_agent`` receives the initial message.
    At each turn, the active agent runs with the full conversation history.
    If the agent's response contains a handoff directive (a line starting
    with ``HANDOFF:``), control is transferred to the named target.
    The loop continues until an agent completes without handing off, or
    ``max_handoffs`` is reached.

    Args:
        name: Agent name.
        description: Human-readable description.
        entry_agent: The first agent to receive the user message.
        handoff_rules: Dict ``{agent_name: [list of agents it can hand off
            to]}``.  Agents not in the map cannot initiate handoffs.
        max_handoffs: Safety limit to prevent infinite handoff chains.
        handoff_keyword: Keyword prefix for handoff directives in agent
            output.  Default ``"HANDOFF:"``.
        max_conversation_entries: Maximum conversation entries to keep in
            the handoff context window.  ``0`` (default) means unlimited.
            When set, only the most recent entries are passed to the
            active agent.

    Example:
        >>> triage = Agent(name="triage", instruction="Route to the right "
        ...     "expert. Write HANDOFF: math_tutor or HANDOFF: history_tutor.", ...)
        >>> math = Agent(name="math_tutor", instruction="Answer math.", ...)
        >>> history = Agent(name="history_tutor", instruction="Answer history.", ...)
        >>> handoff = HandoffAgent(
        ...     name="tutoring",
        ...     entry_agent=triage,
        ...     handoff_rules={
        ...         "triage": [math, history],
        ...         "math_tutor": [triage],
        ...         "history_tutor": [triage],
        ...     },
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        entry_agent: BaseAgent,
        handoff_rules: dict[str, list[BaseAgent]] | None = None,
        max_handoffs: int = 10,
        handoff_keyword: str = "HANDOFF:",
        max_conversation_entries: int = 0,
        **kwargs: Any,
    ) -> None:
        # Collect all unique agents for sub_agents registration.
        all_agents: dict[str, BaseAgent] = {entry_agent.name: entry_agent}
        rules = handoff_rules or {}
        for targets in rules.values():
            for target in targets:
                all_agents[target.name] = target

        super().__init__(
            name=name, description=description,
            sub_agents=list(all_agents.values()), **kwargs,
        )
        self.entry_agent = entry_agent
        self.max_handoffs = max_handoffs
        self.handoff_keyword = handoff_keyword
        self.max_conversation_entries = max_conversation_entries

        # Normalised rules: agent_name → {allowed_target_name: agent}
        self._rules: dict[str, dict[str, BaseAgent]] = {}
        for source_name, targets in rules.items():
            self._rules[source_name] = {t.name: t for t in targets}

    def _extract_handoff(self, text: str) -> str | None:
        """Extract the handoff target name from agent output.

        Looks for a line starting with the ``handoff_keyword`` prefix.

        Args:
            text: Agent response text.

        Returns:
            Target agent name, or ``None`` if no handoff directive found.
        """
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith(self.handoff_keyword.upper()):
                target = stripped[len(self.handoff_keyword):].strip()
                return target
        return None

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Handoff loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each active agent and handoff state updates.
        """
        current_agent = self.entry_agent
        conversation: list[str] = [ctx.user_message]
        handoff_count = 0

        while handoff_count < self.max_handoffs:
            logger.info(
                "[%s] Active agent: %s (handoff %d/%d)",
                self.name, current_agent.name, handoff_count, self.max_handoffs,
            )

            # Build the message with full conversation context.
            if handoff_count == 0:
                message = ctx.user_message
            else:
                message = (
                    "Conversation so far:\n"
                    + "\n---\n".join(conversation)
                    + "\n\nContinue the conversation."
                )

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            agent_output = ""
            for event in current_agent._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    agent_output = event.content
                yield event

            conversation.append(f"[{current_agent.name}]: {agent_output}")
            if self.max_conversation_entries > 0:
                conversation = conversation[-self.max_conversation_entries:]

            # Check for handoff directive.
            target_name = self._extract_handoff(agent_output)
            if target_name is None:
                break

            # Validate handoff.
            allowed = self._rules.get(current_agent.name, {})
            target_agent = allowed.get(target_name)
            if target_agent is None:
                logger.warning(
                    "[%s] Agent %r tried to hand off to %r but not allowed.",
                    self.name, current_agent.name, target_name,
                )
                break

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Handoff: {current_agent.name} → {target_agent.name}",
                data={
                    "from": current_agent.name,
                    "to": target_agent.name,
                    "handoff_count": handoff_count + 1,
                },
            )

            current_agent = target_agent
            handoff_count += 1

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Handoff loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each active agent and handoff state updates.
        """
        current_agent = self.entry_agent
        conversation: list[str] = [ctx.user_message]
        handoff_count = 0

        while handoff_count < self.max_handoffs:
            logger.info(
                "[%s] Active agent (async): %s (handoff %d/%d)",
                self.name, current_agent.name, handoff_count, self.max_handoffs,
            )

            if handoff_count == 0:
                message = ctx.user_message
            else:
                message = (
                    "Conversation so far:\n"
                    + "\n---\n".join(conversation)
                    + "\n\nContinue the conversation."
                )

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=message,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            agent_output = ""
            async for event in current_agent._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    agent_output = event.content
                yield event

            conversation.append(f"[{current_agent.name}]: {agent_output}")
            if self.max_conversation_entries > 0:
                conversation = conversation[-self.max_conversation_entries:]

            target_name = self._extract_handoff(agent_output)
            if target_name is None:
                break

            allowed = self._rules.get(current_agent.name, {})
            target_agent = allowed.get(target_name)
            if target_agent is None:
                logger.warning(
                    "[%s] Agent %r tried to hand off to %r but not allowed.",
                    self.name, current_agent.name, target_name,
                )
                break

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Handoff: {current_agent.name} → {target_agent.name}",
                data={
                    "from": current_agent.name,
                    "to": target_agent.name,
                    "handoff_count": handoff_count + 1,
                },
            )

            current_agent = target_agent
            handoff_count += 1


# ── GroupChatAgent ────────────────────────────────────────────────────────────


class GroupChatAgent(BaseAgent):
    """N-agent group chat with manager-controlled speaker selection.

    A central manager selects which agent speaks next on each round.
    All agents see the full conversation history, enabling collaborative
    refinement.  Supports round-robin, LLM-based, and custom speaker
    selection strategies.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Participant agents.
        speaker_selection: Strategy for selecting the next speaker.

            - ``"round_robin"`` (default): cycle through agents in order.
            - ``"llm"``: use an LLM call to pick the next speaker.
            - ``Callable[[list[dict], list[BaseAgent]], BaseAgent]``: custom
              function receiving ``(messages, agents)`` → next agent.

        max_rounds: Maximum conversation rounds.
        termination_condition: Optional ``(messages) -> bool`` callable.
            Checked after each turn; ``True`` stops the chat.
        termination_keyword: Optional keyword.  If found in the last
            agent message, the chat terminates.
        model: LLM model (required when ``speaker_selection="llm"``).
        provider: LLM provider (for ``"llm"`` selection).
        api_key: API key override.
        result_key: Optional key to store the full transcript in
            ``session.state[result_key]``.
        max_context_messages: Maximum number of messages to include in
            the transcript sent to each speaker.  ``0`` (default) means
            unlimited.  When set, only the most recent messages are shown.

    Example:
        >>> writer = Agent(name="writer",
        ...     instruction="Write marketing copy.", ...)
        >>> reviewer = Agent(name="reviewer",
        ...     instruction="Review copy. Say APPROVED when satisfied.", ...)
        >>> chat = GroupChatAgent(
        ...     name="creative_team",
        ...     sub_agents=[writer, reviewer],
        ...     speaker_selection="round_robin",
        ...     max_rounds=6,
        ...     termination_keyword="APPROVED",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        speaker_selection: str | Callable[..., BaseAgent] = "round_robin",
        max_rounds: int = 10,
        termination_condition: Callable[[list[dict[str, str]]], bool] | None = None,
        termination_keyword: str | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        result_key: str | None = None,
        max_context_messages: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents, **kwargs,
        )
        self.speaker_selection = speaker_selection
        self.max_rounds = max_rounds
        self.termination_condition = termination_condition
        self.termination_keyword = termination_keyword
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service: Any = None
        self.result_key = result_key
        self.max_context_messages = max_context_messages

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service (for LLM speaker selection)."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                    )
        return self._service

    def _select_speaker_round_robin(
        self, round_num: int, _messages: list[dict[str, str]],
    ) -> BaseAgent:
        """Select the next speaker using round-robin.

        Args:
            round_num: Current round number (0-indexed).
            _messages: Conversation history (unused for round-robin).

        Returns:
            The next agent in the rotation.
        """
        idx = round_num % len(self.sub_agents)
        return self.sub_agents[idx]

    def _select_speaker_llm(
        self, _round_num: int, messages: list[dict[str, str]],
    ) -> BaseAgent:
        """Select the next speaker using an LLM call.

        Args:
            _round_num: Current round number (unused).
            messages: Conversation history.

        Returns:
            The agent selected by the LLM.
        """
        agent_list = "\n".join(
            f'  - "{a.name}": {a.description or "No description"}'
            for a in self.sub_agents
        )
        system = (
            "You are a group chat manager. Based on the conversation so far, "
            "pick which agent should speak next.\n\n"
            f"Available agents:\n{agent_list}\n\n"
            "Respond with ONLY the agent name, nothing else."
        )
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for msg in messages[-10:]:
            llm_messages.append(msg)

        response = self.service.generate_completion(
            messages=llm_messages, temperature=0.0, max_tokens=64,
        )
        chosen_name = str(response).strip().strip('"').strip("'")

        agent = self.find_sub_agent(chosen_name)
        if agent is None:
            logger.warning(
                "[%s] LLM picked unknown agent %r, falling back to round-robin.",
                self.name, chosen_name,
            )
            return self.sub_agents[0]
        return agent

    def _select_speaker(
        self, round_num: int, messages: list[dict[str, str]],
    ) -> BaseAgent:
        """Select the next speaker based on the configured strategy.

        Args:
            round_num: Current round number (0-indexed).
            messages: Conversation history.

        Returns:
            The agent that should speak next.
        """
        if callable(self.speaker_selection) and not isinstance(
            self.speaker_selection, str,
        ):
            return self.speaker_selection(messages, self.sub_agents)
        if self.speaker_selection == "llm":
            return self._select_speaker_llm(round_num, messages)
        return self._select_speaker_round_robin(round_num, messages)

    def _should_terminate(
        self, messages: list[dict[str, str]], last_content: str,
    ) -> bool:
        """Check if the group chat should terminate.

        Args:
            messages: Full conversation history.
            last_content: The last agent's response text.

        Returns:
            True if the chat should end.
        """
        if self.termination_keyword and self.termination_keyword in last_content:
            return True
        if self.termination_condition and self.termination_condition(messages):
            return True
        return False

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Group chat loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each speaking agent.
        """
        if not self.sub_agents:
            return

        messages: list[dict[str, str]] = [
            {"role": "user", "content": ctx.user_message},
        ]
        last_content = ""

        for round_num in range(self.max_rounds):
            speaker = self._select_speaker(round_num, messages)

            logger.info(
                "[%s] Round %d/%d — speaker: %s",
                self.name, round_num + 1, self.max_rounds, speaker.name,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_num + 1}/{self.max_rounds}: {speaker.name} speaking",
                data={
                    "round": round_num + 1,
                    "speaker": speaker.name,
                },
            )

            # Build conversation context for the speaker.
            context_msgs = messages
            if self.max_context_messages > 0:
                context_msgs = messages[-self.max_context_messages:]
            transcript = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in context_msgs
            )
            speaker_input = (
                f"Group chat conversation:\n{transcript}\n\n"
                f"You are {speaker.name}. Respond to the conversation."
            )

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=speaker_input,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            agent_output = ""
            for event in speaker._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    agent_output = event.content
                yield event

            messages.append({
                "role": "assistant",
                "content": f"[{speaker.name}]: {_truncate(agent_output)}",
            })
            last_content = agent_output

            if self._should_terminate(messages, last_content):
                logger.info(
                    "[%s] Termination condition met at round %d.",
                    self.name, round_num + 1,
                )
                break

        if self.result_key:
            ctx.session.state_set(self.result_key, messages)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Group chat loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each speaking agent.
        """
        if not self.sub_agents:
            return

        messages: list[dict[str, str]] = [
            {"role": "user", "content": ctx.user_message},
        ]
        last_content = ""

        for round_num in range(self.max_rounds):
            speaker = self._select_speaker(round_num, messages)

            logger.info(
                "[%s] Async round %d/%d — speaker: %s",
                self.name, round_num + 1, self.max_rounds, speaker.name,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_num + 1}/{self.max_rounds}: {speaker.name} speaking",
                data={
                    "round": round_num + 1,
                    "speaker": speaker.name,
                },
            )

            context_msgs = messages
            if self.max_context_messages > 0:
                context_msgs = messages[-self.max_context_messages:]
            transcript = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in context_msgs
            )
            speaker_input = (
                f"Group chat conversation:\n{transcript}\n\n"
                f"You are {speaker.name}. Respond to the conversation."
            )

            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=speaker_input,
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )

            agent_output = ""
            async for event in speaker._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    agent_output = event.content
                yield event

            messages.append({
                "role": "assistant",
                "content": f"[{speaker.name}]: {_truncate(agent_output)}",
            })
            last_content = agent_output

            if self._should_terminate(messages, last_content):
                logger.info(
                    "[%s] Termination condition met at round %d.",
                    self.name, round_num + 1,
                )
                break

        if self.result_key:
            ctx.session.state_set(self.result_key, messages)


# ── HierarchicalAgent ─────────────────────────────────────────────────────────


class HierarchicalAgent(BaseAgent):
    """Multi-level hierarchical orchestration with LLM-powered manager.

    Unlike ``SupervisorAgent`` (which manages a flat pool of workers),
    ``HierarchicalAgent`` creates a tree-shaped command structure where
    each ``sub_agent`` may itself be an orchestration agent with its own
    sub-agents.  The top-level manager sees the full org-chart and
    delegates to department heads, who run their internal pipelines
    autonomously before returning results.

    Hierarchy example::

        HierarchicalAgent("cto")
        ├── SequentialAgent("backend_team")
        │   ├── LlmAgent("architect")
        │   └── LlmAgent("developer")
        ├── SupervisorAgent("qa_team")
        │   ├── LlmAgent("tester")
        │   └── LlmAgent("security_reviewer")
        └── LlmAgent("devops")

    The manager can delegate multiple times, re-delegate if unsatisfied,
    and synthesise a final answer from the collected department outputs.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Department-head agents (may have their own sub-agents).
        model: LLM model for manager decisions.
        provider: LLM provider for manager decisions.
        api_key: API key override.
        manager_instruction: Extra instructions for the manager LLM.
        temperature: LLM temperature for manager decisions.
        max_iterations: Maximum delegation rounds.
        synthesis_prompt: Optional custom prompt used when the manager
            synthesises the final answer from all department outputs.
            Defaults to a built-in prompt asking for a cohesive summary.
        result_key: Optional key to store the final synthesis in
            ``session.state[result_key]``.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> from nono.agent import Agent, SequentialAgent, HierarchicalAgent
        >>> backend = SequentialAgent(
        ...     name="backend_team",
        ...     sub_agents=[Agent(name="architect", ...), Agent(name="dev", ...)],
        ... )
        >>> qa = Agent(name="qa", instruction="Review code for bugs.", ...)
        >>> cto = HierarchicalAgent(
        ...     name="cto",
        ...     provider="google",
        ...     sub_agents=[backend, qa],
        ...     max_iterations=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        manager_instruction: str = "",
        temperature: float = 0.0,
        max_iterations: int = 3,
        synthesis_prompt: str = "",
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents,
        )
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.manager_instruction = manager_instruction
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.synthesis_prompt = synthesis_prompt
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    @staticmethod
    def _describe_org_chart(agents: list[BaseAgent], indent: int = 0) -> str:
        """Build a text representation of the agent hierarchy.

        Args:
            agents: List of agents at the current level.
            indent: Indentation depth for nested display.

        Returns:
            Multi-line string showing the org-chart.
        """
        lines: list[str] = []
        prefix = "  " * indent

        for agent in agents:
            desc = agent.description or "No description"
            kind = type(agent).__name__
            lines.append(f"{prefix}- \"{agent.name}\" ({kind}): {desc}")

            if agent.sub_agents:
                lines.append(
                    HierarchicalAgent._describe_org_chart(
                        agent.sub_agents, indent + 1,
                    ),
                )
        return "\n".join(lines)

    def _build_manager_messages(
        self,
        user_message: str,
        history: list[dict[str, str]],
        collected_outputs: dict[str, str],
    ) -> list[dict[str, str]]:
        """Build messages for the manager's delegation/evaluation call.

        Args:
            user_message: The original user request.
            history: Previous delegation attempts and results.
            collected_outputs: Map of department-name to output collected
                so far across all rounds.

        Returns:
            List of message dicts for the LLM.
        """
        org_chart = self._describe_org_chart(self.sub_agents)
        collected_section = ""

        if collected_outputs:
            parts = [
                f"  - \"{name}\": {out[:200]}"
                for name, out in collected_outputs.items()
            ]
            collected_section = (
                "\nOutputs collected so far:\n" + "\n".join(parts) + "\n"
            )

        system = (
            "You are a hierarchical manager orchestrating a tree of "
            "departments and specialist agents.\n\n"
            f"Organisation chart:\n{org_chart}\n"
            f"{collected_section}\n"
            "Your job:\n"
            "1. Delegate to the best department/agent for (part of) the "
            "task.\n"
            "2. After seeing a department's output, decide: delegate to "
            "another department, or SYNTHESISE a final answer.\n\n"
            "Respond with JSON (no markdown fences):\n"
            "{\n"
            '  "action": "delegate|synthesise",\n'
            '  "agent": "department_name",       // required for delegate\n'
            '  "message": "task for department",  // optional: refine task\n'
            '  "reason": "why this decision"      // brief explanation\n'
            "}\n\n"
            "Rules:\n"
            "- You may delegate to multiple departments across rounds.\n"
            "- 'agent' must be a name from the top-level org chart.\n"
            "- Choose 'synthesise' only when you have enough information "
            "to produce a final answer.\n"
        )

        if self.manager_instruction:
            system += f"\n{self.manager_instruction}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {user_message}"},
        ]
        messages.extend(history)
        return messages

    def _parse_manager_response(self, response: str) -> dict[str, str]:
        """Parse the manager's JSON response.

        Args:
            response: Raw LLM response text.

        Returns:
            Dict with keys: action, agent, message, reason.
        """
        text = response.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[%s] Failed to parse manager response: %s",
                self.name, text,
            )
            return {
                "action": "synthesise", "agent": "",
                "message": "", "reason": "parse failure",
            }

        return {
            "action": data.get("action", "synthesise"),
            "agent": data.get("agent", ""),
            "message": data.get("message", ""),
            "reason": data.get("reason", ""),
        }

    def _build_synthesis_messages(
        self, user_message: str, collected_outputs: dict[str, str],
    ) -> list[dict[str, str]]:
        """Build messages for the final synthesis call.

        Args:
            user_message: The original user request.
            collected_outputs: All department outputs to synthesise.

        Returns:
            List of message dicts for the LLM.
        """
        parts = [
            f"## {name}\n{_truncate(output)}" for name, output in collected_outputs.items()
        ]
        departments_text = "\n\n".join(parts)

        prompt = self.synthesis_prompt or (
            "Synthesise the department outputs below into a single, "
            "cohesive final answer for the original task. "
            "Be concise and well-structured."
        )

        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Original task: {user_message}\n\n"
                    f"Department outputs:\n{departments_text}"
                ),
            },
        ]

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Hierarchical delegation loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from delegated departments and manager decisions.
        """
        if not self.sub_agents:
            yield Event(
                EventType.ERROR, self.name, "No departments configured.",
            )
            return

        history: list[dict[str, str]] = []
        collected_outputs: dict[str, str] = {}

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Manager round %d/%d",
                self.name, iteration, self.max_iterations,
            )

            messages = self._build_manager_messages(
                ctx.user_message, history, collected_outputs,
            )
            response = self.service.generate_completion(
                messages=messages, temperature=self.temperature,
                max_tokens=256,
            )
            decision = self._parse_manager_response(str(response))

            if decision["action"] == "synthesise":
                logger.info(
                    "[%s] Manager chose to synthesise at round %d.",
                    self.name, iteration,
                )
                break

            department = self.find_sub_agent(decision["agent"])

            if department is None:
                logger.warning(
                    "[%s] Unknown department %r, using first.",
                    self.name, decision["agent"],
                )
                department = self.sub_agents[0]

            task_message = decision.get("message") or ctx.user_message

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Delegating to {department.name}: "
                f"{decision.get('reason', '')}",
                data={
                    "department": department.name,
                    "round": iteration,
                    "reason": decision.get("reason", ""),
                },
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=task_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            dept_output = ""

            for event in department._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    dept_output = event.content
                yield event

            collected_outputs[department.name] = _truncate(dept_output)

            history.append({
                "role": "assistant",
                "content": json.dumps(decision),
            })
            history.append({
                "role": "user",
                "content": (
                    f"Department '{department.name}' responded:\n"
                    f"{_truncate(dept_output)}\n\nDelegate to another department "
                    f"or synthesise?"
                ),
            })

        # Synthesise final answer from collected outputs.
        if collected_outputs:
            synth_messages = self._build_synthesis_messages(
                ctx.user_message, collected_outputs,
            )
            synthesis = str(
                self.service.generate_completion(
                    messages=synth_messages, temperature=self.temperature,
                ),
            )

            yield Event(EventType.AGENT_MESSAGE, self.name, synthesis)

            if self.result_key:
                ctx.session.state_set(self.result_key, synthesis)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Hierarchical delegation loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from delegated departments and manager decisions.
        """
        if not self.sub_agents:
            yield Event(
                EventType.ERROR, self.name, "No departments configured.",
            )
            return

        history: list[dict[str, str]] = []
        collected_outputs: dict[str, str] = {}

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "[%s] Async manager round %d/%d",
                self.name, iteration, self.max_iterations,
            )

            messages = self._build_manager_messages(
                ctx.user_message, history, collected_outputs,
            )
            response = await asyncio.to_thread(
                self.service.generate_completion,
                messages=messages, temperature=self.temperature,
                max_tokens=256,
            )
            decision = self._parse_manager_response(str(response))

            if decision["action"] == "synthesise":
                logger.info(
                    "[%s] Manager chose to synthesise at round %d.",
                    self.name, iteration,
                )
                break

            department = self.find_sub_agent(decision["agent"])

            if department is None:
                logger.warning(
                    "[%s] Unknown department %r, using first.",
                    self.name, decision["agent"],
                )
                department = self.sub_agents[0]

            task_message = decision.get("message") or ctx.user_message

            yield Event(
                EventType.AGENT_TRANSFER, self.name,
                f"Delegating to {department.name}: "
                f"{decision.get('reason', '')}",
                data={
                    "department": department.name,
                    "round": iteration,
                    "reason": decision.get("reason", ""),
                },
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=task_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            dept_output = ""

            async for event in department._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    dept_output = event.content
                yield event

            collected_outputs[department.name] = _truncate(dept_output)

            history.append({
                "role": "assistant",
                "content": json.dumps(decision),
            })
            history.append({
                "role": "user",
                "content": (
                    f"Department '{department.name}' responded:\n"
                    f"{_truncate(dept_output)}\n\nDelegate to another department "
                    f"or synthesise?"
                ),
            })

        # Synthesise final answer from collected outputs.
        if collected_outputs:
            synth_messages = self._build_synthesis_messages(
                ctx.user_message, collected_outputs,
            )
            synthesis = str(
                await asyncio.to_thread(
                    self.service.generate_completion,
                    messages=synth_messages,
                    temperature=self.temperature,
                ),
            )

            yield Event(EventType.AGENT_MESSAGE, self.name, synthesis)

            if self.result_key:
                ctx.session.state_set(self.result_key, synthesis)


# ── GuardrailAgent ────────────────────────────────────────────────────────────


class GuardrailAgent(BaseAgent):
    """Wrap an agent with pre- and post-validation guardrails.

    The execution flow is:

    1. **Pre-validator** (optional): checks/transforms the input.
       If it yields an ``ERROR`` event, the main agent is skipped.
    2. **Main agent**: processes the (possibly transformed) message.
    3. **Post-validator** (optional): checks the output.
       If validation fails, the main agent retries (up to ``max_retries``).

    This is the canonical pattern for safety layers — toxicity filters,
    PII detection, schema validation, etc.

    Args:
        name: Agent name.
        description: Human-readable description.
        main_agent: The agent whose input/output is guarded.
        pre_validator: Optional agent that checks the input.  If it
            yields an ``ERROR`` event, execution stops.  Otherwise its
            last ``AGENT_MESSAGE`` replaces the user message (input
            transformation).
        post_validator: Optional agent that checks the output.  Its
            last ``AGENT_MESSAGE`` is inspected for ``rejection_keyword``.
        rejection_keyword: Substring in the post-validator response
            that signals a failed validation (case-insensitive).
        max_retries: How many times to retry the main agent when
            post-validation fails.
        result_key: Store the final validated output in ``session.state``.

    Example:
        >>> guardrail = GuardrailAgent(
        ...     name="safe_writer",
        ...     main_agent=writer,
        ...     pre_validator=pii_checker,
        ...     post_validator=toxicity_filter,
        ...     rejection_keyword="REJECTED",
        ...     max_retries=2,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        main_agent: BaseAgent,
        pre_validator: BaseAgent | None = None,
        post_validator: BaseAgent | None = None,
        rejection_keyword: str = "REJECTED",
        max_retries: int = 2,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        sub = [a for a in [pre_validator, main_agent, post_validator] if a]
        super().__init__(
            name=name, description=description, sub_agents=sub, **kwargs,
        )
        self.main_agent = main_agent
        self.pre_validator = pre_validator
        self.post_validator = post_validator
        self.rejection_keyword = rejection_keyword
        self.max_retries = max_retries
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Guardrail loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from validators and the main agent.
        """
        message = ctx.user_message

        # Pre-validation
        if self.pre_validator:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            pre_output = ""
            for event in self.pre_validator._run_impl_traced(sub_ctx):
                if event.event_type == EventType.ERROR:
                    yield event
                    return
                if event.event_type == EventType.AGENT_MESSAGE:
                    pre_output = event.content
                yield event
            if pre_output:
                message = pre_output

        # Main + post-validation loop
        for attempt in range(1, self.max_retries + 2):
            logger.info(
                "[%s] Main agent attempt %d/%d",
                self.name, attempt, self.max_retries + 1,
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            main_output = ""
            for event in self.main_agent._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    main_output = event.content
                yield event

            if not self.post_validator:
                if self.result_key:
                    ctx.session.state_set(self.result_key, main_output)
                return

            # Post-validation
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=main_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            post_output = ""
            for event in self.post_validator._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    post_output = event.content
                yield event

            if self.rejection_keyword.lower() not in post_output.lower():
                logger.info(
                    "[%s] Post-validation passed at attempt %d.",
                    self.name, attempt,
                )
                if self.result_key:
                    ctx.session.state_set(self.result_key, main_output)
                return

            logger.warning(
                "[%s] Post-validation failed at attempt %d.",
                self.name, attempt,
            )

        yield Event(
            EventType.ERROR, self.name,
            f"Post-validation failed after {self.max_retries + 1} attempts.",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Guardrail loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from validators and the main agent.
        """
        message = ctx.user_message

        # Pre-validation
        if self.pre_validator:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            pre_output = ""
            async for event in self.pre_validator._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.ERROR:
                    yield event
                    return
                if event.event_type == EventType.AGENT_MESSAGE:
                    pre_output = event.content
                yield event
            if pre_output:
                message = pre_output

        # Main + post-validation loop
        for attempt in range(1, self.max_retries + 2):
            logger.info(
                "[%s] Async main agent attempt %d/%d",
                self.name, attempt, self.max_retries + 1,
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            main_output = ""
            async for event in self.main_agent._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    main_output = event.content
                yield event

            if not self.post_validator:
                if self.result_key:
                    ctx.session.state_set(self.result_key, main_output)
                return

            # Post-validation
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=main_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            post_output = ""
            async for event in self.post_validator._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    post_output = event.content
                yield event

            if self.rejection_keyword.lower() not in post_output.lower():
                logger.info(
                    "[%s] Post-validation passed at attempt %d.",
                    self.name, attempt,
                )
                if self.result_key:
                    ctx.session.state_set(self.result_key, main_output)
                return

            logger.warning(
                "[%s] Post-validation failed at attempt %d.",
                self.name, attempt,
            )

        yield Event(
            EventType.ERROR, self.name,
            f"Post-validation failed after {self.max_retries + 1} attempts.",
        )


# ── BestOfNAgent ──────────────────────────────────────────────────────────────


class BestOfNAgent(BaseAgent):
    """Run the same agent N times and pick the best via a scoring function.

    Unlike ``VotingAgent`` (N different agents, majority vote), this agent
    executes a **single** agent ``n`` times — exploiting sampling variance —
    and uses a custom ``score_fn`` to pick the best response.

    This is the canonical *best-of-N sampling* technique used for creative
    generation, code generation, and any task where trying multiple times
    yields different quality.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The agent to execute N times.
        n: Number of executions.  Keep ``n`` small (e.g. ≤10) — all
            candidate outputs are held in memory until scoring completes.
        score_fn: Callable ``(response: str) -> float`` that returns a
            numeric score.  Higher is better.  Default: length of response.
        max_workers: Maximum concurrent threads (sync).
        result_key: Store scored results in ``session.state``.

    Example:
        >>> best = BestOfNAgent(
        ...     name="best_copy",
        ...     agent=copywriter,
        ...     n=5,
        ...     score_fn=lambda r: len(r),  # prefer longer copy
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent,
        n: int = 3,
        score_fn: Callable[[str], float] | None = None,
        max_workers: int | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[agent], **kwargs,
        )
        self.agent = agent
        self.n = n
        self.score_fn = score_fn or (lambda r: float(len(r)))
        self.max_workers = max_workers
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run agent N times (parallel) and pick the best (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from all runs, then the best response.
        """
        workers = min(self.max_workers or self.n, _MAX_DEFAULT_PARALLEL_WORKERS)

        def _run_once(idx: int) -> tuple[int, list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in self.agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return idx, events, output

        candidates: list[tuple[int, str, float]] = []
        all_events: list[list[Event]] = [[] for _ in range(self.n)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_once, i): i for i in range(self.n)
            }
            for future in as_completed(futures):
                try:
                    idx, events, output = future.result(timeout=_FUTURE_TIMEOUT)
                    all_events[idx] = events
                    score = self.score_fn(output)
                    candidates.append((idx, output, score))
                except Exception as exc:
                    logger.error(
                        "[%s] Run %d failed: %s", self.name,
                        futures[future], exc,
                    )

        if not candidates:
            yield Event(
                EventType.ERROR, self.name, "All N runs failed.",
            )
            return

        candidates.sort(key=lambda c: c[2], reverse=True)
        best_idx, best_output, best_score = candidates[0]

        # Yield events from the winning run
        for event in all_events[best_idx]:
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_index": best_idx,
                "best_score": best_score,
                "all_scores": [
                    {"index": c[0], "score": c[2]} for c in candidates
                ],
            })

        logger.info(
            "[%s] Best-of-%d: run %d (score=%.2f)",
            self.name, self.n, best_idx, best_score,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run agent N times (concurrent) and pick the best (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the best run.
        """

        async def _run_once(idx: int) -> tuple[int, list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in self.agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return idx, events, output

        tasks = [_run_once(i) for i in range(self.n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[tuple[int, str, float]] = []
        all_events: list[list[Event]] = [[] for _ in range(self.n)]

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error("[%s] Async run %d failed: %s", self.name, i, result)
            else:
                idx, events, output = result
                all_events[idx] = events
                score = self.score_fn(output)
                candidates.append((idx, output, score))

        if not candidates:
            yield Event(
                EventType.ERROR, self.name, "All N runs failed.",
            )
            return

        candidates.sort(key=lambda c: c[2], reverse=True)
        best_idx, best_output, best_score = candidates[0]

        for event in all_events[best_idx]:
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_index": best_idx,
                "best_score": best_score,
                "all_scores": [
                    {"index": c[0], "score": c[2]} for c in candidates
                ],
            })

        logger.info(
            "[%s] Async best-of-%d: run %d (score=%.2f)",
            self.name, self.n, best_idx, best_score,
        )


# ── BatchAgent ────────────────────────────────────────────────────────────────


class BatchAgent(BaseAgent):
    """Process a list of items through the same agent with concurrency control.

    Unlike ``MapReduceAgent`` (fan-out to different agents + reduce),
    ``BatchAgent`` sends a **list of data items** through a single agent
    pipeline with configurable concurrency.  There is no reduce step —
    results are collected as ``{index: response}``.

    This is the canonical pattern for bulk processing: classify 100
    documents, translate 50 paragraphs, analyse a dataset row by row.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The agent that processes each item.
        items_key: Key in ``session.state`` containing the list of items
            to process.  Each item is converted to ``str`` and sent
            as the user message to the agent.
        items: Optional static list of items (alternative to ``items_key``).
            If both are provided, ``items_key`` takes precedence at runtime.
        max_workers: Maximum concurrent threads.
        result_key: Key to store results as ``{index: response}``.
        template: Optional template string with ``{item}`` placeholder.
            If provided, each item is formatted into the template before
            being sent to the agent.

    Example:
        >>> batch = BatchAgent(
        ...     name="classifier",
        ...     agent=classifier_agent,
        ...     items=["article1", "article2", "article3"],
        ...     max_workers=5,
        ...     result_key="classifications",
        ...     template="Classify this article: {item}",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent,
        items: list[Any] | None = None,
        items_key: str | None = None,
        max_workers: int = 4,
        result_key: str = "batch_results",
        template: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[agent], **kwargs,
        )
        self.agent = agent
        self.items = items
        self.items_key = items_key
        self.max_workers = max_workers
        self.result_key = result_key
        self.template = template

    def _resolve_items(self, ctx: InvocationContext) -> list[Any]:
        """Get the item list from state or static config.

        Args:
            ctx: The invocation context.

        Returns:
            List of items to process.
        """
        if self.items_key and self.items_key in ctx.session.state:
            return list(ctx.session.state[self.items_key])
        return list(self.items or [])

    def _format_message(self, item: Any) -> str:
        """Format an item into the message to send to the agent.

        Args:
            item: The data item.

        Returns:
            Formatted message string.
        """
        text = str(item)
        if self.template:
            return self.template.replace("{item}", text)
        return text

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Batch processing loop (sync, parallel).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each item processing.
        """
        items = self._resolve_items(ctx)

        if not items:
            yield Event(EventType.ERROR, self.name, "No items to process.")
            return

        logger.info(
            "[%s] Processing %d items (max_workers=%d)",
            self.name, len(items), self.max_workers,
        )

        def _process(idx: int, item: Any) -> tuple[int, list[Event], str]:
            msg = self._format_message(item)
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=msg,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in self.agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return idx, events, output

        results: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS)) as executor:
            futures = {
                executor.submit(_process, i, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, events, output = future.result(timeout=_FUTURE_TIMEOUT)
                    results[idx] = output
                    for event in events:
                        yield event
                except Exception as exc:
                    logger.error(
                        "[%s] Item %d failed: %s", self.name, idx, exc,
                    )
                    results[idx] = f"ERROR: {exc}"
                    yield Event(
                        EventType.ERROR, self.name,
                        f"Item {idx} failed: {exc}",
                    )

        ctx.session.state_set(self.result_key, results)

        logger.info(
            "[%s] Batch complete: %d/%d succeeded",
            self.name,
            sum(1 for v in results.values() if not v.startswith("ERROR:")),
            len(items),
        )

        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"Processed {len(items)} items.",
            data={"total": len(items), "results": results},
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Batch processing loop (async, concurrent).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each item processing.
        """
        items = self._resolve_items(ctx)

        if not items:
            yield Event(EventType.ERROR, self.name, "No items to process.")
            return

        logger.info(
            "[%s] Async processing %d items (max_workers=%d)",
            self.name, len(items), self.max_workers,
        )

        semaphore = asyncio.Semaphore(self.max_workers)

        async def _process(idx: int, item: Any) -> tuple[int, list[Event], str]:
            async with semaphore:
                msg = self._format_message(item)
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=msg,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                events: list[Event] = []
                output = ""
                async for event in self.agent._run_async_impl_traced(sub_ctx):
                    events.append(event)
                    if event.event_type == EventType.AGENT_MESSAGE:
                        output = event.content
                return idx, events, output

        tasks = [_process(i, item) for i, item in enumerate(items)]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: dict[int, str] = {}

        for i, result in enumerate(task_results):
            if isinstance(result, BaseException):
                logger.error("[%s] Async item %d failed: %s", self.name, i, result)
                results[i] = f"ERROR: {result}"
                yield Event(
                    EventType.ERROR, self.name, f"Item {i} failed: {result}",
                )
            else:
                idx, events, output = result
                results[idx] = output
                for event in events:
                    yield event

        ctx.session.state_set(self.result_key, results)

        logger.info(
            "[%s] Async batch complete: %d/%d succeeded",
            self.name,
            sum(1 for v in results.values() if not v.startswith("ERROR:")),
            len(items),
        )

        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"Processed {len(items)} items.",
            data={"total": len(items), "results": results},
        )


# ── CascadeAgent ──────────────────────────────────────────────────────────────


class CascadeAgent(BaseAgent):
    """Progressive cascade with confidence gates between stages.

    Unlike ``EscalationAgent`` (keyword-based failure detection),
    ``CascadeAgent`` uses a **scoring function** to decide whether the
    current stage's output meets a quality threshold.  If the score is
    below the threshold, execution cascades to the next stage.

    Typical use: cheap flash model first → if confidence < 0.8 → larger
    model → if still low → specialist model.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ordered stages (cheapest/fastest first).
        score_fn: Callable ``(response: str) -> float`` that returns a
            confidence/quality score in ``[0, 1]``.
        threshold: Minimum score to accept a response.  Stages with
            score ``>= threshold`` stop the cascade.
        result_key: Store cascade details in ``session.state``.

    Example:
        >>> cascade = CascadeAgent(
        ...     name="smart_cascade",
        ...     sub_agents=[flash, pro, expert],
        ...     score_fn=lambda r: 1.0 if "CONFIDENT" in r else 0.3,
        ...     threshold=0.8,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        score_fn: Callable[[str], float],
        threshold: float = 0.8,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents, **kwargs,
        )
        self.score_fn = score_fn
        self.threshold = threshold
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Cascade through stages until confidence met (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the stage that meets the threshold.
        """
        if not self.sub_agents:
            return

        last_events: list[Event] = []
        last_output = ""
        last_score = 0.0

        for i, agent in enumerate(self.sub_agents):
            logger.info(
                "[%s] Cascade stage %d/%d: %s",
                self.name, i + 1, len(self.sub_agents), agent.name,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Stage {i + 1}/{len(self.sub_agents)}: {agent.name}",
                data={"stage": i + 1, "agent": agent.name},
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            last_events = []
            last_output = ""
            for event in agent._run_impl_traced(sub_ctx):
                last_events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_output = event.content

            last_score = self.score_fn(last_output)
            logger.info(
                "[%s] Stage %s scored %.2f (threshold=%.2f)",
                self.name, agent.name, last_score, self.threshold,
            )

            if last_score >= self.threshold:
                yield from last_events

                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "stage": i + 1,
                        "agent": agent.name,
                        "score": last_score,
                    })
                return

        # No stage met the threshold — yield last attempt
        yield from last_events

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "stage": len(self.sub_agents),
                "agent": self.sub_agents[-1].name,
                "score": last_score,
                "met_threshold": False,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Cascade through stages until confidence met (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the stage that meets the threshold.
        """
        if not self.sub_agents:
            return

        last_events: list[Event] = []
        last_output = ""
        last_score = 0.0

        for i, agent in enumerate(self.sub_agents):
            logger.info(
                "[%s] Async cascade stage %d/%d: %s",
                self.name, i + 1, len(self.sub_agents), agent.name,
            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Stage {i + 1}/{len(self.sub_agents)}: {agent.name}",
                data={"stage": i + 1, "agent": agent.name},
            )

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )

            last_events = []
            last_output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                last_events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_output = event.content

            last_score = self.score_fn(last_output)
            logger.info(
                "[%s] Async stage %s scored %.2f (threshold=%.2f)",
                self.name, agent.name, last_score, self.threshold,
            )

            if last_score >= self.threshold:
                for event in last_events:
                    yield event

                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "stage": i + 1,
                        "agent": agent.name,
                        "score": last_score,
                    })
                return

        # No stage met the threshold — yield last attempt
        for event in last_events:
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "stage": len(self.sub_agents),
                "agent": self.sub_agents[-1].name,
                "score": last_score,
                "met_threshold": False,
            })


# ── TreeOfThoughtsAgent ──────────────────────────────────────────────────────


class TreeOfThoughtsAgent(BaseAgent):
    """Tree-of-Thoughts reasoning: generate, evaluate, prune, and deepen.

    Explores multiple reasoning branches at each level, evaluates them with
    ``evaluate_fn``, keeps only the top-k, then expands those to the next
    depth level.  Stops early when a branch meets ``threshold``.

    This is the agent-orchestration equivalent of the Tree of Thoughts
    algorithm (Yao et al., 2023).  No other framework implements this
    natively.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Agent that generates each thought branch — it receives
            a prompt that includes the path so far.
        evaluate_fn: ``(response: str) -> float`` scoring function.
            Higher is better.
        n_branches: Branches to generate at each level.
        top_k: How many branches survive pruning each level.
        max_depth: Maximum tree depth before stopping.
        threshold: Score to accept a branch early.
        max_workers: Maximum concurrent threads.
        result_key: Store the full tree + best path in ``session.state``.

    Example:
        >>> tot = TreeOfThoughtsAgent(
        ...     name="reasoner",
        ...     agent=thinker,
        ...     evaluate_fn=lambda r: 1.0 if "correct" in r else 0.3,
        ...     n_branches=3,
        ...     top_k=2,
        ...     max_depth=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent,
        evaluate_fn: Callable[[str], float],
        n_branches: int = 3,
        top_k: int = 2,
        max_depth: int = 3,
        threshold: float = 0.9,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[agent], **kwargs,
        )
        self.agent = agent
        self.evaluate_fn = evaluate_fn
        self.n_branches = n_branches
        self.top_k = top_k
        self.max_depth = max_depth
        self.threshold = threshold
        self.max_workers = max_workers
        self.result_key = result_key

    def _run_branch(
        self, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        """Run the agent once with the given prompt (sync).

        Args:
            ctx: The invocation context (for session/tracing).
            prompt: The prompt including path-so-far.

        Returns:
            Tuple of (events, output text).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in self.agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    async def _run_branch_async(
        self, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        """Run the agent once with the given prompt (async).

        Args:
            ctx: The invocation context.
            prompt: The prompt including path-so-far.

        Returns:
            Tuple of (events, output text).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in self.agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Explore the thought tree using breadth-first expansion (sync).

        Args:
            ctx: The invocation context.

        Yields:
            STATE_UPDATE per depth level, AGENT_MESSAGE with best result.
        """
        # Each frontier item: (path_so_far, last_output)
        frontier: list[tuple[str, str]] = [
            (ctx.user_message, ctx.user_message),
        ]
        best_output = ""
        best_score = 0.0
        best_path: list[str] = []
        final_depth = 0

        for depth in range(1, self.max_depth + 1):
            final_depth = depth
            candidates: list[tuple[str, str, float, list[Event]]] = []

            def _expand(prefix: str) -> tuple[list[Event], str, float, str]:
                prompt = (
                    f"Continue this reasoning (depth {depth}):\n\n"
                    f"{prefix}\n\n"
                    f"Provide step {depth} of your analysis."
                )
                events, output = self._run_branch(ctx, prompt)
                score = self.evaluate_fn(output)
                return events, output, score, prefix

            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS),
            ) as executor:
                futures = []
                for prefix, _ in frontier:
                    for _ in range(self.n_branches):
                        futures.append(executor.submit(_expand, prefix))
                for future in as_completed(futures):
                    try:
                        events, output, score, prefix = future.result(timeout=_FUTURE_TIMEOUT)
                        path = f"{_truncate(prefix)}\n---\n{_truncate(output)}"
                        candidates.append((path, output, score, events))
                    except Exception as exc:
                        logger.error(
                            "[%s] Branch failed at depth %d: %s",
                            self.name, depth, exc,
                        )

            if not candidates:
                break

            candidates.sort(key=lambda c: c[2], reverse=True)

            # Check threshold
            if candidates[0][2] >= self.threshold:
                best_output = candidates[0][1]
                best_score = candidates[0][2]
                best_path = candidates[0][0].split("\n---\n")
                for event in candidates[0][3]:
                    yield event

                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Depth {depth}: threshold met (score={best_score:.2f})",
                    data={
                        "depth": depth,
                        "score": best_score,
                        "met_threshold": True,
                    },
                )

                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "depth": depth,
                        "score": best_score,
                        "met_threshold": True,
                        "path": [_truncate(p) for p in best_path],
                    })
                return

            # Update best
            if candidates[0][2] > best_score:
                best_output = candidates[0][1]
                best_score = candidates[0][2]
                best_path = candidates[0][0].split("\n---\n")

            # Prune to top-k
            frontier = [
                (c[0], c[1]) for c in candidates[: self.top_k]
            ]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Depth {depth}: {len(candidates)} branches → top {self.top_k} "
                f"(best={best_score:.2f})",
                data={
                    "depth": depth,
                    "candidates": len(candidates),
                    "top_k": self.top_k,
                    "best_score": best_score,
                },
            )

        # Yield best result found
        yield Event(EventType.AGENT_MESSAGE, self.name, best_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "depth": final_depth,
                "score": best_score,
                "met_threshold": False,
                "path": [_truncate(p) for p in best_path],
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Explore the thought tree using breadth-first expansion (async).

        Args:
            ctx: The invocation context.

        Yields:
            STATE_UPDATE per depth level, AGENT_MESSAGE with best result.
        """
        frontier: list[tuple[str, str]] = [
            (ctx.user_message, ctx.user_message),
        ]
        best_output = ""
        best_score = 0.0
        best_path: list[str] = []
        final_depth = 0

        for depth in range(1, self.max_depth + 1):
            final_depth = depth
            candidates: list[tuple[str, str, float, list[Event]]] = []

            async def _expand(
                prefix: str, _depth: int = depth,
            ) -> tuple[list[Event], str, float, str]:
                prompt = (
                    f"Continue this reasoning (depth {_depth}):\n\n"
                    f"{prefix}\n\n"
                    f"Provide step {_depth} of your analysis."
                )
                events, output = await self._run_branch_async(ctx, prompt)
                score = self.evaluate_fn(output)
                return events, output, score, prefix

            tasks = [
                _expand(prefix)
                for prefix, _ in frontier
                for _ in range(self.n_branches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(
                        "[%s] Async branch failed at depth %d: %s",
                        self.name, depth, result,
                    )
                    continue
                events, output, score, prefix = result
                path = f"{_truncate(prefix)}\n---\n{_truncate(output)}"
                candidates.append((path, output, score, events))

            if not candidates:
                break

            candidates.sort(key=lambda c: c[2], reverse=True)

            if candidates[0][2] >= self.threshold:
                best_output = candidates[0][1]
                best_score = candidates[0][2]
                best_path = candidates[0][0].split("\n---\n")
                for event in candidates[0][3]:
                    yield event

                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Depth {depth}: threshold met (score={best_score:.2f})",
                    data={
                        "depth": depth,
                        "score": best_score,
                        "met_threshold": True,
                    },
                )

                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "depth": depth,
                        "score": best_score,
                        "met_threshold": True,
                        "path": [_truncate(p) for p in best_path],
                    })
                return

            if candidates[0][2] > best_score:
                best_output = candidates[0][1]
                best_score = candidates[0][2]
                best_path = candidates[0][0].split("\n---\n")

            frontier = [
                (c[0], c[1]) for c in candidates[: self.top_k]
            ]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Depth {depth}: {len(candidates)} branches → top {self.top_k} "
                f"(best={best_score:.2f})",
                data={
                    "depth": depth,
                    "candidates": len(candidates),
                    "top_k": self.top_k,
                    "best_score": best_score,
                },
            )

        yield Event(EventType.AGENT_MESSAGE, self.name, best_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "depth": final_depth,
                "score": best_score,
                "met_threshold": False,
                "path": [_truncate(p) for p in best_path],
            })


# ── PlannerAgent ──────────────────────────────────────────────────────────────


class PlannerAgent(BaseAgent):
    """Plan-and-execute: LLM decomposes, then executes respecting dependencies.

    The planner uses an LLM call to decompose the task into steps, assigning
    each step to one of the available ``sub_agents``.  Steps without mutual
    dependencies run in parallel; dependent steps run sequentially.

    Inspired by CrewAI's ``planning=True`` and LangGraph's plan-and-execute
    pattern — but as a first-class composable ``BaseAgent``.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of available agents.
        model: LLM model for planning.
        provider: LLM provider.
        api_key: API key override.
        planning_instruction: Extra instruction for the planner prompt.
        max_steps: Maximum steps the planner may produce.
        synthesis_prompt: Custom prompt for the final synthesis.
        result_key: Store plan + results in ``session.state``.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> planner = PlannerAgent(
        ...     name="project_manager",
        ...     provider="google",
        ...     sub_agents=[researcher, writer, reviewer],
        ...     max_steps=5,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        planning_instruction: str = "",
        max_steps: int = 5,
        synthesis_prompt: str = "",
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.planning_instruction = planning_instruction
        self.max_steps = max_steps
        self.synthesis_prompt = synthesis_prompt
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service for planning."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _build_planning_messages(
        self, user_message: str,
    ) -> list[dict[str, str]]:
        """Build the system + user messages for the planning LLM call.

        Args:
            user_message: The user's original message.

        Returns:
            List of message dicts for the LLM.
        """
        agent_list = "\n".join(
            f'  - "{a.name}": {a.description or "No description"}'
            for a in self.sub_agents
        )
        system = (
            "You are a planning assistant. Decompose the user's task into "
            f"a maximum of {self.max_steps} concrete steps.\n\n"
            f"Available agents:\n{agent_list}\n\n"
            "Respond with a JSON array (no markdown fences). Each element:\n"
            "{\n"
            '  "step": 1,\n'
            '  "agent": "agent_name",\n'
            '  "task": "what this agent should do",\n'
            '  "depends_on": []  // list of step numbers this depends on\n'
            "}\n\n"
            "Rules:\n"
            "- step numbers start at 1.\n"
            "- agent MUST be one of the names listed above.\n"
            "- depends_on is a list of step numbers that must complete first.\n"
            "- Steps with no dependencies can run in parallel.\n"
            "- Do NOT answer the user's question — only create the plan."
        )
        if self.planning_instruction:
            system += f"\n\n{self.planning_instruction}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _parse_plan(self, response: str) -> list[dict[str, Any]]:
        """Parse the LLM's plan response into a list of steps.

        Args:
            response: Raw LLM response text.

        Returns:
            List of step dicts with keys: step, agent, task, depends_on.
        """
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            plan = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("[%s] Plan parse error: %s", self.name, text[:200])
            return []
        if not isinstance(plan, list):
            return []
        return plan[: self.max_steps]

    def _build_synthesis_messages(
        self, user_message: str, results: dict[int, str],
    ) -> list[dict[str, str]]:
        """Build messages for the synthesis LLM call.

        Args:
            user_message: Original user message.
            results: Mapping of step number → agent output.

        Returns:
            List of message dicts for the LLM.
        """
        parts = "\n\n".join(
            f"**Step {k}:**\n{v}" for k, v in sorted(results.items())
        )
        prompt = self.synthesis_prompt or (
            "Synthesise the following step results into a single cohesive "
            "response for the user's original request."
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Original request: {user_message}\n\n"
                    f"Step results:\n{parts}"
                ),
            },
        ]

    def _execute_step(
        self, ctx: InvocationContext, step: dict[str, Any],
    ) -> tuple[int, list[Event], str]:
        """Execute a single plan step (sync).

        Args:
            ctx: The invocation context.
            step: Step dict with agent, task, step number.

        Returns:
            Tuple of (step_number, events, output).
        """
        agent = self.find_sub_agent(step.get("agent", ""))
        if not agent:
            logger.warning(
                "[%s] Unknown agent %r in step %d",
                self.name, step.get("agent"), step["step"],
            )
            return step["step"], [], "(agent not found)"

        sub_ctx = InvocationContext(
            session=ctx.session,
            user_message=str(step.get("task", ctx.user_message)),
            parent_agent=self,
            trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return step["step"], events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Plan and execute the task (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from planning, execution, and synthesis.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        # Plan
        messages = self._build_planning_messages(ctx.user_message)
        response = self.service.generate_completion(
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        plan = self._parse_plan(str(response))

        if not plan:
            yield Event(
                EventType.ERROR, self.name, "Failed to generate a valid plan.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Plan: {len(plan)} steps",
            data={"plan": plan},
        )

        # Execute respecting dependencies
        step_results: dict[int, str] = {}
        remaining = list(plan)

        while remaining:
            runnable = [
                s for s in remaining
                if all(d in step_results for d in s.get("depends_on", []))
            ]
            if not runnable:
                logger.warning(
                    "[%s] Deadlock — steps with unresolved deps: %s",
                    self.name, [s["step"] for s in remaining],
                )
                yield Event(
                    EventType.ERROR, self.name,
                    f"Circular dependency detected in plan — "
                    f"stuck steps: {[s['step'] for s in remaining]}",
                )
                break

            if len(runnable) == 1:
                step_num, events, output = self._execute_step(
                    ctx, runnable[0],
                )
                step_results[step_num] = output
                for event in events:
                    yield event
                remaining.remove(runnable[0])
            else:
                with ThreadPoolExecutor(
                    max_workers=min(len(runnable), _MAX_DEFAULT_PARALLEL_WORKERS),
                ) as executor:
                    futures = {
                        executor.submit(
                            self._execute_step, ctx, s,
                        ): s
                        for s in runnable
                    }
                    for future in as_completed(futures):
                        step_num, events, output = future.result(timeout=_FUTURE_TIMEOUT)
                        step_results[step_num] = output
                        for event in events:
                            yield event
                for s in runnable:
                    remaining.remove(s)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Completed {len(step_results)}/{len(plan)} steps",
                data={"completed": len(step_results), "total": len(plan)},
            )

        # Synthesis
        synth_messages = self._build_synthesis_messages(
            ctx.user_message, step_results,
        )
        synthesis = self.service.generate_completion(
            messages=synth_messages, temperature=0.0, max_tokens=2048,
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, str(synthesis))

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "plan": plan,
                "step_results": step_results,
                "synthesis": str(synthesis),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Plan and execute the task (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from planning, execution, and synthesis.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        messages = self._build_planning_messages(ctx.user_message)
        response = await asyncio.to_thread(
            self.service.generate_completion,
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        plan = self._parse_plan(str(response))

        if not plan:
            yield Event(
                EventType.ERROR, self.name, "Failed to generate a valid plan.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Plan: {len(plan)} steps",
            data={"plan": plan},
        )

        step_results: dict[int, str] = {}
        remaining = list(plan)

        while remaining:
            runnable = [
                s for s in remaining
                if all(d in step_results for d in s.get("depends_on", []))
            ]
            if not runnable:
                logger.warning("[%s] Async deadlock", self.name)
                yield Event(
                    EventType.ERROR, self.name,
                    "Circular dependency detected in plan.",
                )
                break

            async def _run_step(
                s: dict[str, Any],
            ) -> tuple[int, list[Event], str]:
                agent = self.find_sub_agent(s.get("agent", ""))
                if not agent:
                    return s["step"], [], "(agent not found)"
                sub_ctx = InvocationContext(
                    session=ctx.session,
                    user_message=str(s.get("task", ctx.user_message)),
                    parent_agent=self,
                    trace_collector=ctx.trace_collector,
                )
                events: list[Event] = []
                output = ""
                async for event in agent._run_async_impl_traced(sub_ctx):
                    events.append(event)
                    if event.event_type == EventType.AGENT_MESSAGE:
                        output = event.content
                return s["step"], events, output

            if len(runnable) == 1:
                step_num, events, output = await _run_step(runnable[0])
                step_results[step_num] = output
                for event in events:
                    yield event
                remaining.remove(runnable[0])
            else:
                results = await asyncio.gather(
                    *[_run_step(s) for s in runnable],
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, BaseException):
                        logger.error("[%s] Planner step failed: %s", self.name, r)
                        continue
                    step_num, events, output = r
                    step_results[step_num] = output
                    for event in events:
                        yield event
                for s in runnable:
                    remaining.remove(s)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Completed {len(step_results)}/{len(plan)} steps",
                data={"completed": len(step_results), "total": len(plan)},
            )

        # Synthesis
        synth_messages = self._build_synthesis_messages(
            ctx.user_message, step_results,
        )
        synthesis = await asyncio.to_thread(
            self.service.generate_completion,
            messages=synth_messages, temperature=0.0, max_tokens=2048,
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, str(synthesis))

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "plan": plan,
                "step_results": step_results,
                "synthesis": str(synthesis),
            })


# ── SubQuestionAgent ──────────────────────────────────────────────────────────


class SubQuestionAgent(BaseAgent):
    """Decompose a complex question into sub-questions, dispatch, and synthesise.

    Uses an LLM to break a multi-faceted question into targeted
    sub-questions, assigns each to the most appropriate sub-agent, runs
    them in parallel, and synthesises the combined answers.

    Inspired by LlamaIndex ``SubQuestionQueryEngine`` but implemented as
    a composable ``BaseAgent``.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of specialist agents.
        model: LLM model for decomposition and synthesis.
        provider: LLM provider.
        api_key: API key override.
        decomposition_instruction: Extra instruction for the decomposer.
        max_sub_questions: Maximum sub-questions to generate.
        max_workers: Maximum concurrent threads (sync).
        synthesis_prompt: Custom prompt for synthesis.
        result_key: Store sub-questions + answers in ``session.state``.
        service_kwargs: Extra kwargs for the connector service constructor.

    Example:
        >>> sqd = SubQuestionAgent(
        ...     name="analyst",
        ...     provider="google",
        ...     sub_agents=[market_agent, tech_agent, finance_agent],
        ...     max_sub_questions=4,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        decomposition_instruction: str = "",
        max_sub_questions: int = 5,
        max_workers: int = 4,
        synthesis_prompt: str = "",
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.decomposition_instruction = decomposition_instruction
        self.max_sub_questions = max_sub_questions
        self.max_workers = max_workers
        self.synthesis_prompt = synthesis_prompt
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialize the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _build_decomposition_messages(
        self, user_message: str,
    ) -> list[dict[str, str]]:
        """Build messages for the decomposition LLM call.

        Args:
            user_message: The user's complex question.

        Returns:
            List of message dicts for the LLM.
        """
        agent_list = "\n".join(
            f'  - "{a.name}": {a.description or "No description"}'
            for a in self.sub_agents
        )
        system = (
            "You are a question decomposition assistant. Break the user's "
            f"question into {self.max_sub_questions} or fewer focused "
            "sub-questions. Assign each to the most appropriate agent.\n\n"
            f"Available agents:\n{agent_list}\n\n"
            "Respond with a JSON array (no markdown fences). Each element:\n"
            "{\n"
            '  "question": "the focused sub-question",\n'
            '  "agent": "agent_name"\n'
            "}\n\n"
            "Rules:\n"
            "- Each sub-question should be self-contained.\n"
            "- Each agent name MUST be from the list above.\n"
            "- Sub-questions should cover different aspects of the main question.\n"
            "- Do NOT answer any question — only decompose."
        )
        if self.decomposition_instruction:
            system += f"\n\n{self.decomposition_instruction}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _parse_sub_questions(
        self, response: str,
    ) -> list[dict[str, str]]:
        """Parse the decomposition response.

        Args:
            response: Raw LLM response text.

        Returns:
            List of dicts with keys: question, agent.
        """
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[%s] Sub-question parse error: %s", self.name, text[:200],
            )
            return []
        if not isinstance(data, list):
            return []
        return data[: self.max_sub_questions]

    def _build_synthesis_messages(
        self, user_message: str,
        sub_qa: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build messages for the synthesis LLM call.

        Args:
            user_message: Original user message.
            sub_qa: List of {question, agent, answer} dicts.

        Returns:
            List of message dicts for the LLM.
        """
        parts = "\n\n".join(
            f"**Q:** {qa['question']}\n"
            f"**A ({qa['agent']}):** {qa.get('answer', '(no answer)')}"
            for qa in sub_qa
        )
        prompt = self.synthesis_prompt or (
            "Synthesise the following sub-answers into a single cohesive "
            "response for the user's original question."
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Original question: {user_message}\n\n"
                    f"Sub-answers:\n{parts}"
                ),
            },
        ]

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Decompose, dispatch, and synthesise (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from sub-question execution and synthesis.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        # Decompose
        messages = self._build_decomposition_messages(ctx.user_message)
        response = self.service.generate_completion(
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        sub_questions = self._parse_sub_questions(str(response))

        if not sub_questions:
            yield Event(
                EventType.ERROR, self.name,
                "Failed to decompose question into sub-questions.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Decomposed into {len(sub_questions)} sub-questions",
            data={"sub_questions": sub_questions},
        )

        # Dispatch in parallel
        def _answer(
            sq: dict[str, str],
        ) -> tuple[dict[str, str], list[Event]]:
            agent = self.find_sub_agent(sq.get("agent", ""))
            if not agent:
                sq["answer"] = "(agent not found)"
                return sq, []
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=sq["question"],
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            sq["answer"] = output
            return sq, events

        sub_qa: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS)) as executor:
            futures = {
                executor.submit(_answer, sq): sq for sq in sub_questions
            }
            for future in as_completed(futures):
                try:
                    sq, events = future.result(timeout=_FUTURE_TIMEOUT)
                    sub_qa.append(sq)
                    for event in events:
                        yield event
                except Exception as exc:
                    logger.error("[%s] Sub-question failed: %s", self.name, exc)

        # Synthesis
        synth_messages = self._build_synthesis_messages(
            ctx.user_message, sub_qa,
        )
        synthesis = self.service.generate_completion(
            messages=synth_messages, temperature=0.0, max_tokens=2048,
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, str(synthesis))

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "sub_questions": sub_qa,
                "synthesis": str(synthesis),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Decompose, dispatch, and synthesise (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from sub-question execution and synthesis.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        messages = self._build_decomposition_messages(ctx.user_message)
        response = await asyncio.to_thread(
            self.service.generate_completion,
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        sub_questions = self._parse_sub_questions(str(response))

        if not sub_questions:
            yield Event(
                EventType.ERROR, self.name,
                "Failed to decompose question into sub-questions.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Decomposed into {len(sub_questions)} sub-questions",
            data={"sub_questions": sub_questions},
        )

        async def _answer(
            sq: dict[str, str],
        ) -> tuple[dict[str, str], list[Event]]:
            agent = self.find_sub_agent(sq.get("agent", ""))
            if not agent:
                sq["answer"] = "(agent not found)"
                return sq, []
            sub_ctx = InvocationContext(
                session=ctx.session,
                user_message=sq["question"],
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            sq["answer"] = output
            return sq, events

        results = await asyncio.gather(
            *[_answer(sq) for sq in sub_questions],
            return_exceptions=True,
        )

        sub_qa: list[dict[str, str]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "[%s] Async sub-question failed: %s", self.name, result,
                )
                continue
            sq, events = result
            sub_qa.append(sq)
            for event in events:
                yield event

        synth_messages = self._build_synthesis_messages(
            ctx.user_message, sub_qa,
        )
        synthesis = await asyncio.to_thread(
            self.service.generate_completion,
            messages=synth_messages, temperature=0.0, max_tokens=2048,
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, str(synthesis))

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "sub_questions": sub_qa,
                "synthesis": str(synthesis),
            })


# ── ContextFilterAgent ────────────────────────────────────────────────────────


class ContextFilterAgent(BaseAgent):
    """Filter context/history before delegating to sub-agents.

    In large multi-agent workflows the accumulated event history can
    confuse downstream agents.  ``ContextFilterAgent`` applies per-agent
    filtering rules before delegation, reducing noise and hallucination.

    Inspired by AutoGen's ``MessageFilterAgent`` and OpenAI SDK's
    ``input_filter``.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Agents that receive filtered context.
        filter_fn: Optional custom ``(agent, events) -> events`` filter
            applied per sub-agent.  When ``None``, built-in filters
            (``max_history``, ``include_sources``, ``exclude_sources``) are
            used.
        max_history: Maximum events each agent sees (most recent).
        include_sources: Whitelist — only events from these agent names.
        exclude_sources: Blacklist — exclude events from these agent names.
        mode: ``"sequential"`` or ``"parallel"`` execution of sub-agents.
        result_key: Store filtered event counts in ``session.state``.

    Example:
        >>> cf = ContextFilterAgent(
        ...     name="focused",
        ...     sub_agents=[analyst, writer],
        ...     max_history=10,
        ...     exclude_sources=["debug_logger"],
        ...     mode="sequential",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        filter_fn: Callable[
            [BaseAgent, list[Event]], list[Event],
        ] | None = None,
        max_history: int | None = None,
        include_sources: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        mode: str = "sequential",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.filter_fn = filter_fn
        self.max_history = max_history
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources
        self.mode = mode
        self.result_key = result_key

    def _apply_filter(
        self, agent: BaseAgent, events: list[Event],
    ) -> list[Event]:
        """Apply filtering rules to events for a specific agent.

        Args:
            agent: The target agent.
            events: The full event history.

        Returns:
            Filtered list of events.
        """
        if self.filter_fn:
            return self.filter_fn(agent, events)

        filtered = list(events)
        if self.include_sources:
            filtered = [
                e for e in filtered if e.author in self.include_sources
            ]
        if self.exclude_sources:
            filtered = [
                e for e in filtered if e.author not in self.exclude_sources
            ]
        if self.max_history is not None:
            filtered = filtered[-self.max_history:]
        return filtered

    def _build_context_message(
        self, events: list[Event], user_message: str,
    ) -> str:
        """Build a message from filtered events + the user message.

        Args:
            events: Filtered events.
            user_message: Original user message.

        Returns:
            Combined message string.
        """
        if not events:
            return user_message
        context_parts = [
            f"[{e.author}]: {e.content}"
            for e in events
            if e.event_type == EventType.AGENT_MESSAGE and e.content
        ]
        if not context_parts:
            return user_message
        context = "\n".join(context_parts)
        return f"Context:\n{context}\n\nCurrent request: {user_message}"

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute sub-agents with filtered context (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from sub-agent executions.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        all_events = list(ctx.session.events)
        filter_stats: dict[str, int] = {}

        if self.mode == "parallel":
            def _run_agent(
                agent: BaseAgent,
            ) -> tuple[str, list[Event]]:
                filtered = self._apply_filter(agent, all_events)
                message = self._build_context_message(
                    filtered, ctx.user_message,
                )
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                events: list[Event] = []
                for event in agent._run_impl_traced(sub_ctx):
                    events.append(event)
                return agent.name, events

            with ThreadPoolExecutor(
                max_workers=min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS),
            ) as executor:
                futures = {
                    executor.submit(_run_agent, a): a
                    for a in self.sub_agents
                }
                for future in as_completed(futures):
                    try:
                        agent_name, events = future.result(timeout=_FUTURE_TIMEOUT)
                        filter_stats[agent_name] = len(
                            self._apply_filter(
                                futures[future], all_events,
                            ),
                        )
                        for event in events:
                            yield event
                    except Exception as exc:
                        logger.error(
                            "[%s] Agent failed: %s", self.name, exc,
                        )
        else:
            for agent in self.sub_agents:
                filtered = self._apply_filter(agent, all_events)
                filter_stats[agent.name] = len(filtered)
                message = self._build_context_message(
                    filtered, ctx.user_message,
                )
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                for event in agent._run_impl_traced(sub_ctx):
                    yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "filter_stats": filter_stats,
                "original_events": len(all_events),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute sub-agents with filtered context (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from sub-agent executions.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        all_events = list(ctx.session.events)
        filter_stats: dict[str, int] = {}

        if self.mode == "parallel":
            async def _run_agent(
                agent: BaseAgent,
            ) -> tuple[str, list[Event]]:
                filtered = self._apply_filter(agent, all_events)
                message = self._build_context_message(
                    filtered, ctx.user_message,
                )
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                events: list[Event] = []
                async for event in agent._run_async_impl_traced(sub_ctx):
                    events.append(event)
                return agent.name, events

            results = await asyncio.gather(
                *[_run_agent(a) for a in self.sub_agents],
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "[%s] Async agent failed: %s", self.name, result,
                    )
                    continue
                agent_name, events = result
                filter_stats[agent_name] = len(
                    self._apply_filter(self.sub_agents[i], all_events),
                )
                for event in events:
                    yield event
        else:
            for agent in self.sub_agents:
                filtered = self._apply_filter(agent, all_events)
                filter_stats[agent.name] = len(filtered)
                message = self._build_context_message(
                    filtered, ctx.user_message,
                )
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                async for event in agent._run_async_impl_traced(sub_ctx):
                    yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "filter_stats": filter_stats,
                "original_events": len(all_events),
            })


# ── ReflexionAgent ────────────────────────────────────────────────────────────


class ReflexionAgent(BaseAgent):
    """Reflexion: iterative self-improvement with persistent memory.

    Unlike ``ProducerReviewerAgent`` (which forgets between sessions),
    ``ReflexionAgent`` accumulates lessons learned across attempts in
    ``session.state[memory_key]`` — each retry includes past failures
    and evaluator feedback so the agent can self-correct.

    Implements the Reflexion algorithm (Shinn et al., 2023).

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Main agent that generates responses.
        evaluator: Agent that evaluates outputs and provides feedback.
        score_fn: ``(evaluator_response: str) -> float``.
        threshold: Score to accept.
        max_attempts: Maximum generation attempts.
        memory_key: Key in ``session.state`` for accumulated lessons.
        result_key: Store attempt history in ``session.state``.
        max_memories: Maximum number of lessons to retain.  Oldest are
            evicted when the limit is reached.  ``0`` means unlimited.

    Example:
        >>> reflex = ReflexionAgent(
        ...     name="learner",
        ...     agent=coder,
        ...     evaluator=reviewer,
        ...     score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
        ...     threshold=0.8,
        ...     max_attempts=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent,
        evaluator: BaseAgent,
        score_fn: Callable[[str], float],
        threshold: float = 0.8,
        max_attempts: int = 3,
        memory_key: str = "reflexion_memory",
        result_key: str | None = None,
        max_memories: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[agent, evaluator], **kwargs,
        )
        self.agent = agent
        self.evaluator = evaluator
        self.score_fn = score_fn
        self.threshold = threshold
        self.max_attempts = max_attempts
        self.memory_key = memory_key
        self.result_key = result_key
        self.max_memories = max_memories

    def _build_prompt_with_memory(
        self, user_message: str, memories: list[dict[str, str]],
    ) -> str:
        """Build a prompt that includes past lessons.

        Args:
            user_message: Original user message.
            memories: List of {attempt, feedback, lesson} dicts.

        Returns:
            Enriched prompt.
        """
        if not memories:
            return user_message
        lessons = "\n".join(
            f"- Attempt {m['attempt']}: {m['lesson']}"
            for m in memories
        )
        return (
            f"{user_message}\n\n"
            f"LESSONS FROM PREVIOUS ATTEMPTS (apply these):\n{lessons}"
        )

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Generate, evaluate, reflect, and retry (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from generation, evaluation, and the accepted output.
        """
        memories: list[dict[str, str]] = list(
            ctx.session.state.get(self.memory_key, []),
        )
        # Enforce cap on memories carried over from previous runs
        if self.max_memories > 0 and len(memories) > self.max_memories:
            memories = memories[-self.max_memories:]
        attempts: list[dict[str, Any]] = []
        gen_events: list[Event] = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_prompt_with_memory(
                ctx.user_message, memories,
            )
            gen_ctx = InvocationContext(
                session=ctx.session, user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            gen_output = ""
            gen_events = []
            for event in self.agent._run_impl_traced(gen_ctx):
                gen_events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    gen_output = event.content

            eval_ctx = InvocationContext(
                session=ctx.session, user_message=gen_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            eval_output = ""
            for event in self.evaluator._run_impl_traced(eval_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    eval_output = event.content

            score = self.score_fn(eval_output)
            logger.info(
                "[%s] Attempt %d/%d: score=%.2f (threshold=%.2f)",
                self.name, attempt, self.max_attempts, score, self.threshold,
            )

            attempts.append({
                "attempt": attempt,
                "output": gen_output,
                "feedback": eval_output,
                "score": score,
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Attempt {attempt}: score={score:.2f}",
                data={
                    "attempt": attempt,
                    "score": score,
                    "feedback": eval_output,
                },
            )

            if score >= self.threshold:
                for event in gen_events:
                    yield event
                ctx.session.state_set(self.memory_key, memories)
                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "attempts": attempts,
                        "accepted_attempt": attempt,
                        "score": score,
                    })
                return

            # Reflect
            lesson = (
                f"Score {score:.2f} — evaluator said: "
                f"{eval_output[:200]}"
            )
            memories.append({
                "attempt": str(attempt),
                "feedback": eval_output[:200],
                "lesson": lesson,
            })
            # Evict oldest memories when limit is set
            if self.max_memories > 0 and len(memories) > self.max_memories:
                memories = memories[-self.max_memories:]

        # Exhausted — yield last output
        for event in gen_events:
            yield event

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"All {self.max_attempts} attempts exhausted "
            f"(best={max(a['score'] for a in attempts):.2f})",
            data={"exhausted": True},
        )

        ctx.session.state_set(self.memory_key, memories)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "attempts": attempts,
                "accepted_attempt": None,
                "score": max(a["score"] for a in attempts),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Generate, evaluate, reflect, and retry (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from generation, evaluation, and the accepted output.
        """
        memories: list[dict[str, str]] = list(
            ctx.session.state.get(self.memory_key, []),
        )
        # Enforce cap on memories carried over from previous runs
        if self.max_memories > 0 and len(memories) > self.max_memories:
            memories = memories[-self.max_memories:]
        attempts: list[dict[str, Any]] = []
        gen_events: list[Event] = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_prompt_with_memory(
                ctx.user_message, memories,
            )
            gen_ctx = InvocationContext(
                session=ctx.session, user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            gen_output = ""
            gen_events = []
            async for event in self.agent._run_async_impl_traced(gen_ctx):
                gen_events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    gen_output = event.content

            eval_ctx = InvocationContext(
                session=ctx.session, user_message=gen_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            eval_output = ""
            async for event in self.evaluator._run_async_impl_traced(eval_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    eval_output = event.content

            score = self.score_fn(eval_output)
            logger.info(
                "[%s] Async attempt %d/%d: score=%.2f",
                self.name, attempt, self.max_attempts, score,
            )

            attempts.append({
                "attempt": attempt,
                "output": gen_output,
                "feedback": eval_output,
                "score": score,
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Attempt {attempt}: score={score:.2f}",
                data={
                    "attempt": attempt,
                    "score": score,
                    "feedback": eval_output,
                },
            )

            if score >= self.threshold:
                for event in gen_events:
                    yield event
                ctx.session.state_set(self.memory_key, memories)
                if self.result_key:
                    ctx.session.state_set(self.result_key, {
                        "attempts": attempts,
                        "accepted_attempt": attempt,
                        "score": score,
                    })
                return

            lesson = (
                f"Score {score:.2f} — evaluator said: "
                f"{eval_output[:200]}"
            )
            memories.append({
                "attempt": str(attempt),
                "feedback": eval_output[:200],
                "lesson": lesson,
            })
            # Evict oldest memories when limit is set
            if self.max_memories > 0 and len(memories) > self.max_memories:
                memories = memories[-self.max_memories:]

        for event in gen_events:
            yield event

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"All {self.max_attempts} attempts exhausted "
            f"(best={max(a['score'] for a in attempts):.2f})",
            data={"exhausted": True},
        )

        ctx.session.state_set(self.memory_key, memories)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "attempts": attempts,
                "accepted_attempt": None,
                "score": max(a["score"] for a in attempts),
            })


# ── SpeculativeAgent ──────────────────────────────────────────────────────────


class SpeculativeAgent(BaseAgent):
    """Speculative execution: race multiple agents, cancel losers early.

    Starts all ``sub_agents`` in parallel. As soon as one agent's output
    passes ``evaluator_fn`` with score >= ``min_confidence``, the winner
    is accepted and slower agents are effectively abandoned.

    Unlike ``BestOfNAgent`` (which waits for *all* to finish), this
    optimises for **latency** — if a fast/cheap model produces a good
    result, the expensive model is never waited for.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Competing agents (typically different models/providers).
        evaluator_fn: ``(response: str) -> float`` scoring function.
        min_confidence: Minimum score to accept a result early.
        result_key: Store winner info in ``session.state``.

    Example:
        >>> spec = SpeculativeAgent(
        ...     name="racer",
        ...     sub_agents=[fast_agent, slow_good_agent],
        ...     evaluator_fn=lambda r: 1.0 if len(r) > 100 else 0.4,
        ...     min_confidence=0.8,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        evaluator_fn: Callable[[str], float] | None = None,
        min_confidence: float = 0.8,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.evaluator_fn = evaluator_fn or (lambda _r: 1.0)
        self.min_confidence = min_confidence
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Race all sub-agents (sync, ThreadPoolExecutor).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the winning agent.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        winner_name: str = ""
        winner_output: str = ""
        winner_score: float = 0.0
        winner_events: list[Event] = []

        def _run_one(
            agent: BaseAgent,
        ) -> tuple[str, list[Event], str, float]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            score = self.evaluator_fn(output)
            return agent.name, events, output, score

        with ThreadPoolExecutor(
            max_workers=min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS),
        ) as executor:
            futures = {
                executor.submit(_run_one, a): a for a in self.sub_agents
            }
            for future in as_completed(futures):
                try:
                    name, events, output, score = future.result(timeout=_FUTURE_TIMEOUT)
                    logger.info(
                        "[%s] Agent %s finished: score=%.2f",
                        self.name, name, score,
                    )
                    if score >= self.min_confidence:
                        winner_name = name
                        winner_output = output
                        winner_score = score
                        winner_events = events
                        for f in futures:
                            f.cancel()
                        break
                    if score > winner_score:
                        winner_name = name
                        winner_output = output
                        winner_score = score
                        winner_events = events
                except Exception as exc:
                    logger.error(
                        "[%s] Agent failed: %s", self.name, exc,
                    )

        for event in winner_events:
            yield event

        if not winner_name:
            yield Event(EventType.AGENT_MESSAGE, self.name, winner_output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Winner: {winner_name} (score={winner_score:.2f})",
            data={
                "winner": winner_name,
                "score": winner_score,
                "early_stop": winner_score >= self.min_confidence,
            },
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name,
                "score": winner_score,
                "early_stop": winner_score >= self.min_confidence,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Race all sub-agents (async, asyncio with cancellation).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the winning agent.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        winner_name: str = ""
        winner_output: str = ""
        winner_score: float = 0.0
        winner_events: list[Event] = []

        async def _run_one(
            agent: BaseAgent,
        ) -> tuple[str, list[Event], str, float]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            score = self.evaluator_fn(output)
            return agent.name, events, output, score

        tasks = [
            asyncio.create_task(_run_one(a)) for a in self.sub_agents
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                name, events, output, score = await coro
                logger.info(
                    "[%s] Async agent %s finished: score=%.2f",
                    self.name, name, score,
                )
                if score >= self.min_confidence:
                    winner_name = name
                    winner_output = output
                    winner_score = score
                    winner_events = events
                    for t in tasks:
                        t.cancel()
                    break
                if score > winner_score:
                    winner_name = name
                    winner_output = output
                    winner_score = score
                    winner_events = events
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("[%s] Async agent failed: %s", self.name, exc)

        for event in winner_events:
            yield event

        if not winner_name:
            yield Event(EventType.AGENT_MESSAGE, self.name, winner_output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Winner: {winner_name} (score={winner_score:.2f})",
            data={
                "winner": winner_name,
                "score": winner_score,
                "early_stop": winner_score >= self.min_confidence,
            },
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name,
                "score": winner_score,
                "early_stop": winner_score >= self.min_confidence,
            })


# ── CircuitBreakerAgent ───────────────────────────────────────────────────────


class CircuitBreakerAgent(BaseAgent):
    """Circuit breaker: protect against cascading agent failures.

    Monitors the wrapped agent's failure rate.  When failures exceed
    ``failure_threshold`` within ``window_size`` calls, the circuit
    "opens" and routes directly to ``fallback_agent`` (or yields an
    error if no fallback).  After ``recovery_timeout`` calls in the
    open state, the circuit enters "half-open" and tries the primary
    agent once — if it succeeds, the circuit closes.

    Inspired by Netflix Hystrix / microservice circuit breaker patterns.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The primary agent to protect.
        fallback_agent: Agent to use when circuit is open (optional).
        failure_threshold: Max failures before opening the circuit.
        window_size: Rolling window of recent calls for counting failures.
        recovery_timeout: Calls in open state before trying half-open.
        failure_detector: Optional ``(output: str) -> bool`` that returns
            ``True`` if the output should be counted as a failure.
        result_key: Store circuit state in ``session.state``.

    Example:
        >>> cb = CircuitBreakerAgent(
        ...     name="protected",
        ...     agent=flaky_agent,
        ...     fallback_agent=safe_agent,
        ...     failure_threshold=3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent,
        fallback_agent: BaseAgent | None = None,
        failure_threshold: int = 3,
        window_size: int = 10,
        recovery_timeout: int = 5,
        failure_detector: Callable[[str], bool] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        sub = [agent] + ([fallback_agent] if fallback_agent else [])
        super().__init__(
            name=name, description=description,
            sub_agents=sub, **kwargs,
        )
        self.agent = agent
        self.fallback_agent = fallback_agent
        self.failure_threshold = failure_threshold
        self.window_size = window_size
        self.recovery_timeout = recovery_timeout
        self.failure_detector = failure_detector or (
            lambda output: not output or len(output.strip()) < 5
        )
        self.result_key = result_key
        self._history: list[bool] = []
        self._state: str = "closed"
        self._open_counter: int = 0
        self._cb_lock = threading.Lock()

    def _record(self, success: bool) -> None:
        """Record a call result and update circuit state.

        Args:
            success: Whether the call succeeded.
        """
        with self._cb_lock:
            self._history.append(success)
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]

            if self._state == "closed":
                failures = sum(1 for h in self._history if not h)
                if failures >= self.failure_threshold:
                    self._state = "open"
                    self._open_counter = 0
                    logger.warning(
                        "[%s] Circuit OPENED — %d failures in last %d calls",
                        self.name, failures, len(self._history),
                    )
            elif self._state == "half-open":
                if success:
                    self._state = "closed"
                    self._history.clear()
                    logger.info("[%s] Circuit CLOSED — recovery", self.name)
                else:
                    self._state = "open"
                    self._open_counter = 0
                    logger.warning("[%s] Circuit re-OPENED", self.name)

    def _should_try_primary(self) -> bool:
        """Check whether the primary agent should be tried.

        Returns:
            True if circuit is closed or half-open.
        """
        with self._cb_lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                self._open_counter += 1
                if self._open_counter >= self.recovery_timeout:
                    self._state = "half-open"
                    logger.info("[%s] Circuit HALF-OPEN", self.name)
                    return True
                return False
            return True

    def _run_sub(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[list[Event], str]:
        """Execute an agent and capture output (sync).

        Args:
            agent: Agent to run.
            ctx: The invocation context.

        Returns:
            Tuple of (events, output).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    async def _run_sub_async(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[list[Event], str]:
        """Execute an agent and capture output (async).

        Args:
            agent: Agent to run.
            ctx: The invocation context.

        Returns:
            Tuple of (events, output).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute with circuit breaker protection (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from primary or fallback agent.
        """
        if self._should_try_primary():
            try:
                events, output = self._run_sub(self.agent, ctx)
                failed = self.failure_detector(output)
                self._record(not failed)
                if not failed:
                    for event in events:
                        yield event
                    self._emit_state(ctx, "primary_success")
                    return
                logger.warning(
                    "[%s] Primary output detected as failure", self.name,
                )
            except Exception as exc:
                self._record(False)
                logger.error("[%s] Primary exception: %s", self.name, exc)

        if self.fallback_agent:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Circuit {self._state} — using fallback",
                data={"circuit_state": self._state, "using": "fallback"},
            )
            events, output = self._run_sub(self.fallback_agent, ctx)
            for event in events:
                yield event
            self._emit_state(ctx, "fallback_used")
        else:
            yield Event(
                EventType.ERROR, self.name,
                f"Circuit {self._state} — no fallback configured.",
            )
            self._emit_state(ctx, "no_fallback")

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute with circuit breaker protection (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from primary or fallback agent.
        """
        if self._should_try_primary():
            try:
                events, output = await self._run_sub_async(self.agent, ctx)
                failed = self.failure_detector(output)
                self._record(not failed)
                if not failed:
                    for event in events:
                        yield event
                    self._emit_state(ctx, "primary_success")
                    return
                logger.warning(
                    "[%s] Async primary failure detected", self.name,
                )
            except Exception as exc:
                self._record(False)
                logger.error(
                    "[%s] Async primary exception: %s", self.name, exc,
                )

        if self.fallback_agent:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Circuit {self._state} — using fallback",
                data={"circuit_state": self._state, "using": "fallback"},
            )
            events, output = await self._run_sub_async(
                self.fallback_agent, ctx,
            )
            for event in events:
                yield event
            self._emit_state(ctx, "fallback_used")
        else:
            yield Event(
                EventType.ERROR, self.name,
                f"Circuit {self._state} — no fallback configured.",
            )
            self._emit_state(ctx, "no_fallback")

    def _emit_state(self, ctx: InvocationContext, outcome: str) -> None:
        """Persist circuit state to result_key if configured.

        Args:
            ctx: The invocation context.
            outcome: Description of what happened.
        """
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "circuit_state": self._state,
                "outcome": outcome,
                "history": list(self._history),
            })


# ── TournamentAgent ───────────────────────────────────────────────────────────


class TournamentAgent(BaseAgent):
    """Tournament bracket: pairwise elimination until one winner remains.

    Runs N agents, pairs them up, and a ``judge_agent`` picks the
    winner of each pair.  Winners advance to the next round until
    one agent remains.

    More accurate than ``VotingAgent`` for subjective tasks — LLMs
    are better at pairwise comparison than ranking N items at once.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Competing agents (N >= 2).
        judge_agent: Agent that compares two responses and picks a winner.
        max_workers: Maximum concurrent threads/tasks.
        result_key: Store bracket progression in ``session.state``.

    Example:
        >>> tourney = TournamentAgent(
        ...     name="bracket",
        ...     sub_agents=[writer_a, writer_b, writer_c, writer_d],
        ...     judge_agent=judge,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        judge_agent: BaseAgent | None = None,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        agents = list(sub_agents or [])
        if judge_agent:
            agents.append(judge_agent)
        super().__init__(
            name=name, description=description,
            sub_agents=agents, **kwargs,
        )
        self._competitors = list(sub_agents or [])
        self.judge_agent = judge_agent
        self.max_workers = max_workers
        self.result_key = result_key

    def _run_one(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a single agent (sync).

        Args:
            agent: Agent to execute.
            ctx: The invocation context.
            message: The user message for this agent.

        Returns:
            Tuple of (output, events).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    async def _run_one_async(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a single agent (async).

        Args:
            agent: Agent to execute.
            ctx: The invocation context.
            message: The user message.

        Returns:
            Tuple of (output, events).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    def _judge_pair(
        self, ctx: InvocationContext,
        name_a: str, output_a: str,
        name_b: str, output_b: str,
    ) -> str:
        """Judge a pair of outputs (sync).

        Args:
            ctx: The invocation context.
            name_a: First agent name.
            output_a: First agent's response.
            name_b: Second agent name.
            output_b: Second agent's response.

        Returns:
            Name of the winner.
        """
        prompt = (
            f"Compare these two responses to: {ctx.user_message}\n\n"
            f"--- Response A ({name_a}) ---\n{output_a}\n\n"
            f"--- Response B ({name_b}) ---\n{output_b}\n\n"
            f"Which is better? Reply with ONLY: "
            f'"{name_a}" or "{name_b}".'
        )
        result, _ = self._run_one(self.judge_agent, ctx, prompt)
        if name_b in result and name_a not in result:
            return name_b
        return name_a

    async def _judge_pair_async(
        self, ctx: InvocationContext,
        name_a: str, output_a: str,
        name_b: str, output_b: str,
    ) -> str:
        """Judge a pair of outputs (async).

        Args:
            ctx: The invocation context.
            name_a: First agent name.
            output_a: First agent's response.
            name_b: Second agent name.
            output_b: Second agent's response.

        Returns:
            Name of the winner.
        """
        prompt = (
            f"Compare these two responses to: {ctx.user_message}\n\n"
            f"--- Response A ({name_a}) ---\n{output_a}\n\n"
            f"--- Response B ({name_b}) ---\n{output_b}\n\n"
            f"Which is better? Reply with ONLY: "
            f'"{name_a}" or "{name_b}".'
        )
        result, _ = await self._run_one_async(self.judge_agent, ctx, prompt)
        if name_b in result and name_a not in result:
            return name_b
        return name_a

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run the tournament bracket (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each round and the final winner.
        """
        if len(self._competitors) < 2:
            yield Event(
                EventType.ERROR, self.name, "Need at least 2 competitors.",
            )
            return
        if not self.judge_agent:
            yield Event(EventType.ERROR, self.name, "No judge_agent configured.")
            return

        responses: dict[str, str] = {}

        def _gen(agent: BaseAgent) -> tuple[str, str, list[Event]]:
            output, events = self._run_one(agent, ctx, ctx.user_message)
            return agent.name, output, events

        with ThreadPoolExecutor(max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS)) as executor:
            futures = [
                executor.submit(_gen, a) for a in self._competitors
            ]
            for future in as_completed(futures):
                name, output, events = future.result(timeout=_FUTURE_TIMEOUT)
                responses[name] = output

        bracket: list[dict[str, Any]] = []
        competitors = list(responses.keys())
        rnd = 0

        while len(competitors) > 1:
            rnd += 1
            next_round: list[str] = []
            pairs: list[tuple[str, str]] = []

            for i in range(0, len(competitors) - 1, 2):
                pairs.append((competitors[i], competitors[i + 1]))
            if len(competitors) % 2 == 1:
                next_round.append(competitors[-1])

            for name_a, name_b in pairs:
                winner = self._judge_pair(
                    ctx, name_a, responses[name_a],
                    name_b, responses[name_b],
                )
                next_round.append(winner)
                bracket.append({
                    "round": rnd, "pair": [name_a, name_b],
                    "winner": winner,
                })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: {len(competitors)} → {len(next_round)}",
                data={"round": rnd, "winners": next_round},
            )
            competitors = next_round

        winner_name = competitors[0]
        yield Event(
            EventType.AGENT_MESSAGE, self.name, responses[winner_name],
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name, "bracket": bracket, "rounds": rnd,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the tournament bracket (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each round and the final winner.
        """
        if len(self._competitors) < 2:
            yield Event(
                EventType.ERROR, self.name, "Need at least 2 competitors.",
            )
            return
        if not self.judge_agent:
            yield Event(EventType.ERROR, self.name, "No judge_agent configured.")
            return

        async def _gen(
            agent: BaseAgent,
        ) -> tuple[str, str, list[Event]]:
            output, events = await self._run_one_async(
                agent, ctx, ctx.user_message,
            )
            return agent.name, output, events

        results = await asyncio.gather(
            *[_gen(a) for a in self._competitors],
            return_exceptions=True,
        )
        responses: dict[str, str] = {}
        for r in results:
            if isinstance(r, BaseException):
                continue
            name, output, events = r
            responses[name] = output

        bracket: list[dict[str, Any]] = []
        competitors = list(responses.keys())
        rnd = 0

        while len(competitors) > 1:
            rnd += 1
            next_round: list[str] = []
            pairs: list[tuple[str, str]] = []

            for i in range(0, len(competitors) - 1, 2):
                pairs.append((competitors[i], competitors[i + 1]))
            if len(competitors) % 2 == 1:
                next_round.append(competitors[-1])

            judge_tasks = [
                self._judge_pair_async(
                    ctx, a, responses[a], b, responses[b],
                )
                for a, b in pairs
            ]
            winners = await asyncio.gather(*judge_tasks, return_exceptions=True)

            for (name_a, name_b), winner in zip(pairs, winners):
                if isinstance(winner, BaseException):
                    winner = name_a  # default to first on judge failure
                next_round.append(winner)
                bracket.append({
                    "round": rnd, "pair": [name_a, name_b],
                    "winner": winner,
                })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: {len(competitors)} → {len(next_round)}",
                data={"round": rnd, "winners": next_round},
            )
            competitors = next_round

        winner_name = competitors[0]
        yield Event(
            EventType.AGENT_MESSAGE, self.name, responses[winner_name],
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name, "bracket": bracket, "rounds": rnd,
            })


# ── ShadowAgent ───────────────────────────────────────────────────────────────


class ShadowAgent(BaseAgent):
    """Shadow deployment: run candidate alongside stable, return stable only.

    Executes ``stable_agent`` and ``shadow_agent`` concurrently.  Only
    the ``stable_agent``'s output is yielded.  The shadow's output is
    logged for offline comparison — enabling "dark launches" of new
    models or prompts without affecting the user experience.

    Args:
        name: Agent name.
        description: Human-readable description.
        stable_agent: The production agent whose output is returned.
        shadow_agent: The candidate agent whose output is only logged.
        diff_logger: Optional ``(stable_output, shadow_output) -> None``.
        result_key: Store both outputs in ``session.state``.

    Example:
        >>> shadow = ShadowAgent(
        ...     name="canary",
        ...     stable_agent=gpt4_agent,
        ...     shadow_agent=gemini_agent,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        stable_agent: BaseAgent,
        shadow_agent: BaseAgent,
        diff_logger: Callable[[str, str], None] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[stable_agent, shadow_agent], **kwargs,
        )
        self.stable_agent = stable_agent
        self.shadow_agent = shadow_agent
        self.diff_logger = diff_logger
        self.result_key = result_key

    def _run_sub(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[list[Event], str]:
        """Run a single agent (sync).

        Args:
            agent: The agent to run.
            ctx: Invocation context.

        Returns:
            Tuple of (events, output).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    async def _run_sub_async(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[list[Event], str]:
        """Run a single agent (async).

        Args:
            agent: The agent to run.
            ctx: Invocation context.

        Returns:
            Tuple of (events, output).
        """
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run stable and shadow in parallel, yield stable only (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the stable agent only.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_stable = executor.submit(self._run_sub, self.stable_agent, ctx)
            f_shadow = executor.submit(self._run_sub, self.shadow_agent, ctx)
            stable_events, stable_output = f_stable.result(timeout=_FUTURE_TIMEOUT)
            try:
                _, shadow_output = f_shadow.result(timeout=_FUTURE_TIMEOUT)
            except Exception as exc:
                logger.warning("[%s] Shadow failed: %s", self.name, exc)
                shadow_output = f"(error: {exc})"

        for event in stable_events:
            yield event

        match = stable_output.strip() == shadow_output.strip()
        if self.diff_logger:
            self.diff_logger(stable_output, shadow_output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Shadow comparison: match={match}",
            data={
                "match": match,
                "stable_agent": self.stable_agent.name,
                "shadow_agent": self.shadow_agent.name,
            },
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "stable_output": stable_output,
                "shadow_output": shadow_output,
                "match": match,
                "stable_agent": self.stable_agent.name,
                "shadow_agent": self.shadow_agent.name,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run stable and shadow in parallel, yield stable only (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the stable agent only.
        """
        stable_task = asyncio.create_task(
            self._run_sub_async(self.stable_agent, ctx),
        )
        shadow_task = asyncio.create_task(
            self._run_sub_async(self.shadow_agent, ctx),
        )

        stable_events, stable_output = await stable_task
        try:
            _, shadow_output = await shadow_task
        except Exception as exc:
            logger.warning("[%s] Async shadow failed: %s", self.name, exc)
            shadow_output = f"(error: {exc})"

        for event in stable_events:
            yield event

        match = stable_output.strip() == shadow_output.strip()
        if self.diff_logger:
            self.diff_logger(stable_output, shadow_output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Shadow comparison: match={match}",
            data={
                "match": match,
                "stable_agent": self.stable_agent.name,
                "shadow_agent": self.shadow_agent.name,
            },
        )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "stable_output": stable_output,
                "shadow_output": shadow_output,
                "match": match,
                "stable_agent": self.stable_agent.name,
                "shadow_agent": self.shadow_agent.name,
            })


# ── CompilerAgent ─────────────────────────────────────────────────────────────


class CompilerAgent(BaseAgent):
    """Prompt compiler: optimise sub-agent instructions against a dataset.

    A meta-agent that runs a ``target_agent`` against ``examples``,
    evaluates with ``metric_fn``, and iteratively refines the agent's
    instructions using an LLM optimiser.

    Inspired by DSPy BootstrapFewShot — but as a composable agent.

    Args:
        name: Agent name.
        description: Human-readable description.
        target_agent: Agent whose ``instruction`` will be optimised.
        examples: List of ``{"input": ..., "expected": ...}`` dicts.
        metric_fn: ``(output: str, expected: str) -> float`` scorer.
        model: LLM model for prompt optimisation.
        provider: LLM provider.
        max_iterations: Maximum optimisation rounds.
        result_key: Store optimisation history in ``session.state``.

    Example:
        >>> compiler = CompilerAgent(
        ...     name="optimizer",
        ...     target_agent=my_agent,
        ...     examples=[{"input": "2+2", "expected": "4"}],
        ...     metric_fn=lambda o, e: 1.0 if e in o else 0.0,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        target_agent: BaseAgent,
        examples: list[dict[str, str]] | None = None,
        metric_fn: Callable[[str, str], float] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        max_iterations: int = 3,
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[target_agent],
        )
        self.target_agent = target_agent
        self.examples = examples or []
        self.metric_fn = metric_fn or (
            lambda o, e: 1.0 if e.lower() in o.lower() else 0.0
        )
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.max_iterations = max_iterations
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialise the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _evaluate(
        self, ctx: InvocationContext,
    ) -> tuple[float, list[dict[str, Any]]]:
        """Run the target agent on all examples and compute average score.

        Args:
            ctx: Base invocation context.

        Returns:
            Tuple of (average_score, detail_list).
        """
        details: list[dict[str, Any]] = []
        total = 0.0
        for ex in self.examples:
            sub_ctx = InvocationContext(
                session=Session(), user_message=ex["input"],
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in self.target_agent._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            score = self.metric_fn(output, ex.get("expected", ""))
            total += score
            details.append({
                "input": ex["input"], "expected": ex.get("expected", ""),
                "output": output, "score": score,
            })
        return total / max(len(self.examples), 1), details

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Optimise the target agent's instruction (sync).

        Args:
            ctx: The invocation context.

        Yields:
            STATE_UPDATE per iteration, AGENT_MESSAGE with best instruction.
        """
        if not self.examples:
            yield Event(EventType.ERROR, self.name, "No examples provided.")
            return

        history: list[dict[str, Any]] = []
        current_instruction = getattr(
            self.target_agent, "instruction", "",
        )
        best_score = 0.0
        best_instruction = current_instruction

        for iteration in range(1, self.max_iterations + 1):
            avg, details = self._evaluate(ctx)
            history.append({
                "iteration": iteration,
                "instruction": current_instruction,
                "score": avg,
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration}: score={avg:.2f}",
                data={"iteration": iteration, "score": avg},
            )

            if avg > best_score:
                best_score = avg
                best_instruction = current_instruction
            if avg >= 1.0:
                break

            failures = [d for d in details if d["score"] < 1.0][:3]
            failure_text = "\n".join(
                f"Input: {f['input']}\nExpected: {f['expected']}\n"
                f"Got: {f['output']}"
                for f in failures
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a prompt engineer. Improve the system "
                        "instruction to fix the failures. Return ONLY "
                        "the improved instruction, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current instruction:\n{current_instruction}\n\n"
                        f"Score: {avg:.2f}\n\nFailures:\n{failure_text}"
                    ),
                },
            ]
            new_instruction = self.service.generate_completion(
                messages=messages, temperature=0.7, max_tokens=1024,
            )
            current_instruction = str(new_instruction)
            if hasattr(self.target_agent, "instruction"):
                self.target_agent.instruction = current_instruction

        if hasattr(self.target_agent, "instruction"):
            self.target_agent.instruction = best_instruction

        yield Event(EventType.AGENT_MESSAGE, self.name, best_instruction)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "best_instruction": best_instruction,
                "iterations": history,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Optimise the target agent's instruction (async).

        Args:
            ctx: The invocation context.

        Yields:
            STATE_UPDATE per iteration, AGENT_MESSAGE with best instruction.
        """
        if not self.examples:
            yield Event(EventType.ERROR, self.name, "No examples provided.")
            return

        history: list[dict[str, Any]] = []
        current_instruction = getattr(
            self.target_agent, "instruction", "",
        )
        best_score = 0.0
        best_instruction = current_instruction

        for iteration in range(1, self.max_iterations + 1):
            details: list[dict[str, Any]] = []
            total = 0.0
            for ex in self.examples:
                sub_ctx = InvocationContext(
                    session=Session(), user_message=ex["input"],
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                output = ""
                async for event in self.target_agent._run_async_impl_traced(
                    sub_ctx,
                ):
                    if event.event_type == EventType.AGENT_MESSAGE:
                        output = event.content
                score = self.metric_fn(output, ex.get("expected", ""))
                total += score
                details.append({
                    "input": ex["input"],
                    "expected": ex.get("expected", ""),
                    "output": output, "score": score,
                })
            avg = total / max(len(self.examples), 1)

            history.append({
                "iteration": iteration,
                "instruction": current_instruction,
                "score": avg,
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration}: score={avg:.2f}",
                data={"iteration": iteration, "score": avg},
            )

            if avg > best_score:
                best_score = avg
                best_instruction = current_instruction
            if avg >= 1.0:
                break

            failures = [d for d in details if d["score"] < 1.0][:3]
            failure_text = "\n".join(
                f"Input: {f['input']}\nExpected: {f['expected']}\n"
                f"Got: {f['output']}"
                for f in failures
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a prompt engineer. Improve the system "
                        "instruction to fix the failures. Return ONLY "
                        "the improved instruction."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current instruction:\n{current_instruction}\n\n"
                        f"Score: {avg:.2f}\n\nFailures:\n{failure_text}"
                    ),
                },
            ]
            new_instruction = await asyncio.to_thread(
                self.service.generate_completion,
                messages=messages, temperature=0.7, max_tokens=1024,
            )
            current_instruction = str(new_instruction)
            if hasattr(self.target_agent, "instruction"):
                self.target_agent.instruction = current_instruction

        if hasattr(self.target_agent, "instruction"):
            self.target_agent.instruction = best_instruction

        yield Event(EventType.AGENT_MESSAGE, self.name, best_instruction)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "best_instruction": best_instruction,
                "iterations": history,
            })


# ── CheckpointableAgent ──────────────────────────────────────────────────────


class CheckpointableAgent(BaseAgent):
    """Checkpoint and resume: pause and resume a multi-step pipeline.

    Wraps a list of sub-agents executed sequentially.  After each
    sub-agent completes, saves a checkpoint to ``session.state``.
    On subsequent runs it resumes from the last completed step.

    Inspired by LangGraph's checkpoint/interrupt/resume model.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ordered list of agents to execute.
        checkpoint_key: Key in ``session.state`` for checkpoint data.
        result_key: Store final result in ``session.state``.

    Example:
        >>> ckpt = CheckpointableAgent(
        ...     name="resilient_pipeline",
        ...     sub_agents=[step1, step2, step3],
        ...     checkpoint_key="pipeline_ckpt",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        checkpoint_key: str = "checkpoint",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.checkpoint_key = checkpoint_key
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute sub-agents sequentially with checkpointing (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each sub-agent, STATE_UPDATEs for checkpoints.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        ckpt = ctx.session.state.get(self.checkpoint_key, {})
        start_idx = ckpt.get("completed_step", 0)
        last_output = ckpt.get("last_output", ctx.user_message)

        if start_idx > 0:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Resuming from step {start_idx + 1}/{len(self.sub_agents)}",
                data={"resumed_from": start_idx},
            )

        outputs: list[str] = list(ckpt.get("outputs", []))

        for idx in range(start_idx, len(self.sub_agents)):
            agent = self.sub_agents[idx]
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=last_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output or last_output
            outputs.append(_truncate(last_output))

            ctx.session.state_set(self.checkpoint_key, {
                "completed_step": idx + 1,
                "last_output": _truncate(last_output),
                "outputs": outputs[-20:],
                "total_steps": len(self.sub_agents),
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Checkpoint: step {idx + 1}/{len(self.sub_agents)}",
                data={"step": idx + 1, "agent": agent.name,
                      "total": len(self.sub_agents)},
            )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "completed": True,
                "total_steps": len(self.sub_agents),
                "outputs": outputs[-20:],
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute sub-agents sequentially with checkpointing (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each sub-agent, STATE_UPDATEs for checkpoints.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        ckpt = ctx.session.state.get(self.checkpoint_key, {})
        start_idx = ckpt.get("completed_step", 0)
        last_output = ckpt.get("last_output", ctx.user_message)

        if start_idx > 0:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Resuming from step {start_idx + 1}/{len(self.sub_agents)}",
                data={"resumed_from": start_idx},
            )

        outputs: list[str] = list(ckpt.get("outputs", []))

        for idx in range(start_idx, len(self.sub_agents)):
            agent = self.sub_agents[idx]
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=last_output,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output or last_output
            outputs.append(_truncate(last_output))

            ctx.session.state_set(self.checkpoint_key, {
                "completed_step": idx + 1,
                "last_output": _truncate(last_output),
                "outputs": outputs[-20:],
                "total_steps": len(self.sub_agents),
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Checkpoint: step {idx + 1}/{len(self.sub_agents)}",
                data={"step": idx + 1, "agent": agent.name,
                      "total": len(self.sub_agents)},
            )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "completed": True,
                "total_steps": len(self.sub_agents),
                "outputs": outputs[-20:],
            })


# ── DynamicFanOutAgent ────────────────────────────────────────────────────────


class DynamicFanOutAgent(BaseAgent):
    """Dynamic fan-out: LLM determines how many parallel workers to launch.

    Uses an LLM to decompose work into N items, spawns one worker
    per item in parallel, and feeds all results to a reducer agent.

    Unlike ``MapReduceAgent`` (fixed at build time), the fan-out
    count is determined **at runtime** by the LLM.

    Inspired by LangGraph's ``Send`` API for dynamic map-reduce.

    Args:
        name: Agent name.
        description: Human-readable description.
        worker_agent: Agent that processes each work item.
        reducer_agent: Agent that combines all worker results.
        model: LLM model for work decomposition.
        provider: LLM provider.
        decomposition_instruction: Extra instruction for decomposition.
        max_items: Maximum items the LLM may produce.
        max_workers: Maximum concurrent threads/tasks.
        result_key: Store items and results in ``session.state``.

    Example:
        >>> fanout = DynamicFanOutAgent(
        ...     name="dynamic_research",
        ...     worker_agent=researcher,
        ...     reducer_agent=summarizer,
        ...     max_items=10,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        worker_agent: BaseAgent,
        reducer_agent: BaseAgent,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        decomposition_instruction: str = "",
        max_items: int = 10,
        max_workers: int = 4,
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[worker_agent, reducer_agent],
        )
        self.worker_agent = worker_agent
        self.reducer_agent = reducer_agent
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.decomposition_instruction = decomposition_instruction
        self.max_items = max_items
        self.max_workers = max_workers
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialise the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _decompose(self, user_message: str) -> list[str]:
        """Use LLM to decompose the task into work items.

        Args:
            user_message: The original user request.

        Returns:
            List of work-item strings.
        """
        system = (
            "Decompose the user's task into concrete, independent work items. "
            f"Return a JSON array of strings (max {self.max_items} items). "
            "No markdown fences. Each item should be self-contained."
        )
        if self.decomposition_instruction:
            system += f"\n\n{self.decomposition_instruction}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        response = self.service.generate_completion(
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        text = str(response).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            items = json.loads(text)
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[%s] Decomposition parse error: %s",
                self.name, text[:200],
            )
            return []
        if not isinstance(items, list):
            return []
        return [str(i) for i in items[: self.max_items]]

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Dynamic fan-out → parallel workers → reduce (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from workers and reducer.
        """
        items = self._decompose(ctx.user_message)
        if not items:
            yield Event(
                EventType.ERROR, self.name,
                "Failed to decompose task into work items.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Fan-out: {len(items)} work items",
            data={"items": items, "count": len(items)},
        )

        results: dict[int, str] = {}

        def _work(idx: int, item: str) -> tuple[int, str, list[Event]]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=item,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in self.worker_agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return idx, output, events

        with ThreadPoolExecutor(max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS)) as executor:
            futures = {
                executor.submit(_work, i, item): i
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                try:
                    idx, output, events = future.result(timeout=_FUTURE_TIMEOUT)
                    results[idx] = output
                    for event in events:
                        yield event
                except Exception as exc:
                    logger.error("[%s] Worker failed: %s", self.name, exc)

        combined = "\n\n".join(
            f"**Item {i + 1}:** {items[i]}\n"
            f"**Result:** {results.get(i, '(failed)')}"
            for i in range(len(items))
        )
        reduce_ctx = InvocationContext(
            session=ctx.session,
            user_message=(
                f"Original request: {ctx.user_message}\n\n"
                f"Worker results:\n{combined}"
            ),
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        for event in self.reducer_agent._run_impl_traced(reduce_ctx):
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "items": items, "results": results, "count": len(items),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Dynamic fan-out → parallel workers → reduce (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from workers and reducer.
        """
        items = await asyncio.to_thread(
            self._decompose, ctx.user_message,
        )
        if not items:
            yield Event(
                EventType.ERROR, self.name,
                "Failed to decompose task into work items.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Fan-out: {len(items)} work items",
            data={"items": items, "count": len(items)},
        )

        async def _work(
            idx: int, item: str,
        ) -> tuple[int, str, list[Event]]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=item,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in self.worker_agent._run_async_impl_traced(
                sub_ctx,
            ):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return idx, output, events

        tasks = await asyncio.gather(
            *[_work(i, item) for i, item in enumerate(items)],
            return_exceptions=True,
        )

        results: dict[int, str] = {}
        for result in tasks:
            if isinstance(result, Exception):
                logger.error(
                    "[%s] Async worker failed: %s", self.name, result,
                )
                continue
            idx, output, events = result
            results[idx] = output
            for event in events:
                yield event

        combined = "\n\n".join(
            f"**Item {i + 1}:** {items[i]}\n"
            f"**Result:** {results.get(i, '(failed)')}"
            for i in range(len(items))
        )
        reduce_ctx = InvocationContext(
            session=ctx.session,
            user_message=(
                f"Original request: {ctx.user_message}\n\n"
                f"Worker results:\n{combined}"
            ),
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        async for event in self.reducer_agent._run_async_impl_traced(
            reduce_ctx,
        ):
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "items": items, "results": results, "count": len(items),
            })


# ── SwarmAgent ────────────────────────────────────────────────────────────────


class SwarmAgent(BaseAgent):
    """Swarm orchestration: fluid agent handoffs with evolving context.

    Maintains ``context_variables`` that evolve as agents process
    requests.  Each agent can update context and designate the next
    agent.  Continues until ``"__done__"`` is set or no next agent.

    Unlike ``HandoffAgent`` (keyword routing in a flat mesh),
    ``SwarmAgent`` provides a **shared context dict** that accumulates
    state — similar to OpenAI Swarm's ``context_variables`` protocol.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of available agents.
        initial_agent: Name of the first agent to run.
        context_variables: Initial context dict (shared + mutable).
        max_handoffs: Maximum handoffs before force-stop.
        next_agent_key: Key in ``session.state`` for next-agent name.
        result_key: Store handoff chain in ``session.state``.

    Example:
        >>> swarm = SwarmAgent(
        ...     name="support",
        ...     sub_agents=[triage, billing, tech],
        ...     initial_agent="triage",
        ...     context_variables={"customer_tier": "premium"},
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        initial_agent: str = "",
        context_variables: dict[str, Any] | None = None,
        max_handoffs: int = 10,
        next_agent_key: str = "__next_agent__",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.initial_agent = initial_agent
        self.context_variables = dict(context_variables or {})
        self.max_handoffs = max_handoffs
        self.next_agent_key = next_agent_key
        self.result_key = result_key

    def _build_message(
        self, user_message: str, context: dict[str, Any],
    ) -> str:
        """Build a message that includes the swarm context.

        Args:
            user_message: The current message.
            context: Current context variables.

        Returns:
            Enriched message string.
        """
        if not context:
            return user_message
        ctx_str = "\n".join(
            f"- {k}: {v}" for k, v in context.items()
            if not k.startswith("__")
        )
        return f"Context:\n{ctx_str}\n\nRequest: {user_message}"

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute the swarm with context handoffs (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent in the handoff chain.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        context = dict(self.context_variables)
        context.update(ctx.session.state.get("swarm_context", {}))
        current_name = self.initial_agent or self.sub_agents[0].name
        chain: list[str] = []
        last_output = ctx.user_message

        for step in range(self.max_handoffs):
            agent = self.find_sub_agent(current_name)
            if not agent:
                yield Event(
                    EventType.ERROR, self.name,
                    f"Agent '{current_name}' not found in swarm.",
                )
                break

            chain.append(current_name)
            message = self._build_message(last_output, context)
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content

            last_output = output or last_output
            next_name = ctx.session.state.pop(self.next_agent_key, "")
            if "__done__" in ctx.session.state:
                ctx.session.state.pop("__done__", None)
                break
            if not next_name:
                break

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Handoff: {current_name} → {next_name}",
                data={"from": current_name, "to": next_name, "step": step + 1},
            )
            current_name = next_name

        ctx.session.state_set("swarm_context", context)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "chain": chain, "handoffs": len(chain) - 1,
                "context": context,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute the swarm with context handoffs (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent in the handoff chain.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        context = dict(self.context_variables)
        context.update(ctx.session.state.get("swarm_context", {}))
        current_name = self.initial_agent or self.sub_agents[0].name
        chain: list[str] = []
        last_output = ctx.user_message

        for step in range(self.max_handoffs):
            agent = self.find_sub_agent(current_name)
            if not agent:
                yield Event(
                    EventType.ERROR, self.name,
                    f"Agent '{current_name}' not found in swarm.",
                )
                break

            chain.append(current_name)
            message = self._build_message(last_output, context)
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content

            last_output = output or last_output
            next_name = ctx.session.state.pop(self.next_agent_key, "")
            if "__done__" in ctx.session.state:
                ctx.session.state.pop("__done__", None)
                break
            if not next_name:
                break

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Handoff: {current_name} → {next_name}",
                data={"from": current_name, "to": next_name, "step": step + 1},
            )
            current_name = next_name

        ctx.session.state_set("swarm_context", context)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "chain": chain, "handoffs": len(chain) - 1,
                "context": context,
            })


# ── MemoryConsolidationAgent ────────────────────────────────────────────────


class MemoryConsolidationAgent(BaseAgent):
    """Consolidate session history when context grows too long.

    Wraps a ``main_agent`` and monitors session event count.
    When it exceeds ``event_threshold``, a ``summarizer_agent``
    compresses history into a summary stored in
    ``session.state[memory_key]``.  The main agent then receives
    only the summary + recent events.

    Args:
        name: Agent name.
        description: Human-readable description.
        main_agent: The agent doing the actual work.
        summarizer_agent: Agent that compresses history.
        event_threshold: Events before triggering consolidation.
        keep_recent: Recent events to keep after consolidation.
        memory_key: Key in ``session.state`` for the summary.
        result_key: Store consolidation stats in ``session.state``.

    Example:
        >>> mc = MemoryConsolidationAgent(
        ...     name="long_runner",
        ...     main_agent=assistant,
        ...     summarizer_agent=summarizer,
        ...     event_threshold=50,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        main_agent: BaseAgent,
        summarizer_agent: BaseAgent,
        event_threshold: int = 50,
        keep_recent: int = 10,
        memory_key: str = "consolidated_memory",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=[main_agent, summarizer_agent], **kwargs,
        )
        self.main_agent = main_agent
        self.summarizer_agent = summarizer_agent
        self.event_threshold = event_threshold
        self.keep_recent = keep_recent
        self.memory_key = memory_key
        self.result_key = result_key

    def _maybe_consolidate(
        self, ctx: InvocationContext,
    ) -> tuple[bool, str]:
        """Check if consolidation is needed and perform it (sync).

        Args:
            ctx: The invocation context.

        Returns:
            Tuple of (consolidation_happened, summary_text).
        """
        events = ctx.session.events
        if len(events) < self.event_threshold:
            return False, ""

        old_events = events[:-self.keep_recent] if self.keep_recent else events
        history = "\n".join(
            f"[{e.author}]: {e.content}"
            for e in old_events
            if e.content and e.event_type == EventType.AGENT_MESSAGE
        )

        existing = ctx.session.state.get(self.memory_key, "")
        prompt = (
            f"Previous summary:\n{existing}\n\n"
            f"New conversation to summarise:\n{history}\n\n"
            "Produce a concise summary preserving all key information."
            if existing else
            f"Summarise this conversation:\n{history}\n\n"
            "Be concise but preserve all key facts and decisions."
        )

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        summary = ""
        for event in self.summarizer_agent._run_impl_traced(sub_ctx):
            if event.event_type == EventType.AGENT_MESSAGE:
                summary = event.content

        ctx.session.state_set(self.memory_key, summary)
        return True, summary

    async def _maybe_consolidate_async(
        self, ctx: InvocationContext,
    ) -> tuple[bool, str]:
        """Check if consolidation is needed and perform it (async).

        Args:
            ctx: The invocation context.

        Returns:
            Tuple of (consolidation_happened, summary_text).
        """
        events = ctx.session.events
        if len(events) < self.event_threshold:
            return False, ""

        old_events = events[:-self.keep_recent] if self.keep_recent else events
        history = "\n".join(
            f"[{e.author}]: {e.content}"
            for e in old_events
            if e.content and e.event_type == EventType.AGENT_MESSAGE
        )

        existing = ctx.session.state.get(self.memory_key, "")
        prompt = (
            f"Previous summary:\n{existing}\n\n"
            f"New conversation to summarise:\n{history}\n\n"
            "Produce a concise summary preserving all key information."
            if existing else
            f"Summarise this conversation:\n{history}\n\n"
            "Be concise but preserve all key facts and decisions."
        )

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        summary = ""
        async for event in self.summarizer_agent._run_async_impl_traced(
            sub_ctx,
        ):
            if event.event_type == EventType.AGENT_MESSAGE:
                summary = event.content

        ctx.session.state_set(self.memory_key, summary)
        return True, summary

    def _build_enriched_message(self, ctx: InvocationContext) -> str:
        """Build a message with summary context if available.

        Args:
            ctx: The invocation context.

        Returns:
            Enriched message.
        """
        summary = ctx.session.state.get(self.memory_key, "")
        if not summary:
            return ctx.user_message
        return (
            f"Conversation summary:\n{summary}\n\n"
            f"Current request: {ctx.user_message}"
        )

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Consolidate if needed, then run the main agent (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from consolidation and main agent.
        """
        consolidated, summary = self._maybe_consolidate(ctx)
        if consolidated:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Consolidated {len(ctx.session.events)} events",
                data={"consolidated": True, "summary_len": len(summary)},
            )

        message = self._build_enriched_message(ctx)
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        for event in self.main_agent._run_impl_traced(sub_ctx):
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "consolidated": consolidated,
                "event_count": len(ctx.session.events),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Consolidate if needed, then run the main agent (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from consolidation and main agent.
        """
        consolidated, summary = await self._maybe_consolidate_async(ctx)
        if consolidated:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Consolidated {len(ctx.session.events)} events",
                data={"consolidated": True, "summary_len": len(summary)},
            )

        message = self._build_enriched_message(ctx)
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        async for event in self.main_agent._run_async_impl_traced(sub_ctx):
            yield event

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "consolidated": consolidated,
                "event_count": len(ctx.session.events),
            })


# ── PriorityQueueAgent ────────────────────────────────────────────────────────


class PriorityQueueAgent(BaseAgent):
    """Priority-queue orchestration: process sub-agents by priority order.

    Each sub-agent has an assigned priority (lower = higher priority).
    Agents are executed in priority order, with optional concurrency
    for agents sharing the same priority level.  A ``stop_condition``
    can halt processing early.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of agents with priorities.
        priority_map: ``{agent_name: priority}`` (lower = higher).
        max_workers: Max concurrent agents at the same priority level.
        stop_condition: Optional ``(session.state) -> bool`` to halt early.
        result_key: Store execution order in ``session.state``.

    Example:
        >>> pq = PriorityQueueAgent(
        ...     name="scheduler",
        ...     sub_agents=[urgent, normal, background],
        ...     priority_map={"urgent": 0, "normal": 1, "background": 2},
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        priority_map: dict[str, int] | None = None,
        max_workers: int = 4,
        stop_condition: Callable[[dict[str, Any]], bool] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.priority_map = priority_map or {}
        self.max_workers = max_workers
        self.stop_condition = stop_condition
        self.result_key = result_key

    def _get_priority(self, agent: BaseAgent) -> int:
        """Get priority for an agent (default 999).

        Args:
            agent: The agent.

        Returns:
            Priority value (lower = higher priority).
        """
        return self.priority_map.get(agent.name, 999)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute agents in priority order (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent in priority order.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        sorted_agents = sorted(self.sub_agents, key=self._get_priority)
        groups: dict[int, list[BaseAgent]] = {}
        for agent in sorted_agents:
            p = self._get_priority(agent)
            groups.setdefault(p, []).append(agent)

        execution_order: list[str] = []

        for priority in sorted(groups.keys()):
            if self.stop_condition and self.stop_condition(ctx.session.state):
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    "Stop condition met — halting queue",
                    data={"stopped_at_priority": priority},
                )
                break

            level_agents = groups[priority]

            if len(level_agents) == 1:
                agent = level_agents[0]
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=ctx.user_message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                for event in agent._run_impl_traced(sub_ctx):
                    yield event
                execution_order.append(agent.name)
            else:
                def _run_priority(a: BaseAgent) -> tuple[str, list[Event]]:
                    sub_ctx = InvocationContext(
                        session=ctx.session,
                        user_message=ctx.user_message,
                        parent_agent=self,
                        trace_collector=ctx.trace_collector,
                    )
                    events: list[Event] = []
                    for event in a._run_impl_traced(sub_ctx):
                        events.append(event)
                    return a.name, events

                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, _MAX_DEFAULT_PARALLEL_WORKERS),
                ) as executor:
                    futures = {
                        executor.submit(_run_priority, a): a
                        for a in level_agents
                    }
                    for future in as_completed(futures):
                        try:
                            name, events = future.result(timeout=_FUTURE_TIMEOUT)
                            execution_order.append(name)
                            for event in events:
                                yield event
                        except Exception as exc:
                            logger.error(
                                "[%s] Agent failed: %s", self.name, exc,
                            )

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Priority {priority}: {len(level_agents)} agents done",
                data={
                    "priority": priority,
                    "agents": [a.name for a in level_agents],
                },
            )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "execution_order": execution_order,
                "total": len(execution_order),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute agents in priority order (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent in priority order.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No sub-agents configured.")
            return

        sorted_agents = sorted(self.sub_agents, key=self._get_priority)
        groups: dict[int, list[BaseAgent]] = {}
        for agent in sorted_agents:
            p = self._get_priority(agent)
            groups.setdefault(p, []).append(agent)

        execution_order: list[str] = []

        for priority in sorted(groups.keys()):
            if self.stop_condition and self.stop_condition(ctx.session.state):
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    "Stop condition met — halting queue",
                    data={"stopped_at_priority": priority},
                )
                break

            level_agents = groups[priority]

            if len(level_agents) == 1:
                agent = level_agents[0]
                sub_ctx = InvocationContext(
                    session=ctx.session, user_message=ctx.user_message,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                async for event in agent._run_async_impl_traced(sub_ctx):
                    yield event
                execution_order.append(agent.name)
            else:
                async def _run_priority_async(
                    a: BaseAgent,
                ) -> tuple[str, list[Event]]:
                    sub_ctx = InvocationContext(
                        session=ctx.session,
                        user_message=ctx.user_message,
                        parent_agent=self,
                        trace_collector=ctx.trace_collector,
                    )
                    events: list[Event] = []
                    async for event in a._run_async_impl_traced(sub_ctx):
                        events.append(event)
                    return a.name, events

                results = await asyncio.gather(
                    *[_run_priority_async(a) for a in level_agents],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(
                            "[%s] Async agent failed: %s",
                            self.name, result,
                        )
                        continue
                    name, events = result
                    execution_order.append(name)
                    for event in events:
                        yield event

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Priority {priority}: {len(level_agents)} agents done",
                data={
                    "priority": priority,
                    "agents": [a.name for a in level_agents],
                },
            )

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "execution_order": execution_order,
                "total": len(execution_order),
            })


# ─────────────────────────────────────────────────────────────────────
# MonteCarloAgent
# ─────────────────────────────────────────────────────────────────────


class MonteCarloAgent(BaseAgent):
    """Monte Carlo Tree Search (MCTS) orchestration with UCT.

    Unlike ``TreeOfThoughtsAgent`` (BFS + prune), MCTS uses simulated
    rollouts and UCT (Upper Confidence bound for Trees) to balance
    exploration vs. exploitation.  Each node represents a thought
    generated by ``agent`` and is scored by ``evaluate_fn``.

    Tree growth is capped at ``_MAX_TREE_NODES`` nodes and node
    responses are truncated to ``_MAX_NODE_RESPONSE_LEN`` characters
    to bound memory usage.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The sub-agent that generates thoughts.
        evaluate_fn: ``(response: str) -> float`` scoring function.
        n_simulations: Total number of MCTS simulations/rollouts.
        max_depth: Maximum tree depth per rollout.
        exploration_weight: UCT exploration constant (default √2).
        result_key: Store best path in ``session.state``.

    Example:
        >>> mcts = MonteCarloAgent(
        ...     name="mcts",
        ...     agent=thinker,
        ...     evaluate_fn=lambda r: 1.0 if "correct" in r.lower() else 0.3,
        ...     n_simulations=20,
        ... )
    """

    _MAX_TREE_NODES: int = 5_000
    _MAX_NODE_RESPONSE_LEN: int = 4_000

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        evaluate_fn: Callable[[str], float] | None = None,
        n_simulations: int = 20,
        max_depth: int = 3,
        exploration_weight: float | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        sub = [agent] if agent else []
        super().__init__(
            name=name, description=description,
            sub_agents=sub, **kwargs,
        )
        self.agent = agent
        self.evaluate_fn = evaluate_fn or (lambda _r: 0.5)
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.exploration_weight = (
            exploration_weight if exploration_weight is not None
            else math.sqrt(2)
        )
        self.result_key = result_key

    # -- internal node -------------------------------------------------

    class _Node:
        """Lightweight MCTS tree node."""

        __slots__ = (
            "response", "score", "visits", "total_value",
            "children", "parent", "depth",
        )

        def __init__(
            self, response: str = "", score: float = 0.0,
            parent: "MonteCarloAgent._Node | None" = None,
            depth: int = 0,
        ) -> None:
            self.response = response
            self.score = score
            self.visits = 0
            self.total_value = 0.0
            self.children: list["MonteCarloAgent._Node"] = []
            self.parent = parent
            self.depth = depth

    def _uct(self, node: _Node, parent_visits: int) -> float:
        """Upper Confidence bound for Trees."""
        if node.visits == 0:
            return float("inf")
        exploit = node.total_value / node.visits
        explore = self.exploration_weight * math.sqrt(
            math.log(parent_visits) / node.visits,
        )
        return exploit + explore

    def _select(self, node: _Node) -> _Node:
        """Select a leaf node using UCT."""
        current = node
        while current.children and current.depth < self.max_depth:
            current = max(
                current.children,
                key=lambda c: self._uct(c, current.visits),
            )
        return current

    @staticmethod
    def _backpropagate(node: _Node, value: float) -> None:
        """Backpropagate the rollout value up the tree."""
        current: MonteCarloAgent._Node | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _best_path(self, root: _Node) -> list[str]:
        """Extract the best path (by average value) from root to leaf."""
        path: list[str] = []
        current = root
        while current.children:
            current = max(
                current.children,
                key=lambda c: (c.total_value / c.visits) if c.visits else 0.0,
            )
            path.append(current.response)
        return path

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run MCTS simulations (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the search process.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        root = self._Node(response="", depth=0)
        _node_count = 1

        for sim in range(self.n_simulations):
            # Selection
            leaf = self._select(root)

            # Cap: skip expansion if tree is too large
            if _node_count >= self._MAX_TREE_NODES:
                logger.debug(
                    "[%s] Tree node cap (%d) reached, skipping expansion.",
                    self.name, self._MAX_TREE_NODES,
                )
                continue

            # Expansion — generate a new child thought
            sub_ctx = InvocationContext(
                session=Session(),
                user_message=(
                    leaf.response or ctx.user_message
                ),
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in self.agent._run_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            score = self.evaluate_fn(output)
            child = self._Node(
                response=output[:self._MAX_NODE_RESPONSE_LEN],
                score=score,
                parent=leaf, depth=leaf.depth + 1,
            )
            leaf.children.append(child)
            _node_count += 1

            # Backpropagation
            self._backpropagate(child, score)

            logger.debug(
                "[%s] Simulation %d: depth=%d score=%.2f",
                self.name, sim + 1, child.depth, score,
            )

        best_path = self._best_path(root)
        best_response = best_path[-1] if best_path else ""
        best_node = root
        while best_node.children:
            best_node = max(
                best_node.children,
                key=lambda c: (c.total_value / c.visits) if c.visits else 0.0,
            )
        best_score = (
            (best_node.total_value / best_node.visits)
            if best_node.visits else 0.0
        )

        yield Event(EventType.AGENT_MESSAGE, self.name, best_response)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"MCTS done: {self.n_simulations} simulations, best_score={best_score:.2f}",
            data={
                "simulations": self.n_simulations,
                "best_score": best_score,
                "path_length": len(best_path),
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_response": best_response,
                "best_score": best_score,
                "best_path": best_path,
                "simulations": self.n_simulations,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run MCTS simulations (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the search process.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        root = self._Node(response="", depth=0)
        _node_count = 1

        for sim in range(self.n_simulations):
            leaf = self._select(root)

            # Cap: skip expansion if tree is too large
            if _node_count >= self._MAX_TREE_NODES:
                logger.debug(
                    "[%s] Tree node cap (%d) reached, skipping expansion.",
                    self.name, self._MAX_TREE_NODES,
                )
                continue

            sub_ctx = InvocationContext(
                session=Session(),
                user_message=(
                    leaf.response or ctx.user_message
                ),
                parent_agent=self,
                trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in self.agent._run_async_impl_traced(sub_ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            score = self.evaluate_fn(output)
            child = self._Node(
                response=output[:self._MAX_NODE_RESPONSE_LEN],
                score=score,
                parent=leaf, depth=leaf.depth + 1,
            )
            leaf.children.append(child)
            _node_count += 1
            self._backpropagate(child, score)

        best_path = self._best_path(root)
        best_response = best_path[-1] if best_path else ""
        best_node = root
        while best_node.children:
            best_node = max(
                best_node.children,
                key=lambda c: (c.total_value / c.visits) if c.visits else 0.0,
            )
        best_score = (
            (best_node.total_value / best_node.visits)
            if best_node.visits else 0.0
        )

        yield Event(EventType.AGENT_MESSAGE, self.name, best_response)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"MCTS done: {self.n_simulations} simulations",
            data={
                "simulations": self.n_simulations,
                "best_score": best_score,
                "path_length": len(best_path),
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_response": best_response,
                "best_score": best_score,
                "best_path": best_path,
                "simulations": self.n_simulations,
            })


# ─────────────────────────────────────────────────────────────────────
# GraphOfThoughtsAgent
# ─────────────────────────────────────────────────────────────────────


class GraphOfThoughtsAgent(BaseAgent):
    """Graph-of-Thoughts: DAG orchestration with generate, aggregate, refine.

    Unlike ``TreeOfThoughtsAgent`` (tree-shaped BFS), GoT allows
    thoughts to **merge** and **refine** in a directed acyclic graph.

    The user provides ``operations``: an ordered list of dicts, each with:
    - ``type``: ``"generate"`` | ``"aggregate"`` | ``"score"``
    - ``agent``: The sub-agent to use (or ``None`` for score).
    - ``k``: Number of branches for generate (default 3).
    - ``source_ids``: For aggregate — list of thought IDs to merge.

    Simpler usage: just supply ``agent``, ``aggregate_agent``, and
    ``score_fn``, and the agent auto-builds a simple generate → aggregate
    → score pipeline.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generation sub-agent.
        aggregate_agent: Agent for merging multiple thoughts.
        score_fn: ``(response: str) -> float`` scoring function.
        n_branches: Branches per generation step.
        n_rounds: Number of generate → aggregate → score rounds.
        result_key: Store best thought in ``session.state``.

    Example:
        >>> got = GraphOfThoughtsAgent(
        ...     name="got",
        ...     agent=thinker,
        ...     aggregate_agent=merger,
        ...     score_fn=lambda r: 1.0 if "correct" in r else 0.3,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        aggregate_agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_branches: int = 3,
        n_rounds: int = 2,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [agent, aggregate_agent] if a]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.agent = agent
        self.aggregate_agent = aggregate_agent
        self.score_fn = score_fn or (lambda _r: 0.5)
        self.n_branches = n_branches
        self.n_rounds = n_rounds
        self.result_key = result_key

    def _run_agent(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> str:
        """Run a sub-agent sync and return its last message."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output

    async def _run_agent_async(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> str:
        """Run a sub-agent async and return its last message."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run GoT pipeline (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the GoT process.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        thoughts: list[str] = []
        best_thought = ""
        best_score = 0.0

        for round_idx in range(self.n_rounds):
            # Generate phase
            new_thoughts: list[str] = []
            seed = (
                best_thought if best_thought
                else ctx.user_message
            )
            for _b in range(self.n_branches):
                t = self._run_agent(self.agent, ctx, seed)
                new_thoughts.append(t)

            # Aggregate phase
            if self.aggregate_agent and len(new_thoughts) > 1:
                combined = "\n---\n".join(
                    f"Thought {i+1}: {_truncate(t)}"
                    for i, t in enumerate(new_thoughts)
                )
                aggregated = self._run_agent(
                    self.aggregate_agent, ctx,
                    f"Merge these thoughts into one:\n{combined}",
                )
                new_thoughts.append(aggregated)

            # Score phase
            for t in new_thoughts:
                s = self.score_fn(t)
                if s > best_score:
                    best_score = s
                    best_thought = t

            thoughts.extend(_truncate(t) for t in new_thoughts)
            thoughts = thoughts[-20:]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"GoT round {round_idx + 1}: best_score={best_score:.2f}",
                data={
                    "round": round_idx + 1,
                    "best_score": best_score,
                    "n_thoughts": len(thoughts),
                },
            )

        yield Event(EventType.AGENT_MESSAGE, self.name, best_thought)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_thought": best_thought,
                "best_score": best_score,
                "total_thoughts": len(thoughts),
                "rounds": self.n_rounds,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run GoT pipeline (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the GoT process.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        thoughts: list[str] = []
        best_thought = ""
        best_score = 0.0

        for round_idx in range(self.n_rounds):
            new_thoughts: list[str] = []
            seed = best_thought if best_thought else ctx.user_message
            tasks = [
                self._run_agent_async(self.agent, ctx, seed)
                for _ in range(self.n_branches)
            ]
            raw_thoughts = await asyncio.gather(*tasks, return_exceptions=True)
            new_thoughts = [r for r in raw_thoughts if not isinstance(r, BaseException)]

            if self.aggregate_agent and len(new_thoughts) > 1:
                combined = "\n---\n".join(
                    f"Thought {i+1}: {_truncate(t)}"
                    for i, t in enumerate(new_thoughts)
                )
                aggregated = await self._run_agent_async(
                    self.aggregate_agent, ctx,
                    f"Merge these thoughts into one:\n{combined}",
                )
                new_thoughts.append(aggregated)

            for t in new_thoughts:
                s = self.score_fn(t)
                if s > best_score:
                    best_score = s
                    best_thought = t

            thoughts.extend(_truncate(t) for t in new_thoughts)
            thoughts = thoughts[-20:]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"GoT round {round_idx + 1}: best_score={best_score:.2f}",
                data={
                    "round": round_idx + 1,
                    "best_score": best_score,
                    "n_thoughts": len(thoughts),
                },
            )

        yield Event(EventType.AGENT_MESSAGE, self.name, best_thought)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_thought": best_thought,
                "best_score": best_score,
                "total_thoughts": len(thoughts),
                "rounds": self.n_rounds,
            })


# ─────────────────────────────────────────────────────────────────────
# BlackboardAgent
# ─────────────────────────────────────────────────────────────────────


class BlackboardAgent(BaseAgent):
    """Blackboard architecture: shared board with expert activation.

    Experts publish partial solutions to a shared ``board`` dict.
    A ``controller_fn`` selects the most relevant expert each
    iteration based on the current board state.  Stops when a
    ``termination_fn`` fires or ``max_iterations`` is reached.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Expert agents.
        controller_fn: ``(board: dict, agents: list) -> BaseAgent``
            picks the next expert.  Default: round-robin.
        termination_fn: ``(board: dict) -> bool`` whether to stop.
        max_iterations: Safety limit on iterations.
        board_key: Key in ``session.state`` where board is stored.
        result_key: Store final board in ``session.state``.

    Example:
        >>> bb = BlackboardAgent(
        ...     name="diagnosis",
        ...     sub_agents=[symptom_agent, lab_agent, radiology_agent],
        ...     controller_fn=lambda board, agents: agents[0],
        ...     termination_fn=lambda board: board.get("diagnosis") is not None,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        controller_fn: Callable[
            [dict[str, Any], list[BaseAgent]], BaseAgent
        ] | None = None,
        termination_fn: Callable[[dict[str, Any]], bool] | None = None,
        max_iterations: int = 10,
        board_key: str = "blackboard",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self._rr_idx = 0
        self._controller_fn = controller_fn or self._round_robin
        self.termination_fn = termination_fn or (lambda _b: False)
        self.max_iterations = max_iterations
        self.board_key = board_key
        self.result_key = result_key

    def _round_robin(
        self, _board: dict[str, Any], agents: list[BaseAgent],
    ) -> BaseAgent:
        """Default controller: round-robin."""
        agent = agents[self._rr_idx % len(agents)]
        self._rr_idx += 1
        return agent

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Blackboard loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each expert activation.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No experts configured.")
            return

        board: dict[str, Any] = ctx.session.state.get(self.board_key, {})
        ctx.session.state_set(self.board_key, board)

        last_output = ""
        for iteration in range(self.max_iterations):
            if self.termination_fn(board):
                break

            expert = self._controller_fn(board, list(self.sub_agents))
            board_summary = _truncate(
                json.dumps(board, default=str, ensure_ascii=False), 8000,
            )
            message = (
                f"{ctx.user_message}\n\n"
                f"Current board state:\n{board_summary}"
            )
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in expert._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output

            board[f"iteration_{iteration}"] = {
                "expert": expert.name,
                "output": _truncate(output),
            }

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Blackboard iteration {iteration + 1}: expert={expert.name}",
                data={"iteration": iteration + 1, "expert": expert.name},
            )

        if not last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, "")

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "board": board,
                "iterations": min(
                    iteration + 1  # type: ignore[possibly-undefined]
                    if self.sub_agents else 0,
                    self.max_iterations,
                ),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Blackboard loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each expert activation.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No experts configured.")
            return

        board: dict[str, Any] = ctx.session.state.get(self.board_key, {})
        ctx.session.state_set(self.board_key, board)

        last_output = ""
        for iteration in range(self.max_iterations):
            if self.termination_fn(board):
                break

            expert = self._controller_fn(board, list(self.sub_agents))
            board_summary = _truncate(
                json.dumps(board, default=str, ensure_ascii=False), 8000,
            )
            message = (
                f"{ctx.user_message}\n\n"
                f"Current board state:\n{board_summary}"
            )
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in expert._run_async_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output

            board[f"iteration_{iteration}"] = {
                "expert": expert.name,
                "output": _truncate(output),
            }

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Blackboard iteration {iteration + 1}: expert={expert.name}",
                data={"iteration": iteration + 1, "expert": expert.name},
            )

        if not last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, "")

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "board": board,
                "iterations": min(
                    iteration + 1  # type: ignore[possibly-undefined]
                    if self.sub_agents else 0,
                    self.max_iterations,
                ),
            })


# ─────────────────────────────────────────────────────────────────────
# MixtureOfExpertsAgent
# ─────────────────────────────────────────────────────────────────────


class MixtureOfExpertsAgent(BaseAgent):
    """Mixture-of-Experts (MoE): gating + weighted expert combination.

    A ``gating_fn`` assigns weights to each expert.  The top-k experts
    are executed and their outputs are combined using ``combine_fn``
    (default: weighted concatenation).

    Unlike ``RouterAgent`` (picks one) or ``VotingAgent`` (majority),
    MoE executes multiple experts and blends their outputs.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Expert agents.
        gating_fn: ``(message: str, agents: list) -> dict[str, float]``
            maps agent names to weights.  Default: uniform.
        top_k: Number of top experts to activate.
        combine_fn: ``(outputs: list[tuple[str, str, float]]) -> str``
            combines ``(agent_name, output, weight)`` tuples.
        result_key: Store expert weights in ``session.state``.

    Example:
        >>> moe = MixtureOfExpertsAgent(
        ...     name="moe",
        ...     sub_agents=[math_expert, code_expert, writing_expert],
        ...     gating_fn=lambda msg, agents: {"math": 0.8, "code": 0.1, "writing": 0.1},
        ...     top_k=2,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        gating_fn: Callable[
            [str, list[BaseAgent]], dict[str, float]
        ] | None = None,
        top_k: int = 2,
        combine_fn: Callable[
            [list[tuple[str, str, float]]], str
        ] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.gating_fn = gating_fn or self._uniform_gate
        self.top_k = top_k
        self.combine_fn = combine_fn or self._default_combine
        self.result_key = result_key

    @staticmethod
    def _uniform_gate(
        _msg: str, agents: list[BaseAgent],
    ) -> dict[str, float]:
        """Uniform gating: equal weight for all agents."""
        w = 1.0 / len(agents) if agents else 0.0
        return {a.name: w for a in agents}

    @staticmethod
    def _default_combine(
        outputs: list[tuple[str, str, float]],
    ) -> str:
        """Weighted concatenation of expert outputs."""
        parts: list[str] = []
        for agent_name, output, weight in outputs:
            parts.append(
                f"[{agent_name} (weight={weight:.2f})]\n{_truncate(output)}"
            )
        return "\n\n".join(parts)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute top-k experts and combine (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from experts and the combined output.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No experts configured.")
            return

        weights = self.gating_fn(ctx.user_message, list(self.sub_agents))
        sorted_experts = sorted(
            self.sub_agents,
            key=lambda a: weights.get(a.name, 0.0),
            reverse=True,
        )
        selected = sorted_experts[: self.top_k]

        expert_outputs: list[tuple[str, str, float]] = []
        for expert in selected:
            w = weights.get(expert.name, 0.0)
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in expert._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            expert_outputs.append((expert.name, output, w))

        combined = self.combine_fn(expert_outputs)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "weights": weights,
                "selected": [e[0] for e in expert_outputs],
                "top_k": self.top_k,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute top-k experts and combine (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from experts and the combined output.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No experts configured.")
            return

        weights = self.gating_fn(ctx.user_message, list(self.sub_agents))
        sorted_experts = sorted(
            self.sub_agents,
            key=lambda a: weights.get(a.name, 0.0),
            reverse=True,
        )
        selected = sorted_experts[: self.top_k]

        async def _run_one(
            agent: BaseAgent,
        ) -> tuple[str, list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return agent.name, events, output

        results = await asyncio.gather(
            *[_run_one(a) for a in selected],
            return_exceptions=True,
        )

        expert_outputs: list[tuple[str, str, float]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("[%s] Expert failed: %s", self.name, result)
                continue
            aname, events, output = result
            for event in events:
                yield event
            w = weights.get(aname, 0.0)
            expert_outputs.append((aname, output, w))

        combined = self.combine_fn(expert_outputs)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "weights": weights,
                "selected": [e[0] for e in expert_outputs],
                "top_k": self.top_k,
            })


# ─────────────────────────────────────────────────────────────────────
# CoVeAgent (Chain-of-Verification)
# ─────────────────────────────────────────────────────────────────────


class CoVeAgent(BaseAgent):
    """Chain-of-Verification: 4-phase anti-hallucination pipeline.

    1. **Draft** — ``drafter`` generates an initial response.
    2. **Plan** — ``planner`` produces verification questions.
    3. **Verify** — ``verifier`` answers each question independently.
    4. **Revise** — ``reviser`` produces the final, verified response.

    Each phase uses a dedicated sub-agent so they can be different
    models or have different instructions.

    Args:
        name: Agent name.
        description: Human-readable description.
        drafter: Agent that generates the initial draft.
        planner: Agent that plans verification questions.
        verifier: Agent that answers verification questions.
        reviser: Agent that produces the final verified response.
        max_questions: Maximum verification questions.
        result_key: Store verification details in ``session.state``.

    Example:
        >>> cove = CoVeAgent(
        ...     name="verifier",
        ...     drafter=draft_agent,
        ...     planner=plan_agent,
        ...     verifier=verify_agent,
        ...     reviser=revise_agent,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        drafter: BaseAgent | None = None,
        planner: BaseAgent | None = None,
        verifier: BaseAgent | None = None,
        reviser: BaseAgent | None = None,
        max_questions: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [drafter, planner, verifier, reviser] if a]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.drafter = drafter
        self.planner = planner
        self.verifier = verifier
        self.reviser = reviser
        self.max_questions = max_questions
        self.result_key = result_key

    def _run_sub(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a sub-agent sync, return (output, events)."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    async def _run_sub_async(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a sub-agent async, return (output, events)."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run the 4-phase CoVe pipeline (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each phase.
        """
        if not all([self.drafter, self.planner, self.verifier, self.reviser]):
            yield Event(
                EventType.ERROR, self.name,
                "CoVe requires drafter, planner, verifier, and reviser.",
            )
            return

        # Phase 1: Draft
        draft, draft_events = self._run_sub(
            self.drafter, ctx, ctx.user_message,  # type: ignore[arg-type]
        )
        for e in draft_events:
            yield e
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Draft complete ({len(draft)} chars)",
            data={"phase": "draft"},
        )

        # Phase 2: Plan verification questions
        plan_prompt = (
            f"Original question: {ctx.user_message}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Generate up to {self.max_questions} verification questions "
            f"to fact-check the draft. One question per line."
        )
        plan_output, plan_events = self._run_sub(
            self.planner, ctx, plan_prompt,  # type: ignore[arg-type]
        )
        for e in plan_events:
            yield e
        questions = [
            q.strip() for q in plan_output.strip().splitlines()
            if q.strip()
        ][: self.max_questions]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Planned {len(questions)} verification questions",
            data={"phase": "plan", "n_questions": len(questions)},
        )

        # Phase 3: Verify each question independently
        verifications: list[dict[str, str]] = []
        for q in questions:
            answer, v_events = self._run_sub(
                self.verifier, ctx, q,  # type: ignore[arg-type]
            )
            for e in v_events:
                yield e
            verifications.append({"question": q, "answer": answer})
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Verified {len(verifications)} questions",
            data={"phase": "verify"},
        )

        # Phase 4: Revise with verification results
        verification_text = "\n".join(
            f"Q: {v['question']}\nA: {v['answer']}"
            for v in verifications
        )
        revise_prompt = (
            f"Original question: {ctx.user_message}\n\n"
            f"Draft answer:\n{_truncate(draft)}\n\n"
            f"Verification results:\n{_truncate(verification_text)}\n\n"
            f"Produce a final, verified answer."
        )
        final, revise_events = self._run_sub(
            self.reviser, ctx, revise_prompt,  # type: ignore[arg-type]
        )
        for e in revise_events:
            yield e

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "draft": _truncate(draft),
                "questions": questions[-10:],
                "verifications": [{"question": v["question"][:200], "answer": _truncate(v["answer"])} for v in verifications[-10:]],
                "final": _truncate(final),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the 4-phase CoVe pipeline (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each phase.
        """
        if not all([self.drafter, self.planner, self.verifier, self.reviser]):
            yield Event(
                EventType.ERROR, self.name,
                "CoVe requires drafter, planner, verifier, and reviser.",
            )
            return

        # Phase 1
        draft, draft_events = await self._run_sub_async(
            self.drafter, ctx, ctx.user_message,  # type: ignore[arg-type]
        )
        for e in draft_events:
            yield e

        # Phase 2
        plan_prompt = (
            f"Original question: {ctx.user_message}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Generate up to {self.max_questions} verification questions "
            f"to fact-check the draft. One question per line."
        )
        plan_output, plan_events = await self._run_sub_async(
            self.planner, ctx, plan_prompt,  # type: ignore[arg-type]
        )
        for e in plan_events:
            yield e
        questions = [
            q.strip() for q in plan_output.strip().splitlines()
            if q.strip()
        ][: self.max_questions]

        # Phase 3 — verify in parallel
        async def _verify(q: str) -> dict[str, str]:
            answer, _ = await self._run_sub_async(
                self.verifier, ctx, q,  # type: ignore[arg-type]
            )
            return {"question": q, "answer": answer}

        verifications_raw = await asyncio.gather(
            *[_verify(q) for q in questions],
            return_exceptions=True,
        )
        verifications = [
            v for v in verifications_raw if not isinstance(v, BaseException)
        ]

        # Phase 4
        verification_text = "\n".join(
            f"Q: {v['question']}\nA: {v['answer']}"
            for v in verifications
        )
        revise_prompt = (
            f"Original question: {ctx.user_message}\n\n"
            f"Draft answer:\n{_truncate(draft)}\n\n"
            f"Verification results:\n{_truncate(verification_text)}\n\n"
            f"Produce a final, verified answer."
        )
        final, revise_events = await self._run_sub_async(
            self.reviser, ctx, revise_prompt,  # type: ignore[arg-type]
        )
        for e in revise_events:
            yield e

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "draft": _truncate(draft),
                "questions": questions[-10:],
                "verifications": [{"question": v["question"][:200], "answer": _truncate(v["answer"])} for v in verifications[-10:]],
                "final": _truncate(final),
            })


# ─────────────────────────────────────────────────────────────────────
# SagaAgent
# ─────────────────────────────────────────────────────────────────────


class SagaAgent(BaseAgent):
    """Saga: distributed transactions with compensating rollback.

    Each step has an ``action`` agent and an optional ``compensate``
    agent.  If step N fails (detected by ``failure_detector``), the
    compensators for steps N-1 … 0 are executed in reverse order.

    Args:
        name: Agent name.
        description: Human-readable description.
        steps: Ordered list of ``{"action": Agent, "compensate": Agent | None}``.
        failure_detector: ``(output: str) -> bool`` — returns ``True``
            if the step is considered failed.
        result_key: Store saga result in ``session.state``.

    Example:
        >>> saga = SagaAgent(
        ...     name="order",
        ...     steps=[
        ...         {"action": reserve_stock, "compensate": release_stock},
        ...         {"action": charge_payment, "compensate": refund_payment},
        ...         {"action": ship_order},
        ...     ],
        ...     failure_detector=lambda o: "ERROR" in o.upper(),
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        steps: list[dict[str, Any]] | None = None,
        failure_detector: Callable[[str], bool] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        all_agents = []
        for step in (steps or []):
            all_agents.append(step["action"])
            if step.get("compensate"):
                all_agents.append(step["compensate"])
        super().__init__(
            name=name, description=description,
            sub_agents=all_agents, **kwargs,
        )
        self.steps = steps or []
        self.failure_detector = failure_detector or (
            lambda o: not o or not o.strip()
        )
        self.result_key = result_key

    def _run_sub_agent(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a sub-agent sync."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    async def _run_sub_agent_async(
        self, agent: BaseAgent, ctx: InvocationContext, message: str,
    ) -> tuple[str, list[Event]]:
        """Run a sub-agent async."""
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            events.append(event)
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content
        return output, events

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Execute saga steps with compensating rollback (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from action and compensation agents.
        """
        if not self.steps:
            yield Event(EventType.ERROR, self.name, "No saga steps configured.")
            return

        completed: list[int] = []
        step_outputs: list[str] = []
        failed_step: int | None = None
        last_output = ctx.user_message

        for idx, step in enumerate(self.steps):
            action = step["action"]
            output, events = self._run_sub_agent(action, ctx, last_output)
            for e in events:
                yield e

            if self.failure_detector(output):
                failed_step = idx
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Saga step {idx} ({action.name}) FAILED — rolling back",
                    data={"step": idx, "agent": action.name, "status": "failed"},
                )
                break

            completed.append(idx)
            step_outputs.append(output)
            last_output = output
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Saga step {idx} ({action.name}) completed",
                data={"step": idx, "agent": action.name, "status": "ok"},
            )

        # Compensating rollback
        compensated: list[int] = []
        if failed_step is not None:
            for idx in reversed(completed):
                compensator = self.steps[idx].get("compensate")
                if compensator:
                    comp_msg = (
                        f"Compensate step {idx}: "
                        f"original output was: {_truncate(step_outputs[completed.index(idx)])}"
                    )
                    c_output, c_events = self._run_sub_agent(
                        compensator, ctx, comp_msg,
                    )
                    for e in c_events:
                        yield e
                    compensated.append(idx)
                    yield Event(
                        EventType.STATE_UPDATE, self.name,
                        f"Compensated step {idx} ({compensator.name})",
                        data={"step": idx, "agent": compensator.name},
                    )

        final_output = step_outputs[-1] if step_outputs else ""
        yield Event(EventType.AGENT_MESSAGE, self.name, final_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "completed": completed,
                "failed_step": failed_step,
                "compensated": compensated,
                "success": failed_step is None,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Execute saga steps with compensating rollback (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from action and compensation agents.
        """
        if not self.steps:
            yield Event(EventType.ERROR, self.name, "No saga steps configured.")
            return

        completed: list[int] = []
        step_outputs: list[str] = []
        failed_step: int | None = None
        last_output = ctx.user_message

        for idx, step in enumerate(self.steps):
            action = step["action"]
            output, events = await self._run_sub_agent_async(
                action, ctx, last_output,
            )
            for e in events:
                yield e

            if self.failure_detector(output):
                failed_step = idx
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Saga step {idx} ({action.name}) FAILED — rolling back",
                    data={"step": idx, "agent": action.name, "status": "failed"},
                )
                break

            completed.append(idx)
            step_outputs.append(output)
            last_output = output

        compensated: list[int] = []
        if failed_step is not None:
            for idx in reversed(completed):
                compensator = self.steps[idx].get("compensate")
                if compensator:
                    comp_msg = (
                        f"Compensate step {idx}: "
                        f"original output was: {_truncate(step_outputs[completed.index(idx)])}"
                    )
                    c_output, c_events = await self._run_sub_agent_async(
                        compensator, ctx, comp_msg,
                    )
                    for e in c_events:
                        yield e
                    compensated.append(idx)

        final_output = step_outputs[-1] if step_outputs else ""
        yield Event(EventType.AGENT_MESSAGE, self.name, final_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "completed": completed,
                "failed_step": failed_step,
                "compensated": compensated,
                "success": failed_step is None,
            })


# ─────────────────────────────────────────────────────────────────────
# LoadBalancerAgent
# ─────────────────────────────────────────────────────────────────────


class LoadBalancerAgent(BaseAgent):
    """Load balancer: distribute requests across equivalent agents.

    Strategies: ``round_robin`` (default), ``random``, ``least_used``,
    or a custom callable.

    Unlike ``ParallelAgent`` (runs all), LoadBalancer picks **one**
    agent per request to distribute load.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of equivalent agents.
        strategy: ``"round_robin"`` | ``"random"`` | ``"least_used"``
            or ``Callable[[list[BaseAgent], dict], BaseAgent]``.
        result_key: Store routing info in ``session.state``.

    Example:
        >>> lb = LoadBalancerAgent(
        ...     name="lb",
        ...     sub_agents=[gpt_agent, gemini_agent, claude_agent],
        ...     strategy="round_robin",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        strategy: str | Callable[..., BaseAgent] = "round_robin",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.strategy = strategy
        self._rr_idx = 0
        # Bounded by len(sub_agents) — one entry per registered agent.
        self._usage: dict[str, int] = {}
        self.result_key = result_key

    def _pick(self) -> BaseAgent:
        """Select the next agent based on strategy."""
        agents = list(self.sub_agents or [])
        if not agents:
            raise ValueError("No agents in pool")

        if callable(self.strategy) and not isinstance(self.strategy, str):
            return self.strategy(agents, self._usage)

        if self.strategy == "random":
            return random.choice(agents)
        if self.strategy == "least_used":
            return min(agents, key=lambda a: self._usage.get(a.name, 0))
        # round_robin (default)
        agent = agents[self._rr_idx % len(agents)]
        self._rr_idx += 1
        return agent

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Route to one agent (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the selected agent.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents in pool.")
            return

        agent = self._pick()
        self._usage[agent.name] = self._usage.get(agent.name, 0) + 1

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        for event in agent._run_impl_traced(sub_ctx):
            yield event

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Routed to {agent.name} (strategy={self.strategy})",
            data={
                "selected": agent.name,
                "strategy": (
                    self.strategy if isinstance(self.strategy, str)
                    else "custom"
                ),
                "usage": dict(self._usage),
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": agent.name,
                "usage": dict(self._usage),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Route to one agent (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the selected agent.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents in pool.")
            return

        agent = self._pick()
        self._usage[agent.name] = self._usage.get(agent.name, 0) + 1

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        async for event in agent._run_async_impl_traced(sub_ctx):
            yield event

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Routed to {agent.name}",
            data={
                "selected": agent.name,
                "strategy": (
                    self.strategy if isinstance(self.strategy, str)
                    else "custom"
                ),
                "usage": dict(self._usage),
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": agent.name,
                "usage": dict(self._usage),
            })


# ─────────────────────────────────────────────────────────────────────
# EnsembleAgent
# ─────────────────────────────────────────────────────────────────────


class EnsembleAgent(BaseAgent):
    """Ensemble: aggregate outputs from multiple agents.

    Unlike ``VotingAgent`` (picks majority) or ``ConsensusAgent``
    (judge synthesises), EnsembleAgent runs all sub-agents and
    combines their outputs via a configurable ``aggregate_fn``.

    Built-in strategies:
    - ``"concat"`` — simple concatenation.
    - ``"weighted"`` — weighted by ``weights`` dict.
    - A custom callable ``(outputs: list[tuple[str, str]]) -> str``.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ensemble members.
        aggregate_fn: Strategy or callable.
        weights: ``{agent_name: float}`` for weighted mode.
        max_workers: Thread pool size for sync parallel execution.
        result_key: Store individual outputs in ``session.state``.

    Example:
        >>> ens = EnsembleAgent(
        ...     name="ensemble",
        ...     sub_agents=[model_a, model_b, model_c],
        ...     aggregate_fn="concat",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        aggregate_fn: str | Callable[..., str] = "concat",
        weights: dict[str, float] | None = None,
        max_workers: int | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.aggregate_fn = aggregate_fn
        self.weights = weights or {}
        self.max_workers = max_workers
        self.result_key = result_key

    def _aggregate(
        self, outputs: list[tuple[str, str]],
    ) -> str:
        """Aggregate outputs using the configured strategy.

        Args:
            outputs: List of ``(agent_name, response)`` tuples.

        Returns:
            Combined string.
        """
        if callable(self.aggregate_fn) and not isinstance(
            self.aggregate_fn, str,
        ):
            return self.aggregate_fn(outputs)

        if self.aggregate_fn == "weighted":
            parts: list[str] = []
            for agent_name, output in outputs:
                w = self.weights.get(agent_name, 1.0)
                parts.append(
                    f"[{agent_name} w={w:.2f}]\n{_truncate(output)}"
                )
            return "\n\n".join(parts)

        # "concat" (default)
        return "\n\n".join(
            f"[{aname}]\n{_truncate(output)}" for aname, output in outputs
        )

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run all agents and aggregate (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent plus the aggregated output.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        def _run_one(
            agent: BaseAgent,
        ) -> tuple[str, list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return agent.name, events, output

        outputs: list[tuple[str, str]] = []
        workers = self.max_workers or min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, a): a for a in self.sub_agents}
            for future in as_completed(futures):
                try:
                    aname, events, output = future.result(timeout=_FUTURE_TIMEOUT)
                    for event in events:
                        yield event
                    outputs.append((aname, output))
                except Exception as exc:
                    logger.error("[%s] Agent failed: %s", self.name, exc)

        combined = self._aggregate(outputs)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "outputs": {aname: out for aname, out in outputs},
                "count": len(outputs),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run all agents and aggregate (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each agent plus the aggregated output.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        async def _run_one(
            agent: BaseAgent,
        ) -> tuple[str, list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return agent.name, events, output

        results = await asyncio.gather(
            *[_run_one(a) for a in self.sub_agents],
            return_exceptions=True,
        )

        outputs: list[tuple[str, str]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("[%s] Agent failed: %s", self.name, result)
                continue
            aname, events, output = result
            for event in events:
                yield event
            outputs.append((aname, output))

        combined = self._aggregate(outputs)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "outputs": {aname: out for aname, out in outputs},
                "count": len(outputs),
            })


# ─────────────────────────────────────────────────────────────────────
# TimeoutAgent
# ─────────────────────────────────────────────────────────────────────


class TimeoutAgent(BaseAgent):
    """Timeout wrapper: enforces a deadline on any sub-agent.

    If the agent exceeds ``timeout_seconds``, the run is abandoned
    and ``fallback_message`` is yielded instead.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The sub-agent to wrap.
        timeout_seconds: Maximum execution time.
        fallback_message: Returned when timeout fires.
        result_key: Store timing info in ``session.state``.

    Example:
        >>> guarded = TimeoutAgent(
        ...     name="guarded",
        ...     agent=slow_agent,
        ...     timeout_seconds=10,
        ...     fallback_message="Request timed out.",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        timeout_seconds: float = 30.0,
        fallback_message: str = "Operation timed out.",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        sub = [agent] if agent else []
        super().__init__(
            name=name, description=description,
            sub_agents=sub, **kwargs,
        )
        self.agent = agent
        self.timeout_seconds = timeout_seconds
        self.fallback_message = fallback_message
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run with timeout (sync, uses ThreadPoolExecutor).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the sub-agent or a timeout fallback.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        def _execute() -> tuple[list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            for event in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return events, output

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_execute)
            try:
                events, output = future.result(
                    timeout=self.timeout_seconds,
                )
                elapsed = time.monotonic() - start
                for event in events:
                    yield event
                timed_out = False
            except TimeoutError:
                elapsed = time.monotonic() - start
                output = self.fallback_message
                yield Event(
                    EventType.AGENT_MESSAGE, self.name,
                    self.fallback_message,
                )
                timed_out = True
            except Exception:
                elapsed = time.monotonic() - start
                logger.exception("[%s] Agent execution failed", self.name)
                output = self.fallback_message
                yield Event(
                    EventType.AGENT_MESSAGE, self.name,
                    self.fallback_message,
                )
                timed_out = False

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"{'Timed out' if timed_out else 'Completed'} in {elapsed:.1f}s",
            data={
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 2),
                "timeout_seconds": self.timeout_seconds,
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 2),
                "output": output,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run with timeout (async, uses asyncio.wait_for).

        Args:
            ctx: The invocation context.

        Yields:
            Events from the sub-agent or a timeout fallback.
        """
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        async def _execute() -> tuple[list[Event], str]:
            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            events: list[Event] = []
            output = ""
            async for event in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
                events.append(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            return events, output

        start = time.monotonic()
        timed_out = False
        output = ""
        try:
            events, output = await asyncio.wait_for(
                _execute(), timeout=self.timeout_seconds,
            )
            for event in events:
                yield event
        except asyncio.TimeoutError:
            timed_out = True
            output = self.fallback_message
            yield Event(
                EventType.AGENT_MESSAGE, self.name,
                self.fallback_message,
            )
        elapsed = time.monotonic() - start

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"{'Timed out' if timed_out else 'Completed'} in {elapsed:.1f}s",
            data={
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 2),
                "output": output,
            })


# ─────────────────────────────────────────────────────────────────────
# AdaptivePlannerAgent
# ─────────────────────────────────────────────────────────────────────


class AdaptivePlannerAgent(BaseAgent):
    """Adaptive planner: re-plans after every step.

    Like ``PlannerAgent`` but after each step the LLM evaluates
    intermediate results and can modify/extend the remaining plan.
    Handles emergent tasks the original plan didn't anticipate.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Available worker agents.
        model: LLM model for planning.
        provider: LLM provider.
        planning_instruction: Custom planning prompt.
        max_steps: Maximum total steps.
        result_key: Store execution history in ``session.state``.

    Example:
        >>> planner = AdaptivePlannerAgent(
        ...     name="adaptive",
        ...     sub_agents=[researcher, writer, reviewer],
        ...     model="gemini-3-flash-preview",
        ...     provider="google",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        planning_instruction: str = "",
        max_steps: int = 10,
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents,
        )
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.planning_instruction = planning_instruction
        self.max_steps = max_steps
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialise the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _plan(
        self, question: str, history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Ask the LLM for the next step(s).

        Returns a list of step dicts with ``agent`` and ``message``.
        """
        agent_names = [a.name for a in (self.sub_agents or [])]
        history_text = ""
        if history:
            history_text = "\n".join(
                f"Step {h['step']}: agent={h['agent']}, "
                f"result={h['output'][:200]}"
                for h in history
            )

        prompt = (
            f"{self.planning_instruction}\n"
            f"Available agents: {agent_names}\n"
            f"Original task: {question}\n\n"
            f"Completed steps:\n{history_text or 'None'}\n\n"
            f"Return a JSON array of the next step(s) to execute. "
            f'Each step: {{"agent": "<name>", "message": "<task>"}}. '
            f"Return an empty array [] if the task is complete."
        )
        messages = [
            {"role": "system", "content": "You are a task planner."},
            {"role": "user", "content": prompt},
        ]
        raw = self.service.generate_completion(
            messages=messages, temperature=0.0, max_tokens=1024,
        )
        raw_str = str(raw).strip()
        # Extract JSON from possible markdown fences
        if "```" in raw_str:
            parts = raw_str.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("["):
                    raw_str = stripped
                    break

        try:
            plan = json.loads(raw_str)
            if isinstance(plan, list):
                return plan  # type: ignore[return-value]
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "[%s] Plan parse error: %s", self.name, raw_str[:200],
            )
        return []

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Adaptive planning loop (sync).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each step plus planning updates.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        history: list[dict[str, Any]] = []
        last_output = ""

        for step_idx in range(self.max_steps):
            # Plan next step(s)
            plan = self._plan(ctx.user_message, history)
            if not plan:
                break  # LLM says task is complete

            next_step = plan[0]  # execute one step at a time
            agent_name = next_step.get("agent", "")
            message = next_step.get("message", ctx.user_message)

            agent = self.find_sub_agent(agent_name)
            if not agent:
                yield Event(
                    EventType.ERROR, self.name,
                    f"Agent {agent_name!r} not found.",
                )
                break

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output

            history.append({
                "step": step_idx + 1,
                "agent": agent_name,
                "message": _truncate(message),
                "output": _truncate(output),
            })

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step_idx + 1}: {agent_name} done",
                data={
                    "step": step_idx + 1,
                    "agent": agent_name,
                    "remaining_plan": plan[1:],
                },
            )

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": history[-20:],
                "total_steps": len(history),
                "completed": True,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Adaptive planning loop (async).

        Args:
            ctx: The invocation context.

        Yields:
            Events from each step plus planning updates.
        """
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        history: list[dict[str, Any]] = []
        last_output = ""

        for step_idx in range(self.max_steps):
            plan = await asyncio.to_thread(
                self._plan, ctx.user_message, history,
            )
            if not plan:
                break

            next_step = plan[0]
            agent_name = next_step.get("agent", "")
            message = next_step.get("message", ctx.user_message)

            agent = self.find_sub_agent(agent_name)
            if not agent:
                yield Event(
                    EventType.ERROR, self.name,
                    f"Agent {agent_name!r} not found.",
                )
                break

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content
            last_output = output

            history.append({
                "step": step_idx + 1,
                "agent": agent_name,
                "message": _truncate(message),
                "output": _truncate(output),
            })

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)

        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": history[-20:],
                "total_steps": len(history),
                "completed": True,
            })


# =====================================================================
# SkeletonOfThoughtAgent
# =====================================================================


class SkeletonOfThoughtAgent(BaseAgent):
    """Skeleton-of-Thought: parallel elaboration of outline points.

    Generates a skeleton (N key points) first, then elaborates each point
    **in parallel** using ``worker_agent``, and finally an ``assembler_agent``
    stitches them into a coherent document.  Optimises latency by
    parallelising the expensive elaboration phase.

    Args:
        name: Agent name.
        description: Human-readable description.
        skeleton_agent: Agent that produces the skeleton (numbered list).
        worker_agent: Agent that elaborates one skeleton point.
        assembler_agent: Agent that merges elaborated sections.
        max_points: Maximum skeleton points to generate.
        result_key: Store breakdown in ``session.state``.

    Example:
        >>> sot = SkeletonOfThoughtAgent(
        ...     name="sot",
        ...     skeleton_agent=outliner,
        ...     worker_agent=writer,
        ...     assembler_agent=assembler,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        skeleton_agent: BaseAgent | None = None,
        worker_agent: BaseAgent | None = None,
        assembler_agent: BaseAgent | None = None,
        max_points: int = 7,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [
            a for a in (skeleton_agent, worker_agent, assembler_agent) if a
        ]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.skeleton_agent = skeleton_agent
        self.worker_agent = worker_agent
        self.assembler_agent = assembler_agent
        self.max_points = max_points
        self.result_key = result_key

    @staticmethod
    def _parse_skeleton(raw: str, max_points: int) -> list[str]:
        """Extract numbered points from skeleton output."""
        points: list[str] = []
        for line in raw.splitlines():
            m = _LIST_ITEM_RE.match(line)
            if m:
                points.append(m.group(1).strip())
                if len(points) >= max_points:
                    break
        if not points:
            points = [
                s.strip() for s in raw.split(".")
                if s.strip()
            ][:max_points]
        return points

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Skeleton → parallel elaborate → assemble (sync)."""
        if not (self.skeleton_agent and self.worker_agent
                and self.assembler_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires skeleton_agent, worker_agent, assembler_agent.",
            )
            return

        # Phase 1: generate skeleton
        skel_ctx = InvocationContext(
            session=Session(), user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        skeleton_raw = ""
        for event in self.skeleton_agent._run_impl_traced(skel_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                skeleton_raw = event.content

        points = self._parse_skeleton(skeleton_raw, self.max_points)
        if not points:
            yield Event(
                EventType.ERROR, self.name, "Empty skeleton.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Skeleton: {len(points)} points extracted.",
            data={"points": points},
        )

        # Phase 2: elaborate each point in parallel
        elaborations: dict[int, str] = {}

        def _elaborate(idx: int, point: str) -> tuple[int, str]:
            msg = (
                f"Elaborate on point {idx + 1}: {point}\n\n"
                f"Original question: {ctx.user_message}"
            )
            sub_ctx = InvocationContext(
                session=Session(), user_message=msg,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            for ev in self.worker_agent._run_impl_traced(sub_ctx):
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return idx, out

        with ThreadPoolExecutor(max_workers=min(len(points), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = {
                pool.submit(_elaborate, i, p): i
                for i, p in enumerate(points)
            }
            for fut in as_completed(futs):
                idx, text = fut.result(timeout=_FUTURE_TIMEOUT)
                elaborations[idx] = text

        ordered = [elaborations.get(i, "") for i in range(len(points))]

        # Phase 3: assemble
        assemble_msg = (
            f"Original question: {ctx.user_message}\n\n"
            "Merge these elaborated sections into a coherent document:\n\n"
            + "\n\n".join(
                f"## Section {i + 1}: {points[i]}\n{ordered[i]}"
                for i in range(len(points))
            )
        )
        asm_ctx = InvocationContext(
            session=Session(), user_message=assemble_msg,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        final = ""
        for event in self.assembler_agent._run_impl_traced(asm_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                final = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "skeleton": points,
                "elaborations": ordered,
                "final": final,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Skeleton → parallel elaborate → assemble (async)."""
        if not (self.skeleton_agent and self.worker_agent
                and self.assembler_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires skeleton_agent, worker_agent, assembler_agent.",
            )
            return

        skel_ctx = InvocationContext(
            session=Session(), user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        skeleton_raw = ""
        async for event in self.skeleton_agent._run_async_impl_traced(
            skel_ctx,
        ):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                skeleton_raw = event.content

        points = self._parse_skeleton(skeleton_raw, self.max_points)
        if not points:
            yield Event(EventType.ERROR, self.name, "Empty skeleton.")
            return

        async def _elaborate_async(
            idx: int, point: str,
        ) -> tuple[int, str]:
            msg = (
                f"Elaborate on point {idx + 1}: {point}\n\n"
                f"Original question: {ctx.user_message}"
            )
            sub_ctx = InvocationContext(
                session=Session(), user_message=msg,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in self.worker_agent._run_async_impl_traced(
                sub_ctx,
            ):
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return idx, out

        results = await asyncio.gather(
            *[_elaborate_async(i, p) for i, p in enumerate(points)],
            return_exceptions=True,
        )
        elaborations: dict[int, str] = {}
        for r in results:
            if isinstance(r, BaseException):
                continue
            idx, text = r
            elaborations[idx] = text
        ordered = [elaborations.get(i, "") for i in range(len(points))]

        assemble_msg = (
            f"Original question: {ctx.user_message}\n\n"
            "Merge these elaborated sections into a coherent document:\n\n"
            + "\n\n".join(
                f"## Section {i + 1}: {points[i]}\n{ordered[i]}"
                for i in range(len(points))
            )
        )
        asm_ctx = InvocationContext(
            session=Session(), user_message=assemble_msg,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        final = ""
        async for event in self.assembler_agent._run_async_impl_traced(
            asm_ctx,
        ):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                final = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "skeleton": points,
                "elaborations": ordered,
                "final": final,
            })


# =====================================================================
# LeastToMostAgent
# =====================================================================


class LeastToMostAgent(BaseAgent):
    """Least-to-Most prompting: solve from easiest to hardest.

    Decomposes a complex problem into sub-problems ordered by
    difficulty (easiest first).  Each sub-problem is solved
    sequentially, with the solutions of **all previous sub-problems**
    accumulated as context for the next.  This mimics curriculum-style
    learning within a single request.

    Args:
        name: Agent name.
        description: Human-readable description.
        decomposer_agent: Agent that breaks the problem into ordered sub-problems.
        solver_agent: Agent that solves each sub-problem.
        synthesizer_agent: Agent that produces the final answer.
        max_subproblems: Maximum sub-problems to generate.
        result_key: Store breakdown in ``session.state``.

    Example:
        >>> l2m = LeastToMostAgent(
        ...     name="l2m",
        ...     decomposer_agent=decomposer,
        ...     solver_agent=solver,
        ...     synthesizer_agent=synth,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        decomposer_agent: BaseAgent | None = None,
        solver_agent: BaseAgent | None = None,
        synthesizer_agent: BaseAgent | None = None,
        max_subproblems: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [
            a for a in (decomposer_agent, solver_agent, synthesizer_agent) if a
        ]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.decomposer_agent = decomposer_agent
        self.solver_agent = solver_agent
        self.synthesizer_agent = synthesizer_agent
        self.max_subproblems = max_subproblems
        self.result_key = result_key

    @staticmethod
    def _parse_subproblems(raw: str, max_items: int) -> list[str]:
        """Extract numbered sub-problems from decomposer output."""
        items: list[str] = []
        for line in raw.splitlines():
            m = _LIST_ITEM_RE.match(line)
            if m:
                items.append(m.group(1).strip())
                if len(items) >= max_items:
                    break
        if not items:
            items = [s.strip() for s in raw.split(".") if s.strip()][:max_items]
        return items

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Decompose → solve sequentially with accumulation → synthesise (sync)."""
        if not (self.decomposer_agent and self.solver_agent
                and self.synthesizer_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires decomposer_agent, solver_agent, synthesizer_agent.",
            )
            return

        # Phase 1: decompose
        dec_ctx = InvocationContext(
            session=Session(),
            user_message=(
                f"Break this problem into sub-problems ordered from "
                f"easiest to hardest (max {self.max_subproblems}):\n\n"
                f"{ctx.user_message}"
            ),
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        raw = ""
        for event in self.decomposer_agent._run_impl_traced(dec_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                raw = event.content

        subproblems = self._parse_subproblems(raw, self.max_subproblems)
        if not subproblems:
            yield Event(EventType.ERROR, self.name, "No sub-problems found.")
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Decomposed into {len(subproblems)} sub-problems.",
            data={"subproblems": subproblems},
        )

        # Phase 2: solve sequentially with accumulation
        solutions: list[dict[str, str]] = []
        accumulated = ""
        for i, sp in enumerate(subproblems):
            msg = (
                f"Original problem: {ctx.user_message}\n\n"
                f"Previously solved:\n{accumulated or 'None'}\n\n"
                f"Now solve sub-problem {i + 1}: {sp}"
            )
            sol_ctx = InvocationContext(
                session=Session(), user_message=msg,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            for event in self.solver_agent._run_impl_traced(sol_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    out = event.content
            solutions.append({"subproblem": sp, "solution": out})
            accumulated += f"\nSub-problem {i + 1}: {sp}\nSolution: {_truncate(out)}\n"

        # Phase 3: synthesise
        synth_msg = (
            f"Original problem: {ctx.user_message}\n\n"
            "All sub-problems solved (easiest to hardest):\n\n"
            + "\n".join(
                f"{i + 1}. {s['subproblem']}: {_truncate(s['solution'])}"
                for i, s in enumerate(solutions)
            )
            + "\n\nProduce a coherent final answer."
        )
        syn_ctx = InvocationContext(
            session=Session(), user_message=synth_msg,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        final = ""
        for event in self.synthesizer_agent._run_impl_traced(syn_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                final = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "subproblems": subproblems,
                "solutions": solutions,
                "final": final,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Decompose → solve sequentially with accumulation → synthesise (async)."""
        if not (self.decomposer_agent and self.solver_agent
                and self.synthesizer_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires decomposer_agent, solver_agent, synthesizer_agent.",
            )
            return

        dec_ctx = InvocationContext(
            session=Session(),
            user_message=(
                f"Break this problem into sub-problems ordered from "
                f"easiest to hardest (max {self.max_subproblems}):\n\n"
                f"{ctx.user_message}"
            ),
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        raw = ""
        async for event in self.decomposer_agent._run_async_impl_traced(
            dec_ctx,
        ):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                raw = event.content

        subproblems = self._parse_subproblems(raw, self.max_subproblems)
        if not subproblems:
            yield Event(EventType.ERROR, self.name, "No sub-problems found.")
            return

        solutions: list[dict[str, str]] = []
        accumulated = ""
        for i, sp in enumerate(subproblems):
            msg = (
                f"Original problem: {ctx.user_message}\n\n"
                f"Previously solved:\n{accumulated or 'None'}\n\n"
                f"Now solve sub-problem {i + 1}: {sp}"
            )
            sol_ctx = InvocationContext(
                session=Session(), user_message=msg,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for event in self.solver_agent._run_async_impl_traced(
                sol_ctx,
            ):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    out = event.content
            solutions.append({"subproblem": sp, "solution": out})
            accumulated += f"\nSub-problem {i + 1}: {sp}\nSolution: {_truncate(out)}\n"

        synth_msg = (
            f"Original problem: {ctx.user_message}\n\n"
            "All sub-problems solved (easiest to hardest):\n\n"
            + "\n".join(
                f"{i + 1}. {s['subproblem']}: {_truncate(s['solution'])}"
                for i, s in enumerate(solutions)
            )
            + "\n\nProduce a coherent final answer."
        )
        syn_ctx = InvocationContext(
            session=Session(), user_message=synth_msg,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        final = ""
        async for event in self.synthesizer_agent._run_async_impl_traced(
            syn_ctx,
        ):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                final = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "subproblems": subproblems,
                "solutions": solutions,
                "final": final,
            })


# =====================================================================
# SelfDiscoverAgent
# =====================================================================


class SelfDiscoverAgent(BaseAgent):
    """Self-Discover: LLMs self-compose reasoning structures.

    Pipeline of 4 phases: SELECT relevant reasoning modules from a pool,
    ADAPT them to the task, IMPLEMENT a composite structure, EXECUTE
    using the composed structure.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The LLM agent used for all phases.
        reasoning_modules: Pool of reasoning strategies to select from.
        result_key: Store phases in ``session.state``.

    Example:
        >>> sd = SelfDiscoverAgent(
        ...     name="discovery",
        ...     agent=thinker,
        ...     reasoning_modules=["critical thinking", "step by step", "analogy"],
        ... )
    """

    DEFAULT_MODULES: list[str] = [
        "Critical Thinking",
        "Step-by-Step Analysis",
        "Analogical Reasoning",
        "Decomposition",
        "Abstraction & Generalisation",
        "Cause-and-Effect Analysis",
        "Hypothesis Testing",
        "Constraint Satisfaction",
        "Pattern Recognition",
        "First Principles Reasoning",
    ]

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        reasoning_modules: list[str] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.agent = agent
        self.reasoning_modules = reasoning_modules or self.DEFAULT_MODULES
        self.result_key = result_key

    def _call_agent(
        self, ctx: InvocationContext, prompt: str,
    ) -> str:
        """Run the agent synchronously and return its output."""
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_agent_async(
        self, ctx: InvocationContext, prompt: str,
    ) -> str:
        """Run the agent asynchronously and return its output."""
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """SELECT → ADAPT → IMPLEMENT → EXECUTE (sync)."""
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        modules_str = "\n".join(
            f"- {m}" for m in self.reasoning_modules
        )

        # Phase 1: SELECT
        select_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Available reasoning modules:\n{modules_str}\n\n"
            "Select the most relevant modules for this task. "
            "List only the selected modules."
        )
        selected = self._call_agent(ctx, select_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"SELECT: {selected[:200]}",
        )

        # Phase 2: ADAPT
        adapt_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Selected reasoning modules:\n{_truncate(selected)}\n\n"
            "Adapt each module to this specific task. "
            "Describe how each will be applied."
        )
        adapted = self._call_agent(ctx, adapt_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"ADAPT: {adapted[:200]}",
        )

        # Phase 3: IMPLEMENT
        implement_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Adapted reasoning strategy:\n{_truncate(adapted)}\n\n"
            "Implement a step-by-step reasoning structure "
            "that combines these modules into one plan."
        )
        structure = self._call_agent(ctx, implement_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"IMPLEMENT: {structure[:200]}",
        )

        # Phase 4: EXECUTE
        execute_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Reasoning structure to follow:\n{_truncate(structure)}\n\n"
            "Now execute each step of the structure and produce the answer."
        )
        final = self._call_agent(ctx, execute_prompt)

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": _truncate(selected),
                "adapted": _truncate(adapted),
                "structure": _truncate(structure),
                "final": _truncate(final),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """SELECT → ADAPT → IMPLEMENT → EXECUTE (async)."""
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        modules_str = "\n".join(
            f"- {m}" for m in self.reasoning_modules
        )

        select_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Available reasoning modules:\n{modules_str}\n\n"
            "Select the most relevant modules for this task."
        )
        selected = await self._call_agent_async(ctx, select_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"SELECT: {selected[:200]}",
        )

        adapt_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Selected reasoning modules:\n{_truncate(selected)}\n\n"
            "Adapt each module to this specific task."
        )
        adapted = await self._call_agent_async(ctx, adapt_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"ADAPT: {adapted[:200]}",
        )

        implement_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Adapted reasoning strategy:\n{_truncate(adapted)}\n\n"
            "Implement a step-by-step reasoning structure."
        )
        structure = await self._call_agent_async(ctx, implement_prompt)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"IMPLEMENT: {structure[:200]}",
        )

        execute_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Reasoning structure to follow:\n{_truncate(structure)}\n\n"
            "Execute each step and produce the answer."
        )
        final = await self._call_agent_async(ctx, execute_prompt)

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": _truncate(selected),
                "adapted": _truncate(adapted),
                "structure": _truncate(structure),
                "final": _truncate(final),
            })


# =====================================================================
# GeneticAlgorithmAgent
# =====================================================================


class GeneticAlgorithmAgent(BaseAgent):
    """Evolutionary optimisation: population → fitness → crossover → mutation.

    Maintains a population of solutions across generations. Each
    generation: (1) agents generate candidate solutions, (2) ``fitness_fn``
    scores them, (3) a crossover agent combines top parents,
    (4) a mutation agent introduces variation. Evolves over
    ``n_generations``.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Agent that generates candidate solutions.
        crossover_agent: Agent that combines two parent solutions.
        mutation_agent: Agent that mutates a solution.
        fitness_fn: ``(response: str) -> float`` fitness function.
        population_size: Number of candidates per generation.
        n_generations: Number of evolutionary generations.
        mutation_rate: Probability of mutation per offspring.
        elite_count: Top solutions carried unchanged to next generation.
        result_key: Store evolution log in ``session.state``.

    Example:
        >>> ga = GeneticAlgorithmAgent(
        ...     name="ga",
        ...     agent=generator,
        ...     crossover_agent=combiner,
        ...     mutation_agent=mutator,
        ...     fitness_fn=lambda r: len(r) / 100,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        crossover_agent: BaseAgent | None = None,
        mutation_agent: BaseAgent | None = None,
        fitness_fn: Callable[[str], float] | None = None,
        population_size: int = 6,
        n_generations: int = 3,
        mutation_rate: float = 0.3,
        elite_count: int = 1,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (agent, crossover_agent, mutation_agent) if a]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.agent = agent
        self.crossover_agent = crossover_agent
        self.mutation_agent = mutation_agent
        self.fitness_fn = fitness_fn or (lambda _r: 0.5)
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.result_key = result_key

    def _generate(
        self, ctx: InvocationContext, prompt: str,
        agent: BaseAgent,
    ) -> str:
        """Run an agent synchronously and return output."""
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Evolve solutions over N generations (sync)."""
        if not (self.agent and self.crossover_agent and self.mutation_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires agent, crossover_agent, mutation_agent.",
            )
            return

        # Initial population
        population: list[tuple[str, float]] = []
        with ThreadPoolExecutor(max_workers=min(self.population_size, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(
                    self._generate, ctx, ctx.user_message, self.agent,
                )
                for _ in range(self.population_size)
            ]
            for fut in as_completed(futs):
                resp = fut.result(timeout=_FUTURE_TIMEOUT)
                score = self.fitness_fn(resp)
                population.append((resp, score))

        generation_log: list[dict[str, Any]] = []

        for gen in range(self.n_generations):
            population.sort(key=lambda x: x[1], reverse=True)

            gen_info = {
                "generation": gen + 1,
                "best_score": population[0][1],
                "avg_score": sum(s for _, s in population) / len(population),
            }
            generation_log.append(gen_info)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gen {gen + 1}: best={gen_info['best_score']:.2f} "
                f"avg={gen_info['avg_score']:.2f}",
                data=gen_info,
            )

            # Elite carry-over
            next_gen: list[tuple[str, float]] = list(
                population[:self.elite_count],
            )

            # Crossover + mutation
            parents = population[:max(2, len(population) // 2)]
            while len(next_gen) < self.population_size:
                p1 = random.choice(parents)[0]
                p2 = random.choice(parents)[0]

                cross_prompt = (
                    f"Original task: {ctx.user_message}\n\n"
                    f"Parent A:\n{_truncate(p1)}\n\nParent B:\n{_truncate(p2)}\n\n"
                    "Combine the best parts of both into a new solution."
                )
                child = self._generate(ctx, cross_prompt, self.crossover_agent)

                if random.random() < self.mutation_rate:
                    mut_prompt = (
                        f"Original task: {ctx.user_message}\n\n"
                        f"Solution:\n{_truncate(child)}\n\n"
                        "Introduce a creative variation or improvement."
                    )
                    child = self._generate(
                        ctx, mut_prompt, self.mutation_agent,
                    )

                score = self.fitness_fn(child)
                next_gen.append((child, score))

            population = next_gen

        population.sort(key=lambda x: x[1], reverse=True)
        best = population[0][0]

        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_response": _truncate(best),
                "best_score": population[0][1],
                "generations": generation_log[-20:],
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Evolve solutions over N generations (async)."""
        if not (self.agent and self.crossover_agent and self.mutation_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires agent, crossover_agent, mutation_agent.",
            )
            return

        async def _gen_async(prompt: str, ag: BaseAgent) -> str:
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in ag._run_async_impl_traced(sub_ctx):
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return out

        coros = [
            _gen_async(ctx.user_message, self.agent)
            for _ in range(self.population_size)
        ]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        population: list[tuple[str, float]] = [
            (r, self.fitness_fn(r)) for r in raw_results if not isinstance(r, BaseException)
        ]

        generation_log: list[dict[str, Any]] = []

        for gen in range(self.n_generations):
            population.sort(key=lambda x: x[1], reverse=True)
            gen_info = {
                "generation": gen + 1,
                "best_score": population[0][1],
                "avg_score": sum(s for _, s in population) / len(population),
            }
            generation_log.append(gen_info)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gen {gen + 1}: best={gen_info['best_score']:.2f}",
                data=gen_info,
            )

            next_gen: list[tuple[str, float]] = list(
                population[:self.elite_count],
            )
            parents = population[:max(2, len(population) // 2)]

            child_coros = []
            for _ in range(self.population_size - self.elite_count):
                p1 = random.choice(parents)[0]
                p2 = random.choice(parents)[0]
                cross_prompt = (
                    f"Original task: {ctx.user_message}\n\n"
                    f"Parent A:\n{_truncate(p1)}\n\nParent B:\n{_truncate(p2)}\n\n"
                    "Combine the best parts into a new solution."
                )
                child_coros.append(
                    _gen_async(cross_prompt, self.crossover_agent),
                )

            raw_children = await asyncio.gather(*child_coros, return_exceptions=True)
            children = [c for c in raw_children if not isinstance(c, BaseException)]
            for child in children:
                if random.random() < self.mutation_rate:
                    mut_prompt = (
                        f"Original task: {ctx.user_message}\n\n"
                        f"Solution:\n{_truncate(child)}\n\n"
                        "Introduce a creative variation."
                    )
                    child = await _gen_async(mut_prompt, self.mutation_agent)
                score = self.fitness_fn(child)
                next_gen.append((child, score))

            population = next_gen

        population.sort(key=lambda x: x[1], reverse=True)
        best = population[0][0]
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_response": _truncate(best),
                "best_score": population[0][1],
                "generations": generation_log[-20:],
            })


# =====================================================================
# MultiArmedBanditAgent
# =====================================================================


class MultiArmedBanditAgent(BaseAgent):
    """Multi-Armed Bandit router: learns which agent performs best online.

    Maintains per-agent statistics (wins/total). Uses a configurable
    strategy (epsilon-greedy, UCB1, or Thompson Sampling) to balance
    exploration vs. exploitation across many requests. After each run,
    ``reward_fn`` scores the output and updates the agent's stats.

    The stats persist in ``session.state[stats_key]`` so learning carries
    across calls when the same session is reused.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pool of candidate agents.
        reward_fn: ``(response: str) -> float`` — reward in [0, 1].
        strategy: ``"epsilon_greedy"``, ``"ucb1"``, or ``"thompson"``.
        epsilon: Exploration rate for epsilon-greedy (default 0.1).
        stats_key: Session state key for persisting arm stats.
        result_key: Store selection log in ``session.state``.

    Example:
        >>> bandit = MultiArmedBanditAgent(
        ...     name="bandit",
        ...     sub_agents=[agent_a, agent_b, agent_c],
        ...     reward_fn=lambda r: 1.0 if "correct" in r.lower() else 0.0,
        ...     strategy="ucb1",
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        reward_fn: Callable[[str], float] | None = None,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.1,
        stats_key: str = "_bandit_stats",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents, **kwargs,
        )
        self.reward_fn = reward_fn or (lambda _r: 0.5)
        self.strategy = strategy
        self.epsilon = epsilon
        self.stats_key = stats_key
        self.result_key = result_key

    def _get_stats(
        self, session: Session,
    ) -> dict[str, dict[str, float]]:
        """Retrieve or initialise per-arm statistics."""
        if self.stats_key not in session.state:
            session.state[self.stats_key] = {
                a.name: {"wins": 0.0, "total": 0}
                for a in (self.sub_agents or [])
            }
        stats = session.state[self.stats_key]
        for a in (self.sub_agents or []):
            if a.name not in stats:
                stats[a.name] = {"wins": 0.0, "total": 0}
        return stats

    def _select_arm(
        self, stats: dict[str, dict[str, float]],
    ) -> str:
        """Select an agent based on the configured strategy."""
        agents = [a.name for a in (self.sub_agents or [])]
        if not agents:
            return ""

        if self.strategy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return random.choice(agents)
            return max(
                agents,
                key=lambda n: (
                    stats[n]["wins"] / stats[n]["total"]
                    if stats[n]["total"] > 0 else float("inf")
                ),
            )

        if self.strategy == "ucb1":
            total_pulls = sum(
                stats[n]["total"] for n in agents
            )
            if total_pulls == 0:
                return random.choice(agents)
            scores = {}
            for n in agents:
                s = stats[n]
                if s["total"] == 0:
                    scores[n] = float("inf")
                else:
                    avg = s["wins"] / s["total"]
                    scores[n] = avg + math.sqrt(
                        2 * math.log(total_pulls) / s["total"],
                    )
            return max(scores, key=scores.get)  # type: ignore[arg-type]

        if self.strategy == "thompson":
            samples = {}
            for n in agents:
                s = stats[n]
                alpha = s["wins"] + 1.0
                beta = (s["total"] - s["wins"]) + 1.0
                samples[n] = random.betavariate(alpha, max(beta, 0.01))
            return max(samples, key=samples.get)  # type: ignore[arg-type]

        return random.choice(agents)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Select arm → run → update reward (sync)."""
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        stats = self._get_stats(ctx.session)
        chosen_name = self._select_arm(stats)
        agent = self.find_sub_agent(chosen_name)
        if not agent:
            yield Event(
                EventType.ERROR, self.name,
                f"Agent {chosen_name!r} not found.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Bandit selected: {chosen_name} (strategy={self.strategy})",
        )

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        for event in agent._run_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        reward = self.reward_fn(output)
        stats[chosen_name]["total"] += 1
        stats[chosen_name]["wins"] += reward

        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": chosen_name,
                "reward": reward,
                "stats": dict(stats),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Select arm → run → update reward (async)."""
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        stats = self._get_stats(ctx.session)
        chosen_name = self._select_arm(stats)
        agent = self.find_sub_agent(chosen_name)
        if not agent:
            yield Event(
                EventType.ERROR, self.name,
                f"Agent {chosen_name!r} not found.",
            )
            return

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Bandit selected: {chosen_name} (strategy={self.strategy})",
        )

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        async for event in agent._run_async_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        reward = self.reward_fn(output)
        stats[chosen_name]["total"] += 1
        stats[chosen_name]["wins"] += reward

        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected": chosen_name,
                "reward": reward,
                "stats": dict(stats),
            })


# =====================================================================
# SocraticAgent
# =====================================================================


class SocraticAgent(BaseAgent):
    """Socratic method: iterative questioning to deepen understanding.

    A questioner agent asks probing questions and a respondent agent
    answers.  The questioner analyses each answer, detects gaps, and
    formulates increasingly deeper follow-up questions.  The loop
    continues until ``max_rounds`` or the questioner declares the
    topic exhaustively explored.

    Args:
        name: Agent name.
        description: Human-readable description.
        questioner_agent: Agent that generates probing questions.
        respondent_agent: Agent that answers each question.
        synthesizer_agent: Optional agent that produces the final summary.
        max_rounds: Maximum question-answer rounds.
        completion_keyword: Keyword from questioner that stops the dialogue.
        result_key: Store dialogue in ``session.state``.

    Example:
        >>> socratic = SocraticAgent(
        ...     name="socratic",
        ...     questioner_agent=teacher,
        ...     respondent_agent=student,
        ...     max_rounds=5,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        questioner_agent: BaseAgent | None = None,
        respondent_agent: BaseAgent | None = None,
        synthesizer_agent: BaseAgent | None = None,
        max_rounds: int = 5,
        completion_keyword: str = "EXPLORATION_COMPLETE",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [
            a for a in (questioner_agent, respondent_agent, synthesizer_agent)
            if a
        ]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.questioner_agent = questioner_agent
        self.respondent_agent = respondent_agent
        self.synthesizer_agent = synthesizer_agent
        self.max_rounds = max_rounds
        self.completion_keyword = completion_keyword
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Question → answer → deeper question loop (sync)."""
        if not (self.questioner_agent and self.respondent_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires questioner_agent and respondent_agent.",
            )
            return

        dialogue: list[dict[str, str]] = []
        topic = ctx.user_message

        for rnd in range(self.max_rounds):
            dialogue_text = "\n".join(
                f"Q{i+1}: {d['question']}\nA{i+1}: {d['answer']}"
                for i, d in enumerate(dialogue)
            )
            q_prompt = (
                f"Topic: {topic}\n\n"
                f"Dialogue so far:\n{dialogue_text or 'None'}\n\n"
                f"Ask a deeper probing question to explore gaps. "
                f"If the topic is exhaustively explored, reply with "
                f"'{self.completion_keyword}'."
            )
            question = self._call_sync(
                self.questioner_agent, ctx, q_prompt,
            )

            if self.completion_keyword in question:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Exploration complete after {rnd} rounds.",
                )
                break

            a_prompt = (
                f"Topic: {topic}\n\n"
                f"Previous context:\n{dialogue_text or 'None'}\n\n"
                f"Question: {_truncate(question)}\n\nProvide a thorough answer."
            )
            answer = self._call_sync(
                self.respondent_agent, ctx, a_prompt,
            )

            dialogue.append({"question": _truncate(question), "answer": _truncate(answer)})
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd + 1}: Q={question[:80]}... A={answer[:80]}...",
                data={"round": rnd + 1},
            )

        if self.synthesizer_agent and dialogue:
            synth_prompt = (
                f"Topic: {topic}\n\n"
                "Full Socratic dialogue:\n"
                + "\n".join(
                    f"Q{i+1}: {d['question']}\nA{i+1}: {d['answer']}"
                    for i, d in enumerate(dialogue)
                )
                + "\n\nSynthesise all insights into a comprehensive answer."
            )
            final = self._call_sync(
                self.synthesizer_agent, ctx, synth_prompt,
            )
        elif dialogue:
            final = dialogue[-1]["answer"]
        else:
            final = ""

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "dialogue": dialogue,
                "rounds": len(dialogue),
                "final": _truncate(final),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Question → answer → deeper question loop (async)."""
        if not (self.questioner_agent and self.respondent_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires questioner_agent and respondent_agent.",
            )
            return

        dialogue: list[dict[str, str]] = []
        topic = ctx.user_message

        for rnd in range(self.max_rounds):
            dialogue_text = "\n".join(
                f"Q{i+1}: {d['question']}\nA{i+1}: {d['answer']}"
                for i, d in enumerate(dialogue)
            )
            q_prompt = (
                f"Topic: {topic}\n\n"
                f"Dialogue so far:\n{dialogue_text or 'None'}\n\n"
                f"Ask a deeper probing question. "
                f"If done, reply '{self.completion_keyword}'."
            )
            question = await self._call_async(
                self.questioner_agent, ctx, q_prompt,
            )

            if self.completion_keyword in question:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Exploration complete after {rnd} rounds.",
                )
                break

            a_prompt = (
                f"Topic: {topic}\n\n"
                f"Previous context:\n{dialogue_text or 'None'}\n\n"
                f"Question: {_truncate(question)}\n\nProvide a thorough answer."
            )
            answer = await self._call_async(
                self.respondent_agent, ctx, a_prompt,
            )
            dialogue.append({"question": _truncate(question), "answer": _truncate(answer)})
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd + 1}: Q={question[:80]}...",
                data={"round": rnd + 1},
            )

        if self.synthesizer_agent and dialogue:
            synth_prompt = (
                f"Topic: {topic}\n\nFull dialogue:\n"
                + "\n".join(
                    f"Q{i+1}: {d['question']}\nA{i+1}: {d['answer']}"
                    for i, d in enumerate(dialogue)
                )
                + "\n\nSynthesise all insights."
            )
            final = await self._call_async(
                self.synthesizer_agent, ctx, synth_prompt,
            )
        elif dialogue:
            final = dialogue[-1]["answer"]
        else:
            final = ""

        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "dialogue": dialogue,
                "rounds": len(dialogue),
                "final": _truncate(final),
            })


# =====================================================================
# MetaOrchestratorAgent
# =====================================================================


class MetaOrchestratorAgent(BaseAgent):
    """Meta-Orchestrator: chooses the orchestration pattern, not the agent.

    Given a task, an LLM analyses its nature and selects the best
    orchestration **pattern** (Sequential, Parallel, Debate, Cascade,
    etc.) from a registry of available patterns, then constructs and
    executes it with the supplied worker agents.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Worker agents available for orchestration.
        patterns: Registry of pattern name → callable factory.
        model: LLM model for pattern selection.
        provider: LLM provider.
        result_key: Store selection in ``session.state``.

    Example:
        >>> meta = MetaOrchestratorAgent(
        ...     name="meta",
        ...     sub_agents=[researcher, writer, reviewer],
        ...     model="gemini-3-flash-preview",
        ...     provider="google",
        ... )
    """

    DEFAULT_PATTERNS: list[str] = [
        "SequentialAgent",
        "ParallelAgent",
        "DebateAgent",
        "MapReduceAgent",
        "CascadeAgent",
        "ConsensusAgent",
        "PlannerAgent",
    ]

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        patterns: dict[str, Callable[..., BaseAgent]] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents,
        )
        self.patterns = patterns or {}
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialise the connector service."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _select_pattern(self, question: str) -> str:
        """Ask the LLM which orchestration pattern to use."""
        available = list(self.patterns.keys()) or self.DEFAULT_PATTERNS
        agent_descs = "\n".join(
            f"- {a.name}: {a.description}" for a in (self.sub_agents or [])
        )
        prompt = (
            f"Task: {question}\n\n"
            f"Available worker agents:\n{agent_descs}\n\n"
            f"Available orchestration patterns: {available}\n\n"
            "Choose the single best pattern for this task. "
            "Reply with ONLY the pattern name, nothing else."
        )
        raw = self.service.generate_completion(
            messages=[
                {"role": "system", "content": "You are an orchestration expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, max_tokens=64,
        )
        return str(raw).strip()

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Select pattern → build → execute (sync)."""
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        pattern_name = self._select_pattern(ctx.user_message)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Selected pattern: {pattern_name}",
        )

        if pattern_name in self.patterns:
            orchestrator = self.patterns[pattern_name](
                name=f"{self.name}_inner",
                sub_agents=self.sub_agents,
            )
        else:
            orchestrator = SequentialAgent(
                name=f"{self.name}_sequential",
                sub_agents=self.sub_agents,
            )
            pattern_name = "SequentialAgent (fallback)"

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        for event in orchestrator._run_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected_pattern": pattern_name,
                "output": output,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Select pattern → build → execute (async)."""
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        pattern_name = await asyncio.to_thread(
            self._select_pattern, ctx.user_message,
        )
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Selected pattern: {pattern_name}",
        )

        if pattern_name in self.patterns:
            orchestrator = self.patterns[pattern_name](
                name=f"{self.name}_inner",
                sub_agents=self.sub_agents,
            )
        else:
            orchestrator = SequentialAgent(
                name=f"{self.name}_sequential",
                sub_agents=self.sub_agents,
            )
            pattern_name = "SequentialAgent (fallback)"

        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        async for event in orchestrator._run_async_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "selected_pattern": pattern_name,
                "output": output,
            })


# =====================================================================
# CacheAgent
# =====================================================================


class CacheAgent(BaseAgent):
    """Semantic caching wrapper: avoids redundant LLM calls.

    If the input is identical (or semantically similar via an optional
    ``similarity_fn``) to a cached request, returns the cached result
    without calling the wrapped agent. Supports TTL expiration and a
    maximum cache size.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The agent to wrap with caching.
        similarity_fn: Optional ``(str, str) -> float`` semantic similarity.
        similarity_threshold: Threshold for semantic match (0.0–1.0).
        ttl_seconds: Time-to-live for cache entries (0 = no expiry).
        max_entries: Maximum cache entries before LRU eviction.
        cache_key: Session state key for the cache store.
        result_key: Store cache hit/miss in ``session.state``.

    Example:
        >>> cached = CacheAgent(
        ...     name="cached",
        ...     agent=expensive_agent,
        ...     ttl_seconds=300,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        similarity_fn: Callable[[str, str], float] | None = None,
        similarity_threshold: float = 0.95,
        ttl_seconds: float = 0,
        max_entries: int = 100,
        cache_key: str = "_cache_store",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.agent = agent
        self.similarity_fn = similarity_fn
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache_key = cache_key
        self.result_key = result_key

    def _get_cache(
        self, session: Session,
    ) -> list[dict[str, Any]]:
        """Get or initialise the cache from session state."""
        if self.cache_key not in session.state:
            session.state[self.cache_key] = []
        return session.state[self.cache_key]

    def _lookup(
        self, query: str, cache: list[dict[str, Any]],
    ) -> str | None:
        """Look up a query in the cache."""
        now = time.time()
        query_hash = hashlib.sha256(
            query.encode("utf-8"),
        ).hexdigest()

        for entry in reversed(cache):
            if self.ttl_seconds > 0:
                age = now - entry.get("timestamp", 0)
                if age > self.ttl_seconds:
                    continue

            if entry.get("hash") == query_hash:
                return entry["response"]

            if self.similarity_fn:
                sim = self.similarity_fn(query, entry.get("query", ""))
                if sim >= self.similarity_threshold:
                    return entry["response"]

        return None

    def _store(
        self, query: str, response: str,
        cache: list[dict[str, Any]],
    ) -> None:
        """Store a new entry, evicting LRU if needed."""
        cache.append({
            "query": query,
            "hash": hashlib.sha256(
                query.encode("utf-8"),
            ).hexdigest(),
            "response": response,
            "timestamp": time.time(),
        })
        while len(cache) > self.max_entries:
            cache.pop(0)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Check cache → run agent if miss → store result (sync)."""
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        cache = self._get_cache(ctx.session)
        cached = self._lookup(ctx.user_message, cache)

        if cached is not None:
            yield Event(
                EventType.STATE_UPDATE, self.name, "Cache HIT",
            )
            yield Event(EventType.AGENT_MESSAGE, self.name, cached)
            if self.result_key:
                ctx.session.state_set(self.result_key, {
                    "cache_hit": True, "response": cached,
                })
            return

        yield Event(EventType.STATE_UPDATE, self.name, "Cache MISS")
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        for event in self.agent._run_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        self._store(ctx.user_message, output, cache)
        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "cache_hit": False, "response": output,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Check cache → run agent if miss → store result (async)."""
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        cache = self._get_cache(ctx.session)
        cached = self._lookup(ctx.user_message, cache)

        if cached is not None:
            yield Event(
                EventType.STATE_UPDATE, self.name, "Cache HIT",
            )
            yield Event(EventType.AGENT_MESSAGE, self.name, cached)
            if self.result_key:
                ctx.session.state_set(self.result_key, {
                    "cache_hit": True, "response": cached,
                })
            return

        yield Event(EventType.STATE_UPDATE, self.name, "Cache MISS")
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        output = ""
        async for event in self.agent._run_async_impl_traced(sub_ctx):
            yield event
            if event.event_type == EventType.AGENT_MESSAGE:
                output = event.content

        self._store(ctx.user_message, output, cache)
        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "cache_hit": False, "response": output,
            })


# =====================================================================
# BudgetAgent
# =====================================================================


class BudgetAgent(BaseAgent):
    """Budget-aware wrapper: caps token/cost consumption.

    Wraps a pipeline of sub-agents with a token budget. After each
    agent completes, the ``cost_fn`` estimates the tokens used. When
    the budget is exhausted, the agent can (a) switch to a cheaper
    fallback, (b) stop execution, or (c) continue with a warning.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Pipeline of agents to run within budget.
        cost_fn: ``(response: str) -> float`` — estimates tokens/cost.
        budget: Maximum total cost.
        fallback_agent: Agent used when budget is exhausted.
        on_exhausted: ``"fallback"``, ``"stop"``, or ``"warn"``.
        result_key: Store budget log in ``session.state``.

    Example:
        >>> budget = BudgetAgent(
        ...     name="budget",
        ...     sub_agents=[agent_a, agent_b],
        ...     cost_fn=lambda r: len(r) / 4,
        ...     budget=1000,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        cost_fn: Callable[[str], float] | None = None,
        budget: float = 1000.0,
        fallback_agent: BaseAgent | None = None,
        on_exhausted: str = "stop",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        all_subs = list(sub_agents or [])
        if fallback_agent and fallback_agent not in all_subs:
            all_subs.append(fallback_agent)
        super().__init__(
            name=name, description=description,
            sub_agents=all_subs, **kwargs,
        )
        self._pipeline = list(sub_agents or [])
        self.cost_fn = cost_fn or (lambda r: len(r) / 4.0)
        self.budget = budget
        self.fallback_agent = fallback_agent
        self.on_exhausted = on_exhausted
        self.result_key = result_key

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Run pipeline within budget (sync)."""
        if not self._pipeline:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        spent = 0.0
        last_output = ""
        log: list[dict[str, Any]] = []

        for agent in self._pipeline:
            if spent >= self.budget:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Budget exhausted ({spent:.1f}/{self.budget:.1f})",
                )
                if self.on_exhausted == "fallback" and self.fallback_agent:
                    sub_ctx = InvocationContext(
                        session=ctx.session,
                        user_message=ctx.user_message,
                        parent_agent=self,
                        trace_collector=ctx.trace_collector,
                    )
                    for event in self.fallback_agent._run_impl_traced(sub_ctx):
                        yield event
                        if event.event_type == EventType.AGENT_MESSAGE:
                            last_output = event.content
                elif self.on_exhausted == "warn":
                    pass
                else:
                    break

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            for event in agent._run_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content

            cost = self.cost_fn(output)
            spent += cost
            last_output = output
            log.append({
                "agent": agent.name, "cost": cost, "total_spent": spent,
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"{agent.name}: cost={cost:.1f}, total={spent:.1f}/{self.budget:.1f}",
                data={"agent": agent.name, "cost": cost, "total": spent},
            )

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "total_spent": spent,
                "budget": self.budget,
                "log": log,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run pipeline within budget (async)."""
        if not self._pipeline:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        spent = 0.0
        last_output = ""
        log: list[dict[str, Any]] = []

        for agent in self._pipeline:
            if spent >= self.budget:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Budget exhausted ({spent:.1f}/{self.budget:.1f})",
                )
                if self.on_exhausted == "fallback" and self.fallback_agent:
                    sub_ctx = InvocationContext(
                        session=ctx.session,
                        user_message=ctx.user_message,
                        parent_agent=self,
                        trace_collector=ctx.trace_collector,
                    )
                    async for event in self.fallback_agent._run_async_impl_traced(sub_ctx):
                        yield event
                        if event.event_type == EventType.AGENT_MESSAGE:
                            last_output = event.content
                elif self.on_exhausted == "warn":
                    pass
                else:
                    break

            sub_ctx = InvocationContext(
                session=ctx.session, user_message=ctx.user_message,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            output = ""
            async for event in agent._run_async_impl_traced(sub_ctx):
                yield event
                if event.event_type == EventType.AGENT_MESSAGE:
                    output = event.content

            cost = self.cost_fn(output)
            spent += cost
            last_output = output
            log.append({
                "agent": agent.name, "cost": cost, "total_spent": spent,
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"{agent.name}: cost={cost:.1f}, total={spent:.1f}/{self.budget:.1f}",
                data={"agent": agent.name, "cost": cost, "total": spent},
            )

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "total_spent": spent,
                "budget": self.budget,
                "log": log,
            })


# =====================================================================
# CurriculumAgent
# =====================================================================


class CurriculumAgent(BaseAgent):
    """Curriculum learning: progressive task generation with skill library.

    The proposer agent generates tasks of increasing difficulty. The
    solver agent attempts each task. Successful solutions are stored
    in a skill library (session state). On future tasks, the library
    is consulted before attempting a fresh solution.

    Args:
        name: Agent name.
        description: Human-readable description.
        proposer_agent: Agent that proposes progressively harder tasks.
        solver_agent: Agent that attempts each task.
        success_fn: ``(response: str) -> bool`` — True if the task was solved.
        max_tasks: Maximum tasks to attempt.
        library_key: Session state key for the skill library.
        result_key: Store curriculum log in ``session.state``.

    Example:
        >>> curriculum = CurriculumAgent(
        ...     name="curriculum",
        ...     proposer_agent=proposer,
        ...     solver_agent=solver,
        ...     success_fn=lambda r: "PASS" in r,
        ... )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        proposer_agent: BaseAgent | None = None,
        solver_agent: BaseAgent | None = None,
        success_fn: Callable[[str], bool] | None = None,
        max_tasks: int = 5,
        library_key: str = "_skill_library",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (proposer_agent, solver_agent) if a]
        super().__init__(
            name=name, description=description,
            sub_agents=subs, **kwargs,
        )
        self.proposer_agent = proposer_agent
        self.solver_agent = solver_agent
        self.success_fn = success_fn or (lambda _r: True)
        self.max_tasks = max_tasks
        self.library_key = library_key
        self.result_key = result_key

    def _get_library(
        self, session: Session,
    ) -> list[dict[str, str]]:
        """Retrieve or initialise the skill library."""
        if self.library_key not in session.state:
            session.state[self.library_key] = []
        return session.state[self.library_key]

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Propose → solve → store → repeat (sync)."""
        if not (self.proposer_agent and self.solver_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires proposer_agent and solver_agent.",
            )
            return

        library = self._get_library(ctx.session)
        curriculum_log: list[dict[str, Any]] = []

        for task_idx in range(self.max_tasks):
            lib_summary = "\n".join(
                f"- {s['task']}: {s['solution'][:100]}"
                for s in library[-10:]
            ) or "Empty"

            propose_prompt = (
                f"Goal: {ctx.user_message}\n\n"
                f"Completed skills:\n{lib_summary}\n\n"
                f"Propose task {task_idx + 1} (progressively harder). "
                "Reply with a single clear task description."
            )
            task = self._call_sync(
                self.proposer_agent, ctx, propose_prompt,
            )

            existing = None
            for s in library:
                if s["task"].lower() == task.lower():
                    existing = s["solution"]
                    break

            if existing:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Task {task_idx + 1}: reused from library.",
                )
                curriculum_log.append({
                    "task": _truncate(task), "solution": existing,
                    "success": True, "from_library": True,
                })
                continue

            solve_prompt = (
                f"Goal: {ctx.user_message}\n\n"
                f"Available skills:\n{lib_summary}\n\n"
                f"Task: {_truncate(task)}\n\nSolve this task."
            )
            solution = self._call_sync(
                self.solver_agent, ctx, solve_prompt,
            )
            success = self.success_fn(solution)

            if success:
                library.append({"task": task, "solution": _truncate(solution)})

            curriculum_log.append({
                "task": _truncate(task), "solution": _truncate(solution),
                "success": success, "from_library": False,
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Task {task_idx + 1}: {'PASS' if success else 'FAIL'} — {task[:80]}",
                data={
                    "task_index": task_idx + 1,
                    "success": success,
                },
            )

        final = (
            f"Curriculum complete. {len(library)} skills acquired.\n"
            + "\n".join(
                f"- {s['task']}" for s in library
            )
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "tasks": curriculum_log[-20:],
                "library_size": len(library),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Propose → solve → store → repeat (async)."""
        if not (self.proposer_agent and self.solver_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires proposer_agent and solver_agent.",
            )
            return

        library = self._get_library(ctx.session)
        curriculum_log: list[dict[str, Any]] = []

        for task_idx in range(self.max_tasks):
            lib_summary = "\n".join(
                f"- {s['task']}: {s['solution'][:100]}"
                for s in library[-10:]
            ) or "Empty"

            propose_prompt = (
                f"Goal: {ctx.user_message}\n\n"
                f"Completed skills:\n{lib_summary}\n\n"
                f"Propose task {task_idx + 1} (progressively harder)."
            )
            task = await self._call_async(
                self.proposer_agent, ctx, propose_prompt,
            )

            existing = None
            for s in library:
                if s["task"].lower() == task.lower():
                    existing = s["solution"]
                    break

            if existing:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Task {task_idx + 1}: reused from library.",
                )
                curriculum_log.append({
                    "task": task, "solution": existing,
                    "success": True, "from_library": True,
                })
                continue

            solve_prompt = (
                f"Goal: {ctx.user_message}\n\n"
                f"Available skills:\n{lib_summary}\n\n"
                f"Task: {_truncate(task)}\n\nSolve this task."
            )
            solution = await self._call_async(
                self.solver_agent, ctx, solve_prompt,
            )
            success = self.success_fn(solution)

            if success:
                library.append({"task": task, "solution": _truncate(solution)})

            curriculum_log.append({
                "task": task, "solution": _truncate(solution),
                "success": success, "from_library": False,
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Task {task_idx + 1}: {'PASS' if success else 'FAIL'}",
                data={"task_index": task_idx + 1, "success": success},
            )

        final = (
            f"Curriculum complete. {len(library)} skills acquired.\n"
            + "\n".join(f"- {s['task']}" for s in library)
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "tasks": curriculum_log[-20:],
                "library_size": len(library),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# SelfConsistencyAgent — Wang et al., 2022
# ═══════════════════════════════════════════════════════════════════════════════


class SelfConsistencyAgent(BaseAgent):
    """Sample N reasoning paths from the same agent and majority-vote the answer.

    Unlike :class:`VotingAgent` (which uses *different* agents) or
    :class:`BestOfNAgent` (which uses an LLM judge), Self-Consistency samples
    from the **same** agent multiple times and picks the most frequent final
    answer via an ``extract_fn`` + majority vote.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The agent to sample from (called ``n_samples`` times).
        n_samples: Number of independent reasoning paths.
        extract_fn: Callable that extracts the "answer" from full output.
            Defaults to stripping whitespace and lowering.
        max_workers: ThreadPoolExecutor parallelism.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        n_samples: int = 5,
        extract_fn: Callable[[str], str] | None = None,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.n_samples = max(n_samples, 1)
        self.extract_fn = extract_fn or (lambda s: s.strip().lower())
        self.max_workers = max_workers
        self.result_key = result_key

    # -- helpers --

    def _sample_sync(
        self, ctx: InvocationContext, idx: int,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in self.agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _sample_async(
        self, ctx: InvocationContext, idx: int,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _vote(self, outputs: list[str]) -> tuple[str, str]:
        """Return (winning_extracted, first_full_output_that_matched)."""
        extracted = [self.extract_fn(o) for o in outputs]
        counter: collections.Counter[str] = collections.Counter(extracted)
        winner = counter.most_common(1)[0][0]
        for ext, full in zip(extracted, outputs):
            if ext == winner:
                return winner, full
        return winner, outputs[0]

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        outputs: list[str] = []
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, self.n_samples),
        ) as pool:
            futures = {
                pool.submit(self._sample_sync, ctx, i): i
                for i in range(self.n_samples)
            }
            for fut in as_completed(futures):
                events, output = fut.result(timeout=_FUTURE_TIMEOUT)
                for ev in events:
                    yield ev
                outputs.append(output)

        winner_ext, winner_full = self._vote(outputs)
        yield Event(
            EventType.AGENT_MESSAGE, self.name, winner_full,
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_samples": self.n_samples,
                "outputs": [_truncate(o) for o in outputs],
                "winner": winner_ext,
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        tasks = [self._sample_async(ctx, i) for i in range(self.n_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.error("[%s] Sample failed: %s", self.name, r)
                continue
            events, output = r
            for ev in events:
                yield ev
            outputs.append(output)

        winner_ext, winner_full = self._vote(outputs)
        yield Event(
            EventType.AGENT_MESSAGE, self.name, winner_full,
        )
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_samples": self.n_samples,
                "outputs": [_truncate(o) for o in outputs],
                "winner": winner_ext,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MixtureOfAgentsAgent — Together AI / Wang et al., 2024
# ═══════════════════════════════════════════════════════════════════════════════


class MixtureOfAgentsAgent(BaseAgent):
    """Multi-layer Mixture-of-Agents: proposers → aggregators × N layers.

    Layer 1 runs all proposers in parallel.  Each subsequent layer receives
    **all** outputs from the previous layer and produces refined aggregations.
    After ``n_layers``, an optional ``final_agent`` synthesises the result.

    Args:
        name: Agent name.
        description: Human-readable description.
        proposer_agents: Agents that generate initial proposals (layer 1).
        aggregator_agents: Agents that refine proposals (layers 2-N).
        final_agent: Optional single agent for final synthesis.
        n_layers: Number of aggregation layers (≥ 1).
        max_workers: ThreadPoolExecutor parallelism.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        proposer_agents: list[BaseAgent] | None = None,
        aggregator_agents: list[BaseAgent] | None = None,
        final_agent: BaseAgent | None = None,
        n_layers: int = 2,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        all_subs: list[BaseAgent] = []
        all_subs.extend(proposer_agents or [])
        all_subs.extend(aggregator_agents or [])
        if final_agent:
            all_subs.append(final_agent)
        super().__init__(
            name=name, description=description, sub_agents=all_subs, **kwargs,
        )
        self.proposer_agents = list(proposer_agents or [])
        self.aggregator_agents = list(aggregator_agents or [])
        self.final_agent = final_agent
        self.n_layers = max(n_layers, 1)
        self.max_workers = max_workers
        self.result_key = result_key

    # -- helpers --

    def _run_agent_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _run_agent_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _build_agg_prompt(
        self, user_message: str, prev_outputs: list[str],
    ) -> str:
        formatted = "\n\n---\n\n".join(
            f"Proposal {i + 1}:\n{o}" for i, o in enumerate(prev_outputs)
        )
        return (
            f"Original task: {user_message}\n\n"
            f"Previous-layer outputs:\n{formatted}\n\n"
            "Synthesise and improve these outputs into a better response."
        )

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.proposer_agents:
            yield Event(EventType.ERROR, self.name, "No proposer_agents configured.")
            return

        # Layer 1 — proposers
        layer_outputs: list[str] = []
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(self.proposer_agents)),
        ) as pool:
            futures = {
                pool.submit(
                    self._run_agent_sync, agent, ctx, ctx.user_message,
                ): agent.name
                for agent in self.proposer_agents
            }
            for fut in as_completed(futures):
                events, output = fut.result(timeout=_FUTURE_TIMEOUT)
                for ev in events:
                    yield ev
                layer_outputs.append(output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Layer 1 (proposers): {len(layer_outputs)} outputs",
        )

        # Layers 2..N — aggregators
        agents_for_agg = self.aggregator_agents or self.proposer_agents
        for layer_idx in range(2, self.n_layers + 1):
            agg_prompt = self._build_agg_prompt(ctx.user_message, layer_outputs)
            new_outputs: list[str] = []
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(agents_for_agg)),
            ) as pool:
                futures = {
                    pool.submit(
                        self._run_agent_sync, agent, ctx, agg_prompt,
                    ): agent.name
                    for agent in agents_for_agg
                }
                for fut in as_completed(futures):
                    events, output = fut.result(timeout=_FUTURE_TIMEOUT)
                    for ev in events:
                        yield ev
                    new_outputs.append(output)

            layer_outputs = new_outputs
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Layer {layer_idx}: {len(layer_outputs)} outputs",
            )

        # Final synthesis
        if self.final_agent:
            final_prompt = self._build_agg_prompt(ctx.user_message, layer_outputs)
            events, final_output = self._run_agent_sync(
                self.final_agent, ctx, final_prompt,
            )
            for ev in events:
                yield ev
            layer_outputs = [final_output]

        winner = layer_outputs[0] if layer_outputs else ""
        yield Event(EventType.AGENT_MESSAGE, self.name, winner)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_layers": self.n_layers,
                "final_output": winner,
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.proposer_agents:
            yield Event(EventType.ERROR, self.name, "No proposer_agents configured.")
            return

        # Layer 1 — proposers
        tasks = [
            self._run_agent_async(agent, ctx, ctx.user_message)
            for agent in self.proposer_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        layer_outputs: list[str] = []
        for r in results:
            if isinstance(r, BaseException):
                logger.error("[%s] Proposer failed: %s", self.name, r)
                continue
            events, output = r
            for ev in events:
                yield ev
            layer_outputs.append(output)

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Layer 1 (proposers): {len(layer_outputs)} outputs",
        )

        agents_for_agg = self.aggregator_agents or self.proposer_agents
        for layer_idx in range(2, self.n_layers + 1):
            agg_prompt = self._build_agg_prompt(ctx.user_message, layer_outputs)
            tasks = [
                self._run_agent_async(agent, ctx, agg_prompt)
                for agent in agents_for_agg
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            layer_outputs = []
            for r in results:
                if isinstance(r, BaseException):
                    logger.error("[%s] Aggregator failed: %s", self.name, r)
                    continue
                events, output = r
                for ev in events:
                    yield ev
                layer_outputs.append(output)

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Layer {layer_idx}: {len(layer_outputs)} outputs",
            )

        if self.final_agent:
            final_prompt = self._build_agg_prompt(ctx.user_message, layer_outputs)
            events, final_output = await self._run_agent_async(
                self.final_agent, ctx, final_prompt,
            )
            for ev in events:
                yield ev
            layer_outputs = [final_output]

        winner = layer_outputs[0] if layer_outputs else ""
        yield Event(EventType.AGENT_MESSAGE, self.name, winner)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_layers": self.n_layers,
                "final_output": winner,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# StepBackAgent — Zheng et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class StepBackAgent(BaseAgent):
    """Step-Back Prompting: abstract first, then reason with the abstraction.

    Phase 1 — the ``abstractor`` generates a higher-level question.
    Phase 2 — the ``reasoner`` answers the original question using the
    abstraction as grounding context.

    Args:
        name: Agent name.
        description: Human-readable description.
        abstractor_agent: Agent that produces the high-level abstraction.
        reasoner_agent: Agent that answers with the abstraction as context.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        abstractor_agent: BaseAgent | None = None,
        reasoner_agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (abstractor_agent, reasoner_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.abstractor_agent = abstractor_agent
        self.reasoner_agent = reasoner_agent
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.abstractor_agent and self.reasoner_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires abstractor_agent and reasoner_agent.",
            )
            return

        # Phase 1 — abstraction
        abs_prompt = (
            f"Question: {ctx.user_message}\n\n"
            "What is the higher-level principle, concept, or abstraction behind "
            "this question? Formulate a more general question that captures the "
            "underlying reasoning needed."
        )
        abs_events, abstraction = self._call_sync(
            self.abstractor_agent, ctx, abs_prompt,
        )
        for ev in abs_events:
            yield ev
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Abstraction: {abstraction[:200]}",
        )

        # Phase 2 — grounded reasoning
        reason_prompt = (
            f"High-level abstraction:\n{_truncate(abstraction)}\n\n"
            f"Original question: {ctx.user_message}\n\n"
            "Using the abstraction above as context, provide a thorough answer "
            "to the original question."
        )
        reason_events, answer = self._call_sync(
            self.reasoner_agent, ctx, reason_prompt,
        )
        for ev in reason_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "abstraction": _truncate(abstraction),
                "answer": _truncate(answer),
                "completed": True,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.abstractor_agent and self.reasoner_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires abstractor_agent and reasoner_agent.",
            )
            return

        abs_prompt = (
            f"Question: {ctx.user_message}\n\n"
            "What is the higher-level principle, concept, or abstraction behind "
            "this question? Formulate a more general question that captures the "
            "underlying reasoning needed."
        )
        abs_events, abstraction = await self._call_async(
            self.abstractor_agent, ctx, abs_prompt,
        )
        for ev in abs_events:
            yield ev
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Abstraction: {abstraction[:200]}",
        )

        reason_prompt = (
            f"High-level abstraction:\n{_truncate(abstraction)}\n\n"
            f"Original question: {ctx.user_message}\n\n"
            "Using the abstraction above as context, provide a thorough answer "
            "to the original question."
        )
        reason_events, answer = await self._call_async(
            self.reasoner_agent, ctx, reason_prompt,
        )
        for ev in reason_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "abstraction": _truncate(abstraction),
                "answer": _truncate(answer),
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# OrchestratorWorkerAgent — Anthropic MCP / OpenAI Agents SDK pattern
# ═══════════════════════════════════════════════════════════════════════════════


class OrchestratorWorkerAgent(BaseAgent):
    """Iterative orchestration loop: plan → delegate → evaluate → re-plan.

    Unlike :class:`SupervisorAgent` (flat, single-round delegation) or
    :class:`PlannerAgent` (static DAG), this orchestrator maintains an
    **interactive loop**: it asks the LLM to pick a worker and formulate a
    sub-task, executes the worker, feeds the result back, and repeats until
    the LLM declares the task complete or ``max_rounds`` is reached.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Worker agents to delegate to.
        model: LLM model for orchestration decisions.
        provider: LLM provider.
        api_key: Optional API key.
        max_rounds: Maximum orchestration rounds.
        completion_keyword: Keyword the LLM uses to signal completion.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        model: str | None = None,
        provider: str = "google",
        api_key: str | None = None,
        max_rounds: int = 10,
        completion_keyword: str = "TASK_COMPLETE",
        result_key: str | None = None,
        **service_kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents,
        )
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._service_kwargs = service_kwargs
        self._service: Any = None
        self.max_rounds = max_rounds
        self.completion_keyword = completion_keyword
        self.result_key = result_key

    @property
    def service(self) -> Any:
        """Lazily initialise the LLM connector."""
        if self._service is None:
            with _SERVICE_INIT_LOCK:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model, self._api_key,
                        **self._service_kwargs,
                    )
        return self._service

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self.service.generate_completion(messages=messages)
        return (response.get("text") or response.get("content") or "").strip()

    async def _call_llm_async(self, messages: list[dict[str, str]]) -> str:
        svc = self.service
        if hasattr(svc, "generate_completion_async"):
            response = await svc.generate_completion_async(messages=messages)
        else:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: svc.generate_completion(messages=messages),
            )
        return (response.get("text") or response.get("content") or "").strip()

    def _run_worker_sync(
        self, agent: BaseAgent, ctx: InvocationContext, task: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=task,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _run_worker_async(
        self, agent: BaseAgent, ctx: InvocationContext, task: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=task,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _build_system_prompt(self) -> str:
        agent_list = "\n".join(
            f"- {a.name}: {a.description or 'No description'}"
            for a in self.sub_agents
        )
        return (
            "You are an orchestrator. You have these workers:\n"
            f"{agent_list}\n\n"
            "On each turn, reply with a JSON object:\n"
            '{"worker": "<agent_name>", "task": "<sub-task description>"}\n'
            f"When the overall task is done, reply with: {self.completion_keyword}"
        )

    def _parse_delegation(self, llm_output: str) -> tuple[str, str] | None:
        """Parse the worker name and task from LLM JSON output."""
        if self.completion_keyword in llm_output:
            return None
        try:
            data = json.loads(llm_output)
            return data.get("worker", ""), data.get("task", "")
        except (json.JSONDecodeError, AttributeError):
            # Try to extract from text
            for agent in self.sub_agents:
                if agent.name in llm_output:
                    return agent.name, llm_output
            return None

    def _find_worker(self, worker_name: str) -> BaseAgent | None:
        for agent in self.sub_agents:
            if agent.name == worker_name:
                return agent
        return None

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No worker agents configured.")
            return

        system_prompt = self._build_system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ctx.user_message},
        ]
        rounds_log: list[dict[str, Any]] = []
        last_output = ""

        for round_idx in range(1, self.max_rounds + 1):
            llm_reply = self._call_llm(messages)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_idx}: {llm_reply[:200]}",
            )

            delegation = self._parse_delegation(llm_reply)
            if delegation is None:
                # Task complete
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Orchestrator declared task complete at round {round_idx}.",
                )
                break

            worker_name, sub_task = delegation
            worker = self._find_worker(worker_name)
            if worker is None:
                messages.append({"role": "assistant", "content": llm_reply})
                messages.append({
                    "role": "user",
                    "content": f"Worker '{worker_name}' not found. Choose from the list.",
                })
                continue

            events, output = self._run_worker_sync(worker, ctx, sub_task)
            for ev in events:
                yield ev
            last_output = output

            rounds_log.append({
                "round": round_idx,
                "worker": worker_name,
                "task": sub_task,
                "output": output[:500],
            })

            messages.append({"role": "assistant", "content": llm_reply})
            messages.append({
                "role": "user",
                "content": f"Worker '{worker_name}' returned:\n{_truncate(output)}\n\n"
                "What's next?",
            })
            messages = messages[:2] + messages[2:][-20:]

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rounds_log,
                "total_rounds": len(rounds_log),
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No worker agents configured.")
            return

        system_prompt = self._build_system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ctx.user_message},
        ]
        rounds_log: list[dict[str, Any]] = []
        last_output = ""

        for round_idx in range(1, self.max_rounds + 1):
            llm_reply = await self._call_llm_async(messages)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_idx}: {llm_reply[:200]}",
            )

            delegation = self._parse_delegation(llm_reply)
            if delegation is None:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Orchestrator declared task complete at round {round_idx}.",
                )
                break

            worker_name, sub_task = delegation
            worker = self._find_worker(worker_name)
            if worker is None:
                messages.append({"role": "assistant", "content": llm_reply})
                messages.append({
                    "role": "user",
                    "content": f"Worker '{worker_name}' not found. Choose from the list.",
                })
                continue

            events, output = await self._run_worker_async(worker, ctx, sub_task)
            for ev in events:
                yield ev
            last_output = output

            rounds_log.append({
                "round": round_idx,
                "worker": worker_name,
                "task": sub_task,
                "output": output[:500],
            })

            messages.append({"role": "assistant", "content": llm_reply})
            messages.append({
                "role": "user",
                "content": f"Worker '{worker_name}' returned:\n{_truncate(output)}\n\n"
                "What's next?",
            })
            messages = messages[:2] + messages[2:][-20:]

        if last_output:
            yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rounds_log,
                "total_rounds": len(rounds_log),
                "completed": True,
            })


class SelfRefineAgent(BaseAgent):
    """Self-Refine: iterative generate → critique → refine loop.

    A single agent generates an initial output, then critiques it (producing
    structured feedback), then refines using the feedback.  The loop repeats
    until the critique contains ``stop_phrase`` or ``max_iterations`` is hit.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: The agent used for generate, critique, and refine.
        max_iterations: Maximum refine rounds.
        stop_phrase: Phrase in critique that signals "good enough".
        critique_instruction: System instruction for the critique step.
        refine_instruction: System instruction for the refine step.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        max_iterations: int = 3,
        stop_phrase: str = "NO_ISSUES_FOUND",
        critique_instruction: str = "",
        refine_instruction: str = "",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.max_iterations = max(max_iterations, 1)
        self.stop_phrase = stop_phrase
        self.critique_instruction = critique_instruction or (
            "Critique the following output. List specific issues and "
            f"improvements. If no issues, reply with: {stop_phrase}"
        )
        self.refine_instruction = refine_instruction or (
            "Refine the output based on the critique below."
        )
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        # Initial generation
        current = self._call_sync(self.agent, ctx, ctx.user_message)
        yield Event(
            EventType.STATE_UPDATE, self.name, f"Initial output: {current[:200]}",
        )

        iterations_log: list[dict[str, str]] = []

        for iteration in range(1, self.max_iterations + 1):
            # Critique
            critique_prompt = (
                f"{self.critique_instruction}\n\nOutput:\n{_truncate(current)}"
            )
            critique = self._call_sync(self.agent, ctx, critique_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration} critique: {critique[:200]}",
            )

            if self.stop_phrase in critique:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Stop phrase detected at iteration {iteration}.",
                )
                iterations_log.append({
                    "iteration": str(iteration),
                    "critique": _truncate(critique),
                    "refined": _truncate(current),
                    "stopped": "true",
                })
                break

            # Refine
            refine_prompt = (
                f"{self.refine_instruction}\n\n"
                f"Current output:\n{_truncate(current)}\n\n"
                f"Critique:\n{_truncate(critique)}\n\n"
                "Produce an improved version."
            )
            current = self._call_sync(self.agent, ctx, refine_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration} refined: {current[:200]}",
            )
            iterations_log.append({
                "iteration": str(iteration),
                "critique": _truncate(critique),
                "refined": _truncate(current),
                "stopped": "false",
            })

        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "iterations": iterations_log,
                "final": current,
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        current = await self._call_async(self.agent, ctx, ctx.user_message)
        yield Event(
            EventType.STATE_UPDATE, self.name, f"Initial output: {current[:200]}",
        )

        iterations_log: list[dict[str, str]] = []

        for iteration in range(1, self.max_iterations + 1):
            critique_prompt = (
                f"{self.critique_instruction}\n\nOutput:\n{_truncate(current)}"
            )
            critique = await self._call_async(self.agent, ctx, critique_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration} critique: {critique[:200]}",
            )

            if self.stop_phrase in critique:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Stop phrase detected at iteration {iteration}.",
                )
                iterations_log.append({
                    "iteration": str(iteration),
                    "critique": _truncate(critique),
                    "refined": _truncate(current),
                    "stopped": "true",
                })
                break

            refine_prompt = (
                f"{self.refine_instruction}\n\n"
                f"Current output:\n{_truncate(current)}\n\n"
                f"Critique:\n{_truncate(critique)}\n\n"
                "Produce an improved version."
            )
            current = await self._call_async(self.agent, ctx, refine_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {iteration} refined: {current[:200]}",
            )
            iterations_log.append({
                "iteration": str(iteration),
                "critique": _truncate(critique),
                "refined": _truncate(current),
                "stopped": "false",
            })

        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "iterations": iterations_log,
                "final": current,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# BacktrackingAgent — Inspired by LATS / Zhou et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class BacktrackingAgent(BaseAgent):
    """Pipeline with validation and backtracking on failure.

    Runs ``sub_agents`` sequentially.  After each agent, ``validate_fn``
    checks the output.  If validation fails, the agent **backtracks** to
    ``backtrack_to`` (default: restart from the beginning) and retries with
    knowledge of prior failed attempts.  After ``max_retries`` backtrack
    cycles the pipeline gives up.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Ordered pipeline of agents to execute.
        validate_fn: Callable[[str], bool] — True if step output is valid.
        backtrack_to: Index of the agent to rewind to on failure (0-based).
        max_retries: Maximum number of full backtrack cycles.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        validate_fn: Callable[[str], bool] | None = None,
        backtrack_to: int = 0,
        max_retries: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description, sub_agents=sub_agents, **kwargs,
        )
        self.validate_fn = validate_fn or (lambda _s: True)
        self.backtrack_to = backtrack_to
        self.max_retries = max(max_retries, 0)
        self.result_key = result_key

    def _run_agent_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _run_agent_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        failed_attempts: list[dict[str, str]] = []
        last_output = ""

        for retry in range(self.max_retries + 1):
            success = True
            current_msg = ctx.user_message

            if failed_attempts:
                failed_summary = "\n".join(
                    f"- Step {f['step']}: failed ({f['reason'][:100]})"
                    for f in failed_attempts[-5:]
                )
                current_msg = (
                    f"{ctx.user_message}\n\n"
                    f"Previous failed attempts:\n{failed_summary}\n\n"
                    "Avoid the same mistakes."
                )

            start_idx = self.backtrack_to if retry > 0 else 0

            for i in range(start_idx, len(self.sub_agents)):
                agent = self.sub_agents[i]
                events, output = self._run_agent_sync(agent, ctx, current_msg)
                for ev in events:
                    yield ev
                current_msg = output
                last_output = output

                if not self.validate_fn(output):
                    failed_attempts.append({
                        "step": agent.name,
                        "reason": output[:200],
                        "retry": str(retry),
                    })
                    yield Event(
                        EventType.STATE_UPDATE, self.name,
                        f"Validation failed at '{agent.name}', "
                        f"backtracking (retry {retry + 1}/{self.max_retries}).",
                    )
                    success = False
                    break

            if success:
                break

        yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "failed_attempts": failed_attempts,
                "total_retries": len(failed_attempts),
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return

        failed_attempts: list[dict[str, str]] = []
        last_output = ""

        for retry in range(self.max_retries + 1):
            success = True
            current_msg = ctx.user_message

            if failed_attempts:
                failed_summary = "\n".join(
                    f"- Step {f['step']}: failed ({f['reason'][:100]})"
                    for f in failed_attempts[-5:]
                )
                current_msg = (
                    f"{ctx.user_message}\n\n"
                    f"Previous failed attempts:\n{failed_summary}\n\n"
                    "Avoid the same mistakes."
                )

            start_idx = self.backtrack_to if retry > 0 else 0

            for i in range(start_idx, len(self.sub_agents)):
                agent = self.sub_agents[i]
                events, output = await self._run_agent_async(
                    agent, ctx, current_msg,
                )
                for ev in events:
                    yield ev
                current_msg = output
                last_output = output

                if not self.validate_fn(output):
                    failed_attempts.append({
                        "step": agent.name,
                        "reason": output[:200],
                        "retry": str(retry),
                    })
                    yield Event(
                        EventType.STATE_UPDATE, self.name,
                        f"Validation failed at '{agent.name}', "
                        f"backtracking (retry {retry + 1}/{self.max_retries}).",
                    )
                    success = False
                    break

            if success:
                break

        yield Event(EventType.AGENT_MESSAGE, self.name, last_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "failed_attempts": failed_attempts,
                "total_retries": len(failed_attempts),
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ChainOfDensityAgent — Adams et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class ChainOfDensityAgent(BaseAgent):
    """Iterative densification: rewrite content adding missing info each round.

    Each round: (1) the agent identifies missing entities/information,
    (2) rewrites to incorporate them **without increasing length**.
    Continues for ``n_rounds`` or until the agent signals ``stop_phrase``.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Agent used for all densification rounds.
        n_rounds: Number of densification iterations.
        stop_phrase: Phrase that signals "fully dense".
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        n_rounds: int = 3,
        stop_phrase: str = "FULLY_DENSE",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.n_rounds = max(n_rounds, 1)
        self.stop_phrase = stop_phrase
        self.result_key = result_key

    def _call_sync(
        self, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        # Initial generation
        current = self._call_sync(ctx, ctx.user_message)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Initial (len={len(current)}): {current[:200]}",
        )
        rounds_log: list[dict[str, Any]] = []

        for round_idx in range(1, self.n_rounds + 1):
            dense_prompt = (
                f"Current text:\n{_truncate(current)}\n\n"
                "Identify key entities, facts, or details that are missing. "
                "Rewrite the text to incorporate them WITHOUT increasing the "
                "overall length. Make it denser and more informative.\n"
                f"If the text is already fully dense, reply: {self.stop_phrase}"
            )
            result = self._call_sync(ctx, dense_prompt)

            if self.stop_phrase in result:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Fully dense at round {round_idx}.",
                )
                rounds_log.append({
                    "round": round_idx, "stopped": True, "length": len(current),
                })
                break

            current = result
            rounds_log.append({
                "round": round_idx, "stopped": False, "length": len(current),
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_idx} (len={len(current)}): {current[:200]}",
            )

        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rounds_log,
                "final_length": len(current),
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        current = await self._call_async(ctx, ctx.user_message)
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Initial (len={len(current)}): {current[:200]}",
        )
        rounds_log: list[dict[str, Any]] = []

        for round_idx in range(1, self.n_rounds + 1):
            dense_prompt = (
                f"Current text:\n{_truncate(current)}\n\n"
                "Identify key entities, facts, or details that are missing. "
                "Rewrite the text to incorporate them WITHOUT increasing the "
                "overall length. Make it denser and more informative.\n"
                f"If the text is already fully dense, reply: {self.stop_phrase}"
            )
            result = await self._call_async(ctx, dense_prompt)

            if self.stop_phrase in result:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Fully dense at round {round_idx}.",
                )
                rounds_log.append({
                    "round": round_idx, "stopped": True, "length": len(current),
                })
                break

            current = result
            rounds_log.append({
                "round": round_idx, "stopped": False, "length": len(current),
            })
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {round_idx} (len={len(current)}): {current[:200]}",
            )

        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rounds_log,
                "final_length": len(current),
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MediatorAgent — Conflict resolution / Abdelnabi et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class MediatorAgent(BaseAgent):
    """Mediator: run N agents, then a neutral mediator synthesises a compromise.

    Unlike :class:`DebateAgent` (judge picks a winner) or :class:`ConsensusAgent`
    (iterate until convergence), the mediator explicitly **analyses** the
    differences between proposals and produces a **compromise** that
    incorporates the best of each.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Agents that produce competing proposals.
        mediator_agent: The neutral agent that synthesises a compromise.
        max_workers: ThreadPoolExecutor parallelism.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        mediator_agent: BaseAgent | None = None,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        all_subs = list(sub_agents or [])
        if mediator_agent and mediator_agent not in all_subs:
            all_subs.append(mediator_agent)
        super().__init__(
            name=name, description=description, sub_agents=all_subs, **kwargs,
        )
        self._proposal_agents = list(sub_agents or [])
        self.mediator_agent = mediator_agent
        self.max_workers = max_workers
        self.result_key = result_key

    def _run_agent_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _run_agent_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _build_mediation_prompt(
        self, user_message: str, proposals: dict[str, str],
    ) -> str:
        formatted = "\n\n---\n\n".join(
            f"**{name}**:\n{text}" for name, text in proposals.items()
        )
        return (
            f"Original task: {user_message}\n\n"
            f"The following agents produced different proposals:\n\n"
            f"{formatted}\n\n"
            "As a neutral mediator:\n"
            "1. Identify points of agreement and disagreement.\n"
            "2. Synthesise a compromise solution that incorporates the "
            "best aspects of each proposal.\n"
            "3. Explain briefly why this compromise is superior."
        )

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self._proposal_agents:
            yield Event(EventType.ERROR, self.name, "No proposal agents configured.")
            return
        if not self.mediator_agent:
            yield Event(EventType.ERROR, self.name, "No mediator_agent configured.")
            return

        # Gather proposals in parallel
        proposals: dict[str, str] = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(self._proposal_agents)),
        ) as pool:
            futures = {
                pool.submit(
                    self._run_agent_sync, agent, ctx, ctx.user_message,
                ): agent.name
                for agent in self._proposal_agents
            }
            for fut in as_completed(futures):
                agent_name = futures[fut]
                events, output = fut.result(timeout=_FUTURE_TIMEOUT)
                for ev in events:
                    yield ev
                proposals[agent_name] = output

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Collected {len(proposals)} proposals.",
        )

        # Mediation
        mediation_prompt = self._build_mediation_prompt(
            ctx.user_message, proposals,
        )
        med_events, compromise = self._run_agent_sync(
            self.mediator_agent, ctx, mediation_prompt,
        )
        for ev in med_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, compromise)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "proposals": {k: _truncate(v) for k, v in proposals.items()},
                "compromise": compromise,
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self._proposal_agents:
            yield Event(EventType.ERROR, self.name, "No proposal agents configured.")
            return
        if not self.mediator_agent:
            yield Event(EventType.ERROR, self.name, "No mediator_agent configured.")
            return

        tasks = [
            self._run_agent_async(agent, ctx, ctx.user_message)
            for agent in self._proposal_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        proposals: dict[str, str] = {}
        for agent, r in zip(self._proposal_agents, results):
            if isinstance(r, BaseException):
                continue
            events, output = r
            for ev in events:
                yield ev
            proposals[agent.name] = output

        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Collected {len(proposals)} proposals.",
        )

        mediation_prompt = self._build_mediation_prompt(
            ctx.user_message, proposals,
        )
        med_events, compromise = await self._run_agent_async(
            self.mediator_agent, ctx, mediation_prompt,
        )
        for ev in med_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, compromise)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "proposals": {k: _truncate(v) for k, v in proposals.items()},
                "compromise": compromise,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# DivideAndConquerAgent — Khot et al., 2022 (Decomposed Prompting)
# ═══════════════════════════════════════════════════════════════════════════════


class DivideAndConquerAgent(BaseAgent):
    """Recursive divide-and-conquer: split → recurse → merge.

    The ``splitter_agent`` decomposes a problem.  If a sub-problem is
    simple (according to ``is_base_case``), the ``solver_agent`` handles it
    directly.  Otherwise, the agent recursively divides further up to
    ``max_depth``.  Results are merged bottom-up by the ``merger_agent``.

    Args:
        name: Agent name.
        description: Human-readable description.
        splitter_agent: Agent that decomposes a problem into sub-problems.
        solver_agent: Agent that solves base-case sub-problems.
        merger_agent: Agent that merges sub-results bottom-up.
        is_base_case: Callable[[str], bool] — True if the problem is simple.
        max_depth: Maximum recursion depth.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        splitter_agent: BaseAgent | None = None,
        solver_agent: BaseAgent | None = None,
        merger_agent: BaseAgent | None = None,
        is_base_case: Callable[[str], bool] | None = None,
        max_depth: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (splitter_agent, solver_agent, merger_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.splitter_agent = splitter_agent
        self.solver_agent = solver_agent
        self.merger_agent = merger_agent
        self.is_base_case = is_base_case or (lambda _p: len(_p) < 100)
        self.max_depth = max(max_depth, 1)
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    @staticmethod
    def _parse_subproblems(text: str) -> list[str]:
        """Extract numbered sub-problems from agent output."""
        items = _LIST_ITEM_RE.findall(text)
        if items:
            return [i.strip() for i in items if i.strip()]
        return [line.strip() for line in text.strip().splitlines() if line.strip()]

    def _solve_sync(
        self, ctx: InvocationContext, problem: str, depth: int,
        events_out: list[Event],
    ) -> str:
        """Recursive sync solver."""
        if depth >= self.max_depth or self.is_base_case(problem):
            result = self._call_sync(self.solver_agent, ctx, problem)
            events_out.append(Event(
                EventType.STATE_UPDATE, self.name,
                f"Base case (depth={depth}): {problem[:80]}",
            ))
            return result

        # Split
        split_prompt = (
            f"Problem: {problem}\n\n"
            "Decompose into 2-4 smaller sub-problems. "
            "Reply with a numbered list."
        )
        raw = self._call_sync(self.splitter_agent, ctx, split_prompt)
        subs = self._parse_subproblems(raw)

        if len(subs) <= 1:
            # Cannot split further → solve directly
            return self._call_sync(self.solver_agent, ctx, problem)

        events_out.append(Event(
            EventType.STATE_UPDATE, self.name,
            f"Split (depth={depth}): {len(subs)} sub-problems",
        ))

        # Recurse
        sub_results: list[str] = []
        for sp in subs:
            sub_results.append(
                self._solve_sync(ctx, sp, depth + 1, events_out),
            )

        # Merge
        merge_prompt = (
            f"Original problem: {problem}\n\n"
            "Sub-problem results:\n"
            + "\n\n".join(
                f"{i + 1}. {_truncate(r)}" for i, r in enumerate(sub_results)
            )
            + "\n\nMerge these results into a single coherent answer."
        )
        merged = self._call_sync(self.merger_agent, ctx, merge_prompt)
        events_out.append(Event(
            EventType.STATE_UPDATE, self.name,
            f"Merged (depth={depth}): {merged[:100]}",
        ))
        return merged

    async def _solve_async(
        self, ctx: InvocationContext, problem: str, depth: int,
        events_out: list[Event],
    ) -> str:
        """Recursive async solver."""
        if depth >= self.max_depth or self.is_base_case(problem):
            result = await self._call_async(self.solver_agent, ctx, problem)
            events_out.append(Event(
                EventType.STATE_UPDATE, self.name,
                f"Base case (depth={depth}): {problem[:80]}",
            ))
            return result

        split_prompt = (
            f"Problem: {problem}\n\n"
            "Decompose into 2-4 smaller sub-problems. "
            "Reply with a numbered list."
        )
        raw = await self._call_async(self.splitter_agent, ctx, split_prompt)
        subs = self._parse_subproblems(raw)

        if len(subs) <= 1:
            return await self._call_async(self.solver_agent, ctx, problem)

        events_out.append(Event(
            EventType.STATE_UPDATE, self.name,
            f"Split (depth={depth}): {len(subs)} sub-problems",
        ))

        # Recurse in parallel
        tasks = [
            self._solve_async(ctx, sp, depth + 1, events_out) for sp in subs
        ]
        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        sub_results = [r if not isinstance(r, BaseException) else "" for r in sub_results]

        merge_prompt = (
            f"Original problem: {problem}\n\n"
            "Sub-problem results:\n"
            + "\n\n".join(
                f"{i + 1}. {_truncate(r)}" for i, r in enumerate(sub_results)
            )
            + "\n\nMerge these results into a single coherent answer."
        )
        merged = await self._call_async(self.merger_agent, ctx, merge_prompt)
        events_out.append(Event(
            EventType.STATE_UPDATE, self.name,
            f"Merged (depth={depth}): {merged[:100]}",
        ))
        return merged

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.splitter_agent and self.solver_agent and self.merger_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires splitter_agent, solver_agent, and merger_agent.",
            )
            return

        events_out: list[Event] = []
        result = self._solve_sync(ctx, ctx.user_message, 0, events_out)
        for ev in events_out:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, result)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "result": _truncate(result),
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.splitter_agent and self.solver_agent and self.merger_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires splitter_agent, solver_agent, and merger_agent.",
            )
            return

        events_out: list[Event] = []
        result = await self._solve_async(ctx, ctx.user_message, 0, events_out)
        for ev in events_out:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, result)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "result": _truncate(result),
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# BeamSearchAgent — Xie et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class BeamSearchAgent(BaseAgent):
    """Beam search over reasoning paths: expand → score → prune per step.

    Maintains ``beam_width`` candidate paths.  At each step, each beam is
    expanded into ``n_expansions`` candidates.  All candidates are scored
    with ``score_fn`` and pruned to the top ``beam_width``.  After
    ``n_steps``, the highest-scoring beam is returned.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Agent that expands each beam candidate.
        score_fn: Callable[[str], float] — score a candidate.
        beam_width: Number of beams to keep.
        n_expansions: Expansions per beam per step.
        n_steps: Number of expansion steps.
        max_workers: ThreadPoolExecutor parallelism.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        beam_width: int = 3,
        n_expansions: int = 2,
        n_steps: int = 3,
        max_workers: int = 4,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _s: 0.0)
        self.beam_width = max(beam_width, 1)
        self.n_expansions = max(n_expansions, 1)
        self.n_steps = max(n_steps, 1)
        self.max_workers = max_workers
        self.result_key = result_key

    def _expand_sync(
        self, ctx: InvocationContext, beam: str, expansion_idx: int,
    ) -> tuple[list[Event], str]:
        prompt = (
            f"Previous reasoning:\n{_truncate(beam)}\n\n"
            "Continue or improve this reasoning. "
            f"Produce variation #{expansion_idx + 1}."
        )
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in self.agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _expand_async(
        self, ctx: InvocationContext, beam: str, expansion_idx: int,
    ) -> tuple[list[Event], str]:
        prompt = (
            f"Previous reasoning:\n{_truncate(beam)}\n\n"
            "Continue or improve this reasoning. "
            f"Produce variation #{expansion_idx + 1}."
        )
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        # Initialise beams
        beams: list[tuple[float, str]] = [(0.0, ctx.user_message)]

        for step in range(1, self.n_steps + 1):
            candidates: list[tuple[float, str]] = []

            with ThreadPoolExecutor(
                max_workers=min(
                    self.max_workers,
                    len(beams) * self.n_expansions,
                    _MAX_DEFAULT_PARALLEL_WORKERS,
                ),
            ) as pool:
                futures: dict[Any, tuple[float, str]] = {}
                for score, beam_text in beams:
                    for exp_idx in range(self.n_expansions):
                        fut = pool.submit(
                            self._expand_sync, ctx, beam_text, exp_idx,
                        )
                        futures[fut] = (score, beam_text)

                for fut in as_completed(futures):
                    events, output = fut.result(timeout=_FUTURE_TIMEOUT)
                    for ev in events:
                        yield ev
                    new_score = self.score_fn(output)
                    candidates.append((new_score, output))

            # Prune to top beam_width
            candidates.sort(key=lambda c: c[0], reverse=True)
            beams = candidates[: self.beam_width]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}/{self.n_steps}: top score={beams[0][0]:.3f}, "
                f"beams={len(beams)}",
            )

        # Best beam
        best_score, best_output = beams[0]
        yield Event(EventType.AGENT_MESSAGE, self.name, best_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "n_steps": self.n_steps,
                "beam_width": self.beam_width,
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "No agent configured.")
            return

        beams: list[tuple[float, str]] = [(0.0, ctx.user_message)]

        for step in range(1, self.n_steps + 1):
            tasks = []
            for _score, beam_text in beams:
                for exp_idx in range(self.n_expansions):
                    tasks.append(
                        self._expand_async(ctx, beam_text, exp_idx),
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            candidates: list[tuple[float, str]] = []
            for r in results:
                if isinstance(r, BaseException):
                    logger.error("[%s] Beam expand failed: %s", self.name, r)
                    continue
                events, output = r
                for ev in events:
                    yield ev
                new_score = self.score_fn(output)
                candidates.append((new_score, output))

            candidates.sort(key=lambda c: c[0], reverse=True)
            beams = candidates[: self.beam_width]

            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}/{self.n_steps}: top score={beams[0][0]:.3f}, "
                f"beams={len(beams)}",
            )

        best_score, best_output = beams[0]
        yield Event(EventType.AGENT_MESSAGE, self.name, best_output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "n_steps": self.n_steps,
                "beam_width": self.beam_width,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# RephraseAndRespondAgent — Deng et al., 2023 (RaR)
# ═══════════════════════════════════════════════════════════════════════════════


class RephraseAndRespondAgent(BaseAgent):
    """Rephrase-and-Respond: improve query clarity before solving.

    Two phases: a ``rephrase_agent`` rewrites the user query for clarity
    and precision, then a ``solver_agent`` answers the improved
    formulation.  Unlike :class:`StepBackAgent` (which abstracts the
    conceptual level), RaR preserves the original intent—only the
    *wording* improves.

    Reference: Deng et al., *Rephrase and Respond*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        rephrase_agent: Agent that rewrites the question.
        solver_agent: Agent that answers the rephrased version.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        rephrase_agent: BaseAgent | None = None,
        solver_agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (rephrase_agent, solver_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.rephrase_agent = rephrase_agent
        self.solver_agent = solver_agent
        self.result_key = result_key

    # -- helpers --

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    # -- sync --

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.rephrase_agent and self.solver_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires rephrase_agent and solver_agent.",
            )
            return

        reph_prompt = (
            f"Rephrase the following question for maximum clarity and "
            f"precision.  Keep the same intent:\n\n{ctx.user_message}"
        )
        reph_events, rephrased = self._call_sync(
            self.rephrase_agent, ctx, reph_prompt,
        )
        for ev in reph_events:
            yield ev
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Rephrased: {rephrased[:200]}",
        )

        solve_events, answer = self._call_sync(
            self.solver_agent, ctx, rephrased,
        )
        for ev in solve_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rephrased": _truncate(rephrased),
                "answer": _truncate(answer),
                "completed": True,
            })

    # -- async --

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.rephrase_agent and self.solver_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires rephrase_agent and solver_agent.",
            )
            return

        reph_prompt = (
            f"Rephrase the following question for maximum clarity and "
            f"precision.  Keep the same intent:\n\n{ctx.user_message}"
        )
        reph_events, rephrased = await self._call_async(
            self.rephrase_agent, ctx, reph_prompt,
        )
        for ev in reph_events:
            yield ev
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Rephrased: {rephrased[:200]}",
        )

        solve_events, answer = await self._call_async(
            self.solver_agent, ctx, rephrased,
        )
        for ev in solve_events:
            yield ev

        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rephrased": _truncate(rephrased),
                "answer": _truncate(answer),
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# CumulativeReasoningAgent — Zhang et al., 2024
# ═══════════════════════════════════════════════════════════════════════════════


class CumulativeReasoningAgent(BaseAgent):
    """Cumulative Reasoning: propose → verify → accumulate → report.

    Three-role cyclic pipeline.  Each round the ``proposer_agent``
    generates a hypothesis, the ``verifier_agent`` accepts or rejects it,
    and accepted facts accumulate.  After *n_rounds* the
    ``reporter_agent`` synthesises all verified facts into a final answer.

    Unlike :class:`CoVeAgent` (which verifies fixed claims) the knowledge
    base *grows* incrementally across rounds.

    Reference: Zhang et al., *Cumulative Reasoning*, 2024.

    Args:
        name: Agent name.
        description: Human-readable description.
        proposer_agent: Generates new hypotheses each round.
        verifier_agent: Accepts or rejects proposals.
        reporter_agent: Synthesises final answer from accepted facts.
        n_rounds: Maximum proposal rounds.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        proposer_agent: BaseAgent | None = None,
        verifier_agent: BaseAgent | None = None,
        reporter_agent: BaseAgent | None = None,
        n_rounds: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (proposer_agent, verifier_agent, reporter_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.proposer_agent = proposer_agent
        self.verifier_agent = verifier_agent
        self.reporter_agent = reporter_agent
        self.n_rounds = n_rounds
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.proposer_agent and self.verifier_agent and self.reporter_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires proposer_agent, verifier_agent, and reporter_agent.",
            )
            return
        accepted_facts: list[str] = []
        for rnd in range(1, self.n_rounds + 1):
            facts_str = (
                "\n".join(f"- {f}" for f in accepted_facts)
                if accepted_facts else "(none yet)"
            )
            prop_prompt = (
                f"Question: {ctx.user_message}\n\n"
                f"Accepted facts so far:\n{facts_str}\n\n"
                "Propose a new hypothesis or fact that helps answer the question."
            )
            _, hypothesis = self._call_sync(self.proposer_agent, ctx, prop_prompt)
            ver_prompt = (
                f"Question: {ctx.user_message}\n\n"
                f"Accepted facts so far:\n{facts_str}\n\n"
                f"Proposed hypothesis: {_truncate(hypothesis)}\n\n"
                "Is this hypothesis correct and useful?  Reply ACCEPT or REJECT "
                "followed by a brief justification."
            )
            _, verdict = self._call_sync(self.verifier_agent, ctx, ver_prompt)
            accepted = "ACCEPT" in verdict.upper()
            if accepted:
                accepted_facts.append(_truncate(hypothesis))
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: {'ACCEPT' if accepted else 'REJECT'} — "
                f"{hypothesis[:120]}",
            )
        facts_str = (
            "\n".join(f"- {f}" for f in accepted_facts)
            if accepted_facts else "(none)"
        )
        rep_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Verified facts:\n{facts_str}\n\n"
            "Synthesise a comprehensive answer using only the verified facts."
        )
        rep_events, answer = self._call_sync(self.reporter_agent, ctx, rep_prompt)
        for ev in rep_events:
            yield ev
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "accepted_facts": accepted_facts,
                "answer": _truncate(answer),
                "n_rounds": self.n_rounds,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.proposer_agent and self.verifier_agent and self.reporter_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires proposer_agent, verifier_agent, and reporter_agent.",
            )
            return
        accepted_facts: list[str] = []
        for rnd in range(1, self.n_rounds + 1):
            facts_str = (
                "\n".join(f"- {f}" for f in accepted_facts)
                if accepted_facts else "(none yet)"
            )
            prop_prompt = (
                f"Question: {ctx.user_message}\n\n"
                f"Accepted facts so far:\n{facts_str}\n\n"
                "Propose a new hypothesis or fact that helps answer the question."
            )
            _, hypothesis = await self._call_async(
                self.proposer_agent, ctx, prop_prompt,
            )
            ver_prompt = (
                f"Question: {ctx.user_message}\n\n"
                f"Accepted facts so far:\n{facts_str}\n\n"
                f"Proposed hypothesis: {_truncate(hypothesis)}\n\n"
                "Is this hypothesis correct and useful?  Reply ACCEPT or REJECT "
                "followed by a brief justification."
            )
            _, verdict = await self._call_async(
                self.verifier_agent, ctx, ver_prompt,
            )
            accepted = "ACCEPT" in verdict.upper()
            if accepted:
                accepted_facts.append(_truncate(hypothesis))
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: {'ACCEPT' if accepted else 'REJECT'} — "
                f"{hypothesis[:120]}",
            )
        facts_str = (
            "\n".join(f"- {f}" for f in accepted_facts)
            if accepted_facts else "(none)"
        )
        rep_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Verified facts:\n{facts_str}\n\n"
            "Synthesise a comprehensive answer using only the verified facts."
        )
        rep_events, answer = await self._call_async(
            self.reporter_agent, ctx, rep_prompt,
        )
        for ev in rep_events:
            yield ev
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "accepted_facts": accepted_facts,
                "answer": _truncate(answer),
                "n_rounds": self.n_rounds,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MultiPersonaAgent — Wang et al., 2024 (Solo Performance Prompting)
# ═══════════════════════════════════════════════════════════════════════════════


class MultiPersonaAgent(BaseAgent):
    """Multi-Persona: one agent adopts multiple expert roles sequentially.

    A single ``agent`` is called once per persona with a role-specific
    prefix.  A ``synthesizer_agent`` (defaults to *agent*) merges all
    perspectives into a final answer.

    Unlike :class:`GroupChatAgent` (multiple distinct agents) this uses a
    single agent shifting perspective.

    Reference: Wang et al., *Solo Performance Prompting*, 2024.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Base agent that adopts each persona.
        personas: List of role descriptions.
        synthesizer_agent: Merges perspectives (defaults to *agent*).
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        personas: list[str] | None = None,
        synthesizer_agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        seen: set[int] = set()
        deduped: list[BaseAgent] = []
        for a in (agent, synthesizer_agent):
            if a and id(a) not in seen:
                seen.add(id(a))
                deduped.append(a)
        super().__init__(name=name, description=description, sub_agents=deduped, **kwargs)
        self.agent = agent
        self.personas = personas or ["domain expert", "devil's advocate", "pragmatist"]
        self.synthesizer_agent = synthesizer_agent
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        perspectives: list[tuple[str, str]] = []
        for persona in self.personas:
            p_prompt = (
                f"You are a {persona}.  Answer strictly from this perspective.\n\n"
                f"Question: {ctx.user_message}"
            )
            _, answer = self._call_sync(self.agent, ctx, p_prompt)
            perspectives.append((persona, answer))
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"[{persona}] {answer[:120]}",
            )
        parts = "\n\n".join(f"**{p}**: {_truncate(a)}" for p, a in perspectives)
        synth_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Perspectives:\n{parts}\n\n"
            "Synthesise all perspectives into one balanced, comprehensive answer."
        )
        synth_agent = self.synthesizer_agent or self.agent
        synth_events, final = self._call_sync(synth_agent, ctx, synth_prompt)
        for ev in synth_events:
            yield ev
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "perspectives": {p: _truncate(a) for p, a in perspectives},
                "answer": _truncate(final),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        perspectives: list[tuple[str, str]] = []
        for persona in self.personas:
            p_prompt = (
                f"You are a {persona}.  Answer strictly from this perspective.\n\n"
                f"Question: {ctx.user_message}"
            )
            _, answer = await self._call_async(self.agent, ctx, p_prompt)
            perspectives.append((persona, answer))
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"[{persona}] {answer[:120]}",
            )
        parts = "\n\n".join(f"**{p}**: {_truncate(a)}" for p, a in perspectives)
        synth_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Perspectives:\n{parts}\n\n"
            "Synthesise all perspectives into one balanced, comprehensive answer."
        )
        synth_agent = self.synthesizer_agent or self.agent
        synth_events, final = await self._call_async(synth_agent, ctx, synth_prompt)
        for ev in synth_events:
            yield ev
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "perspectives": {p: _truncate(a) for p, a in perspectives},
                "answer": _truncate(final),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# AntColonyAgent — Dorigo & Stützle (ACO)
# ═══════════════════════════════════════════════════════════════════════════════


class AntColonyAgent(BaseAgent):
    """Ant Colony Optimisation: pheromone-guided parallel exploration.

    Multiple *ants* (identical agent runs) explore solutions in parallel.
    Each solution is scored and scores serve as *pheromones*.  In later
    rounds the top-scoring solutions are fed as context, biasing
    exploration toward proven paths.

    Reference: Dorigo & Stützle, *Ant Colony Optimization*, 2004.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Agent that generates candidate solutions.
        score_fn: ``(response: str) -> float`` pheromone scorer.
        n_ants: Parallel ant agents per round.
        n_rounds: Iteration rounds.
        evaporation: Pheromone decay factor (0–1, lower = more decay).
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_ants: int = 4,
        n_rounds: int = 3,
        evaporation: float = 0.7,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.n_ants = n_ants
        self.n_rounds = n_rounds
        self.evaporation = evaporation
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        pheromone_trails: list[tuple[float, str]] = []
        best_score = -float("inf")
        best_solution = ""
        for rnd in range(1, self.n_rounds + 1):
            if pheromone_trails:
                pheromone_trails.sort(key=lambda x: x[0], reverse=True)
                top = pheromone_trails[:3]
                trail_ctx = "\n".join(
                    f"[score={s:.2f}] {t[:200]}" for s, t in top
                )
                prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Best solutions found so far:\n{trail_ctx}\n\n"
                    "Generate an improved solution inspired by the best paths."
                )
            else:
                prompt = ctx.user_message
            with ThreadPoolExecutor(max_workers=min(self.n_ants, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
                futs = [
                    pool.submit(self._gen_sync, ctx, prompt)
                    for _ in range(self.n_ants)
                ]
                new_solutions = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
            for sol in new_solutions:
                score = self.score_fn(sol)
                pheromone_trails.append((score, sol))
                if score > best_score:
                    best_score = score
                    best_solution = sol
            pheromone_trails = [
                (s * self.evaporation, t) for s, t in pheromone_trails
            ]
            # Keep only the best trails to bound memory growth.
            pheromone_trails.sort(key=lambda x: x[0], reverse=True)
            pheromone_trails = pheromone_trails[: self.n_ants * 3]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}/{self.n_rounds}: best={best_score:.3f}, "
                f"trails={len(pheromone_trails)}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best_solution)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "n_rounds": self.n_rounds,
                "completed": True,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return

        async def _gen(prompt: str) -> str:
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return out

        pheromone_trails: list[tuple[float, str]] = []
        best_score = -float("inf")
        best_solution = ""
        for rnd in range(1, self.n_rounds + 1):
            if pheromone_trails:
                pheromone_trails.sort(key=lambda x: x[0], reverse=True)
                top = pheromone_trails[:3]
                trail_ctx = "\n".join(
                    f"[score={s:.2f}] {t[:200]}" for s, t in top
                )
                prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Best solutions found so far:\n{trail_ctx}\n\n"
                    "Generate an improved solution inspired by the best paths."
                )
            else:
                prompt = ctx.user_message
            results = await asyncio.gather(
                *[_gen(prompt) for _ in range(self.n_ants)],
                return_exceptions=True,
            )
            for sol in results:
                if isinstance(sol, BaseException):
                    continue
                score = self.score_fn(sol)
                pheromone_trails.append((score, sol))
                if score > best_score:
                    best_score = score
                    best_solution = sol
            pheromone_trails = [
                (s * self.evaporation, t) for s, t in pheromone_trails
            ]
            pheromone_trails.sort(key=lambda x: x[0], reverse=True)
            pheromone_trails = pheromone_trails[: self.n_ants * 3]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}/{self.n_rounds}: best={best_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best_solution)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "n_rounds": self.n_rounds,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineParallelAgent — Pipeline Parallelism pattern
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineParallelAgent(BaseAgent):
    """Pipeline Parallelism: assembly-line concurrency for item lists.

    N stages run concurrently on different items — item 1 enters stage 2
    while item 2 enters stage 1.  Maximises throughput when processing a
    list of items through a multi-step pipeline.

    Items are read from ``session.state[items_key]`` (a list of strings).
    If not found, ``user_message`` is treated as a single item.

    Args:
        name: Agent name.
        description: Human-readable description.
        stages: Ordered list of stage agents (the pipeline).
        items_key: Session state key containing input items.
        result_key: Optional session state key for results list.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        stages: list[BaseAgent] | None = None,
        items_key: str = "items",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=stages or [], **kwargs,
        )
        self.stages = stages or []
        self.items_key = items_key
        self.result_key = result_key

    def _stage_sync(
        self, agent: BaseAgent, ctx: InvocationContext, text: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=text,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.stages:
            yield Event(EventType.ERROR, self.name, "No stages configured.")
            return
        items: list[str] = ctx.session.state.get(self.items_key, [])
        if not items:
            items = [ctx.user_message]
        results: list[str] = []
        for idx, item in enumerate(items):
            current = item
            for stage in self.stages:
                current = self._stage_sync(stage, ctx, current)
            results.append(current)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Item {idx + 1}/{len(items)} completed.",
            )
        combined = "\n\n---\n\n".join(results)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)
        if self.result_key:
            ctx.session.state_set(self.result_key, results)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.stages:
            yield Event(EventType.ERROR, self.name, "No stages configured.")
            return
        items: list[str] = ctx.session.state.get(self.items_key, [])
        if not items:
            items = [ctx.user_message]

        sem = asyncio.Semaphore(_MAX_DEFAULT_PARALLEL_WORKERS)

        async def _process_item(item: str) -> str:
            async with sem:
                current = item
                for stage in self.stages:
                    sub_ctx = InvocationContext(
                        session=Session(), user_message=current,
                        parent_agent=self, trace_collector=ctx.trace_collector,
                    )
                    out = ""
                    async for ev in stage._run_async_impl_traced(sub_ctx):
                        if ev.event_type == EventType.AGENT_MESSAGE:
                            out = ev.content
                    current = out
                return current

        tasks = [_process_item(it) for it in items]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r if not isinstance(r, BaseException) else "" for r in raw]
        for idx in range(len(results)):
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Item {idx + 1}/{len(items)} completed.",
            )
        combined = "\n\n---\n\n".join(results)
        yield Event(EventType.AGENT_MESSAGE, self.name, combined)
        if self.result_key:
            ctx.session.state_set(self.result_key, results)


# ═══════════════════════════════════════════════════════════════════════════════
# ContractNetAgent — FIPA Contract Net Protocol (Smith, 1980)
# ═══════════════════════════════════════════════════════════════════════════════


class ContractNetAgent(BaseAgent):
    """Contract Net Protocol: announce → bid → award → execute.

    The manager announces a task.  Each worker agent produces a *bid*
    (confidence via ``bid_fn``).  The highest bidder wins the contract and
    executes the task.

    Reference: Smith, *The Contract Net Protocol*, 1980.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Worker agents that can bid.
        bid_fn: ``(agent_name: str, response: str) -> float`` scorer.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        bid_fn: Callable[[str, str], float] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents or [], **kwargs,
        )
        self.bid_fn = bid_fn or (lambda _name, resp: float(len(resp)))
        self.result_key = result_key

    def _bid_sync(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[str, str, float]:
        bid_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Produce a brief proposal explaining how you would solve this."
        )
        sub_ctx = InvocationContext(
            session=Session(), user_message=bid_prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        score = self.bid_fn(agent.name, out)
        return agent.name, out, score

    def _execute_sync(
        self, agent: BaseAgent, ctx: InvocationContext,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return events, out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No worker agents.")
            return
        bids: list[tuple[str, str, float]] = []
        with ThreadPoolExecutor(max_workers=min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = {
                pool.submit(self._bid_sync, a, ctx): a
                for a in self.sub_agents
            }
            for fut in as_completed(futs):
                bids.append(fut.result(timeout=_FUTURE_TIMEOUT))
        bids.sort(key=lambda b: b[2], reverse=True)
        winner_name = bids[0][0]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Bids: {[(n, f'{s:.2f}') for n, _, s in bids]}. "
            f"Winner: {winner_name}",
        )
        winner = next(a for a in self.sub_agents if a.name == winner_name)
        events, output = self._execute_sync(winner, ctx)
        for ev in events:
            yield ev
        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name,
                "bids": {n: s for n, _, s in bids},
                "answer": output,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No worker agents.")
            return

        async def _bid(agent: BaseAgent) -> tuple[str, str, float]:
            bid_prompt = (
                f"Task: {ctx.user_message}\n\n"
                "Produce a brief proposal explaining how you would solve this."
            )
            sub_ctx = InvocationContext(
                session=Session(), user_message=bid_prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in agent._run_async_impl_traced(sub_ctx):
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return agent.name, out, self.bid_fn(agent.name, out)

        raw = await asyncio.gather(
            *[_bid(a) for a in self.sub_agents],
            return_exceptions=True,
        )
        bids = sorted(
            [r for r in raw if not isinstance(r, BaseException)],
            key=lambda b: b[2], reverse=True,
        )
        if not bids:
            yield Event(EventType.ERROR, self.name, "All bids failed.")
            return
        winner_name = bids[0][0]
        yield Event(
            EventType.STATE_UPDATE, self.name, f"Winner: {winner_name}",
        )
        winner = next(a for a in self.sub_agents if a.name == winner_name)
        sub_ctx = InvocationContext(
            session=ctx.session, user_message=ctx.user_message,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in winner._run_async_impl_traced(sub_ctx):
            yield ev
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        yield Event(EventType.AGENT_MESSAGE, self.name, out)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner_name,
                "bids": {n: s for n, _, s in bids},
                "answer": out,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# RedTeamAgent — Perez et al., 2022 / Anthropic Red Teaming
# ═══════════════════════════════════════════════════════════════════════════════


class RedTeamAgent(BaseAgent):
    """Red-Team: adversarial attack → defence hardening loop.

    The ``attacker_agent`` probes for weaknesses in the
    ``defender_agent``'s output.  Each round: attack → hardened response.
    The final hardened output is emitted.

    Reference: Perez et al., *Red Teaming Language Models*, 2022.

    Args:
        name: Agent name.
        description: Human-readable description.
        defender_agent: Agent whose output is tested.
        attacker_agent: Agent that probes for weaknesses.
        n_rounds: Attack-defence rounds.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        defender_agent: BaseAgent | None = None,
        attacker_agent: BaseAgent | None = None,
        n_rounds: int = 2,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (defender_agent, attacker_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.defender_agent = defender_agent
        self.attacker_agent = attacker_agent
        self.n_rounds = n_rounds
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        for ev in agent._run_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> tuple[list[Event], str]:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        events: list[Event] = []
        output = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            events.append(ev)
            if ev.event_type == EventType.AGENT_MESSAGE:
                output = ev.content
        return events, output

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.defender_agent and self.attacker_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires defender_agent and attacker_agent.",
            )
            return
        def_events, defence = self._call_sync(
            self.defender_agent, ctx, ctx.user_message,
        )
        for ev in def_events:
            yield ev
        attacks_log: list[dict[str, str]] = []
        for rnd in range(1, self.n_rounds + 1):
            atk_prompt = (
                f"Original task: {ctx.user_message}\n\n"
                f"Current response:\n{_truncate(defence)}\n\n"
                "Find weaknesses, inaccuracies, or edge cases in this response."
            )
            _, attack = self._call_sync(self.attacker_agent, ctx, atk_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Attack round {rnd}: {attack[:150]}",
            )
            def_prompt = (
                f"Original task: {ctx.user_message}\n\n"
                f"Your previous response:\n{_truncate(defence)}\n\n"
                f"Criticism received:\n{_truncate(attack)}\n\n"
                "Produce an improved response that addresses all criticisms."
            )
            def_events, defence = self._call_sync(
                self.defender_agent, ctx, def_prompt,
            )
            for ev in def_events:
                yield ev
            attacks_log.append({"round": str(rnd), "attack": _truncate(attack)})
        yield Event(EventType.AGENT_MESSAGE, self.name, defence)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "attacks": attacks_log[-10:],
                "final_defence": _truncate(defence),
                "n_rounds": self.n_rounds,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.defender_agent and self.attacker_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires defender_agent and attacker_agent.",
            )
            return
        def_events, defence = await self._call_async(
            self.defender_agent, ctx, ctx.user_message,
        )
        for ev in def_events:
            yield ev
        attacks_log: list[dict[str, str]] = []
        for rnd in range(1, self.n_rounds + 1):
            atk_prompt = (
                f"Original task: {ctx.user_message}\n\n"
                f"Current response:\n{_truncate(defence)}\n\n"
                "Find weaknesses, inaccuracies, or edge cases."
            )
            _, attack = await self._call_async(
                self.attacker_agent, ctx, atk_prompt,
            )
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Attack round {rnd}: {attack[:150]}",
            )
            def_prompt = (
                f"Original task: {ctx.user_message}\n\n"
                f"Your previous response:\n{_truncate(defence)}\n\n"
                f"Criticism:\n{_truncate(attack)}\n\n"
                "Produce an improved response addressing all criticisms."
            )
            def_events, defence = await self._call_async(
                self.defender_agent, ctx, def_prompt,
            )
            for ev in def_events:
                yield ev
            attacks_log.append({"round": str(rnd), "attack": _truncate(attack)})
        yield Event(EventType.AGENT_MESSAGE, self.name, defence)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "attacks": attacks_log[-10:],
                "final_defence": _truncate(defence),
                "n_rounds": self.n_rounds,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# FeedbackLoopAgent — Control Theory (closed-loop feedback)
# ═══════════════════════════════════════════════════════════════════════════════


class FeedbackLoopAgent(BaseAgent):
    """Feedback Loop: circular chain pipeline until output converges.

    The output of the last agent feeds back as input to the first.
    Iterates until ``similarity_fn(prev, curr)`` exceeds *threshold*
    or *max_iterations* is reached.

    Unlike :class:`LoopAgent` (one agent repeats) this loops a *chain*
    of distinct agents circularly.

    Reference: Classical control theory — closed-loop feedback.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Chain of agents forming the loop.
        similarity_fn: ``(prev, curr) -> float`` convergence measure.
        threshold: Convergence threshold (0–1, higher = stricter).
        max_iterations: Maximum loop iterations.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        similarity_fn: Callable[[str, str], float] | None = None,
        threshold: float = 0.95,
        max_iterations: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=sub_agents or [], **kwargs,
        )
        self.similarity_fn = similarity_fn or self._default_similarity
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.result_key = result_key

    @staticmethod
    def _default_similarity(a: str, b: str) -> float:
        if a == b:
            return 1.0
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _stage_sync(
        self, agent: BaseAgent, ctx: InvocationContext, text: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=text,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return
        current = ctx.user_message
        prev = ""
        sim = 0.0
        it = 0
        for it in range(1, self.max_iterations + 1):
            for agent in self.sub_agents:
                current = self._stage_sync(agent, ctx, current)
            sim = self.similarity_fn(prev, current)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {it}: similarity={sim:.3f}",
            )
            if sim >= self.threshold:
                break
            prev = current
        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "iterations": it,
                "converged": sim >= self.threshold,
                "answer": _truncate(current),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "No agents configured.")
            return
        current = ctx.user_message
        prev = ""
        sim = 0.0
        it = 0
        for it in range(1, self.max_iterations + 1):
            for agent in self.sub_agents:
                sub_ctx = InvocationContext(
                    session=Session(), user_message=current,
                    parent_agent=self, trace_collector=ctx.trace_collector,
                )
                out = ""
                async for ev in agent._run_async_impl_traced(sub_ctx):
                    if ev.event_type == EventType.AGENT_MESSAGE:
                        out = ev.content
                current = out
            sim = self.similarity_fn(prev, current)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {it}: similarity={sim:.3f}",
            )
            if sim >= self.threshold:
                break
            prev = current
        yield Event(EventType.AGENT_MESSAGE, self.name, current)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "iterations": it,
                "converged": sim >= self.threshold,
                "answer": _truncate(current),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# WinnowingAgent — progressive elimination by global ranking
# ═══════════════════════════════════════════════════════════════════════════════


class WinnowingAgent(BaseAgent):
    """Winnowing: progressive elimination by global ranking.

    Starts with N candidates in parallel.  Each round an
    ``evaluator_agent`` scores all survivors and the bottom
    ``cull_fraction`` are eliminated.  Repeats until one remains.

    Unlike :class:`TournamentAgent` (pair brackets) this uses global
    ranking.  Unlike :class:`BestOfNAgent` (single round) this iterates.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates candidate solutions.
        evaluator_agent: Scores candidates.
        n_candidates: Initial candidate count.
        cull_fraction: Fraction eliminated each round (0–1).
        score_fn: ``(evaluation: str) -> float`` numeric extractor.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        evaluator_agent: BaseAgent | None = None,
        n_candidates: int = 6,
        cull_fraction: float = 0.5,
        score_fn: Callable[[str], float] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in (agent, evaluator_agent) if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.evaluator_agent = evaluator_agent
        self.n_candidates = n_candidates
        self.cull_fraction = cull_fraction
        self.score_fn = score_fn or (lambda r: float(len(r)))
        self.result_key = result_key

    def _gen_sync(
        self, ctx: InvocationContext, prompt: str, agent: BaseAgent,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not (self.agent and self.evaluator_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires agent and evaluator_agent.",
            )
            return
        with ThreadPoolExecutor(max_workers=min(self.n_candidates, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(self._gen_sync, ctx, ctx.user_message, self.agent)
                for _ in range(self.n_candidates)
            ]
            candidates = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
        rnd = 0
        while len(candidates) > 1:
            rnd += 1
            scored: list[tuple[float, str]] = []
            for cand in candidates:
                eval_prompt = (
                    f"Task: {ctx.user_message}\n\nCandidate:\n{cand}\n\n"
                    "Rate quality 0-100. Start with the numeric score."
                )
                evaluation = self._gen_sync(ctx, eval_prompt, self.evaluator_agent)
                score = self.score_fn(evaluation)
                scored.append((score, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            keep = max(1, int(len(scored) * (1 - self.cull_fraction)))
            candidates = [c for _, c in scored[:keep]]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: {len(scored)} → {len(candidates)} candidates",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, candidates[0])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rnd, "winner": candidates[0],
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not (self.agent and self.evaluator_agent):
            yield Event(
                EventType.ERROR, self.name,
                "Requires agent and evaluator_agent.",
            )
            return

        async def _gen(prompt: str, ag: BaseAgent) -> str:
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in ag._run_async_impl_traced(sub_ctx):
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return out

        coros = [
            _gen(ctx.user_message, self.agent)
            for _ in range(self.n_candidates)
        ]
        raw = await asyncio.gather(*coros, return_exceptions=True)
        candidates = [r for r in raw if not isinstance(r, BaseException)]
        rnd = 0
        while len(candidates) > 1:
            rnd += 1
            eval_coros = [
                _gen(
                    f"Task: {ctx.user_message}\n\nCandidate:\n{c}\n\n"
                    "Rate quality 0-100.",
                    self.evaluator_agent,
                )
                for c in candidates
            ]
            eval_raw = await asyncio.gather(*eval_coros, return_exceptions=True)
            evaluations = [
                e if not isinstance(e, BaseException) else "0"
                for e in eval_raw
            ]
            scored = sorted(
                zip([self.score_fn(e) for e in evaluations], candidates),
                key=lambda x: x[0], reverse=True,
            )
            keep = max(1, int(len(scored) * (1 - self.cull_fraction)))
            candidates = [c for _, c in scored[:keep]]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Round {rnd}: → {len(candidates)} candidates",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, candidates[0])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": rnd, "winner": candidates[0],
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MixtureOfThoughtsAgent — multi-strategy reasoning fusion
# ═══════════════════════════════════════════════════════════════════════════════


class MixtureOfThoughtsAgent(BaseAgent):
    """Mixture-of-Thoughts: parallel reasoning strategies then selection.

    Runs the same agent with **different strategy prompts** in parallel
    (e.g. chain-of-thought, direct, step-back).  A ``selector_agent``
    picks or fuses the best answer.

    Unlike :class:`SelfConsistencyAgent` (same strategy, N samples) the
    *reasoning strategies* themselves vary.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Base agent for all strategies.
        strategies: Dict mapping strategy name → prompt prefix.
        selector_agent: Selects/fuses the best answer (defaults to *agent*).
        result_key: Optional session state key.
    """

    _DEFAULT_STRATEGIES: dict[str, str] = {
        "chain-of-thought": "Think step by step before answering.\n\n",
        "direct": "Answer directly and concisely.\n\n",
        "step-back": "First consider the general principle, then answer.\n\n",
        "devil's-advocate": "Consider why the obvious answer might be wrong.\n\n",
    }

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        strategies: dict[str, str] | None = None,
        selector_agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        seen: set[int] = set()
        deduped: list[BaseAgent] = []
        for a in (agent, selector_agent):
            if a and id(a) not in seen:
                seen.add(id(a))
                deduped.append(a)
        super().__init__(name=name, description=description, sub_agents=deduped, **kwargs)
        self.agent = agent
        self.strategies = strategies or self._DEFAULT_STRATEGIES
        self.selector_agent = selector_agent
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(len(self.strategies), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = {}
            for sname, prefix in self.strategies.items():
                prompt = f"{prefix}{ctx.user_message}"
                futs[pool.submit(
                    self._call_sync, self.agent, ctx, prompt,
                )] = sname
            for fut in as_completed(futs):
                sname = futs[fut]
                results[sname] = fut.result(timeout=_FUTURE_TIMEOUT)
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"[{sname}] {results[sname][:100]}",
                )
        parts = "\n\n".join(
            f"**Strategy: {k}**\n{_truncate(v)}" for k, v in results.items()
        )
        sel_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Multiple reasoning strategies produced:\n{parts}\n\n"
            "Select or fuse the best answer into one final response."
        )
        sel_agent = self.selector_agent or self.agent
        final = self._call_sync(sel_agent, ctx, sel_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "strategies": {k: _truncate(v) for k, v in results.items()},
                "answer": final,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return

        async def _run_strat(sname: str, prefix: str) -> tuple[str, str]:
            prompt = f"{prefix}{ctx.user_message}"
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return sname, out

        raw = await asyncio.gather(
            *[_run_strat(sn, px) for sn, px in self.strategies.items()],
            return_exceptions=True,
        )
        results = {sn: out for r in raw if not isinstance(r, BaseException) for sn, out in [r]}
        for sn, out in results.items():
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"[{sn}] {out[:100]}",
            )
        parts = "\n\n".join(
            f"**Strategy: {k}**\n{_truncate(v)}" for k, v in results.items()
        )
        sel_prompt = (
            f"Question: {ctx.user_message}\n\n"
            f"Multiple reasoning strategies produced:\n{parts}\n\n"
            "Select or fuse the best answer."
        )
        sel_agent = self.selector_agent or self.agent
        sub_ctx = InvocationContext(
            session=Session(), user_message=sel_prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        final = ""
        async for ev in sel_agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                final = ev.content
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "strategies": {k: _truncate(v) for k, v in results.items()},
                "answer": final,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# SimulatedAnnealingAgent — Kirkpatrick et al., 1983
# ═══════════════════════════════════════════════════════════════════════════════

_sa_random = random.Random(42)


class SimulatedAnnealingAgent(BaseAgent):
    """Simulated Annealing: accept worse solutions with decaying probability.

    Each iteration a neighbour solution is generated.  If better, it is
    always accepted.  If worse, it is accepted with probability
    ``exp(-delta / temperature)``.  Temperature decays by
    ``cooling_rate`` each step, reducing exploration over time.

    Reference: Kirkpatrick, Gelatt & Vecchi, *Science*, 1983.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates candidate / neighbour solutions.
        score_fn: ``(response: str) -> float`` fitness function.
        n_iterations: Number of annealing steps.
        initial_temp: Starting temperature.
        cooling_rate: Multiplicative cooling factor per step (0–1).
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_iterations: int = 10,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.85,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _gen_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        current = self._gen_sync(ctx, ctx.user_message)
        current_score = self.score_fn(current)
        best, best_score = current, current_score
        temp = self.initial_temp
        for it in range(1, self.n_iterations + 1):
            neighbour_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current solution (score={current_score:.3f}):\n{_truncate(current)}\n\n"
                "Produce a variation or improvement of this solution."
            )
            neighbour = self._gen_sync(ctx, neighbour_prompt)
            n_score = self.score_fn(neighbour)
            delta = n_score - current_score
            if delta > 0 or (temp > 0 and _sa_random.random() < math.exp(delta / temp)):
                current, current_score = neighbour, n_score
            if current_score > best_score:
                best, best_score = current, current_score
            temp *= self.cooling_rate
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {it}: score={current_score:.3f}, best={best_score:.3f}, "
                f"temp={temp:.4f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "iterations": self.n_iterations,
                "completed": True,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        current = await self._gen_async(ctx, ctx.user_message)
        current_score = self.score_fn(current)
        best, best_score = current, current_score
        temp = self.initial_temp
        for it in range(1, self.n_iterations + 1):
            neighbour_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current solution (score={current_score:.3f}):\n{_truncate(current)}\n\n"
                "Produce a variation or improvement."
            )
            neighbour = await self._gen_async(ctx, neighbour_prompt)
            n_score = self.score_fn(neighbour)
            delta = n_score - current_score
            if delta > 0 or (temp > 0 and _sa_random.random() < math.exp(delta / temp)):
                current, current_score = neighbour, n_score
            if current_score > best_score:
                best, best_score = current, current_score
            temp *= self.cooling_rate
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {it}: score={current_score:.3f}, temp={temp:.4f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "iterations": self.n_iterations,
                "completed": True,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# TabuSearchAgent — Glover, 1986
# ═══════════════════════════════════════════════════════════════════════════════


class TabuSearchAgent(BaseAgent):
    """Tabu Search: local search with memory to avoid cycling.

    Maintains a *tabu list* of recent solutions (hashes).  Each iteration
    generates a neighbour; if it is in the tabu list it is rejected and a
    new one is requested.

    Reference: Glover, *Future Paths for Integer Programming*, 1986.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates neighbour solutions.
        score_fn: ``(response: str) -> float`` fitness function.
        n_iterations: Search iterations.
        tabu_size: Maximum tabu list length.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_iterations: int = 10,
        tabu_size: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.n_iterations = n_iterations
        self.tabu_size = tabu_size
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _gen_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    @staticmethod
    def _hash(text: str) -> int:
        return hash(text.strip().lower())

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        current = self._gen_sync(ctx, ctx.user_message)
        current_score = self.score_fn(current)
        best, best_score = current, current_score
        tabu: list[int] = [self._hash(current)]
        for it in range(1, self.n_iterations + 1):
            nb_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current solution:\n{_truncate(current)}\n\n"
                "Produce a DIFFERENT variation.  Do NOT repeat previous solutions."
            )
            neighbour = self._gen_sync(ctx, nb_prompt)
            h = self._hash(neighbour)
            if h in tabu:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Step {it}: tabu hit — skipped",
                )
                continue
            n_score = self.score_fn(neighbour)
            current, current_score = neighbour, n_score
            tabu.append(h)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
            if current_score > best_score:
                best, best_score = current, current_score
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {it}: score={current_score:.3f}, best={best_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "iterations": self.n_iterations,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        current = await self._gen_async(ctx, ctx.user_message)
        current_score = self.score_fn(current)
        best, best_score = current, current_score
        tabu: list[int] = [self._hash(current)]
        for it in range(1, self.n_iterations + 1):
            nb_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current solution:\n{_truncate(current)}\n\n"
                "Produce a DIFFERENT variation."
            )
            neighbour = await self._gen_async(ctx, nb_prompt)
            h = self._hash(neighbour)
            if h in tabu:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Step {it}: tabu hit — skipped",
                )
                continue
            n_score = self.score_fn(neighbour)
            current, current_score = neighbour, n_score
            tabu.append(h)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
            if current_score > best_score:
                best, best_score = current, current_score
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {it}: score={current_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "iterations": self.n_iterations,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ParticleSwarmAgent — Kennedy & Eberhart, 1995
# ═══════════════════════════════════════════════════════════════════════════════


class ParticleSwarmAgent(BaseAgent):
    """Particle Swarm Optimisation: personal-best + global-best guidance.

    Each particle generates a solution.  On subsequent rounds each
    particle sees its own best and the swarm's global best as context.

    Reference: Kennedy & Eberhart, 1995.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates candidate solutions.
        score_fn: ``(response: str) -> float`` fitness function.
        n_particles: Swarm size.
        n_iterations: Iteration count.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_particles: int = 4,
        n_iterations: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        # Initialise particles
        with ThreadPoolExecutor(max_workers=min(self.n_particles, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(self._gen_sync, ctx, ctx.user_message)
                for _ in range(self.n_particles)
            ]
            positions = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
        scores = [self.score_fn(p) for p in positions]
        p_best = list(zip(list(scores), list(positions)))  # personal bests
        g_best_score = max(scores)
        g_best = positions[scores.index(g_best_score)]

        for it in range(1, self.n_iterations + 1):
            new_positions: list[str] = []
            for i, pos in enumerate(positions):
                prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Your current solution:\n{_truncate(pos)}\n\n"
                    f"Your personal best (score={p_best[i][0]:.2f}):\n{p_best[i][1][:200]}\n\n"
                    f"Swarm global best (score={g_best_score:.2f}):\n{g_best[:200]}\n\n"
                    "Generate an improved solution inspired by both bests."
                )
                new_positions.append(self._gen_sync(ctx, prompt))
            positions = new_positions
            scores = [self.score_fn(p) for p in positions]
            for i in range(len(positions)):
                if scores[i] > p_best[i][0]:
                    p_best[i] = (scores[i], positions[i])
                if scores[i] > g_best_score:
                    g_best_score = scores[i]
                    g_best = positions[i]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {it}: global_best={g_best_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, g_best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": g_best_score,
                "iterations": self.n_iterations,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return

        async def _gen(prompt: str) -> str:
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return out

        raw_init = await asyncio.gather(
            *[_gen(ctx.user_message) for _ in range(self.n_particles)],
            return_exceptions=True,
        )
        positions = [r if not isinstance(r, BaseException) else "" for r in raw_init]
        scores = [self.score_fn(p) for p in positions]
        p_best = list(zip(list(scores), list(positions)))
        g_best_score = max(scores)
        g_best = positions[scores.index(g_best_score)]

        for it in range(1, self.n_iterations + 1):
            coros = []
            for i, pos in enumerate(positions):
                prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Your current solution:\n{_truncate(pos)}\n\n"
                    f"Your personal best (score={p_best[i][0]:.2f}):\n{p_best[i][1][:200]}\n\n"
                    f"Global best (score={g_best_score:.2f}):\n{g_best[:200]}\n\n"
                    "Generate an improved solution."
                )
                coros.append(_gen(prompt))
            raw_iter = await asyncio.gather(*coros, return_exceptions=True)
            positions = [r if not isinstance(r, BaseException) else positions[i] for i, r in enumerate(raw_iter)]
            scores = [self.score_fn(p) for p in positions]
            for i in range(len(positions)):
                if scores[i] > p_best[i][0]:
                    p_best[i] = (scores[i], positions[i])
                if scores[i] > g_best_score:
                    g_best_score = scores[i]
                    g_best = positions[i]
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Iteration {it}: global_best={g_best_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, g_best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": g_best_score,
                "iterations": self.n_iterations,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# DifferentialEvolutionAgent — Storn & Price, 1997
# ═══════════════════════════════════════════════════════════════════════════════


class DifferentialEvolutionAgent(BaseAgent):
    """Differential Evolution: mutate by differencing population members.

    Each generation: for every individual, pick three others, build a
    *donor* from their combination, crossover with the current, and
    select the fitter.

    Reference: Storn & Price, *J. of Global Optimization*, 1997.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates / mutates solutions.
        score_fn: ``(response: str) -> float`` fitness function.
        population_size: Population count.
        n_generations: Number of generations.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        population_size: int = 6,
        n_generations: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.population_size = population_size
        self.n_generations = n_generations
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        with ThreadPoolExecutor(max_workers=min(self.population_size, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(self._gen_sync, ctx, ctx.user_message)
                for _ in range(self.population_size)
            ]
            pop = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
        scores = [self.score_fn(p) for p in pop]
        for gen in range(1, self.n_generations + 1):
            new_pop: list[str] = []
            new_scores: list[float] = []
            for i in range(len(pop)):
                idxs = [j for j in range(len(pop)) if j != i]
                a, b, c = random.sample(idxs, min(3, len(idxs)))
                donor_prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Base: {pop[a][:200]}\n\n"
                    f"Variation A: {pop[b][:200]}\n\n"
                    f"Variation B: {pop[c][:200]}\n\n"
                    "Combine elements of Base, A, and B into a novel solution."
                )
                trial = self._gen_sync(ctx, donor_prompt)
                t_score = self.score_fn(trial)
                if t_score >= scores[i]:
                    new_pop.append(trial)
                    new_scores.append(t_score)
                else:
                    new_pop.append(pop[i])
                    new_scores.append(scores[i])
            pop, scores = new_pop, new_scores
            best_score = max(scores)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gen {gen}: best={best_score:.3f}",
            )
        best_idx = scores.index(max(scores))
        yield Event(EventType.AGENT_MESSAGE, self.name, pop[best_idx])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": scores[best_idx],
                "generations": self.n_generations,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return

        async def _gen(prompt: str) -> str:
            sub_ctx = InvocationContext(
                session=Session(), user_message=prompt,
                parent_agent=self, trace_collector=ctx.trace_collector,
            )
            out = ""
            async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
                if ev.event_type == EventType.AGENT_MESSAGE:
                    out = ev.content
            return out

        raw_init = await asyncio.gather(
            *[_gen(ctx.user_message) for _ in range(self.population_size)],
            return_exceptions=True,
        )
        pop = [r if not isinstance(r, BaseException) else "" for r in raw_init]
        scores = [self.score_fn(p) for p in pop]
        for gen in range(1, self.n_generations + 1):
            trial_coros = []
            for i in range(len(pop)):
                idxs = [j for j in range(len(pop)) if j != i]
                a, b, c = random.sample(idxs, min(3, len(idxs)))
                donor_prompt = (
                    f"Task: {ctx.user_message}\n\n"
                    f"Base: {pop[a][:200]}\nA: {pop[b][:200]}\nB: {pop[c][:200]}\n\n"
                    "Combine into a novel solution."
                )
                trial_coros.append(_gen(donor_prompt))
            raw_trials = await asyncio.gather(*trial_coros, return_exceptions=True)
            trials = [r if not isinstance(r, BaseException) else pop[i] for i, r in enumerate(raw_trials)]
            t_scores = [self.score_fn(t) for t in trials]
            new_pop, new_scores = [], []
            for i in range(len(pop)):
                if t_scores[i] >= scores[i]:
                    new_pop.append(trials[i])
                    new_scores.append(t_scores[i])
                else:
                    new_pop.append(pop[i])
                    new_scores.append(scores[i])
            pop, scores = new_pop, new_scores
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gen {gen}: best={max(scores):.3f}",
            )
        best_idx = scores.index(max(scores))
        yield Event(EventType.AGENT_MESSAGE, self.name, pop[best_idx])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": scores[best_idx],
                "generations": self.n_generations,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianOptimizationAgent — Mockus 1975; Snoek et al. 2012
# ═══════════════════════════════════════════════════════════════════════════════


class BayesianOptimizationAgent(BaseAgent):
    """Bayesian Optimisation: surrogate model + acquisition function.

    Keeps a history of (solution, score) pairs.  Each round the agent
    sees the full history and is asked to propose the solution most
    likely to improve the best score.  The ``score_fn`` evaluates it and
    the pair is appended.

    Reference: Snoek, Larochelle & Adams, *NeurIPS*, 2012.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Proposes next candidate given history.
        score_fn: ``(response: str) -> float`` evaluation.
        n_iterations: Number of evaluation rounds.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        score_fn: Callable[[str], float] | None = None,
        n_iterations: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.score_fn = score_fn or (lambda _: 0.5)
        self.n_iterations = n_iterations
        self.result_key = result_key

    def _gen_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _gen_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _build_history_prompt(
        self, ctx: InvocationContext,
        history: list[tuple[str, float]],
    ) -> str:
        capped = history[-20:]
        hist_str = "\n".join(
            f"  [{i+1}] score={s:.3f} → {_truncate(sol, 120)}"
            for i, (sol, s) in enumerate(capped)
        )
        return (
            f"Task: {ctx.user_message}\n\n"
            f"Evaluation history:\n{hist_str}\n\n"
            "Based on the pattern of scores above, propose a NEW solution "
            "most likely to achieve a HIGHER score than the current best."
        )

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        # Initial sample
        first = _truncate(self._gen_sync(ctx, ctx.user_message))
        history: list[tuple[str, float]] = [(first, self.score_fn(first))]
        best, best_score = first, history[0][1]
        for it in range(2, self.n_iterations + 1):
            prompt = self._build_history_prompt(ctx, history)
            candidate = _truncate(self._gen_sync(ctx, prompt))
            score = self.score_fn(candidate)
            history.append((candidate, score))
            if score > best_score:
                best, best_score = candidate, score
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Eval {it}/{self.n_iterations}: score={score:.3f}, "
                f"best={best_score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "evaluations": len(history),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        first = _truncate(await self._gen_async(ctx, ctx.user_message))
        history: list[tuple[str, float]] = [(first, self.score_fn(first))]
        best, best_score = first, history[0][1]
        for it in range(2, self.n_iterations + 1):
            prompt = self._build_history_prompt(ctx, history)
            candidate = _truncate(await self._gen_async(ctx, prompt))
            score = self.score_fn(candidate)
            history.append((candidate, score))
            if score > best_score:
                best, best_score = candidate, score
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Eval {it}/{self.n_iterations}: score={score:.3f}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "best_score": best_score,
                "evaluations": len(history),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# AnalogicalReasoningAgent — Yasunaga et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class AnalogicalReasoningAgent(BaseAgent):
    """Analogical Reasoning: self-generate analogous examples then solve.

    Phase 1 – the agent generates ``n_analogies`` analogous problems and
    their solutions.  Phase 2 – a solver agent receives the original
    problem together with the analogies as exemplar context.

    Reference: Yasunaga et al., *Large Language Models as Analogical
    Reasoners*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates analogies (phase 1) and final answer (phase 2).
        n_analogies: Number of analogous examples to self-generate.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        n_analogies: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.n_analogies = n_analogies
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        analogy_prompt = (
            f"Given the following problem:\n{ctx.user_message}\n\n"
            f"Generate {self.n_analogies} analogous problems with their "
            "solutions.  Use the format:\n"
            "Analogy 1: <problem> → <solution>\n"
            "Analogy 2: …"
        )
        analogies = self._call_sync(ctx, analogy_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, f"Generated {self.n_analogies} analogies")
        solve_prompt = (
            f"Problem:\n{ctx.user_message}\n\n"
            f"Here are analogous examples for guidance:\n{_truncate(analogies)}\n\n"
            "Now solve the original problem using insights from the analogies."
        )
        answer = self._call_sync(ctx, solve_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "analogies": _truncate(analogies), "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        analogy_prompt = (
            f"Given the following problem:\n{ctx.user_message}\n\n"
            f"Generate {self.n_analogies} analogous problems with solutions."
        )
        analogies = await self._call_async(ctx, analogy_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, f"Generated {self.n_analogies} analogies")
        solve_prompt = (
            f"Problem:\n{ctx.user_message}\n\n"
            f"Analogous examples:\n{_truncate(analogies)}\n\n"
            "Solve the original problem using insights from the analogies."
        )
        answer = await self._call_async(ctx, solve_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "analogies": _truncate(analogies), "answer": _truncate(answer),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ThreadOfThoughtAgent — Zhou et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class ThreadOfThoughtAgent(BaseAgent):
    """Thread of Thought (ThoT): walk through a lengthy context segment
    by segment before answering.

    Phase 1 – *segmented analysis*: the agent receives the context with
    the instruction "Walk through this context step by step".
    Phase 2 – *distilled answer*: summarise the walk-through into a final
    answer.

    Reference: Zhou et al., *Thread of Thought Unraveling Chaotic
    Contexts*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Performs the walk-through and final summary.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        walk_prompt = (
            "Walk through the following context and question step by step, "
            "segment by segment.  Summarise each segment before moving on.\n\n"
            f"{ctx.user_message}"
        )
        walkthrough = self._call_sync(ctx, walk_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Walk-through complete")
        distill_prompt = (
            f"Based on this step-by-step walk-through:\n{_truncate(walkthrough)}\n\n"
            "Provide a concise, final answer to the original question."
        )
        answer = self._call_sync(ctx, distill_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "walkthrough": _truncate(walkthrough), "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        walk_prompt = (
            "Walk through the following context step by step, segment by "
            "segment.  Summarise each segment before moving on.\n\n"
            f"{ctx.user_message}"
        )
        walkthrough = await self._call_async(ctx, walk_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Walk-through complete")
        distill_prompt = (
            f"Based on this walk-through:\n{_truncate(walkthrough)}\n\n"
            "Provide a concise, final answer."
        )
        answer = await self._call_async(ctx, distill_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "walkthrough": _truncate(walkthrough), "answer": _truncate(answer),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertPromptingAgent — Xu et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class ExpertPromptingAgent(BaseAgent):
    """Expert Prompting: auto-generate an expert identity then answer.

    Phase 1 – the LLM designs an expert persona best suited for the
    task.  Phase 2 – it answers *as* that expert.

    Reference: Xu et al., *ExpertPrompting: Instructing Large Language
    Models to be Distinguished Experts*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates expert profile and final answer.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        identity_prompt = (
            f"For the following task, describe the ideal expert who would "
            f"be best suited to answer it.  Include their credentials, "
            f"experience, and relevant expertise.\n\nTask: {ctx.user_message}"
        )
        expert_profile = self._call_sync(ctx, identity_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Expert profile created")
        answer_prompt = (
            f"You are the following expert:\n{_truncate(expert_profile)}\n\n"
            f"Now answer the question as this expert:\n{ctx.user_message}"
        )
        answer = self._call_sync(ctx, answer_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "expert_profile": _truncate(expert_profile), "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        identity_prompt = (
            f"For the following task, describe the ideal expert.\n\n"
            f"Task: {ctx.user_message}"
        )
        expert_profile = await self._call_async(ctx, identity_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Expert profile created")
        answer_prompt = (
            f"You are the following expert:\n{_truncate(expert_profile)}\n\n"
            f"Answer:\n{ctx.user_message}"
        )
        answer = await self._call_async(ctx, answer_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "expert_profile": _truncate(expert_profile), "answer": _truncate(answer),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# BufferOfThoughtsAgent — Yang et al., 2024
# ═══════════════════════════════════════════════════════════════════════════════


class BufferOfThoughtsAgent(BaseAgent):
    """Buffer of Thoughts (BoT): distil reusable thought-templates.

    Maintains a *thought buffer* (list of high-level reasoning templates).
    For each new task the agent first retrieves the most relevant
    template, instantiates it, and then reasons.  After solving, a
    *distiller* step extracts a new template if the approach was novel.

    Reference: Yang et al., *Buffer of Thoughts: Thought-Augmented
    Reasoning with Large Language Models*, 2024.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Performs retrieval, reasoning, and distillation.
        initial_buffer: Seed thought-templates.
        max_buffer: Maximum buffer size.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        initial_buffer: list[str] | None = None,
        max_buffer: int = 20,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.buffer: list[str] = list(initial_buffer or [])
        self._buffer_lock = threading.Lock()
        self.max_buffer = max_buffer
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _retrieve(self, ctx: InvocationContext) -> str:
        if not self.buffer:
            return "(no prior templates)"
        buf_str = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(self.buffer))
        ret_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Available thought-templates:\n{buf_str}\n\n"
            "Which template (number) is most relevant?  "
            "Reply with just the number."
        )
        reply = self._call_sync(ctx, ret_prompt).strip()
        try:
            idx = int(reply) - 1
            return self.buffer[idx] if 0 <= idx < len(self.buffer) else self.buffer[0]
        except (ValueError, IndexError):
            return self.buffer[0]

    async def _retrieve_async(self, ctx: InvocationContext) -> str:
        if not self.buffer:
            return "(no prior templates)"
        buf_str = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(self.buffer))
        ret_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Available thought-templates:\n{buf_str}\n\n"
            "Which template is most relevant?  Reply with just the number."
        )
        reply = (await self._call_async(ctx, ret_prompt)).strip()
        try:
            idx = int(reply) - 1
            return self.buffer[idx] if 0 <= idx < len(self.buffer) else self.buffer[0]
        except (ValueError, IndexError):
            return self.buffer[0]

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        template = self._retrieve(ctx)
        yield Event(EventType.STATE_UPDATE, self.name, f"Retrieved template: {template[:80]}")
        reason_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Apply this reasoning template:\n{template}\n\n"
            "Work through the task step by step using the template."
        )
        answer = self._call_sync(ctx, reason_prompt)
        distill_prompt = (
            f"You just solved a task using this approach:\n{answer[:500]}\n\n"
            "Distil a reusable, high-level thought-template (1-2 sentences) "
            "from your reasoning that could help with similar future tasks."
        )
        new_template = self._call_sync(ctx, distill_prompt).strip()
        if new_template and new_template not in self.buffer:
            with self._buffer_lock:
                self.buffer.append(new_template)
                if len(self.buffer) > self.max_buffer:
                    self.buffer.pop(0)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "template_used": template,
                "new_template": new_template,
                "buffer_size": len(self.buffer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        template = await self._retrieve_async(ctx)
        yield Event(EventType.STATE_UPDATE, self.name, f"Retrieved template: {template[:80]}")
        reason_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Apply this reasoning template:\n{template}\n\n"
            "Work through the task step by step."
        )
        answer = await self._call_async(ctx, reason_prompt)
        distill_prompt = (
            f"Approach used:\n{answer[:500]}\n\n"
            "Distil a reusable thought-template (1-2 sentences)."
        )
        new_template = (await self._call_async(ctx, distill_prompt)).strip()
        if new_template and new_template not in self.buffer:
            with self._buffer_lock:
                self.buffer.append(new_template)
                if len(self.buffer) > self.max_buffer:
                    self.buffer.pop(0)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "template_used": template,
                "buffer_size": len(self.buffer),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ChainOfAbstractionAgent — Gao et al., 2024
# ═══════════════════════════════════════════════════════════════════════════════


class ChainOfAbstractionAgent(BaseAgent):
    """Chain of Abstraction (CoA): reason with placeholders then ground.

    Phase 1 – *abstract chain*: the agent produces a reasoning chain
    using abstract placeholders (``[FACT-1]``, ``[FACT-2]``, …) for
    knowledge it would need to look up.
    Phase 2 – *grounding*: a second pass fills in each placeholder with
    concrete evidence.

    Reference: Gao et al., *Chain-of-Abstraction Reasoning*, 2024.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates both abstract chain and grounded answer.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        abstract_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Produce a step-by-step reasoning chain.  Wherever you need an "
            "external fact, write a placeholder like [FACT-1], [FACT-2], etc.  "
            "Do NOT fill them in yet."
        )
        abstract_chain = self._call_sync(ctx, abstract_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Abstract chain generated")
        ground_prompt = (
            f"Here is an abstract reasoning chain:\n{_truncate(abstract_chain)}\n\n"
            "Now fill in ALL placeholders ([FACT-1], [FACT-2], …) with "
            "concrete, accurate information and provide the final answer."
        )
        grounded = self._call_sync(ctx, ground_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, grounded)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "abstract_chain": _truncate(abstract_chain),
                "grounded_answer": grounded,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        abstract_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Produce a reasoning chain with placeholders "
            "([FACT-1], [FACT-2], …) for needed facts."
        )
        abstract_chain = await self._call_async(ctx, abstract_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Abstract chain generated")
        ground_prompt = (
            f"Abstract chain:\n{_truncate(abstract_chain)}\n\n"
            "Fill ALL placeholders with concrete information and answer."
        )
        grounded = await self._call_async(ctx, ground_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, grounded)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "abstract_chain": _truncate(abstract_chain),
                "grounded_answer": grounded,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# VerifierAgent — Cobbe et al., 2021
# ═══════════════════════════════════════════════════════════════════════════════


class VerifierAgent(BaseAgent):
    """Verifier: generate multiple solutions, score each, pick the best.

    Phase 1 – generate ``n_solutions`` candidate answers.
    Phase 2 – a verifier agent scores each on correctness.
    Phase 3 – return the highest-scoring solution.

    Reference: Cobbe et al., *Training Verifiers to Solve Math Word
    Problems*, 2021.

    Args:
        name: Agent name.
        description: Human-readable description.
        generator: Agent that produces candidate solutions.
        verifier: Agent that scores solutions (should output a number 0–10).
        n_solutions: Number of candidates.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        generator: BaseAgent | None = None,
        verifier: BaseAgent | None = None,
        n_solutions: int = 5,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [generator, verifier] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.generator = generator
        self.verifier = verifier or generator
        self.n_solutions = n_solutions
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    @staticmethod
    def _parse_score(text: str) -> float:
        m = _NUMBER_RE.search(text)
        return float(m.group(1)) if m else 0.0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.generator:
            yield Event(EventType.ERROR, self.name, "Requires generator.")
            return
        solutions: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.n_solutions, _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(self._call_sync, self.generator, ctx, ctx.user_message)
                for _ in range(self.n_solutions)
            ]
            solutions = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Generated {len(solutions)} candidates",
        )
        best, best_score = solutions[0], -1.0
        for i, sol in enumerate(solutions):
            verify_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Candidate solution:\n{sol}\n\n"
                "Rate the correctness of this solution from 0 to 10.  "
                "Reply with just the number."
            )
            raw = self._call_sync(self.verifier, ctx, verify_prompt)
            score = self._parse_score(raw)
            if score > best_score:
                best, best_score = sol, score
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_solutions": len(solutions),
                "best_score": best_score,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.generator:
            yield Event(EventType.ERROR, self.name, "Requires generator.")
            return
        coros = [
            self._call_async(self.generator, ctx, ctx.user_message)
            for _ in range(self.n_solutions)
        ]
        raw_solutions = await asyncio.gather(*coros, return_exceptions=True)
        solutions = [r for r in raw_solutions if not isinstance(r, BaseException)]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Generated {len(solutions)} candidates",
        )
        best, best_score = solutions[0], -1.0
        for sol in solutions:
            verify_prompt = (
                f"Task: {ctx.user_message}\n\nCandidate:\n{sol}\n\n"
                "Rate correctness 0-10.  Reply with just the number."
            )
            raw = await self._call_async(self.verifier, ctx, verify_prompt)
            score = self._parse_score(raw)
            if score > best_score:
                best, best_score = sol, score
        yield Event(EventType.AGENT_MESSAGE, self.name, best)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_solutions": len(solutions),
                "best_score": best_score,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ProgOfThoughtAgent — Chen et al., 2023
# ═══════════════════════════════════════════════════════════════════════════════


class ProgOfThoughtAgent(BaseAgent):
    """Program of Thoughts (PoT): generate code, then execute it.

    Phase 1 – the agent generates a Python program that computes the
    answer.  Phase 2 – the code is executed in the built-in sandbox (or
    a restricted ``exec``).  The program output becomes the answer.

    Reference: Chen et al., *Program of Thoughts Prompting*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates the Python program.
        sandbox: If True, use Nono sandboxed exec; else use restricted
                 exec with limited builtins.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        sandbox: bool = False,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.sandbox = sandbox
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    @staticmethod
    def _extract_code(text: str) -> str:
        m = _CODE_BLOCK_RE.search(text)
        return m.group(1).strip() if m else text.strip()

    @staticmethod
    def _safe_exec(code: str) -> str:
        buf = io.StringIO()
        safe_globals: dict[str, Any] = {"__builtins__": {"print": print, "range": range, "len": len, "int": int, "float": float, "str": str, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter}}
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals)  # noqa: S102
        return buf.getvalue().strip()

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        gen_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Write a Python program that computes the answer and prints it.  "
            "Wrap the code in ```python ... ```."
        )
        raw = self._call_sync(ctx, gen_prompt)
        code = self._extract_code(raw)
        yield Event(EventType.STATE_UPDATE, self.name, f"Generated code ({len(code)} chars)")
        try:
            output = self._safe_exec(code)
        except Exception as exc:
            output = f"Execution error: {exc}"
        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"code": _truncate(code), "output": output})

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        gen_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Write a Python program that computes the answer and prints it.  "
            "Wrap in ```python ... ```."
        )
        raw = await self._call_async(ctx, gen_prompt)
        code = self._extract_code(raw)
        yield Event(EventType.STATE_UPDATE, self.name, f"Generated code ({len(code)} chars)")
        try:
            output = self._safe_exec(code)
        except Exception as exc:
            output = f"Execution error: {exc}"
        yield Event(EventType.AGENT_MESSAGE, self.name, output)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"code": _truncate(code), "output": output})


# ═══════════════════════════════════════════════════════════════════════════════
# InnerMonologueAgent — Huang et al., 2022
# ═══════════════════════════════════════════════════════════════════════════════


class InnerMonologueAgent(BaseAgent):
    """Inner Monologue: closed-loop verbal reasoning with environment
    feedback.

    Each step the agent proposes an action.  A ``feedback_fn`` returns
    environment feedback.  The agent sees its full inner monologue
    (history) and continues until it outputs a token matching
    ``done_token`` or ``max_steps`` is reached.

    Reference: Huang et al., *Inner Monologue: Embodied Reasoning
    through Planning with Language Models*, 2022.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Proposes actions / continues reasoning.
        feedback_fn: ``(action: str) -> str`` environment feedback.
        max_steps: Maximum monologue steps.
        done_token: String whose presence signals completion.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        feedback_fn: Callable[[str], str] | None = None,
        max_steps: int = 5,
        done_token: str = "DONE",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.feedback_fn = feedback_fn or (lambda a: f"Feedback: received '{a[:50]}'")
        self.max_steps = max_steps
        self.done_token = done_token
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        monologue: list[str] = [f"Task: {ctx.user_message}"]
        last_action = ""
        for step in range(1, self.max_steps + 1):
            prompt = "\n".join(monologue[-20:]) + "\n\nNext action (or say DONE if finished):"
            action = self._call_sync(ctx, prompt)
            last_action = action
            monologue.append(f"Action {step}: {_truncate(action)}")
            if self.done_token.upper() in action.upper():
                break
            feedback = self.feedback_fn(action)
            monologue.append(f"Feedback {step}: {_truncate(feedback)}")
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}: {action[:80]}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, last_action)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": len(monologue) // 2,
                "monologue_length": len(monologue),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        monologue: list[str] = [f"Task: {ctx.user_message}"]
        last_action = ""
        for step in range(1, self.max_steps + 1):
            prompt = "\n".join(monologue[-20:]) + "\n\nNext action (or say DONE):"
            action = await self._call_async(ctx, prompt)
            last_action = action
            monologue.append(f"Action {step}: {_truncate(action)}")
            if self.done_token.upper() in action.upper():
                break
            feedback = self.feedback_fn(action)
            monologue.append(f"Feedback {step}: {_truncate(feedback)}")
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}: {action[:80]}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, last_action)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": len(monologue) // 2,
                "monologue_length": len(monologue),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# RolePlayingAgent — Li et al., 2023 (CAMEL)
# ═══════════════════════════════════════════════════════════════════════════════


class RolePlayingAgent(BaseAgent):
    """Role-Playing: two agents converse in assigned roles toward a goal.

    *instructor_agent* and *assistant_agent* alternate messages for
    ``n_turns``.  The instructor drives the task and the assistant
    follows instructions.  Both see the growing conversation history.

    Reference: Li et al., *CAMEL: Communicative Agents for "Mind"
    Exploration of Large Language Model Society*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        instructor_agent: Drives the conversation.
        assistant_agent: Follows instructions.
        n_turns: Number of exchange rounds.
        instructor_role: Persona description for the instructor.
        assistant_role: Persona description for the assistant.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        instructor_agent: BaseAgent | None = None,
        assistant_agent: BaseAgent | None = None,
        n_turns: int = 3,
        instructor_role: str = "Instructor",
        assistant_role: str = "Assistant",
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [instructor_agent, assistant_agent] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.instructor_agent = instructor_agent
        self.assistant_agent = assistant_agent
        self.n_turns = n_turns
        self.instructor_role = instructor_role
        self.assistant_role = assistant_role
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.instructor_agent or not self.assistant_agent:
            yield Event(EventType.ERROR, self.name, "Requires instructor and assistant agents.")
            return
        history: list[str] = []
        task = ctx.user_message
        for turn in range(1, self.n_turns + 1):
            inst_prompt = (
                f"You are the {self.instructor_role}.\n"
                f"Task: {task}\n\n"
                + (f"Conversation so far:\n" + "\n".join(history[-20:]) + "\n\n" if history else "")
                + "Give the next instruction to the assistant."
            )
            instruction = self._call_sync(self.instructor_agent, ctx, inst_prompt)
            history.append(f"[{self.instructor_role}]: {_truncate(instruction)}")
            asst_prompt = (
                f"You are the {self.assistant_role}.\n"
                f"Task: {task}\n\n"
                f"Conversation so far:\n" + "\n".join(history[-20:]) + "\n\n"
                "Follow the latest instruction."
            )
            response = self._call_sync(self.assistant_agent, ctx, asst_prompt)
            history.append(f"[{self.assistant_role}]: {_truncate(response)}")
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Turn {turn}/{self.n_turns} complete",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, history[-1])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "turns": self.n_turns,
                "history_length": len(history),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.instructor_agent or not self.assistant_agent:
            yield Event(EventType.ERROR, self.name, "Requires instructor and assistant agents.")
            return
        history: list[str] = []
        task = ctx.user_message
        for turn in range(1, self.n_turns + 1):
            inst_prompt = (
                f"You are the {self.instructor_role}.\nTask: {task}\n\n"
                + (f"Conversation:\n" + "\n".join(history[-20:]) + "\n\n" if history else "")
                + "Give the next instruction."
            )
            instruction = await self._call_async(self.instructor_agent, ctx, inst_prompt)
            history.append(f"[{self.instructor_role}]: {_truncate(instruction)}")
            asst_prompt = (
                f"You are the {self.assistant_role}.\nTask: {task}\n\n"
                f"Conversation:\n" + "\n".join(history[-20:]) + "\n\nFollow the instruction."
            )
            response = await self._call_async(self.assistant_agent, ctx, asst_prompt)
            history.append(f"[{self.assistant_role}]: {_truncate(response)}")
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Turn {turn}/{self.n_turns} complete",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, history[-1])
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "turns": self.n_turns,
                "history_length": len(history),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# GossipProtocolAgent — Demers et al., 1987
# ═══════════════════════════════════════════════════════════════════════════════


class GossipProtocolAgent(BaseAgent):
    """Gossip Protocol: epidemic information spreading among agents.

    Each round, every agent shares its current state with a random peer.
    Peers merge received information with their own.  After ``n_rounds``,
    a *collector* agent summarises the converged knowledge.

    Reference: Demers et al., *Epidemic Algorithms for Replicated
    Database Maintenance*, 1987.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: List of participating agents.
        collector: Agent that produces the final summary.
        n_rounds: Gossip rounds.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        collector: BaseAgent | None = None,
        n_rounds: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        agents = list(sub_agents or [])
        if collector and collector not in agents:
            agents.append(collector)
        super().__init__(name=name, description=description, sub_agents=agents, **kwargs)
        self.peers = list(sub_agents or [])
        self.collector = collector or (self.peers[0] if self.peers else None)
        self.n_rounds = n_rounds
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if len(self.peers) < 2:
            yield Event(EventType.ERROR, self.name, "Need ≥2 peers.")
            return
        states = {a.name: _truncate(self._call_sync(a, ctx, ctx.user_message)) for a in self.peers}
        for rnd in range(1, self.n_rounds + 1):
            for agent in self.peers:
                peer = random.choice([p for p in self.peers if p is not agent])
                merge_prompt = (
                    f"Your current knowledge:\n{states[agent.name]}\n\n"
                    f"Received from {peer.name}:\n{states[peer.name]}\n\n"
                    "Merge the two, keeping the most accurate information."
                )
                states[agent.name] = _truncate(self._call_sync(agent, ctx, merge_prompt))
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gossip round {rnd}/{self.n_rounds}",
            )
        all_states = "\n\n".join(f"[{n}]: {s}" for n, s in states.items())
        summary_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Collected states after gossip:\n{all_states}\n\n"
            "Summarise the converged answer."
        )
        final = self._call_sync(self.collector, ctx, summary_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": self.n_rounds,
                "peers": len(self.peers),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if len(self.peers) < 2:
            yield Event(EventType.ERROR, self.name, "Need ≥2 peers.")
            return
        coros = {a.name: self._call_async(a, ctx, ctx.user_message) for a in self.peers}
        raw_results = await asyncio.gather(*coros.values(), return_exceptions=True)
        states = {}
        for nm, r in zip(coros.keys(), raw_results):
            states[nm] = "" if isinstance(r, BaseException) else _truncate(r)
        for rnd in range(1, self.n_rounds + 1):
            merge_coros = []
            merge_names = []
            for agent in self.peers:
                peer = random.choice([p for p in self.peers if p is not agent])
                merge_prompt = (
                    f"Your knowledge:\n{states[agent.name]}\n\n"
                    f"From {peer.name}:\n{states[peer.name]}\n\nMerge accurately."
                )
                merge_coros.append(self._call_async(agent, ctx, merge_prompt))
                merge_names.append(agent.name)
            raw_merged = await asyncio.gather(*merge_coros, return_exceptions=True)
            for nm, val in zip(merge_names, raw_merged):
                if not isinstance(val, BaseException):
                    states[nm] = _truncate(val)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Gossip round {rnd}/{self.n_rounds}",
            )
        all_states = "\n\n".join(f"[{n}]: {s}" for n, s in states.items())
        summary_prompt = (
            f"Task: {ctx.user_message}\n\nStates:\n{all_states}\n\nSummarise."
        )
        final = await self._call_async(self.collector, ctx, summary_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": self.n_rounds,
                "peers": len(self.peers),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# AuctionAgent — Vickrey, 1961
# ═══════════════════════════════════════════════════════════════════════════════


class AuctionAgent(BaseAgent):
    """Auction: agents bid on tasks, highest bidder executes.

    Each sub-agent receives the task and produces a bid (via
    ``bid_fn``).  The highest bidder wins and executes the full task.
    Useful for self-selection among specialists.

    Reference: Vickrey, *Counterspeculation, Auctions, and Competitive
    Sealed Tenders*, 1961.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Bidding agents.
        bid_fn: ``(agent_name: str, response: str) -> float``.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        bid_fn: Callable[[str, str], float] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=list(sub_agents or []), **kwargs,
        )
        self.bid_fn = bid_fn or (lambda _n, r: float(len(r)))
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "Requires sub_agents.")
            return
        bid_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Explain briefly why you are well-suited for this task and "
            "how confident you are (1-10)."
        )
        bids: list[tuple[float, BaseAgent, str]] = []
        with ThreadPoolExecutor(max_workers=min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = {
                pool.submit(self._call_sync, a, ctx, bid_prompt): a
                for a in self.sub_agents
            }
            for f in as_completed(futs):
                a = futs[f]
                resp = f.result(timeout=_FUTURE_TIMEOUT)
                bids.append((self.bid_fn(a.name, resp), a, resp))
        bids.sort(key=lambda x: x[0], reverse=True)
        winner_score, winner, _ = bids[0]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Winner: {winner.name} (bid={winner_score:.2f})",
        )
        answer = self._call_sync(winner, ctx, ctx.user_message)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner.name,
                "bid": winner_score,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "Requires sub_agents.")
            return
        bid_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Explain why you are suited for this task (1-10 confidence)."
        )
        coros = [self._call_async(a, ctx, bid_prompt) for a in self.sub_agents]
        raw_responses = await asyncio.gather(*coros, return_exceptions=True)
        bids = [
            (self.bid_fn(a.name, r), a)
            for a, r in zip(self.sub_agents, raw_responses)
            if not isinstance(r, BaseException)
        ]
        bids.sort(key=lambda x: x[0], reverse=True)
        winner_score, winner = bids[0]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Winner: {winner.name} (bid={winner_score:.2f})",
        )
        answer = await self._call_async(winner, ctx, ctx.user_message)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "winner": winner.name,
                "bid": winner_score,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# DelphiMethodAgent — Dalkey & Helmer, 1963
# ═══════════════════════════════════════════════════════════════════════════════


class DelphiMethodAgent(BaseAgent):
    """Delphi Method: anonymous iterative expert consensus.

    Each round, every expert agent answers independently.  A facilitator
    shares anonymised summaries back.  Experts revise.  After
    ``n_rounds`` the facilitator produces the final consensus.

    Reference: Dalkey & Helmer, *An Experimental Application of the
    DELPHI Method*, RAND Corp., 1963.

    Args:
        name: Agent name.
        description: Human-readable description.
        experts: List of expert agents.
        facilitator: Summarises and synthesises rounds.
        n_rounds: Delphi rounds.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        experts: list[BaseAgent] | None = None,
        facilitator: BaseAgent | None = None,
        n_rounds: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        agents = list(experts or [])
        if facilitator and facilitator not in agents:
            agents.append(facilitator)
        super().__init__(name=name, description=description, sub_agents=agents, **kwargs)
        self.experts = list(experts or [])
        self.facilitator = facilitator or (self.experts[0] if self.experts else None)
        self.n_rounds = n_rounds
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.experts or not self.facilitator:
            yield Event(EventType.ERROR, self.name, "Requires experts + facilitator.")
            return
        summary = ""
        for rnd in range(1, self.n_rounds + 1):
            prompt_base = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Previous round summary:\n{summary}\n\n" if summary else "")
                + "Provide your independent expert opinion."
            )
            opinions = []
            with ThreadPoolExecutor(max_workers=min(len(self.experts), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
                futs = [pool.submit(self._call_sync, e, ctx, prompt_base) for e in self.experts]
                opinions = [f.result(timeout=_FUTURE_TIMEOUT) for f in as_completed(futs)]
            anon = "\n\n".join(f"Expert {i+1}: {_truncate(o)}" for i, o in enumerate(opinions))
            summary_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Round {rnd} responses (anonymous):\n{anon}\n\n"
                "Summarise the key points of agreement and disagreement."
            )
            summary = self._call_sync(self.facilitator, ctx, summary_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Delphi round {rnd}/{self.n_rounds}",
            )
        final_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Final Delphi summary:\n{summary}\n\n"
            "Produce the consensus answer."
        )
        final = self._call_sync(self.facilitator, ctx, final_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": self.n_rounds,
                "experts": len(self.experts),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.experts or not self.facilitator:
            yield Event(EventType.ERROR, self.name, "Requires experts + facilitator.")
            return
        summary = ""
        for rnd in range(1, self.n_rounds + 1):
            prompt_base = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Previous summary:\n{summary}\n\n" if summary else "")
                + "Provide your independent expert opinion."
            )
            coros = [self._call_async(e, ctx, prompt_base) for e in self.experts]
            raw_opinions = await asyncio.gather(*coros, return_exceptions=True)
            opinions = [r for r in raw_opinions if not isinstance(r, BaseException)]
            anon = "\n\n".join(f"Expert {i+1}: {_truncate(o)}" for i, o in enumerate(opinions))
            summary_prompt = (
                f"Round {rnd} responses:\n{anon}\n\nSummarise agreement/disagreement."
            )
            summary = await self._call_async(self.facilitator, ctx, summary_prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Delphi round {rnd}/{self.n_rounds}",
            )
        final_prompt = (
            f"Task: {ctx.user_message}\n\nFinal summary:\n{summary}\n\nConsensus answer."
        )
        final = await self._call_async(self.facilitator, ctx, final_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, final)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "rounds": self.n_rounds,
                "experts": len(self.experts),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# NominalGroupAgent — Delbecq et al., 1971
# ═══════════════════════════════════════════════════════════════════════════════


class NominalGroupAgent(BaseAgent):
    """Nominal Group Technique: structured idea generation + ranking.

    Phase 1 – Each agent silently generates ideas.
    Phase 2 – Ideas are shared (round-robin).
    Phase 3 – Each agent ranks all ideas.
    Phase 4 – Rankings are tallied; the top idea wins.

    Reference: Delbecq, Van de Ven & Gustafson, *Group Techniques for
    Program Planning*, 1971.

    Args:
        name: Agent name.
        description: Human-readable description.
        sub_agents: Participating agents.
        n_top: Number of top ideas each agent ranks.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        n_top: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, description=description,
            sub_agents=list(sub_agents or []), **kwargs,
        )
        self.n_top = n_top
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "Requires sub_agents.")
            return
        # Phase 1 — silent generation
        ideas: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(self.sub_agents), _MAX_DEFAULT_PARALLEL_WORKERS)) as pool:
            futs = [
                pool.submit(self._call_sync, a, ctx, ctx.user_message)
                for a in self.sub_agents
            ]
            ideas = [_truncate(f.result(timeout=_FUTURE_TIMEOUT)) for f in as_completed(futs)]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Phase 1: {len(ideas)} ideas generated",
        )
        # Phase 2 — share
        idea_list = "\n".join(f"  [{i+1}] {idea[:200]}" for i, idea in enumerate(ideas))
        # Phase 3 — rank
        tally: dict[int, int] = {}
        for agent in self.sub_agents:
            rank_prompt = (
                f"Task: {ctx.user_message}\n\nIdeas:\n{idea_list}\n\n"
                f"Rank the top {self.n_top} ideas by number (best first).  "
                "Reply as a comma-separated list of numbers."
            )
            raw = self._call_sync(agent, ctx, rank_prompt)
            nums = _DIGITS_RE.findall(raw)
            for pos, num_s in enumerate(nums[: self.n_top]):
                idx = int(num_s)
                if 1 <= idx <= len(ideas):
                    tally[idx] = tally.get(idx, 0) + (self.n_top - pos)
        # Phase 4 — winner
        if tally:
            winner_idx = max(tally, key=lambda k: tally[k])
            answer = ideas[winner_idx - 1]
        else:
            answer = ideas[0]
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_ideas": len(ideas),
                "tally": tally,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.sub_agents:
            yield Event(EventType.ERROR, self.name, "Requires sub_agents.")
            return
        coros = [self._call_async(a, ctx, ctx.user_message) for a in self.sub_agents]
        raw_ideas = await asyncio.gather(*coros, return_exceptions=True)
        ideas = [r for r in raw_ideas if not isinstance(r, BaseException)]
        yield Event(
            EventType.STATE_UPDATE, self.name,
            f"Phase 1: {len(ideas)} ideas generated",
        )
        idea_list = "\n".join(f"  [{i+1}] {idea[:200]}" for i, idea in enumerate(ideas))
        tally: dict[int, int] = {}
        rank_coros = []
        for agent in self.sub_agents:
            rank_prompt = (
                f"Task: {ctx.user_message}\n\nIdeas:\n{idea_list}\n\n"
                f"Rank top {self.n_top} by number.  Comma-separated."
            )
            rank_coros.append(self._call_async(agent, ctx, rank_prompt))
        rank_results = await asyncio.gather(*rank_coros, return_exceptions=True)
        for raw in rank_results:
            if isinstance(raw, BaseException):
                continue
            nums = _DIGITS_RE.findall(raw)
            for pos, num_s in enumerate(nums[: self.n_top]):
                idx = int(num_s)
                if 1 <= idx <= len(ideas):
                    tally[idx] = tally.get(idx, 0) + (self.n_top - pos)
        if tally:
            winner_idx = max(tally, key=lambda k: tally[k])
            answer = ideas[winner_idx - 1]
        else:
            answer = ideas[0]
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_ideas": len(ideas),
                "tally": tally,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# ActiveRetrievalAgent — Jiang et al., 2023 (FLARE)
# ═══════════════════════════════════════════════════════════════════════════════


class ActiveRetrievalAgent(BaseAgent):
    """Active Retrieval (FLARE): retrieve only when confidence is low.

    The agent generates a draft answer.  A ``confidence_fn`` checks it.
    If below ``threshold``, a ``retriever`` agent fetches extra context
    and the solver re-generates.  This loop runs up to ``max_retrievals``
    times.

    Reference: Jiang et al., *Active Retrieval Augmented Generation*,
    2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates draft answers.
        retriever: Fetches additional context.
        confidence_fn: ``(response: str) -> float`` in [0, 1].
        threshold: Below this → retrieve more.
        max_retrievals: Maximum retrieval cycles.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        retriever: BaseAgent | None = None,
        confidence_fn: Callable[[str], float] | None = None,
        threshold: float = 0.5,
        max_retrievals: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [agent, retriever] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.retriever = retriever or agent
        self.confidence_fn = confidence_fn or (lambda _: 0.8)
        self.threshold = threshold
        self.max_retrievals = max_retrievals
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        extra_context = ""
        answer = ""
        for r in range(self.max_retrievals + 1):
            gen_prompt = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Additional context:\n{extra_context}\n\n" if extra_context else "")
                + "Provide your answer."
            )
            answer = self._call_sync(self.agent, ctx, gen_prompt)
            conf = self.confidence_fn(answer)
            if conf >= self.threshold or r == self.max_retrievals:
                break
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Confidence {conf:.2f} < {self.threshold} — retrieving more",
            )
            retrieve_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current draft:\n{_truncate(answer)}\n\n"
                "What additional information is needed?  Retrieve it."
            )
            extra_context += "\n" + _truncate(
                self._call_sync(self.retriever, ctx, retrieve_prompt),
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "retrievals": r,
                "final_confidence": self.confidence_fn(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        extra_context = ""
        answer = ""
        for r in range(self.max_retrievals + 1):
            gen_prompt = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Context:\n{extra_context}\n\n" if extra_context else "")
                + "Answer."
            )
            answer = await self._call_async(self.agent, ctx, gen_prompt)
            conf = self.confidence_fn(answer)
            if conf >= self.threshold or r == self.max_retrievals:
                break
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Confidence {conf:.2f} — retrieving",
            )
            ret_prompt = (
                f"Task: {ctx.user_message}\nDraft:\n{_truncate(answer)}\nRetrieve more info."
            )
            extra_context += "\n" + _truncate(
                await self._call_async(self.retriever, ctx, ret_prompt),
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "retrievals": r,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# IterativeRetrievalAgent — Trivedi et al., 2023 (IRCoT)
# ═══════════════════════════════════════════════════════════════════════════════


class IterativeRetrievalAgent(BaseAgent):
    """Iterative Retrieval (IRCoT): interleave CoT reasoning and retrieval.

    Each step the agent writes one reasoning step.  A retriever then
    fetches supporting evidence for that step.  The next reasoning step
    sees all prior steps + evidence.

    Reference: Trivedi et al., *Interleaving Retrieval with Chain-of-
    Thought Reasoning*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Performs reasoning steps.
        retriever: Fetches evidence per step.
        n_steps: Number of reasoning-retrieval cycles.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        retriever: BaseAgent | None = None,
        n_steps: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [agent, retriever] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.retriever = retriever or agent
        self.n_steps = n_steps
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        chain: list[str] = []
        evidence: list[str] = []
        for step in range(1, self.n_steps + 1):
            ctx_str = ""
            for i, (c, e) in enumerate(zip(chain, evidence)):
                ctx_str += f"Step {i+1}: {c}\nEvidence: {e}\n\n"
            reason_prompt = (
                f"Task: {ctx.user_message}\n\n{ctx_str}"
                f"Write reasoning step {step} (one sentence)."
            )
            thought = self._call_sync(self.agent, ctx, reason_prompt)
            chain.append(_truncate(thought))
            ret_prompt = (
                f"Find evidence supporting: {_truncate(thought)}"
            )
            ev_text = self._call_sync(self.retriever, ctx, ret_prompt)
            evidence.append(_truncate(ev_text))
            yield Event(EventType.STATE_UPDATE, self.name, f"Step {step}/{self.n_steps}")
        # Final answer
        full_chain = "\n".join(
            f"Step {i+1}: {c} [Evidence: {e}]"
            for i, (c, e) in enumerate(zip(chain, evidence))
        )
        final_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Reasoning chain:\n{full_chain}\n\n"
            "Based on this chain, give the final answer."
        )
        answer = self._call_sync(self.agent, ctx, final_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": len(chain),
                "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        chain: list[str] = []
        evidence: list[str] = []
        for step in range(1, self.n_steps + 1):
            ctx_str = ""
            for i, (c, e) in enumerate(zip(chain, evidence)):
                ctx_str += f"Step {i+1}: {c}\nEvidence: {e}\n\n"
            reason_prompt = (
                f"Task: {ctx.user_message}\n\n{ctx_str}"
                f"Write step {step} (one sentence)."
            )
            thought = await self._call_async(self.agent, ctx, reason_prompt)
            chain.append(_truncate(thought))
            ev_text = await self._call_async(
                self.retriever, ctx, f"Find evidence for: {_truncate(thought)}",
            )
            evidence.append(_truncate(ev_text))
            yield Event(EventType.STATE_UPDATE, self.name, f"Step {step}/{self.n_steps}")
        full_chain = "\n".join(
            f"Step {i+1}: {c} [Evidence: {e}]"
            for i, (c, e) in enumerate(zip(chain, evidence))
        )
        final_prompt = (
            f"Task: {ctx.user_message}\n\nChain:\n{full_chain}\n\nFinal answer."
        )
        answer = await self._call_async(self.agent, ctx, final_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"steps": len(chain)})


# ═══════════════════════════════════════════════════════════════════════════════
# PromptChainAgent — Wu et al., 2022
# ═══════════════════════════════════════════════════════════════════════════════


class PromptChainAgent(BaseAgent):
    """Prompt Chaining: explicit multi-step prompt pipeline.

    A list of ``prompts`` is executed sequentially.  Each prompt can
    contain ``{previous}`` which is replaced by the output of the
    previous step.

    Reference: Wu et al., *AI Chains: Transparent and Controllable
    Human-AI Interaction by Chaining LLM Steps*, 2022.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Executes each prompt step.
        prompts: List of prompt templates (use ``{input}`` and
                 ``{previous}``).
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        prompts: list[str] | None = None,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.prompts = list(prompts or ["{input}"])
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        previous = ""
        for i, tmpl in enumerate(self.prompts):
            prompt = tmpl.replace("{input}", ctx.user_message).replace("{previous}", _truncate(previous))
            previous = self._call_sync(ctx, prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {i+1}/{len(self.prompts)}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, previous)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": len(self.prompts),
                "answer": _truncate(previous),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        previous = ""
        for i, tmpl in enumerate(self.prompts):
            prompt = tmpl.replace("{input}", ctx.user_message).replace("{previous}", _truncate(previous))
            previous = await self._call_async(ctx, prompt)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {i+1}/{len(self.prompts)}",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, previous)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": len(self.prompts),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# HypothesisTestingAgent — Popper, 1959
# ═══════════════════════════════════════════════════════════════════════════════


class HypothesisTestingAgent(BaseAgent):
    """Hypothesis Testing: generate, test, falsify, refine.

    Phase 1 – *generate* ``n_hypotheses`` candidate hypotheses.
    Phase 2 – *test* each using a critic / tester agent, looking for
    counter-evidence or flaws.
    Phase 3 – *refine*: the surviving or best hypothesis is refined.

    Reference: Popper, *The Logic of Scientific Discovery*, 1959.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates and refines hypotheses.
        tester: Tests / attempts to falsify hypotheses.
        n_hypotheses: Number of initial hypotheses.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        tester: BaseAgent | None = None,
        n_hypotheses: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [agent, tester] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.tester = tester or agent
        self.n_hypotheses = n_hypotheses
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        gen_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Generate {self.n_hypotheses} distinct hypotheses that could "
            "answer this task.  Number them."
        )
        hypotheses_raw = self._call_sync(self.agent, ctx, gen_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Hypotheses generated")
        test_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Hypotheses:\n{_truncate(hypotheses_raw)}\n\n"
            "For each hypothesis, attempt to FALSIFY it.  Identify flaws, "
            "counter-examples, or missing evidence.  Mark surviving ones."
        )
        test_result = self._call_sync(self.tester, ctx, test_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Hypotheses tested")
        refine_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Test results:\n{_truncate(test_result)}\n\n"
            "Based on the surviving evidence, produce a refined final answer."
        )
        answer = self._call_sync(self.agent, ctx, refine_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_hypotheses": self.n_hypotheses,
                "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        gen_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Generate {self.n_hypotheses} hypotheses.  Number them."
        )
        hypotheses_raw = await self._call_async(self.agent, ctx, gen_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Hypotheses generated")
        test_prompt = (
            f"Hypotheses:\n{_truncate(hypotheses_raw)}\n\n"
            "Falsify each.  Mark survivors."
        )
        test_result = await self._call_async(self.tester, ctx, test_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Tested")
        refine_prompt = (
            f"Task: {ctx.user_message}\nTests:\n{_truncate(test_result)}\n\nRefined answer."
        )
        answer = await self._call_async(self.agent, ctx, refine_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_hypotheses": self.n_hypotheses,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# SkillLibraryAgent — Wang et al., 2023 (Voyager)
# ═══════════════════════════════════════════════════════════════════════════════


class SkillLibraryAgent(BaseAgent):
    """Skill Library: accumulate reusable skills and retrieve them.

    Maintains a library of (name, description, code/answer) tuples.
    For each task, the most relevant skill is retrieved.  If none fits,
    the agent solves from scratch and the result is stored as a new skill.

    Reference: Wang et al., *Voyager: An Open-Ended Embodied Agent with
    Large Language Models*, 2023.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Solves tasks and creates new skills.
        initial_skills: Seed library.
        max_skills: Maximum library size.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        initial_skills: list[dict[str, str]] | None = None,
        max_skills: int = 50,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.skills: list[dict[str, str]] = list(initial_skills or [])
        self._skills_lock = threading.Lock()
        self.max_skills = max_skills
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _find_skill(self, ctx: InvocationContext) -> dict[str, str] | None:
        if not self.skills:
            return None
        lib_str = "\n".join(
            f"  [{i+1}] {s.get('name', '?')}: {s.get('description', '')[:80]}"
            for i, s in enumerate(self.skills)
        )
        prompt = (
            f"Task: {ctx.user_message}\n\nSkill library:\n{lib_str}\n\n"
            "Which skill number is most relevant?  Reply '0' if none fits."
        )
        reply = self._call_sync(ctx, prompt).strip()
        try:
            idx = int(reply) - 1
            return self.skills[idx] if 0 <= idx < len(self.skills) else None
        except (ValueError, IndexError):
            return None

    async def _find_skill_async(self, ctx: InvocationContext) -> dict[str, str] | None:
        if not self.skills:
            return None
        lib_str = "\n".join(
            f"  [{i+1}] {s.get('name', '?')}: {s.get('description', '')[:80]}"
            for i, s in enumerate(self.skills)
        )
        prompt = (
            f"Task: {ctx.user_message}\n\nSkills:\n{lib_str}\n\n"
            "Most relevant number?  '0' if none."
        )
        reply = (await self._call_async(ctx, prompt)).strip()
        try:
            idx = int(reply) - 1
            return self.skills[idx] if 0 <= idx < len(self.skills) else None
        except (ValueError, IndexError):
            return None

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        skill = self._find_skill(ctx)
        if skill:
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Reusing skill: {skill.get('name', '?')}",
            )
            adapt_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Reuse this skill:\n{skill.get('code', skill.get('answer', ''))}\n\n"
                "Adapt it to the current task."
            )
            answer = self._call_sync(ctx, adapt_prompt)
        else:
            yield Event(EventType.STATE_UPDATE, self.name, "No matching skill — solving fresh")
            answer = self._call_sync(ctx, ctx.user_message)
            name_prompt = (
                f"Give a short name (2-4 words) for this skill:\n{answer[:200]}"
            )
            skill_name = self._call_sync(ctx, name_prompt).strip()[:60]
            new_skill = {
                "name": skill_name,
                "description": ctx.user_message[:120],
                "answer": answer[:500],
            }
            with self._skills_lock:
                self.skills.append(new_skill)
                if len(self.skills) > self.max_skills:
                    self.skills.pop(0)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "library_size": len(self.skills),
                "skill_reused": skill is not None,
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        skill = await self._find_skill_async(ctx)
        if skill:
            yield Event(EventType.STATE_UPDATE, self.name, f"Reusing: {skill.get('name')}")
            answer = await self._call_async(
                ctx,
                f"Reuse skill:\n{skill.get('answer', '')}\nAdapt to: {ctx.user_message}",
            )
        else:
            yield Event(EventType.STATE_UPDATE, self.name, "Solving fresh")
            answer = await self._call_async(ctx, ctx.user_message)
            skill_name = (await self._call_async(
                ctx, f"Short name for:\n{answer[:200]}",
            )).strip()[:60]
            with self._skills_lock:
                self.skills.append({
                    "name": skill_name,
                    "description": ctx.user_message[:120],
                    "answer": answer[:500],
                })
                if len(self.skills) > self.max_skills:
                    self.skills.pop(0)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "library_size": len(self.skills),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# RecursiveCriticAgent — nested critique & revision
# ═══════════════════════════════════════════════════════════════════════════════


class RecursiveCriticAgent(BaseAgent):
    """Recursive Critic: nested critique at increasing depth.

    Level 0 – generate a draft.
    Level 1 – critique the draft.
    Level 2 – critique the critique.
    …up to ``depth``.  Each deeper level provides meta-feedback.
    The final revision incorporates all layers.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Performs generation and critique.
        depth: Critique nesting depth.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        depth: int = 2,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.depth = depth
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        draft = self._call_sync(ctx, ctx.user_message)
        layers: list[str] = [draft]
        for d in range(1, self.depth + 1):
            critique_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"{'Draft' if d == 1 else f'Level-{d-1} critique'}:\n{_truncate(layers[-1])}\n\n"
                f"Provide a level-{d} critique: identify weaknesses, errors, "
                "and suggest improvements."
            )
            critique = self._call_sync(ctx, critique_prompt)
            layers.append(critique)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Critique depth {d}/{self.depth}",
            )
        all_feedback = "\n\n".join(
            f"Level {i}: {l[:300]}" for i, l in enumerate(layers)
        )
        revise_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Multi-level feedback:\n{all_feedback}\n\n"
            "Produce the final revised answer incorporating all feedback."
        )
        answer = self._call_sync(ctx, revise_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "depth": self.depth,
                "layers": len(layers),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        draft = await self._call_async(ctx, ctx.user_message)
        layers: list[str] = [draft]
        for d in range(1, self.depth + 1):
            critique_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Previous:\n{_truncate(layers[-1])}\n\nLevel-{d} critique."
            )
            layers.append(await self._call_async(ctx, critique_prompt))
            yield Event(EventType.STATE_UPDATE, self.name, f"Depth {d}/{self.depth}")
        all_fb = "\n\n".join(f"Level {i}: {l[:300]}" for i, l in enumerate(layers))
        answer = await self._call_async(
            ctx,
            f"Task: {ctx.user_message}\n\nFeedback:\n{all_fb}\n\nRevised answer.",
        )
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"depth": self.depth})


# ═══════════════════════════════════════════════════════════════════════════════
# DemonstrateSearchPredictAgent — Khattab et al., 2022 (DSP)
# ═══════════════════════════════════════════════════════════════════════════════


class DemonstrateSearchPredictAgent(BaseAgent):
    """Demonstrate-Search-Predict (DSP): 3-stage pipeline.

    Stage 1 – *Demonstrate*: the agent generates few-shot demonstrations
    for the task.
    Stage 2 – *Search*: a retriever finds supporting passages.
    Stage 3 – *Predict*: the agent produces the final answer using
    demos + retrieved passages.

    Reference: Khattab et al., *Demonstrate-Search-Predict*, 2022.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Generates demos and final prediction.
        retriever: Searches for supporting passages.
        n_demos: Number of demonstrations to generate.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        retriever: BaseAgent | None = None,
        n_demos: int = 3,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [a for a in [agent, retriever] if a]
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.retriever = retriever or agent
        self.n_demos = n_demos
        self.result_key = result_key

    def _call_sync(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in agent._run_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(
        self, agent: BaseAgent, ctx: InvocationContext, prompt: str,
    ) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in agent._run_async_impl_traced(sub_ctx):
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        # Demonstrate
        demo_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Create {self.n_demos} example input-output demonstrations "
            "for this type of task."
        )
        demos = self._call_sync(self.agent, ctx, demo_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Demonstrations generated")
        # Search
        search_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Find relevant passages or evidence to help answer this task."
        )
        passages = self._call_sync(self.retriever, ctx, search_prompt)
        yield Event(EventType.STATE_UPDATE, self.name, "Passages retrieved")
        # Predict
        predict_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Demonstrations:\n{_truncate(demos)}\n\n"
            f"Retrieved passages:\n{_truncate(passages)}\n\n"
            "Using the demonstrations and passages, provide the final answer."
        )
        answer = self._call_sync(self.agent, ctx, predict_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "n_demos": self.n_demos,
                "answer": _truncate(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        demo_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Create {self.n_demos} demonstrations."
        )
        raw_pair = await asyncio.gather(
            self._call_async(self.agent, ctx, demo_prompt),
            self._call_async(
                self.retriever, ctx,
                f"Find evidence for: {ctx.user_message}",
            ),
            return_exceptions=True,
        )
        if any(isinstance(r, BaseException) for r in raw_pair):
            yield Event(EventType.ERROR, self.name, "Demo/retrieval failed.")
            return
        demos, passages = raw_pair
        yield Event(EventType.STATE_UPDATE, self.name, "Demos + passages ready")
        predict_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Demos:\n{_truncate(demos)}\nPassages:\n{_truncate(passages)}\n\nFinal answer."
        )
        answer = await self._call_async(self.agent, ctx, predict_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"n_demos": self.n_demos})


# ═══════════════════════════════════════════════════════════════════════════════
# DoubleLoopLearningAgent — Argyris & Schön, 1978
# ═══════════════════════════════════════════════════════════════════════════════


class DoubleLoopLearningAgent(BaseAgent):
    """Double-Loop Learning: question *assumptions*, not just outcomes.

    Loop 1 (single-loop) – solve and evaluate.
    Loop 2 (double-loop) – if the result is unsatisfactory, question the
    underlying assumptions and mental models, revise them, then re-solve.

    Reference: Argyris & Schön, *Organizational Learning*, 1978.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Performs solving, evaluation, and meta-reflection.
        max_loops: Maximum double-loop iterations.
        quality_fn: ``(response: str) -> float`` quality score [0–1].
        threshold: Quality threshold to accept.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        max_loops: int = 3,
        quality_fn: Callable[[str], float] | None = None,
        threshold: float = 0.7,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.max_loops = max_loops
        self.quality_fn = quality_fn or (lambda _: 0.8)
        self.threshold = threshold
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        assumptions = ""
        answer = ""
        for loop in range(1, self.max_loops + 1):
            solve_prompt = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Revised assumptions:\n{_truncate(assumptions)}\n\n" if assumptions else "")
                + "Solve the task."
            )
            answer = self._call_sync(ctx, solve_prompt)
            quality = self.quality_fn(answer)
            if quality >= self.threshold:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Loop {loop}: quality {quality:.2f} ≥ {self.threshold} — accepted",
                )
                break
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Loop {loop}: quality {quality:.2f} < {self.threshold} — double-loop",
            )
            meta_prompt = (
                f"Task: {ctx.user_message}\n\n"
                f"Current answer (quality={quality:.2f}):\n{_truncate(answer)}\n\n"
                "Question your underlying ASSUMPTIONS and mental models.  "
                "What implicit beliefs led to this answer?  How should they "
                "be revised?"
            )
            assumptions = self._call_sync(ctx, meta_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "loops": loop,
                "final_quality": self.quality_fn(answer),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        assumptions = ""
        answer = ""
        for loop in range(1, self.max_loops + 1):
            solve_prompt = (
                f"Task: {ctx.user_message}\n\n"
                + (f"Assumptions:\n{_truncate(assumptions)}\n\n" if assumptions else "")
                + "Solve."
            )
            answer = await self._call_async(ctx, solve_prompt)
            quality = self.quality_fn(answer)
            if quality >= self.threshold:
                yield Event(
                    EventType.STATE_UPDATE, self.name,
                    f"Loop {loop}: accepted ({quality:.2f})",
                )
                break
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Loop {loop}: double-loop ({quality:.2f})",
            )
            assumptions = await self._call_async(
                ctx,
                f"Answer (q={quality:.2f}):\n{_truncate(answer)}\n\nQuestion assumptions.",
            )
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {"loops": loop})


# ═══════════════════════════════════════════════════════════════════════════════
# AgendaAgent — Allen & Ferguson, 1994
# ═══════════════════════════════════════════════════════════════════════════════


class AgendaAgent(BaseAgent):
    """Agenda-Based Planning: maintain a priority queue of sub-goals.

    The agent decomposes the task into an *agenda* of prioritised
    sub-goals.  Each iteration it pops the highest-priority item,
    resolves it, and may add new sub-goals.  Continues until the agenda
    is empty or ``max_steps`` is reached.

    Reference: Allen & Ferguson, *Actions and Events in Interval
    Temporal Logic*, 1994.

    Args:
        name: Agent name.
        description: Human-readable description.
        agent: Executes sub-goals and manages the agenda.
        max_steps: Maximum sub-goal resolution steps.
        result_key: Optional session state key.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        agent: BaseAgent | None = None,
        max_steps: int = 10,
        result_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        subs = [agent] if agent else []
        super().__init__(name=name, description=description, sub_agents=subs, **kwargs)
        self.agent = agent
        self.max_steps = max_steps
        self.result_key = result_key

    def _call_sync(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        for ev in self.agent._run_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    async def _call_async(self, ctx: InvocationContext, prompt: str) -> str:
        sub_ctx = InvocationContext(
            session=Session(), user_message=prompt,
            parent_agent=self, trace_collector=ctx.trace_collector,
        )
        out = ""
        async for ev in self.agent._run_async_impl_traced(sub_ctx):  # type: ignore[union-attr]
            if ev.event_type == EventType.AGENT_MESSAGE:
                out = ev.content
        return out

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        # Decompose into agenda
        decompose_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Decompose this task into a prioritised list of sub-goals.  "
            "Format: one sub-goal per line, most important first."
        )
        raw_agenda = self._call_sync(ctx, decompose_prompt)
        agenda = [line.strip() for line in raw_agenda.strip().splitlines() if line.strip()]
        results: list[str] = []
        step = 0
        while agenda and step < self.max_steps:
            goal = agenda.pop(0)
            step += 1
            resolve_prompt = (
                f"Main task: {ctx.user_message}\n\n"
                f"Sub-goal to resolve: {goal}\n\n"
                + (f"Prior results:\n" + "\n".join(results[-3:]) + "\n\n" if results else "")
                + "Resolve this sub-goal.  If it requires further decomposition, "
                "list new sub-goals prefixed with 'NEW_GOAL: '."
            )
            response = self._call_sync(ctx, resolve_prompt)
            results.append(f"[{goal}]: {_truncate(response)}")
            # Extract new sub-goals
            for line in response.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("NEW_GOAL:"):
                    new_goal = stripped[9:].strip()
                    if new_goal:
                        agenda.append(new_goal)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}: resolved '{goal[:50]}'  (agenda: {len(agenda)} remaining)",
            )
        # Synthesise
        all_results = "\n\n".join(results[-10:])
        synth_prompt = (
            f"Task: {ctx.user_message}\n\n"
            f"Resolved sub-goals:\n{all_results}\n\n"
            "Synthesise a final comprehensive answer."
        )
        answer = self._call_sync(ctx, synth_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": step,
                "remaining_agenda": len(agenda),
            })

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        if not self.agent:
            yield Event(EventType.ERROR, self.name, "Requires agent.")
            return
        decompose_prompt = (
            f"Task: {ctx.user_message}\n\n"
            "Decompose into prioritised sub-goals (one per line)."
        )
        raw_agenda = await self._call_async(ctx, decompose_prompt)
        agenda = [l.strip() for l in raw_agenda.strip().splitlines() if l.strip()]
        results: list[str] = []
        step = 0
        while agenda and step < self.max_steps:
            goal = agenda.pop(0)
            step += 1
            resolve_prompt = (
                f"Main task: {ctx.user_message}\nSub-goal: {goal}\n\n"
                + (f"Prior:\n" + "\n".join(results[-3:]) + "\n\n" if results else "")
                + "Resolve.  Prefix new sub-goals with 'NEW_GOAL: '."
            )
            response = await self._call_async(ctx, resolve_prompt)
            results.append(f"[{goal}]: {_truncate(response)}")
            for line in response.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("NEW_GOAL:"):
                    ng = stripped[9:].strip()
                    if ng:
                        agenda.append(ng)
            yield Event(
                EventType.STATE_UPDATE, self.name,
                f"Step {step}: '{goal[:50]}' (agenda: {len(agenda)})",
            )
        synth_prompt = (
            f"Task: {ctx.user_message}\n\nResults:\n"
            + "\n\n".join(results[-10:]) + "\n\nFinal answer."
        )
        answer = await self._call_async(ctx, synth_prompt)
        yield Event(EventType.AGENT_MESSAGE, self.name, answer)
        if self.result_key:
            ctx.session.state_set(self.result_key, {
                "steps": step, "remaining": len(agenda),
            })
