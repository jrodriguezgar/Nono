"""
Base agent primitives: BaseAgent, Event, Session, InvocationContext.

Core primitives for the Nono Agent Architecture (NAA).  Works standalone
with Nono's connector layer — no external agent-framework dependency
required.

Key concepts:
    - **Event**: An immutable record of something that happened during agent
      execution (user message, agent response, tool call, tool result, etc.).
    - **Session**: A single conversation thread holding a chronological list of
      Events and a mutable ``state`` dict.
    - **InvocationContext**: Everything an agent needs to execute: the session,
      the current user message, and the parent agent reference.
    - **BaseAgent**: Abstract base class that all agents extend.  Defines the
      ``run()`` / ``run_async()`` contract plus lifecycle callbacks
      (``before_agent``, ``after_agent``, ``before_tool``, ``after_tool``).
    - **HookManager integration**: Agents can carry a ``HookManager`` that
      fires lifecycle hooks (PreAgentRun, PostAgentRun, PreToolUse, etc.)
      at deterministic points during execution.
"""

from __future__ import annotations

import logging
import threading
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Iterator, Optional

logger = logging.getLogger("Nono.Agent")


# ── Event ─────────────────────────────────────────────────────────────────────

class EventType(Enum):
    """Types of events that can occur during agent execution."""
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TEXT_CHUNK = "text_chunk"
    TOOL_CALL = "tool_call"
    TOOL_CALL_CHUNK = "tool_call_chunk"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    AGENT_TRANSFER = "agent_transfer"
    HUMAN_INPUT_REQUEST = "human_input_request"
    HUMAN_INPUT_RESPONSE = "human_input_response"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    """An immutable record of an action during agent execution.

    Args:
        event_type: Category of the event.
        author: Name of the agent or "user" that produced this event.
        content: Textual payload (message body, tool output, etc.).
        data: Arbitrary structured data (tool args, state delta, etc.).
        timestamp: UTC timestamp (auto-set on creation).
        event_id: Unique identifier (auto-generated UUID).
    """
    event_type: EventType
    author: str
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ── SharedContent ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContentItem:
    """A single stored content entry.

    Args:
        name: Unique key for this content.
        data: The payload — ``str``, ``bytes``, ``dict``, or any serialisable.
        content_type: MIME type hint (e.g. ``"text/plain"``, ``"image/png"``).
        metadata: Extra information about the content.
        created_by: Name of the agent that saved this item.
        created_at: UTC timestamp when the item was stored.
    """
    name: str
    data: Any
    content_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Default capacity for SharedContent stores (0 = unlimited)
_DEFAULT_SHARED_CONTENT_MAX_ITEMS: int = 200

# Maximum size (in bytes) for a single SharedContent item (10 MB)
_MAX_CONTENT_ITEM_BYTES: int = 10_485_760


class SharedContent:
    """Named content store shared across all agents in a session.

    Agents and tools can **save** and **load** arbitrary content (text, bytes,
    dicts, images, files) by name.  All items are kept in-memory and scoped to
    the owning ``Session``.

    When *max_items* is set to a positive integer the store behaves as an
    LRU cache: the least-recently accessed item is evicted when the
    capacity is exceeded.  ``0`` (the default) means unlimited.

    Nono uses the term "SharedContent" instead of artefact/artifact stores.

    Args:
        max_items: Maximum number of content items to keep.  ``0`` = unlimited.

    Example:
        >>> store = SharedContent()
        >>> store.save("report", "Quarterly results ...", content_type="text/plain")
        >>> store.load("report").data
        'Quarterly results ...'
        >>> store.save("chart", b'\\x89PNG...', content_type="image/png", metadata={"width": 800})
        >>> store.names()
        ['report', 'chart']
    """

    def __init__(self, *, max_items: int = _DEFAULT_SHARED_CONTENT_MAX_ITEMS) -> None:
        self._items: OrderedDict[str, ContentItem] = OrderedDict()
        self.max_items = max_items
        self._lock = threading.Lock()

    def save(
        self,
        name: str,
        data: Any,
        *,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        created_by: str = "",
    ) -> ContentItem:
        """Save a content item (overwrites if the name already exists).

        Args:
            name: Unique key for this content.
            data: The payload to store.
            content_type: MIME type hint.
            metadata: Extra key-value information.
            created_by: Agent name that produced this content.

        Returns:
            The newly created ``ContentItem``.
        """
        # Reject oversized items to prevent memory exhaustion
        _size = len(data) if isinstance(data, (str, bytes, bytearray)) else len(str(data))
        if _size > _MAX_CONTENT_ITEM_BYTES:
            raise ValueError(
                f"Content item {name!r} exceeds size limit "
                f"({_size:,} > {_MAX_CONTENT_ITEM_BYTES:,} bytes)."
            )

        item = ContentItem(
            name=name,
            data=data,
            content_type=content_type,
            metadata=metadata or {},
            created_by=created_by,
        )
        with self._lock:
            # Move to end if overwriting (keeps LRU order correct)
            if name in self._items:
                self._items.move_to_end(name)
            self._items[name] = item

            # Evict oldest entry when capacity is exceeded
            if self.max_items > 0 and len(self._items) > self.max_items:
                evicted_key, _ = self._items.popitem(last=False)
                logger.debug("SharedContent: evicted %r (capacity=%d)", evicted_key, self.max_items)

        logger.debug("SharedContent: saved %r", name)
        return item

    def load(self, name: str) -> ContentItem | None:
        """Load a content item by name.

        Accessing an item marks it as recently used (LRU touch).

        Args:
            name: Key of the content to retrieve.

        Returns:
            The ``ContentItem``, or ``None`` if not found.
        """
        with self._lock:
            item = self._items.get(name)
            if item is not None:
                self._items.move_to_end(name)
            return item

    def names(self) -> list[str]:
        """Return all stored content names."""
        with self._lock:
            return list(self._items.keys())

    def delete(self, name: str) -> bool:
        """Delete a content item.

        Args:
            name: Key of the content to delete.

        Returns:
            ``True`` if the item existed and was deleted.
        """
        with self._lock:
            if name in self._items:
                del self._items[name]
                return True
            return False

    def clear(self) -> None:
        """Remove all stored content."""
        with self._lock:
            self._items.clear()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __repr__(self) -> str:
        with self._lock:
            return f"SharedContent(items={list(self._items.keys())})"


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """A single conversation thread with chronological events and mutable state.

    When a ``KeepInMind`` memory instance is provided, the session
    automatically loads prior conversation history and commits new events
    as they are added.  Memory is **optional** — omit it for stateless use.

    Args:
        session_id: Unique session identifier (auto-generated if omitted).
        state: Initial state dict.
        memory: Optional ``KeepInMind`` instance for persistent memory.
        max_events: Maximum number of events to retain. ``0`` (default)
            means unlimited.  When set, oldest events are evicted first.

    Example:
        >>> session = Session()
        >>> session.add_event(Event(EventType.USER_MESSAGE, "user", "Hello"))
        >>> len(session.events)
        1
    """

    def __init__(
        self,
        session_id: str | None = None,
        state: dict[str, Any] | None = None,
        memory: Any | None = None,
        max_events: int = 0,
    ) -> None:
        self.session_id: str = session_id or uuid.uuid4().hex[:16]
        self.state: dict[str, Any] = state or {}
        self.shared_content: SharedContent = SharedContent()
        self._events: deque[Event] = deque(maxlen=max_events if max_events > 0 else None)
        self.memory: Any | None = memory  # Optional[KeepInMind]
        self.max_events: int = max_events
        self._lock = threading.Lock()
        self.created_at: datetime = datetime.now(timezone.utc)

        # Auto-load prior conversation from memory
        if self.memory is not None:
            self._load_memory()

    @property
    def events(self) -> list[Event]:
        """Returns the chronological list of events."""
        with self._lock:
            return list(self._events)

    @property
    def last_event(self) -> Event | None:
        """Returns the most recent event, or None."""
        with self._lock:
            return self._events[-1] if self._events else None

    def add_event(self, event: Event) -> None:
        """Append an event to the session history.

        When memory is attached, user and agent messages are automatically
        committed to the persistent store.  When ``max_events`` is set,
        the oldest events are evicted to stay within the limit.

        Thread-safe: concurrent calls from ``ParallelAgent`` sub-agents
        are serialised.
        """
        with self._lock:
            self._events.append(event)
        self._commit_to_memory(event)

    def get_messages(self) -> list[dict[str, str]]:
        """Convert session events to the connector message format.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts suitable for
            ``GenerativeAIService.generate_completion()``.
        """
        messages: list[dict[str, str]] = []

        with self._lock:
            events_snapshot = list(self._events)

        for event in events_snapshot:
            if event.event_type == EventType.USER_MESSAGE:
                messages.append({"role": "user", "content": event.content})
            elif event.event_type == EventType.AGENT_MESSAGE:
                messages.append({"role": "assistant", "content": event.content})

        return messages

    # ── Thread-safe state helpers ──────────────────────────────────────

    def state_set(self, key: str, value: Any) -> None:
        """Set a state key under lock — safe for concurrent agents.

        Args:
            key: State key.
            value: Value to store.
        """
        with self._lock:
            self.state[key] = value

    def state_get(self, key: str, default: Any = None) -> Any:
        """Get a state value under lock — safe for concurrent agents.

        Args:
            key: State key.
            default: Value returned when key is absent.

        Returns:
            The stored value, or *default*.
        """
        with self._lock:
            return self.state.get(key, default)

    def state_update(self, mapping: dict[str, Any]) -> None:
        """Merge multiple keys into state under lock.

        Args:
            mapping: Dict of key/value pairs to merge.
        """
        with self._lock:
            self.state.update(mapping)

    # ── Memory helpers ─────────────────────────────────────────────────

    def _load_memory(self) -> None:
        """Load prior conversation turns from the memory store."""
        if self.memory is None:
            return

        try:
            remembered = self.memory.recall(self.session_id)

            for entry in remembered:
                if entry.role == "user":
                    etype = EventType.USER_MESSAGE
                elif entry.role == "assistant":
                    etype = EventType.AGENT_MESSAGE
                else:
                    continue

                self._events.append(Event(
                    event_type=etype,
                    author=entry.agent_name or entry.role,
                    content=entry.content,
                ))

            if remembered:
                logger.debug(
                    "Session %s: loaded %d entries from memory",
                    self.session_id, len(remembered),
                )
        except Exception:
            logger.warning(
                "Session %s: failed to load memory", self.session_id,
                exc_info=True,
            )

    def _commit_to_memory(self, event: Event) -> None:
        """Persist a user or agent message event to memory."""
        if self.memory is None:
            return

        if event.event_type not in (
            EventType.USER_MESSAGE,
            EventType.AGENT_MESSAGE,
        ):
            return

        role = (
            "user" if event.event_type == EventType.USER_MESSAGE
            else "assistant"
        )

        try:
            self.memory.append(
                session_id=self.session_id,
                role=role,
                content=event.content,
                agent_name=event.author,
            )
        except Exception:
            logger.warning(
                "Session %s: failed to commit to memory", self.session_id,
                exc_info=True,
            )

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"Session(id={self.session_id!r}, events={len(self._events)}, "
                f"state_keys={list(self.state.keys())})"
            )


# ── InvocationContext ─────────────────────────────────────────────────────────

# Maximum depth for recursive agent transfers to prevent stack overflow
def _default_max_transfer_depth() -> int:
    try:
        from nono.config import load_config as _lc
        val = _lc().get("agent.max_transfer_depth")
        if val is not None:
            return int(val)
    except Exception:
        pass
    return 10


MAX_TRANSFER_DEPTH: int = _default_max_transfer_depth()


@dataclass
class InvocationContext:
    """Everything an agent needs to execute a single turn.

    Args:
        session: The active session.
        user_message: The current user message to process.
        parent_agent: Reference to the parent agent (for sub-agent scenarios).
        trace_collector: Optional trace collector for observability.
        transfer_depth: Current depth of nested ``transfer_to_agent`` calls.
            Incremented automatically; raises ``RecursionError`` when
            ``MAX_TRANSFER_DEPTH`` is exceeded.
    """
    session: Session
    user_message: str = ""
    parent_agent: Optional[BaseAgent] = None
    trace_collector: Any = None  # Optional[TraceCollector] — avoiding circular import
    transfer_depth: int = 0


# ── Callback types ────────────────────────────────────────────────────────────

BeforeAgentCallback = Callable[["BaseAgent", InvocationContext], Optional[str]]
AfterAgentCallback = Callable[["BaseAgent", InvocationContext, str], Optional[str]]
BeforeToolCallback = Callable[["BaseAgent", str, dict], Optional[dict]]
AfterToolCallback = Callable[["BaseAgent", str, dict, Any], Optional[Any]]

# ── Orchestration lifecycle callbacks ─────────────────────────────────────────

OrchestrationStartCallback = Callable[[str, "Session"], None]
"""``(orchestrator_name, session) -> None``.

Fired once when an orchestration agent (Sequential, Parallel, Loop, …)
begins execution, before the first sub-agent runs."""

OrchestrationEndCallback = Callable[[str, "Session", int], None]
"""``(orchestrator_name, session, agents_executed) -> None``.

Fired once when an orchestration agent finishes, whether it completed
normally or was halted by ``on_between_agents``.  ``agents_executed`` is
the count of sub-agent invocations that ran."""

BetweenAgentsCallback = Callable[[str, Optional[str], "Session"], Optional[bool]]
"""``(completed_agent_name, next_agent_name | None, session) -> Optional[bool]``.

Fired between two sequential sub-agents — after one completes and before the
next begins (only in SequentialAgent, LoopAgent, and similar).
``next_agent_name`` is ``None`` when the orchestration is about to end.
Return ``False`` to halt execution early; any other value continues."""

AgentExecutingCallback = Callable[[str, "Session"], None]
"""``(sub_agent_name, session) -> None``.

Fired right before a sub-agent begins execution."""

AgentExecutedCallback = Callable[[str, "Session", Optional[str]], None]
"""``(sub_agent_name, session, error) -> None``.

Fired right after a sub-agent finishes execution.  ``error`` is ``None``
on success, or the error message string on failure."""


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for all Nono agents.

    Base contract for all Nono agents:
    - ``run()`` / ``run_async()`` execute the agent's logic.
    - Lifecycle callbacks (``before_agent``, ``after_agent``, ``before_tool``,
      ``after_tool``) allow hooking into execution without modifying core logic.
    - Sub-agents are declared via ``sub_agents`` for multi-agent orchestration.

    Each agent has two content scopes:
    - **Session** ``shared_content``: visible to all agents in the session.
    - **Agent** ``local_content``: private to this agent instance.

    Hooks:
    - Agents support a ``HookManager`` for lifecycle hooks (PreAgentRun,
      PostAgentRun, PreToolUse, PostToolUse, etc.) that fire at
      deterministic points during execution, independent of callbacks.

    Args:
        name: Unique agent name (used in logs and event authorship).
        description: Human-readable description of the agent's purpose.
        sub_agents: List of child agents this agent can delegate to.
        before_agent_callback: Called before the agent runs; return a string
            to short-circuit execution with that response.
        after_agent_callback: Called after the agent runs; can transform the
            response.
        before_tool_callback: Called before a tool is invoked.
        after_tool_callback: Called after a tool returns.
        hook_manager: Optional ``HookManager`` for lifecycle hooks.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        before_agent_callback: BeforeAgentCallback | None = None,
        after_agent_callback: AfterAgentCallback | None = None,
        before_tool_callback: BeforeToolCallback | None = None,
        after_tool_callback: AfterToolCallback | None = None,
        hook_manager: Any = None,
    ) -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = sub_agents or []
        self.local_content: SharedContent = SharedContent()
        self.before_agent_callback = before_agent_callback
        self.after_agent_callback = after_agent_callback
        self.before_tool_callback = before_tool_callback
        self.after_tool_callback = after_tool_callback
        self.hook_manager: Any = hook_manager  # Optional[HookManager]

        # Orchestration lifecycle hooks (set via fluent on_* methods)
        self._on_start: OrchestrationStartCallback | None = None
        self._on_end: OrchestrationEndCallback | None = None
        self._between_agents: BetweenAgentsCallback | None = None
        self._agent_executing: AgentExecutingCallback | None = None
        self._agent_executed: AgentExecutedCallback | None = None

    # ── Orchestration hook setters (fluent API) ─────────────────────────

    def on_start(self, callback: OrchestrationStartCallback) -> BaseAgent:
        """Register a callback fired when this orchestrator begins execution.

        Signature: ``(orchestrator_name, session) -> None``

        Args:
            callback: The start callback.

        Returns:
            ``self`` for fluent chaining.
        """
        self._on_start = callback
        return self

    def on_end(self, callback: OrchestrationEndCallback) -> BaseAgent:
        """Register a callback fired when this orchestrator finishes execution.

        Signature: ``(orchestrator_name, session, agents_executed) -> None``

        Args:
            callback: The end callback.

        Returns:
            ``self`` for fluent chaining.
        """
        self._on_end = callback
        return self

    def on_between_agents(self, callback: BetweenAgentsCallback) -> BaseAgent:
        """Register a callback fired between sequential sub-agent executions.

        Signature: ``(completed_agent, next_agent | None, session) -> Optional[bool]``

        Return ``False`` to halt execution early.  Only meaningful for
        agents that run sub-agents in sequence (``SequentialAgent``,
        ``LoopAgent``, etc.).

        Args:
            callback: The between-agents callback.

        Returns:
            ``self`` for fluent chaining.
        """
        self._between_agents = callback
        return self

    def on_agent_start(self, callback: AgentExecutingCallback) -> BaseAgent:
        """Register a callback fired before each sub-agent starts.

        Signature: ``(sub_agent_name, session) -> None``

        Args:
            callback: The agent-executing callback.

        Returns:
            ``self`` for fluent chaining.
        """
        self._agent_executing = callback
        return self

    def on_agent_end(self, callback: AgentExecutedCallback) -> BaseAgent:
        """Register a callback fired after each sub-agent finishes.

        Signature: ``(sub_agent_name, session, error) -> None``

        ``error`` is ``None`` on success, or the error message on failure.

        Args:
            callback: The agent-executed callback.

        Returns:
            ``self`` for fluent chaining.
        """
        self._agent_executed = callback
        return self

    # ── Hook helpers ──────────────────────────────────────────────────────

    def _fire_hook(
        self,
        event_name: str,
        ctx: InvocationContext | None = None,
        **extra_fields: Any,
    ) -> Any:
        """Fire a lifecycle hook if a ``HookManager`` is attached.

        Args:
            event_name: The ``HookEvent`` value string (e.g. ``"PreAgentRun"``).
            ctx: The current invocation context.
            **extra_fields: Additional fields to set on the ``HookContext``.

        Returns:
            ``HookResult`` if a manager fired the hook, else ``None``.
        """
        if self.hook_manager is None:
            return None

        from nono.hooks import HookEvent, HookContext

        try:
            event = HookEvent.from_string(event_name)
        except ValueError:
            return None

        hook_ctx = HookContext(
            event=event,
            session_id=ctx.session.session_id if ctx else "",
            agent_name=self.name,
        )

        # Set extra fields on the context
        for key, value in extra_fields.items():
            if hasattr(hook_ctx, key):
                setattr(hook_ctx, key, value)

        tool_name = extra_fields.get("tool_name", "")
        return self.hook_manager.fire(event, hook_ctx, tool_name=tool_name or "")

    def set_hook_manager(self, manager: Any) -> BaseAgent:
        """Attach a ``HookManager`` to this agent.

        Args:
            manager: A ``HookManager`` instance.

        Returns:
            ``self`` for fluent chaining.
        """
        self.hook_manager = manager
        return self

    # ── Visualization ─────────────────────────────────────────────────────

    def draw(self) -> str:
        """Render the agent hierarchy as an ASCII tree.

        Returns:
            Multi-line ASCII string showing the agent tree with icons,
            metadata, tools, and sub-agents.

        Example:
            >>> print(agent.draw())
        """
        from ..visualize import draw_agent
        return draw_agent(self)

    def agent_card(self, *, url: str = "http://localhost:8080", **kwargs: Any) -> Any:
        """Generate an A2A Agent Card from this agent.

        Extracts metadata, tools, skills, and sub-agents into an
        A2A-compliant ``AgentCard`` for discovery.

        Args:
            url: Base URL where the agent's A2A endpoint is hosted.
            **kwargs: Extra arguments forwarded to ``to_agent_card()``.

        Returns:
            An ``AgentCard`` instance.

        Example:
            >>> card = agent.agent_card(url="https://api.example.com")
            >>> print(card.to_json())
        """
        from .agent_card import to_agent_card
        return to_agent_card(self, url=url, **kwargs)

    # ── Abstract contract ─────────────────────────────────────────────────

    @abstractmethod
    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        """Core sync execution logic.  Yields Events produced by the agent.

        Args:
            ctx: The invocation context for the current turn.

        Yields:
            Event objects produced during execution.
        """

    @abstractmethod
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Core async execution logic.  Yields Events produced by the agent.

        Args:
            ctx: The invocation context for the current turn.

        Yields:
            Event objects produced during execution.
        """
        # https://docs.python.org/3/library/abc.html — yield required for type
        yield  # pragma: no cover

    # ── Traced wrappers for sub-agent calls ──────────────────────────────

    def _run_impl_traced(self, ctx: InvocationContext) -> Iterator[Event]:
        """Wrap ``_run_impl`` with automatic trace start/end.

        Orchestration agents should call this instead of ``_run_impl``
        for sub-agents so that each child gets its own trace.
        """
        collector = ctx.trace_collector
        my_trace = None
        if collector is not None:
            provider = getattr(self, "_provider", "")
            model = getattr(self, "_model", "") or ""
            my_trace = collector.start_trace(
                agent_name=self.name,
                agent_type=type(self).__name__,
                input_message=ctx.user_message,
                provider=provider,
                model=model,
            )

        try:
            last_content = ""
            for event in self._run_impl(ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content
                yield event
            if collector is not None:
                collector.end_trace(output=last_content, trace=my_trace)
        except Exception as exc:
            if collector is not None:
                collector.end_trace(error=str(exc), trace=my_trace)
            raise

    async def _run_async_impl_traced(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Async version of ``_run_impl_traced``."""
        collector = ctx.trace_collector
        my_trace = None
        if collector is not None:
            provider = getattr(self, "_provider", "")
            model = getattr(self, "_model", "") or ""
            my_trace = collector.start_trace(
                agent_name=self.name,
                agent_type=type(self).__name__,
                input_message=ctx.user_message,
                provider=provider,
                model=model,
            )

        try:
            last_content = ""
            async for event in self._run_async_impl(ctx):
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content
                yield event
            if collector is not None:
                collector.end_trace(output=last_content, trace=my_trace)
        except Exception as exc:
            if collector is not None:
                collector.end_trace(error=str(exc), trace=my_trace)
            raise

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, ctx: InvocationContext) -> str:
        """Execute the agent and return the final text response.

        Lifecycle:
            1. ``PreAgentRun`` hook (may block execution)
            2. ``before_agent_callback`` (may short-circuit)
            3. ``_run_impl`` (core logic, yields Events)
            4. ``after_agent_callback`` (may transform response)
            5. ``PostAgentRun`` hook (may inject context)

        When a ``trace_collector`` is present on *ctx*, the invocation is
        automatically traced with input, output, status, duration, and errors.

        Args:
            ctx: The invocation context.

        Returns:
            The agent's text response.
        """
        collector = ctx.trace_collector

        # Start trace if collector is active
        my_trace = None
        if collector is not None:
            provider = getattr(self, "_provider", "")
            model = getattr(self, "_model", "") or ""
            my_trace = collector.start_trace(
                agent_name=self.name,
                agent_type=type(self).__name__,
                input_message=ctx.user_message,
                provider=provider,
                model=model,
            )

        try:
            # 0. PreAgentRun hook
            pre_hook_result = self._fire_hook(
                "PreAgentRun", ctx,
                user_message=ctx.user_message,
            )
            if pre_hook_result is not None and pre_hook_result.should_block:
                reason = pre_hook_result.block_reason or "Blocked by PreAgentRun hook"
                event = Event(EventType.AGENT_MESSAGE, self.name, reason)
                ctx.session.add_event(event)
                if collector is not None:
                    collector.end_trace(output=reason, trace=my_trace)
                return reason

            # 1. before callback
            if self.before_agent_callback:
                early = self.before_agent_callback(self, ctx)
                if early is not None:
                    event = Event(EventType.AGENT_MESSAGE, self.name, early)
                    ctx.session.add_event(event)
                    if collector is not None:
                        collector.end_trace(output=early, trace=my_trace)
                    return early

            # 2. core execution
            last_content = ""
            for event in self._run_impl(ctx):
                ctx.session.add_event(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content

            # 3. after callback
            if self.after_agent_callback:
                transformed = self.after_agent_callback(self, ctx, last_content)
                if transformed is not None:
                    last_content = transformed

            # 4. PostAgentRun hook
            post_hook_result = self._fire_hook(
                "PostAgentRun", ctx,
                agent_response=last_content,
            )
            if post_hook_result is not None and post_hook_result.additional_context:
                last_content += f"\n{post_hook_result.additional_context}"

            if collector is not None:
                collector.end_trace(output=last_content, trace=my_trace)

            return last_content

        except Exception as exc:
            # Error hook
            self._fire_hook("Error", ctx, error=str(exc))
            if collector is not None:
                collector.end_trace(error=str(exc), trace=my_trace)
            raise

    async def run_async(self, ctx: InvocationContext) -> str:
        """Async version of ``run()``.

        Same lifecycle as ``run()`` but uses ``_run_async_impl``.
        Includes PreAgentRun / PostAgentRun / Error hooks.

        Args:
            ctx: The invocation context.

        Returns:
            The agent's text response.
        """
        collector = ctx.trace_collector

        my_trace = None
        if collector is not None:
            provider = getattr(self, "_provider", "")
            model = getattr(self, "_model", "") or ""
            my_trace = collector.start_trace(
                agent_name=self.name,
                agent_type=type(self).__name__,
                input_message=ctx.user_message,
                provider=provider,
                model=model,
            )

        try:
            # 0. PreAgentRun hook
            pre_hook_result = self._fire_hook(
                "PreAgentRun", ctx,
                user_message=ctx.user_message,
            )
            if pre_hook_result is not None and pre_hook_result.should_block:
                reason = pre_hook_result.block_reason or "Blocked by PreAgentRun hook"
                event = Event(EventType.AGENT_MESSAGE, self.name, reason)
                ctx.session.add_event(event)
                if collector is not None:
                    collector.end_trace(output=reason, trace=my_trace)
                return reason

            if self.before_agent_callback:
                early = self.before_agent_callback(self, ctx)
                if early is not None:
                    event = Event(EventType.AGENT_MESSAGE, self.name, early)
                    ctx.session.add_event(event)
                    if collector is not None:
                        collector.end_trace(output=early, trace=my_trace)
                    return early

            last_content = ""
            async for event in self._run_async_impl(ctx):
                ctx.session.add_event(event)
                if event.event_type == EventType.AGENT_MESSAGE:
                    last_content = event.content

            if self.after_agent_callback:
                transformed = self.after_agent_callback(self, ctx, last_content)
                if transformed is not None:
                    last_content = transformed

            # PostAgentRun hook
            post_hook_result = self._fire_hook(
                "PostAgentRun", ctx,
                agent_response=last_content,
            )
            if post_hook_result is not None and post_hook_result.additional_context:
                last_content += f"\n{post_hook_result.additional_context}"

            if collector is not None:
                collector.end_trace(output=last_content, trace=my_trace)

            return last_content

        except Exception as exc:
            self._fire_hook("Error", ctx, error=str(exc))
            if collector is not None:
                collector.end_trace(error=str(exc), trace=my_trace)
            raise

    def find_sub_agent(self, name: str) -> BaseAgent | None:
        """Find a sub-agent by name (recursive search).

        Args:
            name: Agent name to search for.

        Returns:
            The matching agent, or None.
        """
        for agent in self.sub_agents:
            if agent.name == name:
                return agent
            found = agent.find_sub_agent(name)
            if found:
                return found
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
