"""
Tracing — structured observability for agent execution.

Records inputs, outputs, token estimates, tools used, LLM provider/model,
status, errors, and timing for every agent invocation.  Traces nest
hierarchically to reflect orchestration topology (Sequential → children).

Usage:
    from nono.agent import Agent, Runner
    from nono.agent.tracing import TraceCollector

    collector = TraceCollector()
    runner = Runner(agent=my_agent)
    runner.run("Hello", trace_collector=collector)

    for trace in collector.traces:
        print(trace.summary())

    # Export as list of dicts:
    data = collector.export()
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("Nono.Agent.Tracing")

_MAX_TRACE_DEPTH: int = 50
"""Maximum nesting depth for recursive trace serialisation.

Prevents ``RecursionError`` when ``to_dict()`` or aggregate properties
traverse deeply nested orchestration trees.
"""

_MAX_TRACE_CHILDREN: int = 200
"""Maximum child traces per parent.

When exceeded, the oldest children are evicted to prevent unbounded
memory growth in long-running parent agents (e.g. LoopAgent).
"""

_MAX_LLM_CALLS: int = 500
"""Maximum LLM call records per trace before oldest entries are evicted."""

_MAX_TOOL_RECORDS: int = 500
"""Maximum tool invocation records per trace before oldest entries are evicted."""


class TraceStatus(Enum):
    """Outcome of a traced agent invocation."""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Estimated token counts for an LLM call.

    Attributes:
        input_tokens: Estimated tokens in the prompt / input messages.
        output_tokens: Estimated tokens in the LLM response.
        total_tokens: Sum of input and output tokens.
    """
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class ToolRecord:
    """Record of a single tool invocation within an agent.

    Attributes:
        tool_name: Name of the tool called.
        arguments: Arguments passed to the tool.
        result: Return value of the tool (stringified).
        duration_ms: Wall-clock time of the tool call in milliseconds.
        error: Error message if the tool raised an exception.
    """
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class LLMCall:
    """Record of a single LLM completion request.

    Attributes:
        provider: Provider name (google, openai, …).
        model: Model name.
        temperature: Temperature used.
        max_tokens: Max tokens setting.
        token_usage: Estimated token counts.
        duration_ms: Wall-clock time of the LLM call in milliseconds.
        error: Error message if the call failed.
    """
    provider: str = ""
    model: str = ""
    temperature: float | str = 0.7
    max_tokens: int | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class Trace:
    """Structured record of a single agent invocation.

    Attributes:
        trace_id: Unique identifier.
        agent_name: Name of the agent.
        agent_type: Class name (LlmAgent, SequentialAgent, …).
        input_message: User message that triggered the invocation.
        output_message: Final response produced by the agent.
        status: Outcome (running, success, error).
        error: Error message if status is ERROR.
        provider: LLM provider used (if applicable).
        model: LLM model used (if applicable).
        llm_calls: List of individual LLM completion requests.
        tools_used: List of tool invocation records.
        token_usage: Aggregated token counts across all LLM calls.
        duration_ms: Total wall-clock time in milliseconds.
        started_at: UTC timestamp when the invocation started.
        ended_at: UTC timestamp when the invocation ended.
        children: Nested traces for sub-agents.
        parent_id: Trace ID of the parent agent (for nesting).
        metadata: Extra key-value data.
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    agent_name: str = ""
    agent_type: str = ""
    input_message: str = ""
    output_message: str = ""
    status: TraceStatus = TraceStatus.RUNNING
    error: str | None = None
    provider: str = ""
    model: str = ""
    llm_calls: list[LLMCall] = field(default_factory=list)
    tools_used: list[ToolRecord] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    duration_ms: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    children: list[Trace] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal timing helper
    _start_ns: int = field(default=0, repr=False)

    def start(self) -> None:
        """Mark the trace as started."""
        self.status = TraceStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self._start_ns = time.perf_counter_ns()

    def finish(self, output: str = "", error: str | None = None) -> None:
        """Mark the trace as finished.

        Args:
            output: Final agent response.
            error: Error message if the invocation failed.
        """
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self.duration_ms = elapsed_ns / 1_000_000
        self.ended_at = datetime.now(timezone.utc)
        self.output_message = output

        if error:
            self.status = TraceStatus.ERROR
            self.error = error
        else:
            self.status = TraceStatus.SUCCESS

        # Aggregate token usage from all LLM calls
        self.token_usage = TokenUsage(
            input_tokens=sum(c.token_usage.input_tokens for c in self.llm_calls),
            output_tokens=sum(c.token_usage.output_tokens for c in self.llm_calls),
        )

    def add_llm_call(self, llm_call: LLMCall) -> None:
        """Record an LLM completion request.

        Oldest entries are evicted when the list exceeds
        ``_MAX_LLM_CALLS``.

        Args:
            llm_call: The LLM call record to add.
        """
        self.llm_calls.append(llm_call)
        if len(self.llm_calls) > _MAX_LLM_CALLS:
            self.llm_calls = self.llm_calls[-_MAX_LLM_CALLS:]

    def add_tool(self, tool_record: ToolRecord) -> None:
        """Record a tool invocation.

        Oldest entries are evicted when the list exceeds
        ``_MAX_TOOL_RECORDS``.

        Args:
            tool_record: The tool record to add.
        """
        self.tools_used.append(tool_record)
        if len(self.tools_used) > _MAX_TOOL_RECORDS:
            self.tools_used = self.tools_used[-_MAX_TOOL_RECORDS:]

    def add_child(self, child: Trace) -> None:
        """Add a nested sub-agent trace.

        Oldest children are evicted when the list exceeds
        ``_MAX_TRACE_CHILDREN`` to prevent unbounded growth.

        Args:
            child: The child trace to add.
        """
        child.parent_id = self.trace_id
        self.children.append(child)
        if len(self.children) > _MAX_TRACE_CHILDREN:
            self.children.pop(0)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        tokens = self.token_usage.total_tokens
        tools = len(self.tools_used)
        children = len(self.children)
        llm_calls = len(self.llm_calls)
        return (
            f"[{self.status.value}] {self.agent_name} ({self.agent_type}) "
            f"— {self.duration_ms:.0f}ms, "
            f"{llm_calls} LLM call(s), {tokens} tokens, "
            f"{tools} tool(s), {children} child(ren)"
        )

    def to_dict(self, _depth: int = 0) -> dict[str, Any]:
        """Export trace as a serialisable dictionary.

        Args:
            _depth: Internal recursion depth counter (do not set manually).
        """
        if _depth >= _MAX_TRACE_DEPTH:
            return {
                "trace_id": self.trace_id,
                "agent_name": self.agent_name,
                "error": f"Trace depth limit ({_MAX_TRACE_DEPTH}) reached",
                "children": [],
            }
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "input_message": self.input_message,
            "output_message": self.output_message,
            "status": self.status.value,
            "error": self.error,
            "provider": self.provider,
            "model": self.model,
            "llm_calls": [
                {
                    "provider": c.provider,
                    "model": c.model,
                    "temperature": c.temperature,
                    "max_tokens": c.max_tokens,
                    "input_tokens": c.token_usage.input_tokens,
                    "output_tokens": c.token_usage.output_tokens,
                    "total_tokens": c.token_usage.total_tokens,
                    "duration_ms": c.duration_ms,
                    "error": c.error,
                }
                for c in self.llm_calls
            ],
            "tools_used": [
                {
                    "tool_name": t.tool_name,
                    "arguments": t.arguments,
                    "result": t.result,
                    "duration_ms": t.duration_ms,
                    "error": t.error,
                }
                for t in self.tools_used
            ],
            "token_usage": {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens,
            },
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "children": [c.to_dict(_depth=_depth + 1) for c in self.children],
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }


class TraceCollector:
    """Collects and manages traces from agent executions.

    Pass an instance to ``Runner.run()`` or attach to a ``Session`` to
    automatically capture traces for every agent invocation.

    When ``max_traces`` is set to a positive integer, the collector
    automatically evicts the oldest top-level traces to stay within the
    limit, preventing unbounded memory growth in long-running sessions.

    Args:
        max_traces: Maximum number of top-level traces to keep.
            ``0`` (default) means unlimited.

    Example:
        >>> collector = TraceCollector()
        >>> runner = Runner(agent=my_agent)
        >>> runner.run("Hello", trace_collector=collector)
        >>> for t in collector.traces:
        ...     print(t.summary())
    """

    def __init__(self, *, max_traces: int = 0) -> None:
        self._traces: list[Trace] = []
        self._stack: list[Trace] = []
        self.max_traces = max_traces
        self._lock = threading.Lock()

    @property
    def traces(self) -> list[Trace]:
        """All top-level traces collected."""
        with self._lock:
            return list(self._traces)

    @property
    def current(self) -> Trace | None:
        """The currently active trace (innermost in the nesting stack)."""
        with self._lock:
            return self._stack[-1] if self._stack else None

    def start_trace(
        self,
        agent_name: str,
        agent_type: str,
        input_message: str,
        provider: str = "",
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Trace:
        """Begin a new trace for an agent invocation.

        Args:
            agent_name: Name of the agent.
            agent_type: Class name of the agent.
            input_message: The user message / input.
            provider: LLM provider (if applicable).
            model: LLM model (if applicable).
            metadata: Extra key-value data.

        Returns:
            The newly created Trace.
        """
        trace = Trace(
            agent_name=agent_name,
            agent_type=agent_type,
            input_message=input_message,
            provider=provider,
            model=model,
            metadata=metadata or {},
        )
        trace.start()

        with self._lock:
            # Nest under parent if there's an active trace
            if self._stack:
                self._stack[-1].add_child(trace)
            else:
                self._traces.append(trace)
                # Evict oldest traces when max_traces is set
                if self.max_traces > 0:
                    while len(self._traces) > self.max_traces:
                        self._traces.pop(0)

            self._stack.append(trace)
        logger.debug("Trace started: %s (%s)", agent_name, trace.trace_id)
        return trace

    def end_trace(
        self,
        output: str = "",
        error: str | None = None,
        *,
        trace: Trace | None = None,
    ) -> Trace | None:
        """End a trace.

        When *trace* is given the method removes that specific trace from
        the stack (safe for parallel / multi-thread usage).  Otherwise it
        pops the most recent entry (legacy behaviour for sequential agents).

        Args:
            output: Agent's final response.
            error: Error message if the agent failed.
            trace: Explicit trace to end (preferred in concurrent code).

        Returns:
            The finished Trace, or None if no trace was active.
        """
        with self._lock:
            if not self._stack:
                logger.warning("end_trace called with no active trace.")
                return None
            if trace is not None:
                try:
                    self._stack.remove(trace)
                except ValueError:
                    logger.warning("end_trace: trace %s not in stack.", trace.trace_id)
                    return None
            else:
                trace = self._stack.pop()
        trace.finish(output=output, error=error)
        logger.debug("Trace ended: %s [%s] %.0fms",
                      trace.agent_name, trace.status.value, trace.duration_ms)
        return trace

    def record_llm_call(self, llm_call: LLMCall) -> None:
        """Record an LLM call in the current trace.

        Args:
            llm_call: The LLM call record.
        """
        with self._lock:
            if self._stack:
                self._stack[-1].add_llm_call(llm_call)

    def record_tool(self, tool_record: ToolRecord) -> None:
        """Record a tool invocation in the current trace.

        Args:
            tool_record: The tool record.
        """
        with self._lock:
            if self._stack:
                self._stack[-1].add_tool(tool_record)

    def export(self) -> list[dict[str, Any]]:
        """Export all traces as serialisable dicts.

        Returns:
            List of trace dictionaries.
        """
        with self._lock:
            snapshot = list(self._traces)
        return [t.to_dict() for t in snapshot]

    def clear(self) -> None:
        """Remove all collected traces."""
        with self._lock:
            self._traces.clear()
            self._stack.clear()

    @staticmethod
    def _iter_all(traces: list[Trace]) -> list[Trace]:
        """Iterate all traces (including children) iteratively."""
        stack = list(traces)
        result: list[Trace] = []
        while stack:
            t = stack.pop()
            result.append(t)
            stack.extend(t.children)
        return result

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens across all traces (including children)."""
        with self._lock:
            snapshot = list(self._traces)
        return sum(t.token_usage.total_tokens for t in self._iter_all(snapshot))

    @property
    def total_llm_calls(self) -> int:
        """Total number of LLM calls across all traces."""
        with self._lock:
            snapshot = list(self._traces)
        return sum(len(t.llm_calls) for t in self._iter_all(snapshot))

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool invocations across all traces."""
        with self._lock:
            snapshot = list(self._traces)
        return sum(len(t.tools_used) for t in self._iter_all(snapshot))

    def print_summary(self) -> None:
        """Print a human-readable summary of all traces."""
        with self._lock:
            snapshot = list(self._traces)
        # Iterative depth-limited print
        stack: list[tuple[Trace, int]] = [
            (t, 0) for t in reversed(snapshot)
        ]
        while stack:
            trace, indent = stack.pop()
            prefix = "  " * indent
            print(f"{prefix}{trace.summary()}")
            if indent < _MAX_TRACE_DEPTH:
                for child in reversed(trace.children):
                    stack.append((child, indent + 1))

        print(
            f"\nTotal: {len(self._traces)} trace(s), "
            f"{self.total_llm_calls} LLM call(s), "
            f"{self.total_tokens} token(s), "
            f"{self.total_tool_calls} tool call(s)"
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._traces)

    def __repr__(self) -> str:
        return f"TraceCollector(traces={len(self._traces)})"
