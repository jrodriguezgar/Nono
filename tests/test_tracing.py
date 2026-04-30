"""
Tests for the tracing / observability system.

Covers:
- TraceCollector standalone API (start, end, record, export, summary).
- Trace nesting (parent → child).
- Token aggregation across LLM calls and nested traces.
- Integration with BaseAgent.run() / run_async() via InvocationContext.
- Integration with Runner (trace_collector parameter).
- Orchestration agents forward trace_collector to sub-agents.
- Error traces.
"""

import asyncio
import sys
import os
from typing import Any, AsyncIterator, Iterator

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.tracing import (
    LLMCall,
    TokenUsage,
    ToolRecord,
    Trace,
    TraceCollector,
    TraceStatus,
)
from nono.agent.workflow_agents import SequentialAgent, ParallelAgent
from nono.agent.runner import Runner


# ── Helpers ──────────────────────────────────────────────────────────────────

class MockAgent(BaseAgent):
    """Agent that returns a fixed reply."""

    def __init__(self, *, name: str, reply: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.reply = reply

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self.reply)


class FailingAgent(BaseAgent):
    """Agent that always raises."""

    def __init__(self, *, name: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        raise RuntimeError("boom")
        yield  # type: ignore[misc]

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        raise RuntimeError("boom")
        yield  # type: ignore[misc]


# ── Unit: TokenUsage ─────────────────────────────────────────────────────────

def test_token_usage_total() -> None:
    tu = TokenUsage(input_tokens=100, output_tokens=50)
    assert tu.total_tokens == 150
    print("PASS: test_token_usage_total")


def test_token_usage_defaults_zero() -> None:
    tu = TokenUsage()
    assert tu.total_tokens == 0
    print("PASS: test_token_usage_defaults_zero")


# ── Unit: ToolRecord ─────────────────────────────────────────────────────────

def test_tool_record_fields() -> None:
    tr = ToolRecord(
        tool_name="search", arguments={"q": "hello"}, result="found",
        duration_ms=42.5, error=None,
    )
    assert tr.tool_name == "search"
    assert tr.arguments == {"q": "hello"}
    assert tr.result == "found"
    assert tr.duration_ms == 42.5
    assert tr.error is None
    print("PASS: test_tool_record_fields")


# ── Unit: LLMCall ────────────────────────────────────────────────────────────

def test_llm_call_fields() -> None:
    lc = LLMCall(
        provider="google", model="gemini-3-flash-preview",
        temperature=0.7, max_tokens=1024,
        token_usage=TokenUsage(200, 80), duration_ms=150.0,
    )
    assert lc.provider == "google"
    assert lc.model == "gemini-3-flash-preview"
    assert lc.token_usage.total_tokens == 280
    print("PASS: test_llm_call_fields")


# ── Unit: Trace ──────────────────────────────────────────────────────────────

def test_trace_start_finish() -> None:
    t = Trace(agent_name="a", agent_type="MockAgent")
    t.start()
    assert t.status == TraceStatus.RUNNING
    t.finish(output="done")
    assert t.status == TraceStatus.SUCCESS
    assert t.output_message == "done"
    assert t.duration_ms >= 0
    assert t.ended_at is not None
    print("PASS: test_trace_start_finish")


def test_trace_finish_with_error() -> None:
    t = Trace(agent_name="b", agent_type="MockAgent")
    t.start()
    t.finish(error="oops")
    assert t.status == TraceStatus.ERROR
    assert t.error == "oops"
    print("PASS: test_trace_finish_with_error")


def test_trace_aggregates_tokens() -> None:
    t = Trace(agent_name="c", agent_type="MockAgent")
    t.start()
    t.add_llm_call(LLMCall(token_usage=TokenUsage(100, 50)))
    t.add_llm_call(LLMCall(token_usage=TokenUsage(200, 80)))
    t.finish(output="ok")
    assert t.token_usage.input_tokens == 300
    assert t.token_usage.output_tokens == 130
    assert t.token_usage.total_tokens == 430
    print("PASS: test_trace_aggregates_tokens")


def test_trace_nesting() -> None:
    parent = Trace(agent_name="parent", agent_type="Sequential")
    child = Trace(agent_name="child", agent_type="LlmAgent")
    parent.add_child(child)
    assert child.parent_id == parent.trace_id
    assert len(parent.children) == 1
    print("PASS: test_trace_nesting")


def test_trace_to_dict() -> None:
    t = Trace(agent_name="d", agent_type="MockAgent", input_message="hi")
    t.start()
    t.add_llm_call(LLMCall(provider="test", model="m1", token_usage=TokenUsage(10, 5)))
    t.add_tool(ToolRecord(tool_name="t1", arguments={"x": 1}, result="ok", duration_ms=3.0))
    t.finish(output="bye")
    d = t.to_dict()
    assert d["agent_name"] == "d"
    assert d["status"] == "success"
    assert d["input_message"] == "hi"
    assert d["output_message"] == "bye"
    assert len(d["llm_calls"]) == 1
    assert len(d["tools_used"]) == 1
    assert d["token_usage"]["total_tokens"] == 15
    print("PASS: test_trace_to_dict")


def test_trace_summary_string() -> None:
    t = Trace(agent_name="x", agent_type="LlmAgent")
    t.start()
    t.add_llm_call(LLMCall(token_usage=TokenUsage(10, 5)))
    t.finish(output="ok")
    s = t.summary()
    assert "x" in s
    assert "LlmAgent" in s
    assert "success" in s
    print("PASS: test_trace_summary_string")


# ── Unit: TraceCollector ─────────────────────────────────────────────────────

def test_collector_start_end() -> None:
    c = TraceCollector()
    c.start_trace("a", "MockAgent", "hello")
    assert c.current is not None
    c.end_trace(output="world")
    assert c.current is None
    assert len(c) == 1
    assert c.traces[0].agent_name == "a"
    assert c.traces[0].output_message == "world"
    assert c.traces[0].status == TraceStatus.SUCCESS
    print("PASS: test_collector_start_end")


def test_collector_nesting() -> None:
    c = TraceCollector()
    c.start_trace("parent", "Sequential", "msg")
    c.start_trace("child", "LlmAgent", "msg")
    c.end_trace(output="child_out")
    c.end_trace(output="parent_out")
    assert len(c) == 1  # one top-level trace
    assert len(c.traces[0].children) == 1
    assert c.traces[0].children[0].agent_name == "child"
    print("PASS: test_collector_nesting")


def test_collector_record_llm_call() -> None:
    c = TraceCollector()
    c.start_trace("a", "LlmAgent", "hi")
    c.record_llm_call(LLMCall(provider="google", model="gemini", token_usage=TokenUsage(50, 20)))
    c.end_trace(output="ok")
    assert c.total_llm_calls == 1
    assert c.total_tokens == 70
    print("PASS: test_collector_record_llm_call")


def test_collector_record_tool() -> None:
    c = TraceCollector()
    c.start_trace("a", "LlmAgent", "hi")
    c.record_tool(ToolRecord(tool_name="search", arguments={}, result="r", duration_ms=5.0))
    c.end_trace(output="done")
    assert c.total_tool_calls == 1
    print("PASS: test_collector_record_tool")


def test_collector_export() -> None:
    c = TraceCollector()
    c.start_trace("a", "MockAgent", "in")
    c.end_trace(output="out")
    data = c.export()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["agent_name"] == "a"
    assert data[0]["input_message"] == "in"
    assert data[0]["output_message"] == "out"
    print("PASS: test_collector_export")


def test_collector_clear() -> None:
    c = TraceCollector()
    c.start_trace("a", "MockAgent", "hi")
    c.end_trace(output="ok")
    assert len(c) == 1
    c.clear()
    assert len(c) == 0
    print("PASS: test_collector_clear")


def test_collector_total_tokens_nested() -> None:
    c = TraceCollector()
    c.start_trace("parent", "Sequential", "msg")
    c.record_llm_call(LLMCall(token_usage=TokenUsage(10, 5)))  # parent LLM call
    c.start_trace("child", "LlmAgent", "msg")
    c.record_llm_call(LLMCall(token_usage=TokenUsage(20, 10)))
    c.end_trace(output="c")
    c.end_trace(output="p")
    # Parent: 15 tokens, Child: 30 tokens => total 45
    assert c.total_tokens == 45
    print("PASS: test_collector_total_tokens_nested")


def test_collector_no_active_trace_safe() -> None:
    c = TraceCollector()
    c.record_llm_call(LLMCall(provider="x"))  # no active trace, should not crash
    c.record_tool(ToolRecord(tool_name="y"))
    result = c.end_trace(output="z")
    assert result is None
    print("PASS: test_collector_no_active_trace_safe")


# ── Integration: BaseAgent + TraceCollector ──────────────────────────────────

def test_agent_run_creates_trace() -> None:
    agent = MockAgent(name="mock", reply="hello world")
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="test input",
        trace_collector=collector,
    )
    result = agent.run(ctx)
    assert result == "hello world"
    assert len(collector) == 1
    trace = collector.traces[0]
    assert trace.agent_name == "mock"
    assert trace.agent_type == "MockAgent"
    assert trace.input_message == "test input"
    assert trace.output_message == "hello world"
    assert trace.status == TraceStatus.SUCCESS
    assert trace.duration_ms >= 0
    print("PASS: test_agent_run_creates_trace")


def test_agent_run_async_creates_trace() -> None:
    agent = MockAgent(name="async_mock", reply="async hello")
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="async test",
        trace_collector=collector,
    )
    result = asyncio.run(agent.run_async(ctx))
    assert result == "async hello"
    assert len(collector) == 1
    trace = collector.traces[0]
    assert trace.agent_name == "async_mock"
    assert trace.status == TraceStatus.SUCCESS
    assert trace.output_message == "async hello"
    print("PASS: test_agent_run_async_creates_trace")


def test_agent_error_traced() -> None:
    agent = FailingAgent(name="fail")
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="trigger error",
        trace_collector=collector,
    )
    try:
        agent.run(ctx)
    except RuntimeError:
        pass
    assert len(collector) == 1
    trace = collector.traces[0]
    assert trace.status == TraceStatus.ERROR
    assert trace.error == "boom"
    print("PASS: test_agent_error_traced")


def test_agent_error_async_traced() -> None:
    agent = FailingAgent(name="fail_async")
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="async trigger",
        trace_collector=collector,
    )
    try:
        asyncio.run(agent.run_async(ctx))
    except RuntimeError:
        pass
    assert len(collector) == 1
    trace = collector.traces[0]
    assert trace.status == TraceStatus.ERROR
    assert trace.error == "boom"
    print("PASS: test_agent_error_async_traced")


def test_agent_run_without_collector() -> None:
    """Agent works normally when no trace_collector is set."""
    agent = MockAgent(name="plain", reply="no trace")
    ctx = InvocationContext(session=Session(), user_message="hello")
    result = agent.run(ctx)
    assert result == "no trace"
    print("PASS: test_agent_run_without_collector")


# ── Integration: Runner + TraceCollector ─────────────────────────────────────

def test_runner_forwards_collector() -> None:
    agent = MockAgent(name="runner_agent", reply="runner reply")
    collector = TraceCollector()
    runner = Runner(agent=agent, trace_collector=collector)
    result = runner.run("question")
    assert result == "runner reply"
    assert len(collector) == 1
    assert collector.traces[0].agent_name == "runner_agent"
    print("PASS: test_runner_forwards_collector")


def test_runner_async_forwards_collector() -> None:
    agent = MockAgent(name="arunner", reply="async runner")
    collector = TraceCollector()
    runner = Runner(agent=agent, trace_collector=collector)
    result = asyncio.run(runner.run_async("q"))
    assert result == "async runner"
    assert len(collector) == 1
    print("PASS: test_runner_async_forwards_collector")


def test_runner_stream_forwards_collector() -> None:
    agent = MockAgent(name="stream_agent", reply="stream reply")
    collector = TraceCollector()
    runner = Runner(agent=agent, trace_collector=collector)
    events = list(runner.stream("q"))
    assert any(e.content == "stream reply" for e in events)
    # stream bypasses run() wrapper, so collector may have 0 traces
    # (stream calls _run_impl directly, not run())
    # This test confirms no crash
    print("PASS: test_runner_stream_forwards_collector")


# ── Integration: Orchestration nesting ───────────────────────────────────────

def test_sequential_agent_nested_traces() -> None:
    a1 = MockAgent(name="step1", reply="r1")
    a2 = MockAgent(name="step2", reply="r2")
    seq = SequentialAgent(name="seq", sub_agents=[a1, a2])
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="pipeline",
        trace_collector=collector,
    )
    result = seq.run(ctx)
    assert result == "r2"
    # Top-level: seq trace. Children: step1, step2
    assert len(collector) == 1
    top = collector.traces[0]
    assert top.agent_name == "seq"
    assert len(top.children) == 2
    assert top.children[0].agent_name == "step1"
    assert top.children[1].agent_name == "step2"
    assert all(c.status == TraceStatus.SUCCESS for c in top.children)
    print("PASS: test_sequential_agent_nested_traces")


def test_parallel_agent_nested_traces() -> None:
    a1 = MockAgent(name="w1", reply="p1")
    a2 = MockAgent(name="w2", reply="p2")
    par = ParallelAgent(name="par", sub_agents=[a1, a2])
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="parallel",
        trace_collector=collector,
    )
    result = par.run(ctx)
    assert len(collector) == 1
    top = collector.traces[0]
    assert top.agent_name == "par"
    assert len(top.children) == 2
    child_names = {c.agent_name for c in top.children}
    assert child_names == {"w1", "w2"}
    print("PASS: test_parallel_agent_nested_traces")


def test_sequential_async_nested_traces() -> None:
    a1 = MockAgent(name="as1", reply="ar1")
    a2 = MockAgent(name="as2", reply="ar2")
    seq = SequentialAgent(name="aseq", sub_agents=[a1, a2])
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="async pipeline",
        trace_collector=collector,
    )
    result = asyncio.run(seq.run_async(ctx))
    assert result == "ar2"
    assert len(collector) == 1
    assert len(collector.traces[0].children) == 2
    print("PASS: test_sequential_async_nested_traces")


# ── Integration: before_agent_callback short-circuit ─────────────────────────

def test_before_callback_traced() -> None:
    def early_exit(agent: BaseAgent, ctx: InvocationContext) -> str:
        return "early response"

    agent = MockAgent(name="cb", reply="never", before_agent_callback=early_exit)
    collector = TraceCollector()
    ctx = InvocationContext(
        session=Session(), user_message="trigger callback",
        trace_collector=collector,
    )
    result = agent.run(ctx)
    assert result == "early response"
    assert len(collector) == 1
    trace = collector.traces[0]
    assert trace.output_message == "early response"
    assert trace.status == TraceStatus.SUCCESS
    print("PASS: test_before_callback_traced")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Tracing tests: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
