"""Tests for token-level streaming — generate_completion_stream + TEXT_CHUNK."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from nono.agent.base import Event, EventType


# ── EventType ─────────────────────────────────────────────────────────────


class TestTextChunkEvent:
    def test_text_chunk_event_type_exists(self) -> None:
        assert hasattr(EventType, "TEXT_CHUNK")
        assert EventType.TEXT_CHUNK.value == "text_chunk"

    def test_text_chunk_event_creation(self) -> None:
        e = Event(EventType.TEXT_CHUNK, "agent1", "Hello")
        assert e.event_type == EventType.TEXT_CHUNK
        assert e.author == "agent1"
        assert e.content == "Hello"

    def test_all_event_types_count(self) -> None:
        """TOOL_CALL_CHUNK should be the 11th event type."""
        assert len(EventType) == 11


# ── GenerativeAIService.generate_completion_stream default ────────────────


class TestBaseStreamDefault:
    def test_default_yields_full_response(self) -> None:
        """Base class default should yield the full response as one chunk."""
        from nono.connector.connector_genai import GenerativeAIService

        # Create a concrete subclass with minimal implementation
        class FakeService(GenerativeAIService):
            _PROVIDER_NAME = "fake"

            def __init__(self) -> None:
                pass  # skip real init

            @property
            def model_name(self) -> str:
                return "fake-model"

            @property
            def api_key(self) -> str:
                return "***"

            @property
            def provider(self) -> str:
                return "fake"

            @property
            def rate_limit(self) -> dict:
                return {}

            @property
            def config(self) -> dict:
                return {}

            def generate_completion(self, messages, **kwargs) -> str:
                return "Full response text"

        svc = FakeService()
        chunks = list(svc.generate_completion_stream(
            messages=[{"role": "user", "content": "Hi"}],
        ))
        assert chunks == ["Full response text"]


# ── OpenAICompatibleService.generate_completion_stream ────────────────────


class TestOpenAIStreamParsing:
    """Test SSE line parsing in OpenAICompatibleService streaming."""

    def _make_sse_response(self, deltas: list[str]) -> MagicMock:
        """Build a mock requests.Response that yields SSE lines."""
        lines = []
        for d in deltas:
            chunk = {
                "choices": [{"delta": {"content": d}, "index": 0}],
            }
            lines.append(f"data: {json.dumps(chunk)}")
        lines.append("data: [DONE]")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("nono.connector.connector_genai.HTTP_SESSION")
    def test_streams_deltas(self, mock_session: MagicMock) -> None:
        from nono.connector.connector_genai import OpenAIService, ResponseFormat

        mock_resp = self._make_sse_response(["Hello", " world", "!"])
        mock_session.post.return_value = mock_resp

        svc = OpenAIService.__new__(OpenAIService)
        svc._base_url = "https://api.example.com/v1"
        svc._api_key = "test-key"
        svc.headers = {"Authorization": "Bearer test-key"}
        svc._model_name = "gpt-4o-mini"
        svc._max_input_chars = 1_000_000
        svc.rate_limiter = MagicMock()
        svc.rate_limiter.wait_for_permit = MagicMock()

        chunks = list(svc.generate_completion_stream(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=ResponseFormat.TEXT,
        ))
        assert chunks == ["Hello", " world", "!"]

    @patch("nono.connector.connector_genai.HTTP_SESSION")
    def test_skips_empty_deltas(self, mock_session: MagicMock) -> None:
        from nono.connector.connector_genai import OpenAIService, ResponseFormat

        lines = [
            'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}',
            'data: {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_session.post.return_value = mock_resp

        svc = OpenAIService.__new__(OpenAIService)
        svc._base_url = "https://api.example.com/v1"
        svc._api_key = "test-key"
        svc.headers = {"Authorization": "Bearer test-key"}
        svc._model_name = "gpt-4o-mini"
        svc._max_input_chars = 1_000_000
        svc.rate_limiter = MagicMock()
        svc.rate_limiter.wait_for_permit = MagicMock()

        chunks = list(svc.generate_completion_stream(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=ResponseFormat.TEXT,
        ))
        assert chunks == ["Hi"]


# ── LlmAgent._call_llm_stream ────────────────────────────────────────────


class TestCallLlmStream:
    def test_delegates_to_service(self) -> None:
        from nono.agent.llm_agent import LlmAgent

        agent = LlmAgent(name="test")
        mock_svc = MagicMock()
        mock_svc.generate_completion_stream.return_value = iter(["a", "b"])
        agent._service = mock_svc

        result = list(agent._call_llm_stream(
            messages=[{"role": "user", "content": "Hi"}],
        ))
        assert result == ["a", "b"]
        mock_svc.generate_completion_stream.assert_called_once()


# ── LlmAgent._run_stream_impl ────────────────────────────────────────────


class TestRunStreamImpl:
    def test_yields_text_chunks_no_tools(self) -> None:
        """Without tools, _run_stream_impl yields TEXT_CHUNK + AGENT_MESSAGE."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.connector.connector_genai import StreamChunk

        agent = LlmAgent(name="test")
        mock_svc = MagicMock()
        mock_svc.generate_stream.return_value = iter([
            StreamChunk(type="text", content="Hello"),
            StreamChunk(type="text", content=" "),
            StreamChunk(type="text", content="world"),
            StreamChunk(type="text", content="!"),
            StreamChunk(type="finish", finish_reason="stop"),
        ])
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Say hello")

        events = list(agent._run_stream_impl(ctx))

        # USER_MESSAGE + 4 TEXT_CHUNKs + AGENT_MESSAGE
        types = [e.event_type for e in events]
        assert types[0] == EventType.USER_MESSAGE
        assert types[-1] == EventType.AGENT_MESSAGE

        text_chunks = [e for e in events if e.event_type == EventType.TEXT_CHUNK]
        assert len(text_chunks) == 4
        assert text_chunks[0].content == "Hello"
        assert text_chunks[3].content == "!"

        # AGENT_MESSAGE has full assembled text
        final = events[-1]
        assert final.content == "Hello world!"

    def test_yields_text_chunks_with_tools_no_call(self) -> None:
        """With tools but no tool call, final text is yielded as TEXT_CHUNK."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather")
        def get_weather(city: str) -> str:
            return "Sunny"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()
        mock_svc.generate_stream.return_value = iter([
            StreamChunk(type="text", content="It's sunny in Madrid"),
            StreamChunk(type="finish", finish_reason="stop"),
        ])
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Weather?")

        events = list(agent._run_stream_impl(ctx))

        types = [e.event_type for e in events]
        assert EventType.TEXT_CHUNK in types
        assert types[-1] == EventType.AGENT_MESSAGE
        assert events[-1].content == "It's sunny in Madrid"


# ── Runner.stream_text ────────────────────────────────────────────────────


class TestRunnerStreamText:
    def test_stream_text_yields_events(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.runner import Runner
        from nono.connector.connector_genai import StreamChunk

        agent = LlmAgent(name="test")
        mock_svc = MagicMock()
        mock_svc.generate_stream.return_value = iter([
            StreamChunk(type="text", content="Hi"),
            StreamChunk(type="text", content="!"),
            StreamChunk(type="finish", finish_reason="stop"),
        ])
        agent._service = mock_svc

        runner = Runner(agent=agent)
        events = list(runner.stream_text("Hello"))

        types = [e.event_type for e in events]
        assert EventType.TEXT_CHUNK in types
        assert types[-1] == EventType.AGENT_MESSAGE

    def test_stream_text_early_callback(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.runner import Runner

        agent = LlmAgent(name="test")
        agent.before_agent_callback = lambda a, ctx: "Early response"

        runner = Runner(agent=agent)
        events = list(runner.stream_text("Hello"))

        assert len(events) == 1
        assert events[0].event_type == EventType.AGENT_MESSAGE
        assert events[0].content == "Early response"


# ── StreamChunk ───────────────────────────────────────────────────────────


class TestStreamChunk:
    def test_text_chunk(self) -> None:
        from nono.connector.connector_genai import StreamChunk

        c = StreamChunk(type="text", content="Hello")
        assert c.type == "text"
        assert c.content == "Hello"
        assert c.tool_index == 0
        assert c.tool_name == ""

    def test_tool_call_chunk(self) -> None:
        from nono.connector.connector_genai import StreamChunk

        c = StreamChunk(
            type="tool_call",
            tool_index=0,
            tool_call_id="call_abc",
            tool_name="get_weather",
            content='{"city":',
        )
        assert c.type == "tool_call"
        assert c.tool_call_id == "call_abc"
        assert c.tool_name == "get_weather"
        assert c.content == '{"city":'

    def test_finish_chunk(self) -> None:
        from nono.connector.connector_genai import StreamChunk

        c = StreamChunk(type="finish", finish_reason="tool_calls")
        assert c.type == "finish"
        assert c.finish_reason == "tool_calls"

    def test_frozen(self) -> None:
        from nono.connector.connector_genai import StreamChunk

        c = StreamChunk(type="text", content="Hi")
        with pytest.raises(AttributeError):
            c.content = "Bye"  # type: ignore[misc]


class TestToolCallChunkEvent:
    def test_tool_call_chunk_event_type_exists(self) -> None:
        assert hasattr(EventType, "TOOL_CALL_CHUNK")
        assert EventType.TOOL_CALL_CHUNK.value == "tool_call_chunk"


# ── Base class generate_stream default ────────────────────────────────────


class TestBaseGenerateStream:
    def test_default_yields_text_and_finish(self) -> None:
        """Base class generate_stream yields full response + finish."""
        from nono.connector.connector_genai import GenerativeAIService, StreamChunk

        class FakeService(GenerativeAIService):
            _PROVIDER_NAME = "fake"

            def __init__(self) -> None:
                pass

            @property
            def model_name(self) -> str:
                return "fake-model"

            @property
            def api_key(self) -> str:
                return "***"

            @property
            def provider(self) -> str:
                return "fake"

            @property
            def rate_limit(self) -> dict:
                return {}

            @property
            def config(self) -> dict:
                return {}

            def generate_completion(self, messages, **kwargs) -> str:
                return "Full response text"

        svc = FakeService()
        chunks = list(svc.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
        ))
        assert len(chunks) == 2
        assert chunks[0] == StreamChunk(type="text", content="Full response text")
        assert chunks[1] == StreamChunk(type="finish", finish_reason="stop")


# ── OpenAI generate_stream with tool calls ────────────────────────────────


class TestOpenAIGenerateStream:
    """Test generate_stream SSE parsing with tool_calls."""

    @patch("nono.connector.connector_genai.HTTP_SESSION")
    def test_streams_tool_call_chunks(self, mock_session: MagicMock) -> None:
        from nono.connector.connector_genai import (
            OpenAIService, ResponseFormat, StreamChunk,
        )

        # Simulate SSE with tool call deltas
        lines = [
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\""}}]}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": \\"NYC\\"}"}}]}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_session.post.return_value = mock_resp

        svc = OpenAIService.__new__(OpenAIService)
        svc._base_url = "https://api.example.com/v1"
        svc._api_key = "test-key"
        svc.headers = {"Authorization": "Bearer test-key"}
        svc._model_name = "gpt-4o-mini"
        svc._max_input_chars = 1_000_000
        svc.rate_limiter = MagicMock()
        svc.rate_limiter.wait_for_permit = MagicMock()

        chunks = list(svc.generate_stream(
            messages=[{"role": "user", "content": "Weather?"}],
            response_format=ResponseFormat.TEXT,
        ))

        # Should have 3 tool_call chunks + 1 finish
        tool_chunks = [c for c in chunks if c.type == "tool_call"]
        assert len(tool_chunks) == 3
        assert tool_chunks[0].tool_name == "get_weather"
        assert tool_chunks[0].tool_call_id == "call_1"
        assert tool_chunks[0].tool_index == 0

        # Arguments assembled
        args_text = "".join(c.content for c in tool_chunks)
        assert args_text == '{"city": "NYC"}'

        # Finish chunk
        finish = [c for c in chunks if c.type == "finish"]
        assert len(finish) == 1
        assert finish[0].finish_reason == "tool_calls"

    @patch("nono.connector.connector_genai.HTTP_SESSION")
    def test_streams_text_and_finish(self, mock_session: MagicMock) -> None:
        from nono.connector.connector_genai import (
            OpenAIService, ResponseFormat, StreamChunk,
        )

        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
            "data: [DONE]",
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_session.post.return_value = mock_resp

        svc = OpenAIService.__new__(OpenAIService)
        svc._base_url = "https://api.example.com/v1"
        svc._api_key = "test-key"
        svc.headers = {"Authorization": "Bearer test-key"}
        svc._model_name = "gpt-4o-mini"
        svc._max_input_chars = 1_000_000
        svc.rate_limiter = MagicMock()
        svc.rate_limiter.wait_for_permit = MagicMock()

        chunks = list(svc.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=ResponseFormat.TEXT,
        ))

        text_chunks = [c for c in chunks if c.type == "text"]
        assert [c.content for c in text_chunks] == ["Hello", " world"]

        finish = [c for c in chunks if c.type == "finish"]
        assert finish[0].finish_reason == "stop"


# ── Streaming tool calls in _run_stream_impl ──────────────────────────────


class TestStreamingToolCalls:
    """Test that _run_stream_impl handles streaming tool call chunks."""

    def test_native_streaming_tool_call(self) -> None:
        """Native tool_call chunks are accumulated and executed."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather for a city")
        def get_weather(city: str) -> str:
            return "Sunny, 22°C"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()

        # First call: tool call chunks
        # Second call: text response
        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        tool_call_id="call_1", tool_name="get_weather",
                        content='{"city"',
                    ),
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        content=': "Madrid"}',
                    ),
                    StreamChunk(type="finish", finish_reason="tool_calls"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="It's sunny!"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Weather?")

        events = list(agent._run_stream_impl(ctx))
        types = [e.event_type for e in events]

        # Should see: USER_MSG, TOOL_CALL_CHUNKs, TOOL_CALL, TOOL_RESULT,
        #             TEXT_CHUNK, AGENT_MESSAGE
        assert EventType.TOOL_CALL_CHUNK in types
        assert EventType.TOOL_CALL in types
        assert EventType.TOOL_RESULT in types
        assert EventType.TEXT_CHUNK in types
        assert types[-1] == EventType.AGENT_MESSAGE

        # Verify tool was called with correct args
        tool_call_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].data["tool"] == "get_weather"
        assert tool_call_events[0].data["arguments"] == {"city": "Madrid"}

        # Verify tool result
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        assert "Sunny, 22°C" in tool_results[0].content

    def test_tool_call_chunk_events_carry_metadata(self) -> None:
        """Each TOOL_CALL_CHUNK event has tool_name and arguments_delta."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        agent = LlmAgent(name="test", tools=[add])
        mock_svc = MagicMock()

        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        tool_call_id="c1", tool_name="add",
                        content='{"a": 1',
                    ),
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        content=', "b": 2}',
                    ),
                    StreamChunk(type="finish", finish_reason="tool_calls"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="3"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="1+2?")

        events = list(agent._run_stream_impl(ctx))
        tc_chunks = [
            e for e in events if e.event_type == EventType.TOOL_CALL_CHUNK
        ]
        assert len(tc_chunks) == 2
        assert tc_chunks[0].data["tool_name"] == "add"
        assert tc_chunks[0].data["arguments_delta"] == '{"a": 1'
        assert tc_chunks[1].data["arguments_delta"] == ', "b": 2}'

    def test_multiple_parallel_tool_calls(self) -> None:
        """Multiple tool calls with different indices are handled."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather")
        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()

        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        tool_call_id="c1", tool_name="get_weather",
                        content='{"city": "NYC"}',
                    ),
                    StreamChunk(
                        type="tool_call", tool_index=1,
                        tool_call_id="c2", tool_name="get_weather",
                        content='{"city": "LA"}',
                    ),
                    StreamChunk(type="finish", finish_reason="tool_calls"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="NYC: Sunny, LA: Sunny"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Weather?")

        events = list(agent._run_stream_impl(ctx))

        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_calls) == 2
        assert tool_calls[0].data["arguments"]["city"] == "NYC"
        assert tool_calls[1].data["arguments"]["city"] == "LA"

        tool_results = [
            e for e in events if e.event_type == EventType.TOOL_RESULT
        ]
        assert len(tool_results) == 2

    def test_fallback_text_based_tool_detection(self) -> None:
        """When generate_stream returns only text, tool calls are detected
        from assembled text via parse_tool_calls (fallback path)."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather")
        def get_weather(city: str) -> str:
            return "Sunny"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()

        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Base-class fallback: tool call encoded as text
                tc_json = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
                return iter([
                    StreamChunk(type="text", content=tc_json),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="Sunny in NYC"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Weather?")

        events = list(agent._run_stream_impl(ctx))
        types = [e.event_type for e in events]

        # Fallback path should detect tool call from text and execute it
        assert EventType.TOOL_CALL in types
        assert EventType.TOOL_RESULT in types
        assert types[-1] == EventType.AGENT_MESSAGE

    def test_unknown_tool_in_stream_skipped(self) -> None:
        """Native tool call for an unknown tool adds error to messages."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather")
        def get_weather(city: str) -> str:
            return "Sunny"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()

        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        tool_call_id="c1", tool_name="unknown_tool",
                        content='{"x": 1}',
                    ),
                    StreamChunk(type="finish", finish_reason="tool_calls"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="Sorry, tool not found"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Use tool")

        events = list(agent._run_stream_impl(ctx))
        types = [e.event_type for e in events]

        # Unknown tool should NOT produce TOOL_CALL event
        assert EventType.TOOL_CALL not in types
        # Should eventually get a text response
        assert types[-1] == EventType.AGENT_MESSAGE

    def test_malformed_tool_arguments_handled(self) -> None:
        """Malformed JSON arguments default to empty dict."""
        from nono.agent.llm_agent import LlmAgent
        from nono.agent.base import Session, InvocationContext
        from nono.agent.tool import tool
        from nono.connector.connector_genai import StreamChunk

        @tool(description="Get weather")
        def get_weather(city: str = "unknown") -> str:
            return "Sunny"

        agent = LlmAgent(name="test", tools=[get_weather])
        mock_svc = MagicMock()

        call_count = 0

        def fake_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([
                    StreamChunk(
                        type="tool_call", tool_index=0,
                        tool_call_id="c1", tool_name="get_weather",
                        content="not valid json{{{",
                    ),
                    StreamChunk(type="finish", finish_reason="tool_calls"),
                ])
            else:
                return iter([
                    StreamChunk(type="text", content="Done"),
                    StreamChunk(type="finish", finish_reason="stop"),
                ])

        mock_svc.generate_stream.side_effect = fake_stream
        agent._service = mock_svc

        session = Session()
        ctx = InvocationContext(session=session, user_message="Do it")

        events = list(agent._run_stream_impl(ctx))

        # Tool should still be called (with empty args)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_calls) == 1
        assert tool_calls[0].data["arguments"] == {}
