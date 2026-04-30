"""Tests for AnthropicService — native Anthropic Claude connector."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Iterator


# ── Helpers ───────────────────────────────────────────────────────────────


@dataclass
class _FakeTextBlock:
    type: str = "text"
    text: str = "Hello from Claude"


@dataclass
class _FakeResponse:
    content: list = None

    def __post_init__(self):
        if self.content is None:
            self.content = [_FakeTextBlock()]


class _FakeStreamContext:
    """Mock for ``client.messages.stream()`` context manager."""

    def __init__(self, texts: list[str] | None = None, events: list | None = None):
        self._texts = texts or ["Hello", " from", " Claude"]
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def text_stream(self) -> Iterator[str]:
        yield from self._texts

    def __iter__(self):
        if self._events:
            yield from self._events
        else:
            for t in self._texts:
                yield _make_text_delta_event(t)


def _make_text_delta_event(text: str):
    """Create a mock Anthropic text_delta streaming event."""
    ev = MagicMock()
    ev.type = "content_block_delta"
    ev.delta = MagicMock()
    ev.delta.type = "text_delta"
    ev.delta.text = text
    return ev


# ── System/Message Splitting ─────────────────────────────────────────────


class TestSplitSystemAndMessages:
    """Tests for AnthropicService._split_system_and_messages static method."""

    def _get_cls(self):
        from nono.connector.connector_genai import AnthropicService
        return AnthropicService

    def test_extracts_single_system_message(self) -> None:
        cls = self._get_cls()
        messages = [
            {"role": "system", "content": "You are a coder."},
            {"role": "user", "content": "Hi"},
        ]
        system, filtered = cls._split_system_and_messages(messages)
        assert system == "You are a coder."
        assert len(filtered) == 1
        assert filtered[0]["role"] == "user"

    def test_merges_multiple_system_messages(self) -> None:
        cls = self._get_cls()
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hello"},
        ]
        system, filtered = cls._split_system_and_messages(messages)
        assert "Rule 1" in system
        assert "Rule 2" in system
        assert len(filtered) == 1

    def test_no_system_message_returns_none(self) -> None:
        cls = self._get_cls()
        messages = [{"role": "user", "content": "Hi"}]
        system, filtered = cls._split_system_and_messages(messages)
        assert system is None
        assert len(filtered) == 1

    def test_maps_model_role_to_assistant(self) -> None:
        cls = self._get_cls()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "model", "content": "Hello!"},
        ]
        _, filtered = cls._split_system_and_messages(messages)
        assert filtered[1]["role"] == "assistant"


# ── generate_completion ───────────────────────────────────────────────────


class TestAnthropicGenerateCompletion:

    @patch("nono.connector.connector_genai.install_library", return_value=True)
    @patch("nono.connector.connector_genai.resolve_api_key_for_provider", return_value="sk-ant-test")
    @patch("nono.connector.connector_genai.get_rate_limit_config", return_value=None)
    @patch("nono.connector.connector_genai.get_prompt_size", return_value=200000)
    def test_generate_completion_basic(
        self, mock_ps, mock_rl, mock_key, mock_install
    ) -> None:
        from nono.connector.connector_genai import AnthropicService, ResponseFormat

        service = AnthropicService(model_name="claude-sonnet-4")

        # Mock the SDK client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse()
        service.client = mock_client

        result = service.generate_completion(
            messages=[{"role": "user", "content": "Hello"}],
            response_format=ResponseFormat.TEXT,
        )

        assert result == "Hello from Claude"
        mock_client.messages.create.assert_called_once()

        # Verify system prompt is passed as top-level param, not in messages
        call_kwargs = mock_client.messages.create.call_args
        assert "system" in call_kwargs.kwargs
        for msg in call_kwargs.kwargs["messages"]:
            assert msg["role"] != "system"

    @patch("nono.connector.connector_genai.install_library", return_value=True)
    @patch("nono.connector.connector_genai.resolve_api_key_for_provider", return_value="sk-ant-test")
    @patch("nono.connector.connector_genai.get_rate_limit_config", return_value=None)
    @patch("nono.connector.connector_genai.get_prompt_size", return_value=200000)
    def test_generate_completion_strips_json_fences(
        self, mock_ps, mock_rl, mock_key, mock_install
    ) -> None:
        from nono.connector.connector_genai import AnthropicService, ResponseFormat

        service = AnthropicService(model_name="claude-sonnet-4")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse(
            content=[_FakeTextBlock(text='```json\n{"key": "value"}\n```')]
        )
        service.client = mock_client

        result = service.generate_completion(
            messages=[{"role": "user", "content": "Give JSON"}],
            response_format=ResponseFormat.JSON,
        )
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    @patch("nono.connector.connector_genai.install_library", return_value=True)
    @patch("nono.connector.connector_genai.resolve_api_key_for_provider", return_value="sk-ant-test")
    @patch("nono.connector.connector_genai.get_rate_limit_config", return_value=None)
    @patch("nono.connector.connector_genai.get_prompt_size", return_value=200000)
    def test_generate_completion_default_max_tokens(
        self, mock_ps, mock_rl, mock_key, mock_install
    ) -> None:
        from nono.connector.connector_genai import AnthropicService, ResponseFormat

        service = AnthropicService(model_name="claude-sonnet-4")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _FakeResponse()
        service.client = mock_client

        service.generate_completion(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=ResponseFormat.TEXT,
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096


# ── generate_completion_stream ────────────────────────────────────────────


class TestAnthropicStreaming:

    @patch("nono.connector.connector_genai.install_library", return_value=True)
    @patch("nono.connector.connector_genai.resolve_api_key_for_provider", return_value="sk-ant-test")
    @patch("nono.connector.connector_genai.get_rate_limit_config", return_value=None)
    @patch("nono.connector.connector_genai.get_prompt_size", return_value=200000)
    def test_stream_yields_text_chunks(
        self, mock_ps, mock_rl, mock_key, mock_install
    ) -> None:
        from nono.connector.connector_genai import AnthropicService, ResponseFormat

        service = AnthropicService(model_name="claude-sonnet-4")
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _FakeStreamContext(
            texts=["Hello", " world"]
        )
        service.client = mock_client

        chunks = list(service.generate_completion_stream(
            messages=[{"role": "user", "content": "Hi"}],
            response_format=ResponseFormat.TEXT,
        ))

        assert chunks == ["Hello", " world"]


# ── Provider Registration ─────────────────────────────────────────────────


class TestAnthropicProviderRegistration:

    def test_provider_in_provider_map(self) -> None:
        """AnthropicService should be registered in _PROVIDER_MAP."""
        from nono.agent.llm_agent import _PROVIDER_MAP

        assert "anthropic" in _PROVIDER_MAP
        assert "claude" in _PROVIDER_MAP
        assert _PROVIDER_MAP["anthropic"][0] == "AnthropicService"
        assert _PROVIDER_MAP["claude"][0] == "AnthropicService"

    @patch("nono.connector.connector_genai.resolve_api_key_for_provider", return_value="sk-ant-test")
    @patch("nono.connector.connector_genai.get_rate_limit_config", return_value=None)
    @patch("nono.connector.connector_genai.get_prompt_size", return_value=200000)
    def test_factory_returns_anthropic_service(
        self, mock_ps, mock_rl, mock_key
    ) -> None:
        from nono.connector.connector_genai import get_service_for_provider, AnthropicService

        service = get_service_for_provider(
            "anthropic", "claude-sonnet-4", "sk-ant-test"
        )
        assert isinstance(service, AnthropicService)

    def test_provider_name_attribute(self) -> None:
        from nono.connector.connector_genai import AnthropicService
        assert AnthropicService._PROVIDER_NAME == "anthropic"


# ── Model Configuration ──────────────────────────────────────────────────


class TestAnthropicModelConfig:

    def test_model_features_csv_has_anthropic(self) -> None:
        from nono.connector.connector_genai import get_prompt_size
        size = get_prompt_size("anthropic", "claude-sonnet-4")
        assert size == 200000

    def test_rate_limits_csv_has_anthropic(self) -> None:
        from nono.connector.connector_genai import get_rate_limit_config
        config = get_rate_limit_config("anthropic", "claude-sonnet-4")
        assert config is not None
        assert config.rpm == 50
