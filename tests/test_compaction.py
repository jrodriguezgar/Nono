"""Tests for nono.agent.compaction — auto-compaction strategies."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from nono.agent.compaction import (
    CallableStrategy,
    CompactionResult,
    CompactionStrategy,
    SummarizationStrategy,
    TokenAwareStrategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _msgs(n: int, *, with_system: bool = True) -> list[dict[str, str]]:
    """Generate a message list with *n* user/assistant pairs + optional system."""
    result: list[dict[str, str]] = []
    if with_system:
        result.append({"role": "system", "content": "You are helpful."})
    for i in range(n):
        result.append({"role": "user", "content": f"Question {i}"})
        result.append({"role": "assistant", "content": f"Answer {i}"})
    return result


def _long_msg(role: str, chars: int = 2000) -> dict[str, str]:
    return {"role": role, "content": "x" * chars}


# ── CompactionResult ──────────────────────────────────────────────────────


class TestCompactionResult:
    def test_frozen(self) -> None:
        r = CompactionResult(10, 5, "summary", 200)
        with pytest.raises(AttributeError):
            r.original_count = 99  # type: ignore[misc]

    def test_fields(self) -> None:
        r = CompactionResult(
            original_count=20,
            compacted_count=8,
            summary="A summary",
            estimated_tokens_saved=500,
        )
        assert r.original_count == 20
        assert r.compacted_count == 8
        assert r.summary == "A summary"
        assert r.estimated_tokens_saved == 500




# ── CallableStrategy ────────────────────────────────────────────────────


def _keep_recent(messages: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    """Simple custom compaction: keep system + last N."""
    system = [messages[0]] if messages and messages[0].get("role") == "system" else []
    rest = messages[1:] if system else messages
    return system + rest[-(max_messages - len(system)):]


class TestCallableStrategy:
    def test_wraps_function(self) -> None:
        s = CallableStrategy(_keep_recent)
        msgs = _msgs(25)  # 51 messages
        compacted, result = s.compact(msgs, max_messages=10)
        assert len(compacted) == 10
        assert compacted[0]["role"] == "system"
        assert result.original_count == 51
        assert result.compacted_count == 10
        assert result.estimated_tokens_saved > 0

    def test_default_trigger(self) -> None:
        s = CallableStrategy(_keep_recent)
        assert s.should_compact(_msgs(25), max_messages=40)  # 51 > 40
        assert not s.should_compact(_msgs(3), max_messages=40)  # 7 <= 40

    def test_custom_trigger(self) -> None:
        always_trigger = lambda msgs, mx: True
        s = CallableStrategy(_keep_recent, trigger=always_trigger)
        assert s.should_compact(_msgs(1), max_messages=100)

    def test_noop_when_not_triggered(self) -> None:
        s = CallableStrategy(_keep_recent)
        msgs = _msgs(3)  # 7 messages, under 40
        assert not s.should_compact(msgs, max_messages=40)

    def test_tokens_saved_non_negative(self) -> None:
        """Even if the function returns more content, tokens_saved >= 0."""
        def expand(msgs, mx):
            return msgs + [{"role": "user", "content": "extra" * 1000}]
        s = CallableStrategy(expand)
        _, result = s.compact(_msgs(25), max_messages=10)
        assert result.estimated_tokens_saved >= 0

    def test_result_summary_is_empty(self) -> None:
        s = CallableStrategy(_keep_recent)
        _, result = s.compact(_msgs(25), max_messages=10)
        assert result.summary == ""


# ── SummarizationStrategy ────────────────────────────────────────────────


class TestSummarizationStrategy:
    def test_init_defaults(self) -> None:
        s = SummarizationStrategy()
        assert s.trigger_ratio == 0.75
        assert s.keep_recent == 6
        assert s.summary_max_tokens == 500
        assert s.temperature == 0.3
        assert s.service is None

    def test_init_invalid_trigger_ratio(self) -> None:
        with pytest.raises(ValueError, match="trigger_ratio"):
            SummarizationStrategy(trigger_ratio=0)

        with pytest.raises(ValueError, match="trigger_ratio"):
            SummarizationStrategy(trigger_ratio=1.5)

    def test_init_invalid_keep_recent(self) -> None:
        with pytest.raises(ValueError, match="keep_recent"):
            SummarizationStrategy(keep_recent=0)

    def test_service_property(self) -> None:
        s = SummarizationStrategy()
        assert s.service is None
        mock_service = MagicMock()
        s.service = mock_service
        assert s.service is mock_service

    def test_should_compact_below_threshold(self) -> None:
        s = SummarizationStrategy(trigger_ratio=0.75)
        msgs = _msgs(10)  # 21 messages
        assert not s.should_compact(msgs, max_messages=40)  # 21 < 30

    def test_should_compact_above_threshold(self) -> None:
        s = SummarizationStrategy(trigger_ratio=0.75)
        msgs = _msgs(20)  # 41 messages
        assert s.should_compact(msgs, max_messages=40)  # 41 > 30

    def test_compact_with_service(self) -> None:
        mock_svc = MagicMock()
        mock_svc.generate_completion.return_value = "Summary of conversation."
        s = SummarizationStrategy(service=mock_svc, keep_recent=4)
        msgs = _msgs(20)  # 41 messages

        compacted, result = s.compact(msgs, max_messages=40)

        mock_svc.generate_completion.assert_called_once()
        call_kwargs = mock_svc.generate_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_tokens"] == 500

        # Should have: system + summary + 4 recent
        assert compacted[0]["role"] == "system"
        assert "[Context Summary]" in compacted[1]["content"]
        assert "Summary of conversation." in compacted[1]["content"]
        assert len(compacted) == 1 + 1 + 4  # system + summary + keep_recent

        assert result.summary == "Summary of conversation."
        assert result.original_count == 41
        assert result.compacted_count == 6
        assert result.estimated_tokens_saved > 0

    def test_compact_noop_when_below_threshold(self) -> None:
        s = SummarizationStrategy(trigger_ratio=0.75)
        msgs = _msgs(5)  # 11 messages
        compacted, result = s.compact(msgs, max_messages=40)
        assert compacted is msgs
        assert result.summary == ""

    def test_compact_noop_when_rest_fewer_than_keep_recent(self) -> None:
        s = SummarizationStrategy(keep_recent=10, trigger_ratio=0.5)
        # system + 8 = 9 messages, rest=8 < keep_recent=10
        msgs = _msgs(4)  # 9 messages
        # Force should_compact to True by lowering max_messages
        compacted, result = s.compact(msgs, max_messages=10)
        assert compacted is msgs

    def test_compact_fallback_on_no_service(self) -> None:
        s = SummarizationStrategy(keep_recent=2)
        msgs = _msgs(20)  # 41 messages

        compacted, result = s.compact(msgs, max_messages=40)

        assert compacted[0]["role"] == "system"
        assert "[Context Summary]" in compacted[1]["content"]
        # Naive summary has "- [role]" lines
        assert "- [user]" in compacted[1]["content"]
        assert result.summary != ""

    def test_compact_fallback_on_service_error(self) -> None:
        mock_svc = MagicMock()
        mock_svc.generate_completion.side_effect = RuntimeError("API error")
        s = SummarizationStrategy(service=mock_svc, keep_recent=2)
        msgs = _msgs(20)

        compacted, result = s.compact(msgs, max_messages=40)

        # Should still produce a compacted result via naive fallback
        assert "[Context Summary]" in compacted[1]["content"]
        assert result.summary != ""

    def test_format_messages_truncation(self) -> None:
        msgs = [_long_msg("user", 2000)]
        formatted = SummarizationStrategy._format_messages(msgs)
        # Should be capped at ~1000 + "…"
        assert "…" in formatted
        assert len(formatted) < 1200

    def test_naive_summary(self) -> None:
        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        summary = SummarizationStrategy._naive_summary(msgs)
        assert "- [user] Hello world" in summary
        assert "- [assistant] Hi there" in summary

    def test_custom_prompt_template(self) -> None:
        mock_svc = MagicMock()
        mock_svc.generate_completion.return_value = "Custom summary."
        custom = "CUSTOM: {conversation}"
        s = SummarizationStrategy(
            service=mock_svc, keep_recent=2, prompt_template=custom,
        )
        msgs = _msgs(20)

        s.compact(msgs, max_messages=40)

        call_args = mock_svc.generate_completion.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert prompt_content.startswith("CUSTOM: ")

    def test_compact_preserves_recent_order(self) -> None:
        mock_svc = MagicMock()
        mock_svc.generate_completion.return_value = "Summary"
        s = SummarizationStrategy(service=mock_svc, keep_recent=4)
        msgs = _msgs(20)

        compacted, _ = s.compact(msgs, max_messages=40)

        # Last 4 messages should match original last 4
        assert compacted[-4:] == msgs[-4:]


# ── TokenAwareStrategy ────────────────────────────────────────────────────


class TestTokenAwareStrategy:
    def test_init_defaults(self) -> None:
        s = TokenAwareStrategy()
        assert s.max_context_tokens == 100_000

    def test_init_invalid_max_context_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_context_tokens"):
            TokenAwareStrategy(max_context_tokens=0)

    def test_should_compact_below_token_limit(self) -> None:
        s = TokenAwareStrategy(
            max_context_tokens=10_000, trigger_ratio=0.75,
        )
        # Short messages → few tokens
        msgs = _msgs(3)
        assert not s.should_compact(msgs, max_messages=40)

    def test_should_compact_above_token_limit(self) -> None:
        s = TokenAwareStrategy(
            max_context_tokens=100, trigger_ratio=0.5,
        )
        # Long messages → many tokens
        msgs = [_long_msg("user", 500)]
        # 500 chars ≈ 125 tokens > 50 threshold
        assert s.should_compact(msgs, max_messages=40)

    def test_compact_uses_summarization(self) -> None:
        mock_svc = MagicMock()
        mock_svc.generate_completion.return_value = "Token summary."
        s = TokenAwareStrategy(
            service=mock_svc,
            max_context_tokens=100,
            trigger_ratio=0.5,
            keep_recent=2,
        )
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [_long_msg("user", 500) for _ in range(5)]
        )

        compacted, result = s.compact(msgs, max_messages=40)

        mock_svc.generate_completion.assert_called_once()
        assert "[Context Summary]" in compacted[1]["content"]
        assert result.summary == "Token summary."

    def test_inherits_summarization_fallback(self) -> None:
        s = TokenAwareStrategy(
            max_context_tokens=100,
            trigger_ratio=0.5,
            keep_recent=2,
        )
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [_long_msg("user", 500) for _ in range(5)]
        )

        compacted, result = s.compact(msgs, max_messages=40)

        # Naive fallback
        assert "- [user]" in compacted[1]["content"]


# ── LlmAgent integration ─────────────────────────────────────────────────


class TestLlmAgentCompaction:
    """Test that LlmAgent correctly accepts and uses compaction strategies."""

    def test_default_is_none(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        agent = LlmAgent(name="test")
        assert agent._compaction is None

    def test_compaction_true_creates_summarization(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        agent = LlmAgent(name="test", compaction=True)
        assert isinstance(agent._compaction, SummarizationStrategy)

    def test_compaction_false_is_none(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        agent = LlmAgent(name="test", compaction=False)
        assert agent._compaction is None

    def test_compaction_custom_strategy(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        strategy = TokenAwareStrategy(max_context_tokens=50_000)
        agent = LlmAgent(name="test", compaction=strategy)
        assert agent._compaction is strategy

    def test_compaction_callable(self) -> None:
        from nono.agent.llm_agent import LlmAgent

        def my_compact(msgs, mx):
            return msgs[:mx]

        agent = LlmAgent(name="test", compaction=my_compact)
        assert isinstance(agent._compaction, CallableStrategy)

    def test_compact_messages_with_callable(self) -> None:
        from nono.agent.llm_agent import LlmAgent

        def keep_last_five(msgs, mx):
            return [msgs[0]] + msgs[-4:]

        agent = LlmAgent(name="test", compaction=keep_last_five)
        msgs = _msgs(25)
        result = agent._compact_messages(msgs, max_messages=10)
        assert len(result) == 5
        assert result[0]["role"] == "system"

    def test_compaction_string_loads_builtin(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        agent = LlmAgent(
            name="test",
            compaction="nono.agent.compaction.SummarizationStrategy",
        )
        assert isinstance(agent._compaction, SummarizationStrategy)

    def test_compaction_string_loads_token_aware(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        agent = LlmAgent(
            name="test",
            compaction="nono.agent.compaction.TokenAwareStrategy",
        )
        assert isinstance(agent._compaction, TokenAwareStrategy)

    def test_compaction_string_invalid_path(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        with pytest.raises(ImportError):
            LlmAgent(name="test", compaction="NoModule")

    def test_compaction_string_missing_module(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        with pytest.raises(ImportError):
            LlmAgent(name="test", compaction="nonexistent.pkg.Strategy")

    def test_compaction_string_not_a_strategy(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        with pytest.raises(TypeError, match="not a CompactionStrategy"):
            LlmAgent(name="test", compaction="nono.agent.compaction.CompactionResult")

    def test_compact_messages_delegates_to_strategy(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        mock_strategy = MagicMock(spec=CompactionStrategy)
        mock_strategy.should_compact.return_value = True
        mock_strategy.compact.return_value = (
            [{"role": "system", "content": "s"}],
            CompactionResult(10, 1, "summary", 100),
        )

        agent = LlmAgent(name="test")
        agent._compaction = mock_strategy
        msgs = _msgs(20)

        result = agent._compact_messages(msgs)

        mock_strategy.should_compact.assert_called_once()
        mock_strategy.compact.assert_called_once()
        assert len(result) == 1

    def test_compact_messages_noop_when_not_needed(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        mock_strategy = MagicMock(spec=CompactionStrategy)
        mock_strategy.should_compact.return_value = False

        agent = LlmAgent(name="test")
        agent._compaction = mock_strategy
        msgs = _msgs(3)

        result = agent._compact_messages(msgs)

        assert result is msgs
        mock_strategy.compact.assert_not_called()

    def test_compact_messages_injects_service_lazily(self) -> None:
        from nono.agent.llm_agent import LlmAgent
        strategy = SummarizationStrategy(keep_recent=2)
        assert strategy.service is None

        agent = LlmAgent(name="test", compaction=strategy)
        # Mock the service property
        mock_svc = MagicMock()
        mock_svc.generate_completion.return_value = "Summary"
        agent._service = mock_svc

        msgs = _msgs(25)  # 51 messages, triggers compaction
        result = agent._compact_messages(msgs, max_messages=40)

        assert strategy.service is mock_svc
        assert "[Context Summary]" in result[1]["content"]

    def test_prune_messages_still_works(self) -> None:
        """Static _prune_messages is still available for backward compat."""
        from nono.agent.llm_agent import LlmAgent
        msgs = _msgs(25)
        result = LlmAgent._prune_messages(msgs, max_messages=10)
        assert len(result) == 10
        assert result[0]["role"] == "system"
