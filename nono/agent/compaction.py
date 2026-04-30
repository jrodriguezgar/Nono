"""Auto-compaction for agent context window management.

Provides LLM-based summarization to compress old messages when the
context window approaches its limit, preserving key information
instead of silently dropping middle messages.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ── Result ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
        original_count: Number of messages before compaction.
        compacted_count: Number of messages after compaction.
        summary: The generated summary text (empty for sliding-window).
        estimated_tokens_saved: Rough token savings estimate.
    """

    original_count: int
    compacted_count: int
    summary: str
    estimated_tokens_saved: int


# ── Abstract strategy ─────────────────────────────────────────────────────


class CompactionStrategy(ABC):
    """Base class for message compaction strategies."""

    @abstractmethod
    def should_compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> bool:
        """Decide whether compaction should run.

        Args:
            messages: Current message list.
            max_messages: Hard message-count ceiling.

        Returns:
            ``True`` if the strategy wants to compact now.
        """

    @abstractmethod
    def compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> tuple[list[dict[str, str]], CompactionResult]:
        """Compact messages and return the new list plus metadata.

        Args:
            messages: Current message list.
            max_messages: Hard message-count ceiling.

        Returns:
            Tuple of (compacted messages, result).
        """


# ── Callable adapter ──────────────────────────────────────────────────────


class CallableStrategy(CompactionStrategy):
    """Adapter that wraps a plain function as a compaction strategy.

    This allows users to pass a simple function instead of creating
    a full ``CompactionStrategy`` subclass::

        def my_compactor(messages, max_messages):
            kept = [messages[0]] + messages[-max_messages + 1:]
            return kept

        agent = Agent(name="x", compaction=my_compactor)

    The callable signature must be::

        (messages: list[dict[str, str]], max_messages: int)
            -> list[dict[str, str]]

    Args:
        fn: A callable that receives ``(messages, max_messages)`` and
            returns a compacted message list.
        trigger: Optional callable ``(messages, max_messages) -> bool``
            that decides whether to compact.  Defaults to
            ``len(messages) > max_messages``.
    """

    def __init__(
        self,
        fn: Callable[[list[dict[str, str]], int], list[dict[str, str]]],
        *,
        trigger: Callable[[list[dict[str, str]], int], bool] | None = None,
    ) -> None:
        self._fn = fn
        self._trigger = trigger or (lambda msgs, mx: len(msgs) > mx)

    def should_compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> bool:
        return self._trigger(messages, max_messages)

    def compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> tuple[list[dict[str, str]], CompactionResult]:
        result = self._fn(messages, max_messages)
        tokens_saved = (
            sum(len(m.get("content", "")) for m in messages)
            - sum(len(m.get("content", "")) for m in result)
        ) // 4
        logger.debug(
            "CallableStrategy: %d → %d messages (saved ~%d tokens)",
            len(messages), len(result), max(tokens_saved, 0),
        )
        return result, CompactionResult(
            original_count=len(messages),
            compacted_count=len(result),
            summary="",
            estimated_tokens_saved=max(tokens_saved, 0),
        )


# ── LLM summarisation ────────────────────────────────────────────────────

_DEFAULT_COMPACTION_PROMPT = (
    "Summarize the following conversation history into a concise context "
    "paragraph.  Preserve: key facts, decisions made, tool call results, "
    "and any unresolved questions.  Omit: greetings, repeated information, "
    "verbose tool outputs.\n\n"
    "Conversation:\n{conversation}\n\n"
    "Concise summary:"
)

_MSG_CONTENT_CAP: int = 1_000  # per-message truncation for the summary prompt


class SummarizationStrategy(CompactionStrategy):
    """LLM-based compaction via middle-message summarisation.

    When triggered the strategy:

    1. Keeps the system prompt (first message).
    2. Keeps the ``keep_recent`` most recent messages intact.
    3. Summarises the *middle* messages with an LLM call.
    4. Inserts the summary as a ``system`` message directly after the
       original system prompt.

    The strategy never mutates the original list.

    Args:
        service: A ``GenerativeAIService`` instance used for the summary
            LLM call.  May be ``None`` at construction time and set later
            via the :pyattr:`service` property (the agent injects its own
            service lazily).
        trigger_ratio: Fraction of ``max_messages`` at which compaction
            fires (default 0.75 → fires at 30 out of 40 messages).
        keep_recent: Number of most-recent messages to keep verbatim.
        summary_max_tokens: ``max_tokens`` passed to the summary call.
        prompt_template: Custom prompt with ``{conversation}`` placeholder.
        temperature: Temperature for the summary call.
    """

    def __init__(
        self,
        *,
        service: Any | None = None,
        trigger_ratio: float = 0.75,
        keep_recent: int = 6,
        summary_max_tokens: int = 500,
        prompt_template: str | None = None,
        temperature: float = 0.3,
    ) -> None:
        if not 0 < trigger_ratio <= 1:
            raise ValueError("trigger_ratio must be in (0, 1]")

        if keep_recent < 1:
            raise ValueError("keep_recent must be >= 1")

        self._service: Any | None = service
        self.trigger_ratio = trigger_ratio
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.temperature = temperature
        self._prompt_template = prompt_template or _DEFAULT_COMPACTION_PROMPT

    # -- service property (set lazily by the agent) -------------------------

    @property
    def service(self) -> Any | None:
        """Connector service used for the summarisation call."""
        return self._service

    @service.setter
    def service(self, value: Any) -> None:
        self._service = value

    # -- CompactionStrategy interface ---------------------------------------

    def should_compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> bool:
        threshold = int(max_messages * self.trigger_ratio)
        return len(messages) > max(threshold, self.keep_recent + 2)

    def compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> tuple[list[dict[str, str]], CompactionResult]:
        if not self.should_compact(messages, max_messages):
            return messages, CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                summary="",
                estimated_tokens_saved=0,
            )

        # Split: system | middle | recent
        has_system = (
            bool(messages) and messages[0].get("role") == "system"
        )
        system = [messages[0]] if has_system else []
        rest = messages[1:] if has_system else messages

        if len(rest) <= self.keep_recent:
            return messages, CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                summary="",
                estimated_tokens_saved=0,
            )

        middle = rest[:-self.keep_recent]
        recent = rest[-self.keep_recent:]

        # Call LLM for summarisation (or fall back to sliding window)
        summary = self._summarise(middle)

        summary_msg: dict[str, str] = {
            "role": "system",
            "content": f"[Context Summary]\n{summary}",
        }

        result = system + [summary_msg] + recent

        tokens_before = sum(
            len(m.get("content", "")) for m in middle
        ) // 4
        tokens_after = len(summary) // 4
        tokens_saved = max(tokens_before - tokens_after, 0)

        logger.info(
            "Compacted %d → %d messages (summarised %d middle msgs, "
            "saved ~%d tokens)",
            len(messages), len(result), len(middle), tokens_saved,
        )

        return result, CompactionResult(
            original_count=len(messages),
            compacted_count=len(result),
            summary=summary,
            estimated_tokens_saved=tokens_saved,
        )

    # -- internals ----------------------------------------------------------

    def _summarise(self, middle: list[dict[str, str]]) -> str:
        """Call the LLM to produce a summary of *middle* messages.

        If no service is available, falls back to a naive concatenation
        of the first line of each message.

        Args:
            middle: Messages to summarise.

        Returns:
            Summary text.
        """
        conversation = self._format_messages(middle)
        prompt = self._prompt_template.format(conversation=conversation)

        if self._service is None:
            logger.warning(
                "SummarizationStrategy has no service — "
                "falling back to naive compaction"
            )
            return self._naive_summary(middle)

        try:
            return self._service.generate_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.summary_max_tokens,
            )
        except Exception:
            logger.exception("Summarisation LLM call failed — naive fallback")
            return self._naive_summary(middle)

    @staticmethod
    def _format_messages(messages: list[dict[str, str]]) -> str:
        """Format messages for the summarisation prompt.

        Long messages are truncated to ``_MSG_CONTENT_CAP`` characters.

        Args:
            messages: Messages to format.

        Returns:
            Formatted conversation string.
        """
        lines: list[str] = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if len(content) > _MSG_CONTENT_CAP:
                content = content[:_MSG_CONTENT_CAP] + "…"

            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)

    @staticmethod
    def _naive_summary(messages: list[dict[str, str]]) -> str:
        """Produce a short summary without an LLM call.

        Extracts the first 120 characters of each message.

        Args:
            messages: Messages to summarise.

        Returns:
            Concatenated truncated messages.
        """
        parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:120]
            parts.append(f"- [{role}] {content}")

        return "\n".join(parts)


# ── Token-aware variant ───────────────────────────────────────────────────


class TokenAwareStrategy(SummarizationStrategy):
    """Token-aware variant that triggers on estimated token count.

    Instead of counting messages, this strategy estimates the total
    tokens in the message list and fires when the ratio of estimated
    tokens to ``max_context_tokens`` exceeds ``trigger_ratio``.

    Args:
        max_context_tokens: Model's context window size in tokens.
        **kwargs: All other arguments forwarded to
            ``SummarizationStrategy``.
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 100_000,
        **kwargs: Any,
    ) -> None:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")

        super().__init__(**kwargs)
        self.max_context_tokens = max_context_tokens

    def should_compact(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
    ) -> bool:
        estimated = sum(
            len(m.get("content", "")) for m in messages
        ) // 4
        threshold = int(self.max_context_tokens * self.trigger_ratio)
        return estimated > threshold
