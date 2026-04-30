"""SummarizeSkill — Text summarization into key points."""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ..llm_agent import LlmAgent
from ..skill import BaseSkill, SkillDescriptor, registry

_INSTRUCTION = """\
You are an expert summarization specialist. Produce concise, accurate \
summaries that capture the essential information.

Guidelines:
- Preserve key facts, figures, and conclusions.
- Maintain the original meaning without adding interpretation.
- Organize with clear structure when the source is long.
- Use bullet points for key points.
"""


class SummarizeSkill(BaseSkill):
    """Summarize text into a concise overview with key points.

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override (``None`` for provider default).
        max_points: Maximum number of key points to extract.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
        max_points: int = 5,
        temperature: float | str = 0.3,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_points = max_points
        self._temperature = temperature

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="summarize",
            description=f"Summarize text into up to {self._max_points} key points.",
            tags=("text", "summarization", "analysis"),
            input_keys=("input",),
            output_keys=("summary", "key_points"),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        return LlmAgent(
            name="summarizer",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=(
                f"{_INSTRUCTION}\n"
                f"Produce at most {self._max_points} key points."
            ),
            description=self.descriptor.description,
            temperature=overrides.get("temperature", self._temperature),
        )


registry.register(SummarizeSkill())
