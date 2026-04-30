"""ClassifySkill — Input classification and labelling."""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ..llm_agent import LlmAgent
from ..skill import BaseSkill, SkillDescriptor, registry

_INSTRUCTION = """\
You are an expert classification specialist. Analyze inputs and determine \
the most appropriate category based on content, intent, and context.

Always respond in JSON format:
{
  "classification": "primary_category",
  "confidence": 0.95,
  "secondary_labels": ["optional", "labels"],
  "reasoning": "Brief explanation"
}
"""


class ClassifySkill(BaseSkill):
    """Classify text into categories with confidence scores.

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override.
        categories: Optional list of valid categories to constrain output.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
        categories: list[str] | None = None,
        temperature: float | str = 0.1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._categories = categories
        self._temperature = temperature

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="classify",
            description="Classify text into categories with confidence scores.",
            tags=("text", "classification", "routing"),
            input_keys=("input",),
            output_keys=("classification", "confidence"),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        instruction = _INSTRUCTION
        if self._categories:
            cats = ", ".join(self._categories)
            instruction += f"\nValid categories: {cats}"

        return LlmAgent(
            name="classifier",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=instruction,
            description=self.descriptor.description,
            output_format="json",
            temperature=overrides.get("temperature", self._temperature),
        )


registry.register(ClassifySkill())
