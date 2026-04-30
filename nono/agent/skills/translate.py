"""TranslateSkill — Text translation between languages."""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ..llm_agent import LlmAgent
from ..skill import BaseSkill, SkillDescriptor, registry

_INSTRUCTION = """\
You are an expert translator. Translate text accurately while preserving \
the original meaning, tone, and style.

Guidelines:
- Maintain technical terms when appropriate.
- Preserve formatting (bullet points, paragraphs, emphasis).
- Use natural phrasing in the target language — avoid literal translations.
- If the source language is ambiguous, auto-detect it.

Return ONLY the translated text, nothing else.
"""


class TranslateSkill(BaseSkill):
    """Translate text between languages.

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override.
        target_language: Default target language.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
        target_language: str = "English",
        temperature: float | str = 0.3,
    ) -> None:
        self._provider = provider
        self._model = model
        self._target_language = target_language
        self._temperature = temperature

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="translate",
            description=f"Translate text to {self._target_language}.",
            tags=("text", "translation", "language"),
            input_keys=("input",),
            output_keys=("translation",),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        target = overrides.pop("target_language", self._target_language)
        return LlmAgent(
            name="translator",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=(
                f"{_INSTRUCTION}\n"
                f"Target language: {target}."
            ),
            description=self.descriptor.description,
            temperature=overrides.get("temperature", self._temperature),
        )


registry.register(TranslateSkill())
