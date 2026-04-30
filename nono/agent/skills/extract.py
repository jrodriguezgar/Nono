"""ExtractSkill — Structured data extraction from text."""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ..llm_agent import LlmAgent
from ..skill import BaseSkill, SkillDescriptor, registry

_INSTRUCTION = """\
You are an expert data extraction specialist. Extract structured information \
from unstructured text and return it in a clean JSON format.

Guidelines:
- Extract all relevant entities: names, dates, numbers, locations, etc.
- Use null for fields that cannot be determined from the text.
- Preserve original values — do not paraphrase or interpret.

Always respond in JSON format:
{
  "entities": [
    {"text": "extracted value", "type": "entity_type", "context": "surrounding text"}
  ],
  "summary": "Brief description of what was extracted"
}
"""


class ExtractSkill(BaseSkill):
    """Extract structured data (entities, fields) from unstructured text.

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override.
        schema: Optional JSON schema hint for expected output fields.
        temperature: LLM temperature.
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
        schema: dict[str, str] | None = None,
        temperature: float | str = 0.1,
    ) -> None:
        self._provider = provider
        self._model = model
        self._schema = schema
        self._temperature = temperature

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="extract",
            description="Extract structured data from unstructured text.",
            tags=("text", "extraction", "data"),
            input_keys=("input",),
            output_keys=("entities",),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        instruction = _INSTRUCTION
        if self._schema:
            fields = ", ".join(f"{k} ({v})" for k, v in self._schema.items())
            instruction += f"\nExpected fields: {fields}"

        return LlmAgent(
            name="extractor",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=instruction,
            description=self.descriptor.description,
            output_format="json",
            temperature=overrides.get("temperature", self._temperature),
        )


registry.register(ExtractSkill())
