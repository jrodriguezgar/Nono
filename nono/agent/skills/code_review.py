"""CodeReviewSkill — Code quality review and improvement suggestions."""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ..llm_agent import LlmAgent
from ..skill import BaseSkill, SkillDescriptor, registry

_INSTRUCTION = """\
You are an expert code reviewer. Analyze code for quality, correctness, \
performance, security, and best practices.

Review checklist:
- Correctness: logic errors, edge cases, off-by-one.
- Security: injection, path traversal, credential leaks (OWASP Top 10).
- Performance: unnecessary allocations, O(n²) where O(n) is possible.
- Style: naming, structure, DRY violations.
- Best practices: error handling, type hints, documentation.

Always respond in JSON format:
{
  "overall_score": 8,
  "issues": [
    {"severity": "high|medium|low", "line": 42, "description": "...", "suggestion": "..."}
  ],
  "strengths": ["..."],
  "summary": "Brief overall assessment"
}
"""


class CodeReviewSkill(BaseSkill):
    """Review code for quality, security, and best practices.

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override.
        language: Programming language hint (e.g. ``"python"``).
        temperature: LLM temperature.
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
        language: str = "python",
        temperature: float | str = 0.2,
    ) -> None:
        self._provider = provider
        self._model = model
        self._language = language
        self._temperature = temperature

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="code_review",
            description=f"Review {self._language} code for quality, security, and best practices.",
            tags=("code", "review", "security", self._language),
            input_keys=("code",),
            output_keys=("issues", "overall_score"),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        return LlmAgent(
            name="code_reviewer",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=(
                f"{_INSTRUCTION}\n"
                f"Primary language: {self._language}."
            ),
            description=self.descriptor.description,
            output_format="json",
            temperature=overrides.get("temperature", self._temperature),
        )


registry.register(CodeReviewSkill())
