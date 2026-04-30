"""Reviewer agent — Quality review and constructive critique."""

from __future__ import annotations

from ..llm_agent import LlmAgent

REVIEWER_INSTRUCTION = """\
You are an expert reviewer and quality analyst. Your role is to critically \
evaluate content, code, plans, or any output and provide constructive, \
actionable feedback. Be thorough but fair.

Evaluation criteria:
- **Correctness**: Are facts, logic, and results accurate?
- **Completeness**: Is anything missing or incomplete?
- **Clarity**: Is it well-organized and easy to understand?
- **Quality**: Does it meet professional standards?
- **Improvements**: What specific changes would make it better?

Always respond in JSON format with this structure:
{
  "verdict": "approve|revise|reject",
  "score": 8,
  "strengths": ["What works well"],
  "issues": [
    {
      "severity": "critical|major|minor|suggestion",
      "description": "What the issue is",
      "location": "Where it occurs (if applicable)",
      "fix": "Suggested correction"
    }
  ],
  "summary": "Overall assessment in 1-2 sentences"
}
"""


def reviewer_agent(
    *,
    name: str = "reviewer",
    model: str | None = None,
    provider: str = "google",
    instruction: str = REVIEWER_INSTRUCTION,
    description: str = "Reviews content and provides quality feedback.",
    output_format: str = "json",
    temperature: float | str = 0.3,
    **kwargs,
) -> LlmAgent:
    """Create a reviewer agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider.
        instruction: System prompt.
        description: Short description used by routers.
        output_format: Response format.
        temperature: Sampling temperature.
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` for reviewing and critiquing.
    """
    return LlmAgent(
        name=name,
        model=model,
        provider=provider,
        instruction=instruction,
        description=description,
        output_format=output_format,
        temperature=temperature,
        **kwargs,
    )
