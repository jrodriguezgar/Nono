"""Planner agent — Strategic planning and project breakdown."""

from __future__ import annotations

from ..llm_agent import LlmAgent

PLANNER_INSTRUCTION = """\
You are an expert project planner and strategist. Your task is to create \
detailed, actionable plans to achieve goals. You break down complex objectives \
into manageable phases with clear milestones, timelines, and success criteria.

Always respond in JSON format with this structure:
{
  "goal": "The main objective",
  "summary": "Brief overview of the plan",
  "phases": [
    {
      "phase_number": 1,
      "name": "Phase name",
      "description": "What this phase accomplishes",
      "duration": "Estimated time",
      "milestones": ["milestone1", "milestone2"],
      "deliverables": ["deliverable1"],
      "dependencies": ["any prerequisites"],
      "risks": ["potential risks"]
    }
  ],
  "total_duration": "Total estimated time",
  "success_criteria": ["How to measure success"],
  "resources_needed": ["Required resources"]
}
"""


def planner_agent(
    *,
    name: str = "planner",
    model: str | None = None,
    provider: str = "google",
    instruction: str = PLANNER_INSTRUCTION,
    description: str = "Plans and strategizes how to achieve a goal.",
    output_format: str = "json",
    temperature: float | str = 0.4,
    **kwargs,
) -> LlmAgent:
    """Create a planner agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider (``google``, ``openai``, …).
        instruction: System prompt. Override for custom planning styles.
        description: Short description used by routers.
        output_format: Response format (``json`` or ``text``).
        temperature: Sampling temperature.
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` ready for planning tasks.
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
