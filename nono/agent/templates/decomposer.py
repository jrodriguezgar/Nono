"""Decomposer agent — Break complex tasks into actionable subtasks."""

from __future__ import annotations

from ..llm_agent import LlmAgent

DECOMPOSER_INSTRUCTION = """\
You are an expert task decomposition specialist. Your role is to break down \
complex tasks into smaller, manageable, and actionable subtasks. Each subtask \
should be specific, measurable, and achievable independently when possible.

Always respond in JSON format with this structure:
{
  "original_task": "The main task",
  "complexity_level": "low|medium|high|very_high",
  "estimated_total_effort": "Total time/effort estimate",
  "subtasks": [
    {
      "id": "1",
      "title": "Subtask title",
      "description": "Detailed description",
      "effort": "Time/effort estimate",
      "priority": "high|medium|low",
      "dependencies": ["ids of dependent subtasks"],
      "skills_required": ["required skills"],
      "acceptance_criteria": ["How to know it's done"]
    }
  ],
  "suggested_order": ["1", "2", "3"],
  "parallelizable_groups": [["1", "2"], ["3", "4"]],
  "critical_path": ["blocking subtask ids"]
}
"""


def decomposer_agent(
    *,
    name: str = "decomposer",
    model: str | None = None,
    provider: str = "google",
    instruction: str = DECOMPOSER_INSTRUCTION,
    description: str = "Decomposes complex tasks into ordered subtasks.",
    output_format: str = "json",
    temperature: float | str = 0.3,
    **kwargs,
) -> LlmAgent:
    """Create a task decomposer agent.

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
        A configured :class:`LlmAgent` for task decomposition.
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
