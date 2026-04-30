"""Classifier agent — Input classification and routing decisions."""

from __future__ import annotations

from ..llm_agent import LlmAgent

CLASSIFIER_INSTRUCTION = """\
You are an expert classification and routing specialist. Your task is to \
analyze inputs and determine the most appropriate category, label, or route \
based on the content, intent, and context.

You evaluate inputs against defined categories and provide structured \
classification decisions that can drive workflow routing, content tagging, \
or multi-step process control.

Always respond in JSON format with this structure:
{
  "input_summary": "Brief description of the input",
  "classification": "primary_category",
  "confidence": 0.95,
  "secondary_labels": ["optional", "additional", "labels"],
  "reasoning": "Brief explanation of why this classification was chosen",
  "suggested_action": "What to do next based on this classification"
}
"""


def classifier_agent(
    *,
    name: str = "classifier",
    model: str | None = None,
    provider: str = "google",
    instruction: str = CLASSIFIER_INSTRUCTION,
    description: str = "Classifies inputs into categories and suggests actions.",
    output_format: str = "json",
    temperature: float | str = 0.1,
    **kwargs,
) -> LlmAgent:
    """Create a classifier agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider.
        instruction: System prompt.
        description: Short description used by routers.
        output_format: Response format.
        temperature: Sampling temperature (very low for consistent classification).
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` for classification.
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
