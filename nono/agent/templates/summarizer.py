"""Summarizer agent — Text and document summarization."""

from __future__ import annotations

from ..llm_agent import LlmAgent

SUMMARIZER_INSTRUCTION = """\
You are an expert summarization specialist. You produce concise, accurate \
summaries that capture the essential information from any text or document.

Guidelines:
- Preserve key facts, figures, and conclusions.
- Maintain the original meaning without adding interpretation.
- Organize the summary with clear structure when the source is long.
- Adjust length proportionally to the input (short input → brief summary).

Always respond in JSON format with this structure:
{
  "summary": "The concise summary",
  "key_points": ["point1", "point2", "point3"],
  "word_count": 150,
  "compression_ratio": "20%",
  "topics": ["main topic tags"]
}
"""


def summarizer_agent(
    *,
    name: str = "summarizer",
    model: str | None = None,
    provider: str = "google",
    instruction: str = SUMMARIZER_INSTRUCTION,
    description: str = "Summarizes text and documents into concise overviews.",
    output_format: str = "json",
    temperature: float | str = 0.3,
    **kwargs,
) -> LlmAgent:
    """Create a summarizer agent.

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
        A configured :class:`LlmAgent` for summarization.
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
