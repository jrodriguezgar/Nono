"""Extractor agent — Structured data extraction from text."""

from __future__ import annotations

from ..llm_agent import LlmAgent

EXTRACTOR_INSTRUCTION = """\
You are an expert data extraction specialist. Your role is to identify and \
extract structured information from unstructured text. You are precise, \
thorough, and never invent data that is not present in the source.

Guidelines:
- Extract ONLY information explicitly present in the text.
- Use null/empty for fields where information is not available.
- Maintain original values without paraphrasing or interpreting.
- Flag ambiguous extractions with low confidence.

Always respond in JSON format with this structure:
{
  "source_type": "email|article|report|conversation|other",
  "extracted_data": {
    "entities": [
      {
        "type": "person|organization|location|date|amount|product|other",
        "value": "The extracted value",
        "context": "Surrounding text for verification",
        "confidence": "high|medium|low"
      }
    ],
    "facts": ["key factual statements extracted"],
    "dates": ["any dates mentioned"],
    "numbers": ["any numerical values with context"]
  },
  "completeness": "full|partial",
  "notes": ["Any observations about extraction quality"]
}
"""


def extractor_agent(
    *,
    name: str = "extractor",
    model: str | None = None,
    provider: str = "google",
    instruction: str = EXTRACTOR_INSTRUCTION,
    description: str = "Extracts structured data from unstructured text.",
    output_format: str = "json",
    temperature: float | str = 0.1,
    **kwargs,
) -> LlmAgent:
    """Create a data extractor agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider.
        instruction: System prompt.
        description: Short description used by routers.
        output_format: Response format.
        temperature: Sampling temperature (very low for faithful extraction).
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` for data extraction.
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
