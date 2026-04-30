"""Writer agent — Content writing and text generation."""

from __future__ import annotations

from ..llm_agent import LlmAgent

WRITER_INSTRUCTION = """\
You are an expert content writer. You produce clear, engaging, and \
well-structured text adapted to the audience and purpose. Your writing is \
concise, professional, and free of filler.

Guidelines:
- Adapt tone and style to the requested format (article, email, report, etc.).
- Use clear structure: headings, paragraphs, bullet points as appropriate.
- Be factual — do not fabricate information.
- Keep sentences direct and active voice when possible.
- Match the requested length and level of detail.
"""


def writer_agent(
    *,
    name: str = "writer",
    model: str | None = None,
    provider: str = "google",
    instruction: str = WRITER_INSTRUCTION,
    description: str = "Writes clear, structured content for any format.",
    output_format: str = "text",
    temperature: float | str = 0.7,
    **kwargs,
) -> LlmAgent:
    """Create a writer agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider.
        instruction: System prompt.
        description: Short description used by routers.
        output_format: Response format (``text`` for natural prose).
        temperature: Sampling temperature (higher for creative writing).
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` for content writing.
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
