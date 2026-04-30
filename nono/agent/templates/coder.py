"""Coder agent — Code generation and programming."""

from __future__ import annotations

from ..llm_agent import LlmAgent

CODER_INSTRUCTION = """\
You are an expert programmer. Generate production-ready code that is correct, \
concise, and well-structured.

RULES:
1. Generate ONLY valid, executable code.
2. Use type hints for function signatures.
3. Handle errors gracefully with try/except where critical.
4. Use descriptive but concise names.
5. Prefer idiomatic patterns over verbose alternatives.
6. The code must be self-contained and executable.
7. NEVER include explanatory text — output ONLY code.
8. Do NOT wrap code in markdown code blocks.

When asked for multiple languages, default to Python unless specified.
"""


def coder_agent(
    *,
    name: str = "coder",
    model: str | None = None,
    provider: str = "google",
    instruction: str = CODER_INSTRUCTION,
    description: str = "Generates production-ready code from requirements.",
    output_format: str = "text",
    temperature: float | str = 0.2,
    **kwargs,
) -> LlmAgent:
    """Create a code generation agent.

    Args:
        name: Agent name.
        model: LLM model identifier. ``None`` uses the config default.
        provider: AI provider.
        instruction: System prompt.
        description: Short description used by routers.
        output_format: Response format (``text`` for raw code).
        temperature: Sampling temperature (low for deterministic code).
        **kwargs: Extra arguments forwarded to :class:`LlmAgent`.

    Returns:
        A configured :class:`LlmAgent` for code generation.
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
