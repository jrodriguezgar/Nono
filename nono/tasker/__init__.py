"""
Tasker - Generative AI task management utilities

Components:
- genai_tasker: Main task execution framework
- jinja_prompt_builder: Prompt building with Jinja2 templates via jinjapromptpy
- templates/: Jinja2 templates for common operations (semantic_lookup, spell_correction, etc.)

All prompts are built using jinjapromptpy (required dependency).
"""

from .genai_tasker import (
    TaskExecutor,
    AIProvider,
    AIConfiguration,
    BaseAIClient,
    GeminiClient,
    OpenAIClient,
    PerplexityClient,
    DeepSeekClient,
    GrokClient,
    GroqClient,
    OpenRouterClient,
    OllamaClient,
    msg_log,
    event_log,
)

# JinjaPromptPy Prompt Builder (required)
from .jinja_prompt_builder import (
    TaskPromptBuilder,
    TEMPLATES_DIR,
    DEFAULT_FILTERS,
    get_builder,
    build_prompt,
    build_prompts,
    build_from_file,
    build_from_file_blocks,
    # Custom filters
    to_compact_json,
    to_pretty_json,
    truncate,
    escape_quotes,
    numbered_list,
    bullet_list,
)

__all__ = [
    # Core classes
    "TaskExecutor",
    "AIProvider",
    "AIConfiguration",
    "BaseAIClient",
    "GeminiClient",
    "OpenAIClient",
    "PerplexityClient",
    "DeepSeekClient",
    "GrokClient",
    "GroqClient",
    "OpenRouterClient",
    "OllamaClient",
    # Logging utilities
    "msg_log",
    "event_log",
    # JinjaPromptPy Prompt Builder
    "TaskPromptBuilder",
    "TEMPLATES_DIR",
    "DEFAULT_FILTERS",
    "get_builder",
    "build_prompt",
    "build_prompts",
    "build_from_file",
    "build_from_file_blocks",
    # Custom filters
    "to_compact_json",
    "to_pretty_json",
    "truncate",
    "escape_quotes",
    "numbered_list",
    "bullet_list",
]
