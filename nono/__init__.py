"""
Nono - GenAI Framework

A unified framework for executing tasks using Generative AI.
Supports multiple LLM providers: Google Gemini, OpenAI, Perplexity, DeepSeek, Grok, and Ollama.

Submodules:
- config: Central configuration management (paths, settings)
- connector: Shared AI service connectors
- tasker: Task execution framework
  - data_stage: Batch data operations with intelligent throttling
- executer: Code generation and execution

Configuration from external projects:
    from nono.config import NonoConfig, set_templates_dir, set_prompts_dir
    
    # Via environment variables (highest priority)
    os.environ["NONO_TEMPLATES_DIR"] = "/path/to/templates"
    os.environ["NONO_PROMPTS_DIR"] = "/path/to/prompts"
    
    # Via programmatic API
    set_templates_dir("/path/to/templates")
    set_prompts_dir("/path/to/prompts")
"""

__version__ = "1.1.0"

# Export configuration utilities
from .config import (
    NonoConfig,
    get_templates_dir,
    get_prompts_dir,
    set_templates_dir,
    set_prompts_dir,
)
