# GitHub Copilot Instructions - Nono GenAI Tasker

## Project

**NONO** = **N**o **O**verhead, **N**eural **O**perations. Unified framework for GenAI-driven operations.

Providers: Google Gemini, OpenAI, Anthropic Claude, Perplexity, DeepSeek, and 9 more (xAI, Groq, Cerebras, NVIDIA, HuggingFace, GitHub Models, OpenRouter, Azure AI, Ollama).

## Conventions

- English everywhere · PEP 8 · Type hints always · Google Docstrings
- Default model: **`gemini-3-flash-preview`** (configured in `config.toml`)

### File Structure

```
nono/
├── agent/                  # AI agent framework (LlmAgent, orchestration)
│   └── templates/          # Pre-configured agent templates & pipelines
├── cli/                    # Command-line interface
├── config/                 # Configuration management + config.toml
├── connector/              # AI service connectors
├── executer/               # Code generation and execution
├── tasker/                 # Task execution framework
│   ├── prompts/            # JSON task definition files
│   └── templates/          # Jinja2 prompt templates
├── visualize/              # ASCII rendering (workflows + agents)
├── workflows/              # Multi-step execution pipelines
└── examples/               # Usage examples
```

## Design Patterns

**SOLID**: Single Responsibility · Open/Closed (base classes extensible) · Liskov (AI clients interchangeable) · Interface Segregation · Dependency Inversion

| Base Class | Purpose |
|-----------|---------|
| `GenerativeAIService` | Abstract class for all AI connectors |
| `OpenAICompatibleService` | Base for OpenAI-compatible REST APIs |
| `AIProvider` | Enum of supported providers |

## Common Patterns

See [`patterns/common_patterns.md`](patterns/common_patterns.md) — load with `read_file` when implementing new code.

Covers: `@event_log` decorator, `msg_log` function, task config JSON schema, SSL configuration.

## Security

- API keys in `config.toml` or `apikey.txt` — ❌ keys in code, ✓ env vars as alternative
- SSL: `SSLVerificationMode.CERTIFI` (prod) · `INSECURE` (dev only) · `CUSTOM` (corporate certs)

## Supported Providers (top 5)

| Provider | Default Model |
|----------|---------------|
| **Google** | `gemini-3-flash-preview` ← project default |
| OpenAI | `gpt-4o-mini` |
| Anthropic | `claude-sonnet-4` |
| DeepSeek | `deepseek-chat` |
| Groq | `llama-3.3-70b-versatile` |

…and 10 more (xAI, Cerebras, NVIDIA, HuggingFace, GitHub Models, OpenRouter, Azure AI, Vercel, Perplexity, Ollama).

## Best Practices

- New AI service → inherit `BaseService` · New task → JSON in `prompts/`
- Errors → use `msg_log` · Rate limiting → Token Bucket · Validation → `jsonschema`
- Tests: `pytest`, mock external APIs, cover each provider

## Main Dependencies

`google-genai` · `openai` (also Groq/NVIDIA/OpenRouter/xAI/DeepSeek/Perplexity/HuggingFace/GitHub) · `anthropic` · `httpx` · `certifi` — and 4 more (`cerebras-cloud-sdk`, `azure-ai-inference`, `azure-ai-projects`, `jsonschema`)

