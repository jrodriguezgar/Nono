# Provider Fallback — Automatic Failover

> Transparent provider/model failover when the primary LLM fails to respond. Configured in `config.toml`, works automatically with both the Agent and Tasker layers.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Default Fallback Chain](#default-fallback-chain)
- [Usage](#usage)
  - [Agent (automatic)](#agent-automatic)
  - [Tasker (automatic)](#tasker-automatic)
  - [Standalone](#standalone)
  - [Disable Fallback](#disable-fallback)
  - [Custom Chain at Runtime](#custom-chain-at-runtime)
- [API Reference](#api-reference)
  - [FallbackHandler](#fallbackhandler)
  - [FallbackConfig](#fallbackconfig)
  - [FallbackEntry](#fallbackentry)
  - [load_fallback_config()](#load_fallback_config)
- [Logging](#logging)
- [FAQ](#faq)
- [See Also](#see-also)

---

## Overview

When an LLM call fails — network timeout, API error, rate-limit, service outage — Nono can automatically retry the request with a different provider/model, walking through a configurable chain until one succeeds.

**Key characteristics:**

| Feature | Detail |
|---------|--------|
| Zero-overhead on success | If the primary provider responds, fallback adds no extra cost or latency |
| Configurable chain | Edit `config.toml` to set the provider order |
| Per-provider retries | Each provider is retried `max_retries` times before moving to the next |
| Lazy service init | Backup services are only instantiated when first needed |
| Transparent integration | Works automatically with `Agent` and `BaseAIClient` |

---

## How It Works

```
Request ──► Primary Provider ──► Success ──► Return response
                 │
                 ▼ (failure)
             Retry N times
                 │
                 ▼ (still fails)
            Fallback #2 ──► Success ──► Return response
                 │
                 ▼ (failure)
            Fallback #3 ──► ...
                 │
                 ▼ (all exhausted)
            RuntimeError
```

1. The request is sent to the **primary provider** (the one configured on the Agent or Tasker).
2. If it fails, the same provider is retried up to `max_retries` times.
3. If all retries fail, the next provider in the chain is tried.
4. The process repeats until a provider succeeds or the chain is exhausted.
5. If all providers fail, a `RuntimeError` is raised with the last error.

---

## Configuration

Fallback is configured in the `[fallback]` section of `nono/config/config.toml`:

```toml
[fallback]
# Enable or disable automatic fallback
enabled = true

# Number of retry attempts per provider before moving to the next
max_retries = 1

# Timeout in seconds for each provider attempt (0 = no override)
timeout = 30

# Fallback chain — tried in order when the active provider fails.
# Each entry needs a provider name and an optional model override.
# Empty model = use [provider].default_model from config.toml.

[[fallback.chain]]
provider = "google"
model = "gemini-3-flash-preview"

[[fallback.chain]]
provider = "groq"
model = "llama-3.3-70b-versatile"

[[fallback.chain]]
provider = "openai"
model = "gpt-4o-mini"

[[fallback.chain]]
provider = "deepseek"
model = "deepseek-chat"

[[fallback.chain]]
provider = "openrouter"
model = "openrouter/auto"
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable/disable fallback globally |
| `max_retries` | `int` | `1` | Retries per provider before falling back |
| `timeout` | `int` | `30` | Per-attempt timeout in seconds |
| `chain` | `list` | See below | Ordered list of provider/model pairs |

### Chain Entry

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `str` | Yes | Provider identifier (e.g. `"google"`, `"openai"`) |
| `model` | `str` | No | Model override. Empty = provider default |

---

## Default Fallback Chain

The default chain is ordered by speed, cost, and reliability:

| # | Provider | Model | Rationale |
|---|----------|-------|-----------|
| 1 | **Google** | `gemini-3-flash-preview` | Fast, cheap, project default |
| 2 | **Groq** | `llama-3.3-70b-versatile` | Ultra-fast LPU inference |
| 3 | **OpenAI** | `gpt-4o-mini` | Reliable, widely available |
| 4 | **DeepSeek** | `deepseek-chat` | Good quality, affordable |
| 5 | **OpenRouter** | `openrouter/auto` | Auto-routes to best available model |

> **Note:** The primary provider is always tried first, regardless of its position in the chain. Duplicate entries are automatically skipped.

---

## Usage

### Agent (automatic)

Fallback is transparent — no code changes needed. When `[fallback].enabled = true`, every `Agent` / `LlmAgent` call goes through the fallback handler:

```python
from nono.agent import Agent, Session
from nono.agent.base import InvocationContext

agent = Agent(
    name="assistant",
    model="gemini-3-flash-preview",
    provider="google",
    instruction="You are a helpful assistant.",
)

session = Session()
ctx = InvocationContext(session=session, user_message="Hello!")
response = agent.run(ctx)
# If Google fails → Groq → OpenAI → DeepSeek → OpenRouter
```

### Tasker (automatic)

The `BaseAIClient` also uses fallback transparently:

```python
from nono.tasker import BaseAIClient, AIConfiguration, AIProvider

config = AIConfiguration(
    provider=AIProvider.GOOGLE,
    model_name="gemini-3-flash-preview",
    api_key="your-api-key",
)
client = BaseAIClient(config)
result = client.generate_content("Explain quantum computing.")
# Automatic fallback on failure
```

### Standalone

Use `FallbackHandler` directly for custom integrations:

```python
from nono.connector.fallback import FallbackHandler

handler = FallbackHandler(
    primary_provider="google",
    primary_model="gemini-3-flash-preview",
)

response = handler.generate_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=1024,
)
```

### Disable Fallback

Set `enabled = false` in `config.toml`:

```toml
[fallback]
enabled = false
```

Or pass a disabled config at runtime:

```python
from nono.connector.fallback import FallbackHandler, FallbackConfig

handler = FallbackHandler(
    primary_provider="google",
    config=FallbackConfig(enabled=False),
)
# Only the primary provider is used — no fallback
```

### Custom Chain at Runtime

Override the chain without modifying `config.toml`:

```python
from nono.connector.fallback import FallbackHandler, FallbackConfig, FallbackEntry

custom_config = FallbackConfig(
    enabled=True,
    max_retries=2,
    timeout=60,
    chain=[
        FallbackEntry(provider="openai", model="gpt-4o"),
        FallbackEntry(provider="google", model="gemini-2.5-flash"),
        FallbackEntry(provider="ollama", model="llama3"),
    ],
)

handler = FallbackHandler(
    primary_provider="openai",
    primary_model="gpt-4o",
    config=custom_config,
)
```

---

## API Reference

### `FallbackHandler`

Main class that wraps LLM calls with automatic failover.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_provider` | `str` | — | Primary provider name |
| `primary_model` | `str \| None` | `None` | Primary model (None = provider default) |
| `api_key` | `str \| None` | `None` | API key override |
| `config` | `FallbackConfig \| None` | `None` | Pre-loaded config (auto-loaded if None) |
| `**service_kwargs` | `Any` | — | Extra kwargs for service constructors |

**Properties:**

- `enabled` → `bool`: Whether fallback is active and has alternative providers.

**Methods:**

#### `generate_completion(messages, **kwargs) → str`

Call `generate_completion` with automatic fallback. Accepts the same arguments as any `GenerativeAIService.generate_completion`.

**Raises:** `RuntimeError` when all providers in the chain have been exhausted.

---

### `FallbackConfig`

Dataclass holding the parsed `[fallback]` configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Fallback active |
| `max_retries` | `int` | `1` | Retries per provider |
| `timeout` | `int` | `30` | Timeout in seconds |
| `chain` | `list[FallbackEntry]` | `[]` | Ordered fallback entries |

---

### `FallbackEntry`

Dataclass representing a single provider/model in the chain.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | — | Provider identifier |
| `model` | `str` | `""` | Model override (empty = default) |

---

### `load_fallback_config()`

```python
def load_fallback_config(config_path: str | Path | None = None) -> FallbackConfig
```

Load the `[fallback]` section from `config.toml`.

**Parameters:**
- `config_path` (`str | Path | None`): Explicit path. `None` = default `nono/config/config.toml`.

**Returns:** `FallbackConfig` instance.

---

## Logging

The fallback handler logs to `Nono.Connector.Fallback`:

| Level | When |
|-------|------|
| `DEBUG` | Each attempt (provider, model, attempt number) |
| `WARNING` | A provider attempt failed (includes error message) |
| `INFO` | Fallback succeeded on a non-primary provider |

Example output:

```
WARNING  Nono.Connector.Fallback: Provider google (model=gemini-3-flash-preview) failed [attempt 1/1]: 503 Service Unavailable
WARNING  Nono.Connector.Fallback: Provider groq (model=llama-3.3-70b-versatile) failed [attempt 1/1]: Connection timeout
INFO     Nono.Connector.Fallback: Fallback succeeded → provider=openai model=gpt-4o-mini (1.2s)
```

---

## FAQ

**Q: Does fallback add latency on success?**
A: No. When the primary provider responds, the fallback handler is a pass-through with negligible overhead.

**Q: Are backup services initialized eagerly?**
A: No. Services for fallback providers are lazily created only when needed.

**Q: What errors trigger a fallback?**
A: Any exception from `generate_completion` — network errors, HTTP 4xx/5xx, rate-limit errors, timeouts, SDK errors.

**Q: Can I use different API keys per provider?**
A: Each provider reads its API key from the standard sources (`config.toml`, `apikey.txt`, environment variables). The `api_key` parameter on `FallbackHandler` applies only to the primary provider.

**Q: Does the fallback chain include the primary provider by default?**
A: Yes. The primary is always first. If it also appears in the chain, the duplicate is skipped.

---

## See Also

- [Connector Documentation](README_connector_genai.md) — multi-provider client reference
- [API Manager](README_api_manager.md) — rate limiting, circuit breakers, retries
- [Configuration Module](../config/README_config.md) — config.toml reference
