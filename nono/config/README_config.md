# Configuration Management - Nono

> Single-file configuration for the entire Nono framework via `config.toml`.

## Table of Contents

- [Overview](#overview)
- [File Location](#file-location)
- [Priority Resolution](#priority-resolution)
- [config.toml Reference](#configtoml-reference)
  - [Provider Defaults](#provider-defaults)
  - [Rate Limits](#rate-limits)
  - [Agent](#agent)
  - [Provider Fallback](#provider-fallback)
  - [Executer](#executer)
  - [CLI](#cli)
  - [Workflow](#workflow)
  - [Paths](#paths)
- [Environment Variable Overrides](#environment-variable-overrides)
- [Config API (Programmatic Access)](#config-api-programmatic-access)
- [Schema Validation](#schema-validation)
- [CLI Integration](#cli-integration)
- [Testing with Isolated Config](#testing-with-isolated-config)
- [Legacy API](#legacy-api)

---

## Overview

All Nono settings live in a single **TOML** file: `nono/config/config.toml`. Every module — connectors, agent, tasker, executer, CLI, workflows — reads its parameters from this file at startup.

Alternative formats (JSON, YAML) are supported by the `Config` API but the canonical source that ships with the project is TOML.

## File Location

```
nono/
└── config/
    ├── config.py          # Config class & loaders
    └── config.toml        # ← Central configuration file
```

The framework auto-discovers `config.toml` relative to the `config/` package. You can override the path via the `NONO_CONFIG_FILE` environment variable or by passing it explicitly.

## Priority Resolution

When the same key is defined in multiple sources, the **first found wins**:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  CLI Args   │ -> │  Env Vars   │ -> │  TOML File  │ -> │  Defaults   │
│  (highest)  │    │  NONO_*     │    │ config.toml │    │  (lowest)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

| Priority | Source | How to set |
|----------|--------|------------|
| 1 (high) | Arguments | `config.load_args()` / `config.set()` / CLI flags |
| 2 | Environment variables | `NONO_<KEY>` (double `__` for nesting) |
| 3 | Configuration file | `config.toml` |
| 4 (low) | Default values | `Config(defaults={...})` |

---

## config.toml Reference

### Provider Defaults

Each supported provider has a `[provider]` section with at least `default_model`. The model used when no explicit model is passed.

```toml
[google]
default_model = "gemini-3-flash-preview"

[openai]
default_model = "gpt-4o-mini"

[perplexity]
default_model = "sonar"

[deepseek]
default_model = "deepseek-chat"

[xai]
default_model = "grok-3"

[groq]
default_model = "llama-3.3-70b-versatile"

[cerebras]
default_model = "llama-3.3-70b"

[nvidia]
default_model = "meta/llama-3.3-70b-instruct"

[huggingface]
default_model = "meta-llama/Llama-3.3-70B-Instruct"

[github]
default_model = "openai/gpt-5"

[openrouter]
default_model = "openrouter/auto"

[foundry]
default_model = "openai/gpt-4o"

[vercel]
default_model = "anthropic/claude-opus-4.5"

[ollama]
host = "http://localhost:11434"
default_model = "llama3"
```

| Provider | Key | Default |
|----------|-----|---------|
| Google | `google.default_model` | `gemini-3-flash-preview` |
| OpenAI | `openai.default_model` | `gpt-4o-mini` |
| Perplexity | `perplexity.default_model` | `sonar` |
| DeepSeek | `deepseek.default_model` | `deepseek-chat` |
| xAI | `xai.default_model` | `grok-3` |
| Groq | `groq.default_model` | `llama-3.3-70b-versatile` |
| Cerebras | `cerebras.default_model` | `llama-3.3-70b` |
| NVIDIA | `nvidia.default_model` | `meta/llama-3.3-70b-instruct` |
| Hugging Face | `huggingface.default_model` | `meta-llama/Llama-3.3-70B-Instruct` |
| GitHub Models | `github.default_model` | `openai/gpt-5` |
| OpenRouter | `openrouter.default_model` | `openrouter/auto` |
| Azure AI Foundry | `foundry.default_model` | `openai/gpt-4o` |
| Vercel | `vercel.default_model` | `anthropic/claude-opus-4.5` |
| Ollama | `ollama.default_model` | `llama3` |

> **API keys** are stored separately in `apikey.txt` or via environment variables (e.g., `GOOGLE_API_KEY`). Never put keys in `config.toml`.

---

### Rate Limits

```toml
[rate_limits]
delay_between_requests = 0.5
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `delay_between_requests` | `float` | `0.5` | Seconds between consecutive API calls (Token Bucket) |

---

### Agent

Controls `LlmAgent`, `RouterAgent`, and the agent transfer system.

```toml
[agent]
default_provider = "google"
default_model = ""
temperature = 0.7
max_tokens = 4096
router_max_iterations = 3
router_temperature = 0.0
max_tool_iterations = 10
max_loop_messages = 40
max_transfer_depth = 10
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_provider` | `str` | `"google"` | Provider for `LlmAgent` and routing calls |
| `default_model` | `str` | `""` | Model override (empty = use provider's default) |
| `temperature` | `float` | `0.7` | LLM temperature for agent responses |
| `max_tokens` | `int` | `4096` | Maximum tokens per response |
| `router_max_iterations` | `int` | `3` | Max loop iterations for `RouterAgent` |
| `router_temperature` | `float` | `0.0` | Temperature for routing decisions (lower = deterministic) |
| `max_tool_iterations` | `int` | `10` | Max tool-call loop iterations (prevents infinite loops) |
| `max_loop_messages` | `int` | `40` | Max messages in tool-calling loop (prevents context overflow) |
| `max_transfer_depth` | `int` | `10` | Max recursive agent transfers (prevents stack overflow) |

---

### Provider Fallback

Automatic failover when the active provider fails. See also [Fallback Documentation](../connector/README_fallback.md).

```toml
[fallback]
enabled = true
max_retries = 1
timeout = 30

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

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | `bool` | `true` | Enable/disable automatic fallback |
| `max_retries` | `int` | `1` | Retry attempts per provider before moving to next |
| `timeout` | `int` | `30` | Timeout in seconds per attempt (0 = no override) |

Each `[[fallback.chain]]` entry defines a provider + model pair tried in order. The active provider is always tried first; then the chain is walked. Empty `model` = use the provider's `default_model`.

---

### Executer

Controls the code generation and execution engine (`genai_executer`).

```toml
[executer]
mode = "subprocess"
security = "safe"
timeout = 30
max_retries = 2
save_executions = true
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | `str` | `"subprocess"` | `"subprocess"` (isolated process) or `"exec"` (in-process) |
| `security` | `str` | `"safe"` | `"safe"` (blocks dangerous ops) or `"permissive"` (allows all) |
| `timeout` | `int` | `30` | Timeout in seconds for code execution |
| `max_retries` | `int` | `2` | Max retry attempts for failed executions |
| `save_executions` | `bool` | `true` | Persist execution results to disk |

---

### CLI

Default values for the command-line interface. CLI flags override these.

```toml
[cli]
colors_enabled = true
default_output_format = "summary"
default_log_level = "info"
default_provider = "google"
allow_parameter_files = true
require_confirmation = false
dry_run_by_default = false
default_timeout = 60
default_max_tokens = 4096
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `colors_enabled` | `bool` | `true` | Enable ANSI color output |
| `default_output_format` | `str` | `"summary"` | `"summary"`, `"json"`, `"text"`, or `"table"` |
| `default_log_level` | `str` | `"info"` | `"debug"`, `"info"`, `"warning"`, or `"error"` |
| `default_provider` | `str` | `"google"` | Default AI provider for CLI commands |
| `allow_parameter_files` | `bool` | `true` | Allow reading arguments from `@file` |
| `require_confirmation` | `bool` | `false` | Ask confirmation before executing |
| `dry_run_by_default` | `bool` | `false` | Run in dry-run mode by default |
| `default_timeout` | `int` | `60` | Default timeout in seconds |
| `default_max_tokens` | `int` | `4096` | Default maximum tokens for responses |

---

### Workflow

Settings for multi-step execution pipelines.

```toml
[workflow]
log_steps = true
default_input_key = "input"
default_output_key = "output"
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_steps` | `bool` | `true` | Log step transitions |
| `default_input_key` | `str` | `"input"` | Default `input_key` for `tasker_node` / `agent_node` |
| `default_output_key` | `str` | `"output"` | Default `output_key` for `tasker_node` / `agent_node` |

---

### Paths

Override directories for templates and task definitions.

```toml
[paths]
templates_dir = ""
prompts_dir = ""
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `templates_dir` | `str` | `""` | Jinja2 templates directory (empty = `nono/tasker/templates`) |
| `prompts_dir` | `str` | `""` | Task definitions directory (empty = `nono/tasker/prompts`) |

Relative paths are resolved from the project root.

---

## Environment Variable Overrides

Prefix: **`NONO_`**. Use double underscore (`__`) for nested keys.

| Environment Variable | config.toml Key |
|---------------------|-----------------|
| `NONO_GOOGLE__DEFAULT_MODEL` | `google.default_model` |
| `NONO_AGENT__TEMPERATURE` | `agent.temperature` |
| `NONO_EXECUTER__TIMEOUT` | `executer.timeout` |
| `NONO_FALLBACK__ENABLED` | `fallback.enabled` |
| `NONO_CLI__DEFAULT_LOG_LEVEL` | `cli.default_log_level` |
| `NONO_RATE_LIMITS__DELAY_BETWEEN_REQUESTS` | `rate_limits.delay_between_requests` |

### Automatic Type Conversion

```bash
NONO_FALLBACK__ENABLED=false     # → False (bool)
NONO_EXECUTER__TIMEOUT=60        # → 60 (int)
NONO_AGENT__TEMPERATURE=0.3      # → 0.3 (float)
NONO_HOSTS='["a","b"]'           # → ["a", "b"] (list)
```

### Special Environment Variables

| Variable | Description |
|----------|-------------|
| `NONO_CONFIG_FILE` | Override path to `config.toml` |
| `NONO_TEMPLATES_DIR` | Override templates directory |
| `NONO_PROMPTS_DIR` | Override prompts directory |

---

## Config API (Programmatic Access)

### Quick Start

```python
from nono.config import Config, load_config

# One-liner
config = load_config(filepath='config.toml', env_prefix='NONO_')
print(config['google.default_model'])
```

### `Config` Class

```python
config = Config(
    defaults={'app.timeout': 30},    # Default values
    schema=None,                     # Schema for validation (optional)
    auto_discover=False              # Auto-search for config.toml
)

# Load sources
config.load_file('config.toml')     # TOML, JSON, or YAML
config.load_env(prefix='NONO_')     # Environment variables
config.load_args({'debug': True})   # Dict or argparse.Namespace

# Read values
model = config.get('google.default_model')
port = config.get('server.port', default=8080, type=int)
key = config.require('api.key')     # Raises ValueError if missing

# Write values (maximum priority)
config.set('runtime.mode', 'production')

# Dictionary-style access
value = config['key']
config['key'] = 'value'
exists = 'key' in config

# Inspect
all_values = config.all()
source = config.get_source('google.default_model')  # ConfigSource.FILE
```

### Method Chaining

```python
config = (
    Config(defaults={'app.debug': False})
    .load_file('config.toml')
    .load_env(prefix='NONO_')
    .set('runtime.mode', 'production')
)
```

### `load_config()` Shortcut

```python
config = load_config(
    filepath='config.toml',     # File path (optional)
    defaults={'timeout': 30},   # Default values
    env_prefix='NONO_'          # Prefix for environment variables
)
```

### Supported File Formats

| Format | Extension | Dependency |
|--------|-----------|------------|
| TOML | `.toml` | Built-in (3.11+) or `tomli` |
| JSON | `.json` | Built-in |
| YAML | `.yaml` | `pyyaml` |

### API Quick Reference

| Method | Description |
|--------|-------------|
| `Config()` | Create new instance |
| `load_file(path)` | Load from file |
| `load_env(prefix)` | Load from environment variables |
| `load_args(args)` | Load from arguments |
| `get(key, default, type)` | Get value with fallback |
| `set(key, value)` | Set value (maximum priority) |
| `require(key, message)` | Get required value |
| `all()` | Get all resolved values |
| `validate()` | Validate against schema |
| `copy()` | Deep copy (useful for tests) |
| `get_source(key)` | Get value source |
| `load_config()` | Shortcut function |
| `create_sample_config()` | Generate example file |

### Enums

```python
from nono.config import ConfigSource, ConfigFormat

ConfigSource.DEFAULT       # Default value
ConfigSource.FILE          # Configuration file
ConfigSource.ENVIRONMENT   # Environment variable
ConfigSource.ARGUMENT      # Programmatic argument

ConfigFormat.JSON
ConfigFormat.YAML
ConfigFormat.TOML
```

---

## Schema Validation

```python
from nono.config import Config, ConfigSchema

schema = ConfigSchema()
schema.add_field('google.default_model', type=str, required=True)
schema.add_field('rate_limits.delay_between_requests',
                 type=float, min_value=0, max_value=10)
schema.add_field('agent.temperature', type=float, min_value=0, max_value=2)

config = Config(schema=schema)
config.load_file('config.toml')

# Raises ValueError on failure
config.validate()

# Or collect errors
is_valid, errors = config.validate(raise_on_error=False)
```

| Parameter | Description |
|-----------|-------------|
| `type` | Expected type (`str`, `int`, `float`, `bool`) |
| `required` | `True` if field is mandatory |
| `default` | Default value |
| `choices` | List of allowed values |
| `min_value` | Minimum (numeric) |
| `max_value` | Maximum (numeric) |

---

## CLI Integration

The CLI reads defaults from `[cli]` in `config.toml`. Command-line flags override them.

```bash
# config.toml values used as defaults
python -m nono.cli --prompt "Hello"

# Override provider and model via CLI flags
python -m nono.cli --provider openai --model gpt-4o --prompt "Hello"

# Override via environment variable
NONO_CLI__DEFAULT_PROVIDER=openai python -m nono.cli --prompt "Hello"
```

### Programmatic CLI + Config

```python
from nono.config import load_config
from nono.cli import CLIBase

config = load_config(filepath='config.toml', env_prefix='NONO_')
cli = CLIBase(prog="my_tool", version="1.0.0")
args = cli.parse_args()
config.load_args(vars(args))

# Priority: args > env > config.toml > defaults
model = config['google.default_model']
```

---

## Testing with Isolated Config

```python
from nono.config import Config, load_config

def test_feature():
    config = Config(defaults={'test.mode': True})

    # Or isolate from the real config
    original = load_config()
    isolated = original.copy()
    isolated.set('agent.temperature', 0.0)

    assert original.get('agent.temperature') != 0.0
```

---

## Legacy API

```python
from nono.config import (
    NonoConfig,
    get_templates_dir,
    get_prompts_dir,
    set_templates_dir,
    set_prompts_dir,
)

model = NonoConfig.get_config_value('google', 'default_model')
templates = get_templates_dir()
prompts = get_prompts_dir()
```

---

## Author

**DatamanEdge** - MIT License
