# Configuration Module - Nono

## Description

Unified configuration management with multi-source priority resolution and instance-based design for flexibility and testability.

## Installation

The module is included in the project. No additional dependencies required for JSON/TOML (Python 3.11+).

For Python < 3.11:
```bash
pip install tomli
```

For YAML support:
```bash
pip install pyyaml
```

## Quick Start

```python
from nono.config import Config, load_config

# Simple usage
config = load_config(filepath='config.toml', env_prefix='NONO_')
print(config['google.default_model'])

# With method chaining
config = (
    Config(defaults={'timeout': 30})
    .load_file('config.toml')
    .load_env(prefix='NONO_')
)
```

## Priority Resolution

Values are resolved in order (first found wins):

| Priority | Source | Method |
|-----------|--------|--------|
| 1 (High) | Arguments | `load_args()` / `set()` |
| 2 | Environment variables | `load_env(prefix='NONO_')` |
| 3 | Configuration file | `load_file('config.toml')` |
| 4 (Low) | Default values | `Config(defaults={...})` |

## Main API

### `Config` Class

```python
from nono.config import Config

# Create instance
config = Config(
    defaults={'app.timeout': 30},      # Default values
    schema=None,                        # Schema for validation (optional)
    auto_discover=False                 # Auto-search for config.toml
)

# Load from file
config.load_file('config.toml')         # TOML
config.load_file('config.json')         # JSON
config.load_file('config.yaml')         # YAML (requires pyyaml)

# Load from environment variables
config.load_env(prefix='NONO_')

# Load from arguments (argparse.Namespace or dict)
config.load_args({'debug': True, 'port': 8080})

# Get values
model = config.get('google.default_model')
port = config.get('server.port', default=8080, type=int)
api_key = config.require('api.key')  # Raises ValueError if not found

# Set values (maximum priority)
config.set('runtime.mode', 'production')

# Dictionary-style access
value = config['key']
config['key'] = 'value'
exists = 'key' in config

# Get all values
all_config = config.all()

# Get source of a value
source = config.get_source('google.default_model')  # ConfigSource.FILE

# Copy configuration (useful for tests)
isolated = config.copy()
```

### Method Chaining

All load methods return `self` for chaining:

```python
config = (
    Config(defaults={'app.debug': False})
    .load_file('config.toml')
    .load_env(prefix='NONO_')
    .set('runtime.mode', 'production')
)
```

### `load_config()` Function

Shortcut for creating and loading configuration:

```python
from nono.config import load_config

config = load_config(
    filepath='config.toml',     # File path (optional)
    defaults={'timeout': 30},   # Default values
    env_prefix='NONO_'          # Prefix for environment variables
)
```

## File Formats

### TOML (config.toml)

```toml
[google]
default_model = "gemini-3-flash-preview"

[rate_limits]
delay_between_requests = 0.5

[paths]
templates_dir = ""
prompts_dir = ""
```

### JSON (config.json)

```json
{
    "google": {
        "default_model": "gemini-3-flash-preview"
    },
    "rate_limits": {
        "delay_between_requests": 0.5
    }
}
```

### YAML (config.yaml)

```yaml
google:
  default_model: gemini-3-flash-preview

rate_limits:
  delay_between_requests: 0.5
```

## Environment Variables

Prefix: `NONO_`

### Key Mapping

| Environment Variable | Configuration Key |
|---------------------|----------------------|
| `NONO_TIMEOUT` | `timeout` |
| `NONO_GOOGLE__DEFAULT_MODEL` | `google.default_model` |
| `NONO_RATE_LIMITS__DELAY` | `rate_limits.delay` |

> **Note**: Use double underscore (`__`) for nested keys.

### Automatic Type Conversion

```bash
NONO_DEBUG=true          # → True (bool)
NONO_PORT=8080           # → 8080 (int)
NONO_TIMEOUT=30.5        # → 30.5 (float)
NONO_HOSTS=["a","b"]     # → ["a", "b"] (list)
```

## Schema Validation

```python
from nono.config import Config, ConfigSchema

# Define schema
schema = ConfigSchema()
schema.add_field('google.default_model', type=str, required=True)
schema.add_field('rate_limits.delay_between_requests', 
                 type=float, min_value=0, max_value=10)
schema.add_field('ollama.host', type=str, required=True)
schema.add_field('app.mode', type=str, choices=['dev', 'prod'])

# Create config with schema
config = Config(schema=schema)
config.load_file('config.toml')

# Validate (raises ValueError on failure)
config.validate()

# Or without exception
is_valid, errors = config.validate(raise_on_error=False)
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### Validation Options

| Parameter | Description |
|-----------|-------------|
| `type` | Expected type (`str`, `int`, `float`, `bool`) |
| `required` | `True` if field is mandatory |
| `default` | Default value for validation |
| `choices` | List of allowed values |
| `min_value` | Minimum value (numeric) |
| `max_value` | Maximum value (numeric) |

## Legacy API (Backward Compatible)

For compatibility with existing code:

```python
from nono.config import (
    NonoConfig,
    get_templates_dir,
    get_prompts_dir,
    set_templates_dir,
    set_prompts_dir,
)

# Get directories
templates = get_templates_dir()
prompts = get_prompts_dir()

# Set directories
set_templates_dir('/path/to/templates')
set_prompts_dir('/path/to/prompts')

# Get values from TOML file
model = NonoConfig.get_config_value('google', 'default_model')

# Load custom file
NonoConfig.load_from_file('/path/to/config.toml')

# Reset (useful for tests)
NonoConfig.reset()
```

### Legacy Environment Variables

| Variable | Description |
|----------|-------------|
| `NONO_TEMPLATES_DIR` | Path to templates directory |
| `NONO_PROMPTS_DIR` | Path to prompts directory |
| `NONO_CONFIG_FILE` | Path to config.toml file |

## Usage Examples

### Basic Configuration

```python
from nono.config import load_config

config = load_config()
model = config['google.default_model']
delay = config.get('rate_limits.delay_between_requests', type=float)
```

### With Required Values

```python
from nono.config import Config

config = Config().load_file('config.toml')

# Raises ValueError if not found
api_key = config.require('api.key', message='API key is required')
```

### Isolation for Tests

```python
from nono.config import Config

def test_feature():
    # Create isolated configuration
    config = Config(defaults={'test.mode': True})
    
    # Or copy an existing one
    original = load_config()
    isolated = original.copy()
    isolated.set('test.override', 'value')
    
    # Original is unaffected
    assert 'test.override' not in original
```

### Source Tracking

```python
from nono.config import Config, ConfigSource

config = Config(defaults={'app.timeout': 30})
config.load_file('config.toml')
config.load_env(prefix='NONO_')

# Know where each value comes from
source = config.get_source('google.default_model')
if source == ConfigSource.ENVIRONMENT:
    print("Value overridden by environment variable")
elif source == ConfigSource.FILE:
    print("Value from configuration file")
```

## API Reference

| Method/Function | Description |
|----------------|-------------|
| `Config()` | Create new instance |
| `load_file(path)` | Load from file |
| `load_env(prefix)` | Load from environment variables |
| `load_args(args)` | Load from arguments |
| `get(key, default, type)` | Get value with fallback |
| `set(key, value)` | Set value (maximum priority) |
| `require(key, message)` | Get required value |
| `all()` | Get all resolved values |
| `validate()` | Validate against schema |
| `copy()` | Create deep copy |
| `get_source(key)` | Get value source |
| `load_config()` | Helper function for quick loading |
| `create_sample_config()` | Create example file |

## Enums

```python
from nono.config import ConfigSource, ConfigFormat

# Configuration sources
ConfigSource.DEFAULT      # Default value
ConfigSource.FILE         # Configuration file
ConfigSource.ENVIRONMENT  # Environment variable
ConfigSource.ARGUMENT     # Programmatic argument

# File formats
ConfigFormat.JSON
ConfigFormat.YAML
ConfigFormat.TOML
```

## CLI Integration

The configuration module integrates with the [CLI module](README_cli.md) to provide a unified experience.

### Configuration Loading in CLI

The CLI can load configuration from:

1. **Configuration file** via `--config-file`
2. **Environment variables** with prefix `NONO_`
3. **Command line arguments** (maximum priority)

```bash
# CLI with external configuration
python -m nono.cli --config-file config.toml --provider gemini --prompt "Hello"
```

### Joint Programmatic Usage

```python
from nono.config import load_config
from nono.cli import CLIBase, print_info

# Load configuration
config = load_config(filepath='config.toml', env_prefix='NONO_')

# Create CLI with config values
cli = CLIBase(
    prog="my_tool",
    version=config.get('app.version', '1.0.0')
)

# CLI arguments have priority over config
args = cli.parse_args()

# Merge: args override config
config.load_args(vars(args))

# Now config has correct priority:
# args > env > file > defaults
model = config['google.default_model']
print_info(f"Using model: {model}")
```

### Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Defaults   │ -> │  TOML File  │ -> │  Env Vars   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            v
                                   ┌─────────────┐
                                   │  CLI Args   │  <- Maximum priority
                                   └─────────────┘
                                            │
                                            v
                                   ┌─────────────┐
                                   │   Config    │  <- Final value
                                   └─────────────┘
```

📖 See also: [CLI Documentation](README_cli.md)

## Author

**DatamanEdge** - MIT License
