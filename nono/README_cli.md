# CLI Module - Nono

## Description

Command-line interface for Nono GenAI Tasker with multi-provider support, colored output, progress bars, subcommands, and unified argument parsing.

## Installation

The module is included in the project. No additional dependencies required.

For color support on Windows without native ANSI:
```bash
pip install colorama
```

## Quick Start

### Command Line Usage

```bash
# Show help
python -m nono.cli --help

# Execute with direct prompt
python -m nono.cli --provider google --prompt "Explain what Python is"

# Execute with task file
python -m nono.cli --provider openai --task summarize --input document.txt

# Use local Ollama
python -m nono.cli --provider ollama --model llama3 --prompt "Hello"

# Load parameters from file
python -m nono.cli @params.txt

# Dry-run mode
python -m nono.cli --dry-run --provider google --prompt "Test"

# CI mode (no colors, quiet, JSON output)
python -m nono.cli --ci --provider google --prompt "Test"
```

### Usage as Python Module

```python
from nono.cli import create_cli, print_success, print_error

# Create CLI with factory function
cli = create_cli(
    prog="my_tool",
    description="My GenAI tool",
    version="1.0.0",
    with_provider=True,
    with_task=True,
    with_io=True
)

# Add usage examples
cli.add_examples([
    "%(prog)s --provider gemini --prompt 'Hello' -o output.txt",
    "%(prog)s @params.txt"
])

# Parse arguments
args = cli.parse_args()

# Your logic here...
cli.increment_stat('processed', 100)
print_success("Completed!")
cli.print_final_summary()
```

## Command Line

### Global Options

| Option | Description |
|--------|-------------|
| `--version`, `-V` | Show version |
| `--verbose`, `-v` | Increase verbosity (-v=INFO, -vv=DEBUG) |
| `--quiet`, `-q` | Suppress non-essential output |
| `--no-color` | Disable colors |
| `--ci` | CI mode: implies `--no-color --quiet --output-format json` |
| `--dry-run` | Simulate without making API calls |
| `--output-format`, `-F` | Output format: table, json, csv, text, markdown, summary, quiet |
| `--config-file` | Load configuration from TOML/JSON file |
| `--log-file` | Write logs to file |

Exit codes: `0` success, `1` runtime error, `2` usage/argument error, `130` interrupted

### AI Provider Configuration

| Option | Description |
|--------|-------------|
| `--provider`, `-p` | Provider: google, openai, perplexity, deepseek, grok, groq, cerebras, nvidia, openrouter, foundry, vercel, ollama |
| `--model`, `-m` | Model name |
| `--api-key` | API key (or use environment variable) |
| `--api-key-file` | Read API key from file |
| `--temperature` | Generation temperature 0.0-2.0 (default: 0.7) |
| `--max-tokens` | Maximum tokens in response (default: 4096) |
| `--timeout` | Request timeout in seconds (default: 60) |
| `--ollama-host` | Ollama server URL (default: http://localhost:11434) |

### Task Configuration

| Option | Description |
|--------|-------------|
| `--task`, `-t` | Task name or path to JSON file |
| `--prompt` | Direct prompt text |
| `--system-prompt` | System prompt/instructions |
| `--template` | Jinja2 template file |
| `--variables`, `--vars` | JSON string or file with template variables |

### Input/Output

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input file or data |
| `--input-format` | Input format: text, json, csv, file |
| `--output`, `-o` | Output file (stdout if not specified) |
| `--output-type` | Output format: text, json, csv, markdown |
| `--append` | Append to file instead of overwriting |

### Batch Processing

| Option | Description |
|--------|-------------|
| `--batch` | Enable batch processing mode |
| `--batch-file` | File with inputs (one per line or JSON array) |
| `--batch-size` | Number of items per batch (default: 10) |
| `--delay` | Delay between requests in seconds (default: 0.5) |
| `--retry` | Number of retries on failure (default: 3) |
| `--continue-on-error` | Continue processing on individual failures |

## Parameter File (@params.txt)

You can load arguments from a file using `@filename`:

```
--provider
gemini
--model
gemini-3-flash-preview
--prompt
Generate a summary of the following text
--input
document.txt
--output
summary.txt
--verbose
```

## Output Utilities

### Colored Messages

```python
from nono.cli import print_success, print_error, print_warning, print_info, cprint, Colors

print_success("Operation completed")  # ✓ Green
print_error("Something failed")              # ✗ Red
print_warning("Careful")               # ⚠ Yellow
print_info("Information")              # ℹ Cyan

# Custom
cprint("Bold cyan text", Colors.CYAN, bold=True)
cprint("Dim text", Colors.GRAY, dim=True)
```

### Tables

```python
from nono.cli import print_table

headers = ["Provider", "Model", "Status"]
rows = [
    ["Gemini", "gemini-3-flash", "Active"],
    ["OpenAI", "gpt-4o-mini", "Active"],
    ["Ollama", "llama3", "Offline"],
]
print_table(headers, rows)

# With index column
print_table(headers, rows, show_index=True)
```

### Progress Bar

```python
from nono.cli import print_progress

for i in range(101):
    print_progress(i, 100, prefix="Processing", suffix="Complete")
```

### Spinner

```python
from nono.cli import print_spinner
import time

update = print_spinner("Loading data...")
for _ in range(50):
    update()
    time.sleep(0.1)
```

### Statistics Summary

```python
from nono.cli import print_summary

stats = {
    'total_processed': 100,
    'successful': 95,
    'errors': 5,
    'tokens_used': 15000,
}
print_summary(stats, title="RESULTS")
```

### Interactive Confirmation

```python
from nono.cli import confirm_action

if confirm_action("Do you want to continue?", default=False):
    print("Continuing...")
else:
    print("Cancelled")
```

## API Reference

### CLIBase Class

```python
from nono.cli import CLIBase

cli = CLIBase(
    prog="my_cli",           # Program name
    description="My CLI",    # Description for help
    version="1.0.0"          # Version
)
```

| Method | Description |
|--------|-------------|
| `add_ai_provider_group()` | Add AI provider args |
| `add_task_group()` | Add task configuration args |
| `add_io_group(formats)` | Add input/output args |
| `add_batch_group()` | Add batch processing args |
| `add_group(name, title)` | Add custom argument group |
| `add_example(example)` | Add usage example |
| `add_examples(list)` | Add multiple examples |
| `init_subcommands(title, dest)` | Initialize subcommand support |
| `add_subcommand(name, help, handler, aliases)` | Add a subcommand |
| `set_handler(command, handler)` | Set/update subcommand handler |
| `run()` | Execute the parsed subcommand's handler |
| `parse_args()` | Parse command line |
| `increment_stat(name, value)` | Increment statistic |
| `set_stat(name, value)` | Set statistic |
| `get_elapsed_time()` | Get formatted elapsed time |
| `print_final_summary(title)` | Print final summary |
| `exit_success(message)` | Exit with code 0 |
| `exit_with_error(message)` | Exit with error |
| `last_result` | Attribute — set by handlers for `run_api()` |

### Factory Function

```python
from nono.cli import create_cli

cli = create_cli(
    prog="tool",
    description="My tool",
    version="1.0.0",
    with_provider=True,   # Add AI provider group
    with_task=True,       # Add task group
    with_io=True,         # Add I/O group
    with_batch=False      # No batch processing
)
```

### Enums

```python
from nono.cli import OutputFormat, AIProvider, LogLevel

# Output formats
OutputFormat.TABLE
OutputFormat.JSON
OutputFormat.CSV
OutputFormat.TEXT
OutputFormat.MARKDOWN
OutputFormat.SUMMARY
OutputFormat.QUIET

# AI providers
AIProvider.GOOGLE
AIProvider.OPENAI
AIProvider.PERPLEXITY
AIProvider.DEEPSEEK
AIProvider.GROK
AIProvider.GROQ
AIProvider.CEREBRAS
AIProvider.NVIDIA
AIProvider.OPENROUTER
AIProvider.FOUNDRY
AIProvider.VERCEL
AIProvider.OLLAMA

# Log levels
LogLevel.DEBUG
LogLevel.INFO
LogLevel.WARNING
LogLevel.ERROR
LogLevel.QUIET
```

### Colors Class

```python
from nono.cli import Colors

# ANSI colors
Colors.RED
Colors.GREEN
Colors.YELLOW
Colors.BLUE
Colors.CYAN
Colors.MAGENTA
Colors.WHITE
Colors.GRAY

# Semantic colors
Colors.SUCCESS  # Green
Colors.ERROR    # Red
Colors.WARNING  # Yellow
Colors.INFO     # Cyan

# Control
Colors.BOLD
Colors.DIM
Colors.RESET

# Methods
Colors.disable()  # Disable colors
Colors.enable()   # Re-enable colors
Colors.init()     # Initialize (automatic)
```

## Complete Examples

### Simple CLI for Translation

```python
from nono.cli import create_cli, print_success, print_info

def main():
    cli = create_cli(
        prog="translator",
        description="Translate text using AI",
        version="1.0.0"
    )
    
    # Add custom argument
    lang_group = cli.add_group("language", "Languages")
    lang_group.add_argument('--source-lang', default='auto', help="Source language")
    lang_group.add_argument('--target-lang', required=True, help="Target language")
    
    cli.add_examples([
        "%(prog)s --provider gemini --prompt 'Hello world' --target-lang es",
    ])
    
    args = cli.parse_args()
    
    print_info(f"Translating from {args.source_lang} to {args.target_lang}")
    
    # Translation logic here...
    
    cli.increment_stat('translated', 1)
    print_success("Translation completed!")
    cli.print_final_summary()

if __name__ == "__main__":
    main()
```

### CLI with Batch Processing

```python
from nono.cli import create_cli, print_progress, print_summary
import time

def main():
    cli = create_cli(
        prog="batch_processor",
        description="Process multiple files with AI",
        with_batch=True
    )
    
    args = cli.parse_args()
    
    if args.batch and args.batch_file:
        with open(args.batch_file) as f:
            items = f.read().splitlines()
        
        total = len(items)
        for i, item in enumerate(items):
            print_progress(i + 1, total, prefix="Processing")
            
            # Process item...
            time.sleep(args.delay)
            
            cli.increment_stat('processed', 1)
        
        cli.print_final_summary()

if __name__ == "__main__":
    main()
```

## Customization

### Disable Colors

```python
from nono.cli import Colors

# Disable globally
Colors.disable()

# Or use --no-color argument on command line
```

### Change Default Colors

```python
from nono.cli import Colors

Colors.SUCCESS = Colors.BLUE  # Change success to blue
Colors.ERROR = Colors.MAGENTA  # Change error to magenta
```

### Custom Argument Group

```python
from nono.cli import CLIBase

cli = CLIBase(prog="custom", version="1.0.0")

# Add custom group
db_group = cli.add_group("database", "Database Connection")
db_group.add_argument('--db-host', required=True, help="DB host")
db_group.add_argument('--db-port', type=int, default=5432, help="Port")
db_group.add_argument('--db-name', required=True, help="Database name")
db_group.add_argument('--db-user', help="User")
db_group.add_argument('--db-password', help="Password")
```

## Environment Variables

The CLI supports API keys via environment variables:

| Provider | Variable |
|-----------|----------|
| Google | `GOOGLE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Grok | `XAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Foundry | `GITHUB_TOKEN` |
| Vercel | Uses underlying provider key |

You can also use `--api-key` or `--api-key-file` to specify the key.

## Programmatic API (run_api)

Use `run_api()` to invoke CLI logic from Python code, REST endpoints, or test harnesses without spawning a subprocess:

```python
from nono.cli import run_api, CLIResult

result: CLIResult = run_api(["--provider", "google", "--prompt", "Hello world"])

if result.ok:
    print(result.data)    # Handler payload
    print(result.stats)   # {"tasks_executed": 1}
else:
    print(result.error)   # Error message
```

### CLIResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `ok` | `bool` | `True` when the command succeeded |
| `data` | `Any` | Payload set by the handler via `cli.last_result` |
| `stats` | `Dict[str, int]` | Counters from `cli.increment_stat()` |
| `error` | `Optional[str]` | Error message when `ok` is `False` |

### Module-Level Functions

| Function | Description |
|----------|-------------|
| `run_api(argv) -> CLIResult` | Programmatic entry point — run a command without subprocess |
| `main(argv) -> int` | CLI entry point — returns exit code (0/1/2/130) |
| `create_cli(prog, ...) -> CLIBase` | Factory to create configured CLI instance |

## Subcommands

The CLI supports subcommands for organizing related operations:

```python
from nono.cli import CLIBase, print_success, print_info

def run_analyze(args, cli):
    print_info(f"Analyzing {args.input}")
    cli.last_result = {"analyzed": args.input}
    cli.increment_stat('analyzed', 1)
    cli.print_final_summary()

cli = CLIBase(prog="nono", description="GenAI Tasker", version="0.2.0")
cli.init_subcommands()

analyze = cli.add_subcommand("analyze", "Analyze content", handler=run_analyze, aliases=["a"])
analyze.add_argument('--input', '-i', required=True)

args = cli.parse_args()
cli.run()
```

```bash
python -m nono.cli analyze -i document.txt
python -m nono.cli a -i document.txt  # using alias
```

## CI Mode

The `--ci` flag activates all CI-friendly defaults in a single switch:

```bash
python -m nono.cli --ci --provider google --prompt "Test"
echo $?  # 0=success, 1=error, 2=usage
```

`--ci` is equivalent to `--no-color --quiet --output-format json`. Colors are also
auto-disabled when stdout is not a TTY (piped output, CI runners).

## Config Integration

The CLI integrates with the [configuration module](README_config.md) for centralized settings management.

### Automatic Configuration Loading

```python
from nono.config import load_config
from nono.cli import CLIBase

# 1. Load base configuration from file
config = load_config(filepath='config.toml', env_prefix='NONO_')

# 2. Create CLI
cli = CLIBase(prog="my_app", version="1.0.0")
cli.add_ai_provider_group()
args = cli.parse_args()

# 3. Merge CLI arguments into config (maximum priority)
config.load_args(vars(args))

# 4. Use final values (CLI > Env > File > Defaults)
provider = config.get('provider', 'gemini')
model = config.get('model') or config.get('google.default_model')
```

### Default Values from Config

You can use configuration to set CLI defaults:

```python
from nono.config import load_config
from nono.cli import CLIConfig, CLIBase

# Read defaults from config.toml
config = load_config()

# Create CLI with config defaults
cli_config = CLIConfig(
    prog_name="nono",
    default_timeout=config.get('rate_limits.timeout', 60),
)

cli = CLIBase(config=cli_config)
```

### Complete Example

```python
#!/usr/bin/env python
from nono.config import load_config, ConfigSchema
from nono.cli import CLIBase, print_success, print_error

def main():
    # Configuration with validation
    schema = ConfigSchema()
    schema.add_field('google.default_model', required=True)
    
    config = load_config(filepath='config.toml', env_prefix='NONO_')
    
    # CLI
    cli = CLIBase(prog="app", version="1.0.0")
    cli.add_ai_provider_group()
    args = cli.parse_args()
    
    # Merge (CLI has priority)
    config.load_args(vars(args))
    
    # Validate final configuration
    try:
        config.validate()
    except ValueError as e:
        cli.exit_with_error(str(e))
    
    # Use configuration
    model = config['google.default_model']
    print_success(f"Using model: {model}")

if __name__ == "__main__":
    main()
```

### Priority Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VALUE PRIORITY                        │
├──────────────────────────────────────────────────────────────┤
│  CLI Arguments (--provider, --model)      <- Maximum     │
│  Environment Variables (NONO_*)                          │
│  Config File (config.toml)                               │
│  Default Values                           <- Minimum     │
└──────────────────────────────────────────────────────────────┘
```

📖 See also: [Configuration Documentation](README_config.md)

## Author

**DatamanEdge** - MIT License
