# Executer

Module for generating and executing Python code from natural language instructions using LLMs.

## Description

**Executer** wraps `TaskExecutor` from the `tasker` module to provide:

1. Generate Python code from natural language instructions using J2 templates
2. Execute the generated code in a controlled, secure environment
3. Capture and return execution results
4. Save execution history (code and results) to the `executions/` directory

## Architecture

`CodeExecuter` is a high-level wrapper that integrates:

- **TaskExecutor** (from `tasker`): Handles AI provider connections and content generation
- **TaskPromptBuilder** (from `tasker`): Builds prompts using Jinja2 templates
- **code_executer.j2** template: Defines system and user prompts for code generation

This design follows the single responsibility principle - AI operations are delegated to `tasker`, while `executer` focuses on code execution and security.

## Security Modes

The module includes two security modes to control what operations are allowed:

### 🔒 SAFE Mode (Default)

Blocks potentially dangerous operations including:

- **File modifications**: write, delete, move operations
- **Network operations**: requests, sockets, HTTP clients
- **System commands**: os.system, subprocess calls
- **Dynamic code execution**: eval, exec, compile
- **Module manipulation**: __import__, importlib
- **Process manipulation**: fork, kill, multiprocessing

### ⚠️ PERMISSIVE Mode

Allows all operations without restrictions. **Use with caution** - only enable when you fully trust the generated code.

## Installation

The module uses the same infrastructure as `tasker`. Ensure dependencies are installed:

```bash
pip install google-genai openai httpx keyring
```

## Configuration

For API keys configuration, see [main README Configuration section](../../README.md#configuration).

### Using config.json (Execution Settings Only)

Create or edit `config.json` in the `executer/` directory for execution-related settings:

```json
{
  "execution": {
    "mode": "subprocess",
    "security": "safe",
    "timeout": 30,
    "max_retries": 2,
    "save_executions": true
  }
}
```

**Note**: AI settings (provider, model, temperature) are passed as constructor parameters, not in config.json. This allows flexibility to change AI provider at runtime while keeping execution behavior consistent.

### Configuration Parameters

| Category | Parameter | Description | Default |
|----------|-----------|-------------|---------|
| **AI** | `provider` | AI provider (gemini, openai, groq, etc.) | gemini |
| **AI** | `model_name` | Model to use | gemini-3-flash-preview |
| **AI** | `api_key` | API key (resolved via tasker if None) | None |
| **AI** | `temperature` | Sampling temperature | 0.2 |
| **Execution** | `mode` | subprocess or exec | subprocess |
| **Execution** | `security` | safe or permissive | safe |
| **Execution** | `timeout` | Max execution time (seconds) | 30 |
| **Execution** | `max_retries` | Retry attempts on failure | 2 |
| **Execution** | `save_executions` | Save to executions/ | true |

## Basic Usage

### Simple Example (SAFE mode - default)

```python
from nono.executer import CodeExecuter, SecurityMode

# Initialize with AI provider settings
executer = CodeExecuter(
    provider="gemini",
    model_name="gemini-3-flash-preview"
)

# Or use defaults (gemini, gemini-3-flash-preview)
executer = CodeExecuter()

# Generate and execute code
result = executer.run(
    instruction="Calculate the factorial of 10 and display the result"
)

print(f"Execution ID: {result.execution_id}")
print(f"Success: {result.success}")
print(f"Output: {result.output}")
print(f"Generated code:\n{result.code}")
```
```

### Using PERMISSIVE Mode

```python
from nono.executer import CodeExecuter, SecurityMode

# Initialize with PERMISSIVE mode (allows dangerous operations)
executer = CodeExecuter(
    provider="gemini",
    model_name="gemini-3-flash-preview",
    security_mode=SecurityMode.PERMISSIVE
)

# Or change mode at runtime
executer.set_security_mode(SecurityMode.PERMISSIVE)

# Now file operations and network requests are allowed
result = executer.run(
    instruction="List all files in the current directory and save to output.txt"
)
```

### Execution History

All executions are automatically saved to the `executions/` directory:

```python
# List recent executions
executions = executer.list_executions(limit=10)
for exec_data in executions:
    print(f"{exec_data['execution_id']}: {exec_data['success']}")

# Get a specific execution
execution = executer.get_execution("abc12345")

# Clear all execution history
executer.clear_executions()
```

## Module Structure

```
executer/
├── __init__.py              # Module exports
├── genai_executer.py        # Main CodeExecuter class
├── config.json              # Execution settings only
├── README.md                # This documentation
└── executions/              # Saved execution history
    └── YYYYMMDD_HHMMSS_execid_OK.json  # Execution data + code
```

**Related files in tasker/:**
```
tasker/templates/
└── python_programming.j2    # J2 template for code generation prompts
```

## Main Classes

### `CodeExecuter`

Main class for code generation and execution. Wraps `TaskExecutor` for AI operations.

**Initialization Parameters:**

- `config_file`: Path to execution config file (JSON)
- `provider`: AI provider ("gemini", "openai", "groq", "openrouter", etc.) - default: "gemini"
- `model_name`: Model name to use - default: "gemini-3-flash-preview"
- `api_key`: API key (if None, resolved via tasker/connector)
- `temperature`: Sampling temperature for code generation - default: 0.2
- `execution_mode`: Execution method (`SUBPROCESS` or `EXEC`)
- `security_mode`: Security level (`SAFE` or `PERMISSIVE`)

**Key Attributes:**

- `tasker`: `TaskExecutor` instance for AI operations
- `prompt_builder`: `TaskPromptBuilder` for J2 template rendering

**Main Methods:**

- `run(instruction, context, timeout, save_execution)`: Generate and execute code
- `generate_code(instruction, context, previous_error)`: Generate code only (uses J2 template)
- `execute_code(code, timeout)`: Execute code only
- `set_security_mode(mode)`: Change security mode at runtime
- `list_executions(limit)`: List saved executions
- `get_execution(execution_id)`: Get a specific execution
- `clear_executions()`: Delete all saved executions

### `ExecutionResult`

Dataclass containing execution results.

**Attributes:**

- `success`: bool - Whether execution was successful
- `output`: str - Standard output from the code
- `error`: str - Error message if failed
- `code`: str - Generated Python code
- `execution_time`: float - Execution time in seconds
- `execution_id`: str - Unique identifier for the execution
- `instruction`: str - Original instruction
- `timestamp`: str - ISO format timestamp

### `SecurityMode`

Enum for security levels.

**Values:**

- `SAFE`: Default. Blocks dangerous operations.
- `PERMISSIVE`: Allows all operations (use with caution).

## Execution Modes

### Subprocess (default)

- Executes code in a separate process
- More secure and isolated
- Captures complete stdout/stderr

### Exec

- Executes with `exec()` in the same process
- Faster but less isolated
- Useful for trusted code

```python
from nono.executer import CodeExecuter, ExecutionMode

# Subprocess mode (recommended)
executer = CodeExecuter(execution_mode=ExecutionMode.SUBPROCESS)

# Exec mode (faster)
executer = CodeExecuter(execution_mode=ExecutionMode.EXEC)
```

## Execution Saving

Each execution is saved to the `executions/` directory as a single JSON file after all retries are completed:

**File format**: `YYYYMMDD_HHMMSS_microseconds_OK.json` or `_FAIL.json`

The JSON file contains:
- `execution_id`: Unique identifier
- `timestamp`: ISO format timestamp
- `instruction`: Original instruction
- `code`: Generated Python code
- `success`: Execution status (boolean)
- `output`: Standard output from execution
- `error`: Error message if failed
- `execution_time`: Duration in seconds
- `provider`, `model`, `security_mode`, `execution_mode`

### Execution Log API

Use these methods to retrieve execution history programmatically:

```python
from genai_executer import CodeExecuter

executer = CodeExecuter()

# List recent executions (returns list of dicts)
executions = executer.list_executions(limit=10)
for exec_data in executions:
    print(f"ID: {exec_data['execution_id']}")
    print(f"Status: {'OK' if exec_data['success'] else 'FAIL'}")
    print(f"Instruction: {exec_data['instruction']}")
    print(f"Code: {exec_data['code'][:100]}...")
    print(f"Output: {exec_data['output']}")
    print(f"Error: {exec_data['error']}")
    print("---")

# Get a specific execution by ID
execution = executer.get_execution("20260203_190708_059131")
if execution:
    print(f"Found: {execution['instruction']}")
    print(f"Generated code:\n{execution['code']}")

# List only successful executions
successful = executer.list_executions(limit=5, success_only=True)

# Clear old executions, keep last 10
executer.clear_executions(keep_recent=10)

# Clear all executions
executer.clear_executions()
```

### ExecutionResult Object

The `run()` method returns an `ExecutionResult` object with these attributes:

```python
result = executer.run(instruction="Calculate 2+2")

result.execution_id   # str: Unique ID (e.g., "20260203_190708_059131")
result.instruction    # str: Original instruction
result.success        # bool: True if execution succeeded
result.code           # str: Generated Python code
result.output         # str: Standard output from execution
result.error          # str: Error message if failed
result.execution_time # float: Duration in seconds
result.timestamp      # str: ISO format timestamp
```

```python
# Disable saving for a specific execution
result = executer.run(instruction="...", save_execution=False)

# List all saved executions
for exec_data in executer.list_executions():
    print(f"{exec_data['execution_id']}: {exec_data['instruction'][:50]}")
```

## Included Examples

### Arithmetic Calculator

```bash
python examples/calculator_example.py
```

Demonstrates:

- Simple calculations
- Complex mathematical expressions
- Using the math library
- Multiple operations

### File Lister

```bash
python examples/file_lister_example.py
```

Demonstrates:

- Listing files in a directory
- Recursive search
- Filtering by extension
- Directory statistics

## Error Handling

The module includes automatic retries:

```python
result = executer.run(
    instruction="...",
    retry_on_error=True,  # Retry if execution fails
    max_retries=2         # Maximum 2 retries
)
```

If generated code fails, the LLM receives the error and generates a corrected version.

## Security Considerations

⚠️ **Warning**: This module executes AI-generated code. Consider:

1. **Always use SAFE mode** unless absolutely necessary
2. **Set appropriate timeouts** to prevent infinite loops
3. **Review generated code** before using PERMISSIVE mode
4. **Don't use in production** without proper sandboxing
5. In SAFE mode, file read operations are allowed but writes are blocked

### Blocked Operations in SAFE Mode

| Category     | Blocked Patterns                                  |
| ------------ | ------------------------------------------------- |
| File Write   | `open(..., 'w')`, `write()`, `writelines()` |
| File Delete  | `os.remove`, `os.unlink`, `shutil.rmtree`   |
| Network      | `requests.*`, `urllib.*`, `socket.*`        |
| System       | `os.system`, `subprocess.*`                   |
| Dynamic Code | `eval()`, `exec()`, `compile()`             |
| Process      | `os.fork`, `multiprocessing.*`                |

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `google-genai` | >= 1.0.0 | Google Gemini SDK ([docs](https://ai.google.dev/gemini-api/docs)) |
| `requests` | >= 2.28.0 | HTTP library for API calls |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
