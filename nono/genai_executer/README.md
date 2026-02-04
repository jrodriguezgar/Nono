# GenAI Executer

Module for generating and executing Python code from natural language instructions using LLMs.

## Description

**GenAI Executer** extends the capabilities of `genai_tasker` by adding functionality to:

1. Generate Python code from natural language instructions
2. Execute the generated code in a controlled, secure environment
3. Capture and return execution results
4. Save execution history (code and results) to the `executions/` directory

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

The module uses the same infrastructure as `genai_tasker`. Ensure dependencies are installed:

```bash
pip install google-genai openai httpx keyring
```

## Configuration

### API Keys

API keys are resolved in this order:

| Priority | Method | Description |
|----------|--------|-------------|
| 1st | Argument | `CodeExecuter(api_key="...")` |
| 2nd | OS Keyring | `keyring.get_password(provider, "api_key")` |
| 3rd | Key Files | `{provider}_api_key.txt` or `apikey.txt` |

**Recommended: Use OS Keyring (Most Secure)**

```python
import keyring

# Store API key (one-time setup)
keyring.set_password("gemini", "api_key", "your-api-key")
```

> **Auto-Migration**: Keys found in files are automatically saved to keyring for future use.

### Using config.json (Recommended)

Create or edit `config.json` in the `genai_executer/` directory:

```json
{
  "genai": {
    "provider": "gemini",
    "model": "gemini-3-flash-preview",
    "temperature": 0.1
  },
  "execution": {
    "mode": "subprocess",
    "security": "safe",
    "timeout": 30,
    "max_retries": 2,
    "save_executions": true
  }
}
```

The config file is loaded automatically. You can also pass parameters directly (they take priority over config file):

```python
# Uses config.json automatically
executer = CodeExecuter()

# Override specific parameters
executer = CodeExecuter(provider="openai", model_name="gpt-4o")

# Use a custom config file
executer = CodeExecuter(config_file="path/to/my_config.json")
```

## Basic Usage

### Simple Example (SAFE mode - default)

```python
from genai_executer import CodeExecuter, SecurityMode

# Initialize the executor (uses config.json by default)
executer = CodeExecuter()

# Or pass parameters directly (override config)
executer = CodeExecuter(
    provider="gemini",
    model_name="gemini-2.0-flash"
)

# Generate and execute code
result = executer.run(
    instruction="Calculate the factorial of 10 and display the result"
)

print(f"Execution ID: {result.execution_id}")
print(f"Success: {result.success}")
print(f"Output: {result.output}")
print(f"Generated code:\n{result.code}")
```

### Using PERMISSIVE Mode

```python
from genai_executer import CodeExecuter, SecurityMode

# Initialize with PERMISSIVE mode (allows dangerous operations)
executer = CodeExecuter(
    provider="gemini",
    model_name="gemini-2.0-flash",
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
genai_executer/
├── __init__.py              # Module exports
├── genai_executer.py        # Main CodeExecuter class
├── README.md                # This documentation
├── executions/              # Saved execution history
│   └── YYYYMMDD_HHMMSS_execid_OK.json  # Execution data + code
└── examples/                # Usage examples
    ├── calculator_example.py
    └── file_lister_example.py
```

## Main Classes

### `CodeExecuter`

Main class for code generation and execution.

**Initialization Parameters:**

- `config_file`: Path to configuration file (TOML or JSON)
- `provider`: AI provider ("gemini", "openai", "perplexity", "ollama")
- `model_name`: Model name to use
- `api_key`: API key for the provider
- `execution_mode`: Execution method (`SUBPROCESS` or `EXEC`)
- `security_mode`: Security level (`SAFE` or `PERMISSIVE`)

**Main Methods:**

- `run(instruction, context, timeout, save_execution)`: Generate and execute code
- `generate_code(instruction, context)`: Generate code only
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
from genai_executer import CodeExecuter, ExecutionMode

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


## Credits

**Author:** DatamanEdge
**License:** MIT
