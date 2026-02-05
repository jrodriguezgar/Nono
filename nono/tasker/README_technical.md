# GenAI Tasker Module

Advanced module for executing tasks using Generative AI. It provides a unified, extensible, and robust interface for interacting with multiple LLM providers.

---

## Overview

The GenAI Tasker module provides a comprehensive framework for:

- **Multi-Provider Support**: Integrated support for **Google Gemini**, **OpenAI**, **Perplexity**, **DeepSeek**, **Grok (xAI)**, and **Ollama** (local).
- **Task-Based Execution**: Define prompts, schemas, and configurations in reusable JSON files.
- **Advanced Logging**: Decorated traceability system (`@event_log`) combining structured logs with console output.
- **Unified Abstraction**: Base class (`BaseAIClient`) standardizing message preparation and system instruction management.
- **Smart Throttling**: Automatic batch splitting for large data inputs to respect model context windows.
- **Schema Validation**: Optional integration with `jsonschema` for validating structured inputs and outputs.
- **Rate Limiting**: Built-in Token Bucket rate limiter for controlling API request frequency.

---

## Module Structure

| File                                                      | Description                                                          |
| --------------------------------------------------------- | -------------------------------------------------------------------- |
| [genai_tasker.py](genai_tasker.py)                           | Main module with TaskExecutor, AI clients, and task management logic |
| [../connector/connector_genai.py](../connector/connector_genai.py) | Unified connector for multiple AI service providers (shared)   |
| [prompts/](prompts/)                                         | Directory for task definition JSON files                             |

---

## Table of Contents

- [Function Categories](#function-categories)
- [Function Index](#function-index)
- [Detailed Function Documentation](#detailed-function-documentation)
- [Classes](#classes)
- [Enumerations](#enumerations)
- [Credits](#credits)

---

## Function Categories

### Logging & Utilities

- [msg_log()](#msg_log) - Logs a message and prints to console
- [event_log()](#event_log) - Decorator for automatic logging at function entry and exit

### SSL Configuration

- [configure_ssl_verification()](#configure_ssl_verification) - Configures SSL certificate verification mode

### Dependency Management

- [install_library()](#install_library) - Checks and installs required libraries via pip

### Schema Utilities

- [convert_json_schema()](#convert_json_schema) - Converts simplified JSON schema to detailed format

---

## Function Index

| Function                                                 | Module          |
| -------------------------------------------------------- | --------------- |
| [configure_ssl_verification()](#configure_ssl_verification) | connector_genai |
| [convert_json_schema()](#convert_json_schema)               | connector_genai |
| [install_library()](#install_library)                       | connector_genai |
| [msg_log()](#msg_log)                                       | genai_tasker    |
| [event_log()](#event_log)                                   | genai_tasker    |

---

## Detailed Function Documentation

### `msg_log()`

Logs a message using the configured logger and simultaneously prints it to the console with a timestamp.

**Parameters:**

- `message` (str): The message to log and print.
- `level` (int): The logging level (default: `logging.INFO`).

**Returns:**

- `None`

**Example:**

```python
from genai_tasker import msg_log
import logging

msg_log("Task started", logging.INFO)
msg_log("Error occurred", logging.ERROR)
```

**Cost:** O(1)

---

### `event_log()`

Decorator for managing log messages at the beginning and end of function execution. Automatically logs start, completion, and any errors with timestamps (via `msg_log`).

**Parameters:**

- `message` (str, optional): Custom message to display. Defaults to function name.
- `level` (int): The logging level (default: `logging.INFO`).

**Returns:**

- `Callable`: Decorated function wrapper.

**Output Format:**

```
2026-02-03 10:30:00 - [INFO] Starting: Data Processing
2026-02-03 10:30:05 - [INFO] Completed: Data Processing
```

**Example:**

```python
from genai_tasker import event_log

@event_log(message="Data Processing")
def process_data(data):
    # Process data here
    return result

@event_log()  # Uses function name as message
def another_task():
    pass
```

**Cost:** O(1) overhead per function call

---

### `configure_ssl_verification()`

Configures SSL certificate verification for the application. Supports three modes: insecure (development), certifi (production), and custom certificate.

**Parameters:**

- `mode` (SSLVerificationMode): SSL verification mode (INSECURE, CERTIFI, or CUSTOM).
- `custom_cert_path` (str, optional): Path to custom certificate file (required if mode=CUSTOM).

**Returns:**

- `None`

**Raises:**

- `ValueError`: If CUSTOM mode is selected but no certificate path is provided.
- `FileNotFoundError`: If custom certificate file does not exist.

**Example:**

```python
from connector_genai import configure_ssl_verification, SSLVerificationMode

# Option 1: Insecure mode (development only)
configure_ssl_verification(SSLVerificationMode.INSECURE)

# Option 2: Use certifi package (recommended for production)
configure_ssl_verification(SSLVerificationMode.CERTIFI)

# Option 3: Use custom corporate certificate
configure_ssl_verification(
    SSLVerificationMode.CUSTOM, 
    custom_cert_path=r'C:\path\to\corporate-cert.crt'
)
```

**Cost:** O(1)

---

### `install_library()`

Checks if a library is installed and, if not, attempts to install it via pip.

**Parameters:**

- `library_name` (str): The name of the library to install.
- `import_name` (str, optional): The import name if different from library name.

**Returns:**

- `bool`: True if library is available (installed or already present), False otherwise.

**Example:**

```python
from connector_genai import install_library

# Check and install if needed
if install_library("google.genai", package_name="google-genai"):
    from google import genai
else:
    print("Failed to install required library")
```

**Cost:** O(1) if installed, O(n) for installation where n is package size

---

### `convert_json_schema()`

Converts a simplified JSON schema to a more detailed one with proper title attributes for each property.

**Parameters:**

- `input_schema` (dict): The input JSON schema with a 'properties' key.
- `output_title` (str): Title for the output schema (default: "perplexity").

**Returns:**

- `dict`: The converted JSON schema with titles.

**Raises:**

- `ValueError`: If input schema does not contain a 'properties' key.

**Example:**

```python
from connector_genai import convert_json_schema

input_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name"]
}

output_schema = convert_json_schema(input_schema, "person_schema")
```

**Cost:** O(n) where n is the number of properties

---

## Classes

### AIProvider (Enum)

Enumeration of supported AI providers.

| Value          | Description          |
| -------------- | -------------------- |
| `GEMINI`     | Google Gemini models |
| `OPENAI`     | OpenAI GPT models    |
| `PERPLEXITY` | Perplexity AI models |
| `OLLAMA`     | Local Ollama models  |

---

### SSLVerificationMode (Enum)

Enumeration for SSL verification modes.

| Value        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `INSECURE` | Disables SSL verification (development only)                 |
| `CERTIFI`  | Uses certifi package for certificate validation (production) |
| `CUSTOM`   | Uses a custom certificate file path                          |

---

### ResponseFormat (Enum)

Enumeration for response format types.

| Value     | Description           |
| --------- | --------------------- |
| `TEXT`  | Plain text response   |
| `TABLE` | Markdown table format |
| `XML`   | XML format            |
| `JSON`  | JSON format           |
| `CSV`   | CSV format            |

---

### AIConfiguration (Dataclass)

Configuration dataclass for AI service settings.

**Attributes:**

- `provider` (AIProvider): The AI provider to use.
- `model_name` (str): Name of the model.
- `api_key` (str): API key for authentication.
- `temperature` (float): Sampling temperature (default: 0.7).
- `max_tokens` (int): Maximum tokens in response (default: 2048).
- `system_instruction` (str, optional): System prompt instruction.

**Example:**

```python
from genai_tasker import AIConfiguration, AIProvider

config = AIConfiguration(
    provider=AIProvider.GEMINI,
    model_name="gemini-3-flash-preview",
    api_key="your-api-key",
    temperature=0.5,
    max_tokens=4096,
    system_instruction="You are a helpful assistant."
)
```

---

### BaseAIClient (Abstract Class)

Abstract base class for AI service wrappers. Provides unified interface for content generation.

**Methods:**

#### `__init__(config: AIConfiguration)`

Initializes the client with the given configuration.

#### `generate_content(input_data, json_schema, config_overrides) -> str`

Generates content based on the prompt (str) or messages (list[dict]).

**Parameters:**

- `input_data` (Union[str, list[dict]]): String prompt or list of message dictionaries.
- `json_schema` (dict, optional): JSON schema for structured output.
- `config_overrides` (dict, optional): Runtime configuration overrides.

**Returns:**

- `str`: Generated content from the AI model.

---

### Provider Clients

All provider clients inherit from `BaseAIClient` and share the same interface. They differ only in the underlying service connector they use.

| Client | Provider | Notes |
| ------ | -------- | ----- |
| `GeminiClient` | Google Gemini | — |
| `OpenAIClient` | OpenAI | — |
| `PerplexityClient` | Perplexity | — |
| `OllamaClient` | Ollama (local) | No API key required |

**Usage:** All clients are interchangeable thanks to the Liskov Substitution Principle. Select the appropriate client based on your provider:

---

### TaskExecutor

Main executor class to manage AI tasks. Follows S.O.L.I.D principles by separating configuration, usage, and implementation.

**Methods:**

#### `__init__(task_or_config_file, api_key)`

Initializes the executor with optional task file or API key.

**Parameters:**

- `task_or_config_file` (str, optional): Path to task definition JSON file.
- `api_key` (str, optional): API key override.

---

#### `execute(input_data, output_schema, config_overrides) -> str`

Executes the task with the given input.

**Parameters:**

- `input_data` (Union[str, list[dict]]): Input prompt or message list.
- `output_schema` (dict, optional): JSON schema for structured output.
- `config_overrides` (dict, optional): Runtime configuration overrides.

**Returns:**

- `str`: Generated response.

**Example:**

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute("Explain quantum computing in simple terms.")
print(result)
```

**Cost:** O(1) + API call latency

---

#### `run_json_task(task_source, data_input, **data_inputs) -> str`

Executes a task defined in a JSON file with automatic batching support and **multiple data inputs**.

**Parameters:**

- `task_source` (Union[str, Any]): Path to task JSON file or data (if task bound at init).
- `data_input` (Any, optional): Primary input data (replaces `{data_input_json}` placeholder).
- `**data_inputs` (Any): Additional named inputs (replace `{name}` placeholders).

**Returns:**

- `str`: Task execution result.

**Placeholder Mapping:**

| Parameter                | Placeholder in Template |
| ------------------------ | ----------------------- |
| `data_input` (2nd arg) | `{data_input_json}`   |
| `categories=...`       | `{categories}`        |
| `context=...`          | `{context}`           |
| `<any_name>=...`       | `{<any_name>}`        |

**Example (Single Input):**

```python
from genai_tasker import TaskExecutor

# Method 1: Bound task
executor = TaskExecutor("prompts/name_classifier.json")
result = executor.run_json_task(["Alice", "Bob", "ACME Corp"])

# Method 2: Ad-hoc task
executor = TaskExecutor()
result = executor.run_json_task("prompts/name_classifier.json", ["Alice", "Bob"])
```

**Example (Multiple Inputs):**

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor("prompts/product_categorizer.json")

# Primary data -> {data_input_json}
# categories -> {categories}
# rules -> {rules}
result = executor.run_json_task(
    "prompts/product_categorizer.json",
    ["iPhone 15", "Nike Air Max"],              # {data_input_json}
    categories=["Electronics", "Footwear"],      # {categories}
    rules="Prioritize primary function"          # {rules}
)
```

**Cost:** O(n) where n is the number of batches

---

### RateLimiter

Token bucket rate limiter for controlling API request frequency.

**Methods:**

#### `__init__(requests_per_second, burst)`

**Parameters:**

- `requests_per_second` (float): Rate limit in requests per second.
- `burst` (int): Maximum burst capacity (default: 1).

#### `acquire() -> bool`

Attempts to acquire a token for making a request.

**Returns:**

- `bool`: True if token acquired, False otherwise.

#### `wait_for_token()`

Blocks until a token is available.

---

### GenerativeAIService (Abstract Class)

Abstract base class for interacting with generative AI services with rate limiting.

**Class Methods:**

#### `get_recommended_temperature(use_case) -> float`

Returns recommended temperature for a given use case.

**Parameters:**

- `use_case` (str): Use case identifier.

**Returns:**

- `float`: Recommended temperature value.

| Use Case      | Temperature |
| ------------- | ----------- |
| coding        | 0.0         |
| math          | 0.0         |
| data_cleaning | 0.1         |
| data_analysis | 0.3         |
| translation   | 0.3         |
| conversation  | 0.7         |
| creative      | 1.0         |
| poetry        | 1.2         |
| default       | 0.7         |

**Example:**

```python
from connector_genai import GenerativeAIService

temp = GenerativeAIService.get_recommended_temperature("coding")
# Returns 0.0
```

---

#### `get_max_input_chars(model_name) -> int`

Returns the maximum input characters for a given model.

**Parameters:**

- `model_name` (str): Name of the model.

**Returns:**

- `int`: Maximum allowed input characters.

---

### Service Implementations

| Service Class         | Provider   |
| --------------------- | ---------- |
| `GeminiService`     | Google     |
| `OpenAIService`     | OpenAI     |
| `PerplexityService` | Perplexity |
| `DeepSeekService`   | DeepSeek   |
| `GrokService`       | xAI        |
| `OllamaService`     | Local      |

> **Note:** For available models and context limits, consult each provider's official documentation.

---

## Task Definition Schema

Task definitions are JSON files that configure execution parameters.

```json
{
  "task": {
    "name": "task_name",
    "description": "Task description",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-3-flash-preview",
    "temperature": "coding",
    "max_tokens": 4096,
    "batch_size": 0
  },
  "prompts": {
    "system": "System instruction here",
    "user": "Process this data: {data_input_json}",
    "assistant": ""
  },
  "input_schema": {},
  "output_schema": {}
}
```

---

## Usage Examples

### Basic Execution

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute("What is machine learning?")
print(result)
```

### Task-Based Execution

```python
from genai_tasker import TaskExecutor

# Initialize with task file
executor = TaskExecutor("prompts/name_classifier.json")

# Execute with data
data = ["John Doe", "ACME Corp", "Jane Smith"]
result = executor.run_json_task(data)
print(result)
```

### Direct Service Usage

```python
from connector_genai import GeminiService, ResponseFormat

# Initialize service
service = GeminiService(
    model_name="gemini-3-flash-preview",
    api_key="your-api-key"
)

# Generate completion
messages = [{"role": "user", "content": "Hello, how are you?"}]
response = service.generate_completion(
    messages,
    response_format=ResponseFormat.TEXT
)
print(response)
```

### JSON Structured Output

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor("prompts/name_classifier.json")

schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["person", "company"]}
        }
    }
}

result = executor.execute(
    "Classify: John Doe, Microsoft, Alice",
    output_schema=schema
)
```

---

## Requirements

- Python 3.8 or higher
- Third-party libraries (auto-installed per provider):
  - `google-genai` (Gemini)
  - `requests` (OpenAI, Perplexity, Ollama)
  - `jsonschema` (Optional, for input validation)
  - `certifi` (For production SSL)

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `google-genai` | >= 1.0.0 | Google Gemini SDK ([docs](https://ai.google.dev/gemini-api/docs)) |
| `requests` | >= 2.28.0 | HTTP library for API calls |
| `jsonschema` | >= 4.0.0 | JSON schema validation |
| `certifi` | >= 2023.0.0 | SSL certificates |
| `jinja2` | >= 3.0.0 | Template engine for prompts |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
