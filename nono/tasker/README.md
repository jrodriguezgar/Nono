# GenAI Tasker

> Unified framework for executing AI tasks across multiple LLM providers with automatic batching and structured outputs.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Features](#features)
- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Multi-Provider Support**: Google Gemini, OpenAI, Perplexity, DeepSeek, Grok (xAI), and Ollama (local)
- **Task-Based Execution**: Define reusable prompts and schemas in JSON files
- **Automatic Batching**: Smart throttling based on model context windows
- **Structured Outputs**: JSON schema validation for consistent responses
- **Rate Limiting**: Built-in Token Bucket for API request control
- **Flexible Logging**: Decorated traceability with `@event_log`

## Supported Providers

- **Google Gemini**
- **OpenAI**
- **Perplexity**
- **DeepSeek**
- **Grok (xAI)**
- **Ollama** (local models)

## Installation

See [main README Installation section](../../README.md#installation).

## Quick Start

### Basic Usage

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute("Explain quantum computing in simple terms.")
print(result)
```

### Task-Based Execution

```python
from nono.tasker import TaskExecutor

# Initialize with a task definition file
executor = TaskExecutor("prompts/name_classifier.json")

# Execute with data
data = ["John Doe", "ACME Corp", "Jane Smith"]
result = executor.run_json_task(data)
print(result)
```

### Multiple Data Inputs

```python
executor = TaskExecutor("prompts/product_categorizer.json")

result = executor.run_json_task(
    "prompts/product_categorizer.json",
    ["iPhone 15", "Nike Air Max"],     # → {data_input_json}
    categories=["Electronics", "Footwear"],  # → {categories}
    rules="Prioritize primary function"      # → {rules}
)
```

## Documentation

| Document                                                                  | Description                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| [Technical Reference](README_technical.md)                                   | API documentation, classes, and functions        |
| [Task Configuration Guide](README_task_configuration.md)                     | JSON task definitions, placeholders, and schemas |
| [SSL Configuration](../connector/connector_genai_ssl.md)                     | SSL/TLS setup for corporate environments         |
| [Connector Reference](../connector/README_connector_genai.md)                | Low-level service connector documentation        |

## Configuration

For API keys and SSL configuration, see [main README Configuration section](../../README.md#configuration).

### Task Definition Files

Create JSON files in the `prompts/` directory:

```json
{
  "task": {
    "name": "my_task",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-3-flash-preview",
    "temperature": "coding"
  },
  "prompts": {
    "system": "You are a helpful assistant.",
    "user": "Process this data: {data_input_json}"
  }
}
```

See [Task Configuration Guide](README_task_configuration.md) for complete reference.

## Project Structure

See [main README](../../README.md#project-structure) for full project structure.

## Configuration

For API keys, SSL configuration, and other settings, see [main README Configuration section](../../README.md#configuration).

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).
