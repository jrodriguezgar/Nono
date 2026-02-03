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

### Prerequisites

- Python >= 3.8

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/DatamanEdge/Nono.git
   cd Nono/nono
   ```
2. Install dependencies:

   ```bash
   pip install google-genai openai httpx jsonschema certifi
   ```
3. Configure your API key:

   ```bash
   echo "your-api-key" > apikey.txt
   ```

## Quick Start

### Basic Usage

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute("Explain quantum computing in simple terms.")
print(result)
```

### Task-Based Execution

```python
from genai_tasker import TaskExecutor

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
| [Technical Reference](nono/genai_tasker/README_technical.md)                 | API documentation, classes, and functions        |
| [Task Configuration Guide](nono/genai_tasker/README_task_configuration.md)   | JSON task definitions, placeholders, and schemas |
| [SSL Configuration](nono/genai_tasker/connector/connector_genai_ssl.md)      | SSL/TLS setup for corporate environments         |
| [Connector Reference](nono/genai_tasker/connector/README_connector_genai.md) | Low-level service connector documentation        |

## Configuration

### API Keys

Store your API key in one of these files (searched in order):

1. `{provider}_api_key.txt` (e.g., `gemini_api_key.txt`)
2. `google_ai_api_key.txt`
3. `apikey.txt`

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
    "model": "gemini-1.5-flash",
    "temperature": "coding"
  },
  "prompts": {
    "system": "You are a helpful assistant.",
    "user": "Process this data: {data_input_json}"
  }
}
```

See [Task Configuration Guide](nono/genai_tasker/README_task_configuration.md) for complete reference.

### Temperature Presets

| Preset            | Value | Use Case         |
| ----------------- | ----- | ---------------- |
| `coding`        | 0.0   | Code generation  |
| `math`          | 0.0   | Calculations     |
| `data_cleaning` | 0.1   | Data processing  |
| `data_analysis` | 0.3   | Analytics        |
| `translation`   | 0.3   | Translations     |
| `conversation`  | 0.7   | Chatbots         |
| `creative`      | 1.0   | Creative writing |
| `poetry`        | 1.2   | Artistic content |

## Project Structure

```
nono/
├── genai_tasker/
│   ├── genai_tasker.py          # Main module
│   ├── connector/
│   │   └── connector_genai.py   # Provider connectors
│   ├── prompts/                 # Task definition files
│   └── examples/                # Usage examples
├── apikey.txt                   # API key (gitignored)
└── config.toml                  # Configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## Credits

**Author:** DatamanEdge  
**License:** MIT
