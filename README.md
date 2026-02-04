# Nono

> Unified AI-powered framework for executing tasks, operations, and applications using Generative AI as the engine.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-orange)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Modules](#modules)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

**Nono** is a modular Python framework that leverages Large Language Models (LLMs) to perform a variety of tasks—from simple text generation to complex data transformations and code execution. It provides a unified interface across multiple AI providers, allowing you to switch between them seamlessly.

## Features

| Feature                               | Description                                                     |
| ------------------------------------- | --------------------------------------------------------------- |
| **Multi-Provider Support**      | Google Gemini, OpenAI, Perplexity, DeepSeek, Grok (xAI), Ollama |
| **Task-Based Execution**        | Define reusable prompts and schemas in JSON files               |
| **Code Generation & Execution** | Generate and run Python code from natural language              |
| **Batch Data Operations**       | Token-efficient processing with TSV format (70% savings)        |
| **Intelligent Throttling**      | Automatic batching based on model context windows               |
| **Structured Outputs**          | JSON schema validation for consistent responses                 |
| **Rate Limiting**               | Built-in Token Bucket for API request control                   |
| **SSL Flexibility**             | Configurable SSL verification for corporate environments        |

## Supported Providers

| Provider                | Models                                                   | Context Window   |
| ----------------------- | -------------------------------------------------------- | ---------------- |
| **Google Gemini** | gemini-3-flash-preview, gemini-2.0-flash, gemini-1.5-pro | Up to 4M chars   |
| **OpenAI**        | gpt-4o, gpt-4o-mini, gpt-4-turbo                         | Up to 500K chars |
| **Perplexity**    | sonar, llama-3 variants                                  | Up to 120K chars |
| **DeepSeek**      | deepseek-chat, deepseek-coder                            | Up to 120K chars |
| **Grok (xAI)**    | grok-1                                                   | Up to 100K chars |
| **Ollama**        | Any local model                                          | Varies           |

## Installation

### Prerequisites

- Python >= 3.10
- pip (package manager)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/DatamanEdge/Nono.git
   cd Nono
   ```
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:

   ```bash
   pip install google-genai openai httpx jsonschema certifi keyring
   ```

4. Configure your API key (choose one method):

   **Option A: OS Keyring (Recommended - Most Secure)**
   ```python
   import keyring
   keyring.set_password("gemini", "api_key", "your-api-key")
   ```

   **Option B: Key File**
   ```bash
   echo "your-api-key" > nono/apikey.txt
   ```

## Modules

Nono consists of three main modules:

### 🔧 Connector

Low-level unified interface for AI providers with rate limiting and SSL management.

```python
from nono.connector import GeminiService

client = GeminiService(model_name="gemini-3-flash-preview")
response = client.generate_completion(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

📖 [Connector Documentation](nono/connector/README_connector_genai.md)

---

### 📋 GenAI Tasker

Task-based execution framework with JSON prompt definitions and structured outputs.

```python
from nono.genai_tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute_task(
    task_name="name_classifier",
    input_data={"name": "María García"}
)
```

📖 [GenAI Tasker Documentation](nono/genai_tasker/README.md)

---

### ⚡ GenAI Executer

Generate and execute Python code from natural language with security controls.

```python
from nono.genai_executer import CodeExecuter

executer = CodeExecuter()
result = executer.run("Calculate the factorial of 10")
print(result.output)  # 3628800
```

📖 [GenAI Executer Documentation](nono/genai_executer/README.md)

---

### 📊 Data Stage

Token-efficient batch operations for data processing (semantic lookup, spell correction).

```python
from nono.genai_tasker.data_stage import DataStageExecutor

executor = DataStageExecutor()
result = executor.semantic_lookup(
    data=["Madriz", "Barzelona"],
    reference_list=["Madrid", "Barcelona", "Sevilla"]
)
```

📖 [Data Stage Documentation](nono/genai_tasker/data_stage/README.md)

## Quick Start

### Basic Text Generation

```python
from nono.genai_tasker import TaskExecutor

executor = TaskExecutor()
response = executor.execute("Explain quantum computing in simple terms.")
print(response)
```

### Task-Based Execution

```python
from nono.genai_tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute_task(
    task_name="product_categorizer",
    input_data={"product": "iPhone 15 Pro Max 256GB"}
)
print(result)  # {"category": "Electronics", "subcategory": "Smartphones"}
```

### Code Generation

```python
from nono.genai_executer import CodeExecuter

executer = CodeExecuter()
result = executer.run(
    instruction="List all Python files in the current directory"
)
print(result.output)
```

### Batch Data Processing

```python
from nono.genai_tasker.data_stage import DataStageExecutor

executor = DataStageExecutor()

# Spell correction
result = executor.spell_correction(
    data=["Teh quick brwon fox"],
    language="English"
)
# Result: "The quick brown fox"
```

## Configuration

### API Keys

API keys are resolved in this order:

| Priority | Method | Description |
|----------|--------|-------------|
| 1st | Argument | `TaskExecutor(api_key="...")` |
| 2nd | OS Keyring | Secure credential store (auto-installed) |
| 3rd | Key Files | `{provider}_api_key.txt` or `apikey.txt` |

**Recommended: Use OS Keyring**

```python
import keyring

# Store keys (one-time setup)
keyring.set_password("gemini", "api_key", "your-gemini-key")
keyring.set_password("openai", "api_key", "your-openai-key")
keyring.set_password("perplexity", "api_key", "your-perplexity-key")
```

> **Note**: If a key is found in a file, it's automatically migrated to keyring for future use.

**Alternative: Key Files**

```bash
# Provider-specific file
echo "your-key" > nono/gemini_api_key.txt

# Or generic file (used if provider-specific not found)
echo "your-key" > nono/apikey.txt
```

### SSL Configuration

For corporate environments with custom certificates:

```python
from nono.connector import configure_ssl_verification, SSLVerificationMode

# Use certifi (default)
configure_ssl_verification(SSLVerificationMode.CERTIFI)

# Custom certificate
configure_ssl_verification(SSLVerificationMode.CUSTOM, custom_cert_path="path/to/cert.crt")

# Insecure (development only)
configure_ssl_verification(SSLVerificationMode.INSECURE)
```

## Project Structure

```
Nono/
├── LICENSE
├── README.md
└── nono/
    ├── __init__.py
    ├── apikey.txt              # API key storage
    ├── config.toml             # Provider configuration
    ├── connector/              # Low-level AI connectors
    │   ├── connector_genai.py
    │   └── README_connector_genai.md
    ├── genai_tasker/           # Task execution framework
    │   ├── genai_tasker.py
    │   ├── README.md
    │   ├── prompts/            # JSON task definitions
    │   │   ├── name_classifier.json
    │   │   └── product_categorizer.json
    │   ├── data_stage/         # Batch data operations
    │   │   ├── core.py
    │   │   ├── operations.py
    │   │   ├── README.md
    │   │   └── examples/
    │   └── examples/
    └── genai_executer/         # Code generation & execution
        ├── genai_executer.py
        ├── config.json
        ├── README.md
        └── examples/
```

## Documentation

| Document                                                          | Description                     |
| ----------------------------------------------------------------- | ------------------------------- |
| [Connector Guide](nono/connector/README_connector_genai.md)          | Low-level AI provider interface |
| [GenAI Tasker](nono/genai_tasker/README.md)                          | Task-based execution framework  |
| [Task Configuration](nono/genai_tasker/README_task_configuration.md) | JSON prompt definition guide    |
| [Technical Reference](nono/genai_tasker/README_technical.md)         | Architecture and internals      |
| [GenAI Executer](nono/genai_executer/README.md)                      | Code generation and execution   |
| [Data Stage](nono/genai_tasker/data_stage/README.md)                 | Batch data operations           |

## Credits

**Author:** DatamanEdge
**License:** MIT
