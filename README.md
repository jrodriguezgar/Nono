# Nono - No Overhead, Neural Operations

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
- [Contact](#contact)
- [Dependencies](#dependencies)
- [License](#license)

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

### 📋 Tasker

Task-based execution framework with JSON prompt definitions and structured outputs.

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute_task(
    task_name="name_classifier",
    input_data={"name": "María García"}
)
```

📖 [Tasker Documentation](nono/tasker/README.md)

---

### ⚡ Executer

Generate and execute Python code from natural language with security controls.

```python
from nono.executer import CodeExecuter

executer = CodeExecuter()
result = executer.run("Calculate the factorial of 10")
print(result.output)  # 3628800
```

📖 [Executer Documentation](nono/executer/README.md)

---

### 📊 Data Operations (Templates)

Token-efficient batch operations using Jinja2 templates.

**Available Templates:**

| Template                    | Description                            |
| --------------------------- | -------------------------------------- |
| `semantic_lookup.j2`      | Fuzzy matching against reference lists |
| `spell_correction.j2`     | Correct misspellings in text           |
| `data_loss_prevention.j2` | Anonymize PII (GDPR, HIPAA, CCPA)      |
| `planner.j2`              | Generate structured project plans      |
| `decompose_tasks.j2`      | Break tasks into subtasks              |
| `logical_ordering.j2`     | Order items by dependencies            |
| `conditional_flow.j2`     | Route decisions based on conditions    |

```python
from nono.tasker import build_from_file_blocks

# Build prompt from template with system/user blocks
prompts = build_from_file_blocks(
    "data_loss_prevention.j2",
    text="John Smith, email: john@email.com, SSN: 123-45-6789",
    compliance="GDPR"
)

# Execute with AI
executor = TaskExecutor()
response = executor.execute(prompts["user"], system_prompt=prompts["system"])
```

📖 See [templates/](nono/tasker/templates/) for available templates

## Quick Start

### Basic Text Generation

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor()
response = executor.execute("Explain quantum computing in simple terms.")
print(response)
```

### Task-Based Execution

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor()
result = executor.execute_task(
    task_name="product_categorizer",
    input_data={"product": "iPhone 15 Pro Max 256GB"}
)
print(result)  # {"category": "Electronics", "subcategory": "Smartphones"}
```

### Code Generation

```python
from nono.executer import CodeExecuter

executer = CodeExecuter()
result = executer.run(
    instruction="List all Python files in the current directory"
)
print(result.output)
```

### Batch Data Processing (with Templates)

```python
from nono.tasker import TaskExecutor, build_from_file

# Build spell correction prompt
prompts = build_from_file(
    template_name="spell_correction",
    data=["Teh quick brwon fox"],
    language="English"
)

# Execute
executor = TaskExecutor()
response = executor.execute(prompts[0])
# Response contains corrected text in TSV format
```

## Input/Output Examples

Esta sección muestra ejemplos concretos de datos de entrada y las respuestas JSON esperadas para cada template.

### Data Loss Prevention (DLP)

Anonimización de datos personales para cumplimiento normativo.

**Entrada:**
```python
text = """
Customer: John Smith
Email: john.smith@company.com
Phone: +1 (555) 123-4567
SSN: 123-45-6789
Credit Card: 4532-1234-5678-9012
"""
compliance = "GDPR"
```

**Salida JSON:**
```json
{
  "anonymized_text": "Customer: [PERSON_1]\nEmail: [EMAIL_1]\nPhone: [PHONE_1]\nSSN: [SSN_1]\nCredit Card: [CREDIT_CARD_1]",
  "entities_found": [
    {"type": "PERSON", "original": "John Smith", "replacement": "[PERSON_1]"},
    {"type": "EMAIL", "original": "john.smith@company.com", "replacement": "[EMAIL_1]"},
    {"type": "PHONE", "original": "+1 (555) 123-4567", "replacement": "[PHONE_1]"},
    {"type": "SSN", "original": "123-45-6789", "replacement": "[SSN_1]"},
    {"type": "CREDIT_CARD", "original": "4532-1234-5678-9012", "replacement": "[CREDIT_CARD_1]"}
  ],
  "compliance_standard": "GDPR",
  "risk_level": "high",
  "recommendations": ["Store mapping table securely", "Implement data retention policy"]
}
```

---

### Conditional Flow (Decision Routing)

Enrutamiento de decisiones basado en condiciones.

**Entrada:**
```python
input_text = "My order #12345 hasn't arrived after 2 weeks and I want a refund!"
routes = ["billing", "technical_support", "shipping", "general_inquiry", "escalate_to_manager"]
context = "E-commerce customer support system"
```

**Salida JSON:**
```json
{
  "selected_route": "shipping",
  "confidence": 0.85,
  "reasoning": "Customer mentions order delivery delay, which is a shipping-related issue",
  "alternative_routes": [
    {"route": "escalate_to_manager", "confidence": 0.6, "reason": "Refund request may require escalation"}
  ],
  "extracted_entities": {
    "order_id": "12345",
    "issue_type": "delayed_delivery",
    "sentiment": "frustrated"
  }
}
```

---

### Planner (Project Planning)

Generación de planes de proyecto estructurados.

**Entrada:**
```python
goal = "Migrate legacy monolithic application to microservices architecture"
constraints = ["6 month timeline", "Budget: $500K", "Team of 5 developers", "Zero downtime requirement"]
```

**Salida JSON:**
```json
{
  "project_name": "Monolith to Microservices Migration",
  "total_duration": "6 months",
  "phases": [
    {
      "name": "Analysis & Planning",
      "duration": "4 weeks",
      "tasks": [
        "Document current architecture",
        "Identify service boundaries",
        "Define API contracts"
      ],
      "deliverables": ["Architecture document", "Service decomposition diagram"],
      "resources": ["2 senior developers", "1 architect"]
    },
    {
      "name": "Infrastructure Setup",
      "duration": "3 weeks",
      "tasks": [
        "Set up Kubernetes cluster",
        "Configure CI/CD pipelines",
        "Implement service mesh"
      ],
      "deliverables": ["Production-ready K8s environment", "Deployment automation"],
      "resources": ["1 DevOps engineer", "1 developer"]
    }
  ],
  "milestones": [
    {"name": "Architecture approved", "date": "Week 4"},
    {"name": "First service deployed", "date": "Week 10"},
    {"name": "Full migration complete", "date": "Week 24"}
  ],
  "risks": [
    {"risk": "Data consistency during migration", "mitigation": "Implement saga pattern"},
    {"risk": "Performance degradation", "mitigation": "Load testing before each release"}
  ]
}
```

---

### Decompose Tasks (Task Breakdown)

Descomposición de tareas complejas en subtareas.

**Entrada:**
```python
task = "Implement user authentication system with OAuth2 and MFA"
granularity = "detailed"
```

**Salida JSON:**
```json
{
  "original_task": "Implement user authentication system with OAuth2 and MFA",
  "subtasks": [
    {
      "id": 1,
      "title": "Design authentication architecture",
      "description": "Create technical design document for auth flow",
      "estimated_hours": 8,
      "priority": "high",
      "dependencies": []
    },
    {
      "id": 2,
      "title": "Set up OAuth2 provider integration",
      "description": "Integrate with Google, GitHub, Microsoft OAuth2",
      "estimated_hours": 16,
      "priority": "high",
      "dependencies": [1]
    },
    {
      "id": 3,
      "title": "Implement JWT token management",
      "description": "Create token generation, validation, and refresh logic",
      "estimated_hours": 12,
      "priority": "high",
      "dependencies": [1]
    },
    {
      "id": 4,
      "title": "Add MFA support",
      "description": "Implement TOTP-based two-factor authentication",
      "estimated_hours": 16,
      "priority": "medium",
      "dependencies": [2, 3]
    },
    {
      "id": 5,
      "title": "Create user session management",
      "description": "Handle login, logout, session timeout",
      "estimated_hours": 8,
      "priority": "medium",
      "dependencies": [3]
    }
  ],
  "total_estimated_hours": 60,
  "critical_path": [1, 2, 4],
  "parallel_opportunities": [[2, 3], [4, 5]]
}
```

---

### Logical Ordering (Dependency Ordering)

Ordenación de elementos según dependencias lógicas.

**Entrada:**
```python
items = [
    "Deploy to production",
    "Write unit tests",
    "Code review",
    "Implement feature",
    "Create PR",
    "QA testing",
    "Merge to main"
]
context = "Software development workflow"
```

**Salida JSON:**
```json
{
  "ordered_items": [
    {"position": 1, "item": "Implement feature", "reason": "Core work must be done first"},
    {"position": 2, "item": "Write unit tests", "reason": "Tests validate the implementation"},
    {"position": 3, "item": "Create PR", "reason": "Code ready for review"},
    {"position": 4, "item": "Code review", "reason": "Peer review before merge"},
    {"position": 5, "item": "Merge to main", "reason": "Approved code integrated"},
    {"position": 6, "item": "QA testing", "reason": "Integration testing on main branch"},
    {"position": 7, "item": "Deploy to production", "reason": "Final step after all validations"}
  ],
  "dependency_graph": {
    "Implement feature": [],
    "Write unit tests": ["Implement feature"],
    "Create PR": ["Write unit tests"],
    "Code review": ["Create PR"],
    "Merge to main": ["Code review"],
    "QA testing": ["Merge to main"],
    "Deploy to production": ["QA testing"]
  },
  "parallel_groups": [
    ["Write unit tests"]
  ]
}
```

---

### Semantic Lookup (Fuzzy Matching)

Búsqueda semántica con coincidencia aproximada.

**Entrada:**
```python
query = "iphone 15 pro"
candidates = [
    "Apple iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Apple iPhone 14 Pro",
    "Google Pixel 8 Pro",
    "Apple iPhone 15 128GB"
]
threshold = 0.7
```

**Salida JSON:**
```json
{
  "query": "iphone 15 pro",
  "matches": [
    {"candidate": "Apple iPhone 15 Pro Max 256GB", "score": 0.95, "reason": "Exact model match with variant"},
    {"candidate": "Apple iPhone 15 128GB", "score": 0.82, "reason": "Same generation, different variant"},
    {"candidate": "Apple iPhone 14 Pro", "score": 0.75, "reason": "Same product line, previous generation"}
  ],
  "best_match": "Apple iPhone 15 Pro Max 256GB",
  "no_match_candidates": [
    {"candidate": "Samsung Galaxy S24 Ultra", "score": 0.25},
    {"candidate": "Google Pixel 8 Pro", "score": 0.30}
  ]
}
```

---

### Spell Correction

Corrección ortográfica con formato TSV.

**Entrada:**
```python
data = ["Teh quick brwon fox", "Programing is awsome", "Recieve teh mesage"]
language = "English"
```

**Salida TSV:**
```
Original	Corrected	Changes
Teh quick brwon fox	The quick brown fox	Teh→The, brwon→brown
Programing is awsome	Programming is awesome	Programing→Programming, awsome→awesome
Recieve teh mesage	Receive the message	Recieve→Receive, teh→the, mesage→message
```

---

### Product Categorizer

Categorización de productos.

**Entrada:**
```python
product = "Sony WH-1000XM5 Wireless Noise Canceling Headphones"
```

**Salida JSON:**
```json
{
  "product": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
  "category": "Electronics",
  "subcategory": "Audio",
  "type": "Headphones",
  "attributes": {
    "brand": "Sony",
    "model": "WH-1000XM5",
    "connectivity": "Wireless",
    "features": ["Noise Canceling", "Bluetooth"]
  },
  "confidence": 0.98
}
```

---

### Name Classifier

Clasificación de nombres propios.

**Entrada:**
```python
names = ["María García", "Tokyo", "Microsoft", "Amazon River", "Albert Einstein"]
```

**Salida JSON:**
```json
{
  "classifications": [
    {"name": "María García", "type": "PERSON", "subtype": "individual", "confidence": 0.99},
    {"name": "Tokyo", "type": "LOCATION", "subtype": "city", "confidence": 0.98},
    {"name": "Microsoft", "type": "ORGANIZATION", "subtype": "company", "confidence": 0.99},
    {"name": "Amazon River", "type": "LOCATION", "subtype": "geographical_feature", "confidence": 0.95},
    {"name": "Albert Einstein", "type": "PERSON", "subtype": "historical_figure", "confidence": 0.99}
  ],
  "summary": {
    "PERSON": 2,
    "LOCATION": 2,
    "ORGANIZATION": 1
  }
}
```

## Configuration

### Paths Configuration

Configure custom directories for templates and prompts. Useful when using Nono as a library from other projects.

#### Resolution Priority

| Priority | Method | Description |
|----------|--------|-------------|
| 1st | Environment variables | `NONO_TEMPLATES_DIR`, `NONO_PROMPTS_DIR` |
| 2nd | Programmatic | `set_templates_dir()`, `set_prompts_dir()` |
| 3rd | config.toml | `[paths]` section in configuration file |
| 4th | Default | `nono/tasker/templates`, `nono/tasker/prompts` |

#### Environment Variables

```bash
# Set custom directories (highest priority)
export NONO_TEMPLATES_DIR="/path/to/my/templates"
export NONO_PROMPTS_DIR="/path/to/my/prompts"
export NONO_CONFIG_FILE="/path/to/custom/config.toml"
```

#### Programmatic Configuration

```python
from nono.config import NonoConfig, set_templates_dir, set_prompts_dir

# Set directories programmatically
set_templates_dir("/path/to/my/templates")
set_prompts_dir("/path/to/my/prompts")

# Or use the class directly
NonoConfig.set_templates_dir("/path/to/my/templates")
NonoConfig.set_prompts_dir("/path/to/my/prompts")

# Load a custom config file
NonoConfig.load_from_file("/path/to/my/config.toml")

# Get current configuration
config = NonoConfig.get_all_config()
print(config)
```

#### config.toml Configuration

```toml
[paths]
# Relative paths are resolved from project root
templates_dir = "my_templates"
prompts_dir = "my_prompts"

# Or use absolute paths
# templates_dir = "/absolute/path/to/templates"
# prompts_dir = "/absolute/path/to/prompts"
```

### API Keys

API keys are resolved in this order:

| Priority | Method     | Description                                  |
| -------- | ---------- | -------------------------------------------- |
| 1st      | Argument   | `TaskExecutor(api_key="...")`              |
| 2nd      | OS Keyring | Secure credential store (auto-installed)     |
| 3rd      | Key Files  | `{provider}_api_key.txt` or `apikey.txt` |

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
├── main.py                     # CLI entry point
└── nono/
    ├── __init__.py
    ├── config.py               # Central configuration module
    ├── config.toml             # Provider and paths configuration
    ├── connector/              # Low-level AI connectors
    │   ├── connector_genai.py
    │   └── README_connector_genai.md
    ├── tasker/                 # Task execution framework
    │   ├── genai_tasker.py
    │   ├── jinja_prompt_builder.py
    │   ├── prompts/            # JSON task definitions
    │   └── templates/          # Jinja2 templates
    │       ├── conditional_flow.j2      # Decision routing
    │       ├── data_loss_prevention.j2  # PII anonymization (DLP)
    │       ├── decompose_tasks.j2       # Task breakdown
    │       ├── logical_ordering.j2      # Dependency ordering
    │       ├── name_classifier.j2       # Name classification
    │       ├── planner.j2               # Project planning
    │       ├── product_categorizer.j2   # Product categorization
    │       ├── semantic_lookup.j2       # Fuzzy matching
    │       └── spell_correction.j2      # Spell checking
    ├── executer/               # Code generation & execution
    │   ├── genai_executer.py
    │   └── config.json
    └── examples/               # Usage examples
```

## Documentation

| Document                                                    | Description                     |
| ----------------------------------------------------------- | ------------------------------- |
| [Configuration](nono/config.py)                                | Central configuration module    |
| [Connector Guide](nono/connector/README_connector_genai.md)    | Low-level AI provider interface |
| [Tasker](nono/tasker/README.md)                                | Task-based execution framework  |
| [Task Configuration](nono/tasker/README_task_configuration.md) | JSON prompt definition guide    |
| [Technical Reference](nono/tasker/README_technical.md)         | Architecture and internals      |
| [Executer](nono/executer/README.md)                            | Code generation and execution   |

---

## Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| `google-genai` | >= 1.0.0 | Google Gemini SDK ([docs](https://ai.google.dev/gemini-api/docs)) |
| `openai` | >= 1.0.0 | OpenAI SDK |
| `requests` | >= 2.28.0 | HTTP library for API calls |
| `certifi` | >= 2023.0.0 | SSL certificates for secure connections |
| `jsonschema` | >= 4.0.0 | JSON schema validation |
| `jinja2` | >= 3.0.0 | Template engine for prompts |

---

## Contact

- **Author**: [DatamanEdge](https://github.com/DatamanEdge)
- **Email**: [jrodriguezga@outlook.com](mailto:jrodriguezga@outlook.com)
- **LinkedIn**: [Javier Rodríguez](https://es.linkedin.com/in/javier-rodriguez-ga)

---

## License

MIT © 2026 DatamanEdge. See [LICENSE](../../LICENSE).