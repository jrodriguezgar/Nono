# GenAI Tasker - Task Configuration Guide

> Learn how to create, configure, and execute AI tasks using JSON definition files.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Overview](#overview)
- [Task Structure](#task-structure)
- [Configuration Reference](#configuration-reference)
- [Creating Your First Task](#creating-your-first-task)
- [Advanced Configuration](#advanced-configuration)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Overview

Task definition files are JSON documents that configure how the GenAI Tasker executes AI operations. They allow you to:

- Define reusable prompts with placeholders
- Configure AI provider and model settings
- Validate input/output data with JSON schemas
- Enable automatic batching for large datasets

---

## Task Structure

A task definition file has five main sections:

```json
{
  "task": { },       // Metadata
  "genai": { },      // AI configuration
  "prompts": { },    // System/user/assistant prompts
  "input_schema": { },   // Input validation (optional)
  "output_schema": { }   // Structured output (optional)
}
```

### Minimal Example

```json
{
  "task": {
    "name": "my_task",
    "version": "1.0.0"
  },
  "prompts": {
    "user": "Analyze this data: {data_input_json}"
  }
}
```

---

## Configuration Reference

### Task Metadata

| Field           | Type   | Required | Description                      |
| --------------- | ------ | -------- | -------------------------------- |
| `name`        | string | Yes      | Unique identifier for the task   |
| `description` | string | No       | Human-readable description       |
| `version`     | string | Yes      | Semantic version (e.g., "1.0.0") |

### GenAI Configuration

| Field               | Type          | Required | Default          | Description                                                                            |
| ------------------- | ------------- | -------- | ---------------- | -------------------------------------------------------------------------------------- |
| `provider`        | string        | No       | `gemini`       | AI provider:`gemini`, `openai`, `perplexity`, `ollama`, `deepseek`, `grok` |
| `model`           | string        | No       | Provider default | Model identifier (e.g.,`gemini-1.5-flash`, `gpt-4o`)                               |
| `temperature`     | string/number | No       | `0.7`          | Preset or float value (see[Temperature Presets](#temperature-presets))                    |
| `max_tokens`      | integer       | No       | `2048`         | Maximum tokens to generate                                                             |
| `batch_size`      | integer       | No       | Auto             | Split input lists into batches of this size                                            |
| `response_format` | string        | No       | `json`         | Output format:`text`, `json`, `table`, `csv`, `xml`                          |

#### Temperature Presets

Temperature controls the randomness of the model's output. Lower values produce more deterministic results, while higher values increase creativity and variability.

| Preset            | Value | Best For                               |
| ----------------- | ----- | -------------------------------------- |
| `coding`        | 0.0   | Code generation, deterministic outputs |
| `math`          | 0.0   | Mathematical calculations              |
| `data_cleaning` | 0.1   | Data processing, transformations       |
| `data_analysis` | 0.3   | Analytical tasks, consistency          |
| `translation`   | 0.3   | Precise, faithful translations         |
| `conversation`  | 0.7   | Chatbots, dialogue (balanced)          |
| `creative`      | 1.0   | Creative writing, variability          |
| `poetry`        | 1.2   | Poetry, artistic expression            |
| `default`       | 0.7   | General purpose (balanced)             |

#### Advanced Parameters

| Field                 | Type             | Providers                  | Description                  |
| --------------------- | ---------------- | -------------------------- | ---------------------------- |
| `top_p`             | number (0-1)     | All                        | Nucleus sampling probability |
| `top_k`             | integer          | Gemini, Ollama             | Top-K sampling               |
| `frequency_penalty` | number (-2 to 2) | OpenAI, Perplexity, Ollama | Penalize frequent tokens     |
| `presence_penalty`  | number (-2 to 2) | OpenAI, Perplexity         | Penalize repeated tokens     |
| `stop`              | string[]         | All                        | Stop sequences               |
| `candidate_count`   | integer          | Gemini                     | Number of responses          |
| `seed`              | integer          | Ollama                     | Reproducibility seed         |
| `num_ctx`           | integer          | Ollama                     | Context window size          |

### Prompts Configuration

| Field         | Type   | Required | Description                                                 |
| ------------- | ------ | -------- | ----------------------------------------------------------- |
| `system`    | string | No       | System instruction (defines AI behavior)                    |
| `user`      | string | Yes      | User prompt template with placeholders for data injection |
| `assistant` | string | No       | Pre-filled assistant response for context                   |

---

#### Placeholders

**Placeholders** are markers in the prompt that are automatically replaced with the data you provide when executing the task. They use the `{name}` syntax and allow you to inject dynamic data into your prompts.

##### Overview

Placeholders enable you to create reusable prompt templates that can be populated with different data at runtime. This separation of template and data makes tasks flexible and maintainable.

**Key Concepts:**

1. **Template**: The `user` prompt in your JSON task file contains placeholders like `{data_input_json}`
2. **Data**: When you call `run_json_task()`, you pass the actual data
3. **Substitution**: The executor replaces each placeholder with its corresponding value
4. **Serialization**: Complex types (lists, dicts) are automatically converted to compact JSON strings

##### How Placeholder Substitution Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: TEMPLATE (JSON file)                                               │
│  "user": "Analyze: {data_input_json}"                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 2: PYTHON CALL                                                        │
│  executor.run_json_task("task.json", ["apple", "car", "dog"])               │
│                                        │                                    │
│                                        ▼                                    │
│                          data → {data_input_json}                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 3: FINAL PROMPT (sent to AI)                                          │
│  "Analyze: ["apple","car","dog"]"                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Placeholder Types

There are two ways to pass data to placeholders:

| Placeholder | How to Pass | Description |
|-------------|-------------|-------------|
| `{data_input_json}` | 2nd positional argument | **Main placeholder** - The primary data for the task |
| `{custom_name}` | Keyword argument (`name=value`) | **Additional placeholders** - Secondary data or context |

**Syntax:**

```python
# Main data only
executor.run_json_task("task.json", my_data)
#                                   └── replaces {data_input_json}

# Main data + additional placeholders
executor.run_json_task(
    "task.json",
    my_data,                    # → {data_input_json}
    categories=my_categories,   # → {categories}
    rules=my_rules              # → {rules}
)
```

##### Automatic Type Conversion

Values are automatically serialized based on their Python type:

| Python Type | Conversion | Example |
|-------------|------------|----------|
| `list` | Compact JSON | `["a", "b"]` → `["a","b"]` |
| `dict` | Compact JSON | `{"k": "v"}` → `{"k":"v"}` |
| `str` | Direct text (no quotes) | `"hello"` → `hello` |
| `int`, `float` | String representation | `42` → `42` |
| `bool` | String representation | `True` → `True` |

> **Note:** Lists and dictionaries are serialized as compact JSON (no extra spaces) to minimize token usage.

---

##### Example 1: Single Placeholder (Basic)

**Task definition (`classifier.json`):**

```json
{
  "prompts": {
    "system": "Classify each element.",
    "user": "Classify these elements:\n\n{data_input_json}"
  }
}
```

**Python usage:**

```python
data = ["apple", "car", "dog"]

result = executor.run_json_task("classifier.json", data)
#                                                   │
#                            Inserted here ─────────┘
```

**Generated prompt:**

```
Classify these elements:

["apple","car","dog"]
```

---

##### Example 2: Multiple Placeholders

**Task definition (`categorizer.json`):**

```json
{
  "prompts": {
    "system": "You are an expert categorization system.",
    "user": "Products to categorize:\n{data_input_json}\n\nAvailable categories:\n{categories}\n\nAdditional instructions:\n{instructions}"
  }
}
```

**Python usage:**

```python
products = ["iPhone 15", "Nike Air Max", "MacBook Pro"]
cats = ["Electronics", "Clothing", "Footwear"]
instructions = "Prioritize the main function of the product"

result = executor.run_json_task(
    "categorizer.json",
    products,                    # → {data_input_json}
    categories=cats,             # → {categories}
    instructions=instructions    # → {instructions}
)
```

**Generated prompt:**

```
Products to categorize:
["iPhone 15","Nike Air Max","MacBook Pro"]

Available categories:
["Electronics","Clothing","Footwear"]

Additional instructions:
Prioritize the main function of the product
```

---

##### Example 3: Structured Data as Context

You can pass complex structures as additional context:

```python
# Expected data schema
schema = {
    "fields": ["name", "date", "amount"],
    "date_format": "DD/MM/YYYY",
    "currency": "EUR"
}

# Reference examples
examples = [
    {"input": "invoice 001", "output": {"name": "Invoice", "amount": 100}},
    {"input": "order ABC", "output": {"name": "Order", "amount": 250}}
]

result = executor.run_json_task(
    "extractor.json",
    documents,            # → {data_input_json}
    schema=schema,        # → {schema}
    examples=examples     # → {examples}
)
```

---

##### Rules and Best Practices

| ✅ Do | ❌ Avoid |
|-------|----------|
| Use descriptive names: `{categories}` | Generic names: `{data2}` |
| Document placeholders in `description` | Undocumented placeholders |
| Validate data before sending | Send unvalidated data |
| Use placeholders for variable data | Hardcode values in the prompt |

> **💡 Tip:** Placeholder names are case-sensitive. `{Categories}` ≠ `{categories}`

---

### Schema Validation

Both `input_schema` and `output_schema` use [JSON Schema](https://json-schema.org/) format:

```json
{
  "input_schema": {
    "type": "array",
    "items": { "type": "string" }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "item": { "type": "string" },
            "category": { "type": "string" }
          },
          "required": ["item", "category"]
        }
      }
    },
    "required": ["results"]
  }
}
```

---

## Creating Your First Task

### Step 1: Create the Task File

Create a new file in the `prompts/` directory:

```bash
touch prompts/sentiment_analyzer.json
```

### Step 2: Define the Task

```json
{
  "task": {
    "name": "sentiment_analyzer",
    "description": "Analyzes sentiment of text inputs",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "temperature": "data_analysis"
  },
  "prompts": {
    "system": "You are a sentiment analysis expert. Analyze the sentiment of each text and respond with valid JSON only.",
    "user": "Analyze the sentiment of these texts:\n\n{data_input_json}"
  },
  "input_schema": {
    "type": "array",
    "items": { "type": "string" }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "text": { "type": "string" },
            "sentiment": { 
              "type": "string",
              "enum": ["positive", "negative", "neutral"]
            },
            "confidence": { "type": "number" }
          },
          "required": ["text", "sentiment"]
        }
      }
    }
  }
}
```

### Step 3: Execute the Task

```python
from genai_tasker import TaskExecutor

# Initialize with task file
executor = TaskExecutor("prompts/sentiment_analyzer.json")

# Prepare input data
texts = [
    "I love this product!",
    "Terrible experience, never again.",
    "It's okay, nothing special."
]

# Run the task
result = executor.run_json_task(texts)
print(result)
```

### Expected Output

```json
{
  "results": [
    {"text": "I love this product!", "sentiment": "positive", "confidence": 0.95},
    {"text": "Terrible experience, never again.", "sentiment": "negative", "confidence": 0.92},
    {"text": "It's okay, nothing special.", "sentiment": "neutral", "confidence": 0.78}
  ]
}
```

---

## Advanced Configuration

### Automatic Batching

For large datasets, the executor automatically splits input into batches based on model context limits. You can also set explicit batch sizes:

```json
{
  "genai": {
    "batch_size": 50
  }
}
```

### Provider-Specific Settings

#### Gemini

```json
{
  "genai": {
    "provider": "gemini",
    "model": "gemini-1.5-pro-latest",
    "top_k": 40,
    "candidate_count": 1
  }
}
```

#### OpenAI

```json
{
  "genai": {
    "provider": "openai",
    "model": "gpt-4o",
    "frequency_penalty": 0.5,
    "presence_penalty": 0.3
  }
}
```

#### Ollama (Local)

```json
{
  "genai": {
    "provider": "ollama",
    "model": "llama3",
    "num_ctx": 8192,
    "seed": 42
  }
}
```

### Multi-Turn Conversations

Use the `assistant` field to provide context:

```json
{
  "prompts": {
    "system": "You are a helpful coding assistant.",
    "user": "Review this code: {data_input_json}",
    "assistant": "I'll analyze the code for potential issues and improvements."
  }
}
```

---

## Examples

### Name Classifier

```json
{
  "task": {
    "name": "name_classifier",
    "description": "Identifies person names in a list of strings",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "temperature": 0.1
  },
  "prompts": {
    "system": "You are an entity classifier. Determine if each string is a person's name.",
    "user": "Classify these strings:\n\n{data_input_json}"
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "analyzed_string": { "type": "string" },
            "is_person": { "type": "boolean" }
          },
          "required": ["analyzed_string", "is_person"]
        }
      }
    }
  }
}
```

### Product Categorizer (Multi-Input Example)

This example demonstrates the use of **multiple placeholders** for richer context:

```json
{
  "task": {
    "name": "product_categorizer",
    "description": "Categorizes products using multiple data inputs",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-1.5-flash",
    "temperature": 0.2
  },
  "prompts": {
    "system": "You are a product classification expert. Only use categories from the provided list.",
    "user": "Categorize these products:\n\n{data_input_json}\n\nAvailable categories:\n{categories}\n\nClassification rules:\n{rules}"
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "product": { "type": "string" },
            "category": { "type": "string" },
            "confidence": { "type": "number" }
          },
          "required": ["product", "category"]
        }
      }
    }
  }
}
```

**Python Usage:**

```python
from genai_tasker import TaskExecutor

executor = TaskExecutor("prompts/product_categorizer.json")

products = ["iPhone 15", "Nike Air Max", "MacBook Pro"]
categories = ["Electronics - Phones", "Electronics - Computers", "Footwear"]
rules = "Prioritize primary product function"

result = executor.run_json_task(
    "prompts/product_categorizer.json",
    products,              # -> {data_input_json}
    categories=categories, # -> {categories}
    rules=rules            # -> {rules}
)
```

### Text Summarizer

```json
{
  "task": {
    "name": "summarizer",
    "description": "Summarizes long texts into concise paragraphs",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": "conversation",
    "max_tokens": 500
  },
  "prompts": {
    "system": "You are an expert summarizer. Create concise, informative summaries.",
    "user": "Summarize the following text in 2-3 sentences:\n\n{data_input_json}"
  }
}
```

### Code Reviewer

```json
{
  "task": {
    "name": "code_reviewer",
    "description": "Reviews code for best practices and issues",
    "version": "1.0.0"
  },
  "genai": {
    "provider": "gemini",
    "model": "gemini-1.5-pro-latest",
    "temperature": "coding",
    "response_format": "json"
  },
  "prompts": {
    "system": "You are a senior code reviewer. Analyze code for bugs, security issues, and improvements.",
    "user": "Review this code:\n\n```\n{data_input_json}\n```"
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "issues": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "severity": { "type": "string", "enum": ["critical", "warning", "info"] },
            "line": { "type": "integer" },
            "message": { "type": "string" },
            "suggestion": { "type": "string" }
          }
        }
      },
      "overall_score": { "type": "number" }
    }
  }
}
```

---

## Best Practices

### 1. Use Descriptive Names

```json
{
  "task": {
    "name": "email_intent_classifier_v2",
    "description": "Classifies customer email intents for routing",
    "version": "2.1.0"
  }
}
```

### 2. Choose Appropriate Temperature

- Use `coding` or `math` for deterministic outputs
- Use `conversation` for natural language
- Use numeric values for fine control

### 3. Define Output Schemas

Always define `output_schema` for structured data extraction to ensure consistent, parseable responses.

### 4. Keep Prompts Clear

```json
{
  "prompts": {
    "system": "You are X. Your task is Y. Always respond with Z format.",
    "user": "Process this input: {data_input_json}"
  }
}
```

### 5. Version Your Tasks

Use semantic versioning to track changes:

- `1.0.0` → Initial release
- `1.1.0` → New features (backward compatible)
- `2.0.0` → Breaking changes

### 6. Validate Input

Use `input_schema` to catch invalid data before API calls:

```json
{
  "input_schema": {
    "type": "array",
    "items": { "type": "string" },
    "minItems": 1,
    "maxItems": 100
  }
}
```

---

## Credits

**Author:** DatamanEdge  
**License:** MIT
