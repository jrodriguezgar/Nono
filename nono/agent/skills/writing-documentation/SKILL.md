---
name: writing-documentation
description: >
  Generate clear, structured documentation for code, APIs, and projects.
  Use when the user asks to document code, write a README, create
  docstrings, or produce technical documentation.
license: MIT
metadata:
  author: DatamanEdge
  version: "1.0.0"
tags:
  - code
  - documentation
  - writing
version: "1.0.0"
temperature: 0.3
---

# Write Documentation

Generate clear, comprehensive technical documentation for code,
modules, APIs, and projects.

## Documentation types

| Type | When to use |
|---|---|
| **README** | Project overview, setup, usage |
| **Docstrings** | Functions, classes, modules (Google format) |
| **API docs** | Endpoint reference, parameters, responses |
| **Architecture** | System design, component diagrams |
| **Guide** | Step-by-step tutorials, how-tos |
| **CHANGELOG** | Version history, breaking changes |

## Guidelines

1. **Audience**: Write for the target audience — developers, end users,
   or operators. Ask yourself: "What does the reader need to know?"
2. **Structure**: Use clear headings, ordered from general → specific.
3. **Examples**: Always include at least one usage example.
4. **Accuracy**: Document only what exists. Don't invent features.
5. **Conciseness**: Be thorough but not verbose. Prefer bullet points
   and tables over long paragraphs.
6. **Code language**: English for all documentation and code identifiers.

## README template

```markdown
# Project Name

> One-line description.

## Overview

Brief description of what the project does and why it exists.

## Installation

\```bash
pip install project-name
\```

## Quick Start

\```python
from project import main_feature
result = main_feature("input")
\```

## Features

- Feature 1: Description
- Feature 2: Description

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `param1` | `str` | `"default"` | What it controls |

## Contributing

Guidelines for contribution.

## License

MIT
```

## Docstring format (Google style)

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """Brief one-line summary of function purpose.

    Longer description if needed, explaining context, edge cases,
    and important behavior.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        Dictionary with keys 'result' and 'count'.

    Raises:
        ValueError: If param1 is empty.

    Example:
        >>> result = function_name("test", param2=5)
        >>> print(result["count"])
        5
    """
```
