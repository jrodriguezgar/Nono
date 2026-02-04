# Data Stage

> Token-efficient batch data operations for GenAI Tasker with TSV format and intelligent throttling.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

## Why TSV Format?

TSV (Tab-Separated Values) is the recommended format for data operations because:

| Benefit                    | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| **70% fewer tokens** | `001\tMadrid` vs `{"key":"001","value":"Madrid"}`     |
| **Faster parsing**   | Simple `split('\t')` vs JSON deserialization            |
| **LLM-friendly**     | Models handle tabular data more reliably than nested JSON |
| **Preserves order**  | Line-by-line correspondence between input and output      |
| **Error resilient**  | Partial responses are still parseable                     |

When the AI returns malformed JSON, the entire response fails. With TSV, you can recover valid lines even if some are corrupted.

## Table of Contents

- [Why TSV Format?](#why-tsv-format)
- [Installation](#installation)
- [Usage](#usage)
- [Operations](#operations)
- [Configuration](#configuration)
- [Creating Custom Operations](#creating-custom-operations)
- [Architecture](#architecture)
- [License](#license)

## Installation

### Prerequisites

- Python >= 3.10
- `genai_tasker` parent module configured with API keys

### Steps

Data Stage is a submodule of `genai_tasker`. No additional installation required if you have the parent module set up.

```python
from nono.genai_tasker.data_stage import DataStageExecutor
```

## Usage

### Quick Start

```python
from nono.genai_tasker.data_stage import DataStageExecutor

executor = DataStageExecutor(provider="gemini")

# Semantic Lookup
result = executor.semantic_lookup(
    data=["Madriz", "Barzelona", "Tokyo"],
    reference_list=["Madrid", "Barcelona", "Sevilla"]
)

# Spell Correction
result = executor.spell_correction(
    data=["Teh quick brwon fox", "Hello world"],
    language="auto"
)
```

### TSV Format

TSV reduces token usage by ~70% compared to JSON:

```
# JSON (verbose)
[{"key":"001","value":"Madrid"}]  # 32 chars

# TSV (efficient)
001	Madrid                         # 11 chars
```

#### I/O Structure

```
# Input
KEY<TAB>VALUE
001	Madrid
002	Barzelona

# Lookup Output
KEY<TAB>ORIGINAL<TAB>MATCHED<TAB>TYPE
001	Madrid	Madrid	exact
002	Barzelona	Barcelona	fuzzy

# Spell Correction Output
KEY<TAB>CORRECTED<TAB>CHANGED
001	Madrid	0
002	Barcelona	1
```

### Supported Input Formats

```python
# Simple list (auto-generated keys)
data = ["Madrid", "Barcelona"]

# Tuples (key, value)
data = [("ES-MAD", "Madrid"), ("ES-BCN", "Barcelona")]

# Dictionaries
data = [{"key": "ES-MAD", "value": "Madrid"}]

# DataRecord objects
data = [DataRecord(key="ES-MAD", value="Madrid")]
```

## Operations

### SemanticLookupOperation

Matches values against a reference list using semantic similarity:

```python
result = executor.semantic_lookup(
    data=["Sevila", "Madriz"],
    reference_list=["Madrid", "Barcelona", "Sevilla"]
)

# Result:
# record.result = {"matched_value": "Sevilla", "match_type": "fuzzy"}
```

### SpellCorrectionOperation

Corrects spelling errors in text:

```python
result = executor.spell_correction(
    data=["Teh quick brwon fox"],
    language="English"
)

# Result:
# record.result = {"corrected_text": "The quick brown fox", "was_corrected": True}
```

## Configuration

| Parameter                 | Description                                                      | Required | Default                    |
| ------------------------- | ---------------------------------------------------------------- | -------- | -------------------------- |
| `provider`              | AI provider (`gemini`, `openai`, `perplexity`, `ollama`) | No       | `gemini`                 |
| `model_name`            | Model to use                                                     | No       | `gemini-3-flash-preview` |
| `api_key`               | API key (auto-loaded from config if not provided)                | No       | Auto                       |
| `temperature`           | Preset string or float value                                     | No       | `data_cleaning`          |
| `max_tokens`            | Maximum output tokens                                            | No       | `4096`                   |
| `context_reserve_ratio` | Context window reserved for response                             | No       | `0.3`                    |

### Temperature Presets

| Preset            | Value | Use Case                                |
| ----------------- | ----- | --------------------------------------- |
| `coding`        | 0.0   | Deterministic code generation           |
| `math`          | 0.0   | Exact mathematical responses            |
| `data_cleaning` | 0.1   | High precision data transformations     |
| `data_analysis` | 0.3   | Consistent analysis with flexibility    |
| `translation`   | 0.3   | Precise translations                    |
| `conversation`  | 0.7   | Balanced coherence and naturalness      |
| `creative`      | 1.0   | Creative content generation             |
| `poetry`        | 1.2   | High creativity for artistic expression |

### DataStageExecutor

```python
# Simple usage with defaults
executor = DataStageExecutor()

# Or with custom configuration
executor = DataStageExecutor(
    provider="gemini",
    model_name="gemini-3-flash-preview",
    api_key=None,                    # Auto-loaded from config.toml
    temperature="data_cleaning",     # Or float like 0.1
    max_tokens=4096,
    context_reserve_ratio=0.3
)
```

### Intelligent Throttling

The executor automatically splits large datasets into batches:

1. Calculates available space based on model's context window
2. Groups records to fill available space
3. Executes each batch sequentially
4. Combines results at the end

```python
# 10,000 records are automatically batched
result = executor.semantic_lookup(
    data=huge_list,
    reference_list=reference,
    progress_callback=lambda c, t: print(f"Batch {c}/{t}")
)
```

## Creating Custom Operations

Custom operations extend the `DataStageOperation` abstract base class. You must implement three components:

### 1. Required Methods

| Method                              | Description                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- |
| `__init__()`                      | Define the prompt template with `{placeholders}` for dynamic content |
| `operation_type`                  | Property returning `OperationType.CUSTOM`                            |
| `build_prompt(batch, context)`    | Build the messages list from batch data and context                    |
| `parse_response(response, batch)` | Parse AI response and populate `record.result` for each record       |

### 2. Prompt Definition Structure

The prompt definition is a dictionary with system and user prompts:

```python
prompt = {
    "prompts": {
        "system": "Instructions for the AI...",
        "user": "Template with {placeholders}..."
    }
}
```

### 3. Complete Example: Sentiment Analysis Operation

```python
from nono.genai_tasker.data_stage import (
    DataStageOperation, DataBatch, DataRecord, OperationType, DataFormat
)
from typing import Any, Dict, List, Optional


class SentimentAnalysisOperation(DataStageOperation):
    """Analyze sentiment of text values."""
  
    def __init__(self, data_format: DataFormat = DataFormat.TSV):
        prompt = {
            "prompts": {
                "system": """You are a sentiment analysis expert.

RULES:
- Return ONLY TSV format: KEY<TAB>SENTIMENT<TAB>CONFIDENCE
- SENTIMENT must be: positive|negative|neutral
- CONFIDENCE is a float 0.0-1.0
- One line per input, same order
- No headers, no explanations""",
                "user": """Input data (TSV: KEY<TAB>TEXT):
{input_data}

Output TSV (KEY<TAB>SENTIMENT<TAB>CONFIDENCE):"""
            }
        }
        super().__init__(prompt, data_format)
  
    @property
    def operation_type(self) -> OperationType:
        return OperationType.CUSTOM
  
    def build_prompt(
        self,
        batch: DataBatch,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build messages for sentiment analysis."""
        prompts = self.prompt_definition.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")
  
        # Convert batch to TSV format
        input_data = batch.to_tsv(include_result=False)
  
        # Replace placeholder
        user_content = user_template.replace("{input_data}", input_data)
  
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
  
        return messages
  
    def parse_response(
        self,
        response: str,
        batch: DataBatch
    ) -> List[DataRecord]:
        """Parse sentiment analysis response."""
        results = []
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
  
        # Map keys to records for lookup
        key_map = {str(r.key): r for r in batch.records}
  
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
    
            key = parts[0]
            sentiment = parts[1] if len(parts) > 1 else "neutral"
            confidence = float(parts[2]) if len(parts) > 2 else 0.5
    
            if key in key_map:
                record = key_map[key]
                record.result = {
                    "sentiment": sentiment.lower(),
                    "confidence": confidence
                }
                results.append(record)
  
        # Handle missing records
        for key, record in key_map.items():
            if record not in results:
                record.result = {"sentiment": "unknown", "confidence": 0.0}
                results.append(record)
  
        return results


# Usage
executor = DataStageExecutor()
result = executor.execute(
    operation=SentimentAnalysisOperation(),
    data=["I love this product!", "Terrible experience", "It's okay I guess"]
)

for record in result.records:
    print(f"{record.value}: {record.result['sentiment']} ({record.result['confidence']:.0%})")
```

### 4. Using Context Parameters

For operations that need external data (like reference lists), use the `context` parameter:

```python
def build_prompt(self, batch: DataBatch, context: Optional[Dict[str, Any]] = None):
    if not context or 'categories' not in context:
        raise ValueError("This operation requires 'categories' in context")
  
    categories = '\n'.join(context['categories'])
    # ... use categories in prompt
```

Then pass context when executing:

```python
result = executor.execute(
    operation=MyCategoryOperation(),
    data=items,
    context={"categories": ["Electronics", "Clothing", "Food"]}
)
```

## Architecture

```
nono/genai_tasker/
├── genai_tasker.py          # TaskExecutor, AIConfiguration, Clients
└── data_stage/              # Extension submodule
    ├── __init__.py          # Public exports
    ├── core.py              # DataStageExecutor, DataRecord, DataBatch
    ├── operations.py        # SemanticLookup, SpellCorrection
    ├── README.md
    └── examples/
        ├── semantic_lookup_example.py
        ├── spell_correction_example.py
        ├── custom_operation_example.py
        └── keyed_data_example.py
```

**Composition over duplication**: `DataStageExecutor` uses `TaskExecutor` internally for AI client management.

### Core Classes

| Class                  | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `DataStageExecutor`  | Main executor with batch processing and throttling |
| `DataRecord`         | Single data record with key, value, and result     |
| `DataBatch`          | Collection of records for batch processing         |
| `DataStageResult`    | Execution result with records and statistics       |
| `DataStageOperation` | Abstract base class for custom operations          |

### DataRecord

```python
record = DataRecord(key="001", value="Madrid", result=None)
record.to_tsv_input()   # "001\tMadrid"
record.to_tsv_output()  # "001\tMadrid\t<result>"
```

### DataStageResult

```python
result.success              # True/False
result.records              # List[DataRecord]
result.batches_processed    # int
result.total_tokens_used    # int
result.to_tsv()             # Export TSV
result.get_results_as_dict() # {key: result}
```

## Credits

**Author:** DatamanEdge
**License:** MIT
