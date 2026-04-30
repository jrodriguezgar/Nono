# Skills

> Reusable, composable AI capabilities — define once, use everywhere. Compatible with the [Anthropic Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) standard.

## Table of Contents

- [Overview](#overview)
- [Anthropic Agent Skills Standard](#anthropic-agent-skills-standard)
- [Architecture](#architecture)
- [Skill Formats](#skill-formats)
  - [Python Skills (Programmatic)](#python-skills-programmatic)
  - [SKILL.md Files (Anthropic Standard)](#skillmd-files-anthropic-standard)
- [Built-in Skills](#built-in-skills)
- [Usage: Python Code](#usage-python-code)
  - [Standalone Execution](#standalone-execution)
  - [Attach to an Agent](#attach-to-an-agent)
  - [Load from SKILL.md](#load-from-skillmd)
  - [Scan a Directory](#scan-a-directory)
  - [Compose in Workflows](#compose-in-workflows)
- [Usage: CLI](#usage-cli)
- [Usage: REST API](#usage-rest-api)
- [Creating Custom Skills](#creating-custom-skills)
  - [Option A — SKILL.md File (Recommended)](#option-a--skillmd-file-recommended)
  - [Option B — Python Class](#option-b--python-class)
  - [Option C — Wrap an Existing Agent](#option-c--wrap-an-existing-agent)
- [Tutorial: Building a Complex SKILL.md Skill](#tutorial-building-a-complex-skillmd-skill)
- [Tools in Skills](#tools-in-skills)
  - [YAML tools Field (Anthropic Standard)](#yaml-tools-field-anthropic-standard)
  - [Programmatic Tools](#programmatic-tools)
  - [Tools in Python Skills](#tools-in-python-skills)
  - [Tool Security](#tool-security)
  - [How Tools Propagate: Skill → Agent](#how-tools-propagate-skill--agent)
- [Skill Discovery and Registry](#skill-discovery-and-registry)
  - [Dynamic Discovery at Runtime](#dynamic-discovery-at-runtime)
- [SKILL.md Reference](#skillmd-reference)
  - [YAML Frontmatter Fields](#yaml-frontmatter-fields)
  - [Tools Entry Format](#tools-entry-format)
  - [Supporting Directories](#supporting-directories)
  - [File References](#file-references)
  - [Markdown Body](#markdown-body)
  - [Naming Conventions](#naming-conventions)
  - [Best Practices](#best-practices)
- [Comparison: Nono vs Anthropic Skills](#comparison-nono-vs-anthropic-skills)
- [API Reference](#api-reference)

---

## Overview

A **Skill** is a reusable unit of AI capability that packages an agent, domain-specific instructions, and metadata into a single discoverable component. Skills let you:

| Capability | Description |
|---|---|
| **Define once** | Package expert instructions, output format, and agent config |
| **Discover** | Find skills by name or tag via the global registry |
| **Compose** | Attach multiple skills to an agent as callable tools |
| **Reuse** | Same skill works across Python, CLI, and REST API |

Nono supports **two formats** for defining skills:

| Format | Best For | Anthropic Compatible |
|---|---|---|
| **SKILL.md** files | Instructions-heavy skills, shareable definitions | Yes |
| **Python classes** | Skills with custom tools, validation, complex logic | Conceptually aligned |

---

## Anthropic Agent Skills Standard

Nono's skill system is fully compatible with the [Agent Skills](https://agentskills.io/specification) open standard (originally developed by Anthropic, adopted by Claude Code, Roo Code, JetBrains Junie, and others). The standard defines skills as:

- **Filesystem-based** `SKILL.md` files with YAML frontmatter
- **Three-level loading**: metadata (always) → instructions (on trigger) → resources (as needed)
- **Auto-discovered** by the agent at startup from configured directories
- **Model-invoked** — the agent autonomously chooses when to use them

Nono implements the full standard including:

- **All standard frontmatter fields**: `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`
- **Supporting directories**: `scripts/`, `references/`, `assets/` with `load_resource()` access
- **Progressive disclosure**: metadata → instructions → resources (3-level loading)
- **File references**: `.load_resource("references/REFERENCE.md")` for on-demand resource loading

Nono also extends the standard with:

- **Programmatic Python skills** for complex logic and custom tools
- **YAML `tools` field** — reference Python scripts as tools directly from SKILL.md
- **A global registry** for skill discovery across all interfaces
- **`as_tool()` conversion** — any skill becomes a function-calling tool
- **`build_tools()` injection** — skill tools are automatically wired into the inner agent
- **Multi-provider support** — skills work with all 14 Nono LLM providers

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    SkillRegistry                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Python Skill│  │ SKILL.md File│  │ Wrapped Factory│  │
│  │ (BaseSkill) │  │(MarkdownSkill│  │(skill_from_agent│ │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
│         └────────────────┼──────────────────┘            │
│                    SkillDescriptor                        │
│              (name, description, tags)                    │
└──────────────────────────┬───────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Python  │    │   CLI    │    │ REST API │
    │   Code   │    │ nono sk  │    │ /skill/  │
    └──────────┘    └──────────┘    └──────────┘
```

### Three-Level Loading (Anthropic Compatible)

| Level | What Loads | When | Token Cost |
|---|---|---|---|
| **1. Metadata** | `name` + `description` | At startup / import | ~10 tokens per skill |
| **2. Instructions** | Full SKILL.md body or Python instruction | When skill is invoked | Depends on content |
| **3. Resources** | Referenced files, scripts, data | As needed by the agent | On demand |

---

## Skill Formats

### Python Skills (Programmatic)

Python skills inherit from `BaseSkill` and implement `descriptor` and `build_agent()`:

```python
from nono.agent.skill import BaseSkill, SkillDescriptor, registry
from nono.agent import LlmAgent

class MySkill(BaseSkill):

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="my_skill",
            description="What this skill does and when to use it.",
            tags=("text", "analysis"),
        )

    def build_agent(self, **overrides) -> LlmAgent:
        return LlmAgent(
            name="my_agent",
            provider=overrides.get("provider", "google"),
            instruction="Your expert instructions here.",
        )

registry.register(MySkill())
```

### SKILL.md Files (Anthropic Standard)

SKILL.md files follow the [Agent Skills](https://agentskills.io/specification) open standard:

```markdown
---
name: analyzing-logs
description: >
  Analyze application logs to identify errors, patterns, and anomalies.
  Use when the user asks to analyze, parse, or review log files.
license: MIT
compatibility: Requires Python 3.10+
metadata:
  author: my-team
  version: "2.0"
allowed-tools: Read Grep
tags:
  - text
  - analysis
  - logs
temperature: 0.2
output_format: json
---

# Analyze Logs

Parse application logs and identify:

1. **Errors**: Stack traces, exceptions, error codes
2. **Patterns**: Repeated sequences, frequency spikes
3. **Anomalies**: Unusual timing, unexpected values

## Output format

Return a JSON object with `errors`, `patterns`, and `anomalies` arrays.
```

SKILL.md files can also reference **utility scripts as tools** (see [Tools in Skills](#tools-in-skills)):

```markdown
---
name: processing-forms
description: Process PDF forms with validation and field extraction.
tags:
  - pdf
  - forms
tools:
  - name: validate_form
    script: scripts/validate_form.py
    description: Validate form field mappings against the PDF.
  - name: extract_fields
    script: scripts/extract_fields.py
    description: Extract all form fields and their locations.
---

# Form Processing

Use the validate_form tool before filling and extract_fields to discover fields.
```

Load it:

```python
from nono.agent.skill_loader import load_skill_md, scan_skills_dir
from nono.agent.skill import registry

# Single file
skill = load_skill_md("skills/analyzing-logs/SKILL.md", register_in=registry)

# Entire directory
skills = scan_skills_dir(".claude/skills", register_in=registry)
```

---

## Built-in Skills

Nono ships with five built-in skills, available both as Python classes and SKILL.md files:

| Skill | Name | Tags | Description |
|---|---|---|---|
| **SummarizeSkill** | `summarize` | `text`, `summarization`, `analysis` | Summarize text into key points |
| **ClassifySkill** | `classify` | `text`, `classification`, `routing` | Classify text with confidence scores |
| **ExtractSkill** | `extract` | `text`, `extraction`, `data` | Extract structured data from text |
| **CodeReviewSkill** | `code_review` | `code`, `review`, `security` | Review code for quality and security |
| **TranslateSkill** | `translate` | `text`, `translation`, `language` | Translate text between languages |

All built-in SKILL.md skills include **supporting directories** following the Agent Skills standard:

| Skill | `references/` | `assets/` | `scripts/` | `tools:` |
|---|---|---|---|---|
| **reviewing-code** | OWASP Top 10 checklist, severity guide | `report_schema.json` | `score.py` | `score_review` |
| **extracting-data** | Entity types, extraction guidelines | `output_schema.json` | — | — |
| **classifying-data** | Confidence scoring, common domains | `output_schema.json` | — | — |
| **summarizing-text** | Techniques by content type, length guide | — | — | — |
| **translating-text** | ISO 639-1 language codes, quality guide | — | — | — |

All skills also include `license: MIT` and `metadata` (author, version) in their frontmatter.

Built-in skills auto-register in the global `registry` on import:

```python
import nono.agent.skills  # triggers registration

from nono.agent.skill import registry
print(registry.names)
# ['classify', 'code_review', 'extract', 'summarize', 'translate']
```

### SKILL.md Equivalents

The SKILL.md versions are located in `nono/agent/skills/`:

```
nono/agent/skills/
├── summarizing-text/
│   ├── SKILL.md
│   └── references/REFERENCE.md          # Summarization techniques & length guide
├── classifying-data/
│   ├── SKILL.md
│   ├── references/REFERENCE.md          # Confidence scoring & common domains
│   └── assets/output_schema.json        # JSON schema for classification output
├── extracting-data/
│   ├── SKILL.md
│   ├── references/REFERENCE.md          # Entity types & extraction guidelines
│   └── assets/output_schema.json        # JSON schema for extraction output
├── reviewing-code/
│   ├── SKILL.md
│   ├── references/REFERENCE.md          # OWASP Top 10 checklist & severity guide
│   ├── assets/report_schema.json        # JSON schema for review reports
│   └── scripts/score.py                 # Scoring tool (main() → FunctionTool)
├── translating-text/
│   ├── SKILL.md
│   └── references/REFERENCE.md          # ISO 639-1 codes & quality guidelines
├── analyzing-sentiment/
│   ├── SKILL.md
│   ├── references/REFERENCE.md          # Emotion taxonomy & scoring
│   └── assets/output_schema.json        # JSON schema for sentiment output
├── analyzing-data/
│   ├── SKILL.md
│   └── assets/output_schema.json        # JSON schema for analysis output
├── generating-sql/
│   └── SKILL.md
├── generating-tests/
│   └── SKILL.md
├── generating-api-docs/
│   ├── SKILL.md
│   └── scripts/validate_schema.py       # OpenAPI schema validator
├── writing-documentation/
│   └── SKILL.md
├── writing-emails/
│   └── SKILL.md
├── explaining-code/
│   └── SKILL.md
├── converting-formats/
│   └── SKILL.md
└── researching-topics/
    └── SKILL.md
```

Access supporting files programmatically:

```python
from nono.agent.skill_loader import scan_skills_dir
from nono.agent.skill import registry

skills = scan_skills_dir("nono/agent/skills", register_in=registry)

# Load reference material from a skill
review = registry.get("reviewing-code")
owasp = review.load_resource("references/REFERENCE.md")
schema = review.load_resource("assets/report_schema.json")

# List available references
for path in review.list_references():
    print(path.name)  # REFERENCE.md

# The reviewing-code skill also bundles a scoring tool
tools = review.build_tools()
print(tools[0].name)  # "score_review"
```

---

## Usage: Python Code

### Standalone Execution

```python
from nono.agent.skills import SummarizeSkill

skill = SummarizeSkill()
result = skill.run("Long article text here...")
print(result)
```

### Attach to an Agent

Skills auto-convert to function-calling tools when attached to an `LlmAgent`:

```python
from nono.agent import Agent
from nono.agent.skills import SummarizeSkill, ClassifySkill, TranslateSkill

agent = Agent(
    name="analyst",
    instruction="You analyze and process text. Use your skills when needed.",
    skills=[SummarizeSkill(), ClassifySkill(), TranslateSkill()],
)
```

When the LLM decides to use a skill, it calls the skill's `as_tool()` wrapper, which executes the skill's inner agent and returns the result.

### Load from SKILL.md

```python
from nono.agent.skill_loader import load_skill_md

# Load a single skill
skill = load_skill_md(".claude/skills/analyzing-logs/SKILL.md")
print(skill.descriptor.name)        # "analyzing-logs"
print(skill.instruction[:100])       # First 100 chars of the markdown body

result = skill.run("2024-01-15 ERROR NullPointerException at line 42...")
```

### Scan a Directory

```python
from nono.agent.skill_loader import scan_skills_dir
from nono.agent.skill import registry

# Load all SKILL.md files from a directory tree
skills = scan_skills_dir(".claude/skills", register_in=registry)
print(f"Loaded {len(skills)} skills")

# Now use any skill by name
skill = registry.get("analyzing-logs")
if skill:
    result = skill.run("log content here...")
```

Directory layout (Anthropic standard):

```
.claude/skills/
├── analyzing-logs/
│   ├── SKILL.md
│   └── references/            # On-demand reference docs
│       └── REFERENCE.md
├── processing-pdfs/
│   ├── SKILL.md
│   ├── scripts/               # Executable tools
│   │   └── extract.py
│   ├── references/
│   │   └── REFERENCE.md
│   └── assets/                # Schemas, templates
│       └── output_schema.json
└── generating-reports/
    └── SKILL.md
```

### Compose in Workflows

```python
from nono.workflows import Workflow

flow = Workflow(name="content_pipeline")

flow.add_step("translate", skill_translate.build_agent())
flow.add_step("summarize", skill_summarize.build_agent())

result = flow.run(text="Texto en español aquí...")
```

---

## Usage: CLI

```bash
# Run a skill directly
nono skill summarize "Long text to summarize..."

# With provider override
nono skill classify "Is this spam?" --provider openai

# Alias: 'sk'
nono sk translate "Hola, ¿cómo estás?"

# List available skills
nono info
```

The CLI auto-loads all registered built-in skills. Output includes skill name, result, and optional trace information.

---

## Usage: REST API

### Run a Skill

```bash
curl -X POST http://localhost:8000/skill/summarize \
  -H "Content-Type: application/json" \
  -d '{"input": "Long text to summarize...", "trace": true}'
```

Response:

```json
{
  "result": "## Summary\n...",
  "skill": "summarize",
  "duration_ms": 1234,
  "trace": { ... }
}
```

### List Available Skills

```bash
curl http://localhost:8000/info
```

Response includes:

```json
{
  "skills": [
    {"name": "summarize", "description": "Summarize text into up to 5 key points."},
    {"name": "classify", "description": "Classify text into categories."},
    ...
  ]
}
```

---

## Creating Custom Skills

### Option A — SKILL.md File (Recommended)

The simplest way to create a skill. Follows the Anthropic standard and requires no Python code.

**1. Create the directory structure:**

```
my_skills/
└── sentiment-analysis/
    ├── SKILL.md             # Required: metadata + instructions
    ├── scripts/             # Optional: executable code as tools
    │   └── lookup.py
    ├── references/          # Optional: documentation loaded on demand
    │   └── REFERENCE.md
    └── assets/              # Optional: templates, schemas, data files
        └── output_schema.json
```

**2. Write the SKILL.md:**

```markdown
---
name: sentiment-analysis
description: >
  Analyze text sentiment and emotional tone. Use when
  the user asks about sentiment, mood, or emotional tone.
tags:
  - text
  - sentiment
  - analysis
temperature: 0.1
output_format: json
---

# Sentiment Analysis

Analyze the emotional tone of text content.

## Output format

```json
{
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.92,
  "emotions": ["joy", "surprise"],
  "explanation": "Brief explanation"
}
```

## Guidelines

- Consider context and nuance, not just keywords.
- Sarcasm should be flagged in the explanation.
- For mixed sentiments, list all detected emotions.
```

**3. Load and use:**

```python
from nono.agent.skill_loader import load_skill_md
from nono.agent.skill import registry

skill = load_skill_md(
    "my_skills/sentiment-analysis/SKILL.md",
    register_in=registry,
)
result = skill.run("I absolutely love this product! Best purchase ever.")
```

### Option B — Python Class

For skills that need custom tools, validation logic, or complex agent pipelines.

```python
from nono.agent.skill import BaseSkill, SkillDescriptor, registry
from nono.agent import LlmAgent, FunctionTool

class SentimentSkill(BaseSkill):

    def __init__(self, *, provider: str = "google"):
        self._provider = provider

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="sentiment",
            description="Analyze text sentiment and emotional tone.",
            tags=("text", "sentiment", "analysis"),
        )

    def build_agent(self, **overrides):
        return LlmAgent(
            name="sentiment_analyzer",
            provider=overrides.get("provider", self._provider),
            instruction="Analyze sentiment. Return JSON.",
            output_format="json",
            temperature=0.1,
        )

    def build_tools(self):
        """Optional: add domain-specific tools to the agent."""
        return [
            FunctionTool(
                fn=self._lookup_lexicon,
                name="sentiment_lexicon",
                description="Look up word in sentiment lexicon.",
            )
        ]

    @staticmethod
    def _lookup_lexicon(word: str) -> str:
        lexicon = {"love": "+0.9", "hate": "-0.9", "okay": "+0.1"}
        return lexicon.get(word.lower(), "0.0")

registry.register(SentimentSkill())
```

### Option C — Wrap an Existing Agent

Use `skill_from_agent` to wrap any agent factory function as a skill:

```python
from nono.agent.skill import skill_from_agent

def my_agent_factory(**kwargs):
    from nono.agent import Agent
    return Agent(
        name="formatter",
        provider=kwargs.get("provider", "google"),
        instruction="Format the input as a clean markdown document.",
    )

skill = skill_from_agent(
    name="format_markdown",
    description="Format text as clean markdown.",
    agent_factory=my_agent_factory,
    tags=("text", "formatting"),
    register=True,  # auto-register in global registry
)
```

---

## Tutorial: Building a Complex SKILL.md Skill

This step-by-step guide walks through creating a complete skill with scripts, references, and assets — from zero to a fully functional agent tool.

### Goal

Build an **invoice-processing** skill that extracts line items from PDF invoices, validates totals, and outputs structured JSON.

### Step 1 — Create the Directory Structure

```
my_skills/
└── processing-invoices/
    ├── SKILL.md                  # Metadata + instructions
    ├── scripts/
    │   ├── extract_lines.py      # Tool: extract line items from text
    │   └── validate_totals.py    # Tool: verify amounts add up
    ├── references/
    │   └── REFERENCE.md          # Accounting rules, edge cases
    └── assets/
        └── output_schema.json    # Expected JSON output schema
```

### Step 2 — Write the SKILL.md

```markdown
---
name: processing-invoices
description: >
  Extract line items and validate totals from invoices.
  Use when the user provides invoice text or asks to parse billing documents.
license: MIT
compatibility: Requires Python 3.10+
metadata:
  author: my-team
  version: "1.0"
tags:
  - finance
  - extraction
  - validation
temperature: 0.1
output_format: json
tools:
  - name: extract_lines
    script: scripts/extract_lines.py
    description: Extract line items (description, quantity, unit price) from invoice text.
  - name: validate_totals
    script: scripts/validate_totals.py
    description: Validate that line item amounts sum to the invoice total.
---

# Invoice Processing

Extract and validate invoice data following these steps:

1. Use **extract_lines** to parse all line items from the input.
2. Use **validate_totals** to verify the amounts are consistent.
3. Return structured JSON matching the schema in `assets/output_schema.json`.

## Output Format

Return a JSON object:

```json
{
  "vendor": "string",
  "date": "YYYY-MM-DD",
  "lines": [
    {"description": "string", "qty": 1, "unit_price": 10.0, "total": 10.0}
  ],
  "subtotal": 100.0,
  "tax": 21.0,
  "grand_total": 121.0,
  "valid": true
}
```

## Edge Cases

Refer to `.load_resource("references/REFERENCE.md")` for accounting rules
and common edge cases (discounts, multi-currency, rounding).
```

### Step 3 — Implement the Tool Scripts

Each script must expose a `main(input: str) -> str` function:

**`scripts/extract_lines.py`**:

```python
import json
import re

def main(input: str) -> str:
    """Extract line items from invoice text."""
    lines = []
    for match in re.finditer(
        r"(.+?)\s+(\d+)\s+x\s+([\d.]+)\s*€?", input
    ):
        desc, qty, price = match.groups()
        qty, price = int(qty), float(price)
        lines.append({
            "description": desc.strip(),
            "qty": qty,
            "unit_price": price,
            "total": round(qty * price, 2),
        })
    return json.dumps({"lines": lines, "count": len(lines)})
```

**`scripts/validate_totals.py`**:

```python
import json

def main(input: str) -> str:
    """Validate that line totals sum correctly."""
    data = json.loads(input)
    lines = data.get("lines", [])
    computed = round(sum(l["total"] for l in lines), 2)
    expected = data.get("subtotal", computed)
    valid = abs(computed - expected) < 0.01
    return json.dumps({
        "computed_subtotal": computed,
        "expected_subtotal": expected,
        "valid": valid,
        "difference": round(computed - expected, 2),
    })
```

### Step 4 — Write the Reference Document

**`references/REFERENCE.md`**:

```markdown
# Invoice Processing Reference

## Rounding Rules
- Round line totals to 2 decimal places before summing.
- Apply tax after subtotal rounding.

## Multi-Currency
- If currencies are mixed, flag as `"valid": false`.

## Discounts
- Discounts appear as negative line items. Include in subtotal.
```

### Step 5 — Define the Output Schema

**`assets/output_schema.json`**:

```json
{
  "type": "object",
  "required": ["vendor", "date", "lines", "grand_total", "valid"],
  "properties": {
    "vendor": {"type": "string"},
    "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
    "lines": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["description", "qty", "unit_price", "total"],
        "properties": {
          "description": {"type": "string"},
          "qty": {"type": "integer"},
          "unit_price": {"type": "number"},
          "total": {"type": "number"}
        }
      }
    },
    "subtotal": {"type": "number"},
    "tax": {"type": "number"},
    "grand_total": {"type": "number"},
    "valid": {"type": "boolean"}
  }
}
```

### Step 6 — Load, Test, and Use

```python
from nono.agent.skill_loader import load_skill_md
from nono.agent.skill import registry
from nono.agent import Agent

# Load the skill
skill = load_skill_md(
    "my_skills/processing-invoices/SKILL.md",
    register_in=registry,
)

# 1. Standalone execution
result = skill.run("Acme Corp - 2025-03-15\nWidget 3 x 25.00€\nGadget 1 x 99.50€")
print(result)  # Structured JSON output

# 2. Attach to an agent as a tool
agent = Agent(
    name="finance_assistant",
    instruction="You help users process financial documents.",
    skills=[skill],
)

# 3. Compose in a pipeline
from nono.workflows import Workflow

flow = Workflow(name="invoice_pipeline")
flow.add_step("process", skill.build_agent())
flow.add_step("review", review_agent)  # a separate reviewer agent
result = flow.run(text="Invoice text here...")

# 4. Access reference material
owasp = skill.load_resource("references/REFERENCE.md")
schema = skill.load_resource("assets/output_schema.json")

# 5. List bundled tools
for t in skill.build_tools():
    print(f"{t.name}: {t.description}")
    # extract_lines: Extract line items from invoice text.
    # validate_totals: Validate that line totals sum correctly.
```

---

## Tools in Skills

Skills can provide **domain-specific tools** that the inner LLM agent can call during execution.  This follows the Anthropic convention of bundling utility scripts alongside SKILL.md files.

Tools are automatically injected into the agent created by `build_agent()` when the skill is executed via `run()` or `as_tool()`.

### YAML `tools` Field (Anthropic Standard)

Declare tools directly in the SKILL.md frontmatter.  Each tool references a Python script that must expose a `main(input: str) -> str` function:

```yaml
---
name: form-processor
description: Process and validate PDF forms.
tools:
  - name: validate_form
    script: scripts/validate_form.py
    description: Validate form field mappings.
  - name: analyze_pdf
    script: scripts/analyze_pdf.py
    description: Extract form fields from a PDF.
---
```

Directory layout:

```
form-processing/
├── SKILL.md
└── scripts/
    ├── validate_form.py    # must define main(input: str) -> str
    └── analyze_pdf.py      # must define main(input: str) -> str
```

Example script (`scripts/validate_form.py`):

```python
import json

def main(input: str) -> str:
    """Validate form fields against expected schema."""
    fields = json.loads(input)
    errors = []
    for name, spec in fields.items():
        if "type" not in spec:
            errors.append(f"Field '{name}' missing 'type'")
    if errors:
        return json.dumps({"valid": False, "errors": errors})
    return json.dumps({"valid": True, "errors": []})
```

When the skill runs, the LLM agent sees `validate_form` and `analyze_pdf` as callable tools and can invoke them via function calling.

### Programmatic Tools

Pass `FunctionTool` instances when loading a SKILL.md:

```python
from nono.agent.tool import FunctionTool
from nono.agent.skill_loader import load_skill_md

def search_db(query: str) -> str:
    return f"Results for: {query}"

db_tool = FunctionTool(fn=search_db, name="search_db", description="Search the database.")

# Programmatic tools are combined with YAML-declared tools
skill = load_skill_md("skills/my-skill/SKILL.md", tools=[db_tool])
```

Or pass them directly to the `MarkdownSkill` constructor:

```python
from nono.agent.skill_loader import MarkdownSkill

skill = MarkdownSkill(meta, body, source_path=path, tools=[db_tool])
```

### Tools in Python Skills

Python skills define tools by overriding `build_tools()`:

```python
from nono.agent.skill import BaseSkill, SkillDescriptor
from nono.agent import LlmAgent, FunctionTool

class DataSkill(BaseSkill):

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="data-lookup",
            description="Look up data in local databases.",
            tags=("data", "lookup"),
        )

    def build_agent(self, **overrides):
        return LlmAgent(
            name="data_agent",
            instruction="Use the lookup tool to find data.",
        )

    def build_tools(self):
        return [
            FunctionTool(
                fn=self._lookup,
                name="lookup",
                description="Look up a record by ID.",
            )
        ]

    @staticmethod
    def _lookup(id: str) -> str:
        db = {"001": "Alice", "002": "Bob"}
        return db.get(id, "Not found")
```

Tools returned by `build_tools()` are automatically injected into the agent when calling `skill.run()` or `skill.as_tool()`.

### Tool Security

| Protection | Description |
|---|---|
| **Path traversal blocked** | Scripts must reside inside the skill directory |
| **Required fields validated** | Each tool entry must have `name`, `script`, and `description` |
| **`main()` required** | Script must expose a callable `main` function |
| **No duplicate injection** | Tools are de-duplicated by name when injected |

Example of a rejected path traversal:

```yaml
tools:
  - name: evil
    script: ../../../etc/exploit.py   # ❌ ValueError: must be inside the skill directory
    description: Malicious script
```

### How Tools Propagate: Skill → Agent

Understanding how tools flow when a skill is used in different modes is key to building correct multi-skill agents.

#### Mode 1: Standalone (`skill.run()`)

```
skill.run("message")
  ├── skill.build_agent()        → creates the inner LlmAgent
  ├── skill.build_tools()        → returns [FunctionTool, ...]
  ├── _inject_tools(agent, tools)→ appends tools to agent.tools (deduped by name)
  └── Runner(agent).run(msg)     → inner agent can call skill's tools via function-calling
```

The inner agent sees **only its own tools** (from `build_tools()` or YAML `tools:`).  It does not see tools from the parent agent.

#### Mode 2: As a Tool (`skills=[...]` on a parent agent)

When you attach skills to an `LlmAgent`, two separate tool scopes exist:

```
Parent Agent (LlmAgent)
  ├── tools: [get_weather, search_db]        ← parent's own tools
  ├── skills: [InvoiceSkill, ReviewSkill]    ← converted via as_tool()
  └── _all_tools (combined):
        [get_weather, search_db, processing-invoices, reviewing-code]
        ↑ flat list: parent tools + one FunctionTool per skill

When the LLM calls "processing-invoices":
  skill.as_tool()._invoke(input)
    ├── skill.build_agent()       → fresh inner agent
    ├── skill.build_tools()       → inner tools (extract_lines, validate_totals)
    ├── _inject_tools(agent, ...)  → injected into the inner agent
    └── Runner(inner_agent).run() → inner agent uses its own tools
```

**Key points:**

| Aspect | Parent Agent | Inner Skill Agent |
|---|---|---|
| **Sees parent tools?** | Yes | No — isolated scope |
| **Sees skill tools?** | No — only sees `skill.as_tool()` wrapper | Yes — injected via `build_tools()` |
| **`allowed-tools` effect** | Not applied | Stored in `SkillDescriptor.allowed_tools` for reference |
| **Tool name collisions** | De-duplicated by name at each level | De-duplicated independently |

#### Mode 3: As a Pipeline (`skill.build_agent()`)

When you extract the inner agent, tools are **not** auto-injected — you must inject them explicitly:

```python
skill = InvoiceSkill()
agent = skill.build_agent()

# Tools are NOT injected yet — agent.tools is empty
from nono.agent.skill import _inject_tools
_inject_tools(agent, skill.build_tools())

# Now the agent has the skill's tools
pipeline = SequentialAgent(sub_agents=[agent, reviewer_agent])
```

Alternatively, use `skill.run()` which handles injection automatically.

#### The `allowed-tools` Field

The `allowed-tools` YAML field (from the Anthropic standard) declares which tools the skill is *expected* to use.  In Nono, this is stored as `SkillDescriptor.allowed_tools` for documentation and discovery purposes.  It does **not** restrict which tools the inner agent can actually call — enforcement is the responsibility of the outer orchestration layer.

```yaml
allowed-tools: Read Grep Write
```

```python
skill = load_skill_md("skills/analyzing-logs/SKILL.md")
print(skill.descriptor.allowed_tools)  # ("Read", "Grep", "Write")
```

---

## Skill Discovery and Registry

The `SkillRegistry` provides discovery and lookup:

```python
from nono.agent.skill import registry

# List all skills
for desc in registry.list_skills():
    print(f"{desc.name}: {desc.description}")

# Find by tag
text_skills = registry.find_by_tag("text")

# Get by name
skill = registry.get("summarize")

# Check existence
if "translate" in registry:
    print("Translation available")

# Count
print(f"{len(registry)} skills registered")
```

### Registration Methods

| Method | When to Use |
|---|---|
| `registry.register(instance)` | Register a skill instance |
| `registry.register(MySkillClass)` | Register by class (auto-instantiated) |
| `@registry.register` | Decorator on a class definition |
| `load_skill_md(path, register_in=registry)` | Load and register a SKILL.md file |
| `scan_skills_dir(dir, register_in=registry)` | Scan and register all SKILL.md in a directory |
| `skill_from_agent(..., register=True)` | Wrap a factory and register |

### Dynamic Discovery at Runtime

Beyond static registration, you can build agents that **discover and attach skills dynamically** based on user intent or task context.

#### Auto-attach by Tag

Select skills at runtime based on the task domain:

```python
from nono.agent import Agent
from nono.agent.skill import registry
import nono.agent.skills  # ensure built-ins are registered

def build_agent_for_domain(domain: str) -> Agent:
    """Create an agent with skills matching a domain tag."""
    matching = registry.find_by_tag(domain)
    skills = [registry.get(desc.name) for desc in matching]

    return Agent(
        name=f"{domain}_agent",
        instruction=f"You are a {domain} expert. Use your skills when needed.",
        skills=[s for s in skills if s is not None],
    )

# Build a text-processing agent with all "text" skills
agent = build_agent_for_domain("text")
# → skills: [ClassifySkill, ExtractSkill, SummarizeSkill, TranslateSkill]
```

#### User-Driven Skill Selection

Let the user choose which skills to enable:

```python
from nono.agent import Agent
from nono.agent.skill import registry

def build_agent_with_skills(skill_names: list[str]) -> Agent:
    """Create an agent with user-selected skills."""
    skills = []
    for name in skill_names:
        skill = registry.get(name)
        if skill is None:
            raise ValueError(
                f"Unknown skill {name!r}. "
                f"Available: {registry.names}"
            )
        skills.append(skill)

    return Agent(
        name="custom_agent",
        instruction="Use your skills to help the user.",
        skills=skills,
    )

# User selects skills
agent = build_agent_with_skills(["summarize", "translate"])
```

#### Hot-Reload from a Skills Directory

Scan a directory at startup (or on demand) to pick up newly added SKILL.md files:

```python
from nono.agent.skill_loader import scan_skills_dir
from nono.agent.skill import registry
from nono.agent import Agent

def refresh_skills(skill_dirs: list[str]) -> None:
    """Re-scan directories and register any new skills."""
    for d in skill_dirs:
        scan_skills_dir(d, register_in=registry)

# On startup
refresh_skills([".claude/skills", "company_skills/", "nono/agent/skills"])

# Build an agent with ALL registered skills
all_skills = [registry.get(d.name) for d in registry.list_skills()]
agent = Agent(
    name="universal",
    instruction="You have access to all available skills.",
    skills=[s for s in all_skills if s is not None],
)

print(f"Agent loaded with {len(agent.skills)} skills: {registry.names}")
```

#### Skill Metadata Inspection

Use descriptors to build dynamic UIs or routing logic:

```python
from nono.agent.skill import registry

for desc in registry.list_skills():
    print(f"  Name:        {desc.name}")
    print(f"  Description: {desc.description}")
    print(f"  Tags:        {', '.join(desc.tags)}")
    print(f"  Version:     {desc.version}")
    print(f"  License:     {desc.license or 'Not specified'}")
    print(f"  Tools:       {', '.join(desc.allowed_tools) or 'None declared'}")
    print()
```

---

## SKILL.md Reference

### YAML Frontmatter Fields

| Field | Required | Type | Constraint | Description |
|---|---|---|---|---|
| `name` | Yes | `string` | Max 64 chars, `[a-z0-9-]` only | Unique skill identifier |
| `description` | Yes | `string` | Max 1024 chars, non-empty | What the skill does and when to use it |
| `license` | No | `string` | — | License name or reference to a bundled LICENSE file |
| `compatibility` | No | `string` | Max 500 chars | Environment requirements (product, packages, network) |
| `metadata` | No | `map[str, str]` | String keys → string values | Arbitrary key-value map for extra properties |
| `allowed-tools` | No | `string \| list` | Space-separated or YAML list | Pre-approved tools the skill may use |
| `tags` | No | `list[string]` | — | Classification tags for discovery |
| `version` | No | `string` | Default `"1.0.0"` | Semantic version |
| `temperature` | No | `float` | `0.0` – `2.0` | LLM temperature override |
| `output_format` | No | `string` | `"json"` or `null` | Force structured output |
| `provider` | No | `string` | Default `"google"` | LLM provider |
| `model` | No | `string` | — | Model override |
| `input_keys` | No | `list[string]` | Default `["input"]` | Session state keys the skill reads |
| `output_keys` | No | `list[string]` | Default `["output"]` | Session state keys the skill writes |
| `tools` | No | `list[object]` | See [Tools Entry Format](#tools-entry-format) | Utility scripts exposed as agent tools |

### Tools Entry Format

Each entry in the `tools` list is a mapping with three required fields:

| Field | Required | Type | Description |
|---|---|---|---|
| `name` | Yes | `string` | Tool name (used in LLM function calling) |
| `script` | Yes | `string` | Relative path to a Python script (from SKILL.md directory) |
| `description` | Yes | `string` | Description shown to the LLM when selecting tools |

The referenced script must define a `main(input: str) -> str` function.  Nono imports the script as a Python module and wraps `main` as a `FunctionTool`.

```yaml
tools:
  - name: validate_form
    script: scripts/validate_form.py
    description: Validate form field mappings against the PDF schema.
  - name: fill_form
    script: scripts/fill_form.py
    description: Fill PDF form fields with provided values.
```

### Supporting Directories

The Agent Skills standard defines three optional subdirectories:

```
my-skill/
├── SKILL.md           # Required: metadata + instructions
├── scripts/           # Optional: executable code
├── references/        # Optional: documentation (REFERENCE.md, domain files)
└── assets/            # Optional: templates, schemas, data files
```

Nono provides methods to discover and load these files:

```python
from nono.agent.skill_loader import load_skill_md

skill = load_skill_md("skills/pdf-processing/SKILL.md")

# List files in each directory
refs = skill.list_references()     # references/*.md
assets = skill.list_assets()       # assets/*.json, etc.
scripts = skill.list_scripts()     # scripts/*.py

# skill_dir gives you the root
print(skill.skill_dir)  # Path to the skill directory
```

| Directory | Purpose | Typical Contents |
|---|---|---|
| `scripts/` | Executable code the agent can run | Python scripts, shell scripts |
| `references/` | Documentation loaded on demand | `REFERENCE.md`, domain files (`finance.md`, `legal.md`) |
| `assets/` | Static resources | Templates, schemas, lookup tables, images |

### File References

SKILL.md can reference supporting files via relative paths.  Use `load_resource()` to read them at runtime:

```python
skill = load_skill_md("skills/pdf-processing/SKILL.md")

# Load a reference document
api_docs = skill.load_resource("references/REFERENCE.md")

# Load an asset
template = skill.load_resource("assets/template.json")
```

Path traversal protection blocks attempts to escape the skill directory:

```python
skill.load_resource("../../etc/passwd")  # ❌ ValueError: escapes the skill directory
```

### Markdown Body

The content below the frontmatter becomes the agent's **system instruction**. Follow these principles:

- **Be concise** — Only include information the LLM doesn't already know
- **Use structure** — Headers, bullet points, code blocks
- **Define output format** — Show the expected response structure
- **Keep under 500 lines** — Split large content into referenced files

### Naming Conventions

Follow the Anthropic standard — **gerund form** (verb + -ing):

| Good | Avoid |
|---|---|
| `summarizing-text` | `summarizer` |
| `analyzing-logs` | `log-analysis` |
| `reviewing-code` | `code-reviewer` |
| `processing-pdfs` | `pdf-utils` |

Rules:
- Lowercase letters, numbers, and hyphens only
- Maximum 64 characters
- No reserved words: `anthropic`, `claude`

### Best Practices

Following the [Anthropic Skill Authoring Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices):

| Principle | Guidance |
|---|---|
| **Concise is key** | Only add context the LLM doesn't already have |
| **Set appropriate freedom** | Match specificity to task fragility |
| **Write third-person descriptions** | "Analyzes logs..." not "I analyze logs..." |
| **Include trigger keywords** | Add terms users might say to invoke the skill |
| **Test across models** | Verify with different providers and models |
| **Use progressive disclosure** | Link to separate files for advanced content |
| **Avoid time-sensitive info** | No dates, versions that will become outdated |
| **Consistent terminology** | Pick one term and use it throughout |

---

## Comparison: Nono vs Anthropic Skills

| Concept | Anthropic Agent Skills | Nono Skills |
|---|---|---|
| **Definition format** | `SKILL.md` with YAML frontmatter | `SKILL.md` files **and** Python classes |
| **Discovery** | Filesystem scan at startup | `SkillRegistry` + `scan_skills_dir()` |
| **Invocation** | Model-invoked via `Skill` tool | `as_tool()` → LLM function-calling |
| **Loading** | Progressive (metadata → body → resources) | Same pattern via `MarkdownSkill` |
| **Metadata** | `name`, `description`, `license`, `compatibility`, `metadata` | `SkillDescriptor` — all standard fields supported |
| **Supporting dirs** | `scripts/`, `references/`, `assets/` | Same — `list_references()`, `list_assets()`, `list_scripts()` |
| **File references** | Relative paths from SKILL.md | `load_resource("references/REFERENCE.md")` |
| **Storage** | `.claude/skills/*/SKILL.md` | Any directory + global registry |
| **Tool restrictions** | `allowed-tools` in frontmatter | `allowed-tools` in frontmatter + agent-level `tools` parameter |
| **Bundled scripts** | `scripts/` dir, executed via bash | `tools:` YAML field → `FunctionTool` via `main()` |
| **Multi-provider** | Claude only | 14 LLM providers |
| **Programmatic API** | Not supported | `BaseSkill`, `skill_from_agent()` |
| **Standalone execution** | Not supported | `skill.run("message")` |
| **CLI / REST** | Via Claude Code CLI | `nono skill` / `POST /skill/{name}` |

---

## API Reference

### Classes

| Class | Module | Description |
|---|---|---|
| `BaseSkill` | `nono.agent.skill` | Abstract base class for all skills |
| `SkillDescriptor` | `nono.agent.skill` | Immutable skill metadata (name, description, license, compatibility, metadata, allowed_tools, tags, version) |
| `SkillRegistry` | `nono.agent.skill` | Registry for skill discovery |
| `MarkdownSkill` | `nono.agent.skill_loader` | Skill loaded from a SKILL.md file |

### Functions

| Function | Module | Description |
|---|---|---|
| `skill_from_agent()` | `nono.agent.skill` | Wrap an agent factory as a skill |
| `load_skill_md()` | `nono.agent.skill_loader` | Load a single SKILL.md file (accepts `tools` parameter) |
| `scan_skills_dir()` | `nono.agent.skill_loader` | Scan directory for SKILL.md files |

### Key Methods

| Method | Returns | Description |
|---|---|---|
| `skill.run(message)` | `str` | Execute the skill standalone (injects `build_tools()`) |
| `skill.as_tool()` | `FunctionTool` | Convert to LLM function-calling tool (injects `build_tools()`) |
| `skill.build_agent()` | `BaseAgent` | Get the skill's inner agent |
| `skill.build_tools()` | `list[FunctionTool]` | Get domain-specific tools for the agent |
| `skill.descriptor` | `SkillDescriptor` | Get skill metadata |
| `skill.skill_dir` | `Path \| None` | Directory containing the SKILL.md file |
| `skill.list_references()` | `list[Path]` | List files in `references/` directory |
| `skill.list_assets()` | `list[Path]` | List files in `assets/` directory |
| `skill.list_scripts()` | `list[Path]` | List files in `scripts/` directory |
| `skill.load_resource(path)` | `str` | Load a supporting file by relative path |
| `registry.register(skill)` | same | Register a skill |
| `registry.get(name)` | `BaseSkill \| None` | Look up by name |
| `registry.list_skills()` | `list[SkillDescriptor]` | List all registered skills |
| `registry.find_by_tag(tag)` | `list[SkillDescriptor]` | Find by tag |
| `registry.names` | `list[str]` | Sorted list of registered names |

### Singleton

```python
from nono.agent.skill import registry  # global SkillRegistry instance
```

---

*Author: DatamanEdge — MIT License*
