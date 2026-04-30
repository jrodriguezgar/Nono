# Skills

> SKILL.md files following the [Anthropic Agent Skills](https://agentskills.io/specification)
> standard — the same layout used by Claude Code, Roo Code, and others.

## Overview

This directory contains both **Python skill classes** and **SKILL.md skill
definitions**. Each SKILL.md skill lives in its own subdirectory, following
the Claude/Anthropic convention:

```
skills/
├── classify.py                # Python skill classes
├── code_review.py
├── extract.py
├── summarize.py
├── translate.py
├── __init__.py
├── summarizing-text/          # SKILL.md skills (Anthropic standard)
│   └── SKILL.md
├── classifying-data/
│   └── SKILL.md
├── analyzing-sentiment/
│   └── SKILL.md
└── ...
```

Skills are auto-discovered by `scan_skills_dir()` which searches for
`*/SKILL.md` in the given directory — no nesting required.

## Available Templates

### Built-in Skills (SKILL.md equivalents)

These are the SKILL.md versions of the built-in Python skills:

| Template | Category | Description |
|---|---|---|
| [summarizing-text](summarizing-text/) | Text / Analysis | Summarize text into key points |
| [classifying-data](classifying-data/) | Text / Classification | Classify text with confidence scores |
| [extracting-data](extracting-data/) | Text / Extraction | Extract structured data from text |
| [reviewing-code](reviewing-code/) | Code / Security | Review code for quality and security |
| [translating-text](translating-text/) | Text / Language | Translate text between languages |

### Additional Templates

| Template | Category | Description |
|---|---|---|
| [analyzing-sentiment](analyzing-sentiment/) | Text / NLP | Sentiment analysis with emotion and tone detection |
| [generating-sql](generating-sql/) | Code / Data | Natural language to SQL query generation |
| [writing-documentation](writing-documentation/) | Code / Docs | Auto-generate documentation for code |
| [analyzing-data](analyzing-data/) | Data / Analytics | Data analysis, insights, and visualization advice |
| [converting-formats](converting-formats/) | Data / Utility | Convert between JSON, YAML, CSV, XML, TOML |
| [generating-tests](generating-tests/) | Code / Testing | Generate unit tests for code |
| [researching-topics](researching-topics/) | Research / Knowledge | Topic research and synthesis |
| [writing-emails](writing-emails/) | Text / Communication | Professional email composition |
| [explaining-code](explaining-code/) | Code / Education | Explain code in plain language |
| [generating-api-docs](generating-api-docs/) | Code / Docs | OpenAPI/REST API documentation generation |

## Quick Start

```python
from nono.agent.skill_loader import load_skill_md, scan_skills_dir
from nono.agent.skill import registry

# Load a single skill
skill = load_skill_md(
    "nono/agent/skills/analyzing-sentiment/SKILL.md",
    register_in=registry,
)
result = skill.run("I love this product! Best purchase ever.")

# Load all SKILL.md skills at once
skills = scan_skills_dir("nono/agent/skills", register_in=registry)
```

## Creating Your Own

Use any skill as a starting point:

```bash
# Copy an existing skill
cp -r nono/agent/skills/analyzing-sentiment/ my_project/skills/my-custom-skill/

# Edit the SKILL.md
# - Change name, description, tags
# - Customize instructions
# - Add references/ or scripts/ as needed
```

## Skill Directory Structure

Each SKILL.md skill follows the Anthropic standard layout:

```
skill-name/
├── SKILL.md              # Main skill definition (YAML frontmatter + instructions)
├── references/           # Optional: reference documents for the agent
│   └── REFERENCE.md
├── assets/               # Optional: schemas, configs, static files
│   └── output_schema.json
└── scripts/              # Optional: Python scripts exposed as tools
    └── helper.py
```
