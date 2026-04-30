# Nono Projects

> Organise AI work into isolated, self-contained project directories.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Manifest Format](#manifest-format)
- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Resources](#resources)
- [Examples](#examples)

## Overview

A **Nono project** is a directory containing a `nono.toml` manifest file.
Projects group skills, prompts, templates, and workflows together with
their own configuration, keeping concerns isolated across codebases or
use-cases.

Key features:

| Feature | Description |
|---------|-------------|
| **Manifest-based** | Single `nono.toml` file defines the project |
| **List & discover** | `list_projects()` scans a directory for projects |
| **Resource isolation** | Skills, prompts, templates, workflows, and data per project |
| **Local config** | Optional `config.toml` with provider/model overrides |
| **CLI integration** | `nono init`, `nono project` subcommands |
| **REST API** | `GET /projects`, `GET /project/{name}` endpoints |

## Quick Start

```bash
# Scaffold a new project
nono init my-ai-project --name "My AI Project" --description "Demo project"

# Navigate into it
cd my-ai-project

# Check project info
nono project

# Add a skill, prompt, or template and start working
```

## Project Structure

After running `nono init`, you get:

```
my-ai-project/
├── nono.toml          # Project manifest (required)
├── config.toml        # Provider / model overrides (optional)
├── skills/            # SKILL.md files (Agent Skills standard)
├── prompts/           # JSON task definitions
├── templates/         # Jinja2 prompt templates
├── workflows/         # Workflow definitions (JSON / YAML)
└── data/              # Project data files (CSV, JSON, TXT, …)
```

| Directory | Glob Pattern | Purpose |
|-----------|-------------|--------|
| `skills/` | `*/SKILL.md` | Agent Skills (Anthropic standard) |
| `prompts/` | `*.json` | JSON task definitions for the Tasker |
| `templates/` | `*.j2` | Jinja2 prompt templates |
| `workflows/` | `*.json`, `*.yaml`, `*.yml` | Multi-step workflow pipelines |
| `data/` | `*` | Project data files (CSV, JSON, TXT, images, etc.) |

## Manifest Format

The `nono.toml` manifest follows standard TOML syntax:

```toml
# Nono Project Manifest

[project]
name = "my-ai-project"
description = "A demo project for text processing"
version = "0.1.0"
authors = ["Alice <alice@example.com>"]
default_provider = "google"
default_model = "gemini-3-flash-preview"

# Directory layout (relative to project root)
skills_dir = "skills"
prompts_dir = "prompts"
templates_dir = "templates"
workflows_dir = "workflows"
data_dir = "data"
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | No | directory name | Project identifier |
| `description` | string | No | `""` | One-line purpose |
| `version` | string | No | `"0.1.0"` | Semantic version |
| `authors` | list[string] | No | `[]` | `"Name <email>"` entries |
| `default_provider` | string | No | `"google"` | Default LLM provider |
| `default_model` | string | No | `""` | Default LLM model |
| `skills_dir` | string | No | `"skills"` | Skills directory (relative) |
| `prompts_dir` | string | No | `"prompts"` | Prompts directory (relative) |
| `templates_dir` | string | No | `"templates"` | Templates directory (relative) |
| `workflows_dir` | string | No | `"workflows"` | Workflows directory (relative) |
| `data_dir` | string | No | `"data"` | Data directory (relative) |

Extra TOML sections (e.g. `[providers.google]`) are preserved in `manifest.extra`.

## CLI Commands

### `nono init`

Scaffold a new project.

```bash
nono init [path] [--name NAME] [--description DESC] [--provider PROVIDER]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `path` | Directory to create | `.` (current dir) |
| `--name`, `-n` | Project name | directory name |
| `--description`, `-d` | One-line description | `""` |
| `--provider`, `-p` | Default LLM provider | `google` |

```bash
# Create in current directory
nono init --name "My Project"

# Create in a subdirectory
nono init my-project -n "My Project" -d "Text analysis pipeline" -p openai
```

### `nono project` / `nono proj`

Show information about the current project.

```bash
nono project [--path PATH]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--path` | Explicit project directory | auto-detect (walk upward) |

```bash
# Auto-detect project from cwd
nono project

# Explicit path
nono project --path /path/to/my-project
```

Output includes: name, version, description, provider, resource counts (skills, prompts, templates, workflows), and skill details.

## Python API

### `init_project()` — create a new project

```python
from nono.project import init_project

project = init_project(
    "my-project",
    name="My Project",
    description="Text analysis pipeline",
    default_provider="google",
)
# Creates: my-project/nono.toml, config.toml, skills/, prompts/, templates/, workflows/, data/
```

### `load_project()` — open an existing project

```python
from nono.project import load_project

project = load_project("my-project")
print(project.name)          # "My Project"
print(project.version)       # "0.1.0"
print(project.skills_dir)    # Path("my-project/skills")
```

### `list_projects()` — discover all projects in a directory

Scans direct children of a directory looking for `nono.toml` manifests.

```python
from nono.project import list_projects

# Scan a workspace directory
projects = list_projects("/path/to/workspace")
for p in projects:
    print(f"{p.name} v{p.version} — {p.root}")

# Defaults to cwd
projects = list_projects()
```

### `Project` class

```python
project = load_project("my-project")

# Properties
project.root           # Path — absolute project root
project.name           # str — from manifest or directory name
project.description    # str
project.version        # str — "0.1.0"
project.manifest       # ProjectManifest dataclass
project.skills_dir     # Path — resolved skills directory
project.prompts_dir    # Path — resolved prompts directory
project.templates_dir  # Path — resolved templates directory
project.workflows_dir  # Path — resolved workflows directory
project.data_dir       # Path — resolved data directory
project.config_file    # Path — path to config.toml

# Load project-local configuration
config = project.config()

# List resources
skills    = project.load_skills()        # list[MarkdownSkill]
prompts   = project.list_prompts()       # list[Path] — *.json
templates = project.list_templates()     # list[Path] — *.j2
workflows = project.list_workflows()     # list[Path] — *.json, *.yaml, *.yml

# Data resources
data_files = project.list_data()              # list[Path] — all files
csv_files  = project.list_data("*.csv")       # list[Path] — filtered by glob
all_deep   = project.list_data_recursive()    # list[Path] — recursive
content    = project.read_data("input.csv")   # str — read as text
raw        = project.read_data_bytes("img.png")  # bytes — read as binary
```

### `ProjectManifest` dataclass

```python
from nono.project import ProjectManifest

m = project.manifest
m.name               # str
m.description         # str
m.version             # str
m.authors             # list[str]
m.default_provider    # str
m.default_model       # str
m.skills_dir          # str (relative)
m.prompts_dir         # str (relative)
m.templates_dir       # str (relative)
m.workflows_dir       # str (relative)
m.data_dir            # str (relative)
m.extra               # dict — any extra TOML sections
```

## Configuration

Project configuration layering (highest to lowest priority):

1. **Environment variables** — `NONO_` prefix
2. **Project `config.toml`** — local provider/model overrides
3. **Manifest defaults** — `default_provider`, `default_model`
4. **Global defaults** — from `nono/config/config.toml`

The starter `config.toml` created by `nono init`:

```toml
# Nono project configuration
# API keys should be set via environment variables or apikey.txt,
# NOT in this file.

[agent]
default_provider = "google"

[rate_limits]
delay_between_requests = 0.5
```

## Resources

Add resources to their respective directories:

### Skills

```bash
mkdir skills/my-skill
cat > skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: Describe what this skill does.
tags:
  - text
---

# My Skill

Instructions for the AI agent.
EOF
```

### Prompts

```bash
cat > prompts/classify.json << 'EOF'
{
  "task_name": "classify",
  "system_prompt": "Classify the input text.",
  "user_prompt_template": "Classify: {text}",
  "output_schema": { "category": "string", "confidence": "float" }
}
EOF
```

### Templates

```bash
cat > templates/analysis.j2 << 'EOF'
Analyse the following {{ data_type }}:

{{ content }}

Provide a structured summary.
EOF
```

### Data

```bash
# Copy a CSV into the data directory
cp customers.csv data/

# Or create inline
cat > data/categories.json << 'EOF'
["billing", "technical_support", "shipping", "general_inquiry"]
EOF
```

Access from code:

```python
project = load_project(".")

# List all CSV files
for f in project.list_data("*.csv"):
    print(f.name)

# Read file contents
text = project.read_data("categories.json")
raw  = project.read_data_bytes("image.png")

# Recursive listing (subdirectories)
all_files = project.list_data_recursive("**/*.csv")
```

## Examples

### Full workflow: create project, add skill, run

```python
from nono.project import init_project

# 1. Create project
project = init_project("text-analysis", name="Text Analysis")

# 2. Add a skill programmatically
skill_dir = project.skills_dir / "summarise"
skill_dir.mkdir()
(skill_dir / "SKILL.md").write_text("""\
---
name: summarise
description: Summarise long documents into key points.
tags:
  - text
  - summarisation
---

# Summarise

Given a document, extract the 5 most important points.
Return a JSON array of strings.
""")

# 3. Load and inspect
skills = project.load_skills()
print(f"Loaded {len(skills)} skill(s): {skills[0].descriptor.name}")

# 4. Use project config
config = project.config()
print(f"Provider: {config.get('agent.default_provider')}")
```

### Integration with agents

```python
from nono.project import load_project
from nono.agent import Agent, Runner

project = load_project("my-project")
config = project.config()
skills = project.load_skills()

agent = Agent(
    name="analyst",
    model=config.get("agent.default_model", "gemini-3-flash-preview"),
    provider=config.get("agent.default_provider", "google"),
    instruction=skills[0].body if skills else "You are a helpful assistant.",
)

response = Runner(agent=agent).run("Analyse this data\u2026")
```

### List all projects in a workspace

```python
from nono.project import list_projects

for project in list_projects("/workspace"):
    skills = project.load_skills()
    data   = project.list_data()
    print(f"{project.name}: {len(skills)} skills, {len(data)} data files")
```

---

*Module: `nono.project` — Author: DatamanEdge — License: MIT*
