# Workspace & Manifest — Declarative Agent I/O

> Describe **what data an agent reads and writes** without coupling it to a specific execution environment.

[← Back to main README](../README.md) · [Sandbox API](sandbox/README_sandbox.md) · [Sandbox Guide](README_sandbox_guide.md)

---

## Table of Contents

- [Overview](#overview)
- [Why Workspace](#why-workspace)
- [Architecture](#architecture)
- [Part 1 — Workspace Basics](#part-1--workspace-basics)
- [Part 2 — Entry Types](#part-2--entry-types)
- [Part 3 — Cloud Storage](#part-3--cloud-storage)
- [Part 4 — Templates and Inline Data](#part-4--templates-and-inline-data)
- [Part 5 — Outputs](#part-5--outputs)
- [Part 6 — Serialisation](#part-6--serialisation)
- [Part 7 — Manifest Bridge](#part-7--manifest-bridge)
- [Part 8 — Practical Patterns](#part-8--practical-patterns)
- [Quick Reference](#quick-reference)

---

## Overview

A **Workspace** is a provider-agnostic declaration of the input files, data sources, and output destinations an agent needs.  The same `Workspace` definition works whether the agent runs:

- **Locally** — paths resolve directly on the host filesystem.
- **In a sandbox** — converted to a `Manifest` for E2B, Modal, Daytona, etc.
- **In the cloud** — cloud storage entries resolve to S3, GCS, Azure Blob, or R2.

```
┌─────────────────────────────────────┐
│              Workspace              │
│                                     │
│  inputs:                            │
│    "data"    → FileEntry            │
│    "config"  → InlineEntry          │
│    "models"  → CloudStorageEntry    │
│    "api"     → URLEntry             │
│    "prompt"  → TemplateEntry        │
│                                     │
│  outputs:                           │
│    "report"  → OutputEntry          │
│                                     │
│  metadata: {version: "1.0"}         │
└───────────────┬─────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
   Local Agent     Sandbox Agent
   (reads files    (to_manifest() →
    directly)       materialise in
                    container)
```

---

## Why Workspace

| Without Workspace | With Workspace |
|-------------------|----------------|
| Hardcoded file paths scattered in agent instructions | Centralised, named I/O declarations |
| Different code for local vs. sandbox execution | Same definition, different materialisation |
| Cloud credentials mixed with agent logic | `credentials_env` references env-vars by name |
| No structured output contract | `OutputEntry` declares what the agent produces |
| Manual JSON construction for manifests | `to_manifest()` converts automatically |

---

## Architecture

```
WorkspaceEntry (ABC)
    │
    ├── FileEntry          # Local file or directory
    ├── URLEntry           # HTTP resource
    ├── InlineEntry        # Literal embedded data
    ├── CloudStorageEntry  # S3 / GCS / Azure Blob / R2
    ├── TemplateEntry      # Jinja2 template
    └── OutputEntry        # Declared output destination

Workspace
    ├── inputs:  dict[str, WorkspaceEntry]
    ├── outputs: dict[str, OutputEntry]
    └── metadata: dict[str, Any]

Bidirectional bridge:
    Workspace.to_manifest()  →  sandbox.Manifest
    Manifest.to_workspace()  →  agent.Workspace
```

---

## Part 1 — Workspace Basics

### Creating a Workspace

```python
from nono.agent.workspace import Workspace, FileEntry, OutputEntry

ws = Workspace(
    inputs={
        "sales_data": FileEntry(path="/data/sales.csv"),
        "regions":    FileEntry(path="/data/regions/", glob="*.json"),
    },
    outputs={
        "report": OutputEntry(path="output/summary.md"),
    },
    metadata={"owner": "analytics-team", "version": "2.1"},
)

print(ws.describe())
# Workspace: 2 inputs, 1 outputs
```

### Adding and Removing Entries

```python
from nono.agent.workspace import InlineEntry

# Add dynamically
ws.add_input("threshold", InlineEntry(data=0.95, description="Alert threshold"))

# Check
assert ws.get_input("threshold") is not None
assert ws.input_names() == ["sales_data", "regions", "threshold"]

# Remove
removed = ws.remove_input("threshold")
```

### Filtered Accessors

```python
# Get only file-based inputs
files = ws.file_entries()        # dict[str, FileEntry]

# Get only cloud storage inputs
cloud = ws.cloud_entries()       # dict[str, CloudStorageEntry]

# Get only URL inputs
urls = ws.url_entries()          # dict[str, URLEntry]

# Get only inline data inputs
inline = ws.inline_entries()     # dict[str, InlineEntry]

# Get only template inputs
templates = ws.template_entries()  # dict[str, TemplateEntry]
```

---

## Part 2 — Entry Types

### FileEntry

Mount a local file or directory.

```python
from nono.agent.workspace import FileEntry

# Single file
csv = FileEntry(path="/data/sales.csv", read_only=True)

# Directory with glob filter
logs = FileEntry(
    path="/var/log/app/",
    glob="*.log",
    read_only=True,
    description="Application logs from the last 24h",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str \| Path` | — | Host filesystem path |
| `read_only` | `bool` | `True` | Prevent agent from modifying |
| `glob` | `str` | `""` | Filter files in a directory |
| `description` | `str` | `""` | Human-readable hint |

### URLEntry

Fetch a remote HTTP resource.

```python
from nono.agent.workspace import URLEntry

api_data = URLEntry(
    url="https://api.example.com/v1/metrics",
    headers={"Authorization": "Bearer ${API_TOKEN}"},
    cache=True,
    description="Daily metrics endpoint",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `str` | — | HTTP/HTTPS URL |
| `headers` | `dict[str, str]` | `{}` | Request headers (excluded from serialisation) |
| `cache` | `bool` | `True` | Cache for session lifetime |
| `description` | `str` | `""` | Human-readable hint |

> **Security**: Headers are intentionally **omitted** from `to_dict()` / `to_json()` to prevent credential leakage in serialised workspaces.

### InlineEntry

Embed literal data directly in the workspace.

```python
from nono.agent.workspace import InlineEntry

# String
raw_csv = InlineEntry(data="name,age\nAlice,30\nBob,25", content_type="text/csv")

# Dict (JSON-like)
config = InlineEntry(data={"model": "gemini-3-flash-preview", "temperature": 0.7})

# Bytes
image = InlineEntry(data=b"\x89PNG...", content_type="image/png")
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `str \| bytes \| dict \| list` | — | The payload |
| `content_type` | `str` | `"text/plain"` | MIME type hint |
| `description` | `str` | `""` | Human-readable hint |

> When `data` is `bytes`, `to_dict()` emits `data_size` instead of the raw content (to keep serialisation safe).

---

## Part 3 — Cloud Storage

### CloudStorageEntry

A unified type for S3, GCS, Azure Blob, and Cloudflare R2.

```python
from nono.agent.workspace import CloudStorageEntry, StorageKind

# AWS S3
models = CloudStorageEntry(
    kind=StorageKind.S3,
    bucket="ml-models",
    prefix="prod/v3/",
    region="eu-west-1",
    credentials_env="AWS_PROFILE",
    read_only=True,
)

# Google Cloud Storage
datasets = CloudStorageEntry(
    kind=StorageKind.GCS,
    bucket="data-lake",
    prefix="2026/Q1/",
    credentials_env="GOOGLE_APPLICATION_CREDENTIALS",
)

# Azure Blob Storage
reports = CloudStorageEntry(
    kind=StorageKind.AZURE_BLOB,
    bucket="financial-reports",   # = container name
    prefix="quarterly/",
    credentials_env="AZURE_STORAGE_CONNECTION_STRING",
)

# Cloudflare R2
assets = CloudStorageEntry(
    kind=StorageKind.CLOUDFLARE_R2,
    bucket="static-assets",
    prefix="images/",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kind` | `StorageKind` | — | S3, GCS, AZURE_BLOB, CLOUDFLARE_R2 |
| `bucket` | `str` | — | Bucket / container name |
| `prefix` | `str` | `""` | Key prefix to scope |
| `region` | `str` | `""` | Cloud region |
| `credentials_env` | `str` | `""` | Env-var holding credentials |
| `read_only` | `bool` | `True` | Prevent writes |
| `description` | `str` | `""` | Human-readable hint |

---

## Part 4 — Templates and Inline Data

### TemplateEntry

A Jinja2 template rendered at resolve time with session state or explicit variables.

```python
from nono.agent.workspace import TemplateEntry

prompt = TemplateEntry(
    template="Analyse {{ metric }} for {{ region }} in {{ period }}.",
    variables={"metric": "revenue", "region": "EMEA", "period": "Q1 2026"},
    description="Dynamic analysis prompt",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template` | `str` | — | Jinja2 template string |
| `variables` | `dict[str, Any]` | `{}` | Default variable values |
| `description` | `str` | `""` | Human-readable hint |

### Combining Entry Types

```python
from nono.agent.workspace import (
    Workspace, FileEntry, URLEntry, InlineEntry,
    CloudStorageEntry, StorageKind, TemplateEntry, OutputEntry,
)

ws = Workspace(
    inputs={
        "raw_data":  FileEntry(path="/data/raw/"),
        "api_feed":  URLEntry(url="https://api.example.com/feed"),
        "config":    InlineEntry(data={"threshold": 0.9}),
        "models":    CloudStorageEntry(kind=StorageKind.S3, bucket="ml"),
        "prompt":    TemplateEntry(template="Summarise {{ topic }}."),
    },
    outputs={
        "report":    OutputEntry(path="output/report.md"),
        "dashboard": OutputEntry(path="output/dash.html", content_type="text/html"),
    },
)
```

---

## Part 5 — Outputs

### OutputEntry

Declares where the agent should write results.

```python
from nono.agent.workspace import OutputEntry

report = OutputEntry(
    path="output/quarterly_report.pdf",
    content_type="application/pdf",
    description="Final quarterly report with charts",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | `"output"` | Relative or absolute output path |
| `content_type` | `str` | `"application/octet-stream"` | Expected MIME type |
| `description` | `str` | `""` | Human-readable hint |

Properties:

| Property | Value |
|----------|-------|
| `is_readable` | `False` |
| `is_writable` | `True` |

---

## Part 6 — Serialisation

### to_dict / from_dict

```python
ws = Workspace(inputs={"data": FileEntry(path="/tmp/data")})

# Serialise
d = ws.to_dict()
# {'inputs': {'data': {'type': 'file', 'path': '/tmp/data', 'read_only': True}},
#  'outputs': {}, 'metadata': {}}

# Deserialise
restored = Workspace.from_dict(d)
assert restored.input_names() == ["data"]
```

### to_json / from_json

```python
# Full JSON round-trip
json_str = ws.to_json()
restored = Workspace.from_json(json_str)
```

### Deserialisation Safety

- Unknown entry types are **logged and skipped** — no crash on forward-compatible schemas.
- `URLEntry.headers` are **never serialised** to prevent credential leakage.
- `InlineEntry` with `bytes` data emits `data_size` instead of raw content.

---

## Part 7 — Manifest Bridge

### Workspace → Manifest (for sandbox execution)

```python
from nono.agent.workspace import (
    Workspace, FileEntry, CloudStorageEntry, StorageKind, OutputEntry,
)

ws = Workspace(
    inputs={
        "data":   FileEntry(path="/tmp/reports"),
        "models": CloudStorageEntry(kind=StorageKind.S3, bucket="ml-models"),
    },
    outputs={
        "result": OutputEntry(path="results"),
    },
)

# Convert to sandbox Manifest
manifest = ws.to_manifest()

# Use with SandboxAgent
from nono.sandbox import SandboxAgent, SandboxRunConfig, SandboxProvider

agent = SandboxAgent(
    name="Analyst",
    sandbox_config=SandboxRunConfig(
        provider=SandboxProvider.E2B,
        manifest=manifest,
    ),
)
```

### Manifest → Workspace (from sandbox to agent-level)

```python
from nono.sandbox.manifest import Manifest, LocalDir, S3Bucket, OutputDir

manifest = Manifest(
    entries={
        "data":   LocalDir(src="/tmp/data"),
        "models": S3Bucket(bucket="ml-models", prefix="v2/"),
    },
    output=OutputDir(path="output"),
)

# Convert to Workspace
ws = manifest.to_workspace()

print(ws.input_names())   # ['data', 'models']
print(ws.output_names())  # ['output']
```

### Conversion Table

| Workspace Type | → Manifest Type | ← Manifest Type |
|----------------|-----------------|------------------|
| `FileEntry` (file) | `LocalFile` | `LocalFile` → `FileEntry` |
| `FileEntry` (dir) | `LocalDir` | `LocalDir` → `FileEntry` |
| `CloudStorageEntry(S3)` | `S3Bucket` | `S3Bucket` → `CloudStorageEntry(S3)` |
| `CloudStorageEntry(GCS)` | `GCSBucket` | `GCSBucket` → `CloudStorageEntry(GCS)` |
| `CloudStorageEntry(AZURE_BLOB)` | `AzureBlob` | `AzureBlob` → `CloudStorageEntry(AZURE_BLOB)` |
| `CloudStorageEntry(CLOUDFLARE_R2)` | `CloudflareR2` | `CloudflareR2` → `CloudStorageEntry(CLOUDFLARE_R2)` |
| `OutputEntry` | `OutputDir` | `OutputDir` → `OutputEntry` |
| `URLEntry` | *(not mapped)* | — |
| `InlineEntry` | *(not mapped)* | — |
| `TemplateEntry` | *(not mapped)* | — |

> `URLEntry`, `InlineEntry`, and `TemplateEntry` are **agent-level only** — they have no sandbox-filesystem equivalent.  They are resolved by the agent before code reaches the sandbox.

---

## Part 8 — Practical Patterns

### Pattern 1: Data Analysis Pipeline

```python
from nono.agent.workspace import (
    Workspace, FileEntry, InlineEntry, OutputEntry,
)

ws = Workspace(
    inputs={
        "raw_csv":   FileEntry(path="/data/sales_2026.csv"),
        "schema":    InlineEntry(data={
            "columns": ["date", "region", "amount", "product"],
            "date_format": "%Y-%m-%d",
        }),
    },
    outputs={
        "summary":   OutputEntry(path="output/summary.json", content_type="application/json"),
        "chart":     OutputEntry(path="output/chart.png", content_type="image/png"),
    },
    metadata={"pipeline": "quarterly-analysis", "version": "3.0"},
)
```

### Pattern 2: Multi-Cloud ETL Agent

```python
from nono.agent.workspace import (
    Workspace, CloudStorageEntry, StorageKind, OutputEntry,
)

ws = Workspace(
    inputs={
        "source_s3":   CloudStorageEntry(
            kind=StorageKind.S3, bucket="raw-events", prefix="2026/04/",
            region="us-east-1", credentials_env="AWS_PROFILE",
        ),
        "reference_gcs": CloudStorageEntry(
            kind=StorageKind.GCS, bucket="reference-data",
            credentials_env="GOOGLE_APPLICATION_CREDENTIALS",
        ),
    },
    outputs={
        "transformed": OutputEntry(path="output/parquet/"),
    },
)

# Run locally or convert for sandbox
manifest = ws.to_manifest()
```

### Pattern 3: Durable Sandbox with Workspace

```python
from nono.agent.workspace import Workspace, FileEntry, OutputEntry
from nono.sandbox import (
    DurableSandboxAgent, SandboxRunConfig, SandboxProvider,
)

ws = Workspace(
    inputs={"data": FileEntry(path="/data/large_dataset.parquet")},
    outputs={"result": OutputEntry(path="output/processed.csv")},
)

agent = DurableSandboxAgent(
    name="ResilientETL",
    sandbox_config=SandboxRunConfig(
        provider=SandboxProvider.E2B,
        manifest=ws.to_manifest(),
        snapshot=True,
    ),
    max_retries=3,
)
```

### Pattern 4: Dynamic Workspace from Session State

```python
from nono.agent.workspace import Workspace, TemplateEntry, FileEntry, OutputEntry

def build_workspace(session_state: dict) -> Workspace:
    """Build a workspace dynamically from session state."""
    return Workspace(
        inputs={
            "instructions": TemplateEntry(
                template="Analyse {{ metric }} for {{ region }}.",
                variables={
                    "metric": session_state.get("metric", "revenue"),
                    "region": session_state.get("region", "global"),
                },
            ),
            "data": FileEntry(path=session_state["data_path"]),
        },
        outputs={
            "report": OutputEntry(path=f"output/{session_state['run_id']}.md"),
        },
    )
```

### Pattern 5: Workspace as Agent Configuration

```python
import json

# Save workspace config to file
ws.to_json()  # → send to config store, database, or API

# Load and reconstruct later
with open("workspace_config.json") as f:
    ws = Workspace.from_json(f.read())
```

---

## Quick Reference

### Entry Types

| Type | `entry_type()` | Readable | Writable | Mapped to Manifest |
|------|----------------|----------|----------|--------------------|
| `FileEntry` | `"file"` | Yes | No | `LocalDir` / `LocalFile` |
| `URLEntry` | `"url"` | Yes | No | No |
| `InlineEntry` | `"inline"` | Yes | No | No |
| `CloudStorageEntry` | `"s3"` / `"gcs"` / `"azure_blob"` / `"cloudflare_r2"` | Yes | No | `S3Bucket` / `GCSBucket` / `AzureBlob` / `CloudflareR2` |
| `TemplateEntry` | `"template"` | Yes | No | No |
| `OutputEntry` | `"output"` | No | Yes | `OutputDir` |

### Workspace API

| Method | Returns | Description |
|--------|---------|-------------|
| `add_input(name, entry)` | `None` | Add or replace an input |
| `add_output(name, entry)` | `None` | Add or replace an output |
| `remove_input(name)` | `entry \| None` | Remove an input |
| `remove_output(name)` | `entry \| None` | Remove an output |
| `get_input(name)` | `entry \| None` | Look up an input |
| `get_output(name)` | `entry \| None` | Look up an output |
| `input_names()` | `list[str]` | All input names |
| `output_names()` | `list[str]` | All output names |
| `describe()` | `str` | One-line summary |
| `file_entries()` | `dict[str, FileEntry]` | Only file inputs |
| `url_entries()` | `dict[str, URLEntry]` | Only URL inputs |
| `inline_entries()` | `dict[str, InlineEntry]` | Only inline inputs |
| `cloud_entries()` | `dict[str, CloudStorageEntry]` | Only cloud inputs |
| `template_entries()` | `dict[str, TemplateEntry]` | Only template inputs |
| `to_dict()` | `dict` | JSON-safe serialisation |
| `to_json()` | `str` | JSON string |
| `from_dict(d)` | `Workspace` | Deserialise from dict |
| `from_json(s)` | `Workspace` | Deserialise from JSON |
| `to_manifest()` | `Manifest` | Convert to sandbox manifest |

### Enums

| Enum | Values |
|------|--------|
| `IODirection` | `INPUT`, `OUTPUT` |
| `StorageKind` | `S3`, `GCS`, `AZURE_BLOB`, `CLOUDFLARE_R2` |

### Imports

```python
# All types from agent.workspace
from nono.agent.workspace import (
    Workspace, WorkspaceEntry,
    FileEntry, URLEntry, InlineEntry,
    CloudStorageEntry, StorageKind,
    TemplateEntry, OutputEntry, IODirection,
)

# Or from the agent package
from nono.agent import (
    Workspace, FileEntry, OutputEntry, CloudStorageEntry, StorageKind,
)
```
