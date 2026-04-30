# Multi-Provider Sandbox Execution Guide

> Run agent code in isolated external environments — switch between E2B, Modal, Daytona, Blaxel, Cloudflare, Runloop, and Vercel with a single config change.

[← Back to main README](../README.md) · [API Reference](sandbox/README_sandbox.md) · [Workspace & Manifest](README_workspace.md)

---

## Table of Contents

- [Overview](#overview)
- [Why Sandbox Execution](#why-sandbox-execution)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Part 1 — Getting Started](#part-1--getting-started)
  - [1.1 Install a Provider SDK](#11-install-a-provider-sdk)
  - [1.2 Set API Keys](#12-set-api-keys)
  - [1.3 Your First Sandbox Execution](#13-your-first-sandbox-execution)
- [Part 2 — The Manifest System](#part-2--the-manifest-system)
  - [2.1 Local Files and Directories](#21-local-files-and-directories)
  - [2.2 AWS S3](#22-aws-s3)
  - [2.3 Google Cloud Storage](#23-google-cloud-storage)
  - [2.4 Azure Blob Storage](#24-azure-blob-storage)
  - [2.5 Cloudflare R2](#25-cloudflare-r2)
  - [2.6 Output Collection](#26-output-collection)
  - [2.7 Combining Multiple Sources](#27-combining-multiple-sources)
- [Part 3 — Provider Deep Dive](#part-3--provider-deep-dive)
  - [3.1 E2B — Cloud Code Interpreters](#31-e2b--cloud-code-interpreters)
  - [3.2 Modal — Serverless Containers](#32-modal--serverless-containers)
  - [3.3 Daytona — Dev Environments](#33-daytona--dev-environments)
  - [3.4 Blaxel — Cloud Sandboxes](#34-blaxel--cloud-sandboxes)
  - [3.5 Cloudflare — Workers Containers](#35-cloudflare--workers-containers)
  - [3.6 Runloop — AI Dev-Boxes](#36-runloop--ai-dev-boxes)
  - [3.7 Vercel — Serverless Sandboxes](#37-vercel--serverless-sandboxes)
- [Part 4 — SandboxAgent Integration](#part-4--sandboxagent-integration)
  - [4.1 Standalone SandboxAgent](#41-standalone-sandboxagent)
  - [4.2 SandboxAgent as Sub-Agent](#42-sandboxagent-as-sub-agent)
  - [4.3 Output Files in SharedContent](#43-output-files-in-sharedcontent)
- [Part 5 — Switching Providers at Runtime](#part-5--switching-providers-at-runtime)
  - [5.1 Config-Driven Provider Selection](#51-config-driven-provider-selection)
  - [5.2 Provider Fallback Pattern](#52-provider-fallback-pattern)
  - [5.3 Environment-Based Routing](#53-environment-based-routing)
- [Part 6 — Advanced Patterns](#part-6--advanced-patterns)
  - [6.1 Parallel Execution Across Providers](#61-parallel-execution-across-providers)
  - [6.2 Custom Sandbox Client](#62-custom-sandbox-client)
  - [6.3 Manifest Serialisation](#63-manifest-serialisation)
- [Part 7 — Security Considerations](#part-7--security-considerations)
- [Part 8 — Configuration Reference](#part-8--configuration-reference)
- [Comparison: When to Use Each Provider](#comparison-when-to-use-each-provider)

---

## Overview

The Nono sandbox module separates **what code runs** from **where it runs**. You write one agent, one manifest, and one config — then swap the execution environment by changing a single enum value.

```
Your Agent Code  →  Manifest (workspace)  →  SandboxRunConfig (provider)  →  External Sandbox
```

All seven providers implement the same `BaseSandboxClient` interface, so your agent logic never depends on a specific cloud vendor.

---

## Why Sandbox Execution

| Problem | How Sandboxing Solves It |
|---------|--------------------------|
| Agent-generated code running on production servers | Code executes in an isolated container, not on your machine |
| Credential leaks via prompt injection | API keys stay in the harness — the sandbox has no access |
| Uncontrolled pip installs or system modifications | Sandbox is ephemeral — destroyed after execution |
| Long-running tasks blocking the main process | Sandboxes run remotely; your process remains responsive |
| Need to test same code on different infra | Swap `SandboxProvider.E2B` for `SandboxProvider.MODAL` — nothing else changes |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Your Application                     │
│                                                          │
│   LlmAgent  ──▶  SandboxAgent  ──▶  get_sandbox_client()│
│                        │                     │           │
│                   SandboxRunConfig       Provider enum   │
│                   + Manifest             resolution      │
└──────────────────────┬───────────────────────┬───────────┘
                       │                       │
         ┌─────────────┼─────────────┐         │
         ▼             ▼             ▼         ▼
    ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐
    │   E2B   │  │  Modal   │  │ Daytona │  │ Blaxel   │
    └─────────┘  └──────────┘  └─────────┘  └──────────┘
    ┌──────────────┐  ┌──────────┐  ┌──────────┐
    │ Cloudflare   │  │ Runloop  │  │  Vercel  │
    └──────────────┘  └──────────┘  └──────────┘
```

Each client follows the same contract:

1. **Provision** — create or reuse a sandbox instance
2. **Materialise** — upload Manifest entries into the sandbox filesystem
3. **Execute** — run the code
4. **Collect** — gather stdout, stderr, exit code, and output files
5. **Clean up** — destroy the sandbox (unless `keep_alive=True`)

---

## Prerequisites

- Python >= 3.13
- Nono framework installed
- An account and API key for at least one sandbox provider

---

## Part 1 — Getting Started

### 1.1 Install a Provider SDK

Sandbox provider SDKs are **optional dependencies** — install only what you need:

```bash
# Pick one (or several)
pip install e2b-code-interpreter   # E2B
pip install modal                  # Modal
pip install daytona-sdk            # Daytona
pip install blaxel                 # Blaxel
pip install cloudflare             # Cloudflare
pip install runloop-api-client     # Runloop
pip install httpx                  # Vercel (httpx is likely already installed)
```

### 1.2 Set API Keys

Each provider reads credentials from environment variables. **Never hardcode keys in source code.**

```bash
# Linux / macOS
export E2B_API_KEY="your-e2b-key"
export MODAL_TOKEN_ID="your-modal-id"
export MODAL_TOKEN_SECRET="your-modal-secret"
export DAYTONA_API_KEY="your-daytona-key"
export BLAXEL_API_KEY="your-blaxel-key"
export CLOUDFLARE_API_TOKEN="your-cf-token"
export CLOUDFLARE_ACCOUNT_ID="your-cf-account"
export RUNLOOP_API_KEY="your-runloop-key"
export VERCEL_TOKEN="your-vercel-token"
```

```powershell
# Windows PowerShell
$env:E2B_API_KEY = "your-e2b-key"
```

### 1.3 Your First Sandbox Execution

```python
from nono.sandbox import get_sandbox_client, SandboxRunConfig
from nono.sandbox.base import SandboxProvider

# Create a client for E2B
client = get_sandbox_client(SandboxProvider.E2B)

# Run code
result = client.execute(
    "print('Hello from the sandbox!')",
    SandboxRunConfig(timeout=60),
)

print(result.stdout)     # "Hello from the sandbox!"
print(result.success)    # True
print(result.exit_code)  # 0
```

To switch to Modal, change **one line**:

```python
client = get_sandbox_client(SandboxProvider.MODAL)
```

Everything else stays the same.

---

## Part 2 — The Manifest System

A `Manifest` tells the sandbox **what files the agent needs**. It maps logical mount paths (what the code sees inside the sandbox) to concrete sources on your host or in the cloud.

### 2.1 Local Files and Directories

```python
from nono.sandbox.manifest import Manifest, LocalDir, LocalFile, OutputDir

manifest = Manifest(
    entries={
        # Mount an entire directory
        "data": LocalDir(src="/home/user/project/data", read_only=True),

        # Mount a single file
        "config": LocalFile(src="/home/user/project/settings.yaml", read_only=True),
    },
    output=OutputDir(path="results"),
)
```

Inside the sandbox the agent sees:

```
/home/user/
├── data/           ← contents of /home/user/project/data
├── config/         ← contains settings.yaml
└── results/        ← agent writes output here
```

### 2.2 AWS S3

```python
from nono.sandbox.manifest import S3Bucket

manifest = Manifest(entries={
    "training_data": S3Bucket(
        bucket="my-ml-bucket",
        prefix="datasets/v3/",
        region="eu-west-1",
        credentials_profile="prod",  # Named AWS CLI profile
        read_only=True,
    ),
})
```

### 2.3 Google Cloud Storage

```python
from nono.sandbox.manifest import GCSBucket

manifest = Manifest(entries={
    "models": GCSBucket(
        bucket="my-gcs-models",
        prefix="checkpoints/",
        project="my-gcp-project",
        credentials_json="/path/to/service-account.json",
    ),
})
```

### 2.4 Azure Blob Storage

```python
from nono.sandbox.manifest import AzureBlob

manifest = Manifest(entries={
    "reports": AzureBlob(
        container="quarterly-reports",
        prefix="2026/Q1/",
        account_name="mycompanystorage",
        connection_string_env="AZURE_STORAGE_CONN",  # env var name
    ),
})
```

### 2.5 Cloudflare R2

```python
from nono.sandbox.manifest import CloudflareR2

manifest = Manifest(entries={
    "assets": CloudflareR2(
        bucket="media-assets",
        prefix="images/",
        account_id="abc123",
        access_key_id_env="R2_ACCESS_KEY",
        secret_access_key_env="R2_SECRET_KEY",
    ),
})
```

### 2.6 Output Collection

Declare an `OutputDir` to tell the sandbox client which directory to collect after execution:

```python
from nono.sandbox.manifest import Manifest, LocalDir, OutputDir

manifest = Manifest(
    entries={"data": LocalDir(src="/tmp/input")},
    output=OutputDir(path="output"),
)

# After execution:
result = client.execute(code, SandboxRunConfig(manifest=manifest))

for filename, content in result.output_files.items():
    print(f"{filename}: {len(content)} bytes")
    with open(f"/tmp/collected/{filename}", "wb") as f:
        f.write(content)
```

### 2.7 Combining Multiple Sources

A single manifest can mix local and cloud sources:

```python
manifest = Manifest(
    entries={
        "raw_data": S3Bucket(bucket="data-lake", prefix="raw/2026/"),
        "config": LocalFile(src="./pipeline.yaml", read_only=True),
        "reference": GCSBucket(bucket="ref-db", prefix="v2/"),
        "templates": AzureBlob(container="templates", prefix="reports/"),
    },
    output=OutputDir(path="results"),
    metadata={"pipeline": "quarterly-analysis", "version": "2.1"},
)
```

---

## Part 3 — Provider Deep Dive

### 3.1 E2B — Cloud Code Interpreters

Best for: **data analysis, code generation, Jupyter-like execution**

```python
from nono.sandbox.clients.e2b import E2BSandboxClient

client = E2BSandboxClient(template="base")

result = client.execute(
    """
import pandas as pd
df = pd.DataFrame({"revenue": [100, 200, 300]})
print(df.describe().to_string())
    """,
    SandboxRunConfig(
        packages=["pandas"],
        timeout=120,
    ),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `e2b-code-interpreter` |
| Env var | `E2B_API_KEY` |
| Filesystem upload | Yes — local files uploaded via API |
| Output collection | Yes — reads from output directory |
| Custom template | `E2BSandboxClient(template="my-template")` |

### 3.2 Modal — Serverless Containers

Best for: **GPU workloads, ML inference, heavy compute**

```python
from nono.sandbox.clients.modal import ModalSandboxClient

client = ModalSandboxClient(image="python:3.12")

result = client.execute(
    "import torch; print(torch.cuda.is_available())",
    SandboxRunConfig(packages=["torch"]),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `modal` |
| Env var | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| GPU support | Yes (via Modal image configuration) |
| Auto-scaling | Built-in |

### 3.3 Daytona — Dev Environments

Best for: **full development environments, multi-language support**

```python
from nono.sandbox.clients.daytona import DaytonaSandboxClient

client = DaytonaSandboxClient(target="us")

result = client.execute(
    "print('Daytona dev environment')",
    SandboxRunConfig(packages=["requests"]),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `daytona-sdk` |
| Env var | `DAYTONA_API_KEY` |
| Full dev env | Yes — persistent workspace |
| Multi-language | Python, Node.js, Go, etc. |

### 3.4 Blaxel — Cloud Sandboxes

Best for: **agent-native sandboxing, AI-first workflows**

```python
from nono.sandbox.clients.blaxel import BlaxelSandboxClient

client = BlaxelSandboxClient()

result = client.execute(
    "print('Running in Blaxel')",
    SandboxRunConfig(environment={"MODE": "production"}),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `blaxel` |
| Env var | `BLAXEL_API_KEY` |
| Agent-native | Designed for AI agent workloads |

### 3.5 Cloudflare — Workers Containers

Best for: **edge execution, low-latency, globally distributed**

```python
from nono.sandbox.clients.cloudflare import CloudflareSandboxClient

client = CloudflareSandboxClient()

result = client.execute(
    "print('Edge computing with Cloudflare')",
    SandboxRunConfig(timeout=30),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `cloudflare` |
| Env var | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` |
| Edge locations | Global network |
| Container image | `python:3.12-slim` (default) |

### 3.6 Runloop — AI Dev-Boxes

Best for: **AI coding agents, persistent dev environments**

```python
from nono.sandbox.clients.runloop import RunloopSandboxClient

client = RunloopSandboxClient(blueprint="default")

result = client.execute(
    "print('Runloop devbox')",
    SandboxRunConfig(packages=["numpy"]),
)
```

| Feature | Detail |
|---------|--------|
| SDK | `runloop-api-client` |
| Env var | `RUNLOOP_API_KEY` |
| Persistent devbox | Yes — with blueprints |
| AI-optimised | Built for coding agents |

### 3.7 Vercel — Serverless Sandboxes

Best for: **web-oriented execution, quick prototyping**

```python
from nono.sandbox.clients.vercel import VercelSandboxClient

client = VercelSandboxClient()

result = client.execute(
    "print('Vercel sandbox')",
    SandboxRunConfig(timeout=60),
)
```

| Feature | Detail |
|---------|--------|
| HTTP client | `httpx` (no custom SDK) |
| Env var | `VERCEL_TOKEN` (optional: `VERCEL_TEAM_ID`) |
| REST API | Standard Vercel Sandbox API |

---

## Part 4 — SandboxAgent Integration

### 4.1 Standalone SandboxAgent

`SandboxAgent` is a full Nono `BaseAgent` — it receives code as input and executes it in the configured sandbox:

```python
from nono.sandbox import SandboxAgent, SandboxRunConfig, Manifest, LocalDir
from nono.sandbox.base import SandboxProvider
from nono.agent.base import InvocationContext, Session

agent = SandboxAgent(
    name="DataAnalyst",
    sandbox_config=SandboxRunConfig(
        provider=SandboxProvider.E2B,
        packages=["pandas", "matplotlib"],
        manifest=Manifest(entries={
            "data": LocalDir(src="/tmp/reports"),
        }),
    ),
)

# Execute via Runner or directly
session = Session()
ctx = InvocationContext(session=session, user_message="""
import pandas as pd
df = pd.read_csv('/home/user/data/sales.csv')
print(df.groupby('region').sum().to_string())
""")

response = agent.run(ctx)
print(response)
```

### 4.2 SandboxAgent as Sub-Agent

Pair a `SandboxAgent` with an `LlmAgent` so the LLM generates code and the sandbox executes it:

```python
from nono.agent import LlmAgent
from nono.sandbox import SandboxAgent, SandboxRunConfig
from nono.sandbox.base import SandboxProvider

# The executor — runs code in E2B
executor = SandboxAgent(
    name="CodeRunner",
    sandbox_config=SandboxRunConfig(
        provider=SandboxProvider.E2B,
        packages=["pandas", "matplotlib"],
    ),
)

# The planner — generates code, delegates to CodeRunner
planner = LlmAgent(
    name="Analyst",
    model="gemini-3-flash-preview",
    instructions=(
        "You are a data analyst. When you need to run code, "
        "delegate to CodeRunner with the Python code to execute."
    ),
    sub_agents=[executor],
)
```

### 4.3 Output Files in SharedContent

When the sandbox produces output files (via `OutputDir`), they are automatically stored in the session's `SharedContent`:

```python
# After SandboxAgent runs, output files are accessible:
report = session.shared_content.load("sandbox:report.csv")
if report:
    print(f"Report size: {len(report.data)} bytes")
    
    # Write to disk
    with open("report.csv", "wb") as f:
        f.write(report.data)
```

Files are stored with the key pattern `sandbox:<filename>`.

---

## Part 5 — Switching Providers at Runtime

### 5.1 Config-Driven Provider Selection

Read the provider from `config.toml`:

```toml
# config.toml
[sandbox]
default_provider = "e2b"
timeout = 300
```

```python
from nono.config import NonoConfig
from nono.sandbox import get_sandbox_client, SandboxRunConfig
from nono.sandbox.base import SandboxProvider

config = NonoConfig()
provider_name = config.get("sandbox.default_provider", "e2b")
provider = SandboxProvider(provider_name)

client = get_sandbox_client(provider)
result = client.execute("print('dynamic provider')", SandboxRunConfig())
```

### 5.2 Provider Fallback Pattern

Try one provider, fall back to another on failure:

```python
from nono.sandbox import get_sandbox_client, SandboxRunConfig
from nono.sandbox.base import SandboxProvider, SandboxStatus

FALLBACK_CHAIN = [
    SandboxProvider.E2B,
    SandboxProvider.MODAL,
    SandboxProvider.DAYTONA,
]

def execute_with_fallback(code: str, config: SandboxRunConfig) -> "SandboxResult":
    """Try providers in order until one succeeds."""
    last_result = None

    for provider in FALLBACK_CHAIN:
        try:
            client = get_sandbox_client(provider)
            config.provider = provider
            result = client.execute(code, config)

            if result.success:
                return result

            last_result = result
        except Exception as exc:
            print(f"{provider.value} failed: {exc}")
            continue

    # All providers failed — return the last result
    return last_result

result = execute_with_fallback(
    "print('resilient execution')",
    SandboxRunConfig(timeout=60),
)
```

### 5.3 Environment-Based Routing

Use different providers per deployment stage:

```python
import os
from nono.sandbox.base import SandboxProvider

_ENV_PROVIDER_MAP = {
    "development": SandboxProvider.E2B,        # Quick iteration
    "staging": SandboxProvider.MODAL,          # GPU testing
    "production": SandboxProvider.CLOUDFLARE,  # Edge performance
}

def get_provider() -> SandboxProvider:
    env = os.environ.get("DEPLOY_ENV", "development")
    return _ENV_PROVIDER_MAP.get(env, SandboxProvider.E2B)
```

---

## Part 6 — Advanced Patterns

### 6.1 Parallel Execution Across Providers

Run the same code on multiple providers simultaneously (useful for benchmarking or redundancy):

```python
from concurrent.futures import ThreadPoolExecutor
from nono.sandbox import get_sandbox_client, SandboxRunConfig
from nono.sandbox.base import SandboxProvider

providers = [SandboxProvider.E2B, SandboxProvider.MODAL, SandboxProvider.DAYTONA]
code = "import sys; print(f'Python {sys.version}')"

def run_on(provider):
    client = get_sandbox_client(provider)
    return provider.value, client.execute(code, SandboxRunConfig(timeout=60))

with ThreadPoolExecutor(max_workers=3) as pool:
    results = dict(pool.map(run_on, providers))

for name, result in results.items():
    print(f"{name}: {result.stdout.strip()} ({result.duration_seconds:.1f}s)")
```

### 6.2 Custom Sandbox Client

Implement `BaseSandboxClient` to add your own provider:

```python
from nono.sandbox.base import BaseSandboxClient, SandboxResult, SandboxRunConfig, SandboxStatus

class DockerSandboxClient(BaseSandboxClient):
    """Run code in a local Docker container."""

    def __init__(self, image: str = "python:3.12-slim") -> None:
        super().__init__()
        self._image = image

    def execute(self, code: str, config: SandboxRunConfig) -> SandboxResult:
        import subprocess, time

        start = time.monotonic()
        proc = subprocess.run(
            ["docker", "run", "--rm", self._image, "python", "-c", code],
            capture_output=True, text=True, timeout=config.timeout,
        )

        return SandboxResult(
            status=SandboxStatus.COMPLETED if proc.returncode == 0 else SandboxStatus.FAILED,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_seconds=time.monotonic() - start,
        )

# Use it directly
client = DockerSandboxClient()
result = client.execute("print('Docker sandbox')", SandboxRunConfig())

# Or inject it into SandboxAgent
from nono.sandbox import SandboxAgent

agent = SandboxAgent(name="DockerRunner", client=client)
```

### 6.3 Manifest Serialisation

Manifests can be serialised to JSON for storage, logging, or transmission:

```python
import json
from nono.sandbox.manifest import Manifest, LocalDir, S3Bucket, OutputDir

manifest = Manifest(
    entries={
        "data": LocalDir(src="/tmp/data"),
        "models": S3Bucket(bucket="ml-models", prefix="v2/"),
    },
    output=OutputDir(path="results"),
    metadata={"pipeline": "analysis", "created_by": "data-team"},
)

# Serialise
manifest_json = json.dumps(manifest.to_dict(), indent=2)
print(manifest_json)

# Output:
# {
#   "entries": {
#     "data": {"type": "local_dir", "src": "/tmp/data", "read_only": false},
#     "models": {"type": "s3", "bucket": "ml-models", "prefix": "v2/", "read_only": false}
#   },
#   "output": {"type": "output_dir", "path": "results"},
#   "metadata": {"pipeline": "analysis", "created_by": "data-team"}
# }
```

---

## Part 7 — Security Considerations

| Concern | Mitigation |
|---------|------------|
| **Credential isolation** | API keys remain in the harness process. The sandbox only receives what you explicitly pass via `environment` |
| **Code injection** | Sandboxes are ephemeral — destroyed after execution. No persistent access to host systems |
| **Manifest read-only** | Set `read_only=True` on entries that should not be modified by agent code |
| **Timeout enforcement** | Always set `timeout` to prevent infinite loops consuming provider credits |
| **Network access** | Most providers restrict outbound network by default. Check your provider's docs for egress controls |
| **Secret env vars** | Only pass necessary variables via `SandboxRunConfig.environment`. Never pass `*_API_KEY` vars to the sandbox |
| **Output validation** | Always validate `output_files` content before writing to your filesystem or databases |

```python
# Good — only pass what the sandbox needs
config = SandboxRunConfig(
    environment={
        "DATA_SOURCE": "s3://my-bucket/data/",
        "LOG_LEVEL": "INFO",
    },
)

# Bad — leaking credentials into the sandbox
config = SandboxRunConfig(
    environment=dict(os.environ),  # NEVER do this
)
```

---

## Part 8 — Configuration Reference

### config.toml

```toml
[sandbox]
# Default provider: e2b | modal | daytona | blaxel | cloudflare | runloop | vercel
default_provider = "e2b"

# Default execution timeout in seconds
timeout = 300

# Working directory inside the sandbox
working_dir = "/home/user"

# Keep sandbox alive after execution for inspection / debugging
keep_alive = false

# Enable state snapshots for durability (provider must support it)
snapshot = false
```

### SandboxRunConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `SandboxProvider` | `E2B` | Target provider |
| `timeout` | `int` | `300` | Max execution time (seconds) |
| `environment` | `dict[str, str]` | `{}` | Env vars for the sandbox |
| `packages` | `list[str]` | `[]` | `pip install` before execution |
| `working_dir` | `str` | `/home/user` | Working directory |
| `manifest` | `Manifest \| None` | `None` | Workspace description |
| `keep_alive` | `bool` | `False` | Keep sandbox running |
| `snapshot` | `bool` | `False` | Enable state snapshots |
| `metadata` | `dict` | `{}` | Provider-specific options |

### Environment Variables by Provider

| Provider | Required | Optional |
|----------|----------|----------|
| E2B | `E2B_API_KEY` | — |
| Modal | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | — |
| Daytona | `DAYTONA_API_KEY` | — |
| Blaxel | `BLAXEL_API_KEY` | — |
| Cloudflare | `CLOUDFLARE_API_TOKEN` | `CLOUDFLARE_ACCOUNT_ID` |
| Runloop | `RUNLOOP_API_KEY` | — |
| Vercel | `VERCEL_TOKEN` | `VERCEL_TEAM_ID` |

---

## Comparison: When to Use Each Provider

| Provider | Best For | Cold Start | GPU | Persistent |
|----------|----------|------------|-----|------------|
| **E2B** | Data analysis, code interpretation | Fast | No | No |
| **Modal** | ML inference, GPU workloads | Medium | Yes | No |
| **Daytona** | Full dev environments | Medium | No | Yes |
| **Blaxel** | Agent-native workflows | Fast | No | No |
| **Cloudflare** | Edge, low-latency, global | Very fast | No | No |
| **Runloop** | AI coding agents | Medium | No | Yes |
| **Vercel** | Web apps, quick prototyping | Fast | No | No |

### Decision Tree

```
Need GPU?
├── Yes → Modal
└── No
    ├── Need persistent environment?
    │   ├── Yes → Daytona or Runloop
    │   └── No
    │       ├── Need global edge distribution?
    │       │   ├── Yes → Cloudflare
    │       │   └── No
    │       │       ├── Data analysis / code interpretation?
    │       │       │   ├── Yes → E2B
    │       │       │   └── No
    │       │       │       ├── Agent-first workflow?
    │       │       │       │   ├── Yes → Blaxel
    │       │       │       │   └── No → Vercel
```
