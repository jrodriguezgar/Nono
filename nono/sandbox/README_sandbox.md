# Sandbox — External Sandbox Execution

Run agent-generated code inside isolated, external sandbox environments instead of the local machine.

[← Back to main README](../README.md) · [Workspace & Manifest Guide](../README_workspace.md)

---

## Supported Providers

| Provider | Package | Env Var |
|----------|---------|---------|
| **E2B** | `e2b-code-interpreter` | `E2B_API_KEY` |
| **Modal** | `modal` | `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` |
| **Daytona** | `daytona-sdk` | `DAYTONA_API_KEY` |
| **Blaxel** | `blaxel` | `BLAXEL_API_KEY` |
| **Cloudflare** | `cloudflare` | `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID` |
| **Runloop** | `runloop-api-client` | `RUNLOOP_API_KEY` |
| **Vercel** | `httpx` | `VERCEL_TOKEN` |

Install the provider package you need — sandbox dependencies are **not** included by default.

---

## Quick Start

```python
from nono.sandbox import (
    SandboxAgent, SandboxRunConfig, SandboxProvider,
    Manifest, LocalDir, S3Bucket, OutputDir,
)

# 1. Describe the workspace
manifest = Manifest(
    entries={
        "data": LocalDir(src="/tmp/reports"),
        "models": S3Bucket(bucket="ml-models", prefix="v2/"),
    },
    output=OutputDir(path="results"),
)

# 2. Configure the sandbox
config = SandboxRunConfig(
    provider=SandboxProvider.E2B,
    timeout=300,
    packages=["pandas", "matplotlib"],
    manifest=manifest,
)

# 3. Create the agent
agent = SandboxAgent(
    name="Analyst",
    sandbox_config=config,
)
```

---

## Manifest Entries

A `Manifest` maps logical mount paths (what the agent sees) to concrete sources:

| Entry | Source | Key Args |
|-------|--------|----------|
| `LocalDir` | Local directory | `src`, `read_only` |
| `LocalFile` | Single local file | `src`, `read_only` |
| `S3Bucket` | AWS S3 | `bucket`, `prefix`, `region`, `credentials_profile` |
| `GCSBucket` | Google Cloud Storage | `bucket`, `prefix`, `project`, `credentials_json` |
| `AzureBlob` | Azure Blob Storage | `container`, `prefix`, `account_name`, `connection_string_env` |
| `CloudflareR2` | Cloudflare R2 | `bucket`, `prefix`, `account_id`, `access_key_id_env` |
| `OutputDir` | Output directory | `path` (collected after execution) |

```python
from nono.sandbox.manifest import Manifest, LocalDir, S3Bucket, AzureBlob

m = Manifest(entries={
    "input": LocalDir(src="/data/raw", read_only=True),
    "models": S3Bucket(bucket="my-models", prefix="prod/", region="eu-west-1"),
    "reports": AzureBlob(container="reports", account_name="myaccount"),
})

# Serialise to JSON-safe dict
print(m.to_dict())
```

---

## Configuration (`config.toml`)

```toml
[sandbox]
default_provider = "e2b"    # e2b | modal | daytona | blaxel | cloudflare | runloop | vercel
timeout = 300                # seconds
working_dir = "/home/user"
keep_alive = false           # keep sandbox after execution
snapshot = false             # enable state snapshots
```

---

## SandboxRunConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `SandboxProvider` | `E2B` | Target sandbox provider |
| `timeout` | `int` | `300` | Max execution time (seconds) |
| `environment` | `dict[str, str]` | `{}` | Env vars passed to sandbox |
| `packages` | `list[str]` | `[]` | Python packages to install |
| `working_dir` | `str` | `/home/user` | Working directory inside sandbox |
| `manifest` | `Manifest \| None` | `None` | Workspace description |
| `keep_alive` | `bool` | `False` | Keep sandbox after execution |
| `snapshot` | `bool` | `False` | Enable snapshots |
| `metadata` | `dict` | `{}` | Provider-specific options |

---

## SandboxResult

| Field | Type | Description |
|-------|------|-------------|
| `status` | `SandboxStatus` | `COMPLETED`, `FAILED`, `TIMEOUT` |
| `stdout` | `str` | Captured standard output |
| `stderr` | `str` | Captured standard error |
| `exit_code` | `int` | Process exit code |
| `output_files` | `dict[str, bytes]` | Files from `OutputDir` |
| `sandbox_id` | `str` | Provider sandbox identifier |
| `snapshot_id` | `str` | Snapshot ID (if enabled) |
| `success` | `bool` | Property: `status == COMPLETED and exit_code == 0` |

---

## Using a Provider Directly

```python
from nono.sandbox import get_sandbox_client, SandboxRunConfig
from nono.sandbox.base import SandboxProvider

client = get_sandbox_client(SandboxProvider.MODAL)
result = client.execute(
    "import pandas as pd; print(pd.__version__)",
    SandboxRunConfig(packages=["pandas"]),
)
print(result.stdout)
```

---

## Architecture

```
SandboxAgent ──────────────────────────────────┐
    │                                          │
    ▼                                          │
get_sandbox_client(provider)                   │
    │                                          │
    ▼                                          │
BaseSandboxClient  ◄── E2BSandboxClient        │
                   ◄── ModalSandboxClient      │
                   ◄── DaytonaSandboxClient    │
                   ◄── BlaxelSandboxClient     │
                   ◄── CloudflareSandboxClient │
                   ◄── RunloopSandboxClient    │
                   ◄── VercelSandboxClient     │
    │                                          │
    ▼                                          │
SandboxResult  →  SharedContent (output files) │
                                               │
DurableSandboxAgent ───extends─────────────────┘
    │
    ▼
HarnessRuntime  ←→  SnapshotStore
    │                   │
    │  checkpoint        └── in-memory / disk (.json)
    │  execute
    │  retry + rehydrate
    ▼
BaseSandboxClient.execute()
    │
    ▼
SandboxSnapshot  →  provider snapshot (optional)
```

---

## Harness / Compute Separation

The **harness** (your local process) manages session state, events, and
orchestration logic.  The **compute** (remote sandbox) runs untrusted code.
Credentials never leave the harness.

If a sandbox container crashes, the harness:

1. Snapshots accumulated state (`SandboxSnapshot`).
2. Optionally takes a provider-native snapshot (E2B, Modal).
3. Provisions a fresh container (or restores from provider snapshot).
4. Resumes execution from the last checkpoint.

### DurableSandboxAgent

Drop-in replacement for `SandboxAgent` with automatic fault tolerance:

```python
from nono.sandbox import DurableSandboxAgent, SandboxRunConfig, SandboxProvider

agent = DurableSandboxAgent(
    name="ResilientAnalyst",
    sandbox_config=SandboxRunConfig(
        provider=SandboxProvider.E2B,
        snapshot=True,
    ),
    max_retries=3,
    retry_delay=2.0,
)
```

### HarnessRuntime (low-level)

Use `HarnessRuntime` directly for fine-grained control:

```python
from nono.sandbox.harness import HarnessRuntime, SnapshotStore
from nono.sandbox import SandboxRunConfig

store = SnapshotStore.on_disk("/tmp/nono-snapshots")
runtime = HarnessRuntime(max_retries=3, store=store, retry_delay=1.0)

result = runtime.execute(
    code="import pandas; print(pandas.__version__)",
    config=SandboxRunConfig(provider=SandboxProvider.E2B),
    session_state={"step": 1},
)

# Resume from a saved snapshot
result = runtime.resume(snapshot_id="abc123", config=SandboxRunConfig())
```

### SandboxSnapshot

Serialisable checkpoint — JSON round-trip supported:

```python
from nono.sandbox.harness import SandboxSnapshot

snap = SandboxSnapshot(code="print(1)", session_state={"k": "v"})
raw = snap.to_json()                    # persist anywhere
restored = SandboxSnapshot.from_json(raw)  # rehydrate
```

### SnapshotStore

| Factory | Storage | Persistence |
|---------|---------|-------------|
| `SnapshotStore.in_memory()` | dict | Session only |
| `SnapshotStore.on_disk(dir)` | JSON files | Survives restarts |
