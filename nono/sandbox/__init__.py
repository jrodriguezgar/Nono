"""
Sandbox — external sandbox execution for Nono agents.

This module provides a unified interface for running agent-generated code
inside isolated, external sandbox environments.  Supported providers:

- **E2B** — cloud code interpreters (``e2b-code-interpreter``)
- **Modal** — serverless containers (``modal``)
- **Daytona** — dev-environments-as-a-service (``daytona-sdk``)
- **Blaxel** — cloud sandboxes (``blaxel``)
- **Cloudflare** — Workers containers (``cloudflare``)
- **Runloop** — AI dev-boxes (``runloop-api-client``)
- **Vercel** — serverless sandboxes (``httpx``)

A ``Manifest`` describes the agent's workspace (mounted files from local
directories, S3, GCS, Azure Blob Storage, or Cloudflare R2).

Quick start:
    >>> from nono.sandbox import (
    ...     SandboxAgent, SandboxRunConfig, SandboxProvider,
    ...     Manifest, LocalDir, S3Bucket,
    ... )
    >>>
    >>> config = SandboxRunConfig(
    ...     provider=SandboxProvider.E2B,
    ...     manifest=Manifest(entries={
    ...         "data": LocalDir(src="/tmp/data"),
    ...         "models": S3Bucket(bucket="my-models"),
    ...     }),
    ... )
    >>> agent = SandboxAgent(name="Analyst", sandbox_config=config)
"""

from .base import (
    BaseSandboxClient,
    SandboxProvider,
    SandboxResult,
    SandboxRunConfig,
    SandboxStatus,
)
from .manifest import (
    AzureBlob,
    CloudflareR2,
    GCSBucket,
    LocalDir,
    LocalFile,
    Manifest,
    ManifestEntry,
    OutputDir,
    S3Bucket,
)
from .sandbox_agent import SandboxAgent, get_sandbox_client
from .harness import (
    CheckpointStatus,
    HarnessEvent,
    HarnessRuntime,
    SandboxSnapshot,
    SnapshotStore,
)
from .durable_agent import DurableSandboxAgent

__all__ = [
    # Base
    "BaseSandboxClient",
    "SandboxProvider",
    "SandboxResult",
    "SandboxRunConfig",
    "SandboxStatus",
    # Manifest
    "AzureBlob",
    "CloudflareR2",
    "GCSBucket",
    "LocalDir",
    "LocalFile",
    "Manifest",
    "ManifestEntry",
    "OutputDir",
    "S3Bucket",
    # Agent
    "SandboxAgent",
    "get_sandbox_client",
    # Harness / Durability
    "CheckpointStatus",
    "DurableSandboxAgent",
    "HarnessEvent",
    "HarnessRuntime",
    "SandboxSnapshot",
    "SnapshotStore",
]
