"""
Manifest system for describing sandbox workspaces.

A ``Manifest`` declares the files and directories an agent needs inside its
sandbox environment.  Each entry maps a logical mount path (where the agent
sees it) to a concrete source — a local directory, an S3 bucket prefix, a
GCS path, an Azure Blob container, or a Cloudflare R2 bucket.

Part of the Nono Agent Architecture (NAA).

Example:
    >>> from nono.sandbox.manifest import Manifest, LocalDir, S3Bucket
    >>> m = Manifest(entries={
    ...     "data": LocalDir(src="/tmp/reports"),
    ...     "models": S3Bucket(bucket="ml-models", prefix="v2/"),
    ... })
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Nono.Sandbox.Manifest")


# ── Entry base ────────────────────────────────────────────────────────────────

class ManifestEntry(ABC):
    """Abstract base for a single mount-point source.

    Subclasses describe *where* the data comes from.  The sandbox client
    is responsible for materialising these entries inside the sandbox
    filesystem before the agent starts executing.
    """

    @abstractmethod
    def entry_type(self) -> str:
        """Return a short identifier for this entry kind."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise the entry to a plain dict (JSON-safe)."""


# ── Local entries ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LocalDir(ManifestEntry):
    """Mount a local directory into the sandbox.

    Args:
        src: Absolute path on the host filesystem.
        read_only: If ``True`` the sandbox cannot modify these files.
    """

    src: str | Path
    read_only: bool = False

    def entry_type(self) -> str:
        return "local_dir"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.entry_type(),
            "src": str(self.src),
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class LocalFile(ManifestEntry):
    """Mount a single local file into the sandbox.

    Args:
        src: Absolute path to the file on the host.
        read_only: If ``True`` the sandbox cannot modify the file.
    """

    src: str | Path
    read_only: bool = False

    def entry_type(self) -> str:
        return "local_file"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.entry_type(),
            "src": str(self.src),
            "read_only": self.read_only,
        }


# ── Cloud storage entries ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class S3Bucket(ManifestEntry):
    """Mount an AWS S3 bucket (or prefix) into the sandbox.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to scope the mount (e.g. ``"reports/2024/"``).
        region: AWS region.  ``None`` uses the SDK default.
        credentials_profile: Named AWS CLI profile.  ``None`` uses the
            default credential chain.
        read_only: Prevent writes back to S3.
    """

    bucket: str
    prefix: str = ""
    region: str | None = None
    credentials_profile: str | None = None
    read_only: bool = False

    def entry_type(self) -> str:
        return "s3"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "read_only": self.read_only,
        }

        if self.region:
            d["region"] = self.region

        if self.credentials_profile:
            d["credentials_profile"] = self.credentials_profile

        return d


@dataclass(frozen=True)
class GCSBucket(ManifestEntry):
    """Mount a Google Cloud Storage bucket (or prefix) into the sandbox.

    Args:
        bucket: GCS bucket name.
        prefix: Object prefix to scope the mount.
        project: GCP project ID.  ``None`` uses the SDK default.
        credentials_json: Path to a service-account JSON key file.
        read_only: Prevent writes back to GCS.
    """

    bucket: str
    prefix: str = ""
    project: str | None = None
    credentials_json: str | None = None
    read_only: bool = False

    def entry_type(self) -> str:
        return "gcs"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "read_only": self.read_only,
        }

        if self.project:
            d["project"] = self.project

        if self.credentials_json:
            d["credentials_json"] = self.credentials_json

        return d


@dataclass(frozen=True)
class AzureBlob(ManifestEntry):
    """Mount an Azure Blob Storage container (or prefix) into the sandbox.

    Args:
        container: Azure Blob container name.
        prefix: Blob name prefix to scope the mount.
        account_name: Storage account name.
        connection_string_env: Name of the env-var holding the connection
            string.  ``None`` uses ``DefaultAzureCredential``.
        read_only: Prevent writes back to Azure.
    """

    container: str
    prefix: str = ""
    account_name: str = ""
    connection_string_env: str | None = None
    read_only: bool = False

    def entry_type(self) -> str:
        return "azure_blob"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "container": self.container,
            "prefix": self.prefix,
            "read_only": self.read_only,
        }

        if self.account_name:
            d["account_name"] = self.account_name

        if self.connection_string_env:
            d["connection_string_env"] = self.connection_string_env

        return d


@dataclass(frozen=True)
class CloudflareR2(ManifestEntry):
    """Mount a Cloudflare R2 bucket (or prefix) into the sandbox.

    Args:
        bucket: R2 bucket name.
        prefix: Key prefix to scope the mount.
        account_id: Cloudflare account ID.
        access_key_id_env: Env-var name for the R2 access key ID.
        secret_access_key_env: Env-var name for the R2 secret access key.
        read_only: Prevent writes back to R2.
    """

    bucket: str
    prefix: str = ""
    account_id: str = ""
    access_key_id_env: str | None = None
    secret_access_key_env: str | None = None
    read_only: bool = False

    def entry_type(self) -> str:
        return "cloudflare_r2"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "read_only": self.read_only,
        }

        if self.account_id:
            d["account_id"] = self.account_id

        if self.access_key_id_env:
            d["access_key_id_env"] = self.access_key_id_env

        if self.secret_access_key_env:
            d["secret_access_key_env"] = self.secret_access_key_env

        return d


# ── Output entry ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputDir(ManifestEntry):
    """Declare an output directory that the sandbox will produce.

    After the sandbox run completes the client will collect the contents
    of this directory and return them as execution artefacts.

    Args:
        path: Relative path inside the sandbox (e.g. ``"output"``).
    """

    path: str = "output"

    def entry_type(self) -> str:
        return "output_dir"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.entry_type(),
            "path": self.path,
        }


# ── Manifest ──────────────────────────────────────────────────────────────────

@dataclass
class Manifest:
    """Declarative description of a sandbox workspace.

    Maps logical mount paths (seen by the agent inside the sandbox) to
    concrete ``ManifestEntry`` sources.

    Args:
        entries: Mapping of mount-path → entry source.
        output: Optional output directory entry.
        metadata: Free-form metadata attached to the manifest.

    Example:
        >>> m = Manifest(
        ...     entries={
        ...         "data": LocalDir(src="/tmp/data"),
        ...         "models": S3Bucket(bucket="my-models", prefix="v3/"),
        ...     },
        ...     output=OutputDir(path="results"),
        ... )
        >>> m.to_dict()
        {'entries': {...}, 'output': {...}, 'metadata': {}}
    """

    entries: dict[str, ManifestEntry] = field(default_factory=dict)
    output: OutputDir | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────────────

    def add_entry(self, mount_path: str, entry: ManifestEntry) -> None:
        """Add or replace a mount-point entry.

        Args:
            mount_path: Logical path inside the sandbox.
            entry: The source to mount.
        """
        self.entries[mount_path] = entry
        logger.debug("Manifest: added %s → %s", mount_path, entry.entry_type())

    def remove_entry(self, mount_path: str) -> ManifestEntry | None:
        """Remove and return an entry by mount path.

        Args:
            mount_path: Key to remove.

        Returns:
            The removed entry, or ``None`` if not found.
        """
        return self.entries.pop(mount_path, None)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full manifest to a JSON-safe dict."""
        d: dict[str, Any] = {
            "entries": {
                path: entry.to_dict() for path, entry in self.entries.items()
            },
            "metadata": self.metadata,
        }

        if self.output:
            d["output"] = self.output.to_dict()

        return d

    def mount_paths(self) -> list[str]:
        """Return all mount paths defined in this manifest."""
        return list(self.entries.keys())

    def to_workspace(self) -> Any:
        """Convert this manifest to an agent ``Workspace``.

        Maps ``LocalDir``/``LocalFile`` → ``FileEntry``,
        ``S3Bucket``/``GCSBucket``/``AzureBlob``/``CloudflareR2`` →
        ``CloudStorageEntry``, and ``OutputDir`` → ``OutputEntry``.

        Returns:
            A ``nono.agent.workspace.Workspace`` instance.
        """
        from nono.agent.workspace import (
            CloudStorageEntry,
            FileEntry,
            OutputEntry,
            StorageKind,
            Workspace,
        )

        inputs: dict[str, Any] = {}

        for mount_path, entry in self.entries.items():
            if isinstance(entry, (LocalDir, LocalFile)):
                inputs[mount_path] = FileEntry(
                    path=str(entry.src),
                    read_only=entry.read_only,
                )
            elif isinstance(entry, S3Bucket):
                inputs[mount_path] = CloudStorageEntry(
                    kind=StorageKind.S3,
                    bucket=entry.bucket,
                    prefix=entry.prefix,
                    region=entry.region or "",
                    read_only=entry.read_only,
                )
            elif isinstance(entry, GCSBucket):
                inputs[mount_path] = CloudStorageEntry(
                    kind=StorageKind.GCS,
                    bucket=entry.bucket,
                    prefix=entry.prefix,
                    read_only=entry.read_only,
                )
            elif isinstance(entry, AzureBlob):
                inputs[mount_path] = CloudStorageEntry(
                    kind=StorageKind.AZURE_BLOB,
                    bucket=entry.container,
                    prefix=entry.prefix,
                    credentials_env=entry.connection_string_env or "",
                    read_only=entry.read_only,
                )
            elif isinstance(entry, CloudflareR2):
                inputs[mount_path] = CloudStorageEntry(
                    kind=StorageKind.CLOUDFLARE_R2,
                    bucket=entry.bucket,
                    prefix=entry.prefix,
                    read_only=entry.read_only,
                )

        outputs: dict[str, OutputEntry] = {}

        if self.output:
            outputs["output"] = OutputEntry(path=self.output.path)

        return Workspace(
            inputs=inputs,
            outputs=outputs,
            metadata=self.metadata,
        )
