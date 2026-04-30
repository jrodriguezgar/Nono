"""
Workspace — declarative I/O description for agents.

A ``Workspace`` declares the input files, data sources, and output
destinations an agent needs to perform its task.  It is **provider-agnostic**
— the same workspace definition works whether the agent runs locally, in a
sandbox (E2B, Modal, …), or in a remote environment.

The sandbox ``Manifest`` is a materialisation target: a workspace can be
converted to a manifest for sandbox execution, but a workspace can also
drive local file resolution, URL fetching, or in-memory data injection.

Part of the Nono Agent Architecture (NAA).

Key types:
    - ``WorkspaceEntry``: Abstract base for a single I/O resource.
    - ``FileEntry``: A local file or directory.
    - ``URLEntry``: A remote resource to fetch.
    - ``InlineEntry``: Literal data embedded in the workspace.
    - ``CloudStorageEntry``: S3 / GCS / Azure Blob / R2 reference.
    - ``TemplateEntry``: A Jinja2 template rendered at resolve time.
    - ``OutputEntry``: A declared output destination.
    - ``Workspace``: Container grouping inputs and outputs.

Example:
    >>> from nono.agent.workspace import Workspace, FileEntry, OutputEntry
    >>> ws = Workspace(
    ...     inputs={"data": FileEntry(path="/tmp/reports")},
    ...     outputs={"result": OutputEntry(path="output/report.csv")},
    ... )
    >>> ws.describe()
    'Workspace: 1 inputs, 1 outputs'
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("Nono.Agent.Workspace")


# ── Enums ─────────────────────────────────────────────────────────────────────

class IODirection(Enum):
    """Whether an entry is an input or output."""

    INPUT = "input"
    OUTPUT = "output"


class StorageKind(Enum):
    """Cloud storage provider kind."""

    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    CLOUDFLARE_R2 = "cloudflare_r2"


# ── Entry base ────────────────────────────────────────────────────────────────

class WorkspaceEntry(ABC):
    """Abstract base for a single I/O resource in a workspace.

    Subclasses describe the concrete source or destination (file, URL,
    inline data, cloud storage, template, output location).
    """

    @abstractmethod
    def entry_type(self) -> str:
        """Short identifier for this entry kind."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""

    @property
    def is_readable(self) -> bool:
        """Return ``True`` if the entry can be read (input source)."""
        return True

    @property
    def is_writable(self) -> bool:
        """Return ``True`` if the entry can be written to (output)."""
        return False


# ── Input entries ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FileEntry(WorkspaceEntry):
    """A local file or directory.

    Args:
        path: Absolute or relative path on the host filesystem.
        read_only: If ``True`` the agent must not modify this resource.
        glob: Optional glob pattern to filter files inside a directory
            (e.g. ``"*.csv"``).  Ignored for single files.
        description: Human-readable hint about this entry's purpose.
    """

    path: str | Path
    read_only: bool = True
    glob: str = ""
    description: str = ""

    def entry_type(self) -> str:
        return "file"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "path": str(self.path),
            "read_only": self.read_only,
        }
        if self.glob:
            d["glob"] = self.glob
        if self.description:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class URLEntry(WorkspaceEntry):
    """A remote resource to fetch (HTTP/HTTPS).

    The workspace resolver downloads the content and makes it available
    to the agent as a local file or in-memory blob.

    Args:
        url: The URL to fetch.
        headers: Optional request headers (e.g. authentication).
        cache: Cache the response for the session lifetime.
        description: Human-readable hint.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    cache: bool = True
    description: str = ""

    def entry_type(self) -> str:
        return "url"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "url": self.url,
            "cache": self.cache,
        }
        if self.description:
            d["description"] = self.description
        # headers intentionally omitted from serialisation (may contain secrets)
        return d


@dataclass(frozen=True)
class InlineEntry(WorkspaceEntry):
    """Literal data embedded directly in the workspace definition.

    Args:
        data: The payload — ``str``, ``bytes``, ``dict``, or ``list``.
        content_type: MIME type hint (e.g. ``"text/csv"``).
        description: Human-readable hint.
    """

    data: str | bytes | dict | list
    content_type: str = "text/plain"
    description: str = ""

    def entry_type(self) -> str:
        return "inline"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "content_type": self.content_type,
        }
        if isinstance(self.data, bytes):
            d["data_size"] = len(self.data)
        else:
            d["data"] = self.data
        if self.description:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class CloudStorageEntry(WorkspaceEntry):
    """Reference to cloud object storage (S3, GCS, Azure Blob, R2).

    A single entry type covers all cloud providers; ``kind`` selects
    which provider the resolver uses.

    Args:
        kind: Cloud storage provider.
        bucket: Bucket or container name.
        prefix: Key / blob name prefix to scope the mount.
        region: Cloud region (provider-dependent).
        credentials_env: Env-var holding credentials or connection string.
        read_only: Prevent writes.
        description: Human-readable hint.
    """

    kind: StorageKind
    bucket: str
    prefix: str = ""
    region: str = ""
    credentials_env: str = ""
    read_only: bool = True
    description: str = ""

    def entry_type(self) -> str:
        return self.kind.value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "read_only": self.read_only,
        }
        if self.region:
            d["region"] = self.region
        if self.credentials_env:
            d["credentials_env"] = self.credentials_env
        if self.description:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class TemplateEntry(WorkspaceEntry):
    """A Jinja2 template rendered at resolve time.

    The template is rendered with the provided variables (or session state)
    and the result is injected as a text input.

    Args:
        template: The Jinja2 template string.
        variables: Default variable values.
        description: Human-readable hint.
    """

    template: str
    variables: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def entry_type(self) -> str:
        return "template"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "template": self.template,
        }
        if self.variables:
            d["variables"] = self.variables
        if self.description:
            d["description"] = self.description
        return d


# ── Output entries ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OutputEntry(WorkspaceEntry):
    """A declared output destination.

    Tells the agent (or sandbox) where to write results.  After execution
    the workspace resolver collects the output for downstream consumption.

    Args:
        path: Relative or absolute path for the output.
        content_type: Expected MIME type of the output.
        description: Human-readable hint.
    """

    path: str = "output"
    content_type: str = "application/octet-stream"
    description: str = ""

    def entry_type(self) -> str:
        return "output"

    @property
    def is_readable(self) -> bool:
        return False

    @property
    def is_writable(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.entry_type(),
            "path": self.path,
            "content_type": self.content_type,
        }
        if self.description:
            d["description"] = self.description
        return d


# ── Workspace ─────────────────────────────────────────────────────────────────

@dataclass
class Workspace:
    """Declarative description of an agent's I/O resources.

    A workspace groups **inputs** (data the agent reads) and **outputs**
    (destinations the agent writes to) under logical names.  It is
    provider-agnostic — the same definition drives local execution,
    sandbox materialisation, or cloud orchestration.

    Args:
        inputs: Mapping of logical name → input entry.
        outputs: Mapping of logical name → output entry.
        metadata: Free-form metadata (tags, version, owner, etc.).

    Example:
        >>> ws = Workspace(
        ...     inputs={
        ...         "raw_data": FileEntry(path="/data/sales.csv"),
        ...         "config": InlineEntry(data={"threshold": 0.95}),
        ...     },
        ...     outputs={
        ...         "report": OutputEntry(path="output/report.md"),
        ...     },
        ... )
    """

    inputs: dict[str, WorkspaceEntry] = field(default_factory=dict)
    outputs: dict[str, OutputEntry] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── mutators ──────────────────────────────────────────────────────

    def add_input(self, name: str, entry: WorkspaceEntry) -> None:
        """Add or replace an input entry.

        Args:
            name: Logical name for the input.
            entry: The input entry.
        """
        self.inputs[name] = entry
        logger.debug("Workspace: added input %r → %s", name, entry.entry_type())

    def add_output(self, name: str, entry: OutputEntry) -> None:
        """Add or replace an output entry.

        Args:
            name: Logical name for the output.
            entry: The output entry.
        """
        self.outputs[name] = entry
        logger.debug("Workspace: added output %r → %s", name, entry.entry_type())

    def remove_input(self, name: str) -> WorkspaceEntry | None:
        """Remove and return an input by name.

        Args:
            name: Key to remove.

        Returns:
            The removed entry, or ``None``.
        """
        return self.inputs.pop(name, None)

    def remove_output(self, name: str) -> OutputEntry | None:
        """Remove and return an output by name.

        Args:
            name: Key to remove.

        Returns:
            The removed entry, or ``None``.
        """
        return self.outputs.pop(name, None)

    # ── queries ───────────────────────────────────────────────────────

    def input_names(self) -> list[str]:
        """Return all input names."""
        return list(self.inputs.keys())

    def output_names(self) -> list[str]:
        """Return all output names."""
        return list(self.outputs.keys())

    def describe(self) -> str:
        """Human-readable one-line summary.

        Returns:
            Summary string like ``'Workspace: 3 inputs, 1 outputs'``.
        """
        return f"Workspace: {len(self.inputs)} inputs, {len(self.outputs)} outputs"

    def get_input(self, name: str) -> WorkspaceEntry | None:
        """Retrieve an input entry by name.

        Args:
            name: Logical input name.

        Returns:
            The entry, or ``None``.
        """
        return self.inputs.get(name)

    def get_output(self, name: str) -> OutputEntry | None:
        """Retrieve an output entry by name.

        Args:
            name: Logical output name.

        Returns:
            The entry, or ``None``.
        """
        return self.outputs.get(name)

    def file_entries(self) -> dict[str, FileEntry]:
        """Return only ``FileEntry`` inputs.

        Returns:
            Filtered mapping of name → ``FileEntry``.
        """
        return {
            k: v for k, v in self.inputs.items() if isinstance(v, FileEntry)
        }

    def cloud_entries(self) -> dict[str, CloudStorageEntry]:
        """Return only ``CloudStorageEntry`` inputs.

        Returns:
            Filtered mapping of name → ``CloudStorageEntry``.
        """
        return {
            k: v for k, v in self.inputs.items()
            if isinstance(v, CloudStorageEntry)
        }

    def url_entries(self) -> dict[str, URLEntry]:
        """Return only ``URLEntry`` inputs.

        Returns:
            Filtered mapping of name → ``URLEntry``.
        """
        return {
            k: v for k, v in self.inputs.items() if isinstance(v, URLEntry)
        }

    def inline_entries(self) -> dict[str, InlineEntry]:
        """Return only ``InlineEntry`` inputs.

        Returns:
            Filtered mapping of name → ``InlineEntry``.
        """
        return {
            k: v for k, v in self.inputs.items() if isinstance(v, InlineEntry)
        }

    def template_entries(self) -> dict[str, TemplateEntry]:
        """Return only ``TemplateEntry`` inputs.

        Returns:
            Filtered mapping of name → ``TemplateEntry``.
        """
        return {
            k: v for k, v in self.inputs.items()
            if isinstance(v, TemplateEntry)
        }

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the workspace to a JSON-safe dict.

        Returns:
            Dict with ``inputs``, ``outputs``, and ``metadata`` keys.
        """
        return {
            "inputs": {
                name: entry.to_dict() for name, entry in self.inputs.items()
            },
            "outputs": {
                name: entry.to_dict() for name, entry in self.outputs.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workspace:
        """Deserialise from a plain dict.

        Reconstructs input entries using the ``type`` field to select the
        correct entry class.  Unknown types are logged and skipped.

        Args:
            data: Dict previously produced by :meth:`to_dict` (or
                manually constructed with the same schema).

        Returns:
            A ``Workspace`` instance.
        """
        inputs: dict[str, WorkspaceEntry] = {}

        for name, entry_data in data.get("inputs", {}).items():
            entry = _entry_from_dict(entry_data)
            if entry is not None:
                inputs[name] = entry

        outputs: dict[str, OutputEntry] = {}

        for name, entry_data in data.get("outputs", {}).items():
            outputs[name] = OutputEntry(
                path=entry_data.get("path", "output"),
                content_type=entry_data.get(
                    "content_type", "application/octet-stream"
                ),
                description=entry_data.get("description", ""),
            )

        return cls(
            inputs=inputs,
            outputs=outputs,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, raw: str) -> Workspace:
        """Deserialise from a JSON string.

        Args:
            raw: JSON string.

        Returns:
            A ``Workspace``.
        """
        return cls.from_dict(json.loads(raw))

    # ── sandbox bridge ────────────────────────────────────────────────

    def to_manifest(self) -> Any:
        """Convert this workspace to a sandbox ``Manifest``.

        Maps ``FileEntry`` → ``LocalDir``/``LocalFile``,
        ``CloudStorageEntry`` → ``S3Bucket``/``GCSBucket``/``AzureBlob``/
        ``CloudflareR2``, and the first ``OutputEntry`` → ``OutputDir``.

        Returns:
            A ``nono.sandbox.manifest.Manifest`` instance.

        Raises:
            ImportError: If the sandbox module is not available.
        """
        from nono.sandbox.manifest import (
            AzureBlob,
            CloudflareR2,
            GCSBucket,
            LocalDir,
            LocalFile,
            Manifest,
            OutputDir,
            S3Bucket,
        )

        entries: dict[str, Any] = {}

        for name, entry in self.inputs.items():
            if isinstance(entry, FileEntry):
                p = Path(entry.path)
                if p.is_file() or (not p.exists() and p.suffix):
                    entries[name] = LocalFile(
                        src=entry.path, read_only=entry.read_only
                    )
                else:
                    entries[name] = LocalDir(
                        src=entry.path, read_only=entry.read_only
                    )

            elif isinstance(entry, CloudStorageEntry):
                if entry.kind == StorageKind.S3:
                    entries[name] = S3Bucket(
                        bucket=entry.bucket,
                        prefix=entry.prefix,
                        region=entry.region or None,
                        read_only=entry.read_only,
                    )
                elif entry.kind == StorageKind.GCS:
                    entries[name] = GCSBucket(
                        bucket=entry.bucket,
                        prefix=entry.prefix,
                        read_only=entry.read_only,
                    )
                elif entry.kind == StorageKind.AZURE_BLOB:
                    entries[name] = AzureBlob(
                        container=entry.bucket,
                        prefix=entry.prefix,
                        connection_string_env=entry.credentials_env or None,
                        read_only=entry.read_only,
                    )
                elif entry.kind == StorageKind.CLOUDFLARE_R2:
                    entries[name] = CloudflareR2(
                        bucket=entry.bucket,
                        prefix=entry.prefix,
                        read_only=entry.read_only,
                    )

        # First output entry → OutputDir
        output_dir: OutputDir | None = None
        if self.outputs:
            first_output = next(iter(self.outputs.values()))
            output_dir = OutputDir(path=first_output.path)

        return Manifest(
            entries=entries,
            output=output_dir,
            metadata=self.metadata,
        )


# ── Deserialisation helpers ───────────────────────────────────────────────────

_ENTRY_BUILDERS: dict[str, type[WorkspaceEntry]] = {}


def _entry_from_dict(data: dict[str, Any]) -> WorkspaceEntry | None:
    """Reconstruct a WorkspaceEntry from a serialised dict.

    Args:
        data: Entry dict with a ``type`` key.

    Returns:
        The reconstructed entry, or ``None`` for unknown types.
    """
    entry_type = data.get("type", "")

    if entry_type == "file":
        return FileEntry(
            path=data.get("path", ""),
            read_only=data.get("read_only", True),
            glob=data.get("glob", ""),
            description=data.get("description", ""),
        )

    if entry_type == "url":
        return URLEntry(
            url=data.get("url", ""),
            cache=data.get("cache", True),
            description=data.get("description", ""),
        )

    if entry_type == "inline":
        return InlineEntry(
            data=data.get("data", ""),
            content_type=data.get("content_type", "text/plain"),
            description=data.get("description", ""),
        )

    if entry_type in ("s3", "gcs", "azure_blob", "cloudflare_r2"):
        return CloudStorageEntry(
            kind=StorageKind(entry_type),
            bucket=data.get("bucket", ""),
            prefix=data.get("prefix", ""),
            region=data.get("region", ""),
            credentials_env=data.get("credentials_env", ""),
            read_only=data.get("read_only", True),
            description=data.get("description", ""),
        )

    if entry_type == "template":
        return TemplateEntry(
            template=data.get("template", ""),
            variables=data.get("variables", {}),
            description=data.get("description", ""),
        )

    if entry_type == "output":
        return OutputEntry(
            path=data.get("path", "output"),
            content_type=data.get(
                "content_type", "application/octet-stream"
            ),
            description=data.get("description", ""),
        )

    logger.warning("Workspace: unknown entry type %r — skipping", entry_type)
    return None
