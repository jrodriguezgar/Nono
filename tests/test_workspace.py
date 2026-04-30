"""Tests for nono.agent.workspace module."""

import json
import tempfile
from pathlib import Path

import pytest


# ── WorkspaceEntry types ──────────────────────────────────────────────────────


class TestFileEntry:
    """Test FileEntry dataclass."""

    def test_defaults(self):
        from nono.agent.workspace import FileEntry

        e = FileEntry(path="/tmp/data")

        assert e.path == "/tmp/data"
        assert e.read_only is True
        assert e.glob == ""
        assert e.entry_type() == "file"
        assert e.is_readable is True
        assert e.is_writable is False

    def test_to_dict(self):
        from nono.agent.workspace import FileEntry

        e = FileEntry(path="/tmp/reports", glob="*.csv", description="Sales")
        d = e.to_dict()

        assert d["type"] == "file"
        assert d["path"] == "/tmp/reports"
        assert d["glob"] == "*.csv"
        assert d["description"] == "Sales"

    def test_to_dict_minimal(self):
        from nono.agent.workspace import FileEntry

        d = FileEntry(path="/data").to_dict()

        assert "glob" not in d
        assert "description" not in d


class TestURLEntry:
    """Test URLEntry dataclass."""

    def test_defaults(self):
        from nono.agent.workspace import URLEntry

        e = URLEntry(url="https://example.com/data.json")

        assert e.url == "https://example.com/data.json"
        assert e.headers == {}
        assert e.cache is True
        assert e.entry_type() == "url"

    def test_to_dict_omits_headers(self):
        from nono.agent.workspace import URLEntry

        e = URLEntry(
            url="https://api.example.com",
            headers={"Authorization": "Bearer secret"},
        )
        d = e.to_dict()

        assert "headers" not in d
        assert d["url"] == "https://api.example.com"

    def test_to_dict_with_description(self):
        from nono.agent.workspace import URLEntry

        d = URLEntry(url="https://x.com", description="API").to_dict()

        assert d["description"] == "API"


class TestInlineEntry:
    """Test InlineEntry dataclass."""

    def test_string_data(self):
        from nono.agent.workspace import InlineEntry

        e = InlineEntry(data="hello world")

        assert e.data == "hello world"
        assert e.content_type == "text/plain"
        assert e.entry_type() == "inline"

    def test_to_dict_string(self):
        from nono.agent.workspace import InlineEntry

        d = InlineEntry(data="csv content", content_type="text/csv").to_dict()

        assert d["data"] == "csv content"
        assert d["content_type"] == "text/csv"

    def test_to_dict_bytes(self):
        from nono.agent.workspace import InlineEntry

        d = InlineEntry(data=b"\x89PNG").to_dict()

        assert "data" not in d
        assert d["data_size"] == 4

    def test_to_dict_dict(self):
        from nono.agent.workspace import InlineEntry

        d = InlineEntry(data={"key": "val"}).to_dict()

        assert d["data"] == {"key": "val"}


class TestCloudStorageEntry:
    """Test CloudStorageEntry dataclass."""

    def test_s3(self):
        from nono.agent.workspace import CloudStorageEntry, StorageKind

        e = CloudStorageEntry(
            kind=StorageKind.S3,
            bucket="my-bucket",
            prefix="data/",
            region="us-east-1",
        )

        assert e.entry_type() == "s3"
        assert e.bucket == "my-bucket"

    def test_to_dict(self):
        from nono.agent.workspace import CloudStorageEntry, StorageKind

        d = CloudStorageEntry(
            kind=StorageKind.GCS,
            bucket="gcs-bucket",
            credentials_env="GCS_KEY",
        ).to_dict()

        assert d["type"] == "gcs"
        assert d["bucket"] == "gcs-bucket"
        assert d["credentials_env"] == "GCS_KEY"

    def test_all_providers(self):
        from nono.agent.workspace import CloudStorageEntry, StorageKind

        for kind in StorageKind:
            e = CloudStorageEntry(kind=kind, bucket="b")
            assert e.entry_type() == kind.value


class TestTemplateEntry:
    """Test TemplateEntry dataclass."""

    def test_defaults(self):
        from nono.agent.workspace import TemplateEntry

        e = TemplateEntry(template="Hello {{ name }}")

        assert e.template == "Hello {{ name }}"
        assert e.variables == {}
        assert e.entry_type() == "template"

    def test_to_dict(self):
        from nono.agent.workspace import TemplateEntry

        d = TemplateEntry(
            template="{{ x }} + {{ y }}",
            variables={"x": 1, "y": 2},
            description="Math",
        ).to_dict()

        assert d["template"] == "{{ x }} + {{ y }}"
        assert d["variables"] == {"x": 1, "y": 2}
        assert d["description"] == "Math"


class TestOutputEntry:
    """Test OutputEntry dataclass."""

    def test_defaults(self):
        from nono.agent.workspace import OutputEntry

        e = OutputEntry()

        assert e.path == "output"
        assert e.content_type == "application/octet-stream"
        assert e.entry_type() == "output"
        assert e.is_readable is False
        assert e.is_writable is True

    def test_to_dict(self):
        from nono.agent.workspace import OutputEntry

        d = OutputEntry(
            path="results/report.csv",
            content_type="text/csv",
            description="Final report",
        ).to_dict()

        assert d["path"] == "results/report.csv"
        assert d["content_type"] == "text/csv"
        assert d["description"] == "Final report"


# ── Workspace container ───────────────────────────────────────────────────────


class TestWorkspace:
    """Test the Workspace container."""

    def test_empty_workspace(self):
        from nono.agent.workspace import Workspace

        ws = Workspace()

        assert ws.inputs == {}
        assert ws.outputs == {}
        assert ws.metadata == {}
        assert ws.describe() == "Workspace: 0 inputs, 0 outputs"

    def test_with_entries(self):
        from nono.agent.workspace import (
            FileEntry,
            InlineEntry,
            OutputEntry,
            Workspace,
        )

        ws = Workspace(
            inputs={
                "data": FileEntry(path="/tmp/data"),
                "config": InlineEntry(data={"k": "v"}),
            },
            outputs={"report": OutputEntry(path="out.csv")},
        )

        assert ws.input_names() == ["data", "config"]
        assert ws.output_names() == ["report"]
        assert ws.describe() == "Workspace: 2 inputs, 1 outputs"

    def test_add_remove_input(self):
        from nono.agent.workspace import FileEntry, Workspace

        ws = Workspace()
        ws.add_input("x", FileEntry(path="/tmp/x"))

        assert "x" in ws.inputs
        assert ws.get_input("x") is not None

        removed = ws.remove_input("x")

        assert removed is not None
        assert ws.inputs == {}

    def test_remove_nonexistent_input(self):
        from nono.agent.workspace import Workspace

        ws = Workspace()

        assert ws.remove_input("missing") is None

    def test_add_remove_output(self):
        from nono.agent.workspace import OutputEntry, Workspace

        ws = Workspace()
        ws.add_output("res", OutputEntry(path="result"))

        assert "res" in ws.outputs
        assert ws.get_output("res") is not None

        removed = ws.remove_output("res")

        assert removed is not None
        assert ws.outputs == {}

    def test_remove_nonexistent_output(self):
        from nono.agent.workspace import Workspace

        ws = Workspace()

        assert ws.remove_output("nope") is None

    def test_get_input_missing(self):
        from nono.agent.workspace import Workspace

        assert Workspace().get_input("no") is None

    def test_get_output_missing(self):
        from nono.agent.workspace import Workspace

        assert Workspace().get_output("no") is None

    def test_filtered_accessors(self):
        from nono.agent.workspace import (
            CloudStorageEntry,
            FileEntry,
            InlineEntry,
            StorageKind,
            TemplateEntry,
            URLEntry,
            Workspace,
        )

        ws = Workspace(inputs={
            "f": FileEntry(path="/data"),
            "u": URLEntry(url="https://x.com"),
            "i": InlineEntry(data="hi"),
            "c": CloudStorageEntry(kind=StorageKind.S3, bucket="b"),
            "t": TemplateEntry(template="{{ x }}"),
        })

        assert list(ws.file_entries().keys()) == ["f"]
        assert list(ws.url_entries().keys()) == ["u"]
        assert list(ws.inline_entries().keys()) == ["i"]
        assert list(ws.cloud_entries().keys()) == ["c"]
        assert list(ws.template_entries().keys()) == ["t"]


# ── Serialisation ─────────────────────────────────────────────────────────────


class TestWorkspaceSerialization:
    """Test to_dict / from_dict / JSON round-trips."""

    def test_to_dict(self):
        from nono.agent.workspace import (
            FileEntry,
            OutputEntry,
            Workspace,
        )

        ws = Workspace(
            inputs={"data": FileEntry(path="/tmp")},
            outputs={"out": OutputEntry(path="result")},
            metadata={"version": "1"},
        )
        d = ws.to_dict()

        assert "data" in d["inputs"]
        assert d["inputs"]["data"]["type"] == "file"
        assert "out" in d["outputs"]
        assert d["metadata"] == {"version": "1"}

    def test_roundtrip_dict(self):
        from nono.agent.workspace import (
            CloudStorageEntry,
            FileEntry,
            InlineEntry,
            OutputEntry,
            StorageKind,
            TemplateEntry,
            URLEntry,
            Workspace,
        )

        original = Workspace(
            inputs={
                "f": FileEntry(path="/data", read_only=False, glob="*.txt"),
                "u": URLEntry(url="https://api.com", description="API"),
                "i": InlineEntry(data="raw text", content_type="text/csv"),
                "s3": CloudStorageEntry(
                    kind=StorageKind.S3, bucket="bkt", prefix="p/"
                ),
                "gcs": CloudStorageEntry(kind=StorageKind.GCS, bucket="g"),
                "az": CloudStorageEntry(
                    kind=StorageKind.AZURE_BLOB, bucket="c"
                ),
                "r2": CloudStorageEntry(
                    kind=StorageKind.CLOUDFLARE_R2, bucket="r"
                ),
                "t": TemplateEntry(
                    template="{{ x }}", variables={"x": 1}
                ),
            },
            outputs={"out": OutputEntry(path="out", content_type="text/csv")},
            metadata={"v": 2},
        )

        restored = Workspace.from_dict(original.to_dict())

        assert restored.input_names() == original.input_names()
        assert restored.output_names() == original.output_names()
        assert restored.metadata == {"v": 2}

        # Spot-check entry fields
        assert restored.get_input("f").path == "/data"
        assert restored.get_input("u").url == "https://api.com"
        assert restored.get_input("i").data == "raw text"
        assert restored.get_input("s3").bucket == "bkt"
        assert restored.get_input("t").template == "{{ x }}"
        assert restored.get_output("out").content_type == "text/csv"

    def test_roundtrip_json(self):
        from nono.agent.workspace import FileEntry, OutputEntry, Workspace

        ws = Workspace(
            inputs={"data": FileEntry(path="/tmp/f.csv")},
            outputs={"report": OutputEntry(path="report.md")},
        )

        restored = Workspace.from_json(ws.to_json())

        assert restored.input_names() == ["data"]
        assert restored.output_names() == ["report"]

    def test_from_dict_unknown_type_skipped(self):
        from nono.agent.workspace import Workspace

        data = {
            "inputs": {
                "good": {"type": "file", "path": "/ok"},
                "bad": {"type": "unknown_xyz"},
            },
            "outputs": {},
        }
        ws = Workspace.from_dict(data)

        assert "good" in ws.inputs
        assert "bad" not in ws.inputs

    def test_from_dict_empty(self):
        from nono.agent.workspace import Workspace

        ws = Workspace.from_dict({})

        assert ws.inputs == {}
        assert ws.outputs == {}


# ── Manifest bridge ──────────────────────────────────────────────────────────


class TestWorkspaceToManifest:
    """Test Workspace → Manifest conversion."""

    def test_file_entries_to_manifest(self):
        from nono.agent.workspace import FileEntry, OutputEntry, Workspace

        ws = Workspace(
            inputs={"data": FileEntry(path="/tmp/reports", read_only=True)},
            outputs={"result": OutputEntry(path="results")},
        )
        manifest = ws.to_manifest()

        assert "data" in manifest.entries
        assert manifest.output is not None
        assert manifest.output.path == "results"

    def test_cloud_entries_to_manifest(self):
        from nono.agent.workspace import (
            CloudStorageEntry,
            StorageKind,
            Workspace,
        )

        ws = Workspace(inputs={
            "s3": CloudStorageEntry(
                kind=StorageKind.S3, bucket="bkt", prefix="p/", region="eu"
            ),
            "gcs": CloudStorageEntry(kind=StorageKind.GCS, bucket="g"),
            "az": CloudStorageEntry(
                kind=StorageKind.AZURE_BLOB,
                bucket="container",
                credentials_env="AZ_CONN",
            ),
            "r2": CloudStorageEntry(
                kind=StorageKind.CLOUDFLARE_R2, bucket="r"
            ),
        })
        manifest = ws.to_manifest()

        assert len(manifest.entries) == 4

        s3_entry = manifest.entries["s3"]
        assert s3_entry.bucket == "bkt"
        assert s3_entry.region == "eu"

    def test_empty_workspace_to_manifest(self):
        from nono.agent.workspace import Workspace

        manifest = Workspace().to_manifest()

        assert manifest.entries == {}
        assert manifest.output is None


class TestManifestToWorkspace:
    """Test Manifest → Workspace conversion."""

    def test_local_entries(self):
        from nono.sandbox.manifest import (
            LocalDir,
            LocalFile,
            Manifest,
            OutputDir,
        )

        m = Manifest(
            entries={
                "data": LocalDir(src="/tmp/data"),
                "file": LocalFile(src="/tmp/f.csv", read_only=True),
            },
            output=OutputDir(path="out"),
        )
        ws = m.to_workspace()

        assert "data" in ws.inputs
        assert "file" in ws.inputs
        assert ws.inputs["data"].path == "/tmp/data"
        assert ws.inputs["file"].read_only is True
        assert "output" in ws.outputs
        assert ws.outputs["output"].path == "out"

    def test_cloud_entries(self):
        from nono.sandbox.manifest import (
            AzureBlob,
            CloudflareR2,
            GCSBucket,
            Manifest,
            S3Bucket,
        )

        m = Manifest(entries={
            "s3": S3Bucket(bucket="b", prefix="p/", region="us-east-1"),
            "gcs": GCSBucket(bucket="g"),
            "az": AzureBlob(
                container="c", connection_string_env="AZ"
            ),
            "r2": CloudflareR2(bucket="r"),
        })
        ws = m.to_workspace()

        assert len(ws.inputs) == 4
        assert ws.inputs["s3"].bucket == "b"
        assert ws.inputs["az"].credentials_env == "AZ"

    def test_empty_manifest(self):
        from nono.sandbox.manifest import Manifest

        ws = Manifest().to_workspace()

        assert ws.inputs == {}
        assert ws.outputs == {}

    def test_roundtrip_workspace_manifest_workspace(self):
        from nono.agent.workspace import (
            CloudStorageEntry,
            FileEntry,
            OutputEntry,
            StorageKind,
            Workspace,
        )

        original = Workspace(
            inputs={
                "data": FileEntry(path="/tmp/data"),
                "models": CloudStorageEntry(
                    kind=StorageKind.S3, bucket="ml", prefix="v1/"
                ),
            },
            outputs={"result": OutputEntry(path="out")},
        )

        manifest = original.to_manifest()
        restored = manifest.to_workspace()

        assert set(restored.input_names()) == {"data", "models"}
        assert "output" in restored.outputs


# ── IODirection & StorageKind enums ───────────────────────────────────────────


class TestEnums:
    """Test enum values."""

    def test_io_direction(self):
        from nono.agent.workspace import IODirection

        assert IODirection.INPUT.value == "input"
        assert IODirection.OUTPUT.value == "output"

    def test_storage_kind(self):
        from nono.agent.workspace import StorageKind

        assert StorageKind.S3.value == "s3"
        assert StorageKind.GCS.value == "gcs"
        assert StorageKind.AZURE_BLOB.value == "azure_blob"
        assert StorageKind.CLOUDFLARE_R2.value == "cloudflare_r2"


# ── Module import ─────────────────────────────────────────────────────────────


class TestModuleImport:
    """Verify the module is importable."""

    def test_import_all_types(self):
        from nono.agent.workspace import (
            CloudStorageEntry,
            FileEntry,
            InlineEntry,
            IODirection,
            OutputEntry,
            StorageKind,
            TemplateEntry,
            URLEntry,
            Workspace,
            WorkspaceEntry,
        )

        assert Workspace is not None
        assert WorkspaceEntry is not None
