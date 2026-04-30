"""Tests for nono.sandbox module — manifest, base types, and agent."""

import pytest
from unittest.mock import MagicMock, patch


# ── Manifest tests ────────────────────────────────────────────────────────────


class TestManifestEntries:
    """Test individual manifest entry types."""

    def test_local_dir_to_dict(self):
        from nono.sandbox.manifest import LocalDir

        entry = LocalDir(src="/tmp/data", read_only=True)
        d = entry.to_dict()

        assert d["type"] == "local_dir"
        assert d["src"] == "/tmp/data"
        assert d["read_only"] is True

    def test_local_file_to_dict(self):
        from nono.sandbox.manifest import LocalFile

        entry = LocalFile(src="/tmp/report.csv")
        d = entry.to_dict()

        assert d["type"] == "local_file"
        assert d["src"] == "/tmp/report.csv"
        assert d["read_only"] is False

    def test_s3_bucket_to_dict_minimal(self):
        from nono.sandbox.manifest import S3Bucket

        entry = S3Bucket(bucket="my-bucket")
        d = entry.to_dict()

        assert d["type"] == "s3"
        assert d["bucket"] == "my-bucket"
        assert d["prefix"] == ""
        assert "region" not in d
        assert "credentials_profile" not in d

    def test_s3_bucket_to_dict_full(self):
        from nono.sandbox.manifest import S3Bucket

        entry = S3Bucket(
            bucket="ml-models",
            prefix="v2/",
            region="us-east-1",
            credentials_profile="prod",
            read_only=True,
        )
        d = entry.to_dict()

        assert d["region"] == "us-east-1"
        assert d["credentials_profile"] == "prod"
        assert d["read_only"] is True

    def test_gcs_bucket_to_dict(self):
        from nono.sandbox.manifest import GCSBucket

        entry = GCSBucket(bucket="my-gcs", prefix="data/", project="my-project")
        d = entry.to_dict()

        assert d["type"] == "gcs"
        assert d["project"] == "my-project"

    def test_azure_blob_to_dict(self):
        from nono.sandbox.manifest import AzureBlob

        entry = AzureBlob(
            container="reports",
            account_name="myaccount",
            connection_string_env="AZURE_CONN",
        )
        d = entry.to_dict()

        assert d["type"] == "azure_blob"
        assert d["container"] == "reports"
        assert d["account_name"] == "myaccount"
        assert d["connection_string_env"] == "AZURE_CONN"

    def test_cloudflare_r2_to_dict(self):
        from nono.sandbox.manifest import CloudflareR2

        entry = CloudflareR2(
            bucket="r2-bucket",
            account_id="abc123",
            access_key_id_env="R2_KEY",
            secret_access_key_env="R2_SECRET",
        )
        d = entry.to_dict()

        assert d["type"] == "cloudflare_r2"
        assert d["account_id"] == "abc123"
        assert d["access_key_id_env"] == "R2_KEY"

    def test_output_dir_to_dict(self):
        from nono.sandbox.manifest import OutputDir

        entry = OutputDir(path="results")
        d = entry.to_dict()

        assert d["type"] == "output_dir"
        assert d["path"] == "results"


class TestManifest:
    """Test the Manifest container."""

    def test_empty_manifest(self):
        from nono.sandbox.manifest import Manifest

        m = Manifest()

        assert m.entries == {}
        assert m.output is None
        assert m.mount_paths() == []

    def test_manifest_with_entries(self):
        from nono.sandbox.manifest import Manifest, LocalDir, S3Bucket

        m = Manifest(entries={
            "data": LocalDir(src="/tmp/data"),
            "models": S3Bucket(bucket="models"),
        })

        assert m.mount_paths() == ["data", "models"]

    def test_manifest_add_remove(self):
        from nono.sandbox.manifest import Manifest, LocalDir

        m = Manifest()
        m.add_entry("data", LocalDir(src="/tmp"))

        assert "data" in m.entries

        removed = m.remove_entry("data")

        assert removed is not None
        assert m.entries == {}

    def test_manifest_remove_nonexistent(self):
        from nono.sandbox.manifest import Manifest

        m = Manifest()
        result = m.remove_entry("nope")

        assert result is None

    def test_manifest_to_dict(self):
        from nono.sandbox.manifest import Manifest, LocalDir, OutputDir

        m = Manifest(
            entries={"data": LocalDir(src="/tmp/data")},
            output=OutputDir(path="output"),
            metadata={"version": "1.0"},
        )
        d = m.to_dict()

        assert "entries" in d
        assert "data" in d["entries"]
        assert d["entries"]["data"]["type"] == "local_dir"
        assert d["output"]["type"] == "output_dir"
        assert d["metadata"]["version"] == "1.0"


# ── Base types tests ──────────────────────────────────────────────────────────


class TestSandboxProvider:
    """Test SandboxProvider enum."""

    def test_all_providers_exist(self):
        from nono.sandbox.base import SandboxProvider

        providers = [p.value for p in SandboxProvider]

        assert "e2b" in providers
        assert "modal" in providers
        assert "daytona" in providers
        assert "blaxel" in providers
        assert "cloudflare" in providers
        assert "runloop" in providers
        assert "vercel" in providers

    def test_provider_count(self):
        from nono.sandbox.base import SandboxProvider

        assert len(SandboxProvider) == 7


class TestSandboxResult:
    """Test SandboxResult dataclass."""

    def test_success_property_true(self):
        from nono.sandbox.base import SandboxResult, SandboxStatus

        result = SandboxResult(status=SandboxStatus.COMPLETED, exit_code=0)

        assert result.success is True

    def test_success_property_false_on_failure(self):
        from nono.sandbox.base import SandboxResult, SandboxStatus

        result = SandboxResult(status=SandboxStatus.FAILED, exit_code=1)

        assert result.success is False

    def test_success_property_false_on_nonzero_exit(self):
        from nono.sandbox.base import SandboxResult, SandboxStatus

        result = SandboxResult(status=SandboxStatus.COMPLETED, exit_code=1)

        assert result.success is False

    def test_default_values(self):
        from nono.sandbox.base import SandboxResult, SandboxStatus

        result = SandboxResult()

        assert result.status == SandboxStatus.COMPLETED
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.output_files == {}


class TestSandboxRunConfig:
    """Test SandboxRunConfig dataclass."""

    def test_default_config(self):
        from nono.sandbox.base import SandboxRunConfig, SandboxProvider

        config = SandboxRunConfig()

        assert config.provider == SandboxProvider.E2B
        assert config.timeout == 300
        assert config.working_dir == "/home/user"
        assert config.keep_alive is False
        assert config.snapshot is False


# ── Client factory tests ─────────────────────────────────────────────────────


class TestGetSandboxClient:
    """Test get_sandbox_client factory."""

    def test_all_providers_resolve(self):
        from nono.sandbox import get_sandbox_client
        from nono.sandbox.base import SandboxProvider, BaseSandboxClient

        for provider in SandboxProvider:
            client = get_sandbox_client(provider)
            assert isinstance(client, BaseSandboxClient)

    def test_invalid_provider_raises(self):
        from nono.sandbox.sandbox_agent import get_sandbox_client

        with pytest.raises((ValueError, KeyError)):
            get_sandbox_client("not_a_provider")


# ── BaseSandboxClient tests ──────────────────────────────────────────────────


class TestBaseSandboxClient:
    """Test base client helper methods."""

    def test_get_api_key_missing_raises(self, monkeypatch):
        from nono.sandbox.clients.e2b import E2BSandboxClient

        monkeypatch.delenv("E2B_API_KEY", raising=False)
        client = E2BSandboxClient()

        with pytest.raises(EnvironmentError, match="E2B_API_KEY"):
            client._get_api_key()

    def test_get_api_key_present(self, monkeypatch):
        from nono.sandbox.clients.e2b import E2BSandboxClient

        monkeypatch.setenv("E2B_API_KEY", "test-key-123")
        client = E2BSandboxClient()

        assert client._get_api_key() == "test-key-123"

    def test_snapshot_not_implemented(self):
        from nono.sandbox.clients.e2b import E2BSandboxClient

        client = E2BSandboxClient()

        with pytest.raises(NotImplementedError):
            client.snapshot("some-id")

    def test_restore_not_implemented(self):
        from nono.sandbox.clients.e2b import E2BSandboxClient
        from nono.sandbox.base import SandboxRunConfig

        client = E2BSandboxClient()

        with pytest.raises(NotImplementedError):
            client.restore("snap-id", SandboxRunConfig())


# ── E2B client tests (mocked) ────────────────────────────────────────────────


class TestE2BSandboxClient:
    """Test E2B client with mocked SDK."""

    @patch("nono.sandbox.clients.e2b.E2BSandboxClient._get_api_key", return_value="fake")
    def test_execute_import_error(self, mock_key):
        from nono.sandbox.clients.e2b import E2BSandboxClient
        from nono.sandbox.base import SandboxRunConfig

        client = E2BSandboxClient()

        with pytest.raises(ImportError, match="e2b-code-interpreter"):
            client.execute("print('hi')", SandboxRunConfig())


# ── Modal client tests (mocked) ──────────────────────────────────────────────


class TestModalSandboxClient:
    """Test Modal client with mocked SDK."""

    def test_execute_import_error(self):
        from nono.sandbox.clients.modal import ModalSandboxClient
        from nono.sandbox.base import SandboxRunConfig

        client = ModalSandboxClient()

        with pytest.raises(ImportError, match="modal"):
            client.execute("print('hi')", SandboxRunConfig())


# ── Daytona client tests (mocked) ────────────────────────────────────────────


class TestDaytonaSandboxClient:
    """Test Daytona client with mocked SDK."""

    def test_execute_import_error(self, monkeypatch):
        from nono.sandbox.clients.daytona import DaytonaSandboxClient
        from nono.sandbox.base import SandboxRunConfig

        monkeypatch.setenv("DAYTONA_API_KEY", "fake")
        client = DaytonaSandboxClient()

        with pytest.raises(ImportError, match="daytona-sdk"):
            client.execute("print('hi')", SandboxRunConfig())


# ── SandboxAgent tests ────────────────────────────────────────────────────────


class TestSandboxAgent:
    """Test the SandboxAgent wrapper."""

    def test_agent_creation(self):
        from nono.sandbox import SandboxAgent, SandboxRunConfig
        from nono.sandbox.base import SandboxProvider

        config = SandboxRunConfig(provider=SandboxProvider.E2B)
        agent = SandboxAgent(name="TestAgent", sandbox_config=config)

        assert agent.name == "TestAgent"
        assert agent.sandbox_config.provider == SandboxProvider.E2B

    def test_agent_run_success(self):
        from nono.sandbox import SandboxAgent, SandboxRunConfig
        from nono.sandbox.base import SandboxResult, SandboxStatus
        from nono.agent.base import EventType

        mock_client = MagicMock()
        mock_client.execute.return_value = SandboxResult(
            status=SandboxStatus.COMPLETED,
            stdout="Hello World",
            exit_code=0,
            sandbox_id="sb-123",
            duration_seconds=1.5,
        )

        agent = SandboxAgent(name="TestAgent", client=mock_client)

        # Build a minimal InvocationContext
        from nono.agent.base import InvocationContext, Session

        session = Session()
        ctx = InvocationContext(
            session=session,
            user_message="print('Hello World')",
        )

        events = list(agent._run_impl(ctx))

        assert len(events) == 3  # STATE_UPDATE + TOOL_RESULT + AGENT_MESSAGE
        assert events[0].event_type == EventType.STATE_UPDATE
        assert events[1].event_type == EventType.TOOL_RESULT
        assert events[1].data["sandbox_id"] == "sb-123"
        assert events[2].event_type == EventType.AGENT_MESSAGE
        assert "Hello World" in events[2].content

    def test_agent_run_failure(self):
        from nono.sandbox import SandboxAgent
        from nono.sandbox.base import SandboxResult, SandboxStatus
        from nono.agent.base import EventType, InvocationContext, Session

        mock_client = MagicMock()
        mock_client.execute.return_value = SandboxResult(
            status=SandboxStatus.FAILED,
            stderr="SyntaxError: invalid syntax",
            exit_code=1,
        )

        agent = SandboxAgent(name="FailAgent", client=mock_client)
        session = Session()
        ctx = InvocationContext(
            session=session,
            user_message="invalid code",
        )

        events = list(agent._run_impl(ctx))
        agent_msg = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]

        assert len(agent_msg) == 1
        assert "failed" in agent_msg[0].content.lower()

    def test_agent_run_exception(self):
        from nono.sandbox import SandboxAgent
        from nono.agent.base import EventType, InvocationContext, Session

        mock_client = MagicMock()
        mock_client.execute.side_effect = RuntimeError("Connection lost")

        agent = SandboxAgent(name="ErrorAgent", client=mock_client)
        session = Session()
        ctx = InvocationContext(
            session=session,
            user_message="print('hi')",
        )

        events = list(agent._run_impl(ctx))
        error_events = [e for e in events if e.event_type == EventType.ERROR]

        assert len(error_events) == 1
        assert "Connection lost" in error_events[0].content

    def test_agent_stores_output_files(self):
        from nono.sandbox import SandboxAgent
        from nono.sandbox.base import SandboxResult, SandboxStatus
        from nono.agent.base import InvocationContext, Session

        mock_client = MagicMock()
        mock_client.execute.return_value = SandboxResult(
            status=SandboxStatus.COMPLETED,
            stdout="done",
            exit_code=0,
            sandbox_id="sb-456",
            output_files={"report.csv": b"a,b\n1,2\n"},
        )

        agent = SandboxAgent(name="FileAgent", client=mock_client)
        session = Session()
        ctx = InvocationContext(
            session=session,
            user_message="generate_report()",
        )

        list(agent._run_impl(ctx))

        item = session.shared_content.load("sandbox:report.csv")

        assert item is not None
        assert item.data == b"a,b\n1,2\n"


# ── Module import tests ──────────────────────────────────────────────────────


class TestModuleImports:
    """Verify the public API surface is importable."""

    def test_import_all_public_names(self):
        from nono.sandbox import (
            BaseSandboxClient,
            SandboxProvider,
            SandboxResult,
            SandboxRunConfig,
            SandboxStatus,
            AzureBlob,
            CloudflareR2,
            GCSBucket,
            LocalDir,
            LocalFile,
            Manifest,
            ManifestEntry,
            OutputDir,
            S3Bucket,
            SandboxAgent,
            get_sandbox_client,
        )

        # Just verify all names are importable
        assert SandboxProvider.E2B.value == "e2b"

    def test_import_clients(self):
        from nono.sandbox.clients import (
            E2BSandboxClient,
            ModalSandboxClient,
            DaytonaSandboxClient,
            BlaxelSandboxClient,
            CloudflareSandboxClient,
            RunloopSandboxClient,
            VercelSandboxClient,
        )

        assert E2BSandboxClient is not None
