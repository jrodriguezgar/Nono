"""Tests for :mod:`nono.connector.mcp_client`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nono.connector.mcp_client import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    _mcp_schema_to_parameters,
    _parse_call_result,
    _save_mcp_servers,
    load_mcp_config,
    mcp_tools,
)
from nono.agent.tool import FunctionTool


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mcp_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    input_schema: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    return tool


def _make_call_result(
    text: str = "result text",
    *,
    is_error: bool = False,
    structured: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock CallToolResult."""
    result = MagicMock()
    result.isError = is_error

    content_item = MagicMock()
    content_item.text = text
    result.content = [content_item]

    result.structuredContent = structured or None
    return result


# ── Schema Conversion ────────────────────────────────────────────────────────


class TestMcpSchemaToParameters:
    """Tests for ``_mcp_schema_to_parameters``."""

    def test_none_returns_empty_object(self):
        schema = _mcp_schema_to_parameters(None)

        assert schema == {"type": "object", "properties": {}}

    def test_empty_dict_returns_empty_object(self):
        schema = _mcp_schema_to_parameters({})

        assert schema == {"type": "object", "properties": {}}

    def test_full_schema_preserved(self):
        input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

        schema = _mcp_schema_to_parameters(input_schema)

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_schema_without_required(self):
        input_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }

        schema = _mcp_schema_to_parameters(input_schema)

        assert "required" not in schema


# ── Result Parsing ───────────────────────────────────────────────────────────


class TestParseCallResult:
    """Tests for ``_parse_call_result``."""

    def test_text_content(self):
        result = _make_call_result("Hello, world!")

        assert _parse_call_result(result) == "Hello, world!"

    def test_error_result(self):
        result = _make_call_result("Something went wrong", is_error=True)

        parsed = _parse_call_result(result)

        assert "[ERROR]" in parsed
        assert "Something went wrong" in parsed

    def test_structured_content(self):
        data = {"name": "Alice", "age": 30}
        result = _make_call_result(structured=data)

        parsed = _parse_call_result(result)

        assert json.loads(parsed) == data

    def test_empty_content(self):
        result = MagicMock()
        result.isError = False
        result.structuredContent = None
        result.content = []

        assert _parse_call_result(result) == ""


# ── MCPServerConfig ──────────────────────────────────────────────────────────


class TestMCPServerConfig:
    """Tests for ``MCPServerConfig``."""

    def test_stdio_config(self):
        cfg = MCPServerConfig(
            transport="stdio",
            command="uvx",
            args=("mcp-server-filesystem", "/tmp"),
        )

        assert cfg.transport == "stdio"
        assert cfg.command == "uvx"
        assert cfg.args == ("mcp-server-filesystem", "/tmp")

    def test_http_config(self):
        cfg = MCPServerConfig(
            transport="http",
            url="http://localhost:8000/mcp",
        )

        assert cfg.transport == "http"
        assert cfg.url == "http://localhost:8000/mcp"

    def test_frozen(self):
        cfg = MCPServerConfig(transport="stdio", command="test")

        with pytest.raises(AttributeError):
            cfg.transport = "http"  # type: ignore[misc]


# ── Factory Methods ──────────────────────────────────────────────────────────


class TestMCPClientFactories:
    """Tests for ``MCPClient`` factory methods."""

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_stdio_factory(self):
        client = MCPClient.stdio("uvx", args=["server", "/tmp"], name="fs")

        assert client._config.transport == "stdio"
        assert client._config.command == "uvx"
        assert client._config.args == ("server", "/tmp")
        assert client._config.name == "fs"

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_http_factory(self):
        client = MCPClient.http(
            "http://localhost:8000/mcp",
            headers={"Authorization": "Bearer token"},
        )

        assert client._config.transport == "http"
        assert client._config.url == "http://localhost:8000/mcp"
        assert client._config.headers == {"Authorization": "Bearer token"}

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_sse_factory(self):
        client = MCPClient.sse("http://localhost:8000/sse")

        assert client._config.transport == "sse"
        assert client._config.url == "http://localhost:8000/sse"

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", False)
    def test_missing_mcp_raises(self):
        with pytest.raises(ImportError, match="mcp"):
            MCPClient.stdio("uvx")


# ── Tool Wrapping ────────────────────────────────────────────────────────────


class TestMCPClientGetTools:
    """Tests for ``MCPClient.get_tools`` with mocked MCP sessions."""

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_make_function_tool(self):
        client = MCPClient(MCPServerConfig(transport="stdio", command="test"))
        mcp_tool = _make_mcp_tool(
            name="search",
            description="Search documents",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        ft = client._make_function_tool(mcp_tool)

        assert isinstance(ft, FunctionTool)
        assert ft.name == "search"
        assert ft.description == "Search documents"
        assert ft.parameters_schema["properties"]["query"]["type"] == "string"
        assert ft.parameters_schema["required"] == ["query"]

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_make_function_tool_no_description(self):
        client = MCPClient(MCPServerConfig(transport="stdio", command="test"))
        mcp_tool = _make_mcp_tool(name="fetch", description="")

        ft = client._make_function_tool(mcp_tool)

        assert ft.name == "fetch"
        assert "fetch" in ft.description

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_get_tools_caching(self):
        """Cached results are returned without re-discovery."""
        client = MCPClient(MCPServerConfig(transport="stdio", command="test"))
        fake_tools = [FunctionTool(fn=lambda: None, name="cached")]
        client._tools_cache = fake_tools

        result = client.get_tools()

        assert len(result) == 1
        assert result[0].name == "cached"
        # Returns a copy, not the same list
        assert result is not client._tools_cache

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_list_tool_names(self):
        client = MCPClient(MCPServerConfig(transport="stdio", command="test"))
        client._tools_cache = [
            FunctionTool(fn=lambda: None, name="beta"),
            FunctionTool(fn=lambda: None, name="alpha"),
        ]

        names = client.list_tool_names()

        assert names == ["alpha", "beta"]

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_repr(self):
        client = MCPClient(MCPServerConfig(transport="stdio", command="uvx"))

        assert "stdio" in repr(client)
        assert "uvx" in repr(client)


# ── Async Discovery Integration ──────────────────────────────────────────────


class TestMCPClientAsync:
    """Tests for async internals with fully mocked transport."""

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_wrap_tools_converts_all(self):
        """``_wrap_tools`` converts every MCP tool to FunctionTool."""
        client = MCPClient(MCPServerConfig(transport="stdio", command="test"))

        mock_session = AsyncMock()
        list_response = MagicMock()
        list_response.tools = [
            _make_mcp_tool("tool_a", "Tool A"),
            _make_mcp_tool("tool_b", "Tool B"),
        ]
        mock_session.list_tools = AsyncMock(return_value=list_response)

        tools = asyncio.run(client._wrap_tools(mock_session))

        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"
        assert all(isinstance(t, FunctionTool) for t in tools)

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_unsupported_transport_raises(self):
        client = MCPClient(MCPServerConfig(transport="grpc", command="test"))

        with pytest.raises(ValueError, match="grpc"):
            asyncio.run(client._discover_and_wrap_tools())


# ── Convenience Function ─────────────────────────────────────────────────────


class TestMcpToolsConvenience:
    """Tests for the ``mcp_tools`` convenience function."""

    def test_no_command_no_url_raises(self):
        with pytest.raises(ValueError, match="command.*url"):
            mcp_tools()

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    @patch.object(MCPClient, "get_tools")
    def test_auto_detects_stdio(self, mock_get_tools: MagicMock):
        mock_get_tools.return_value = [
            FunctionTool(fn=lambda: None, name="fs_tool"),
        ]

        tools = mcp_tools(command="uvx", args=["server"])

        assert len(tools) == 1
        assert tools[0].name == "fs_tool"

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    @patch.object(MCPClient, "get_tools")
    def test_auto_detects_http(self, mock_get_tools: MagicMock):
        mock_get_tools.return_value = []

        tools = mcp_tools(url="http://localhost:8000/mcp")

        assert tools == []

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    @patch.object(MCPClient, "get_tools")
    def test_explicit_sse(self, mock_get_tools: MagicMock):
        mock_get_tools.return_value = []

        mcp_tools(url="http://localhost/sse", transport="sse")

        mock_get_tools.assert_called_once()

    def test_stdio_without_command_raises(self):
        with pytest.raises(ValueError, match="command"):
            mcp_tools(transport="stdio")

    def test_http_without_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            mcp_tools(transport="http")

    def test_unknown_transport_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            mcp_tools(command="test", transport="unknown")


# ── Config Loading ───────────────────────────────────────────────────────────

_SAMPLE_TOML = """\
[google]
default_model = "gemini-3-flash-preview"

[mcp]

[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "/tmp"]
timeout = 30.0

[[mcp.servers]]
name = "my-api"
transport = "http"
url = "http://localhost:8000/mcp"
headers = { Authorization = "Bearer token" }
enabled = false
"""


class TestLoadMcpConfig:
    """Tests for ``load_mcp_config``."""

    def test_loads_servers_from_toml(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(_SAMPLE_TOML, encoding="utf-8")

        servers = load_mcp_config(cfg)

        assert len(servers) == 2
        assert servers[0]["name"] == "filesystem"
        assert servers[0]["command"] == "uvx"
        assert servers[1]["name"] == "my-api"
        assert servers[1]["url"] == "http://localhost:8000/mcp"
        assert servers[1]["enabled"] is False

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_mcp_config(tmp_path / "nonexistent.toml") == []

    def test_no_mcp_section_returns_empty(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[google]\ndefault_model = "test"\n', encoding="utf-8")

        assert load_mcp_config(cfg) == []


class TestSaveMcpServers:
    """Tests for ``_save_mcp_servers``."""

    def test_save_creates_mcp_section(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('[google]\ndefault_model = "gemini"\n', encoding="utf-8")

        servers = [
            {"name": "fs", "transport": "stdio", "command": "uvx", "args": ["server"]},
        ]
        _save_mcp_servers(servers, cfg)

        content = cfg.read_text(encoding="utf-8")
        assert "[[mcp.servers]]" in content
        assert 'name = "fs"' in content
        assert 'command = "uvx"' in content
        assert '[google]' in content  # preserved

    def test_save_replaces_existing_mcp(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(_SAMPLE_TOML, encoding="utf-8")

        _save_mcp_servers([{"name": "new", "transport": "http", "url": "http://x"}], cfg)

        content = cfg.read_text(encoding="utf-8")
        assert 'name = "new"' in content
        assert 'name = "filesystem"' not in content  # old removed

    def test_save_empty_removes_mcp(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(_SAMPLE_TOML, encoding="utf-8")

        _save_mcp_servers([], cfg)

        content = cfg.read_text(encoding="utf-8")
        assert "[[mcp.servers]]" not in content
        assert '[google]' in content

    def test_roundtrip(self, tmp_path):
        """Save then load returns equivalent data."""
        cfg = tmp_path / "config.toml"
        cfg.write_text("[google]\n", encoding="utf-8")

        original = [
            {"name": "a", "transport": "stdio", "command": "cmd", "args": ["x"]},
            {"name": "b", "transport": "http", "url": "http://localhost"},
        ]
        _save_mcp_servers(original, cfg)
        loaded = load_mcp_config(cfg)

        assert len(loaded) == 2
        assert loaded[0]["name"] == "a"
        assert loaded[1]["name"] == "b"


# ── MCPManager ───────────────────────────────────────────────────────────────


class TestMCPManager:
    """Tests for ``MCPManager``."""

    def test_from_config(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(_SAMPLE_TOML, encoding="utf-8")

        mgr = MCPManager.from_config(cfg)

        assert len(mgr) == 2
        assert "filesystem" in mgr
        assert "my-api" in mgr

    def test_add_and_list(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")

        mgr.add("test-server", command="npx", args=["-y", "server"])

        assert "test-server" in mgr
        servers = mgr.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "test-server"
        assert servers[0]["command"] == "npx"

    def test_remove(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("deleteme", command="test")

        assert mgr.remove("deleteme") is True
        assert "deleteme" not in mgr
        assert mgr.remove("nonexistent") is False

    def test_enable_disable(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("srv", command="test")

        assert mgr.disable("srv") is True
        servers = mgr.list_servers()
        assert servers[0].get("enabled") is False

        assert mgr.enable("srv") is True
        servers = mgr.list_servers()
        assert servers[0].get("enabled") is True

        assert mgr.enable("nonexistent") is False
        assert mgr.disable("nonexistent") is False

    def test_names_sorted(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("zebra", command="z")
        mgr.add("alpha", command="a")

        assert mgr.names == ["alpha", "zebra"]

    def test_save_and_reload(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("[google]\n", encoding="utf-8")

        mgr = MCPManager(config_path=cfg)
        mgr.add("saved", command="uvx", args=["server"])
        mgr.save()

        mgr2 = MCPManager.from_config(cfg)
        assert "saved" in mgr2
        assert mgr2.list_servers()[0]["command"] == "uvx"

    @patch("nono.connector.mcp_client._MCP_AVAILABLE", True)
    def test_get_client_disabled_raises(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("disabled", command="test")
        mgr.disable("disabled")

        with pytest.raises(ValueError, match="disabled"):
            mgr.get_client("disabled")

    def test_get_client_unknown_raises(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")

        with pytest.raises(KeyError, match="unknown"):
            mgr.get_client("unknown")

    def test_repr(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("a", command="test")
        mgr.add("b", command="test")
        mgr.disable("b")

        r = repr(mgr)
        assert "servers=2" in r
        assert "enabled=1" in r

    def test_add_updates_existing(self, tmp_path):
        mgr = MCPManager(config_path=tmp_path / "config.toml")
        mgr.add("srv", command="old")
        mgr.add("srv", command="new")

        assert len(mgr) == 1
        assert mgr.list_servers()[0]["command"] == "new"
