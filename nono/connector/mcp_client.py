"""
MCP Client — connect to external Model Context Protocol servers.

Bridges MCP servers (stdio, SSE, Streamable HTTP) into Nono's tool system,
converting remote MCP tools into ``FunctionTool`` instances that any Nono
agent can use.

Requires the ``mcp`` optional dependency::

    pip install nono[mcp]

Usage::

    from nono.connector.mcp_client import MCPClient

    # Stdio server
    client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/tmp"])
    tools = client.get_tools()

    # HTTP server
    client = MCPClient.http("http://localhost:8000/mcp")
    tools = client.get_tools()

    # Attach to an agent
    from nono.agent import Agent
    agent = Agent(
        name="assistant",
        instruction="You are a helpful assistant.",
        tools=client.get_tools(),
    )

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nono.agent.tool import FunctionTool

logger = logging.getLogger("Nono.Connector.MCP")

try:
    from mcp import ClientSession, StdioServerParameters, types as mcp_types
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.sse import sse_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "load_mcp_config",
    "mcp_tools",
]


def _require_mcp() -> None:
    """Raise if the ``mcp`` package is not installed."""
    if not _MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. "
            "Install it with: pip install nono[mcp]"
        )


# ── Server Configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for connecting to an MCP server.

    Attributes:
        transport: Transport type (``"stdio"``, ``"http"``, ``"sse"``).
        command: Executable command (stdio only).
        args: Command arguments (stdio only).
        env: Environment variables (stdio only).
        url: Server URL (http/sse only).
        headers: HTTP headers (http/sse only).
        timeout: Connection timeout in seconds.
        name: Human-readable server name for logging.
    """

    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    name: str = ""


# ── MCP → FunctionTool Conversion ────────────────────────────────────────────


def _mcp_schema_to_parameters(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Convert an MCP tool input schema to Nono parameters schema.

    Args:
        input_schema: The ``inputSchema`` from an MCP tool definition.

    Returns:
        OpenAI-compatible JSON Schema for ``FunctionTool.parameters_schema``.
    """
    if not input_schema:
        return {"type": "object", "properties": {}}

    schema: dict[str, Any] = {
        "type": input_schema.get("type", "object"),
        "properties": input_schema.get("properties", {}),
    }

    if "required" in input_schema:
        schema["required"] = input_schema["required"]

    return schema


def _parse_call_result(result: Any) -> str:
    """Extract text from an MCP ``CallToolResult``.

    Handles text content, structured content, error results, and
    embedded resources.

    Args:
        result: ``CallToolResult`` from MCP ``session.call_tool()``.

    Returns:
        String representation of the result.
    """
    if result.isError:
        texts = []
        for content in result.content:
            if hasattr(content, "text"):
                texts.append(content.text)
        error_msg = "\n".join(texts) if texts else "MCP tool returned an error"
        logger.warning("MCP tool error: %s", error_msg)
        return f"[ERROR] {error_msg}"

    # Prefer structured content if available
    if hasattr(result, "structuredContent") and result.structuredContent:
        return json.dumps(result.structuredContent, ensure_ascii=False)

    # Fall back to text content blocks
    parts: list[str] = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        elif hasattr(content, "resource"):
            resource = content.resource
            if hasattr(resource, "text"):
                parts.append(resource.text)
            else:
                parts.append(f"[binary resource: {getattr(resource, 'uri', 'unknown')}]")
        elif hasattr(content, "data"):
            parts.append(f"[image: {getattr(content, 'mimeType', 'unknown')}]")

    return "\n".join(parts) if parts else ""


# ── Core Client ──────────────────────────────────────────────────────────────

_MCP_SYNC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)


class MCPClient:
    """Client that connects to an MCP server and exposes its tools as ``FunctionTool``.

    Use the factory methods :meth:`stdio`, :meth:`http`, or :meth:`sse` to
    create an instance.  Then call :meth:`get_tools` to retrieve Nono-compatible
    tools.

    The client manages its own async event loop internally so that callers
    do not need to use ``async``/``await``.

    Example::

        client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/tmp"])
        tools = client.get_tools()

        # Use in an agent
        agent = Agent(name="fs", instruction="...", tools=tools)
    """

    def __init__(self, config: MCPServerConfig) -> None:
        _require_mcp()
        self._config = config
        self._tools_cache: list[FunctionTool] | None = None
        self._server_name = config.name or config.command or config.url or "MCP"

    # ── Factory Methods ───────────────────────────────────────────────

    @classmethod
    def stdio(
        cls,
        command: str,
        *,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        name: str = "",
    ) -> MCPClient:
        """Create a client for a stdio-based MCP server.

        Args:
            command: Executable to launch (e.g. ``"uvx"``, ``"npx"``,
                ``"python"``).
            args: Command-line arguments for the server process.
            env: Environment variables for the subprocess.
            timeout: Connection timeout in seconds.
            name: Human-readable name for logging.

        Returns:
            Configured ``MCPClient`` instance.

        Example::

            client = MCPClient.stdio(
                "uvx",
                args=["mcp-server-filesystem", "/home/user/docs"],
            )
        """
        return cls(MCPServerConfig(
            transport="stdio",
            command=command,
            args=tuple(args or []),
            env=env or {},
            timeout=timeout,
            name=name or command,
        ))

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        name: str = "",
    ) -> MCPClient:
        """Create a client for a Streamable HTTP MCP server.

        Args:
            url: Server endpoint URL (e.g. ``"http://localhost:8000/mcp"``).
            headers: Additional HTTP headers.
            timeout: Connection timeout in seconds.
            name: Human-readable name for logging.

        Returns:
            Configured ``MCPClient`` instance.

        Example::

            client = MCPClient.http("http://localhost:8000/mcp")
        """
        return cls(MCPServerConfig(
            transport="http",
            url=url,
            headers=headers or {},
            timeout=timeout,
            name=name or url,
        ))

    @classmethod
    def sse(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        name: str = "",
    ) -> MCPClient:
        """Create a client for an SSE-based MCP server (legacy).

        Args:
            url: Server SSE endpoint URL.
            headers: Additional HTTP headers.
            timeout: Connection timeout in seconds.
            name: Human-readable name for logging.

        Returns:
            Configured ``MCPClient`` instance.

        Example::

            client = MCPClient.sse("http://localhost:8000/sse")
        """
        return cls(MCPServerConfig(
            transport="sse",
            url=url,
            headers=headers or {},
            timeout=timeout,
            name=name or url,
        ))

    # ── Tool Discovery ────────────────────────────────────────────────

    def get_tools(self, *, refresh: bool = False) -> list[FunctionTool]:
        """Discover MCP tools and return them as ``FunctionTool`` instances.

        Results are cached after the first call.  Pass ``refresh=True``
        to re-query the server.

        Args:
            refresh: Force re-discovery of tools.

        Returns:
            List of ``FunctionTool`` wrapping each remote MCP tool.

        Raises:
            ImportError: If the ``mcp`` package is not installed.
            ConnectionError: If the server cannot be reached.
        """
        if self._tools_cache is not None and not refresh:
            return list(self._tools_cache)

        tools = self._run_async(self._discover_and_wrap_tools())
        self._tools_cache = tools
        logger.info(
            "Discovered %d tools from MCP server %r",
            len(tools), self._server_name,
        )
        return list(tools)

    def list_tool_names(self) -> list[str]:
        """Return names of all available MCP tools.

        Returns:
            Sorted list of tool names.
        """
        return sorted(t.name for t in self.get_tools())

    # ── Async Internals ───────────────────────────────────────────────

    async def _discover_and_wrap_tools(self) -> list[FunctionTool]:
        """Connect to the MCP server, list tools, and wrap each one.

        Returns:
            List of ``FunctionTool`` for each remote tool.
        """
        config = self._config

        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=config.env if config.env else None,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await self._wrap_tools(session)

        elif config.transport == "http":
            async with streamable_http_client(
                config.url,
                headers=config.headers if config.headers else None,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await self._wrap_tools(session)

        elif config.transport == "sse":
            async with sse_client(
                config.url,
                headers=config.headers if config.headers else None,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await self._wrap_tools(session)

        else:
            raise ValueError(
                f"Unsupported MCP transport: {config.transport!r}. "
                "Use 'stdio', 'http', or 'sse'."
            )

    async def _wrap_tools(self, session: ClientSession) -> list[FunctionTool]:
        """List tools from a connected session and wrap each as ``FunctionTool``.

        Args:
            session: Active MCP ``ClientSession``.

        Returns:
            List of wrapped tools.
        """
        response = await session.list_tools()
        tools: list[FunctionTool] = []

        for mcp_tool in response.tools:
            ft = self._make_function_tool(mcp_tool)
            tools.append(ft)
            logger.debug(
                "Wrapped MCP tool %r from server %r",
                mcp_tool.name, self._server_name,
            )

        return tools

    def _make_function_tool(self, mcp_tool: Any) -> FunctionTool:
        """Create a ``FunctionTool`` that calls the MCP tool on invocation.

        Each invocation opens a fresh connection to the server, calls
        the tool, and returns the result.  This is stateless by design —
        MCP servers handle each request independently.

        Args:
            mcp_tool: An MCP ``Tool`` object from ``list_tools()``.

        Returns:
            A ``FunctionTool`` wrapping the remote call.
        """
        tool_name: str = mcp_tool.name
        description: str = mcp_tool.description or f"MCP tool: {tool_name}"
        params_schema = _mcp_schema_to_parameters(
            mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else None,
        )

        def _invoke(**kwargs: Any) -> str:
            return self._run_async(self._call_remote_tool(tool_name, kwargs))

        return FunctionTool(
            fn=_invoke,
            name=tool_name,
            description=description,
            parameters_schema=params_schema,
        )

    async def _call_remote_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> str:
        """Open a connection and call a single MCP tool.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass.

        Returns:
            String result.
        """
        config = self._config

        if config.transport == "stdio":
            server_params = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=config.env if config.env else None,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=config.timeout,
                    )
                    return _parse_call_result(result)

        elif config.transport == "http":
            async with streamable_http_client(
                config.url,
                headers=config.headers if config.headers else None,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=config.timeout,
                    )
                    return _parse_call_result(result)

        elif config.transport == "sse":
            async with sse_client(
                config.url,
                headers=config.headers if config.headers else None,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=config.timeout,
                    )
                    return _parse_call_result(result)

        else:
            raise ValueError(f"Unsupported transport: {config.transport!r}")

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Run an async coroutine from sync context.

        Uses the running loop if available, otherwise creates a new one.

        Args:
            coro: Awaitable to run.

        Returns:
            The coroutine's result.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = _MCP_SYNC_POOL.submit(asyncio.run, coro)
            return future.result(timeout=300)
        else:
            return asyncio.run(coro)

    def __repr__(self) -> str:
        cfg = self._config
        transport = cfg.transport
        target = cfg.command if transport == "stdio" else cfg.url
        cached = len(self._tools_cache) if self._tools_cache is not None else "?"
        return f"MCPClient({transport}={target!r}, tools={cached})"


# ── Convenience Function ─────────────────────────────────────────────────────


def mcp_tools(
    *,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    transport: str = "auto",
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    name: str = "",
) -> list[FunctionTool]:
    """Convenience function to get MCP tools in one call.

    Automatically detects the transport: stdio if ``command`` is provided,
    HTTP if ``url`` is provided.

    Args:
        command: Executable for stdio transport.
        args: Arguments for stdio transport.
        url: URL for HTTP/SSE transport.
        transport: ``"auto"``, ``"stdio"``, ``"http"``, or ``"sse"``.
        headers: HTTP headers (http/sse only).
        timeout: Connection timeout in seconds.
        name: Human-readable name for logging.

    Returns:
        List of ``FunctionTool`` from the MCP server.

    Raises:
        ValueError: If neither ``command`` nor ``url`` is specified.

    Example::

        from nono.connector.mcp_client import mcp_tools
        from nono.agent import Agent

        agent = Agent(
            name="assistant",
            instruction="You can access the filesystem.",
            tools=mcp_tools(command="uvx", args=["mcp-server-filesystem", "/tmp"]),
        )
    """
    if transport == "auto":
        if command:
            transport = "stdio"
        elif url:
            transport = "http"
        else:
            raise ValueError("Provide 'command' (stdio) or 'url' (http/sse).")

    if transport == "stdio":
        if not command:
            raise ValueError("'command' is required for stdio transport.")
        client = MCPClient.stdio(
            command, args=args, timeout=timeout, name=name,
        )
    elif transport == "http":
        if not url:
            raise ValueError("'url' is required for HTTP transport.")
        client = MCPClient.http(
            url, headers=headers, timeout=timeout, name=name,
        )
    elif transport == "sse":
        if not url:
            raise ValueError("'url' is required for SSE transport.")
        client = MCPClient.sse(
            url, headers=headers, timeout=timeout, name=name,
        )
    else:
        raise ValueError(f"Unknown transport: {transport!r}")

    return client.get_tools()


# ── Configuration Loading ────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"


def load_mcp_config(
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load the ``[[mcp.servers]]`` array from *config.toml*.

    Args:
        config_path: Explicit path to config.toml.  When *None*, the
                     default ``nono/config/config.toml`` is used.

    Returns:
        List of server dicts with keys ``name``, ``transport``, etc.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.debug("Config file not found at %s — no MCP servers.", config_path)
        return []

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        logger.warning("Failed to read MCP config: %s", exc)
        return []

    mcp_section = data.get("mcp", {})
    return list(mcp_section.get("servers", []))


def _save_mcp_servers(
    servers: list[dict[str, Any]],
    config_path: str | Path | None = None,
) -> None:
    """Write the ``[[mcp.servers]]`` section back to *config.toml*.

    Preserves all other sections in the file.  Creates the file if it
    does not exist.

    Args:
        servers: List of server dicts to write.
        config_path: Target config file path.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    # Read existing content (preserve everything outside [mcp])
    existing_lines: list[str] = []
    if config_path.exists():
        existing_lines = config_path.read_text(encoding="utf-8").splitlines()

    # Remove existing [[mcp.servers]] blocks and [mcp] section
    cleaned: list[str] = []
    in_mcp = False
    for line in existing_lines:
        stripped = line.strip()

        if stripped == "[mcp]" or stripped.startswith("[[mcp."):
            in_mcp = True
            continue

        if in_mcp:
            # Exit MCP section when we hit another top-level section
            if stripped.startswith("[") and not stripped.startswith("[[mcp."):
                in_mcp = False
                cleaned.append(line)
            continue

        cleaned.append(line)

    # Remove trailing blank lines
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    # Build new MCP section
    def _esc(val: str) -> str:
        """Escape a value for TOML double-quoted string."""
        return val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")

    mcp_lines: list[str] = []
    if servers:
        mcp_lines.append("")
        mcp_lines.append("[mcp]")
        mcp_lines.append(f"# {len(servers)} configured MCP server(s)")
        mcp_lines.append("")

        for srv in servers:
            mcp_lines.append("[[mcp.servers]]")
            mcp_lines.append(f'name = "{_esc(srv["name"])}"')
            mcp_lines.append(f'transport = "{_esc(srv.get("transport", "stdio"))}"')

            if srv.get("command"):
                mcp_lines.append(f'command = "{_esc(srv["command"])}"')

            if srv.get("args"):
                args_str = ", ".join(f'"{_esc(a)}"' for a in srv["args"])
                mcp_lines.append(f"args = [{args_str}]")

            if srv.get("url"):
                mcp_lines.append(f'url = "{_esc(srv["url"])}"')

            if srv.get("headers"):
                pairs = ", ".join(
                    f'{_esc(k)} = "{_esc(v)}"' for k, v in srv["headers"].items()
                )
                mcp_lines.append(f"headers = {{ {pairs} }}")

            if srv.get("env"):
                pairs = ", ".join(
                    f'{_esc(k)} = "{_esc(v)}"' for k, v in srv["env"].items()
                )
                mcp_lines.append(f"env = {{ {pairs} }}")

            if "timeout" in srv:
                mcp_lines.append(f"timeout = {int(srv['timeout'])}")

            if srv.get("enabled") is False:
                mcp_lines.append("enabled = false")

            mcp_lines.append("")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    final = "\n".join(cleaned + mcp_lines) + "\n"
    config_path.write_text(final, encoding="utf-8")
    logger.info("Saved %d MCP server(s) to %s", len(servers), config_path)


# ── MCPManager ───────────────────────────────────────────────────────────────


class MCPManager:
    """Central registry for MCP servers — load from config, add, remove, query.

    Manages the lifecycle of MCP server connections declared in
    ``config.toml`` under ``[[mcp.servers]]``.

    Usage::

        from nono.connector.mcp_client import MCPManager

        mgr = MCPManager.from_config()          # load from config.toml
        mgr.add("github", command="uvx", args=["mcp-server-github"])
        mgr.save()                               # persist to config.toml

        tools = mgr.get_tools("github")          # tools from one server
        all_tools = mgr.get_all_tools()           # tools from all servers

        mgr.remove("github")
        mgr.save()

    Args:
        config_path: Path to config.toml for load/save.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._servers: dict[str, dict[str, Any]] = {}
        self._clients: dict[str, MCPClient] = {}

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
    ) -> MCPManager:
        """Load MCP servers from ``config.toml``.

        Args:
            config_path: Explicit path.  ``None`` uses the default.

        Returns:
            Populated ``MCPManager``.
        """
        mgr = cls(config_path=config_path)
        raw = load_mcp_config(mgr._config_path)

        for entry in raw:
            name = entry.get("name", "")
            if not name:
                logger.warning("Skipping MCP server entry without 'name'.")
                continue
            mgr._servers[name] = dict(entry)

        logger.info("Loaded %d MCP server(s) from config.", len(mgr._servers))
        return mgr

    # ── CRUD ──────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        *,
        transport: str = "stdio",
        command: str = "",
        args: list[str] | None = None,
        url: str = "",
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        enabled: bool = True,
    ) -> None:
        """Add or update an MCP server.

        Args:
            name: Unique server identifier.
            transport: ``"stdio"``, ``"http"``, or ``"sse"``.
            command: Executable (stdio).
            args: Command arguments (stdio).
            url: Server URL (http/sse).
            headers: HTTP headers (http/sse).
            env: Environment variables (stdio).
            timeout: Connection timeout.
            enabled: Whether the server is active.
        """
        entry: dict[str, Any] = {
            "name": name,
            "transport": transport,
            "enabled": enabled,
            "timeout": timeout,
        }

        if command:
            entry["command"] = command
        if args:
            entry["args"] = list(args)
        if url:
            entry["url"] = url
        if headers:
            entry["headers"] = dict(headers)
        if env:
            entry["env"] = dict(env)

        self._servers[name] = entry
        self._clients.pop(name, None)  # invalidate cached client
        logger.info("Added MCP server %r (%s)", name, transport)

    def remove(self, name: str) -> bool:
        """Remove an MCP server by name.

        Args:
            name: Server identifier.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        if name not in self._servers:
            return False

        del self._servers[name]
        self._clients.pop(name, None)
        logger.info("Removed MCP server %r", name)
        return True

    def enable(self, name: str) -> bool:
        """Enable a previously disabled server.

        Args:
            name: Server identifier.

        Returns:
            ``True`` if found and enabled.
        """
        if name not in self._servers:
            return False
        self._servers[name]["enabled"] = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a server without removing it.

        Args:
            name: Server identifier.

        Returns:
            ``True`` if found and disabled.
        """
        if name not in self._servers:
            return False
        self._servers[name]["enabled"] = False
        self._clients.pop(name, None)
        return True

    # ── Query ─────────────────────────────────────────────────────────

    def list_servers(self) -> list[dict[str, Any]]:
        """Return metadata for all configured servers.

        Returns:
            Sorted list of server dicts.
        """
        return sorted(self._servers.values(), key=lambda s: s["name"])

    @property
    def names(self) -> list[str]:
        """Sorted list of server names."""
        return sorted(self._servers.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._servers

    def __len__(self) -> int:
        return len(self._servers)

    # ── Tool Access ───────────────────────────────────────────────────

    def get_client(self, name: str) -> MCPClient:
        """Get or create an ``MCPClient`` for a configured server.

        Args:
            name: Server identifier.

        Returns:
            ``MCPClient`` instance.

        Raises:
            KeyError: If the server is not configured.
            ValueError: If the server is disabled.
        """
        if name not in self._servers:
            raise KeyError(f"MCP server {name!r} not configured.")

        srv = self._servers[name]
        if srv.get("enabled") is False:
            raise ValueError(f"MCP server {name!r} is disabled.")

        if name not in self._clients:
            self._clients[name] = self._build_client(srv)

        return self._clients[name]

    def get_tools(self, name: str) -> list[FunctionTool]:
        """Get tools from a specific MCP server.

        Args:
            name: Server identifier.

        Returns:
            ``FunctionTool`` list from that server.
        """
        return self.get_client(name).get_tools()

    def get_all_tools(self) -> list[FunctionTool]:
        """Get tools from all enabled MCP servers.

        Returns:
            Combined list of ``FunctionTool`` from every enabled server.
        """
        all_tools: list[FunctionTool] = []
        for name in self.names:
            srv = self._servers[name]
            if srv.get("enabled") is False:
                continue
            try:
                all_tools.extend(self.get_tools(name))
            except Exception as exc:
                logger.warning(
                    "Failed to get tools from MCP server %r: %s", name, exc,
                )
        return all_tools

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, config_path: str | Path | None = None) -> None:
        """Write current servers to ``config.toml``.

        Args:
            config_path: Override destination (default: original path).
        """
        dest = Path(config_path) if config_path else self._config_path
        _save_mcp_servers(self.list_servers(), dest)

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _build_client(srv: dict[str, Any]) -> MCPClient:
        """Create an ``MCPClient`` from a server config dict.

        Args:
            srv: Server configuration dict.

        Returns:
            Configured ``MCPClient``.
        """
        transport = srv.get("transport", "stdio")
        name = srv.get("name", "")
        timeout = float(srv.get("timeout", 30.0))

        if transport == "stdio":
            return MCPClient.stdio(
                srv.get("command", ""),
                args=srv.get("args"),
                env=srv.get("env"),
                timeout=timeout,
                name=name,
            )
        elif transport == "http":
            return MCPClient.http(
                srv.get("url", ""),
                headers=srv.get("headers"),
                timeout=timeout,
                name=name,
            )
        elif transport == "sse":
            return MCPClient.sse(
                srv.get("url", ""),
                headers=srv.get("headers"),
                timeout=timeout,
                name=name,
            )
        else:
            raise ValueError(
                f"Unsupported transport {transport!r} for server {name!r}."
            )

    def __repr__(self) -> str:
        enabled = sum(
            1 for s in self._servers.values() if s.get("enabled", True)
        )
        return f"MCPManager(servers={len(self)}, enabled={enabled})"
