# MCP Client Integration

> Connect external [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) servers to Nono agents — use remote tools as if they were native.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Transport Types](#transport-types)
  - [Stdio (Subprocess)](#stdio-subprocess)
  - [Streamable HTTP](#streamable-http)
  - [SSE (Legacy)](#sse-legacy)
- [Managing Servers (CLI)](#managing-servers-cli)
  - [List Servers](#list-servers)
  - [Add a Server](#add-a-server)
  - [Remove a Server](#remove-a-server)
  - [Enable / Disable](#enable--disable)
  - [Discover Tools](#discover-tools)
- [Managing Servers (config.toml)](#managing-servers-configtoml)
- [Managing Servers (Python)](#managing-servers-python)
- [Usage Patterns](#usage-patterns)
  - [Attach Tools to an Agent](#attach-tools-to-an-agent)
  - [Combine with Native Tools](#combine-with-native-tools)
  - [Use with Skills](#use-with-skills)
  - [Use with Workflows](#use-with-workflows)
  - [Convenience Function](#convenience-function)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Popular MCP Servers](#popular-mcp-servers)
- [Troubleshooting](#troubleshooting)

---

## Overview

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that allows AI applications to connect to external tools, data, and services. Nono's MCP client lets you:

| Capability         | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **Connect**  | Reach any MCP server via stdio, HTTP, or SSE                 |
| **Discover** | Auto-discover tools exposed by the server                    |
| **Convert**  | Convert MCP tools into Nono `FunctionTool` instances       |
| **Use**      | Attach converted tools to any Nono agent, skill, or workflow |

```
┌───────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Nono Agent      │      │   MCPClient      │      │   MCP Server     │
│                   │      │                  │      │                  │
│  tools=[ft1, ft2] │◄─────│  get_tools()     │◄─────│  list_tools()    │
│                   │      │                  │      │  call_tool()     │
│  LLM calls ft1()  │─────►│  _invoke()       │─────►│  execute tool    │
└───────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## Installation

MCP support is an optional dependency:

```bash
# With pip
pip install nono[mcp]

# With uv
uv add "nono[mcp]"
```

This installs the [`mcp`](https://pypi.org/project/mcp/) Python SDK (v1.0+).

---

## Quick Start

```python
from nono.agent import Agent, Runner
from nono.connector.mcp_client import MCPClient

# Connect to an MCP server
client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/home/user/docs"])

# Get tools from the server
tools = client.get_tools()
print([t.name for t in tools])
# ['read_file', 'write_file', 'list_directory', ...]

# Use them in an agent
agent = Agent(
    name="file_assistant",
    instruction="You help users manage their files.",
    tools=tools,
)
result = Runner(agent).run("List all markdown files in the docs directory")
print(result)
```

---

## Transport Types

### Stdio (Subprocess)

Launches the MCP server as a subprocess and communicates via stdin/stdout. Best for local tools.

```python
from nono.connector.mcp_client import MCPClient

client = MCPClient.stdio(
    "uvx",                                        # executable
    args=["mcp-server-filesystem", "/tmp"],        # server arguments
    env={"HOME": "/home/user"},                    # optional env vars
    timeout=30.0,                                  # connection timeout
    name="filesystem",                             # for logging
)
```

### Streamable HTTP

Connects to an MCP server exposed over HTTP (the recommended remote transport).

```python
client = MCPClient.http(
    "http://localhost:8000/mcp",
    headers={"Authorization": "Bearer <token>"},   # optional auth
    timeout=30.0,
)
```

### SSE (Legacy)

For servers that use the older Server-Sent Events transport.

```python
client = MCPClient.sse(
    "http://localhost:8000/sse",
    headers={"Authorization": "Bearer <token>"},
)
```

> **Note:** SSE is considered legacy. Prefer Streamable HTTP for new servers.

---

## Managing Servers (CLI)

The `nono mcp` command manages the MCP servers declared in `config.toml`.

### List Servers

```bash
nono mcp list
# or simply
nono mcp
```

```
========================== MCP Servers ===========================
Name        Transport  Target                              Enabled
──────────  ─────────  ──────────────────────────────────  ───────
filesystem  stdio      uvx mcp-server-filesystem /tmp      yes
my-api      http       http://localhost:8000/mcp            no
```

### Add a Server

```bash
# Stdio server
nono mcp add filesystem --command uvx --args mcp-server-filesystem /home/user/docs

# HTTP server
nono mcp add my-api --transport http --url http://localhost:8000/mcp

# With environment variables and headers
nono mcp add github --command uvx --args mcp-server-github \
    --env GITHUB_TOKEN=ghp_xxx

nono mcp add remote --transport http --url https://api.example.com/mcp \
    --header Authorization="Bearer token" --timeout 60
```

### Remove a Server

```bash
nono mcp remove filesystem
# alias
nono mcp rm filesystem
```

### Enable / Disable

```bash
nono mcp disable my-api    # keep config, stop using
nono mcp enable my-api     # re-activate
```

### Discover Tools

```bash
# Tools from one server
nono mcp tools filesystem

# Tools from all enabled servers
nono mcp tools
```

---

## Managing Servers (config.toml)

MCP servers are declared in `config.toml` under `[[mcp.servers]]`:

```toml
[mcp]

[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "/home/user/docs"]
timeout = 30.0

[[mcp.servers]]
name = "my-api"
transport = "http"
url = "http://localhost:8000/mcp"
headers = { Authorization = "Bearer <token>" }
enabled = false
```

| Field         | Type          | Default            | Description                                  |
| ------------- | ------------- | ------------------ | -------------------------------------------- |
| `name`      | `str`       | **required** | Unique server identifier                     |
| `transport` | `str`       | `"stdio"`        | `"stdio"`, `"http"`, or `"sse"`        |
| `command`   | `str`       | `""`             | Executable (stdio only)                      |
| `args`      | `list[str]` | `[]`             | Command arguments (stdio only)               |
| `env`       | `dict`      | `{}`             | Environment variables (stdio only)           |
| `url`       | `str`       | `""`             | Server URL (http/sse only)                   |
| `headers`   | `dict`      | `{}`             | HTTP headers (http/sse only)                 |
| `timeout`   | `float`     | `30.0`           | Connection timeout (seconds)                 |
| `enabled`   | `bool`      | `true`           | Set to `false` to disable without removing |

---

## Managing Servers (Python)

Use `MCPManager` for programmatic server management:

```python
from nono.connector.mcp_client import MCPManager

# Load servers from config.toml
mgr = MCPManager.from_config()

# Add a new server
mgr.add("github", command="uvx", args=["mcp-server-github"])

# Save changes to config.toml
mgr.save()

# Remove a server
mgr.remove("github")
mgr.save()

# Enable / disable
mgr.disable("my-api")
mgr.enable("my-api")
mgr.save()

# List all servers
for srv in mgr.list_servers():
    print(srv["name"], srv.get("enabled", True))

# Get tools from one or all servers
tools = mgr.get_tools("filesystem")
all_tools = mgr.get_all_tools()
```

---

## Usage Patterns

### Attach Tools to an Agent

```python
from nono.agent import Agent, Runner
from nono.connector.mcp_client import MCPClient

# Filesystem tools
fs_client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/data"])

agent = Agent(
    name="data_analyst",
    instruction="Analyze data files. Use your tools to read files when needed.",
    tools=fs_client.get_tools(),
)

result = Runner(agent).run("Read and summarize the contents of report.csv")
```

### Combine with Native Tools

```python
from nono.agent import Agent, Runner, tool
from nono.connector.mcp_client import MCPClient

@tool(description="Calculate the average of a list of numbers.")
def calculate_average(numbers: str) -> str:
    nums = [float(n) for n in numbers.split(",")]
    return str(sum(nums) / len(nums))

fs_tools = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/data"]).get_tools()

agent = Agent(
    name="analyst",
    instruction="Read files and compute statistics.",
    tools=[calculate_average, *fs_tools],
)
```

### Use with Skills

```python
from nono.agent import Agent
from nono.agent.skill import BaseSkill, SkillDescriptor, registry
from nono.agent.llm_agent import LlmAgent
from nono.agent.tool import FunctionTool
from nono.connector.mcp_client import MCPClient


class FileAnalysisSkill(BaseSkill):
    """Skill that uses MCP filesystem tools."""

    def __init__(self, root_path: str = "/data") -> None:
        self._client = MCPClient.stdio(
            "uvx", args=["mcp-server-filesystem", root_path],
        )

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="file_analysis",
            description="Analyze files using the filesystem MCP server.",
            tags=("files", "analysis"),
        )

    def build_agent(self, **overrides) -> LlmAgent:
        return LlmAgent(
            name="file_analyzer",
            instruction="Analyze files. Use tools to read and inspect them.",
            provider=overrides.get("provider", "google"),
        )

    def build_tools(self) -> list[FunctionTool]:
        return self._client.get_tools()


registry.register(FileAnalysisSkill())
```

### Use with Workflows

```python
from nono.agent import Agent, SequentialAgent, Runner
from nono.connector.mcp_client import MCPClient

fs_tools = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/data"]).get_tools()

reader = Agent(
    name="reader",
    instruction="Read the requested file and output its contents.",
    tools=fs_tools,
)

analyzer = Agent(
    name="analyzer",
    instruction="Analyze the data from the previous agent and produce a summary.",
)

pipeline = SequentialAgent(
    name="read_and_analyze",
    sub_agents=[reader, analyzer],
)

result = Runner(pipeline).run("Analyze the file sales_2024.csv")
```

### Convenience Function

For one-liners when you just need the tools:

```python
from nono.connector.mcp_client import mcp_tools

# Auto-detects stdio from the command parameter
tools = mcp_tools(command="uvx", args=["mcp-server-filesystem", "/tmp"])

# Auto-detects HTTP from the url parameter
tools = mcp_tools(url="http://localhost:8000/mcp")

# Explicit transport
tools = mcp_tools(url="http://localhost:8000/sse", transport="sse")
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      MCPManager                               │
│  (Central registry — load from config, add, remove, query)    │
│                         │                                     │
│              ┌──────────┼───────────┐                        │
│              ▼          ▼           ▼                         │
│          MCPClient  MCPClient   MCPClient                    │
│          "fs"       "github"    "my-api"                     │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                      MCPClient                                │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │  stdio() │    │    http()    │    │      sse()        │   │
│  └────┬─────┘    └──────┬───────┘    └─────────┬─────────┘   │
│       └─────────────────┼──────────────────────┘             │
│                         ▼                                     │
│              ClientSession.list_tools()                        │
│                         │                                     │
│                         ▼                                     │
│              _make_function_tool()  ×N                         │
│                         │                                     │
│                         ▼                                     │
│              list[FunctionTool]  → ready for Agent.tools      │
└──────────────────────────────────────────────────────────────┘
```

### Tool Invocation Flow

When an LLM calls an MCP-backed `FunctionTool`:

1. The agent framework calls `FunctionTool.invoke(args)`
2. The tool opens a fresh connection to the MCP server
3. Calls `session.call_tool(name, args)`
4. Parses the `CallToolResult` (text, structured, or error)
5. Returns the string result to the agent

---

## API Reference

### `MCPManager`

| Method                                               | Description                       |
| ---------------------------------------------------- | --------------------------------- |
| `MCPManager.from_config(config_path)`              | Load servers from `config.toml` |
| `add(name, *, transport, command, args, url, ...)` | Add or update a server            |
| `remove(name)`                                     | Remove a server                   |
| `enable(name)` / `disable(name)`                 | Toggle without removing           |
| `list_servers()`                                   | All server configs                |
| `get_tools(name)`                                  | Tools from one server             |
| `get_all_tools()`                                  | Tools from all enabled servers    |
| `save(config_path)`                                | Persist to `config.toml`        |

### `MCPClient`

| Method                                                    | Description                                        |
| --------------------------------------------------------- | -------------------------------------------------- |
| `MCPClient.stdio(command, *, args, env, timeout, name)` | Factory for stdio servers                          |
| `MCPClient.http(url, *, headers, timeout, name)`        | Factory for HTTP servers                           |
| `MCPClient.sse(url, *, headers, timeout, name)`         | Factory for SSE servers                            |
| `get_tools(*, refresh=False)`                           | Discover and return `FunctionTool` list (cached) |
| `list_tool_names()`                                     | Sorted list of tool names                          |

### `mcp_tools(**kwargs)`

Convenience function — auto-detects transport and returns tools in one call.

### CLI Commands

| Command                                     | Description               |
| ------------------------------------------- | ------------------------- |
| `nono mcp list`                           | List configured servers   |
| `nono mcp add <name> --command <cmd>`     | Add a stdio server        |
| `nono mcp add <name> -t http --url <url>` | Add an HTTP server        |
| `nono mcp remove <name>`                  | Remove a server           |
| `nono mcp enable <name>`                  | Enable a server           |
| `nono mcp disable <name>`                 | Disable a server          |
| `nono mcp tools [name]`                   | List tools from server(s) |

---

## Popular MCP Servers

| Server                                                                                  | Install                                        | Description                     |
| --------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------- |
| [Filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)     | `uvx mcp-server-filesystem`                  | Read/write local files          |
| [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/github)             | `uvx mcp-server-github`                      | GitHub API (issues, PRs, repos) |
| [PostgreSQL](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres)       | `uvx mcp-server-postgres`                    | Query PostgreSQL databases      |
| [Puppeteer](https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer)       | `npx @modelcontextprotocol/server-puppeteer` | Browser automation              |
| [Brave Search](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search) | `uvx mcp-server-brave-search`                | Web search                      |

See the full catalog: [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)

### Developer MCP Server Projects

| Server    | Repository                                                                    | Description                                                                             |
| --------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ShortFx | [github.com/jrodriguezgar/shortFx](https://github.com/jrodriguezgar/shortFx) | Excel-style formula functions for Python (dates, strings, numeric, VLOOKUP, VBA compat) |

---

## Troubleshooting

| Problem                                | Solution                                                 |
| -------------------------------------- | -------------------------------------------------------- |
| `ImportError: mcp package not found` | Install with `pip install nono[mcp]`                   |
| `ConnectionError` on stdio           | Verify the command is installed (`uvx`, `npx`, etc.) |
| `ValueError: Unsupported transport`  | Use `"stdio"`, `"http"`, or `"sse"`                |
| Tools not appearing                    | Call `get_tools(refresh=True)` to bypass cache         |
| Timeout errors                         | Increase `timeout` parameter                           |

---

*See also: [Skills](./README_skills.md) · [Connector](./connector/README_connector_genai.md) · [Agent Guide](./README_guide.md)*
