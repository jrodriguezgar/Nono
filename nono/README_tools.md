# Tools

> Define custom functions, bridge external services, and extend agent capabilities — all through a unified tool interface.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Defining Tools](#defining-tools)
  - [The @tool Decorator](#the-tool-decorator)
  - [FunctionTool (Manual Wrapping)](#functiontool-manual-wrapping)
  - [Type Hints and JSON Schema](#type-hints-and-json-schema)
- [Registering Tools on an Agent](#registering-tools-on-an-agent)
- [Who Can Use Tools](#who-can-use-tools)
  - [Agents with Tools](#agents-with-tools)
  - [Skills with Tools](#skills-with-tools)
- [How Tool Selection Works](#how-tool-selection-works)
- [ToolContext (State and Content)](#toolcontext-state-and-content)
  - [State Management](#state-management)
  - [Content Stores](#content-stores)
- [Built-in Tools](#built-in-tools)
  - [transfer_to_agent](#transfer_to_agent)
  - [Tasker Tools](#tasker-tools)
  - [MCP Tools](#mcp-tools)
  - [Skills as Tools](#skills-as-tools)
  - [Built-in Tool Collections](#built-in-tool-collections-nonoagenttools)
- [Execution Flow](#execution-flow)
- [Configuration](#configuration)
- [ACI Quality Validation](#aci-quality-validation)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [See Also](#see-also)

---

## Overview

A **tool** is a Python function with metadata (name, description, parameter schema) that an LLM agent can invoke via function calling. When the LLM decides a tool is needed, the agent framework:

1. Extracts the tool call from the LLM response.
2. Executes the corresponding Python function locally.
3. Sends the result back to the LLM for the next reasoning step.

This loop repeats until the LLM produces a final text response.

| Concept | Description |
|---|---|
| **FunctionTool** | Core wrapper: function + name + description + JSON Schema |
| **@tool** | Decorator that creates a `FunctionTool` automatically |
| **ToolContext** | Injected context giving tools access to session state and content stores |
| **Tool loop** | Agent iterates LLM ↔ tool execution until a text answer is produced |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Agent.run()                       │
│                                                       │
│   ┌──────────┐       ┌──────────┐       ┌──────────┐ │
│   │   LLM    │──────→│ Extract  │──────→│ Execute  │ │
│   │   Call    │       │ Tool Call│       │  Tool    │ │
│   └──────────┘       └──────────┘       └──────────┘ │
│        ↑                                     │        │
│        └─────────── Result ──────────────────┘        │
│                                                       │
│   Repeats up to max_tool_iterations (default: 10)     │
└──────────────────────────────────────────────────────┘
```

Tool definitions are sent to the LLM as JSON Schema via the connector. The connector handles provider-specific formatting (Google `function_declarations`, OpenAI `tools`, etc.) transparently.

---

## Defining Tools

### The @tool Decorator

The simplest way to create a tool. Type hints are used to auto-generate the JSON Schema:

```python
from nono.agent import tool

@tool(description="Get the current weather for a city.")
def get_weather(city: str, unit: str = "celsius") -> str:
    return f"22°{unit[0].upper()} in {city}"
```

Without arguments (uses the first docstring line as description):

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"22°C in {city}"
```

With a custom name:

```python
@tool(name="weather_lookup", description="Look up weather by city name.")
def get_weather(city: str) -> str:
    return f"22°C in {city}"
```

### FunctionTool (Manual Wrapping)

For existing functions that you don't want to decorate:

```python
from nono.agent import FunctionTool

def search_database(query: str) -> list[dict]:
    """Search the internal database."""
    return [{"id": 1, "title": "Result"}]

search_tool = FunctionTool(search_database, description="Search the internal database.")
```

### Type Hints and JSON Schema

The parameter schema is auto-generated from type annotations:

| Python Type | JSON Schema Type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |

Parameters **without defaults** are marked as `required`. Parameters with defaults are optional.

```python
@tool(description="Search with filters.")
def search(query: str, max_results: int = 10, include_metadata: bool = False) -> str:
    ...
# Schema: query (required, string), max_results (optional, integer),
#          include_metadata (optional, boolean)
```

---

## Registering Tools on an Agent

Pass tools to the `tools=` parameter when creating an `Agent`:

```python
from nono.agent import Agent, Runner, tool

@tool(description="Add two numbers together.")
def add(a: int, b: int) -> str:
    return str(a + b)

@tool(description="Multiply two numbers together.")
def multiply(a: int, b: int) -> str:
    return str(a * b)

agent = Agent(
    name="calculator",
    model="gemini-3-flash-preview",
    provider="google",
    instruction="You are a calculator assistant. Use tools for all math operations.",
    tools=[add, multiply],
)

runner = Runner(agent=agent)
response = runner.run("What is 7 + 3, and then multiply the result by 5?")
```

---

## Who Can Use Tools

Both **agents** and **skills** can use tools, but through different mechanisms.

### Agents with Tools

An `Agent` (LlmAgent) receives tools via three sources, all merged into a single list at runtime:

| Source | How it's added | Example |
|---|---|---|
| **`tools=`** parameter | Explicitly passed at construction | `Agent(tools=[add, multiply])` |
| **`skills=`** parameter | Each skill is auto-converted via `skill.as_tool()` | `Agent(skills=[SummarizeSkill()])` |
| **`sub_agents=`** parameter | Auto-generates a `transfer_to_agent` tool | `Agent(sub_agents=[researcher])` |

```python
from nono.agent import Agent
from nono.agent.tools import calculate, fetch_webpage
from nono.agent.skills import SummarizeSkill

agent = Agent(
    name="analyst",
    instruction="You analyse data. Use tools and skills as needed.",
    tools=[calculate, fetch_webpage],        # ← explicit tools
    skills=[SummarizeSkill()],               # ← auto-converted to tool
    sub_agents=[researcher],                 # ← auto-generates transfer_to_agent
)
# The LLM sees 4 tools: calculate, fetch_webpage, summarize, transfer_to_agent
```

### Skills with Tools

A skill's inner agent can also have its own tools. Override `build_tools()` to provide domain-specific tools that are **automatically injected** when the skill runs.

Skills can use tools from **any source**: the built-in catalog (`nono.agent.tools`), custom `FunctionTool` instances, MCP remote tools, or any combination.

#### Catalog tools in a skill

```python
from nono.agent.skill import BaseSkill, SkillDescriptor
from nono.agent import LlmAgent, FunctionTool
from nono.agent.tools import calculate, fetch_webpage, text_stats  # ← from catalog

class DataAnalysisSkill(BaseSkill):

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(name="data-analysis", description="Analyse datasets.")

    def build_agent(self, **overrides):
        return LlmAgent(name="analyst", instruction="Use tools to analyse data.")

    def build_tools(self):
        return [calculate, fetch_webpage, text_stats]  # ← all from catalog
```

#### Catalog + custom tools in a skill

```python
from nono.agent.tools import calculate

class MixedSkill(BaseSkill):
    # ...
    def build_tools(self):
        return [
            calculate,                         # ← from catalog
            FunctionTool(                      # ← custom
                fn=self._query_db,
                name="query_db",
                description="Query the internal database.",
            ),
        ]

    @staticmethod
    def _query_db(sql: str) -> str:
        return f"[Results for: {sql}]"
```

#### MCP tools in a skill

```python
from nono.connector.mcp_client import MCPClient

class FileSkill(BaseSkill):
    def __init__(self, root: str = "/data") -> None:
        self._client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", root])

    # ...
    def build_tools(self):
        return self._client.get_tools()  # ← remote MCP tools
```

#### Full category list in a skill

```python
from nono.agent.tools import DATETIME_TOOLS, WEB_TOOLS

class ResearchSkill(BaseSkill):
    # ...
    def build_tools(self):
        return [*DATETIME_TOOLS, *WEB_TOOLS]  # ← entire categories
```

When `skill.run()` or `skill.as_tool()` is called, `build_tools()` is invoked and the returned tools are injected into the inner agent before execution. Duplicates (by name) are automatically skipped.

---

## How Tool Selection Works

The LLM (not the framework) decides which tools to call. The process:

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Assembly                             │
│                                                              │
│  Agent.tools        ──┐                                      │
│  + skill.as_tool()  ──┼──→  _all_tools  ──→  JSON Schema    │
│  + transfer_to_agent──┘     (merged)          sent to LLM    │
│                                                              │
│                    Tool Selection                            │
│                                                              │
│  LLM receives all tool schemas in the prompt                 │
│  LLM decides which tool(s) to call based on:                │
│    • Tool name and description                               │
│    • Parameter schema                                        │
│    • Current conversation context                            │
│    • User's request                                          │
│                                                              │
│  The framework does NOT filter or pre-select tools.          │
│  All registered tools are always available to the LLM.       │
└─────────────────────────────────────────────────────────────┘
```

**Key points:**

| Aspect | Behaviour |
|---|---|
| **Assembly** | `_all_tools` merges `tools` + `skills` + `transfer_to_agent` (cached after first call) |
| **Deduplication** | Skills inject tools by name — duplicates are skipped |
| **Selection** | The LLM chooses tools based on descriptions — write good descriptions |
| **No filtering** | All tools are sent to the LLM every turn; use fewer, focused tools for best results |
| **Validation** | ACI quality checks warn about poor names/descriptions at construction time |

**Practical guidance — how many tools?**

| Scenario | Recommendation |
|---|---|
| **< 10 tools** | Pass all tools directly — the LLM handles this well |
| **10–20 tools** | Group by category; consider using separate sub-agents with focused tool sets |
| **20+ tools** | Use `sub_agents` to create specialist agents, each with a focused tool set. The coordinator delegates via `transfer_to_agent` |

```python
# Too many tools for one agent? Split into specialists:
math_agent = Agent(name="math", tools=[calculate, ...], ...)
web_agent = Agent(name="web", tools=[fetch_webpage, fetch_json, ...], ...)

coordinator = Agent(
    name="coordinator",
    instruction="Delegate math to math agent, web tasks to web agent.",
    sub_agents=[math_agent, web_agent],   # LLM routes via transfer_to_agent
)
```

---

## ToolContext (State and Content)

When a tool function declares a parameter typed as `ToolContext`, the framework **excludes** it from the JSON Schema and **injects** a populated instance at invocation time.

```python
from nono.agent import tool, ToolContext

@tool(description="Save a research finding to memory.")
def save_finding(text: str, tool_context: ToolContext) -> str:
    findings = tool_context.state.setdefault("findings", [])
    findings.append(text)
    return f"Saved finding #{len(findings)}"
```

### State Management

| Method | Description |
|---|---|
| `tool_context.state_set(key, value)` | Thread-safe write to session state |
| `tool_context.state_get(key, default)` | Thread-safe read from session state |
| `tool_context.state_update(mapping)` | Merge multiple keys into state |
| `tool_context.state[key]` | Direct access (not thread-safe) |

State persists across all tool calls and turns within a session.

### Content Stores

Two content scopes are available:

| Scope | Access | Visibility |
|---|---|---|
| `shared_content` | `tool_context.shared_content` | All agents in the session |
| `local_content` | `tool_context.local_content` | Only the invoking agent |

```python
@tool(description="Save a report visible to all agents.")
def save_report(text: str, tool_context: ToolContext) -> str:
    tool_context.shared_content.save("report", text)       # visible to all
    tool_context.local_content.save("draft", text)          # private
    tool_context.state_set("has_report", True)
    return "Report saved"
```

Convenience method:

```python
tool_context.save_content("report", data, scope="shared")   # or "local"
item = tool_context.load_content("report", scope="shared")
```

---

## Built-in Tools

### transfer_to_agent

Automatically registered when an agent has `sub_agents`. The LLM can call it to delegate tasks to a specific sub-agent:

```python
researcher = Agent(name="researcher", instruction="You research topics.", ...)
writer = Agent(name="writer", instruction="You write articles.", ...)

coordinator = Agent(
    name="coordinator",
    instruction="Delegate research to researcher and writing to writer.",
    sub_agents=[researcher, writer],   # ← creates transfer_to_agent tool
)
```

The coordinator's LLM sees a `transfer_to_agent(agent_name, message)` tool and decides when to delegate.

### Tasker Tools

Bridge `TaskExecutor` tasks into agent tools. Two factory functions:

**Inline configuration:**

```python
from nono.agent.tasker_tool import tasker_tool

summarise = tasker_tool(
    name="summarise",
    description="Summarise a document using a dedicated model.",
    provider="google",
    model="gemini-3-flash-preview",
    system_prompt="You are a professional summariser.",
)

agent = Agent(name="analyst", tools=[summarise], ...)
```

**From a JSON task file:**

```python
from nono.agent.tasker_tool import json_task_tool

classify = json_task_tool("nono/tasker/prompts/name_classifier.json")
agent = Agent(name="analyst", tools=[classify], ...)
```

The JSON file defines the task configuration (provider, model, system prompt, output schema). The tool accepts a `prompt` or `data` string from the LLM.

### MCP Tools

Connect to external [Model Context Protocol](https://modelcontextprotocol.io/) servers and use their tools as native Nono tools:

```python
from nono.connector.mcp_client import MCPClient, mcp_tools

# Stdio server
client = MCPClient.stdio("uvx", args=["mcp-server-filesystem", "/tmp"])
tools = client.get_tools()

# HTTP server
client = MCPClient.http("http://localhost:8000/mcp")
tools = client.get_tools()

# One-liner convenience
tools = mcp_tools(command="uvx", args=["mcp-server-filesystem", "/tmp"])

# Attach to agent
agent = Agent(name="assistant", instruction="Help with files.", tools=tools)
```

Requires the `mcp` optional dependency: `pip install nono[mcp]`

See [MCP Client Documentation](./README_mcp.md) for full details.

### Skills as Tools

Skills (reusable AI capabilities) are automatically converted to tools when attached to an agent:

```python
from nono.agent import Agent
from nono.agent.skills import SummarizeSkill, ClassifySkill

agent = Agent(
    name="analyst",
    skills=[SummarizeSkill(), ClassifySkill()],
    ...
)
# SummarizeSkill and ClassifySkill become callable tools
```

See [Skills Documentation](./README_skills.md) for creating and managing skills.

### Built-in Tool Collections (`nono.agent.tools`)

Nono ships with ready-to-use tools organised by category in `nono/agent/tools/`. Import individual tools or full category lists:

```python
from nono.agent.tools import ALL_TOOLS, DATETIME_TOOLS, calculate, fetch_webpage

# Give the agent every built-in tool
agent = Agent(name="power_agent", tools=ALL_TOOLS)

# Or pick specific categories
agent = Agent(name="time_agent", tools=DATETIME_TOOLS)

# Or cherry-pick individual tools
agent = Agent(name="math_agent", tools=[calculate, fetch_webpage])
```

#### DateTime Tools (`datetime_tools.py`)

| Tool | Description |
|---|---|
| `current_datetime` | Current date/time in a given IANA timezone |
| `convert_timezone` | Convert a datetime string between timezones |
| `days_between` | Days between two dates |
| `list_timezones` | List IANA timezone names by region |

#### Text Tools (`text_tools.py`)

| Tool | Description |
|---|---|
| `text_stats` | Word, character, sentence, and line counts |
| `extract_urls` | Extract all unique URLs from text |
| `extract_emails` | Extract all email addresses from text |
| `find_replace` | Find and replace (plain text or regex) |
| `truncate_text` | Truncate to a word limit |
| `transform_text` | Convert to uppercase, lowercase, title case, or slug |

#### Web Tools (`web_tools.py`)

| Tool | Description |
|---|---|
| `fetch_webpage` | Fetch a web page and return text content |
| `fetch_json` | GET a JSON API and return formatted response |
| `check_url` | HEAD request to check URL availability |

#### Python Tools (`python_tools.py`)

| Tool | Description |
|---|---|
| `calculate` | Evaluate math expressions safely (supports `math` functions) |
| `run_python` | Execute Python code in a sandboxed environment |
| `format_json` | Parse and pretty-print a JSON string |

---

## Execution Flow

The complete tool-calling cycle:

```
1. Agent.run(user_message)
       │
2. Build tool definitions (JSON Schema) from all registered tools
       │
3. Send message + tool definitions to LLM via connector
       │
4. LLM response contains tool_call?
       ├── YES → 5. Fire PreToolUse hooks (optional, can modify/block)
       │         6. Create ToolContext with session state + content stores
       │         7. Call tool.invoke(arguments, tool_context=ctx)
       │         8. Append result to message history
       │         9. Go to step 3 (loop)
       │
       └── NO  → 10. Return final text response
```

The loop is bounded by `max_tool_iterations` (default: 10) and `max_loop_messages` (default: 40).

---

## Configuration

Tool-related settings in `config.toml` under `[agent]`:

```toml
[agent]
# Maximum tool-call loop iterations to prevent infinite loops
max_tool_iterations = 10

# Maximum messages in tool-calling loop to avoid context overflow
max_loop_messages = 40

# Maximum depth for recursive agent transfers (prevents stack overflow)
max_transfer_depth = 10
```

---

## ACI Quality Validation

Following Anthropic's guidance that *"tool definitions deserve as much prompt engineering as your prompts"*, Nono validates tool descriptions automatically when creating an Agent:

```python
from nono.agent import Agent, FunctionTool

# ⚠️ This triggers warnings at construction time:
bad = FunctionTool(lambda q: q, name="s", description="search")
agent = Agent(name="a", provider="google", tools=[bad])
# WARNING: Tool 's': Description is only 6 chars (minimum recommended: 10).
# WARNING: Tool 's': Tool name is too short.
```

Programmatic validation for CI/CD:

```python
from nono.agent import validate_tools

issues = validate_tools(agent.tools, warn=False)
assert len(issues) == 0, f"Fix tool descriptions: {issues}"
```

---

## Best Practices

| Practice | Why |
|---|---|
| Return `str` from tools | LLMs process text — return structured text, not raw objects |
| Write clear descriptions | The description is part of the prompt — vague descriptions cause wrong tool selection |
| Use type hints on all parameters | Enables accurate JSON Schema generation |
| Keep tools focused | One tool = one capability. Prefer multiple small tools over one large one |
| Use `ToolContext` for state | Avoids global variables; state is scoped to the session |
| Use `shared_content` for cross-agent data | Better than passing everything through the LLM |
| Validate in CI | Use `validate_tools()` to catch poor descriptions before deployment |

---

## API Reference

### Classes

| Class | Module | Description |
|---|---|---|
| `FunctionTool` | `nono.agent.tool` | Core tool wrapper with invoke, schema generation, and validation |
| `ToolContext` | `nono.agent.tool` | Injected context with state, content stores, and session info |
| `ToolIssue` | `nono.agent.tool` | Quality issue found during ACI validation |

### Functions

| Function | Module | Description |
|---|---|---|
| `tool()` | `nono.agent.tool` | Decorator to create a `FunctionTool` from a function |
| `validate_tools()` | `nono.agent.tool` | Validate tool descriptions and schemas |
| `parse_tool_calls()` | `nono.agent.tool` | Parse tool call instructions from LLM response text |
| `tasker_tool()` | `nono.agent.tasker_tool` | Create a tool from TaskExecutor configuration |
| `json_task_tool()` | `nono.agent.tasker_tool` | Create a tool from a JSON task definition file |
| `mcp_tools()` | `nono.connector.mcp_client` | Discover and return MCP server tools in one call |

### FunctionTool Methods

| Method | Description |
|---|---|
| `invoke(args, tool_context=)` | Execute the tool with a dict of arguments |
| `to_function_declaration()` | Convert to OpenAI-compatible function declaration dict |

---

## See Also

- [Step-by-Step Guide — Adding Tools](./README_guide.md#step-22--adding-tools) — tutorial walkthrough
- [Skills Documentation](./README_skills.md) — reusable AI capabilities as tools
- [MCP Client Documentation](./README_mcp.md) — external tool servers
- [Agent Documentation](./agent/README_agent.md) — agent framework reference
- [Tasker vs Agent](./README_tasker_vs_agent.md) — when to use tools vs direct execution
