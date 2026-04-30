# ShortFx Integration

> Deterministic formulas for AI agents — 3,000+ tested functions, zero hallucinated calculations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Integration Models](#integration-models)
  - [Model 1 — Curated Tools (Direct)](#model-1--curated-tools-direct)
  - [Model 2 — Discovery Tools (Meta-tools)](#model-2--discovery-tools-meta-tools)
  - [Model 3 — MCP Server](#model-3--mcp-server)
  - [Model 4 — ShortFxSkill](#model-4--shortfxskill)
- [Decision Guide](#decision-guide)
- [Combining Models](#combining-models)
- [ShortFx Modules](#shortfx-modules)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [See Also](#see-also)

---

## Overview

[ShortFx](https://github.com/jrodriguezgar/shortFx) is the largest open-source Python formula library, with **3,000+ functions** covering dates, math, finance, statistics, strings, Excel formulas, and VBA compatibility — all with **zero dependencies**.

LLMs are excellent at reasoning but produce inconsistent results when performing calculations. ShortFx solves this by providing **deterministic, tested functions** that always return the same output for the same input.

Nono integrates ShortFx through a dedicated module (`nono/agent/tools/shortfx_tools.py`) that offers **four integration models**, from zero-code MCP to curated high-performance tools.

```
┌──────────────────────────────────────────────────────────┐
│                    Nono Agent                             │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │ Model 1     │  │ Model 2     │  │ Model 3          │ │
│  │ Curated     │  │ Discovery   │  │ MCP Server       │ │
│  │ tools=[     │  │ tools=[     │  │ tools=           │ │
│  │  fx_calc,   │  │  list_fx,   │  │  mcp_tools()    │ │
│  │  fx_fv,     │  │  inspect,   │  │                  │ │
│  │  fx_vlook]  │  │  call_fx]   │  │ ┌──────────────┐│ │
│  └──────┬──────┘  └──────┬──────┘  │ │shortfx-mcp  ││ │
│         │                │         │ │ (subprocess)  ││ │
│         │                │         │ └──────────────┘│ │
│         ▼                ▼         └────────┬─────────┘ │
│  ┌──────────────────────────────────────────┴──────────┐│
│  │           shortfx.registry / direct import          ││
│  │                  3,000+ functions                    ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Core ShortFx (enough for Models 1, 2, and 4)
pip install shortfx

# Or from GitHub (latest)
pip install git+https://github.com/jrodriguezgar/shortFx.git

# With MCP support (required for Model 3)
pip install shortfx[mcp]
```

---

## Integration Models

### Model 1 — Curated Tools (Direct)

Pre-wrapped ShortFx functions for the most common operations. **One LLM step** — the agent calls the tool directly. Maximum performance.

```python
from nono.agent import Agent, Runner
from nono.agent.tools.shortfx_tools import SHORTFX_TOOLS

agent = Agent(
    name="calculator",
    model="gemini-3-flash-preview",
    instruction="Use ShortFx tools for all calculations.",
    tools=SHORTFX_TOOLS,
)

runner = Runner(agent)
print(runner.run("What is the future value at 5% rate, 10 periods, -100 payment, -1000 PV?"))
```

Cherry-pick individual tools:

```python
from nono.agent.tools.shortfx_tools import fx_future_value, fx_calculate, fx_vlookup

agent = Agent(name="finance", tools=[fx_future_value, fx_calculate], ...)
```

#### Available Curated Tools

| Tool | Description |
|---|---|
| `fx_future_value` | Future Value (FV) of an investment |
| `fx_present_value` | Present Value (PV) of an investment |
| `fx_add_time` | Add days/months/years to a date |
| `fx_is_valid_date` | Validate a date string |
| `fx_vlookup` | Excel-style VLOOKUP on a JSON table |
| `fx_calculate` | Evaluate math expressions (AST-based, safe) |
| `fx_find_positions` | Find all positions of a substring in text |
| `fx_text_similarity` | Compute text similarity (0.0–1.0) |

---

### Model 2 — Discovery Tools (Meta-tools)

Access **all 3,000+ ShortFx functions** via four meta-tools. The LLM follows a **search/list → inspect → call** workflow.

```python
from nono.agent import Agent, Runner
from nono.agent.tools.shortfx_tools import SHORTFX_DISCOVERY_TOOLS

agent = Agent(
    name="universal_calc",
    model="gemini-3-flash-preview",
    instruction=(
        "You have access to ShortFx's 3,000+ functions via discovery tools. "
        "Workflow: search_shortfx/list_shortfx → inspect_shortfx → call_shortfx. "
        "Always use these for calculations — never compute via inference."
    ),
    tools=SHORTFX_DISCOVERY_TOOLS,
)

runner = Runner(agent)
print(runner.run("Calculate the Altman Z-score for a company"))
```

#### Discovery Meta-tools

| Tool | Purpose | LLM Step |
|---|---|---|
| `list_shortfx(module)` | Browse functions by module prefix | 1. Search |
| `search_shortfx(query)` | Semantic search by description | 1. Search |
| `inspect_shortfx(tool_name)` | Get full parameter schema | 2. Inspect |
| `call_shortfx(tool_name, args_json)` | Execute a function | 3. Execute |

Example LLM workflow:

```
1. list_shortfx("fxNumeric.finance")
   → [{"name": "fxNumeric.finance_functions.future_value", ...}, ...]

2. inspect_shortfx("fxNumeric.finance_functions.future_value")
   → {"name": ..., "parameters": {"rate": "float", "nper": "int", ...}}

3. call_shortfx("fxNumeric.finance_functions.future_value",
                '{"rate": 0.05, "nper": 10, "pmt": -100, "pv": -1000}')
   → "2886.68"
```

---

### Model 3 — MCP Server

Connect to ShortFx's built-in MCP server. **Zero code** — just configure and go. The server exposes its own meta-tools (search, inspect, call, scientific_calculate).

#### Python

```python
from nono.agent import Agent, Runner
from nono.agent.tools.shortfx_tools import shortfx_mcp_tools

agent = Agent(
    name="mcp_calc",
    instruction="Use ShortFx MCP tools for calculations.",
    tools=shortfx_mcp_tools(),
)
```

#### config.toml

```toml
[[mcp.servers]]
name = "shortfx"
transport = "stdio"
command = "shortfx-mcp"
```

Then load via `MCPManager`:

```python
from nono.connector.mcp_client import MCPManager

mgr = MCPManager.from_config()
tools = mgr.get_tools("shortfx")
agent = Agent(name="calc", tools=tools, ...)
```

#### CLI

```bash
# Add ShortFx MCP server
nono mcp add shortfx --command shortfx-mcp

# Check available tools
nono mcp tools shortfx
```

> **Requires:** `pip install shortfx[mcp]` and `pip install nono[mcp]`

---

### Model 4 — ShortFxSkill

A reusable skill that wraps the discovery tools with a pre-configured LLM agent. The agent autonomously searches, inspects, and executes ShortFx functions.

#### Standalone

```python
from nono.agent.tools.shortfx_tools import ShortFxSkill

skill = ShortFxSkill()
result = skill.run("What is the future value at 5% rate, 10 periods, -100 payment, -1000 PV?")
print(result)
```

#### Attached to an Agent

```python
from nono.agent import Agent, Runner
from nono.agent.tools.shortfx_tools import ShortFxSkill

agent = Agent(
    name="analyst",
    instruction="You analyze data. Use ShortFx for any calculations.",
    skills=[ShortFxSkill()],
)

runner = Runner(agent)
print(runner.run("Calculate compound interest on $10,000 at 3% for 5 years"))
```

When attached to an agent, the skill is auto-converted to a single `shortfx` tool. When invoked, it spawns an inner agent that handles the discovery workflow.

---

## Decision Guide

| Scenario | Model | Why |
|---|---|---|
| Few specific calculations (finance, dates) | **1 — Curated** | One LLM step, maximum speed, no discovery overhead |
| Access to all 3,000+ functions | **2 — Discovery** | Full library via meta-tools, no external process |
| Zero-code, server-based | **3 — MCP** | Just config.toml, ShortFx's native MCP server |
| Reusable component in agents/workflows | **4 — Skill** | Encapsulated, composable, discoverable |
| Top functions fast + full library on demand | **1 + 2** | Curated for common ops, discovery for the rest |
| Corporate/remote deployment | **3 — MCP** | MCP servers can run on separate hosts |

---

## Combining Models

Mix models in a single agent for the best of both worlds:

```python
from nono.agent import Agent, Runner
from nono.agent.tools.shortfx_tools import (
    SHORTFX_TOOLS,            # Model 1 — fast, curated
    SHORTFX_DISCOVERY_TOOLS,  # Model 2 — full access
)

agent = Agent(
    name="power_calc",
    instruction=(
        "Use the direct tools (fx_*) for common operations. "
        "For anything else, use list_shortfx → inspect → call."
    ),
    tools=[*SHORTFX_TOOLS, *SHORTFX_DISCOVERY_TOOLS],
)
```

Or combine with other Nono tools:

```python
from nono.agent.tools import DATETIME_TOOLS, fetch_webpage
from nono.agent.tools.shortfx_tools import fx_calculate, fx_future_value

agent = Agent(
    name="research_analyst",
    tools=[*DATETIME_TOOLS, fetch_webpage, fx_calculate, fx_future_value],
    ...
)
```

---

## ShortFx Modules

| Module | Functions | Scope |
|---|---:|---|
| `fxNumeric` | 1,602 | Finance, statistics, geometry, transforms, series, number theory |
| `fxExcel` | 489 | Excel-compatible formulas (VLOOKUP, PMT, CONCATENATE, etc.) |
| `fxString` | 331 | Text manipulation, regex, hashing, validation, encoding |
| `fxDate` | 261 | Date operations, evaluations, conversions |
| `fxVBA` | 133 | VBA/Access-compatible functions (Left, InStr, Format, etc.) |
| `fxPython` | 116 | Iterable utilities, type conversions, logic helpers |

Use `list_shortfx("fxNumeric.finance")` to browse any module.

---

## Configuration

No special configuration needed for Models 1, 2, and 4 — just `pip install shortfx`.

For Model 3 (MCP), add to `config.toml`:

```toml
[[mcp.servers]]
name = "shortfx"
transport = "stdio"
command = "shortfx-mcp"
timeout = 30.0
```

---

## API Reference

### Curated Tools (Model 1)

| Symbol | Type | Description |
|---|---|---|
| `SHORTFX_TOOLS` | `list[FunctionTool]` | All curated tools as a list |
| `fx_future_value` | `FunctionTool` | Future Value calculation |
| `fx_present_value` | `FunctionTool` | Present Value calculation |
| `fx_add_time` | `FunctionTool` | Add time to a date |
| `fx_is_valid_date` | `FunctionTool` | Validate a date string |
| `fx_vlookup` | `FunctionTool` | Excel-style VLOOKUP |
| `fx_calculate` | `FunctionTool` | Safe math expression evaluator |
| `fx_find_positions` | `FunctionTool` | Find substring positions |
| `fx_text_similarity` | `FunctionTool` | Text similarity score |

### Discovery Tools (Model 2)

| Symbol | Type | Description |
|---|---|---|
| `SHORTFX_DISCOVERY_TOOLS` | `list[FunctionTool]` | All discovery meta-tools |
| `list_shortfx` | `FunctionTool` | Browse functions by module |
| `search_shortfx` | `FunctionTool` | Semantic search by description |
| `inspect_shortfx` | `FunctionTool` | Get parameter schema |
| `call_shortfx` | `FunctionTool` | Execute any function |

### MCP (Model 3)

| Symbol | Type | Description |
|---|---|---|
| `shortfx_mcp_tools()` | `() → list[FunctionTool]` | Get tools via MCP server |

### Skill (Model 4)

| Symbol | Type | Description |
|---|---|---|
| `ShortFxSkill` | `class` | Skill wrapping discovery tools |
| `ShortFxSkill.run(msg)` | `str` | Execute standalone |
| `ShortFxSkill.as_tool()` | `FunctionTool` | Convert to agent tool |
| `ShortFxSkill.build_tools()` | `list[FunctionTool]` | Get inner tools |

---

## See Also

- [ShortFx GitHub](https://github.com/jrodriguezgar/shortFx) — source repository
- [ShortFx MCP Server](https://github.com/jrodriguezgar/shortFx/blob/main/shortfx/mcp/README.md) — native MCP documentation
- [Tools Guide](./README_tools.md) — Nono tool system reference
- [MCP Client](./README_mcp.md) — Nono MCP integration
- [Skills Documentation](./README_skills.md) — Nono skill system
