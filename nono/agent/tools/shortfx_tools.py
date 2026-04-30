"""
ShortFx integration — 3000+ deterministic functions as agent tools.

Bridges `ShortFx <https://github.com/jrodriguezgar/shortFx>`_ into Nono's
tool system, providing four integration models:

1. **Direct tools** — Curated ShortFx functions wrapped as ``FunctionTool``.
2. **Discovery tools** — Search, inspect, and execute any of ShortFx's 3,000+
   functions via meta-tools (registry-based).
3. **MCP tools** — Connect to ShortFx's built-in MCP server.
4. **ShortFxSkill** — A reusable skill combining discovery + execution.

Requires ``shortfx`` (``pip install shortfx``).  MCP model additionally
requires ``pip install shortfx[mcp]``.

Usage::

    # Model 1 — Curated tools (fastest for common operations)
    from nono.agent.tools.shortfx_tools import SHORTFX_TOOLS
    agent = Agent(name="calc", tools=SHORTFX_TOOLS, ...)

    # Model 2 — Discovery (access to all 3,000+ functions)
    from nono.agent.tools.shortfx_tools import SHORTFX_DISCOVERY_TOOLS
    agent = Agent(name="calc", tools=SHORTFX_DISCOVERY_TOOLS, ...)

    # Model 3 — MCP (zero-code, via config.toml)
    from nono.agent.tools.shortfx_tools import shortfx_mcp_tools
    agent = Agent(name="calc", tools=shortfx_mcp_tools(), ...)

    # Model 4 — Skill (composable, reusable)
    from nono.agent.tools.shortfx_tools import ShortFxSkill
    agent = Agent(name="calc", skills=[ShortFxSkill()], ...)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..tool import FunctionTool, ToolContext, tool

logger = logging.getLogger("Nono.Agent.Tools.ShortFx")

_SHORTFX_AVAILABLE: bool | None = None


def _require_shortfx() -> None:
    """Raise ``ImportError`` if ShortFx is not installed."""
    global _SHORTFX_AVAILABLE

    if _SHORTFX_AVAILABLE is None:
        try:
            import shortfx  # noqa: F401

            _SHORTFX_AVAILABLE = True
        except ImportError:
            _SHORTFX_AVAILABLE = False

    if not _SHORTFX_AVAILABLE:
        raise ImportError(
            "ShortFx is required for this tool. "
            "Install it with: pip install shortfx  "
            "(or: pip install git+https://github.com/jrodriguezgar/shortFx.git)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Curated Direct Tools
# ═══════════════════════════════════════════════════════════════════════════════
# Pre-wrapped ShortFx functions for the most common operations.
# One LLM step: the agent calls the tool directly.  Maximum performance.
# ═══════════════════════════════════════════════════════════════════════════════


@tool(description=(
    "Calculate the Future Value (FV) of an investment. "
    "Args: rate (annual interest as decimal, e.g. 0.05), nper (number of periods), "
    "pmt (payment per period, negative = outflow), pv (present value, negative = outflow)."
))
def fx_future_value(rate: float, nper: int, pmt: float, pv: float) -> str:
    """Compute future value using ShortFx's finance module.

    Args:
        rate: Interest rate per period (e.g. 0.05 for 5%).
        nper: Total number of payment periods.
        pmt: Payment per period (negative for outflows).
        pv: Present value (negative for outflows).

    Returns:
        Future value as a string.
    """
    _require_shortfx()
    from shortfx.fxNumeric import finance_functions

    return str(finance_functions.future_value(rate=rate, nper=nper, pmt=pmt, pv=pv))


@tool(description=(
    "Calculate the Present Value (PV) of an investment. "
    "Args: rate (decimal), nper (periods), pmt (payment/period), fv (future value)."
))
def fx_present_value(rate: float, nper: int, pmt: float, fv: float) -> str:
    """Compute present value using ShortFx's finance module.

    Args:
        rate: Interest rate per period.
        nper: Total number of payment periods.
        pmt: Payment per period.
        fv: Future value.

    Returns:
        Present value as a string.
    """
    _require_shortfx()
    from shortfx.fxNumeric import finance_functions

    return str(finance_functions.present_value(rate=rate, nper=nper, pmt=pmt, fv=fv))


@tool(description=(
    "Add a time delta to a date. "
    "Args: start_date (YYYY-MM-DD), amount (integer), unit ('days', 'months', 'years')."
))
def fx_add_time(start_date: str, amount: int, unit: str = "days") -> str:
    """Add time to a date using ShortFx.

    Args:
        start_date: Date in ``YYYY-MM-DD`` format.
        amount: Number of units to add.
        unit: ``"days"``, ``"months"``, or ``"years"``.

    Returns:
        Resulting date as a string.
    """
    _require_shortfx()
    from datetime import datetime

    from shortfx.fxDate import date_operations

    dt = datetime.strptime(start_date, "%Y-%m-%d")
    result = date_operations.add_time_to_date(dt, amount, unit)
    return str(result)


@tool(description=(
    "Validate whether a date string is a real calendar date. "
    "Args: date_string (e.g. '2025-02-30')."
))
def fx_is_valid_date(date_string: str) -> str:
    """Check if a date string represents a valid date.

    Args:
        date_string: Date to validate.

    Returns:
        ``"True"`` or ``"False"``.
    """
    _require_shortfx()
    from shortfx.fxDate import date_operations

    return str(date_operations.is_valid_date(date_string))


@tool(description=(
    "Excel-style VLOOKUP: search for a value in the first column of a table "
    "and return the value from a specified column. "
    "Args: lookup_value (string to find), table_json (JSON 2D array), col_index (1-based column number)."
))
def fx_vlookup(lookup_value: str, table_json: str, col_index: int) -> str:
    """Excel-compatible VLOOKUP via ShortFx.

    Args:
        lookup_value: Value to search in the first column.
        table_json: Table as a JSON 2D array string.
        col_index: 1-based column index to return.

    Returns:
        The matching value as a string.
    """
    _require_shortfx()
    from shortfx import fxExcel

    table = json.loads(table_json)
    return str(fxExcel.VLOOKUP(lookup_value, table, col_index))


@tool(description=(
    "Evaluate a mathematical expression safely using ShortFx's AST-based parser. "
    "Supports arithmetic, trig, log, sqrt, etc. No eval() — fully sandboxed. "
    "Example: '(3 + 4) * sqrt(16) / 2'."
))
def fx_calculate(expression: str) -> str:
    """Evaluate a math expression via ShortFx's safe parser.

    Args:
        expression: Mathematical expression string.

    Returns:
        Computed result as a string.
    """
    _require_shortfx()
    from shortfx.fxNumeric import math_functions

    return str(math_functions.evaluate_expression(expression))


@tool(description=(
    "Find all positions of a substring in a text. "
    "Returns a JSON array of 0-based positions, or '[]' if not found."
))
def fx_find_positions(text: str, substring: str) -> str:
    """Find all occurrences of a substring.

    Args:
        text: Source text to search.
        substring: Pattern to find.

    Returns:
        JSON array of positions.
    """
    _require_shortfx()
    from shortfx.fxString import string_operations

    positions = string_operations.position_in_string(text, substring)
    return json.dumps(positions)


@tool(description=(
    "Calculate the similarity between two text strings. "
    "Returns a score between 0.0 (no match) and 1.0 (identical). "
    "Args: text1, text2."
))
def fx_text_similarity(text1: str, text2: str) -> str:
    """Compute text similarity using ShortFx.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Similarity score as a string (0.0–1.0).
    """
    _require_shortfx()
    from shortfx.fxString import string_similarity

    return str(string_similarity.similarity_ratio(text1, text2))


SHORTFX_TOOLS: list[FunctionTool] = [
    fx_future_value,
    fx_present_value,
    fx_add_time,
    fx_is_valid_date,
    fx_vlookup,
    fx_calculate,
    fx_find_positions,
    fx_text_similarity,
]
"""Curated ShortFx tools — the most common operations, maximum performance."""


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Discovery Tools (meta-tools)
# ═══════════════════════════════════════════════════════════════════════════════
# Expose ShortFx's full registry (3,000+ functions) via search/inspect/call
# meta-tools.  The LLM uses a search → inspect → execute workflow.
# ═══════════════════════════════════════════════════════════════════════════════


@tool(description=(
    "List available ShortFx functions, optionally filtered by module prefix. "
    "Valid prefixes: fxDate, fxNumeric, fxString, fxPython, fxExcel, fxVBA. "
    "Returns name and description for each function. "
    "Example: list_shortfx('fxNumeric.finance')."
))
def list_shortfx(module: str = "") -> str:
    """Browse ShortFx functions by module.

    Args:
        module: Module prefix filter (e.g. ``"fxExcel.math"``).

    Returns:
        JSON array of ``{name, description}`` dicts.
    """
    _require_shortfx()
    from shortfx.registry import get_tool_schemas

    schemas = get_tool_schemas(module_filter=module or None)
    summaries = [
        {"name": s["name"], "description": s.get("description", "")}
        for s in schemas
    ]

    # Cap to avoid overwhelming LLM context
    if len(summaries) > 80:
        total = len(summaries)
        summaries = summaries[:80]
        summaries.append({"name": f"... and {total - 80} more", "description": "Use a more specific filter."})

    return json.dumps(summaries, indent=2, ensure_ascii=False)


@tool(description=(
    "Search ShortFx functions using natural language. Uses semantic search "
    "to find the most relevant functions from 3,000+. "
    "Example: search_shortfx('calculate compound interest')."
))
def search_shortfx(query: str, top_k: int = 10) -> str:
    """Semantic search across all ShortFx functions.

    Args:
        query: Natural language description of what you need.
        top_k: Maximum number of results to return.

    Returns:
        JSON array of matching functions with scores.
    """
    _require_shortfx()
    from shortfx.registry import search_tools

    results = search_tools(query, top_k=top_k)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool(description=(
    "Get full details (parameters, types, description) for a specific ShortFx function. "
    "Use the qualified name from list_shortfx or search_shortfx. "
    "Example: inspect_shortfx('fxNumeric.finance_functions.future_value')."
))
def inspect_shortfx(tool_name: str) -> str:
    """Get the parameter schema for a ShortFx function.

    Args:
        tool_name: Qualified function name (e.g.
            ``"fxDate.date_operations.add_time_to_date"``).

    Returns:
        JSON object with name, description, and parameters.
    """
    _require_shortfx()
    from shortfx.registry import get_tool_schemas

    schemas = get_tool_schemas(module_filter=None)
    for s in schemas:
        if s["name"] == tool_name:
            return json.dumps(s, indent=2, ensure_ascii=False)

    return json.dumps({"error": f"Function '{tool_name}' not found."})


@tool(description=(
    "Execute any ShortFx function by its qualified name. "
    "Pass arguments as a JSON object string. "
    "Example: call_shortfx('fxNumeric.finance_functions.future_value', "
    "'{\"rate\": 0.05, \"nper\": 10, \"pmt\": -100, \"pv\": -1000}')."
))
def call_shortfx(tool_name: str, arguments_json: str = "{}") -> str:
    """Invoke a ShortFx function dynamically.

    Args:
        tool_name: Qualified function name.
        arguments_json: JSON string with keyword arguments.

    Returns:
        Function result as a string.
    """
    _require_shortfx()
    from shortfx.registry import invoke_tool

    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON arguments: {exc}"

    try:
        result = invoke_tool(tool_name, args)
    except Exception as exc:
        logger.warning("call_shortfx(%s) failed: %s", tool_name, exc)
        return f"Error: {exc}"

    return str(result)


SHORTFX_DISCOVERY_TOOLS: list[FunctionTool] = [
    list_shortfx,
    search_shortfx,
    inspect_shortfx,
    call_shortfx,
]
"""Discovery meta-tools — search, inspect, and execute any of ShortFx's 3,000+ functions."""


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — MCP Tools
# ═══════════════════════════════════════════════════════════════════════════════
# Connect to ShortFx's built-in MCP server as a subprocess.
# Requires: pip install shortfx[mcp]
# ═══════════════════════════════════════════════════════════════════════════════


def shortfx_mcp_tools(
    *,
    timeout: float = 30.0,
) -> list[FunctionTool]:
    """Get ShortFx tools via its MCP server.

    Launches the ``shortfx-mcp`` server as a subprocess and converts
    its meta-tools into Nono ``FunctionTool`` instances.

    Requires ``pip install shortfx[mcp]``.

    Args:
        timeout: Connection timeout in seconds.

    Returns:
        List of ``FunctionTool`` from the ShortFx MCP server.

    Raises:
        ImportError: If Nono's MCP client or ShortFx MCP are not installed.

    Example::

        from nono.agent.tools.shortfx_tools import shortfx_mcp_tools
        agent = Agent(name="calc", tools=shortfx_mcp_tools(), ...)
    """
    from ...connector.mcp_client import MCPClient

    client = MCPClient.stdio(
        "shortfx-mcp",
        timeout=timeout,
        name="shortfx",
    )
    return client.get_tools()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — ShortFxSkill
# ═══════════════════════════════════════════════════════════════════════════════
# A reusable skill that combines discovery tools with an LLM agent.
# The agent searches, inspects, and executes functions autonomously.
# ═══════════════════════════════════════════════════════════════════════════════


class ShortFxSkill:
    """Skill that provides access to ShortFx's full function library.

    The inner agent uses discovery tools (list, search, inspect, call) to find
    and execute the right ShortFx function autonomously.

    Can be used standalone or attached to an agent::

        # Standalone
        skill = ShortFxSkill()
        result = skill.run("What is the future value at 5% rate, 10 periods?")

        # Attached to an agent
        agent = Agent(name="analyst", skills=[ShortFxSkill()], ...)

    Args:
        provider: AI provider (default ``"google"``).
        model: Model override (``None`` for provider default).
    """

    def __init__(
        self,
        *,
        provider: str = "google",
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model

    @property
    def descriptor(self) -> Any:
        """Return the skill's metadata descriptor."""
        from ..skill import SkillDescriptor

        return SkillDescriptor(
            name="shortfx",
            description=(
                "Execute deterministic calculations using ShortFx's 3,000+ functions "
                "covering dates, math, finance, statistics, strings, Excel, and VBA."
            ),
            tags=("math", "finance", "dates", "strings", "excel", "vba", "formulas"),
            input_keys=("input",),
            output_keys=("result",),
        )

    def build_agent(self, **overrides: Any) -> Any:
        """Create the LLM agent that drives ShortFx discovery.

        Args:
            **overrides: Optional provider/model overrides.

        Returns:
            Configured ``LlmAgent``.
        """
        from ..llm_agent import LlmAgent

        return LlmAgent(
            name="shortfx_agent",
            provider=overrides.get("provider", self._provider),
            model=overrides.get("model", self._model),
            instruction=(
                "You are a deterministic calculation assistant powered by ShortFx. "
                "ShortFx has 3,000+ functions across 6 modules: "
                "fxDate (dates), fxNumeric (math/finance/stats), fxString (text), "
                "fxPython (utilities), fxExcel (Excel-compatible), fxVBA (VBA-compatible).\n\n"
                "Workflow:\n"
                "1. Use search_shortfx to find functions by description.\n"
                "2. Use list_shortfx to browse functions by module.\n"
                "3. Use inspect_shortfx to get the parameter schema.\n"
                "4. Use call_shortfx to execute the function.\n\n"
                "Always use ShortFx functions for calculations — never compute "
                "results via inference. ShortFx guarantees deterministic answers."
            ),
            description=self.descriptor.description,
            temperature=0.0,
        )

    def build_tools(self) -> list[FunctionTool]:
        """Return discovery tools for the inner agent.

        Returns:
            The four discovery meta-tools (including semantic search).
        """
        return list(SHORTFX_DISCOVERY_TOOLS)

    def as_tool(self) -> FunctionTool:
        """Convert this skill to a ``FunctionTool`` for LLM function-calling.

        Returns:
            A ``FunctionTool`` wrapping the full discovery workflow.
        """
        desc = self.descriptor

        def _invoke(input: str) -> str:  # noqa: A002
            return self.run(input)

        return FunctionTool(
            fn=_invoke,
            name=desc.name,
            description=desc.description,
        )

    def run(
        self,
        user_message: str,
        **overrides: Any,
    ) -> str:
        """Execute the skill standalone.

        Args:
            user_message: Calculation request in natural language.
            **overrides: Forwarded to :meth:`build_agent`.

        Returns:
            The agent's final text response.
        """
        from ..runner import Runner

        agent = self.build_agent(**overrides)

        # Inject discovery tools into the agent
        existing = {t.name for t in agent.tools}
        for t in self.build_tools():
            if t.name not in existing:
                agent.tools.append(t)

        return Runner(agent).run(user_message)

    def __repr__(self) -> str:
        return "ShortFxSkill()"
