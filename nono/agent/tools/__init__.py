"""
Built-in tools for Nono agents.

Ready-to-use tool collections organised by category.  Import individual
tools or complete category lists and pass them to an ``Agent``:

    from nono.agent.tools import DATETIME_TOOLS, calculate, fetch_webpage

    agent = Agent(
        name="assistant",
        tools=[calculate, fetch_webpage, *DATETIME_TOOLS],
    )

Or use ``ALL_TOOLS`` to give the agent every built-in tool:

    from nono.agent.tools import ALL_TOOLS

    agent = Agent(name="power_agent", tools=ALL_TOOLS)

Categories:
    - **datetime_tools**: Date, time, timezone operations.
    - **text_tools**: Text statistics, extraction, transformation.
    - **web_tools**: HTTP fetch, JSON APIs, URL checking.
    - **python_tools**: Math evaluation, code execution, JSON formatting.
    - **officebridge_tools**: Document I/O, conversion, Excel, translation, censoring.
    - **shortfx_tools**: 3,000+ deterministic functions (math, finance, dates, strings, Excel, VBA).
"""

from __future__ import annotations

from .datetime_tools import (
    DATETIME_TOOLS,
    convert_timezone,
    current_datetime,
    days_between,
    list_timezones,
)
from .text_tools import (
    TEXT_TOOLS,
    extract_emails,
    extract_urls,
    find_replace,
    text_stats,
    transform_text,
    truncate_text,
)
from .web_tools import (
    WEB_TOOLS,
    check_url,
    fetch_json,
    fetch_webpage,
)
from .python_tools import (
    PYTHON_TOOLS,
    calculate,
    format_json,
    run_python,
)
from .shortfx_tools import (
    SHORTFX_TOOLS,
    SHORTFX_DISCOVERY_TOOLS,
    ShortFxSkill,
    call_shortfx,
    fx_add_time,
    fx_calculate,
    fx_find_positions,
    fx_future_value,
    fx_is_valid_date,
    fx_present_value,
    fx_vlookup,
    shortfx_mcp_tools,
    inspect_shortfx,
    list_shortfx,
    search_shortfx,
    fx_text_similarity,
)
from .officebridge_tools import (
    OFFICEBRIDGE_TOOLS,
    OFFICEBRIDGE_DISCOVERY_TOOLS,
    OfficeBridgeSkill,
    ob_convert_document,
    ob_read_document,
    ob_create_word,
    ob_create_excel,
    ob_read_excel,
    ob_translate_document,
    ob_censor_document,
    ob_create_html,
    list_officebridge,
    inspect_officebridge,
    call_officebridge,
)

ALL_TOOLS = [*DATETIME_TOOLS, *TEXT_TOOLS, *WEB_TOOLS, *PYTHON_TOOLS]
"""Every built-in tool as a flat list (excludes ShortFx/OfficeBridge — require separate install)."""

__all__ = [
    # Category lists
    "ALL_TOOLS",
    "DATETIME_TOOLS",
    "TEXT_TOOLS",
    "WEB_TOOLS",
    "PYTHON_TOOLS",
    "SHORTFX_TOOLS",
    "SHORTFX_DISCOVERY_TOOLS",
    "OFFICEBRIDGE_TOOLS",
    "OFFICEBRIDGE_DISCOVERY_TOOLS",
    # DateTime
    "current_datetime",
    "convert_timezone",
    "days_between",
    "list_timezones",
    # Text
    "text_stats",
    "extract_urls",
    "extract_emails",
    "find_replace",
    "truncate_text",
    "transform_text",
    # Web
    "fetch_webpage",
    "fetch_json",
    "check_url",
    # Python / Code
    "calculate",
    "run_python",
    "format_json",
    # ShortFx — Curated
    "fx_future_value",
    "fx_present_value",
    "fx_add_time",
    "fx_is_valid_date",
    "fx_vlookup",
    "fx_calculate",
    "fx_find_positions",
    "fx_text_similarity",
    # ShortFx — Discovery
    "list_shortfx",
    "search_shortfx",
    "inspect_shortfx",
    "call_shortfx",
    # ShortFx — MCP
    "shortfx_mcp_tools",
    # ShortFx — Skill
    "ShortFxSkill",
    # OfficeBridge — Curated
    "ob_convert_document",
    "ob_read_document",
    "ob_create_word",
    "ob_create_excel",
    "ob_read_excel",
    "ob_translate_document",
    "ob_censor_document",
    "ob_create_html",
    # OfficeBridge — Discovery
    "list_officebridge",
    "inspect_officebridge",
    "call_officebridge",
    # OfficeBridge — Skill
    "OfficeBridgeSkill",
]
