"""DateTime tools — date, time, and timezone utilities for agents."""

from __future__ import annotations

import datetime
import zoneinfo
from typing import Optional

from ..tool import FunctionTool, ToolContext, tool


@tool(description="Get the current date and time, optionally in a specific timezone (e.g. 'Europe/Madrid', 'US/Eastern', 'UTC').")
def current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time in the given timezone.

    Args:
        timezone: IANA timezone name (default ``"UTC"``).

    Returns:
        Formatted datetime string with timezone info.
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except (KeyError, zoneinfo.ZoneInfoNotFoundError):
        return f"Unknown timezone: {timezone}. Use IANA names like 'Europe/Madrid', 'US/Eastern'."

    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


@tool(description="Convert a datetime string from one timezone to another. Input format: 'YYYY-MM-DD HH:MM:SS'.")
def convert_timezone(
    datetime_str: str,
    from_timezone: str,
    to_timezone: str,
) -> str:
    """Convert a datetime between timezones.

    Args:
        datetime_str: Datetime in ``YYYY-MM-DD HH:MM:SS`` format.
        from_timezone: Source IANA timezone.
        to_timezone: Target IANA timezone.

    Returns:
        Converted datetime string.
    """
    try:
        from_tz = zoneinfo.ZoneInfo(from_timezone)
        to_tz = zoneinfo.ZoneInfo(to_timezone)
    except (KeyError, zoneinfo.ZoneInfoNotFoundError) as exc:
        return f"Invalid timezone: {exc}"

    try:
        dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Invalid format. Use 'YYYY-MM-DD HH:MM:SS'."

    dt = dt.replace(tzinfo=from_tz)
    converted = dt.astimezone(to_tz)
    return converted.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


@tool(description="Calculate the number of days between two dates. Format: 'YYYY-MM-DD'.")
def days_between(date1: str, date2: str) -> str:
    """Return the number of days between two dates.

    Args:
        date1: First date in ``YYYY-MM-DD`` format.
        date2: Second date in ``YYYY-MM-DD`` format.

    Returns:
        Number of days as a string.
    """
    try:
        d1 = datetime.date.fromisoformat(date1)
        d2 = datetime.date.fromisoformat(date2)
    except ValueError:
        return "Invalid date format. Use 'YYYY-MM-DD'."

    delta = abs((d2 - d1).days)
    return str(delta)


@tool(description="List common IANA timezone names for a region (e.g. 'Europe', 'America', 'Asia').")
def list_timezones(region: str = "") -> str:
    """List available timezone names, optionally filtered by region prefix.

    Args:
        region: Region prefix to filter (e.g. ``"Europe"``).

    Returns:
        Newline-separated list of matching timezone names.
    """
    all_zones = sorted(zoneinfo.available_timezones())

    if region:
        prefix = region.rstrip("/") + "/"
        filtered = [z for z in all_zones if z.startswith(prefix)]
    else:
        filtered = [z for z in all_zones if "/" in z]

    if not filtered:
        return f"No timezones found for region '{region}'."

    # Cap output to avoid overwhelming the LLM context
    if len(filtered) > 50:
        return "\n".join(filtered[:50]) + f"\n... and {len(filtered) - 50} more"

    return "\n".join(filtered)


# ── Convenience collection ────────────────────────────────────────────────────

DATETIME_TOOLS: list[FunctionTool] = [
    current_datetime,
    convert_timezone,
    days_between,
    list_timezones,
]
"""All datetime tools as a ready-to-use list."""
