"""Web tools — HTTP fetch and web content utilities for agents."""

from __future__ import annotations

import json
import logging
from typing import Optional

from ..tool import FunctionTool, ToolContext, tool

logger = logging.getLogger("Nono.Agent.Tools.Web")


@tool(description="Fetch the text content of a web page given its URL. Returns the first 5000 characters by default.")
def fetch_webpage(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return its text content.

    Uses ``httpx`` if available, otherwise falls back to ``urllib``.
    Only ``http`` and ``https`` schemes are allowed.

    Args:
        url: The URL to fetch (must start with ``http://`` or ``https://``).
        max_chars: Maximum characters to return from the response body.

    Returns:
        Response text, truncated to *max_chars*.
    """
    if not url.startswith(("http://", "https://")):
        return "Invalid URL. Only http:// and https:// URLs are supported."

    try:
        import httpx

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text

    except ImportError:
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Nono/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            return f"Request failed: {exc.reason}"

    except Exception as exc:
        logger.warning("fetch_webpage failed for %s: %s", url, exc)
        return f"Request failed: {exc}"

    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated, {len(text)} total chars)"

    return text


@tool(description="Make an HTTP GET request to a JSON API and return the parsed response.")
def fetch_json(url: str) -> str:
    """Fetch a JSON API endpoint and return the formatted result.

    Args:
        url: The API URL (must start with ``http://`` or ``https://``).

    Returns:
        Pretty-printed JSON string.
    """
    if not url.startswith(("http://", "https://")):
        return "Invalid URL. Only http:// and https:// URLs are supported."

    try:
        import httpx

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()

    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Nono/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
        except Exception as exc:
            return f"Request failed: {exc}"

    except Exception as exc:
        logger.warning("fetch_json failed for %s: %s", url, exc)
        return f"Request failed: {exc}"

    text = json.dumps(data, indent=2, ensure_ascii=False)

    if len(text) > 8000:
        return text[:8000] + "\n... (truncated)"

    return text


@tool(description="Check if a URL is reachable. Returns the HTTP status code and headers.")
def check_url(url: str) -> str:
    """Send a HEAD request to check URL availability.

    Args:
        url: The URL to check (must start with ``http://`` or ``https://``).

    Returns:
        Status code and selected headers as a formatted string.
    """
    if not url.startswith(("http://", "https://")):
        return "Invalid URL. Only http:// and https:// URLs are supported."

    try:
        import httpx

        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.head(url)

        info = {
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type", "unknown"),
            "server": resp.headers.get("server", "unknown"),
        }

    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Nono/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                info = {
                    "status": resp.status,
                    "content_type": resp.headers.get("Content-Type", "unknown"),
                    "server": resp.headers.get("Server", "unknown"),
                }
        except Exception as exc:
            return f"Unreachable: {exc}"

    except Exception as exc:
        return f"Unreachable: {exc}"

    return json.dumps(info, indent=2)


# ── Convenience collection ────────────────────────────────────────────────────

WEB_TOOLS: list[FunctionTool] = [
    fetch_webpage,
    fetch_json,
    check_url,
]
"""All web tools as a ready-to-use list."""
