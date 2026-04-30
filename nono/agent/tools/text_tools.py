"""Text tools — string analysis and manipulation utilities for agents."""

from __future__ import annotations

import json
import re
from typing import Optional

from ..tool import FunctionTool, ToolContext, tool


@tool(description="Count words, characters, sentences, and lines in a text.")
def text_stats(text: str) -> str:
    """Return basic statistics about a text string.

    Args:
        text: Input text to analyse.

    Returns:
        JSON string with word_count, char_count, sentence_count, line_count.
    """
    words = len(text.split())
    chars = len(text)
    sentences = len(re.split(r"[.!?]+", text.strip())) if text.strip() else 0
    lines = text.count("\n") + 1 if text else 0

    return json.dumps({
        "word_count": words,
        "char_count": chars,
        "sentence_count": sentences,
        "line_count": lines,
    })


@tool(description="Extract all unique URLs from a text.")
def extract_urls(text: str) -> str:
    """Find and return all URLs present in the given text.

    Args:
        text: Input text to search.

    Returns:
        Newline-separated list of URLs, or a message if none found.
    """
    pattern = r"https?://[^\s<>\"')\]]+"
    urls = sorted(set(re.findall(pattern, text)))
    return "\n".join(urls) if urls else "No URLs found."


@tool(description="Extract all email addresses from a text.")
def extract_emails(text: str) -> str:
    """Find and return all email addresses present in the given text.

    Args:
        text: Input text to search.

    Returns:
        Newline-separated list of emails, or a message if none found.
    """
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    emails = sorted(set(re.findall(pattern, text)))
    return "\n".join(emails) if emails else "No email addresses found."


@tool(description="Find and replace a pattern in text. Supports plain text or regex (set is_regex=true).")
def find_replace(text: str, find: str, replace: str, is_regex: bool = False) -> str:
    """Replace occurrences of a pattern in text.

    Args:
        text: Input text.
        find: Pattern to search for.
        replace: Replacement string.
        is_regex: Whether *find* is a regular expression.

    Returns:
        Modified text with replacements applied.
    """
    if is_regex:
        try:
            return re.sub(find, replace, text)
        except re.error as exc:
            return f"Invalid regex pattern: {exc}"

    return text.replace(find, replace)


@tool(description="Truncate text to a maximum number of words, appending '...' if truncated.")
def truncate_text(text: str, max_words: int = 100) -> str:
    """Truncate text to the given word limit.

    Args:
        text: Input text.
        max_words: Maximum number of words to keep.

    Returns:
        Truncated text, with ``...`` appended if it was shortened.
    """
    words = text.split()

    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + "..."


@tool(description="Convert text between formats: uppercase, lowercase, title case, or slug.")
def transform_text(text: str, format: str = "lowercase") -> str:
    """Transform text to the requested format.

    Args:
        text: Input text.
        format: Target format — ``"uppercase"``, ``"lowercase"``,
            ``"title"``, or ``"slug"``.

    Returns:
        Transformed text.
    """
    fmt = format.lower()

    if fmt == "uppercase":
        return text.upper()
    elif fmt == "lowercase":
        return text.lower()
    elif fmt == "title":
        return text.title()
    elif fmt == "slug":
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-")

    return f"Unknown format '{format}'. Use: uppercase, lowercase, title, slug."


# ── Convenience collection ────────────────────────────────────────────────────

TEXT_TOOLS: list[FunctionTool] = [
    text_stats,
    extract_urls,
    extract_emails,
    find_replace,
    truncate_text,
    transform_text,
]
"""All text tools as a ready-to-use list."""
