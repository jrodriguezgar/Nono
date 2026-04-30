"""Compute a numeric quality score from a code review issues list.

Usage from the LLM via function calling::

    result = main('{"issues": [{"severity": "high"}, {"severity": "low"}]}')
    # => '{"score": 6, "breakdown": {"high": 1, "medium": 0, "low": 1}}'
"""

from __future__ import annotations

import json


_WEIGHTS = {"high": 3, "medium": 1.5, "low": 0.5}


def main(input: str) -> str:
    """Score code quality from a list of review issues.

    Args:
        input: JSON string with an ``issues`` array, each having a
            ``severity`` field (``high``, ``medium``, or ``low``).

    Returns:
        JSON string with ``score`` (1-10) and ``breakdown`` counts.
    """
    data = json.loads(input)
    issues = data.get("issues", [])

    breakdown = {"high": 0, "medium": 0, "low": 0}
    for issue in issues:
        sev = issue.get("severity", "low")
        if sev in breakdown:
            breakdown[sev] += 1

    penalty = sum(
        breakdown[sev] * _WEIGHTS[sev] for sev in breakdown
    )
    score = max(1, min(10, round(10 - penalty)))

    return json.dumps({"score": score, "breakdown": breakdown})
