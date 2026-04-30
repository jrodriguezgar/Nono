"""Data enrichment pipeline workflow with conditional branching.

Steps:
    1. parse — Extract structured fields from raw input.
    2. validate — Check data quality and completeness.
    3. enrich (if valid) — Add contextual information.
    4. flag (if invalid) — Flag data quality issues.

This workflow demonstrates ``branch_if`` for conditional routing.

Usage::

    flow = build_data_enrichment()
    result = flow.run(input="John Smith, CEO at TechCorp, john@techcorp.com")
    # result["enriched"] or result["flags"] depending on validation
"""

from __future__ import annotations

import json as _json

from nono.workflows import END, Workflow, tasker_node


def _check_valid(state: dict) -> bool:
    """Return True if the validation step found no critical issues."""
    validation = state.get("validation", "")

    if isinstance(validation, str):
        lower = validation.lower()
        return "invalid" not in lower and "missing" not in lower

    return True


def build_data_enrichment() -> Workflow:
    """Build the data enrichment pipeline with conditional branching.

    Returns:
        Configured ``Workflow`` with four steps and one branch.
    """
    flow = Workflow("data_enrichment")

    flow.step(
        "parse",
        tasker_node(
            system_prompt=(
                "You are a data parser. Extract structured fields from the "
                "raw input text. Return a JSON object with all identified "
                "fields: name, role, company, email, phone, location. "
                "Use null for fields not found in the input."
            ),
            input_key="input",
            output_key="parsed",
            temperature=0.0,
        ),
    )

    flow.step(
        "validate",
        tasker_node(
            system_prompt=(
                "You are a data quality checker. Given a parsed JSON record, "
                "check for: missing required fields (name, email), invalid "
                "email format, inconsistencies. Return a JSON object with: "
                "is_valid (bool), issues (list of strings), completeness "
                "(float 0-1). If all checks pass, issues should be empty."
            ),
            input_key="parsed",
            output_key="validation",
            temperature=0.0,
        ),
    )

    flow.step(
        "enrich",
        tasker_node(
            system_prompt=(
                "You are a data enrichment specialist. Given a parsed record, "
                "add contextual information: industry category, company size "
                "estimate, geographic region, and a professional summary. "
                "Return the original record merged with the new fields as JSON."
            ),
            input_key="parsed",
            output_key="enriched",
            temperature=0.3,
        ),
    )

    flow.step(
        "flag",
        tasker_node(
            system_prompt=(
                "You are a data quality reporter. Given a parsed record and "
                "its validation results, produce a quality report with: "
                "severity (low, medium, high), recommended_actions (list), "
                "and a human-readable summary. Return as JSON."
            ),
            input_key="validation",
            output_key="flags",
            temperature=0.1,
        ),
    )

    flow.connect("parse", "validate")
    flow.branch_if("validate", _check_valid, then="enrich", otherwise="flag")

    return flow
