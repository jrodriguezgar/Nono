"""Validate an OpenAPI schema against the specification.

Usage (by the skill's inner agent):
    validate_schema(spec_text) -> {"valid": True/False, "errors": [...]}
"""

from __future__ import annotations

import json
import sys


def validate_schema(spec_text: str) -> dict:
    """Validate OpenAPI spec structure (basic checks).

    Args:
        spec_text: OpenAPI spec as a JSON or YAML string.

    Returns:
        Dict with ``valid`` (bool) and ``errors`` (list of strings).
    """
    errors: list[str] = []

    try:
        import yaml  # noqa: F811
        spec = yaml.safe_load(spec_text)
    except Exception:
        try:
            spec = json.loads(spec_text)
        except Exception:
            return {"valid": False, "errors": ["Could not parse as JSON or YAML."]}

    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["Spec must be a mapping/object."]}

    # Required top-level fields
    for field in ("openapi", "info", "paths"):
        if field not in spec:
            errors.append(f"Missing required field: '{field}'")

    # Info must have title and version
    info = spec.get("info", {})
    if isinstance(info, dict):
        if "title" not in info:
            errors.append("info.title is required.")
        if "version" not in info:
            errors.append("info.version is required.")

    # Paths must be a dict
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        errors.append("'paths' must be an object.")

    return {"valid": len(errors) == 0, "errors": errors}


if __name__ == "__main__":
    text = sys.stdin.read()
    result = validate_schema(text)
    print(json.dumps(result, indent=2))
