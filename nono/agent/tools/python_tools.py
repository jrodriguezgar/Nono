"""Python tools — safe code execution and evaluation for agents."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from ..tool import FunctionTool, ToolContext, tool

logger = logging.getLogger("Nono.Agent.Tools.Python")


@tool(description="Evaluate a mathematical expression safely (e.g. '2**10 + sqrt(144)'). Supports basic math functions.")
def calculate(expression: str) -> str:
    """Evaluate a math expression in a restricted namespace.

    Only ``math`` module functions and basic operators are available.
    No access to builtins, imports, or file system.

    Args:
        expression: Mathematical expression to evaluate.

    Returns:
        The result as a string.
    """
    allowed_names: dict[str, Any] = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    allowed_names["min"] = min
    allowed_names["max"] = max
    allowed_names["sum"] = sum
    allowed_names["len"] = len
    allowed_names["int"] = int
    allowed_names["float"] = float

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
    except Exception as exc:
        return f"Error evaluating expression: {exc}"

    return str(result)


@tool(description="Execute a Python code snippet using Nono's sandboxed CodeExecuter. Returns stdout output and any errors.")
def run_python(code: str, tool_context: ToolContext = None) -> str:
    """Execute Python code in a sandboxed environment.

    Delegates to ``nono.executer.CodeExecuter`` for safe execution.
    Falls back to a restricted ``exec()`` if the executer is unavailable.

    Args:
        code: Python source code to execute.
        tool_context: Optional context (used to store execution results).

    Returns:
        Execution output or error message.
    """
    # Try the full sandboxed executer first
    try:
        from ...executer.genai_executer import CodeExecuter

        executer = CodeExecuter()
        result = executer.execute_code(code)
        output = getattr(result, "output", str(result))

        if tool_context is not None:
            tool_context.state_set("last_code", code)
            tool_context.state_set("last_code_output", output)

        return output

    except ImportError:
        logger.debug("CodeExecuter not available, using restricted exec")
    except Exception as exc:
        return f"Execution error: {exc}"

    # Fallback: restricted exec with captured stdout
    import io
    import contextlib

    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {"print": print, "range": range, "len": len}})  # noqa: S102
    except Exception as exc:
        return f"Error: {exc}"

    output = stdout.getvalue()

    if tool_context is not None:
        tool_context.state_set("last_code", code)
        tool_context.state_set("last_code_output", output)

    return output if output else "(no output)"


@tool(description="Convert a JSON string to a formatted, pretty-printed version.")
def format_json(json_string: str) -> str:
    """Parse and pretty-print a JSON string.

    Args:
        json_string: Raw JSON text.

    Returns:
        Formatted JSON with 2-space indentation.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    return json.dumps(data, indent=2, ensure_ascii=False)


# ── Convenience collection ────────────────────────────────────────────────────

PYTHON_TOOLS: list[FunctionTool] = [
    calculate,
    run_python,
    format_json,
]
"""All Python/code tools as a ready-to-use list."""
