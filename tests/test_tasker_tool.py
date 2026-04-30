"""Tests for nono.agent.tasker_tool integration.

Verifies that ``tasker_tool`` and ``json_task_tool`` correctly create
``FunctionTool`` instances that delegate to ``TaskExecutor``.
All external calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import tempfile
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — allow running as ``python tests/test_tasker_tool.py``
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock jinjapromptpy if not installed (prevents import error in genai_tasker)
# ---------------------------------------------------------------------------
if "jinjapromptpy" not in sys.modules:
    _jpp_mock = MagicMock()
    sys.modules["jinjapromptpy"] = _jpp_mock
    # genai_tasker imports specific names from jinjapromptpy
    sys.modules["jinjapromptpy.prompt_generator"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_template"] = _jpp_mock
    sys.modules["jinjapromptpy.batch_generator"] = _jpp_mock

from nono.agent.tasker_tool import tasker_tool, json_task_tool
from nono.agent.tool import FunctionTool

# Counter for pass/fail
_pass = 0
_fail = 0


def _ok(label: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def _ko(label: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    msg = f"  FAIL  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _assert(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        _ok(label)
    else:
        _ko(label, detail)


# ===== Tests for tasker_tool =====

def test_tasker_tool_returns_function_tool() -> None:
    """tasker_tool returns a FunctionTool with correct metadata."""
    t = tasker_tool(
        name="my_task",
        description="Does something useful.",
        provider="openai",
        model="gpt-4o-mini",
    )
    _assert(isinstance(t, FunctionTool), "tasker_tool returns FunctionTool")
    _assert(t.name == "my_task", "tool name matches")
    _assert(t.description == "Does something useful.", "tool description matches")


def test_tasker_tool_schema_has_prompt_param() -> None:
    """The generated tool schema exposes a 'prompt' parameter."""
    t = tasker_tool(name="test_schema")
    schema = t.parameters_schema
    _assert("prompt" in schema.get("properties", {}), "schema has 'prompt' property")
    _assert("prompt" in schema.get("required", []), "'prompt' is required")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_tool_invoke_delegates_to_executor(mock_executor_cls: MagicMock) -> None:
    """Invoking the tool calls TaskExecutor.execute with the prompt."""
    mock_executor_instance = MagicMock()
    mock_executor_instance.execute.return_value = "mocked result"
    mock_executor_cls.return_value = mock_executor_instance

    t = tasker_tool(
        name="delegator",
        description="Test delegation.",
        provider="google",
        model="gemini-3-flash-preview",
        temperature=0.5,
        max_tokens=1024,
    )

    result = t.invoke({"prompt": "Summarise this text."})

    _assert(result == "mocked result", "invoke returns executor result")
    mock_executor_cls.assert_called_once_with(
        provider="google",
        model="gemini-3-flash-preview",
        api_key=None,
        temperature=0.5,
        max_tokens=1024,
    )
    mock_executor_instance.execute.assert_called_once()
    call_args = mock_executor_instance.execute.call_args
    _assert(call_args[0][0] == "Summarise this text.", "prompt passed as input_data")


def test_tasker_tool_with_system_prompt() -> None:
    """When system_prompt is set, input_data becomes a message list."""
    mock_executor_cls = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_instance.execute.return_value = "system result"
    mock_executor_cls.return_value = mock_executor_instance

    t = tasker_tool(
        name="with_system",
        description="Has system prompt.",
        system_prompt="You are an expert analyst.",
    )

    with patch("nono.tasker.genai_tasker.TaskExecutor", mock_executor_cls):
        result = t.invoke({"prompt": "Analyse data."})

    _assert(result == "system result", "result returned with system prompt")
    call_args = mock_executor_instance.execute.call_args
    input_data = call_args[0][0]
    _assert(isinstance(input_data, list), "input_data is message list when system_prompt set")
    _assert(input_data[0]["role"] == "system", "first message is system")
    _assert(input_data[0]["content"] == "You are an expert analyst.", "system content matches")
    _assert(input_data[1]["role"] == "user", "second message is user")
    _assert(input_data[1]["content"] == "Analyse data.", "user content matches")


def test_tasker_tool_with_output_schema() -> None:
    """output_schema is forwarded to TaskExecutor.execute."""
    mock_executor_cls = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_instance.execute.return_value = '{"key": "value"}'
    mock_executor_cls.return_value = mock_executor_instance

    schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    t = tasker_tool(name="schema_task", output_schema=schema)

    with patch("nono.tasker.genai_tasker.TaskExecutor", mock_executor_cls):
        t.invoke({"prompt": "Generate JSON."})

    call_args = mock_executor_instance.execute.call_args
    passed_schema = call_args[1].get("output_schema") if call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else None
    _assert(passed_schema == schema, "output_schema forwarded to execute")


def test_tasker_tool_default_values() -> None:
    """Default name and description are set when not overridden."""
    t = tasker_tool()
    _assert(t.name == "execute_task", "default name is 'execute_task'")
    _assert("TaskExecutor" in t.description, "default description mentions TaskExecutor")


# ===== Tests for json_task_tool =====

def _make_task_json(tmp_dir: str, task_def: dict) -> str:
    """Write a task JSON to a temp file and return its path."""
    path = os.path.join(tmp_dir, "test_task.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(task_def, f)
    return path


def test_json_task_tool_reads_metadata() -> None:
    """json_task_tool extracts name/description from the JSON file."""
    task_def = {
        "task": {
            "name": "classifier",
            "description": "Classifies input data.",
        },
        "genai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        "prompts": {"system": "", "user": "{data_input_json}"},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _make_task_json(tmp_dir, task_def)
        t = json_task_tool(path)

    _assert(t.name == "classifier", "name from JSON task.name")
    _assert(t.description == "Classifies input data.", "description from JSON task.description")


def test_json_task_tool_overrides_metadata() -> None:
    """Explicit name/description override JSON values."""
    task_def = {
        "task": {"name": "original", "description": "Original desc."},
        "genai": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prompts": {"system": "", "user": "{data_input_json}"},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _make_task_json(tmp_dir, task_def)
        t = json_task_tool(path, name="custom_name", description="Custom desc.")

    _assert(t.name == "custom_name", "overridden name")
    _assert(t.description == "Custom desc.", "overridden description")


def test_json_task_tool_schema_has_data_param() -> None:
    """The generated tool schema exposes a 'data' parameter."""
    task_def = {
        "task": {"name": "test"},
        "genai": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prompts": {"system": "", "user": ""},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _make_task_json(tmp_dir, task_def)
        t = json_task_tool(path)

    schema = t.parameters_schema
    _assert("data" in schema.get("properties", {}), "schema has 'data' property")
    _assert("data" in schema.get("required", []), "'data' is required")


def test_json_task_tool_invoke_delegates() -> None:
    """Invoking calls TaskExecutor.run_json_task with the file and data."""
    task_def = {
        "task": {"name": "delegated"},
        "genai": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prompts": {"system": "", "user": "{data_input_json}"},
    }

    mock_executor_cls = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_instance.run_json_task.return_value = '{"results": []}'
    mock_executor_cls.return_value = mock_executor_instance

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _make_task_json(tmp_dir, task_def)
        t = json_task_tool(path)

        with patch("nono.tasker.genai_tasker.TaskExecutor", mock_executor_cls):
            result = t.invoke({"data": '["Alice", "Bob"]'})

    _assert(result == '{"results": []}', "invoke returns run_json_task result")
    mock_executor_instance.run_json_task.assert_called_once()
    call_args = mock_executor_instance.run_json_task.call_args
    _assert(call_args[0][1] == '["Alice", "Bob"]', "data passed to run_json_task")


def test_json_task_tool_file_not_found() -> None:
    """json_task_tool raises FileNotFoundError for missing files."""
    try:
        json_task_tool("/nonexistent/task.json")
        _ko("FileNotFoundError on missing file", "no exception raised")
    except FileNotFoundError:
        _ok("FileNotFoundError on missing file")


def test_json_task_tool_fallback_name() -> None:
    """When task.name is missing, filename (without extension) is used."""
    task_def = {
        "task": {},
        "genai": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prompts": {"system": "", "user": ""},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "my_custom_task.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task_def, f)
        t = json_task_tool(path)

    _assert(t.name == "my_custom_task", "fallback name from filename")


def test_json_task_tool_provider_override() -> None:
    """Provider/model overrides take precedence over JSON values."""
    task_def = {
        "task": {"name": "test_override"},
        "genai": {"provider": "google", "model": "gemini-3-flash-preview"},
        "prompts": {"system": "", "user": ""},
    }

    mock_executor_cls = MagicMock()
    mock_executor_instance = MagicMock()
    mock_executor_instance.run_json_task.return_value = "ok"
    mock_executor_cls.return_value = mock_executor_instance

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = _make_task_json(tmp_dir, task_def)
        t = json_task_tool(path, provider="openai", model="gpt-4o-mini")

        with patch("nono.tasker.genai_tasker.TaskExecutor", mock_executor_cls):
            t.invoke({"data": "test"})

    mock_executor_cls.assert_called_once_with(
        provider="openai",
        model="gpt-4o-mini",
        api_key=None,
    )
    _ok("provider/model overrides forwarded to TaskExecutor")


# ===== Integration shape test =====

def test_tool_usable_in_agent_tools_list() -> None:
    """The tool can be added to an Agent's tools list."""
    t = tasker_tool(name="agent_ready", description="Ready for agent use.")

    # Verify it has the interface an Agent expects
    _assert(hasattr(t, "name"), "has .name")
    _assert(hasattr(t, "description"), "has .description")
    _assert(hasattr(t, "invoke"), "has .invoke()")
    _assert(hasattr(t, "to_function_declaration"), "has .to_function_declaration()")

    decl = t.to_function_declaration()
    _assert(decl["type"] == "function", "declaration type is 'function'")
    _assert(decl["function"]["name"] == "agent_ready", "declaration name matches")


# ===== Runner =====

if __name__ == "__main__":
    print("=" * 60)
    print("  tasker_tool integration tests")
    print("=" * 60)
    print()

    tests = [
        test_tasker_tool_returns_function_tool,
        test_tasker_tool_schema_has_prompt_param,
        test_tasker_tool_invoke_delegates_to_executor,
        test_tasker_tool_with_system_prompt,
        test_tasker_tool_with_output_schema,
        test_tasker_tool_default_values,
        test_json_task_tool_reads_metadata,
        test_json_task_tool_overrides_metadata,
        test_json_task_tool_schema_has_data_param,
        test_json_task_tool_invoke_delegates,
        test_json_task_tool_file_not_found,
        test_json_task_tool_fallback_name,
        test_json_task_tool_provider_override,
        test_tool_usable_in_agent_tools_list,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            _ko(test_fn.__name__, str(exc))

    print()
    total = _pass + _fail
    print(f"Results: {_pass}/{total} passed, {_fail} failed")
    if _fail:
        sys.exit(1)
    print("All tests passed!")
