"""Tests for the Nono hooks engine (nono/hooks.py).

Covers:
- HookEvent parsing from various string formats
- HookContext serialization (to_dict, to_json)
- HookResult merging and parsing (most restrictive wins)
- Hook execution (Python callable and shell command)
- Hook matcher filtering (regex on tool names)
- HookManager registration, firing, config loading, and serialization
- Integration with BaseAgent lifecycle (PreAgentRun, PostAgentRun, Error)
- Integration with LlmAgent tool hooks (PreToolUse blocking, PostToolUse context)
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nono.hooks import (
    Hook,
    HookContext,
    HookEvent,
    HookManager,
    HookResult,
    HookType,
    _parse_hook_entry,
    discover_hooks,
    load_agent_scoped_hooks,
    load_hooks_from_file,
    HOOK_DISCOVERY_PATHS,
)


# ══════════════════════════════════════════════════════════════════════════════
# HookEvent
# ══════════════════════════════════════════════════════════════════════════════


class TestHookEvent:
    """HookEvent.from_string parses various formats."""

    def test_pascal_case(self) -> None:
        assert HookEvent.from_string("PreToolUse") == HookEvent.PRE_TOOL_USE

    def test_upper_snake_case(self) -> None:
        assert HookEvent.from_string("PRE_TOOL_USE") == HookEvent.PRE_TOOL_USE

    def test_session_start(self) -> None:
        assert HookEvent.from_string("SessionStart") == HookEvent.SESSION_START

    def test_post_agent_run(self) -> None:
        assert HookEvent.from_string("PostAgentRun") == HookEvent.POST_AGENT_RUN

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown hook event"):
            HookEvent.from_string("NonExistentEvent")

    def test_all_events_have_values(self) -> None:
        for event in HookEvent:
            assert isinstance(event.value, str)
            assert len(event.value) > 0


# ══════════════════════════════════════════════════════════════════════════════
# HookContext
# ══════════════════════════════════════════════════════════════════════════════


class TestHookContext:
    """HookContext serialization."""

    def test_to_dict_minimal(self) -> None:
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            session_id="s123",
            agent_name="test_agent",
        )
        d = ctx.to_dict()

        assert d["hookEventName"] == "SessionStart"
        assert d["sessionId"] == "s123"
        assert d["agent_name"] == "test_agent"
        assert "tool_name" not in d

    def test_to_dict_with_tool_fields(self) -> None:
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            session_id="s1",
            agent_name="a1",
            tool_name="get_weather",
            tool_input={"city": "Madrid"},
        )
        d = ctx.to_dict()

        assert d["tool_name"] == "get_weather"
        assert d["tool_input"] == {"city": "Madrid"}

    def test_to_json_roundtrip(self) -> None:
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            session_id="s2",
            tool_response="Sunny, 22°C",
        )
        raw = ctx.to_json()
        parsed = json.loads(raw)

        assert parsed["hookEventName"] == "PostToolUse"
        assert parsed["tool_response"] == "Sunny, 22°C"

    def test_to_dict_excludes_none_fields(self) -> None:
        ctx = HookContext(event=HookEvent.STOP, session_id="s3")
        d = ctx.to_dict()

        assert "error" not in d
        assert "tool_name" not in d
        assert "llm_response" not in d


# ══════════════════════════════════════════════════════════════════════════════
# HookResult
# ══════════════════════════════════════════════════════════════════════════════


class TestHookResult:
    """HookResult merging, parsing, and properties."""

    def test_default_is_success(self) -> None:
        r = HookResult()

        assert r.should_block is False
        assert r.should_stop is False
        assert r.exit_code == 0

    def test_should_block_explicit(self) -> None:
        r = HookResult(block=True, block_reason="Policy violation")

        assert r.should_block is True

    def test_should_block_exit_code_2(self) -> None:
        r = HookResult(exit_code=2)

        assert r.should_block is True

    def test_should_block_deny_permission(self) -> None:
        r = HookResult(permission_decision="deny")

        assert r.should_block is True

    def test_should_stop(self) -> None:
        r = HookResult(continue_execution=False, stop_reason="Done")

        assert r.should_stop is True

    def test_from_json_empty(self) -> None:
        r = HookResult.from_json("")

        assert r.continue_execution is True

    def test_from_json_valid(self) -> None:
        data = {
            "continue": False,
            "stopReason": "Security",
            "systemMessage": "Warning: blocked",
        }
        r = HookResult.from_json(json.dumps(data))

        assert r.continue_execution is False
        assert r.stop_reason == "Security"
        assert r.system_message == "Warning: blocked"

    def test_from_json_hook_specific_output(self) -> None:
        data = {
            "hookSpecificOutput": {
                "permissionDecision": "deny",
                "permissionDecisionReason": "Blocked",
                "additionalContext": "Read-only",
                "updatedInput": {"safe": True},
            }
        }
        r = HookResult.from_json(json.dumps(data))

        assert r.permission_decision == "deny"
        assert r.block is True
        assert r.additional_context == "Read-only"
        assert r.updated_input == {"safe": True}

    def test_from_json_invalid(self) -> None:
        r = HookResult.from_json("not json {")

        assert r.raw_output == "not json {"

    def test_merge_most_restrictive_wins(self) -> None:
        r1 = HookResult(permission_decision="allow")
        r2 = HookResult(permission_decision="deny", block_reason="Policy")

        merged = r1.merge(r2)

        assert merged.permission_decision == "deny"
        assert merged.block_reason == "Policy"

    def test_merge_stop_propagates(self) -> None:
        r1 = HookResult(continue_execution=True)
        r2 = HookResult(continue_execution=False, stop_reason="Halt")

        merged = r1.merge(r2)

        assert merged.should_stop is True
        assert merged.stop_reason == "Halt"

    def test_merge_block_propagates(self) -> None:
        r1 = HookResult()
        r2 = HookResult(block=True)

        merged = r1.merge(r2)

        assert merged.block is True

    def test_merge_messages_concatenate(self) -> None:
        r1 = HookResult(system_message="Msg1")
        r2 = HookResult(system_message="Msg2")

        merged = r1.merge(r2)

        assert "Msg1" in merged.system_message
        assert "Msg2" in merged.system_message

    def test_merge_exit_code_max(self) -> None:
        r1 = HookResult(exit_code=0)
        r2 = HookResult(exit_code=2)

        merged = r1.merge(r2)

        assert merged.exit_code == 2


# ══════════════════════════════════════════════════════════════════════════════
# Hook
# ══════════════════════════════════════════════════════════════════════════════


class TestHook:
    """Hook definition, matching, and execution."""

    def test_requires_fn_or_command(self) -> None:
        with pytest.raises(ValueError, match="'fn'.*'command'.*'prompt'.*'task'.*'skill'.*'tool'"):
            Hook()

    def test_callable_hook_executes(self) -> None:
        def my_hook(ctx: HookContext) -> HookResult:
            return HookResult(system_message=f"Agent: {ctx.agent_name}")

        hook = Hook(fn=my_hook, name="test_fn")
        ctx = HookContext(agent_name="helper")
        result = hook.execute(ctx)

        assert result.system_message == "Agent: helper"

    def test_callable_hook_returns_none(self) -> None:
        hook = Hook(fn=lambda ctx: None, name="noop")
        result = hook.execute(HookContext())

        assert result.exit_code == 0
        assert result.should_block is False

    def test_callable_hook_exception(self) -> None:
        def bad_hook(ctx: HookContext) -> HookResult:
            raise RuntimeError("Boom!")

        hook = Hook(fn=bad_hook, name="bad")
        result = hook.execute(HookContext())

        assert result.exit_code == 1
        assert "Boom!" in result.raw_error

    def test_disabled_hook_skips(self) -> None:
        hook = Hook(fn=lambda ctx: HookResult(block=True), enabled=False)
        result = hook.execute(HookContext())

        assert result.should_block is False

    def test_matcher_matches_tool_name(self) -> None:
        hook = Hook(command="echo ok", matcher="Write|Edit")

        assert hook.matches("Write") is True
        assert hook.matches("EditFile") is True
        assert hook.matches("Read") is False

    def test_matcher_empty_matches_all(self) -> None:
        hook = Hook(command="echo ok")

        assert hook.matches("anything") is True
        assert hook.matches("") is True

    def test_matcher_regex(self) -> None:
        hook = Hook(command="echo ok", matcher=r"^get_.*")

        assert hook.matches("get_weather") is True
        assert hook.matches("set_config") is False

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only command test",
    )
    def test_shell_command_echo(self) -> None:
        hook = Hook(
            command='powershell -Command "echo hello"',
            name="echo_test",
        )
        ctx = HookContext(event=HookEvent.POST_TOOL_USE, session_id="t1")
        result = hook.execute(ctx)

        assert result.exit_code == 0
        assert "hello" in result.raw_output

    def test_shell_command_timeout(self) -> None:
        if sys.platform == "win32":
            cmd = 'powershell -Command "Start-Sleep -Seconds 60"'
        else:
            cmd = "sleep 60"

        hook = Hook(command=cmd, timeout=1, name="slow")
        result = hook.execute(HookContext())

        assert result.exit_code == 1
        assert "timed out" in result.raw_error

    def test_shell_command_not_found(self) -> None:
        hook = Hook(command="nonexistent_command_xyz_12345")
        result = hook.execute(HookContext(event=HookEvent.STOP))

        assert result.exit_code == 1


# ══════════════════════════════════════════════════════════════════════════════
# HookManager
# ══════════════════════════════════════════════════════════════════════════════


class TestHookManager:
    """HookManager registration, firing, and configuration."""

    def test_register_and_count(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.SESSION_START, Hook(fn=lambda ctx: None))
        mgr.register(HookEvent.SESSION_START, Hook(fn=lambda ctx: None))
        mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None))

        assert mgr.count(HookEvent.SESSION_START) == 2
        assert mgr.count(HookEvent.STOP) == 1
        assert mgr.count() == 3

    def test_fluent_register(self) -> None:
        mgr = HookManager()
        result = mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None))

        assert result is mgr

    def test_events_list(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.PRE_TOOL_USE, Hook(fn=lambda ctx: None))

        assert HookEvent.PRE_TOOL_USE in mgr.events
        assert HookEvent.STOP not in mgr.events

    def test_unregister(self) -> None:
        hook = Hook(fn=lambda ctx: None, name="removable")
        mgr = HookManager()
        mgr.register(HookEvent.STOP, hook)

        assert mgr.unregister(HookEvent.STOP, hook) is True
        assert mgr.count(HookEvent.STOP) == 0

    def test_unregister_not_found(self) -> None:
        mgr = HookManager()
        hook = Hook(fn=lambda ctx: None)

        assert mgr.unregister(HookEvent.STOP, hook) is False

    def test_clear_event(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None))
        mgr.register(HookEvent.ERROR, Hook(fn=lambda ctx: None))
        mgr.clear(HookEvent.STOP)

        assert mgr.count(HookEvent.STOP) == 0
        assert mgr.count(HookEvent.ERROR) == 1

    def test_clear_all(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None))
        mgr.register(HookEvent.ERROR, Hook(fn=lambda ctx: None))
        mgr.clear()

        assert mgr.count() == 0

    def test_fire_no_hooks(self) -> None:
        mgr = HookManager()
        result = mgr.fire(HookEvent.STOP, HookContext())

        assert result.should_block is False
        assert result.exit_code == 0

    def test_fire_single_hook(self) -> None:
        calls: list[str] = []

        def on_start(ctx: HookContext) -> HookResult:
            calls.append(ctx.session_id)
            return HookResult(system_message="Started")

        mgr = HookManager()
        mgr.register(HookEvent.SESSION_START, Hook(fn=on_start))
        result = mgr.fire(
            HookEvent.SESSION_START,
            HookContext(session_id="s1"),
        )

        assert calls == ["s1"]
        assert result.system_message == "Started"

    def test_fire_multiple_hooks_merge(self) -> None:
        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(fn=lambda ctx: HookResult(system_message="Hook1")),
        )
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(fn=lambda ctx: HookResult(system_message="Hook2")),
        )

        result = mgr.fire(HookEvent.PRE_TOOL_USE, HookContext())

        assert "Hook1" in result.system_message
        assert "Hook2" in result.system_message

    def test_fire_stops_on_block(self) -> None:
        calls: list[str] = []

        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(fn=lambda ctx: HookResult(block=True, block_reason="Denied")),
        )
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(fn=lambda ctx: (calls.append("second"), HookResult())[1]),
        )

        result = mgr.fire(HookEvent.PRE_TOOL_USE, HookContext())

        assert result.should_block is True
        assert calls == []  # second hook never ran

    def test_fire_matcher_filtering(self) -> None:
        calls: list[str] = []

        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(
                fn=lambda ctx: (calls.append("write_hook"), HookResult())[1],
                matcher="Write",
            ),
        )
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(
                fn=lambda ctx: (calls.append("all_hook"), HookResult())[1],
            ),
        )

        mgr.fire(HookEvent.PRE_TOOL_USE, HookContext(), tool_name="Read")

        assert "write_hook" not in calls
        assert "all_hook" in calls

    def test_fire_stops_on_session_stop(self) -> None:
        calls: list[str] = []

        mgr = HookManager()
        mgr.register(
            HookEvent.STOP,
            Hook(fn=lambda ctx: HookResult(continue_execution=False, stop_reason="Done")),
        )
        mgr.register(
            HookEvent.STOP,
            Hook(fn=lambda ctx: (calls.append("second"), HookResult())[1]),
        )

        result = mgr.fire(HookEvent.STOP, HookContext())

        assert result.should_stop is True
        assert calls == []

    def test_fire_disabled_hooks_skipped(self) -> None:
        calls: list[str] = []

        mgr = HookManager()
        mgr.register(
            HookEvent.STOP,
            Hook(
                fn=lambda ctx: (calls.append("disabled"), HookResult())[1],
                enabled=False,
            ),
        )
        mgr.register(
            HookEvent.STOP,
            Hook(fn=lambda ctx: (calls.append("enabled"), HookResult())[1]),
        )

        mgr.fire(HookEvent.STOP, HookContext())

        assert "disabled" not in calls
        assert "enabled" in calls


class TestHookManagerConfig:
    """HookManager.load_config and to_config."""

    def test_load_vscode_format(self) -> None:
        config = {
            "hooks": {
                "PreToolUse": [
                    {"type": "command", "command": "echo validate"}
                ],
                "PostToolUse": [
                    {"type": "command", "command": "echo format"}
                ],
            }
        }
        mgr = HookManager()
        loaded = mgr.load_config(config)

        assert loaded == 2
        assert mgr.count(HookEvent.PRE_TOOL_USE) == 1
        assert mgr.count(HookEvent.POST_TOOL_USE) == 1

    def test_load_claude_code_format(self) -> None:
        config = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {"type": "command", "command": "python -m black ."}
                        ],
                    }
                ],
            }
        }
        mgr = HookManager()
        loaded = mgr.load_config(config)

        assert loaded == 1

    def test_load_unknown_event_warns(self) -> None:
        config = {
            "hooks": {
                "UnknownEvent": [{"type": "command", "command": "echo hi"}]
            }
        }
        mgr = HookManager()
        loaded = mgr.load_config(config)

        assert loaded == 0

    def test_load_file(self, tmp_path) -> None:
        hook_file = tmp_path / "hooks.json"
        hook_file.write_text(json.dumps({
            "hooks": {
                "SessionStart": [
                    {"type": "command", "command": "echo start"}
                ]
            }
        }))
        mgr = HookManager()
        loaded = mgr.load_file(hook_file)

        assert loaded == 1

    def test_load_file_not_found(self) -> None:
        mgr = HookManager()

        with pytest.raises(FileNotFoundError):
            mgr.load_file("/nonexistent/path/hooks.json")

    def test_load_directory(self, tmp_path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({
            "hooks": {
                "SessionStart": [{"type": "command", "command": "echo a"}]
            }
        }))
        (tmp_path / "b.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{"type": "command", "command": "echo b"}]
            }
        }))

        mgr = HookManager()
        total = mgr.load_directory(tmp_path)

        assert total == 2

    def test_load_directory_nonexistent(self) -> None:
        mgr = HookManager()
        total = mgr.load_directory("/nonexistent/dir")

        assert total == 0

    def test_to_config_roundtrip(self) -> None:
        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_TOOL_USE,
            Hook(command="echo pre", matcher="Write", name="pre_hook"),
        )
        config = mgr.to_config()

        assert "hooks" in config
        assert "PreToolUse" in config["hooks"]
        assert len(config["hooks"]["PreToolUse"]) == 1
        assert config["hooks"]["PreToolUse"][0]["command"] == "echo pre"
        assert config["hooks"]["PreToolUse"][0]["matcher"] == "Write"

    def test_repr(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None))

        assert "Stop" in repr(mgr)


class TestLoadHooksFromFile:
    """Convenience function load_hooks_from_file."""

    def test_creates_manager(self, tmp_path) -> None:
        hook_file = tmp_path / "hooks.json"
        hook_file.write_text(json.dumps({
            "hooks": {
                "PostToolUse": [
                    {"type": "command", "command": "echo done"}
                ]
            }
        }))

        mgr = load_hooks_from_file(hook_file)

        assert isinstance(mgr, HookManager)
        assert mgr.count(HookEvent.POST_TOOL_USE) == 1


class TestParseHookEntry:
    """_parse_hook_entry handles various formats."""

    def test_vscode_flat(self) -> None:
        hook = _parse_hook_entry({
            "type": "command",
            "command": "echo hello",
            "timeout": 10,
        })

        assert hook is not None
        assert hook.command == "echo hello"
        assert hook.timeout == 10

    def test_claude_code_nested(self) -> None:
        hook = _parse_hook_entry({
            "matcher": "Write|Edit",
            "hooks": [
                {"type": "command", "command": "black ."}
            ],
        })

        assert hook is not None
        assert hook.command == "black ."
        assert hook.matcher == "Write|Edit"

    def test_with_os_specific(self) -> None:
        hook = _parse_hook_entry({
            "type": "command",
            "command": "./format.sh",
            "windows": "powershell -File format.ps1",
            "linux": "./format-linux.sh",
            "osx": "./format-mac.sh",
        })

        assert hook is not None
        assert hook.command_windows == "powershell -File format.ps1"
        assert hook.command_linux == "./format-linux.sh"

    def test_empty_nested_hooks(self) -> None:
        hook = _parse_hook_entry({"hooks": []})

        assert hook is None

    def test_unrecognized_format(self) -> None:
        hook = _parse_hook_entry({"unknown_key": "value"})

        assert hook is None

    def test_prompt_type(self) -> None:
        hook = _parse_hook_entry({
            "type": "prompt",
            "prompt": "Validate: {tool_response}",
            "provider": "google",
            "model": "gemini-3-flash-preview",
        })

        assert hook is not None
        assert hook.hook_type == HookType.PROMPT
        assert hook.prompt == "Validate: {tool_response}"
        assert hook.provider == "google"
        assert hook.model == "gemini-3-flash-preview"

    def test_task_type(self) -> None:
        hook = _parse_hook_entry({
            "type": "task",
            "task": "review_output",
            "provider": "openai",
        })

        assert hook is not None
        assert hook.hook_type == HookType.TASK
        assert hook.task == "review_output"
        assert hook.provider == "openai"

    def test_skill_type(self) -> None:
        hook = _parse_hook_entry({
            "type": "skill",
            "skill": "summarize_skill",
        })

        assert hook is not None
        assert hook.hook_type == HookType.SKILL
        assert hook.skill == "summarize_skill"

    def test_tool_type(self) -> None:
        hook = _parse_hook_entry({
            "type": "tool",
            "tool": "audit_logger",
            "tool_args": {"action": "call", "target": "{tool_name}"},
        })

        assert hook is not None
        assert hook.hook_type == HookType.TOOL
        assert hook.tool == "audit_logger"
        assert hook.tool_args == {"action": "call", "target": "{tool_name}"}

    def test_prompt_implicit_type(self) -> None:
        """Detect prompt type from 'prompt' key without explicit type."""
        hook = _parse_hook_entry({"prompt": "Hello {user_message}"})

        assert hook is not None
        assert hook.hook_type == HookType.PROMPT

    def test_task_implicit_type(self) -> None:
        """Detect task type from 'task' key without explicit type."""
        hook = _parse_hook_entry({"task": "my_task"})

        assert hook is not None
        assert hook.hook_type == HookType.TASK

    def test_skill_implicit_type(self) -> None:
        """Detect skill type from 'skill' key without explicit type."""
        hook = _parse_hook_entry({"skill": "my_skill"})

        assert hook is not None
        assert hook.hook_type == HookType.SKILL


# ══════════════════════════════════════════════════════════════════════════════
# HookType
# ══════════════════════════════════════════════════════════════════════════════


class TestHookType:
    """HookType enum and Hook.hook_type property."""

    def test_function_type(self) -> None:
        hook = Hook(fn=lambda ctx: None)
        assert hook.hook_type == HookType.FUNCTION

    def test_command_type(self) -> None:
        hook = Hook(command="echo hi")
        assert hook.hook_type == HookType.COMMAND

    def test_prompt_type(self) -> None:
        hook = Hook(prompt="Summarize: {tool_response}")
        assert hook.hook_type == HookType.PROMPT

    def test_task_type(self) -> None:
        hook = Hook(task="review_output")
        assert hook.hook_type == HookType.TASK

    def test_skill_type(self) -> None:
        hook = Hook(skill="summarize_skill")
        assert hook.hook_type == HookType.SKILL

    def test_tool_type(self) -> None:
        hook = Hook(tool="audit_logger")
        assert hook.hook_type == HookType.TOOL

    def test_all_types_present(self) -> None:
        assert len(HookType) == 6
        expected = {"function", "command", "prompt", "task", "skill", "tool"}
        assert {t.value for t in HookType} == expected


# ══════════════════════════════════════════════════════════════════════════════
# Extended Hook Execution (prompt, task, skill, tool)
# ══════════════════════════════════════════════════════════════════════════════


class TestPromptHook:
    """Prompt hook execution with mocked GenAI service."""

    def test_prompt_resolves_template(self) -> None:
        mock_service = MagicMock()
        mock_service.generate_completion.return_value = "Looks good"

        hook = Hook(prompt="Check: {tool_name} output: {tool_response}", name="validator")

        with patch.object(hook, "_get_genai_service", return_value=mock_service):
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                tool_name="write_file",
                tool_response="File saved",
            )
            result = hook.execute(ctx)

        call_args = mock_service.generate_completion.call_args[0][0]
        assert call_args[0]["content"] == "Check: write_file output: File saved"
        assert result.raw_output == "Looks good"
        assert result.additional_context == "Looks good"

    def test_prompt_returns_structured_json(self) -> None:
        json_response = json.dumps({
            "decision": "block",
            "reason": "Unsafe operation",
        })
        mock_service = MagicMock()
        mock_service.generate_completion.return_value = json_response

        hook = Hook(prompt="Is {tool_name} safe?", name="safety")

        with patch.object(hook, "_get_genai_service", return_value=mock_service):
            ctx = HookContext(tool_name="delete_all")
            result = hook.execute(ctx)

        assert result.block is True
        assert "Unsafe operation" in result.block_reason

    def test_prompt_service_failure(self) -> None:
        hook = Hook(prompt="Test prompt", name="fail_prompt")

        with patch.object(
            hook, "_get_genai_service",
            side_effect=RuntimeError("Service unavailable"),
        ):
            result = hook.execute(HookContext())

        assert result.exit_code == 1
        assert "Service unavailable" in result.raw_error

    def test_prompt_unknown_placeholders_kept(self) -> None:
        mock_service = MagicMock()
        mock_service.generate_completion.return_value = "ok"

        hook = Hook(prompt="Hello {unknown_var}", name="safe_template")

        with patch.object(hook, "_get_genai_service", return_value=mock_service):
            result = hook.execute(HookContext())

        call_args = mock_service.generate_completion.call_args[0][0]
        assert "{unknown_var}" in call_args[0]["content"]


class TestTaskHook:
    """Task hook execution with mocked GenAI service and task file."""

    def test_task_loads_and_executes(self, tmp_path) -> None:
        task_file = tmp_path / "my_task.json"
        task_file.write_text(json.dumps({
            "task_name": "my_task",
            "system_prompt": "You are a reviewer.",
            "user_prompt_template": "Review: {tool_response}",
        }))

        mock_service = MagicMock()
        mock_service.generate_completion.return_value = "LGTM"

        hook = Hook(task="my_task", name="review")

        # Must mock both the TaskExecutor import guard AND the prompts dir
        mock_tasker = MagicMock()
        mock_tasker.TaskExecutor = MagicMock

        with (
            patch.dict("sys.modules", {"nono.tasker.genai_tasker": mock_tasker}),
            patch("nono.config.get_prompts_dir", return_value=str(tmp_path)),
            patch.object(hook, "_get_genai_service", return_value=mock_service),
        ):
            ctx = HookContext(tool_response="File written successfully")
            result = hook.execute(ctx)

        messages = mock_service.generate_completion.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a reviewer."
        assert "File written successfully" in messages[1]["content"]
        assert result.raw_output == "LGTM"

    def test_task_file_not_found(self, tmp_path) -> None:
        hook = Hook(task="nonexistent_task", name="missing")

        mock_tasker = MagicMock()
        mock_tasker.TaskExecutor = MagicMock

        with (
            patch.dict("sys.modules", {"nono.tasker.genai_tasker": mock_tasker}),
            patch("nono.config.get_prompts_dir", return_value=str(tmp_path)),
        ):
            result = hook.execute(HookContext())

        assert result.exit_code == 1
        assert "not found" in result.raw_error


class TestSkillHook:
    """Skill hook execution with mocked skill registry."""

    def test_skill_executes(self) -> None:
        mock_skill_cls = MagicMock()
        mock_skill_instance = MagicMock()
        mock_skill_instance.run.return_value = "Summary of content"
        mock_skill_cls.return_value = mock_skill_instance

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill_cls

        hook = Hook(skill="summarize_skill", name="summarizer")

        with patch("nono.hooks.registry", mock_registry, create=True):
            with patch.dict("sys.modules", {"nono.agent.skill": MagicMock(registry=mock_registry)}):
                ctx = HookContext(agent_response="Long response text")
                result = hook.execute(ctx)

        assert result.raw_output == "Summary of content"
        assert result.additional_context == "Summary of content"

    def test_skill_not_found(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        hook = Hook(skill="nonexistent_skill", name="missing_skill")

        with patch.dict("sys.modules", {"nono.agent.skill": MagicMock(registry=mock_registry)}):
            result = hook.execute(HookContext())

        assert result.exit_code == 1
        assert "not found" in result.raw_error


class TestToolHook:
    """Tool hook execution with mocked tools."""

    def test_tool_executes_from_context(self) -> None:
        mock_tool = MagicMock()
        mock_tool.return_value = "Logged"

        hook = Hook(
            tool="audit_logger",
            tool_args={"action": "call", "target": "{tool_name}"},
            name="audit",
        )

        ctx = HookContext(
            tool_name="write_file",
            extra={"_tools": {"audit_logger": mock_tool}},
        )
        result = hook.execute(ctx)

        mock_tool.assert_called_once_with(action="call", target="write_file")
        assert result.raw_output == "Logged"

    def test_tool_not_found(self) -> None:
        hook = Hook(tool="missing_tool", name="missing")

        with patch.dict("sys.modules", {}):
            ctx = HookContext(extra={"_tools": {}})
            result = hook.execute(ctx)

        assert result.exit_code == 1
        assert "not found" in result.raw_error

    def test_tool_executes_function_tool(self) -> None:
        from nono.agent.tool import FunctionTool

        def log_fn(action: str = "", target: str = "") -> str:
            return f"Logged: {action} on {target}"

        func_tool = FunctionTool(log_fn, description="Log an action")

        hook = Hook(
            tool="logger",
            tool_args={"action": "invoke", "target": "{tool_name}"},
            name="ft_hook",
        )

        ctx = HookContext(
            tool_name="read_file",
            extra={"_tools": {"logger": func_tool}},
        )
        result = hook.execute(ctx)

        assert "Logged: invoke on read_file" in result.raw_output


class TestHookManagerConfigExtended:
    """HookManager.load_config and to_config with extended types."""

    def test_load_all_types(self) -> None:
        config = {
            "hooks": {
                "PostToolUse": [
                    {"type": "command", "command": "echo done"},
                    {"type": "prompt", "prompt": "Check: {tool_response}"},
                    {"type": "task", "task": "review_output"},
                    {"type": "skill", "skill": "summarize_skill"},
                    {"type": "tool", "tool": "log_action", "tool_args": {"msg": "{tool_name}"}},
                ]
            }
        }
        mgr = HookManager()
        loaded = mgr.load_config(config)

        assert loaded == 5
        assert mgr.count(HookEvent.POST_TOOL_USE) == 5

    def test_to_config_roundtrip_all_types(self) -> None:
        mgr = HookManager()
        mgr.register(HookEvent.STOP, Hook(fn=lambda ctx: None, name="fn_hook"))
        mgr.register(HookEvent.STOP, Hook(command="echo done", name="cmd_hook"))
        mgr.register(HookEvent.STOP, Hook(prompt="Test", name="prompt_hook"))
        mgr.register(HookEvent.STOP, Hook(task="my_task", name="task_hook"))
        mgr.register(HookEvent.STOP, Hook(skill="my_skill", name="skill_hook"))
        mgr.register(HookEvent.STOP, Hook(
            tool="my_tool", tool_args={"k": "v"}, name="tool_hook",
        ))

        config = mgr.to_config()
        entries = config["hooks"]["Stop"]

        assert len(entries) == 6
        types = [e["type"] for e in entries]
        assert types == ["function", "command", "prompt", "task", "skill", "tool"]

    def test_load_prompt_with_provider_model(self) -> None:
        config = {
            "hooks": {
                "PreToolUse": [
                    {
                        "type": "prompt",
                        "prompt": "Is {tool_name} safe?",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "matcher": "delete.*",
                    }
                ]
            }
        }
        mgr = HookManager()
        mgr.load_config(config)

        exported = mgr.to_config()
        entry = exported["hooks"]["PreToolUse"][0]

        assert entry["type"] == "prompt"
        assert entry["provider"] == "openai"
        assert entry["model"] == "gpt-4o-mini"
        assert entry["matcher"] == "delete.*"


# ══════════════════════════════════════════════════════════════════════════════
# Integration: BaseAgent + Hooks
# ══════════════════════════════════════════════════════════════════════════════

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)


class StubAgent(BaseAgent):
    """Minimal agent for testing hooks integration."""

    def __init__(self, name: str = "stub", reply: str = "ok", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self._reply = reply

    def _run_impl(self, ctx: InvocationContext):  # type: ignore[override]
        yield Event(EventType.AGENT_MESSAGE, self.name, self._reply)

    async def _run_async_impl(self, ctx: InvocationContext):  # type: ignore[override]
        yield Event(EventType.AGENT_MESSAGE, self.name, self._reply)


class TestBaseAgentHooks:
    """Hooks integration with BaseAgent.run()."""

    def test_pre_agent_run_hook_fires(self) -> None:
        calls: list[str] = []

        def on_pre(ctx: HookContext) -> HookResult:
            calls.append(ctx.agent_name)
            return HookResult()

        mgr = HookManager()
        mgr.register(HookEvent.PRE_AGENT_RUN, Hook(fn=on_pre))

        agent = StubAgent(name="test_agent", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")
        agent.run(ctx)

        assert calls == ["test_agent"]

    def test_pre_agent_run_blocks_execution(self) -> None:
        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_AGENT_RUN,
            Hook(fn=lambda ctx: HookResult(block=True, block_reason="Policy")),
        )

        agent = StubAgent(name="blocked", reply="should not appear", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")
        result = agent.run(ctx)

        assert result == "Policy"

    def test_post_agent_run_hook_fires(self) -> None:
        calls: list[str] = []

        def on_post(ctx: HookContext) -> HookResult:
            calls.append(ctx.agent_response or "")
            return HookResult()

        mgr = HookManager()
        mgr.register(HookEvent.POST_AGENT_RUN, Hook(fn=on_post))

        agent = StubAgent(name="test", reply="hello world", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")
        result = agent.run(ctx)

        assert calls == ["hello world"]
        assert result == "hello world"

    def test_post_agent_run_injects_context(self) -> None:
        mgr = HookManager()
        mgr.register(
            HookEvent.POST_AGENT_RUN,
            Hook(fn=lambda ctx: HookResult(additional_context="Extra info")),
        )

        agent = StubAgent(name="test", reply="base", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")
        result = agent.run(ctx)

        assert "base" in result
        assert "Extra info" in result

    def test_error_hook_fires_on_exception(self) -> None:
        calls: list[str] = []

        class FailAgent(BaseAgent):
            def _run_impl(self, ctx):  # type: ignore[override]
                raise RuntimeError("test error")
                yield  # pragma: no cover

            async def _run_async_impl(self, ctx):  # type: ignore[override]
                raise RuntimeError("test error")
                yield  # pragma: no cover

        mgr = HookManager()
        mgr.register(
            HookEvent.ERROR,
            Hook(fn=lambda ctx: (calls.append(ctx.error or ""), HookResult())[1]),
        )

        agent = FailAgent(name="fail", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")

        with pytest.raises(RuntimeError, match="test error"):
            agent.run(ctx)

        assert calls == ["test error"]

    def test_no_hook_manager_runs_normally(self) -> None:
        agent = StubAgent(name="no_hooks", reply="works")
        ctx = InvocationContext(session=Session(), user_message="hi")
        result = agent.run(ctx)

        assert result == "works"

    def test_set_hook_manager_fluent(self) -> None:
        mgr = HookManager()
        agent = StubAgent(name="fluent")

        assert agent.set_hook_manager(mgr) is agent
        assert agent.hook_manager is mgr


class TestBaseAgentHooksAsync:
    """Hooks integration with BaseAgent.run_async()."""

    def test_pre_agent_run_hook_fires_async(self) -> None:
        calls: list[str] = []

        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_AGENT_RUN,
            Hook(fn=lambda ctx: (calls.append(ctx.agent_name), HookResult())[1]),
        )

        agent = StubAgent(name="async_agent", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")

        import asyncio
        asyncio.run(agent.run_async(ctx))

        assert calls == ["async_agent"]

    def test_pre_agent_run_blocks_async(self) -> None:
        mgr = HookManager()
        mgr.register(
            HookEvent.PRE_AGENT_RUN,
            Hook(fn=lambda ctx: HookResult(block=True, block_reason="Blocked")),
        )

        agent = StubAgent(name="blocked_async", hook_manager=mgr)
        ctx = InvocationContext(session=Session(), user_message="hi")

        import asyncio
        result = asyncio.run(agent.run_async(ctx))

        assert result == "Blocked"


# ══════════════════════════════════════════════════════════════════════════════
# VS Code-compatible context fields
# ══════════════════════════════════════════════════════════════════════════════


class TestHookContextVSCodeFields:
    """HookContext VS Code-compatible camelCase serialization and new fields."""

    def test_camel_case_keys(self) -> None:
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            session_id="s1",
        )
        d = ctx.to_dict()

        assert "sessionId" in d
        assert "hookEventName" in d
        assert "session_id" not in d
        assert "hook_event_name" not in d

    def test_session_start_source(self) -> None:
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            session_id="s1",
            source="new",
        )
        d = ctx.to_dict()

        assert d["source"] == "new"

    def test_pre_tool_use_tool_use_id(self) -> None:
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="editFiles",
            tool_input={"files": ["src/main.ts"]},
            tool_use_id="tool-123",
        )
        d = ctx.to_dict()

        assert d["tool_use_id"] == "tool-123"
        assert d["tool_name"] == "editFiles"

    def test_post_tool_use_includes_response(self) -> None:
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_name="editFiles",
            tool_input={"files": ["src/main.ts"]},
            tool_use_id="tool-123",
            tool_response="File edited successfully",
        )
        d = ctx.to_dict()

        assert d["tool_response"] == "File edited successfully"

    def test_stop_hook_active(self) -> None:
        ctx = HookContext(
            event=HookEvent.STOP,
            stop_hook_active=True,
        )
        d = ctx.to_dict()

        assert d["stop_hook_active"] is True

    def test_stop_hook_active_false_excluded(self) -> None:
        ctx = HookContext(
            event=HookEvent.STOP,
            stop_hook_active=False,
        )
        d = ctx.to_dict()

        assert "stop_hook_active" not in d

    def test_subagent_start_fields(self) -> None:
        ctx = HookContext(
            event=HookEvent.SUBAGENT_START,
            subagent_id="subagent-456",
            subagent_type="Plan",
        )
        d = ctx.to_dict()

        assert d["agent_id"] == "subagent-456"
        assert d["agent_type"] == "Plan"

    def test_subagent_stop_with_stop_hook_active(self) -> None:
        ctx = HookContext(
            event=HookEvent.SUBAGENT_STOP,
            subagent_id="subagent-456",
            subagent_type="Plan",
            stop_hook_active=True,
        )
        d = ctx.to_dict()

        assert d["agent_id"] == "subagent-456"
        assert d["stop_hook_active"] is True

    def test_pre_compact_trigger(self) -> None:
        ctx = HookContext(
            event=HookEvent.PRE_COMPACT,
            trigger="auto",
        )
        d = ctx.to_dict()

        assert d["trigger"] == "auto"

    def test_user_prompt_submit_prompt_field(self) -> None:
        ctx = HookContext(
            event=HookEvent.USER_PROMPT_SUBMIT,
            user_message="Hello, what is the weather?",
        )
        d = ctx.to_dict()

        assert d["prompt"] == "Hello, what is the weather?"
        assert "user_message" not in d

    def test_transcript_path(self) -> None:
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            transcript_path="/path/to/transcript.json",
        )
        d = ctx.to_dict()

        assert d["transcript_path"] == "/path/to/transcript.json"


# ══════════════════════════════════════════════════════════════════════════════
# Auto-discovery
# ══════════════════════════════════════════════════════════════════════════════


class TestHookDiscovery:
    """HookManager.discover and discover_hooks."""

    def test_discover_github_hooks_dir(self, tmp_path) -> None:
        hooks_dir = tmp_path / ".github" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "format.json").write_text(json.dumps({
            "hooks": {
                "PostToolUse": [{"type": "command", "command": "echo fmt"}]
            }
        }))

        mgr = HookManager()
        total = mgr.discover(tmp_path)

        assert total >= 1
        assert mgr.count(HookEvent.POST_TOOL_USE) == 1

    def test_discover_claude_settings(self, tmp_path) -> None:
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{"type": "command", "command": "echo pre"}]
            }
        }))

        mgr = HookManager()
        total = mgr.discover(tmp_path)

        assert total >= 1
        assert mgr.count(HookEvent.PRE_TOOL_USE) == 1

    def test_discover_extra_paths(self, tmp_path) -> None:
        extra_dir = tmp_path / "custom_hooks"
        extra_dir.mkdir()
        (extra_dir / "my.json").write_text(json.dumps({
            "hooks": {
                "Stop": [{"type": "command", "command": "echo stop"}]
            }
        }))

        mgr = HookManager()
        total = mgr.discover(tmp_path, extra_paths=[extra_dir])

        assert total >= 1
        assert mgr.count(HookEvent.STOP) == 1

    def test_discover_empty_workspace(self, tmp_path) -> None:
        mgr = HookManager()
        total = mgr.discover(tmp_path)

        assert total == 0

    def test_discover_hooks_convenience(self, tmp_path) -> None:
        hooks_dir = tmp_path / ".github" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "sec.json").write_text(json.dumps({
            "hooks": {
                "PreToolUse": [{"type": "command", "command": "echo sec"}]
            }
        }))

        mgr = discover_hooks(tmp_path)

        assert isinstance(mgr, HookManager)
        assert mgr.count(HookEvent.PRE_TOOL_USE) == 1

    def test_hook_discovery_paths_constant(self) -> None:
        assert len(HOOK_DISCOVERY_PATHS) >= 5
        assert ".github/hooks" in HOOK_DISCOVERY_PATHS


# ══════════════════════════════════════════════════════════════════════════════
# Agent-scoped hooks
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentScopedHooks:
    """load_agent_scoped_hooks from .agent.md frontmatter."""

    def test_load_from_agent_md(self, tmp_path) -> None:
        agent_file = tmp_path / "formatter.agent.md"
        agent_file.write_text(
            "---\n"
            "name: Strict Formatter\n"
            "description: Agent that auto-formats\n"
            "hooks:\n"
            "  PostToolUse:\n"
            "    - type: command\n"
            "      command: echo format\n"
            "---\n\n"
            "You are a formatting agent.\n"
        )

        mgr = load_agent_scoped_hooks(agent_file)

        assert isinstance(mgr, HookManager)
        assert mgr.count(HookEvent.POST_TOOL_USE) >= 1

    def test_no_hooks_in_frontmatter(self, tmp_path) -> None:
        agent_file = tmp_path / "simple.agent.md"
        agent_file.write_text(
            "---\nname: Simple Agent\n---\n\nJust a plain agent.\n"
        )

        mgr = load_agent_scoped_hooks(agent_file)

        assert mgr.count() == 0

    def test_no_frontmatter(self, tmp_path) -> None:
        agent_file = tmp_path / "bare.agent.md"
        agent_file.write_text("No frontmatter here.\n")

        mgr = load_agent_scoped_hooks(agent_file)

        assert mgr.count() == 0

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_agent_scoped_hooks("/nonexistent/agent.md")
