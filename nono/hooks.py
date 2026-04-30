"""
Lifecycle hooks engine for Nono framework and agents.

Provides a hook system inspired by Claude Code Hooks.  Hooks are
deterministic, code-driven automations that execute
at specific lifecycle points during agent sessions.

Key concepts:
    - **HookEvent**: Enum of lifecycle events (SessionStart, PreToolUse, …).
    - **HookContext**: Structured input passed to every hook execution.
    - **HookResult**: Structured output returned by hooks to influence behaviour.
    - **Hook**: A single hook definition (callable, shell command, prompt,
      task, skill, or tool).
    - **HookType**: Enum of supported hook execution types.
    - **HookManager**: Registry that loads, stores, and fires hooks.

Supported hook types:
    - ``function``: Python callable ``(HookContext) -> HookResult``.
    - ``command``: Shell command executed as a subprocess.
    - ``prompt``: Inline GenAI prompt sent to an LLM provider.
    - ``task``: Predefined JSON task from the ``prompts/`` directory.
    - ``skill``: Registered skill executed via the skill registry.
    - ``tool``: Registered ``FunctionTool`` invoked with context-derived args.

Usage:
    from nono.hooks import HookManager, HookEvent, Hook

    manager = HookManager()

    # Register a Python callable hook
    def on_session_start(ctx):
        print(f"Session {ctx.session_id} started")
        return HookResult()

    manager.register(HookEvent.SESSION_START, Hook(fn=on_session_start))

    # Register a prompt hook
    manager.register(HookEvent.POST_TOOL_USE, Hook(
        prompt="Summarize: {tool_response}",
        provider="google",
        model="gemini-3-flash-preview",
    ))

    # Register a task hook
    manager.register(HookEvent.USER_PROMPT_SUBMIT, Hook(
        task="validate_input",  # matches prompts/validate_input.json
    ))

    # Register a skill hook
    manager.register(HookEvent.POST_AGENT_RUN, Hook(
        skill="summarize_skill",
    ))

    # Register a tool hook
    manager.register(HookEvent.PRE_TOOL_USE, Hook(
        tool="security_scanner",
        tool_args={"target": "{tool_name}"},
    ))

    # Register from a dict / JSON configuration
    manager.load_config({
        "hooks": {
            "PostToolUse": [
                {"type": "command", "command": "echo tool done"},
                {"type": "prompt", "prompt": "Validate output: {tool_response}"},
                {"type": "task", "task": "review_output"},
                {"type": "skill", "skill": "summarize_skill"},
                {"type": "tool", "tool": "log_action"}
            ]
        }
    })

    # Fire hooks
    result = manager.fire(HookEvent.SESSION_START, HookContext(session_id="abc"))
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("Nono.Hooks")

__all__ = [
    "HookEvent",
    "HookType",
    "HookContext",
    "HookResult",
    "Hook",
    "HookManager",
    "load_hooks_from_file",
    "discover_hooks",
    "load_agent_scoped_hooks",
    "HOOK_DISCOVERY_PATHS",
]


# ── Hook Types ───────────────────────────────────────────────────────────────

class HookType(Enum):
    """Supported hook execution types.

    Each value corresponds to a different execution strategy:
    - ``FUNCTION``: Python callable invoked directly.
    - ``COMMAND``: Shell command executed as a subprocess.
    - ``PROMPT``: Inline GenAI prompt sent to an LLM provider.
    - ``TASK``: Predefined JSON task from the ``prompts/`` directory.
    - ``SKILL``: Registered skill from the skill registry.
    - ``TOOL``: Registered ``FunctionTool`` invoked with arguments.
    """

    FUNCTION = "function"
    COMMAND = "command"
    PROMPT = "prompt"
    TASK = "task"
    SKILL = "skill"
    TOOL = "tool"


# ── Hook Events ──────────────────────────────────────────────────────────────

class HookEvent(Enum):
    """Lifecycle events where hooks can fire.

    Mirrors the events defined by Claude Code Hooks, adapted for the Nono
    framework.
    """

    SESSION_START = "SessionStart"
    """Fires when a new agent session begins."""

    SESSION_END = "SessionEnd"
    """Fires when an agent session closes."""

    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    """Fires when the user submits a prompt, before processing."""

    PRE_TOOL_USE = "PreToolUse"
    """Fires before an agent invokes a tool."""

    POST_TOOL_USE = "PostToolUse"
    """Fires after a tool completes."""

    PRE_AGENT_RUN = "PreAgentRun"
    """Fires before an agent's core execution logic runs."""

    POST_AGENT_RUN = "PostAgentRun"
    """Fires after an agent finishes execution."""

    PRE_LLM_CALL = "PreLLMCall"
    """Fires before an LLM API call is made."""

    POST_LLM_CALL = "PostLLMCall"
    """Fires after an LLM API call returns."""

    PRE_COMPACT = "PreCompact"
    """Fires before conversation context is compacted / pruned."""

    SUBAGENT_START = "SubagentStart"
    """Fires when a sub-agent is spawned."""

    SUBAGENT_STOP = "SubagentStop"
    """Fires when a sub-agent completes."""

    STOP = "Stop"
    """Fires when the agent session ends (final response)."""

    NOTIFICATION = "Notification"
    """Fires when the system generates a notification."""

    ERROR = "Error"
    """Fires when an error occurs during execution."""

    @classmethod
    def from_string(cls, value: str) -> HookEvent:
        """Parse a hook event from its string name.

        Accepts PascalCase (``"PreToolUse"``), UPPER_SNAKE_CASE
        (``"PRE_TOOL_USE"``), or lower-snake (``"pre_tool_use"``).

        Args:
            value: Event name string.

        Returns:
            The matching ``HookEvent``.

        Raises:
            ValueError: If no match is found.
        """
        # Try by value first (PascalCase)
        for member in cls:
            if member.value == value:
                return member

        # Try by name (UPPER_SNAKE_CASE)
        upper = value.upper().replace("-", "_")
        try:
            return cls[upper]
        except KeyError:
            pass

        # Try converting camelCase/PascalCase to UPPER_SNAKE_CASE
        snake = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", value).upper()
        try:
            return cls[snake]
        except KeyError:
            pass

        raise ValueError(
            f"Unknown hook event {value!r}. "
            f"Available: {[e.value for e in cls]}"
        )


# ── Hook Context (input) ────────────────────────────────────────────────────

@dataclass
class HookContext:
    """Structured input passed to hooks when they fire.

    Different events populate different fields.  Hooks should check for
    ``None`` before accessing event-specific data.

    Serialized field names follow VS Code camelCase convention for
    compatibility with external shell hooks (``sessionId``,
    ``hookEventName``, ``toolName``, etc.).

    Args:
        event: The hook event being fired.
        session_id: Current session identifier.
        agent_name: Name of the agent triggering the hook.
        cwd: Working directory.
        timestamp: ISO-8601 timestamp (auto-set).
        transcript_path: Path to the session transcript file.
        user_message: User prompt text (for ``UserPromptSubmit``).
        tool_name: Tool name (for ``PreToolUse`` / ``PostToolUse``).
        tool_input: Tool arguments (for ``PreToolUse`` / ``PostToolUse``).
        tool_use_id: Unique ID for the tool invocation.
        tool_response: Tool return value (for ``PostToolUse``).
        tool_error: Tool error string (for ``PostToolUse`` on failure).
        llm_messages: Messages sent to the LLM (for ``PreLLMCall``).
        llm_response: LLM response text (for ``PostLLMCall``).
        agent_response: Final agent response (for ``PostAgentRun``).
        subagent_name: Sub-agent name (for ``SubagentStart/Stop``).
        subagent_id: Unique ID of the sub-agent (for ``SubagentStart/Stop``).
        subagent_type: Type/name of the sub-agent (for ``SubagentStart/Stop``).
        source: How the session was started (for ``SessionStart``).
        trigger: How compaction was triggered (for ``PreCompact``).
        stop_hook_active: ``True`` when continuing from a previous stop hook
            (for ``Stop`` / ``SubagentStop``). Check to prevent infinite loops.
        error: Error details (for ``Error`` event).
        extra: Arbitrary extra data for custom hooks.
    """

    event: HookEvent | None = None
    session_id: str = ""
    agent_name: str = ""
    cwd: str = field(default_factory=os.getcwd)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    transcript_path: str = ""
    user_message: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_use_id: str = ""
    tool_response: Any = None
    tool_error: str | None = None
    llm_messages: list[dict[str, str]] | None = None
    llm_response: str | None = None
    agent_response: str | None = None
    subagent_name: str | None = None
    subagent_id: str = ""
    subagent_type: str = ""
    source: str = ""
    trigger: str = ""
    stop_hook_active: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context to a dict (for JSON stdin to shell commands).

        Field names use camelCase to match the VS Code / Claude Code
        hook protocol (``sessionId``, ``hookEventName``, ``toolName``, …).

        Returns:
            Dictionary with all non-None fields.
        """
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "cwd": self.cwd,
            "sessionId": self.session_id,
            "hookEventName": self.event.value if self.event else "",
        }

        if self.transcript_path:
            result["transcript_path"] = self.transcript_path

        # -- UserPromptSubmit --
        if self.user_message is not None:
            result["prompt"] = self.user_message

        # -- PreToolUse / PostToolUse --
        if self.tool_name is not None:
            result["tool_name"] = self.tool_name
        if self.tool_input is not None:
            result["tool_input"] = self.tool_input
        if self.tool_use_id:
            result["tool_use_id"] = self.tool_use_id
        if self.tool_response is not None:
            result["tool_response"] = (
                self.tool_response
                if isinstance(self.tool_response, (str, dict, list, int, float, bool))
                else str(self.tool_response)
            )
        if self.tool_error is not None:
            result["tool_error"] = self.tool_error

        # -- SessionStart --
        if self.source:
            result["source"] = self.source

        # -- Stop / SubagentStop --
        if self.stop_hook_active:
            result["stop_hook_active"] = self.stop_hook_active

        # -- SubagentStart / SubagentStop --
        if self.subagent_id:
            result["agent_id"] = self.subagent_id
        if self.subagent_type:
            result["agent_type"] = self.subagent_type
        if self.subagent_name is not None:
            result["subagent_name"] = self.subagent_name

        # -- PreCompact --
        if self.trigger:
            result["trigger"] = self.trigger

        # -- Nono-specific extensions --
        if self.llm_response is not None:
            result["llm_response"] = self.llm_response
        if self.agent_response is not None:
            result["agent_response"] = self.agent_response
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.error is not None:
            result["error"] = self.error
        if self.extra:
            result["extra"] = self.extra

        return result

    def to_json(self) -> str:
        """Serialize context to JSON string.

        Returns:
            JSON representation of the context.
        """
        return json.dumps(self.to_dict(), default=str)


# ── Hook Result (output) ────────────────────────────────────────────────────

@dataclass
class HookResult:
    """Structured output returned by a hook to influence agent behaviour.

    Args:
        continue_execution: ``False`` to stop the agent session entirely.
        stop_reason: Reason for stopping (shown to user).
        block: ``True`` to block the current operation (e.g. tool call).
        block_reason: Reason for blocking (shown to model).
        system_message: Warning/info message displayed to the user.
        additional_context: Extra context injected into the conversation.
        updated_input: Modified tool input (for ``PreToolUse``).
        permission_decision: ``"allow"``, ``"deny"``, or ``"ask"`` (PreToolUse).
        exit_code: Shell hook exit code (0=success, 2=blocking, other=warning).
        raw_output: Raw stdout from shell hook execution.
        raw_error: Raw stderr from shell hook execution.
    """

    continue_execution: bool = True
    stop_reason: str = ""
    block: bool = False
    block_reason: str = ""
    system_message: str = ""
    additional_context: str = ""
    updated_input: dict[str, Any] | None = None
    permission_decision: str = ""  # "allow", "deny", "ask"
    exit_code: int = 0
    raw_output: str = ""
    raw_error: str = ""

    @property
    def should_block(self) -> bool:
        """Whether this result blocks the current operation.

        Returns:
            ``True`` if the hook explicitly blocks or exit code is 2.
        """
        return self.block or self.exit_code == 2 or self.permission_decision == "deny"

    @property
    def should_stop(self) -> bool:
        """Whether this result stops the entire session.

        Returns:
            ``True`` if ``continue_execution`` is ``False``.
        """
        return not self.continue_execution

    @classmethod
    def from_json(cls, raw: str) -> HookResult:
        """Parse a HookResult from JSON output (stdout of a shell hook).

        Args:
            raw: Raw JSON string.

        Returns:
            Parsed ``HookResult``.
        """
        if not raw.strip():
            return cls()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls(raw_output=raw)

        hook_specific = data.get("hookSpecificOutput", {})

        return cls(
            continue_execution=data.get("continue", True),
            stop_reason=data.get("stopReason", ""),
            block=data.get("decision") == "block"
                  or hook_specific.get("permissionDecision") == "deny",
            block_reason=(
                data.get("reason", "")
                or hook_specific.get("permissionDecisionReason", "")
            ),
            system_message=data.get("systemMessage", ""),
            additional_context=hook_specific.get("additionalContext", ""),
            updated_input=hook_specific.get("updatedInput"),
            permission_decision=hook_specific.get("permissionDecision", ""),
            raw_output=raw,
        )

    def merge(self, other: HookResult) -> HookResult:
        """Merge another result into this one (most restrictive wins).

        Args:
            other: The other ``HookResult`` to merge.

        Returns:
            A new ``HookResult`` combining both.
        """
        # Permission priority: deny > ask > allow
        _perm_priority = {"deny": 3, "ask": 2, "allow": 1, "": 0}
        p1 = _perm_priority.get(self.permission_decision, 0)
        p2 = _perm_priority.get(other.permission_decision, 0)
        merged_perm = (
            self.permission_decision if p1 >= p2
            else other.permission_decision
        )

        return HookResult(
            continue_execution=self.continue_execution and other.continue_execution,
            stop_reason=self.stop_reason or other.stop_reason,
            block=self.block or other.block,
            block_reason=self.block_reason or other.block_reason,
            system_message="\n".join(
                m for m in [self.system_message, other.system_message] if m
            ),
            additional_context="\n".join(
                c for c in [self.additional_context, other.additional_context] if c
            ),
            updated_input=other.updated_input or self.updated_input,
            permission_decision=merged_perm,
            exit_code=max(self.exit_code, other.exit_code),
        )


# ── Hook Definition ──────────────────────────────────────────────────────────

# Maximum time (seconds) for a shell hook to execute
_DEFAULT_HOOK_TIMEOUT: int = 30


@dataclass
class Hook:
    """A single hook definition — callable, command, prompt, task, skill, or tool.

    Supply exactly one of ``fn``, ``command``, ``prompt``, ``task``,
    ``skill``, or ``tool``.

    Args:
        fn: Python callable ``(HookContext) -> HookResult | None``.
        command: Shell command string to execute.
        command_windows: Windows-specific command override.
        command_linux: Linux-specific command override.
        command_osx: macOS-specific command override.
        prompt: Inline GenAI prompt template. Supports ``{variable}``
            placeholders resolved from ``HookContext`` fields.
        task: Name of a JSON task definition in the ``prompts/`` directory
            (without ``.json`` extension).
        skill: Name of a registered skill in the skill registry.
        tool: Name of a registered ``FunctionTool`` to invoke.
        tool_args: Keyword arguments for tool invocation. Values may contain
            ``{variable}`` placeholders resolved from ``HookContext``.
        provider: AI provider name for ``prompt`` / ``task`` hooks
            (e.g. ``"google"``, ``"openai"``). Defaults to config.
        model: Model name for ``prompt`` / ``task`` hooks. Defaults to config.
        matcher: Regex pattern to match tool names (for Pre/PostToolUse).
        timeout: Execution timeout in seconds.
        name: Optional descriptive name for logging.
        enabled: Whether the hook is active.
        env: Additional environment variables for shell execution.
        cwd: Working directory for shell execution.
    """

    fn: Callable[[HookContext], HookResult | None] | None = None
    command: str = ""
    command_windows: str = ""
    command_linux: str = ""
    command_osx: str = ""
    prompt: str = ""
    task: str = ""
    skill: str = ""
    tool: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    matcher: str = ""
    timeout: int = _DEFAULT_HOOK_TIMEOUT
    name: str = ""
    enabled: bool = True
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""

    def __post_init__(self) -> None:
        has_fn = self.fn is not None
        has_cmd = bool(self.command or self.command_windows
                       or self.command_linux or self.command_osx)
        has_prompt = bool(self.prompt)
        has_task = bool(self.task)
        has_skill = bool(self.skill)
        has_tool = bool(self.tool)

        sources = sum([has_fn, has_cmd, has_prompt, has_task, has_skill, has_tool])

        if sources == 0:
            raise ValueError(
                "Hook must have one of 'fn', 'command', 'prompt', "
                "'task', 'skill', or 'tool' set."
            )

        if self.matcher:
            self._matcher_re: re.Pattern[str] | None = re.compile(self.matcher)
        else:
            self._matcher_re = None

    @property
    def hook_type(self) -> HookType:
        """Determine the hook execution type.

        Returns:
            The ``HookType`` for this hook.
        """
        if self.fn is not None:
            return HookType.FUNCTION
        if self.prompt:
            return HookType.PROMPT
        if self.task:
            return HookType.TASK
        if self.skill:
            return HookType.SKILL
        if self.tool:
            return HookType.TOOL
        return HookType.COMMAND

    def matches(self, tool_name: str) -> bool:
        """Check if this hook's matcher matches a tool name.

        Args:
            tool_name: The tool name to check.

        Returns:
            ``True`` if the matcher matches or no matcher is set.
        """
        if not self._matcher_re:
            return True
        return bool(self._matcher_re.search(tool_name))

    def _resolve_command(self) -> str:
        """Resolve the OS-specific command to execute.

        Returns:
            The command string for the current platform.
        """
        import platform
        system = platform.system().lower()

        if system == "windows" and self.command_windows:
            return self.command_windows
        if system == "linux" and self.command_linux:
            return self.command_linux
        if system == "darwin" and self.command_osx:
            return self.command_osx

        return self.command

    def execute(self, ctx: HookContext) -> HookResult:
        """Execute this hook with the given context.

        Dispatches to the appropriate execution strategy based on hook type:
        - ``function``: Invokes the Python callable directly.
        - ``command``: Runs a shell command with context on stdin.
        - ``prompt``: Sends an inline prompt to a GenAI provider.
        - ``task``: Executes a predefined JSON task via the tasker.
        - ``skill``: Runs a registered skill from the skill registry.
        - ``tool``: Invokes a registered FunctionTool.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` from the hook execution.
        """
        if not self.enabled:
            return HookResult()

        hook_type = self.hook_type

        if hook_type == HookType.FUNCTION:
            return self._execute_fn(ctx)
        if hook_type == HookType.COMMAND:
            return self._execute_command(ctx)
        if hook_type == HookType.PROMPT:
            return self._execute_prompt(ctx)
        if hook_type == HookType.TASK:
            return self._execute_task(ctx)
        if hook_type == HookType.SKILL:
            return self._execute_skill(ctx)
        if hook_type == HookType.TOOL:
            return self._execute_tool(ctx)

        return HookResult(
            exit_code=1,
            raw_error=f"Unknown hook type: {hook_type}",
        )

    def _execute_fn(self, ctx: HookContext) -> HookResult:
        """Execute a Python callable hook.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` from the callable.
        """
        try:
            result = self.fn(ctx)  # type: ignore[misc]

            if result is None:
                return HookResult()

            if isinstance(result, HookResult):
                return result

            return HookResult(raw_output=str(result))

        except Exception as e:
            logger.error("Hook %r failed: %s", self.name or "unnamed", e)
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Hook error: {e}",
            )

    def _execute_command(self, ctx: HookContext) -> HookResult:
        """Execute a shell command hook.

        Sends context as JSON on stdin. Parses stdout as JSON for the result.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` from the command execution.
        """
        cmd = self._resolve_command()

        if not cmd:
            logger.warning("Hook %r has no command for this platform.", self.name)
            return HookResult()

        # Substitute environment variable placeholders
        env = {**os.environ, **self.env}
        if ctx.tool_name:
            env["TOOL_NAME"] = ctx.tool_name
        if ctx.tool_input:
            file_path = ctx.tool_input.get("file_path") or ctx.tool_input.get("filePath", "")
            env["TOOL_INPUT_FILE_PATH"] = str(file_path)
        env["HOOK_EVENT_NAME"] = ctx.event.value if ctx.event else ""
        env["SESSION_ID"] = ctx.session_id
        env["AGENT_NAME"] = ctx.agent_name

        stdin_data = ctx.to_json()
        work_dir = self.cwd or ctx.cwd or None

        try:
            import platform
            shell = platform.system().lower() == "windows"

            proc = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=work_dir,
                env=env,
                shell=shell,
            )

            result = HookResult(
                exit_code=proc.returncode,
                raw_output=proc.stdout,
                raw_error=proc.stderr,
            )

            # Parse stdout as JSON if exit code is 0
            if proc.returncode == 0 and proc.stdout.strip():
                parsed = HookResult.from_json(proc.stdout)
                parsed.exit_code = 0
                parsed.raw_output = proc.stdout
                parsed.raw_error = proc.stderr
                return parsed

            # Exit code 2 = blocking error
            if proc.returncode == 2:
                result.block = True
                result.block_reason = proc.stderr.strip() or "Blocked by hook"
                return result

            # Other non-zero = warning
            if proc.returncode != 0:
                result.system_message = proc.stderr.strip() or f"Hook exited with code {proc.returncode}"

            return result

        except subprocess.TimeoutExpired:
            logger.error(
                "Hook %r timed out after %ds",
                self.name or cmd[:50], self.timeout,
            )
            return HookResult(
                exit_code=1,
                raw_error=f"Hook timed out after {self.timeout}s",
                system_message=f"Hook timed out after {self.timeout}s",
            )
        except FileNotFoundError as e:
            logger.error("Hook command not found: %s", e)
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Hook command not found: {e}",
            )
        except Exception as e:
            logger.error("Hook %r execution failed: %s", self.name or cmd[:50], e)
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Hook execution error: {e}",
            )

    # ── GenAI hook executors ─────────────────────────────────────────────

    def _resolve_template(self, template: str, ctx: HookContext) -> str:
        """Resolve ``{variable}`` placeholders in a template string.

        Substitutes placeholders with values from ``HookContext`` fields
        and ``ctx.extra``.  Unknown placeholders are left unchanged.

        Args:
            template: String with ``{field_name}`` placeholders.
            ctx: The hook context providing values.

        Returns:
            The resolved string.
        """
        ctx_dict = ctx.to_dict()
        # Flatten extra into top-level for convenience
        extras = ctx_dict.pop("extra", {})
        ctx_dict.update(extras)

        # Safe format: only replace known keys
        try:
            return template.format_map(_SafeFormatDict(ctx_dict))
        except Exception:
            return template

    def _get_genai_service(self) -> Any:
        """Create a GenAI service instance for prompt / task execution.

        Uses ``self.provider`` and ``self.model`` if set, otherwise falls
        back to the default provider and model from ``nono.config``.

        Returns:
            A ``GenerativeAIService`` instance.

        Raises:
            RuntimeError: If the connector module is not available.
        """
        try:
            from nono.connector.connector_genai import create_service
        except ImportError:
            raise RuntimeError(
                "nono.connector.connector_genai is required for "
                "prompt/task hooks. Install the nono package with "
                "connector support."
            )

        provider = self.provider
        model = self.model

        if not provider or not model:
            try:
                from nono.config import NonoConfig
                cfg = NonoConfig()
                provider = provider or cfg.get("provider", "google")
                model = model or cfg.get("model", "gemini-3-flash-preview")
            except Exception:
                provider = provider or "google"
                model = model or "gemini-3-flash-preview"

        return create_service(provider=provider, model_name=model)

    def _execute_prompt(self, ctx: HookContext) -> HookResult:
        """Execute an inline GenAI prompt hook.

        Resolves ``{variable}`` placeholders in the prompt template, sends
        it to the configured LLM provider, and wraps the response in a
        ``HookResult``.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` with the LLM response as ``raw_output`` and
            ``additional_context``.
        """
        resolved_prompt = self._resolve_template(self.prompt, ctx)

        try:
            service = self._get_genai_service()
            messages = [{"role": "user", "content": resolved_prompt}]
            response = service.generate_completion(messages)

            return _parse_genai_response(response, self.name or "prompt_hook")

        except Exception as e:
            logger.error(
                "Prompt hook %r failed: %s", self.name or "unnamed", e,
            )
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Prompt hook error: {e}",
            )

    def _execute_task(self, ctx: HookContext) -> HookResult:
        """Execute a predefined JSON task hook.

        Loads the task definition from the ``prompts/`` directory, resolves
        template variables from the hook context, and runs the task via
        the tasker framework.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` with the task output.
        """
        try:
            from nono.tasker.genai_tasker import TaskExecutor
        except ImportError:
            try:
                from nono.tasker.genai_tasker import BaseAIClient as TaskExecutor
            except ImportError:
                return HookResult(
                    exit_code=1,
                    raw_error="TaskExecutor not available",
                    system_message="Task hook requires nono.tasker",
                )

        try:
            from nono.config import get_prompts_dir
            prompts_dir = Path(get_prompts_dir())
        except Exception:
            prompts_dir = Path(__file__).parent / "tasker" / "prompts"

        task_file = prompts_dir / f"{self.task}.json"

        if not task_file.exists():
            return HookResult(
                exit_code=1,
                raw_error=f"Task file not found: {task_file}",
                system_message=f"Task '{self.task}' not found in prompts directory",
            )

        try:
            with open(task_file, encoding="utf-8") as f:
                task_def = json.load(f)

            # Resolve template variables in user_prompt_template
            user_template = task_def.get("user_prompt_template", "")
            resolved_prompt = self._resolve_template(user_template, ctx)
            system_prompt = task_def.get("system_prompt", "")

            service = self._get_genai_service()
            messages: list[dict[str, str]] = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": resolved_prompt})

            response = service.generate_completion(messages)

            return _parse_genai_response(response, self.name or f"task:{self.task}")

        except Exception as e:
            logger.error(
                "Task hook %r (%s) failed: %s",
                self.name or "unnamed", self.task, e,
            )
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Task hook error: {e}",
            )

    def _execute_skill(self, ctx: HookContext) -> HookResult:
        """Execute a registered skill hook.

        Looks up the skill in the global skill registry and invokes it
        with a message derived from the hook context.

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` with the skill output.
        """
        try:
            from nono.agent.skill import registry
        except ImportError:
            return HookResult(
                exit_code=1,
                raw_error="Skill registry not available",
                system_message="Skill hook requires nono.agent.skill",
            )

        skill_cls = registry.get(self.skill)

        if skill_cls is None:
            return HookResult(
                exit_code=1,
                raw_error=f"Skill '{self.skill}' not found in registry",
                system_message=f"Skill '{self.skill}' not registered",
            )

        try:
            skill_instance = skill_cls()

            # Build input message from context
            input_message = _build_skill_input(ctx)
            result_text = skill_instance.run(input_message)

            return HookResult(
                raw_output=str(result_text),
                additional_context=str(result_text),
            )

        except Exception as e:
            logger.error(
                "Skill hook %r (%s) failed: %s",
                self.name or "unnamed", self.skill, e,
            )
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Skill hook error: {e}",
            )

    def _execute_tool(self, ctx: HookContext) -> HookResult:
        """Execute a registered FunctionTool hook.

        Looks up the tool by name in the agent tool registry and invokes
        it with the configured ``tool_args`` (with placeholder resolution).

        Args:
            ctx: The hook context.

        Returns:
            ``HookResult`` with the tool output.
        """
        try:
            from nono.agent.tool import FunctionTool
        except ImportError:
            return HookResult(
                exit_code=1,
                raw_error="FunctionTool not available",
                system_message="Tool hook requires nono.agent.tool",
            )

        # Try to find tool in context extra or resolve from global registry
        tool_fn = ctx.extra.get("_tools", {}).get(self.tool)

        if tool_fn is None:
            try:
                from nono.agent.tool import tool_registry
                tool_fn = tool_registry.get(self.tool)
            except (ImportError, AttributeError):
                pass

        if tool_fn is None:
            return HookResult(
                exit_code=1,
                raw_error=f"Tool '{self.tool}' not found",
                system_message=f"Tool '{self.tool}' not registered or available",
            )

        try:
            # Resolve placeholders in tool_args
            resolved_args: dict[str, Any] = {}

            for key, value in self.tool_args.items():
                if isinstance(value, str):
                    resolved_args[key] = self._resolve_template(value, ctx)
                else:
                    resolved_args[key] = value

            # Invoke the tool
            if isinstance(tool_fn, FunctionTool):
                result_text = tool_fn(**resolved_args)
            elif callable(tool_fn):
                result_text = tool_fn(**resolved_args)
            else:
                return HookResult(
                    exit_code=1,
                    raw_error=f"Tool '{self.tool}' is not callable",
                )

            return HookResult(
                raw_output=str(result_text),
                additional_context=str(result_text),
            )

        except Exception as e:
            logger.error(
                "Tool hook %r (%s) failed: %s",
                self.name or "unnamed", self.tool, e,
            )
            return HookResult(
                exit_code=1,
                raw_error=str(e),
                system_message=f"Tool hook error: {e}",
            )


# ── Hook Manager ─────────────────────────────────────────────────────────────

class HookManager:
    """Registry that loads, stores, and fires hooks.

    Thread-safe.  Supports Python callable hooks, shell command hooks,
    GenAI prompt hooks, task hooks, skill hooks, and tool hooks — all
    loadable from JSON configuration files.

    Args:
        hooks: Optional initial mapping of events to hook lists.

    Example:
        >>> manager = HookManager()
        >>> manager.register(HookEvent.SESSION_START, Hook(
        ...     fn=lambda ctx: print(f"Session {ctx.session_id} started"),
        ... ))
        >>> manager.fire(HookEvent.SESSION_START, HookContext(session_id="abc"))
    """

    def __init__(
        self,
        hooks: dict[HookEvent, list[Hook]] | None = None,
    ) -> None:
        self._hooks: dict[HookEvent, list[Hook]] = hooks or {}
        self._lock = threading.Lock()

    @property
    def events(self) -> list[HookEvent]:
        """List all events that have registered hooks.

        Returns:
            List of ``HookEvent`` values with at least one hook.
        """
        with self._lock:
            return [e for e, h in self._hooks.items() if h]

    def count(self, event: HookEvent | None = None) -> int:
        """Count registered hooks.

        Args:
            event: Count only hooks for this event, or all if ``None``.

        Returns:
            Number of registered hooks.
        """
        with self._lock:
            if event is not None:
                return len(self._hooks.get(event, []))
            return sum(len(h) for h in self._hooks.values())

    def register(self, event: HookEvent, hook: Hook) -> HookManager:
        """Register a hook for a lifecycle event.

        Args:
            event: The event to hook into.
            hook: The hook definition.

        Returns:
            ``self`` for fluent chaining.
        """
        with self._lock:
            self._hooks.setdefault(event, []).append(hook)

        logger.debug(
            "Registered hook %r for %s",
            hook.name or "unnamed", event.value,
        )
        return self

    def unregister(self, event: HookEvent, hook: Hook) -> bool:
        """Remove a specific hook.

        Args:
            event: The event the hook is registered for.
            hook: The hook to remove.

        Returns:
            ``True`` if the hook was found and removed.
        """
        with self._lock:
            hooks = self._hooks.get(event, [])
            try:
                hooks.remove(hook)
                return True
            except ValueError:
                return False

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all hooks, or all hooks for a specific event.

        Args:
            event: Clear only hooks for this event, or all if ``None``.
        """
        with self._lock:
            if event is not None:
                self._hooks.pop(event, None)
            else:
                self._hooks.clear()

    def fire(
        self,
        event: HookEvent,
        ctx: HookContext,
        *,
        tool_name: str = "",
    ) -> HookResult:
        """Fire all hooks registered for an event.

        Hooks run sequentially.  Results are merged (most restrictive wins).
        For ``PreToolUse`` / ``PostToolUse`` events, only hooks whose
        ``matcher`` matches the ``tool_name`` are executed.

        Args:
            event: The event to fire.
            ctx: The hook context with event-specific data.
            tool_name: Tool name for matcher filtering (Pre/PostToolUse).

        Returns:
            Merged ``HookResult`` from all executed hooks.
        """
        ctx.event = event

        with self._lock:
            hooks = list(self._hooks.get(event, []))

        if not hooks:
            return HookResult()

        merged = HookResult()

        for hook in hooks:
            if not hook.enabled:
                continue

            # Matcher filtering for tool-related events
            if event in (HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE):
                if tool_name and not hook.matches(tool_name):
                    continue

            logger.debug(
                "Firing hook %r for %s",
                hook.name or "unnamed", event.value,
            )

            _start = time.perf_counter_ns()
            result = hook.execute(ctx)
            _elapsed_ms = (time.perf_counter_ns() - _start) / 1_000_000

            logger.debug(
                "Hook %r completed in %.1fms (exit=%d, block=%s)",
                hook.name or "unnamed",
                _elapsed_ms,
                result.exit_code,
                result.should_block,
            )

            merged = merged.merge(result)

            # Stop early if a hook blocks or stops execution
            if merged.should_stop:
                logger.info(
                    "Hook stopped session: %s",
                    merged.stop_reason or "no reason given",
                )
                break

            if merged.should_block and event == HookEvent.PRE_TOOL_USE:
                logger.info(
                    "Hook blocked tool %r: %s",
                    tool_name, merged.block_reason or "no reason given",
                )
                break

        return merged

    def load_config(self, config: dict[str, Any]) -> int:
        """Load hooks from a configuration dict (nested / flat format).

        Supports the standard format::

            {
                "hooks": {
                    "PreToolUse": [
                        {"type": "command", "command": "...", "matcher": "..."}
                    ],
                    "PostToolUse": [
                        {"type": "command", "command": "..."}
                    ]
                }
            }

        Args:
            config: Configuration dict with a ``"hooks"`` key.

        Returns:
            Number of hooks loaded.
        """
        hooks_section = config.get("hooks", config)
        loaded = 0

        for event_name, hook_list in hooks_section.items():
            if event_name == "hooks":
                continue

            try:
                event = HookEvent.from_string(event_name)
            except ValueError:
                logger.warning("Unknown hook event %r, skipping.", event_name)
                continue

            if not isinstance(hook_list, list):
                logger.warning(
                    "Hook event %r value is not a list, skipping.",
                    event_name,
                )
                continue

            for entry in hook_list:
                hook = _parse_hook_entry(entry)
                if hook:
                    self.register(event, hook)
                    loaded += 1

        logger.info("Loaded %d hooks from configuration.", loaded)
        return loaded

    def load_file(self, path: str | Path) -> int:
        """Load hooks from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Number of hooks loaded.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Hook file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config = json.load(f)

        loaded = self.load_config(config)
        logger.info("Loaded %d hooks from %s", loaded, path)
        return loaded

    def load_directory(self, directory: str | Path) -> int:
        """Load all ``*.json`` hook files from a directory.

        Args:
            directory: Path to the directory to scan.

        Returns:
            Total number of hooks loaded.
        """
        directory = Path(directory)

        if not directory.is_dir():
            return 0

        total = 0

        for json_file in sorted(directory.glob("*.json")):
            try:
                total += self.load_file(json_file)
            except Exception as e:
                logger.warning("Failed to load hooks from %s: %s", json_file, e)

        return total

    def to_config(self) -> dict[str, Any]:
        """Serialize all hooks to a configuration dict.

        Returns:
            Configuration dict in the standard hook format.
        """
        config: dict[str, list[dict[str, Any]]] = {}

        with self._lock:
            for event, hooks in self._hooks.items():
                entries: list[dict[str, Any]] = []

                for hook in hooks:
                    entry: dict[str, Any] = {"type": hook.hook_type.value}

                    if hook.fn:
                        entry["name"] = hook.name or hook.fn.__name__
                    elif hook.prompt:
                        entry["prompt"] = hook.prompt
                    elif hook.task:
                        entry["task"] = hook.task
                    elif hook.skill:
                        entry["skill"] = hook.skill
                    elif hook.tool:
                        entry["tool"] = hook.tool
                        if hook.tool_args:
                            entry["tool_args"] = hook.tool_args
                    else:
                        entry["command"] = hook.command
                        if hook.command_windows:
                            entry["windows"] = hook.command_windows
                        if hook.command_linux:
                            entry["linux"] = hook.command_linux
                        if hook.command_osx:
                            entry["osx"] = hook.command_osx

                    if hook.provider:
                        entry["provider"] = hook.provider
                    if hook.model:
                        entry["model"] = hook.model
                    if hook.matcher:
                        entry["matcher"] = hook.matcher
                    if hook.timeout != _DEFAULT_HOOK_TIMEOUT:
                        entry["timeout"] = hook.timeout
                    if hook.env:
                        entry["env"] = hook.env
                    if hook.cwd:
                        entry["cwd"] = hook.cwd
                    if not hook.enabled:
                        entry["enabled"] = False
                    if hook.name and "name" not in entry:
                        entry["name"] = hook.name

                    entries.append(entry)

                if entries:
                    config[event.value] = entries

        return {"hooks": config}

    def __repr__(self) -> str:
        with self._lock:
            counts = {
                e.value: len(h) for e, h in self._hooks.items() if h
            }
        return f"HookManager({counts})"

    def discover(
        self,
        workspace: str | Path | None = None,
        *,
        extra_paths: list[str | Path] | None = None,
    ) -> int:
        """Auto-discover and load hooks from standard locations.

        Searches the locations defined by VS Code and Claude Code:
        1. ``<workspace>/.github/hooks/*.json``
        2. ``<workspace>/.claude/settings.json``
        3. ``<workspace>/.claude/settings.local.json``
        4. ``~/.copilot/hooks/*.json``
        5. ``~/.claude/settings.json``

        Workspace hooks take precedence over user hooks for the same
        event type (loaded first, so they appear earlier in the list).

        Args:
            workspace: Workspace root directory.  Defaults to ``cwd``.
            extra_paths: Additional file or directory paths to load.

        Returns:
            Total number of hooks loaded.
        """
        workspace = Path(workspace) if workspace else Path.cwd()
        home = Path.home()
        total = 0

        # Workspace-level paths (higher precedence, loaded first)
        ws_paths: list[Path] = [
            workspace / ".github" / "hooks",
            workspace / ".claude" / "settings.json",
            workspace / ".claude" / "settings.local.json",
        ]

        # User-level paths
        user_paths: list[Path] = [
            home / ".copilot" / "hooks",
            home / ".claude" / "settings.json",
        ]

        for p in ws_paths + user_paths:
            try:
                if p.is_dir():
                    total += self.load_directory(p)
                elif p.is_file() and p.suffix == ".json":
                    total += self.load_file(p)
            except Exception as e:
                logger.warning("Failed to load hooks from %s: %s", p, e)

        # Extra paths from caller
        for ep in extra_paths or []:
            ep = Path(ep)
            try:
                if ep.is_dir():
                    total += self.load_directory(ep)
                elif ep.is_file():
                    total += self.load_file(ep)
            except Exception as e:
                logger.warning("Failed to load hooks from %s: %s", ep, e)

        logger.info("Auto-discovery loaded %d hooks total.", total)
        return total


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_hook_entry(entry: dict[str, Any]) -> Hook | None:
    """Parse a single hook entry from a configuration dict.

    Supports Claude Code format (nested ``hooks`` array), flat
    format, and extended Nono types (prompt, task, skill, tool).

    Args:
        entry: Hook entry dict.

    Returns:
        Parsed ``Hook``, or ``None`` on error.
    """
    # Common fields shared by all hook types
    common = {
        "matcher": entry.get("matcher", ""),
        "timeout": entry.get("timeout", _DEFAULT_HOOK_TIMEOUT),
        "name": entry.get("name", ""),
        "enabled": entry.get("enabled", True),
        "env": entry.get("env", {}),
        "cwd": entry.get("cwd", ""),
        "provider": entry.get("provider", ""),
        "model": entry.get("model", ""),
    }

    # Claude Code nested format: {"matcher": "Write", "hooks": [{"type": "command", ...}]}
    if "hooks" in entry and isinstance(entry["hooks"], list):
        matcher = entry.get("matcher", "")
        hooks_list = entry["hooks"]

        if hooks_list:
            inner = hooks_list[0]
            inner_type = inner.get("type", "command")

            # Nested entries can also be prompt/task/skill/tool
            merged = {**common, **inner, "matcher": matcher}
            return _build_hook_from_type(inner_type, merged)
        return None

    hook_type = entry.get("type", "")

    # Prompt hook: {"type": "prompt", "prompt": "..."}
    if hook_type == "prompt" or ("prompt" in entry and hook_type != "command"):
        return Hook(
            prompt=entry.get("prompt", ""),
            **common,
        )

    # Task hook: {"type": "task", "task": "task_name"}
    if hook_type == "task" or ("task" in entry and hook_type != "command"):
        return Hook(
            task=entry.get("task", ""),
            **common,
        )

    # Skill hook: {"type": "skill", "skill": "skill_name"}
    if hook_type == "skill" or ("skill" in entry and hook_type != "command"):
        return Hook(
            skill=entry.get("skill", ""),
            **common,
        )

    # Tool hook: {"type": "tool", "tool": "tool_name"}
    if hook_type == "tool" or ("tool" in entry and "command" not in entry):
        return Hook(
            tool=entry.get("tool", ""),
            tool_args=entry.get("tool_args", {}),
            **common,
        )

    # Command hook (default): {"type": "command", "command": "..."}
    if hook_type == "command" or "command" in entry:
        return Hook(
            command=entry.get("command", ""),
            command_windows=entry.get("windows", ""),
            command_linux=entry.get("linux", ""),
            command_osx=entry.get("osx", ""),
            **common,
        )

    logger.warning("Unrecognized hook entry format: %s", entry)
    return None


def _build_hook_from_type(hook_type: str, params: dict[str, Any]) -> Hook | None:
    """Build a Hook from a type string and merged parameter dict.

    Args:
        hook_type: One of ``"command"``, ``"prompt"``, ``"task"``,
            ``"skill"``, ``"tool"``.
        params: Merged parameters for the hook.

    Returns:
        A ``Hook`` instance, or ``None`` on error.
    """
    common = {
        "matcher": params.get("matcher", ""),
        "timeout": params.get("timeout", _DEFAULT_HOOK_TIMEOUT),
        "name": params.get("name", ""),
        "enabled": params.get("enabled", True),
        "env": params.get("env", {}),
        "cwd": params.get("cwd", ""),
        "provider": params.get("provider", ""),
        "model": params.get("model", ""),
    }

    try:
        if hook_type == "prompt":
            return Hook(prompt=params.get("prompt", ""), **common)
        if hook_type == "task":
            return Hook(task=params.get("task", ""), **common)
        if hook_type == "skill":
            return Hook(skill=params.get("skill", ""), **common)
        if hook_type == "tool":
            return Hook(
                tool=params.get("tool", ""),
                tool_args=params.get("tool_args", {}),
                **common,
            )
        # Default: command
        return Hook(
            command=params.get("command", ""),
            command_windows=params.get("windows", ""),
            command_linux=params.get("linux", ""),
            command_osx=params.get("osx", ""),
            **common,
        )
    except (ValueError, TypeError) as e:
        logger.warning("Failed to build hook (type=%s): %s", hook_type, e)
        return None


# ── Helper utilities ─────────────────────────────────────────────────────────

class _SafeFormatDict(dict):
    """Dict subclass for safe ``str.format_map`` — missing keys stay as-is."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _parse_genai_response(response: Any, hook_name: str) -> HookResult:
    """Convert a GenAI service response into a ``HookResult``.

    If the response looks like JSON with hook-specific fields, parse it.
    Otherwise wrap the raw text.

    Args:
        response: The response from ``generate_completion``.
        hook_name: Name for logging.

    Returns:
        A ``HookResult``.
    """
    text = str(response).strip() if response else ""

    # Try to parse as structured HookResult JSON
    if text.startswith("{"):
        try:
            data = json.loads(text)

            if any(k in data for k in ("block", "continue", "decision",
                                        "system_message", "additional_context")):
                parsed = HookResult.from_json(text)
                parsed.raw_output = text
                return parsed
        except json.JSONDecodeError:
            pass

    return HookResult(
        raw_output=text,
        additional_context=text,
    )


def _build_skill_input(ctx: HookContext) -> str:
    """Build a descriptive input message for skill execution from context.

    Args:
        ctx: The hook context.

    Returns:
        A string summarizing the relevant context for skill processing.
    """
    parts: list[str] = []

    if ctx.user_message:
        parts.append(f"User message: {ctx.user_message}")
    if ctx.tool_name:
        parts.append(f"Tool: {ctx.tool_name}")
    if ctx.tool_response:
        parts.append(f"Tool response: {ctx.tool_response}")
    if ctx.llm_response:
        parts.append(f"LLM response: {ctx.llm_response}")
    if ctx.agent_response:
        parts.append(f"Agent response: {ctx.agent_response}")
    if ctx.error:
        parts.append(f"Error: {ctx.error}")

    return "\n".join(parts) if parts else ctx.to_json()


def load_hooks_from_file(path: str | Path) -> HookManager:
    """Convenience function: create a ``HookManager`` from a JSON file.

    Args:
        path: Path to the JSON hook configuration file.

    Returns:
        A new ``HookManager`` with hooks loaded from the file.

    Example:
        >>> manager = load_hooks_from_file(".github/hooks/format.json")
    """
    manager = HookManager()
    manager.load_file(path)
    return manager


# ── Auto-discovery paths ─────────────────────────────────────────────────────

HOOK_DISCOVERY_PATHS: list[str] = [
    ".github/hooks",
    ".claude/settings.json",
    ".claude/settings.local.json",
    "~/.copilot/hooks",
    "~/.claude/settings.json",
]
"""Default locations searched by :meth:`HookManager.discover`."""


def discover_hooks(
    workspace: str | Path | None = None,
    *,
    extra_paths: list[str | Path] | None = None,
) -> HookManager:
    """Create a ``HookManager`` pre-loaded with auto-discovered hooks.

    Searches workspace and user directories for hook configuration files
    following the VS Code / Claude Code conventions.

    Args:
        workspace: Workspace root directory. Defaults to ``cwd``.
        extra_paths: Additional file or directory paths to load.

    Returns:
        A new ``HookManager`` with discovered hooks loaded.

    Example:
        >>> manager = discover_hooks("/path/to/project")
    """
    manager = HookManager()
    manager.discover(workspace, extra_paths=extra_paths)
    return manager


# ── Agent-scoped hooks ───────────────────────────────────────────────────────

def load_agent_scoped_hooks(agent_md_path: str | Path) -> HookManager:
    """Load hooks from a custom agent's ``.agent.md`` YAML frontmatter.

    Agent-scoped hooks only run when that custom agent is active.  The
    ``hooks`` field in the frontmatter uses the same structure as JSON
    hook configuration files.

    Frontmatter example::

        ---
        name: "Strict Formatter"
        description: "Agent that auto-formats code after every edit"
        hooks:
          PostToolUse:
            - type: command
              command: "./scripts/format-changed-files.sh"
        ---

    Args:
        agent_md_path: Path to the ``.agent.md`` file.

    Returns:
        A new ``HookManager`` with agent-scoped hooks loaded.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(agent_md_path)

    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Extract YAML frontmatter between --- markers
    frontmatter = _extract_yaml_frontmatter(text)
    if not frontmatter:
        return HookManager()

    hooks_section = frontmatter.get("hooks")
    if not hooks_section or not isinstance(hooks_section, dict):
        return HookManager()

    manager = HookManager()
    manager.load_config({"hooks": hooks_section})
    logger.info(
        "Loaded %d agent-scoped hooks from %s",
        manager.count(), path.name,
    )
    return manager


def _extract_yaml_frontmatter(text: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter delimited by ``---`` markers.

    Args:
        text: Full file text.

    Returns:
        Parsed dict, or ``None`` if no valid frontmatter found.
    """
    text = text.strip()
    if not text.startswith("---"):
        return None

    end = text.find("---", 3)
    if end == -1:
        return None

    yaml_text = text[3:end].strip()
    if not yaml_text:
        return None

    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load(yaml_text)
    except ImportError:
        # Fallback: simple key-value parser for basic YAML
        return _simple_yaml_parse(yaml_text)
    except Exception as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
        return None


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Minimal YAML-like parser for frontmatter without PyYAML.

    Handles top-level scalar keys and nested list-of-dict structures
    used by hook configuration.  Not a full YAML parser.

    Args:
        text: YAML-like text.

    Returns:
        Parsed dict.
    """
    result: dict[str, Any] = {}
    current_key: str = ""
    current_list: list[dict[str, Any]] = []
    current_item: dict[str, Any] = {}
    in_list = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())

        # Top-level key: value
        if indent == 0 and ":" in stripped:
            # Save previous list
            if in_list and current_key:
                if current_item:
                    current_list.append(current_item)
                result[current_key] = current_list

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if value:
                result[key] = value
                in_list = False
                current_key = ""
            else:
                current_key = key
                current_list = []
                current_item = {}
                in_list = True
            continue

        if in_list:
            # Nested event key (e.g. "  PostToolUse:")
            if indent == 2 and stripped.endswith(":"):
                if current_item:
                    current_list.append(current_item)
                    current_item = {}
                # This is a sub-key under hooks
                sub_key = stripped[:-1].strip()
                if current_key == "hooks":
                    if current_list:
                        result[current_key] = current_list
                    current_key = "hooks"
                    if "hooks" not in result:
                        result["hooks"] = {}
                    if not isinstance(result["hooks"], dict):
                        result["hooks"] = {}
                    result["hooks"][sub_key] = []
                    current_list = result["hooks"][sub_key]
                    current_item = {}
                continue

            # List item start
            if stripped.startswith("- "):
                if current_item:
                    current_list.append(current_item)
                item_content = stripped[2:].strip()
                current_item = {}
                if ":" in item_content:
                    k, _, v = item_content.partition(":")
                    current_item[k.strip()] = v.strip().strip('"').strip("'")
                continue

            # Continuation of list item
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                current_item[k.strip()] = v.strip().strip('"').strip("'")

    # Save last pending items
    if in_list and current_key:
        if current_item:
            current_list.append(current_item)
        if current_key != "hooks":
            result[current_key] = current_list

    return result
