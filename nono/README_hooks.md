# Nono Hooks — Lifecycle Hook Engine

Deterministic, code-driven automations that execute at specific lifecycle points during agent sessions. Inspired by [Claude Code Hooks](https://docs.anthropic.com/en/docs/claude-code/hooks).

## Overview

Hooks provide guaranteed execution of your code at precise moments — before a tool runs, after an agent finishes, when a session starts, or when an error occurs. Unlike instructions or prompts that guide behaviour, hooks enforce policies deterministically.

| Use case | How hooks help |
|----------|---------------|
| **Security policies** | Block dangerous tools before they execute |
| **Code quality** | Run formatters/linters after file modifications |
| **Audit trails** | Log every tool invocation and command execution |
| **Context injection** | Add project-specific information to agent conversations |
| **Approval control** | Auto-approve safe operations, deny sensitive ones |

## Quick Start

```python
from nono.hooks import HookManager, HookEvent, Hook, HookContext, HookResult
from nono.agent import Agent, Runner

# 1. Create a hook manager
manager = HookManager()

# 2. Register a Python hook
def block_dangerous_tools(ctx: HookContext) -> HookResult:
    if ctx.tool_name in ("delete_files", "drop_table"):
        return HookResult(block=True, block_reason="Tool blocked by security policy")
    return HookResult()

manager.register(HookEvent.PRE_TOOL_USE, Hook(fn=block_dangerous_tools))

# 3. Attach to an agent
agent = Agent(
    name="assistant",
    model="gemini-3-flash-preview",
    provider="google",
    instruction="You are a helpful assistant.",
    hook_manager=manager,
)

# 4. Run — PreToolUse hooks fire automatically before each tool call
result = Runner(agent).run("Help me with my data")
```

## Hook Events

| Event | When it fires | Primary use cases |
|-------|---------------|-------------------|
| `SESSION_START` | New agent session begins | Initialize resources, log session start |
| `SESSION_END` | Agent session closes | Cleanup, send notifications |
| `USER_PROMPT_SUBMIT` | User submits a prompt | Audit requests, inject context |
| `PRE_TOOL_USE` | Before agent invokes a tool | Block dangerous ops, modify input |
| `POST_TOOL_USE` | After tool completes | Run formatters, log results |
| `PRE_AGENT_RUN` | Before agent execution logic | Validate state, block execution |
| `POST_AGENT_RUN` | After agent finishes | Inject context, generate reports |
| `PRE_LLM_CALL` | Before LLM API call | Modify messages, add context |
| `POST_LLM_CALL` | After LLM API returns | Log responses, validate output |
| `SUBAGENT_START` | Sub-agent spawned | Track nested usage |
| `SUBAGENT_STOP` | Sub-agent completes | Aggregate results |
| `STOP` | Final response generated | Final checks, reports |
| `ERROR` | Error occurs | Error logging, alerts |
| `NOTIFICATION` | System notification | Handle alerts |
| `PRE_COMPACT` | Before context pruning | Save important state |

## Hook Types

Six execution types are supported. Use `HookType` to inspect a hook's type programmatically.

| Type | Config `"type"` | Requires | Description |
|------|----------------|----------|-------------|
| **Function** | `function` | `fn` | Python callable invoked directly |
| **Command** | `command` | `command` | Shell command (subprocess) |
| **Prompt** | `prompt` | `prompt` | Inline GenAI prompt sent to LLM |
| **Task** | `task` | `task` | Predefined JSON task from `prompts/` |
| **Skill** | `skill` | `skill` | Registered skill from the registry |
| **Tool** | `tool` | `tool` | Registered `FunctionTool` invocation |

### Python Callable Hooks

```python
def my_hook(ctx: HookContext) -> HookResult:
    """Receives context, returns result to influence behaviour."""
    print(f"Agent {ctx.agent_name} is calling tool {ctx.tool_name}")
    return HookResult(system_message="Logged tool call")

manager.register(HookEvent.PRE_TOOL_USE, Hook(fn=my_hook, name="logger"))
```

### Shell Command Hooks

```python
manager.register(
    HookEvent.POST_TOOL_USE,
    Hook(
        command="python scripts/format.py",
        command_windows="powershell -File scripts\\format.ps1",
        matcher="Write|Edit",  # Only fire for Write/Edit tools
        timeout=15,
        name="auto_format",
    ),
)
```

Shell hooks receive `HookContext` as JSON on stdin and return `HookResult` as JSON on stdout. Exit codes: `0` = success, `2` = blocking error, other = warning.

### GenAI Prompt Hooks

Send an inline prompt to any LLM provider. Template variables (`{field}`) are resolved from `HookContext` fields.

```python
# Validate tool output with an LLM
manager.register(
    HookEvent.POST_TOOL_USE,
    Hook(
        prompt="Analyze this tool output for errors: {tool_response}",
        provider="google",
        model="gemini-3-flash-preview",
        matcher="Write|Edit",
        name="output_validator",
    ),
)
```

If the LLM returns valid JSON with `HookResult` fields (`block`, `continue`, `system_message`, etc.), it is parsed as a structured result. Otherwise the raw text is returned as `additional_context`.

### Task Hooks

Execute a predefined JSON task from the `prompts/` directory. The task's `user_prompt_template` is resolved with context variables.

```python
# Run the "review_output" task after every tool call
manager.register(
    HookEvent.POST_TOOL_USE,
    Hook(
        task="review_output",  # loads prompts/review_output.json
        provider="openai",
        model="gpt-4o-mini",
        name="auto_review",
    ),
)
```

Task JSON format (in `prompts/review_output.json`):

```json
{
  "task_name": "review_output",
  "system_prompt": "You are a code reviewer.",
  "user_prompt_template": "Review the following output: {tool_response}"
}
```

### Skill Hooks

Invoke a registered skill from the [skill registry](./README_skills.md). The skill receives a summary of the hook context as input.

```python
# Run summarization after agent completes
manager.register(
    HookEvent.POST_AGENT_RUN,
    Hook(
        skill="summarize_skill",
        name="auto_summarize",
    ),
)
```

### Tool Hooks

Invoke a registered `FunctionTool` with explicit arguments. Values support `{variable}` placeholders.

```python
# Log every tool invocation
manager.register(
    HookEvent.PRE_TOOL_USE,
    Hook(
        tool="audit_logger",
        tool_args={"action": "tool_call", "target": "{tool_name}"},
        name="audit_hook",
    ),
)
```

Tools are resolved from `ctx.extra["_tools"]` (agent-provided) or a global `tool_registry` if available.

## HookContext — VS Code-Compatible Input

The `HookContext` serializes to JSON using **camelCase keys** matching the [VS Code hook protocol](https://code.visualstudio.com/docs/copilot/customization/hooks):

| Python field | JSON key | Events |
|---|---|---|
| `session_id` | `sessionId` | All |
| `event` | `hookEventName` | All |
| `cwd` | `cwd` | All |
| `timestamp` | `timestamp` | All |
| `transcript_path` | `transcript_path` | All (when set) |
| `user_message` | `prompt` | UserPromptSubmit |
| `tool_name` | `tool_name` | PreToolUse, PostToolUse |
| `tool_input` | `tool_input` | PreToolUse, PostToolUse |
| `tool_use_id` | `tool_use_id` | PreToolUse, PostToolUse |
| `tool_response` | `tool_response` | PostToolUse |
| `source` | `source` | SessionStart |
| `trigger` | `trigger` | PreCompact |
| `stop_hook_active` | `stop_hook_active` | Stop, SubagentStop |
| `subagent_id` | `agent_id` | SubagentStart, SubagentStop |
| `subagent_type` | `agent_type` | SubagentStart, SubagentStop |

## Auto-Discovery

Hooks can be auto-discovered from standard locations following the VS Code / Claude Code conventions:

```python
from nono.hooks import discover_hooks, HookManager

# Discover from current workspace
manager = discover_hooks("/path/to/workspace")

# Or use the method on an existing manager
manager = HookManager()
manager.discover("/path/to/workspace", extra_paths=["custom/hooks"])
```

Search locations (in priority order):

| Scope | Path |
|-------|------|
| Workspace | `.github/hooks/*.json` |
| Workspace | `.claude/settings.json` |
| Workspace | `.claude/settings.local.json` |
| User | `~/.copilot/hooks/*.json` |
| User | `~/.claude/settings.json` |

Workspace hooks take precedence over user hooks for the same event type.

## Agent-Scoped Hooks

Load hooks from a custom agent's `.agent.md` YAML frontmatter. These hooks only run when that agent is active.

```markdown
---
name: "Strict Formatter"
description: "Agent that auto-formats code after every edit"
hooks:
  PostToolUse:
    - type: command
      command: "./scripts/format-changed-files.sh"
---

You are a code editing agent.
```

```python
from nono.hooks import load_agent_scoped_hooks

agent_hooks = load_agent_scoped_hooks(".github/agents/formatter.agent.md")
```

## Configuration Files

Load hooks from JSON files. All six hook types are supported in configuration.

### Nono Extended Format

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "type": "command",
        "command": "./scripts/validate-tool.sh",
        "timeout": 15
      },
      {
        "type": "prompt",
        "prompt": "Is tool {tool_name} safe to execute?",
        "provider": "google",
        "model": "gemini-3-flash-preview",
        "matcher": "delete.*|drop.*"
      }
    ],
    "PostToolUse": [
      {
        "type": "task",
        "task": "review_output",
        "matcher": "Write|Edit"
      },
      {
        "type": "skill",
        "skill": "summarize_skill"
      },
      {
        "type": "tool",
        "tool": "audit_logger",
        "tool_args": {"action": "tool_completed", "target": "{tool_name}"}
      }
    ]
  }
}
```

### Flat Format

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "type": "command",
        "command": "./scripts/validate-tool.sh",
        "timeout": 15
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "npx prettier --write \"$TOOL_INPUT_FILE_PATH\""
      }
    ]
  }
}
```

### Claude Code Format

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python -m black ."
          }
        ]
      }
    ]
  }
}
```

### Loading Configuration

```python
# From a dict
manager.load_config({"hooks": {"PreToolUse": [...]}})

# From a single file
manager.load_file(".github/hooks/security.json")

# From a directory (all *.json files)
manager.load_directory(".github/hooks/")

# Convenience: create manager from file
from nono.hooks import load_hooks_from_file
manager = load_hooks_from_file(".github/hooks/format.json")
```

## Matcher Filtering

For `PreToolUse` and `PostToolUse` events, matchers filter which tools trigger the hook:

```python
# Match specific tools (regex)
Hook(command="echo ok", matcher="Write|Edit")

# Match tools by prefix
Hook(command="echo ok", matcher=r"^get_.*")

# Match all tools (default when no matcher)
Hook(command="echo ok")
```

## HookResult — Controlling Behaviour

### Block a Tool Call (PreToolUse)

```python
def security_check(ctx: HookContext) -> HookResult:
    if "production" in (ctx.tool_input or {}).get("path", ""):
        return HookResult(
            block=True,
            block_reason="Cannot modify production files",
            permission_decision="deny",
        )
    return HookResult(permission_decision="allow")
```

### Modify Tool Input

```python
def sanitize_input(ctx: HookContext) -> HookResult:
    safe_input = {k: v for k, v in (ctx.tool_input or {}).items() if k != "password"}
    return HookResult(updated_input=safe_input)
```

### Stop the Session

```python
def budget_guard(ctx: HookContext) -> HookResult:
    if ctx.extra.get("tokens_used", 0) > 100_000:
        return HookResult(continue_execution=False, stop_reason="Token budget exceeded")
    return HookResult()
```

### Inject Additional Context

```python
def add_project_info(ctx: HookContext) -> HookResult:
    return HookResult(additional_context="Project: my-app v2.1.0 | Branch: main")
```

## Merging Multiple Hooks

When multiple hooks fire for the same event, results merge with **most restrictive wins**:

- Permission: `deny` > `ask` > `allow`
- Block: any `True` wins
- Stop: any `False` in `continue_execution` wins
- Messages: concatenated
- Exit codes: max value wins

## Agent Integration

### Constructor Parameter

```python
agent = Agent(
    name="my_agent",
    model="gemini-3-flash-preview",
    provider="google",
    hook_manager=manager,
)
```

### Fluent API

```python
agent = Agent(name="my_agent", ...)
agent.set_hook_manager(manager)
```

### Automatic Hook Points

| Agent method | Hooks fired |
|-------------|-------------|
| `agent.run()` | PreAgentRun → (execution) → PostAgentRun |
| `agent.run_async()` | PreAgentRun → (execution) → PostAgentRun |
| Tool call in LlmAgent | PreToolUse → (tool execution) → PostToolUse |
| Exception during run | Error |

## API Reference

| Class | Purpose |
|-------|---------|
| `HookEvent` | Enum of lifecycle events |
| `HookType` | Enum of hook execution types (function, command, prompt, task, skill, tool) |
| `HookContext` | Structured input passed to hooks |
| `HookResult` | Structured output returned by hooks |
| `Hook` | Single hook definition (callable, command, prompt, task, skill, or tool) |
| `HookManager` | Registry that loads, stores, and fires hooks |
| `load_hooks_from_file()` | Convenience: create HookManager from JSON file |
| `discover_hooks()` | Convenience: create HookManager with auto-discovered hooks |
| `load_agent_scoped_hooks()` | Load hooks from `.agent.md` YAML frontmatter |
| `HOOK_DISCOVERY_PATHS` | Default search locations for auto-discovery |

## Related Documentation

- [Agent Architecture](./agent/README_agent.md)
- [Orchestration Hooks](./agent/README_orchestration.md)
- [Configuration](./README_config.md)
