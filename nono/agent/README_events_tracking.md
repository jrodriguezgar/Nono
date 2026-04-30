# Events and Tracing Guide

> Comprehensive guide to the event system and structured tracing/observability across **all** Nono modules: Agents, TaskExecutor, Workflows, and CodeExecuter.

## Table of Contents

- [Overview](#overview)
- [Event System](#event-system)
  - [EventType Enum](#eventtype-enum)
  - [Event Dataclass](#event-dataclass)
  - [Session Events](#session-events)
  - [Streaming Events](#streaming-events)
- [Lifecycle Callbacks (Agents)](#lifecycle-callbacks-agents)
  - [before_agent / after_agent](#before_agent--after_agent)
  - [before_tool / after_tool](#before_tool--after_tool)
- [Workflow Step Callbacks](#workflow-step-callbacks)
  - [on_before_step](#on_before_step)
  - [on_after_step](#on_after_step)
- [Event Logging (Tasker)](#event-logging-tasker)
  - [@event_log Decorator](#event_log-decorator)
  - [msg_log Function](#msg_log-function)
- [Tracing System](#tracing-system)
  - [Quick Start](#quick-start)
  - [Unified Imports](#unified-imports)
  - [TraceCollector](#tracecollector)
  - [Trace](#trace)
  - [LLMCall](#llmcall)
  - [ToolRecord](#toolrecord)
  - [TokenUsage](#tokenusage)
  - [TraceStatus](#tracestatus)
- [Nested Traces](#nested-traces)
- [Token Tracking](#token-tracking)
- [Tool Tracking](#tool-tracking)
- [LLM Call Tracking](#llm-call-tracking)
- [Error Tracing](#error-tracing)
- [Exporting Traces](#exporting-traces)
- [Integration with Runner](#integration-with-runner)
- [Integration with Orchestration Agents](#integration-with-orchestration-agents)
- [Integration with TaskExecutor](#integration-with-taskexecutor)
- [Integration with Workflows](#integration-with-workflows)
- [Integration with CodeExecuter](#integration-with-codeexecuter)
- [Cross-Module Tracing](#cross-module-tracing)
- [Observability Summary](#observability-summary)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Overview

Nono provides four complementary observability layers:

| Layer | Purpose | Granularity | Scope |
| --- | --- | --- | --- |
| **Events** | Stream of immutable records during agent execution | Per-action (message, tool call, state update, error) | Agents |
| **Traces** | Structured performance/observability data | Per-component invocation (nested, with timing and tokens) | All modules |
| **Callbacks** | Runtime interception and modification of I/O | Per-agent or per-workflow-step | Agents, Workflows |
| **Event Logging** | Function entry/exit timestamped logging | Per-function call | Taskers, Executer |

**Events** tell you *what happened*. **Traces** tell you *how it performed*. **Callbacks** let you *intercept and modify*. **Event Logging** gives you *timestamped monitoring*.

Tracing is available across **all** execution modules:

| Module | Parameter | Trace type |
| --- | --- | --- |
| `Runner` (Agents) | `Runner(agent, trace_collector=)` | `LlmAgent`, `SequentialAgent`, etc. |
| `TaskExecutor` | `execute(..., trace_collector=)` / `run_json_task(..., trace_collector=)` | `TaskExecutor` |
| `Workflow` | `run(trace_collector=)` / `stream(trace_collector=)` | `Workflow` → `WorkflowStep` children |
| `CodeExecuter` | `run(..., trace_collector=)` | `CodeExecuter` |
| API Server | `"trace": true` in request body | All endpoint types |

---

## Event System

### EventType Enum

Every action during agent execution emits an `Event` with one of these types:

| EventType | Description | Typical author |
| --- | --- | --- |
| `USER_MESSAGE` | User input recorded to session | `"user"` |
| `AGENT_MESSAGE` | Agent's text response | Agent name |
| `TOOL_CALL` | Tool invocation initiated | Agent name |
| `TOOL_RESULT` | Tool returned a result | Agent name |
| `STATE_UPDATE` | Session state was modified | Agent name |
| `AGENT_TRANSFER` | Control transferred to another agent | Agent name |
| `HUMAN_INPUT_REQUEST` | Execution paused — awaiting human input | `HumanInputAgent` name |
| `HUMAN_INPUT_RESPONSE` | Human responded — execution resumes | `HumanInputAgent` name |
| `ERROR` | An error occurred | Agent name |

```python
from nono.agent import EventType
```

### Event Dataclass

`Event` is a **frozen** (immutable) dataclass:

```python
@dataclass(frozen=True)
class Event:
    event_type: EventType
    author: str          # Who produced the event (agent name or "user")
    content: str         # Human-readable description
    data: dict           # Structured payload (tool args, state delta, etc.)
    timestamp: datetime  # UTC timestamp (auto-generated)
    event_id: str        # Unique ID (auto-generated)
```

**Example:**

```python
from nono.agent import Event, EventType

event = Event(EventType.TOOL_CALL, "my_agent", "Calling search", data={"tool": "search", "arguments": {"q": "Nono"}})
print(event.event_type)   # EventType.TOOL_CALL
print(event.author)       # "my_agent"
print(event.data["tool"]) # "search"
```

### Session Events

All events are automatically recorded in the session:

```python
from nono.agent import Agent, Runner

runner = Runner(my_agent)
runner.run("What is the capital of Spain?")

for event in runner.session.events:
    print(f"[{event.event_type.name}] {event.author}: {event.content}")
```

Output:

```
[USER_MESSAGE] user: What is the capital of Spain?
[AGENT_MESSAGE] my_agent: The capital of Spain is Madrid.
```

### Streaming Events

Use `Runner.stream()` or `Runner.astream()` to receive events as they happen:

```python
runner = Runner(my_agent)

for event in runner.stream("Explain quantum computing"):
    if event.event_type == EventType.TOOL_CALL:
        print(f"  → Tool: {event.data['tool']}")
    elif event.event_type == EventType.AGENT_MESSAGE:
        print(f"  ✓ {event.content[:80]}...")
```

---

## Lifecycle Callbacks (Agents)

All agent classes inheriting from `BaseAgent` support four lifecycle hooks. These are invoked automatically by the framework during agent execution.

### before_agent / after_agent

| Hook | Signature | Behavior |
| --- | --- | --- |
| `before_agent` | `(agent, ctx) -> Optional[str]` | Return a `str` to **short-circuit** the agent (skip LLM call). Return `None` to proceed normally. |
| `after_agent` | `(agent, ctx, response) -> Optional[str]` | Return a `str` to **replace** the agent's response. Return `None` to keep the original. |

```python
from nono.agent import Agent, Runner

def log_input(agent, ctx):
    print(f"[BEFORE] {agent.name} received: {ctx.user_message}")
    return None  # proceed normally

def redact_output(agent, ctx, response):
    print(f"[AFTER] {agent.name} produced: {response[:80]}...")
    return response.replace("confidential", "[REDACTED]")

agent = Agent(
    name="assistant",
    provider="google",
    instruction="You are a helpful assistant.",
    before_agent_callback=log_input,
    after_agent_callback=redact_output,
)

runner = Runner(agent)
runner.run("Tell me about the confidential project")
```

### before_tool / after_tool

| Hook | Signature | Behavior |
| --- | --- | --- |
| `before_tool` | `(agent, tool_name, args) -> Optional[dict]` | Return a `dict` to **replace** tool arguments. Return `None` to keep original args. |
| `after_tool` | `(agent, tool_name, args, result) -> Optional[Any]` | Return a value to **replace** the tool result. Return `None` to keep original result. |

```python
from nono.agent import Agent, Runner, tool

@tool(description="Search the web")
def search_web(query: str) -> str:
    return f"Results for: {query}"

def audit_tool_call(agent, tool_name, args):
    print(f"  [TOOL] {tool_name}({args})")
    return None

def filter_tool_result(agent, tool_name, args, result):
    print(f"  [RESULT] {tool_name} -> {result[:50]}")
    return None

agent = Agent(
    name="researcher",
    provider="google",
    instruction="Research topics using the search tool.",
    tools=[search_web],
    before_tool_callback=audit_tool_call,
    after_tool_callback=filter_tool_result,
)
```

> **Scope:** Lifecycle callbacks are available on all agent types: `LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `RouterAgent`.

---

## Workflow Step Callbacks

Workflows support seven lifecycle hooks for controlling and observing execution. Register them via fluent API methods.

### on_start

```python
Workflow.on_start(callback: (workflow_name, state) -> None)
```

Called **once** when the workflow starts, before the first step runs.

### on_end

```python
Workflow.on_end(callback: (workflow_name, state, steps_executed) -> None)
```

Called **once** when the workflow finishes, regardless of how it ended.

### on_before_step

```python
Workflow.on_before_step(callback: (step_name, state) -> Optional[dict])
```

Called **before** each step executes. Return a `dict` to **skip** the step entirely (the dict is used as the step result). Return `None` to let the step run normally.

### on_after_step

```python
Workflow.on_after_step(callback: (step_name, state, result) -> Optional[dict])
```

Called **after** each step executes. Return a `dict` to **replace** the step's result. Return `None` to keep the original result.

### on_between_steps

```python
Workflow.on_between_steps(callback: (completed_step, next_step, state) -> Optional[bool])
```

Called **between** steps — after one step completes and before the next begins. Return `False` to **halt** the workflow. Any other return value continues normally.

### on_step_executing

```python
Workflow.on_step_executing(callback: (step_name, state, attempt) -> None)
```

Called **right before each execution attempt** — inside the retry loop. Fires once per attempt. The `attempt` parameter is 1-based.

### on_step_executed

```python
Workflow.on_step_executed(callback: (step_name, state, attempt, error) -> None)
```

Called **right after each execution attempt** — inside the retry loop. The counterpart of `on_step_executing`. `error` is `None` on success or the error message on failure.

### Example: I/O logging

```python
from nono.workflows import Workflow

def log_step_io(step_name, state):
    print(f"  → [{step_name}] input keys: {list(state.keys())}")
    return None  # proceed normally

def log_step_result(step_name, state, result):
    keys = list(result.keys()) if isinstance(result, dict) else str(result)[:50]
    print(f"  ← [{step_name}] output keys: {keys}")
    return None  # keep original result

flow = Workflow("my_pipeline")
flow.step("preprocess", lambda s: {"clean": s["input"].strip()})
flow.step("analyze", lambda s: {"score": len(s["clean"])})
flow.on_before_step(log_step_io)
flow.on_after_step(log_step_result)

flow.run(input="  hello world  ")
```

Output:

```
  → [preprocess] input keys: ['input']
  ← [preprocess] output keys: ['clean']
  → [analyze] input keys: ['input', 'clean']
  ← [analyze] output keys: ['score']
```

### Example: step skipping (cache)

```python
cache = {}

def check_cache(step_name, state):
    key = f"{step_name}:{hash(frozenset(state.items()))}"
    if key in cache:
        print(f"  [CACHE HIT] {step_name}")
        return cache[key]
    return None

def store_cache(step_name, state, result):
    key = f"{step_name}:{hash(frozenset(state.items()))}"
    if isinstance(result, dict):
        cache[key] = result
    return None

flow.on_before_step(check_cache)
flow.on_after_step(store_cache)
```

### Example: security filter

```python
def block_sensitive_steps(step_name, state):
    if step_name == "deploy" and not state.get("approved"):
        print(f"  [BLOCKED] {step_name} — not approved")
        return {"error": "Deployment requires approval"}
    return None

flow.on_before_step(block_sensitive_steps)
```

### Example: between-steps progress and halt

```python
def progress_and_gate(prev, nxt, state):
    print(f"  {prev} ✓ → {nxt or 'END'}")
    if state.get("quality", 1.0) < 0.3:
        print(f"  [HALT] quality too low after {prev}")
        return False
    return None

flow.on_between_steps(progress_and_gate)
```

### Example: retry telemetry

```python
def log_retries(name, state, attempt):
    if attempt > 1:
        print(f"  ⟳ {name} retry #{attempt}")

flow.on_step_executing(log_retries)
```

### Hook Comparison

| Hook | When it fires | Can alter execution | Invocations |
|:---|:---|:---|:---|
| `on_start` | **Start** of the workflow | No — observation only | 1 per run |
| `on_before_step` | **Before** a step runs | Yes — return `dict` to **skip** the step | 1 per step |
| `on_step_executing` | **Before each attempt** (inside retry loop) | No — observation only | N (1 per attempt) |
| `on_step_executed` | **After each attempt** (inside retry loop) | No — observation only | N (1 per attempt) |
| `on_after_step` | **After** a step runs | Yes — return `dict` to **replace** the result | 1 per step |
| `on_between_steps` | **Between** steps | Yes — return `False` to **halt** the workflow | 1 per transition |
| `on_end` | **End** of the workflow | No — observation only | 1 per run |

**Key differences:**
- `on_start` / `on_end`: workflow-level bookends for setup/teardown and telemetry.
- `on_before_step` vs `on_step_executing`: `on_before_step` runs once and can skip; `on_step_executing` runs per retry attempt and is informational only.
- `on_step_executing` vs `on_step_executed`: mirror pair inside the retry loop — before and after each attempt.
- `on_after_step` vs `on_between_steps`: `on_after_step` modifies the step result; `on_between_steps` controls workflow continuation.
- `on_before_step` / `on_after_step` control a **single step**; `on_between_steps` controls the **overall flow**.

> **Scope:** All seven workflow callbacks work in every execution method: `run()`, `run_async()`, `stream()`, `astream()`, and `replay_from()`.

---

## Event Logging (Tasker)

The Tasker and Executer modules use a lightweight logging system based on the `@event_log` decorator and `msg_log` function. This provides timestamped entry/exit logging for task operations.

### @event_log Decorator

Automatically logs function entry ("Starting"), exit ("Completed"), and errors:

```python
from nono.tasker.genai_tasker import event_log

@event_log("Processing customer feedback")
def process_feedback(data: list[dict]) -> dict:
    # ... implementation ...
    return {"processed": len(data)}
```

Log output:

```
2026-03-31 10:30:00 - [INFO] Starting: Processing customer feedback
2026-03-31 10:30:02 - [INFO] Completed: Processing customer feedback
```

On error:

```
2026-03-31 10:30:00 - [INFO] Starting: Processing customer feedback
2026-03-31 10:30:01 - [ERROR] Error in Processing customer feedback: KeyError: 'missing_field'
```

Functions decorated with `@event_log` in Nono:

| Module | Function | Message |
| --- | --- | --- |
| `genai_tasker` | `BaseAIClient.generate_content()` | `"AI Content Generation"` |
| `genai_tasker` | `TaskExecutor.execute()` | `"Task Execution"` |
| `genai_executer` | `CodeExecuter.run()` | `"Code Execution"` |

### msg_log Function

For ad-hoc logging at any point:

```python
from nono.tasker.genai_tasker import msg_log
import logging

msg_log("Processing batch 3/5", logging.INFO)
msg_log("Rate limit reached, backing off", logging.WARNING)
msg_log("API connection failed", logging.ERROR)
```

> **Note:** `@event_log` captures function entry/exit timing but does **not** record the actual prompt or LLM response content. For full I/O tracing, use `TraceCollector` with `execute(..., trace_collector=collector)` or `run_json_task(..., trace_collector=collector)`.

---

## Tracing System

The tracing system provides **structured observability** for agent executions, capturing inputs, outputs, token estimates, tools used, LLM provider/model, status, errors, and timing.

### Quick Start

```python
from nono import TraceCollector  # Unified import (works for agents, workflows, tasker)
from nono.agent import Agent, Runner

collector = TraceCollector()
runner = Runner(agent=my_agent, trace_collector=collector)
response = runner.run("Summarize this document")
collector.print_summary()
```

### Unified Imports

All tracing classes can be imported from three equivalent locations:

```python
# 1. Top-level package (recommended)
from nono import TraceCollector, Trace, TraceStatus, LLMCall, TokenUsage, ToolRecord

# 2. Unified tracing module
from nono.tracing import TraceCollector, Trace, TraceStatus, LLMCall, TokenUsage, ToolRecord

# 3. Agent subpackage (original location)
from nono.agent import TraceCollector, Trace, TraceStatus, LLMCall, TokenUsage, ToolRecord
```

All three are equivalent. Use `from nono import TraceCollector` for cross-module code.

### Quick Start (original agent-only example)

```python
from nono.agent import Agent, Runner, TraceCollector

collector = TraceCollector()
runner = Runner(agent=my_agent, trace_collector=collector)

response = runner.run("Summarize this document")

# Print human-readable summary
collector.print_summary()

# Export as serializable dicts (for logging, dashboards, etc.)
data = collector.export()
```

Output:

```
[success] my_agent (LlmAgent) — 1234ms, 1 LLM call(s), 350 tokens, 0 tool(s), 0 child(ren)

Total: 1 trace(s), 1 LLM call(s), 350 token(s), 0 tool call(s)
```

### TraceCollector

`TraceCollector` manages the lifecycle of traces across an agent execution tree.

```python
from nono.agent import TraceCollector

collector = TraceCollector()
```

| Property / Method | Description |
| --- | --- |
| `traces` | List of top-level `Trace` objects |
| `current` | Currently active trace (or `None`) |
| `total_tokens` | Sum of all tokens across all traces (recursive) |
| `total_llm_calls` | Total LLM completion calls (recursive) |
| `total_tool_calls` | Total tool invocations (recursive) |
| `start_trace(...)` | Begin a new trace |
| `end_trace(output=, error=)` | End the active trace |
| `record_llm_call(llm_call)` | Record an LLM call in the active trace |
| `record_tool(tool_record)` | Record a tool call in the active trace |
| `export()` | Export all traces as `list[dict]` |
| `clear()` | Remove all collected traces |
| `print_summary()` | Print human-readable summary |

### Trace

Each agent invocation produces a `Trace` with:

```python
from nono.agent import Trace, TraceStatus

trace: Trace
trace.trace_id        # Unique ID (hex string)
trace.agent_name      # "my_agent"
trace.agent_type      # "LlmAgent", "SequentialAgent", etc.
trace.input_message   # User message that triggered the invocation
trace.output_message  # Final agent response
trace.status          # TraceStatus.SUCCESS / ERROR / RUNNING
trace.error           # Error message (if status == ERROR)
trace.provider        # "google", "openai", etc.
trace.model           # "gemini-3-flash-preview"
trace.llm_calls       # list[LLMCall]
trace.tools_used      # list[ToolRecord]
trace.token_usage     # TokenUsage (aggregated)
trace.duration_ms     # Total wall-clock time in ms
trace.started_at      # UTC datetime
trace.ended_at        # UTC datetime
trace.children        # list[Trace] (nested sub-agent traces)
trace.parent_id       # Parent trace ID
trace.metadata        # Extra key-value data
```

### LLMCall

Records a single LLM completion request:

```python
from nono.agent import LLMCall, TokenUsage

llm_call = LLMCall(
    provider="google",
    model="gemini-3-flash-preview",
    temperature=0.7,
    max_tokens=1024,
    token_usage=TokenUsage(input_tokens=200, output_tokens=80),
    duration_ms=1500.0,
    error=None,
)
```

### ToolRecord

Records a single tool invocation:

```python
from nono.agent import ToolRecord

tool_record = ToolRecord(
    tool_name="search_web",
    arguments={"query": "Nono framework"},
    result="Found 3 results...",
    duration_ms=250.0,
    error=None,  # or "ConnectionError: ..."
)
```

### TokenUsage

Estimated token counts:

```python
from nono.agent import TokenUsage

usage = TokenUsage(input_tokens=100, output_tokens=50)
print(usage.total_tokens)  # 150
```

> **Note:** Token counts are estimated using a `len(text) // 4` heuristic. For precise counts, use provider-specific tokenizers.

### TraceStatus

```python
from nono.agent import TraceStatus

TraceStatus.RUNNING  # Invocation in progress
TraceStatus.SUCCESS  # Completed successfully
TraceStatus.ERROR    # Failed with an error
```

---

## Nested Traces

Orchestration agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`, `RouterAgent`) automatically produce **nested traces**. The parent trace contains child traces for each sub-agent:

```python
from nono.agent import (
    Agent, SequentialAgent, Runner, TraceCollector,
)

researcher = Agent(name="researcher", provider="google", instruction="Research the topic.")
writer = Agent(name="writer", provider="google", instruction="Write a summary.")

pipeline = SequentialAgent(name="pipeline", sub_agents=[researcher, writer])

collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
runner.run("Explain quantum computing")

collector.print_summary()
```

Output:

```
[success] pipeline (SequentialAgent) — 3200ms, 0 LLM call(s), 0 tokens, 0 tool(s), 2 child(ren)
  [success] researcher (LlmAgent) — 1800ms, 1 LLM call(s), 420 tokens, 0 tool(s), 0 child(ren)
  [success] writer (LlmAgent) — 1400ms, 1 LLM call(s), 380 tokens, 0 tool(s), 0 child(ren)

Total: 1 trace(s), 2 LLM call(s), 800 token(s), 0 tool call(s)
```

The trace tree mirrors the agent composition:

```
pipeline (SequentialAgent)
├── researcher (LlmAgent) — 1 LLM call, 420 tokens
└── writer (LlmAgent) — 1 LLM call, 380 tokens
```

---

## Token Tracking

Token tracking is automatic for every LLM call. Tokens are estimated per-call and aggregated at the trace level:

```python
collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
runner.run("...")

# Per-trace tokens
for trace in collector.traces:
    print(f"{trace.agent_name}: {trace.token_usage.total_tokens} tokens")
    for call in trace.llm_calls:
        print(f"  LLM call: {call.token_usage.input_tokens} in, {call.token_usage.output_tokens} out")

# Global totals (recursive across nested traces)
print(f"Total tokens: {collector.total_tokens}")
```

---

## Tool Tracking

Every tool invocation is recorded with arguments, result, duration, and any errors:

```python
for trace in collector.traces:
    for tool in trace.tools_used:
        print(f"Tool: {tool.tool_name}")
        print(f"  Arguments: {tool.arguments}")
        print(f"  Result: {tool.result[:100]}")
        print(f"  Duration: {tool.duration_ms:.0f}ms")
        if tool.error:
            print(f"  ERROR: {tool.error}")
```

---

## LLM Call Tracking

Each LLM completion request is recorded with provider, model, parameters, and timing:

```python
for trace in collector.traces:
    for call in trace.llm_calls:
        print(f"Provider: {call.provider}, Model: {call.model}")
        print(f"Temperature: {call.temperature}, Max tokens: {call.max_tokens}")
        print(f"Duration: {call.duration_ms:.0f}ms")
        print(f"Tokens: {call.token_usage.total_tokens}")
```

---

## Error Tracing

When an agent raises an exception, the trace captures the error and marks the status:

```python
collector = TraceCollector()
runner = Runner(agent=risky_agent, trace_collector=collector)

try:
    runner.run("risky input")
except Exception:
    pass

trace = collector.traces[0]
print(trace.status)  # TraceStatus.ERROR
print(trace.error)   # "RuntimeError: something went wrong"
```

---

## Exporting Traces

### As Python dicts

```python
data = collector.export()
# Returns: list[dict] — one dict per top-level trace, with nested children
```

### As JSON

```python
import json

json_str = json.dumps(collector.export(), indent=2, default=str)
print(json_str)
```

### Export schema

Each trace dict contains:

```json
{
  "trace_id": "abc123...",
  "agent_name": "my_agent",
  "agent_type": "LlmAgent",
  "input_message": "What is Nono?",
  "output_message": "Nono is an AI framework...",
  "status": "success",
  "error": null,
  "provider": "google",
  "model": "gemini-3-flash-preview",
  "llm_calls": [
    {
      "provider": "google",
      "model": "gemini-3-flash-preview",
      "temperature": 0.7,
      "max_tokens": 1024,
      "input_tokens": 120,
      "output_tokens": 85,
      "total_tokens": 205,
      "duration_ms": 1500.0,
      "error": null
    }
  ],
  "tools_used": [
    {
      "tool_name": "search",
      "arguments": {"q": "Nono"},
      "result": "Found...",
      "duration_ms": 200.0,
      "error": null
    }
  ],
  "token_usage": {
    "input_tokens": 120,
    "output_tokens": 85,
    "total_tokens": 205
  },
  "duration_ms": 1800.0,
  "started_at": "2025-01-15T10:30:00+00:00",
  "ended_at": "2025-01-15T10:30:01.800000+00:00",
  "children": [],
  "parent_id": null,
  "metadata": {}
}
```

---

## Integration with Runner

Pass a `TraceCollector` to the `Runner` constructor:

```python
from nono.agent import Runner, TraceCollector

collector = TraceCollector()
runner = Runner(agent=my_agent, trace_collector=collector)

# All these methods are traced:
runner.run("question")           # Sync single-turn
runner.stream("question")        # Sync streaming (sub-agent traces via _run_impl_traced)
await runner.run_async("question")   # Async single-turn
async for event in runner.astream("question"):  # Async streaming
    ...
```

The collector persists across multiple `.run()` calls, accumulating traces:

```python
runner.run("first question")
runner.run("second question")
print(len(collector))  # 2
```

Use `collector.clear()` to reset.

---

## Integration with Orchestration Agents

All orchestration agents automatically forward the `trace_collector` to sub-agents:

| Agent | Trace behavior |
| --- | --- |
| `SequentialAgent` | Parent trace + 1 child per sub-agent (in order) |
| `ParallelAgent` | Parent trace + 1 child per sub-agent (concurrent) |
| `LoopAgent` | Parent trace + N children per iteration × sub-agents |
| `MapReduceAgent` | Parent trace + 1 child per mapper (concurrent) + 1 child for reducer |
| `ConsensusAgent` | Parent trace + 1 child per voter (concurrent) + 1 child for judge |
| `ProducerReviewerAgent` | Parent trace + 2 children per iteration (producer + reviewer) |
| `DebateAgent` | Parent trace + 1 child per debater per round (concurrent) + 1 child for judge |
| `EscalationAgent` | Parent trace + 1 child per attempted agent (sequential, stops on success) |
| `SupervisorAgent` | Parent trace + 1 child per delegation + 1 child per evaluation (iterative) |
| `VotingAgent` | Parent trace + 1 child per voter (concurrent) |
| `HandoffAgent` | Parent trace + 1 child per handoff step (sequential chain) |
| `GroupChatAgent` | Parent trace + 1 child per speaker turn (sequential rounds) |
| `HierarchicalAgent` | Parent trace + 1 child per department delegation + 1 child for synthesis (iterative) |
| `GuardrailAgent` | Parent trace + child for pre-validator + child for main agent + child for post-validator (with retry children) |
| `BestOfNAgent` | Parent trace + N children (one per parallel run) |
| `BatchAgent` | Parent trace + 1 child per item (concurrent) |
| `CascadeAgent` | Parent trace + 1 child per stage attempted (sequential, early-stop) |
| `TreeOfThoughtsAgent` | Parent trace + N children per depth level (concurrent branches) |
| `PlannerAgent` | Parent trace + 1 child per plan step (parallel where deps allow) + LLM planning/synthesis |
| `SubQuestionAgent` | Parent trace + 1 child per sub-question (concurrent) + LLM decomposition/synthesis |
| `ContextFilterAgent` | Parent trace + 1 child per sub-agent (sequential or parallel) |
| `ReflexionAgent` | Parent trace + 2 children per attempt (agent + evaluator, serial loop) |
| `SpeculativeAgent` | Parent trace + 1 child per sub-agent (concurrent race, early cancel) |
| `CircuitBreakerAgent` | Parent trace + 1 child for primary or fallback |
| `TournamentAgent` | Parent trace + 2 children per match (concurrent pairs) + 1 child for judge per match |
| `ShadowAgent` | Parent trace + 2 children (stable + shadow, concurrent) |
| `CompilerAgent` | Parent trace + 1 child per example eval + LLM optimisation per iteration |
| `CheckpointableAgent` | Parent trace + 1 child per sub-agent step (sequential, skips completed) |
| `DynamicFanOutAgent` | Parent trace + LLM decomposition + 1 child per work item (concurrent) + 1 child for reducer |
| `SwarmAgent` | Parent trace + 1 child per handoff step (sequential chain) |
| `MemoryConsolidationAgent` | Parent trace + optional summariser child + 1 child for main agent |
| `PriorityQueueAgent` | Parent trace + 1 child per agent (grouped by priority, concurrent within group) |
| `MonteCarloAgent` | Parent trace + 1 child per simulation rollout (sequential) |
| `GraphOfThoughtsAgent` | Parent trace + N children per round (branches) + optional aggregate child |
| `BlackboardAgent` | Parent trace + 1 child per iteration (expert activation) |
| `MixtureOfExpertsAgent` | Parent trace + 1 child per selected expert (top-k, concurrent in async) |
| `CoVeAgent` | Parent trace + 4 phase children (draft, plan, verify×N, revise) |
| `SagaAgent` | Parent trace + 1 child per action step + 1 child per compensator (reverse) |
| `LoadBalancerAgent` | Parent trace + 1 child for the selected agent |
| `EnsembleAgent` | Parent trace + 1 child per agent (concurrent) |
| `TimeoutAgent` | Parent trace + 1 child for wrapped agent (may be interrupted) |
| `AdaptivePlannerAgent` | Parent trace + 1 child per executed step + LLM planning calls |
| `RouterAgent` | Parent trace + child for the ephemeral orchestrated agent |
| `LlmAgent` (transfer) | Parent trace + child for the transferred-to agent |

---

## Integration with TaskExecutor

Pass `trace_collector` to `execute()` or `run_json_task()` to record the LLM call:

```python
from nono.tasker import TaskExecutor
from nono import TraceCollector

tasker = TaskExecutor(
    system_prompt="Classify this article into a category.",
    provider="google",
)

collector = TraceCollector()

# Direct execution
result = tasker.execute(
    "AI-powered diagnostics improve early cancer detection by 35%.",
    trace_collector=collector,
)

# JSON task file execution
result = tasker.run_json_task(
    "nono/tasker/prompts/classify.json",
    data_input="AI-powered diagnostics improve early cancer detection by 35%.",
    trace_collector=collector,
)

collector.print_summary()
# [success] TaskExecutor (TaskExecutor) — 820ms, 1 LLM call(s), ~312 tokens, 0 tool(s)
```

The trace records:

| Field | Value |
| --- | --- |
| `agent_name` | `"TaskExecutor"` |
| `agent_type` | `"TaskExecutor"` |
| `llm_calls` | 1 `LLMCall` with provider, model, token estimates, and timing |
| `status` | `SUCCESS` or `ERROR` (if the LLM call fails) |

Both `execute()` and `run_json_task()` accept `trace_collector`. When using `run_json_task()` with batching, each batch execution is traced individually.

---

## Integration with Workflows

Pass `trace_collector` as a **keyword-only** argument to `run()`, `run_async()`, `stream()`, or `astream()`:

```python
from nono.workflows import Workflow
from nono import TraceCollector

flow = Workflow("my_pipeline")
flow.step("preprocess", lambda s: {"clean": s["input"].strip()})
flow.step("analyze", lambda s: {"result": len(s["clean"])})

collector = TraceCollector()
result = flow.run(trace_collector=collector, input="  hello world  ")

collector.print_summary()
```

Output shows a **hierarchical** structure — the workflow as parent, steps as children:

```
[success] my_pipeline (Workflow) — 5ms, 0 LLM call(s), 0 tokens, 0 tool(s), 2 child(ren)
  [success] preprocess (WorkflowStep) — 1ms, 0 LLM call(s), 0 tokens, 0 tool(s)
  [success] analyze (WorkflowStep) — 1ms, 0 LLM call(s), 0 tokens, 0 tool(s)
```

**Streaming** builds the trace incrementally as each step completes:

```python
collector = TraceCollector()
for step_name, state in flow.stream(trace_collector=collector):
    print(f"  completed: {step_name}")

collector.print_summary()
```

**Branching** traces only the executed path:

```python
flow = Workflow("branched")
flow.step("check", lambda s: {"score": 90})
flow.step("approve", lambda s: {"status": "approved"})
flow.step("reject", lambda s: {"status": "rejected"})
flow.branch_if("check", lambda s: s["score"] >= 80, then="approve", otherwise="reject")

collector = TraceCollector()
result = flow.run(trace_collector=collector)
# Only "check" and "approve" appear in traces — "reject" is skipped
```

**Error handling** — if a step raises, the workflow trace is marked `ERROR`:

```python
collector = TraceCollector()
try:
    flow.run(trace_collector=collector)
except Exception:
    pass

trace = collector.traces[0]
print(trace.status)  # TraceStatus.ERROR
```

---

## Integration with CodeExecuter

Pass `trace_collector` to `run()` to record code generation and execution:

```python
from nono.executer import CodeExecuter
from nono import TraceCollector

executer = CodeExecuter(provider="google")

collector = TraceCollector()
result = executer.run(
    "Calculate the factorial of 10",
    trace_collector=collector,
)

collector.print_summary()
# [success] CodeExecuter (CodeExecuter) — 1500ms, 0 LLM call(s), 0 tokens, 0 tool(s)
```

The trace includes metadata about the security mode:

```python
trace = collector.traces[0]
print(trace.metadata)  # {"security_mode": "restricted"}
```

---

## Cross-Module Tracing

Use a **single `TraceCollector`** across agents, workflows, and task executors for unified observability:

```python
from nono.agent import Agent, SequentialAgent, Runner
from nono.workflows import Workflow, agent_node, tasker_node
from nono import TraceCollector

# Build agents
researcher = Agent(name="researcher", provider="google",
                   instruction="Research AI healthcare trends.")
writer = Agent(name="writer", provider="google",
               instruction="Write a healthcare AI article.")
pipeline = SequentialAgent(name="article_team", sub_agents=[researcher, writer])

# Build workflow with mixed node types
flow = Workflow("publish_pipeline")
flow.step("draft", agent_node(pipeline, input_key="topic", output_key="draft"))
flow.step("quality", tasker_node(
    system_prompt="Rate article quality 0-100. Return JSON: {\"score\": N}",
    input_key="draft", output_key="quality",
))
flow.step("decide", lambda s: {"approved": int(s.get("quality", "0")) >= 80})
flow.connect("draft", "quality", "decide")

# Single collector captures everything
collector = TraceCollector()
result = flow.run(trace_collector=collector, topic="AI diagnostics 2026")

collector.print_summary()
print(f"Total tokens: {collector.total_tokens}")
print(f"Total LLM calls: {collector.total_llm_calls}")
```

---

## Observability Summary

Complete coverage matrix of all I/O recording mechanisms across Nono modules:

| Mechanism | Agents | Workflows | Taskers | CodeExecuter |
| --- | --- | --- | --- | --- |
| **TraceCollector** | Full (nested, auto) | Full (parent + step children) | Full (`execute` + `run_json_task`) | Full |
| **Session / Events** | Full (7 event types) | N/A | N/A | N/A |
| **Lifecycle Callbacks** | 4 hooks (`before/after_agent`, `before/after_tool`) | 2 hooks (`on_before_step`, `on_after_step`) | N/A | N/A |
| **@event_log / msg_log** | N/A (uses Events + Traces) | Entry/exit logging | Entry/exit logging | Entry/exit logging |

### What each mechanism captures

| Mechanism | Prompt/Input | LLM Response | Token Usage | Timing | Errors | Tool I/O |
| --- | --- | --- | --- | --- | --- | --- |
| **TraceCollector** | Yes (truncated) | Yes (truncated) | Yes (estimated) | Yes (ms) | Yes | Yes |
| **Session Events** | Yes (full) | Yes (full) | No | Yes (timestamp) | Yes | Yes (args + result) |
| **Callbacks** | Yes (full access) | Yes (full access) | No | No | No | Yes (args + result) |
| **@event_log** | No | No | No | Yes (entry/exit) | Yes | No |

### Choosing the right mechanism

| Need | Recommended |
| --- | --- |
| Performance metrics, cost tracking | `TraceCollector` |
| Full conversation history / audit trail | `Session.events` |
| Intercept and modify I/O at runtime | Agent callbacks / Workflow `on_before_step` |
| Simple function entry/exit monitoring | `@event_log` |
| Unified cross-module observability | Single `TraceCollector` passed everywhere |

### API Server tracing

All three API endpoints support `trace: true` in the request body to include trace data in the response:

```bash
# Agent with tracing
curl -X POST http://localhost:8000/agent/summarizer \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize this text", "trace": true}'

# Task with tracing
curl -X POST http://localhost:8000/task/classify \
  -H "Content-Type: application/json" \
  -d '{"data_input": "AI in healthcare", "trace": true}'

# Workflow with tracing
curl -X POST http://localhost:8000/workflow/sentiment_pipeline \
  -H "Content-Type: application/json" \
  -d '{"state": {"input": "Great product!"}, "trace": true}'
```

The response includes a `trace` field with the full trace tree when `trace: true`.

---

## Examples

### Basic: single agent with tracing

```python
from nono.agent import Agent, Runner, TraceCollector

agent = Agent(
    name="assistant",
    provider="google",
    model="gemini-3-flash-preview",
    instruction="You are a helpful assistant.",
)

collector = TraceCollector()
runner = Runner(agent=agent, trace_collector=collector)
response = runner.run("What is the capital of France?")

print(response)
collector.print_summary()
print(f"\nTotal tokens used: {collector.total_tokens}")
```

### Pipeline with tool tracking

```python
from nono.agent import (
    Agent, SequentialAgent, Runner, TraceCollector, tool,
)

@tool(description="Search the web")
def search_web(query: str) -> str:
    return f"Results for: {query}"

researcher = Agent(
    name="researcher",
    provider="google",
    instruction="Research the topic using the search tool.",
    tools=[search_web],
)
writer = Agent(
    name="writer",
    provider="google",
    instruction="Write a summary based on the research.",
)

pipeline = SequentialAgent(name="pipeline", sub_agents=[researcher, writer])

collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
runner.run("Latest AI trends")

# Detailed inspection
for trace in collector.traces:
    print(trace.summary())
    for child in trace.children:
        print(f"  {child.summary()}")
        for t in child.tools_used:
            print(f"    Tool: {t.tool_name}({t.arguments}) → {t.result[:50]}")

# Export for dashboard/logging
import json
print(json.dumps(collector.export(), indent=2, default=str))
```

### Multi-turn with accumulated traces

```python
collector = TraceCollector()
runner = Runner(agent=agent, trace_collector=collector)

runner.run("What is Python?")
runner.run("How does it compare to JavaScript?")
runner.run("Which should I learn first?")

print(f"3 turns: {len(collector)} traces, {collector.total_tokens} total tokens")
collector.print_summary()
```

---

## API Reference

### Classes

| Class | Module | Also importable from | Description |
| --- | --- | --- | --- |
| `Event` | `nono.agent.base` | `nono.agent` | Immutable record of agent action |
| `EventType` | `nono.agent.base` | `nono.agent` | Enum of event types |
| `TraceCollector` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Manages trace lifecycle and aggregation |
| `Trace` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Structured record of a single invocation |
| `LLMCall` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Record of a single LLM completion request |
| `ToolRecord` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Record of a single tool invocation |
| `TokenUsage` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Estimated token counts (input, output, total) |
| `TraceStatus` | `nono.agent.tracing` | `nono`, `nono.tracing`, `nono.agent` | Enum: RUNNING, SUCCESS, ERROR |
| `BeforeStepCallback` | `nono.workflows.workflow` | `nono.workflows` | Type alias for workflow before-step hooks |
| `AfterStepCallback` | `nono.workflows.workflow` | `nono.workflows` | Type alias for workflow after-step hooks |
| `HumanInputAgent` | `nono.agent.human_input` | `nono.agent` | Agent that pauses for human input |
| `HumanInputResponse` | `nono.hitl` | `nono` | Dataclass returned by human-input handlers |
| `HumanInputHandler` | `nono.hitl` | `nono` | Sync handler type alias |
| `AsyncHumanInputHandler` | `nono.hitl` | — | Async handler type alias |
| `HumanRejectError` | `nono.hitl` | `nono` | Raised when human rejects with no reject branch |

### Recommended import (cross-module):

```python
from nono import TraceCollector, Trace, TraceStatus, LLMCall, TokenUsage, ToolRecord
```

### All classes are also importable from `nono.agent`:

```python
from nono.agent import (
    # Events
    Event, EventType,
    # Tracing
    TraceCollector, Trace, TraceStatus,
    LLMCall, ToolRecord, TokenUsage,
)
```
