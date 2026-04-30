# State Introspection Guide

> How to query states, tasks, dependencies, transitions, events, and traces — during and after execution.

## Table of Contents

- [Overview](#overview)
- [Real-Time Introspection (During Execution)](#real-time-introspection-during-execution)
  - [Workflow Streaming](#workflow-streaming)
  - [Agent Event Streaming](#agent-event-streaming)
  - [Lifecycle Callbacks](#lifecycle-callbacks)
- [Post-Execution Introspection (After Execution)](#post-execution-introspection-after-execution)
  - [Workflow State and Transitions](#workflow-state-and-transitions)
  - [Time-Travel API](#time-travel-api)
  - [Execution History](#execution-history)
  - [Graph Introspection](#graph-introspection)
  - [Checkpoints (Persistent State)](#checkpoints-persistent-state)
  - [Session Events and State](#session-events-and-state)
  - [Tracing and Observability](#tracing-and-observability)
- [REST API Introspection (External Tools)](#rest-api-introspection-external-tools)
  - [Workflow Introspection via REST](#workflow-introspection-via-rest)
  - [Agent Introspection via REST](#agent-introspection-via-rest)
  - [REST API Quick Reference](#rest-api-quick-reference)
- [Quick Reference](#quick-reference)
- [See Also](#see-also)

---

## Overview

Nono exposes **all** execution data through explicit APIs — nothing is hidden. You can query states, tasks, dependencies, transitions, events, and performance traces both in real-time and after execution completes.

| When | Mechanism | Granularity |
|------|-----------|-------------|
| **During** execution | `stream()` / `astream()` | Per-step (Workflow) or per-event (Agent) |
| **During** execution | Lifecycle callbacks | Per-step or per-agent invocation |
| **After** execution | Properties and methods | Full audit trail, time-travel, graph |

---

## Real-Time Introspection (During Execution)

### Workflow Streaming

`stream()` and `astream()` yield `(step_name, state_snapshot)` after each step completes:

```python
from nono.workflows import Workflow

flow = Workflow("pipeline")
flow.step("fetch", lambda s: {"data": [1, 2, 3]})
flow.step("process", lambda s: {"total": sum(s["data"])})
flow.connect("fetch", "process")

for step_name, state in flow.stream():
    print(f"[{step_name}] state = {state}")
# [fetch] state = {'data': [1, 2, 3]}
# [process] state = {'data': [1, 2, 3], 'total': 6}
```

Async variant:

```python
async for step_name, state in flow.astream(topic="AI"):
    print(f"[{step_name}] {state}")
```

### Agent Event Streaming

`Runner.stream()` and `Runner.astream()` yield `Event` objects as they happen:

```python
from nono.agent import Runner, EventType

runner = Runner(my_agent)

for event in runner.stream("Explain quantum computing"):
    if event.event_type == EventType.TOOL_CALL:
        print(f"  Tool: {event.data['tool']}")
    elif event.event_type == EventType.AGENT_MESSAGE:
        print(f"  Response: {event.content[:80]}...")
    elif event.event_type == EventType.STATE_UPDATE:
        print(f"  State changed: {event.data}")
```

All 9 event types are available for filtering:

| EventType | Description |
|-----------|-------------|
| `USER_MESSAGE` | User input recorded |
| `AGENT_MESSAGE` | Agent text response |
| `TOOL_CALL` | Tool invocation initiated |
| `TOOL_RESULT` | Tool returned a result |
| `STATE_UPDATE` | Session state modified |
| `AGENT_TRANSFER` | Control transferred to another agent |
| `HUMAN_INPUT_REQUEST` | Awaiting human input |
| `HUMAN_INPUT_RESPONSE` | Human responded |
| `ERROR` | Error occurred |

### Lifecycle Callbacks

#### Workflow Callbacks

Intercept execution at fine-grained points:

```python
flow = Workflow("traced")
flow.step("a", lambda s: {"x": 1})
flow.step("b", lambda s: {"y": 2})
flow.connect("a", "b")

# Before each step
flow.on_before_step(lambda name, state: print(f"  >> Starting {name}"))

# After each step
flow.on_after_step(lambda name, state: print(f"  << Finished {name}: {state}"))

# Between steps
flow.on_between_steps(lambda prev, next_, state: print(f"  {prev} -> {next_}"))

flow.run()
```

#### Agent Callbacks

```python
from nono.agent import Agent

agent = Agent(
    name="my_agent",
    model="gemini-2.0-flash",
    before_agent=lambda agent, ctx: print(f"Input: {ctx.user_message}"),
    after_agent=lambda agent, ctx, resp: print(f"Output: {resp[:50]}..."),
    before_tool=lambda agent, ctx, name, args: print(f"Calling {name}"),
    after_tool=lambda agent, ctx, name, args, result: print(f"{name} returned"),
)
```

---

## Post-Execution Introspection (After Execution)

### Workflow State and Transitions

After `run()` or `stream()`, the full audit trail is available:

```python
flow = Workflow("analysis")
flow.step("fetch", lambda s: {"rows": 100})
flow.step("clean", lambda s: {"rows": 95, "dropped": 5})
flow.step("analyze", lambda s: {"score": 0.87})
flow.connect("fetch", "clean", "analyze")

result = flow.run()

# Final state
print(result)
# {'rows': 95, 'dropped': 5, 'score': 0.87}

# Audit trail — one StateTransition per step
for t in flow.transitions:
    print(
        f"{t.step}: "
        f"keys={t.keys_changed} "
        f"duration={t.duration_ms:.1f}ms "
        f"retries={t.retries} "
        f"error={t.error}"
    )
# fetch: keys=frozenset({'rows'}) duration=0.1ms retries=0 error=None
# clean: keys=frozenset({'rows', 'dropped'}) duration=0.1ms retries=0 error=None
# analyze: keys=frozenset({'score'}) duration=0.1ms retries=0 error=None
```

`StateTransition` fields:

| Field | Type | Description |
|-------|------|-------------|
| `step` | `str` | Step name |
| `keys_changed` | `frozenset[str]` | State keys added or modified |
| `branch_taken` | `str \| None` | Next step chosen by a branch |
| `duration_ms` | `float` | Wall-clock time in milliseconds |
| `retries` | `int` | Retry attempts (0 = first-try success) |
| `error` | `str \| None` | Error message if step failed |
| `state_snapshot` | `dict` | Full state copy after the step |

### Time-Travel API

Query the exact state at any past step — no disk access required:

```python
# State after a specific step
state_at_clean = flow.get_state_at("clean")
print(state_at_clean)
# {'rows': 95, 'dropped': 5}

# Returns None if the step was not recorded
state_at_missing = flow.get_state_at("nonexistent")
# None
```

### Execution History

Get the complete step-by-step history as `(step_name, state)` tuples:

```python
for step_name, snapshot in flow.get_history():
    print(f"{step_name}: {snapshot}")
# fetch: {'rows': 100}
# clean: {'rows': 95, 'dropped': 5}
# analyze: {'rows': 95, 'dropped': 5, 'score': 0.87}
```

#### Replay with "What-If" Overrides

Re-execute from any step with modified state:

```python
# What if we had 200 rows instead of 100?
new_result = flow.replay_from("fetch", rows=200)
print(new_result)
```

### Graph Introspection

Query the workflow structure — steps, edges, and branches:

```python
# Ordered list of step names
print(flow.steps)
# ['fetch', 'clean', 'analyze']

# Human-readable graph description
print(flow.describe())
# Workflow: analysis
# Steps (3):
#   0. fetch (entry)
#   1. clean
#   2. analyze
# Edges:
#   fetch -> clean
#   clean -> analyze

# ASCII diagram
print(flow.draw())

# State schema (if defined)
print(flow.schema)
```

### Checkpoints (Persistent State)

When checkpointing is enabled, state is persisted to disk after each step:

```python
flow = Workflow("pipeline")
flow.step("a", lambda s: {"x": 1})
flow.step("b", lambda s: {"y": 2})
flow.connect("a", "b")

# Enable checkpointing
flow.enable_checkpoint("./checkpoints")

flow.run()

# List all checkpoints on disk
for index, step_name in flow.list_checkpoints():
    print(f"  [{index}] {step_name}")
# [0] a
# [1] b

# Load a specific checkpoint
state = flow.get_checkpoint_at("a")
print(state)
# {'x': 1}

# Resume from last checkpoint after crash
result = flow.run(resume=True)
```

### Session Events and State

After agent execution, the full event history and state are available:

```python
runner = Runner(my_agent)
response = runner.run("What is the capital of Spain?")

# All events in chronological order
for event in runner.session.events:
    print(f"[{event.event_type.name}] {event.author}: {event.content}")
# [USER_MESSAGE] user: What is the capital of Spain?
# [AGENT_MESSAGE] my_agent: The capital of Spain is Madrid.

# Most recent event
print(runner.session.last_event)

# Session state (mutable dict)
print(runner.session.state)

# Thread-safe state access
runner.session.state_set("counter", 42)
val = runner.session.state_get("counter", default=0)

# Connector message format
messages = runner.session.get_messages()
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Session metadata
print(runner.session.session_id)
print(runner.session.created_at)
```

### Tracing and Observability

`TraceCollector` captures structured performance data across all modules:

```python
from nono.tracing import TraceCollector

collector = TraceCollector()

# Pass to any execution method
runner = Runner(my_agent, trace_collector=collector)
runner.run("Hello")

# Or with workflows
result = flow.run(trace_collector=collector)

# Inspect traces
for trace in collector.traces:
    print(trace.summary())
# [OK] my_agent (LlmAgent) — 1200ms, 1 LLM call(s), 350 tokens, 0 tool(s), 0 child(ren)

# Aggregate metrics
print(f"Total tokens: {collector.total_tokens}")
print(f"Total LLM calls: {collector.total_llm_calls}")
print(f"Total tool calls: {collector.total_tool_calls}")

# Human-readable summary
collector.print_summary()

# Export as serializable dicts (for JSON, logging, dashboards)
data = collector.export()
import json
print(json.dumps(data, indent=2))

# Per-trace detail
trace = collector.traces[0]
print(trace.to_dict())  # Full nested dict with LLM calls, tools, children
```

`Trace.to_dict()` output includes:

| Key | Content |
|-----|---------|
| `trace_id` | Unique identifier |
| `agent_name` / `agent_type` | Agent metadata |
| `input_message` / `output_message` | I/O |
| `status` | `"ok"` or `"error"` |
| `llm_calls` | List of LLM call records (provider, model, tokens, duration) |
| `tools_used` | List of tool records (name, args, result, duration) |
| `token_usage` | Input/output/total token counts |
| `duration_ms` | Total execution time |
| `children` | Nested sub-agent traces |

---

## REST API Introspection (External Tools)

All introspection data is also available via the Nono API Server (`nono.server`), enabling dashboards, monitoring tools, and external orchestrators to query execution details over HTTP.

### Workflow Introspection via REST

#### Run with introspection

Set `"introspect": true` in the request body to receive transitions, state history, and graph description alongside the result:

```bash
curl -X POST http://localhost:8000/workflow/sentiment_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"input": "I love this product!"},
    "introspect": true
  }'
```

Response includes:

```json
{
  "result": {"sentiment": "positive", "score": 0.95},
  "workflow": "sentiment_pipeline",
  "steps_executed": ["analyze", "score", "format"],
  "duration_ms": 1234.5,
  "introspection": {
    "transitions": [
      {
        "step": "analyze",
        "keys_changed": ["sentiment"],
        "branch_taken": null,
        "duration_ms": 800.12,
        "retries": 0,
        "error": null
      }
    ],
    "state_history": [
      {"step": "analyze", "state": {"input": "...", "sentiment": "positive"}}
    ],
    "graph": "Workflow: sentiment_pipeline\nSteps (3):\n  0. analyze (entry)\n  ...",
    "ascii_diagram": "..."
  }
}
```

#### Describe workflow graph (no execution)

```bash
curl http://localhost:8000/workflow/sentiment_pipeline/describe
```

Returns the graph structure, steps, ASCII diagram, and schema — without running the workflow:

```json
{
  "workflow": "sentiment_pipeline",
  "steps": ["analyze", "score", "format"],
  "graph": "Workflow: sentiment_pipeline\nSteps (3): ...",
  "ascii_diagram": "...",
  "schema": {
    "fields": {"score": "float", "sentiment": "str"},
    "reducers": []
  }
}
```

### Agent Introspection via REST

#### Run with introspection

```bash
curl -X POST http://localhost:8000/agent/summarizer \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize this document...",
    "introspect": true
  }'
```

Response includes session events and final state:

```json
{
  "result": "The document discusses...",
  "agent": "summarizer",
  "duration_ms": 2100.3,
  "introspection": {
    "events": [
      {
        "event_type": "user_message",
        "author": "user",
        "content": "Summarize this document...",
        "timestamp": "2026-04-21T10:30:00+00:00",
        "event_id": "a1b2c3d4"
      },
      {
        "event_type": "agent_message",
        "author": "summarizer",
        "content": "The document discusses...",
        "timestamp": "2026-04-21T10:30:02+00:00",
        "event_id": "e5f6g7h8"
      }
    ],
    "state": {},
    "event_count": 2
  }
}
```

#### Stream events in real-time (SSE)

```bash
curl -X POST http://localhost:8000/agent/summarizer/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing"}'
```

Each event arrives as a Server-Sent Event:

```
data: {"agent": "summarizer", "type": "user_message", "content": "...", "timestamp": "..."}
data: {"agent": "summarizer", "type": "agent_message", "content": "...", "timestamp": "..."}
data: [DONE]
```

### Combining trace + introspect

Both flags can be used together for maximum observability:

```bash
curl -X POST http://localhost:8000/workflow/content_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"topic": "AI safety"},
    "trace": true,
    "introspect": true
  }'
```

### REST API Quick Reference

| Endpoint | Method | Introspection Data |
|----------|--------|-------------------|
| `/workflow/{name}` | POST | `introspect=true` → transitions, state history, graph |
| `/workflow/{name}/describe` | GET | Graph structure, steps, schema (no execution) |
| `/agent/{name}` | POST | `introspect=true` → session events, state |
| `/agent/{name}/stream` | POST | Real-time SSE events |
| `/task/{name}` | POST | `trace=true` → LLM calls, tools, tokens |

---

## Quick Reference

| What | Python API | REST API |
|------|-----------|----------|
| **Workflow state** | `flow.run()` / `stream()` | `POST /workflow/{name}` → `result` |
| **State at step N** | `flow.get_state_at("step")` | `introspect=true` → `state_history` |
| **Full history** | `flow.get_history()` | `introspect=true` → `state_history` |
| **Replay / what-if** | `flow.replay_from("step", **overrides)` | — (Python only) |
| **Transitions** | `flow.transitions` | `introspect=true` → `transitions` |
| **Steps list** | `flow.steps` | `GET /workflow/{name}/describe` → `steps` |
| **Graph description** | `flow.describe()` / `flow.draw()` | `GET /workflow/{name}/describe` |
| **Checkpoints** | `flow.list_checkpoints()` | — (Python only) |
| **Agent events** | `runner.session.events` | `introspect=true` → `events` / SSE stream |
| **Session state** | `runner.session.state` | `introspect=true` → `state` |
| **Traces** | `collector.traces` / `export()` | `trace=true` → `trace` |
| **Token totals** | `collector.total_tokens` | `trace=true` → trace data |
| **Callbacks** | `on_before_step`, `before_agent` | — (Python only) |

---

## See Also

- [Events and Tracing](agent/README_events_tracking.md) — Full event system and tracing reference
- [Workflows](workflows/README_workflow.md) — DAG engine, schemas, reducers, and checkpointing
- [Agent](agent/README_agent.md) — `LlmAgent`, `Runner`, sessions, and tools
- [Progressive Disclosure](README_progressive_disclosure.md) — Complexity levels and architecture overview
- [Hooks](README_hooks.md) — 6 hook types × 15 lifecycle events
