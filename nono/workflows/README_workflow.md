# Workflow - Multi-step Execution Pipeline with Conditional Branching

> Standalone workflow engine for building directed graphs of operations. Each step receives a shared state (`dict`) and returns updates. Supports synchronous functions, async coroutines, and AI-powered callables through Nono's connector. Inspired by the LangFrame API, without LangGraph dependency.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [Quickstart](#quickstart)
- [Key Concepts](#key-concepts)
- [API by Categories](#api-by-categories)
  - [Step registration](#step-registration)
  - [Node factories](#node-factories)
  - [Connection and branching](#connection-and-branching)
  - [Dynamic manipulation](#dynamic-manipulation)
  - [Execution](#execution)
  - [Checkpointing](#checkpointing)
  - [Introspection](#introspection)
- [Function Index](#function-index)
- [Detailed Reference](#detailed-reference)
- [Branch Expressions](#branch-expressions)
- [Exceptions](#exceptions)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [Error Recovery and Retry](#error-recovery-and-retry)
- [State Transition Audit Trail](#state-transition-audit-trail)
- [State Schema and Reducers](#state-schema-and-reducers)
- [Lifecycle Hooks](#lifecycle-hooks)
- [Deterministic Orchestration](#deterministic-orchestration)
- [Declarative Workflows](#declarative-workflows)
- [Integration with Nono Connector](#integration-with-nono-connector)
- [Advanced Examples](#advanced-examples)
- [Workflow vs Agent Orchestration](#workflow-vs-agent-orchestration)


## Overview

`Workflow` allows you to define pipelines as directed graphs of functions. The engine executes steps in order, propagates a state dictionary between them, and supports:

| Feature | Description |
|:---|:---|
| **Fluent API** | All methods return `self` for chaining calls |
| **Linear execution** | Without explicit edges, steps run in registration order |
| **Explicit edges** | `connect("a", "b", "c")` defines the execution graph |
| **Conditional branching** | Dynamic routing with callables or string expressions |
| **Parallel step** | `parallel_step()` fans-out to concurrent functions and merges results |
| **Loop step** | `loop_step()` repeats a function while a condition holds |
| **Join node** | `join()` explicit wait-for-all barrier before continuing |
| **Dynamic manipulation** | Insert, remove, reorder, and replace steps at runtime |
| **Streaming** | `stream()` yields `(step_name, state)` after each step |
| **Async** | `run_async()` and `astream()` for coroutines |
| **Cycle detection** | Prevents infinite loops in pipelines without branches |
| **Human-in-the-Loop** | Pause execution for human approval with `human_step()` / `human_node()` |
| **Checkpointing** | Persist state to disk after each step; resume from last checkpoint |
| **Declarative loading** | Build workflows from JSON or YAML files with `load_workflow()` |
| **Error recovery** | Per-step `on_error` routing to fallback steps on failure |
| **Retry** | Per-step `retry` count for transient failure handling |
| **Audit trail** | `flow.transitions` records a `StateTransition` per step execution |
| **State schema** | Optional `StateSchema` with type validation and reducer functions |
| **Lifecycle hooks** | `on_before_step`, `on_after_step`, `on_between_steps`, `on_step_executing` for fine-grained control |
| **Logging** | Integrated with `logging.getLogger("Nono.Workflow")` |


## Module Structure

```
nono/workflows/
├── __init__.py       # Exports: Workflow, END, DEFAULT_STEP_RETRIES, WorkflowError, StepNotFoundError, DuplicateStepError, StepTimeoutError, JoinPredecessorError, StateTransition, StateSchema, ReducerFn, BeforeStepCallback, AfterStepCallback, BetweenStepsCallback, StepExecutingCallback, StepExecutedCallback, OnStartCallback, OnEndCallback, tasker_node, agent_node, human_node, load_workflow
└── workflow.py       # Workflow engine + node factory functions + declarative loader
```


## Quickstart

### Linear pipeline (no edges)

```python
from nono.workflows import Workflow

def research(state):
    return {"notes": f"Research about {state['topic']}"}

def write(state):
    return {"draft": f"Article based on: {state['notes']}"}

flow = Workflow("article")
flow.step("research", research)
flow.step("write", write)

result = flow.run(topic="Generative AI")
print(result["draft"])
# Article based on: Research about Generative AI
```

### Pipeline with explicit edges

```python
flow = Workflow("pipeline")
flow.step("a", lambda s: {"x": 1})
flow.step("b", lambda s: {"y": s["x"] + 10})
flow.step("c", lambda s: {"z": s["y"] * 2})
flow.connect("a", "b", "c")

result = flow.run()
# {'x': 1, 'y': 11, 'z': 22}
```

### Pipeline with branching

```python
from nono.workflows import Workflow

flow = Workflow("review")
flow.step("score", lambda s: {"quality": len(s["text"]) / 100})
flow.step("approve", lambda s: {"decision": "approved"})
flow.step("revise", lambda s: {"decision": "needs_revision"})

flow.branch_if("score", lambda s: s["quality"] > 0.5, then="approve", otherwise="revise")

result = flow.run(text="A" * 80)
print(result["decision"])  # approved
```

### Pipeline with TaskExecutor (tasker_node)

```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("classify")
flow.step("classify", tasker_node(
    system_prompt="Classify sentiment as positive or negative.",
    input_key="text",
    output_key="sentiment",
))
result = flow.run(text="I love this product!")
print(result["sentiment"])
```

### Pipeline with Agent (agent_node)

```python
from nono.workflows import Workflow, agent_node
from nono.agent import Agent

writer = Agent(name="writer", instruction="Write a blog post.", provider="google")

flow = Workflow("blog")
flow.step("write", agent_node(writer, input_key="topic", output_key="draft"))
result = flow.run(topic="AI trends 2026")
print(result["draft"])
```

### Mixed pipeline: function + tasker_node + agent_node

```python
from nono.workflows import Workflow, tasker_node, agent_node
from nono.agent import Agent

def gather(state):
    return {"research": f"Key findings about {state['topic']}"}

reviewer = Agent(name="reviewer", instruction="Review and polish.", provider="google")

flow = Workflow("article")
flow.step("gather", gather)                                         # function
flow.step("summarise", tasker_node(                                  # tasker
    system_prompt="Summarise in 3 sentences.",
    input_key="research", output_key="summary"))
flow.step("review", agent_node(reviewer,                             # agent
    input_key="summary", output_key="article"))
flow.connect("gather", "summarise", "review")

result = flow.run(topic="GenAI")
# gather → summarise → review (each output feeds the next input)
```


## Key Concepts

### State

The state is a `dict` passed to each step and updated with the result. Each step function receives the full state and returns a `dict` with the keys to update (merge, not full replacement).

```
Initial state: {"topic": "AI"}
     │
     ▼
  research(state) → {"notes": "..."}
     │
     ▼
State: {"topic": "AI", "notes": "..."}
     │
     ▼
  write(state) → {"draft": "..."}
     │
     ▼
Final state: {"topic": "AI", "notes": "...", "draft": "..."}
```

### Execution Order

| Mode | Behavior |
|:---|:---|
| No edges or branches | Steps run in registration order (`step()`) |
| With `connect()` | Follows the edge chain from the first step |
| With `branch()` / `branch_if()` | After the branched step, the condition is evaluated to determine the next step |
| Mixed | Branches take priority over edges; edges over registration order |

### END Constant

Import `END` and use it in branch functions to terminate the workflow early:

```python
from nono.workflows import Workflow, END

flow.branch("validate", lambda s: END if s["error"] else "process")
```


## API by Categories

### Step registration

| Method | Description |
|:---|:---|
| [`step(name, fn)`](#step) | Register a step at the end of the pipeline |
| [`step(name, fn, retry=, on_error=)`](#step) | Register a step with retry and/or error recovery |
| [`human_step(name, handler=, prompt=, ...)`](#human_step) | Register a human-in-the-loop checkpoint |
| [`parallel_step(name, fns, max_workers=)`](#parallel_step) | Register a parallel fan-out/join step |
| [`loop_step(name, fn, condition=, max_iterations=)`](#loop_step) | Register a deterministic loop step |
| [`join(name, wait_for=, reducer=, strict=)`](#join) | Register an explicit join (wait-for-all) node |
| [`insert_at(index, name, fn)`](#insert_at) | Insert a step at a specific position (0-based) |
| [`insert_before(ref, name, fn)`](#insert_before) | Insert before an existing step |
| [`insert_after(ref, name, fn)`](#insert_after) | Insert after an existing step |

### Node factories

Module-level factory functions that return `Callable[[dict], dict]` for use with `step()`.
Since they produce regular callables, they share all manipulation methods (insert, replace, swap, etc.).

| Function | Description |
|:---|:---|
| [`tasker_node(...)`](#tasker_node) | Create a callable backed by `TaskExecutor` |
| [`agent_node(agent, ...)`](#agent_node) | Create a callable that runs a Nono `Agent` |
| [`human_node(handler=, ...)`](#human_node) | Create a callable that pauses for human input |
| [`load_workflow(path, step_registry=)`](#load_workflow) | Load a Workflow from a JSON or YAML file |

### Connection and branching

| Method | Description |
|:---|:---|
| [`connect(*names)`](#connect) | Connect steps in sequence: a→b→c |
| [`branch(from_step, condition)`](#branch) | Conditional routing with a callable |
| [`branch_if(from_step, expr, then, otherwise)`](#branch_if) | Routing with a string expression |

### Dynamic manipulation

| Method | Description |
|:---|:---|
| [`remove_step(name)`](#remove_step) | Remove a step and its edges/branches |
| [`replace_step(name, fn)`](#replace_step) | Replace function keeping position |
| [`move_to(name, index)`](#move_to) | Move a step to a specific position |
| [`move_before(ref, name)`](#move_before) | Move a step before another |
| [`move_after(ref, name)`](#move_after) | Move a step after another |
| [`swap_steps(a, b)`](#swap_steps) | Swap the position of two steps |

### Execution

| Method | Description |
|:---|:---|
| [`run(**state)`](#run) | Run the complete workflow (sync) |
| [`run(resume=True)`](#run) | Resume from the last checkpoint |
| [`run_async(**state)`](#run_async) | Run the complete workflow (async) |
| [`stream(**state)`](#stream) | Run with yield per step (sync) |
| [`astream(**state)`](#astream) | Run with yield per step (async) |

### Checkpointing

| Method | Description |
|:---|:---|
| [`enable_checkpoints(directory)`](#enable_checkpoints) | Enable automatic state persistence |
| [`checkpoint(state, step_name)`](#checkpoint) | Manually persist state to disk |
| [`resume()`](#resume) | Load last checkpoint (state + step) |

### Lifecycle hooks

| Method | Description |
|:---|:---|
| [`on_start(callback)`](#on_start) | Register a callback invoked when the workflow starts |
| [`on_end(callback)`](#on_end) | Register a callback invoked when the workflow finishes |
| [`on_before_step(callback)`](#on_before_step) | Register a callback invoked before each step |
| [`on_after_step(callback)`](#on_after_step) | Register a callback invoked after each step |
| [`on_between_steps(callback)`](#on_between_steps) | Register a callback invoked between steps; return `False` to halt |
| [`on_step_executing(callback)`](#on_step_executing) | Register a callback invoked before each execution attempt (including retries) |
| [`on_step_executed(callback)`](#on_step_executed) | Register a callback invoked after each execution attempt (including retries) |

### Introspection

| Property / Method | Description |
|:---|:---|
| `name` | Workflow name |
| `steps` | Ordered list of step names |
| `len(flow)` | Number of steps |
| `"step" in flow` | Check if a step exists |
| `transitions` | List of `StateTransition` records from the last run |
| `schema` | Attached `StateSchema` or `None` |
| [`describe()`](#describe) | Human-readable graph summary |


## Function Index

| Function | Category |
|:---|:---|
| [`agent_node()`](#agent_node) | Node factory |
| [`astream()`](#astream) | Execution |
| [`checkpoint()`](#checkpoint) | Checkpointing |
| [`enable_checkpoints()`](#enable_checkpoints) | Checkpointing |
| [`human_node()`](#human_node) | Node factory |
| [`human_step()`](#human_step) | Registration |
| [`branch()`](#branch) | Connection |
| [`branch_if()`](#branch_if) | Connection |
| [`connect()`](#connect) | Connection |
| [`describe()`](#describe) | Introspection |
| [`insert_after()`](#insert_after) | Registration |
| [`insert_at()`](#insert_at) | Registration |
| [`insert_before()`](#insert_before) | Registration |
| [`join()`](#join) | Registration |
| [`load_workflow()`](#load_workflow) | Declarative loading |
| [`loop_step()`](#loop_step) | Registration |
| [`move_after()`](#move_after) | Manipulation |
| [`move_before()`](#move_before) | Manipulation |
| [`move_to()`](#move_to) | Manipulation |
| [`parallel_step()`](#parallel_step) | Registration |
| [`remove_step()`](#remove_step) | Manipulation |
| [`replace_step()`](#replace_step) | Manipulation |
| [`resume()`](#resume) | Checkpointing |
| [`run()`](#run) | Execution |
| [`run_async()`](#run_async) | Execution |
| [`step()`](#step) | Registration |
| [`stream()`](#stream) | Execution |
| [`swap_steps()`](#swap_steps) | Manipulation |
| [`tasker_node()`](#tasker_node) | Node factory |
| [`on_after_step()`](#on_after_step) | Lifecycle hooks |
| [`on_before_step()`](#on_before_step) | Lifecycle hooks |
| [`on_between_steps()`](#on_between_steps) | Lifecycle hooks |
| [`on_end()`](#on_end) | Lifecycle hooks |
| [`on_start()`](#on_start) | Lifecycle hooks |
| [`on_step_executed()`](#on_step_executed) | Lifecycle hooks |
| [`on_step_executing()`](#on_step_executing) | Lifecycle hooks |
| `transitions` | Introspection (property) |
| `schema` | Introspection (property) |
| `DEFAULT_STEP_RETRIES` | Constant |
| `StateTransition` | Data class |
| `StateSchema` | Class |
| `ReducerFn` | Type alias |
| `BeforeStepCallback` | Type alias |
| `AfterStepCallback` | Type alias |
| `BetweenStepsCallback` | Type alias |
| `StepExecutingCallback` | Type alias |
| `StepExecutedCallback` | Type alias |
| `OnStartCallback` | Type alias |
| `OnEndCallback` | Type alias |


## Detailed Reference

### `Workflow()`

Main class constructor.

**Parameters:**
- `name` (str): Name for logging and identification. Default: `"workflow"`.
- `schema` (StateSchema | None): Optional state schema for type validation and reducers. Default: `None`.

**Example:**
```python
from nono.workflows import Workflow, StateSchema

# Without schema
flow = Workflow("my_pipeline")

# With schema
schema = StateSchema(
    fields={"topic": str, "score": float, "notes": list},
    reducers={"notes": lambda old, new: (old or []) + new},
)
flow = Workflow("validated_pipeline", schema=schema)
```

---

### `step()`

Registers a step at the end of the pipeline.

**Parameters:**
- `name` (str): Unique step name.
- `fn` (Callable): Function that receives `state: dict` and returns a `dict` of updates.
- `retry` (int): Number of retry attempts on failure. Default: `0` (no retries).
- `on_error` (str | None): Step name to route to when the step fails after all retries. Default: `None` (re-raise).

**Returns:**
- `Workflow`: Self for fluent chaining.

**Raises:**
- `DuplicateStepError`: If a step with the same name already exists.

**Error handling behaviour:**
- When a step fails and `retry > 0`, the engine retries the step up to `retry` times.
- If all retries are exhausted and `on_error` is set, the workflow routes to the named fallback step with `state["__error__"]` containing `{"step": ..., "type": ..., "message": ...}`.
- If `on_error` is not set, the exception is re-raised.

**Example:**
```python
flow = Workflow()
flow.step("greet", lambda s: {"message": f"Hello {s['name']}"})
flow.step("upper", lambda s: {"message": s["message"].upper()})
result = flow.run(name="World")
# {'name': 'World', 'message': 'HELLO WORLD'}
```

**Example with retry and error recovery:**
```python
def risky_call(state):
    resp = call_external_api(state["query"])
    return {"data": resp}

def handle_error(state):
    return {"data": "fallback value", "error_info": state["__error__"]}

flow = Workflow()
flow.step("fetch", risky_call, retry=3, on_error="recover")
flow.step("recover", handle_error)
result = flow.run(query="test")
```

---

### `insert_at()`

Inserts a step at a specific position (0-based).

**Parameters:**
- `index` (int): Insertion position.
- `name` (str): Unique step name.
- `fn` (Callable): Step function.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.step("a", fn_a)
flow.step("c", fn_c)
flow.insert_at(1, "b", fn_b)
# steps: ['a', 'b', 'c']
```

---

### `insert_before()`

Inserts a step immediately before an existing one.

**Parameters:**
- `ref` (str): Reference step (must exist).
- `name` (str): New step name.
- `fn` (Callable): Step function.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.step("write", write_fn)
flow.insert_before("write", "outline", outline_fn)
# steps: ['outline', 'write']
```

---

### `insert_after()`

Inserts a step immediately after an existing one.

**Parameters:**
- `ref` (str): Reference step (must exist).
- `name` (str): New step name.
- `fn` (Callable): Step function.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.step("write", write_fn)
flow.insert_after("write", "fact_check", check_fn)
# steps: ['write', 'fact_check']
```

---

### `parallel_step()`

Registers a parallel execution step. All sub-functions run concurrently on the same state snapshot. Results are merged into the state when all complete.

**Parameters:**
- `name` (str): Unique step name (appears as one node in the graph).
- `fns` (dict[str, Callable]): Mapping `{sub_name: callable}` — each callable receives `state: dict` and returns `dict`.
- `max_workers` (int | None): Thread pool size. `None` = one thread per function.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.parallel_step("fetch", {
    "api":  lambda s: {"api_data": call_api(s["query"])},
    "db":   lambda s: {"db_data": query_db(s["query"])},
    "cache": lambda s: {"cache_data": check_cache(s["query"])},
}, max_workers=3)
# All three run concurrently; results merged: {api_data, db_data, cache_data}
```

**In a pipeline:**
```python
flow = Workflow("data_pipeline")
flow.step("prepare", lambda s: {"query": "AI trends"})
flow.parallel_step("fetch", {
    "news":  lambda s: {"news": get_news(s["query"])},
    "papers": lambda s: {"papers": get_papers(s["query"])},
})
flow.step("combine", lambda s: {"report": f"{s['news']}\n{s['papers']}"})
flow.connect("prepare", "fetch", "combine")
```

**Visualization:**
```
📋 data_pipeline (Workflow, 3 steps)
├── ○ prepare
├── ⏸ fetch  (parallel: news, papers)
└── ○ combine
```

---

### `loop_step()`

Registers a deterministic loop step. The function executes repeatedly while the condition holds, up to `max_iterations`.

**Parameters:**
- `name` (str): Unique step name.
- `fn` (Callable): Function that receives `state: dict` and returns `dict`.
- `condition` (Callable[[dict], bool]): Predicate. Loop continues while `True`.
- `max_iterations` (int): Safety cap. Default: `10`.

**Returns:**
- `Workflow`: Self for fluent chaining.

**State keys set:**
- `__loop_iterations__` (int): Number of iterations actually executed.

**Example:**
```python
flow.loop_step(
    "refine",
    lambda s: {"quality": s["quality"] + 0.15},
    condition=lambda s: s["quality"] < 0.9,
    max_iterations=10,
)
```

**In a pipeline:**
```python
flow = Workflow("quality_pipeline")
flow.step("draft", lambda s: {"text": "rough draft", "quality": 0.3})
flow.loop_step(
    "improve",
    improve_text_fn,
    condition=lambda s: s["quality"] < 0.9,
    max_iterations=5,
)
flow.step("publish", lambda s: {"status": "published"})
flow.connect("draft", "improve", "publish")
```

**Visualization:**
```
📋 quality_pipeline (Workflow, 3 steps)
├── ○ draft
├── 🔁 improve  (loop max 5x)
└── ○ publish
```

---

### `join()`

Registers an explicit join (wait-for-all) node. Validates that all required predecessor steps have executed. An optional reducer can post-process the merged state.

**Parameters:**
- `name` (str): Unique step name for the join node.
- `wait_for` (list[str]): Step names that must complete before this node.
- `reducer` (Callable[[dict], dict] | None): Optional post-processing function.
- `strict` (bool): When `True`, raise `JoinPredecessorError` if any predecessor has not executed. Default `False` (log warning only).

**Returns:**
- `Workflow`: Self for fluent chaining.

**Raises:**
- `JoinPredecessorError`: When `strict=True` and predecessors are missing.

**Example:**
```python
flow.step("branch_a", lambda s: {"a_result": "done"})
flow.step("branch_b", lambda s: {"b_result": "done"})
flow.join("merge", wait_for=["branch_a", "branch_b"],
          reducer=lambda s: {"summary": f"{s['a_result']} + {s['b_result']}"})
flow.connect("branch_a", "branch_b", "merge")
```

**Visualization:**
```
📋 pipeline (Workflow, 3 steps)
├── ○ branch_a
├── ○ branch_b
└── ⏩ merge  (join: branch_a, branch_b)
```

---

### `enable_checkpoints()`

Enables automatic state checkpointing. After each step, state is persisted to a JSON file.

**Parameters:**
- `directory` (str | Path): Folder for checkpoint files.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow = Workflow("long_pipeline")
flow.enable_checkpoints("./checkpoints")
flow.step("expensive_step", expensive_fn)
flow.run()
# State saved to ./checkpoints/long_pipeline_checkpoint.json after each step
```

---

### `resume()`

Loads the last checkpoint and returns the saved state and last completed step.

**Returns:**
- `tuple[dict, str | None]`: `(state_dict, last_step)` or `({}, None)` if no checkpoint.

**Example:**
```python
flow = Workflow("pipeline")
flow.enable_checkpoints("./checkpoints")
flow.step("a", step_a)
flow.step("b", step_b)

# First run crashes during "b"
# Second run resumes after "a"
result = flow.run(resume=True)
```

---

### `load_workflow()`

Module-level function that builds a `Workflow` from a JSON or YAML file.

**Parameters:**
- `path` (str | Path): Path to the `.json`, `.yaml`, or `.yml` file.
- `step_registry` (dict[str, Callable] | None): Mapping `{step_name: callable}` for steps whose logic can't be expressed declaratively.

**Returns:**
- `Workflow`: A fully-wired instance ready to `run()`.

**Raises:**
- `FileNotFoundError`: If the file doesn't exist.
- `WorkflowError`: On parse or validation errors.

**JSON format:**
```json
{
  "name": "my_pipeline",
  "steps": [
    {"name": "fetch", "type": "passthrough"},
    {"name": "process", "type": "tasker", "provider": "google",
     "system_prompt": "Summarise.", "input_key": "data", "output_key": "summary"}
  ],
  "edges": [["fetch", "process"]],
  "branches": [
    {"from": "process", "condition": "score > 0.8",
     "then": "publish", "otherwise": "revise"}
  ],
  "checkpoint_dir": "./checkpoints"
}
```

**YAML format:**
```yaml
name: my_pipeline
steps:
  - name: fetch
    type: passthrough
  - name: process
    type: tasker
    provider: google
    system_prompt: "Summarise."
    input_key: data
    output_key: summary
edges:
  - [fetch, process]
branches:
  - from: process
    condition: "score > 0.8"
    then: publish
    otherwise: revise
checkpoint_dir: ./checkpoints
```

**Example:**
```python
from nono.workflows import load_workflow

flow = load_workflow("pipelines/review.yaml", step_registry={
    "fetch": my_fetch_fn,
    "publish": my_publish_fn,
    "revise": my_revise_fn,
})
result = flow.run(input="some data")
```

---

### `tasker_node()`

Module-level factory that returns a `Callable[[dict], dict]` backed by `TaskExecutor`. Reads from `state[input_key]`, passes it through a TaskExecutor, and writes the result to `state[output_key]`.

Since it returns a plain callable, it is used with `step()` and inherits all dynamic manipulation methods (`insert_before`, `replace_step`, `swap_steps`, etc.).

Two modes:
- **Inline**: set `provider`, `model`, `system_prompt`, etc. Calls `TaskExecutor.execute()`.
- **JSON task file**: set `task_file`. Calls `TaskExecutor.run_json_task()`.

**Parameters:**
- `provider` (str): AI provider. Default: `"google"`.
- `model` (str): Model name. Default: `"gemini-3-flash-preview"`.
- `api_key` (str | None): API key (auto-resolved if `None`).
- `temperature` (float | str): Sampling temperature. Default: `0.7`.
- `max_tokens` (int): Maximum response tokens. Default: `2048`.
- `system_prompt` (str | None): System instruction prepended to the call.
- `output_schema` (dict | None): JSON schema for structured output.
- `input_key` (str): State key to read input from. Default: `"input"`.
- `output_key` (str): State key to write result to. Default: `"output"`.
- `task_file` (str | None): Path to a JSON task definition file.

**Returns:**
- `Callable[[dict], dict]`: A step function for `Workflow.step()`.

**Example — inline:**
```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("sentiment")
flow.step("classify", tasker_node(
    system_prompt="Classify as positive or negative.",
    input_key="text",
    output_key="sentiment",
    temperature=0.2,
))
result = flow.run(text="Great product!")
```

**Example — JSON task file:**
```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("names")
flow.step("classify_names", tasker_node(
    task_file="nono/tasker/prompts/name_classifier.json",
    input_key="names",
    output_key="classification",
))
result = flow.run(names='["Alice Smith", "Acme Corp"]')
```

**Example — chaining two tasker nodes:**
```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("chain")
flow.step("translate", tasker_node(
    system_prompt="Translate to English.",
    input_key="text", output_key="english"))
flow.step("summarise", tasker_node(
    system_prompt="Summarise in one sentence.",
    input_key="english", output_key="summary"))
flow.connect("translate", "summarise")
result = flow.run(text="Texto en espanol...")
# translate.output_key == "english" → summarise.input_key == "english"
```

---

### `agent_node()`

Module-level factory that returns a `Callable[[dict], dict]` which runs a Nono `Agent` via `Runner`. Reads from `state[input_key]`, invokes `Runner(agent).run()`, and writes the response to `state[output_key]`.

Since it returns a plain callable, it is used with `step()` and inherits all dynamic manipulation methods.

**Parameters:**
- `agent` (BaseAgent): A Nono agent instance (`Agent`, `LlmAgent`, etc.).
- `input_key` (str): State key to read the user message from. Default: `"input"`.
- `output_key` (str): State key to write the agent response to. Default: `"output"`.
- `state_keys` (dict[str, str] | None): Mapping `{runner_state_key: workflow_state_key}` to forward workflow state entries to the Runner session.

**Returns:**
- `Callable[[dict], dict]`: A step function for `Workflow.step()`.

**Example:**
```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node

writer = Agent(name="writer", instruction="Write a blog post.", provider="google")

flow = Workflow("blog")
flow.step("write", agent_node(writer, input_key="topic", output_key="draft"))
result = flow.run(topic="AI trends 2026")
```

**Example — with state_keys:**
```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node

writer = Agent(name="writer", instruction="Write a blog post.", provider="google")

flow = Workflow("styled_blog")
flow.step("write", agent_node(
    writer,
    input_key="topic",
    output_key="draft",
    state_keys={"style": "writing_style", "length": "max_words"},
))
result = flow.run(topic="AI", writing_style="formal", max_words="500")
```

---

### `remove_step()`

Removes a step and all edges/branches that reference it.

**Parameters:**
- `name` (str): Step to remove.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Raises:**
- `StepNotFoundError`: If the step does not exist.

**Example:**
```python
flow.step("a", fn_a)
flow.step("b", fn_b)
flow.remove_step("b")
# steps: ['a']
```

---

### `replace_step()`

Replaces a step's function without changing its position or edges.

**Parameters:**
- `name` (str): Step to replace.
- `fn` (Callable): New function.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.step("write", old_write_fn)
flow.replace_step("write", improved_write_fn)
```

---

### `move_to()`

Moves an existing step to a specific position (0-based).

**Parameters:**
- `name` (str): Step to move.
- `index` (int): Destination position.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
# steps: ['a', 'b', 'c']
flow.move_to("c", 0)
# steps: ['c', 'a', 'b']
```

---

### `move_before()`

Moves a step immediately before another.

**Parameters:**
- `ref` (str): Reference step.
- `name` (str): Step to move.

**Returns:**
- `Workflow`: Self for fluent chaining.

---

### `move_after()`

Moves a step immediately after another.

**Parameters:**
- `ref` (str): Reference step.
- `name` (str): Step to move.

**Returns:**
- `Workflow`: Self for fluent chaining.

---

### `swap_steps()`

Swaps the position of two steps.

**Parameters:**
- `a` (str): First step.
- `b` (str): Second step.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
# steps: ['a', 'b', 'c']
flow.swap_steps("a", "c")
# steps: ['c', 'b', 'a']
```

---

### `connect()`

Connects steps in sequence by defining explicit edges.

**Parameters:**
- `*names` (str): Step names in order.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.connect("research", "write", "review")
# Creates edges: research→write, write→review
```

---

### `branch()`

Adds conditional routing from a step using a callable.

**Parameters:**
- `from_step` (str): Step that triggers the branch.
- `condition` (Callable[[dict], str]): Function that receives the state and returns the next step name or `END`.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
from nono.workflows import END

flow.branch("review", lambda s: "publish" if s["score"] > 0.8 else "rewrite")
flow.branch("validate", lambda s: END if s.get("error") else "process")
```

---

### `branch_if()`

Adds conditional routing using a boolean predicate.

**Parameters:**
- `from_step` (str): Step that triggers the branch.
- `condition` (Callable[[dict], bool]): Predicate function.
- `then` (str): Destination step if the condition returns `True`.
- `otherwise` (str): Destination step if `False`.

**Returns:**
- `Workflow`: Self for fluent chaining.

**Example:**
```python
flow.branch_if("check", lambda s: s["score"] > 0.8, then="approve", otherwise="reject")
flow.branch_if("route", lambda s: s["status"] == "error", then="retry", otherwise="done")
```

---

### `run()`

Runs the complete workflow and returns the final state.

**Parameters:**
- `**initial_state` (Any): Key-value pairs for the initial state.

**Returns:**
- `dict`: Final state after running all steps.

**Raises:**
- `WorkflowError`: If the workflow has no steps.
- `StepNotFoundError`: If a referenced step does not exist during execution.

**Example:**
```python
result = flow.run(topic="AI", max_words=500)
print(result)
```

---

### `run_async()`

Async version of `run()`. Supports steps defined as coroutines.

**Parameters:**
- `**initial_state` (Any): Key-value pairs for the initial state.

**Returns:**
- `dict`: Final state.

**Example:**
```python
import asyncio

async def fetch_data(state):
    # async operation
    return {"data": await some_api_call(state["query"])}

flow.step("fetch", fetch_data)
result = asyncio.run(flow.run_async(query="test"))
```

---

### `stream()`

Runs the workflow and yields `(step_name, state_snapshot)` after each step.

**Parameters:**
- `**initial_state` (Any): Key-value pairs for the initial state.

**Yields:**
- `tuple[str, dict]`: Step name and deep copy of the state at that point.

**Example:**
```python
for step_name, snapshot in flow.stream(topic="AI"):
    print(f"[{step_name}] keys={list(snapshot.keys())}")
```

---

### `astream()`

Async version of `stream()`.

**Parameters:**
- `**initial_state` (Any): Key-value pairs for the initial state.

**Yields:**
- `tuple[str, dict]`: Step name and deep copy of the state.

**Example:**
```python
async for step_name, snapshot in flow.astream(topic="AI"):
    print(f"[{step_name}] -> {snapshot}")
```

---

### `describe()`

Returns a human-readable summary of the workflow graph.

**Returns:**
- `str`: Multi-line description with steps, edges, and branches.

**Example:**
```python
print(flow.describe())
# Workflow: my_pipeline
# Steps (3):
#   0. research (entry)
#   1. write
#   2. review
# Edges:
#   research -> write
#   write -> review
# Branches:
#   review -> (conditional)
```


## Branch Expressions

`branch_if()` supports simple comparison expressions that are compiled to callables internally.

### Format

```
key operator value
```

### Supported Operators

| Operator | Example | Description |
|:---|:---|:---|
| `==` | `"status == 'done'"` | Equality |
| `!=` | `"status != 'error'"` | Inequality |
| `>` | `"score > 0.8"` | Greater than |
| `<` | `"score < 0.2"` | Less than |
| `>=` | `"count >= 10"` | Greater than or equal |
| `<=` | `"count <= 5"` | Less than or equal |
| `in` | `"category in ['A', 'B']"` | Membership |
| `not in` | `"status not in ['error', 'failed']"` | Non-membership |
| `is` | `"result is None"` | Identity |
| `is not` | `"result is not None"` | Non-identity |

### Dotted Keys

Dotted keys resolve nested dicts:

```python
# state = {"result": {"score": 0.9, "label": "positive"}}
flow.branch_if("check", "result.score > 0.8", then="accept", otherwise="reject")
```

### Supported Values

| Type | Example |
|:---|:---|
| `None` | `"result is None"` |
| `True` / `False` | `"is_valid == True"` |
| Integers | `"count > 10"` |
| Floats | `"score > 0.8"` |
| Strings | `"status == 'done'"` |
| Lists | `"category in ['A', 'B']"` |


## Human-in-the-Loop (HITL)

Workflows can pause execution and wait for human approval, rejection, or a
custom message.  Two equivalent APIs are provided:

| API | Registration | Reject branch |
|:---|:---|:---|
| `human_step(name, handler=, ...)` | Fluent method on `Workflow` | Auto-registered via `on_reject` |
| `human_node(handler=, ...)` | Factory function for `step()` | Manual — use `branch()` / `branch_if()` |

Both call a **handler** function that blocks until the human responds:

```python
from nono.hitl import HumanInputHandler, HumanInputResponse

def console_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    print(f"[{step_name}] {prompt}")
    answer = input("Approve? (y/n) or message: ")
    if answer.lower() in ("y", "yes"):
        return HumanInputResponse(approved=True)
    if answer.lower() in ("n", "no"):
        return HumanInputResponse(approved=False, message="Rejected")
    return HumanInputResponse(approved=True, message=answer)
```

### `human_step()` — approve or reject with branch

```python
from nono.workflows import Workflow

flow = Workflow("review_pipeline")
flow.step("draft", write_draft)
flow.human_step("review", handler=console_handler, prompt="Approve the draft?",
                on_reject="revise")
flow.step("publish", publish_fn)
flow.step("revise", revise_fn)
flow.connect("draft", "review")
flow.connect("review", "publish")  # taken on approval

result = flow.run(topic="AI trends")
# If approved  → publish executes
# If rejected  → revise executes (on_reject branch)
```

### `human_node()` — factory style

```python
from nono.workflows import Workflow, human_node

flow = Workflow("review")
flow.step("draft", write_draft)
flow.step("review", human_node(handler=console_handler, prompt="Approve?"))
flow.step("publish", publish_fn)
```

### Data injection

The handler can inject data into the workflow state:

```python
def handler(step, state, prompt):
    return HumanInputResponse(
        approved=True,
        message="Approved with edits",
        data={"revised_topic": "AI safety"},
    )
# After this step, state["revised_topic"] == "AI safety"
```

### `HumanInputResponse`

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `approved` | `bool` | `True` | Whether the human approved |
| `message` | `str` | `""` | Free-text feedback or prompt |
| `data` | `dict` | `{}` | Extra key-value pairs merged into state |

The full response is stored in `state[state_key]` (default `"human_input"`).


## Exceptions

| Exception | Inherits from | When raised |
|:---|:---|:---|
| `WorkflowError` | `Exception` | Empty workflow at execution, general errors |
| `StepNotFoundError` | `WorkflowError` | Referenced step does not exist |
| `DuplicateStepError` | `WorkflowError` | Duplicate step name in `step()` / `insert_*()` |
| `StepTimeoutError` | `WorkflowError` | Step exceeds configured timeout |
| `JoinPredecessorError` | `WorkflowError` | `join(strict=True)` and required predecessors have not executed |
| `HumanRejectError` | `Exception` | Human rejected a `human_step` / `human_node` with no reject branch |

**Special state keys:**

| Key | Type | Set by | Description |
|:---|:---|:---|:---|
| `__executed_steps__` | `set` | Execution loop | Names of all steps that have run (serialized as `list` in JSON checkpoints) |
| `__error__` | `dict` | Error recovery | Contains `step`, `type`, and `message` when `on_error` routes to a fallback |
| `__loop_iterations__` | `int` | `loop_step` | Number of iterations executed |

```python
from nono.workflows import Workflow, WorkflowError, StepNotFoundError, DuplicateStepError

try:
    flow.run()
except StepNotFoundError as e:
    print(f"Step not found: {e}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
```


## Error Recovery and Retry

Steps can declare automatic retry and a fallback route using parameters on `step()`.

### Retry

When a step raises an exception and `retry > 0`, the engine re-executes the step up to `retry` times. A warning is logged on each retry attempt.

```python
flow = Workflow("api_pipeline")
flow.step("call_api", call_api_fn, retry=3)
# If call_api_fn raises, it is retried up to 3 times.
# Total attempts = 1 original + 3 retries = 4.
```

### Error routing with `on_error`

If a step fails after all retries and `on_error` is set, execution routes to the named fallback step instead of raising. The error details are stored in `state["__error__"]`:

```python
def fetch_data(state):
    return {"data": external_api(state["query"])}

def handle_failure(state):
    err = state["__error__"]
    return {"data": "default", "error_detail": f"{err['type']}: {err['message']}"}

flow = Workflow("resilient")
flow.step("fetch", fetch_data, retry=2, on_error="fallback")
flow.step("fallback", handle_failure)
flow.step("process", lambda s: {"result": transform(s["data"])})
flow.connect("fetch", "process")

result = flow.run(query="test")
# If fetch fails after 2 retries → fallback runs → process continues
```

The `__error__` dict has three keys:

| Key | Type | Description |
|:---|:---|:---|
| `step` | `str` | Name of the step that failed |
| `type` | `str` | Exception class name (e.g. `"ConnectionError"`) |
| `message` | `str` | Exception message string |

> **Important:** If `on_error` is **not** set and all retries are exhausted, the exception propagates normally.


## State Transition Audit Trail

Every step execution is recorded as a `StateTransition` in `flow.transitions`. The list is reset at the beginning of each `run()` / `run_async()` / `stream()` / `astream()` call.

### `StateTransition`

Frozen dataclass with the following fields:

| Field | Type | Description |
|:---|:---|:---|
| `step` | `str` | Step name |
| `keys_changed` | `frozenset[str]` | State keys added or modified by this step |
| `branch_taken` | `str \| None` | Next step chosen by a branch, or `None` |
| `duration_ms` | `float` | Wall-clock execution time in milliseconds |
| `retries` | `int` | Number of retry attempts (0 = succeeded on first try) |
| `error` | `str \| None` | Error message if the step failed, or `None` |

### Example

```python
from nono.workflows import Workflow

flow = Workflow("traced")
flow.step("a", lambda s: {"x": 1})
flow.step("b", lambda s: {"y": s["x"] + 1})
flow.connect("a", "b")

result = flow.run()
for t in flow.transitions:
    print(f"{t.step}: keys={t.keys_changed} duration={t.duration_ms:.1f}ms")
# a: keys=frozenset({'x'}) duration=0.1ms
# b: keys=frozenset({'y'}) duration=0.1ms
```

### Accessing after streaming

```python
for step_name, snapshot in flow.stream(topic="AI"):
    pass  # process snapshots

# After iteration completes:
for t in flow.transitions:
    print(f"{t.step}: retries={t.retries} error={t.error}")
```


## State Schema and Reducers

Attach a `StateSchema` to a workflow for optional type validation and custom merge logic (reducers).

### `StateSchema`

**Constructor parameters:**
- `fields` (dict[str, type]): Mapping of expected state key names to Python types.
- `reducers` (dict[str, ReducerFn] | None): Optional mapping of key names to reducer functions.

**Properties:**
- `fields` → `dict[str, type]`
- `reducers` → `dict[str, ReducerFn]`

**Methods:**
- `validate(state, step_name="")` → `list[str]`: Returns validation error strings for keys with wrong types. Missing keys are not errors.
- `apply_reducers(state, update)` → `dict`: Merges `update` into `state` using reducers where defined; plain replacement otherwise.

### Type validation

After each step, the schema validates returned values against declared types. Mismatches are logged as warnings but do **not** raise exceptions:

```python
from nono.workflows import Workflow, StateSchema

schema = StateSchema(fields={"score": float, "label": str})
flow = Workflow("typed", schema=schema)

flow.step("compute", lambda s: {"score": 0.95, "label": "positive"})
result = flow.run()
# No warnings — types match
```

### Reducers

Reducers control how a key is updated when a step returns new values. Without a reducer, values are replaced. With a reducer, the function `reducer(old_value, new_value)` determines the merged result:

```python
from nono.workflows import Workflow, StateSchema

schema = StateSchema(
    fields={"notes": list, "count": int},
    reducers={
        "notes": lambda old, new: (old or []) + new,  # append
        "count": lambda old, new: old + new,           # accumulate
    },
)

flow = Workflow("accumulator", schema=schema)
flow.step("a", lambda s: {"notes": ["first"], "count": 1})
flow.step("b", lambda s: {"notes": ["second"], "count": 1})
flow.connect("a", "b")

result = flow.run()
# result["notes"] == ["first", "second"]  (appended, not replaced)
# result["count"] == 2                     (accumulated, not replaced)
```

### Accessing the schema

```python
flow = Workflow("pipeline", schema=schema)
print(flow.schema.fields)     # {"notes": list, "count": int}
print(flow.schema.reducers)   # {"notes": <function>, "count": <function>}
```


## Lifecycle Hooks

Seven lifecycle hooks provide fine-grained control over workflow execution. All are registered via fluent API methods and fire automatically in every execution mode (`run`, `run_async`, `stream`, `astream`, `replay_from`).

### Hook Comparison

| Hook | When it fires | Signature | Can alter execution | Invocations |
|:---|:---|:---|:---|:---|
| `on_start` | **Start** of the workflow | `(workflow_name, state) → None` | No — observation only | 1 per run |
| `on_before_step` | **Before** a step runs | `(step_name, state) → Optional[dict]` | Yes — return `dict` to **skip** the step | 1 per step |
| `on_step_executing` | **Before each attempt** (inside retry loop) | `(step_name, state, attempt) → None` | No — observation only | N (1 per attempt) |
| `on_step_executed` | **After each attempt** (inside retry loop) | `(step_name, state, attempt, error) → None` | No — observation only | N (1 per attempt) |
| `on_after_step` | **After** a step runs (result available) | `(step_name, state, result) → Optional[dict]` | Yes — return `dict` to **replace** the result | 1 per step |
| `on_between_steps` | **Between** steps (step N done, step N+1 not started) | `(completed_step, next_step, state) → Optional[bool]` | Yes — return `False` to **halt** the workflow | 1 per transition |
| `on_end` | **End** of the workflow | `(workflow_name, state, steps_executed) → None` | No — observation only | 1 per run |

### Hook Execution Order

```
on_start(workflow_name, state)                  ← once, at workflow start
  │
  for each step:
  │
  on_before_step(step_name, state)              ← can skip the entire step
    │
    on_step_executing(step_name, state, attempt)  ← fires for every retry attempt
      │
      fn(state)                                   ← the step function
      │
    on_step_executed(step_name, state, attempt, error)  ← fires after every attempt
    │
  on_after_step(step_name, state, result)       ← can replace the result
    │
  on_between_steps(prev, next, state)           ← can halt the workflow
  │
on_end(workflow_name, state, steps_executed)     ← once, at workflow end
```

### Key Differences

1. **`on_before_step` vs `on_step_executing`** — `on_before_step` runs **once** and can **skip** the step entirely. `on_step_executing` runs **per attempt** (including retries) and is purely informational.

2. **`on_step_executing` vs `on_step_executed`** — Mirror pair inside the retry loop. `on_step_executing` fires **before** each attempt, `on_step_executed` fires **after** (with `error=None` on success or the error message on failure).

3. **`on_after_step` vs `on_between_steps`** — `on_after_step` acts on the **step result** (can modify it). `on_between_steps` acts on the **workflow transition** (can stop the pipeline).

4. **`on_start` / `on_end`** — Workflow-level bookends. `on_start` fires before the first step, `on_end` fires after the last step (regardless of how the workflow ended).

5. **Individual vs global control** — `on_before_step` / `on_after_step` control a **single step**. `on_between_steps` controls the **overall flow**. `on_start` / `on_end` observe the **entire lifecycle**.

### `on_start()`

```python
Workflow.on_start(callback: OnStartCallback) -> Workflow
```

Called **once** when the workflow begins execution, after initial validation and before the first step runs.

**Signature:** `OnStartCallback = Callable[[str, dict], None]`

---

### `on_end()`

```python
Workflow.on_end(callback: OnEndCallback) -> Workflow
```

Called **once** when the workflow finishes execution. Fires regardless of whether the workflow completed normally, was halted by `on_between_steps`, or encountered a cycle.

**Signature:** `OnEndCallback = Callable[[str, dict, int], None]`

**Parameters received:**
- `workflow_name` (str): Name of the workflow.
- `state` (dict): Final workflow state.
- `steps_executed` (int): Number of steps that ran.

### `on_before_step()`

```python
Workflow.on_before_step(callback: BeforeStepCallback) -> Workflow
```

Called **before** each step executes. Return a `dict` to **skip** the step (the dict is used as its result). Return `None` to let the step run normally.

**Signature:** `BeforeStepCallback = Callable[[str, dict], Optional[dict]]`

---

### `on_after_step()`

```python
Workflow.on_after_step(callback: AfterStepCallback) -> Workflow
```

Called **after** each step executes. Return a `dict` to **replace** the step's result. Return `None` to keep the original result.

**Signature:** `AfterStepCallback = Callable[[str, dict, Any], Optional[dict]]`

---

### `on_between_steps()`

```python
Workflow.on_between_steps(callback: BetweenStepsCallback) -> Workflow
```

Called **between** steps — after step N completes (including audit trail recording) and before step N+1 begins. Return `False` to **halt** execution immediately. Any other return value (including `None` and `True`) continues normally.

**Signature:** `BetweenStepsCallback = Callable[[str, Optional[str], dict], Optional[bool]]`

**Parameters received:**
- `completed_step` (str): Name of the step that just finished.
- `next_step` (str | None): Name of the next step, or `None` if the workflow would end.
- `state` (dict): Current workflow state.

**Example: progress reporting**

```python
flow.on_between_steps(
    lambda prev, nxt, state: print(f"[{prev}] done → next: {nxt or 'END'}")
)
```

**Example: conditional halt**

```python
def stop_on_low_quality(prev, nxt, state):
    if state.get("quality", 1.0) < 0.3:
        print(f"Halting after {prev}: quality too low")
        return False
    return None

flow.on_between_steps(stop_on_low_quality)
```

---

### `on_step_executing()`

```python
Workflow.on_step_executing(callback: StepExecutingCallback) -> Workflow
```

Called **right before each execution attempt** of a step function — inside the retry loop. Fires once per attempt, so a step with `retry=2` that fails twice and succeeds on the third attempt triggers this hook three times.

**Signature:** `StepExecutingCallback = Callable[[str, dict, int], None]`

**Parameters received:**
- `step_name` (str): Name of the step about to execute.
- `state` (dict): Current workflow state.
- `attempt` (int): 1-based attempt number.

**Example: retry logging**

```python
def log_attempts(name, state, attempt):
    if attempt > 1:
        print(f"  ⟳ Retrying {name} (attempt {attempt})")

flow.on_step_executing(log_attempts)
```

**Example: telemetry**

```python
import time

timings = {}

def start_timer(name, state, attempt):
    timings[name] = time.perf_counter()

flow.on_step_executing(start_timer)
flow.on_after_step(lambda name, state, result: (
    print(f"  {name}: {(time.perf_counter() - timings[name]) * 1000:.1f}ms")
))
```

### `on_step_executed()`

```python
Workflow.on_step_executed(callback: StepExecutedCallback) -> Workflow
```

Called **right after each execution attempt** — inside the retry loop. This is the counterpart of `on_step_executing()`. `error` is `None` on success or the error message string on failure.

**Signature:** `StepExecutedCallback = Callable[[str, dict, int, Optional[str]], None]`

**Parameters received:**
- `step_name` (str): Name of the step that just executed.
- `state` (dict): Current workflow state.
- `attempt` (int): 1-based attempt number.
- `error` (str | None): `None` on success, error message on failure.

**Example: per-attempt result logging**

```python
def log_attempt_result(name, state, attempt, error):
    if error:
        print(f"  ✗ {name} attempt {attempt} failed: {error}")
    else:
        print(f"  ✓ {name} attempt {attempt} succeeded")

flow.on_step_executed(log_attempt_result)
```

### Combined Example: Full Lifecycle

```python
from nono.workflows import Workflow

flow = Workflow("monitored")
flow.step("fetch", fetch_data, retry=2)
flow.step("transform", transform_data)
flow.step("load", load_data)
flow.connect("fetch", "transform", "load")

flow.on_start(lambda name, s: print(f"=== {name} started ==="))
flow.on_before_step(lambda n, s: print(f"▶ {n}"))
flow.on_step_executing(lambda n, s, a: print(f"  attempt {a}") if a > 1 else None)
flow.on_step_executed(lambda n, s, a, err: print(f"  ✗ attempt {a}") if err else None)
flow.on_after_step(lambda n, s, r: print(f"✓ {n}"))
flow.on_between_steps(lambda prev, nxt, s: print(f"  {prev} → {nxt or 'END'}"))
flow.on_end(lambda name, s, n: print(f"=== {name} finished ({n} steps) ==="))

flow.run(source="api")
```

Output (when `fetch` fails once):

```
=== monitored started ===
▶ fetch
  attempt 2
  ✗ attempt 1
✓ fetch
  fetch → transform
▶ transform
✓ transform
  transform → load
▶ load
✓ load
  load → END
=== monitored finished (3 steps) ===
```

### Type Aliases

| Type alias | Signature | Used by |
|:---|:---|:---|
| `OnStartCallback` | `(str, dict) → None` | `on_start()` |
| `OnEndCallback` | `(str, dict, int) → None` | `on_end()` |
| `BeforeStepCallback` | `(str, dict) → Optional[dict]` | `on_before_step()` |
| `AfterStepCallback` | `(str, dict, Any) → Optional[dict]` | `on_after_step()` |
| `BetweenStepsCallback` | `(str, Optional[str], dict) → Optional[bool]` | `on_between_steps()` |
| `StepExecutingCallback` | `(str, dict, int) → None` | `on_step_executing()` |
| `StepExecutedCallback` | `(str, dict, int, Optional[str]) → None` | `on_step_executed()` |


## Deterministic Orchestration

The Workflow engine supports full deterministic orchestration with the following control-flow nodes. No LLM calls are needed for routing — all decisions are based on Python predicates.

### Control-Flow Node Summary

| Node type | Method | Icon | Description |
|:---|:---|:---|:---|
| **Start** | first `step()` | `○` | Implicit — first registered step |
| **End** | `END` sentinel | — | Return `END` from a branch to stop |
| **Sequential** | `connect()` | `○` | Steps run in order |
| **If/Else** | `branch_if()` | `◆` | Predicate → then / otherwise |
| **Switch** | `branch()` | `◆` | Callable returns next step name |
| **Parallel** | `parallel_step()` | `⏸` | Fan-out to concurrent functions, auto-join |
| **Loop** | `loop_step()` | `🔁` | Repeat while condition holds |
| **Join** | `join()` | `⏩` | Wait-for-all barrier with optional reducer |
| **Human gate** | `human_step()` | `○` | Pause for approval / rejection |

### Complete deterministic example

```python
from nono.workflows import Workflow, END

flow = Workflow("etl_pipeline")

# 1. Start
flow.step("extract", lambda s: {"raw": load_data(s["source"])})

# 2. Parallel processing
flow.parallel_step("transform", {
    "clean":     lambda s: {"clean": clean(s["raw"])},
    "validate":  lambda s: {"valid": validate(s["raw"])},
    "enrich":    lambda s: {"enriched": enrich(s["raw"])},
})

# 3. Join barrier
flow.join("sync", wait_for=["transform"],
          reducer=lambda s: {"ready": s["valid"] and s["clean"] is not None})

# 4. Loop until quality is acceptable
flow.loop_step(
    "improve",
    improve_fn,
    condition=lambda s: s.get("score", 0) < 0.9,
    max_iterations=5,
)

# 5. Branch: pass or fail
flow.step("load", lambda s: {"status": "loaded"})
flow.step("quarantine", lambda s: {"status": "quarantined"})

flow.connect("extract", "transform", "sync", "improve")
flow.branch_if("improve", lambda s: s["ready"], then="load", otherwise="quarantine")

result = flow.run(source="data.csv")
```

### Visualization output

```
📋 etl_pipeline (Workflow, 6 steps)
├── ○ extract
├── ⏸ transform  (parallel: clean, validate, enrich)
├── ⏩ sync  (join: transform)
├── 🔁 improve  (loop max 5x)
├── ◆ improve
│   ├── ○ load
│   └── ○ quarantine
```


## Declarative Workflows

Instead of building workflows in Python code, you can define them in JSON or YAML files and load them with `load_workflow()`.

### JSON example

```json
{
  "name": "review_pipeline",
  "steps": [
    {"name": "draft", "type": "passthrough"},
    {"name": "evaluate", "type": "tasker",
     "provider": "google", "system_prompt": "Score from 0 to 1.",
     "input_key": "draft", "output_key": "score"},
    {"name": "publish", "type": "passthrough"},
    {"name": "revise", "type": "passthrough"}
  ],
  "edges": [
    ["draft", "evaluate"],
    ["evaluate", "publish"],
    ["evaluate", "revise"]
  ],
  "branches": [
    {"from": "evaluate", "condition": "score > 0.8",
     "then": "publish", "otherwise": "revise"}
  ],
  "checkpoint_dir": "./checkpoints"
}
```

### YAML example

```yaml
name: review_pipeline
steps:
  - name: draft
  - name: evaluate
    type: tasker
    provider: google
    system_prompt: "Score from 0 to 1."
    input_key: draft
    output_key: score
  - name: publish
  - name: revise
edges:
  - [draft, evaluate]
branches:
  - from: evaluate
    condition: "score > 0.8"
    then: publish
    otherwise: revise
```

### Loading and running

```python
from nono.workflows import load_workflow

flow = load_workflow("pipelines/review.yaml", step_registry={
    "draft": write_draft_fn,
    "publish": publish_fn,
    "revise": revise_fn,
})
result = flow.run()
```

### Supported step types

| `type` | Behavior |
|:---|:---|
| `passthrough` (default) | No-op unless overridden by `step_registry` |
| `tasker` | Creates a `tasker_node()` with `provider`, `model`, `system_prompt`, etc. |

### Supported branch conditions

Simple comparison expressions: `key operator value`.

```yaml
branches:
  - from: check
    condition: "score > 0.8"     # float comparison
    then: pass
    otherwise: fail
  - from: route
    condition: "status == done"   # string equality
    then: finish
    otherwise: retry
```

> **Note:** For YAML files, install PyYAML: `pip install pyyaml`.


## Integration with Nono Tasker and Agent

### Architecture

All node types use the same `step()` method, so they share every manipulation
operation (insert, replace, swap, branch, etc.).

```
                    Workflow
                    (state dict pipeline)
                        |
          step(name, callable)
       +--------+--------+--------+
       |        |                  |
    plain fn  tasker_node()   agent_node()
              (TaskExecutor)  (Agent+Runner)
                   |                |
              connector_genai  connector_genai
              (14 providers)   (14 providers)
```

### Data flow: output of one node → input of the next

Each step reads from `state[input_key]` and writes to `state[output_key]`. By aligning keys across steps, the pipeline chains automatically:

```python
from nono.workflows import Workflow, tasker_node, agent_node
from nono.agent import Agent

def gather(state):
    return {"research": f"Key findings about {state['topic']}"}

reviewer = Agent(name="reviewer", instruction="Polish the text.", provider="google")

flow = Workflow("article_pipeline")
flow.step("gather", gather)
flow.step("summarise", tasker_node(
    system_prompt="Summarise in 3 sentences.",
    input_key="research",      # reads gather's output
    output_key="summary"))
flow.step("review", agent_node(reviewer,
    input_key="summary",       # reads summarise's output
    output_key="article"))
flow.connect("gather", "summarise", "review")

result = flow.run(topic="GenAI")
# state: {topic, research, summary, article}
```

### Using JSON task files

```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("classify")
flow.step("classify_names", tasker_node(
    task_file="nono/tasker/prompts/name_classifier.json",
    input_key="names",
    output_key="classification",
))
result = flow.run(names='["Alice Smith", "Acme Corp"]')
```

### Branching with tasker/agent nodes

Mix functions, `tasker_node`, `agent_node`, and `branch_if` freely — they all
use `step()`, so they share every manipulation operation:

```python
from nono.workflows import Workflow, tasker_node, agent_node
from nono.agent import Agent

editor = Agent(name="editor", instruction="Improve text quality.", provider="google")

flow = Workflow("review_pipeline")
flow.step("evaluate", lambda s: {"quality": score_fn(s["draft"])})
flow.step("publish", lambda s: {"result": s["draft"]})
flow.step("enhance", tasker_node(
    system_prompt="Improve the following text.",
    input_key="draft", output_key="result"))
flow.step("deep_edit", agent_node(editor,
    input_key="draft", output_key="result"))

flow.branch("evaluate", lambda s:
    "publish"   if s["quality"] > 0.8 else
    "enhance"   if s["quality"] > 0.5 else
    "deep_edit"
)

result = flow.run(draft="Initial content...")
```

### Chaining multiple TaskExecutor nodes

```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("multi_tasker")
flow.step("translate", tasker_node(
    system_prompt="Translate to English.",
    provider="google", model="gemini-3-flash-preview",
    input_key="text", output_key="english"))
flow.step("analyse", tasker_node(
    system_prompt="Extract key entities.",
    provider="openai", model="gpt-4o-mini",
    input_key="english", output_key="entities"))
flow.connect("translate", "analyse")

result = flow.run(text="Texto en espanol sobre inteligencia artificial...")
# translate (Google) → analyse (OpenAI): each step can use a different provider
```

### Direct connector usage (low-level)

Steps can also use the connector layer directly when you need full control:

```python
from nono.workflows import Workflow
from nono.connector.connector_genai import GeminiService, ResponseFormat

def custom_step(state):
    client = GeminiService(model_name="gemini-3-flash-preview", api_key="...")
    messages = [
        {"role": "system", "content": "Classify as: tech, science, or other."},
        {"role": "user", "content": state["text"]},
    ]
    return {"category": client.generate_completion(messages, response_format=ResponseFormat.TEXT)}

flow = Workflow("custom")
flow.step("classify", custom_step)
result = flow.run(text="Large Language Models are transforming...")
```


## Advanced Examples

### Pipeline with dynamic manipulation

```python
flow = Workflow("dynamic")
flow.step("a", lambda s: {"log": ["A"]})
flow.step("b", lambda s: {"log": s["log"] + ["B"]})
flow.step("c", lambda s: {"log": s["log"] + ["C"]})

# Insert step between A and B
flow.insert_after("a", "a2", lambda s: {"log": s["log"] + ["A2"]})

# Swap B and C
flow.swap_steps("b", "c")

print(flow.steps)  # ['a', 'a2', 'c', 'b']
result = flow.run()
print(result["log"])  # ['A', 'A2', 'C', 'B']
```

### Workflow with streaming and logging

```python
import logging

logging.basicConfig(level=logging.INFO)

flow = Workflow("monitored")
flow.step("init", lambda s: {"count": 0})
flow.step("process", lambda s: {"count": s["count"] + 1, "status": "processed"})
flow.step("finalize", lambda s: {"status": "complete"})
flow.connect("init", "process", "finalize")

for step_name, snapshot in flow.stream():
    print(f"After [{step_name}]: count={snapshot.get('count')}, status={snapshot.get('status')}")
```

### Branch with complex callable

```python
from nono.workflows import Workflow, END

def route_by_priority(state):
    score = state.get("score", 0)
    category = state.get("category", "")
    
    if score > 0.9 and category == "urgent":
        return "escalate"
    elif score > 0.5:
        return "approve"
    elif score < 0.2:
        return END  # Terminate without processing
    else:
        return "review"

flow = Workflow("triage")
flow.step("evaluate", lambda s: {"score": 0.85, "category": "normal"})
flow.step("approve", lambda s: {"decision": "approved"})
flow.step("review", lambda s: {"decision": "pending_review"})
flow.step("escalate", lambda s: {"decision": "escalated"})
flow.branch("evaluate", route_by_priority)

result = flow.run()
print(result["decision"])  # approved
```

### Async workflow

```python
import asyncio
from nono.workflows import Workflow

async def async_fetch(state):
    await asyncio.sleep(0.1)  # simulate I/O
    return {"data": f"fetched for {state['query']}"}

async def async_process(state):
    await asyncio.sleep(0.1)
    return {"result": state["data"].upper()}

flow = Workflow("async_pipeline")
flow.step("fetch", async_fetch)
flow.step("process", async_process)
flow.connect("fetch", "process")

result = asyncio.run(flow.run_async(query="test"))
print(result["result"])  # FETCHED FOR TEST
```

---

## Workflow vs Agent Orchestration

Nono provides **two independent orchestration systems** that are fully composable. Each can be used alone, or combined in the same pipeline:

- **Deterministic** (`Workflow`) — you define the execution graph: steps, parallel fan-out, loops, joins, branches.
- **Agentic** (`Agent` + orchestration agents) — the LLM decides what to do: tool calls, delegation, routing.

| Aspect | Workflow | Agent (Sequential / Parallel / Router) |
| --- | --- | --- |
| Control | Developer-defined DAG | Agent nesting or LLM routing |
| State | Shared `dict` (merged after each step) | `Session` (events + state + shared_content) |
| Unit of work | Any callable | `BaseAgent` subclass |
| Parallel | `parallel_step()` (deterministic) | `ParallelAgent` |
| Loop | `loop_step()` (deterministic) | `LoopAgent` |
| Join | `join()` (explicit barrier) | Implicit in ParallelAgent |
| Dynamic routing | `branch()` / `branch_if()` (deterministic) | `RouterAgent` (LLM-driven) |
| Runtime manipulation | `insert_before`, `swap_steps`, etc. | Not supported |
| Checkpointing | `enable_checkpoints()` / `resume()` | Not built-in |
| Declarative | `load_workflow()` (JSON/YAML) | Python only |

### Bridging the two

- **Agent inside Workflow**: use `agent_node()` to wrap any agent as a step.
- **Workflow inside Agent**: call `flow.run()` from a `FunctionTool`.

The two layers are orthogonal: the Workflow controls execution order, while the agent reasons freely within its step.

```python
# Hybrid example — deterministic + agentic in the same pipeline
from nono.workflows import Workflow, agent_node, tasker_node
from nono.agent import Agent

researcher = Agent(name="researcher", tools=[search_tool], provider="google")

flow = Workflow("hybrid")
flow.step("ingest", ingest_fn)                              # deterministic
flow.parallel_step("enrich", [ner_fn, sentiment_fn])         # deterministic parallel
flow.step("research", agent_node(researcher))                # agentic (LLM decides tools)
flow.loop_step("refine", refine_fn,                          # deterministic loop
               condition="quality < 0.9", max_iterations=5)
flow.step("publish", tasker_node("prompts/publish.json"))    # deterministic + LLM
flow.connect_chain(["ingest", "enrich", "research", "refine", "publish"])

result = flow.run(raw_data="...")
```

For a full comparison with examples, see [Workflow vs Agent Orchestration](../agent/README_orchestration.md#workflow-vs-agent-orchestration).
