# Routines — Autonomous Execution Infrastructure

> **Inspired by [Claude Code Routines](https://code.claude.com/docs/routines), adapted for Nono's local/programmatic paradigm.**

A **routine** is a saved configuration: a prompt, an executable (agent, workflow, or task), a set of tools, and one or more **triggers** — packaged once and executed automatically.

## Quick Start

```python
from nono.agent import Agent
from nono.routines import Routine, RoutineRunner, ScheduleTrigger

# 1. Define an agent (or workflow, or any callable)
reviewer = Agent(
    name="code_reviewer",
    provider="google",
    model="gemini-3-flash-preview",
    instruction="Review the provided code for quality and security issues.",
)

# 2. Create a routine with a schedule trigger
routine = Routine(
    name="nightly_review",
    description="Run code review every night at 2 AM",
    executable=reviewer,
    input_template="Review the latest changes in {repo}",
    triggers=[ScheduleTrigger(cron="0 2 * * *")],
)

# 3. Register and start
runner = RoutineRunner()
runner.register(routine)
runner.start()

# 4. Manual fire (anytime)
result = runner.fire("nightly_review", context={"repo": "myproject"})
print(result.output)

# 5. Stop when done
runner.stop()
```

## Architecture

```
nono/routines/
├── __init__.py      # Public API exports
├── routine.py       # Routine, RoutineConfig, RoutineResult, RoutineStatus
├── triggers.py      # ScheduleTrigger, EventTrigger, WebhookTrigger, ManualTrigger
├── runner.py        # RoutineRunner — lifecycle management & execution
└── store.py         # RoutineStore — JSON persistence
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RoutineRunner                           │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Scheduler │  │ ThreadPool   │  │ Event Dispatcher     │ │
│  │ (tick     │  │ Executor     │  │ (emit_event →        │ │
│  │  loop)    │  │ (workers)    │  │  match triggers)     │ │
│  └─────┬─────┘  └──────┬───────┘  └──────────┬───────────┘ │
│        │               │                     │              │
│        └───────────┐    │    ┌────────────────┘              │
│                    ▼    ▼    ▼                               │
│              ┌─────────────────────┐                        │
│              │  _execute_routine() │                        │
│              └──────────┬──────────┘                        │
│                         │                                   │
│         ┌───────────────┼────────────────┐                  │
│         ▼               ▼                ▼                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐          │
│  │   Agent    │  │  Workflow  │  │   Callable   │          │
│  │  (Runner)  │  │  (.run())  │  │  (**context) │          │
│  └────────────┘  └────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Routine

A `Routine` bundles:

| Field | Description |
|---|---|
| `name` | Unique identifier |
| `description` | Human-readable purpose |
| `executable` | Agent, Workflow, or callable to run |
| `triggers` | List of trigger definitions (when to fire) |
| `config` | Timeout, retries, environment, tags |
| `instruction` | System prompt for agent executables |
| `input_template` | User message template with `{variable}` placeholders |
| `tools` | Tools to attach (for agent executables) |
| `status` | Lifecycle state (idle → active → running → …) |

### Routine Lifecycle

```
    register()         start()           trigger fires
  ┌──────────┐     ┌──────────┐     ┌──────────────────┐
  │   IDLE   │────▶│  ACTIVE  │────▶│     RUNNING      │
  └──────────┘     └──────────┘     └────────┬─────────┘
       ▲                ▲                    │
       │                │           ┌────────┴────────┐
       │                │           ▼                 ▼
       │           ┌─────────┐  ┌────────┐     ┌──────────┐
       │           │ resume()│  │SUCCESS │     │  ERROR   │
       │           └────┬────┘  └────┬───┘     └──────────┘
       │                │            │
       │           ┌────┴────┐       │
       │           │ PAUSED  │       │
       │           └─────────┘       │
       └─────────────────────────────┘
```

### RoutineStatus

| Status | Description |
|---|---|
| `IDLE` | Registered but scheduler not started |
| `ACTIVE` | Triggers armed, ready to fire |
| `RUNNING` | Currently executing |
| `PAUSED` | Temporarily suspended |
| `ERROR` | Last execution failed |
| `DISABLED` | Permanently deactivated |

## Triggers

Every routine can combine multiple triggers. Any match starts a new execution.

### ScheduleTrigger

Cron expression (5-field) or fixed interval:

```python
from nono.routines import ScheduleTrigger

# Cron: daily at 2 AM UTC
ScheduleTrigger(cron="0 2 * * *")

# Cron: every 15 minutes
ScheduleTrigger(cron="*/15 * * * *")

# Cron: weekdays at 9 AM
ScheduleTrigger(cron="0 9 * * 1-5")

# Interval: every hour
ScheduleTrigger(interval_seconds=3600)
```

**Cron field format**: `minute hour day_of_month month day_of_week`

| Field | Range | Special |
|---|---|---|
| minute | 0-59 | `*`, `*/N`, `N-M`, `N-M/S`, comma-separated |
| hour | 0-23 | same |
| day of month | 1-31 | same |
| month | 1-12 | same |
| day of week | 0-6 (Mon=0) | same |

### EventTrigger

Fires when the application emits a matching event:

```python
from nono.routines import EventTrigger

# Exact event name
EventTrigger(event_name="pr.opened")

# With payload filter
EventTrigger(
    event_name="alert.fired",
    filter_fn=lambda data: data.get("severity") == "critical",
)

# Regex pattern
EventTrigger(event_pattern=r"deploy\..*")
```

Emit events from anywhere in your application:

```python
runner.emit_event("pr.opened", {"repo": "myproject", "pr_number": 42})
```

### WebhookTrigger

HTTP endpoint trigger with optional HMAC-SHA256 validation:

```python
from nono.routines import WebhookTrigger

# Basic (auto-generated path)
WebhookTrigger()

# With HMAC secret
WebhookTrigger(secret="sk-my-secret-token")

# With IP whitelist
WebhookTrigger(allowed_ips=["10.0.0.1", "10.0.0.2"])
```

When the Nono server is running, webhook triggers create endpoints at:
```
POST /routine/{routine_name}/fire
```

### ManualTrigger

Every routine supports manual firing via `runner.fire()`. Attach `ManualTrigger` explicitly only for documentation:

```python
from nono.routines import ManualTrigger
ManualTrigger(description="Run via CLI: nono routine fire nightly_review")
```

### Combining Triggers

```python
routine = Routine(
    name="pr_review",
    executable=reviewer_agent,
    triggers=[
        ScheduleTrigger(cron="0 22 * * *", description="Nightly at 10 PM"),
        EventTrigger(event_name="pr.opened", description="On new PR"),
        WebhookTrigger(description="Via CI/CD pipeline"),
    ],
)
```

## RoutineRunner

The central coordinator that manages registration, scheduling, and execution.

### Basic Usage

```python
from nono.routines import RoutineRunner

runner = RoutineRunner(tick_seconds=30, max_workers=4)
runner.register(routine)
runner.start()

# ... routines fire automatically ...

runner.stop()
```

### Context Manager

```python
with RoutineRunner() as runner:
    runner.register(routine)
    # scheduler runs until the block exits
```

### Manual Execution

```python
# Fire and wait for result
result = runner.fire("nightly_review", context={"repo": "myproject"})
print(result.output)
print(result.status)        # ResultStatus.SUCCESS
print(result.duration_seconds)

# Fire without waiting
result = runner.fire("nightly_review", wait=False)
```

### Event Dispatch

```python
# Emit an event — all routines with matching EventTriggers fire
results = runner.emit_event("pr.opened", {
    "repo": "myproject",
    "pr_number": 42,
    "author": "developer",
})
for r in results:
    print(f"{r.routine_name}: {r.status.value}")
```

### Pause / Resume

```python
runner.pause("nightly_review")    # triggers ignored
runner.resume("nightly_review")   # re-armed
```

### Lifecycle Callbacks

```python
runner.on_start(lambda name, ctx: print(f"Starting {name}"))
runner.on_complete(lambda name, result: print(f"{name} done: {result.status.value}"))
runner.on_error(lambda name, exc: print(f"{name} failed: {exc}"))
```

### Execution History

```python
history = runner.get_history("nightly_review")
for record in history:
    print(f"  {record.run_id}: {record.status} ({record.duration_seconds:.1f}s)")
```

### Status Introspection

```python
status = runner.status()
print(status)
# {
#     "running": True,
#     "tick_seconds": 30.0,
#     "max_workers": 4,
#     "total_routines": 3,
#     "routines": [...]
# }
```

## Executable Types

The `Routine.executable` field accepts three types:

### Agent

```python
from nono.agent import Agent

agent = Agent(
    name="reviewer",
    provider="google",
    instruction="Review code for quality issues.",
)

routine = Routine(
    name="code_review",
    executable=agent,
    input_template="Review changes in {repo}: {description}",
)
```

The runner uses `nono.agent.Runner` internally, creating a fresh `Session` per execution.

### Workflow

```python
from nono.workflows import Workflow

flow = Workflow("analysis")
flow.step("fetch", fetch_data)
flow.step("analyze", analyze_data)
flow.connect("fetch", "analyze")

routine = Routine(
    name="data_analysis",
    executable=flow,
)

# Context keys become workflow initial state
runner.fire("data_analysis", context={"source": "database", "query": "SELECT ..."})
```

### Callable

Any Python callable:

```python
def cleanup_task(repo: str = "", max_age_days: int = 30, **kwargs) -> str:
    # ... perform cleanup ...
    return f"Cleaned {repo}, removed items older than {max_age_days} days"

routine = Routine(
    name="cleanup",
    executable=cleanup_task,
)

runner.fire("cleanup", context={"repo": "myproject", "max_age_days": 7})
```

## Persistence

Save and load routine definitions with `RoutineStore`:

```python
from nono.routines import RoutineStore

store = RoutineStore("routines.json")

# Save
store.save(runner.list_routines())

# Load (executables are None — must be re-bound)
routines = store.load()
for r in routines:
    r.executable = my_executables[r.name]
    runner.register(r)
```

## API Server Integration

When running the Nono API server, routines are exposed as REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/routines` | List all routines and their status |
| `POST` | `/routine/{name}/fire` | Fire a routine with optional context |
| `GET` | `/routine/{name}/history` | Get execution history |
| `POST` | `/routine/{name}/pause` | Pause a routine |
| `POST` | `/routine/{name}/resume` | Resume a paused routine |

### Fire via API

```bash
curl -X POST http://localhost:8000/routine/nightly_review/fire \
  -H "Content-Type: application/json" \
  -d '{"text": "Review PR #42", "context": {"repo": "myproject"}}'
```

Response:
```json
{
    "routine_name": "nightly_review",
    "run_id": "a1b2c3d4e5f6",
    "status": "success",
    "output": "Found 3 issues...",
    "duration_seconds": 12.5
}
```

## Visualization

```python
from nono.visualize import draw, draw_routine, draw_runner

# Single routine
print(draw_routine(routine))
# ⚙️  nightly_review (Routine, active)
# │   "Run code review every night at 2 AM"
# ├── Triggers
# │   ├── ⏰ schedule [0 2 * * *]
# │   └── 📡 event [pr.opened]
# ├── Config
# │   ├── timeout: 300s
# │   └── retries: 0
# └── Executable: code_reviewer (Agent)

# All routines in a runner
print(draw_runner(runner))
# 🏭 RoutineRunner (running, 3 routines)
# ├── ⚙️  nightly_review  [active]  (schedule, event)
# │      "Run code review every night at 2 AM"
# ├── ⚙️  deploy_check  [active]  (webhook)
# │      "Verify deployment health"
# └── ⚙️  doc_sync  [paused]  (schedule)
#        "Weekly documentation sync"

# Auto-detect with draw()
print(draw(routine))
print(draw(runner))
```

## Configuration

### RoutineConfig

| Field | Default | Description |
|---|---|---|
| `timeout_seconds` | `300` | Max execution time (0 = no limit) |
| `max_retries` | `0` | Retries on transient failure |
| `retry_delay_seconds` | `5.0` | Delay between retries |
| `max_history` | `100` | Run records to keep per routine |
| `environment` | `{}` | Key-value pairs injected as context |
| `tags` | `[]` | Labels for filtering/grouping |

```python
from nono.routines import Routine, RoutineConfig

routine = Routine(
    name="resilient_task",
    executable=my_agent,
    config=RoutineConfig(
        timeout_seconds=600,
        max_retries=3,
        retry_delay_seconds=10.0,
        environment={"API_KEY": "...", "ENV": "production"},
        tags=["critical", "nightly"],
    ),
)
```

## Use Case Examples

### 1. Nightly Code Review

```python
routine = Routine(
    name="nightly_review",
    description="Review PRs merged today for quality issues",
    executable=reviewer_agent,
    input_template="Review today's merged PRs in {repo}. Focus on security and performance.",
    triggers=[ScheduleTrigger(cron="0 22 * * 1-5", description="Weekdays at 10 PM")],
)
```

### 2. Alert Triage

```python
routine = Routine(
    name="alert_triage",
    description="Analyze production alerts and propose fixes",
    executable=triage_agent,
    input_template="Analyze alert: {text}. Correlate with recent deployments.",
    triggers=[
        EventTrigger(
            event_name="alert.fired",
            filter_fn=lambda d: d.get("severity") in ("critical", "high"),
            description="Critical/high alerts only",
        ),
    ],
    config=RoutineConfig(timeout_seconds=120, max_retries=1),
)
```

### 3. Deployment Verification

```python
routine = Routine(
    name="deploy_verify",
    description="Smoke-test after each production deployment",
    executable=verification_workflow,
    triggers=[
        WebhookTrigger(
            secret="sk-deploy-webhook-secret",
            description="Called by CD pipeline",
        ),
    ],
)
```

### 4. Documentation Drift

```python
routine = Routine(
    name="doc_sync",
    description="Weekly scan for stale documentation",
    executable=doc_scanner_agent,
    input_template="Scan docs/ for references to modified APIs since last week.",
    triggers=[ScheduleTrigger(cron="0 9 * * 1", description="Monday 9 AM")],
)
```

### 5. Multi-Trigger Routine

```python
routine = Routine(
    name="security_scan",
    description="Security analysis — scheduled + on-demand",
    executable=security_agent,
    triggers=[
        ScheduleTrigger(cron="0 3 * * *", description="Daily at 3 AM"),
        EventTrigger(event_name="pr.opened", description="On new PR"),
        WebhookTrigger(description="Manual trigger from dashboard"),
    ],
    config=RoutineConfig(
        timeout_seconds=900,
        max_retries=2,
        tags=["security", "critical"],
    ),
)
```

## Comparison with Claude Code Routines

| Feature | Claude Code Routines | Nono Routines |
|---|---|---|
| Execution | Cloud (Anthropic infra) | Local (in-process threads) |
| Triggers | Schedule, API, GitHub | Schedule, Event, Webhook, Manual |
| Persistence | Cloud-managed | JSON file store |
| Executables | Claude Code sessions | Agents, Workflows, Callables |
| API | Anthropic REST API | FastAPI server endpoints |
| Repositories | GitHub clone per run | Local workspace |
| Connectors | MCP connectors | Nono tools & connectors |
| Visualization | Web UI | ASCII terminal rendering |
| Context manager | ✗ | `with RoutineRunner() as runner:` |
| Event dispatch | GitHub webhooks | `runner.emit_event()` |
| Lifecycle hooks | ✗ | `on_start`, `on_complete`, `on_error` |
