# Human-in-the-Loop (HITL)

> Pause AI pipelines and wait for human approval, rejection, or data injection — via code, CLI, or REST API.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Types](#core-types)
- [Built-in Handlers](#built-in-handlers)
- [Usage: Python Code](#usage-python-code)
  - [Workflow with human\_step()](#workflow-with-human_step)
  - [Reject and Redirect](#reject-and-redirect)
  - [human\_node() Factory](#human_node-factory)
  - [HumanInputAgent (Agentic Workflows)](#humaninputagent-agentic-workflows)
  - [Conditional Intervention](#conditional-intervention)
  - [Data Injection](#data-injection)
- [Usage: CLI](#usage-cli)
  - [Interactive Mode](#interactive-mode)
  - [Pre-configured Responses](#pre-configured-responses)
- [Usage: REST API](#usage-rest-api)
  - [Run with HITL Responses](#run-with-hitl-responses)
  - [Streaming with HITL](#streaming-with-hitl)
- [Usage: Graphical UI](#usage-graphical-ui)
  - [Handler Interaction Model](#handler-interaction-model)
  - [Data Flow: UI ↔ Pipeline](#data-flow-ui--pipeline)
  - [Events for UI Updates](#events-for-ui-updates)
  - [Sync Handler (Thread-blocking UI)](#sync-handler-thread-blocking-ui)
  - [Async Handler (Non-blocking UI)](#async-handler-non-blocking-ui)
  - [Complete Example: GUI Review Panel](#complete-example-gui-review-panel)
- [Events and Observability](#events-and-observability)
- [Error Handling](#error-handling)
- [Custom Handlers](#custom-handlers)
- [API Reference](#api-reference)

---

## Overview

Human-in-the-Loop (HITL) lets you insert **checkpoints** in any Nono pipeline — Workflow or Agent — where execution pauses and a human decides what happens next. The human can:

- **Approve** — continue with the current state.
- **Reject** — redirect to a revision step or raise an error.
- **Inject data** — add corrections, annotations, or overrides to the state.

HITL is supported across all interfaces:

| Interface | How HITL Works |
|-----------|---------------|
| **Python code** | Pass a handler function that blocks until the human responds |
| **CLI** | `--interactive` prompts on stdin; `--hitl-responses` for automation |
| **REST API** | `hitl_responses` field pre-configures answers for each step |
| **Graphical UI** | Custom handler bridges the pipeline thread and the UI event loop |

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   draft      │────▶│   review     │────▶│   publish    │
│  (AI step)   │     │  (HITL step) │     │  (AI step)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │ reject
                            ▼
                     ┌──────────────┐
                     │   revise     │
                     │  (AI step)   │
                     └──────────────┘
```

When a pipeline reaches a HITL step:

1. An **HUMAN_INPUT_REQUEST** event is emitted.
2. The configured handler is called and **blocks** until the human responds.
3. The response (`HumanInputResponse`) is stored in the workflow state.
4. An **HUMAN_INPUT_RESPONSE** event is emitted.
5. Execution continues — either forward (approved) or to a reject branch.

---

## Core Types

All types are importable from `nono.hitl`:

```python
from nono.hitl import HumanInputResponse, HumanInputHandler, HumanRejectError
```

### HumanInputResponse

```python
@dataclass
class HumanInputResponse:
    approved: bool = True          # Did the human approve?
    message: str = ""              # Free-text feedback
    data: dict[str, Any] = {}     # Extra key-value pairs merged into state
```

### HumanInputHandler

```python
HumanInputHandler = Callable[[str, dict, str], HumanInputResponse]
#                             step_name, state, prompt
```

### HumanRejectError

Raised when a human rejects a step and no reject branch is configured:

```python
class HumanRejectError(Exception):
    step_name: str      # Step where rejection occurred
    human_message: str  # Human's feedback
```

---

## Built-in Handlers

Nono provides two ready-to-use handlers:

### console_handler

Interactive terminal handler — prompts via stdin/stdout and displays
the **current state values** so the human can review the content before
approving:

```python
from nono.hitl import console_handler

# Usage in a Workflow:
flow.human_step("review", handler=console_handler, prompt="Approve the draft?")
```

When executed, the user sees the actual content to review:

```
────────────────────────────────────────────────────────────
  ⏸  Human input required: review
────────────────────────────────────────────────────────────
  Prompt:  Approve the draft?

  Content to review:
  ▸ topic: AI safety in autonomous vehicles
  ▸ draft: Autonomous vehicles rely on AI systems that must
           make split-second safety decisions…

  Approve? (y/n) or type a message: _
```

> **Tip:** Use the `display_keys` parameter on `human_step()` or
> `HumanInputAgent` to select which state keys are shown in the prompt.
> When `display_keys` is set, only those values are appended to the prompt
> — useful when the state contains many internal keys.

### make_auto_handler

Factory for non-interactive handlers — pre-configure responses per step:

```python
from nono.hitl import make_auto_handler

handler = make_auto_handler({
    "review": {"approved": True, "message": "LGTM"},
    "final_check": {"approved": False, "message": "Needs work"},
})

# Steps not in the dict get the default (approved=True)
handler = make_auto_handler(default_approved=True, default_message="Auto-approved")
```

---

## Usage: Python Code

### Workflow with human_step()

The simplest HITL pattern — add a human checkpoint to a Workflow:

```python
from nono.workflows import Workflow
from nono.hitl import console_handler

flow = Workflow("review_pipeline")

# Step 1: Generate a draft
flow.step("draft", lambda s: {"draft": f"Article about {s['topic']}"})

# Step 2: Human reviews the draft — display_keys shows the draft content
flow.human_step(
    "review",
    handler=console_handler,
    prompt="Approve the draft?",
    display_keys=["draft"],  # show draft value in the prompt
)

# Step 3: Publish
flow.step("publish", lambda s: {"published": True, "final": s["draft"]})

result = flow.run(topic="AI safety in 2026")
# The pipeline pauses at "review" and waits for terminal input
print(result["published"])  # True (if approved)
```

After the human responds, the response is available in `state["human_input"]`:

```python
state["human_input"]["approved"]  # bool
state["human_input"]["message"]   # str
```

### Reject and Redirect

Use `on_reject` to redirect to a different step when the human rejects:

```python
flow = Workflow("review_pipeline")

flow.step("draft", write_draft)
flow.human_step(
    "review",
    handler=console_handler,
    prompt="Approve the draft?",
    on_reject="revise",  # redirect here on rejection
)
flow.step("publish", publish_fn)
flow.step("revise", revision_fn)  # handles rejection

flow.connect("draft", "review")
flow.connect("review", "publish")

result = flow.run(topic="AI trends")
# If rejected: flow goes draft → review → revise (skips publish)
# If approved: flow goes draft → review → publish
```

When `on_reject` is **not set** and the human rejects, `HumanRejectError` is raised.

### human_node() Factory

Alternative factory-function style — returns a step callable:

```python
from nono.workflows import Workflow, human_node
from nono.hitl import console_handler

review_fn = human_node(
    handler=console_handler,
    prompt="Approve before publishing?",
    on_reject="continue",  # "error" to raise, "continue" to proceed
)

flow = Workflow("pipeline")
flow.step("draft", draft_fn)
flow.step("review", review_fn)  # regular step() with the HITL callable
flow.step("publish", publish_fn)
```

### HumanInputAgent (Agentic Workflows)

For agentic workflows using `SequentialAgent`, `LoopAgent`, etc.:

```python
from nono.agent import Runner, Session, SequentialAgent
from nono.agent.human_input import HumanInputAgent
from nono.hitl import console_handler

research = Agent(name="researcher", ...)
writer = Agent(name="writer", ...)

# Human gate between research and writing
human_gate = HumanInputAgent(
    name="review",
    handler=console_handler,
    prompt="Approve the research before writing?",
    on_reject="error",  # or "continue"
)

pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[research, human_gate, writer],
)

runner = Runner(pipeline)
result = runner.run("Write about AI trends")
```

### Conditional Intervention

Use the `before_human` callback to skip the human for certain conditions:

```python
def before_human_check(agent, ctx):
    """Only ask human when quality is in the uncertain zone."""
    quality = ctx.session.state.get("quality", 0)

    if quality > 0.9:
        return "Auto-approved: high quality"  # skip human
    if quality < 0.3:
        ctx.session.state["human_rejected"] = True
        return "Auto-rejected: low quality"   # skip human

    return None  # proceed to human handler

human_review = HumanInputAgent(
    name="review",
    handler=console_handler,
    prompt="Review the draft?",
    on_reject="continue",
    before_human=before_human_check,
)
```

Quality zones:
- `quality > 0.9` → auto-approve, no human needed.
- `quality < 0.3` → auto-reject, no human needed.
- `0.3 ≤ quality ≤ 0.9` → ask the human.

### Data Injection

The human can inject data into the workflow state:

```python
def editor_handler(step_name, state, prompt):
    return HumanInputResponse(
        approved=True,
        message="Approved with edits",
        data={"corrections": "Fix paragraph 2, add references"},
    )

flow.human_step("editor", handler=editor_handler, prompt="Review and annotate?")
# After this step: state["corrections"] == "Fix paragraph 2, add references"
```

---

## Usage: CLI

The CLI supports HITL through the `workflow` subcommand with two modes.

### Interactive Mode

Use `--interactive` to enable terminal prompts at each human step:

```bash
# Interactive: the CLI prompts you at each HITL step
python -m nono.cli workflow content_review_pipeline \
    --state '{"topic": "AI safety"}' \
    --interactive
```

Output:

```
──────────────────────────────────────────────────
  ⏸  Human input required: review
──────────────────────────────────────────────────
  Prompt:  Review and approve the draft before publishing.
  State:   ['topic', 'draft']

  Approve? (y/n) or type a message: y

{
  "topic": "AI safety",
  "draft": "Draft article about AI safety",
  "published": true,
  "final": "Draft article about AI safety",
  "review_message": "Approved"
}
```

### Pre-configured Responses

Use `--hitl-responses` for CI/CD or automation — no terminal interaction:

```bash
# Auto-approve the review step
python -m nono.cli workflow content_review_pipeline \
    --state '{"topic": "AI safety"}' \
    --hitl-responses '{"review": {"approved": true, "message": "LGTM"}}'

# Auto-reject the review step
python -m nono.cli workflow content_review_pipeline \
    --state '{"topic": "AI safety"}' \
    --hitl-responses '{"review": {"approved": false, "message": "Needs more detail"}}'
```

### Available HITL Workflows

| Workflow | HITL Steps | Description |
|----------|-----------|-------------|
| `content_review_pipeline` | `review` | Draft → Human review → Publish or Revise |

Use `nono info` to list all available workflows.

---

## Usage: REST API

The Nono API server supports HITL through the `hitl_responses` field in workflow requests.

### Run with HITL Responses

```bash
# Start the server
uvicorn nono.server:app --reload
```

**Approve the review step:**

```bash
curl -X POST http://localhost:8000/workflow/content_review_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"topic": "AI safety in 2026"},
    "hitl_responses": {
      "review": {
        "approved": true,
        "message": "Looks good, publish it"
      }
    }
  }'
```

Response:

```json
{
  "result": {
    "topic": "AI safety in 2026",
    "draft": "Draft article about AI safety in 2026",
    "human_input": {"approved": true, "message": "Looks good, publish it"},
    "published": true,
    "final": "Draft article about AI safety in 2026",
    "review_message": "Looks good, publish it"
  },
  "workflow": "content_review_pipeline",
  "steps_executed": ["draft", "review", "publish"],
  "duration_ms": 1.2
}
```

**Reject — triggers the revise branch:**

```bash
curl -X POST http://localhost:8000/workflow/content_review_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"topic": "AI safety in 2026"},
    "hitl_responses": {
      "review": {
        "approved": false,
        "message": "Add more examples and references"
      }
    }
  }'
```

Response:

```json
{
  "result": {
    "topic": "AI safety in 2026",
    "draft": "Draft article about AI safety in 2026 [revised: Add more examples and references]",
    "human_input": {"approved": false, "message": "Add more examples and references"},
    "human_rejected": true,
    "revised": true
  },
  "workflow": "content_review_pipeline",
  "steps_executed": ["draft", "review", "revise"],
  "duration_ms": 1.5
}
```

**Inject data with the response:**

```bash
curl -X POST http://localhost:8000/workflow/content_review_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"topic": "AI safety"},
    "hitl_responses": {
      "review": {
        "approved": true,
        "message": "Approved with edits",
        "data": {"corrections": "Fix intro paragraph"}
      }
    }
  }'
```

### Streaming with HITL

HITL events are visible in the agent streaming endpoint:

```bash
curl -N http://localhost:8000/agent/plan_and_execute/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a report"}'
```

Events include `human_input_request` and `human_input_response` types.

### API Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `state` | `object` | No | Initial workflow state |
| `hitl_responses` | `object` | No | Pre-configured human responses per step |
| `hitl_responses.{step}.approved` | `bool` | No | Approve or reject (default: `true`) |
| `hitl_responses.{step}.message` | `string` | No | Human feedback message |
| `hitl_responses.{step}.data` | `object` | No | Extra data to merge into state |
| `trace` | `bool` | No | Include trace data in response |

---

## Usage: Graphical UI

This section explains how to integrate HITL with **any graphical UI framework** — tkinter, Qt, web (React, Vue, Svelte), Electron, mobile, etc. The pattern is the same regardless of toolkit: the handler acts as a **bridge** between the Nono pipeline thread and the UI event loop.

### Handler Interaction Model

```
┌──────────────────┐                        ┌──────────────────┐
│  Nono Pipeline    │                        │   UI Framework    │
│  (worker thread)  │                        │   (main thread)   │
│                   │                        │                   │
│  1. Reaches HITL  ├──── handler called ───▶│  2. Show dialog   │
│     step          │                        │     with prompt   │
│                   │     (blocks here)      │     + state data  │
│  4. Continues     │◀── response returned ──│  3. User clicks   │
│     pipeline      │                        │     Approve/Reject│
└──────────────────┘                        └──────────────────┘
```

The critical constraint: the pipeline **blocks** inside the handler until the human responds. In a GUI context this means:

| Approach | Pipeline runs on | Handler returns when |
|---|---|---|
| **Sync + thread** | Background thread | UI signals the thread via `Event`, `Queue`, or `Future` |
| **Async + await** | `asyncio` event loop | UI resolves a `Future` / `asyncio.Event` |

> **Never run the pipeline on the UI main thread.** It blocks the event loop and freezes the window.

### Data Flow: UI ↔ Pipeline

When the pipeline reaches a HITL step, the handler receives three parameters and must return a `HumanInputResponse`:

```
┌─────────────────────────── Handler Input ───────────────────────────────┐
│                                                                         │
│  step_name: str    →  "review"              (identify which HITL step)  │
│  state: dict       →  {"topic": "AI", "draft": "..."}  (current data)  │
│  prompt: str       →  "Approve the draft?"  (message to display)        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

                              ▼  User interacts with the UI  ▼

┌─────────────────────────── Handler Output ──────────────────────────────┐
│                                                                         │
│  HumanInputResponse(                                                    │
│      approved=True/False,            ← button click / checkbox          │
│      message="Human feedback text",  ← text field                      │
│      data={"key": "value"},          ← form fields, edits, overrides   │
│  )                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**What to display in the UI:**

| Handler Input | UI Element | Purpose |
|---|---|---|
| `step_name` | Window title / header label | Identify the checkpoint |
| `prompt` | Main text label | Tell the user what to decide |
| `state` keys and values | Read-only text area or key-value table | Show current pipeline data |

**What to collect from the UI:**

| User Action | Maps To | Type |
|---|---|---|
| Approve / Reject button | `response.approved` | `bool` |
| Feedback text field | `response.message` | `str` |
| Additional form fields | `response.data` | `dict[str, Any]` |

### Events for UI Updates

Use Nono events to update your UI with real-time pipeline progress. This lets you show status indicators, progress bars, or logs **before and after** each HITL step:

| Event Type | When Emitted | UI Action |
|---|---|---|
| `AGENT_START` | Agent begins | Show "Processing..." spinner |
| `AGENT_MESSAGE` | Agent produces output | Update output text area |
| `HUMAN_INPUT_REQUEST` | HITL step reached | Open the review dialog |
| `HUMAN_INPUT_RESPONSE` | Human responded | Close dialog, show result |
| `AGENT_END` | Agent finishes | Hide spinner, show final result |

To capture events, use the `Runner` with event streaming:

```python
from nono.agent import Runner, Session
from nono.agent.base import EventType

runner = Runner(pipeline, session=session)

# Stream events — update UI as they arrive
for event in runner.run_stream("message"):
    if event.type == EventType.HUMAN_INPUT_REQUEST:
        ui.show_review_dialog(event.data)
    elif event.type == EventType.HUMAN_INPUT_RESPONSE:
        ui.close_review_dialog()
    elif event.type == EventType.AGENT_MESSAGE:
        ui.append_output(event.data)
```

### Sync Handler (Thread-blocking UI)

Use `threading.Event` to bridge the pipeline thread and the UI thread. This works with **any UI framework** (tkinter, Qt, GTK, WinForms, etc.):

```python
import threading
from nono.hitl import HumanInputHandler, HumanInputResponse


def make_gui_handler(show_dialog_fn):
    """Create a HITL handler that delegates to a UI dialog.

    Args:
        show_dialog_fn: Callable that receives (step_name, state, prompt,
            response_holder, result_event) and is invoked on the UI thread.
            It must call ``result_event.set()`` after storing the response.

    Returns:
        A ``HumanInputHandler`` safe for use from a background thread.
    """
    result_event = threading.Event()
    response_holder: list[HumanInputResponse] = []

    def handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        result_event.clear()
        response_holder.clear()

        # Schedule the dialog on the UI thread
        # (use framework-specific mechanism: after(), signals, dispatch, etc.)
        show_dialog_fn(step_name, state, prompt, response_holder, result_event)

        # Block the pipeline thread until the UI responds
        result_event.wait()
        return response_holder[0]

    return handler
```

**Generic UI dialog pattern** (framework-agnostic pseudocode):

```python
def show_dialog_on_ui_thread(step_name, state, prompt, response_holder, result_event):
    """Schedule this on the UI main thread via your framework's dispatcher."""

    # 1. Create a modal dialog / panel
    dialog = ui.create_dialog(title=f"Review: {step_name}")

    # 2. Display the prompt and state
    dialog.add_label(prompt)
    dialog.add_readonly_text(format_state(state))

    # 3. Add input controls
    feedback_field = dialog.add_text_input(placeholder="Feedback (optional)")
    data_fields = dialog.add_form(fields=["corrections", "priority"])

    # 4. Add action buttons
    def on_approve():
        response_holder.append(HumanInputResponse(
            approved=True,
            message=feedback_field.get_text(),
            data=data_fields.get_values(),
        ))
        dialog.close()
        result_event.set()  # Unblock the pipeline thread

    def on_reject():
        response_holder.append(HumanInputResponse(
            approved=False,
            message=feedback_field.get_text(),
            data=data_fields.get_values(),
        ))
        dialog.close()
        result_event.set()  # Unblock the pipeline thread

    dialog.add_button("Approve", on_click=on_approve)
    dialog.add_button("Reject", on_click=on_reject)
    dialog.show()
```

### Async Handler (Non-blocking UI)

For async UI frameworks (web backends with WebSocket, async desktop toolkits):

```python
import asyncio
from nono.hitl import AsyncHumanInputHandler, HumanInputResponse


def make_async_gui_handler(send_to_ui, receive_from_ui):
    """Create an async HITL handler for event-driven UIs.

    Args:
        send_to_ui: Async callable to push data to the UI.
            Signature: ``(data: dict) -> None``
        receive_from_ui: Async callable that blocks until the
            user responds. Signature: ``() -> dict``

    Returns:
        An ``AsyncHumanInputHandler``.
    """

    async def handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        # 1. Send the review request to the UI
        await send_to_ui({
            "type": "human_input_request",
            "step": step_name,
            "prompt": prompt,
            "state": state,
        })

        # 2. Wait for the user response from the UI
        user_data = await receive_from_ui()

        # 3. Convert to HumanInputResponse
        return HumanInputResponse(
            approved=user_data.get("approved", True),
            message=user_data.get("message", ""),
            data=user_data.get("data", {}),
        )

    return handler
```

**WebSocket usage** (works with any web framework):

```python
async def ws_send(data):
    await websocket.send_json(data)

async def ws_receive():
    return await websocket.receive_json()

handler = make_async_gui_handler(send_to_ui=ws_send, receive_from_ui=ws_receive)
```

**UI-side JavaScript** (framework-agnostic):

```javascript
// Receive the HITL request from the backend
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "human_input_request") {
    showReviewDialog(data.step, data.prompt, data.state);
  }
};

// Send the user's decision back
function submitReview(approved, message, extraData) {
  ws.send(JSON.stringify({
    approved: approved,
    message: message,
    data: extraData,
  }));
}
```

### Complete Example: GUI Review Panel

This example shows a full working pattern — pipeline on a background thread, UI handler bridging both sides. Replace `ui.*` calls with your framework equivalents.

```python
import threading
from nono.hitl import HumanInputResponse
from nono.workflows import Workflow


# ── 1. Define the handler bridge ─────────────────────────────────────────

class GUIHumanHandler:
    """Thread-safe HITL handler for any GUI framework.

    The pipeline calls ``__call__()`` on a worker thread.
    The UI calls ``submit()`` on the main thread.
    """

    def __init__(self):
        self._event = threading.Event()
        self._response: HumanInputResponse | None = None
        self.on_request = None  # Set this to your UI callback

    def __call__(self, step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        """Called by the pipeline (worker thread). Blocks until submit()."""
        self._event.clear()
        self._response = None

        # Notify the UI that input is needed
        if self.on_request:
            self.on_request(step_name, state, prompt)

        # Block until the UI calls submit()
        self._event.wait()
        return self._response

    def submit(self, approved: bool, message: str = "", data: dict = None):
        """Called by the UI (main thread). Unblocks the pipeline."""
        self._response = HumanInputResponse(
            approved=approved,
            message=message,
            data=data or {},
        )
        self._event.set()


# ── 2. Build the pipeline ────────────────────────────────────────────────

handler = GUIHumanHandler()

flow = Workflow("review_pipeline")
flow.step("draft", lambda s: {"draft": f"Article about {s['topic']}"})
flow.human_step("review", handler=handler, prompt="Approve the draft?")
flow.step("publish", lambda s: {"published": True, "final": s["draft"]})


# ── 3. Connect the UI ────────────────────────────────────────────────────

def on_hitl_request(step_name, state, prompt):
    """Called when the pipeline needs human input.

    Schedule this on your UI thread using your framework's mechanism:
      - tkinter:  root.after(0, lambda: ...)
      - Qt:       QMetaObject.invokeMethod(...) or signal.emit(...)
      - Web:      await websocket.send_json(...)
      - Electron: ipcMain.emit(...)
    """
    ui.set_title(f"Review: {step_name}")
    ui.set_prompt_text(prompt)
    ui.set_state_display(state)
    ui.show_review_panel()

handler.on_request = on_hitl_request


def on_approve_clicked():
    """User clicks the Approve button."""
    handler.submit(
        approved=True,
        message=ui.get_feedback_text(),
        data=ui.get_form_data(),
    )

def on_reject_clicked():
    """User clicks the Reject button."""
    handler.submit(
        approved=False,
        message=ui.get_feedback_text(),
        data=ui.get_form_data(),
    )

# Bind buttons (adapt to your framework)
ui.approve_button.on_click = on_approve_clicked
ui.reject_button.on_click = on_reject_clicked


# ── 4. Run the pipeline on a background thread ───────────────────────────

def run_pipeline():
    try:
        result = flow.run(topic="AI safety in 2026")
        ui.show_result(result)  # Schedule on UI thread
    except Exception as e:
        ui.show_error(str(e))   # Schedule on UI thread

thread = threading.Thread(target=run_pipeline, daemon=True)
thread.start()
```

**Sequence diagram:**

```
 UI Thread                Handler                 Pipeline Thread
 ─────────                ───────                 ───────────────
     │                       │                         │
     │                       │     __call__(...)        │
     │     on_request(...)   │◀────────────────────────│ (blocks)
     │◀──────────────────────│                         │
     │                       │                         │
     │  Show dialog          │                         │
     │  User clicks Approve  │                         │
     │                       │                         │
     │     submit(True, ...) │                         │
     │──────────────────────▶│     return response     │
     │                       │────────────────────────▶│ (continues)
     │                       │                         │
```

**Framework-specific `on_request` dispatching:**

| Framework | How to Schedule on UI Thread |
|---|---|
| **tkinter** | `root.after(0, lambda: on_hitl_request(step, state, prompt))` |
| **Qt / PySide** | `signal.emit(step, state, prompt)` or `QTimer.singleShot(0, fn)` |
| **GTK** | `GLib.idle_add(on_hitl_request, step, state, prompt)` |
| **Web (FastAPI + WS)** | `await websocket.send_json({"step": step, ...})` |
| **Electron** | `ipcMain.emit("hitl-request", {step, state, prompt})` |
| **React / Vue / Svelte** | WebSocket message → state update → component re-render |
| **WinForms (.NET)** | `control.Invoke(Action(lambda: ...))` |

---

## Events and Observability

HITL steps emit two event types visible in traces and streams:

| Event | When | Data |
|-------|------|------|
| `HUMAN_INPUT_REQUEST` | Before handler is called | `{state_keys: [...], display_keys: [...] or null}` |
| `HUMAN_INPUT_RESPONSE` | After handler returns | `{approved: bool, data: {...}}` |

Enable tracing to capture HITL events:

```python
from nono.agent import Runner, TraceCollector

collector = TraceCollector()
runner = Runner(pipeline, trace_collector=collector)
runner.run("message")

for trace in collector.export():
    print(trace)
```

Via API, set `"trace": true`:

```bash
curl -X POST http://localhost:8000/workflow/content_review_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"topic": "AI"},
    "hitl_responses": {"review": {"approved": true}},
    "trace": true
  }'
```

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Human rejects, `on_reject` set | Flow redirects to the named step |
| Human rejects, no `on_reject` | `HumanRejectError` raised |
| No handler configured | `ValueError` raised |
| API call without `hitl_responses` on HITL workflow | Default auto-approve handler used |

```python
from nono.hitl import HumanRejectError

try:
    result = flow.run(topic="AI")
except HumanRejectError as e:
    print(f"Rejected at step '{e.step_name}': {e.human_message}")
```

---

## Custom Handlers

### Web / WebSocket Handler

```python
import asyncio
from nono.hitl import AsyncHumanInputHandler, HumanInputResponse

async def websocket_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Send prompt via WebSocket, await human response."""
    await ws.send_json({"step": step_name, "prompt": prompt, "state_keys": list(state.keys())})
    response = await ws.receive_json()
    return HumanInputResponse(
        approved=response.get("approved", True),
        message=response.get("message", ""),
        data=response.get("data", {}),
    )
```

### Slack / Teams Handler

```python
def slack_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Post to Slack, poll for reaction."""
    msg = slack.post_message(channel="#reviews", text=f"[{step_name}] {prompt}")
    # Block until a reaction is added (thumbsup = approve, thumbsdown = reject)
    reaction = slack.wait_for_reaction(msg.ts, timeout=3600)
    return HumanInputResponse(
        approved=reaction == "thumbsup",
        message=reaction,
    )
```

### Database Queue Handler

```python
def db_queue_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Insert a pending review into a database, poll until resolved."""
    review_id = db.insert("pending_reviews", step=step_name, prompt=prompt, state=state)
    while True:
        row = db.get("pending_reviews", review_id)
        if row["status"] != "pending":
            return HumanInputResponse(
                approved=row["status"] == "approved",
                message=row.get("feedback", ""),
            )
        time.sleep(5)
```

---

## API Reference

### Workflow API

| Method | Signature | Description |
|--------|-----------|-------------|
| `Workflow.human_step()` | `(name, handler, prompt, on_reject, state_key, display_keys)` | Register a HITL checkpoint |
| `human_node()` | `(handler, prompt, state_key, on_reject) -> Callable` | Factory for HITL step functions |

### Agent API

| Class | Key Parameters | Description |
|-------|---------------|-------------|
| `HumanInputAgent` | `handler`, `prompt`, `on_reject`, `before_human`, `after_human`, `display_keys` | Agent that pauses for human input |

### Handler Types

| Type | Signature |
|------|-----------|
| `HumanInputHandler` | `(step_name: str, state: dict, prompt: str) -> HumanInputResponse` |
| `AsyncHumanInputHandler` | `(step_name: str, state: dict, prompt: str) -> Awaitable[HumanInputResponse]` |

### Built-in Handlers

| Function | Use Case |
|----------|----------|
| `console_handler` | Interactive terminal / CLI — shows state values |
| `make_auto_handler(responses, default_approved, default_message)` | Testing, CI/CD, API |
| `format_state_for_review(state, display_keys, max_value_len)` | Format state values for human review |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--interactive` | Enable interactive stdin prompts at HITL steps |
| `--hitl-responses '{...}'` | Pre-configured JSON responses per step |

### REST API Fields

| Field | Type | Description |
|-------|------|-------------|
| `hitl_responses` | `dict[str, dict]` | Pre-configured responses keyed by step name |
| `hitl_responses.{step}.approved` | `bool` | Approve or reject |
| `hitl_responses.{step}.message` | `str` | Human feedback |
| `hitl_responses.{step}.data` | `dict` | Extra data to merge |
