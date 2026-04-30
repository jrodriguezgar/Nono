# Building a Data Pipeline — Step-by-Step Guide

> How to build an end-to-end project with data input, agents, tasker tools, custom tools, and file output.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Step 1 — Define the Input](#step-1--define-the-input)
- [Step 2 — Create Tasker Tools](#step-2--create-tasker-tools)
- [Step 3 — Create Custom Tools](#step-3--create-custom-tools)
- [Step 4 — Build the Agents](#step-4--build-the-agents)
- [Step 5 — Wire the Pipeline](#step-5--wire-the-pipeline)
- [Step 6 — Execute](#step-6--execute)
- [Step 7 — Stream Events](#step-7--stream-events)
- [Step 8 — Add Tracing](#step-8--add-tracing)
- [Step 9 — Save Output](#step-9--save-output)
- [Complete Code](#complete-code)
- [Variations](#variations)

---

## Overview

This guide walks through building a **Customer Feedback Analysis Pipeline** that:

1. Reads raw feedback text (input)
2. Extracts structured data using a `tasker_tool`
3. Classifies each item using another `tasker_tool`
4. Analyses patterns using an `Agent` with custom `@tool` functions
5. Redacts PII using a `tasker_tool`
6. Produces a Markdown report (output)

The pipeline mixes **three types of tools**:

| Tool type | Created with | Best for |
|-----------|-------------|----------|
| `tasker_tool()` | `from nono.agent import tasker_tool` | Structured LLM tasks with schemas |
| `@tool` decorator | `from nono.agent import tool` | Custom Python logic |
| `json_task_tool()` | `from nono.agent import json_task_tool` | Reusing JSON task files |

---

## Architecture

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
│ Raw data │──▶│  Extractor  │──▶│ Classifier  │──▶│ Analyst  │──▶│Guardrail │
│  (input) │    │(tasker_tool)│   │(tasker_tool)│   │(@tool)   │   │(tasker_tool)│
└──────────┘    └────────────┘    └────────────┘    └──────────┘    └────┬─────┘
                                                                         │
                                                                   ┌─────▼──────┐
                                                                   │   Report   │
                                                                   │  (output)  │
                                                                   └────────────┘
```

Each box is an **Agent**. They are connected in a **SequentialAgent** so the output of one flows as input to the next.

---

## Step 1 — Define the Input

The pipeline accepts any text. For this example, raw customer feedback:

```python
SAMPLE_FEEDBACK = """\
1. "The app crashes every time I open settings." - John Smith, john@example.com, 2026-03-15
2. "Love the new dark mode feature!" - María García, 2026-03-16
3. "Billing charged me twice. Card ends in 4242." - Bob Jones, bob@corp.net, 2026-03-17
4. "Search is slow when I type more than 3 words." - Alice W., 2026-03-18
5. "Onboarding tutorial was very helpful." - Chen Wei, chen.wei@mail.cn, 2026-03-19
"""
```

In a real project, this could come from a file, database, or API.

---

## Step 2 — Create Tasker Tools

`tasker_tool()` wraps a `TaskExecutor` call as a tool that any Agent can invoke. The LLM decides **when** to call it.

### 2.1 — Extractor tool

```python
from nono.agent import tasker_tool

extract_tool = tasker_tool(
    name="extract_feedback",
    description="Extract structured fields from raw customer feedback.",
    system_prompt=(
        "You are a data extraction specialist. Extract structured data from "
        "raw feedback. Return a JSON array with fields: "
        "id, user, message, contact_info, date. Use null for missing fields."
    ),
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "user": {"type": "string"},
                "message": {"type": "string"},
                "contact_info": {"type": "string"},
                "date": {"type": "string"},
            },
        },
    },
    temperature=0.1,
)
```

Key points:
- `system_prompt` defines the task behaviour.
- `output_schema` triggers constrained JSON decoding.
- `temperature=0.1` for consistent, deterministic extraction.
- The tool accepts a `prompt: str` parameter — the Agent passes the raw data.

### 2.2 — Classifier tool

```python
classify_tool = tasker_tool(
    name="classify_feedback",
    description="Classify each feedback item by category and sentiment.",
    system_prompt=(
        "You are a feedback classifier. For each item, assign:\n"
        "- category: bug, feature_request, billing, performance, praise, other\n"
        "- sentiment: positive, negative, neutral\n"
        "- priority: critical, high, medium, low\n"
        "Return a JSON array with id, category, sentiment, priority."
    ),
    output_schema={...},  # Same pattern as above
    temperature=0.1,
)
```

### 2.3 — PII Redaction tool

```python
redact_tool = tasker_tool(
    name="redact_pii",
    description="Remove PII (emails, names, card numbers) from text.",
    system_prompt=(
        "Replace all PII with placeholders: [NAME], [EMAIL], [CARD], [PHONE]. "
        "Return the cleaned text only."
    ),
    temperature=0.0,  # Zero for deterministic redaction
)
```

### Alternative: Using JSON task files

If you have a task defined in `prompts/classifier.json`, use `json_task_tool()` instead:

```python
from nono.agent import json_task_tool

classify_tool = json_task_tool(
    "nono/tasker/prompts/name_classifier.json",
    name="classify",
    description="Classify items using the name_classifier task.",
)
```

---

## Step 3 — Create Custom Tools

For logic that is **not** an LLM call (computation, I/O, API calls), use the `@tool` decorator:

### 3.1 — Statistics tool with ToolContext

```python
from nono.agent import tool, ToolContext
import json

@tool(description="Compute statistics from classified feedback data.")
def compute_stats(classified_json: str, tool_context: ToolContext) -> str:
    """Parse classified feedback and compute summary statistics."""
    items = json.loads(classified_json)

    categories = {}
    sentiments = {}
    for item in items:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        sent = item.get("sentiment", "unknown")
        sentiments[sent] = sentiments.get(sent, 0) + 1

    stats = {
        "total": len(items),
        "by_category": categories,
        "by_sentiment": sentiments,
    }

    # Save to shared content — accessible by all agents
    tool_context.save_content("feedback_stats", stats, scope="shared")

    return json.dumps(stats, indent=2)
```

Key points:
- `tool_context: ToolContext` is **auto-injected** — it doesn't appear in the LLM's tool schema.
- `save_content(..., scope="shared")` stores data accessible by all agents in the session.
- The function returns a string — the LLM sees this as the tool result.

### 3.2 — Report saver tool

```python
@tool(description="Save a report to the output file.")
def save_report(report_text: str, tool_context: ToolContext) -> str:
    """Save the final report text."""
    tool_context.save_content(
        "final_report",
        report_text,
        content_type="text/markdown",
        scope="shared",
    )
    return "Report saved successfully."
```

---

## Step 4 — Build the Agents

Each agent has a focused **instruction** and the tools it needs:

### 4.1 — Extractor agent

```python
from nono.agent import Agent

extractor = Agent(
    name="extractor",
    description="Extracts structured data from raw feedback.",
    instruction=(
        "You receive raw customer feedback text. "
        "Use the extract_feedback tool to parse it into structured JSON. "
        "Return ONLY the JSON result from the tool."
    ),
    tools=[extract_tool],
    temperature=0.1,
    output_format="json",
)
```

### 4.2 — Classifier agent

```python
classifier = Agent(
    name="classifier",
    description="Classifies feedback by category, sentiment, and priority.",
    instruction=(
        "You receive structured feedback JSON. "
        "Use the classify_feedback tool to classify each item. "
        "Return ONLY the JSON result."
    ),
    tools=[classify_tool],
    temperature=0.1,
    output_format="json",
)
```

### 4.3 — Analyst agent (custom tools)

```python
analyst = Agent(
    name="analyst",
    description="Analyses feedback patterns and writes a summary report.",
    instruction=(
        "You are a customer feedback analyst.\n\n"
        "Steps:\n"
        "1. Use compute_stats to get summary statistics.\n"
        "2. Write a Markdown report with:\n"
        "   - Executive summary\n"
        "   - Key metrics table\n"
        "   - Top issues (critical/high priority)\n"
        "   - Recommendations\n"
        "3. Use save_report to save the report.\n\n"
        "Return the report text."
    ),
    tools=[compute_stats, save_report],
    temperature=0.5,  # Slightly creative for writing
)
```

### 4.4 — Guardrail agent

```python
guardrail = Agent(
    name="guardrail",
    description="Redacts PII from the final report.",
    instruction=(
        "Use the redact_pii tool to clean the report. "
        "Return the redacted report."
    ),
    tools=[redact_tool],
    temperature=0.0,
)
```

---

## Step 5 — Wire the Pipeline

Connect agents in a `SequentialAgent`. Output of each agent flows as input to the next:

```python
from nono.agent import SequentialAgent

pipeline = SequentialAgent(
    name="feedback_pipeline",
    description="Full pipeline: extract → classify → analyse → redact.",
    sub_agents=[extractor, classifier, analyst, guardrail],
)
```

The execution flow:

```
User message (raw feedback)
    │
    ▼
extractor.run()  →  structured JSON
    │
    ▼
classifier.run() →  classified JSON
    │
    ▼
analyst.run()    →  Markdown report  (calls compute_stats + save_report)
    │
    ▼
guardrail.run()  →  redacted report
    │
    ▼
Final result returned to Runner
```

---

## Step 6 — Execute

### Basic execution

```python
from nono.agent import Runner

runner = Runner(agent=pipeline)
result = runner.run(SAMPLE_FEEDBACK)
print(result)  # The final redacted report
```

### With initial state

```python
runner = Runner(agent=pipeline)
result = runner.run(
    SAMPLE_FEEDBACK,
    project="Mobile App",       # Added to session.state
    date="2026-03-20",          # Accessible via tool_context.state
)
```

---

## Step 7 — Stream Events

See events as they happen — useful for UIs and debugging:

```python
runner = Runner(agent=pipeline)

for event in runner.stream(SAMPLE_FEEDBACK):
    print(f"[{event.author:<12}] {event.event_type.value:<16} "
          f"{event.content[:80] if event.content else ''}")
```

Output:

```
[extractor   ] tool_call        {"name": "extract_feedback", "arguments": {...}}
[extractor   ] tool_result      [{"id": 1, "user": "John Smith", ...}]
[extractor   ] agent_message    [{"id": 1, "user": "John Smith", ...}]
[classifier  ] tool_call        {"name": "classify_feedback", ...}
[classifier  ] tool_result      [{"id": 1, "category": "bug", ...}]
[classifier  ] agent_message    [{"id": 1, "category": "bug", ...}]
[analyst     ] tool_call        {"name": "compute_stats", ...}
[analyst     ] tool_result      {"total": 5, "by_category": {...}}
[analyst     ] tool_call        {"name": "save_report", ...}
[analyst     ] tool_result      Report saved successfully.
[analyst     ] agent_message    # Customer Feedback Report ...
[guardrail   ] tool_call        {"name": "redact_pii", ...}
[guardrail   ] tool_result      # Customer Feedback Report (redacted)
[guardrail   ] agent_message    # Customer Feedback Report (redacted)
```

Each event has: `event_type`, `author` (agent name), `content`, `data`, `timestamp`, `event_id`.

---

## Step 8 — Add Tracing

Record LLM calls, tool invocations, and timing:

```python
from nono.agent import TraceCollector

collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
result = runner.run(SAMPLE_FEEDBACK)

# Print summary
collector.print_summary()

# Export to dict for analysis
traces = collector.export()
```

The trace summary shows:

```
Trace: feedback_pipeline (SequentialAgent) — 4.2s — SUCCESS
├── Trace: extractor (LlmAgent) — 1.1s
│   ├── LLM Call: google/gemini-3-flash-preview — 0.8s — 450 tokens
│   └── Tool: extract_feedback — 0.3s
├── Trace: classifier (LlmAgent) — 0.9s
│   ├── LLM Call: google/gemini-3-flash-preview — 0.6s — 380 tokens
│   └── Tool: classify_feedback — 0.3s
├── Trace: analyst (LlmAgent) — 1.5s
│   ├── LLM Call: google/gemini-3-flash-preview — 0.4s — 200 tokens
│   ├── Tool: compute_stats — 0.001s
│   ├── LLM Call: google/gemini-3-flash-preview — 0.8s — 600 tokens
│   └── Tool: save_report — 0.001s
└── Trace: guardrail (LlmAgent) — 0.7s
    ├── LLM Call: google/gemini-3-flash-preview — 0.4s — 350 tokens
    └── Tool: redact_pii — 0.3s
```

---

## Step 9 — Save Output

### Option A — Via tool (inside the pipeline)

The `save_report` tool already stores the report via `ToolContext.save_content()`. Access it after execution:

```python
runner = Runner(agent=pipeline)
result = runner.run(SAMPLE_FEEDBACK)

# Access shared content from the session
report_item = runner.session.shared_content.load("final_report")
if report_item:
    print(report_item.data)            # The report text
    print(report_item.content_type)    # "text/markdown"
    print(report_item.created_at)      # Timestamp
```

### Option B — Save the final result to a file

```python
from pathlib import Path

result = runner.run(SAMPLE_FEEDBACK)
Path("report.md").write_text(result, encoding="utf-8")
```

### Option C — Export traces to JSON

```python
import json

collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
runner.run(SAMPLE_FEEDBACK)

Path("traces.json").write_text(
    json.dumps(collector.export(), indent=2, default=str),
    encoding="utf-8",
)
```

---

## Complete Code

See the full working example: [`nono/examples/example_data_pipeline.py`](../examples/example_data_pipeline.py)

Run it:

```bash
# With built-in sample data
python -m nono.examples.example_data_pipeline

# With custom input file
python -m nono.examples.example_data_pipeline --input feedback.txt --output report.md

# With streaming + tracing
python -m nono.examples.example_data_pipeline --stream --trace
```

---

## Variations

### A — Using a Workflow instead of SequentialAgent

For deterministic pipelines with more control (branching, state dict):

```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node, tasker_node

flow = Workflow("feedback_pipeline")

flow.step("extract", tasker_node(
    system_prompt="Extract structured data from feedback...",
    output_schema={...},
    input_key="raw_feedback",
    output_key="extracted",
))

flow.step("classify", tasker_node(
    system_prompt="Classify each item...",
    input_key="extracted",
    output_key="classified",
))

flow.step("analyse", agent_node(
    Agent(name="analyst", tools=[compute_stats, save_report]),
    input_key="classified",
    output_key="report",
))

flow.connect("extract", "classify")
flow.connect("classify", "analyse")

result = flow.run(raw_feedback=SAMPLE_FEEDBACK)
print(result["report"])
```

### B — Using ParallelAgent for independent tasks

When stages are independent, run them in parallel:

```python
from nono.agent import ParallelAgent

parallel = ParallelAgent(
    name="parallel_analysis",
    sub_agents=[
        Agent(name="sentiment_analyst", tools=[classify_tool]),
        Agent(name="topic_analyst",     tools=[extract_tool]),
    ],
    message_map={
        "sentiment_analyst": "Analyse sentiment: " + data,
        "topic_analyst":     "Extract topics: " + data,
    },
    result_key="analysis_results",
)
```

### C — Using LoopAgent for iterative refinement

Repeat analysis until quality is met:

```python
from nono.agent import LoopAgent

loop = LoopAgent(
    name="refine_report",
    sub_agents=[analyst, reviewer],
    max_iterations=3,
    stop_condition=lambda state: state.get("approved", False),
)
```

### D — Using RouterAgent for dynamic routing

Let the LLM decide which specialist handles each item:

```python
from nono.agent import RouterAgent

router = RouterAgent(
    name="feedback_router",
    sub_agents=[bug_handler, billing_handler, praise_handler],
    routing_instruction="Route each feedback item to the appropriate handler.",
)
```

### E — Mixed providers

Use different models for different stages:

```python
extractor = Agent(name="extractor", provider="google", model="gemini-3-flash-preview", ...)
analyst   = Agent(name="analyst",   provider="openai", model="gpt-4o-mini", ...)
guardrail = Agent(name="guardrail", provider="groq",   model="llama-3.3-70b-versatile", ...)
```
