# Docker Deployment Guide

> Deploy Nono as an API server and execute agents via HTTP.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How to Run a Task](#how-to-run-a-task)
- [How to Run an Agent](#how-to-run-an-agent)
- [How to Run a Workflow](#how-to-run-a-workflow)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Custom Agents](#custom-agents)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
                       ┌─────────────────────────────────────┐
  HTTP Request ────────▶│      Nono API Server (FastAPI)      │
                       │                                      │
  /task/{name}         │  ┌────────────────────────────────┐  │
  /agent/{name}        │  │  Named resource registries:    │  │
  /agent/{name}        │  │   tasks ─ prompts/*.json       │  │
      /stream          │  │   agents ─ 14 agent templates  │  │
  /workflow/{name}     │  │   workflows ─ 3 workflows      │  │
                       │  └────────────┬───────────────────┘  │
                       │               │                       │
                       │  ┌────────────▼───────────────────┐  │
                       │  │  Connector (connector_genai)    │  │
                       │  │  Google · OpenAI · Groq …      │  │
                       │  └────────────────────────────────┘  │
  HTTP Response ◀──────│                                      │
                       └─────────────────────────────────────┘
                                  Docker Container
```

Every call references a **pre-registered resource by name** — the API
does not accept raw prompts or create ad-hoc tasks/agents.

| Endpoint | Layer | What it runs |
|----------|-------|--------------|
| `POST /task/{task_name}` | **TaskExecutor** — run a built-in JSON task | `name_classifier`, `task_template`, … |
| `POST /agent/{agent_name}` | **Agent** — run a pre-built agent (single or multi-agent) | `planner`, `coder`, `feedback_analysis`, … |
| `POST /agent/{agent_name}/stream` | **Agent** — stream events (SSE) | Same agents, real-time |
| `POST /workflow/{workflow_name}` | **Workflow** — run a graph-based workflow | `sentiment_pipeline`, `content_pipeline`, … |

---

## Prerequisites

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- At least one LLM provider API key

---

## Quick Start

### 1. Configure API keys

Create a `.env` file at the project root:

```bash
# .env — at least one key is required
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
# GROQ_API_KEY=...
# DEEPSEEK_API_KEY=...
```

### 2. Build and run

```bash
docker compose up --build
```

The server starts on `http://localhost:8000`.

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.1.0"}

curl http://localhost:8000/info
# {"tasks": ["name_classifier", ...], "agents": ["coder", "planner", "feedback_analysis", ...], "workflows": ["sentiment_pipeline", ...]}
```

### 4. Run a task and an agent

```bash
# Run a built-in task by name
curl -X POST http://localhost:8000/task/name_classifier \
  -H "Content-Type: application/json" \
  -d '{"data_input": ["María García", "Telefónica S.A."]}'

# Run a pre-built agent by name
curl -X POST http://localhost:8000/agent/summarizer \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain microservices vs monoliths."}'

# Run a multi-agent composition by name
curl -X POST http://localhost:8000/agent/feedback_analysis \
  -H "Content-Type: application/json" \
  -d '{
    "message": "1. App crashes on settings - john@mail.com\n2. Love dark mode! - María",
    "trace": true
  }'
```

---

## Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `XAI_API_KEY` | xAI (Grok) API key | — |
| `PERPLEXITY_API_KEY` | Perplexity API key | — |
### Local Ollama (offline)

Uncomment the `ollama` service in `docker-compose.yml` for local inference:

```bash
docker compose up --build
```

Pull a model first:

```bash
docker exec nono-ollama ollama pull llama3.2
```

---

## How to Run a Task

The `POST /task/{task_name}` endpoint runs a **built-in JSON task** by name.
Tasks are pre-defined in `nono/tasker/prompts/*.json` — each one includes
system/user prompts, Jinja2 templates, output schemas, batching config,
and AI parameters.  The API user only provides the **data to process**.

```
┌──────────┐  POST /task/name_classifier  ┌────────────────────────────┐
│  Client   │ ──────────────────────────▶ │  TaskExecutor.run_json_task│
│           │  {"data_input": [...]}      │                            │
│           │                             │  • Loads name_classifier   │
│           │                             │  • Renders Jinja2 template │──▶ LLM
│           │                             │  • Auto-batches if large   │
│           │                             │  • Validates output schema │
│           │ ◀────────────────────────── │  • Merges batch results   │◀── response
└──────────┘       JSON result            └────────────────────────────┘
```

### Discover available tasks

```bash
curl http://localhost:8000/info | python -m json.tool
```

```json
{
  "tasks": ["name_classifier", "task_template"],
  "templates": ["conditional_flow", "data_loss_prevention", "decompose_tasks", "..."],
  "agents": ["classifier", "classify_and_route", "coder", "decomposer", "draft_review_loop", "extractor", "feedback_analysis", "guardrail", "plan_and_execute", "planner", "research_and_write", "reviewer", "summarizer", "writer"]
}
```

### Run `name_classifier`

```bash
curl -X POST http://localhost:8000/task/name_classifier \
  -H "Content-Type: application/json" \
  -d '{
    "data_input": ["María García", "Telefónica S.A.", "Barcelona", "John Doe"]
  }'
```

Response:

```json
{
  "result": "{\"results\": [{\"input\": \"María García\", \"category\": \"person\"}, {\"input\": \"Telefónica S.A.\", \"category\": \"company\"}, ...]}",
  "task": "name_classifier",
  "provider": "gemini",
  "model": "gemini-3-flash-preview",
  "duration_ms": 890.2
}
```

### Run with additional named inputs

Some tasks use multiple placeholders (`{data_input_json}`, `{language}`, etc.).
Pass them via `data_inputs`:

```bash
curl -X POST http://localhost:8000/task/name_classifier \
  -H "Content-Type: application/json" \
  -d '{
    "data_input": ["María García", "Google LLC"],
    "data_inputs": {"language": "es"}
  }'
```

### Python client — Tasks

```python
import requests

BASE = "http://localhost:8000"

# List available resources
info = requests.get(f"{BASE}/info").json()
print("Tasks:", info["tasks"])
print("Agents:", info["agents"])

# Run name_classifier
resp = requests.post(f"{BASE}/task/name_classifier", json={
    "data_input": ["Alice Smith", "IBM Corp", "Tokyo"],
})
print(resp.json()["result"])
```

### Task request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_input` | any | `null` | Primary data → `{data_input_json}` placeholder |
| `data_inputs` | object | `{}` | Named data → `{key}` placeholders |

---

## How to Run an Agent

The `POST /agent/{agent_name}` endpoint runs a **pre-built agent** by name.
Both single agents (LlmAgent templates) and multi-agent compositions
(SequentialAgent, LoopAgent, MapReduceAgent, ConsensusAgent, ProducerReviewerAgent, DebateAgent, EscalationAgent, SupervisorAgent, VotingAgent, HandoffAgent, GroupChatAgent, HierarchicalAgent, GuardrailAgent, BestOfNAgent, BatchAgent, CascadeAgent, RouterAgent) share the same endpoint.
The API user only provides the **message** to process.

```
┌──────────┐  POST /agent/planner  ┌───────────────────┐
│  Client   │ ───────────────────▶ │   planner_agent() │
│           │  {"message": "..."}  │   + Runner         │──▶ LLM
│           │                      │   + Session         │
│           │ ◀─────────────────── │   + Tracing         │◀── response
└──────────┘     JSON + trace      └───────────────────┘
```

### Available agents

#### Single agents (LlmAgent templates)

| Name | Specialty | Default output |
|------|-----------|----------------|
| `planner` | Strategic planning and project breakdown | JSON |
| `decomposer` | Complex task decomposition into subtasks | JSON |
| `summarizer` | Text and document summarization | text |
| `reviewer` | Output review and quality critique | text |
| `coder` | Code generation and programming | text |
| `classifier` | Input classification and routing | JSON |
| `extractor` | Structured data extraction from text | JSON |
| `writer` | Content writing and text generation | text |
| `guardrail` | Safety checks and PII redaction | text |

#### Multi-agent compositions

| Name | Type | Agents | Description |
|------|------|--------|-------------|
| `feedback_analysis` | Sequential | extractor → classifier → analyst → guardrail | Analyse customer feedback with PII redaction |
| `plan_and_execute` | Sequential | planner → decomposer → coder | Plan a goal, decompose, and generate code |
| `research_and_write` | Sequential | extractor → writer → reviewer | Extract data, write content, review quality |
| `draft_review_loop` | Loop | writer ↔ reviewer (×3 iterations) | Iteratively draft and review until approved |
| `classify_and_route` | Router | classifier → (coder \| writer \| summarizer \| extractor) | Classify input and route to specialist |

### Run the `planner` agent

```bash
curl -X POST http://localhost:8000/agent/planner \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Build a REST API for a task management app with auth, CRUD, and notifications."
  }'
```

Response:

```json
{
  "result": "{\"goal\": \"Build a REST API...\", \"phases\": [...], \"total_duration\": \"6 weeks\"}",
  "agent": "planner",
  "provider": "google",
  "model": "gemini-3-flash-preview",
  "duration_ms": 3200.5,
  "trace": null
}
```

### Run the `coder` agent

```bash
curl -X POST http://localhost:8000/agent/coder \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a Python function to merge two sorted lists in O(n)."
  }'
```

### Run with tracing

```bash
curl -X POST http://localhost:8000/agent/extractor \
  -H "Content-Type: application/json" \
  -d '{
    "message": "John Smith, 35, works at Google in London. Email: john@google.com",
    "trace": true
  }'
```

Response:

```json
{
  "result": "{\"name\": \"John Smith\", \"age\": 35, \"company\": \"Google\", ...}",
  "agent": "extractor",
  "provider": "google",
  "model": "gemini-3-flash-preview",
  "duration_ms": 1820.3,
  "trace": {
    "traces": [
      {"agent": "extractor", "status": "SUCCESS", "duration_ms": 1820, "llm_calls": [...], "tools_used": []}
    ]
  }
}
```

### Python client — Agents

```python
import requests

BASE = "http://localhost:8000"

# List available agents
info = requests.get(f"{BASE}/info").json()
print("Agents:", info["agents"])

# Run planner
resp = requests.post(f"{BASE}/agent/planner", json={
    "message": "Plan a data migration from PostgreSQL to BigQuery.",
    "trace": True,
})
data = resp.json()
print(data["result"])

# Run summarizer
resp = requests.post(f"{BASE}/agent/summarizer", json={
    "message": "Kubernetes is an open-source container orchestration platform...",
})
print(resp.json()["result"])

# Run coder
resp = requests.post(f"{BASE}/agent/coder", json={
    "message": "Implement a TokenBucket rate limiter in Python.",
})
print(resp.json()["result"])
```

### Agent request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | string | *required* | User message to send to the agent |
| `trace` | bool | `false` | Include trace data in response |

### Run a multi-agent composition

Multi-agent compositions use the same endpoint and request format:

```bash
curl -X POST http://localhost:8000/agent/feedback_analysis \
  -H "Content-Type: application/json" \
  -d '{
    "message": "1. App crashes on settings page - john@mail.com\n2. Love the dark mode!\n3. Billing charged twice, card 4111-1111-1111-1111 - María López"
  }'
```

```bash
curl -X POST http://localhost:8000/agent/plan_and_execute \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Build a Python CLI that converts CSV to JSON with schema validation.",
    "trace": true
  }'
```

```bash
curl -X POST http://localhost:8000/agent/classify_and_route \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a function to sort a list of tuples by the second element."
  }'
```

### Stream events in real time (SSE)

Any agent supports streaming via `POST /agent/{name}/stream`:

```bash
curl -N -X POST http://localhost:8000/agent/feedback_analysis/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bug: login fails. Feature: export to PDF."
  }'
```

Output:

```
data: {"agent": "extractor", "type": "tool_call", "content": "extract_feedback", ...}
data: {"agent": "extractor", "type": "agent_message", "content": "[{...}]", ...}
data: {"agent": "classifier", "type": "tool_call", "content": "classify_feedback", ...}
data: {"agent": "analyst", "type": "agent_message", "content": "## Report...", ...}
data: {"agent": "guardrail", "type": "agent_message", "content": "## Report (redacted)...", ...}
data: [DONE]
```

### Python client — Multi-agent compositions

```python
import requests
import json

BASE = "http://localhost:8000"

# Run feedback_analysis with trace
resp = requests.post(f"{BASE}/agent/feedback_analysis", json={
    "message": "1. Login broken\n2. Love the UI\n3. Billing error",
    "trace": True,
})
data = resp.json()
print(data["result"])
for t in data["trace"]["traces"]:
    print(f"  {t['agent']}: {t['status']} ({t['duration_ms']}ms)")

# Run plan_and_execute
resp = requests.post(f"{BASE}/agent/plan_and_execute", json={
    "message": "Create a rate limiter middleware for FastAPI.",
})
print(resp.json()["result"])

# Stream events
resp = requests.post(
    f"{BASE}/agent/feedback_analysis/stream",
    json={"message": "Bug: crash on save"},
    stream=True,
)
for line in resp.iter_lines():
    if line:
        text = line.decode("utf-8")
        if text.startswith("data: ") and text != "data: [DONE]":
            event = json.loads(text[6:])
            print(f"[{event['agent']}] {event['type']}: {event['content'][:80]}")
```

### Which endpoint should I use?

| Need | Endpoint | Example |
|------|----------|---------|
| Classify names | `POST /task/name_classifier` | Pre-defined task with schema |
| Plan a project | `POST /agent/planner` | Returns phased plan as JSON |
| Generate code | `POST /agent/coder` | Writes code from description |
| Summarize text | `POST /agent/summarizer` | Concise summary |
| Extract + write + review | `POST /agent/research_and_write` | 3-agent chain |
| Classify and route | `POST /agent/classify_and_route` | Auto-dispatches to specialist |
| Full feedback analysis | `POST /agent/feedback_analysis` | 4-agent + PII redaction |
| Real-time events | `POST /agent/{name}/stream` | SSE for any agent |
| Sentiment analysis | `POST /workflow/sentiment_pipeline` | 3-step graph: classify → score → summarise |
| Generate content | `POST /workflow/content_pipeline` | 4-step graph: outline → draft → review → polish |
| Enrich data | `POST /workflow/data_enrichment` | Branching graph: parse → validate → enrich/flag |

---

## How to Run a Workflow

The `POST /workflow/{workflow_name}` endpoint runs a **graph-based workflow**
by name.  Workflows are different from agents — they execute a directed graph
of step functions that share and transform a state dict.  Workflows support
conditional branching, cycles, and fine-grained state control.

```
┌──────────┐  POST /workflow/sentiment_pipeline  ┌─────────────────────────────┐
│  Client   │ ─────────────────────────────────▶ │         Workflow             │
│           │  {"state": {"input": "..."}}       │                             │
│           │                                    │  classify → score →         │──▶ LLM
│           │                                    │    summarise                │
│           │ ◀───────────────────────────────── │                             │◀── response
└──────────┘    JSON: full state + steps         └─────────────────────────────┘
```

### Available workflows

| Name | Steps | Description |
|------|-------|-------------|
| `sentiment_pipeline` | classify → score → summarise | Sentiment analysis with confidence scoring |
| `content_pipeline` | outline → draft → review → polish | Full content generation and editing |
| `data_enrichment` | parse → validate → enrich/flag | Data parsing with conditional branching |

### Run `sentiment_pipeline`

```bash
curl -X POST http://localhost:8000/workflow/sentiment_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"input": "The new dark mode is amazing!"},
    "trace": true
  }'
```

Response:

```json
{
  "result": {
    "input": "The new dark mode is amazing!",
    "classification": "{\"sentiment\": \"positive\", ...}",
    "score": "{\"confidence\": 0.95, ...}",
    "summary": "Positive sentiment (0.95) — user praises dark mode feature."
  },
  "workflow": "sentiment_pipeline",
  "steps_executed": ["classify", "score", "summarise"],
  "duration_ms": 4200.3,
  "trace": { "traces": [...] }
}
```

### Run `content_pipeline`

```bash
curl -X POST http://localhost:8000/workflow/content_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"input": "Write a blog post about async Python patterns."}
  }'
```

### Run `data_enrichment` (with branching)

```bash
curl -X POST http://localhost:8000/workflow/data_enrichment \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"input": "John Smith, CEO at TechCorp, john@techcorp.com"}
  }'
```

The workflow branches after `validate` — if the data is valid it goes to
`enrich`, otherwise to `flag`.

### Python client — Workflows

```python
import requests

BASE = "http://localhost:8000"

# Run sentiment analysis
resp = requests.post(f"{BASE}/workflow/sentiment_pipeline", json={
    "state": {"input": "App crashes on login. Terrible experience."},
    "trace": True,
})
data = resp.json()
print(f"Steps: {data['steps_executed']}")
print(f"Summary: {data['result'].get('summary', '')}")

# Run content generation
resp = requests.post(f"{BASE}/workflow/content_pipeline", json={
    "state": {"input": "Write about Python type hints best practices."},
})
print(resp.json()["result"]["final"][:200])

# Run data enrichment with branching
resp = requests.post(f"{BASE}/workflow/data_enrichment", json={
    "state": {"input": "Jane Doe, jane@acme.com, VP Engineering"},
})
data = resp.json()
print(f"Steps: {data['steps_executed']}")
if "enriched" in data["result"]:
    print("Enriched:", data["result"]["enriched"])
else:
    print("Flagged:", data["result"]["flags"])
```

### Workflow request parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state` | object | `{}` | Initial state key-value pairs |
| `trace` | bool | `false` | Include trace data in response |

---

## API Endpoints

### `GET /health`

Returns `{"status": "ok", "version": "..."}`.

### `GET /info`

Lists all named resources registered on the server:

```json
{
  "name": "Nono API Server",
  "version": "1.1.0",
  "endpoints": {
    "POST /task/{task_name}": "Run a built-in JSON task",
    "POST /agent/{agent_name}": "Run a pre-built agent template",
    "POST /agent/{agent_name}/stream": "Stream agent events (SSE)",
    "POST /workflow/{workflow_name}": "Run a pre-built workflow"
  },
  "tasks": ["name_classifier", "task_template"],
  "templates": ["conditional_flow", "data_loss_prevention", "decompose_tasks", "..."],
  "agents": ["classifier", "classify_and_route", "coder", "decomposer", "draft_review_loop", "extractor", "feedback_analysis", "guardrail", "plan_and_execute", "planner", "research_and_write", "reviewer", "summarizer", "writer"],
  "workflows": ["content_pipeline", "data_enrichment", "sentiment_pipeline"]
}
```

### `POST /task/{task_name}` — Run a built-in task

See [How to Run a Task](#how-to-run-a-task) for full examples.

### `POST /agent/{agent_name}` — Run a named agent

See [How to Run an Agent](#how-to-run-an-agent) for full examples.

### `POST /agent/{agent_name}/stream` — Stream agent events (SSE)

Same request body as the agent endpoint. Returns Server-Sent Events.

### `POST /workflow/{workflow_name}` — Run a named workflow

See [How to Run a Workflow](#how-to-run-a-workflow) for full examples.

---

## Usage Examples

### cURL — all three layers

```bash
# ── Task ────────────────────────────────────
curl -X POST http://localhost:8000/task/name_classifier \
  -H "Content-Type: application/json" \
  -d '{"data_input": ["María García", "Telefónica S.A."]}'

# ── Single Agent ─────────────────────────────
curl -X POST http://localhost:8000/agent/planner \
  -H "Content-Type: application/json" \
  -d '{"message": "Build a REST API for a task management app."}'

curl -X POST http://localhost:8000/agent/coder \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Python decorator for retry logic."}'

curl -X POST http://localhost:8000/agent/summarizer \
  -H "Content-Type: application/json" \
  -d '{"message": "Kubernetes is an open-source container orchestration..."}'

# ── Multi-Agent Composition ───────────────────
curl -X POST http://localhost:8000/agent/feedback_analysis \
  -H "Content-Type: application/json" \
  -d '{"message": "App crashes on login. Love dark mode!", "trace": true}'

curl -X POST http://localhost:8000/agent/plan_and_execute \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a rate limiter for FastAPI."}'

# ── Streaming ───────────────────────────────
curl -N -X POST http://localhost:8000/agent/feedback_analysis/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Bug: login fails. Feature: add export."}'

# ── Workflow ────────────────────────────────
curl -X POST http://localhost:8000/workflow/sentiment_pipeline \
  -H "Content-Type: application/json" \
  -d '{"state": {"input": "I love the new dark mode!"}, "trace": true}'

curl -X POST http://localhost:8000/workflow/content_pipeline \
  -H "Content-Type: application/json" \
  -d '{"state": {"input": "Write about Python async patterns"}}'

curl -X POST http://localhost:8000/workflow/data_enrichment \
  -H "Content-Type: application/json" \
  -d '{"state": {"input": "John Smith, john@acme.com, CTO"}}'
```

### Python — all three layers

```python
import requests

BASE = "http://localhost:8000"

# ── 1. Task ───────────────────────────────────
resp = requests.post(f"{BASE}/task/name_classifier", json={
    "data_input": ["Alice Smith", "IBM Corp", "Tokyo"],
})
print("Task:", resp.json()["result"])

# ── 2. Single Agent ────────────────────────────
resp = requests.post(f"{BASE}/agent/planner", json={
    "message": "Migrate a monolith to microservices.",
    "trace": True,
})
data = resp.json()
print("Agent:", data["result"][:100], "...")
print(f"  Model: {data['model']} | Duration: {data['duration_ms']}ms")

# ── 3. Multi-Agent Composition ─────────────────
resp = requests.post(f"{BASE}/agent/feedback_analysis", json={
    "message": "App crashes on login. Love the new UI!",
    "trace": True,
})
data = resp.json()
print("Composition:", data["result"][:100], "...")
for t in data["trace"]["traces"]:
    print(f"  {t['agent']}: {t['status']} ({t['duration_ms']}ms)")

# ── 4. Workflow ────────────────────────────────
resp = requests.post(f"{BASE}/workflow/sentiment_pipeline", json={
    "state": {"input": "The new update is great but login is broken."},
    "trace": True,
})
data = resp.json()
print(f"Workflow: {data['steps_executed']} in {data['duration_ms']}ms")
print(f"  Summary: {data['result'].get('summary', '')}")
```

---

## Custom Agents

Register custom agents or multi-agent compositions by editing `nono/server.py`:

```python
from nono.agent import Agent, SequentialAgent
from nono.agent.templates import writer_agent, reviewer_agent

def _build_content_agent(**kwargs):
    """Custom composition: write → review."""
    return SequentialAgent(
        name="content_agent",
        sub_agents=[writer_agent(), reviewer_agent()],
    )

# Register it in the agent registry
_AGENT_BUILDERS["content"] = _build_content_agent
```

Then call it by name:

```bash
curl -X POST http://localhost:8000/agent/content \
  -H "Content-Type: application/json" \
  -d '{"message": "Write about async Python"}'
```

---

## Production Deployment

### Multi-worker setup

```bash
# Run with multiple workers behind uvicorn
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=... \
  nono-api \
  uvicorn nono.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Behind a reverse proxy (nginx)

```nginx
upstream nono {
    server nono-api:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://nono;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;          # Required for SSE streaming
        proxy_cache off;
    }
}
```

### Resource limits (docker-compose)

```yaml
services:
  nono-api:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 256M
```

### Logging

```bash
# Follow logs
docker compose logs -f nono-api

# JSON structured logs
docker compose up -d
docker logs nono-api --since 1h
```

### SSL / TLS

For production, terminate TLS at the reverse proxy level or use:

```bash
uvicorn nono.server:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` from LLM provider | Check API key in `.env` or container env vars |
| `Connection refused` on Ollama | Ensure Ollama service is running and accessible at `http://ollama:11434` |
| Agent timeout | Increase `--timeout` in uvicorn or use `--workers` for parallelism |
| Image too large | Check `.dockerignore` excludes tests, .git, venv |
| SSE not streaming | Disable proxy buffering (`proxy_buffering off` in nginx) |
| `ModuleNotFoundError: fastapi` | Rebuild image: `docker compose build --no-cache` |
