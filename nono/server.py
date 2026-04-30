"""
Nono API Server — FastAPI application exposing named platform resources.

The API is a thin invocation layer: it does **not** accept raw prompts
or create ad-hoc tasks/agents.  Every call references a pre-registered
resource by name:

Endpoints:
    GET  /health                         — Health check.
    GET  /info                           — List tasks, agents, workflows, and projects.
    POST /task/{task_name}               — Run a built-in JSON task.
    POST /agent/{agent_name}             — Run a pre-built agent.
    POST /agent/{agent_name}/stream      — Stream agent events (SSE).
    POST /workflow/{workflow_name}        — Run a pre-built workflow.
    GET  /workflow/{workflow_name}/describe — Workflow graph structure.
    POST /skill/{skill_name}             — Run a registered skill.
    GET  /projects                       — List discovered projects.
    GET  /project/{project_name}         — Project details and resources.

Usage::

    # Local development
    pip install uvicorn fastapi
    uvicorn nono.server:app --reload

    # Docker
    docker compose up
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nono import __version__
from nono.agent import (
    Agent,
    Runner,
    Session,
    SequentialAgent,
    TraceCollector,
    tool,
    tasker_tool,
)
from nono.agent.templates import (
    planner_agent,
    decomposer_agent,
    summarizer_agent,
    reviewer_agent,
    coder_agent,
    classifier_agent,
    extractor_agent,
    writer_agent,
    guardrail_agent,
    plan_and_execute,
    research_and_write,
    draft_review_loop,
    classify_and_route,
)
from nono.agent.tracing import TraceStatus
from nono.workflows.templates import (
    build_sentiment_pipeline,
    build_content_pipeline,
    build_data_enrichment,
    build_content_review_pipeline,
)
from nono.hitl import make_auto_handler
from nono.agent.skill import registry as skill_registry
from nono.project import load_project, list_projects, MANIFEST_FILE
from nono.routines import RoutineRunner, RoutineResult
import nono.agent.skills  # noqa: F401 — load built-in skills into registry

# ── Routine runner (singleton for the server) ────────────────────
_routine_runner = RoutineRunner()

logger = logging.getLogger("Nono.Server")

# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Nono API",
    description=(
        "Agentic AI server powered by Nono.  Every endpoint references "
        "a pre-built resource by name — no raw-prompt execution."
    ),
    version=__version__,
)

_MAX_REQUEST_BODY_BYTES: int = 1_048_576  # 1 MB


@app.middleware("http")
async def _limit_request_body(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Reject requests whose Content-Length exceeds the global limit."""
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > _MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Payload Too Large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
    return await call_next(request)


# ── Request / Response models ────────────────────────────────────


class TaskRequest(BaseModel):
    """Input for ``POST /task/{task_name}`` — run a registered JSON task.

    Only data inputs are required.  Provider, model, temperature, and
    all other AI parameters are defined inside the task JSON file.
    """

    data_input: Any = Field(
        None,
        description=(
            "Primary input data — maps to {data_input_json} placeholder. "
            "Can be a string, list, dict, or any JSON-serialisable value."
        ),
    )
    data_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional named inputs — each key maps to a {key} placeholder "
            "in the user prompt template."
        ),
    )
    trace: bool = Field(False, description="Include trace data in response.")


class AgentRequest(BaseModel):
    """Input for ``POST /agent/{agent_name}`` — run a registered agent.

    Only the user message is required.  Provider, model, instruction,
    temperature, and output format are defined in the agent template.
    """

    message: str = Field(..., description="User message to send to the agent.", max_length=100_000)
    trace: bool = Field(False, description="Include trace data in response.")
    introspect: bool = Field(
        False,
        description="Include session events and state in the response.",
    )


class EventRecord(BaseModel):
    """Serialised :class:`Event` from the agent session."""

    event_type: str
    author: str
    content: str
    timestamp: str
    event_id: str


class AgentIntrospectionData(BaseModel):
    """Session introspection data returned when ``introspect=true``."""

    events: list[EventRecord] = Field(
        default_factory=list,
        description="Chronological list of session events.",
    )
    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Final session state.",
    )
    event_count: int = Field(0, description="Total number of events.")


class TaskResponse(BaseModel):
    """Output from ``POST /task/{task_name}``."""

    result: str
    task: str
    provider: str
    model: str
    duration_ms: float
    trace: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    """Output from ``POST /agent/{agent_name}``."""

    result: str
    agent: str
    provider: str | None = None
    model: str | None = None
    duration_ms: float
    trace: dict[str, Any] | None = None
    introspection: AgentIntrospectionData | None = None


class WorkflowRequest(BaseModel):
    """Input for ``POST /workflow/{workflow_name}`` — run a named workflow.

    The caller provides the initial state as key-value pairs.
    All AI parameters are defined inside the workflow template.
    """

    state: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial state key-value pairs for the workflow.",
    )
    hitl_responses: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Pre-configured human-in-the-loop responses keyed by step name. "
            "Each value is an object with 'approved' (bool), 'message' (str), "
            "and optional 'data' (dict).  When provided, HITL steps use these "
            "instead of blocking for real human input."
        ),
    )
    trace: bool = Field(False, description="Include trace data in response.")
    introspect: bool = Field(
        False,
        description=(
            "Include detailed introspection data: transitions audit trail, "
            "per-step state history, and graph description."
        ),
    )


class TransitionRecord(BaseModel):
    """Serialised :class:`StateTransition` from the workflow audit trail."""

    step: str
    keys_changed: list[str]
    branch_taken: str | None = None
    duration_ms: float = 0.0
    retries: int = 0
    error: str | None = None


class IntrospectionData(BaseModel):
    """Detailed execution introspection returned when ``introspect=true``."""

    transitions: list[TransitionRecord] = Field(
        default_factory=list,
        description="Audit trail — one record per step execution.",
    )
    state_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step state snapshots in execution order.",
    )
    graph: str = Field(
        "",
        description="Human-readable graph description (steps, edges, branches).",
    )
    ascii_diagram: str = Field(
        "",
        description="ASCII tree rendering of the workflow.",
    )


class WorkflowResponse(BaseModel):
    """Output from ``POST /workflow/{workflow_name}``."""

    result: dict[str, Any]
    workflow: str
    steps_executed: list[str]
    duration_ms: float
    trace: dict[str, Any] | None = None
    introspection: IntrospectionData | None = None


class SkillRequest(BaseModel):
    """Input for ``POST /skill/{skill_name}`` — run a registered skill.

    Only the input message is required.  Provider and model are defined
    inside the skill but can be overridden.
    """

    input: str = Field(..., description="Input text for the skill.", max_length=100_000)
    provider: str | None = Field(None, description="Provider override.")
    trace: bool = Field(False, description="Include trace data in response.")


class SkillResponse(BaseModel):
    """Output from ``POST /skill/{skill_name}``."""

    result: str
    skill: str
    duration_ms: float
    trace: dict[str, Any] | None = None


# ── Built-in agent: feedback analysis ─────────────────────────────


def _build_feedback_pipeline(
    provider: str = "google",
    model: str | None = None,
    **kwargs: Any,
) -> SequentialAgent:
    """Build the feedback analysis demo pipeline.

    Stages: extract → classify → analyse → redact.
    """
    common: dict[str, Any] = {}
    if model:
        common["model"] = model

    extract = tasker_tool(
        name="extract_feedback",
        description="Extract structured fields from raw feedback text.",
        system_prompt=(
            "You are a data extraction specialist. Extract structured data "
            "from raw customer feedback. Return a JSON array of objects with "
            "fields: id, user, message, contact_info, date. "
            "Use null for missing fields."
        ),
        provider=provider,
        temperature=0.1,
        **common,
    )

    classify = tasker_tool(
        name="classify_feedback",
        description="Classify each feedback item by category and sentiment.",
        system_prompt=(
            "You are a feedback classifier. For each item, assign:\n"
            "- category: bug, feature_request, billing, performance, praise\n"
            "- sentiment: positive, negative, neutral\n"
            "- priority: critical, high, medium, low\n"
            "Return a JSON array with id, category, sentiment, priority."
        ),
        provider=provider,
        temperature=0.1,
        **common,
    )

    redact = tasker_tool(
        name="redact_pii",
        description="Remove PII from text.",
        system_prompt=(
            "Replace all personally identifiable information with "
            "placeholders: [NAME], [EMAIL], [CARD], [PHONE]. "
            "Return the cleaned text only."
        ),
        provider=provider,
        temperature=0.0,
        **common,
    )

    extractor = Agent(
        name="extractor",
        provider=provider,
        instruction="Use extract_feedback on the input. Return ONLY the JSON.",
        tools=[extract],
        temperature=0.1,
        output_format="json",
        **common,
    )

    classifier = Agent(
        name="classifier",
        provider=provider,
        instruction="Use classify_feedback on the data. Return ONLY the JSON.",
        tools=[classify],
        temperature=0.1,
        output_format="json",
        **common,
    )

    analyst = Agent(
        name="analyst",
        provider=provider,
        instruction=(
            "You are a customer feedback analyst. Analyse the classified data "
            "and write a concise Markdown report with: executive summary, "
            "key metrics, top issues, positive highlights, and "
            "3 recommendations."
        ),
        temperature=0.5,
        **common,
    )

    guardrail = Agent(
        name="guardrail",
        provider=provider,
        instruction="Use redact_pii on the report. Return the redacted text.",
        tools=[redact],
        temperature=0.0,
        **common,
    )

    return SequentialAgent(
        name="feedback_pipeline",
        description="Extract → Classify → Analyse → Redact",
        sub_agents=[extractor, classifier, analyst, guardrail],
    )


# ── Agent registry ───────────────────────────────────────────────

_AGENT_BUILDERS: dict[str, Any] = {
    # Single agents (LlmAgent templates)
    "planner": planner_agent,
    "decomposer": decomposer_agent,
    "summarizer": summarizer_agent,
    "reviewer": reviewer_agent,
    "coder": coder_agent,
    "classifier": classifier_agent,
    "extractor": extractor_agent,
    "writer": writer_agent,
    "guardrail": guardrail_agent,
    # Multi-agent compositions (SequentialAgent / LoopAgent / RouterAgent)
    "feedback_analysis": _build_feedback_pipeline,
    "plan_and_execute": plan_and_execute,
    "research_and_write": research_and_write,
    "draft_review_loop": draft_review_loop,
    "classify_and_route": classify_and_route,
}


# ── Workflow registry ───────────────────────────────────────────

_WORKFLOW_BUILDERS: dict[str, Any] = {
    "sentiment_pipeline": build_sentiment_pipeline,
    "content_pipeline": build_content_pipeline,
    "data_enrichment": build_data_enrichment,
    "content_review_pipeline": build_content_review_pipeline,
}

# Builders that accept a handler keyword argument for HITL.
_HITL_WORKFLOW_BUILDERS: set[str] = {"content_review_pipeline"}


# ── Helper: resolve tasks directory ──────────────────────────────

_TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasker", "prompts")
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "tasker", "templates")

_PROJECTS_DIR = os.environ.get("NONO_PROJECTS_DIR", os.getcwd())


@functools.lru_cache(maxsize=1)
def _list_tasks() -> tuple[str, ...]:
    """Return names of available JSON task files (cached)."""
    if not os.path.isdir(_TASKS_DIR):
        return ()
    return tuple(
        f.removesuffix(".json")
        for f in sorted(os.listdir(_TASKS_DIR))
        if f.endswith(".json")
    )


@functools.lru_cache(maxsize=1)
def _list_templates() -> tuple[str, ...]:
    """Return names of available Jinja2 templates (cached)."""
    if not os.path.isdir(_TEMPLATES_DIR):
        return ()
    return tuple(
        f.removesuffix(".j2")
        for f in sorted(os.listdir(_TEMPLATES_DIR))
        if f.endswith(".j2")
    )


def _list_projects() -> dict[str, str]:
    """Scan ``_PROJECTS_DIR`` for directories containing ``nono.toml``.

    Returns:
        Dict mapping project name → absolute directory path.
    """
    projects = list_projects(_PROJECTS_DIR)
    return {p.name: str(p.root) for p in projects}


# ── Helper: extract trace data ───────────────────────────────────


def _export_trace(collector: TraceCollector) -> dict[str, Any]:
    """Convert a TraceCollector to a serialisable dict."""
    exported = collector.export()
    return {
        "traces": [
            {
                "agent": t.get("agent_name", ""),
                "status": t.get("status", ""),
                "duration_ms": t.get("duration_ms", 0),
                "llm_calls": t.get("llm_calls", []),
                "tools_used": t.get("tools_used", []),
            }
            for t in exported
        ],
    }


# ── Endpoints ────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/info")
async def info() -> dict[str, Any]:
    """List all available named resources on the server."""
    return {
        "name": "Nono API Server",
        "version": __version__,
        "endpoints": {
            "GET  /health": "Health check",
            "GET  /info": "This endpoint — list named resources",
            "POST /task/{task_name}": "Run a built-in JSON task",
            "POST /agent/{agent_name}": "Run a pre-built agent template",
            "POST /agent/{agent_name}/stream": "Stream agent events (SSE)",
            "POST /workflow/{workflow_name}": "Run a pre-built workflow",
            "GET  /workflow/{workflow_name}/describe": "Workflow graph structure",
            "POST /skill/{skill_name}": "Run a registered skill",
            "GET  /projects": "List discovered projects",
            "GET  /project/{project_name}": "Project details and resources",
        },
        "tasks": _list_tasks(),
        "templates": _list_templates(),
        "agents": sorted(_AGENT_BUILDERS.keys()),
        "workflows": sorted(_WORKFLOW_BUILDERS.keys()),
        "skills": skill_registry.names,
        "projects": list(_list_projects().keys()),
    }


# ── Task endpoint ────────────────────────────────────────────────


@app.post("/task/{task_name}", response_model=TaskResponse)
def run_task(
    task_name: str = Path(..., description="Name of the built-in task (from GET /info)."),
    req: TaskRequest = TaskRequest(),
) -> TaskResponse:
    """Run a built-in JSON task by name.

    The ``task_name`` must match a ``.json`` file in
    ``nono/tasker/prompts/``.  Use ``GET /info`` to list available tasks.

    Provider, model, temperature, and all AI parameters are read from
    the task JSON file.  The caller only supplies data inputs.
    """
    from nono.tasker import TaskExecutor

    task_path = os.path.join(_TASKS_DIR, f"{task_name}.json")

    # Path traversal protection
    real_tasks_dir = os.path.realpath(_TASKS_DIR) + os.sep
    real_task_path = os.path.realpath(task_path)
    if not real_task_path.startswith(real_tasks_dir):
        raise HTTPException(status_code=400, detail="Invalid task name.")

    if not os.path.isfile(task_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Task '{task_name}' not found. "
                f"Available: {_list_tasks()}"
            ),
        )

    # Read provider/model from the task's genai section
    with open(task_path, "r", encoding="utf-8") as f:
        task_def = json.load(f)

    genai_cfg = task_def.get("genai", {})
    provider = genai_cfg.get("provider", "google")
    model = genai_cfg.get("model", "")

    start = time.perf_counter()

    executor = TaskExecutor(
        provider=provider,
        model=model,
    )

    collector = TraceCollector() if req.trace else None

    result = executor.run_json_task(
        task_file=task_path,
        data_input=req.data_input,
        trace_collector=collector,
        **req.data_inputs,
    )
    duration_ms = (time.perf_counter() - start) * 1000

    return TaskResponse(
        result=result,
        task=task_name,
        provider=provider,
        model=executor.config.model_name,
        duration_ms=round(duration_ms, 1),
        trace=_export_trace(collector) if collector else None,
    )


# ── Agent endpoint ───────────────────────────────────────────────


@app.post("/agent/{agent_name}", response_model=AgentResponse)
def run_agent(
    req: AgentRequest,
    agent_name: str = Path(
        ...,
        description="Name of the agent template (from GET /info).",
    ),
) -> AgentResponse:
    """Run a pre-built agent by name.

    This endpoint handles both single agents (LlmAgent templates) and
    multi-agent compositions (SequentialAgent, LoopAgent, RouterAgent).
    Use ``GET /info`` to list all available agents.
    """
    builder = _AGENT_BUILDERS.get(agent_name)
    if not builder:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Agent '{agent_name}' not found. "
                f"Available: {sorted(_AGENT_BUILDERS.keys())}"
            ),
        )

    agent = builder()
    collector = TraceCollector() if req.trace else None
    runner = Runner(agent=agent, trace_collector=collector)

    start = time.perf_counter()
    result = runner.run(req.message)
    duration_ms = (time.perf_counter() - start) * 1000

    # Single LlmAgents expose provider/model; compositions do not.
    provider = getattr(agent, "_provider", None)
    model = agent.service.model_name if hasattr(agent, "service") else None

    introspection = None
    if req.introspect:
        introspection = AgentIntrospectionData(
            events=[
                EventRecord(
                    event_type=e.event_type.value,
                    author=e.author,
                    content=e.content,
                    timestamp=e.timestamp.isoformat(),
                    event_id=e.event_id,
                )
                for e in runner.session.events
            ],
            state=runner.session.state,
            event_count=len(runner.session),
        )

    return AgentResponse(
        result=result,
        agent=agent_name,
        provider=provider,
        model=model,
        duration_ms=round(duration_ms, 1),
        trace=_export_trace(collector) if collector else None,
        introspection=introspection,
    )


@app.post("/agent/{agent_name}/stream")
async def run_agent_stream(
    req: AgentRequest,
    agent_name: str = Path(
        ...,
        description="Name of the agent to stream.",
    ),
):
    """Stream agent events as Server-Sent Events (SSE).

    Each event is a JSON line with ``agent``, ``type``, and ``content``.
    Works with both single agents and multi-agent compositions.

    Uses an async generator so Starlette can cancel on client disconnect.
    """
    builder = _AGENT_BUILDERS.get(agent_name)
    if not builder:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found.",
        )

    agent = builder()
    runner = Runner(agent=agent)

    async def event_generator():
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=1024)
        loop = asyncio.get_running_loop()

        def _produce() -> None:
            try:
                for event in runner.stream(req.message):
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "agent": event.author,
                            "type": event.event_type.value,
                            "content": event.content,
                            "timestamp": event.timestamp.isoformat(),
                        },
                    )
            except Exception as exc:
                logger.error("SSE stream error: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {
                        "agent": "system",
                        "type": "error",
                        "content": "Internal error during streaming.",
                        "timestamp": __import__("datetime").datetime.now(
                            tz=__import__("datetime").timezone.utc
                        ).isoformat(),
                    },
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        await task
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ── Workflow endpoint ────────────────────────────────────────────


@app.post("/workflow/{workflow_name}", response_model=WorkflowResponse)
def run_workflow(
    req: WorkflowRequest,
    workflow_name: str = Path(
        ...,
        description="Name of the workflow template (from GET /info).",
    ),
) -> WorkflowResponse:
    """Run a pre-built workflow by name.

    Workflows are graph-based execution pipelines with conditional
    branching.  Each step processes a shared state dict.
    Use ``GET /info`` to list available workflows.
    """
    builder = _WORKFLOW_BUILDERS.get(workflow_name)
    if not builder:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Workflow '{workflow_name}' not found. "
                f"Available: {sorted(_WORKFLOW_BUILDERS.keys())}"
            ),
        )

    # Inject HITL handler when pre-configured responses are provided.
    if workflow_name in _HITL_WORKFLOW_BUILDERS and req.hitl_responses:
        handler = make_auto_handler(req.hitl_responses)
        workflow = builder(handler=handler)
    else:
        workflow = builder()

    collector = TraceCollector() if req.trace else None

    start = time.perf_counter()
    steps_executed: list[str] = []
    final_state: Any = {}

    if req.trace:
        # Use streaming to capture step names
        for step_name, _state in workflow.stream(
            trace_collector=collector, **req.state
        ):
            steps_executed.append(step_name)
            final_state = _state
    else:
        # Simple run — collect step names via streaming
        for step_name, _state in workflow.stream(**req.state):
            steps_executed.append(step_name)
            final_state = _state

    duration_ms = (time.perf_counter() - start) * 1000

    introspection = None
    if req.introspect:
        introspection = IntrospectionData(
            transitions=[
                TransitionRecord(
                    step=t.step,
                    keys_changed=sorted(t.keys_changed),
                    branch_taken=t.branch_taken,
                    duration_ms=round(t.duration_ms, 2),
                    retries=t.retries,
                    error=t.error,
                )
                for t in workflow.transitions
            ],
            state_history=[
                {"step": step, "state": snapshot}
                for step, snapshot in workflow.get_history()
            ],
            graph=workflow.describe(),
            ascii_diagram=workflow.draw(),
        )

    return WorkflowResponse(
        result=final_state,
        workflow=workflow_name,
        steps_executed=steps_executed,
        duration_ms=round(duration_ms, 1),
        trace=_export_trace(collector) if collector else None,
        introspection=introspection,
    )


# ── Workflow introspection endpoint ──────────────────────────────


@app.get("/workflow/{workflow_name}/describe")
def describe_workflow(
    workflow_name: str = Path(
        ...,
        description="Name of the workflow template (from GET /info).",
    ),
) -> dict[str, Any]:
    """Return the graph structure of a workflow without executing it.

    Useful for external tools, dashboards, and documentation generators
    that need to visualise or validate workflow topology before running.
    """
    builder = _WORKFLOW_BUILDERS.get(workflow_name)
    if not builder:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Workflow '{workflow_name}' not found. "
                f"Available: {sorted(_WORKFLOW_BUILDERS.keys())}"
            ),
        )

    workflow = builder()

    schema_info = None
    if workflow.schema is not None:
        schema_info = {
            "fields": {k: v.__name__ for k, v in workflow.schema.fields.items()},
            "reducers": sorted(workflow.schema.reducers.keys()),
        }

    return {
        "workflow": workflow_name,
        "steps": workflow.steps,
        "graph": workflow.describe(),
        "ascii_diagram": workflow.draw(),
        "schema": schema_info,
    }


# ── Skill endpoint ───────────────────────────────────────────────


@app.post("/skill/{skill_name}", response_model=SkillResponse)
def run_skill(
    req: SkillRequest,
    skill_name: str = Path(
        ...,
        description="Name of the registered skill (from GET /info).",
    ),
) -> SkillResponse:
    """Run a registered skill by name.

    Skills are reusable AI capabilities (summarize, classify, extract,
    etc.) that can be invoked standalone.  Use ``GET /info`` to list
    all available skills.
    """
    skill = skill_registry.get(skill_name)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Skill '{skill_name}' not found. "
                f"Available: {skill_registry.names}"
            ),
        )

    overrides: dict[str, Any] = {}
    if req.provider:
        overrides["provider"] = req.provider

    collector = TraceCollector() if req.trace else None
    start = time.perf_counter()

    agent = skill.build_agent(**overrides)
    runner = Runner(agent=agent, trace_collector=collector)
    result = runner.run(req.input)

    duration_ms = (time.perf_counter() - start) * 1000

    return SkillResponse(
        result=result,
        skill=skill_name,
        duration_ms=round(duration_ms, 1),
        trace=_export_trace(collector) if collector else None,
    )


# ── Project endpoints ────────────────────────────────────────────


@app.get("/projects")
async def list_projects_endpoint() -> dict[str, Any]:
    """List all discovered projects.

    Scans the directory configured by ``NONO_PROJECTS_DIR`` (defaults
    to the server's working directory) for subdirectories containing a
    ``nono.toml`` manifest.
    """
    projects = _list_projects()
    summaries: list[dict[str, Any]] = []

    for name, root in projects.items():
        try:
            proj = load_project(root)
            summaries.append({
                "name": proj.name,
                "description": proj.description,
                "version": proj.version,
                "root": str(proj.root),
            })
        except Exception:
            summaries.append({"name": name, "root": root, "error": "failed to load"})

    return {
        "projects_dir": _PROJECTS_DIR,
        "count": len(summaries),
        "projects": summaries,
    }


@app.get("/project/{project_name}")
async def get_project(
    project_name: str = Path(
        ...,
        description="Project name (from GET /projects).",
    ),
) -> dict[str, Any]:
    """Get details and resource counts for a specific project."""
    projects = _list_projects()
    root = projects.get(project_name)

    if not root:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Project '{project_name}' not found. "
                f"Available: {sorted(projects.keys())}"
            ),
        )

    proj = load_project(root)

    skills = proj.load_skills()
    prompts = proj.list_prompts()
    templates = proj.list_templates()
    workflows = proj.list_workflows()
    data_files = proj.list_data()

    return {
        "name": proj.name,
        "description": proj.description,
        "version": proj.version,
        "root": str(proj.root),
        "provider": proj.manifest.default_provider or None,
        "model": proj.manifest.default_model or None,
        "resources": {
            "skills": len(skills),
            "prompts": len(prompts),
            "templates": len(templates),
            "workflows": len(workflows),
            "data_files": len(data_files),
        },
        "skills": [
            {"name": s.descriptor.name, "description": s.descriptor.description}
            for s in skills
        ],
        "prompts": [p.stem for p in prompts],
        "templates": [t.stem for t in templates],
        "workflows": [w.stem for w in workflows],
        "data_files": [f.name for f in data_files],
    }


# ── Routine endpoints ────────────────────────────────────────────


class RoutineFireRequest(BaseModel):
    """Input for ``POST /routine/{routine_name}/fire``."""

    text: str = Field("", description="Free-form context text for the routine execution.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value context data.",
    )


class RoutineFireResponse(BaseModel):
    """Output from ``POST /routine/{routine_name}/fire``."""

    routine_name: str
    run_id: str
    status: str
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0


@app.get("/routines")
async def list_routines() -> dict[str, Any]:
    """List all registered routines and their current status."""
    return _routine_runner.status()


@app.post("/routine/{routine_name}/fire", response_model=RoutineFireResponse)
async def fire_routine(
    routine_name: str = Path(
        ...,
        description="Name of the routine to fire.",
    ),
    req: RoutineFireRequest = RoutineFireRequest(),
) -> RoutineFireResponse:
    """Manually fire a registered routine with optional context."""
    try:
        context = req.context.copy()
        if req.text:
            context["text"] = req.text

        result = await asyncio.to_thread(
            _routine_runner.fire,
            routine_name,
            context=context,
            trigger_type="api",
        )
        return RoutineFireResponse(
            routine_name=result.routine_name,
            run_id=result.run_id,
            status=result.status.value,
            output=result.output[:1000],
            error=result.error,
            duration_seconds=result.duration_seconds,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Routine '{routine_name}' not found",
        )


@app.get("/routine/{routine_name}/history")
async def routine_history(
    routine_name: str = Path(
        ...,
        description="Name of the routine.",
    ),
) -> dict[str, Any]:
    """Get execution history for a routine."""
    try:
        _routine_runner.get(routine_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Routine '{routine_name}' not found",
        )

    history = _routine_runner.get_history(routine_name)
    return {
        "routine_name": routine_name,
        "total_runs": len(history),
        "runs": [r.to_dict() for r in history],
    }


@app.post("/routine/{routine_name}/pause")
async def pause_routine(
    routine_name: str = Path(
        ...,
        description="Name of the routine to pause.",
    ),
) -> dict[str, str]:
    """Pause a routine — triggers are ignored until resumed."""
    try:
        _routine_runner.pause(routine_name)
        return {"status": "paused", "routine": routine_name}
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Routine '{routine_name}' not found",
        )


@app.post("/routine/{routine_name}/resume")
async def resume_routine(
    routine_name: str = Path(
        ...,
        description="Name of the routine to resume.",
    ),
) -> dict[str, str]:
    """Resume a paused routine."""
    try:
        _routine_runner.resume(routine_name)
        return {"status": "resumed", "routine": routine_name}
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Routine '{routine_name}' not found",
        )
