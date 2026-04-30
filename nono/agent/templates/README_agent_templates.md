# Agent Templates

> Pre-configured agents and multi-agent pipelines for common GenAI tasks.

## Table of Contents

- [Overview](#overview)
- [Individual Agents](#individual-agents)
- [Pipelines (Combinations)](#pipelines-combinations)
- [Quick Start](#quick-start)
- [Customisation](#customisation)
- [Agent–Tasker Correspondence](#agenttasker-correspondence)

---

## Overview

The `templates` package provides **factory functions** that return ready-to-use agents.
Each function creates a configured `LlmAgent` (or composite agent) with sensible defaults
that you can override via keyword arguments.

```
nono/agent/templates/
├── __init__.py          # Re-exports everything
├── planner.py           # Strategic planning
├── decomposer.py        # Task decomposition
├── summarizer.py        # Text summarization
├── reviewer.py          # Quality review & critique
├── coder.py             # Code generation
├── classifier.py        # Classification & routing
├── extractor.py         # Structured data extraction
├── writer.py            # Content writing
├── guardrail.py         # PII / safety checks
└── pipelines.py         # Multi-agent combinations
```

---

## Individual Agents

| Factory | Default name | Output | Temp | Purpose |
|---------|-------------|--------|------|---------|
| `planner_agent()` | `planner` | JSON | 0.4 | Generate phased project plans with milestones |
| `decomposer_agent()` | `decomposer` | JSON | 0.3 | Break complex tasks into ordered subtasks |
| `summarizer_agent()` | `summarizer` | JSON | 0.3 | Summarize text preserving key facts |
| `reviewer_agent()` | `reviewer` | JSON | 0.3 | Evaluate quality and provide actionable feedback |
| `coder_agent()` | `coder` | text | 0.2 | Generate production-ready code |
| `classifier_agent()` | `classifier` | JSON | 0.1 | Classify inputs into categories |
| `extractor_agent()` | `extractor` | JSON | 0.1 | Extract entities/facts from unstructured text |
| `writer_agent()` | `writer` | text | 0.7 | Write articles, emails, reports, etc. |
| `guardrail_agent()` | `guardrail` | JSON | 0.0 | Detect PII, safety issues, policy violations |

---

## Pipelines (Combinations)

### Original Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `plan_and_execute()` | `SequentialAgent` | planner → decomposer → coder | Deterministic sequence |
| `research_and_write()` | `SequentialAgent` | extractor → writer → reviewer | Deterministic sequence |
| `draft_review_loop()` | `LoopAgent` | writer ↔ reviewer × N | Iterative refinement |
| `classify_and_route()` | `RouterAgent` | classifier → {coder, writer, summarizer, extractor} | LLM routing |

### Development Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `bug_fix()` | `SequentialAgent` | triager → debugger → fixer → tester → reviewer | Deterministic sequence |
| `refactoring()` | `SequentialAgent` | code_analyzer → planner → refactorer → tester → reviewer | Deterministic sequence |
| `product_development()` | `SequentialAgent` | product_designer → planner → developer → reviewer | Deterministic sequence |
| `code_review_automation()` | `SequentialAgent` | diff_analyzer → [style ‖ logic ‖ security] → summary | Fan-out/fan-in |
| `performance_optimization()` | `SequentialAgent` | profiler → bottleneck_analyzer → optimizer → benchmarker → reviewer | Deterministic sequence |
| `test_suite_generation()` | `SequentialAgent` | code_analyzer → test_planner → test_writer → coverage → mutation | Deterministic sequence |

### Architecture Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `system_design()` | `SequentialAgent` | requirements_analyst → architect → reviewer → decision_logger | Deterministic sequence |
| `database_design()` | `SequentialAgent` | domain_modeler → schema_designer → migrator → validator | Deterministic sequence |
| `api_design()` | `SequentialAgent` | domain_expert → api_designer → implementer → doc_gen → consumer_tester | Deterministic sequence |

### Operations Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `incident_response()` | `SequentialAgent` | detector → diagnostician → responder → rca_analyst → postmortem | Deterministic sequence |
| `devops_deployment()` | `SequentialAgent` | build_agent → security_scanner → deployer → monitor | Deterministic sequence |
| `cost_optimization()` | `SequentialAgent` | resource_scanner → usage_analyzer → optimizer → validator | Deterministic sequence |
| `observability_setup()` | `SequentialAgent` | signal_identifier → instrumenter → dashboard_builder → alert_tuner | Deterministic sequence |
| `disaster_recovery()` | `SequentialAgent` | risk_assessor → runbook_writer → simulator → validator → certifier | Deterministic sequence |
| `migration()` | `SequentialAgent` | legacy_analyzer → target_designer → migrator → validator → deployer | Deterministic sequence |

### Data Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `data_quality()` | `SequentialAgent` | profiler → rule_designer → validator → cleaner → reporter | Deterministic sequence |
| `etl_pipeline_design()` | `SequentialAgent` | source_analyzer → transform_designer → implementer → validator → scheduler | Deterministic sequence |

### AI/ML Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `prompt_engineering()` | `LoopAgent` | task_analyzer → prompt_drafter → variation_gen → evaluator → optimizer | Iterative refinement |
| `rag_pipeline_design()` | `LoopAgent` | corpus_analyzer → chunking → embedding → retriever → e2e_evaluator | Iterative refinement |
| `model_fine_tuning()` | `LoopAgent` | data_curator → preprocessor → trainer → evaluator → publisher | Iterative refinement |
| `ai_safety_guardrails()` | `LoopAgent` | risk_cataloger → red_teamer → guardrail_designer → tester → certifier | Iterative refinement |

### Content & Knowledge Pipelines

| Factory | Type | Agents | Pattern |
|---------|------|--------|---------|
| `content_documentation()` | `SequentialAgent` | researcher → writer → tech_reviewer → publisher | Deterministic sequence |
| `research()` | `SequentialAgent` | question_formulator → source_finder → analyzer → report_writer | Deterministic sequence |
| `security_audit()` | `SequentialAgent` | threat_modeler → static_analyzer → pen_tester → remediator → verifier | Deterministic sequence |
| `compliance()` | `SequentialAgent` | evidence_collector → gap_analyzer → remediator → auditor → reporter | Deterministic sequence |

---

## Quick Start

### Single agent

```python
from nono.agent.templates import planner_agent
from nono.agent import Runner

planner = planner_agent()
runner = Runner(agent=planner)
events = runner.run("Plan a REST API migration from Flask to FastAPI")

for event in events:
    print(event.content)
```

### Pipeline — Plan & Execute

```python
from nono.agent.templates import plan_and_execute
from nono.agent import Runner

pipeline = plan_and_execute(provider="openai", model="gpt-4o-mini")
runner = Runner(agent=pipeline)

for event in runner.run("Build a CLI tool that converts CSV to Parquet"):
    print(f"[{event.agent}] {event.content[:120]}")
```

### Pipeline — Draft & Review Loop

```python
from nono.agent.templates import draft_review_loop
from nono.agent import Runner

loop = draft_review_loop(max_iterations=3)
runner = Runner(agent=loop)

for event in runner.run("Write a technical blog post about async Python"):
    print(f"[{event.agent}] {event.content[:120]}")
```

### Pipeline — Classify & Route

```python
from nono.agent.templates import classify_and_route
from nono.agent import Runner

router = classify_and_route()
runner = Runner(agent=router)

# The router picks the best specialist automatically
for event in runner.run("Write a Python function that sorts a list"):
    print(f"[{event.agent}] {event.content[:120]}")
```

### Pipeline with custom routes

```python
from nono.agent.templates import classify_and_route, coder_agent, guardrail_agent
from nono.agent import Runner

router = classify_and_route(
    routes={
        "code": coder_agent(name="code_specialist"),
        "safety": guardrail_agent(name="safety_check"),
    },
    routing_instruction="Route code requests to code_specialist, "
                        "everything else to safety_check.",
)
runner = Runner(agent=router)
```

---

## Customisation

Every factory accepts keyword arguments forwarded to `LlmAgent`:

```python
from nono.agent.templates import reviewer_agent

# Override model, provider, and instruction
strict_reviewer = reviewer_agent(
    name="strict_reviewer",
    model="gpt-4o",
    provider="openai",
    instruction="You are a ruthless code reviewer. Reject anything "
                "that does not have 100% test coverage.",
    temperature=0.0,
)
```

Common overrides:

| Parameter | Description |
|-----------|-------------|
| `model` | LLM model identifier (default: config.toml value) |
| `provider` | `google`, `openai`, `groq`, `cerebras`, … |
| `instruction` | Full system prompt override |
| `temperature` | Sampling temperature |
| `output_format` | `"json"` or `"text"` |
| `tools` | List of `FunctionTool` instances |
| `api_key` | Provider API key (overrides config) |

---

## Agent–Tasker Correspondence

Several agent templates mirror existing Jinja2 tasker templates:

| Agent template | Tasker template | Shared role |
|---------------|----------------|-------------|
| `planner_agent` | `planner.j2` | Strategic planning |
| `decomposer_agent` | `decompose_tasks.j2` | Task decomposition |
| `coder_agent` | `python_programming.j2` | Code generation |
| `classifier_agent` | `conditional_flow.j2` | Classification & routing |
| `extractor_agent` | `semantic_lookup.j2` | Data extraction |
| `guardrail_agent` | `data_loss_prevention.j2` | PII & safety |

The difference: **tasker templates** are stateless prompt templates rendered once;
**agent templates** are interactive agents with memory, tool calling, and
multi-turn conversation.
