"""Pipelines — Pre-built multi-agent combinations for common patterns.

Includes development, architecture, operations, data, AI/ML, and content
pipelines mapped from the Prompter catalog.
"""

from __future__ import annotations

from ..llm_agent import LlmAgent
from ..workflow_agents import (
    LoopAgent,
    ParallelAgent,
    RouterAgent,
    SequentialAgent,
)
from .planner import planner_agent
from .decomposer import decomposer_agent
from .summarizer import summarizer_agent
from .reviewer import reviewer_agent
from .coder import coder_agent
from .classifier import classifier_agent
from .extractor import extractor_agent
from .writer import writer_agent
from .guardrail import guardrail_agent


# ═══════════════════════════════════════════════════════════════════════════
# Helper — inline agent factory
# ═══════════════════════════════════════════════════════════════════════════

def _agent(
    name: str,
    instruction: str,
    description: str,
    *,
    output_format: str = "text",
    temperature: float = 0.3,
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> LlmAgent:
    """Create a lightweight inline agent for pipeline stages."""
    return LlmAgent(
        name=name,
        model=model,
        provider=provider,
        instruction=instruction,
        description=description,
        output_format=output_format,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Plan & Execute — Planner → Decomposer → Coder
# ---------------------------------------------------------------------------

def plan_and_execute(
    *,
    name: str = "plan_and_execute",
    model: str | None = None,
    provider: str = "google",
    **kwargs,
) -> SequentialAgent:
    """Pipeline: plan a goal → decompose into tasks → generate code.

    Stages:
        1. **Planner** creates a phased plan.
        2. **Decomposer** breaks it into actionable subtasks.
        3. **Coder** implements each subtask.

    The planner output flows to the decomposer via conversation history,
    then the decomposer output flows to the coder.

    Args:
        name: Pipeline name.
        model: LLM model (shared by all stages). ``None`` for config default.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the three stages.
    """
    common = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Plan a goal, decompose into tasks, and generate code.",
        sub_agents=[
            planner_agent(**common),
            decomposer_agent(**common),
            coder_agent(**common),
        ],
    )


# ---------------------------------------------------------------------------
# 2. Research & Write — Extractor → Writer → Reviewer
# ---------------------------------------------------------------------------

def research_and_write(
    *,
    name: str = "research_and_write",
    model: str | None = None,
    provider: str = "google",
    **kwargs,
) -> SequentialAgent:
    """Pipeline: extract data → write content → review quality.

    Stages:
        1. **Extractor** pulls structured data from raw sources.
        2. **Writer** composes a document from the extracted data.
        3. **Reviewer** evaluates quality and flags issues.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the three stages.
    """
    common = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Extract data, write content, and review it.",
        sub_agents=[
            extractor_agent(**common),
            writer_agent(**common),
            reviewer_agent(**common),
        ],
    )


# ---------------------------------------------------------------------------
# 3. Draft & Review Loop — Writer ↔ Reviewer (iterative)
# ---------------------------------------------------------------------------

def draft_review_loop(
    *,
    name: str = "draft_review_loop",
    model: str | None = None,
    provider: str = "google",
    max_iterations: int = 3,
    **kwargs,
) -> LoopAgent:
    """Pipeline: writer drafts → reviewer critiques → repeat until approved.

    The :class:`LoopAgent` runs writer → reviewer for up to
    *max_iterations* rounds. A custom ``stop_condition`` can inspect
    ``session.state`` to break early (e.g. when the reviewer verdict
    is ``"approve"``).

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        max_iterations: Maximum write/review cycles.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`LoopAgent` alternating writer and reviewer.
    """
    common = dict(model=model, provider=provider, **kwargs)
    return LoopAgent(
        name=name,
        description="Iteratively draft and review until quality is met.",
        sub_agents=[
            writer_agent(**common),
            reviewer_agent(**common),
        ],
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# 4. Classify & Route — Classifier → specialised agents
# ---------------------------------------------------------------------------

def classify_and_route(
    *,
    name: str = "classify_and_route",
    model: str | None = None,
    provider: str = "google",
    routes: dict[str, LlmAgent] | None = None,
    routing_instruction: str = "",
    **kwargs,
) -> RouterAgent:
    """Pipeline: LLM router dispatches to specialised agents by category.

    By default provides four routes: ``coder``, ``writer``, ``summarizer``,
    and ``extractor``.  Pass *routes* to override with custom agents.

    Args:
        name: Pipeline name.
        model: LLM model for the router and default sub-agents.
        provider: AI provider.
        routes: ``{label: agent}`` mapping. Overrides the defaults.
        routing_instruction: Extra routing hints for the LLM.
        **kwargs: Extra args forwarded to default sub-agents.

    Returns:
        A :class:`RouterAgent` that classifies and dispatches input.
    """
    common = dict(model=model, provider=provider, **kwargs)
    if routes:
        agents = list(routes.values())
    else:
        agents = [
            coder_agent(**common),
            writer_agent(**common),
            summarizer_agent(**common),
            extractor_agent(**common),
        ]
    return RouterAgent(
        name=name,
        description="Classify input and route to the best specialist.",
        sub_agents=agents,
        model=model,
        provider=provider,
        routing_instruction=routing_instruction,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DEVELOPMENT PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 5. Bug Fix — Triager → Debugger → Fixer → Tester → Reviewer
# ---------------------------------------------------------------------------

def bug_fix(
    *,
    name: str = "bug_fix",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: triage a bug → debug root cause → fix → test → review.

    Stages:
        1. **Triager** classifies, prioritises, and reproduces the bug.
        2. **Debugger** locates root cause via code reading and tracing.
        3. **Fixer** implements the minimal fix plus a regression test.
        4. **Tester** validates the fix and runs the full suite.
        5. **Reviewer** reviews fix quality and approves/rejects.

    Args:
        name: Pipeline name.
        model: LLM model. ``None`` for config default.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Triage, debug, fix, test, and review a bug.",
        sub_agents=[
            _agent(
                "triager",
                "You are a bug triage specialist. Classify severity, identify "
                "the affected component, and write clear reproduction steps. "
                "Output a structured bug report with severity, component, "
                "impact, and step-by-step repro.",
                "Classifies and reproduces bugs.",
                **c,
            ),
            _agent(
                "debugger",
                "You are an expert debugger. Given a bug report, locate the "
                "root cause by analysing code, logs, and traces. Identify the "
                "exact file, function, and failing condition. Output a root "
                "cause document.",
                "Locates root cause of bugs.",
                **c,
            ),
            coder_agent(name="fixer", description="Implements minimal bug fix with regression test.", **c),
            _agent(
                "tester",
                "You are a QA engineer. Given a code fix, verify it resolves "
                "the original bug, run the full test suite, and report any "
                "regressions. Output a test report with coverage and results.",
                "Validates fixes and detects regressions.",
                **c,
            ),
            reviewer_agent(**c),
        ],
    )


# ---------------------------------------------------------------------------
# 6. Refactoring — Code Analyzer → Planner → Refactorer → Tester → Reviewer
# ---------------------------------------------------------------------------

def refactoring(
    *,
    name: str = "refactoring",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: analyse smells → plan refactoring → execute → test → review.

    Stages:
        1. **Code Analyzer** identifies code smells and tech debt.
        2. **Planner** designs a step-by-step refactoring strategy.
        3. **Refactorer** executes the changes.
        4. **Tester** verifies no regressions.
        5. **Reviewer** confirms metrics improved without behaviour changes.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Analyse code smells, plan and execute refactoring safely.",
        sub_agents=[
            _agent(
                "code_analyzer",
                "You are a code quality analyst. Identify code smells, "
                "duplication, excessive complexity, tight coupling, and tech "
                "debt. Output a prioritised smell report.",
                "Identifies code smells and tech debt.",
                **c,
            ),
            planner_agent(**c),
            coder_agent(name="refactorer", description="Executes refactoring changes.", **c),
            _agent(
                "tester",
                "You are a QA engineer. Verify that refactored code passes "
                "all existing tests and coverage has not decreased. Output a "
                "test report with before/after coverage comparison.",
                "Verifies no regressions after refactoring.",
                **c,
            ),
            reviewer_agent(**c),
        ],
    )


# ---------------------------------------------------------------------------
# 7. Product Development — Designer → Planner → Developer → Reviewer
# ---------------------------------------------------------------------------

def product_development(
    *,
    name: str = "product_development",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: design product → plan implementation → develop → review.

    Stages:
        1. **Product Designer** generates a PRD with acceptance criteria.
        2. **Planner** creates an implementation plan with ordered tasks.
        3. **Developer** implements the plan with tests.
        4. **Reviewer** validates quality and PRD conformity.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Design, plan, develop, and review a product feature.",
        sub_agents=[
            _agent(
                "product_designer",
                "You are a product designer. Define the WHAT and WHY: write a "
                "Product Requirements Document (PRD) with problem statement, "
                "user stories, acceptance criteria, non-functional requirements, "
                "and success metrics.",
                "Generates PRD with acceptance criteria.",
                **c,
            ),
            planner_agent(**c),
            coder_agent(name="developer", description="Implements the plan with tests.", **c),
            reviewer_agent(**c),
        ],
    )


# ---------------------------------------------------------------------------
# 8. Code Review Automation — Diff Analyzer → [Style ‖ Logic ‖ Security] → Summary
# ---------------------------------------------------------------------------

def code_review_automation(
    *,
    name: str = "code_review_automation",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: analyse diff → parallel review (style, logic, security) → summary.

    Stages:
        1. **Diff Analyzer** parses the PR diff and identifies scope/risk.
        2. **Parallel** fan-out: Style Checker, Logic Reviewer, Security Scanner.
        3. **Summary Writer** consolidates all reports into a unified review.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` with an embedded :class:`ParallelAgent`.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Automated code review with parallel style/logic/security checks.",
        sub_agents=[
            _agent(
                "diff_analyzer",
                "You are a diff analysis specialist. Parse the PR diff, "
                "identify changed files, scope of change, and risk level per "
                "file. Output a structured diff context summary.",
                "Parses PR diff and identifies scope and risk.",
                **c,
            ),
            ParallelAgent(
                name="parallel_reviewers",
                description="Style, logic, and security reviews in parallel.",
                sub_agents=[
                    _agent(
                        "style_checker",
                        "You are a code style reviewer. Check adherence to "
                        "coding standards, naming conventions, formatting, and "
                        "consistency. Output a style report with findings.",
                        "Checks code style and conventions.",
                        **c,
                    ),
                    _agent(
                        "logic_reviewer",
                        "You are a logic reviewer. Analyse business logic "
                        "correctness, edge cases, error handling, and algorithm "
                        "efficiency. Output a logic report with findings.",
                        "Reviews business logic and edge cases.",
                        **c,
                    ),
                    _agent(
                        "security_scanner",
                        "You are a security scanner. Detect vulnerabilities in "
                        "changed code: injection, XSS, auth issues, secrets, "
                        "insecure dependencies. Output a security report.",
                        "Detects security vulnerabilities in code changes.",
                        **c,
                    ),
                ],
            ),
            _agent(
                "summary_writer",
                "You are a review summariser. Consolidate the style, logic, "
                "and security reports into a unified PR review with an overall "
                "verdict (approve/request changes/reject), inline comment "
                "suggestions, and prioritised action items.",
                "Consolidates review reports into unified PR review.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 9. Performance Optimization — Profiler → Analyzer → Optimizer → Benchmarker → Reviewer
# ---------------------------------------------------------------------------

def performance_optimization(
    *,
    name: str = "performance_optimization",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: profile → analyse bottlenecks → optimise → benchmark → review.

    Stages:
        1. **Profiler** collects CPU, memory, I/O, and latency data.
        2. **Bottleneck Analyzer** identifies and prioritises hot paths.
        3. **Optimizer** implements targeted optimisations.
        4. **Benchmarker** measures before/after improvement.
        5. **Reviewer** reviews correctness and maintainability trade-offs.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Profile, analyse bottlenecks, optimise, and benchmark.",
        sub_agents=[
            _agent(
                "profiler",
                "You are a performance profiling expert. Collect CPU, memory, "
                "I/O, and latency data. Identify hot functions, flame graph "
                "hotspots, and resource-intensive operations. Output a "
                "structured profile report.",
                "Collects performance profiling data.",
                **c,
            ),
            _agent(
                "bottleneck_analyzer",
                "You are a bottleneck analysis specialist. Given profiling "
                "data, identify and prioritise performance bottlenecks with "
                "evidence (hot paths, root cause, impact). Output a ranked "
                "bottleneck report.",
                "Identifies and prioritises bottlenecks.",
                **c,
            ),
            coder_agent(name="optimizer", description="Implements targeted optimisations.", **c),
            _agent(
                "benchmarker",
                "You are a benchmarking specialist. Measure before/after "
                "performance with p50, p95, and p99 latency stats, throughput, "
                "and resource usage. Output a benchmark comparison report.",
                "Measures before/after performance improvement.",
                **c,
            ),
            reviewer_agent(**c),
        ],
    )


# ---------------------------------------------------------------------------
# 10. Test Suite Generation — Analyzer → Planner → Writer → Coverage → Mutation
# ---------------------------------------------------------------------------

def test_suite_generation(
    *,
    name: str = "test_suite_generation",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: analyse code → plan tests → write tests → check coverage → mutation test.

    Stages:
        1. **Code Analyzer** inventories testable functions and edge cases.
        2. **Test Planner** designs testing strategy with cases per function.
        3. **Test Writer** implements unit/integration tests.
        4. **Coverage Checker** measures line/branch coverage and gaps.
        5. **Mutation Tester** verifies tests detect real bugs.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Generate a complete test suite with coverage and mutation testing.",
        sub_agents=[
            _agent(
                "code_analyzer",
                "You are a code analysis specialist. Inventory all testable "
                "functions, branches, edge cases, and error paths. Identify "
                "complex logic that needs thorough testing. Output a code map.",
                "Analyses code for testable functions and edge cases.",
                **c,
            ),
            _agent(
                "test_planner",
                "You are a test planning specialist. Given a code map, design "
                "a testing strategy: test cases per function, priority, "
                "boundary values, error scenarios, and mocking strategy. "
                "Output a structured test plan.",
                "Designs testing strategy with prioritised cases.",
                **c,
            ),
            coder_agent(name="test_writer", description="Implements unit and integration tests.", **c),
            _agent(
                "coverage_checker",
                "You are a test coverage analyst. Measure line, branch, and "
                "function coverage. Identify untested code paths and gaps. "
                "Output a coverage report with improvement suggestions.",
                "Measures coverage and identifies gaps.",
                **c,
            ),
            _agent(
                "mutation_tester",
                "You are a mutation testing specialist. Apply mutations to "
                "source code and verify tests detect them. Report killed vs "
                "survived mutants and suggest stronger assertions. Output a "
                "mutation report.",
                "Verifies test quality via mutation testing.",
                **c,
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 11. System Design — Requirements Analyst → Architect → Reviewer → Decision Logger
# ---------------------------------------------------------------------------

def system_design(
    *,
    name: str = "system_design",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: gather requirements → design architecture → review → log decisions.

    Stages:
        1. **Requirements Analyst** extracts functional/non-functional requirements.
        2. **Architect** designs system architecture with diagrams.
        3. **Reviewer** evaluates feasibility, scalability, and cost.
        4. **Decision Logger** documents ADRs (Architecture Decision Records).

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Design system architecture with traceable decisions.",
        sub_agents=[
            _agent(
                "requirements_analyst",
                "You are a requirements analyst. Extract and formalise "
                "functional requirements, non-functional requirements (NFRs), "
                "SLAs, and constraints from the input. Output a structured "
                "requirements document.",
                "Extracts and formalises system requirements.",
                **c,
            ),
            _agent(
                "architect",
                "You are a system architect. Design the system architecture: "
                "components, interactions, data flow, technology stack. "
                "Generate C4 diagrams (context, container, component) using "
                "Mermaid or PlantUML. Output an architecture document.",
                "Designs system architecture with diagrams.",
                **c,
            ),
            reviewer_agent(**c),
            _agent(
                "decision_logger",
                "You are an ADR writer. Document each architectural decision "
                "as an Architecture Decision Record: context, options "
                "considered, trade-offs, decision, and status. Output ADRs "
                "in standard format.",
                "Documents architectural decisions as ADRs.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 12. Database Design — Domain Modeler → Schema Designer → Migrator → Validator
# ---------------------------------------------------------------------------

def database_design(
    *,
    name: str = "database_design",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: model domain → design schema → generate migrations → validate.

    Stages:
        1. **Domain Modeler** models entities, relationships, and business rules.
        2. **Schema Designer** translates to physical schema with indexes.
        3. **Migrator** generates versioned, reversible migrations.
        4. **Validator** validates query performance and integrity.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Design database schema from domain model to validated migrations.",
        sub_agents=[
            _agent(
                "domain_modeler",
                "You are a domain modelling expert. Analyse the domain and "
                "model entities, relationships, cardinality, and business "
                "rules. Generate an ERD using Mermaid. Output a domain model.",
                "Models domain entities and relationships.",
                **c,
            ),
            _agent(
                "schema_designer",
                "You are a database schema designer. Translate the domain "
                "model to a physical schema: tables, columns, types, indexes, "
                "constraints, and partitioning strategy. Output DDL statements.",
                "Designs physical database schema.",
                **c,
            ),
            _agent(
                "migrator",
                "You are a database migration specialist. Generate versioned, "
                "reversible migration scripts (up/down) plus seed data. "
                "Ensure zero-downtime compatibility. Output migration files.",
                "Generates versioned database migrations.",
                **c,
            ),
            _agent(
                "validator",
                "You are a database validation specialist. Validate query "
                "performance via EXPLAIN ANALYSE, check referential integrity, "
                "and identify missing indexes. Output a validation report.",
                "Validates schema performance and integrity.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 13. API Design — Domain Expert → Designer → Implementer → Doc Gen → Consumer Tester
# ---------------------------------------------------------------------------

def api_design(
    *,
    name: str = "api_design",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: define domain → design API → implement → generate docs → test DX.

    Stages:
        1. **Domain Expert** defines entities, actions, and vocabulary.
        2. **API Designer** designs endpoints, schemas, auth, versioning.
        3. **Implementer** implements the API with contract tests.
        4. **Doc Generator** generates interactive API documentation.
        5. **Consumer Tester** validates from the client/DX perspective.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Design, implement, document, and validate an API.",
        sub_agents=[
            _agent(
                "domain_expert",
                "You are a domain expert. Define the domain model: entities, "
                "actions, invariants, and ubiquitous language. Output a "
                "structured domain model document.",
                "Defines domain model and vocabulary.",
                **c,
            ),
            _agent(
                "api_designer",
                "You are an API designer. Design the API: endpoints, request/"
                "response schemas, authentication, rate limiting, versioning, "
                "and error codes. Output an OpenAPI specification.",
                "Designs API endpoints and schemas.",
                output_format="json",
                **c,
            ),
            coder_agent(name="implementer", description="Implements the API with contract tests.", **c),
            writer_agent(name="doc_generator", description="Generates interactive API documentation.", **c),
            _agent(
                "consumer_tester",
                "You are a developer experience (DX) tester. Validate the API "
                "from a client perspective: onboarding friction, error clarity, "
                "documentation quality, and SDK usability. Output a DX report "
                "with pain points and suggestions.",
                "Validates API from the developer experience perspective.",
                **c,
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPERATIONS PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 14. Incident Response — Detector → Diagnostician → Responder → RCA → Postmortem
# ---------------------------------------------------------------------------

def incident_response(
    *,
    name: str = "incident_response",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: detect → diagnose → respond → root cause → postmortem.

    Stages:
        1. **Detector** identifies anomalies and generates an alert.
        2. **Diagnostician** narrows down the affected service and blast radius.
        3. **Responder** mitigates the incident (rollback, restart, etc.).
        4. **Root Cause Analyst** investigates the underlying cause.
        5. **Postmortem Writer** documents the full incident.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Detect, diagnose, respond, analyse, and document incidents.",
        sub_agents=[
            _agent(
                "detector",
                "You are an incident detection specialist. Identify anomalies "
                "from monitoring data, generate an initial alert with severity, "
                "impact scope, and affected services. Output a structured alert.",
                "Detects anomalies and generates alerts.",
                **c,
            ),
            _agent(
                "diagnostician",
                "You are an incident diagnostician. Narrow down the problem: "
                "which service, what is the blast radius, what changed "
                "recently, and what is the timeline. Output a diagnosis.",
                "Diagnoses incident scope and root service.",
                **c,
            ),
            _agent(
                "responder",
                "You are an incident responder. Mitigate the incident using "
                "rollback, restart, feature flags, scaling, or traffic "
                "rerouting. Prioritise restoring service over root cause. "
                "Output a mitigation record.",
                "Mitigates incidents and restores service.",
                **c,
            ),
            _agent(
                "rca_analyst",
                "You are a root cause analyst. After service is restored, "
                "investigate the root cause: contributing factors, timeline, "
                "and systemic issues. Output a root cause document.",
                "Investigates root cause after mitigation.",
                **c,
            ),
            _agent(
                "postmortem_writer",
                "You are a postmortem writer. Document the full incident with "
                "timeline, impact, root cause, what went well, what went "
                "wrong, and action items with owners. Use a blameless format.",
                "Documents blameless incident postmortem.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 15. DevOps Deployment — Build → Security Scan → Deploy → Monitor
# ---------------------------------------------------------------------------

def devops_deployment(
    *,
    name: str = "devops_deployment",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: build → security scan → deploy → monitor.

    Stages:
        1. **Build Agent** compiles, packages, and generates artefacts.
        2. **Security Scanner** analyses vulnerabilities in code and deps.
        3. **Deployer** deploys to target environment with health checks.
        4. **Monitor** observes production post-deploy for anomalies.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Build, scan, deploy, and monitor a release.",
        sub_agents=[
            _agent(
                "build_agent",
                "You are a build engineer. Compile, lint, test, package, and "
                "generate deployable artefacts (container images, binaries). "
                "Output build status with artefact details.",
                "Compiles and packages deployable artefacts.",
                **c,
            ),
            _agent(
                "security_scanner",
                "You are a security scanning specialist. Analyse code and "
                "dependencies for CVEs, OWASP issues, and licence risks. "
                "Output a security report with severity and remediation.",
                "Scans for security vulnerabilities.",
                **c,
            ),
            _agent(
                "deployer",
                "You are a deployment specialist. Deploy the artefact to the "
                "target environment, run health checks, and prepare rollback "
                "procedures. Output a deployment confirmation.",
                "Deploys artefacts with health checks.",
                **c,
            ),
            _agent(
                "monitor",
                "You are a production monitoring specialist. Observe the "
                "deployed service for anomalies: error rates, latency spikes, "
                "resource usage. Alert on degradation and recommend rollback "
                "if needed. Output a health report.",
                "Monitors production health post-deploy.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 16. Cost Optimization — Scanner → Analyzer → Optimizer → Validator
# ---------------------------------------------------------------------------

def cost_optimization(
    *,
    name: str = "cost_optimization",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: scan resources → analyse usage → optimise → validate savings.

    Stages:
        1. **Resource Scanner** inventories all cloud resources and costs.
        2. **Usage Analyzer** analyses actual vs provisioned utilisation.
        3. **Optimizer** proposes rightsizing, reservations, cleanup.
        4. **Validator** verifies no performance degradation and reports savings.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Scan, analyse, optimise, and validate cloud cost savings.",
        sub_agents=[
            _agent(
                "resource_scanner",
                "You are a cloud resource auditor. Inventory all cloud "
                "resources with type, cost, tags, and owner. Identify idle "
                "and untagged resources. Output a resource map.",
                "Inventories cloud resources and costs.",
                **c,
            ),
            _agent(
                "usage_analyzer",
                "You are a utilisation analyst. Compare actual usage against "
                "provisioned capacity over 30+ days. Identify overprovisioned, "
                "idle, and waste. Output a usage report.",
                "Analyses actual vs provisioned utilisation.",
                **c,
            ),
            _agent(
                "cost_optimizer",
                "You are a cost optimisation specialist. Propose changes: "
                "rightsizing, reserved instances, spot usage, unused resource "
                "cleanup. Estimate savings and risk per change. Output an "
                "optimisation plan.",
                "Proposes cost optimisation changes.",
                **c,
            ),
            _agent(
                "savings_validator",
                "You are a savings validation specialist. Verify that "
                "optimisations caused no performance degradation. Generate a "
                "savings report with before/after costs, percentage reduction, "
                "and ROI.",
                "Validates savings without performance impact.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 17. Observability Setup — Signals → Implement → Dashboards → Alerts
# ---------------------------------------------------------------------------

def observability_setup(
    *,
    name: str = "observability_setup",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: identify signals → instrument → build dashboards → tune alerts.

    Stages:
        1. **Signal Identifier** identifies critical logs, metrics, traces.
        2. **Implementer** instruments code and configures collectors.
        3. **Dashboard Builder** creates operational dashboards.
        4. **Alert Tuner** configures and tunes alerts to minimise noise.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Set up complete observability: signals, dashboards, alerts.",
        sub_agents=[
            _agent(
                "signal_identifier",
                "You are an observability architect. Identify critical signals "
                "(logs, metrics, traces) per service using USE/RED methods. "
                "Output a signal catalogue with type and criticality.",
                "Identifies critical observability signals.",
                **c,
            ),
            coder_agent(
                name="instrumenter",
                description="Instruments code and configures collectors (OpenTelemetry).",
                **c,
            ),
            _agent(
                "dashboard_builder",
                "You are a dashboard designer. Create operational and SLO "
                "dashboards with key metrics, graphs, and drill-down views. "
                "Output dashboard definitions (Grafana/Datadog JSON).",
                "Creates operational dashboards.",
                output_format="json",
                **c,
            ),
            _agent(
                "alert_tuner",
                "You are an alert tuning specialist. Configure alerts with "
                "proper thresholds, escalation policies, and runbook links. "
                "Target <5%% false positive rate. Output alert configurations.",
                "Configures and tunes alerts.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 18. Disaster Recovery — Risk Assessor → Runbook → Simulator → Validator → Certifier
# ---------------------------------------------------------------------------

def disaster_recovery(
    *,
    name: str = "disaster_recovery",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: assess risks → write runbooks → simulate → validate → certify.

    Stages:
        1. **Risk Assessor** identifies risks and defines RTO/RPO per service.
        2. **Runbook Writer** writes step-by-step recovery runbooks.
        3. **Simulator** runs disaster drills via chaos engineering.
        4. **Validator** validates recovery times meet targets.
        5. **Certifier** generates formal DR certification.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Assess risks, write runbooks, drill, validate, and certify DR plan.",
        sub_agents=[
            _agent(
                "risk_assessor",
                "You are a disaster recovery risk assessor. Identify risks, "
                "define RTO/RPO per service, classify single points of failure "
                "(SPOF), and assess business impact. Output a risk assessment.",
                "Assesses DR risks and defines RTO/RPO.",
                **c,
            ),
            _agent(
                "runbook_writer",
                "You are a runbook author. Write step-by-step recovery "
                "runbooks for each disaster scenario. Include automated and "
                "manual steps, decision points, and escalation. Output runbooks.",
                "Writes step-by-step DR runbooks.",
                **c,
            ),
            _agent(
                "simulator",
                "You are a chaos engineering specialist. Design and run "
                "disaster drills: inject failures, measure actual recovery "
                "times, and identify gaps. Output drill results.",
                "Runs disaster drills and measures recovery.",
                **c,
            ),
            _agent(
                "dr_validator",
                "You are a DR validation specialist. Compare actual recovery "
                "times against RTO/RPO targets. Identify gaps and required "
                "improvements. Output a validation report.",
                "Validates recovery times against targets.",
                **c,
            ),
            _agent(
                "certifier",
                "You are a DR certification specialist. Generate a formal "
                "disaster recovery certification with compliance status, "
                "validity period, and next drill schedule.",
                "Generates formal DR certification.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 19. Migration — Legacy Analyzer → Target Designer → Migrator → Validator → Deployer
# ---------------------------------------------------------------------------

def migration(
    *,
    name: str = "migration",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: analyse legacy → design target → migrate → validate → deploy.

    Stages:
        1. **Legacy Analyzer** maps the legacy system components and tech debt.
        2. **Target Designer** defines the target architecture and strategy.
        3. **Migrator** executes the migration component-by-component.
        4. **Validator** verifies 100%% functional parity.
        5. **Deployer** deploys with cutover strategy.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Analyse, design, migrate, validate, and deploy system migration.",
        sub_agents=[
            _agent(
                "legacy_analyzer",
                "You are a legacy system analyst. Map the existing system: "
                "components, APIs, data stores, integrations, and tech debt. "
                "Output a legacy system inventory.",
                "Maps legacy system components and tech debt.",
                **c,
            ),
            _agent(
                "target_designer",
                "You are a migration architect. Design the target architecture "
                "and choose the migration strategy (big bang, strangler fig, "
                "incremental). Output a target architecture document.",
                "Designs target architecture and migration strategy.",
                **c,
            ),
            coder_agent(name="migrator_impl", description="Executes migration code and scripts.", **c),
            _agent(
                "parity_validator",
                "You are a migration validation specialist. Verify 100%% "
                "functional parity between legacy and migrated systems: API "
                "compatibility, data integrity, and behaviour. Output a "
                "validation report.",
                "Verifies functional parity after migration.",
                **c,
            ),
            _agent(
                "migration_deployer",
                "You are a migration deployment specialist. Deploy the "
                "migrated system with a cutover strategy (blue-green, canary), "
                "health checks, and rollback plan. Output a deployment "
                "confirmation.",
                "Deploys migrated system with cutover strategy.",
                **c,
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 20. Data Quality — Profiler → Rule Designer → Validator → Cleaner → Reporter
# ---------------------------------------------------------------------------

def data_quality(
    *,
    name: str = "data_quality",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: profile data → design rules → validate → clean → report.

    Stages:
        1. **Profiler** analyses data distribution, types, and anomalies.
        2. **Rule Designer** defines quality rules (nullability, ranges, formats).
        3. **Validator** executes rules and reports violations.
        4. **Cleaner** fixes or quarantines violating data.
        5. **Reporter** generates quality reports with trends.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Profile, validate, clean, and report on data quality.",
        sub_agents=[
            _agent(
                "data_profiler",
                "You are a data profiling specialist. Analyse data "
                "distributions, types, patterns, null rates, and anomalies "
                "per column. Output a data profile report.",
                "Analyses data distributions and anomalies.",
                **c,
            ),
            _agent(
                "rule_designer",
                "You are a data quality rule designer. Define quality rules: "
                "nullability, valid ranges, format patterns, referential "
                "integrity, and uniqueness constraints. Output rules in a "
                "structured format.",
                "Designs data quality validation rules.",
                output_format="json",
                **c,
            ),
            _agent(
                "data_validator",
                "You are a data validation specialist. Execute quality rules "
                "against the data and report violations with row references "
                "and severity. Output a violations report.",
                "Executes quality rules and reports violations.",
                **c,
            ),
            _agent(
                "data_cleaner",
                "You are a data cleaning specialist. Fix or quarantine "
                "violating data — never delete without explicit approval. "
                "Log all transformations for audit. Output cleaned data "
                "summary and quarantine list.",
                "Fixes or quarantines violating data.",
                **c,
            ),
            _agent(
                "quality_reporter",
                "You are a data quality reporter. Generate a quality report "
                "with overall score, trends, root causes of issues, and SLA "
                "compliance. Include visualisation suggestions.",
                "Generates data quality reports with trends.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 21. ETL Pipeline Design — Source Analyzer → Transform → Implement → Validate → Schedule
# ---------------------------------------------------------------------------

def etl_pipeline_design(
    *,
    name: str = "etl_pipeline_design",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: analyse sources → design transforms → implement → validate → schedule.

    Stages:
        1. **Source Analyzer** analyses data sources, schemas, and SLAs.
        2. **Transform Designer** designs the transformation DAG and logic.
        3. **Implementer** implements the ETL/ELT pipeline.
        4. **Data Validator** validates transformed data integrity.
        5. **Scheduler** configures scheduling, monitoring, and retries.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Design, implement, validate, and schedule an ETL pipeline.",
        sub_agents=[
            _agent(
                "source_analyzer",
                "You are a data source analyst. Analyse data sources: schemas, "
                "volume, frequency, quality, access patterns, and SLAs. "
                "Output a source map.",
                "Analyses data source schemas and characteristics.",
                **c,
            ),
            _agent(
                "transform_designer",
                "You are a data transformation designer. Design the "
                "transformation DAG: business logic, joins, aggregations, "
                "SCD handling, and data lineage. Output a transform spec.",
                "Designs ETL transformation logic and DAG.",
                **c,
            ),
            coder_agent(name="etl_implementer", description="Implements the ETL/ELT pipeline.", **c),
            _agent(
                "etl_validator",
                "You are a data validation specialist. Validate transformed "
                "data: row counts, checksums, referential integrity, and "
                "business rule compliance. Output a validation report.",
                "Validates transformed data integrity.",
                **c,
            ),
            _agent(
                "scheduler",
                "You are a pipeline scheduling specialist. Configure "
                "scheduling (cron, event-driven), monitoring, retry policies, "
                "dead-letter queues, and alerting. Output a schedule "
                "configuration.",
                "Configures ETL scheduling and monitoring.",
                **c,
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# AI / ML PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 22. Prompt Engineering — Task Analyzer → Drafter → Variations → Evaluator → Optimizer
# ---------------------------------------------------------------------------

def prompt_engineering(
    *,
    name: str = "prompt_engineering",
    model: str | None = None,
    provider: str = "google",
    max_iterations: int = 3,
    **kwargs: object,
) -> LoopAgent:
    """Pipeline: analyse task → draft prompt → generate variants → evaluate → optimise.

    Uses a :class:`LoopAgent` to iterate until the evaluation score
    meets a threshold or *max_iterations* is reached.

    Stages (per iteration):
        1. **Task Analyzer** defines objectives, format, and constraints.
        2. **Prompt Drafter** creates a prompt using best practices.
        3. **Variation Generator** produces 3-5 prompt variants.
        4. **Evaluator** scores variants against test cases.
        5. **Optimizer** refines the best prompt from failure analysis.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        max_iterations: Maximum optimisation cycles.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`LoopAgent` iterating until quality threshold.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return LoopAgent(
        name=name,
        description="Iteratively design and optimise LLM prompts.",
        sub_agents=[
            _agent(
                "task_analyzer",
                "You are a prompt engineering analyst. Define what the prompt "
                "should achieve: objective, expected output format, "
                "constraints, and evaluation criteria. Output a task spec.",
                "Defines prompt objectives and constraints.",
                **c,
            ),
            _agent(
                "prompt_drafter",
                "You are a prompt engineer. Create a prompt using best "
                "practices: chain-of-thought, few-shot examples, role "
                "assignment, and structured output. Output the draft prompt.",
                "Creates prompts using best practices.",
                **c,
            ),
            _agent(
                "variation_generator",
                "You are a prompt variation specialist. Generate 3-5 "
                "materially different prompt variants exploring different "
                "techniques (CoT, ReAct, few-shot, zero-shot). Output all "
                "variants.",
                "Generates diverse prompt variants.",
                **c,
            ),
            _agent(
                "prompt_evaluator",
                "You are a prompt evaluation specialist. Score each variant "
                "against test cases using objective metrics: accuracy, "
                "relevance, format compliance, and consistency. Output an "
                "evaluation report with scores and failure analysis.",
                "Evaluates prompt variants with metrics.",
                output_format="json",
                **c,
            ),
            _agent(
                "prompt_optimizer",
                "You are a prompt optimiser. Refine the best-scoring prompt "
                "based on failure analysis. Fix edge cases and improve "
                "robustness. Output the optimised prompt and test results.",
                "Refines prompts from failure analysis.",
                **c,
            ),
        ],
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# 23. RAG Pipeline Design — Corpus → Chunking → Embedding → Retriever → E2E Eval
# ---------------------------------------------------------------------------

def rag_pipeline_design(
    *,
    name: str = "rag_pipeline_design",
    model: str | None = None,
    provider: str = "google",
    max_iterations: int = 3,
    **kwargs: object,
) -> LoopAgent:
    """Pipeline: analyse corpus → chunk → select embeddings → tune retriever → evaluate.

    Uses a :class:`LoopAgent` to iterate until retrieval quality
    meets a threshold or *max_iterations* is reached.

    Stages (per iteration):
        1. **Corpus Analyzer** analyses document types, structure, and gaps.
        2. **Chunking Strategist** designs the optimal chunking strategy.
        3. **Embedding Selector** selects the best embedding model.
        4. **Retriever Tuner** configures top-k, reranking, hybrid search.
        5. **E2E Evaluator** evaluates faithfulness, relevance, hallucination.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        max_iterations: Maximum optimisation cycles.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`LoopAgent` iterating until quality threshold.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return LoopAgent(
        name=name,
        description="Design and optimise a RAG pipeline iteratively.",
        sub_agents=[
            _agent(
                "corpus_analyzer",
                "You are a corpus analysis specialist. Analyse the document "
                "corpus: types, structure, average length, coverage, and "
                "gaps. Output a corpus profile.",
                "Analyses document corpus characteristics.",
                **c,
            ),
            _agent(
                "chunking_strategist",
                "You are a chunking strategy specialist. Design the optimal "
                "chunking approach (semantic, recursive, section-based) with "
                "overlap and metadata configuration. Output a chunking "
                "strategy.",
                "Designs optimal document chunking strategy.",
                **c,
            ),
            _agent(
                "embedding_selector",
                "You are an embedding model specialist. Select the optimal "
                "embedding model based on MTEB benchmarks, domain evaluation, "
                "dimensionality, and cost. Output an embedding selection with "
                "justification.",
                "Selects optimal embedding model.",
                **c,
            ),
            _agent(
                "retriever_tuner",
                "You are a retriever tuning specialist. Configure the "
                "retriever: top-k, reranking model, hybrid search weights, "
                "metadata filters, and query expansion. Output a retriever "
                "configuration.",
                "Tunes retriever parameters and search strategy.",
                output_format="json",
                **c,
            ),
            _agent(
                "e2e_evaluator",
                "You are a RAG evaluation specialist. Evaluate end-to-end "
                "quality: faithfulness, answer relevance, context precision, "
                "hallucination rate (RAGAS metrics). Output an evaluation "
                "report with pass/fail and improvement suggestions.",
                "Evaluates RAG pipeline end-to-end quality.",
                output_format="json",
                **c,
            ),
        ],
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# 24. Model Fine-Tuning — Curator → Preprocessor → Trainer → Evaluator → Publisher
# ---------------------------------------------------------------------------

def model_fine_tuning(
    *,
    name: str = "model_fine_tuning",
    model: str | None = None,
    provider: str = "google",
    max_iterations: int = 3,
    **kwargs: object,
) -> LoopAgent:
    """Pipeline: curate data → preprocess → train → evaluate → publish.

    Uses a :class:`LoopAgent` to iterate until evaluation metrics
    exceed the baseline or *max_iterations* is reached.

    Stages (per iteration):
        1. **Data Curator** selects, cleans, and balances training data.
        2. **Preprocessor** transforms data to model format.
        3. **Trainer** runs training with hyperparameters.
        4. **Evaluator** evaluates against test set and baselines.
        5. **Registry Publisher** publishes to model registry.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        max_iterations: Maximum training cycles.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`LoopAgent` iterating until metrics meet baseline.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return LoopAgent(
        name=name,
        description="Curate, train, evaluate, and publish fine-tuned models.",
        sub_agents=[
            _agent(
                "data_curator",
                "You are a training data curator. Select, clean, validate, "
                "deduplicate, remove PII, and balance the training dataset. "
                "Output curated dataset stats and train/val/test splits.",
                "Curates and cleans training data.",
                **c,
            ),
            _agent(
                "preprocessor",
                "You are a data preprocessing specialist. Transform data to "
                "model-required format: tokenise, pad, augment, and generate "
                "instruction-response pairs. Output preprocessing config.",
                "Transforms data to model training format.",
                **c,
            ),
            _agent(
                "trainer",
                "You are a model training specialist. Configure and run "
                "training: learning rate, batch size, epochs, LoRA/QLoRA "
                "settings, checkpointing. Output training logs and config.",
                "Runs model training with hyperparameters.",
                **c,
            ),
            _agent(
                "model_evaluator",
                "You are a model evaluation specialist. Evaluate the model "
                "against test set and baselines: accuracy, F1, perplexity, "
                "human-eval. Compare with previous versions. Output an "
                "evaluation report.",
                "Evaluates model against baselines.",
                output_format="json",
                **c,
            ),
            _agent(
                "registry_publisher",
                "You are a model registry specialist. Publish the approved "
                "model to the registry with version, metrics, model card, "
                "and deployment config. Output the registration record.",
                "Publishes model to registry with metadata.",
                output_format="json",
                **c,
            ),
        ],
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# 25. AI Safety & Guardrails — Risk Cataloger → Red Teamer → Designer → Tester → Certifier
# ---------------------------------------------------------------------------

def ai_safety_guardrails(
    *,
    name: str = "ai_safety_guardrails",
    model: str | None = None,
    provider: str = "google",
    max_iterations: int = 3,
    **kwargs: object,
) -> LoopAgent:
    """Pipeline: catalogue risks → red team → design guardrails → test → certify.

    Uses a :class:`LoopAgent` to iterate when guardrail bypasses
    are found, up to *max_iterations*.

    Stages (per iteration):
        1. **Risk Cataloger** catalogues AI risks by category.
        2. **Red Teamer** attempts adversarial attacks.
        3. **Guardrail Designer** designs input/output filters and policies.
        4. **Tester** validates guardrails (catch rate >95%%, FP <2%%).
        5. **Compliance Certifier** certifies regulatory compliance.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        max_iterations: Maximum guardrail design cycles.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`LoopAgent` iterating until guardrails pass.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return LoopAgent(
        name=name,
        description="Red-team, design, test, and certify AI safety guardrails.",
        sub_agents=[
            _agent(
                "risk_cataloger",
                "You are an AI risk analyst. Catalogue risks by category: "
                "bias, toxicity, hallucination, privacy, misuse, robustness. "
                "Rate severity and probability. Output a risk catalogue.",
                "Catalogues AI risks by category and severity.",
                **c,
            ),
            _agent(
                "red_teamer",
                "You are an AI red teamer. Attempt adversarial attacks: "
                "prompt injection, jailbreak, PII extraction, bias "
                "exploitation. Document successful attacks with severity. "
                "Output a red team report.",
                "Attempts adversarial attacks on AI systems.",
                **c,
            ),
            _agent(
                "guardrail_designer",
                "You are a guardrail designer. Design input/output filters, "
                "content policies, rate limiters, and PII detectors per "
                "identified risk. Output guardrail specifications.",
                "Designs AI guardrails and safety filters.",
                **c,
            ),
            _agent(
                "guardrail_tester",
                "You are a guardrail testing specialist. Validate guardrails: "
                "catch rate >95%%, false positive rate <2%%, latency impact. "
                "Test with adversarial and benign inputs. Output a test report.",
                "Tests guardrail effectiveness and false positives.",
                output_format="json",
                **c,
            ),
            _agent(
                "compliance_certifier",
                "You are an AI compliance specialist. Certify compliance with "
                "EU AI Act, NIST AI RMF, and organisational policies. Output "
                "a compliance certificate with audit trail.",
                "Certifies AI regulatory compliance.",
                **c,
            ),
        ],
        max_iterations=max_iterations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTENT & KNOWLEDGE PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 26. Content Documentation — Researcher → Writer → Tech Reviewer → Publisher
# ---------------------------------------------------------------------------

def content_documentation(
    *,
    name: str = "content_documentation",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: research → write → technical review → publish.

    Stages:
        1. **Researcher** gathers information and technical context.
        2. **Writer** composes the technical document.
        3. **Technical Reviewer** validates technical accuracy.
        4. **Publisher** edits for clarity and publishes.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Research, write, review, and publish technical documentation.",
        sub_agents=[
            _agent(
                "researcher",
                "You are a technical researcher. Gather information, code "
                "examples, and context from source code, APIs, and existing "
                "docs. Output structured research notes with sources.",
                "Gathers technical information and context.",
                **c,
            ),
            writer_agent(**c),
            reviewer_agent(
                name="tech_reviewer",
                description="Validates technical accuracy of documentation.",
                **c,
            ),
            _agent(
                "publisher",
                "You are a technical editor and publisher. Edit the document "
                "for clarity, grammar, and style consistency. Prepare for "
                "publication with metadata and formatting. Output the final "
                "published document.",
                "Edits and publishes technical documentation.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 27. Research — Question Formulator → Source Finder → Analyzer → Report Writer
# ---------------------------------------------------------------------------

def research(
    *,
    name: str = "research",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: formulate questions → find sources → analyse → write report.

    Stages:
        1. **Question Formulator** transforms topic into scoped research questions.
        2. **Source Finder** searches and selects reliable sources.
        3. **Analyzer** extracts findings per source.
        4. **Report Writer** synthesises findings into a report.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the four stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Structured research from questions to synthesised report.",
        sub_agents=[
            _agent(
                "question_formulator",
                "You are a research question specialist. Transform the topic "
                "into concrete, scoped research questions with main and sub "
                "questions, scope boundaries, and success criteria. Output a "
                "research question framework.",
                "Formulates structured research questions.",
                **c,
            ),
            _agent(
                "source_finder",
                "You are a research source specialist. Search and select "
                "reliable, diverse sources: academic papers, technical docs, "
                "repositories, and industry reports. Output a curated source "
                "catalogue with relevance ratings.",
                "Finds and curates reliable research sources.",
                **c,
            ),
            _agent(
                "research_analyzer",
                "You are a research analyst. Extract findings from each source "
                "relevant to the research questions. Note key data, citations, "
                "and contradictions. Output structured analysis notes.",
                "Extracts and analyses findings from sources.",
                **c,
            ),
            _agent(
                "report_writer",
                "You are a research report writer. Synthesise all findings "
                "into conclusions and recommendations. Include an executive "
                "summary, methodology, findings, and references. Output a "
                "complete research report.",
                "Synthesises research into conclusions and recommendations.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 28. Security Audit — Threat Modeler → Static Analyzer → Pen Tester → Remediator → Verifier
# ---------------------------------------------------------------------------

def security_audit(
    *,
    name: str = "security_audit",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: threat model → static analysis → pen test → remediate → verify.

    Stages:
        1. **Threat Modeler** identifies attack surfaces using STRIDE/DREAD.
        2. **Static Analyzer** runs SAST to find code vulnerabilities.
        3. **Pen Tester** actively exploits vulnerabilities with PoC.
        4. **Remediator** implements security fixes.
        5. **Verifier** re-runs tests to confirm fixes.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Threat model, analyse, pen test, remediate, and verify security.",
        sub_agents=[
            _agent(
                "threat_modeler",
                "You are a threat modelling specialist. Identify attack "
                "surfaces and potential threats using STRIDE or DREAD. "
                "Generate attack trees and identify at-risk assets. Output a "
                "threat model.",
                "Identifies attack surfaces and threats.",
                **c,
            ),
            _agent(
                "static_analyzer",
                "You are a static analysis specialist. Run SAST to find code "
                "vulnerabilities: CWE references, severity, affected lines, "
                "and remediation guidance. Output a SAST report.",
                "Finds code vulnerabilities via static analysis.",
                **c,
            ),
            _agent(
                "pen_tester",
                "You are a penetration testing specialist. Actively attempt "
                "to exploit identified vulnerabilities with proof-of-concept. "
                "Document exploitable vulnerabilities with impact. Output a "
                "pen test report.",
                "Exploits vulnerabilities with proof-of-concept.",
                **c,
            ),
            coder_agent(
                name="security_remediator",
                description="Implements security fixes and regression tests.",
                **c,
            ),
            _agent(
                "security_verifier",
                "You are a security verification specialist. Re-run all "
                "security tests to confirm all HIGH/CRITICAL vulnerabilities "
                "are closed. Output a verification report with pass/fail per "
                "vulnerability.",
                "Verifies all security fixes are effective.",
                **c,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 29. Compliance — Evidence Collector → Gap Analyzer → Remediator → Auditor → Report
# ---------------------------------------------------------------------------

def compliance(
    *,
    name: str = "compliance",
    model: str | None = None,
    provider: str = "google",
    **kwargs: object,
) -> SequentialAgent:
    """Pipeline: collect evidence → analyse gaps → remediate → audit → report.

    Stages:
        1. **Evidence Collector** gathers compliance artefacts.
        2. **Gap Analyzer** identifies control gaps against framework.
        3. **Remediator** implements missing controls.
        4. **Auditor** performs an internal audit.
        5. **Report Generator** produces audit-ready documentation.

    Args:
        name: Pipeline name.
        model: LLM model.
        provider: AI provider.
        **kwargs: Extra args forwarded to each agent.

    Returns:
        A :class:`SequentialAgent` wiring the five stages.
    """
    c = dict(model=model, provider=provider, **kwargs)
    return SequentialAgent(
        name=name,
        description="Collect evidence, analyse gaps, remediate, audit, and report compliance.",
        sub_agents=[
            _agent(
                "evidence_collector",
                "You are a compliance evidence collector. Gather all artefacts "
                "required by the compliance framework (SOC 2, ISO 27001, etc.): "
                "policies, configs, logs, access reviews. Output an evidence "
                "inventory.",
                "Gathers compliance evidence artefacts.",
                **c,
            ),
            _agent(
                "gap_analyzer",
                "You are a compliance gap analyst. Compare collected evidence "
                "against the compliance framework controls. Identify gaps, "
                "missing controls, and weak areas. Output a gap analysis.",
                "Identifies compliance control gaps.",
                **c,
            ),
            coder_agent(
                name="compliance_remediator",
                description="Implements missing compliance controls.",
                **c,
            ),
            _agent(
                "auditor",
                "You are an internal auditor. Perform an internal audit: test "
                "control effectiveness, verify evidence, and assess risk. "
                "Output audit findings with recommendations.",
                "Performs internal compliance audit.",
                **c,
            ),
            _agent(
                "compliance_reporter",
                "You are a compliance report writer. Generate audit-ready "
                "documentation: executive summary, control matrix, gaps "
                "addressed, residual risks, and certification readiness. "
                "Output the compliance report.",
                "Generates audit-ready compliance documentation.",
                **c,
            ),
        ],
    )
