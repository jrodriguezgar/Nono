"""
Agent Templates — Pre-configured agents for common tasks.

Individual agents:
    - **planner**: Strategic planning and project breakdown.
    - **decomposer**: Complex task decomposition into subtasks.
    - **summarizer**: Text and document summarization.
    - **reviewer**: Output review and quality critique.
    - **coder**: Code generation and programming.
    - **classifier**: Input classification and routing.
    - **extractor**: Structured data extraction from text.
    - **writer**: Content writing and text generation.
    - **guardrail**: Safety checks and PII redaction.

Pipelines (multi-agent combinations):
    - **plan_and_execute**: Planner → Decomposer → Coder pipeline.
    - **research_and_write**: Extractor → Writer → Reviewer pipeline.
    - **draft_review_loop**: Writer + Reviewer iterative loop.
    - **classify_and_route**: Classifier router to specialized agents.

    Development:
    - **bug_fix**: Triager → Debugger → Fixer → Tester → Reviewer.
    - **refactoring**: Code Analyzer → Planner → Refactorer → Tester → Reviewer.
    - **product_development**: Product Designer → Planner → Developer → Reviewer.
    - **code_review_automation**: Diff Analyzer → [Style ‖ Logic ‖ Security] → Summary.
    - **performance_optimization**: Profiler → Analyzer → Optimizer → Benchmarker → Reviewer.
    - **test_suite_generation**: Code Analyzer → Planner → Writer → Coverage → Mutation.

    Architecture:
    - **system_design**: Requirements → Architect → Reviewer → Decision Logger.
    - **database_design**: Domain Modeler → Schema Designer → Migrator → Validator.
    - **api_design**: Domain Expert → Designer → Implementer → Doc Gen → Consumer Tester.

    Operations:
    - **incident_response**: Detector → Diagnostician → Responder → RCA → Postmortem.
    - **devops_deployment**: Build → Security Scan → Deploy → Monitor.
    - **cost_optimization**: Scanner → Analyzer → Optimizer → Validator.
    - **observability_setup**: Signals → Instrument → Dashboards → Alerts.
    - **disaster_recovery**: Risk Assessor → Runbook → Simulator → Validator → Certifier.
    - **migration**: Legacy Analyzer → Target Designer → Migrator → Validator → Deployer.

    Data:
    - **data_quality**: Profiler → Rule Designer → Validator → Cleaner → Reporter.
    - **etl_pipeline_design**: Source Analyzer → Transform → Implement → Validate → Schedule.

    AI/ML:
    - **prompt_engineering**: Task Analyzer → Drafter → Variations → Evaluator → Optimizer (loop).
    - **rag_pipeline_design**: Corpus → Chunking → Embedding → Retriever → E2E Eval (loop).
    - **model_fine_tuning**: Curator → Preprocessor → Trainer → Evaluator → Publisher (loop).
    - **ai_safety_guardrails**: Risk Cataloger → Red Teamer → Designer → Tester → Certifier (loop).

    Content & Knowledge:
    - **content_documentation**: Researcher → Writer → Tech Reviewer → Publisher.
    - **research**: Question Formulator → Source Finder → Analyzer → Report Writer.
    - **security_audit**: Threat Modeler → Static Analyzer → Pen Tester → Remediator → Verifier.
    - **compliance**: Evidence Collector → Gap Analyzer → Remediator → Auditor → Reporter.
"""

# Individual agents
from .planner import planner_agent
from .decomposer import decomposer_agent
from .summarizer import summarizer_agent
from .reviewer import reviewer_agent
from .coder import coder_agent
from .classifier import classifier_agent
from .extractor import extractor_agent
from .writer import writer_agent
from .guardrail import guardrail_agent

# Pipelines (multi-agent combinations)
from .pipelines import (
    # Original pipelines
    plan_and_execute,
    research_and_write,
    draft_review_loop,
    classify_and_route,
    # Development
    bug_fix,
    refactoring,
    product_development,
    code_review_automation,
    performance_optimization,
    test_suite_generation,
    # Architecture
    system_design,
    database_design,
    api_design,
    # Operations
    incident_response,
    devops_deployment,
    cost_optimization,
    observability_setup,
    disaster_recovery,
    migration,
    # Data
    data_quality,
    etl_pipeline_design,
    # AI/ML
    prompt_engineering,
    rag_pipeline_design,
    model_fine_tuning,
    ai_safety_guardrails,
    # Content & Knowledge
    content_documentation,
    research,
    security_audit,
    compliance,
)

__all__ = [
    # Individual agents
    "planner_agent",
    "decomposer_agent",
    "summarizer_agent",
    "reviewer_agent",
    "coder_agent",
    "classifier_agent",
    "extractor_agent",
    "writer_agent",
    "guardrail_agent",
    # Original pipelines
    "plan_and_execute",
    "research_and_write",
    "draft_review_loop",
    "classify_and_route",
    # Development
    "bug_fix",
    "refactoring",
    "product_development",
    "code_review_automation",
    "performance_optimization",
    "test_suite_generation",
    # Architecture
    "system_design",
    "database_design",
    "api_design",
    # Operations
    "incident_response",
    "devops_deployment",
    "cost_optimization",
    "observability_setup",
    "disaster_recovery",
    "migration",
    # Data
    "data_quality",
    "etl_pipeline_design",
    # AI/ML
    "prompt_engineering",
    "rag_pipeline_design",
    "model_fine_tuning",
    "ai_safety_guardrails",
    # Content & Knowledge
    "content_documentation",
    "research",
    "security_audit",
    "compliance",
]
