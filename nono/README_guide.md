# Nono Step-by-Step Guide

> Complete walkthrough for creating agents, workflows, and task executors — from zero to production-ready AI pipelines.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Running Example](#running-example)
- [Part 1 — Task Executor (Tasker)](#part-1--task-executor-tasker)
  - [Step 1.1 — Your First Task](#step-11--your-first-task)
  - [Step 1.2 — Structured Output with JSON Schema](#step-12--structured-output-with-json-schema)
  - [Step 1.3 — JSON Task Files](#step-13--json-task-files)
  - [Step 1.4 — Multiple Inputs and Placeholders](#step-14--multiple-inputs-and-placeholders)
  - [Step 1.5 — Switching Providers at Runtime](#step-15--switching-providers-at-runtime)
- [Part 2 — Agents](#part-2--agents)
  - [Step 2.1 — Your First Agent](#step-21--your-first-agent)
  - [Step 2.2 — Adding Tools](#step-22--adding-tools)
  - [Step 2.3 — Tools with State (ToolContext)](#step-23--tools-with-state-toolcontext)
  - [Step 2.4 — Agent Delegation (transfer\_to\_agent)](#step-24--agent-delegation-transfer_to_agent)
  - [Step 2.5 — Streaming Events](#step-25--streaming-events)
  - [Step 2.6 — Lifecycle Callbacks](#step-26--lifecycle-callbacks)
  - [Step 2.7 — Tracing and Observability](#step-27--tracing-and-observability)
- [Part 3 — Orchestration Agents](#part-3--orchestration-agents)
  - [Step 3.1 — SequentialAgent (Pipeline)](#step-31--sequentialagent-pipeline)
  - [Step 3.2 — ParallelAgent (Fan-out)](#step-32--parallelagent-fan-out)
  - [Step 3.3 — ParallelAgent with message\_map and result\_key](#step-33--parallelagent-with-message_map-and-result_key)
  - [Step 3.4 — LoopAgent (Iterative)](#step-34--loopagent-iterative)
  - [Step 3.5 — MapReduceAgent (Fan-out + Reduce)](#step-35--mapreduceagent-fan-out--reduce)
  - [Step 3.6 — ConsensusAgent (Vote + Judge)](#step-36--consensusagent-vote--judge)
  - [Step 3.7 — ProducerReviewerAgent (Produce + Review)](#step-37--producerrevieweragent-produce--review)
  - [Step 3.8 — DebateAgent (Adversarial Debate)](#step-38--debateagent-adversarial-debate)
  - [Step 3.9 — EscalationAgent (Tiered Escalation)](#step-39--escalationagent-tiered-escalation)
  - [Step 3.10 — SupervisorAgent (LLM Supervisor)](#step-310--supervisoragent-llm-supervisor)
  - [Step 3.11 — VotingAgent (Majority Vote)](#step-311--votingagent-majority-vote)
  - [Step 3.12 — HandoffAgent (Peer-to-Peer Handoff)](#step-312--handoffagent-peer-to-peer-handoff)
  - [Step 3.13 — GroupChatAgent (N-Agent Group Chat)](#step-313--groupchatagent-n-agent-group-chat)
  - [Step 3.14 — HierarchicalAgent (Multi-Level Hierarchy)](#step-314--hierarchicalagent-multi-level-hierarchy)
  - [Step 3.15 — GuardrailAgent (Pre/Post Validation)](#step-315--guardrailagent-prepost-validation)
  - [Step 3.16 — BestOfNAgent (Best-of-N Sampling)](#step-316--bestofnagent-best-of-n-sampling)
  - [Step 3.17 — BatchAgent (Batch Processing)](#step-317--batchagent-batch-processing)
  - [Step 3.18 — CascadeAgent (Progressive Cascade)](#step-318--cascadeagent-progressive-cascade)
  - [Step 3.19 — TreeOfThoughtsAgent (Tree-of-Thoughts)](#step-319--treeofthoughtsagent-tree-of-thoughts)
  - [Step 3.20 — PlannerAgent (Plan-and-Execute)](#step-320--planneragent-plan-and-execute)
  - [Step 3.21 — SubQuestionAgent (Question Decomposition)](#step-321--subquestionagent-question-decomposition)
  - [Step 3.22 — ContextFilterAgent (Context Filtering)](#step-322--contextfilteragent-context-filtering)
  - [Step 3.23 — ReflexionAgent (Self-Improvement)](#step-323--reflexionagent-self-improvement)
  - [Step 3.24 — RouterAgent (Dynamic LLM Routing)](#step-324--routeragent-dynamic-llm-routing)
  - [Step 3.25 — SpeculativeAgent (Speculative Execution)](#step-325--speculativeagent-speculative-execution)
  - [Step 3.26 — CircuitBreakerAgent (Failure Recovery)](#step-326--circuitbreakeragent-failure-recovery)
  - [Step 3.27 — TournamentAgent (Bracket Elimination)](#step-327--tournamentagent-bracket-elimination)
  - [Step 3.28 — ShadowAgent (Shadow Testing)](#step-328--shadowagent-shadow-testing)
  - [Step 3.29 — CompilerAgent (Prompt Optimisation)](#step-329--compileragent-prompt-optimisation)
  - [Step 3.30 — CheckpointableAgent (Checkpoint/Resume)](#step-330--checkpointableagent-checkpointresume)
  - [Step 3.31 — DynamicFanOutAgent (LLM Decomposition)](#step-331--dynamicfanoutagent-llm-decomposition)
  - [Step 3.32 — SwarmAgent (Agent Handoff Swarm)](#step-332--swarmagent-agent-handoff-swarm)
  - [Step 3.33 — MemoryConsolidationAgent (History Summarisation)](#step-333--memoryconsolidationagent-history-summarisation)
  - [Step 3.34 — PriorityQueueAgent (Priority Execution)](#step-334--priorityqueueagent-priority-execution)
  - [Step 3.35 — MonteCarloAgent (MCTS Search)](#step-335--montecarloagent-mcts-search)
  - [Step 3.36 — GraphOfThoughtsAgent (DAG Reasoning)](#step-336--graphofthoughtsagent-dag-reasoning)
  - [Step 3.37 — BlackboardAgent (Expert Blackboard)](#step-337--blackboardagent-expert-blackboard)
  - [Step 3.38 — MixtureOfExpertsAgent (Gated Experts)](#step-338--mixtureofexpertsagent-gated-experts)
  - [Step 3.39 — CoVeAgent (Chain-of-Verification)](#step-339--coveagent-chain-of-verification)
  - [Step 3.40 — SagaAgent (Compensating Transactions)](#step-340--sagaagent-compensating-transactions)
  - [Step 3.41 — LoadBalancerAgent (Request Distribution)](#step-341--loadbalanceragent-request-distribution)
  - [Step 3.42 — EnsembleAgent (Output Aggregation)](#step-342--ensembleagent-output-aggregation)
  - [Step 3.43 — TimeoutAgent (Deadline Wrapper)](#step-343--timeoutagent-deadline-wrapper)
  - [Step 3.44 — AdaptivePlannerAgent (Re-planning)](#step-344--adaptiveplanneragent-re-planning)
  - [Step 3.45 — Composite Patterns](#step-345--composite-patterns)
- [Part 4 — Workflows](#part-4--workflows)
  - [Step 4.1 — Your First Workflow](#step-41--your-first-workflow)
  - [Step 4.2 — Connecting Steps](#step-42--connecting-steps)
  - [Step 4.3 — Conditional Branching](#step-43--conditional-branching)
  - [Step 4.4 — Using tasker\_node()](#step-44--using-tasker_node)
  - [Step 4.5 — Using agent\_node()](#step-45--using-agent_node)
  - [Step 4.6 — Dynamic Pipeline Manipulation](#step-46--dynamic-pipeline-manipulation)
  - [Step 4.7 — Streaming Workflow Execution](#step-47--streaming-workflow-execution)
  - [Step 4.8 — Parallel Step (concurrent execution)](#step-48--parallel-step-concurrent-execution)
  - [Step 4.9 — Loop Step (deterministic iteration)](#step-49--loop-step-deterministic-iteration)
  - [Step 4.10 — Join Node (barrier synchronization)](#step-410--join-node-barrier-synchronization)
  - [Step 4.11 — Checkpointing and Resume](#step-411--checkpointing-and-resume)
  - [Step 4.12 — Declarative Workflows (YAML / JSON)](#step-412--declarative-workflows-yaml--json)
  - [Step 4.13 — Error Recovery and Retry](#step-413--error-recovery-and-retry)
  - [Step 4.14 — State Transition Audit Trail](#step-414--state-transition-audit-trail)
  - [Step 4.15 — State Schema and Reducers](#step-415--state-schema-and-reducers)
- [Part 5 — Connecting Everything](#part-5--connecting-everything)
  - [Step 5.1 — Tasker as Agent Tool](#step-51--tasker-as-agent-tool)
  - [Step 5.2 — Agent Inside Workflow](#step-52--agent-inside-workflow)
  - [Step 5.3 — Workflow Inside Agent (via FunctionTool)](#step-53--workflow-inside-agent-via-functiontool)
  - [Step 5.4 — Full Pipeline Example](#step-54--full-pipeline-example)
- [Part 6 — ASCII Visualization](#part-6--ascii-visualization)
  - [Step 6.1 — Drawing a Workflow](#step-61--drawing-a-workflow)
  - [Step 6.2 — Drawing an Agent Tree](#step-62--drawing-an-agent-tree)
  - [Step 6.3 — Unified draw()](#step-63--unified-draw)
  - [Step 6.4 — Convenience Methods](#step-64--convenience-methods)
- [Part 7 — Unified Tracing and Observability](#part-7--unified-tracing-and-observability)
  - [Step 7.1 — TraceCollector Basics](#step-71--tracecollector-basics)
  - [Step 7.2 — Tracing a TaskExecutor](#step-72--tracing-a-taskexecutor)
  - [Step 7.3 — Tracing a Workflow](#step-73--tracing-a-workflow)
  - [Step 7.4 — Tracing Agents](#step-74--tracing-agents)
  - [Step 7.5 — Cross-Module Tracing](#step-75--cross-module-tracing)
  - [Step 7.6 — Exporting Traces](#step-76--exporting-traces)
- [Configuration Reference](#configuration-reference)
- [Decision Guide](#decision-guide)
- [See Also](#see-also)

---

## Overview

Nono provides three layers for AI-driven operations, each suited to a different level of complexity:

| Layer | Module | Purpose | Complexity |
| --- | --- | --- | --- |
| **Tasker** | `nono.tasker` | Atomic AI tasks (one prompt → one response) | Low |
| **Agent** | `nono.agent` | Autonomous LLM agents with tools and delegation | Medium |
| **Workflow** | `nono.workflows` | Multi-step pipelines with branching, parallel, loops, joins, and state | Medium–High |

```
┌─────────────────────────────────────────────────┐
│                   Workflow                       │
│   ┌────────┐   ┌────────┐   ┌────────┐         │
│   │ Step 1 │──→│ Step 2 │──→│ Step 3 │         │
│   │(tasker)│   │(agent) │   │(custom)│         │
│   └────────┘   └────────┘   └────────┘         │
│       ↓             ↓                           │
│   TaskExecutor   Agent + Tools                  │
│       ↓             ↓                           │
│   ┌─────────────────────────────────┐           │
│   │     Connector (connector_genai) │           │
│   │  Google · OpenAI · Groq · ...   │           │
│   └─────────────────────────────────┘           │
└─────────────────────────────────────────────────┘
```

---

## Prerequisites

1. Python >= 3.10
2. Nono installed:

```bash
pip install -e .
```

3. API key configured for at least one provider. Options:

| Method | Location |
| --- | --- |
| `config.toml` | `nono/config.toml` → add `api_key = "..."` under `[google]` |
| CSV file | `nono/connector/apikeys.csv` |
| Environment var | `GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc. |
| OS keyring | Stored via `keyring` library |

4. Default provider/model: **Google Gemini** (`gemini-3-flash-preview`)

---

## Running Example

All code examples in this guide build on a single scenario: an **AI Article Pipeline** that researches, writes, reviews, and publishes articles about AI trends in healthcare.

As you progress through the guide, you'll:

1. **Part 1** — Use `TaskExecutor` to classify articles and extract entities.
2. **Part 2** — Create `researcher`, `writer`, and `reviewer` agents with tools.
3. **Part 3** — Compose those agents into orchestration patterns (sequential, parallel, loop, router).
4. **Part 4** — Build the same pipeline as a `Workflow` with branching and state.
5. **Part 5** — Connect all three layers into a unified publishing system.
6. **Part 6** — Visualize the complete pipeline as ASCII diagrams.

Each section reuses and extends components from previous sections, so you can follow the natural progression from a simple LLM call to a complete production pipeline.

---

## Part 1 — Task Executor (Tasker)

The **TaskExecutor** handles atomic AI tasks: send a prompt, get a response. No agents, no tools, no loops — just direct LLM communication.

We start by using it to analyze articles about AI in healthcare.

### Step 1.1 — Your First Task

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor(
    provider="google",
    model="gemini-3-flash-preview",
)

response = executor.execute(
    "Summarize the main AI trends in healthcare for 2026."
)
print(response)
# → "Key AI trends in healthcare for 2026 include AI-powered diagnostics,
#     personalized treatment plans, drug discovery acceleration, ..."
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | `str` | — (required) | AI provider (`google`, `openai`, `groq`, etc.) |
| `model` | `str` | — (required) | Model name |
| `api_key` | `str \| None` | `None` | API key (auto-resolved if omitted) |
| `temperature` | `float \| str` | `0.7` | Temperature or preset name (`"creative"`, `"coding"`, `"data_cleaning"`) |
| `max_tokens` | `int` | `2048` | Max response tokens |

### Step 1.2 — Structured Output with JSON Schema

Now we need more than free text — we want a **structured classification** of an article. Force the LLM to return JSON matching a schema:

```python
schema = {
    "type": "object",
    "required": ["sentiment", "confidence", "topics"],
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
        },
        "confidence": {"type": "number"},
        "topics": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

article = (
    "AI is revolutionizing healthcare with breakthrough diagnostic tools "
    "and personalized treatment plans, reducing misdiagnosis rates by 30%."
)

response = executor.execute(
    f"Classify the sentiment and extract the main topics:\n\n{article}",
    output_schema=schema,
)
print(response)
# → '{"sentiment": "positive", "confidence": 0.93,
#     "topics": ["AI", "healthcare", "diagnostics", "personalized treatment"]}'
```

### Step 1.3 — JSON Task Files

The classification above is useful enough to reuse. Define it as a JSON task file in `nono/tasker/prompts/article_classifier.json`:

```json
{
    "task": {
        "name": "article_classifier",
        "description": "Classify article sentiment and extract topics",
        "version": "1.0.0"
    },
    "genai": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "temperature": "balanced",
        "max_tokens": 1024,
        "response_format": "json"
    },
    "prompts": {
        "system": "You are an article analysis expert. Always respond with valid JSON.",
        "user": "Classify the sentiment and extract key topics from this article:\n\n{data_input_json}\n\nReturn: {\"sentiment\": \"positive|negative|neutral\", \"confidence\": 0.0-1.0, \"topics\": [...]}"
    },
    "output_schema": {
        "type": "object",
        "required": ["sentiment", "confidence", "topics"],
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
            "topics": {"type": "array", "items": {"type": "string"}}
        }
    }
}
```

Now classify any article in a single call:

```python
article = (
    "AI is revolutionizing healthcare with breakthrough diagnostic tools "
    "and personalized treatment plans, reducing misdiagnosis rates by 30%."
)

result = executor.run_json_task(
    "nono/tasker/prompts/article_classifier.json",
    article,
)
print(result)
# → '{"sentiment": "positive", "confidence": 0.93, "topics": ["AI", "healthcare", ...]}'
```

### Step 1.4 — Multiple Inputs and Placeholders

Our classifier can be enhanced with extra context. JSON task templates support named placeholders beyond `{data_input_json}`:

```python
result = executor.run_json_task(
    "nono/tasker/prompts/article_classifier.json",
    article,
    audience="healthcare professionals",   # → {audience}
    focus_area="diagnostics",              # → {focus_area}
)
```

### Step 1.5 — Switching Providers at Runtime

Perhaps we want to compare how OpenAI classifies the same article. Override the provider for a single call without changing the executor:

```python
result = executor.execute(
    f"Classify the sentiment of this article:\n\n{article}",
    config_overrides={"provider": "openai", "model_name": "gpt-4o-mini"},
)
print(result)
```

> At this point we can classify articles, but the process is manual. In Part 2 we'll create **agents** that can research and write articles autonomously.

---

## Part 2 — Agents

Agents are **autonomous LLM-powered entities** that can reason, call tools, and delegate to other agents. They go beyond simple prompt→response by supporting multi-turn tool loops.

We'll create the core agents for our article pipeline: a **researcher** that gathers facts, a **writer** that produces drafts, and eventually a **coordinator** that delegates between them.

### Step 2.1 — Your First Agent

Start with a basic researcher agent — no tools yet, just LLM reasoning:

```python
from nono.agent import Agent, Runner

researcher = Agent(
    name="researcher",
    provider="google",
    model="gemini-3-flash-preview",
    instruction=(
        "You are an AI research specialist focused on healthcare technology. "
        "Provide detailed, factual analysis of AI trends."
    ),
)

runner = Runner(researcher)
response = runner.run("What are the top 3 AI trends in healthcare for 2026?")
print(response)
# → "1. AI-powered diagnostics — ...
#    2. Personalized treatment plans — ...
#    3. Drug discovery acceleration — ..."
```

**Agent constructor parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | — (required) | Unique agent name |
| `provider` | `str` | `"google"` | AI provider |
| `model` | `str \| None` | `None` (provider default) | Model name |
| `instruction` | `str` | `"You are a helpful assistant."` | System prompt |
| `description` | `str` | `""` | Short description (used in delegation) |
| `tools` | `list[FunctionTool]` | `None` | Tools the agent can call |
| `sub_agents` | `list[BaseAgent]` | `None` | Agents for delegation |
| `temperature` | `float \| str` | `0.7` | LLM temperature |
| `max_tokens` | `int \| None` | `None` | Max response tokens |
| `output_format` | `str` | `"text"` | `"text"` or `"json"` |

**Runner methods:**

| Method | Description |
| --- | --- |
| `run(message, **state)` | Single turn → string response |
| `stream(message, **state)` | Yields `Event` objects in real-time |
| `run_async(message, **state)` | Async version of `run()` |
| `astream(message, **state)` | Async version of `stream()` |
| `reset()` | Create a fresh session |

### Step 2.2 — Adding Tools

Our researcher needs to find real data. Define tools with the `@tool` decorator — the JSON schema is auto-generated from type hints:

```python
from nono.agent import Agent, Runner, tool

@tool(description="Search the web for recent articles and papers.")
def web_search(query: str) -> str:
    # In production, call a real search API (SerpApi, Perplexity, etc.)
    return f"[Search results for: {query}]"

@tool(description="Calculate a relevance score (0-100) for a source.")
def score_relevance(title: str, topic: str) -> str:
    overlap = len(set(title.lower().split()) & set(topic.lower().split()))
    return str(min(overlap * 25, 100))

researcher = Agent(
    name="researcher",
    provider="google",
    instruction=(
        "You research AI trends in healthcare. "
        "Use web_search to find articles and score_relevance to rank them."
    ),
    tools=[web_search, score_relevance],
)

runner = Runner(researcher)
print(runner.run("Find recent articles about AI-powered diagnostics"))
# Agent calls web_search("AI diagnostics healthcare 2026") → results
# Then score_relevance(...) → ranks them → produces a summary
```

Or use `FunctionTool` manually:

```python
from nono.agent import FunctionTool

def fetch_pubmed(query: str) -> str:
    """Fetch papers from PubMed."""
    return f"[PubMed results for: {query}]"

pubmed = FunctionTool(fetch_pubmed, description="Search PubMed for medical papers.")
```

### Step 2.3 — Tools with State (ToolContext)

Our researcher should **remember** findings across multiple tool calls within a session. Add a `tool_context: ToolContext` parameter to access session state:

```python
from nono.agent import tool, ToolContext

@tool(description="Save a research finding to memory.")
def save_finding(text: str, tool_context: ToolContext) -> str:
    findings = tool_context.state.setdefault("findings", [])
    findings.append(text)
    # Shared content — visible to ALL agents in the session
    tool_context.shared_content.save("latest_finding", text)
    return f"Saved finding #{len(findings)}"

@tool(description="List all saved research findings.")
def list_findings(tool_context: ToolContext) -> str:
    findings = tool_context.state.get("findings", [])
    return "\n".join(f"- {f}" for f in findings) or "No findings yet."

researcher = Agent(
    name="researcher",
    provider="google",
    instruction=(
        "You research AI trends in healthcare. "
        "Use web_search to find data and save_finding to store important facts. "
        "When done, use list_findings to review everything."
    ),
    tools=[web_search, save_finding, list_findings],
)
```

> **Note:** `ToolContext` is automatically excluded from the JSON Schema sent to the LLM and injected at invocation time.

### Step 2.3b — ACI Quality: Validating Tool Descriptions

Anthropic's *"Building effective agents"* emphasises that **tool definitions deserve as much prompt engineering as your prompts**. A tool with a vague or missing description forces the model to guess — leading to wrong tool selection and wasted tokens.

Nono validates tool descriptions **automatically when you create an Agent**:

```python
from nono.agent import Agent, FunctionTool

# ⚠️ This triggers warnings at construction time:
bad = FunctionTool(lambda q: q, name="s", description="search")
agent = Agent(name="a", provider="google", tools=[bad])
# WARNING: Tool 's': Description is only 6 chars (minimum recommended: 10).
# WARNING: Tool 's': Tool name is too short.
```

For CI/CD or test suites, use `validate_tools()` programmatically:

```python
from nono.agent import validate_tools

issues = validate_tools(agent.tools, warn=False)
assert len(issues) == 0, f"Fix tool descriptions: {issues}"
```

**Rule of thumb:** describe *what* the tool does, *when* to use it, and *what* it returns. See [ACI Quality — Tool Description Validation](agent/README_agent.md#aci-quality--tool-description-validation) for the full reference.

### Step 2.4 — Agent Delegation (transfer_to_agent)

Now let's add a **writer** agent and a **coordinator** that decides who should handle each part. When an agent has `sub_agents`, a `transfer_to_agent` tool is **automatically registered** — the LLM decides when to delegate:

```python
writer = Agent(
    name="writer",
    provider="google",
    instruction=(
        "You write clear, engaging articles about AI trends in healthcare. "
        "Use the research findings provided to you."
    ),
    description="Writes polished articles from research findings.",
)

coordinator = Agent(
    name="coordinator",
    provider="google",
    instruction=(
        "You coordinate the article creation process. "
        "First, delegate to 'researcher' to gather facts about the topic. "
        "Then, delegate to 'writer' to produce the article."
    ),
    sub_agents=[researcher, writer],
)

runner = Runner(coordinator)
print(runner.run("Create an article about AI-powered diagnostics in healthcare."))
# coordinator → transfer_to_agent("researcher") → gathers data
#             → transfer_to_agent("writer") → writes the article
```

### Step 2.5 — Streaming Events

Instead of waiting for the full article, process events as the coordinator works:

```python
from nono.agent import Runner, EventType

runner = Runner(coordinator)

for event in runner.stream("Create an article about AI-powered diagnostics"):
    match event.event_type:
        case EventType.TOOL_CALL:
            print(f"  🔧 Calling: {event.data['tool']}")
        case EventType.TOOL_RESULT:
            print(f"  ✓ Result: {event.content[:80]}")
        case EventType.AGENT_MESSAGE:
            print(f"  💬 {event.content[:80]}")
        case EventType.AGENT_TRANSFER:
            print(f"  → Delegating to: {event.data['target_agent']}")
```

**Event types:**

| EventType | When |
| --- | --- |
| `USER_MESSAGE` | User input recorded |
| `AGENT_MESSAGE` | Agent produces a response |
| `TOOL_CALL` | Tool invocation starts |
| `TOOL_RESULT` | Tool returns a result |
| `STATE_UPDATE` | Session state changes |
| `AGENT_TRANSFER` | Control passes to another agent |
| `ERROR` | An error occurred |

### Step 2.6 — Lifecycle Callbacks

Intercept execution at key points. For example, ensure our writer always produces Markdown output:

```python
from nono.agent import Agent, InvocationContext, BaseAgent

def log_before(agent: BaseAgent, ctx: InvocationContext) -> str | None:
    print(f"[BEFORE] {agent.name} receiving: {ctx.user_message[:60]}...")
    # Return a string to short-circuit (skip LLM call)
    # Return None to continue normally
    return None

def ensure_markdown(agent: BaseAgent, ctx: InvocationContext, response: str) -> str | None:
    """Ensure the article starts with a Markdown heading."""
    if not response.startswith("#"):
        return f"# AI Trends in Healthcare\n\n{response}"
    # Return None to keep the original
    return None

writer = Agent(
    name="writer",
    provider="google",
    instruction="Write an article about AI trends in healthcare.",
    before_agent_callback=log_before,
    after_agent_callback=ensure_markdown,
)
```

### Step 2.7 — Tracing and Observability

Track how many tokens and tool calls the coordinator pipeline consumes:

```python
from nono.agent import Agent, Runner, TraceCollector

collector = TraceCollector()
runner = Runner(agent=coordinator, trace_collector=collector)

runner.run("Create an article about AI-powered diagnostics in healthcare")

# Human-readable summary
collector.print_summary()

# Programmatic access
print(f"Total tokens: {collector.total_tokens}")
print(f"LLM calls: {collector.total_llm_calls}")
print(f"Tool calls: {collector.total_tool_calls}")

# Export for dashboards/logging
import json
print(json.dumps(collector.export(), indent=2, default=str))
```

> We now have a `researcher` with tools, a `writer` with callbacks, and a `coordinator` that wires them together. In Part 3 we'll compose these agents into powerful orchestration patterns.

---

## Part 3 — Orchestration Agents

Orchestration agents coordinate multiple agents **without writing custom loop logic**. They are fully composable — you can nest them.

All examples below reuse the `researcher`, `writer`, and other agents introduced in Part 2.

```
Deterministic ◄──────────────────────────────────────────────► Dynamic

Sequential  Parallel  Loop               RouterAgent       transfer_to_agent
(fixed)     (all)     (repeat)     (LLM picks agents+mode) (LLM decides when)
```

### Step 3.1 — SequentialAgent (Pipeline)

Run agents one after another. The researcher gathers data, the writer produces a draft, the reviewer improves it:

```python
from nono.agent import Agent, SequentialAgent, Runner

# researcher and writer defined in Part 2

reviewer = Agent(
    name="reviewer",
    provider="google",
    instruction=(
        "You review articles about AI in healthcare. "
        "Check facts, improve clarity, and suggest concrete edits."
    ),
)

article_pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer, reviewer],
)

runner = Runner(article_pipeline)
result = runner.run("AI-powered diagnostics in healthcare for 2026")
print(result)
# researcher produces research → writer creates draft → reviewer polishes it
```

### Step 3.2 — ParallelAgent (Fan-out)

For a comprehensive article, we need **multiple perspectives** analyzed simultaneously:

```python
from nono.agent import Agent, ParallelAgent, Runner

tech_analyst = Agent(
    name="tech_analyst",
    provider="google",
    instruction="Analyze AI in healthcare from a technology perspective.",
)
business_analyst = Agent(
    name="business_analyst",
    provider="google",
    instruction="Analyze AI in healthcare from a business and market perspective.",
)
ethics_analyst = Agent(
    name="ethics_analyst",
    provider="google",
    instruction="Analyze ethical implications of AI in healthcare.",
)

multi_analysis = ParallelAgent(
    name="multi_analysis",
    sub_agents=[tech_analyst, business_analyst, ethics_analyst],
    max_workers=3,
)

runner = Runner(multi_analysis)
result = runner.run("AI-powered diagnostic tools in hospitals")
# All three analysts run concurrently on the same input
```

### Step 3.3 — ParallelAgent with message_map and result_key

Give each analyst a **specific angle** and **collect** their results in session state:

```python
perspectives = ParallelAgent(
    name="perspectives",
    sub_agents=[tech_analyst, business_analyst, ethics_analyst],
    message_map={
        "tech_analyst": "Analyze the technical capabilities of AI diagnostics: accuracy, speed, integration with EHR systems.",
        "business_analyst": "Analyze the market opportunity for AI diagnostics: TAM, competition, reimbursement models.",
        "ethics_analyst": "Analyze ethical concerns: bias in training data, liability, patient consent.",
    },
    result_key="analysis",
)

runner = Runner(perspectives)
runner.run("analyze")

# All results collected in one dict:
print(runner.session.state["analysis"])
# → {"tech_analyst": "...", "business_analyst": "...", "ethics_analyst": "..."}
```

### Step 3.4 — LoopAgent (Iterative)

Our reviewer can iteratively improve a draft until it reaches a quality threshold:

```python
from nono.agent import Agent, LoopAgent, Runner

improver = Agent(
    name="improver",
    provider="google",
    instruction=(
        "Review and improve the article about AI in healthcare. "
        "Rate its quality 0-100 and set state['quality_score']. "
        "Focus on accuracy, clarity, and engagement."
    ),
)

refinement = LoopAgent(
    name="refinement",
    sub_agents=[improver],
    max_iterations=3,
    stop_condition=lambda state: state.get("quality_score", 0) >= 85,
)

runner = Runner(refinement)
result = runner.run("Improve this draft about AI diagnostics in healthcare...")
# Loops until quality_score >= 85 or 3 iterations
```

### Step 3.5 — MapReduceAgent (Fan-out + Reduce)

Fan-out to multiple agents in parallel, then reduce all outputs with a single reducer agent:

```python
from nono.agent import Agent, MapReduceAgent, Runner

search_web = Agent(name="search_web", instruction="Search the web.", provider="google")
search_db = Agent(name="search_db", instruction="Search the database.", provider="google")
summariser = Agent(name="summariser", instruction="Combine all results.", provider="google")

mapreduce = MapReduceAgent(
    name="summarise_all",
    sub_agents=[search_web, search_db],
    reduce_agent=summariser,
)

runner = Runner(mapreduce)
result = runner.run("What do we know about quantum computing?")
```

### Step 3.6 — ConsensusAgent (Vote + Judge)

Run multiple agents on the same input, then a judge synthesises a single consensus:

```python
from nono.agent import Agent, ConsensusAgent, Runner

model_a = Agent(name="model_a", instruction="Answer the question.", provider="google")
model_b = Agent(name="model_b", instruction="Answer the question.", provider="openai")
judge = Agent(name="judge", instruction="Synthesise a consensus.", provider="google")

consensus = ConsensusAgent(
    name="fact_check",
    sub_agents=[model_a, model_b],
    judge_agent=judge,
)

runner = Runner(consensus)
result = runner.run("What is the capital of Australia?")
```

### Step 3.7 — ProducerReviewerAgent (Produce + Review)

Iterative produce-then-review loop until the reviewer approves:

```python
from nono.agent import Agent, ProducerReviewerAgent, Runner

writer = Agent(name="writer", instruction="Write a blog post.", provider="google")
editor = Agent(name="editor", instruction="Review. Say APPROVED if good.", provider="google")

pr = ProducerReviewerAgent(
    name="blog_pipeline",
    producer=writer,
    reviewer=editor,
    max_iterations=3,
    approval_keyword="APPROVED",
)

runner = Runner(pr)
result = runner.run("Write a blog post about AI in healthcare")
```

### Step 3.8 — DebateAgent (Adversarial Debate)

Two agents argue in rounds, then a judge renders a verdict:

```python
from nono.agent import Agent, DebateAgent, Runner

optimist = Agent(name="optimist", instruction="Argue the positive case.", provider="google")
pessimist = Agent(name="pessimist", instruction="Argue the negative case.", provider="google")
moderator = Agent(name="moderator", instruction="Judge the debate.", provider="google")

debate = DebateAgent(
    name="ethics_debate",
    agent_a=optimist,
    agent_b=pessimist,
    judge=moderator,
    max_rounds=3,
)

runner = Runner(debate)
result = runner.run("Should AI replace radiologists?")
```

### Step 3.9 — EscalationAgent (Tiered Escalation)

Try agents in order; stop at the first success. Cost-efficient fallback chains:

```python
from nono.agent import Agent, EscalationAgent, Runner

fast = Agent(name="fast", instruction="Answer quickly. Say 'I don't know' if unsure.", provider="groq")
powerful = Agent(name="powerful", instruction="Answer thoroughly.", provider="openai")

escalation = EscalationAgent(
    name="smart_fallback",
    sub_agents=[fast, powerful],
    failure_keyword="I don't know",
)

runner = Runner(escalation)
result = runner.run("Explain the Riemann hypothesis")
```

### Step 3.10 — SupervisorAgent (LLM Supervisor)

An LLM-powered supervisor delegates to workers, evaluates results, and re-delegates if unsatisfied:

```python
from nono.agent import Agent, SupervisorAgent, Runner

coder = Agent(name="coder", description="Writes code.", instruction="Write Python.", provider="google")
writer = Agent(name="writer", description="Writes prose.", instruction="Write text.", provider="google")

supervisor = SupervisorAgent(
    name="manager",
    provider="google",
    sub_agents=[coder, writer],
    max_iterations=3,
)

runner = Runner(supervisor)
result = runner.run("Write a Python function to calculate BMI")
```

### Step 3.11 — VotingAgent (Majority Vote)

N agents answer in parallel, the most frequent answer wins — no LLM judge required:

```python
from nono.agent import Agent, VotingAgent, Runner

model_a = Agent(name="model_a", instruction="Answer the math question.", provider="google")
model_b = Agent(name="model_b", instruction="Answer the math question.", provider="openai")
model_c = Agent(name="model_c", instruction="Answer the math question.", provider="deepseek")

voting = VotingAgent(
    name="majority_vote",
    sub_agents=[model_a, model_b, model_c],
)

runner = Runner(voting)
result = runner.run("What is 7 * 8?")
```

### Step 3.12 — HandoffAgent (Peer-to-Peer Handoff)

Agents transfer **full control** to each other via handoff directives. Unlike `transfer_to_agent`, the receiving agent takes full ownership:

```python
from nono.agent import Agent, HandoffAgent, Runner

triage = Agent(
    name="triage",
    instruction="Route to the right expert. Write HANDOFF: math_tutor or HANDOFF: history_tutor.",
    provider="google",
)
math = Agent(name="math_tutor", instruction="Answer math questions.", provider="google")
history = Agent(name="history_tutor", instruction="Answer history questions.", provider="google")

handoff = HandoffAgent(
    name="tutoring",
    entry_agent=triage,
    handoff_rules={"triage": [math, history], "math_tutor": [triage]},
)

runner = Runner(handoff)
result = runner.run("What year did the French Revolution start?")
```

### Step 3.13 — GroupChatAgent (N-Agent Group Chat)

N-agent group chat with manager-controlled speaker selection. Supports round-robin, LLM-based, or custom strategies:

```python
from nono.agent import Agent, GroupChatAgent, Runner

writer = Agent(name="writer", instruction="Write marketing copy.", provider="google")
reviewer = Agent(
    name="reviewer",
    instruction="Review copy. Say APPROVED when satisfied.",
    provider="google",
)

chat = GroupChatAgent(
    name="creative_team",
    sub_agents=[writer, reviewer],
    speaker_selection="round_robin",
    max_rounds=6,
    termination_keyword="APPROVED",
)

runner = Runner(chat)
result = runner.run("Create a tagline for an AI startup")
```

### Step 3.14 — HierarchicalAgent (Multi-Level Hierarchy)

A multi-level tree-shaped orchestration where an LLM manager delegates to department heads — which may themselves be orchestration agents with their own sub-agents:

```python
from nono.agent import Agent, SequentialAgent, HierarchicalAgent, Runner

architect = Agent(name="architect", instruction="Design the system.", provider="google")
developer = Agent(name="developer", instruction="Implement the code.", provider="google")
tester = Agent(name="tester", instruction="Write and run tests.", provider="google")

backend_team = SequentialAgent(
    name="backend_team",
    description="Backend development pipeline.",
    sub_agents=[architect, developer],
)
qa = Agent(name="qa", description="Quality assurance.", instruction="Review code.", provider="google")

cto = HierarchicalAgent(
    name="cto",
    provider="google",
    sub_agents=[backend_team, qa],
    max_iterations=3,
    manager_instruction="Delegate implementation to backend_team first, then QA.",
)

runner = Runner(cto)
result = runner.run("Build a REST API for user management")
```

### Step 3.15 — GuardrailAgent (Pre/Post Validation)

Wrap a main agent with pre- and post-validators that check inputs and outputs. If the post-validator rejects the output, the main agent retries automatically:

```python
from nono.agent import Agent, GuardrailAgent, Runner

writer = Agent(name="writer", instruction="Write marketing copy.", provider="google")
checker = Agent(name="checker", instruction="Reply REJECTED if the text contains toxic language, otherwise APPROVED.", provider="google")

safe_writer = GuardrailAgent(
    name="safe_writer",
    main_agent=writer,
    post_validator=checker,
    rejection_keyword="REJECTED",
    max_retries=2,
)

runner = Runner(safe_writer)
result = runner.run("Write a tagline for our product")
```

### Step 3.16 — BestOfNAgent (Best-of-N Sampling)

Run the same agent N times in parallel and pick the best response using a scoring function:

```python
from nono.agent import Agent, BestOfNAgent, Runner

writer = Agent(name="writer", instruction="Write a creative headline.", provider="google")

best_writer = BestOfNAgent(
    name="best_writer",
    agent=writer,
    n=3,
    score_fn=lambda r: float(len(r)),  # Pick the longest response
    result_key="scoring",
)

runner = Runner(best_writer)
result = runner.run("Headline for an AI conference")
# session.state["scoring"] → {"best_index": 2, "best_score": 47.0, "all_scores": [...]}
```

### Step 3.17 — BatchAgent (Batch Processing)

Process a list of items through one agent with concurrency control:

```python
from nono.agent import Agent, BatchAgent, Runner

classifier = Agent(name="classifier", instruction="Classify the sentiment: positive, negative, neutral.", provider="google")

batch = BatchAgent(
    name="batch_classify",
    agent=classifier,
    items=["I love this!", "Terrible product.", "It's okay."],
    template="Classify: {item}",
    max_workers=3,
    result_key="classifications",
)

runner = Runner(batch)
result = runner.run("Classify all items")
# session.state["classifications"] → {0: "positive", 1: "negative", 2: "neutral"}
```

### Step 3.18 — CascadeAgent (Progressive Cascade)

Try progressively more capable (and expensive) agents in sequence, stopping when a quality threshold is met:

```python
from nono.agent import Agent, CascadeAgent, Runner

flash = Agent(name="flash", instruction="Answer concisely.", provider="google", model="gemini-3-flash-preview")
pro = Agent(name="pro", instruction="Answer thoroughly.", provider="google", model="gemini-2.5-pro-preview-06-05")

cascade = CascadeAgent(
    name="smart_cascade",
    sub_agents=[flash, pro],
    score_fn=lambda r: 1.0 if len(r) > 200 else 0.3,
    threshold=0.8,
    result_key="cascade_info",
)

runner = Runner(cascade)
result = runner.run("Explain quantum entanglement")
# If flash gives a long-enough answer, pro never runs
```

### Step 3.19 — RouterAgent (Dynamic LLM Routing)

Instead of fixed pipelines, let an LLM **decide** which agent handles each request. This is useful for an editorial assistant that might need research, writing, review, or analysis depending on the user's question:

```python
from nono.agent import RouterAgent, Runner

editorial_router = RouterAgent(
    name="editorial_router",
    provider="google",
    model="gemini-3-flash-preview",
    sub_agents=[researcher, writer, reviewer, tech_analyst],
)

runner = Runner(editorial_router)

# Router picks researcher:
print(runner.run("Find recent papers about AI diagnostics in radiology"))

# Router picks writer:
print(runner.run("Write an introduction for the healthcare AI article"))

# Router picks reviewer:
print(runner.run("Check if the statistics in section 2 are accurate"))

# Router picks tech_analyst:
print(runner.run("What are the technical limitations of current AI diagnostic tools?"))
```

### Step 3.20 — SpeculativeAgent (Speculative Execution)

Race multiple agents in parallel; the first to pass an evaluator threshold wins. Slower agents are abandoned.

```python
from nono.agent import SpeculativeAgent, Agent, Runner

fast = Agent(name="fast", instruction="Answer briefly.", provider="groq")
slow = Agent(name="slow", instruction="Answer thoroughly.", provider="openai")

spec = SpeculativeAgent(
    name="racer",
    sub_agents=[fast, slow],
    evaluator_fn=lambda r: 1.0 if len(r) > 100 else 0.3,
    min_confidence=0.8,
    result_key="spec_result",
)

runner = Runner(spec)
print(runner.run("Explain transformers"))
```

### Step 3.21 — CircuitBreakerAgent (Failure Recovery)

Tracks failures and switches to a fallback agent when the primary repeatedly fails.

```python
from nono.agent import CircuitBreakerAgent, Agent, Runner

primary = Agent(name="primary", instruction="Answer questions.", provider="openai")
fallback = Agent(name="fallback", instruction="Provide a safe answer.", provider="google")

cb = CircuitBreakerAgent(
    name="resilient",
    agent=primary,
    fallback_agent=fallback,
    failure_threshold=3,
    recovery_timeout=60,
    result_key="cb_result",
)

runner = Runner(cb)
print(runner.run("What is Python?"))
```

### Step 3.22 — TournamentAgent (Bracket Elimination)

Runs a single-elimination bracket: agents compete in pairs and a judge picks the winner each round.

```python
from nono.agent import TournamentAgent, Agent, Runner

agents = [Agent(name=f"writer_{i}", instruction=f"Write a poem (style {i}).", provider="google") for i in range(4)]
judge = Agent(name="judge", instruction="Pick the better poem. Reply only with the author name.", provider="google")

tourney = TournamentAgent(
    name="poetry_slam",
    sub_agents=agents,
    judge_agent=judge,
    result_key="tourney_result",
)

runner = Runner(tourney)
print(runner.run("Write a haiku about the ocean"))
```

### Step 3.23 — ShadowAgent (Shadow Testing)

Runs a stable agent and a shadow agent in parallel. Only the stable output is returned; the shadow is logged for comparison.

```python
from nono.agent import ShadowAgent, Agent, Runner

stable = Agent(name="production", instruction="Answer accurately.", provider="openai")
shadow = Agent(name="candidate", instruction="Answer accurately.", provider="google")

sa = ShadowAgent(
    name="shadow_test",
    stable_agent=stable,
    shadow_agent=shadow,
    diff_logger=lambda s, sh: print(f"Match: {s == sh}"),
    result_key="shadow_result",
)

runner = Runner(sa)
print(runner.run("Capital of France?"))
```

### Step 3.24 — CompilerAgent (Prompt Optimisation)

Iteratively refines a target agent's instruction using examples and an LLM optimiser (DSPy-inspired).

```python
from nono.agent import CompilerAgent, Agent, Runner

target = Agent(name="classifier", instruction="Classify text sentiment.", provider="google")

compiler = CompilerAgent(
    name="optimiser",
    target_agent=target,
    examples=[
        {"input": "I love this!", "expected": "positive"},
        {"input": "Terrible experience.", "expected": "negative"},
    ],
    metric_fn=lambda output, expected: 1.0 if expected in output.lower() else 0.0,
    max_iterations=3,
    result_key="compiler_result",
)

runner = Runner(compiler)
print(runner.run("Optimise the classifier"))
```

### Step 3.25 — CheckpointableAgent (Checkpoint/Resume)

Executes sub-agents sequentially, saving progress after each step. Can resume from the last checkpoint.

```python
from nono.agent import CheckpointableAgent, Agent, Runner

steps = [
    Agent(name="research", instruction="Research the topic.", provider="google"),
    Agent(name="draft", instruction="Write a draft.", provider="google"),
    Agent(name="polish", instruction="Polish the draft.", provider="google"),
]

ckpt = CheckpointableAgent(
    name="pipeline",
    sub_agents=steps,
    checkpoint_key="pipeline_ckpt",
    result_key="pipeline_result",
)

runner = Runner(ckpt)
print(runner.run("Write an article about AI safety"))
```

### Step 3.26 — DynamicFanOutAgent (LLM Decomposition)

Uses an LLM to dynamically decompose a task into work items, fans out to workers in parallel, then reduces.

```python
from nono.agent import DynamicFanOutAgent, Agent, Runner

worker = Agent(name="analyst", instruction="Analyse the given aspect.", provider="google")
reducer = Agent(name="synthesiser", instruction="Synthesise all analyses.", provider="google")

fanout = DynamicFanOutAgent(
    name="research",
    worker_agent=worker,
    reducer_agent=reducer,
    model="gemini-3-flash-preview",
    provider="google",
    max_items=5,
    result_key="fanout_result",
)

runner = Runner(fanout)
print(runner.run("Analyse the impact of AI on healthcare"))
```

### Step 3.27 — SwarmAgent (Agent Handoff Swarm)

Agents hand off to each other dynamically using session state, similar to OpenAI's Swarm pattern.

```python
from nono.agent import SwarmAgent, Agent, Runner

triage = Agent(name="triage", instruction="Classify the request. Set __next_agent__='billing' or __next_agent__='support' in state.", provider="google")
billing = Agent(name="billing", instruction="Handle billing. Set __done__=True when finished.", provider="google")
support = Agent(name="support", instruction="Handle support. Set __done__=True when finished.", provider="google")

swarm = SwarmAgent(
    name="helpdesk",
    sub_agents=[triage, billing, support],
    initial_agent="triage",
    max_handoffs=5,
    result_key="swarm_result",
)

runner = Runner(swarm)
print(runner.run("I need to update my payment method"))
```

### Step 3.28 — MemoryConsolidationAgent (History Summarisation)

Automatically summarises old conversation events when history exceeds a threshold, keeping context manageable.

```python
from nono.agent import MemoryConsolidationAgent, Agent, Runner

main = Agent(name="assistant", instruction="Help the user.", provider="google")
summarizer = Agent(name="summarizer", instruction="Summarise the conversation history concisely.", provider="google")

mc = MemoryConsolidationAgent(
    name="smart_assistant",
    main_agent=main,
    summarizer_agent=summarizer,
    event_threshold=50,
    keep_recent=10,
    memory_key="conversation_summary",
    result_key="mc_result",
)

runner = Runner(mc)
print(runner.run("Continue our discussion about AI ethics"))
```

### Step 3.29 — PriorityQueueAgent (Priority Execution)

Executes agents grouped by priority level — higher-priority groups run first, agents within a group run in parallel.

```python
from nono.agent import PriorityQueueAgent, Agent, Runner

critical = Agent(name="security_check", instruction="Check for security issues.", provider="google")
normal = Agent(name="analysis", instruction="Analyse the data.", provider="google")
background = Agent(name="logging", instruction="Log the interaction.", provider="google")

pq = PriorityQueueAgent(
    name="processor",
    sub_agents=[critical, normal, background],
    priority_map={"security_check": 0, "analysis": 1, "logging": 2},
    result_key="pq_result",
)

runner = Runner(pq)
print(runner.run("Process this user request"))
```

### Step 3.35 — MonteCarloAgent (MCTS Search)

Monte Carlo Tree Search with UCT — exploration vs exploitation for complex reasoning.

```python
from nono.agent import MonteCarloAgent, Agent, Runner

thinker = Agent(name="thinker", instruction="Propose a solution.", provider="google")

mcts = MonteCarloAgent(
    name="mcts",
    agent=thinker,
    evaluate_fn=lambda r: 1.0 if "correct" in r.lower() else 0.3,
    n_simulations=20,
    max_depth=3,
    result_key="mcts_result",
)

runner = Runner(mcts)
print(runner.run("Find the optimal approach to solve this problem"))
```

### Step 3.36 — GraphOfThoughtsAgent (DAG Reasoning)

DAG-based reasoning: generate, aggregate, and score thoughts with merge support.

```python
from nono.agent import GraphOfThoughtsAgent, Agent, Runner

generator = Agent(name="gen", instruction="Propose an idea.", provider="google")
merger = Agent(name="merge", instruction="Merge these ideas into one.", provider="google")

got = GraphOfThoughtsAgent(
    name="got",
    agent=generator,
    aggregate_agent=merger,
    score_fn=lambda r: 1.0 if len(r) > 50 else 0.5,
    n_branches=3,
    n_rounds=2,
    result_key="got_result",
)

runner = Runner(got)
print(runner.run("Design a novel caching strategy"))
```

### Step 3.37 — BlackboardAgent (Expert Blackboard)

Shared blackboard where experts contribute partial solutions iteratively.

```python
from nono.agent import BlackboardAgent, Agent, Runner

symptom_agent = Agent(name="symptoms", instruction="Analyse symptoms.", provider="google")
lab_agent = Agent(name="lab", instruction="Interpret lab results.", provider="google")

bb = BlackboardAgent(
    name="diagnosis",
    sub_agents=[symptom_agent, lab_agent],
    termination_fn=lambda board: board.get("diagnosis") is not None,
    max_iterations=5,
    result_key="bb_result",
)

runner = Runner(bb)
print(runner.run("Patient presents with fatigue and joint pain"))
```

### Step 3.38 — MixtureOfExpertsAgent (Gated Experts)

Gating function selects top-k experts and blends their outputs.

```python
from nono.agent import MixtureOfExpertsAgent, Agent, Runner

math = Agent(name="math", instruction="Solve math problems.", provider="google")
code = Agent(name="code", instruction="Write code solutions.", provider="google")
writing = Agent(name="writing", instruction="Write essays.", provider="google")

moe = MixtureOfExpertsAgent(
    name="moe",
    sub_agents=[math, code, writing],
    gating_fn=lambda msg, agents: {
        "math": 0.8 if "calculate" in msg else 0.1,
        "code": 0.8 if "code" in msg else 0.1,
        "writing": 0.8 if "write" in msg else 0.1,
    },
    top_k=2,
    result_key="moe_result",
)

runner = Runner(moe)
print(runner.run("Calculate the area of a circle with radius 5"))
```

### Step 3.39 — CoVeAgent (Chain-of-Verification)

4-phase anti-hallucination: draft → plan verification questions → verify → revise.

```python
from nono.agent import CoVeAgent, Agent, Runner

drafter = Agent(name="drafter", instruction="Draft an answer.", provider="google")
planner = Agent(name="planner", instruction="Generate verification questions.", provider="google")
verifier = Agent(name="verifier", instruction="Verify the claim.", provider="google")
reviser = Agent(name="reviser", instruction="Produce verified final answer.", provider="google")

cove = CoVeAgent(
    name="verifier",
    drafter=drafter,
    planner=planner,
    verifier=verifier,
    reviser=reviser,
    max_questions=3,
    result_key="cove_result",
)

runner = Runner(cove)
print(runner.run("Who won the 2024 Nobel Prize in Physics?"))
```

### Step 3.40 — SagaAgent (Compensating Transactions)

Multi-step transactions with automatic rollback on failure.

```python
from nono.agent import SagaAgent, Agent, Runner

reserve = Agent(name="reserve", instruction="Reserve inventory.", provider="google")
charge = Agent(name="charge", instruction="Charge payment.", provider="google")
ship = Agent(name="ship", instruction="Ship the order.", provider="google")
release = Agent(name="release", instruction="Release reserved inventory.", provider="google")
refund = Agent(name="refund", instruction="Refund payment.", provider="google")

saga = SagaAgent(
    name="order",
    steps=[
        {"action": reserve, "compensate": release},
        {"action": charge, "compensate": refund},
        {"action": ship},
    ],
    failure_detector=lambda o: "ERROR" in o.upper(),
    result_key="saga_result",
)

runner = Runner(saga)
print(runner.run("Process order #12345"))
```

### Step 3.41 — LoadBalancerAgent (Request Distribution)

Distribute requests across equivalent agents using round-robin, random, or least-used.

```python
from nono.agent import LoadBalancerAgent, Agent, Runner

agent_a = Agent(name="model_a", instruction="Answer the question.", provider="google")
agent_b = Agent(name="model_b", instruction="Answer the question.", provider="openai")

lb = LoadBalancerAgent(
    name="lb",
    sub_agents=[agent_a, agent_b],
    strategy="round_robin",
    result_key="lb_result",
)

runner = Runner(lb)
print(runner.run("What is machine learning?"))
```

### Step 3.42 — EnsembleAgent (Output Aggregation)

Run multiple agents and aggregate their outputs.

```python
from nono.agent import EnsembleAgent, Agent, Runner

model_a = Agent(name="gemini", instruction="Answer.", provider="google")
model_b = Agent(name="gpt", instruction="Answer.", provider="openai")

ens = EnsembleAgent(
    name="ensemble",
    sub_agents=[model_a, model_b],
    aggregate_fn="concat",
    result_key="ens_result",
)

runner = Runner(ens)
print(runner.run("Explain quantum computing"))
```

### Step 3.43 — TimeoutAgent (Deadline Wrapper)

Enforce a time limit with automatic fallback.

```python
from nono.agent import TimeoutAgent, Agent, Runner

slow_agent = Agent(name="thinker", instruction="Think deeply.", provider="google")

guarded = TimeoutAgent(
    name="guarded",
    agent=slow_agent,
    timeout_seconds=10.0,
    fallback_message="Request timed out. Please try again.",
    result_key="timeout_result",
)

runner = Runner(guarded)
print(runner.run("Solve this complex problem"))
```

### Step 3.44 — AdaptivePlannerAgent (Re-planning)

Re-plans after every step, handling emergent tasks dynamically.

```python
from nono.agent import AdaptivePlannerAgent, Agent, Runner

researcher = Agent(name="researcher", instruction="Research the topic.", provider="google")
writer = Agent(name="writer", instruction="Write based on research.", provider="google")

planner = AdaptivePlannerAgent(
    name="adaptive",
    sub_agents=[researcher, writer],
    model="gemini-3-flash-preview",
    provider="google",
    max_steps=5,
    result_key="plan_result",
)

runner = Runner(planner)
print(runner.run("Write a comprehensive report on quantum computing"))
```

### Step 3.30 — Composite Patterns

Orchestration agents are fully nestable. Here we build the **complete article pipeline**:

1. **Research** in parallel (tech + business perspectives)
2. **Write + review** sequentially
3. **Refine** the write+review loop up to 2 times

```python
from nono.agent import (
    Agent, SequentialAgent, ParallelAgent, LoopAgent, Runner,
)

# Stage 1: parallel research
research_stage = ParallelAgent(
    name="research",
    sub_agents=[tech_analyst, business_analyst],
    max_workers=2,
)

# Stage 2: sequential write → review
write_review = SequentialAgent(
    name="write_review",
    sub_agents=[writer, reviewer],
)

# Stage 3: iterate write+review up to 2x
refine_loop = LoopAgent(
    name="refine",
    sub_agents=[write_review],
    max_iterations=2,
)

# Complete composite pipeline
full_pipeline = SequentialAgent(
    name="full_pipeline",
    sub_agents=[research_stage, refine_loop],
)

runner = Runner(full_pipeline)
result = runner.run("AI trends in healthcare for 2026")
```

> We now have a sophisticated multi-agent pipeline. In Part 4, we'll build the **same workflow** using state-machine pipelines, which give us explicit branching and deterministic control flow.

---

## Part 4 — Workflows

Workflows are **state-machine pipelines** where each step receives a shared `state` dict and returns updates. They are ideal for deterministic multi-step processes.

We'll rebuild parts of our article pipeline as a Workflow, adding features like conditional branching (approve/revise) that agents alone can't express declaratively.

### Step 4.1 — Your First Workflow

Start with the simplest version: take a topic and build an article outline.

```python
from nono.workflows import Workflow

def fetch_topic(state: dict) -> dict:
    """Prepare the article structure from the topic."""
    topic = state["topic"]
    return {
        "headline": f"AI Trends in {topic}",
        "sections": ["Introduction", "Key Trends", "Challenges", "Conclusion"],
    }

def format_outline(state: dict) -> dict:
    """Format sections into a Markdown outline."""
    outline = "\n".join(f"## {s}" for s in state["sections"])
    return {"outline": f"# {state['headline']}\n\n{outline}"}

flow = Workflow("outline_builder")
flow.step("fetch_topic", fetch_topic)
flow.step("format_outline", format_outline)

result = flow.run(topic="Healthcare")
print(result["outline"])
# → # AI Trends in Healthcare
#   ## Introduction
#   ## Key Trends
#   ## Challenges
#   ## Conclusion
```

**Key points:**
- Steps execute in **registration order** by default.
- Each step function takes `state: dict` and returns a `dict` of updates.
- State is shared and accumulated across all steps.

### Step 4.2 — Connecting Steps

Add a word-count step and explicitly define the execution order:

```python
def count_words(state: dict) -> dict:
    """Count words in the draft and check if it meets the minimum."""
    word_count = len(state.get("draft", "").split())
    return {"word_count": word_count, "meets_minimum": word_count >= 500}

flow = Workflow("article_flow")
flow.step("fetch_topic", fetch_topic)
flow.step("format_outline", format_outline)
flow.step("count_words", count_words)

# Explicit execution order (overrides registration order)
flow.connect("fetch_topic", "format_outline", "count_words")

result = flow.run(topic="Healthcare")
print(f"Words: {result['word_count']}, meets minimum: {result['meets_minimum']}")
```

### Step 4.3 — Conditional Branching

After quality-checking an article, route to **approve** or **revise**:

```python
from nono.workflows import Workflow, END

def quality_check(state: dict) -> dict:
    """Score the article quality (simplified: based on length)."""
    draft = state.get("draft", "")
    score = min(len(draft) // 10, 100)
    return {"quality_score": score}

def approve_article(state: dict) -> dict:
    """Mark the article as approved for publication."""
    return {"status": "approved", "final_article": state["draft"]}

def request_revision(state: dict) -> dict:
    """Request revision with feedback."""
    return {"status": "revision_needed", "feedback": "Needs more depth on diagnostic accuracy."}

flow = Workflow("review_flow")
flow.step("quality_check", quality_check)
flow.step("approve", approve_article)
flow.step("revise", request_revision)

# branch_if: predicate with then/otherwise
flow.branch_if(
    "quality_check",
    lambda s: s["quality_score"] >= 80,
    then="approve",
    otherwise="revise",
)

# Or use branch() — function returns the next step name
# flow.branch("quality_check", lambda s: "approve" if s["quality_score"] >= 80 else "revise")

# END stops the workflow early
# flow.branch("revise", lambda s: "quality_check" if s["retries"] < 3 else END)

result = flow.run(draft="A comprehensive article about AI diagnostics in healthcare"
                  " covering recent advances in radiology, pathology, and genomics...")
print(f"Status: {result['status']}")  # → "approved" or "revision_needed"
```

### Step 4.4 — Using tasker_node()

`tasker_node()` creates a step that delegates to `TaskExecutor`. Here we add AI-powered classification and entity extraction to our workflow — using the same article data from Part 1:

```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("ai_analysis")

# Classify the article using the LLM
flow.step("classify", tasker_node(
    provider="google",
    model="gemini-3-flash-preview",
    system_prompt=(
        "Classify the article's main topic area. "
        "Return JSON: {\"category\": \"...\", \"confidence\": 0.0-1.0}"
    ),
    input_key="article_text",
    output_key="classification",
))

# Extract named entities (people, organizations, technologies)
flow.step("extract_entities", tasker_node(
    provider="google",
    model="gemini-3-flash-preview",
    system_prompt="Extract all named entities (people, organizations, technologies) from the text.",
    input_key="article_text",
    output_key="entities",
))

flow.connect("classify", "extract_entities")

result = flow.run(
    article_text="Google DeepMind announced a new AI diagnostic tool for radiology, "
                 "partnering with NHS England to deploy it across 50 hospitals."
)
print(result["classification"])  # → '{"category": "healthcare AI", "confidence": 0.95}'
print(result["entities"])        # → "Google DeepMind, NHS England, ..."
```

**`tasker_node()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | `str` | `"google"` | AI provider |
| `model` | `str` | `"gemini-3-flash-preview"` | Model name |
| `temperature` | `float \| str` | `0.7` | Temperature |
| `max_tokens` | `int` | `2048` | Max tokens |
| `system_prompt` | `str \| None` | `None` | System instruction |
| `output_schema` | `dict \| None` | `None` | JSON schema |
| `input_key` | `str` | `"input"` | State key to read from |
| `output_key` | `str` | `"output"` | State key to write to |
| `task_file` | `str \| None` | `None` | Path to JSON task file |

### Step 4.5 — Using agent_node()

`agent_node()` embeds a full Nono Agent as a workflow step. Here we plug the `researcher` and `writer` agents from Part 2 into a workflow with quality control:

```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node

# researcher and writer defined in Part 2

flow = Workflow("article_workflow")
flow.step("research", agent_node(researcher, input_key="topic", output_key="research_notes"))
flow.step("write", agent_node(writer, input_key="research_notes", output_key="draft"))
flow.step("quality_check", quality_check)  # from Step 4.3
flow.connect("research", "write", "quality_check")

result = flow.run(topic="AI-powered diagnostics in healthcare for 2026")
print(f"Draft: {result['draft'][:200]}...")
print(f"Quality score: {result['quality_score']}")
```

**`agent_node()` parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | — (required) | Agent instance |
| `input_key` | `str` | `"input"` | State key to read the user message from |
| `output_key` | `str` | `"output"` | State key to write the response to |
| `state_keys` | `dict \| None` | `None` | Map `{runner_key: workflow_state_key}` to forward extra state |

### Step 4.6 — Dynamic Pipeline Manipulation

Modify the article workflow before (or between) runs:

```python
flow = Workflow("flexible")
flow.step("research", agent_node(researcher, input_key="topic", output_key="research_notes"))
flow.step("write", agent_node(writer, input_key="research_notes", output_key="draft"))
flow.step("quality_check", quality_check)

# Insert an outline step between research and writing
flow.insert_after("research", "outline", format_outline)

# Insert a review step after quality check
flow.insert_after(
    "quality_check",
    "review",
    agent_node(reviewer, input_key="draft", output_key="feedback"),
)

# Inspect the pipeline
print(flow.steps)
# → ['research', 'outline', 'write', 'quality_check', 'review']

print(flow.describe())
# Human-readable graph description

# Swap, move, replace, remove as needed
flow.swap_steps("outline", "write")
flow.move_before("quality_check", "review")
flow.remove_step("outline")
```

### Step 4.7 — Streaming Workflow Execution

Watch each step complete in real-time:

```python
flow = Workflow("article_workflow")
flow.step("research", agent_node(researcher, input_key="topic", output_key="research_notes"))
flow.step("write", agent_node(writer, input_key="research_notes", output_key="draft"))
flow.step("quality_check", quality_check)
flow.connect("research", "write", "quality_check")

for step_name, state_snapshot in flow.stream(topic="AI diagnostics in healthcare"):
    print(f"[{step_name}] keys={list(state_snapshot.keys())}")
# → [research] keys=['topic', 'research_notes']
#   [write] keys=['topic', 'research_notes', 'draft']
#   [quality_check] keys=['topic', 'research_notes', 'draft', 'quality_score']
```

Async version:

```python
async for step_name, state_snapshot in flow.astream(topic="AI diagnostics"):
    print(f"[{step_name}] done")
```

> We've rebuilt our article pipeline as a Workflow with explicit state, branching, and streaming. In Part 5, we'll connect **all three layers** — Tasker, Agents, and Workflows — into a unified publishing system.

### Step 4.8 — Parallel Step (concurrent execution)

Run multiple functions concurrently within a workflow. Each function receives an isolated copy of the state and results are merged:

```python
from nono.workflows import Workflow

def extract_entities(state: dict) -> dict:
    """Extract named entities from the text."""
    return {"entities": ["Google DeepMind", "NHS England"]}

def detect_sentiment(state: dict) -> dict:
    """Detect sentiment of the article."""
    return {"sentiment": "positive"}

def extract_keywords(state: dict) -> dict:
    """Extract keywords from the text."""
    return {"keywords": ["AI", "diagnostics", "healthcare"]}

flow = Workflow("parallel_analysis")
flow.step("ingest", lambda s: {"article": s["raw_text"]})
flow.parallel_step(
    "analyze",
    [extract_entities, detect_sentiment, extract_keywords],
    max_workers=3,
)
flow.step("report", lambda s: {
    "summary": f"{len(s['entities'])} entities, sentiment={s['sentiment']}"
})
flow.connect_chain(["ingest", "analyze", "report"])

result = flow.run(raw_text="Google DeepMind announced ...")
print(result["entities"])   # → ["Google DeepMind", "NHS England"]
print(result["sentiment"])  # → "positive"
print(result["keywords"])   # → ["AI", "diagnostics", "healthcare"]
```

**Key points:**
- Each function runs in a thread via `ThreadPoolExecutor`.
- State copies are isolated: functions **cannot** see each other's updates during execution.
- All returned dicts are merged into the shared state after completion.

### Step 4.9 — Loop Step (deterministic iteration)

Repeat a step until a condition fails or a maximum iteration count is reached:

```python
from nono.workflows import Workflow

def refine_draft(state: dict) -> dict:
    """Simulate iterative refinement — quality improves each iteration."""
    quality = state.get("quality", 0.5)
    quality += 0.15
    return {"quality": min(quality, 1.0)}

flow = Workflow("refinement_loop")
flow.step("draft", lambda s: {"quality": 0.5, "draft": "initial draft"})
flow.loop_step(
    "refine",
    refine_draft,
    condition="quality < 0.9",
    max_iterations=10,
)
flow.step("publish", lambda s: {"status": "published"})
flow.connect_chain(["draft", "refine", "publish"])

result = flow.run()
print(f"Quality: {result['quality']:.2f}")  # → 0.95 (after 3 iterations)
print(result["status"])                     # → "published"
```

**`loop_step()` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — (required) | Step name |
| `fn` | `Callable` | — (required) | Function to repeat |
| `condition` | `str` | — (required) | Condition expression (e.g. `"quality < 0.9"`) |
| `max_iterations` | `int` | `100` | Safety limit to prevent infinite loops |

The `condition` is evaluated against the current state. The loop stops when the condition evaluates to `False` or `max_iterations` is reached.

### Step 4.10 — Join Node (barrier synchronization)

A `join()` node waits for specific upstream steps to complete before allowing execution to continue. This is useful when parallel branches must converge:

```python
flow = Workflow("fan_out_fan_in")
flow.step("start", lambda s: {"data": "raw input"})
flow.parallel_step("enrich", [enrich_a, enrich_b, enrich_c])
flow.join("barrier", wait_for=["enrich"], reducer=merge_results)
flow.step("final", lambda s: {"done": True})
flow.connect_chain(["start", "enrich", "barrier", "final"])

result = flow.run(data="raw input")
```

**`join()` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — (required) | Step name |
| `wait_for` | `list[str]` | — (required) | Step names that must have executed |
| `reducer` | `Callable \| None` | `None` | Optional function to reduce/merge results |

If a `wait_for` step has **not** been executed, the join raises a `RuntimeError`.

### Step 4.11 — Checkpointing and Resume

Enable automatic state persistence after each step. If a long pipeline fails halfway, resume from the last checkpoint:

```python
import tempfile
from nono.workflows import Workflow

flow = Workflow("resilient_pipeline")
flow.enable_checkpoints(tempfile.mkdtemp())

flow.step("fetch",   fetch_data)
flow.step("enrich",  enrich_data)
flow.step("publish", publish_data)
flow.connect_chain(["fetch", "enrich", "publish"])

# First run — crashes at "publish"
try:
    flow.run(source="api")
except Exception:
    pass

# Resume — skips "fetch" and "enrich", starts at "publish"
result = flow.run(source="api", resume=True)
```

**Key points:**
- `enable_checkpoints(directory)` — checkpoints are saved as JSON files in the directory.
- Each step writes an atomic checkpoint after successful execution.
- `resume=True` loads the latest checkpoint and skips already-completed steps.

### Step 4.12 — Declarative Workflows (YAML / JSON)

Define a workflow in a YAML or JSON file instead of Python code:

```yaml
# pipeline.yaml
name: article_pipeline
steps:
  - name: classify
    fn: classify_article
  - name: enrich
    parallel: [extract_entities, detect_sentiment]
    max_workers: 2
  - name: validate
    fn: validate_output
connections:
  - [classify, enrich]
  - [enrich, validate]
```

Load and run:

```python
from nono.workflows import load_workflow

# step_registry maps step names to functions
registry = {
    "classify_article": classify_article,
    "extract_entities": extract_entities,
    "detect_sentiment": detect_sentiment,
    "validate_output": validate_output,
}

flow = load_workflow("pipeline.yaml", step_registry=registry)
result = flow.run(article="...")
```

**`load_workflow()` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — (required) | Path to YAML or JSON definition |
| `step_registry` | `dict \| None` | `None` | Maps function names to callables. Required when `fn` refers to function names. |

> With control-flow nodes (`parallel_step`, `loop_step`, `join`), checkpointing, and declarative definitions, Workflows now support full deterministic orchestration — from simple linear pipelines to complex fan-out/fan-in patterns with fault recovery.

---

### Step 4.13 — Error Recovery and Retry

Steps can declare automatic retry and a fallback route. The `retry` parameter controls how many times a failing step is re-executed, and `on_error` names a fallback step:

```python
from nono.workflows import Workflow

def call_api(state):
    resp = requests.get(f"https://api.example.com/{state['query']}")
    resp.raise_for_status()
    return {"data": resp.json()}

def use_cache(state):
    err = state["__error__"]  # {"step": "fetch", "type": "HTTPError", "message": "..."}
    return {"data": cached_lookup(state["query"]), "fallback_used": True}

flow = Workflow("resilient_pipeline")
flow.step("fetch", call_api, retry=3, on_error="cache_fallback")
flow.step("cache_fallback", use_cache)
flow.step("process", lambda s: {"result": transform(s["data"])})
flow.connect("fetch", "process")

result = flow.run(query="weather")
# If fetch fails after 3 retries → cache_fallback runs → process continues
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `retry` | `int` | `0` | Number of extra attempts on failure |
| `on_error` | `str \| None` | `None` | Fallback step name; `None` = re-raise |

---

### Step 4.14 — State Transition Audit Trail

Every step records a `StateTransition` in `flow.transitions`. The list resets at each run:

```python
from nono.workflows import Workflow

flow = Workflow("traced")
flow.step("a", lambda s: {"x": 1})
flow.step("b", lambda s: {"y": s["x"] + 1})
flow.connect("a", "b")

result = flow.run()
for t in flow.transitions:
    print(f"{t.step}: changed={t.keys_changed} time={t.duration_ms:.1f}ms retries={t.retries}")
```

`StateTransition` fields: `step`, `keys_changed` (frozenset), `branch_taken`, `duration_ms`, `retries`, `error`.

---

### Step 4.15 — State Schema and Reducers

Attach a `StateSchema` to validate types and control how keys are merged:

```python
from nono.workflows import Workflow, StateSchema

schema = StateSchema(
    fields={"topic": str, "notes": list, "score": float},
    reducers={"notes": lambda old, new: (old or []) + new},
)

flow = Workflow("validated", schema=schema)
flow.step("a", lambda s: {"notes": ["first"], "score": 0.5})
flow.step("b", lambda s: {"notes": ["second"], "score": 0.9})
flow.connect("a", "b")

result = flow.run(topic="AI")
# result["notes"] == ["first", "second"]  (appended via reducer)
# result["score"] == 0.9                   (replaced — no reducer)
```

Without a reducer, values are overwritten. With a reducer, the function `reducer(old, new)` merges them.

> With error recovery, audit trails, and state schemas, Workflows provide enterprise-grade reliability — retry transient failures, trace every step execution, and enforce type contracts on your pipeline state.

---

## Part 5 — Connecting Everything

Nono provides **two independent orchestration systems** that are fully composable:

| System | Module | Who controls execution |
|---|---|---|
| **Deterministic** | `Workflow` | You — via `step`, `parallel_step`, `loop_step`, `join`, `branch_if` |
| **Agentic** | `Agent` + orchestrators | The LLM — via tool calls, delegation, routing |

The two layers are orthogonal: a Workflow controls execution order, while an agent reasons freely within its step. This means you can mix them in the same pipeline — for example, a `parallel_step` that fans out to `agent_node` steps, or a `loop_step` that iterates over an agent's output.

The examples below show how to bridge the layers. Each reuses components from the previous parts.

### Step 5.1 — Tasker as Agent Tool

Use `tasker_tool()` or `json_task_tool()` to give our editor agent access to the classification task from Part 1:

```python
from nono.agent import Agent, Runner
from nono.agent.tasker_tool import tasker_tool, json_task_tool

# Wraps a TaskExecutor call as a tool
quality_scorer = tasker_tool(
    name="score_quality",
    description="Score an article's quality from 0-100.",
    provider="google",
    system_prompt="Rate the quality of this article from 0 to 100. Consider accuracy, clarity, and engagement. Return only the number.",
)

# Wraps the JSON task file from Step 1.3 as a tool
classifier = json_task_tool("nono/tasker/prompts/article_classifier.json")

editor = Agent(
    name="editor",
    provider="google",
    instruction=(
        "You are an editor for healthcare AI articles. "
        "Use score_quality to evaluate articles and article_classifier to categorize them."
    ),
    tools=[quality_scorer, classifier],
)

runner = Runner(editor)
print(runner.run(
    "Evaluate and classify this article: "
    "AI is revolutionizing healthcare with breakthrough diagnostic tools "
    "and personalized treatment plans..."
))
# editor calls score_quality → "87", classifier → {"sentiment": "positive", ...}
```

### Step 5.2 — Agent Inside Workflow

Use `agent_node()` to combine the `researcher` and `writer` from Part 2 with the `tasker_node` classification from Part 1 — all in a single workflow:

```python
from nono.workflows import Workflow, agent_node, tasker_node

flow = Workflow("full_article_pipeline")

# Step 1: Classify the topic (TaskExecutor via tasker_node)
flow.step("classify", tasker_node(
    system_prompt="Classify the article topic area. Return the category as plain text.",
    input_key="topic",
    output_key="category",
))

# Step 2: Research the topic (Agent via agent_node)
flow.step("research", agent_node(researcher, input_key="topic", output_key="research_notes"))

# Step 3: Write the article (Agent via agent_node)
flow.step("write", agent_node(writer, input_key="research_notes", output_key="draft"))

# Step 4: Format the final output (pure Python)
def format_final(state: dict) -> dict:
    return {
        "final_article": f"Category: {state['category']}\n\n{state['draft']}",
    }

flow.step("format", format_final)
flow.connect("classify", "research", "write", "format")

result = flow.run(topic="AI-powered diagnostics in healthcare for 2026")
print(result["final_article"][:300])
```

### Step 5.3 — Workflow Inside Agent (via FunctionTool)

Wrap the review workflow from Step 4.3 as a tool, so the editor agent can invoke it:

```python
from nono.agent import Agent, Runner, FunctionTool
from nono.workflows import Workflow, tasker_node

# Build the review workflow (from Step 4.3 + classification)
review_flow = Workflow("review_pipeline")
review_flow.step("quality_check", quality_check)  # from Step 4.3
review_flow.step("classify", tasker_node(
    system_prompt="Classify the article sentiment as positive, negative, or neutral.",
    input_key="draft",
    output_key="sentiment",
))
review_flow.connect("quality_check", "classify")

# Wrap as a tool
def run_review(article_text: str) -> str:
    """Run quality review and sentiment classification on an article."""
    result = review_flow.run(draft=article_text)
    return f"Score: {result.get('quality_score')}/100, Sentiment: {result.get('sentiment')}"

review_tool = FunctionTool(run_review, description="Review and classify an article.")

# The editor agent uses the workflow as a tool
editor = Agent(
    name="editor",
    provider="google",
    instruction=(
        "You are an editor for healthcare AI articles. "
        "Use the review tool to evaluate articles before approving them."
    ),
    tools=[review_tool],
)

runner = Runner(editor)
print(runner.run(
    "Review this article before publishing: "
    "AI-powered diagnostic tools are transforming radiology departments "
    "across Europe, with early detection rates improving by 35%..."
))
# editor calls run_review(...) → "Score: 85/100, Sentiment: positive"
```

### Step 5.4 — Full Pipeline Example (Hybrid Orchestration)

Here we bring together **every layer** from the entire guide into a single publishing system. The Workflow provides deterministic structure (parallel fan-out, loops, checkpoints), while agents provide LLM-driven reasoning within individual steps:

```python
from nono.agent import (
    Agent, SequentialAgent, ParallelAgent, Runner, TraceCollector, tool,
)
from nono.workflows import Workflow, agent_node, tasker_node

# ── Tools (from Part 2) ──────────────────────────────────────────────────

@tool(description="Search the web for recent articles and papers.")
def web_search(query: str) -> str:
    return f"[Search results for: {query}]"

# ── Agents (from Parts 2–3) ─────────────────────────────────────────────

researcher = Agent(
    name="researcher",
    provider="google",
    instruction=(
        "You research AI trends in healthcare. "
        "Use web_search to find recent articles and papers."
    ),
    tools=[web_search],
)

writer = Agent(
    name="writer",
    provider="google",
    instruction="You write clear, engaging articles about AI in healthcare.",
)

# ── Orchestration (from Part 3) ─────────────────────────────────────────

article_pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer],
)

# ── Workflow (from Parts 4–5) ───────────────────────────────────────────

flow = Workflow("publish_pipeline")

# Generate the article using the agent pipeline
flow.step("generate", agent_node(
    article_pipeline, input_key="topic", output_key="draft",
))

# Score quality using a TaskExecutor
flow.step("quality_check", tasker_node(
    system_prompt=(
        "Rate the quality of this article from 0 to 100. "
        "Return JSON: {\"score\": N}"
    ),
    input_key="draft",
    output_key="quality",
))

# Evaluate the score with plain Python
def evaluate(state: dict) -> dict:
    import json
    score = json.loads(state.get("quality", "{}")).get("score", 0)
    return {"score": score, "approved": score >= 80}

flow.step("evaluate", evaluate)
flow.connect("generate", "quality_check", "evaluate")

# ── Execute with tracing ────────────────────────────────────────────────

result = flow.run(topic="AI trends in healthcare for 2026")
print(f"Draft: {result.get('draft', '')[:200]}...")
print(f"Quality: {result.get('score')}/100")
print(f"Approved: {result.get('approved')}")
```

---

## Part 6 — ASCII Visualization

Nono includes a built-in ASCII renderer that draws Workflow pipelines and Agent orchestration trees directly in the terminal. No external dependencies needed.

Let's visualize the structures we've built throughout this guide.

```python
from nono import draw, draw_workflow, draw_agent
```

### Step 6.1 — Drawing a Workflow

Visualize the publishing workflow from Step 5.4:

```python
from nono import draw_workflow

print(draw_workflow(flow))
```

Output:

```
📋 publish_pipeline (Workflow, 3 steps)
├── ○ generate
├── ○ quality_check
└── ○ evaluate
```

**Branched workflows** (like the review flow from Step 4.3) show the branch step with the `◆` icon, and its targets nested underneath:

```python
print(draw_workflow(review_flow))
```

```
📋 review_flow (Workflow, 3 steps)
└── ◆ quality_check
    ├── ○ approve
    └── ○ revise
```

Pass `title=False` to omit the root label.

### Step 6.2 — Drawing an Agent Tree

Visualize the composite agent pipeline from Step 3.6:

```python
from nono import draw_agent

print(draw_agent(full_pipeline))
```

Output:

```
⏩ full_pipeline (SequentialAgent)
├── ⏸ research (ParallelAgent, 2 workers)
│   ├── 🤖 tech_analyst (LlmAgent, google/gemini-3-flash-preview)
│   └── 🤖 business_analyst (LlmAgent, google/gemini-3-flash-preview)
└── 🔁 refine (LoopAgent, max 2x)
    └── ⏩ write_review (SequentialAgent)
        ├── 🤖 writer (LlmAgent, google/gemini-3-flash-preview)
        └── 🤖 reviewer (LlmAgent, google/gemini-3-flash-preview)
```

**Icons reference:**

| Icon | Meaning |
| --- | --- |
| 📋 | Workflow root |
| ○ | Workflow step / custom agent |
| ◆ | Branch step (with nested targets) |
| 🤖 | `LlmAgent` / `Agent` |
| ⏩ | `SequentialAgent` |
| ⏸ | `ParallelAgent` |
| 🔁 | `LoopAgent` |
| 🔀 | `RouterAgent` |
| 🔧 | Tool (shown as children) |

Agents with tools show them as children. The `researcher` from Part 2:

```python
print(draw_agent(researcher))
```

```
🤖 researcher (LlmAgent, google/gemini-3-flash-preview, 3 tools)
├── 🔧 web_search
├── 🔧 save_finding
└── 🔧 list_findings
```

### Step 6.3 — Unified draw()

The `draw()` function auto-detects the object type:

```python
from nono import draw

print(draw(flow))           # calls draw_workflow()
print(draw(full_pipeline))  # calls draw_agent()
```

Raises a `TypeError` for unsupported types.

### Step 6.4 — Convenience Methods

Both `Workflow` and `BaseAgent` expose a `.draw()` method directly:

```python
# Workflow — same as draw_workflow(flow)
print(flow.draw())

# Agent — same as draw_agent(full_pipeline)
print(full_pipeline.draw())
```

---

## Part 7 — Unified Tracing and Observability

Every Nono module — **TaskExecutor**, **Workflow**, and **Agent** — shares the same `TraceCollector` system. Pass a single collector across all layers and get a unified view of token usage, execution times, LLM calls, and tool invocations.

### Step 7.1 — TraceCollector Basics

Import from the top-level package:

```python
from nono import TraceCollector, Trace, TraceStatus, LLMCall, TokenUsage
```

A `TraceCollector` records hierarchical `Trace` objects. Each trace captures:

| Field | Type | Description |
| --- | --- | --- |
| `agent_name` | `str` | Name of the component (agent, workflow, task executor) |
| `agent_type` | `str` | Type: `LlmAgent`, `Workflow`, `WorkflowStep`, `TaskExecutor`, `CodeExecuter` |
| `status` | `TraceStatus` | `RUNNING`, `SUCCESS`, or `ERROR` |
| `duration_ms` | `float` | Wall-clock time in milliseconds |
| `llm_calls` | `list[LLMCall]` | Individual LLM requests with token counts |
| `tools_used` | `list[ToolRecord]` | Tool invocations with arguments and results |
| `children` | `list[Trace]` | Nested sub-traces (workflow steps, sub-agents) |
| `metadata` | `dict` | Arbitrary key-value pairs |

Create a collector and inspect it after execution:

```python
collector = TraceCollector()

# ... pass collector to any Nono module ...

# Quick summary
collector.print_summary()

# Aggregate metrics
print(f"Tokens:     {collector.total_tokens}")
print(f"LLM calls:  {collector.total_llm_calls}")
print(f"Tool calls:  {collector.total_tool_calls}")
```

### Step 7.2 — Tracing a TaskExecutor

Pass `trace_collector` to `execute()` to record the LLM call:

```python
from nono.tasker import TaskExecutor
from nono import TraceCollector

tasker = TaskExecutor(
    system_prompt="Classify this article into a category.",
    provider="google",
)

collector = TraceCollector()
result = tasker.execute(
    "AI-powered diagnostics improve early cancer detection by 35%.",
    trace_collector=collector,
)

collector.print_summary()
# [success] TaskExecutor (TaskExecutor) — 820ms, 1 LLM call(s), ~312 tokens, 0 tool(s)
```

The trace records provider, model, approximate token counts, and timing. If the call fails, the trace captures the error with `TraceStatus.ERROR`.

### Step 7.3 — Tracing a Workflow

Pass `trace_collector` as a keyword argument to `run()`, `run_async()`, `stream()`, or `astream()`:

```python
from nono.workflows import Workflow, tasker_node
from nono import TraceCollector

flow = Workflow("article_pipeline")

flow.step("classify", tasker_node(
    system_prompt="Classify this article: {input}. Return JSON: {\"category\": \"...\"}",
    input_key="input",
    output_key="category",
))

def evaluate(state: dict) -> dict:
    import json
    cat = json.loads(state.get("category", "{}")).get("category", "unknown")
    return {"label": cat}

flow.step("evaluate", evaluate)
flow.connect("classify", "evaluate")

collector = TraceCollector()
result = flow.run(trace_collector=collector, input="AI in healthcare trends 2026")

collector.print_summary()
```

The output shows a hierarchical structure — the workflow as parent with each step as a child:

```
[success] article_pipeline (Workflow) — 1250ms, 0 LLM call(s), 0 tokens, 0 tool(s), 2 child(ren)
  [success] classify (WorkflowStep) — 1100ms, 0 LLM call(s), 0 tokens, 0 tool(s)
  [success] evaluate (WorkflowStep) — 2ms, 0 LLM call(s), 0 tokens, 0 tool(s)
```

**Streaming** works the same way — trace is built incrementally as each step completes:

```python
collector = TraceCollector()
for step_name, state in flow.stream(trace_collector=collector):
    print(f"  completed: {step_name}")

collector.print_summary()
```

### Step 7.4 — Tracing Agents

Agents accept the collector through `Runner`:

```python
from nono.agent import Agent, Runner, TraceCollector

researcher = Agent(
    name="researcher",
    provider="google",
    instruction="Research AI trends in healthcare.",
    tools=[web_search],
)

collector = TraceCollector()
runner = Runner(agent=researcher, trace_collector=collector)
runner.run("Find the latest AI diagnostic tools in European hospitals")

collector.print_summary()
# [success] researcher (LlmAgent) — 2340ms, 2 LLM call(s), 1580 tokens, 1 tool(s)
```

Orchestration agents (Sequential, Parallel, Loop, Router) create nested traces automatically:

```python
pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer],
)

collector = TraceCollector()
runner = Runner(agent=pipeline, trace_collector=collector)
runner.run("Create an article about AI diagnostics in healthcare")

collector.print_summary()
# [success] article_pipeline (SequentialAgent) — 5200ms, ...
#   [success] researcher (LlmAgent) — 2800ms, ...
#   [success] writer (LlmAgent) — 2400ms, ...
```

### Step 7.5 — Cross-Module Tracing

Use **one collector** across workflow + agents + tasker for a unified view of the entire pipeline:

```python
from nono.agent import Agent, SequentialAgent, Runner
from nono.workflows import Workflow, agent_node, tasker_node
from nono import TraceCollector

# Reuse agents from Part 2–3
researcher = Agent(name="researcher", provider="google",
                   instruction="Research AI healthcare trends.", tools=[web_search])
writer = Agent(name="writer", provider="google",
               instruction="Write a healthcare AI article.")
article_team = SequentialAgent(name="article_team", sub_agents=[researcher, writer])

# Build workflow with mixed node types
flow = Workflow("full_publish_pipeline")
flow.step("draft", agent_node(article_team, input_key="topic", output_key="draft"))
flow.step("quality", tasker_node(
    system_prompt="Rate article quality 0–100. Return JSON: {\"score\": N}",
    input_key="draft", output_key="quality",
))
flow.step("decide", lambda s: {
    "approved": int(s.get("quality", "0")) >= 80
})
flow.connect("draft", "quality", "decide")

# Single collector captures everything
collector = TraceCollector()
result = flow.run(trace_collector=collector, topic="AI diagnostics in healthcare 2026")

collector.print_summary()
```

Output shows the full execution tree:

```
[success] full_publish_pipeline (Workflow) — 8500ms, 0 LLM call(s), 0 tokens, 0 tool(s), 3 child(ren)
  [success] draft (WorkflowStep) — 5200ms, ...
  [success] quality (WorkflowStep) — 3100ms, ...
  [success] decide (WorkflowStep) — 2ms, ...
```

### Step 7.6 — Exporting Traces

Export traces as JSON dicts for dashboards, logging, or analysis:

```python
import json

data = collector.export()
print(json.dumps(data, indent=2, default=str))
```

Each trace dict includes:

```json
[
  {
    "agent_name": "full_publish_pipeline",
    "agent_type": "Workflow",
    "status": "success",
    "start_time": "2025-01-15T10:30:00",
    "end_time": "2025-01-15T10:30:08",
    "duration_ms": 8500.0,
    "llm_calls": [],
    "tools_used": [],
    "children": [
      {
        "agent_name": "draft",
        "agent_type": "WorkflowStep",
        "status": "success",
        "duration_ms": 5200.0,
        "llm_calls": [],
        "children": []
      }
    ]
  }
]
```

Use `total_tokens`, `total_llm_calls`, and `total_tool_calls` for cost tracking:

```python
# Cost estimation (approximate, using OpenAI-style pricing)
INPUT_COST_PER_1K = 0.00015
OUTPUT_COST_PER_1K = 0.0006

total = collector.total_tokens
estimated_cost = (total / 1000) * INPUT_COST_PER_1K
print(f"~{total} tokens, estimated cost: ${estimated_cost:.4f}")
```

> The unified `TraceCollector` gives you full observability across TaskExecutor, Workflow, and Agent — from a single entry point.

---

## Configuration Reference

Settings in `nono/config.toml`:

### Provider settings

```toml
[google]
default_model = "gemini-3-flash-preview"
# api_key = "your-key"  # Optional

[openai]
default_model = "gpt-4o-mini"
```

### Agent defaults

```toml
[agent]
default_provider = "google"
default_model = ""               # Empty = use provider's default
temperature = 0.7
max_tokens = 4096
router_max_iterations = 3
router_temperature = 0.0
```

### Workflow defaults

```toml
[workflow]
log_steps = true
default_input_key = "input"
default_output_key = "output"
```

### Supported providers

| Provider | Key | Default Model |
| --- | --- | --- |
| Google Gemini | `google` | `gemini-3-flash-preview` |
| OpenAI | `openai` | `gpt-4o-mini` |
| Perplexity | `perplexity` | `sonar` |
| DeepSeek | `deepseek` | `deepseek-chat` |
| xAI | `xai` | `grok-3` |
| Groq | `groq` | `llama-3.3-70b-versatile` |
| Cerebras | `cerebras` | `llama-3.3-70b` |
| NVIDIA | `nvidia` | `meta/llama-3.3-70b-instruct` |
| GitHub Models | `github` | `openai/gpt-5` |
| OpenRouter | `openrouter` | `openrouter/auto` |
| Azure AI | `azure` | `openai/gpt-4o` |
| Vercel | `vercel` | `anthropic/claude-opus-4.5` |
| Ollama | `ollama` | `llama3` |

---

## Decision Guide

Use this flowchart to choose the right approach:

```
Is it a single prompt → response?
├── YES → TaskExecutor (Part 1)
└── NO
    ├── Does the LLM need tools or multi-turn reasoning?
    │   ├── YES → Agent (Part 2)
    │   └── NO
    │       ├── Is the execution order fixed?
    │       │   ├── YES → Workflow (Part 4)
    │       │   └── NO → RouterAgent (Step 3.5)
    │       └── Do you need concurrent execution?
    │           ├── YES → ParallelAgent or Workflow
    │           └── NO → SequentialAgent or Workflow
    └── Do you need conditional branching on state?
        ├── YES → Workflow with branch_if() (Step 4.3)
        └── NO → SequentialAgent (Step 3.1)
```

| Scenario | Recommended approach |
| --- | --- |
| Classify 1000 articles | `TaskExecutor` + batch processing (Part 1) |
| Research assistant with tools | Single `Agent` + `@tool` (Part 2) |
| Research → Write → Review | `SequentialAgent` (Step 3.1) or `Workflow` (Step 4.5) |
| Multi-perspective analysis | `ParallelAgent` (Step 3.2) |
| Iterative article improvement | `LoopAgent` (Step 3.4) |
| "Route to the right specialist" | `RouterAgent` (Step 3.5) |
| Conditional approve/revise | `Workflow` + `branch_if()` (Step 4.3) |
| Mix agents + pure functions | `Workflow` + `agent_node()` + `tasker_node()` (Part 5) |

---

## See Also

- [README_orchestration.md](../agent/README_orchestration.md) — Agent orchestration deep-dive
- [README_events_tracking.md](../agent/README_events_tracking.md) — Events and tracing guide
- [README_workflow.md](../workflows/README_workflow.md) — Workflow engine reference
- [README_task_configuration.md](../tasker/README_task_configuration.md) — JSON task file format
- [README_connector_genai.md](../connector/README_connector_genai.md) — Connector layer
- [README_cli.md](../cli/README_cli.md) — CLI arguments reference
- [README_config.md](../config/README_config.md) — Configuration reference
- [visualize](../visualize/) — ASCII visualization source
