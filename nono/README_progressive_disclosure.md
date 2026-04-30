# Progressive Disclosure — Implementation Reference

> How Nono implements the principles from Anthropic's ["Building effective agents"](https://www.anthropic.com/engineering/building-effective-agents) (2024).

---

## Table of Contents

- [Overview](#overview)
- [Principle 1 — Maintain Simplicity](#principle-1--maintain-simplicity)
  - [The Complexity Ladder](#the-complexity-ladder)
  - [Decision Wizard](#decision-wizard)
  - [Escalation and De-escalation](#escalation-and-de-escalation)
  - [Complexity Budget](#complexity-budget)
  - [Agent Complexity Map](#agent-complexity-map)
- [Principle 2 — Prioritize Transparency](#principle-2--prioritize-transparency)
  - [Immutable Events](#immutable-events)
  - [Real-time Streaming](#real-time-streaming)
  - [Lifecycle Hooks](#lifecycle-hooks)
  - [Structured Tracing](#structured-tracing)
  - [Workflow Audit Trail](#workflow-audit-trail)
- [Principle 3 — Craft Your ACI](#principle-3--craft-your-aci)
  - [Automatic Tool Validation](#automatic-tool-validation)
  - [Programmatic Validation API](#programmatic-validation-api)
  - [Validation Checks](#validation-checks)
  - [Writing Good Descriptions](#writing-good-descriptions)
- [Architecture Map](#architecture-map)
- [File Reference](#file-reference)

---

## Overview

Anthropic defines three core principles for building effective agents:

1. **Maintain simplicity** — start with the simplest solution, increase complexity only when needed.
2. **Prioritize transparency** — explicitly show the agent's planning steps.
3. **Carefully craft your ACI** — tool documentation deserves as much engineering as prompts.

Nono implements each principle through concrete code mechanisms, not just documentation. This document describes the implementation, how to use each mechanism, and where to find it in the codebase.

---

## Principle 1 — Maintain Simplicity

> *"Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short."*
> — Anthropic

### The Complexity Ladder

Nono defines six levels of orchestration complexity. The rule: **always start at Level 0 and move down only when the current level is demonstrably insufficient.**

| Level | Name | Pattern | Nono class |
|---|---|---|---|
| 0 | Single LLM call | One prompt, one response | `TaskExecutor.execute()` |
| 1 | Augmented LLM | LLM + tools | `Agent` + `@tool` |
| 2 | Simple workflow | Sequential pipeline | `Workflow` / `SequentialAgent` |
| 3 | Branching workflow | Conditional, parallel, loop | `branch_if()` / `ParallelAgent` / `LoopAgent` |
| 4 | LLM routing | Semantic decisions | `RouterAgent` / `transfer_to_agent` |
| 5 | Advanced | MCTS, evolutionary, sagas | `MonteCarloAgent` / `SagaAgent` / etc. |

Each level has a ready-to-run code snippet accessible via `get_recommendation(level).snippet`.

**Implementation:** `nono/wizard.py` → `_RECOMMENDATIONS` dict.

### Decision Wizard

An interactive tool that asks 5 yes/no questions and recommends the **minimum** level needed. Short-circuits as soon as the level is determined — users answering "yes" to question 1 never see questions 2–5.

**CLI:**

```bash
nono wizard            # interactive mode
nono wizard --json     # JSON output
```

**Programmatic:**

```python
from nono.wizard import recommend

# Pass answers directly
rec = recommend({
    "single_call_enough": False,
    "needs_tools": True,
    "needs_multiple_steps": False,
})
print(rec.level)       # 1
print(rec.pattern)     # "Agent + @tool / FunctionTool"
print(rec.snippet)     # Ready-to-run code
print(rec.next_if)     # "Escalate to Level 2 if..."
```

**Questions asked:**

| # | Key | Question |
|---|---|---|
| 1 | `single_call_enough` | Can a single, well-crafted LLM prompt solve your task? |
| 2 | `needs_tools` | Does the task require external tools, APIs, or data retrieval? |
| 3 | `needs_multiple_steps` | Does the task have distinct steps that should execute in sequence? |
| 4 | `needs_branching` | Do you need conditional logic, parallelism, or iteration? |
| 5 | `needs_semantic_routing` | Do routing decisions require the LLM to understand intent? |

**Implementation:** `nono/wizard.py` → `recommend()`, `recommend_interactive()`, `QUESTIONS`.
**CLI handler:** `nono/cli/cli.py` → `_handle_wizard()`.

### Escalation and De-escalation

Explicit guidance for moving up or down the Complexity Ladder:

```python
from nono.wizard import suggest_next, suggest_simpler, complexity_for_agent

# "I'm at Level 2 — what comes next?"
nxt = suggest_next(2)
print(nxt.level)       # 3
print(nxt.pattern)     # "branch_if() / parallel_step() / loop_step()"
print(nxt.next_if)     # When to escalate further

# "I might be over-engineering — what's simpler?"
sim = suggest_simpler(4)
print(sim.level)       # 3

# "What level is this agent class?"
complexity_for_agent("MonteCarloAgent")    # 5
complexity_for_agent("SequentialAgent")    # 2
complexity_for_agent("Agent")             # 1
```

**Implementation:** `nono/wizard.py` → `suggest_next()`, `suggest_simpler()`, `complexity_for_agent()`.

### Complexity Budget

Audits an agent tree and warns when cumulative complexity exceeds a configurable threshold. Each agent's score equals its Complexity Ladder level. A `SequentialAgent` (level 2) with 3 `Agent` sub-agents (level 1 each) scores 2 + 1 + 1 + 1 = 5.

```python
from nono.wizard import audit_agent_tree

report = audit_agent_tree(my_pipeline, max_score=10)
print(report.summary_table())
```

Output:

```
Complexity Budget: 12/10
Status: OVER BUDGET

Agent                          Type                         Level
-----------------------------------------------------------------
pipeline                       SequentialAgent                  2
  researcher                   LlmAgent                         1
  router                       RouterAgent                      4
    billing                    LlmAgent                         1
    support                    LlmAgent                         1
  writer                       LlmAgent                         1
  reviewer                     LlmAgent                         1
  publisher                    LlmAgent                         1

Suggestion: Total complexity 12 exceeds budget 10. The most complex
component is 'router' (RouterAgent, level 4). Consider replacing it
with a Level 3 pattern (branch_if() / parallel_step() / loop_step()).
```

The `audit_agent_tree()` convenience function also emits a `logging.WARNING` when over budget (disable with `warn=False`).

**Implementation:** `nono/wizard.py` → `ComplexityBudget`, `BudgetReport`, `BudgetEntry`, `audit_agent_tree()`.

### Agent Complexity Map

Every agent class in Nono is assigned a Complexity Ladder level. The full mapping:

| Level | Agents |
|---|---|
| **0** | `TaskExecutor` |
| **1** | `Agent`, `LlmAgent` |
| **2** | `SequentialAgent`, `EscalationAgent`, `BatchAgent`, `TimeoutAgent`, `CacheAgent`, `BudgetAgent` |
| **3** | `ParallelAgent`, `LoopAgent`, `MapReduceAgent`, `PriorityQueueAgent`, `LoadBalancerAgent`, `EnsembleAgent`, `ProducerReviewerAgent`, `GuardrailAgent`, `BestOfNAgent`, `CascadeAgent`, `ContextFilterAgent`, `SkeletonOfThoughtAgent`, `LeastToMostAgent`, `CheckpointableAgent` |
| **4** | `RouterAgent`, `HandoffAgent`, `GroupChatAgent`, `SupervisorAgent`, `HierarchicalAgent`, `VotingAgent`, `ConsensusAgent`, `DebateAgent`, `DynamicFanOutAgent`, `SwarmAgent`, `SubQuestionAgent`, `PlannerAgent`, `AdaptivePlannerAgent`, `ReflexionAgent`, `CoVeAgent`, `MemoryConsolidationAgent`, `SocraticAgent`, `MetaOrchestratorAgent`, `CurriculumAgent`, `ShadowAgent`, `HumanInputAgent` |
| **5** | `TreeOfThoughtsAgent`, `MonteCarloAgent`, `GraphOfThoughtsAgent`, `BlackboardAgent`, `MixtureOfExpertsAgent`, `SagaAgent`, `GeneticAlgorithmAgent`, `MultiArmedBanditAgent`, `SelfDiscoverAgent`, `SpeculativeAgent`, `CircuitBreakerAgent`, `TournamentAgent`, `CompilerAgent` |

**Implementation:** `nono/wizard.py` → `_AGENT_COMPLEXITY` dict.

---

## Principle 2 — Prioritize Transparency

> *"Prioritize transparency by explicitly showing the agent's planning steps."*
> — Anthropic

### Immutable Events

Every action during agent execution is recorded as an immutable `Event`:

```python
from nono.agent import EventType

# EventType enum:
# USER_MESSAGE, AGENT_MESSAGE, TOOL_CALL, TOOL_RESULT,
# STATE_UPDATE, AGENT_TRANSFER, HUMAN_INPUT_REQUEST,
# HUMAN_INPUT_RESPONSE, ERROR
```

Events are stored chronologically in `Session.events` and accessible after execution:

```python
runner = Runner(agent)
response = runner.run("Analyse this data")

for event in runner.session.events:
    print(f"[{event.event_type.value}] {event.author}: {event.content[:80]}")
```

**Implementation:** `nono/agent/base.py` → `Event`, `EventType`, `Session`.

### Real-time Streaming

`Runner.stream()` yields events as they are produced — before the full execution completes:

```python
for event in runner.stream("Process this request"):
    if event.event_type == EventType.TOOL_CALL:
        print(f"  → Calling tool: {event.content}")
    elif event.event_type == EventType.AGENT_MESSAGE:
        print(f"  ← Agent response: {event.content[:100]}")
```

**Implementation:** `nono/agent/runner.py` → `Runner.stream()`.

### Lifecycle Hooks

Orchestration agents (Sequential, Parallel, Loop, etc.) fire hooks at every stage. All use fluent API:

```python
pipeline = SequentialAgent(name="pipe", sub_agents=[a, b, c])

pipeline.on_start(lambda name, session:
    print(f"▶ {name} started"))

pipeline.on_end(lambda name, session, n:
    print(f"■ {name} finished ({n} agents ran)"))

pipeline.on_between_agents(lambda prev, nxt, session:
    print(f"  {prev} → {nxt}"))  # return False to halt

pipeline.on_agent_start(lambda name, session:
    print(f"  ▷ {name} starting"))

pipeline.on_agent_end(lambda name, session, err:
    print(f"  ▪ {name} done, error={err}"))
```

Additionally, every individual agent supports:

| Callback | Signature | Purpose |
|---|---|---|
| `before_agent` | `(agent, ctx) → Optional[str]` | Short-circuit before execution |
| `after_agent` | `(agent, ctx, response) → Optional[str]` | Transform response |
| `before_tool` | `(agent, tool_name, args) → Optional[dict]` | Inspect/modify tool args |
| `after_tool` | `(agent, tool_name, args, result) → Optional[Any]` | Inspect/modify tool result |

**Implementation:** `nono/agent/base.py` → `BaseAgent.on_start()`, `on_end()`, `on_between_agents()`, `on_agent_start()`, `on_agent_end()`.

### Structured Tracing

`TraceCollector` gathers detailed telemetry across all agents — LLM calls, token usage, tool executions, durations, and errors:

```python
from nono.tracing import TraceCollector

collector = TraceCollector()
runner = Runner(agent, trace_collector=collector)
runner.run("Hello")

for trace in collector.traces:
    print(f"{trace.agent_name}: {trace.status.value} "
          f"({trace.duration_ms:.0f}ms, {trace.total_tokens} tokens)")
    for tool_rec in trace.tool_records:
        print(f"  tool: {tool_rec.name} → {tool_rec.result[:50]}")
```

**Exports:** `TraceCollector`, `Trace`, `TraceStatus`, `LLMCall`, `TokenUsage`, `ToolRecord`.

**Implementation:** `nono/agent/tracing.py` (canonical), re-exported via `nono/tracing.py`.

### Workflow Audit Trail

Every step in a `Workflow` records a `StateTransition`:

```python
result = flow.run(topic="AI")
for t in flow.transitions:
    print(f"{t.step}: changed={t.keys_changed}, "
          f"branch={t.branch_taken}, "
          f"duration={t.duration_ms:.1f}ms, "
          f"retries={t.retries}, error={t.error}")
```

Fields: `step`, `keys_changed` (frozenset), `branch_taken`, `duration_ms`, `retries`, `error`.

**Implementation:** `nono/workflows/workflow.py` → `StateTransition`.

---

## Principle 3 — Craft Your ACI

> *"Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts."*
> — Anthropic, Appendix 2

### Automatic Tool Validation

`LlmAgent.__init__` calls `validate_tools()` on every user-supplied tool at construction time. No opt-in needed — it runs automatically:

```python
from nono.agent import Agent, FunctionTool

# This agent construction triggers 3 log messages:
bad = FunctionTool(lambda: None, name="x", description="")
agent = Agent(name="a", provider="google", tools=[bad])
# ERROR  Tool 'x': Missing description. The LLM relies on the description
#        to decide when and how to call this tool.
# WARNING Tool 'x': Tool name is too short.
# WARNING Tool 'x': Tool has no parameters.
```

The validation is **advisory** — it logs via Python's `logging` module but never raises or blocks construction.

**Implementation:** `nono/agent/llm_agent.py` → `LlmAgent.__init__`, line 228.

### Programmatic Validation API

For CI/CD, tests, or custom quality gates:

```python
from nono.agent import validate_tools, ToolIssue

issues = validate_tools(agent.tools, warn=False)
assert len(issues) == 0, f"Fix tool descriptions: {issues}"

# Custom minimum description length
issues = validate_tools(tools, min_description_len=20, warn=False)
```

**Implementation:** `nono/agent/tool.py` → `validate_tools()`, `ToolIssue`.

### Validation Checks

| Check | Severity | Trigger |
|---|---|---|
| Description is empty | `error` | `description=""` and no docstring |
| Description too short | `warning` | < 10 characters (configurable via `min_description_len`) |
| Tool name too short | `warning` | Single-character names like `"x"` |
| Missing parameter types | `warning` | Parameters without type annotations |
| Zero parameters | `warning` | Tool takes no arguments (suspicious but valid) |

### Writing Good Descriptions

Anthropic recommends four practices (Appendix 2 of "Building effective agents"):

**1. Put yourself in the model's shoes.**
Would a junior developer understand the tool from its description alone?

```python
# ❌ Bad — the LLM has to guess
@tool(description="Search.")
def search(q: str) -> str: ...

# ✅ Good — clear, specific, actionable
@tool(description=(
    "Search the product catalogue by keyword. Returns a JSON array of "
    "matching products with name, price, and stock count. Use when the "
    "user asks about product availability or pricing. Returns [] if no "
    "matches found."
))
def search_catalogue(query: str, max_results: int = 10) -> str: ...
```

**2. Include edge cases and boundaries.**
What happens on empty input? What's the return format?

**3. Poka-yoke your tools.**
Design parameters so it's hard to make mistakes — e.g., require absolute paths instead of relative ones.

**4. Test with real inputs.**
Run examples to see what mistakes the model makes, then iterate on the description.

**Docstring fallback:** When no explicit `description` is provided, `FunctionTool` automatically extracts the first line of the function's docstring:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city in Celsius."""
    ...
# description = "Get the current weather for a city in Celsius."
```

---

## Architecture Map

```
                   nono wizard (CLI)
                        │
                        ▼
               ┌─────────────────┐
               │  Decision Wizard │ ← recommend(), QUESTIONS
               │  (nono/wizard.py)│
               └────────┬────────┘
                        │ recommends Level 0–5
                        ▼
┌───────────────────────────────────────────────────────┐
│                  Complexity Ladder                      │
│                                                         │
│  Level 0 ─── TaskExecutor.execute()                    │
│  Level 1 ─── Agent + @tool                             │
│  Level 2 ─── Workflow / SequentialAgent                │
│  Level 3 ─── branch_if / ParallelAgent / LoopAgent    │
│  Level 4 ─── RouterAgent / transfer_to_agent           │
│  Level 5 ─── TreeOfThoughts / MonteCarlo / Saga       │
│                                                         │
│  suggest_next(level) ↑↓ suggest_simpler(level)         │
└───────────────────────────────────────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ComplexityBudget  │ ← audit_agent_tree()
               │  (walk tree,    │
               │   sum scores,   │
               │   warn if over) │
               └─────────────────┘

         ┌──────────────────────────────────┐
         │      Transparency Layer           │
         │                                    │
         │  Event (immutable records)         │
         │  Runner.stream() (real-time)       │
         │  on_start/end/between (hooks)      │
         │  TraceCollector (telemetry)        │
         │  StateTransition (audit trail)     │
         └──────────────────────────────────┘

         ┌──────────────────────────────────┐
         │      ACI Quality Layer            │
         │                                    │
         │  validate_tools() (auto + manual) │
         │  FunctionTool (schema generation) │
         │  ToolContext (auto-excluded)       │
         │  Docstring → description fallback │
         └──────────────────────────────────┘
```

---

## File Reference

| File | What it provides |
|---|---|
| `nono/wizard.py` | Decision Wizard, Complexity Ladder, Budget, agent complexity map |
| `nono/agent/tool.py` | `FunctionTool`, `@tool`, `validate_tools()`, `ToolIssue` |
| `nono/agent/llm_agent.py` | Auto-validation in `LlmAgent.__init__` |
| `nono/agent/base.py` | `Event`, `EventType`, `Session`, lifecycle hooks, callbacks |
| `nono/agent/runner.py` | `Runner.run()`, `Runner.stream()` |
| `nono/agent/tracing.py` | `TraceCollector`, `Trace`, `LLMCall`, `TokenUsage` |
| `nono/tracing.py` | Re-export of tracing classes for convenience |
| `nono/workflows/workflow.py` | `StateTransition`, audit trail |
| `nono/cli/cli.py` | `nono wizard` subcommand |
| `nono/agent/README_agent.md` | ACI Quality section (full reference) |
| `nono/README_guide.md` | Step 2.3b — ACI tutorial |
| `nono/workflows/README_orchestration_guide.md` | "Choosing the Right Level of Complexity" |

---

*Based on Anthropic's ["Building effective agents"](https://www.anthropic.com/engineering/building-effective-agents) (December 2024).*
