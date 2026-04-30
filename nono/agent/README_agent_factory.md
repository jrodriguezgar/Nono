# Dynamic Agent Factory — Generate Agents from Natural Language

> Create fully configured agents and multi-agent pipelines from a description.
> The factory generates system prompts, selects tools, chooses orchestration patterns, and builds the agents — all gated by security controls.

**Module**: [`agent/agent_factory.py`](agent_factory.py) · **Parent**: [Agent Framework](README_agent.md) · **Orchestration patterns**: [Orchestration Guide](README_orchestration.md)

---

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Single Agent Generation](#single-agent-generation)
  - [AgentBlueprint](#agentblueprint)
  - [SystemPromptGenerator](#systempromptgenerator)
  - [ToolSelector](#toolselector)
  - [AgentConfigurator](#agentconfigurator)
- [Orchestrated Multi-Agent Generation](#orchestrated-multi-agent-generation)
  - [OrchestrationSelector](#orchestrationselector)
  - [OrchestrationBlueprint](#orchestrationblueprint)
  - [Supported Patterns](#supported-patterns)
- [Security](#security)
  - [Config Flag](#config-flag)
  - [Prompt Injection Sanitisation](#prompt-injection-sanitisation)
  - [Tool Allowlist](#tool-allowlist)
  - [Provider Restrictions](#provider-restrictions)
- [Human-in-the-Loop Review](#human-in-the-loop-review)
- [Extending the Factory](#extending-the-factory)
- [API Reference](#api-reference)

---

## Overview

The `AgentFactory` dynamically generates agents from natural-language descriptions. It follows a modular pipeline where each step is handled by a replaceable component:

```
Description ──► SystemPromptGenerator ──► ToolSelector ──► AgentConfigurator ──► AgentBlueprint ──► LlmAgent
                                                │
                                    OrchestrationSelector
                                                │
                                    OrchestrationBlueprint ──► Workflow Agent
```

Two generation modes:

| Mode | Method | Output |
|------|--------|--------|
| **Single agent** | `generate_blueprint()` → `build()` | One `LlmAgent` |
| **Orchestrated** | `generate_orchestrated_blueprint()` → `build_orchestrated()` | Workflow agent + sub-agents |

---

## Quickstart

### Single agent

```python
from nono.agent import AgentFactory, create_agent_from_prompt
from nono.agent.tool import tool

@tool(description="Search the web for information.")
def web_search(query: str) -> str:
    return f"Results for: {query}"

# One-liner (requires agent.factory.allow_dynamic_creation = true)
agent = create_agent_from_prompt(
    "A research assistant that searches the web and summarises findings.",
    available_tools=[web_search],
)
```

### Orchestrated multi-agent

```python
factory = AgentFactory()

orch_bp = factory.generate_orchestrated_blueprint(
    "Research a topic, then write an article, then review it for quality.",
    available_tools=[web_search],
)

print(orch_bp.pattern)              # "sequential"
print(len(orch_bp.sub_agent_blueprints))  # 3

# Review before building
for bp in orch_bp.sub_agent_blueprints:
    print(f"  {bp.name}: {bp.description}")

# Build the full pipeline
agent = factory.build_orchestrated(orch_bp, available_tools=[web_search])
```

### Blueprint review before build

```python
factory = AgentFactory()
blueprint = factory.generate_blueprint(
    "An agent that analyses CSV data.",
    available_tools=[csv_reader],
)

# Inspect everything the LLM decided
print(blueprint.to_dict())

# Only build if satisfied
agent = factory.build(blueprint, available_tools=[csv_reader])
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentFactory                             │
│                                                                 │
│  ┌──────────────────────┐   ┌─────────────────┐                │
│  │ SystemPromptGenerator│   │  ToolSelector    │                │
│  │ (LLM meta-prompt)    │   │ (LLM or keyword) │                │
│  └──────────┬───────────┘   └────────┬────────┘                │
│             │                        │                          │
│             ▼                        ▼                          │
│  ┌──────────────────────────────────────────────┐               │
│  │           AgentConfigurator                   │               │
│  │  (sanitise · allowlist · validate · assemble) │               │
│  └──────────────────────┬───────────────────────┘               │
│                         │                                       │
│              ┌──────────┴──────────┐                            │
│              ▼                     ▼                             │
│     AgentBlueprint      OrchestrationBlueprint                  │
│     (single agent)      (pattern + sub-blueprints)              │
│              │                     │                            │
│              ▼                     ▼                             │
│          LlmAgent           Workflow Agent                      │
│                          (Sequential, Planner, …)               │
│                                                                 │
│  ┌──────────────────────────┐                                   │
│  │  OrchestrationSelector   │ ◄── analyse task ──► pattern      │
│  │  (LLM or keyword heuristic)│                                  │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility | Replaceable |
|-----------|---------------|:-----------:|
| `SystemPromptGenerator` | Converts a description into a structured system prompt via LLM | ✓ |
| `ToolSelector` | Selects tools from a pool (LLM-assisted or keyword fallback) | ✓ |
| `OrchestrationSelector` | Recommends orchestration pattern + sub-agent decomposition | ✓ |
| `AgentConfigurator` | Validates names, sanitises instructions, enforces allowlists | ✓ |

Pass custom instances to the `AgentFactory` constructor to replace any component.

---

## Configuration

Add to `config.toml`:

```toml
[agent.factory]
# SECURITY: Must be explicitly enabled
allow_dynamic_creation = false

# Maximum tools a generated agent can use
max_tools_per_agent = 10

# Maximum system prompt length (characters)
max_instruction_length = 4000

# Default provider/model for the factory's own LLM calls
default_provider = "google"
# default_model = ""

# Restrict providers generated agents may use (empty = all)
# allowed_providers = ["google", "openai"]

# Restrict tools generated agents may use (empty = all)
# tool_allowlist = ["web_search", "summarise_text"]
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allow_dynamic_creation` | `bool` | `false` | Master switch — must be `true` to use the factory |
| `max_tools_per_agent` | `int` | `10` | Cap on tools per generated agent |
| `max_instruction_length` | `int` | `4000` | Max system prompt characters |
| `default_provider` | `str` | `"google"` | Provider for factory LLM calls |
| `default_model` | `str` | `""` | Model for factory LLM calls (empty = provider default) |
| `allowed_providers` | `list` | `[]` | Whitelist of providers for generated agents |
| `tool_allowlist` | `list` | `[]` | Whitelist of tool names for generated agents |

---

## Single Agent Generation

### AgentBlueprint

Immutable, frozen dataclass produced by `generate_blueprint()`. Inspect before calling `build()`.

```python
@dataclass(frozen=True)
class AgentBlueprint:
    name: str                        # snake_case agent name
    description: str                 # human-readable purpose
    instruction: str                 # generated system prompt
    provider: str = "google"         # LLM provider
    model: str | None = None         # model name
    temperature: float = 0.7
    tool_names: tuple[str, ...] = () # selected tool names
    output_format: str = "text"      # "text" or "json"
    metadata: dict[str, Any] = {}
```

Serialisation:

```python
data = blueprint.to_dict()           # → dict
restored = AgentBlueprint.from_dict(data)  # → AgentBlueprint
```

### SystemPromptGenerator

Uses an LLM meta-prompt to convert a description into a well-structured system prompt.

```python
gen = SystemPromptGenerator(provider="google")
prompt = gen.generate("A data analyst that processes CSVs")
# → "You are an expert data analyst. You process CSV files..."
```

Rules enforced by the meta-prompt:
- Writes in second person ("You are…")
- Keeps under 2000 characters
- Does not include tool-calling instructions (framework handles those)

### ToolSelector

Selects relevant tools from a pool. Two modes:

| Mode | When | How |
|------|------|-----|
| **LLM-assisted** | `use_llm=True` (default) | LLM reads tool descriptions and picks the best matches |
| **Keyword** | `use_llm=False` | Matches tool names/descriptions against the agent description |

```python
sel = ToolSelector(max_tools=5)
names = sel.select(
    "Search the web and summarise results",
    available_tools=[web_search, summarise, dangerous_tool],
    use_llm=False,  # keyword fallback
)
# → ["web_search", "summarise_text"]
```

### AgentConfigurator

Final validation and assembly step. Applies:

1. **Name sanitisation** — lowercase, alphanumeric + underscores only
2. **Instruction sanitisation** — injection pattern detection, length limit
3. **Provider restriction** — reject if not in `allowed_providers`
4. **Tool allowlist** — remove tools not in `tool_allowlist`
5. **Tool count cap** — truncate if exceeding `max_tools_per_agent`

---

## Orchestrated Multi-Agent Generation

### OrchestrationSelector

Analyses a task description and recommends:
1. **Which pattern** from `OrchestrationRegistry` fits best
2. **How to decompose** the task into sub-agent roles (LLM mode only)
3. **Pattern kwargs** (e.g. `max_iterations`, `max_steps`)

Two modes:

| Mode | Trigger words example | Output |
|------|----------------------|--------|
| **LLM** (`use_llm=True`) | Any description | Full JSON with pattern, sub-agents, kwargs |
| **Keyword** (`use_llm=False`) | "first…then…finally" → `sequential` | Pattern only (no sub-agent decomposition) |

Keyword heuristics:

| Pattern | Trigger words |
|---------|--------------|
| `sequential` | then, after, followed by, step by step, first, next, finally |
| `parallel` | simultaneously, concurrently, at the same time, in parallel |
| `loop` | iterate, repeat, until, refine, improve iteratively |
| `planner` | plan, decompose, break down, complex task, dependencies |
| `producer_reviewer` | review, critique, revise, draft and review |
| `debate` | debate, argue, pros and cons, adversarial, opposing views |
| `map_reduce` | each item, for every, aggregate, map and reduce |
| `supervisor` | supervise, delegate, evaluate, manager, oversee |
| `router` | route, classify first, depending on, if it's about |
| `escalation` | try first, fallback, escalate, if fails |
| `hierarchical` | departments, hierarchy, multiple teams, levels |

### OrchestrationBlueprint

Extends the blueprint concept to multi-agent pipelines:

```python
@dataclass(frozen=True)
class OrchestrationBlueprint:
    name: str                                    # orchestrator name
    description: str                             # pipeline purpose
    pattern: str                                 # "sequential", "planner", etc.
    sub_agent_blueprints: tuple[AgentBlueprint, ...]  # sub-agent specs
    pattern_kwargs: dict[str, Any] = {}          # e.g. max_iterations
    provider: str = "google"
    model: str | None = None
    metadata: dict[str, Any] = {}                # includes "reasoning"
```

### Supported Patterns

| Pattern | Workflow Agent | Min Sub-agents | Description |
|---------|---------------|:--------------:|-------------|
| `none` | `LlmAgent` | 1 | Single agent, no orchestration |
| `sequential` | `SequentialAgent` | 1 | Run sub-agents one after another |
| `parallel` | `ParallelAgent` | 1 | Run sub-agents concurrently |
| `loop` | `LoopAgent` | 1 | Repeat until condition or max iterations |
| `router` | `RouterAgent` | 1 | LLM dynamically picks the best sub-agent |
| `planner` | `PlannerAgent` | 1 | LLM decomposes into dependency-aware steps |
| `map_reduce` | `MapReduceAgent` | 2 | Mappers + 1 reducer (last sub-agent) |
| `supervisor` | `SupervisorAgent` | 1 | LLM supervisor delegates and evaluates |
| `producer_reviewer` | `ProducerReviewerAgent` | 2 | Producer + reviewer loop |
| `debate` | `DebateAgent` | 3 | 2 debaters + 1 judge |
| `pipeline_parallel` | `PipelineParallelAgent` | 1 | Assembly-line stages for item lists |
| `dynamic_fan_out` | `DynamicFanOutAgent` | 2 | Worker + reducer, LLM determines N items |
| `hierarchical` | `HierarchicalAgent` | 1 | Multi-level tree with LLM manager |
| `guardrail` | `GuardrailAgent` | 1 | Pre/post validation wrapper |
| `escalation` | `EscalationAgent` | 1 | Try in order, stop at first success |

---

## Security

### Config Flag

The factory is **disabled by default**. All methods raise `DynamicCreationDisabledError` unless:

```toml
[agent.factory]
allow_dynamic_creation = true
```

### Prompt Injection Sanitisation

All generated instructions pass through `sanitise_instruction()` which blocks:

| Pattern | Example blocked |
|---------|----------------|
| Role hijacking | "ignore all previous instructions" |
| Identity override | "you are now a hacker assistant" |
| System tag injection | `<\|system\|>`, `system:` |
| Rule bypass | "do not follow your rules" |
| Safety override | "override all safety guardrails" |
| Prompt leak | "reveal your system prompt" |

Detection raises `BlueprintValidationError` immediately — no partial results.

### Tool Allowlist

When `tool_allowlist` is configured, only listed tools can be assigned to generated agents. Unlisted tools are silently removed with a warning log.

```toml
[agent.factory]
tool_allowlist = ["web_search", "summarise_text", "csv_reader"]
```

### Provider Restrictions

When `allowed_providers` is configured, generated agents can only use listed providers.

```toml
[agent.factory]
allowed_providers = ["google", "openai"]
```

Attempting to generate an agent with `provider="ollama"` raises `BlueprintValidationError`.

---

## Human-in-the-Loop Review

### Blueprint inspection

```python
factory = AgentFactory()
blueprint = factory.generate_blueprint("A data analyst agent.")

# Review everything
print(blueprint.name)           # "a_data_analyst"
print(blueprint.instruction)    # "You are an expert data analyst..."
print(blueprint.tool_names)     # ("csv_reader",)
print(blueprint.provider)       # "google"

# Only build if satisfied
agent = factory.build(blueprint, available_tools=[csv_reader])
```

### Review callback

```python
agent = create_agent_from_prompt(
    "A research agent.",
    available_tools=[web_search],
    review_callback=lambda bp: input(f"Create '{bp.name}'? (y/n) ") == "y",
)
```

If the callback returns `False`, `BlueprintValidationError` is raised and no agent is created.

### Orchestration review

```python
orch_bp = factory.generate_orchestrated_blueprint(
    "Research, write, and review an article.",
)

print(f"Pattern: {orch_bp.pattern}")
print(f"Reasoning: {orch_bp.metadata.get('reasoning')}")
for i, bp in enumerate(orch_bp.sub_agent_blueprints):
    print(f"  [{i}] {bp.name}: {bp.instruction[:60]}...")

# Save for later review
import json
json.dump(orch_bp.to_dict(), open("pipeline.json", "w"), indent=2)

# Load and build later
data = json.load(open("pipeline.json"))
restored = OrchestrationBlueprint.from_dict(data)
agent = factory.build_orchestrated(restored, available_tools=[...])
```

---

## Extending the Factory

### Custom Components

Each pipeline component can be replaced by passing a custom instance:

```python
class MyPromptGenerator(SystemPromptGenerator):
    """Custom prompt generator with domain-specific rules."""

    def generate(self, description: str) -> str:
        base = super().generate(description)
        return base + "\n\nAlways respond in JSON format."


class MyToolSelector(ToolSelector):
    """Always include the audit_log tool."""

    def select(self, description, available_tools, *, use_llm=True):
        names = super().select(description, available_tools, use_llm=use_llm)
        if "audit_log" not in names:
            names.append("audit_log")
        return names


factory = AgentFactory(
    prompt_generator=MyPromptGenerator(),
    tool_selector=MyToolSelector(),
    orchestration_selector=OrchestrationSelector(provider="openai"),
)
```

### Custom Orchestration Patterns

The `OrchestrationRegistry` is the single source of truth for available patterns. All 15 built-in patterns are registered at module load. Third-party code can register additional patterns at any time:

```python
from nono.agent import register_pattern

def my_factory(*, name, description, sub_agents, provider, model, pattern_kwargs):
    """Custom factory — receives standard kwargs, returns a BaseAgent."""
    return MyCustomAgent(sub_agents=sub_agents, name=name, ...)

register_pattern(
    key="my_custom",
    class_name="MyCustomAgent",
    description="My domain-specific orchestration",
    keyword_hints=["custom", "special workflow"],
    factory=my_factory,
    min_sub_agents=2,
)
```

Once registered, the new pattern is available to `OrchestrationSelector` (both LLM and keyword modes) and `build_orchestrated()`.

`PatternRegistration` is the immutable dataclass stored per pattern:

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Unique snake_case pattern key |
| `class_name` | `str` | Workflow-agent class name (display) |
| `description` | `str` | Human-readable description |
| `keyword_hints` | `tuple[str, ...]` | Trigger words for heuristic selection |
| `factory` | `OrchestrationFactory \| None` | Callable to instantiate the pattern |
| `min_sub_agents` | `int` | Minimum required sub-agents |

`OrchestrationRegistry` class methods:

| Method | Description |
|--------|-------------|
| `register(key, class_name, description, *, ...)` | Add or overwrite a pattern |
| `unregister(key)` | Remove a pattern |
| `get(key) → PatternRegistration` | Retrieve a registration |
| `contains(key) → bool` | Check if a key exists |
| `catalog() → dict` | Legacy format `{key: (class_name, desc)}` |
| `keyword_hints() → dict` | All patterns with keyword hints |
| `list_patterns() → list[str]` | Sorted list of registered keys |
```

---

## API Reference

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_agent_from_prompt` | `(description, *, available_tools, name, provider, model, temperature, output_format, review_callback, config) → BaseAgent` | One-liner: generate and build in one call |
| `sanitise_instruction` | `(text, *, max_length) → str` | Validate and clean instruction text |
| `register_pattern` | `(key, class_name, description, *, keyword_hints, factory, min_sub_agents) → None` | Register a custom orchestration pattern |

### Classes

| Class | Description |
|-------|-------------|
| `AgentFactory` | Main factory — orchestrates the full pipeline |
| `AgentBlueprint` | Immutable single-agent specification |
| `OrchestrationBlueprint` | Immutable multi-agent pipeline specification |
| `SystemPromptGenerator` | LLM-powered system prompt generation |
| `ToolSelector` | LLM or keyword tool selection |
| `OrchestrationSelector` | LLM or keyword orchestration pattern selection |
| `AgentConfigurator` | Security validation and blueprint assembly |
| `OrchestrationRegistry` | Extensible pattern registry (class methods) |
| `PatternRegistration` | Immutable specification for a registered pattern |

### Type Aliases

| Alias | Signature | Description |
|-------|-----------|-------------|
| `OrchestrationFactory` | `Callable[..., BaseAgent]` | `(*, name, description, sub_agents, provider, model, pattern_kwargs) → BaseAgent` |

### Exceptions

| Exception | When |
|-----------|------|
| `DynamicCreationDisabledError` | `allow_dynamic_creation = false` in config |
| `BlueprintValidationError` | Injection detected, disallowed provider, review rejected |

### Constants

| Constant | Type | Description |
|----------|------|-------------|
| `OrchestrationRegistry.catalog()` | `dict[str, tuple[str, str]]` | Pattern → (class name, description) live mapping |

---

*Back to [Agent Framework](README_agent.md) · [Orchestration Guide](README_orchestration.md) · [Main README](../../README.md)*
