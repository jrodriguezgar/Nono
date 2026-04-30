# Tasker vs Agent — When to Use Each

> A practical guide to choosing between `TaskExecutor` and `Agent` in Nono.

## Table of Contents

- [Overview](#overview)
- [How to Execute](#how-to-execute)
- [At a Glance](#at-a-glance)
- [TaskExecutor (Tasker)](#taskexecutor-tasker)
- [Agent (LlmAgent)](#agent-llmagent)
- [Key Differences](#key-differences)
- [Decision Flowchart](#decision-flowchart)
- [Combining Both](#combining-both)
- [Migration Path](#migration-path)
- [Examples Side by Side](#examples-side-by-side)

---

## How to Execute

### Execute a TaskExecutor

```python
from nono.tasker import TaskExecutor

# 1. Create the executor (provider + optional model)
executor = TaskExecutor(provider="google")

# 2a. Direct prompt — pass input inline
result = executor.execute("Summarise this article: {data_input_json}", "Long text here...")
print(result)

# 2b. Direct prompt with named placeholders
result = executor.execute(
    "Translate from {source_lang} to {target_lang}: {data_input_json}",
    "Hello, how are you?",
    source_lang="English",
    target_lang="French",
)
print(result)

# 2c. Structured output with schema validation
result = executor.execute(
    "Extract name and age from: {data_input_json}",
    "Alice is 30 years old.",
    output_schema={"name": "string", "age": "integer"},
)
print(result)  # '{"name": "Alice", "age": 30}'

# 3. JSON task file — declarative, with auto-batching
result = executor.run_json_task(
    "nono/tasker/prompts/name_classifier.json",
    data_input=["Alice Smith", "IBM Corp", "Tokyo"],
)
print(result)

# 4. Override provider/model at runtime
result = executor.execute(
    "Explain quantum computing",
    config_overrides={"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
)
print(result)

# 5. With tracing
from nono.agent.tracing import TraceCollector

collector = TraceCollector()
result = executor.execute("Summarise: {data_input_json}", "Text...", trace_collector=collector)
for trace in collector.export():
    print(f"  {trace['provider']} | {trace['duration_ms']}ms")
```

### Execute an Agent

```python
from nono.agent import Agent, Runner

# 1. Create the agent
agent = Agent(
    name="assistant",
    provider="google",
    instruction="You are a helpful assistant.",
)

# 2. Create a Runner (wraps the agent with session management)
runner = Runner(agent=agent)

# 3a. Synchronous execution — returns a string
result = runner.run("What is the capital of France?")
print(result)  # "The capital of France is Paris."

# 3b. Multi-turn conversation — session remembers previous messages
result = runner.run("And what about Germany?")
print(result)  # "The capital of Germany is Berlin."

# 3c. Streaming — yields Event objects in real time
for event in runner.stream("Explain quantum computing"):
    print(f"[{event.event_type.value}] {event.content}")

# 3d. Async execution
import asyncio
result = asyncio.run(runner.run_async("Hello"))
```

### Execute an Agent with Tools

```python
from nono.agent import Agent, Runner, tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # simplified — use safe eval in production

@tool
def lookup(query: str) -> str:
    """Look up information."""
    return f"Result for '{query}': relevant data here."

agent = Agent(
    name="research_assistant",
    provider="google",
    instruction="You are a research assistant. Use tools to answer questions.",
    tools=[calculate, lookup],
)

runner = Runner(agent=agent)
result = runner.run("What is 15% of 2340?")
print(result)  # Agent calls calculate("2340 * 0.15") → "351.0"
```

### Execute a Multi-Agent Pipeline

```python
from nono.agent import Agent, Runner, SequentialAgent

researcher = Agent(
    name="researcher",
    provider="google",
    instruction="Research the topic and provide key facts.",
)

writer = Agent(
    name="writer",
    provider="google",
    instruction="Write a concise article based on the research provided.",
)

reviewer = Agent(
    name="reviewer",
    provider="google",
    instruction="Review the article. Suggest improvements.",
)

pipeline = SequentialAgent(
    name="content_pipeline",
    sub_agents=[researcher, writer, reviewer],
)

runner = Runner(agent=pipeline)
result = runner.run("Write about renewable energy trends in 2026")
print(result)
```

### Execute an Agent with Tracing

```python
from nono.agent import Agent, Runner
from nono.agent.tracing import TraceCollector

collector = TraceCollector()

agent = Agent(name="analyst", provider="google", instruction="Analyse data.")
runner = Runner(agent=agent, trace_collector=collector)

result = runner.run("Analyse: sales grew 15% in Q1")

# Inspect traces
for trace in collector.export():
    print(f"Agent: {trace['agent_name']}")
    print(f"Duration: {trace['duration_ms']}ms")
    for call in trace.get("llm_calls", []):
        print(f"  LLM: {call['provider']}/{call['model']} ({call['duration_ms']}ms)")
```

---

## Overview

Nono provides two ways to interact with LLMs:

| Layer | Class | Analogy |
|-------|-------|---------|
| **Tasker** | `TaskExecutor` | A function call — one prompt in, one result out |
| **Agent** | `LlmAgent` | A conversation — multi-turn, tools, memory, delegation |

Both share the same **Connector** layer underneath (`connector_genai`), so they support the same providers and models. The difference is in **how** they use the LLM.

```
┌──────────────────────────────────────┐
│            Your Application          │
│                                      │
│   TaskExecutor        Agent/Runner   │
│   (one-shot)          (multi-turn)   │
│        ↓                   ↓         │
│   ┌──────────────────────────────┐   │
│   │   Connector (connector_genai)│   │
│   │   Google · OpenAI · Groq … │   │
│   └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

---

## At a Glance

| Aspect | TaskExecutor | Agent |
|--------|-------------|-------|
| **Interaction** | Single call (fire-and-forget) | Multi-turn conversation |
| **State** | Stateless | Session with memory and state dict |
| **Events** | No | 7 event types (message, tool, transfer, error…) |
| **Prompts** | Jinja2 templates + JSON task files | `instruction` string |
| **Tools** | No tool calling | `FunctionTool` / `@tool` with auto-dispatch loop |
| **Output validation** | `output_schema` + `jsonschema` | `output_format` (text/json) |
| **Input validation** | `input_schema` + `jsonschema` | No |
| **Batching** | Built-in auto-batching + merge | Not built-in |
| **Rate limiting** | Built-in `RateLimiter` (token bucket) | Not built-in |
| **Multi-agent** | No | `sub_agents`, `transfer_to_agent`, orchestration |
| **Streaming** | No | `Runner.stream()` / `astream()` |
| **Async** | No | Full async (`run_async`, `astream`) |
| **Callbacks** | No | 4 hooks (before/after agent/tool) |
| **Shared content** | No | `SharedContent` (shared + local scopes) |
| **Configuration** | JSON files in `prompts/` | Python code |
| **Providers** | 8 providers | 14 providers |
| **Temperature** | String presets + float | String presets + float |
| **Tracing** | `trace_collector` parameter | `trace_collector` in Runner |
| **Visualization** | No | `draw()` ASCII agent tree |
| **Orchestration** | No | Sequential, Parallel, Loop, MapReduce, Consensus, ProducerReviewer, Debate, Escalation, Supervisor, Voting, Handoff, GroupChat, Hierarchical, Guardrail, BestOfN, Batch, Cascade, TreeOfThoughts, Planner, SubQuestion, ContextFilter, Reflexion, Speculative, CircuitBreaker, Tournament, Shadow, Compiler, Checkpointable, DynamicFanOut, Swarm, MemoryConsolidation, PriorityQueue, MonteCarlo, GraphOfThoughts, Blackboard, MixtureOfExperts, CoVe, Saga, LoadBalancer, Ensemble, Timeout, AdaptivePlanner, Router agents |
| **Typical use** | Data processing, classification, extraction | Chatbots, reasoning, tool-use, orchestration |

---

## TaskExecutor (Tasker)

### What it does

`TaskExecutor` sends a single prompt to an LLM and returns the result. It is **stateless** — each call is independent with no conversation history.

### Constructor

```python
TaskExecutor(
    provider: str,             # "google", "openai", "groq", …
    model: str = ...,          # Model name (provider default if omitted)
    api_key: str | None = None,# Auto-resolved via connector if omitted
    temperature: float | str = 0.7,  # Float or preset name
    max_tokens: int = 2048,    # Maximum output tokens
)
```

### Methods

| Method | Description |
|--------|-------------|
| `execute(input_data, output_schema=None, config_overrides=None, trace_collector=None)` | Send prompt, get response |
| `run_json_task(task_file, data_input=None, **data_inputs)` | Execute a JSON-defined task with auto-batching and validation |

### Features

**Providers** — 8 providers: Google, OpenAI, Perplexity, DeepSeek, xAI, Groq, OpenRouter, Ollama. Runtime switching via `config_overrides["provider"]`. Auto-resolves API keys from keyring or CSV.

**Jinja2 Templates** — Prompt templates (`*.j2`) separate logic from code. Token-aware batching via `TaskPromptBuilder` (`jinjapromptpy`). Available templates: `planner.j2`, `decompose_tasks.j2`, `python_programming.j2`, `conditional_flow.j2`, `data_loss_prevention.j2`, `spell_correction.j2`, `semantic_lookup.j2`, `logical_ordering.j2`, `name_classifier.j2`.

**JSON Task Files** — Declarative task definitions in `prompts/*.json`. Sections: `task` (metadata), `genai` (AI config), `prompts` (system/user/assistant), `input_schema`, `output_schema`. Change behaviour without modifying code.

**Structured Output** — `output_schema` triggers JSON response format with constrained decoding. `input_schema` validates input with `jsonschema`.

**Auto-Batching** — `run_json_task()` splits large data into token-limited batches. Results merged automatically (supports JSON, CSV, Markdown table, XML, plain text).

**Temperature Presets** — Named presets: `coding`=0.0, `math`=0.0, `data_cleaning`=0.1, `data_analysis`=0.3, `translation`=0.3, `balanced`=0.5, `conversation`=0.7, `creative`=1.0, `poetry`=1.2.

**Rate Limiting** — Built-in `RateLimiter` with token bucket algorithm. Controls request frequency per provider.

**Runtime Config Overrides** — `config_overrides` dict: `temperature`, `max_tokens`, `model`, `provider`, `api_key`, `response_format`. Extra kwargs forwarded to the LLM call.

**Placeholder System** — `{data_input_json}` (positional arg), `{custom_name}` (keyword args). Auto-serialises lists/dicts to JSON, strings passed directly.

**Tracing** — `trace_collector` parameter records `LLMCall` with provider, model, token estimates, timing.

**Logging** — `@event_log` decorator logs start/complete/error with timestamps. `msg_log()` utility for custom log entries.

### What it does NOT have

- No conversation history or session.
- No tool calling.
- No streaming.
- No async execution.
- No sub-agent delegation or orchestration.
- No event system.
- No callbacks.

### When to use

- Classify, extract, or transform data in bulk.
- Run reproducible tasks defined as configuration files.
- Process batches of inputs with rate limiting.
- Generate structured JSON output with schema validation.
- Tasks that don't need conversation history or tool calling.

### Minimal example

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor(provider="google")

# Direct prompt
result = executor.execute("Summarise this: {text}", text="Long article...")

# JSON task file
result = executor.run_json_task("prompts/name_classifier.json", data=names_list)
```

---

## Agent (LlmAgent)

### What it does

`Agent` (alias for `LlmAgent`) maintains a **conversation session** with memory, can call **tools**, delegate to **sub-agents**, emit **events**, and participate in **multi-agent orchestration**.

### Constructor

```python
Agent(
    name: str,                              # Unique agent name
    model: str | None = None,               # Model (config default if None)
    provider: str = "google",               # AI provider
    instruction: str = "You are a helpful assistant.",
    description: str = "",                  # Used by routers/delegation
    tools: list[FunctionTool] | None = None,
    api_key: str | None = None,
    temperature: float | str = 0.7,
    max_tokens: int | None = None,
    output_format: str = "text",            # "text" or "json"
    sub_agents: list[BaseAgent] | None = None,
    before_agent_callback = None,           # Pre-execution hook
    after_agent_callback = None,            # Post-execution hook
    before_tool_callback = None,            # Pre-tool hook
    after_tool_callback = None,             # Post-tool hook
    **service_kwargs,                       # Extra connector kwargs
)
```

### Execution via Runner

Agents are executed through a `Runner`:

```python
Runner(agent, session=None, trace_collector=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `run(message, **state)` | `str` | Sync single-turn, returns final response |
| `run_async(message, **state)` | `str` | Async single-turn |
| `stream(message, **state)` | `Iterator[Event]` | Sync streaming — yields events as produced |
| `astream(message, **state)` | `AsyncIterator[Event]` | Async streaming |
| `reset()` | — | Discard session, start fresh |
| `history` | `list[Event]` | All session events |

### Features

**Providers** — 14 providers: Google, OpenAI, Perplexity, DeepSeek, xAI, Groq, Cerebras, NVIDIA, GitHub Models, OpenRouter, Azure AI, Vercel, Ollama. Lazy service initialization on first call.

**Event System** — Every action emits an immutable `Event` with type, author, content, data, timestamp, and ID.

| Event Type | Emitted when |
|------------|-------------|
| `USER_MESSAGE` | User sends a message |
| `AGENT_MESSAGE` | Agent produces a response |
| `TOOL_CALL` | Agent invokes a tool |
| `TOOL_RESULT` | Tool returns a result |
| `STATE_UPDATE` | Session state changes (e.g. loop iteration) |
| `AGENT_TRANSFER` | Agent delegates to a sub-agent |
| `ERROR` | An error occurs |

**Session & State** — `Session` tracks the full conversation history as events, holds a mutable `state` dict, and records creation time.

**Shared Content** — Two scopes for inter-agent data exchange:

| Scope | Access | Description |
|-------|--------|-------------|
| `session.shared_content` | All agents | Session-wide shared store |
| `agent.local_content` | Single agent | Private content per agent |

Methods: `save()`, `load()`, `names()`, `delete()`, `clear()`. Each item has `name`, `data`, `content_type`, `metadata`, `created_by`, `created_at`.

**Tool Calling** — Iterative tool-calling loop (up to 10 rounds): LLM → extract tool calls → execute → feed results back → LLM. Automatic `ToolContext` injection with `state`, `shared_content`, `local_content`, `agent_name`, `session_id`. Context helpers: `save_content()` / `load_content()` with `scope="shared"|"local"`.

**`@tool` Decorator & `FunctionTool`** — Auto-generates JSON Schema from type hints. Excludes `ToolContext` from schema. Supports `name`, `description` overrides.

**Sub-Agent Delegation** — When `sub_agents` is set, an auto-generated `transfer_to_agent` tool is registered. The LLM sees agent names + descriptions and can delegate at any point. Emits `AGENT_TRANSFER` and `TOOL_RESULT` events.

**Lifecycle Callbacks** — 4 hooks for intercepting the execution pipeline:

| Callback | Signature | Can |
|----------|-----------|-----|
| `before_agent_callback` | `(agent, ctx) → Optional[str]` | Short-circuit with a return value |
| `after_agent_callback` | `(agent, ctx, response) → Optional[str]` | Transform the final response |
| `before_tool_callback` | `(agent, tool_name, args) → Optional[dict]` | Modify tool arguments |
| `after_tool_callback` | `(agent, tool_name, args, result) → Optional[Any]` | Modify tool result |

**Orchestration Agents** — Composite agents that coordinate sub-agents:

| Type | Pattern | Key params |
|------|---------|-----------|
| `SequentialAgent` | Run sub-agents in order, output flows to next | — |
| `ParallelAgent` | Run sub-agents concurrently | `max_workers`, `message_map`, `result_key` |
| `LoopAgent` | Repeat sub-agents until condition met | `max_iterations`, `stop_condition` |
| `MapReduceAgent` | Fan-out to mappers, then reduce into one result | `reduce_agent`, `max_workers`, `result_key` |
| `ConsensusAgent` | Multiple agents vote, judge synthesises consensus | `judge_agent`, `max_workers`, `result_key` |
| `ProducerReviewerAgent` | Iterative produce-then-review until approval | `producer`, `reviewer`, `approval_keyword` |
| `DebateAgent` | Adversarial debate — two agents argue, judge renders verdict | `agent_a`, `agent_b`, `judge`, `max_rounds`, `resolution_keyword` |
| `EscalationAgent` | Try agents in order, escalate on failure | `sub_agents`, `failure_keyword`, `on_escalation` |
| `SupervisorAgent` | LLM supervisor that delegates, evaluates, re-delegates | `sub_agents`, `model`, `provider`, `supervisor_instruction`, `max_iterations` |
| `VotingAgent` | Majority-vote — N agents answer, most frequent wins | `sub_agents`, `max_workers`, `normalize`, `result_key` |
| `HandoffAgent` | Peer-to-peer handoff — agents transfer full control to each other | `entry_agent`, `handoff_rules`, `max_handoffs`, `handoff_keyword` |
| `GroupChatAgent` | N-agent group chat with manager-controlled speaker selection | `sub_agents`, `speaker_selection`, `max_rounds`, `termination_keyword` |
| `HierarchicalAgent` | Multi-level tree orchestration — LLM manager delegates to departments | `sub_agents`, `model`, `provider`, `manager_instruction`, `max_iterations`, `synthesis_prompt`, `result_key` |
| `GuardrailAgent` | Pre/post validation pipeline with automatic retry on rejection | `main_agent`, `pre_validator`, `post_validator`, `rejection_keyword`, `max_retries`, `result_key` |
| `BestOfNAgent` | Best-of-N sampling — runs same agent N times, picks best by score | `agent`, `n`, `score_fn`, `max_workers`, `result_key` |
| `BatchAgent` | Batch processing — processes item list through one agent with concurrency | `agent`, `items`, `items_key`, `template`, `max_workers`, `result_key` |
| `CascadeAgent` | Progressive cascade — sequential stages with quality threshold gate | `sub_agents`, `score_fn`, `threshold`, `result_key` |
| `RouterAgent` | LLM decides mode + agents at runtime | `routing_instruction`, `temperature` |
| `SpeculativeAgent` | Speculative execution — race multiple agents, cancel losers early | `sub_agents`, `evaluator_fn`, `min_confidence`, `result_key` |
| `CircuitBreakerAgent` | Circuit breaker — failure detection with auto-recovery and fallback | `agent`, `fallback_agent`, `failure_threshold`, `recovery_timeout`, `result_key` |
| `TournamentAgent` | Tournament — bracket-style elimination with judge | `sub_agents`, `judge_agent`, `result_key` |
| `ShadowAgent` | Shadow testing — parallel stable + shadow comparison | `stable_agent`, `shadow_agent`, `diff_logger`, `result_key` |
| `CompilerAgent` | Compiler — iterative prompt optimisation via DSPy-style compilation | `target_agent`, `examples`, `metric_fn`, `max_iterations`, `result_key` |
| `CheckpointableAgent` | Checkpointable — sequential execution with checkpoint/resume | `sub_agents`, `checkpoint_key`, `result_key` |
| `DynamicFanOutAgent` | Dynamic fan-out — LLM-driven task decomposition with parallel workers | `worker_agent`, `reducer_agent`, `model`, `provider`, `max_items`, `result_key` |
| `SwarmAgent` | Swarm — OpenAI-style agent handoff swarm with context variables | `sub_agents`, `initial_agent`, `context_variables`, `max_handoffs`, `result_key` |
| `MemoryConsolidationAgent` | Memory consolidation — auto-summarise long conversation history | `main_agent`, `summarizer_agent`, `event_threshold`, `keep_recent`, `result_key` |
| `PriorityQueueAgent` | Priority queue — priority-based execution ordering with parallel groups | `sub_agents`, `priority_map`, `stop_condition`, `result_key` |
| `MonteCarloAgent` | Monte Carlo Tree Search — MCTS with UCT exploration/exploitation | `agent`, `evaluate_fn`, `n_simulations`, `max_depth`, `exploration_weight`, `result_key` |
| `GraphOfThoughtsAgent` | Graph-of-Thoughts — DAG-based generation, aggregation, and scoring | `agent`, `aggregate_agent`, `score_fn`, `n_branches`, `n_rounds`, `result_key` |
| `BlackboardAgent` | Blackboard architecture — shared board with expert activation loop | `sub_agents`, `controller_fn`, `termination_fn`, `max_iterations`, `board_key`, `result_key` |
| `MixtureOfExpertsAgent` | Mixture-of-Experts — gating function + weighted multi-expert blend | `sub_agents`, `gating_fn`, `top_k`, `combine_fn`, `result_key` |
| `CoVeAgent` | Chain-of-Verification — 4-phase anti-hallucination pipeline | `drafter`, `planner`, `verifier`, `reviser`, `max_questions`, `result_key` |
| `SagaAgent` | Saga — distributed transactions with compensating rollback | `steps`, `failure_detector`, `result_key` |
| `LoadBalancerAgent` | Load balancer — round-robin, random, or least-used distribution | `sub_agents`, `strategy`, `result_key` |
| `EnsembleAgent` | Ensemble — aggregate outputs from multiple agents | `sub_agents`, `aggregate_fn`, `weights`, `max_workers`, `result_key` |
| `TimeoutAgent` | Timeout wrapper — enforce deadline with fallback response | `agent`, `timeout_seconds`, `fallback_message`, `result_key` |
| `AdaptivePlannerAgent` | Adaptive planner — re-plan after every step based on results | `sub_agents`, `model`, `provider`, `planning_instruction`, `max_steps`, `result_key` |

`RouterAgent` dynamically selects from 4 modes: `single`, `sequential`, `parallel`, `loop`.

**Streaming** — `Runner.stream()` / `astream()` yield `Event` objects as they are produced. Each event includes type, author, content, and timestamp.

**Async Support** — Full async via `run_async()` / `astream()`. LLM calls and tool invocations offloaded to threads via `asyncio.to_thread()`.

**Tracing** — Automatic `start_trace()` / `end_trace()`. Records each `LLMCall` (provider, model, temperature, tokens, duration) and each `ToolRecord` (name, args, result, duration, error). Traces nest hierarchically for orchestration agents.

**Visualization** — `agent.draw()` renders an ASCII tree of the agent hierarchy.

### What it does NOT have

- No Jinja2 template system.
- No JSON task file definitions.
- No auto-batching for large datasets.
- No input schema validation (`jsonschema`).
- No built-in rate limiting (use `RateLimiter` externally or Workflow).

### When to use

- Interactive chat or assistant scenarios.
- Tasks requiring tool use (search, calculations, API calls).
- Multi-step reasoning that needs conversation context.
- Pipelines where agents collaborate or delegate.
- Scenarios requiring streaming responses.
- Dynamic routing — let the LLM decide what to do next.

### Minimal example

```python
from nono.agent import Agent, Runner, tool

@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"Results for: {query}"

agent = Agent(
    name="assistant",
    instruction="You are a helpful research assistant.",
    tools=[search],
)

runner = Runner(agent=agent)

# Synchronous — returns the final response as a string
result = runner.run("Find information about async Python")
print(result)

# Streaming — yields Event objects as they are produced
for event in runner.stream("Find information about async Python"):
    print(f"[{event.event_type.value}] {event.content}")
```

---

## Key Differences

### 1. Statefulness

```
TaskExecutor:  prompt ──→ LLM ──→ result       (no memory)

Agent:         msg1 ──→ LLM ──→ response1
               msg2 ──→ LLM ──→ response2      (remembers msg1)
               msg3 ──→ LLM ──→ response3      (remembers msg1 + msg2)
```

### 2. Tool calling

TaskExecutor **cannot** call tools — it sends a prompt and receives text/JSON.

Agent can invoke `FunctionTool` instances. The LLM decides when and which tool to call, the framework executes the function, and feeds the result back to the LLM automatically.

### 3. Prompt management

| | TaskExecutor | Agent |
|---|---|---|
| Definition | Jinja2 template or JSON task file | `instruction` parameter (string) |
| Variables | Template placeholders `{{ var }}` | Conversation messages |
| Reuse | Share `.j2` / `.json` files across projects | Share agent factory functions |

### 4. Output handling

| | TaskExecutor | Agent |
|---|---|---|
| Format | `response_format` in JSON task | `output_format="json"` or `"text"` |
| Validation | `output_schema` + `jsonschema` | Manual (or use tools to validate) |
| Structure | Guaranteed by schema | Best-effort via instruction |

### 5. Scaling

| | TaskExecutor | Agent | Workflow |
|---|---|---|---|
| Batch | Native (`batch_size`, `RateLimiter`) | Not native — use `ParallelAgent` or Workflow | `parallel_step()` for concurrent node execution |
| Concurrency | Sequential with rate limiting | `ParallelAgent` for concurrent execution | `parallel_step()` with `max_workers` |
| Loops | Not applicable | `LoopAgent` (LLM-decided) | `loop_step()` with deterministic condition |
| Join / barrier | Not applicable | Not native | `join()` — wait for named steps |
| Large data | Designed for bulk processing | Designed for interactive sessions | Designed for deterministic pipelines |

---

## Decision Flowchart

```
                    ┌─────────────────────┐
                    │  What's the task?   │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │ Needs multi-turn     │
              ┌─NO─│ conversation?        │─YES─┐
              │     └─────────────────────┘     │
              │                                  │
    ┌─────────▼───────────┐           ┌─────────▼───────────┐
    │ Needs tool calling? │           │       Agent          │
    └─────────┬───────────┘           └─────────────────────┘
        NO    │    YES
        │     │     │
        │     │  ┌──▼────────────────┐
        │     │  │      Agent        │
        │     │  └───────────────────┘
        │     │
  ┌─────▼─────▼──────────┐
  │ Batch processing or   │
  │ structured output?    │
  └─────────┬─────────────┘
      YES   │    NO
       │    │     │
  ┌────▼──┐ │  ┌──▼──────────────────┐
  │Tasker │ │  │  Either works.      │
  │       │ │  │  Tasker = simpler.  │
  └───────┘ │  └─────────────────────┘
            │
   ┌────────▼────────────┐
   │  Needs delegation   │
   │  or orchestration?  │
   └─────────┬───────────┘
       YES   │    NO
        │    │     │
   ┌────▼──┐ │  ┌──▼──────────┐
   │Agent  │ │  │   Tasker    │
   └───────┘ │  └─────────────┘
```

**Rule of thumb**: Start with `TaskExecutor`. Upgrade to `Agent` when you need tools, memory, or multi-agent collaboration.

---

## Combining Both

The most powerful pattern is **using both together**. Nono provides two bridges:

### Tasker inside Agent — `tasker_tool()`

Wrap a `TaskExecutor` call as a tool an Agent can invoke:

```python
from nono.agent import Agent, Runner, tasker_tool

# TaskExecutor becomes a tool
classify = tasker_tool(
    task_name="classify_input",
    system_prompt="Classify the input as: code, text, or data.",
    user_prompt_template="Input: {text}",
)

agent = Agent(
    name="router",
    instruction="Use the classify tool, then respond accordingly.",
    tools=[classify],
)
```

**Why?** The Agent gets conversation context and tool-calling, while Tasker provides structured output validation and specialised prompts.

### Agent inside Workflow — `agent_node()`

Wrap an Agent as a step in a deterministic Workflow:

```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node, tasker_node

flow = Workflow("pipeline")
flow.step("classify",  tasker_node("prompts/classifier.json"))
flow.step("process",   agent_node(Agent(name="processor", tools=[...])))
flow.step("validate",  tasker_node("prompts/validator.json"))
flow.connect("classify", "process")
flow.connect("process",  "validate")
```

**Why?** Deterministic ordering with full control over the pipeline, mixing stateless tasks and stateful agents.

### Workflow control-flow nodes

Workflow also provides deterministic control-flow primitives that complement Agent orchestration:

```python
flow = Workflow("advanced_pipeline")
flow.step("ingest",   ingest_fn)
flow.parallel_step("enrich", [enrich_a, enrich_b], max_workers=2)
flow.step("merge",    merge_fn)
flow.loop_step("refine", refine_fn, condition="quality < 0.9", max_iterations=5)
flow.join("barrier", wait_for=["enrich", "refine"])
flow.step("publish",  publish_fn)
flow.connect_chain(["ingest", "enrich", "merge", "refine", "barrier", "publish"])
```

| Node | Purpose |
|---|---|
| `parallel_step()` | Run N functions concurrently, merge results |
| `loop_step()` | Repeat until condition fails or max iterations |
| `join()` | Barrier — wait for listed steps before continuing |
| `enable_checkpoints()` | Persist state after each step for fault recovery |
| `load_workflow()` | Build a Workflow from a YAML/JSON definition |

### Independent and composable

Deterministic orchestration (`Workflow`) and agentic orchestration (`Agent`) are **independent systems** that combine naturally. The Workflow controls execution order, while the agent reasons freely within its step — they are orthogonal:

```python
flow = Workflow("hybrid")
flow.step("ingest", ingest_fn)                              # deterministic
flow.parallel_step("enrich", [ner_fn, sentiment_fn])         # deterministic parallel
flow.step("research", agent_node(researcher))                # agentic step
flow.loop_step("refine", refine_fn,                          # deterministic loop
               condition="quality < 0.9", max_iterations=5)
flow.step("publish", tasker_node("prompts/publish.json"))    # deterministic + LLM
flow.connect_chain(["ingest", "enrich", "research", "refine", "publish"])
```

### Summary — Integration patterns

| Pattern | How | Best for |
|---------|-----|----------|
| Tasker alone | `TaskExecutor.execute()` | Bulk processing, simple tasks |
| Agent alone | `Runner.run()` | Interactive, tools, delegation |
| Tasker as Agent tool | `tasker_tool()` | Agent decides *when* to run a structured task |
| Agent as Workflow step | `agent_node()` | Deterministic pipeline with agent reasoning |
| Workflow control-flow | `parallel_step()`, `loop_step()`, `join()` | Concurrent, iterative, or barrier-based pipelines |
| Declarative Workflow | `load_workflow("pipeline.yaml")` | No-code pipeline definition |
| Full stack | Workflow → agent_node + tasker_node + control-flow | Complex pipelines mixing both |

---

## Migration Path

Moving from Tasker to Agent when your needs grow:

### Step 1 — Simple Tasker

```python
executor = TaskExecutor(provider="google")
result = executor.execute("Summarise: {text}", text=article)
```

### Step 2 — Add tools → Agent

```python
agent = Agent(
    name="summariser",
    instruction="Summarise the text. Use the word_count tool to check length.",
    tools=[word_count_tool],
)
runner = Runner(agent=agent)
events = runner.run(article)
```

### Step 3 — Add orchestration → Pipeline

```python
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[
        Agent(name="researcher", tools=[search_tool]),
        Agent(name="writer", instruction="Write based on the research."),
        Agent(name="reviewer", instruction="Review and score."),
    ],
)
runner = Runner(agent=pipeline)
```

### Step 4 — Mix both → `tasker_tool` + Agent

```python
validate = tasker_tool(
    task_name="validate",
    system_prompt="Validate JSON against schema.",
    user_prompt_template="JSON: {data}",
    output_schema={"valid": "boolean", "errors": "array"},
)

agent = Agent(
    name="writer",
    instruction="Write JSON output. Use validate tool before returning.",
    tools=[validate],
)
```

Each step adds capability without discarding the previous one. Tasker tasks can always be reused as agent tools.

---

## Examples Side by Side

### Data classification

**Tasker approach** — best for bulk processing:

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor(provider="google")
result = executor.run_json_task(
    "prompts/name_classifier.json",
    data=["Alice Smith", "IBM Corp", "Tokyo"]
)
# Returns validated JSON with schema-checked fields
```

**Agent approach** — best for interactive reasoning:

```python
from nono.agent import Agent, Runner

agent = Agent(
    name="classifier",
    instruction="Classify each item as person, company, or location. "
                "Explain your reasoning.",
    output_format="json",
)

runner = Runner(agent=agent)

# Synchronous — returns the final string
result = runner.run("Classify: Alice Smith, IBM Corp, Tokyo")
print(result)

# Streaming — yields events incrementally
for event in runner.stream("Classify: Alice Smith, IBM Corp, Tokyo"):
    print(f"[{event.event_type.value}] {event.content}")
```

### Code generation

**Tasker approach** — template-driven, deterministic:

```python
from nono.tasker import TaskExecutor

executor = TaskExecutor(provider="google", template="python_programming")
result = executor.execute(topic="binary search", requirements=["type hints"])
```

**Agent approach** — interactive with tool execution:

```python
from nono.agent import Agent, Runner, tool
from nono.executer import CodeExecuter

@tool
def run_code(code: str) -> str:
    """Execute Python code and return the output."""
    return CodeExecuter().execute(code)

coder = Agent(
    name="coder",
    instruction="Write Python code. Test it with run_code before returning.",
    tools=[run_code],
)

runner = Runner(agent=coder)
result = runner.run("Write binary search with type hints")
print(result)
```

---

## Summary

| | **TaskExecutor** | **Agent** |
|---|---|---|
| Think of it as | A smart function | A smart assistant |
| Complexity | Low | Medium–High |
| Providers | 8 | 14 |
| Best for | Atomic, repeatable, batch tasks | Interactive, reasoning, tool-use |
| Configuration | JSON files + Jinja2 templates | Python code + instruction strings |
| Events | No | 7 types (`USER_MESSAGE`…`ERROR`) |
| Tools | No | `FunctionTool` / `@tool` (auto loop) |
| Streaming | No | `Runner.stream()` / `astream()` |
| Async | No | Full async |
| Callbacks | No | 4 lifecycle hooks |
| Session & memory | Stateless | Session + shared/local content |
| Orchestration | No | Sequential, Parallel, Loop, MapReduce, Consensus, ProducerReviewer, Debate, Escalation, Supervisor, Voting, Router, Handoff, GroupChat, Speculative, CircuitBreaker, Tournament, Shadow, Compiler, Checkpointable, DynamicFanOut, Swarm, MemoryConsolidation, PriorityQueue, MonteCarlo, GraphOfThoughts, Blackboard, MixtureOfExperts, CoVe, Saga, LoadBalancer, Ensemble, Timeout, AdaptivePlanner + Workflow (`parallel_step`, `loop_step`, `join`, `load_workflow`) |
| Batching | Native auto-batching + merge | Not built-in |
| Rate limiting | Built-in token bucket | Not built-in |
| Input validation | `input_schema` + `jsonschema` | No |
| Output validation | `output_schema` + `jsonschema` | `output_format` (best-effort) |
| Visualization | No | `draw()` ASCII tree |
| Tracing | Yes | Yes (nested, hierarchical) |
| Scaling | Native batching + rate limiting | Orchestration agents + Workflow (`parallel_step`, checkpointing, declarative YAML) |
| Start here when | You have structured input/output | You need tools, memory, or delegation |
