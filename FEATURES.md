# Features

> **NONO** = **N**o **O**verhead, **N**eural **O**perations — a unified AI framework that replaces complex multi-library setups with a single, batteries-included package for tasks, agents, workflows, and code execution across 16 LLM providers.

## Overview

| Category | Features | Status | Differential |
|----------|----------|--------|--------------|
| Structured Output | 7 | ✅ Stable | Pydantic model → validate → retry across all 16 providers; 4 parsers (JSON, Pydantic, Regex, CSV); agent-level `output_model` |
| Auto-Compaction | 3 | ✅ Stable | LLM-based context summarisation; 3 strategies + `_prune_messages` fallback; callable adapter; pluggable via `compaction` param |
| Token-Level Streaming | 6 | ✅ Stable | Native token streaming for 11+ providers; `TEXT_CHUNK` + `TOOL_CALL_CHUNK` events; streaming tool calls; `Runner.stream_text()`; SSE endpoint |
| Agent Execution Model | 10 | ✅ Stable | Typed task packets, worker state machine, failure taxonomy, executable policies, verification contract, worktree isolation, stale-branch detection, checkpoints, plan mode |
| Agent Framework (NAA) | 20 | ✅ Stable | 9 event types, 5 orchestration callbacks, hook-enabled tool loop, content scopes, ACI validation |
| Orchestration Patterns | 100 | ✅ Stable | Largest built-in pattern library in the ecosystem |
| Provider Connectors | 16 | ✅ Stable | Native multi-provider with provider-specific features (Gemini schemas, Anthropic system prompts, Perplexity citations, OpenRouter plugins) |
| Workflow Engine | 12 | ✅ Stable | Time-travel, runtime graph manipulation, StateSchema reducers |
| Tasker System | 4 | ✅ Stable | Prompts-as-data (JSON assets, not code strings) |
| Agent Skills | 20 | ✅ Stable | Full [agentskills.io](https://agentskills.io) compliance + 11 backward-compatible extensions |
| Agent Templates | 14 | ✅ Stable | Single-agent + multi-agent compositions |
| Built-in Tool Collections | 7 | ✅ Stable | DateTime, Text, Web, Python, ShortFx, OfficeBridge — ready to use |
| Dynamic Agent Factory | 4 | ✅ Stable | Generate agents from natural language with security controls |
| MCP Integration | 5 | ✅ Stable | Connect to any MCP server, auto schema normalization, response parsing |
| Sandbox Execution | 7 | ✅ Stable | Remote sandboxed code execution across 7 cloud providers |
| Lifecycle Hooks | 6 types, 15 events | ✅ Stable | Functions, commands, prompts, tasks, skills, and tools as hooks |
| ShortFx Integration | 4 | ✅ Stable | 3,000+ financial/math functions as agent tools |
| OfficeBridge Integration | 4 | ✅ Stable | Document automation (Word/PDF/HTML/Excel), translation, PII censoring |
| ShortFx Integration | 4 | ✅ Stable | 3,000+ deterministic functions with semantic search |
| Decision Wizard | 3 | ✅ Stable | No equivalent in other frameworks |
| Project System | 4 | ✅ Stable | Git-style discovery, isolated manifests |
| Code Execution | 2 | ✅ Stable | Sandboxed LLM-generated code runner |
| API Server | 9 | ✅ Stable | Named-resource invocation, SSE streaming, 1 MB DoS protection |
| Observability | 7 | ✅ Stable | Built-in tracing with memory-safe eviction, rich metadata |
| Workspace System | 7 | ✅ Stable | Declarative I/O — File, URL, Cloud, Template, Inline entries |
| Enterprise | 10 | ✅ Stable | 3 rate-limit algorithms, circuit breaker, retry policies, API metrics, SSL, fallback, model DB, temperature presets |
| CLI | 15 | ✅ Stable | Full framework access — run, agent, workflow, skill, config, project, wizard, MCP, 6 output formats |
| Agent Card (A2A) | 5 | ✅ Stable | A2A protocol Agent Card generation, well-known URI server, automatic skill extraction from agents/workflows/tools |

---

## What Makes Nono Different

### vs LangChain

| Aspect | LangChain | Nono |
|--------|-----------|------|
| Orchestration patterns | ~5 (Sequential, Router, Map-Reduce, few others) | **100 built-in** (Tree of Thoughts, Monte Carlo, Genetic Algorithm, Circuit Breaker…) |
| Event system | Callbacks, raw strings | **9 typed immutable events** with IDs, timestamps, structured data |
| Orchestration callbacks | None | **5 lifecycle callbacks** (on_start, on_end, on_between_agents, on_agent_start, on_agent_end) |
| Tool-call middleware | None | **Hooks that block, modify arguments, and inject context** |
| Content management | Global state only | **Dual-scope LRU stores** (shared + local) with eviction |
| Inter-agent data passing | `RunnablePassthrough` + `itemgetter` (manual LCEL piping) | **3 built-in mechanisms**: auto `user_message` chaining, `result_key`, `message_map` — zero boilerplate |
| State isolation | Not native | **3 levels**: `local_content`, `ContextFilterAgent`, new `Session` |
| Session constructor | Not applicable | **Pluggable memory** (`KeepInMind`), `max_events`, initial state |
| Provider switching | Requires swapping chain objects | **One config change** — same code, any provider |
| Prompt management | Code strings or LangSmith (external) | **JSON assets** + Jinja2 templates (version-controlled, no code changes) |
| Workflow engine | LCEL (code-only, linear) | **DAG engine** with checkpointing, time-travel, parallel, loops, join barriers |
| Complexity guidance | None | **Decision Wizard** recommends the optimal pattern |
| Tool ecosystem | External integrations only | **7 built-in tool collections** + ShortFx + OfficeBridge (3,000+ functions) + MCP client |
| Tool validation | None | **ACI validation** at construction — checks descriptions, names, parameters |
| Skill system | Tools only | **Triple-mode skills**: standalone, as tool, as pipeline component |
| Dependencies | 100+ transitive packages | **Minimal** — core has ~6 dependencies |

### vs CrewAI

| Aspect | CrewAI | Nono |
|--------|--------|------|
| Agent patterns | Role-based crews (~3 patterns) | **100 orchestration patterns + 29 domain pipelines** across 17 categories |
| Event system | Logs | **9 typed immutable events** safe for replay and audit |
| Orchestration control | None | **5 lifecycle callbacks** with early-halt support |
| Hook types | None | **6 types**: function, command, prompt, task, skill, tool — across 15 events |
| Content scopes | Shared state | **shared + local LRU** with capacity limits |
| Inter-agent data passing | Implicit via role context (limited) | **3 mechanisms**: auto chaining, `result_key`, `message_map` |
| State isolation | Not native | **3 levels**: `local_content`, `ContextFilterAgent`, new `Session` |
| Workflow support | Sequential/Hierarchical only | **Full DAG** with branching, loops, checkpoints, time-travel |
| Provider support | OpenAI-centric + LiteLLM | **15 native connectors** with automatic fallback |
| Code execution | External tools | **Built-in sandboxed executor** + 7 cloud sandbox providers |
| Human-in-the-Loop | Basic input | **HumanInputAgent** with display_keys, callbacks, configurable rejection |
| Agent generation | Manual setup | **Dynamic Agent Factory** — generate agents from natural language |

### vs AutoGen / AG2

| Aspect | AutoGen | Nono |
|--------|---------|------|
| Setup complexity | Multi-file config, Docker agents | **Single import** — `from nono.agent import Agent, Runner` |
| Event system | Messages | **9 typed immutable events** with structured data |
| Orchestration | Conversation-based (GroupChat) | **Pattern-based** — pick from 100 orchestrators + 29 domain pipelines |
| Inter-agent data passing | Via shared chat history (implicit) | **3 mechanisms**: auto chaining, `result_key`, `message_map` |
| State isolation | Not native | **3 levels**: `local_content`, `ContextFilterAgent`, new `Session` |
| Orchestration callbacks | None | **5 lifecycle callbacks** — halt, monitor, inject logic between agents |
| Tool middleware | None | **Hooks that block, modify, and extend tool calls** |
| Deterministic workflows | Not built-in | **Graph-based engine** with state schemas and reducers |
| Observability | External (Langfuse, etc.) | **Hierarchical TraceCollector** with memory-safe eviction |
| Lifecycle hooks | Not built-in | **6 hook types** (function, command, prompt, task, skill, tool) across **15 events** |
| External tools | Custom tool wrappers | **MCP client** — connect to any Model Context Protocol server |

### vs Google ADK

| Aspect | Google ADK | Nono |
|--------|------------|------|
| Event types | 4 (content, tool_call, tool_response, state_delta) | **9 typed events** including transfers, HITL, state, errors |
| Orchestration callbacks | None | **5 fluent lifecycle callbacks** per orchestration agent |
| Hook types | None | **6 types**: function, command, prompt, task, skill, tool — hooks can run AI operations |
| Content management | State + artifacts | **Dual-scope LRU** (shared + local) with eviction and size limits |
| Inter-agent data passing | Via `state` dict (manual) | **3 mechanisms**: auto chaining, `result_key`, `message_map` |
| State isolation | Not native | **3 levels**: `local_content`, `ContextFilterAgent`, new `Session` |
| Orchestration patterns | ~4 (Sequential, Parallel, Loop, Pipeline) | **100 built-in + 29 domain pipelines** across 17 categories |
| Skill system | None | **Triple-mode skills** (standalone / tool / pipeline) |
| Thread safety | Not documented | **Lock-based state** for parallel sub-agents |
| Complexity guidance | None | **Wizard + ComplexityBudget + audit_agent_tree()** |
| ACI validation | None | **Automatic at construction** |
| Providers | Google Gemini only | **14 native providers** with transparent fallback |

---

## Agent Framework (NAA — Nono Agent Architecture)

### LlmAgent

- **Description**: LLM-powered agent with tool-calling loops, sub-agent delegation, and `transfer_to_agent`
- **Module**: `nono/agent/agent.py`
- **Status**: ✅ Stable
- **Differential**: Single class handles tools, sub-agents, and inter-agent transfers — no separate "chain" or "crew" abstractions

### Context Management

- **Description**: Automatic safeguards to prevent context overflow and token waste during tool-calling loops
- **Module**: `nono/agent/llm_agent.py`
- **Status**: ✅ Stable
- **Differential**: Built-in protection — no manual token counting or message trimming needed

| Guard | Default | Config Key | Description |
|-------|---------|------------|-------------|
| `_MAX_TOOL_ITERATIONS` | 10 | `agent.max_tool_iterations` | Maximum tool-call loop iterations to prevent infinite loops |
| `_MAX_LOOP_MESSAGES` | 40 | `agent.max_loop_messages` | Sliding window pruning — oldest messages are dropped to stay within context limits |
| `_MAX_TOOL_RESULT_CHARS` | 20,000 | `agent.max_tool_result_chars` | Tool result truncation — prevents a single large tool output from consuming the entire context |

All values are configurable via `config.toml` under `[agent]`.

### Runner

- **Description**: Manage agent session lifecycle with state updates and event streaming
- **Module**: `nono/agent/runner.py`
- **Status**: ✅ Stable
- **Methods**: `run()`, `stream()`, `run_async()`, `astream()` — sync and async with event yielding

### 9 Immutable Event Types

- **Description**: Structured, frozen event records with unique IDs, UTC timestamps, and typed payloads
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable
- **Differential**: Other frameworks yield raw strings or simple deltas. Nono yields typed `Event` objects safe for replay, audit trails, and deterministic debugging.

| Event | When it fires |
|-------|---------------|
| `USER_MESSAGE` | User sends a message |
| `AGENT_MESSAGE` | Agent produces a response |
| `TOOL_CALL` | Before a tool is invoked (includes tool name + arguments) |
| `TOOL_RESULT` | After a tool returns (includes result data) |
| `STATE_UPDATE` | Session state changes |
| `AGENT_TRANSFER` | Agent delegates to another agent |
| `HUMAN_INPUT_REQUEST` | Workflow pauses for human review |
| `HUMAN_INPUT_RESPONSE` | Human responds with approval/rejection/feedback |
| `ERROR` | Error during execution |

### 5 Orchestration Lifecycle Callbacks

- **Description**: Typed callbacks for every orchestration agent (Sequential, Parallel, Loop…) via fluent API
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable
- **Differential**: No other framework exposes typed orchestration lifecycle hooks. Google ADK has before/after model callbacks but not inter-agent callbacks.

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `on_start` | `(orchestrator, session) → None` | Fired once when orchestration begins |
| `on_end` | `(orchestrator, session, agents_executed) → None` | Fired once when orchestration finishes |
| `on_between_agents` | `(completed, next, session) → bool?` | Return `False` to halt early |
| `on_agent_start` | `(sub_agent, session) → None` | Before each sub-agent starts |
| `on_agent_end` | `(sub_agent, session, error) → None` | After each sub-agent finishes |

### 4 Agent-Level Callbacks

- **Description**: Before/after callbacks for agent execution and tool invocation
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `before_agent_callback` | `(agent, ctx) → str?` | Short-circuit execution with a response |
| `after_agent_callback` | `(agent, ctx, response) → str?` | Transform the final response |
| `before_tool_callback` | `(agent, tool_name, args) → dict?` | Modify tool arguments |
| `after_tool_callback` | `(agent, tool_name, args, result) → Any?` | Transform tool result |

### Hook-Enabled Tool Loop

- **Description**: `PreToolUse` hooks can **block** tool calls or **modify arguments**; `PostToolUse` hooks can **inject additional context** into results
- **Module**: `nono/agent/llm_agent.py`
- **Status**: ✅ Stable
- **Differential**: Deterministic middleware for the tool-calling loop — not just logging. No other framework allows blocking or modifying tool calls via hooks.

| Hook | Capability |
|------|-----------|
| `PreToolUse` → `should_block=True` | Tool is skipped, LLM receives block message |
| `PreToolUse` → `updated_input={...}` | Tool receives modified arguments |
| `PostToolUse` → `additional_context="..."` | Extra text appended to tool result |

### Content Scopes (SharedContent)

- **Description**: Session-wide and agent-private LRU content stores with automatic eviction
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable
- **Differential**: No other framework provides dual-scope LRU content stores with size limits and thread-safe access

| Scope | Visibility | Default capacity |
|-------|-----------|-----------------|
| `shared_content` | All agents in session | 200 items |
| `local_content` | Only the owning agent | 200 items |
| Per-item limit | — | 10 MB |

### ToolContext

- **Description**: Rich context injected into tools at invocation — includes state, both content scopes, agent name, and session ID
- **Module**: `nono/agent/tool.py`
- **Status**: ✅ Stable
- **Differential**: Tools in Nono receive thread-safe state access (`state_set`/`state_get`), dual content scopes, and agent identity — not just raw arguments

### ACI Validation (`validate_tools`)

- **Description**: Automatic validation of tool descriptions and schemas at agent construction time
- **Module**: `nono/agent/tool.py`
- **Status**: ✅ Stable
- **Differential**: No other framework validates tool descriptions for LLM quality. Checks: description length, name meaningfulness, parameter types, parameter existence.

### InvocationContext

- **Description**: Everything an agent needs for a single turn: session, user message, parent agent ref, trace collector, transfer depth
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable

### Thread-Safe Session State

- **Description**: `state_set()`, `state_get()`, `state_update()` use `threading.Lock` — safe for `ParallelAgent` scenarios
- **Module**: `nono/agent/base.py`
- **Status**: ✅ Stable
- **Differential**: Most frameworks assume sequential access. Nono's session state is safe for concurrent sub-agents.

### Native `transfer_to_agent`

- **Description**: Automatic tool registration for sub-agent delegation via LLM function-calling
- **Module**: `nono/agent/llm_agent.py`
- **Status**: ✅ Stable
- **Differential**: No manager class needed. `MAX_TRANSFER_DEPTH` (configurable, default 10) prevents infinite delegation loops.

### Human-in-the-Loop (HITL)

- **Description**: Pause AI execution and request human approval, rejection, or feedback with structured data flow
- **Module**: `nono/hitl.py`
- **Status**: ✅ Stable
- **Differential**: Typed `HumanInputResponse` with `approved`, `message`, and `data` fields — works in both workflows and agents

| Component | Description |
|-----------|-------------|
| `HumanInputResponse` | Typed response: `approved`, `message`, `data` |
| `HumanInputHandler` | Sync callback protocol `(step_name, state, prompt) → HumanInputResponse` |
| `AsyncHumanInputHandler` | Async variant for web API, WebSocket, or queue backends |
| `console_handler` | Built-in interactive CLI handler with state review |
| `make_auto_handler` | Pre-configured responses per step — for testing and CI/CD |
| `format_state_for_review` | State formatting with truncation, indentation, and key filtering (`display_keys`) |
| `HumanRejectError` | Structured exception when human rejects and no reject-branch is configured |

### HumanInputAgent

- **Description**: First-class agent that plugs into any pipeline — emits `HUMAN_INPUT_REQUEST`, blocks, emits `HUMAN_INPUT_RESPONSE`
- **Module**: `nono/agent/human_input.py`
- **Status**: ✅ Stable
- **Differential**: Supports `display_keys` (show only relevant state to human), `before_human`/`after_human` callbacks, configurable rejection (`on_reject="error"` or `"continue"`)

### KeepInMind Integration

- **Description**: Persistent long-term memory for agents via file-based JSON Lines store
- **Module**: `nono/agent/keepinmind.py`
- **Status**: ✅ Stable
- **Differential**: Auto-load prior conversations on session start, auto-commit new messages. `MemoryStore` ABC allows pluggable backends. Zero external dependencies.

| Component | Description |
|-----------|-------------|
| `KeepInMind` | High-level façade: `remember()` / `commit()` helpers |
| `MemoryStore` (ABC) | Backend contract: `save`, `load`, `list_sessions`, `delete`, `clear` |
| `FileMemoryStore` | Default backend — JSON Lines on disk, zero dependencies |
| `MemoryEntry` | Frozen dataclass: role, content, agent_name, timestamp, metadata |
| `append_entry()` | O(1) file-append mode — no full rewrite on each message |
| `max_entries` | Configurable cap — prunes oldest entries automatically |
| Thread safety | All operations protected by `threading.Lock` |

### Triple-Mode Skills

- **Description**: Skills are composable AI capabilities with three usage modes: standalone, as tool, in pipeline
- **Module**: `nono/agent/skill.py`
- **Status**: ✅ Stable
- **Differential**: Other frameworks treat tools as atoms. Nono skills bundle an agent + tools + metadata (`SkillDescriptor` following [agentskills.io](https://agentskills.io) spec) into a composable unit.

| Mode | Method | Use case |
|------|--------|----------|
| Standalone | `skill.run("message")` | Direct execution |
| Tool | `skill.as_tool()` → `FunctionTool` | LLM calls the skill as a function |
| Pipeline | `skill.build_agent()` → `BaseAgent` | Compose into orchestration pipelines |

### Tasker Tool Bridge

- **Description**: Expose JSON-defined Tasker tasks as agent-callable tools
- **Module**: `nono/agent/tasker_tool.py`
- **Status**: ✅ Stable
- **Differential**: Reuse prompt assets as tools without rewriting them

### Tool System

- **Description**: `FunctionTool`, `@tool` decorator, `ToolContext`, and tool validation
- **Module**: `nono/agent/tools.py`
- **Status**: ✅ Stable

| Feature | Description |
|---------|-------------|
| `FunctionTool` | Wraps any Python function as an LLM-callable tool |
| `@tool` decorator | Zero-config decorator with optional name/description override |
| `ToolContext` auto-injection | Functions accepting a `ToolContext` parameter get it injected automatically — excluded from the JSON schema sent to the LLM |
| `parse_tool_calls()` | Parse tool call instructions from raw LLM responses |
| `validate_tools()` | ACI quality validation at construction time |

---

## 100 Orchestration Patterns + 29 Domain Pipelines

> The largest built-in orchestration library in the AI framework ecosystem. Other frameworks offer 3–5 patterns; Nono ships 100 orchestration agents across 17 categories plus 29 ready-to-use multi-agent pipelines for development, architecture, operations, data, AI/ML, content, and security.

### Flow Control

| Pattern | Description |
|---------|-------------|
| `SequentialAgent` | Execute agents in strict order — each agent's response becomes the next agent's `user_message` automatically |
| `ParallelAgent` | Run agents concurrently and merge results |
| `LoopAgent` | Repeat agent execution until exit condition |
| `RouterAgent` | Route input to a specialist agent based on content |

### Collective Reasoning

| Pattern | Description |
|---------|-------------|
| `MapReduceAgent` | Split input → parallel processing → aggregate results |
| `ConsensusAgent` | Multiple agents vote on a shared answer |
| `VotingAgent` | Majority-vote selection across agent outputs |
| `DebateAgent` | Structured multi-round debate between agents |

### Quality Assurance

| Pattern | Description |
|---------|-------------|
| `ProducerReviewerAgent` | Generate + review cycle with feedback loop |
| `GuardrailAgent` | Validate outputs against safety/quality rules |
| `BestOfNAgent` | Generate N candidates, select the best |
| `SelfRefineAgent` | Iterative self-improvement of outputs |
| `SelfConsistencyAgent` | Sample multiple reasoning paths, pick the most consistent |
| `VerifierAgent` | Verify and validate agent outputs against criteria |
| `RecursiveCriticAgent` | Recursive critique and refinement loop |

### Hierarchical

| Pattern | Description |
|---------|-------------|
| `SupervisorAgent` | Manager agent delegates and monitors worker agents |
| `EscalationAgent` | Escalate tasks to higher-capability agents on failure |
| `HierarchicalAgent` | Multi-level chain of command |
| `OrchestratorWorkerAgent` | Orchestrator plans, workers execute |

### Delegation

| Pattern | Description |
|---------|-------------|
| `HandoffAgent` | Transfer control to another agent mid-execution |
| `GroupChatAgent` | Multi-agent conversation with turn management |
| `SwarmAgent` | Dynamic swarm of agents with emergent coordination |

### Batch & Scaling

| Pattern | Description |
|---------|-------------|
| `BatchAgent` | Process items in batches with concurrency control |
| `CascadeAgent` | Try cheap models first, escalate to expensive on failure |
| `LoadBalancerAgent` | Distribute requests across multiple agent instances |
| `EnsembleAgent` | Combine outputs from diverse agents/models |
| `DynamicFanOutAgent` | Spawn agents dynamically based on input analysis |
| `PipelineParallelAgent` | Stage-parallel pipeline with overlapping execution |

### Advanced Reasoning

| Pattern | Description |
|---------|-------------|
| `TreeOfThoughtsAgent` | Explore branching reasoning paths with backtracking |
| `GraphOfThoughtsAgent` | Non-linear reasoning with graph-structured exploration |
| `MonteCarloAgent` | Stochastic sampling with best-path selection |
| `BeamSearchAgent` | Maintain top-K reasoning paths in parallel |
| `AnalogicalReasoningAgent` | Solve problems by analogy to known examples |
| `ThreadOfThoughtAgent` | Thread-based progressive reasoning |
| `BufferOfThoughtsAgent` | Buffered thought accumulation and synthesis |
| `ChainOfAbstractionAgent` | Abstract reasoning chain before concretizing |
| `ProgOfThoughtAgent` | Program-of-thought: generate code to solve reasoning tasks |

### Planning

| Pattern | Description |
|---------|-------------|
| `PlannerAgent` | Generate execution plan before acting |
| `AdaptivePlannerAgent` | Replan dynamically based on intermediate results |
| `SubQuestionAgent` | Decompose complex questions into sub-questions |
| `SkeletonOfThoughtAgent` | Generate outline first, fill in details |
| `LeastToMostAgent` | Solve from simplest to most complex sub-problem |
| `SelfDiscoverAgent` | Discover which reasoning modules to apply |
| `AgendaAgent` | Manage a structured agenda of tasks to execute |

### Prompting Strategies

| Pattern | Description |
|---------|-------------|
| `ChainOfDensityAgent` | Progressive summarization with increasing density |
| `StepBackAgent` | Abstract the problem before solving |
| `CoVeAgent` | Chain-of-Verification: generate → verify → refine |
| `SocraticAgent` | Guide through questions rather than direct answers |
| `ReflexionAgent` | Reflect on past attempts to improve next attempt |
| `ContextFilterAgent` | Filter irrelevant context before processing |
| `RephraseAndRespondAgent` | Rephrase input for clarity before responding |
| `ExpertPromptingAgent` | Adopt expert personas for domain-specific reasoning |
| `PromptChainAgent` | Chain multiple prompts in sequence with context passing |

### Resilience

| Pattern | Description |
|---------|-------------|
| `CircuitBreakerAgent` | Stop calling failing providers after threshold |
| `TimeoutAgent` | Enforce execution time limits per agent |
| `SagaAgent` | Distributed transaction with compensating rollbacks |
| `CheckpointableAgent` | Save/restore agent state for crash recovery |
| `BacktrackingAgent` | Revert to previous state on failure and try alternatives |

### Evolutionary & Optimization

| Pattern | Description |
|---------|-------------|
| `GeneticAlgorithmAgent` | Evolve solutions through mutation and selection |
| `MultiArmedBanditAgent` | Explore/exploit trade-off for agent selection |
| `TournamentAgent` | Bracket-style competition between solution candidates |
| `SimulatedAnnealingAgent` | Probabilistic search with temperature-based acceptance |
| `TabuSearchAgent` | Search with memory of visited states to avoid cycling |
| `ParticleSwarmAgent` | Swarm-based optimization with velocity tracking |
| `DifferentialEvolutionAgent` | Evolve solutions via differential mutation |
| `BayesianOptimizationAgent` | Surrogate-model-guided optimization |
| `AntColonyAgent` | Pheromone-based path optimization |

### Meta

| Pattern | Description |
|---------|-------------|
| `MetaOrchestratorAgent` | Orchestrator that selects which orchestration pattern to use |
| `MixtureOfExpertsAgent` | Route to specialized expert agents based on input |
| `MixtureOfAgentsAgent` | Blend outputs from multiple agents |
| `BlackboardAgent` | Shared knowledge board for collaborative problem-solving |
| `MixtureOfThoughtsAgent` | Combine diverse reasoning strategies |

### Multi-Agent Communication

| Pattern | Description |
|---------|-------------|
| `RolePlayingAgent` | CAMEL-style role-playing conversation between agents |
| `GossipProtocolAgent` | Decentralized information propagation across agents |
| `AuctionAgent` | Auction-based task allocation among agents |
| `DelphiMethodAgent` | Multi-round anonymous expert polling for consensus |
| `NominalGroupAgent` | Structured group technique with independent ideation |
| `ContractNetAgent` | Contract-net protocol for task delegation |

### Retrieval-Augmented

| Pattern | Description |
|---------|-------------|
| `ActiveRetrievalAgent` | FLARE-style active retrieval during generation |
| `IterativeRetrievalAgent` | Multi-round retrieval with query refinement |
| `DemonstrateSearchPredictAgent` | DSP: demonstrate → search → predict pipeline |

### Observability

| Pattern | Description |
|---------|-------------|
| `ShadowAgent` | Run in parallel for comparison without affecting output |
| `CompilerAgent` | Transform agent output into structured formats |
| `SpeculativeAgent` | Predict likely outputs and validate speculatively |

### Resource Management

| Pattern | Description |
|---------|-------------|
| `CacheAgent` | Cache agent responses to avoid redundant LLM calls |
| `BudgetAgent` | Enforce token/cost budgets across execution |
| `CurriculumAgent` | Progressive difficulty scaling |
| `PriorityQueueAgent` | Process tasks by priority order |
| `MemoryConsolidationAgent` | Consolidate and compress agent memory over time |

### Mediation & Learning

| Pattern | Description |
|---------|-------------|
| `MediatorAgent` | Resolve conflicts between agent outputs |
| `DivideAndConquerAgent` | Split problem → solve parts → merge solutions |
| `CumulativeReasoningAgent` | Build reasoning incrementally across iterations |
| `MultiPersonaAgent` | Generate outputs from multiple persona perspectives |
| `RedTeamAgent` | Adversarial testing of agent outputs |
| `FeedbackLoopAgent` | Continuous feedback-driven improvement |
| `WinnowingAgent` | Progressively filter and narrow candidate solutions |
| `InnerMonologueAgent` | Internal reasoning monologue before responding |
| `HypothesisTestingAgent` | Generate and test hypotheses systematically |
| `SkillLibraryAgent` | Accumulate and reuse learned skills |
| `DoubleLoopLearningAgent` | Question assumptions, not just correct errors |

---

## Agent Templates

### Single-Agent Templates

| Template | Description |
|----------|-------------|
| `classifier_agent` | Categorize inputs into predefined classes |
| `coder_agent` | Generate and review code |
| `decomposer_agent` | Break complex tasks into sub-tasks |
| `extractor_agent` | Extract structured data from unstructured text |
| `guardrail_agent` | PII detection, GDPR/HIPAA compliance, prompt injection detection, content safety — structured JSON output with risk levels |
| `planner_agent` | Generate step-by-step execution plans |
| `reviewer_agent` | Review and score content quality |
| `summarizer_agent` | Summarize long-form content |
| `writer_agent` | Generate written content |

### Multi-Agent Pipelines (29)

#### Original

| Pipeline | Description |
|----------|-------------|
| `plan_and_execute` | Planner → Decomposer → Coder pipeline |
| `research_and_write` | Extractor → Writer → Reviewer pipeline |
| `draft_review_loop` | Writer ↔ Reviewer iterative loop |
| `classify_and_route` | Classifier → Specialist Router |

#### Development

| Pipeline | Description |
|----------|-------------|
| `bug_fix` | Triager → Debugger → Fixer → Tester → Reviewer |
| `refactoring` | Code Analyzer → Planner → Refactorer → Tester → Reviewer |
| `product_development` | Product Designer → Planner → Developer → Reviewer |
| `code_review_automation` | Diff Analyzer → [Style ‖ Logic ‖ Security] → Summary (fan-out/fan-in) |
| `performance_optimization` | Profiler → Bottleneck Analyzer → Optimizer → Benchmarker → Reviewer |
| `test_suite_generation` | Code Analyzer → Test Planner → Writer → Coverage → Mutation |

#### Architecture

| Pipeline | Description |
|----------|-------------|
| `system_design` | Requirements Analyst → Architect → Reviewer → Decision Logger |
| `database_design` | Domain Modeler → Schema Designer → Migrator → Validator |
| `api_design` | Domain Expert → API Designer → Implementer → Doc Gen → Consumer Tester |

#### Operations

| Pipeline | Description |
|----------|-------------|
| `incident_response` | Detector → Diagnostician → Responder → RCA → Postmortem |
| `devops_deployment` | Build → Security Scan → Deploy → Monitor |
| `cost_optimization` | Resource Scanner → Usage Analyzer → Optimizer → Validator |
| `observability_setup` | Signal Identifier → Instrumenter → Dashboards → Alerts |
| `disaster_recovery` | Risk Assessor → Runbook → Simulator → Validator → Certifier |
| `migration` | Legacy Analyzer → Target Designer → Migrator → Validator → Deployer |

#### Data

| Pipeline | Description |
|----------|-------------|
| `data_quality` | Profiler → Rule Designer → Validator → Cleaner → Reporter |
| `etl_pipeline_design` | Source Analyzer → Transform Designer → Implementer → Validator → Scheduler |

#### AI/ML

| Pipeline | Description |
|----------|-------------|
| `prompt_engineering` | Task Analyzer → Drafter → Variations → Evaluator → Optimizer (loop) |
| `rag_pipeline_design` | Corpus → Chunking → Embedding → Retriever → E2E Evaluator (loop) |
| `model_fine_tuning` | Data Curator → Preprocessor → Trainer → Evaluator → Publisher (loop) |
| `ai_safety_guardrails` | Risk Cataloger → Red Teamer → Designer → Tester → Certifier (loop) |

#### Content & Knowledge

| Pipeline | Description |
|----------|-------------|
| `content_documentation` | Researcher → Writer → Tech Reviewer → Publisher |
| `research` | Question Formulator → Source Finder → Analyzer → Report Writer |
| `security_audit` | Threat Modeler → Static Analyzer → Pen Tester → Remediator → Verifier |
| `compliance` | Evidence Collector → Gap Analyzer → Remediator → Auditor → Reporter |

- **Module**: `nono/agent/templates/`
- **Status**: ✅ Stable
- **Differential**: Ready-to-use compositions — import and run, no wiring required. 100+ agents across 7 domains

---

## Agent Skills

### Class-Based Skills

| Skill | Description | Bundled Assets |
|-------|-------------|----------------|
| `classify` | Categorize text with configurable taxonomy | Output schema, reference data |
| `code_review` | Automated code review with scoring | Report schema, scoring script |
| `extract` | Extract entities and structured data | Output schema, reference data |
| `summarize` | Summarize text with density control | Reference data |
| `translate` | Translate text between languages | Reference data |

### SKILL.md Template Skills (Anthropic/Claude Standard)

| Skill | Description |
|-------|-------------|
| `analyzing-data` | Analyze datasets and extract insights |
| `analyzing-sentiment` | Detect and classify sentiment in text |
| `classifying-data` | Categorize data into predefined classes |
| `converting-formats` | Transform data between formats (JSON, CSV, XML…) |
| `explaining-code` | Generate clear code explanations |
| `extracting-data` | Extract structured data from unstructured text |
| `generating-api-docs` | Generate API documentation from code |
| `generating-sql` | Generate SQL queries from natural language |
| `generating-tests` | Generate unit and integration tests |
| `researching-topics` | Research and synthesize information on topics |
| `reviewing-code` | Perform automated code reviews |
| `summarizing-text` | Summarize long-form content |
| `translating-text` | Translate between languages |
| `writing-documentation` | Write technical documentation |
| `writing-emails` | Compose professional emails |

- **Module**: `nono/agent/skills/`
- **Status**: ✅ Stable
- **Differential**: SKILL.md standard with bundled assets (schemas, scripts, references) — skills are self-contained packages, not just functions. 20 skills across two paradigms: class-based (`BaseSkill`) and template-based (SKILL.md directories)

### Agent Skills Standard Compliance (agentskills.io)

Nono's skill system is **fully compatible** with the [Agent Skills](https://agentskills.io/specification) open standard — the interoperable format originally developed by Anthropic and adopted by Claude Code, OpenHands, Spring AI, Mistral AI Vibe, JetBrains Junie, Qodo, Snowflake Cortex Code, Databricks Genie, and others.

#### Standard Specification Coverage

| Requirement | Spec | Nono | Implementation |
|---|---|---|---|
| Directory structure (`skill-name/SKILL.md`) | Required | ✅ | 15 built-in skill directories in `nono/agent/skills/` |
| YAML frontmatter + Markdown body | Required | ✅ | `MarkdownSkill` parses frontmatter → `SkillDescriptor` |
| `name` (max 64, lowercase + hyphens, matches directory) | Required | ✅ | All skills use kebab-case matching parent dir |
| `description` (max 1024, non-empty) | Required | ✅ | Every `SkillDescriptor` includes description |
| `license` | Optional | ✅ | Built-in skills use `license: MIT` |
| `compatibility` (max 500) | Optional | ✅ | Supported in frontmatter |
| `metadata` (key-value map) | Optional | ✅ | Built-in skills include `author`, `version` |
| `allowed-tools` (experimental) | Optional | ✅ | Supported in frontmatter |
| Progressive disclosure (3 levels) | Required | ✅ | Metadata → Instructions → Resources |
| `scripts/` directory | Optional | ✅ | `list_scripts()`, e.g. `reviewing-code/scripts/score.py` |
| `references/` directory | Optional | ✅ | `list_references()`, e.g. `references/REFERENCE.md` |
| `assets/` directory | Optional | ✅ | `list_assets()`, e.g. `assets/output_schema.json` |
| File references (relative paths) | Required | ✅ | `skill.load_resource("references/REFERENCE.md")` |

#### Progressive Disclosure (3-Level Loading)

| Level | What Loads | When | Token Budget |
|---|---|---|---|
| **1. Metadata** | `name` + `description` | At startup / import | ~100 tokens total |
| **2. Instructions** | Full SKILL.md body | When skill is activated | < 5 000 tokens recommended |
| **3. Resources** | `scripts/`, `references/`, `assets/` | As needed by the agent | On demand |

#### Nono Extensions (Backward-Compatible)

Nono extends the standard with capabilities not available in the base spec. Extensions use additional YAML fields that are safely ignored by other compliant clients.

| Extension | Description | Standard Impact |
|---|---|---|
| **Python skills** (`BaseSkill`) | Programmatic skills with custom tools, validation, complex logic | Additive — SKILL.md format unaffected |
| **`tools:` YAML field** | Reference `scripts/` as `FunctionTool` directly from frontmatter | Ignored by clients that don't support it |
| **`tags` field** | Categorize skills for `registry.find_by_tag()` discovery | Ignored by clients that don't support it |
| **`temperature` field** | Set LLM temperature from frontmatter | Ignored by clients that don't support it |
| **`output_format` field** | Declare expected output format (json, text, table…) | Ignored by clients that don't support it |
| **`SkillRegistry`** | Global registry with `get()`, `find_by_tag()`, `names` | Framework-side — no file format change |
| **`as_tool()` conversion** | Any skill becomes an LLM function-calling tool | Framework-side — no file format change |
| **`skill.run()` standalone** | Execute a skill without a host agent | Framework-side — no file format change |
| **`build_tools()` injection** | Skill tools auto-wired into the inner agent | Framework-side — no file format change |
| **Multi-provider** | Skills work with all 14 Nono LLM providers | Framework-side — Claude-only in original spec |
| **CLI + REST API** | `nono skill {name}` / `POST /skill/{name}` | Framework-side — no file format change |

- **Spec**: [agentskills.io/specification](https://agentskills.io/specification)
- **Compatibility**: Any SKILL.md directory valid under the agentskills.io spec works in Nono without modification
- **Portability**: Nono SKILL.md directories are valid agentskills.io skills (extra YAML fields are ignored by other clients)
- **Documentation**: [`nono/README_skills.md`](nono/README_skills.md) — full reference with comparison table

---

## Structured Output & Validation

- **Description**: Type-safe structured output from LLM responses with Pydantic model validation, reusable output parsers, and automatic retry on parse failure — works across all 14 providers
- **Module**: `nono/connector/structured_output.py`
- **Status**: ✅ Stable
- **Differential**: Other frameworks require external libraries (Instructor, Outlines) or provider-specific features. Nono provides a unified parsing + validation + retry pipeline that works with any provider.

### Output Parsers

| Parser | Input | Output | Validation |
|--------|-------|--------|------------|
| `PydanticOutputParser` | Raw LLM text | Pydantic model instance | Pydantic model validation + JSON schema derivation |
| `JsonOutputParser` | Raw LLM text | `dict` | Optional `jsonschema` validation |
| `RegexOutputParser` | Raw LLM text | `str` (captured group) | Regex pattern matching |
| `CsvOutputParser` | Raw LLM text | `list[dict]` | Header row + expected column validation |

### StructuredGenerator

| Feature | Description |
|---------|-------------|
| `StructuredGenerator(service, model=MyModel)` | Wrap any service with Pydantic validation |
| `StructuredGenerator(service, parser=parser)` | Wrap any service with a custom parser |
| Format injection | Automatically appends format instructions to prompts |
| Repair prompts | On parse failure, sends error details back to the LLM |
| Configurable retries | `max_retries` parameter (default 2) |
| Provider-aware | Sets `ResponseFormat.JSON` + `json_schema` for JSON parsers |

### LlmAgent Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_model` | `type` | `None` | Pydantic `BaseModel` class — auto-validates and retries |
| `output_parser` | `OutputParser` | `None` | Custom parser instance for any format |
| `output_retries` | `int` | `2` | Max retry attempts on parse failure |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `parse_json(text, schema=None)` | One-shot JSON parse with optional schema validation |
| `parse_pydantic(text, model)` | One-shot Pydantic model parse |
| `parse_csv(text, delimiter, expected_columns)` | One-shot CSV parse |

### Code Fence Extraction

All parsers automatically extract content from markdown code fences (`` ```json ``, `` ```csv ``) — no preprocessing needed when the LLM wraps output in fences.

---

## Auto-Compaction

- **Description**: LLM-based context window management that summarises old messages instead of silently dropping them
- **Module**: `nono/agent/compaction.py`
- **Status**: ✅ Stable
- **Differential**: Other frameworks either have no compaction or require external libraries. Nono provides pluggable strategies with graceful fallbacks.

### Strategies

| Strategy | Trigger | Description |
|----------|---------|-------------|
| *(default)* | Message count > max | `_prune_messages` sliding window — drop oldest messages |
| `SummarizationStrategy` | Message count > ratio × max | Summarise middle messages via LLM call |
| `TokenAwareStrategy` | Estimated tokens > ratio × max_context_tokens | Token-count-aware summarisation |
| `CallableStrategy` | `len(messages) > max` (or custom) | Wraps a plain function `(msgs, max) → msgs` |

### Key Properties

| Feature | Description |
|---------|-------------|
| Pluggable | `compaction=True`, `compaction=strategy_instance`, `compaction=my_func`, `compaction="pkg.mod.Class"`, or `compaction=False` |
| Lazy injection | Agent's own service is injected on first compaction |
| Graceful fallback | On LLM failure, naive text summary is used |
| Zero overhead default | `compaction=None` uses `_prune_messages` directly — no extra indirection |
| `CompactionResult` | Metadata: original/compacted count, summary text, estimated tokens saved |

---

## Token-Level Streaming

- **Description**: Native token-by-token streaming from LLM connectors through the agent framework to the API layer. Enables real-time text output for chat interfaces and progressive rendering.
- **Modules**: `nono/connector/connector_genai.py`, `nono/agent/llm_agent.py`, `nono/agent/runner.py`, `nono/server.py`

### Provider Support

| Provider | Streaming Method | Native |
|----------|-----------------|--------|
| OpenAI-compatible (OpenAI, Groq, DeepSeek, xAI, Perplexity, NVIDIA, Hugging Face, GitHub Models, OpenRouter) | SSE `stream=True` | ✅ |
| Gemini | `generate_content_stream()` SDK | ✅ |
| Cerebras | SDK `stream=True` | ✅ |
| Ollama | NDJSON `stream=True` | ✅ |
| Azure AI Foundry | Fallback (full response) | ⚠️ |
| Vercel AI SDK | Fallback (full response) | ⚠️ |

### Key Components

| Component | Description |
|-----------|-------------|
| `GenerativeAIService.generate_completion_stream()` | Base class method; default yields full response as single chunk |
| `TEXT_CHUNK` event type | New `EventType` member for per-token events |
| `LlmAgent._call_llm_stream()` | Delegates to connector's streaming method |
| `LlmAgent._run_stream_impl()` | Tool calls non-streaming; final answer streams token-by-token |
| `Runner.stream_text()` | Public API for token-level streaming |
| `POST /agent/{name}/stream/text` | SSE endpoint with 4096-item async queue |

### Streaming Tool Calls

| Component | Description |
|-----------|-------------|
| `StreamChunk` | Frozen dataclass: `type` ("text" / "tool_call" / "finish"), `content`, `tool_index`, `tool_call_id`, `tool_name`, `finish_reason` |
| `generate_stream()` | Base-class + native overrides yield `StreamChunk` objects with text and tool-call deltas |
| `TOOL_CALL_CHUNK` event | Per-fragment event carrying `tool_index`, `tool_name`, `arguments_delta` |
| Incremental arguments | Tool-call arguments arrive as JSON fragments, accumulated and parsed on finish |
| Fallback detection | Providers without native streaming tool calls fall back to text-based `parse_tool_calls` |

---

## Agent Execution Model

- **Description**: Infrastructure for autonomous multi-agent execution — typed task assignment, worker lifecycle management, failure classification with recovery, machine-enforced policies, progressive verification, git worktree isolation, stale-branch detection, conversation checkpoints, and plan-mode exploration.
- **Module**: `nono/agent/execution.py`
- **Source**: Inspired by [Claude Code's agent execution model](https://www.anthropic.com)
- **Differential**: No other framework provides a complete, integrated execution model combining task packets, state machines, failure taxonomy, executable policies, verification contracts, and worktree isolation in a single module.

### Key Components

| Component | Class(es) | Purpose |
|-----------|-----------|---------|
| Typed Task Packets | `TaskPacket`, `EscalationPolicy`, `ReportingContract` | Structured task assignment with scope, acceptance tests, branch/commit policies; JSON round-trip via `to_json()`/`from_json()` |
| Worker State Machine | `WorkerStateMachine`, `WorkerState`, `WorkerTransition` | 8-state lifecycle (`SPAWNING` → `FINISHED`/`FAILED`); thread-safe transitions, validation, history, listener callbacks |
| Failure Taxonomy | `FailureClassifier`, `FailureCategory`, `ClassifiedFailure`, `RecoveryRecipe` | 10 failure categories with keyword heuristics + custom classifiers; maps each failure to a recovery recipe with max attempts and backoff |
| Executable Policies | `PolicyEngine`, `PolicyRule`, `CallablePolicy` | 5 built-in policies; customisable via subclassing, callable adapter, `register_type()` decorator; enable/disable/replace/unregister at runtime; priority ordering; `from_config()`/`to_config()` serialisation |
| Verification Contract | `VerificationContract`, `VerificationLevel`, `VerificationResult` | 4-level progressive verification (`TARGETED_TEST` → `MERGE_READY`); customisable commands per level; stop-on-failure |
| Worktree Manager | `WorktreeManager`, `WorktreeInfo`, `WorktreeError`, `WorkspaceMismatchError` | Git worktree isolation per session — create/remove/validate; prevents phantom completions |
| Stale-Branch Detector | `StaleBranchDetector`, `BranchStatus` | Detect branches behind reference; auto-fix via `merge_forward()` or `rebase()` with conflict abort |
| Conversation Checkpoints | `ConversationCheckpointManager`, `ConversationCheckpoint` | Save/rewind agent conversation state; per-session history with configurable max checkpoints |
| Plan Mode | `PlanModeAgent`, `PlanResult` | Read-only exploration — restricts to read-only tools, injects plan instruction, extracts numbered steps |

### Built-in Policies

| Policy | Trigger | Action |
|--------|---------|--------|
| `AutoMergePolicy` | All tests pass + no conflicts | Approve auto-merge |
| `StaleBranchPolicy` | Branch behind reference by N commits | Require merge-forward |
| `StartupRecoveryPolicy` | Worker failed on startup | Retry with clean state |
| `LaneCompletionPolicy` | All tasks in swim-lane complete | Allow lane merge |
| `DegradedModePolicy` | Persistent failures exceed threshold | Enter degraded mode |

### PolicyEngine Extensibility

| Feature | Mechanism |
|---------|-----------|
| Subclass | Inherit from `PolicyRule`, implement `evaluate()` |
| Callable | `engine.register_callable("name", fn, priority=n)` |
| Decorator | `@PolicyEngine.register_type()` on a `PolicyRule` subclass |
| Import path | `from_config()` with `{"type": "pkg.mod.MyPolicy"}` |
| Runtime toggle | `engine.enable("name")`, `engine.disable("name")` |
| Replace | `engine.replace("name", new_policy)` |
| Priority | Lower `priority` value = evaluated first |

### Failure Categories

| Category | Recovery Action |
|----------|----------------|
| `TOOL_ERROR` | Retry with corrected parameters |
| `PERMISSION_DENIED` | Escalate to human |
| `CONTEXT_OVERFLOW` | Compact and retry |
| `RATE_LIMITED` | Backoff and retry |
| `MODEL_REFUSAL` | Rephrase prompt |
| `PARSE_ERROR` | Retry with format hint |
| `NETWORK_ERROR` | Retry with backoff |
| `TIMEOUT` | Retry with increased timeout |
| `RESOURCE_EXHAUSTED` | Wait and retry |
| `UNKNOWN` | Escalate |

---

## Unified Connector System

### 15 Native Providers

| Provider | Service Class | SDK |
|----------|--------------|-----|
| Google Gemini | `GeminiService` | `google-genai` (native) |
| OpenAI | `OpenAIService` | `openai` |
| Perplexity | `PerplexityService` | OpenAI-compatible |
| DeepSeek | `DeepSeekService` | OpenAI-compatible |
| xAI (Grok) | `XAIService` | OpenAI-compatible |
| Groq | `GroqService` | OpenAI-compatible |
| Cerebras | `CerebrasService` | `cerebras-cloud-sdk` |
| NVIDIA NIM | `NvidiaService` | OpenAI-compatible |
| Hugging Face | `HuggingFaceService` | OpenAI-compatible |
| GitHub Models | `GitHubModelsService` | OpenAI-compatible |
| OpenRouter | `OpenRouterService` | OpenAI-compatible |
| Azure AI | `AzureAIService` | `azure-ai-inference` |
| Vercel AI SDK | `VercelAIService` | OpenAI-compatible |
| Ollama | `OllamaService` | OpenAI-compatible |

- **Module**: `nono/connector/`
- **Status**: ✅ Stable
- **Differential**: Switch provider with one config change — same code runs on all 15. No LiteLLM or wrapper needed.

### Provider-Specific Features

Beyond the unified API, several providers expose native capabilities:

| Provider | Feature | Description |
|----------|---------|-------------|
| **Gemini** | `response_schema` | Native JSON schema enforcement (not prompt-based) |
| **Gemini** | `system_instruction` | Dedicated system instruction field (not prepended to messages) |
| **Gemini** | `top_k` | Gemini-specific sampling parameter |
| **Perplexity** | `return_citations` | Get source URLs with responses |
| **Perplexity** | `search_domain_filter` | Restrict search to specific domains |
| **Perplexity** | `return_images` | Include images in search responses |
| **Perplexity** | `return_related_questions` | Suggest follow-up questions |
| **Perplexity** | `search_recency_filter` | Filter by content recency |
| **Perplexity** | `json_schema` | Native structured output via `response_format` |
| **OpenRouter** | `plugins` | Web, file-parser, and response-healing plugins |
| **OpenRouter** | `prediction` | Predicted output for latency optimization |
| **OpenRouter** | `fetch_api_key_info()` | Real-time API key info (label, usage, credits, rate limits) |
| **OpenRouter** | `get_rate_limit_status()` | Live rate limit and credit status |

- **Description**: Automatic system prompt injection for structured output formats — the LLM receives format-specific instructions without manual prompt engineering
- **Module**: `nono/connector/connector_genai.py`
- **Status**: ✅ Stable

| Format | Injected Instruction |
|--------|---------------------|
| `TEXT` | (none — default) |
| `JSON` | “Provide output in JSON format” + optional schema enforcement |
| `TABLE` | “Provide output in markdown table format” |
| `CSV` | “Provide output in CSV format with header row” |
| `XML` | “Provide output in valid XML format” |

### Automatic Dependency Management

- **Description**: Safe auto-installation of required SDK packages from an allowlisted set
- **Module**: `nono/connector/connector_genai.py`
- **Status**: ✅ Stable
- **Differential**: Only packages in `_ALLOWED_PIP_PACKAGES` can be installed — prevents arbitrary code execution

| Function | Description |
|----------|-------------|
| `install_library()` | Check import → install from allowlist if missing |
| `_ALLOWED_PIP_PACKAGES` | Frozen set of 12 permitted packages (google-genai, openai, cerebras-cloud-sdk, etc.) |

### Provider Fallback

- **Description**: Automatic failover chain across providers when the primary fails
- **Module**: `nono/connector/fallback.py`
- **Status**: ✅ Stable
- **Differential**: Built-in cascading failover with per-provider retry — no external middleware

| Feature | Description |
|---------|-------------|
| Cascading failover | Try providers in configured order until one succeeds |
| Service LRU cache | Reuse connector instances (limited to 20) to avoid repeated initialization |
| Deduplication | Skip primary provider in fallback chain to avoid double-counting |
| Per-agent integration | `LlmAgent` creates fallback handler lazily — zero overhead when disabled |

### API Manager

- **Description**: Comprehensive API management with rate limiting, circuit breaker, retry policies, health monitoring, and metrics
- **Module**: `nono/connector/api_manager.py`
- **Status**: ✅ Stable
- **Differential**: Enterprise-grade API resilience — circuit breaker + retry + rate limiting in one module

| Component | Description |
|-----------|-------------|
| `APIManager` | Central registry for all API endpoints with configuration |
| `ManagedAPI` | Single API endpoint with rate limiter + circuit breaker + retry |
| `CircuitBreaker` | Fault tolerance with CLOSED → OPEN → HALF_OPEN states |
| `RetryConfig` | Exponential backoff with configurable strategy |
| `APIMetrics` | Real-time success rate, latency, and error tracking |
| `APIConfigPresets` | Pre-built configurations for common providers |

### Rate Limiting

- **Description**: Multi-algorithm rate limiting with real-time status tracking
- **Module**: `nono/connector/connector_genai.py`, `nono/connector/api_manager.py`
- **Status**: ✅ Stable

| Algorithm | Description |
|-----------|-------------|
| Token Bucket | Default — smooth burst-friendly limiting |
| Sliding Window | Precise per-window tracking |
| Fixed Window | Simple time-based counters |

| Limit Type | Description |
|------------|-------------|
| RPM / RPD | Requests per minute / day |
| TPM / TPD | Tokens per minute / day (AI-specific) |
| RPS | Requests per second |
| Concurrent | Max simultaneous requests |

### RateLimitStatus

- **Description**: Real-time rate limit status with usage percentage, credit tracking, and tier detection
- **Module**: `nono/connector/connector_genai.py`
- **Status**: ✅ Stable

| Property | Description |
|----------|-------------|
| `is_rate_limited` | Whether currently at capacity |
| `has_credits` | Whether credits remain (paid APIs) |
| `usage_percentage` | Percentage of limit used (0–100) |
| `credits_remaining` / `credits_used` | Billing tracking |
| `is_free_tier` | Free tier detection |

### Model Features Database

- **Description**: CSV-based model capabilities database with prompt size, rate limits, and feature flags per model
- **Module**: `nono/connector/connector_genai.py`, `nono/connector/model_features.csv`
- **Status**: ✅ Stable
- **Differential**: Automatic context window discovery per model — no manual configuration needed

| Function | Description |
|----------|-------------|
| `get_prompt_size(provider, model)` | Get maximum input size in characters for any model |
| `get_rate_limits(provider, model)` | Get provider-specific rate limits (RPM, RPD, TPM, TPD) |
| `model_features.csv` | Extensible database of model capabilities |

### Temperature Recommendations

- **Description**: Domain-specific temperature presets for common use cases
- **Module**: `nono/connector/connector_genai.py`
- **Status**: ✅ Stable

| Use Case | Temperature | Rationale |
|----------|-------------|----------|
| `coding` | 0.0 | Maximum precision, deterministic |
| `math` | 0.0 | Exact answers |
| `data_cleaning` | 0.1 | High precision for transformations |
| `data_analysis` | 0.3 | Consistency with some flexibility |
| `translation` | 0.3 | Precise and faithful |
| `conversation` | 0.7 | Balance coherence and naturalness |
| `creative` | 1.0 | Higher variability |
| `poetry` | 1.2 | Maximum creativity |

```python
# Use by name instead of magic numbers
service.generate(prompt, temperature="coding")  # → 0.0
service.generate(prompt, temperature="creative") # → 1.0
```

### SSL Configuration

- **Description**: Three SSL modes for corporate/development environments
- **Module**: `nono/connector/connector_genai.py`
- **Status**: ✅ Stable

| Mode | Use Case |
|------|----------|
| `CERTIFI` | Production — system certificates |
| `INSECURE` | Development/testing only |
| `CUSTOM` | Corporate proxy with custom CA |

---

## Built-in Tool Collections

### Standard Tools

| Collection | Tools | Description |
|------------|-------|-------------|
| **DateTime** | `current_datetime`, `convert_timezone`, `days_between`, `list_timezones` | Date, time, and timezone utilities |
| **Text** | `text_stats`, `extract_urls`, `extract_emails`, `find_replace`, `truncate_text`, `transform_text` | String analysis and manipulation |
| **Web** | `fetch_webpage`, `fetch_json`, `check_url` | HTTP fetch and URL verification |
| **Python** | `calculate`, `run_python`, `format_json` | Safe math evaluation, sandboxed code execution, JSON formatting |
| **OfficeBridge** | `ob_convert_document`, `ob_read_document`, `ob_create_word`, `ob_create_excel`, `ob_read_excel`, `ob_translate_document`, `ob_censor_document`, `ob_create_html` | Document automation and office file management |
| **ShortFx** | `fx_future_value`, `fx_present_value`, `fx_add_time`, `fx_is_valid_date`, `fx_vlookup`, `fx_calculate`, `fx_find_positions`, `fx_text_similarity` | 3,000+ deterministic functions for math, finance, dates, strings |

- **Module**: `nono/agent/tools/`
- **Status**: ✅ Stable
- **Convenience lists**: `DATETIME_TOOLS`, `TEXT_TOOLS`, `WEB_TOOLS`, `PYTHON_TOOLS`, `OFFICEBRIDGE_TOOLS`, `SHORTFX_TOOLS`, `ALL_TOOLS` for bulk registration

---

## ShortFx Integration

- **Description**: Deep integration with the ShortFx library, exposing 3,000+ financial, mathematical, date, and lookup functions as agent tools with semantic search
- **Module**: `nono/agent/tools/shortfx_tools.py`
- **Status**: ✅ Stable
- **Differential**: No other AI framework ships native access to 3,000+ Excel-like functions with semantic discovery

| Integration Model | Description |
|-------------------|-------------|
| `SHORTFX_TOOLS` | 8 pre-wrapped functions (future value, present value, add time, validate date, VLOOKUP, calculate, find positions, text similarity) |
| `SHORTFX_DISCOVERY_TOOLS` | 4 meta-tools (`list_shortfx`, `search_shortfx`, `inspect_shortfx`, `call_shortfx`) including semantic search |
| `shortfx_mcp_tools()` | Connect to ShortFx's built-in MCP server as a subprocess |
| `ShortFxSkill` | Reusable skill wrapping discovery tools with an inner LLM agent — supports `.run()`, `.as_tool()`, and `skills=` attachment |

---

## OfficeBridge Integration

- **Description**: Deep integration with the OfficeBridge library, providing document automation (Word, PDF, HTML, Markdown, Text, Excel), AI-powered translation, PII censoring, and format conversion as agent tools
- **Module**: `nono/agent/tools/officebridge_tools.py`
- **Status**: ✅ Stable
- **Differential**: No other AI framework provides native document automation tools — agents can read, create, convert, translate, and censor documents autonomously

| Integration Model | Description |
|-------------------|-------------|
| `OFFICEBRIDGE_TOOLS` | 8 pre-wrapped functions (convert, read, create Word/HTML/Excel, translate, censor) |
| `OFFICEBRIDGE_DISCOVERY_TOOLS` | 3 meta-tools (`list_officebridge`, `inspect_officebridge`, `call_officebridge`) to browse and execute any capability |
| `OfficeBridgeSkill` | Reusable skill wrapping document tools with an inner LLM agent — supports `.run()`, `.as_tool()`, and `skills=` attachment |

### Supported Operations

| Category | Operations |
|----------|------------|
| **Document I/O** | Read/create Word (.docx), HTML, Markdown, Text, PDF |
| **Conversion** | Word ↔ PDF ↔ HTML ↔ Markdown ↔ Text (N-to-N via DocumentTree IR) |
| **Excel** | Read/write workbooks, add charts (Pie, Bar, Line), cell formatting |
| **Translation** | AI-powered document translation to any language (via Nono GenAI) |
| **Censoring** | PII detection and redaction (emails, phones, IDs, addresses, financial data) |
| **Viewing** | Open documents with system default viewer |

---

## ShortFx Integration

- **Description**: Deep integration with the ShortFx library, exposing 3,000+ deterministic functions as agent tools with semantic search for function discovery
- **Module**: `nono/agent/tools/shortfx_tools.py`
- **Status**: ✅ Stable
- **Differential**: 3,000+ functions with semantic search, MCP server, and broad domain coverage

| Integration Model | Description |
|-------------------|-------------|
| `SHORTFX_TOOLS` | 8 pre-wrapped functions (future value, present value, add time, validate date, VLOOKUP, calculate, find positions, text similarity) |
| `SHORTFX_DISCOVERY_TOOLS` | 4 meta-tools (`list_shortfx`, `search_shortfx`, `inspect_shortfx`, `call_shortfx`) including semantic search |
| `shortfx_mcp_tools()` | Connect to ShortFx's built-in MCP server as a subprocess |
| `ShortFxSkill` | Reusable skill wrapping discovery tools with an inner LLM agent |

### Function Modules

| Module | Functions | Domain |
|--------|-----------|--------|
| **fxDate** | Date conversions, operations, evaluations, system dates | Date/time processing |
| **fxNumeric** | Arithmetic, finance, geometry, trigonometry, statistics, numerical methods | Mathematics and finance |
| **fxString** | Similarity, regex, hashing, Spanish-specific operations | Text processing |
| **fxExcel** | Math, text, lookup, date, financial, engineering, information formulas | Excel-compatible formulas |
| **fxPython** | Conversions, itertools, utilities | Pythonic helpers |
| **fxVBA** | Array, date, financial, string, math functions | VBA/MS Access compatibility |

---

## Workspace System

- **Description**: Declarative I/O description for agents — define input files, URLs, cloud storage, templates, and output destinations in a provider-agnostic way
- **Module**: `nono/agent/workspace.py`
- **Status**: ✅ Stable
- **Differential**: Same workspace definition works locally, in sandbox, or in cloud — no code changes per environment

| Entry Type | Description |
|------------|-------------|
| `FileEntry` | Local file or directory |
| `URLEntry` | Remote resource to fetch |
| `InlineEntry` | Literal data embedded in the workspace |
| `CloudStorageEntry` | S3 / GCS / Azure Blob / Cloudflare R2 reference |
| `TemplateEntry` | Jinja2 template rendered at resolve time |
| `OutputEntry` | Declared output destination |
| `Workspace` | Container grouping inputs and outputs |

### Sandbox Bridge

- **Description**: Convert a `Workspace` into a sandbox-ready `Manifest` with automatic entry type mapping
- **Differential**: Same workspace works locally and in sandbox — `to_manifest()` handles the translation

| Workspace Entry | Sandbox Mapping |
|-----------------|----------------|
| `FileEntry` (file) | `LocalFile` |
| `FileEntry` (directory) | `LocalDir` |
| `CloudStorageEntry` (S3) | `S3Bucket` |
| `CloudStorageEntry` (GCS) | `GCSBucket` |
| `CloudStorageEntry` (Azure) | `AzureBlob` |
| `CloudStorageEntry` (R2) | `CloudflareR2` |
| `OutputEntry` | `OutputDir` |

---

## Dynamic Agent Factory

- **Description**: Generate fully configured agents from natural-language descriptions using LLM, with security controls and human-in-the-loop review
- **Module**: `nono/agent/agent_factory.py`
- **Status**: ✅ Stable
- **Differential**: No equivalent in LangChain, CrewAI, or AutoGen — describe what you need, get a production agent

| Component | Description |
|-----------|-------------|
| `AgentFactory` | Modular pipeline to generate agents from descriptions |
| `AgentBlueprint` | Immutable, reviewable specification before instantiation |
| `SystemPromptGenerator` | LLM-powered description → system prompt converter |
| `ToolSelector` | LLM-assisted or keyword-based tool matching from a pool |
| `AgentConfigurator` | Validates and assembles blueprints with security constraints |
| `OrchestrationSelector` | Recommends the best orchestration pattern for a task |
| `OrchestrationBlueprint` | Immutable spec for multi-agent pipelines |
| `OrchestrationRegistry` | Extensible pattern registry with 15 built-in patterns |
| `create_agent_from_prompt()` | One-liner convenience function with optional `review_callback` |

### Security Controls

| Control | Description |
|---------|-------------|
| Prompt injection sanitisation | Blocks role hijacking, instruction override, system prompt leaks |
| Tool allowlist | Restrict which tools agents can access |
| Provider restrictions | Limit which providers agents can use |
| Max tools per agent | Cap tool count per generated agent |
| Instruction length limit | Prevent excessively long system prompts |
| Config flag | `allow_dynamic_creation = false` by default (opt-in) |

---

## MCP Client Integration

- **Description**: Connect to external Model Context Protocol servers and convert remote tools into Nono `FunctionTool` instances
- **Module**: `nono/connector/mcp_client.py`
- **Status**: ✅ Stable
- **Differential**: Native MCP support — connect to any MCP server (stdio, HTTP, SSE) and use remote tools as local agent tools

| Component | Description |
|-----------|-------------|
| `MCPClient` | Client with factory methods: `.stdio()`, `.http()`, `.sse()` |
| `mcp_tools()` | Convenience function to discover and return tools in one call |
| `MCPServerConfig` | Frozen dataclass for server connection parameters |
| `MCPManager` | Central registry — load from config, add, remove, enable/disable, query tools |
| Schema normalization | Automatic conversion of MCP tool schemas to OpenAI-compatible JSON schemas |
| Response parsing | Intelligent extraction of text, structured JSON, or resource references from `CallToolResult` |
| Config integration | `[[mcp.servers]]` section in `config.toml` for persistent server declarations |
| CLI commands | `nono mcp list|add|remove|enable|disable|tools` |

---

## Sandbox Execution

- **Description**: Delegate code execution to external cloud sandbox environments instead of the local machine
- **Module**: `nono/sandbox/`
- **Status**: ✅ Stable
- **Differential**: 7 cloud sandbox providers with a unified API — no vendor lock-in

| Component | Description |
|-----------|-------------|
| `SandboxAgent` | Agent that delegates code execution to external sandboxes |
| `BaseSandboxClient` | Abstract base class for all sandbox providers (execute, snapshot, restore, terminate) |
| `SandboxRunConfig` | Configuration dataclass (provider, timeout, environment, packages, working directory, manifest) |
| `SandboxResult` | Structured result with status, stdout/stderr, exit code, output files, sandbox ID, snapshot ID |

### Supported Providers

| Provider | Client | SDK |
|----------|--------|-----|
| E2B | `E2BSandboxClient` | `e2b-code-interpreter` |
| Modal | `ModalSandboxClient` | `modal` |
| Daytona | `DaytonaSandboxClient` | `daytona-sdk` |
| Blaxel | `BlaxelSandboxClient` | `blaxel` |
| Cloudflare | `CloudflareSandboxClient` | `cloudflare` |
| Runloop | `RunloopSandboxClient` | `runloop-api-client` |
| Vercel | `VercelSandboxClient` | `httpx` |

### Manifest System

| Source | Description |
|--------|-------------|
| `LocalDir` / `LocalFile` | Mount local filesystem paths |
| `S3Bucket` | Mount AWS S3 bucket or prefix |
| `GCSBucket` | Mount Google Cloud Storage bucket or prefix |
| `AzureBlob` | Mount Azure Blob Storage container or prefix |
| `CloudflareR2` | Mount Cloudflare R2 bucket or prefix |
| `OutputDir` | Declare output directory for artefact collection |

---

## Lifecycle Hooks Engine

- **Description**: Thread-safe hook system that executes custom logic at lifecycle points — supports 6 execution types: Python functions, shell commands, GenAI prompts, JSON tasks, skills, and tools
- **Module**: `nono/hooks.py`
- **Status**: ✅ Stable
- **Differential**: No other AI framework supports hooks that execute LLM prompts, GenAI tasks, skills, and tools — not just scripts. Compatible with Claude Code and VS Code Copilot hook protocols.

### 6 Hook Execution Types

| Type | Description | Example |
|------|-------------|---------|
| `function` | Python callable `(HookContext) → HookResult` | `Hook(fn=my_validator)` |
| `command` | Shell command executed as subprocess | `Hook(command="echo done")` |
| `prompt` | Inline GenAI prompt sent to an LLM provider | `Hook(prompt="Validate: {tool_response}", provider="google")` |
| `task` | Predefined JSON task from `prompts/` directory | `Hook(task="review_output")` |
| `skill` | Registered skill from the skill registry | `Hook(skill="summarize_skill")` |
| `tool` | Registered `FunctionTool` invoked with context-derived args | `Hook(tool="security_scanner", tool_args={"target": "{tool_name}"})` |

### 15 Lifecycle Events

| Event | Description |
|-------|-------------|
| `SessionStart` | Fired when a session begins |
| `SessionEnd` | Fired when a session closes |
| `UserPromptSubmit` | Fired when user submits a prompt, before processing |
| `PreToolUse` | Before a tool is called — can block or modify tool input |
| `PostToolUse` | After a tool completes — can inject additional context |
| `PreAgentRun` | Before an agent starts executing |
| `PostAgentRun` | After an agent completes |
| `PreLLMCall` | Before an LLM API call is made |
| `PostLLMCall` | After an LLM API call returns |
| `PreCompact` | Before conversation context is compacted/pruned |
| `SubagentStart` | When a sub-agent is spawned |
| `SubagentStop` | When a sub-agent completes |
| `Stop` | When the agent session ends (final response) |
| `Notification` | When the system generates a notification |
| `Error` | When an error occurs during execution |

### Infrastructure

| Component | Description |
|-----------|-------------|
| `HookManager` | Thread-safe registry to load, store, and fire hooks |
| `HookContext` / `HookResult` | Structured input/output following Claude Code JSON protocols |
| Config loading | `load_config()` / `load_file()` / `load_directory()` — nested and flat config formats |
| Matcher filtering | Regex-based tool name filtering for PreToolUse / PostToolUse hooks |
| Result merging | Most-restrictive-wins semantics (deny > ask > allow) |
| Exit code protocol | Exit 0 = success, exit 2 = blocking error, other = warning |

---

## Workflow Engine

### DAG Pipeline Engine

- **Description**: Directed Acyclic Graph execution with state persistence and conditional branching
- **Module**: `nono/workflows/workflow.py`
- **Status**: ✅ Stable
- **Differential**: Full DAG engine with time-travel debugging — replay from any checkpoint

### Key Capabilities

| Capability | Description |
|------------|-------------|
| `step()` / `connect()` | Define nodes and edges in the graph |
| `parallel_step()` | Execute steps concurrently with join barriers |
| `loop_step()` | Repeat steps until exit condition |
| `branch_if()` | Conditional branching based on state predicates |
| `score_gate()` | Conditional branching based on quality scores |
| `checkpointing` | Save/restore workflow state at any point |
| `time-travel` | `replay_from()`, `get_state_at()`, `get_history()` |
| `state_schema` | Typed state with custom `ReducerFn` (append vs replace, like Redux) |
| `lifecycle hooks` | `on_start`, `on_end`, `on_before_step`, `on_after_step`, `on_between_steps` |
| `error recovery` | Per-step `on_error` handler with retry configuration |
| `declarative loading` | Define workflows in JSON/YAML, load at runtime |
| `tasker_node` / `agent_node` | Bridge Tasker tasks and Agents as workflow steps |

### Runtime Graph Manipulation

- **Description**: Modify workflow structure at runtime — no other framework supports live DAG editing
- **Differential**: Insert, remove, replace, or swap steps in a running workflow without rebuilding

| Method | Description |
|--------|-------------|
| `insert_before(ref, name, fn)` | Insert a new step before an existing step |
| `insert_after(ref, name, fn)` | Insert a new step after an existing step |
| `remove_step(name)` | Remove a step and re-wire connections |
| `replace_step(name, fn)` | Replace a step's function while keeping connections |
| `swap_steps(a, b)` | Swap the positions of two steps in the graph |

### Workflow Templates

| Template | Description |
|----------|-------------|
| `sentiment_pipeline` | Analyze text sentiment with multi-step validation |
| `content_pipeline` | Generate, review, and refine content |
| `content_review_pipeline` | Quality assurance with scoring gates |
| `data_enrichment` | Enrich structured data with LLM-powered analysis |

---

## Tasker System

### Task Executor

- **Description**: Execute reusable, schema-validated tasks defined in JSON files
- **Module**: `nono/tasker/genai_tasker.py`
- **Status**: ✅ Stable
- **Differential**: Prompts-as-data — version-controlled JSON assets, not code strings. Change prompts without changing code.

### Jinja Prompt Builder

- **Description**: Template-based prompt construction with variables, conditionals, loops, and custom filters
- **Module**: `nono/tasker/jinja_prompt_builder.py`
- **Status**: ✅ Stable
- **Differential**: Block-based templates, automatic legacy conversion, and token-aware batching in a single builder

| Feature | Description |
|---------|-------------|
| Template files | Load from `.j2` files with configurable search paths |
| Template strings | Build from inline Jinja2 strings |
| Block rendering | Extract `{% block system %}` and `{% block user %}` as separate prompts |
| Legacy conversion | Automatic `{placeholder}` → `{{ placeholder }}` for backward compatibility |
| Token-aware batching | Split large datasets across multiple prompts respecting token limits |
| Overlap items | Configurable item overlap between batches for context continuity |

### Custom Jinja2 Filters

| Filter | Description |
|--------|-------------|
| `to_compact_json` | Convert to compact JSON (no whitespace) |
| `to_pretty_json` | Convert to pretty-printed JSON |
| `truncate` | Truncate string to length with suffix |
| `escape_quotes` | Escape double quotes |
| `numbered_list` | Format items as numbered list |
| `bullet_list` | Format items as bullet list |

### Batch Processing

- **Description**: High-volume request processing with concurrency control and TSV format (70% token savings)
- **Module**: `nono/connector/genai_batch_processing.py`
- **Status**: ✅ Stable
- **Differential**: TSV batch format reduces token consumption by 70% vs JSON

### Native Provider Batch APIs

- **Description**: Asynchronous large-scale processing via Google Gemini and OpenAI Batch APIs at 50% cost reduction
- **Module**: `nono/connector/genai_batch_processing.py`
- **Status**: ✅ Stable
- **Differential**: Native batch API support for both providers — not just local concurrency, actual provider-side batch jobs

| Provider | Capabilities |
|----------|-------------|
| **Gemini** (`GeminiBatchService`) | Inline batch (< 20 MB), file batch (< 2 GB), Google Search integration, structured JSON output, image generation batch |
| **OpenAI** (`OpenAIBatchService`) | JSONL file-based batch jobs, job monitoring with polling and callbacks |

| Helper | Description |
|--------|-------------|
| `build_requests_from_prompts()` | Build batch requests from a list of prompts |
| `build_request_with_search()` | Build a request with Google Search tool integration |
| `build_structured_request()` | Build a request with JSON schema validation |
| `build_image_request()` | Build a batch request for image generation |
| `create_jsonl_file()` | Create JSONL file for OpenAI batch submission |
| `wait_for_completion()` | Poll until batch job completes |

---

## Decision Wizard

### Complexity Ladder

- **Description**: Interactive wizard that analyzes a task and recommends the optimal orchestration pattern (Level 0–5)
- **Module**: `nono/wizard.py`
- **Status**: ✅ Stable
- **Differential**: No equivalent in LangChain, CrewAI, or AutoGen — guides users from simple prompts to complex orchestration

### Capabilities

| Function | Description |
|----------|-------------|
| `recommend()` | Programmatic recommendation from answers dict |
| `recommend_interactive()` | Interactive CLI questionnaire |
| `suggest_next()` | Suggest escalation to next complexity level |
| `suggest_simpler()` | Suggest simplification when over-engineering |
| `complexity_for_agent()` | Calculate complexity score for an agent tree |
| `ComplexityBudget` | Enforce complexity limits on agent hierarchies |
| `audit_agent_tree()` | Audit an agent tree for complexity violations |

---

## Project System

- **Description**: Isolated, self-contained project directories with `nono.toml` manifests
- **Module**: `nono/project.py`
- **Status**: ✅ Stable
- **Differential**: Git-style project discovery with per-project skills, templates, prompts, and config — no monolithic setup

| Function | Description |
|----------|-------------|
| `init_project()` | Scaffold a new project directory |
| `load_project()` | Load an existing project from manifest |
| `list_projects()` | Discover all projects in a directory tree |
| Per-project `config.toml` | Override provider/model settings per project |

---

## Code Execution

### GenAI Executer

- **Description**: Sandboxed execution of LLM-generated Python and Bash code
- **Module**: `nono/executer/genai_executer.py`
- **Status**: ✅ Stable
- **Differential**: Built-in secure sandbox — no external Docker or E2B setup required

| Mode | Description |
|------|-------------|
| `subprocess` | Isolated subprocess execution (default) |
| `exec` | In-process execution (faster, less isolated) |
| `safe` | Security controls enabled |
| `permissive` | Reduced restrictions for trusted environments |

---

## API Server

- **Description**: FastAPI-based REST/SSE server exposing pre-registered resources as HTTP endpoints
- **Module**: `nono/server.py`
- **Status**: ✅ Stable
- **Differential**: Named-resource invocation only — no raw prompt injection via API. Every call references a pre-registered task, agent, or workflow by name.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /info` | List available tasks, agents, workflows, projects |
| `POST /task/{name}` | Run a built-in JSON task |
| `POST /agent/{name}` | Run a pre-built agent |
| `POST /agent/{name}/stream` | Stream agent events via SSE |
| `POST /workflow/{name}` | Run a pre-built workflow |
| `POST /skill/{name}` | Run a registered skill |
| `GET /routines` | List all routines and their status |
| `POST /routine/{name}/fire` | Fire a routine with optional context |
| `GET /routine/{name}/history` | Get routine execution history |
| `POST /routine/{name}/pause` | Pause a routine |
| `POST /routine/{name}/resume` | Resume a paused routine |
| `GET /projects` | List discovered projects |
| `GET /project/{name}` | Get project details (resources, manifest) |

### Security

| Control | Description |
|---------|-------------|
| Request body size limit | Automatic 1 MB limit to prevent DoS |
| Named-resource only | No raw prompt injection — all calls reference pre-registered resources |

---

## Routines — Autonomous Execution Infrastructure

- **Description**: Saved configurations (agent/workflow/callable + triggers) that execute autonomously on schedule, events, webhooks, or manual fire. Inspired by Claude Code Routines.
- **Module**: `nono/routines/`
- **Status**: ✅ Stable
- **Differential**: Local-first autonomous execution — no cloud dependency. Supports agents, workflows, and callables as executables with built-in retry, history, and lifecycle callbacks.

### Core Components

| Component | Description |
|-----------|-------------|
| `Routine` | Saved configuration: executable + triggers + config |
| `RoutineRunner` | Central coordinator: registration, scheduling, execution |
| `RoutineConfig` | Timeout, retries, environment, tags |
| `RoutineResult` | Execution outcome with timing and output |
| `RoutineRunRecord` | Persistent history record for audit |
| `RoutineStore` | JSON-based persistence for routine definitions |

### Trigger Types

| Trigger | Description |
|---------|-------------|
| `ScheduleTrigger` | Cron expression (5-field) or fixed interval |
| `EventTrigger` | Application events with optional payload filter |
| `WebhookTrigger` | HTTP POST with HMAC-SHA256 validation |
| `ManualTrigger` | Explicit `fire()` call |

### Routine Lifecycle

| Status | Description |
|--------|-------------|
| `IDLE` | Registered but scheduler not started |
| `ACTIVE` | Triggers armed, ready to fire |
| `RUNNING` | Currently executing |
| `PAUSED` | Temporarily suspended |
| `ERROR` | Last execution failed |
| `DISABLED` | Permanently deactivated |

### Executable Types

| Type | Dispatch |
|------|----------|
| Agent | Via `Runner` with fresh `Session` per execution |
| Workflow | Direct `.run()` with context as initial state |
| Callable | Called with `**context` keyword arguments |

---

## Observability

### TraceCollector

- **Description**: Built-in tracing for LLM calls, tool usage, and token consumption
- **Module**: `nono/tracing.py`, `nono/agent/tracing.py`
- **Status**: ✅ Stable
- **Differential**: Zero-dependency observability — no Langfuse, LangSmith, or OpenTelemetry setup needed

| Component | Description |
|-----------|-------------|
| `Trace` | Single trace record with timing, tokens, and status |
| `LLMCall` | Structured LLM call log (model, tokens, latency) |
| `ToolRecord` | Tool invocation record (name, args, result) |
| `TokenUsage` | Token consumption tracking (prompt, completion, total) |
| `TraceCollector` | Aggregate collector for session-wide observability |

### Memory-Safe Tracing

- **Description**: Hierarchical eviction limits prevent unbounded memory growth during long-running sessions
- **Differential**: No trace explosion — other frameworks can OOM on long sessions

| Limit | Default | Description |
|-------|---------|-------------|
| `_MAX_TRACE_CHILDREN` | Configurable | Maximum child traces per parent |
| `_MAX_LLM_CALLS` | Configurable | Maximum LLM call records retained |
| `_MAX_TOOL_RECORDS` | Configurable | Maximum tool invocation records retained |

### Rich Trace Metadata

| Field | Description |
|-------|-------------|
| `agent_type` | Type of agent that produced the trace |
| `duration_ms` | Execution duration in milliseconds |
| `TokenUsage` | Estimated input/output token counts |
| `ToolRecord` | Full tool call details (name, arguments, result) |

---

## Visualization

### ASCII Renderer

- **Description**: Terminal-based rendering of workflow DAGs and agent hierarchies
- **Module**: `nono/visualize/visualize.py`
- **Status**: ✅ Stable
- **Differential**: Visualize complex agent trees and workflow graphs directly in the terminal — no Graphviz or web UI needed

| Target | Function | Description |
|--------|----------|-------------|
| Workflows | `draw_workflow()` | Render DAG with steps, branches, loops, parallels, and joins |
| Agents | `draw_agent()` | Render agent orchestration tree with sub-agent hierarchy |

### Agent Icons

80+ agent types mapped to distinct emoji icons for instant visual identification:

| Agent | Icon | Agent | Icon | Agent | Icon |
|-------|------|-------|------|-------|------|
| `LlmAgent` | 🤖 | `SequentialAgent` | ⏩ | `ParallelAgent` | ⏸ |
| `RouterAgent` | 🔀 | `DebateAgent` | ⚔️ | `SwarmAgent` | 🐝 |
| `TreeOfThoughtsAgent` | 🌳 | `MonteCarloAgent` | 🎲 | `GeneticAlgorithmAgent` | 🧬 |
| `CircuitBreakerAgent` | 🔌 | `GuardrailAgent` | 🛡️ | `BudgetAgent` | 💰 |

---

## CLI

| Command | Alias | Description |
|---------|-------|-------------|
| `nono run` | `r` | Execute a prompt, task, or JSON task file |
| `nono agent` | `a` | Run a named agent template |
| `nono workflow` | `wf` | Execute a named workflow |
| `nono skill` | `sk` | Run a registered skill |
| `nono info` | `ls` | List all available tasks, agents, workflows, skills, and MCP servers |
| `nono providers` | — | Show all supported AI providers with default models |
| `nono project` | — | Show current project info (resources, manifest, paths) |
| `nono config init` | — | Generate a minimal or full `config.toml` |
| `nono config show` | — | Display resolved config values with their source (default/file/env/arg) |
| `nono wizard` | — | Interactive Decision Wizard — choose the right pattern |
| `nono mcp list` | — | List configured MCP servers |
| `nono mcp add` | — | Add an MCP server (stdio, http, sse) |
| `nono mcp remove` | — | Remove an MCP server |
| `nono mcp enable/disable` | — | Enable or disable an MCP server |
| `nono mcp tools` | — | List tools from an MCP server |

### Output Formats

| Format | Description |
|--------|-------------|
| `TABLE` | Formatted table (default) |
| `JSON` | Raw JSON output |
| `CSV` | Comma-separated values |
| `TEXT` | Plain text |
| `MARKDOWN` | Markdown-formatted output |
| `SUMMARY` | Condensed summary |

- **Module**: `nono/cli/cli.py`
- **Status**: ✅ Stable
- **Differential**: Full framework access from terminal — run tasks, agents, workflows, skills, manage MCP servers, inspect projects, and configure TOML from the command line

---

## Configuration

### Hierarchical Config

- **Description**: Configuration loading with priority resolution: CLI → Environment → TOML → Defaults
- **Module**: `nono/config/`
- **Status**: ✅ Stable

| Source | Priority | Example |
|--------|----------|---------|
| CLI arguments | Highest | `--provider openai --model gpt-4o` |
| Environment variables | High | `NONO_GOOGLE__DEFAULT_MODEL=gemini-3-flash-preview` |
| `config.toml` | Medium | `[google] default_model = "gemini-3-flash-preview"` |
| Programmatic defaults | Lowest | Built-in fallback values |

### ConfigSchema Validation

- **Description**: Declarative validation for all configuration keys with type checking, choices, and range constraints
- **Module**: `nono/config/config.py`
- **Status**: ✅ Stable
- **Differential**: Most frameworks only support key-value config. Nono validates types, min/max values, and allowed choices at load time.

| Feature | Description |
|---------|-------------|
| Type checking | Validate that values match expected types |
| `choices` | Restrict values to a predefined set |
| `min_value` / `max_value` | Enforce numeric range constraints |
| `required` | Mark fields as mandatory |
| Source tracking | Every resolved value reports its source (default, file, env, arg) |

---

## Infrastructure

### Docker Deployment

- **Description**: Production-ready containerized deployment with Docker Compose
- **Status**: ✅ Stable

### Comprehensive Test Suite

- **Description**: 42 test files covering orchestration, agents, workflows, skills, tracing, hooks, sandbox, MCP, and integrations
- **Status**: ✅ Stable

### Examples

- **Description**: 23 example scripts demonstrating tasks, agents, workflows, batch processing, HITL, data pipelines, and advanced orchestration patterns
- **Module**: `nono/examples/`
- **Status**: ✅ Stable
