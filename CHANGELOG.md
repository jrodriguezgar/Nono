# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

#### Agent Card — A2A Protocol Discovery (`agent/agent_card.py`)

- Implement A2A Agent Card protocol (v0.3) for agent/workflow discovery
- Add `AgentCard`, `AgentSkill`, `AgentCapabilities`, `AgentProvider`, `AgentInterface` data model
- Add `to_agent_card()` for automatic card generation from `BaseAgent` and `Workflow` instances
- Automatic skill extraction from tools, skills, sub-agents, and workflow steps
- Add `serve_agent_card()` HTTP server for `/.well-known/agent-card.json` well-known URI
- Add `save_agent_card()` / `load_agent_card()` for file persistence
- Add `BaseAgent.agent_card()` convenience method on all agent subclasses
- Full A2A-compliant JSON serialization with camelCase field names (§5.5)
- Bidirectional serialization: `to_dict()`, `to_json()`, `from_dict()`, `from_json()`
- Export all Agent Card types from `nono.agent`
- Add `README_agent_card.md` documentation

#### Anthropic Claude Native Connector (`connector_genai.py`)

- Add `AnthropicService` class inheriting from `GenerativeAIService` with native Anthropic SDK
- Support `generate_completion`, `generate_completion_stream`, and `generate_stream` (structured `StreamChunk`)
- Automatic system prompt extraction (Anthropic requires system as top-level parameter, not a message role)
- Lazy SDK initialization with thread-safe client creation
- Tool-call streaming with `input_json_delta` parsing for incremental function-call arguments
- Register `anthropic` provider in `get_service_for_provider` factory and `_PROVIDER_MAP` (aliases: `anthropic`, `claude`)
- Add Claude model entries to `model_features.csv` and `model_rate_limits.csv` (Opus 4, Sonnet 4, Haiku 4, 3.5 Sonnet, 3.5 Haiku)
- Add `anthropic` to `_ALLOWED_PIP_PACKAGES` allowlist and `pyproject.toml` optional dependencies
- Provider count updated from 14 → 15 across README, FEATURES, and module docstrings

#### 29 Multi-Agent Pipelines (`agent/templates/pipelines.py`)

- Expand pipeline catalog from 4 to **29 ready-to-use multi-agent pipelines** across 7 domains
- **Development** (6): `bug_fix`, `refactoring`, `product_development`, `code_review_automation` (fan-out/fan-in with `ParallelAgent`), `performance_optimization`, `test_suite_generation`
- **Architecture** (3): `system_design`, `database_design`, `api_design`
- **Operations** (6): `incident_response`, `devops_deployment`, `cost_optimization`, `observability_setup`, `disaster_recovery`, `migration`
- **Data** (2): `data_quality`, `etl_pipeline_design`
- **AI/ML** (4 `LoopAgent`): `prompt_engineering`, `rag_pipeline_design`, `model_fine_tuning`, `ai_safety_guardrails`
- **Content & Knowledge** (4): `content_documentation`, `research`, `security_audit`, `compliance`
- Add `_agent()` helper factory for lightweight inline pipeline stages
- Update `__init__.py` exports and `__all__` for all 29 pipelines

#### Structured Output & Validation (`connector/structured_output.py`)

- Add `PydanticOutputParser` — parse and validate LLM output against any Pydantic `BaseModel` with automatic JSON schema derivation
- Add `JsonOutputParser` — parse JSON with optional `jsonschema` validation and code-fence extraction
- Add `RegexOutputParser` — extract text via regex capture groups
- Add `CsvOutputParser` — parse CSV with header row and expected-column validation
- Add `StructuredGenerator` — wraps any `GenerativeAIService` + `OutputParser` with automatic retry on parse failure (configurable `max_retries`, repair prompts)
- Add `ParseError` and `MaxRetriesExceededError` exceptions
- Add convenience functions `parse_json()`, `parse_pydantic()`, `parse_csv()` for one-shot parsing
- Add `output_model` parameter to `LlmAgent` — pass a Pydantic `BaseModel` class for automatic structured output validation with retry
- Add `output_parser` parameter to `LlmAgent` — pass any `OutputParser` instance for custom parsing with retry
- Add `output_retries` parameter to `LlmAgent` — configurable retry count (default 2) when structured output validation fails
- Export all structured output classes from `nono.connector`

#### Auto-Compaction (`agent/compaction.py`)

- Add `CompactionStrategy` abstract base class for pluggable message compaction
- Add `SummarizationStrategy` — LLM-based middle-message summarisation with configurable `trigger_ratio`, `keep_recent`, `summary_max_tokens`, and custom `prompt_template`; graceful fallback to naive summary on service failure
- Add `TokenAwareStrategy` — token-count-based trigger variant of `SummarizationStrategy` with configurable `max_context_tokens`
- Add `CompactionResult` dataclass with `original_count`, `compacted_count`, `summary`, `estimated_tokens_saved`
- Add `compaction` parameter to `LlmAgent` — accepts `True` (auto-summarisation), custom `CompactionStrategy`, a plain callable `(msgs, max) -> msgs`, or `False`/`None` (legacy sliding-window prune via `_prune_messages`)
- Add `CallableStrategy` adapter — wraps any `(messages, max_messages) -> messages` callable as a `CompactionStrategy` with optional custom trigger
- Add string import path support — `compaction="pkg.module.ClassName"` dynamically imports and instantiates external `CompactionStrategy` subclasses
- Add `_compact_messages()` method — delegates to strategy or falls back to `_prune_messages`; lazy service injection for LLM-based strategies
- Replace all `_prune_messages` call sites in `_run_impl` and `_run_async_impl` with `_compact_messages`
- Export `CallableStrategy`, `CompactionResult`, `CompactionStrategy`, `SummarizationStrategy`, `TokenAwareStrategy` from `nono.agent`

#### Token-Level Streaming (`connector/connector_genai.py`, `agent/llm_agent.py`, `agent/runner.py`, `server.py`)

- Add `generate_completion_stream()` to `GenerativeAIService` base class — default fallback yields full response as single chunk
- Native token streaming for `OpenAICompatibleService` (SSE `stream=True`), `GeminiService` (SDK `generate_content_stream`), `CerebrasService` (SDK `stream=True`), `OllamaService` (NDJSON `stream=True`)
- Add `TEXT_CHUNK` event type to `EventType` enum
- Add `LlmAgent._call_llm_stream()` — delegates to connector's `generate_completion_stream()`
- Add `LlmAgent._run_stream_impl()` — tool calls use non-streaming `_call_llm()`, final text response streams token-by-token yielding `TEXT_CHUNK` events; assembles full text for structured output parsing
- Add `Runner.stream_text()` — token-level streaming counterpart to `Runner.stream()`; falls back to `_run_impl_traced()` for non-`LlmAgent` agents
- Add `POST /agent/{name}/stream/text` SSE endpoint — async queue bridge with 4096-item buffer

#### Streaming Tool Calls (`connector/connector_genai.py`, `agent/llm_agent.py`)

- Add `StreamChunk` frozen dataclass — typed chunk with `type` ("text" | "tool_call" | "finish"), `content`, `tool_index`, `tool_call_id`, `tool_name`, `finish_reason`
- Add `generate_stream()` to `GenerativeAIService` base class — yields `StreamChunk` objects; default falls back to `generate_completion()` + single text chunk + finish
- Native `generate_stream()` for `OpenAICompatibleService` — parses both `delta.content` and `delta.tool_calls` from SSE chunks; yields incremental tool-call argument fragments
- Native `generate_stream()` for `CerebrasService` — SDK streaming with tool-call delta parsing
- Add `TOOL_CALL_CHUNK` event type to `EventType` enum
- Add `LlmAgent._stream_llm_full()` — delegates to connector's `generate_stream()`
- Rewrite `_run_stream_impl()` to fully stream all LLM calls — tool-call arguments stream incrementally via `TOOL_CALL_CHUNK` events; assembled tool calls executed normally; fallback text-based tool detection for providers without native streaming tool support
- Export `StreamChunk` from `nono.agent`

#### Agent Execution Model (`agent/execution.py`)

- **`TaskPacket`**: Typed alternative to natural-language prompts for autonomous/multi-agent execution — includes `objective`, `scope`, `worktree`, `branch_policy`, `acceptance_tests`, `commit_policy`, `ReportingContract`, and `EscalationPolicy`; full JSON round-trip via `to_json()`/`from_json()`
- **`WorkerStateMachine`**: 8-state lifecycle manager (`SPAWNING` → `TRUST_REQUIRED` → `READY_FOR_PROMPT` → `PROMPT_ACCEPTED` → `RUNNING` → `BLOCKED`/`FINISHED`/`FAILED`); thread-safe transitions with validation, history tracking, and listener callbacks; `InvalidTransitionError` on illegal moves
- **`FailureClassifier`**: 10-class failure taxonomy (`FailureCategory` enum) with keyword-based heuristic classification and extensible custom classifiers via `register_classifier()`; each category maps to a `RecoveryRecipe` with auto-recovery action, max attempts, and backoff
- **`PolicyEngine`**: Machine-enforced executable policy engine with 5 built-in policies (`AutoMergePolicy`, `StaleBranchPolicy`, `StartupRecoveryPolicy`, `LaneCompletionPolicy`, `DegradedModePolicy`); customisable via subclassing `PolicyRule`, `CallablePolicy` adapter for inline functions, `register_type()` decorator, `enable()`/`disable()`/`replace()`/`unregister()` runtime management, `priority` ordering, `from_config()`/`from_callables()`/`from_policies()` factory methods, and `to_config()` serialisation
- **`VerificationContract`**: 4-level green-ness verification (`TARGETED_TEST` → `MODULE_GREEN` → `WORKSPACE_GREEN` → `MERGE_READY`); progressive `verify_up_to()` with stop-on-failure, customisable commands per level, `highest_passed` property
- **`WorktreeManager`**: Git worktree isolation for parallel sessions — `create()` / `remove()` / `validate_cwd()` per session; prevents phantom completions with `WorkspaceMismatchError`
- **`StaleBranchDetector`**: Detect branches behind a reference (`check()` returns `BranchStatus` with `commits_behind`/`commits_ahead`); auto-fix via `merge_forward()` or `rebase()` with conflict abort
- **`ConversationCheckpointManager`**: Agent-conversation-level checkpoints — `save()` snapshots events, state dict, and message history; `rewind()` restores to any checkpoint; per-session history with configurable max checkpoints
- **`PlanModeAgent`**: Read-only exploration wrapper — restricts agent to read-only tools via heuristic or explicit allowlist, injects plan-mode instruction, extracts numbered plan steps from output; returns `PlanResult` with `plan`, `tools_available`, `tools_blocked`

#### Workflow State Management

- Add `JoinPredecessorError` exception for strict join validation
- Add `strict` parameter to `Workflow.join()` — raises `JoinPredecessorError` when required predecessors have not executed (`strict=True`); default `False` preserves backward-compatible warning behaviour
- Add `_ensure_executed_steps()` helper for safe `set` ↔ `list` round-trips in JSON checkpoints

#### Agent State Management Documentation

- Add `README_state_introspection.md` — guide for querying states, transitions, events, and traces during and after execution

#### REST API Introspection (`server.py`)

- Add `introspect` parameter to `POST /workflow/{name}` — returns transitions audit trail, per-step state history, graph description, and ASCII diagram
- Add `introspect` parameter to `POST /agent/{name}` — returns session events and final state
- Add `GET /workflow/{name}/describe` endpoint — returns graph structure, steps, schema, and ASCII diagram without executing the workflow
- Add `TransitionRecord`, `IntrospectionData`, `EventRecord`, and `AgentIntrospectionData` response models

- Document `Session` constructor (`session_id`, `state`, `memory`, `max_events`) in `README_agent.md`
- Document `InvocationContext` fields (`transfer_depth`, `MAX_TRANSFER_DEPTH`) in `README_agent.md`
- Document `SharedContent` capacity limits (200 items, 10 MB/item) in `README_agent.md`
- Add "State Isolation Patterns" section to `README_agent.md` — three levels: `local_content`, `ContextFilterAgent`, new `Session`
- Document thread-safe `state_set`/`state_get`/`state_update` helpers in `README_agent.md`
- Expand `ContextFilterAgent` section in `README_orchestration.md` with custom `filter_fn` example and state isolation guidance
- Add state management FAQ entries to `README_orchestration.md`

### Changed

#### Workflow Execution Loop Refactoring

- Extract `_step_loop_sync()` and `_step_loop_async()` shared generators used by `run()`, `run_async()`, `stream()`, `astream()`, and `replay_from()` — eliminates ~400 lines of duplicated execution logic
- `replay_from()` now uses `_retry_step_sync()`, `_apply_result()`, `_record_step_transition()`, and `_handle_error_recovery()` instead of its own inline copies

### Fixed

#### Workflow State Management

- Fix `__executed_steps__` lost on checkpoint round-trip — `set` is now serialized as a sorted `list` in JSON and reconstructed on `resume()`, `get_checkpoint_at()`, and `run(resume=True)`
- Fix `run()` / `run_async()` emitting a spurious second `trace_collector.end_trace()` on step error that prematurely closed the workflow-level trace

#### Skills + Agents Documentation (`README_skills.md`)

- Add **Tutorial: Building a Complex SKILL.md Skill** — step-by-step guide with scripts, references, assets, and all three usage modes
- Add **How Tools Propagate: Skill → Agent** — detailed explanation of tool scoping across standalone, as-tool, and pipeline modes; clarify `allowed-tools` behavior
- Add **Dynamic Discovery at Runtime** — programmatic examples for auto-attach by tag, user-driven skill selection, hot-reload from directories, and metadata inspection

#### Agent Workspace (`agent/workspace.py`)

- **`Workspace`**: Provider-agnostic declarative description of an agent's input files, data sources, and output destinations — works for local, sandbox, or remote execution
- **`WorkspaceEntry`**: Abstract base for I/O resources with `entry_type()`, `to_dict()`, `is_readable`, `is_writable`
- **`FileEntry`**: Local file or directory input with optional glob filtering
- **`URLEntry`**: Remote HTTP resource to fetch (headers omitted from serialisation for security)
- **`InlineEntry`**: Literal data (string, bytes, dict, list) embedded in the workspace
- **`CloudStorageEntry`**: Unified cloud storage reference (S3, GCS, Azure Blob, Cloudflare R2) via `StorageKind` enum
- **`TemplateEntry`**: Jinja2 template rendered at resolve time with variables
- **`OutputEntry`**: Declared output destination with path and content type
- **`IODirection`** enum: INPUT, OUTPUT
- **`StorageKind`** enum: S3, GCS, AZURE_BLOB, CLOUDFLARE_R2
- Full JSON serialisation: `to_dict()` / `from_dict()` / `to_json()` / `from_json()` round-trip
- Filtered accessors: `file_entries()`, `url_entries()`, `inline_entries()`, `cloud_entries()`, `template_entries()`
- **Bidirectional Manifest bridge**: `Workspace.to_manifest()` converts to sandbox `Manifest`; `Manifest.to_workspace()` converts back — seamless interop between agent-level I/O and sandbox materialisation

#### Extended Hook Types (`hooks.py`)

- **`HookType`** enum: `FUNCTION`, `COMMAND`, `PROMPT`, `TASK`, `SKILL`, `TOOL` — classify hook execution strategy
- **Prompt hooks**: Inline GenAI prompt with `{variable}` template resolution, sent to any configured LLM provider
- **Task hooks**: Execute predefined JSON tasks from the `prompts/` directory with context-resolved templates
- **Skill hooks**: Invoke registered skills from the skill registry with auto-built context input
- **Tool hooks**: Invoke registered `FunctionTool` instances with placeholder-resolved arguments
- **`_SafeFormatDict`**: Safe template resolution — unknown placeholders are left unchanged
- **`_parse_genai_response()`**: Auto-detect structured `HookResult` JSON in LLM responses
- **`_build_skill_input()`**: Build descriptive input messages for skills from `HookContext`
- **`_build_hook_from_type()`**: Factory for constructing hooks from type string and params dict
- Add `provider`, `model`, `prompt`, `task`, `skill`, `tool`, `tool_args` fields to `Hook` dataclass
- Extend `_parse_hook_entry()` to parse all six hook types from JSON configuration
- Extend `HookManager.to_config()` to serialize all six hook types for round-trip fidelity

#### ShortFx Integration (`agent/tools/shortfx_tools.py`)

- **`SHORTFX_TOOLS`**: Curated list of 8 pre-wrapped ShortFx functions (future value, present value, add time, validate date, VLOOKUP, calculate, find positions, text similarity)
- **`SHORTFX_DISCOVERY_TOOLS`**: Four meta-tools (`list_shortfx`, `search_shortfx`, `inspect_shortfx`, `call_shortfx`) to search, inspect, and execute any of ShortFx's 3,000+ functions via registry
- **`shortfx_mcp_tools()`**: Connect to ShortFx's built-in MCP server as a subprocess (`pip install shortfx[mcp]`)
- **`ShortFxSkill`**: Reusable skill wrapping discovery tools with an inner LLM agent — supports standalone `.run()`, `.as_tool()`, and attachment to agents via `skills=`
- **`README_shortfx.md`**: Comprehensive documentation covering all four integration models with decision guide and examples

#### Sandbox Execution (`sandbox/`)

- **`SandboxAgent`**: Agent that delegates code execution to external sandbox environments instead of the local machine
- **`BaseSandboxClient`**: Abstract base class for all sandbox provider integrations (execute, snapshot, restore, terminate)
- **`SandboxRunConfig`**: Configuration dataclass for sandbox execution (provider, timeout, environment, packages, working directory, manifest)
- **`SandboxResult`**: Structured result with status, stdout/stderr, exit code, output files, sandbox ID, and snapshot ID
- **`SandboxProvider`** enum: E2B, Modal, Daytona, Blaxel, Cloudflare, Runloop, Vercel
- **`get_sandbox_client()`**: Factory function to instantiate the correct client for a given provider
- **Manifest system**: Declarative workspace description for sandbox environments
  - **`Manifest`**: Container mapping logical mount paths to concrete sources
  - **`LocalDir`** / **`LocalFile`**: Mount local filesystem paths
  - **`S3Bucket`**: Mount AWS S3 bucket or prefix
  - **`GCSBucket`**: Mount Google Cloud Storage bucket or prefix
  - **`AzureBlob`**: Mount Azure Blob Storage container or prefix
  - **`CloudflareR2`**: Mount Cloudflare R2 bucket or prefix
  - **`OutputDir`**: Declare output directory for artefact collection
- **Provider clients**: E2B (`e2b-code-interpreter`), Modal (`modal`), Daytona (`daytona-sdk`), Blaxel (`blaxel`), Cloudflare (`cloudflare`), Runloop (`runloop-api-client`), Vercel (via `httpx`)
- **Config**: `[sandbox]` section in `config.toml` (default_provider, timeout, working_dir, keep_alive, snapshot)

#### Harness/Compute Separation & Durable Execution (`sandbox/harness.py`, `sandbox/durable_agent.py`)

- **`SandboxSnapshot`**: Serialisable checkpoint capturing code, cursor position, session state, accumulated output, environment, and provider-native snapshot ID — supports `to_dict()`/`from_dict()`/`to_json()`/`from_json()` round-trip
- **`CheckpointStatus`** enum: CREATED, ACTIVE, RESTORED, EXPIRED
- **`SnapshotStore`**: Pluggable persistence for snapshots — in-memory (`SnapshotStore.in_memory()`) or disk-backed (`SnapshotStore.on_disk(dir)`) with JSON files
- **`HarnessRuntime`**: Orchestrator that formally separates **harness** (local state, session, events) from **compute** (remote sandbox). Features:
  - Automatic checkpointing before and after each execution step
  - Retry with rehydration — on sandbox failure, restores state in a fresh container
  - Provider-native snapshot/restore when supported (E2B, Modal)
  - Accumulated stdout/stderr across retry attempts
  - Structured `HarnessEvent` trace of all lifecycle actions
  - `resume()` method to continue from a previously saved snapshot
- **`DurableSandboxAgent`**: Extends `SandboxAgent` with fault-tolerant execution — configurable `max_retries`, `retry_delay`, and `snapshot_store`. Emits harness events as `STATE_UPDATE` events for full observability
- **`HarnessEvent`**: Structured record of harness lifecycle events (checkpoint, execute, restore, retry, completed)

#### Lifecycle Hooks Engine (`hooks.py`)

- **HookManager**: Thread-safe registry to load, store, and fire hooks at lifecycle points (SessionStart, PreToolUse, PostToolUse, PreAgentRun, PostAgentRun, Error, etc.)
- **Hook**: Supports Python callable hooks and shell command hooks with OS-specific overrides, regex matchers, timeout, and environment variables
- **HookContext / HookResult**: Structured input/output following Claude Code hook JSON protocols
- **Config loading**: `load_config()` / `load_file()` / `load_directory()` parse both nested and flat hook config formats
- **Matcher filtering**: Regex-based tool name filtering for PreToolUse / PostToolUse hooks
- **Result merging**: Multiple hooks merge results with most-restrictive-wins semantics (deny > ask > allow)
- **BaseAgent integration**: `hook_manager` parameter and `set_hook_manager()` fluent API; PreAgentRun/PostAgentRun/Error hooks fire automatically in `run()` and `run_async()`
- **LlmAgent integration**: PreToolUse hooks can block or modify tool input; PostToolUse hooks can inject additional context
- **Exit code protocol**: Exit 0 = success, exit 2 = blocking error, other = warning (compatible with Claude Code)

#### Dynamic Agent Factory (`agent/agent_factory.py`)

- **AgentFactory**: Modular pipeline to generate fully configured agents from natural-language descriptions using LLM
- **AgentBlueprint**: Immutable, reviewable specification produced before instantiation — supports serialisation via `to_dict()`/`from_dict()`
- **SystemPromptGenerator**: LLM-powered module that converts a description into a structured system prompt
- **ToolSelector**: Selects tools from a pool using LLM-assisted or keyword-based matching
- **AgentConfigurator**: Validates and assembles blueprints with security constraints (allowlists, injection detection, length limits)
- **create_agent_from_prompt()**: One-liner convenience function with optional `review_callback` for human-in-the-loop approval
- **Prompt injection sanitisation**: Blocks common injection patterns (role hijacking, instruction override, system prompt leaks)
- **Config flag**: `[agent.factory]` section in `config.toml` — `allow_dynamic_creation = false` by default (must be explicitly enabled)
- **Security controls**: tool allowlist, provider restrictions, max tools per agent, instruction length limit
- **OrchestrationSelector**: LLM-powered or keyword-based component that analyses a task and recommends the best orchestration pattern (sequential, parallel, planner, supervisor, etc.)
- **OrchestrationBlueprint**: Immutable spec for multi-agent pipelines — contains pattern, sub-agent blueprints, and pattern kwargs
- **OrchestrationRegistry**: Extensible pattern registry with 15 built-in patterns — register custom orchestration patterns via `register_pattern()` with factory functions, keyword hints, and sub-agent constraints
- **PatternRegistration**: Immutable dataclass describing a registered orchestration pattern (key, class_name, description, keyword_hints, factory, min_sub_agents)
- **register_pattern()**: Module-level convenience to register custom patterns so `AgentFactory` and `OrchestrationSelector` can discover and instantiate them
- **generate_orchestrated_blueprint()**: Analyse a task description and generate a full multi-agent pipeline spec
- **build_orchestrated()**: Instantiate the appropriate workflow agent from an orchestration blueprint
- **OrchestrationRegistry**: Extensible pattern registry — register, unregister, and query orchestration patterns at runtime
- **PatternRegistration**: Immutable dataclass for each registered pattern (key, class name, description, keyword hints, factory, min sub-agents)
- **register_pattern()**: Module-level convenience function to register custom orchestration patterns

#### MCP Client Integration (`connector/mcp_client.py`)

- **MCPClient**: Connect to external Model Context Protocol servers (stdio, Streamable HTTP, SSE) and convert remote tools into Nono `FunctionTool` instances
- **Factory methods**: `MCPClient.stdio()`, `MCPClient.http()`, `MCPClient.sse()` for each transport type
- **mcp_tools()**: Convenience function to discover and return tools in one call with auto-detected transport
- **MCPServerConfig**: Frozen dataclass for server connection parameters
- **MCPManager**: Central registry for MCP servers — load from config, add, remove, enable/disable, query tools
- **Config integration**: `[[mcp.servers]]` section in `config.toml` for persistent server declarations
- **CLI commands**: `nono mcp list|add|remove|enable|disable|tools` for managing MCP servers from the terminal
- **Optional dependency**: `pip install nono[mcp]` adds the `mcp` Python SDK

#### Documentation

- **Tools Guide** (`nono/README_tools.md`): Add comprehensive documentation covering @tool decorator, FunctionTool, ToolContext, built-in tools (transfer_to_agent, Tasker tools, MCP tools, Skills), execution flow, configuration, and ACI quality validation
- **Dynamic Agent Factory Guide** (`nono/agent/README_agent_factory.md`): Add documentation covering single-agent generation, orchestrated multi-agent pipelines, security controls, blueprint review, human-in-the-loop, and extending the factory

#### Built-in Tool Collections (`agent/tools/`)

- **DateTime tools** (`datetime_tools.py`): `current_datetime`, `convert_timezone`, `days_between`, `list_timezones` — date, time, and timezone utilities
- **Text tools** (`text_tools.py`): `text_stats`, `extract_urls`, `extract_emails`, `find_replace`, `truncate_text`, `transform_text` — string analysis and manipulation
- **Web tools** (`web_tools.py`): `fetch_webpage`, `fetch_json`, `check_url` — HTTP fetch and URL verification
- **Python tools** (`python_tools.py`): `calculate`, `run_python`, `format_json` — safe math evaluation, sandboxed code execution, JSON formatting
- **Category lists**: `DATETIME_TOOLS`, `TEXT_TOOLS`, `WEB_TOOLS`, `PYTHON_TOOLS`, `ALL_TOOLS` for convenient bulk registration

#### Agent Framework (NAA — Nono Agent Architecture)

- **Core agent** (`LlmAgent`): LLM-powered agent with tool-calling loops and sub-agent delegation
- **Runner**: Session lifecycle manager with state updates and event streaming
- **Base classes**: `BaseAgent` abstract class and shared agent infrastructure
- **Human-in-the-Loop** (`hitl.py`): Standardized protocols for pausing AI execution and requesting human approval or feedback
- **Human input** (`agent/human_input.py`): Interactive input collection within agent execution loops
- **KeepInMind integration** (`agent/keepinmind.py`): Long-term memory integration for agents via the KeepInMind library
- **Tasker tool** (`agent/tasker_tool.py`): Bridge to expose GenAI Tasker tasks as agent-callable tools

#### 100 Orchestration Agents (`agent/workflow_agents.py`)

- **Flow control**: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `RouterAgent`
- **Collective reasoning**: `MapReduceAgent`, `ConsensusAgent`, `VotingAgent`, `DebateAgent`
- **Quality assurance**: `ProducerReviewerAgent`, `GuardrailAgent`, `BestOfNAgent`, `SelfRefineAgent`, `SelfConsistencyAgent`, `VerifierAgent`, `RecursiveCriticAgent`
- **Hierarchical**: `SupervisorAgent`, `EscalationAgent`, `HierarchicalAgent`, `OrchestratorWorkerAgent`
- **Delegation**: `HandoffAgent`, `GroupChatAgent`, `SwarmAgent`
- **Batch & scaling**: `BatchAgent`, `CascadeAgent`, `LoadBalancerAgent`, `EnsembleAgent`, `DynamicFanOutAgent`, `PipelineParallelAgent`
- **Advanced reasoning**: `TreeOfThoughtsAgent`, `GraphOfThoughtsAgent`, `MonteCarloAgent`, `BeamSearchAgent`, `AnalogicalReasoningAgent`, `ThreadOfThoughtAgent`, `BufferOfThoughtsAgent`, `ChainOfAbstractionAgent`, `ProgOfThoughtAgent`
- **Planning**: `PlannerAgent`, `AdaptivePlannerAgent`, `SubQuestionAgent`, `SkeletonOfThoughtAgent`, `LeastToMostAgent`, `SelfDiscoverAgent`, `AgendaAgent`
- **Prompting strategies**: `ChainOfDensityAgent`, `StepBackAgent`, `CoVeAgent`, `SocraticAgent`, `ReflexionAgent`, `ContextFilterAgent`, `RephraseAndRespondAgent`, `ExpertPromptingAgent`, `PromptChainAgent`
- **Resilience**: `CircuitBreakerAgent`, `TimeoutAgent`, `SagaAgent`, `CheckpointableAgent`, `BacktrackingAgent`
- **Evolutionary & optimisation**: `GeneticAlgorithmAgent`, `MultiArmedBanditAgent`, `TournamentAgent`, `SimulatedAnnealingAgent`, `TabuSearchAgent`, `ParticleSwarmAgent`, `DifferentialEvolutionAgent`, `BayesianOptimizationAgent`, `AntColonyAgent`
- **Meta**: `MetaOrchestratorAgent`, `MixtureOfExpertsAgent`, `MixtureOfAgentsAgent`, `BlackboardAgent`, `MixtureOfThoughtsAgent`
- **Multi-agent communication**: `RolePlayingAgent`, `GossipProtocolAgent`, `AuctionAgent`, `DelphiMethodAgent`, `NominalGroupAgent`, `ContractNetAgent`
- **Retrieval-augmented**: `ActiveRetrievalAgent`, `IterativeRetrievalAgent`, `DemonstrateSearchPredictAgent`
- **Observability**: `ShadowAgent`, `CompilerAgent`, `SpeculativeAgent`
- **Resource management**: `CacheAgent`, `BudgetAgent`, `CurriculumAgent`, `PriorityQueueAgent`, `MemoryConsolidationAgent`
- **Mediation & learning**: `MediatorAgent`, `DivideAndConquerAgent`, `CumulativeReasoningAgent`, `MultiPersonaAgent`, `RedTeamAgent`, `FeedbackLoopAgent`, `WinnowingAgent`, `InnerMonologueAgent`, `HypothesisTestingAgent`, `SkillLibraryAgent`, `DoubleLoopLearningAgent`

#### Agent Templates (`agent/templates/`)

- Pre-configured agent templates: `classifier`, `coder`, `decomposer`, `extractor`, `guardrail`, `planner`, `reviewer`, `summarizer`, `writer`
- `pipelines.py`: **29 multi-agent pipelines** across 7 domains — development (`bug_fix`, `refactoring`, `product_development`, `code_review_automation`, `performance_optimization`, `test_suite_generation`), architecture (`system_design`, `database_design`, `api_design`), operations (`incident_response`, `devops_deployment`, `cost_optimization`, `observability_setup`, `disaster_recovery`, `migration`), data (`data_quality`, `etl_pipeline_design`), AI/ML (`prompt_engineering`, `rag_pipeline_design`, `model_fine_tuning`, `ai_safety_guardrails`), content (`content_documentation`, `research`, `security_audit`, `compliance`), plus originals (`plan_and_execute`, `research_and_write`, `draft_review_loop`, `classify_and_route`)

#### Agent Skills (`agent/skills/`)

- Pluggable skill system with `BaseSkill` and `@skill` registry
- Skill loader (`skill_loader.py`): Dynamic skill discovery and registration
- Built-in skills: `classify`, `code_review`, `extract`, `summarize`, `translate`
- SKILL.md skill directories (Anthropic/Claude standard): `classifying-data`, `extracting-data`, `reviewing-code`, `summarizing-text`, `translating-text`, `analyzing-sentiment`, `analyzing-data`, `generating-sql`, `generating-tests`, `generating-api-docs`, `writing-documentation`, `writing-emails`, `explaining-code`, `converting-formats`, `researching-topics`

#### Configuration Consolidation

- **Centralized config.toml**: Consolidate executer, CLI, and agent constants into `config.toml` as sole configuration source
- **`[executer]` section**: `mode`, `security`, `timeout`, `max_retries`, `save_executions`
- **`[cli]` section**: `colors_enabled`, `default_output_format`, `default_log_level`, `default_provider`, `allow_parameter_files`, `require_confirmation`, `dry_run_by_default`, `default_timeout`, `default_max_tokens`
- **`[agent]` extended**: `max_tool_iterations`, `max_loop_messages`, `max_transfer_depth` now configurable via TOML
- **Provider Fallback** (`connector/fallback.py`): Automatic provider/model failover with configurable `[fallback]` chain in `config.toml`
- **Configuration management guide** (`config/README_config.md`): Comprehensive documentation covering all `config.toml` sections, environment variable overrides, Config API, and CLI integration

#### Unified Connector Enhancements

- **Groq** (`GroqService`): Fast inference via OpenAI-compatible API
- **Cerebras** (`CerebrasService`): Native Cerebras Cloud SDK
- **NVIDIA** (`NvidiaService`): NVIDIA NIM via OpenAI-compatible API
- **GitHub Models** (`GitHubModelsService`): GitHub-hosted models via OpenAI-compatible API
- **OpenRouter** (`OpenRouterService`): Multi-provider routing via OpenAI-compatible API
- **Azure AI** (`AzureAIService`): Azure AI Inference client
- **Vercel AI SDK** (`VercelAIService`): Vercel-hosted models via OpenAI-compatible API

#### Workflow Engine (`workflows/`)

- **DAG pipeline engine** (`workflow.py`): Directed Acyclic Graph execution with state persistence, conditional branching, error recovery, audit trail, state schema, and lifecycle hooks
- **Workflow templates**: `sentiment_pipeline`, `content_pipeline`, `content_review_pipeline`, `data_enrichment`

#### Core Modules

- **Tracing** (`tracing.py`): Unified observability for LLM calls, tool usage, and token consumption across all components
- **Agent tracing** (`agent/tracing.py`): Agent-specific structured event tracking
- **Project system** (`project.py`): Self-contained project directories with `nono.toml` manifests, per-project skills and templates
- **Decision Wizard** (`wizard.py`): Interactive "Complexity Ladder" recommending optimal agent/orchestration pattern for a given problem
- **API Server** (`server.py`): FastAPI-based REST/SSE server exposing tasks, agents, and workflows as endpoints
- **Visualization** (`visualize.py`): ASCII rendering of workflow DAGs and agent hierarchies

#### Infrastructure

- **Docker support**: `Dockerfile` and `docker-compose.yml` for containerized deployment
- **LLMs.txt**: Machine-readable project API reference (`llms.txt`)
- **Comprehensive test suite**: 34 test files covering orchestration, agents, workflows, skills, tracing, and integrations


### Changed

- `genai_executer.py`: Remove `config_file` parameter and JSON config loading — `config.toml` is the sole configuration source
- `CLIConfig`: Add `from_config_toml()` class method; CLI defaults now read from `[cli]` section
- `LlmAgent`: Tool-call limits (`_MAX_TOOL_ITERATIONS`, `_MAX_LOOP_MESSAGES`) loaded from `[agent]` in `config.toml`
- `MAX_TRANSFER_DEPTH`: Loaded from `agent.max_transfer_depth` in `config.toml`
- `README_config.md`: Rewrite as comprehensive configuration management guide
- Model catalog (`model_features.csv` + `model_rate_limits.csv`): Merge into config.toml `[models]` section — single inline-table per `"provider/model"` with `prompt_size`, `rpm`, `rpd`, `tpm`, `tpd`
- `_load_model_features()`, `_load_rate_limits()`, `load_models_from_csv()`: Rewrite to read from config.toml `[models]` instead of CSV files
- `task_cli.py`: Rewrite `_load_model_catalog()` to read from config.toml `[models]`

### Removed

- `executer/config.json`: Fully replaced by `[executer]` section in `config.toml`
- `connector/model_features.csv`: Merged into config.toml `[models]`
- `connector/model_rate_limits.csv`: Merged into config.toml `[models]`

---

## [0.1.0] — 2026-02-13

First public release on GitHub. Built from 10 commits spanning initial scaffold through multi-provider AI framework with tasker, executer, CLI, and batch processing.

---

### `d6a90b6` — init: Project scaffold and repository creation (2025-12-22)

Project scaffold and repository creation.

#### Added

- MIT `LICENSE` file
- Initial `README.md` (placeholder)
- `.gitignore` for Python projects
- Basic `Help.md` documentation file

---

### `bd0d78a` — feat: GenAI tasker engine with multi-provider support (2026-02-03)

Complete rewrite of the task execution engine with multi-provider support and structured JSON task definitions.

**24 files changed** | +4,175 −1,716 lines

#### Added

- `nono/genai_tasker/genai_tasker.py` — Core `TaskExecutor` class with multi-provider AI client abstraction (+859 lines rewrite)
- `nono/genai_tasker/connector/connector_genai.py` — Unified connector supporting Gemini, OpenAI, Perplexity, DeepSeek, Grok, Ollama (+789 lines)
- `nono/genai_tasker/connector/connector_genai_ssl.md` — SSL configuration guide (INSECURE, CERTIFI, CUSTOM modes)
- `nono/genai_tasker/connector/README_connector_genai.md` — Connector API documentation (+535 lines)
- `nono/genai_tasker/prompts/product_categorizer.json` — Multi-input product categorization task definition
- `nono/genai_tasker/prompts/task_definition_schema.json` — JSON Schema for task definition files (provider, model, temperature presets, batch_size, top_p, top_k, penalties, stop sequences, etc.)
- `nono/genai_tasker/README.md` — Tasker documentation with features, supported providers, temperature presets table
- `nono/genai_tasker/README_task_configuration.md` — Task configuration guide (+729 lines)
- `nono/genai_tasker/README_technical.md` — Technical API reference (+679 lines)
- `nono/demo_formulite.py` — Demo integration with ShortFx library (then named FormuLite)

#### Changed

- `nono/genai_tasker/prompts/name_classifier.json` — Restructured with `task`, `genai`, `prompts`, `input_schema` sections (new standard format)

#### Removed

- `nono/genai_tasker/delete/` — Purged legacy files: `connector_genai_old.py`, `genai_tasker_Delete.py`, `genai_tasker_old.py`, `name_classifier_Delete.json`
- `nono/genai_tasker/genai_config.json` — Replaced by task-level configuration
- `nono/genai_tasker/api_keys.txt` — Moved to `.gitignore` pattern
- `nono/requeriments.txt` — Replaced by proper dependency management

---

### `513af3f` — feat: Code generation and execution module (2026-02-03)

New code generation and execution module with safe/permissive security modes. Connector extracted to shared location.

**14 files changed** | +1,434 −14 lines

#### Added

- `nono/__init__.py` — Package initialization with `__version__ = "1.0.0"` and submodule documentation
- `nono/genai_executer/genai_executer.py` — `CodeExecuter` class: generates Python code from natural language via LLM, executes in subprocess sandbox (+981 lines)
- `nono/genai_executer/__init__.py` — Exports: `CodeExecuter`, `ExecutionResult`, `ExecutionMode`, `SecurityMode`, `CodeExecutionError`
- `nono/genai_executer/config.json` — Default execution config (subprocess mode, safe security, 30s timeout, 3 retries)
- `nono/genai_executer/README.md` — Documentation: security modes (SAFE blocks writes/network/system; PERMISSIVE allows all), execution history, retry strategy (+375 lines)
- `.gitignore` — Added `nono/genai_executer/executions/` to ignore execution history files

#### Changed

- `nono/genai_tasker/connector/` → `nono/connector/` — Extracted connector to shared location (4 files renamed)
- `nono/genai_tasker/genai_tasker.py` — Updated imports to use shared `..connector`
- `nono/genai_tasker/README.md` — Updated connector paths and project structure diagram
- `nono/genai_tasker/README_technical.md` — Updated file reference table

---

### `0eb56a3` — feat: Batch data processing and keyring credential management (2026-02-04)

Batch data processing with token-efficient TSV format (60-70% savings vs JSON) and OS keyring-based credential management.

**17 files changed** | +1,876 −261 lines

#### Added

- `nono/genai_tasker/data_stage/core.py` — `DataStageExecutor` base class: TSV format (60-70% token savings vs JSON), smart throttling based on context windows, batch record management (+617 lines)
- `nono/genai_tasker/data_stage/operations.py` — Built-in operations: `SemanticLookupOperation` (match values against reference list), `SpellCorrectionOperation` (fix typos with language hints) (+256 lines)
- `nono/genai_tasker/data_stage/__init__.py` — Package exports with feature summary
- `nono/genai_tasker/data_stage/README.md` — Documentation: TSV format rationale, throttling strategy, custom operations guide (+414 lines)
- `README.md` — Comprehensive project README: overview, features table, supported providers (Gemini, OpenAI, Perplexity, DeepSeek, Grok, Ollama), installation guide, usage examples for all submodules, project structure, documentation links (+314 lines)
- `nono/genai_tasker/__init__.py` — Public API exports: `TaskExecutor`, `AIProvider`, `AIConfiguration`, `BaseAIClient`, `GeminiClient`, etc.
- Keyring integration in `genai_tasker.py` and `genai_executer.py`:
  - `_get_api_key_from_keyring()` — Retrieve from OS credential store
  - `_set_api_key_to_keyring()` — Store/update in credential store
  - Resolution order: argument → keyring → key files (with auto-migration to keyring)
- `.gitignore` — Glob patterns for all key/secret files (`**/*key*.txt`, `**/*secret*.json`, etc.)

#### Changed

- `nono/__init__.py` — Version bump to `1.1.0`, added `data_stage` to submodule docs
- `nono/genai_executer/README.md` — Added API key resolution priority table and keyring docs
- `nono/genai_tasker/README.md` — Added API key resolution priority table
- `nono/genai_tasker/genai_tasker.py` — Keyring-first API key resolution, improved error messages

#### Removed

- `nono/Organizer/` — Deleted: `file_analizer.py`, `file_analizer.db` (1.22 MB), `analizer_prompt.txt`
- `nono/demo_formulite.py` — Removed demo file
- `nono/system12/design.txt` — Removed internal design notes

---

### `de9f7ac` — refactor!: Major architecture restructure with JinjaPromptPy (2026-02-05)

Major architecture restructure: module renaming, JinjaPromptPy integration, new API manager, batch processing engine, and model catalogs.

**46 files changed** | +10,756 −2,744 lines

#### Added

- `nono/connector/api_manager.py` — `ApiManager`: centralized API key and endpoint management with `apikeys.csv` support
- `nono/connector/genai_batch_processing.py` — `BatchProcessor`: high-volume request processing with concurrency control
- `nono/connector/genai_batch_processing_example.py` — Usage examples for batch processing
- `nono/connector/model_features.csv` — TSV catalog of model capabilities per provider
- `nono/connector/model_rate_limits.csv` — TSV catalog of rate limits per model
- `nono/connector/README_api_manager.md` — API manager documentation
- `nono/connector/README_genai_batch_processing.md` — Batch processing documentation
- `nono/connector/README_connector_genai_ssl.md` — SSL docs (moved from connector_genai_ssl.md)
- `nono/tasker/jinja_prompt_builder.py` — JinjaPromptPy integration for template-based prompt construction
- `nono/tasker/templates/` — Jinja2 prompt template directory
- `nono/config.toml` — Project-level configuration file

#### Changed

- `nono/genai_tasker/` → `nono/tasker/` — **Module renamed** (all files moved)
- `nono/genai_executer/` → `nono/executer/` — **Module renamed** (all files moved)
- `nono/connector/connector_genai.py` — Rewritten with `GenerativeAIService` abstract base, `OpenAICompatibleService` shared REST base, Token Bucket rate limiting
- `README.md` — Major rewrite (+446 −58 lines): updated structure, JinjaPromptPy integration docs
- `nono/__init__.py` — Updated submodule names: `genai_tasker` → `tasker`, `genai_executer` → `executer`
- `.gitignore` — Added `**/*key*.csv`, `**/*secret*.csv`

#### Removed

- `nono/apikey.txt` — Deleted (key management via `api_manager.py` and keyring)

---

### `c84d0a7` — feat: CLI entry point and centralized config module (2026-02-09)

CLI entry point with argparse, centralized configuration module, and hierarchical path resolution system.

**8 files changed** | +808 −5 lines

#### Added

- `main.py` — CLI entry point with argparse (+310 lines): `--prompt` (simple execution), `--task` (task file execution), `--list-providers`, `--list-templates`, `--list-tasks`, interactive mode with menu
- `nono/config.py` — `NonoConfig` class (+291 lines): hierarchical config resolution (env vars → programmatic API → config.toml), `get_templates_dir()`, `get_prompts_dir()`, `get_config_value()`, TOML loading via `tomllib`
- `nono/config.toml` — Added `[paths]` section: `templates_dir`, `prompts_dir` (configurable, relative or absolute)
- `nono/tasker/prompts/name_classifier.json` — New version with enhanced prompt structure for multi-culture name analysis
- `nono/tasker/prompts/task_template.json` — Generic task template with input/output/metadata schema

#### Changed

- `nono/__init__.py` — Added `config` submodule docs, `NonoConfig` usage examples with env vars and programmatic API
- `nono/tasker/jinja_prompt_builder.py` — Integrated with `NonoConfig.get_templates_dir()` instead of hardcoded paths
- `README.md` — Added paths configuration section with resolution priority table

---

### `d8d2516` — fix: Main entry point refinements and model version update (2026-02-09)

Minor fixes to main entry point and default model version update.

**2 files changed** | +20 −8 lines

#### Changed

- `main.py` — Refactored `interactive_mode()` function (cleaned up header formatting)
- `nono/tasker/prompts/name_classifier.json` — Model updated: `gemini-2.0-flash` → `gemini-3-flash-preview`

---

### `7dada32` — feat: Dedicated CLI module with colored output and progress bars (2026-02-09)

Full CLI implementation with subcommand architecture, colored output (via colorama), progress bars, and enhanced configuration system.

#### Added

- `nono/cli.py` — Full CLI implementation: subcommand architecture, colored output (via colorama), progress bars, `--dry-run` mode, `--output-format` (table/json/csv/text/markdown/summary/quiet), `--config-file`, `--log-file` options
- `nono/README_cli.md` — CLI documentation: installation, usage, argument reference, config integration

#### Changed

- Configuration system enhancements (provider-specific settings, path resolution)

---

### `633fd59` — docs: Translate all documentation from Spanish to English (2026-02-11)

Internationalization: all README and documentation files translated from Spanish to English for broader accessibility.

#### Changed

- All README and documentation files — Translated from Spanish to English:
  - `nono/connector/connector_genai_ssl.md` (was fully in Spanish: "Opciones Disponibles", "Modo INSECURE", etc.)
  - `nono/genai_tasker/README_task_configuration.md`
  - `nono/genai_tasker/README_technical.md`
  - Other internal documentation files

---

### `96acdf5` — feat: Extended provider support and dynamic model catalog (2026-02-13)

Extended provider support with dynamic model catalog system, CLI subcommands with aliases, CI mode, and programmatic API.

**10 files changed** | +1,275 −40 lines

#### Added

- Dynamic model catalog system in `main.py`:
  - `_load_model_catalog()` — Parses `model_features.csv` to build provider→models mapping
  - `get_available_providers()` — Lists providers from catalog (falls back to hardcoded list)
  - `get_models_for_provider()` — Returns known models for a given provider
  - `get_default_model()` — Returns default model per provider from catalog
- `nono/cli.py` enhancements:
  - `run_api()` — Programmatic entry point (no subprocess needed) returning `CLIResult`
  - `CLIResult` — Structured result: `ok`, `data`, `stats`, `error` fields
  - Subcommand support with aliases (e.g., `analyze` / `a`)
  - `--ci` flag: no colors, quiet, JSON output (CI-friendly defaults)
  - Exit codes: `0` success, `1` runtime error, `2` usage error, `130` interrupted
  - Auto-disables colors when stdout is not a TTY
- `README.md` — Added CI/CD section with Jenkins example pipeline reference

#### Changed

- `nono/connector/connector_genai.py` — Additional provider services
- `nono/connector/api_manager.py` — Extended provider endpoint support
- `nono/connector/model_features.csv` — Updated model catalog
- `nono/connector/model_rate_limits.csv` — Updated rate limits data
- `nono/README_cli.md` — Added subcommands, CI mode, `run_api()`, `CLIResult` docs, updated provider names (`gemini` → `google`)
- `main.py` — Interactive mode now uses dynamic provider/model discovery from catalog (+122 −10 lines)

---

[Unreleased]: https://github.com/jrodriguezgar/Nono/commits/main
[0.1.0]: https://github.com/jrodriguezgar/Nono/commits/main
