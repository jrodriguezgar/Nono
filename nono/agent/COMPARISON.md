# Nono vs Agent Frameworks — Feature Comparison

Comprehensive comparison of Nono against LangChain, LangGraph, CrewAI, AutoGen, Google ADK, OpenAI Agents SDK, and LlamaIndex.

Legend: **Y** = Yes/Supported | **N** = No/Not supported | **P** = Partial/Via plugins | **E** = Via external library

---

## 1. LLM Provider Support

| Feature                        | Nono         | LangChain   | CrewAI      | AutoGen     | Google ADK  | OpenAI Agents SDK | LlamaIndex  |
| ------------------------------ | ------------ | ----------- | ----------- | ----------- | ----------- | ----------------- | ----------- |
| Native providers (no plugins)  | **14** | Many (P)    | ~5          | ~3          | 1           | 1                 | Many (P)    |
| 1-line provider switch         | **Y**  | N           | N           | N           | N           | N                 | N           |
| Google Gemini                  | **Y**  | Y (P)       | Y           | Y           | **Y** | N                 | Y (P)       |
| OpenAI                         | **Y**  | **Y** | **Y** | **Y** | N           | **Y**       | **Y** |
| Groq                           | **Y**  | Y (P)       | Y           | N           | N           | N                 | Y (P)       |
| DeepSeek                       | **Y**  | Y (P)       | N           | N           | N           | N                 | Y (P)       |
| xAI (Grok)                     | **Y**  | Y (P)       | N           | N           | N           | N                 | N           |
| Cerebras                       | **Y**  | N           | N           | N           | N           | N                 | N           |
| NVIDIA NIM                     | **Y**  | Y (P)       | N           | N           | N           | N                 | Y (P)       |
| Perplexity                     | **Y**  | Y (P)       | N           | N           | N           | N                 | Y (P)       |
| GitHub Models                  | **Y**  | N           | N           | N           | N           | N                 | N           |
| OpenRouter (300+ models)       | **Y**  | Y (P)       | N           | N           | N           | N                 | N           |
| Azure AI (Inference + Foundry) | **Y**  | Y (P)       | N           | Y           | N           | N                 | Y (P)       |
| Vercel AI SDK                  | **Y**  | N           | N           | N           | N           | N                 | N           |
| Ollama (local)                 | **Y**  | Y (P)       | Y           | Y           | N           | N                 | Y (P)       |

> Nono embeds all 14 provider connectors natively. LangChain requires installing separate `langchain-*` packages. OpenAI Agents SDK supports only OpenAI models. LlamaIndex requires `llama-index-llms-*` packages per provider.

---

## 2. Agent Architecture

| Feature                            | Nono        | LangChain   | CrewAI        | AutoGen           | Google ADK  | OpenAI Agents SDK       | LlamaIndex          |
| ---------------------------------- | ----------- | ----------- | ------------- | ----------------- | ----------- | ----------------------- | ------------------- |
| LLM Agent with tools               | **Y** | **Y** | **Y**   | **Y**       | **Y** | **Y**             | **Y**         |
| Tool calling (function calling)    | **Y** | **Y** | **Y**   | **Y**       | **Y** | **Y**             | **Y**         |
| `@tool` decorator                | **Y** | **Y** | Y             | N                 | **Y** | **Y** `@function_tool` | N (from_defaults)   |
| ToolContext auto-injection         | **Y** | N           | N             | N                 | **Y** | N                       | N                   |
| Context excluded from LLM schema   | **Y** | N/A         | N/A           | N/A               | **Y** | N/A                     | N/A                 |
| transfer_to_agent (LLM delegation) | **Y** | N           | Manager agent | Speaker selection | **Y** | **Y** (handoffs)  | N                   |
| Sub-agents (hierarchical)          | **Y** | P           | **Y**   | **Y**       | **Y** | **Y**             | P (sub-question)    |
| Agent instruction/system prompt    | **Y** | **Y** | **Y**   | **Y**       | **Y** | **Y**             | **Y**         |
| Output format enforcement (JSON)   | **Y** | **Y** | P             | N                 | **Y** | **Y**             | **Y** (parsers) |
| Temperature/max_tokens per agent   | **Y** | **Y** | **Y**   | **Y**       | **Y** | **Y**             | **Y**         |

### Features Nono lacks vs others

| Feature                                     | Who has it                              | Nono                     |
| ------------------------------------------- | --------------------------------------- | ------------------------ |
| Human-in-the-loop approval                  | LangGraph, AutoGen                      | **Y** (`HumanInputAgent`, `human_step`) |
| Multi-turn conversation memory (persistent) | LangChain (Memory), AutoGen             | **Y** (`KeepInMind`)   |
| Agent cloning / serialization               | AutoGen                                 | N                        |
| Planning with self-reflection               | AutoGen, CrewAI                         | N (manual via LoopAgent) |
| Built-in RAG / retrieval tools              | LangChain, CrewAI, LlamaIndex           | N                        |
| Code interpreter tool (sandbox)             | AutoGen, Google ADK, OpenAI Agents SDK  | N (Executer is separate) |
| Guardrails (input/output validation)        | OpenAI Agents SDK                       | N                        |
| Query engines (vector, keyword, hybrid)     | LlamaIndex                              | N                        |

---

## 3. Orchestration

| Feature                           | Nono                                        | LangChain / LangGraph          | CrewAI                | AutoGen           | Google ADK  | OpenAI Agents SDK          | LlamaIndex                |
| --------------------------------- | ------------------------------------------- | ------------------------------ | --------------------- | ----------------- | ----------- | -------------------------- | ------------------------- |
| Sequential pipeline               | **Y** `SequentialAgent`             | **Y** (LangGraph)        | **Y** (Process) | P (GroupChat)     | **Y** | Chain via handoffs         | Custom orchestration      |
| Parallel execution                | **Y** `ParallelAgent`               | **Y** (LangGraph)        | N                     | N                 | **Y** | N                          | N                         |
| Loop / iterative refinement       | **Y** `LoopAgent`                   | **Y** (LangGraph cycles) | N                     | N                 | **Y** | Manual loop                | N                         |
| Map-Reduce                        | **Y** `MapReduceAgent`              | Manual (fan-out + merge) | N                     | N                 | N           | N                          | P (`MapReduce` in QE)    |
| Consensus / voting                | **Y** `ConsensusAgent`              | N                        | N                     | N                 | N           | N                          | N                         |
| Producer-Reviewer                 | **Y** `ProducerReviewerAgent`       | Manual (cycle edges)     | N                     | N                 | N           | N                          | N                         |
| Adversarial debate                | **Y** `DebateAgent`                 | N                        | N                     | N                 | N           | N                          | N                         |
| Tiered escalation                 | **Y** `EscalationAgent`             | N                        | N                     | N                 | N           | N                          | N                         |
| LLM supervisor                    | **Y** `SupervisorAgent`             | N                        | N                     | N                 | N           | N                          | N                         |
| Majority voting                   | **Y** `VotingAgent`                 | N                        | N                     | N                 | N           | N                          | N                         |
| Peer-to-peer handoff              | **Y** `HandoffAgent`                | N                        | N                     | N                 | N           | **Y** (native handoffs) | N                         |
| N-agent group chat                | **Y** `GroupChatAgent`              | N                        | N                     | **Y** (GroupChat) | N           | N                          | N                         |
| Hierarchical orchestration        | **Y** `HierarchicalAgent`           | N                        | **Y** (Process) | N                 | N           | N                          | N                         |
| Pre/post validation guardrails    | **Y** `GuardrailAgent`              | N                        | N               | N                 | N           | N                          | N                         |
| Best-of-N sampling                | **Y** `BestOfNAgent`                | N                        | N               | N                 | N           | N                          | N                         |
| Batch item processing             | **Y** `BatchAgent`                  | N                        | N               | N                 | N           | N                          | N                         |
| Progressive cascade (cost-aware)  | **Y** `CascadeAgent`                | N                        | N               | N                 | N           | N                          | N                         |
| Tree-of-Thoughts reasoning        | **Y** `TreeOfThoughtsAgent`         | N                        | N               | N                 | N           | N                          | N                         |
| Plan-and-execute                   | **Y** `PlannerAgent`                | Plan-and-execute         | `planning=True` | N                 | N           | N                          | N                         |
| Sub-question decomposition         | **Y** `SubQuestionAgent`            | N                        | N               | N                 | N           | N                          | **Y** `SubQuestionQueryEngine` |
| Context/message filtering          | **Y** `ContextFilterAgent`          | N                        | N               | **Y** `MessageFilterAgent` | N | N                    | N                         |
| Reflexion (self-improvement)       | **Y** `ReflexionAgent`              | N                        | N               | N                 | N           | N                          | N                         |
| Speculative execution              | **Y** `SpeculativeAgent`            | N                        | N               | N                 | N           | N                          | N                         |
| Circuit breaker (failure recovery) | **Y** `CircuitBreakerAgent`         | N                        | N               | N                 | N           | N                          | N                         |
| Tournament (bracket elimination)   | **Y** `TournamentAgent`             | N                        | N               | N                 | N           | N                          | N                         |
| Shadow testing                     | **Y** `ShadowAgent`                 | N                        | N               | N                 | N           | N                          | N                         |
| Prompt compilation (DSPy-style)    | **Y** `CompilerAgent`               | N                        | N               | N                 | N           | N                          | P (DSPy integration)      |
| Checkpointable pipeline            | **Y** `CheckpointableAgent`         | P (LangGraph persistence) | N              | N                 | N           | N                          | N                         |
| Dynamic fan-out (LLM decompose)    | **Y** `DynamicFanOutAgent`          | Manual                   | N               | N                 | N           | N                          | N                         |
| Agent handoff swarm                | **Y** `SwarmAgent`                  | N                        | N               | N                 | N           | **Y** (Swarm)           | N                         |
| Memory consolidation               | **Y** `MemoryConsolidationAgent`    | N                        | N               | N                 | N           | N                          | N                         |
| Priority-based execution           | **Y** `PriorityQueueAgent`          | N                        | N               | N                 | N           | N                          | N                         |
| Monte Carlo Tree Search            | **Y** `MonteCarloAgent`             | N                        | N               | N                 | N           | N                          | N                         |
| Graph-of-Thoughts (DAG)            | **Y** `GraphOfThoughtsAgent`        | N                        | N               | N                 | N           | N                          | N                         |
| Blackboard architecture            | **Y** `BlackboardAgent`             | N                        | N               | N                 | N           | N                          | N                         |
| Mixture-of-Experts (gating)        | **Y** `MixtureOfExpertsAgent`       | N                        | N               | N                 | N           | N                          | N                         |
| Chain-of-Verification              | **Y** `CoVeAgent`                   | N                        | N               | N                 | N           | N                          | N                         |
| Saga (compensating transactions)   | **Y** `SagaAgent`                   | N                        | N               | N                 | N           | N                          | N                         |
| Load balancing                     | **Y** `LoadBalancerAgent`           | N                        | N               | N                 | N           | N                          | N                         |
| Ensemble aggregation               | **Y** `EnsembleAgent`               | N                        | N               | N                 | N           | N                          | N                         |
| Timeout / deadline wrapper         | **Y** `TimeoutAgent`                | N                        | N               | N                 | N           | N                          | N                         |
| Adaptive re-planning               | **Y** `AdaptivePlannerAgent`        | P (LangGraph)            | N               | N                 | N           | N                          | N                         |
| LLM-powered router                | **Y** `RouterAgent`                 | N                              | Manager agent         | Speaker selection | N           | Triage agent (handoffs)    | `RouterQueryEngine`       |
| Graph-based workflows             | **Y** `Workflow`                    | **Y** (LangGraph)        | N                     | N                 | N           | N                          | **Y** `Workflow` (v0.11+) |
| Conditional branching             | **Y** `branch_if`                   | **Y** (LangGraph edges)  | N                     | N                 | N           | N                          | **Y** (events)      |
| Dynamic step manipulation         | **Y** (insert/remove/swap at runtime) | N                              | N                     | N                 | N           | N                          | N                         |
| Streaming (step-by-step)          | **Y**                                 | **Y**                    | N                     | P                 | **Y** | **Y**                | P                         |
| Dual orchestration (static + LLM) | **Y**                                 | P                              | N                     | P                 | **Y** | P (handoffs only)          | N                         |

### Features Nono lacks vs others

| Feature                           | Who has it                 | Nono               |
| --------------------------------- | -------------------------- | ------------------ |
| Visual graph editor (UI)          | LangGraph Studio           | N                  |
| Checkpointing / state persistence | LangGraph                  | **Y** `enable_checkpoints()` / `run(resume=True)` |
| Time travel / replay              | LangGraph                  | N                  |
| Crew process types (hierarchical) | CrewAI                     | **Y** `HierarchicalAgent` |
| Built-in guardrails               | OpenAI Agents SDK          | N                  |
| RAG query engines                 | LlamaIndex                 | N                  |

---

## 4. Content and State Management

| Feature                        | Nono                                   | LangChain               | CrewAI        | AutoGen      | Google ADK     | OpenAI Agents SDK | LlamaIndex       |
| ------------------------------ | -------------------------------------- | ----------------------- | ------------- | ------------ | -------------- | ----------------- | ---------------- |
| Session state (shared dict)    | **Y**                            | **Y** (LangGraph) | N             | **Y**  | **Y**    | N                 | P (Context)      |
| Shared content (session-level) | **Y** `shared_content`         | Memory modules          | Shared memory | Chat history | Artifact Store | N                 | N                |
| Private content (agent-level)  | **Y** `local_content`          | N                       | N             | N            | N              | N                 | N                |
| Content versioning             | **Y**                            | N                       | N             | N            | N              | N                 | N                |
| Content scope param in tools   | **Y** `scope="shared"/"local"` | N                       | N             | N            | N              | N                 | N                |

### Features Nono lacks vs others

| Feature                             | Who has it                          | Nono |
| ----------------------------------- | ----------------------------------- | ---- |
| Long-term persistent memory         | LangChain (vector stores)           | **Y** (`KeepInMind` + custom stores) |
| Conversation buffer / window memory | LangChain, AutoGen                  | **Y** (`KeepInMind.max_turns`) |
| Entity memory                       | LangChain                           | N    |
| Knowledge graph integration         | LangChain, LlamaIndex               | N    |
| Vector store indices                | LlamaIndex                          | N    |

---

## 5. Task Execution (Tasker)

| Feature                             | Nono                        | LangChain                    | CrewAI     | AutoGen | Google ADK  | OpenAI Agents SDK          | LlamaIndex              |
| ----------------------------------- | --------------------------- | ---------------------------- | ---------- | ------- | ----------- | -------------------------- | ----------------------- |
| JSON task definitions               | **Y**                 | N                            | YAML tasks | N       | N           | N                          | N                       |
| Jinja2 prompt templates             | **Y** (9 built-in)    | **Y** (PromptTemplate) | N          | N       | N           | N                          | **Y** (PromptTemplate) |
| Structured JSON output validation   | **Y** (jsonschema)    | **Y** (Pydantic)       | P          | N       | **Y** | **Y** (structured outputs) | **Y** (Pydantic parsers) |
| Tasker-as-tool bridge               | **Y** `tasker_tool` | N                            | N          | N       | N           | N                          | N                       |
| Batch processing (TSV, 70% savings) | **Y**                 | N                            | N          | N       | N           | N                          | P (ingestion pipelines) |
| Pre-built data operation templates  | **Y** (9 templates)   | N                            | N          | N       | N           | N                          | N                       |

### Features Nono lacks vs others

| Feature                    | Who has it                        | Nono                |
| -------------------------- | --------------------------------- | ------------------- |
| Pydantic output parsing    | LangChain, Google ADK, LlamaIndex | N (uses jsonschema) |
| Output retry/fixing parser | LangChain                         | N                   |
| Few-shot example selector  | LangChain                         | N                   |
| RAG ingestion pipeline     | LlamaIndex                        | N                   |

---

## 6. Code Generation and Execution

| Feature                         | Nono                          | LangChain     | CrewAI | AutoGen              | Google ADK  | OpenAI Agents SDK             | LlamaIndex     |
| ------------------------------- | ----------------------------- | ------------- | ------ | -------------------- | ----------- | ----------------------------- | -------------- |
| Code generation from NL         | **Y** `CodeExecuter`  | P (via tools) | N      | **Y**          | **Y** | N                             | N              |
| Sandboxed execution (SAFE mode) | **Y**                   | N             | N      | **Y** (Docker) | **Y** | **Y** (Code Interpreter) | P (via plugin) |
| Permissive mode                 | **Y**                   | N/A           | N/A    | N/A                  | N/A         | N/A                           | N/A            |
| Execution timeout               | **Y** (30s default)     | N/A           | N/A    | **Y**          | **Y** | **Y**                   | N/A            |
| Auto-retry on failure           | **Y**                   | N/A           | N/A    | **Y**          | N           | N                             | N              |
| Result persistence              | **Y** (`executions/`) | N             | N      | N                    | N           | N                             | N              |

### Features Nono lacks vs others

| Feature                      | Who has it                        | Nono                   |
| ---------------------------- | --------------------------------- | ---------------------- |
| Docker-isolated execution    | AutoGen                           | N (process-level only) |
| Multi-language execution     | AutoGen (Python, JS, shell)       | N (Python only)        |
| Jupyter notebook integration | AutoGen, LangChain                | N                      |
| Hosted code interpreter      | OpenAI Agents SDK                 | N                      |

---

## 7. API Server and Deployment

| Feature                        | Nono                                      | LangChain / LangServe        | CrewAI      | AutoGen                      | Google ADK  | OpenAI Agents SDK | LlamaIndex               |
| ------------------------------ | ----------------------------------------- | ---------------------------- | ----------- | ---------------------------- | ----------- | ----------------- | ------------------------ |
| Built-in HTTP API server       | **Y** (FastAPI)                     | **Y** (LangServe)      | N           | **Y** (AutoGen Studio) | N           | N                 | P (`llama-deploy`)       |
| Named-resource-only endpoints  | **Y**                               | N                            | N/A         | N                            | N/A         | N/A               | N/A                      |
| Task endpoint                  | **Y** `POST /task/{name}`         | N                            | N           | N                            | N           | N                 | N                        |
| Agent endpoint                 | **Y** `POST /agent/{name}`        | **Y** (chain endpoint) | N           | **Y**                  | N           | N                 | P (`llama-deploy`)       |
| Workflow endpoint              | **Y** `POST /workflow/{name}`     | N                            | N           | N                            | N           | N                 | P (`llama-deploy`)       |
| SSE streaming                  | **Y** `POST /agent/{name}/stream` | **Y**                  | N           | N                            | N           | **Y** (Runner) | N                       |
| Resource discovery (`/info`) | **Y**                               | N                            | N           | N                            | N           | N                 | N                        |
| Docker Compose deployment      | **Y**                               | **Y**                  | N           | **Y**                  | N           | N                 | N                        |
| CLI                            | **Y** (subcommands + CI mode)       | N                            | **Y** | N                            | **Y** | N                 | N                        |
| Programmatic API (`run_api`) | **Y** `CLIResult`                 | N                            | N           | N                            | N           | N                 | N                        |

### Features Nono lacks vs others

| Feature                                | Who has it                         | Nono                     |
| -------------------------------------- | ---------------------------------- | ------------------------ |
| Playground / web UI                    | LangServe, AutoGen Studio          | N                        |
| OpenAPI spec auto-generation           | LangServe                          | P (FastAPI generates it) |
| Authentication / API keys on endpoints | LangServe                          | N (bring your own)       |
| Cloud-hosted deployment                | CrewAI Enterprise, LangSmith       | N                        |
| Kubernetes / Helm charts               | LangServe                          | N                        |
| `llama-deploy` managed serving         | LlamaIndex                         | N                        |

---

## 8. Resilience and Enterprise Features

| Feature                                           | Nono                                     | LangChain                 | CrewAI | AutoGen | Google ADK | OpenAI Agents SDK | LlamaIndex |
| ------------------------------------------------- | ---------------------------------------- | ------------------------- | ------ | ------- | ---------- | ----------------- | ---------- |
| Token Bucket rate limiter                         | **Y** (per-model CSV config)       | N                         | N      | N       | N          | N                 | N          |
| Circuit Breaker                                   | **Y** `api_manager`              | N                         | N      | N       | N          | N                 | N          |
| Retry strategies (exponential, fibonacci, jitter) | **Y**                              | E (tenacity)              | N      | P       | N          | N                 | N          |
| API health monitoring                             | **Y** (HEALTHY/DEGRADED/UNHEALTHY) | N                         | N      | N       | N          | N                 | N          |
| Configurable SSL (CERTIFI, CUSTOM, INSECURE)      | **Y**                              | N                         | N      | N       | N          | N                 | N          |
| Custom corporate certificates                     | **Y**                              | N (manual `os.environ`) | N      | N       | N          | N                 | N          |
| Keyring-based API key storage                     | **Y**                              | N                         | N      | N       | N          | N                 | N          |

### Features Nono lacks vs others

| Feature                             | Who has it              | Nono                           |
| ----------------------------------- | ----------------------- | ------------------------------ |
| Observability / tracing (LangSmith) | LangChain (LangSmith)   | P (TraceCollector, local only) |
| Cost tracking per request           | LangSmith, some AutoGen | N                              |
| Prompt versioning / management      | LangSmith               | N                              |
| A/B testing / evaluation framework  | LangSmith, AutoGen      | N                              |

---

## 9. Visualization and Debugging

| Feature                    | Nono        | LangChain     | CrewAI | AutoGen | Google ADK  | OpenAI Agents SDK    | LlamaIndex  |
| -------------------------- | ----------- | ------------- | ------ | ------- | ----------- | -------------------- | ----------- |
| ASCII workflow rendering   | **Y** | N             | N      | N       | N           | N                    | N           |
| ASCII agent tree rendering | **Y** | N             | N      | N       | N           | N                    | N           |
| Trace collector (local)    | **Y** | P (callbacks) | N      | P       | **Y** | **Y** (built-in) | P (callbacks) |
| Event streaming (SSE)      | **Y** | **Y**   | N      | P       | **Y** | **Y**          | N           |

### Features Nono lacks vs others

| Feature                         | Who has it                    | Nono |
| ------------------------------- | ----------------------------- | ---- |
| Interactive graph visualization | LangGraph Studio              | N    |
| Real-time debug dashboard       | AutoGen Studio                | N    |
| Trace export (OpenTelemetry)    | LangSmith, OpenAI Agents SDK  | N    |

---

## 10. Agent Templates and Reusability

| Feature                              | Nono                                    | LangChain | CrewAI         | AutoGen | Google ADK | OpenAI Agents SDK | LlamaIndex  |
| ------------------------------------ | --------------------------------------- | --------- | -------------- | ------- | ---------- | ----------------- | ----------- |
| Pre-built agent templates            | **Y** (9 single + 5 compositions) | N         | P (role-based) | N       | N          | N                 | N           |
| Pre-built workflow templates         | **Y** (3 pipelines)               | N         | N              | N       | N          | N                 | N           |
| Template registry (API-accessible)   | **Y** (`_AGENT_BUILDERS`)       | N         | N              | N       | N          | N                 | N           |
| Pre-built data op templates (Jinja2) | **Y** (9 templates)               | N         | N              | N       | N          | N                 | N           |

### Features Nono lacks vs others

| Feature                       | Who has it                  | Nono                     |
| ----------------------------- | --------------------------- | ------------------------ |
| Hub / marketplace for sharing | LangChain Hub, LlamaHub     | N                        |
| Community-contributed agents  | CrewAI, LangChain           | N                        |
| Agent config from YAML/JSON   | CrewAI                      | N (Python builders only) |
| Pre-built RAG templates       | LlamaIndex                  | N                        |

---

## 11. Developer Experience

| Feature                              | Nono                                       | LangChain   | CrewAI     | AutoGen     | Google ADK  | OpenAI Agents SDK | LlamaIndex  |
| ------------------------------------ | ------------------------------------------ | ----------- | ---------- | ----------- | ----------- | ----------------- | ----------- |
| Zero agent dependencies              | **Y** (~1500 lines)                  | N (heavy)   | N (medium) | N (medium)  | N (medium)  | N (light)         | N (heavy)   |
| Pure Python (no framework lock-in)   | **Y**                                | N           | N          | N           | N           | N                 | N           |
| Sync + Async (same API)              | **Y**                                | **Y** | N          | **Y** | **Y** | **Y**       | **Y** |
| Unified config module (multi-source) | **Y** (Args > Env > File > Defaults) | N           | N          | N           | N           | N                 | N           |
| Method chaining (fluent API)         | **Y**                                | **Y** | P          | N           | P           | N                 | **Y** |
| `@tool` + type hint auto-schema    | **Y**                                | **Y** | P          | N           | **Y** | **Y**       | N           |
| Multimodal file attachments          | **Y**                                | **Y** | N          | P           | **Y** | **Y**       | **Y** |
| Web search tool (built-in)           | **Y**                                | E (SerpAPI) | E          | N           | **Y** | **Y**       | E (plugins) |

### Features Nono lacks vs others

| Feature                           | Who has it                  | Nono            |
| --------------------------------- | --------------------------- | --------------- |
| Extensive documentation ecosystem | LangChain, LlamaIndex       | P (growing)     |
| Large community / Stack Overflow  | LangChain, AutoGen           | N               |
| TypeScript / JavaScript SDK       | LangChain.js, Vercel AI     | N (Python only) |
| Notebook tutorials                | LangChain, AutoGen, LlamaIndex | N            |
| OpenAI-native tracing dashboard   | OpenAI Agents SDK           | N               |

---

## Summary Matrix

Total feature count across all categories (Y or equivalent).

| Category                | Nono      | LangChain      | CrewAI | AutoGen | Google ADK | OpenAI Agents SDK | LlamaIndex     |
| ----------------------- | --------- | -------------- | ------ | ------- | ---------- | ----------------- | -------------- |
| Provider Support        | 14 native | Many (plugins) | ~5     | ~3      | 1          | 1                 | Many (plugins) |
| Agent Architecture      | 10/10     | 7/10           | 6/10   | 5/10    | 8/10       | 8/10              | 6/10           |
| Orchestration           | 14/14     | 6/14           | 2/14   | 3/14    | 5/14       | 2/14              | 2/14           |
| Content & State         | 5/5       | 2/5            | 1/5    | 2/5     | 2/5        | 0/5               | 0/5            |
| Task Execution          | 6/6       | 2/6            | 1/6    | 0/6     | 1/6        | 1/6               | 2/6            |
| Code Execution          | 5/6       | 1/6            | 0/6    | 4/6     | 3/6        | 2/6               | 1/6            |
| API Server & Deployment | 10/10     | 4/10           | 1/10   | 3/10    | 1/10       | 1/10              | 2/10           |
| Resilience & Enterprise | 7/7       | 1/7            | 0/7    | 1/7     | 0/7        | 0/7               | 0/7            |
| Visualization & Debug   | 4/4       | 2/4            | 0/4    | 2/4     | 2/4        | 2/4               | 1/4            |
| Templates & Reuse       | 4/4       | 0/4            | 1/4    | 0/4     | 0/4        | 0/4               | 0/4            |
| Developer Experience    | 8/8       | 4/8            | 2/8    | 2/8     | 4/8        | 5/8               | 4/8            |

### Where Nono leads

- **Enterprise resilience**: Rate limiting, circuit breaker, SSL, health monitoring — no other framework includes all four natively.
- **Unified platform**: Tasks + Agents + Workflows + API Server + CLI in one package.
- **Zero dependencies**: ~1500 lines, no framework lock-in.
- **Templates**: 9 agent templates + 5 compositions + 3 workflows + 9 data op templates, all API-accessible.
- **Dual content scope**: Session + agent-level content with versioning — unique to Nono.

### Where Nono trails

- **RAG / Retrieval**: LlamaIndex is the clear leader for vector indexing, hybrid search, and retrieval-augmented generation.
- **Persistence**: No checkpoint/replay. Persistent memory via `KeepInMind` (file-based, pluggable stores).
- **Guardrails**: OpenAI Agents SDK includes built-in input/output guardrails; Nono relies on the `guardrail` agent template.
- **Community / Ecosystem**: LangChain and LlamaIndex have the largest communities, hubs, and plugin ecosystems.

---

*Last updated: April 2026*
