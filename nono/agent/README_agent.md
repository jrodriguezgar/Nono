# Agent — AI Agent Framework for Nono

> Nono Agent Architecture (NAA) — multi-agent framework built on top of Nono's unified connector layer. Fully standalone — no external agent framework dependencies.

## Table of Contents

- [Introduction](#introduction)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [LlmAgent / Agent](#llmagent--agent)
- [Structured Output](#structured-output)
- [Auto-Compaction](#auto-compaction)
- [Agent Execution Model](#agent-execution-model)
- [Tools and @tool](#tools-and-tool)
- [ACI Quality — Tool Description Validation](#aci-quality--tool-description-validation)
- [ToolContext](#toolcontext)
- [SharedContent — Two Storage Levels](#sharedcontent--two-storage-levels)
- [Session](#session)
- [InvocationContext](#invocationcontext)
- [State Isolation Patterns](#state-isolation-patterns)
- [transfer\_to\_agent](#transfer_to_agent)
- [Orchestration Agents](#orchestration-agents)
- [RouterAgent](#routeragent)
- [Tasker Integration](#tasker-integration)
- [Workflow Integration](#workflow-integration)
- [Runner](#runner)
- [Sync and Async](#sync-and-async)
- [Callbacks](#callbacks)
- [What Nono Has That Other Frameworks Don't](#what-nono-has-that-other-frameworks-dont)
- [API Reference](#api-reference)

---

## Introduction

The **Agent** module allows you to create LLM-powered agents that can:

- Call **tools** (Python functions) via function calling
- **Delegate tasks** to sub-agents dynamically (`transfer_to_agent`)
- Store **content** at two levels: session (shared) and agent (private)
- Run **synchronously or asynchronously** with the same API
- Compose into **deterministic pipelines** with orchestration agents
- Switch between **14 providers** without changing a single line of logic

---

## Quickstart

```python
from nono.agent import Agent, Runner, tool

@tool(description="Adds two numbers.")
def add(a: int, b: int) -> str:
    return str(a + b)

agent = Agent(
    name="calculator",
    model="gemini-3-flash-preview",
    provider="google",
    instruction="You are a calculator assistant. Use tools for operations.",
    tools=[add],
)

runner = Runner(agent=agent)
response = runner.run("What is 7 + 3?")
print(response)
```

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    Runner                         │
│  (session lifecycle, run/stream, sync+async)      │
└─────────────────────┬────────────────────────────┘
                      │
            ┌─────────▼──────────┐
            │     LlmAgent       │
            │  (LLM + tools +    │
            │  transfer_to_agent) │
            └──┬───────────┬─────┘
               │           │
       ┌───────▼──┐   ┌────▼───────────┐
       │  Tools   │   │  Sub-agents    │
       │  (@tool) │   │  (delegation)  │
       └──────────┘   └────────────────┘

    ┌────────────────────────────────────────┐
    │        Workflow Agents                  │
    │  Sequential · Parallel · Loop          │
    │  (deterministic orchestration)          │
    └────────────────────────────────────────┘

    ┌────────────────────────────────────────┐
    │        SharedContent                    │
    │  Session (shared) + Agent (local)       │
    │  (named content store)                  │
    └────────────────────────────────────────┘
```

**Main components:**

| Class | Description |
| --- | --- |
| `BaseAgent` | Abstract base class — defines `run()` / `run_async()` contract |
| `LlmAgent` / `Agent` | LLM-connected agent with tool calling and delegation |
| `SequentialAgent` | Runs sub-agents in order |
| `ParallelAgent` | Runs sub-agents in parallel (`ThreadPoolExecutor` / `asyncio.gather`) |
| `LoopAgent` | Repeats sub-agents until condition or max iterations |
| `MapReduceAgent` | Fan-out to mappers in parallel, then reduce into one result |
| `ConsensusAgent` | Multiple agents vote, a judge synthesises the consensus |
| `ProducerReviewerAgent` | Iterative produce-then-review loop until approval |
| `DebateAgent` | Adversarial debate — two agents argue, a judge renders the verdict |
| `EscalationAgent` | Try agents in order, escalate on failure |
| `SupervisorAgent` | LLM supervisor that delegates, evaluates, and re-delegates |
| `VotingAgent` | Majority-vote orchestration — N agents answer, most frequent wins |
| `HandoffAgent` | Peer-to-peer handoff mesh — agents transfer full control to each other |
| `GroupChatAgent` | N-agent group chat with manager-controlled speaker selection |
| `HierarchicalAgent` | Multi-level tree orchestration — LLM manager delegates to department heads |
| `GuardrailAgent` | Pre/post validation pipeline — validators check input/output with automatic retry |
| `BestOfNAgent` | Best-of-N sampling — runs same agent N times in parallel, picks best by score |
| `BatchAgent` | Batch processing — processes item list through one agent with concurrency control |
| `CascadeAgent` | Progressive cascade — sequential stages with quality threshold gate |
| `TreeOfThoughtsAgent` | Tree-of-Thoughts — BFS branching with evaluate, prune, and deepen |
| `PlannerAgent` | Plan-and-execute — LLM decomposes task into dependency-aware steps |
| `SubQuestionAgent` | Sub-question decomposition — break complex questions, dispatch, synthesise |
| `ContextFilterAgent` | Context filtering — per-agent event filtering before delegation |
| `ReflexionAgent` | Reflexion — iterative self-improvement with persistent memory |
| `SpeculativeAgent` | Speculative execution — race multiple agents, cancel losers early |
| `CircuitBreakerAgent` | Circuit breaker — failure detection with auto-recovery and fallback |
| `TournamentAgent` | Tournament — bracket-style elimination with judge |
| `ShadowAgent` | Shadow testing — parallel stable + shadow comparison |
| `CompilerAgent` | Compiler — iterative prompt optimisation via DSPy-style compilation |
| `CheckpointableAgent` | Checkpointable — sequential execution with checkpoint/resume |
| `DynamicFanOutAgent` | Dynamic fan-out — LLM-driven task decomposition with parallel workers |
| `SwarmAgent` | Swarm — OpenAI-style agent handoff swarm with context variables |
| `MemoryConsolidationAgent` | Memory consolidation — auto-summarise long conversation history |
| `PriorityQueueAgent` | Priority queue — priority-based execution ordering with parallel groups |
| `MonteCarloAgent` | Monte Carlo Tree Search — MCTS with UCT exploration/exploitation |
| `GraphOfThoughtsAgent` | Graph-of-Thoughts — DAG-based generation, aggregation, and scoring |
| `BlackboardAgent` | Blackboard architecture — shared board with expert activation loop |
| `MixtureOfExpertsAgent` | Mixture-of-Experts — gating function + weighted multi-expert blend |
| `CoVeAgent` | Chain-of-Verification — 4-phase anti-hallucination pipeline |
| `SagaAgent` | Saga transactions — sequential steps with compensating rollback |
| `LoadBalancerAgent` | Load balancer — round-robin, random, or least-used distribution |
| `EnsembleAgent` | Ensemble — aggregate outputs from multiple agents (concat, weighted, custom) |
| `TimeoutAgent` | Timeout wrapper — enforce deadline with fallback response |
| `AdaptivePlannerAgent` | Adaptive planner — re-plan after every step based on results |
| `RouterAgent` | LLM-powered dynamic routing — picks the best sub-agent per request |
| `Runner` | Session management + convenient execution (`run`, `stream`, `run_async`, `astream`) |
| `Session` | Conversation thread: chronological events + mutable state + shared content |
| `Event` | Immutable record of actions (7 types: user, agent, tool, transfer, state, error) |
| `FunctionTool` | Python function wrapper → LLM function calling |
| `ToolContext` | Auto-injected context in tools (state + shared/local content) |
| `SharedContent` | Named content store (session-level and agent-level) |
| `ContentItem` | Immutable entry with data, content_type, metadata, created_by, timestamp |

---

## LlmAgent / Agent

The main agent that connects to an LLM and runs a tool calling loop.

```python
from nono.agent import Agent

agent = Agent(
    name="assistant",
    model="gemini-3-flash-preview",   # Model
    provider="google",                 # Provider (14 supported)
    instruction="You are a helpful assistant.",  # System prompt
    description="General assistant",   # Description for delegation
    tools=[...],                       # List of FunctionTool
    sub_agents=[...],                  # Child agents (auto-registers transfer_to_agent)
    temperature=0.7,
    max_tokens=None,
    output_format="text",              # "text" or "json"
    output_model=None,                 # Pydantic BaseModel for structured output
    output_parser=None,                # Custom OutputParser instance
    output_retries=2,                  # Retry count on parse failure
    compaction=True,                   # LLM-based auto-compaction (or strategy)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Unique agent name |
| `model` | `str` | per provider | LLM model |
| `provider` | `str` | `"google"` | Provider (`google`, `openai`, `groq`, `deepseek`, `xai`, `cerebras`, `nvidia`, `perplexity`, `github`, `openrouter`, `azure`, `vercel`, `ollama`) |
| `instruction` | `str` | `"You are a helpful assistant."` | System prompt |
| `description` | `str` | `""` | Description for `transfer_to_agent` |
| `tools` | `list[FunctionTool]` | `[]` | Available tools |
| `sub_agents` | `list[BaseAgent]` | `[]` | Child agents for dynamic delegation |
| `temperature` | `float` | `0.7` | LLM temperature |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens |
| `output_format` | `str` | `"text"` | Response format (`"text"` or `"json"`) |
| `output_model` | `type \| None` | `None` | Pydantic `BaseModel` class — auto-validates and retries structured output |
| `output_parser` | `OutputParser \| None` | `None` | Custom `OutputParser` instance for parsing with retry |
| `output_retries` | `int` | `2` | Max retry attempts when structured output validation fails |
| `compaction` | `CompactionStrategy \| Callable \| str \| bool \| None` | `None` | Context compaction: `True` = LLM summarisation, strategy instance, callable `(msgs, max) -> msgs`, dotted import path `"pkg.mod.Class"`, or `False`/`None` = sliding window |
| `api_key` | `str \| None` | `None` | API key override |

**Properties and methods:**

| Method | Description |
| --- | --- |
| `service` | Property — lazily initializes the provider connector |
| `local_content` | Agent's private `SharedContent` |
| `run(ctx)` | Synchronous execution → `str` |
| `run_async(ctx)` | Asynchronous execution → `str` |
| `find_sub_agent(name)` | Recursive search for sub-agents by name |

---

## Structured Output

Agents can automatically parse and validate LLM responses against a Pydantic model or custom parser, with retry on failure.

### Using `output_model` (Pydantic)

```python
from pydantic import BaseModel
from nono.agent import Agent, Runner

class Sentiment(BaseModel):
    label: str
    score: float
    reasoning: str

agent = Agent(
    name="classifier",
    instruction="Classify the sentiment of the given text.",
    output_model=Sentiment,   # Pydantic model → auto JSON schema + validation
    output_retries=3,         # retry up to 3 times on parse failure
)

response = Runner(agent).run("Analyze: 'This product is amazing!'")
# response is a validated JSON string matching Sentiment schema
```

### Using `output_parser` (Custom)

```python
from nono.connector import RegexOutputParser

# Extract a score from free-form text
parser = RegexOutputParser(pattern=r"Score:\s*(\d+(?:\.\d+)?)")

agent = Agent(
    name="scorer",
    instruction="Evaluate the quality. Include 'Score: X.X' in your response.",
    output_parser=parser,
    output_retries=2,
)
```

### How It Works

1. On each run, the agent builds the parser from `output_model` or `output_parser`
2. Format instructions are injected into the user message (e.g., "Respond with valid JSON matching this schema: …")
3. `response_format` is set to `JSON` and `json_schema` is passed for JSON-based parsers
4. After the LLM responds, the parser validates the output
5. On validation failure, a repair prompt is sent with the error details, up to `output_retries` times
6. If all retries fail, a `MaxRetriesExceededError` is raised

---

## Auto-Compaction

By default, when the message list exceeds `max_loop_messages` (40), old messages are silently dropped via a sliding window.  Enable LLM-based auto-compaction to **summarise** old messages instead of discarding them.

### Enabling Compaction

```python
from nono.agent import Agent, Runner

# Option 1: Simple — uses SummarizationStrategy with defaults
agent = Agent(
    name="researcher",
    instruction="You are a research assistant.",
    compaction=True,  # LLM-based summarisation
)

# Option 2: Custom strategy
from nono.agent import SummarizationStrategy, TokenAwareStrategy

agent = Agent(
    name="researcher",
    instruction="You are a research assistant.",
    compaction=SummarizationStrategy(
        trigger_ratio=0.75,      # compact when > 75% of max_loop_messages
        keep_recent=6,           # keep 6 most recent messages verbatim
        summary_max_tokens=500,  # max tokens for the summary
    ),
)

# Option 3: Token-aware (triggers on estimated token count)
agent = Agent(
    name="researcher",
    instruction="You are a research assistant.",
    compaction=TokenAwareStrategy(
        max_context_tokens=100_000,  # model context window
        trigger_ratio=0.75,
        keep_recent=6,
    ),
)

# Option 4: Plain function (no subclassing needed)
def my_compactor(messages, max_messages):
    system = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[1:] if system else messages
    return system + rest[-(max_messages - len(system)):]

agent = Agent(
    name="researcher",
    instruction="You are a research assistant.",
    compaction=my_compactor,  # auto-wrapped in CallableStrategy
)

# Option 5: External strategy via import path (e.g. from config.toml or a plugin)
agent = Agent(
    name="researcher",
    instruction="You are a research assistant.",
    compaction="mypackage.compaction.MyCustomStrategy",  # dynamic import
)
```

### How It Works

1. Before each LLM call, `_compact_messages()` checks the strategy's `should_compact()` trigger
2. If triggered, the strategy splits messages into: **system** | **middle** | **recent**
3. The **middle** messages are summarised via an LLM call (using the agent's own service)
4. A `[Context Summary]` system message replaces the middle messages
5. The **recent** messages (default 6) are kept intact
6. If the LLM summary call fails, a naive text summary is used as fallback
7. When `compaction=None` (default), `_prune_messages` sliding window is used directly

### Available Strategies

| Strategy | Trigger | Description |
|----------|---------|-------------|
| *(default)* | Message count > max | `_prune_messages` sliding window — drop oldest messages |
| `SummarizationStrategy` | Message count > ratio × max | Summarise middle messages via LLM |
| `TokenAwareStrategy` | Estimated tokens > ratio × max_context_tokens | Token-count-aware summarisation |
| `CallableStrategy` | `len(messages) > max` (or custom trigger) | Wraps a plain function `(msgs, max) → msgs` |
| *any callable* | `len(messages) > max` | Auto-wrapped in `CallableStrategy` |
| *string import path* | depends on loaded class | `"pkg.module.ClassName"` — dynamic import & instantiation |

---

## Token-Level Streaming

Native token-by-token streaming delivers text as it is generated, enabling real-time chat UIs and progressive rendering.

### `Runner.stream_text()`

```python
from nono.agent import Agent, Runner

agent = Agent(name="writer", model="gemini-3-flash-preview", instruction="Write a story.")
runner = Runner(agent)

for event in runner.stream_text(user_input="Tell me a tale"):
    if event.type.value == "text_chunk":
        print(event.data["chunk"], end="", flush=True)
    elif event.type.value == "agent_message":
        print("\n--- Full response ---")
        print(event.data["content"])
```

### Event Types

| Event | Data | When |
|-------|------|------|
| `TEXT_CHUNK` | `{"chunk": "token..."}` | Each token/delta from the LLM |
| `AGENT_MESSAGE` | `{"content": "full text"}` | After all tokens are assembled |
| `TOOL_CALL` / `TOOL_RESULT` | tool details | During tool-calling loops (non-streaming) |

### SSE Endpoint

```bash
curl -N -X POST http://localhost:8000/agent/writer/stream/text \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Tell me a tale"}'
```

Returns `text/event-stream` with one SSE event per token chunk.

### Provider Coverage

Native streaming: OpenAI-compatible (8 providers), Gemini, Cerebras, Ollama.  
Fallback (single chunk): Azure AI Foundry, Vercel AI SDK.

### Streaming Tool Calls

When tools are available, the LLM's tool-call arguments stream incrementally:

| Event | Data | When |
|-------|------|------|
| `TOOL_CALL_CHUNK` | `{"tool_index": 0, "tool_name": "fn", "arguments_delta": "{\"ci..."}` | Each argument fragment from the LLM |
| `TOOL_CALL` | `{"tool": "fn", "arguments": {...}}` | After arguments are fully assembled |
| `TOOL_RESULT` | `{"tool": "fn", "result": "..."}` | After tool execution |

Native streaming tool calls: OpenAI-compatible (8 providers), Cerebras.  
Fallback (text-based detection): all other providers.

---

## Agent Execution Model

Infrastructure for autonomous multi-agent execution, inspired by production agent systems. All classes live in `nono/agent/execution.py`.

### TaskPacket — Typed Task Assignment

Replace natural-language prompts with structured task packets:

```python
from nono.agent import TaskPacket, EscalationPolicy, ReportingContract

packet = TaskPacket(
    objective="Fix the login timeout bug",
    scope="module:src/auth/",
    acceptance_tests=["pytest tests/test_auth.py"],
    branch_policy="feature/{task_id}",
    commit_policy="squash_on_green",
    escalation_policy=EscalationPolicy(retry=3, then="notify_human"),
    reporting_contract=ReportingContract(format="json", fields=["files_changed", "tests_run", "diff_summary"]),
)

# JSON round-trip
data = packet.to_json()
restored = TaskPacket.from_json(data)
```

### WorkerStateMachine — Lifecycle Management

Track agent worker state through 8 states with thread-safe transitions:

```python
from nono.agent import WorkerStateMachine, WorkerState

sm = WorkerStateMachine()
sm.on_transition(lambda t: print(f"{t.from_state.value} → {t.to_state.value}"))

sm.transition(WorkerState.TRUST_REQUIRED)
sm.transition(WorkerState.READY_FOR_PROMPT)
sm.transition(WorkerState.PROMPT_ACCEPTED)
sm.transition(WorkerState.RUNNING)
sm.transition(WorkerState.FINISHED)

print(sm.history)  # list of WorkerTransition
```

### FailureClassifier — Failure Taxonomy with Recovery

Classify errors into 10 categories, each with a recovery recipe:

```python
from nono.agent import FailureClassifier

classifier = FailureClassifier()
result = classifier.classify(Exception("rate limit exceeded"))
print(result.category)  # FailureCategory.RATE_LIMITED
print(result.recipe)    # RecoveryRecipe(action="backoff_retry", ...)
```

### PolicyEngine — Executable Policies

Machine-enforced policies with full runtime control:

```python
from nono.agent import PolicyEngine, PolicyResult, CallablePolicy

# Default engine with 5 built-in policies
engine = PolicyEngine.default()

# Add a custom policy via callable
engine.register_callable(
    "max_file_size",
    lambda ctx: PolicyResult(triggered=True, action="reject")
    if ctx.get("file_size", 0) > 1_000_000 else PolicyResult(triggered=False),
    priority=10,
)

# Toggle policies at runtime
engine.disable("auto_merge")
engine.enable("auto_merge")

# Evaluate all enabled policies
results = engine.evaluate({"tests_pass": True, "conflicts": False})
triggered = engine.triggered({"tests_pass": False})

# Serialise / restore
config = engine.to_config()
restored = PolicyEngine.from_config(config)
```

#### Custom Policy via Subclass

```python
from nono.agent import PolicyRule, PolicyResult, PolicyEngine

@PolicyEngine.register_type()
class MaxTokenPolicy(PolicyRule):
    name = "max_tokens"
    priority = 5

    def evaluate(self, context: dict) -> PolicyResult:
        if context.get("tokens", 0) > 100_000:
            return PolicyResult(triggered=True, action="compact")
        return PolicyResult(triggered=False)
```

### VerificationContract — Progressive Verification

4-level green-ness checks:

```python
from nono.agent import VerificationContract, VerificationLevel

contract = VerificationContract()
result = contract.verify_up_to(VerificationLevel.WORKSPACE_GREEN)
print(contract.highest_passed)
```

### WorktreeManager — Git Worktree Isolation

```python
from nono.agent import WorktreeManager

mgr = WorktreeManager(repo_root="/path/to/repo")
info = mgr.create(branch="feature/fix-123", path="/tmp/wt-123")
mgr.validate_cwd("/tmp/wt-123")
mgr.remove("/tmp/wt-123")
```

### StaleBranchDetector

```python
from nono.agent import StaleBranchDetector

detector = StaleBranchDetector(repo_root="/path/to/repo")
status = detector.check("feature/old-branch", reference="main")
if status.commits_behind > 0:
    detector.merge_forward("feature/old-branch", reference="main")
```

### ConversationCheckpointManager

```python
from nono.agent import ConversationCheckpointManager

mgr = ConversationCheckpointManager(max_checkpoints=10)
mgr.save("session-1", events=[...], state={"step": 3}, messages=[...])
mgr.rewind("session-1", checkpoint_id="cp-001")
```

### PlanModeAgent — Read-Only Exploration

```python
from nono.agent import PlanModeAgent, Agent

agent = Agent(name="coder", instruction="Write code.", tools=[...])
planner = PlanModeAgent(agent)
result = planner.plan("How would you refactor the auth module?")
print(result.plan)           # ["1. Analyse ...", "2. Extract ...", ...]
print(result.tools_blocked)  # ["write_file", "run_command"]
```

---

## Tools and @tool

Tools allow the LLM to execute Python functions via function calling.

### Decorator @tool

```python
from nono.agent import tool

@tool(description="Searches information in the database.")
def search_db(query: str, max_results: int = 5) -> str:
    return f"Found {max_results} results for '{query}'"
```

### Manual FunctionTool

```python
from nono.agent import FunctionTool

def my_function(x: int, y: int) -> str:
    return str(x + y)

my_tool = FunctionTool(my_function, description="Adds two numbers.")
```

### Automatic schema

The JSON schema is generated automatically from type hints:

```python
@tool(description="Example.")
def example(name: str, count: int, active: bool = True) -> str: ...

print(example.parameters_schema)
# {"type": "object", "properties": {"name": {"type": "string"}, "count": {"type": "integer"}, "active": {"type": "boolean"}}, "required": ["name", "count"]}
```

### FunctionTool API

| Method | Description |
| --- | --- |
| `invoke(args, tool_context)` | Executes the function with an args dict |
| `to_function_declaration()` | Generates an OpenAI-compatible declaration for the LLM |
| `name` | Tool name (default: `fn.__name__`) |
| `description` | Description for the LLM |
| `parameters_schema` | Auto-generated JSON Schema |

---

## ACI Quality — Tool Description Validation

> *"Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts."*
> — Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

The **Agent-Computer Interface (ACI)** is the surface through which the LLM understands and invokes your tools. A tool with a vague or missing description forces the model to guess — leading to wrong tool selection, malformed arguments, and wasted tokens.

Nono enforces ACI quality at two levels:

### Automatic validation at construction time

When you create an `Agent` with tools, Nono **automatically validates** every tool description and logs warnings for issues it finds. No extra code required:

```python
from nono.agent import Agent, FunctionTool

# ❌ Bad tool — triggers ERROR + WARNING at construction
bad = FunctionTool(lambda: None, name="x", description="")
agent = Agent(name="a", provider="google", tools=[bad])
# ERROR  Nono.Agent.Tool: Tool 'x': Missing description. The LLM relies on
#        the description to decide when and how to call this tool.
# WARNING Nono.Agent.Tool: Tool 'x': Tool name is too short.
# WARNING Nono.Agent.Tool: Tool 'x': Tool has no parameters.
```

The validation is **advisory** — it never raises or blocks construction. It uses the standard `logging` module so messages appear in your configured log handler.

### Programmatic validation with `validate_tools()`

For CI pipelines, tests, or custom checks, call `validate_tools()` directly:

```python
from nono.agent import validate_tools, ToolIssue

issues = validate_tools([my_tool, other_tool], warn=False)
for issue in issues:
    print(f"[{issue.severity}] {issue.tool_name}: {issue.message}")

assert len(issues) == 0, "Fix tool descriptions before deploying"
```

### What gets checked

| Check | Severity | Trigger |
| --- | --- | --- |
| Description is empty | `error` | `description=""` or no docstring |
| Description too short | `warning` | < 10 characters (configurable via `min_description_len`) |
| Tool name too short | `warning` | Single-character names like `"x"` |
| Missing parameter types | `warning` | Parameters without type annotations |
| Zero parameters | `warning` | Tool accepts no arguments (suspicious but not always wrong) |

### Writing good tool descriptions

Anthopic recommends investing as much effort in tool documentation as in prompts. A good description should tell the LLM **what** the tool does, **when** to use it, and **what** it returns:

```python
# ❌ Bad — the LLM has to guess
@tool(description="Search.")
def search(q: str) -> str: ...

# ✅ Good — clear, specific, actionable
@tool(description=(
    "Search the product catalogue by keyword. Returns a JSON array of "
    "matching products with name, price, and stock. Use this when the "
    "user asks about product availability or pricing. Returns an empty "
    "array if no matches are found."
))
def search_catalogue(query: str, max_results: int = 10) -> str: ...
```

**Tips from Anthropic's Appendix 2:**

1. **Put yourself in the model's shoes.** Would a junior developer understand the tool from its description alone?
2. **Include edge cases and boundaries.** What happens on empty input? What's the format of the return value?
3. **Use poka-yoke.** Design parameters so it's hard to make mistakes — e.g., require absolute paths instead of relative ones.
4. **Test with real inputs.** Run examples to see what mistakes the model makes, then improve the description.

### ToolIssue API

| Field | Type | Description |
| --- | --- | --- |
| `tool_name` | `str` | Name of the tool with the issue |
| `severity` | `str` | `"error"` or `"warning"` |
| `message` | `str` | Human-readable explanation with fix suggestion |

---

## ToolContext

Auto-injected context in tools that request it. Provides access to session state and two content storage levels.

```python
from nono.agent import tool, ToolContext

@tool(description="Saves a report.")
def save_report(text: str, tool_context: ToolContext) -> str:
    # Session state (persists across turns)
    tool_context.state["has_report"] = True

    # Shared content (visible to all agents)
    tool_context.save_content("report", text)

    # Local content (private to this agent)
    tool_context.save_content("draft", f"WIP: {text}", scope="local")

    return "Report saved"
```

**Important**: The `ToolContext` parameter is **automatically excluded** from the JSON schema sent to the LLM and is **injected at runtime**. You only need to declare the type hint.

**Attributes:**

| Attribute | Type | Description |
| --- | --- | --- |
| `state` | `dict[str, Any]` | Mutable session state — persists across turns |
| `shared_content` | `SharedContent` | Session content (visible to **all** agents) |
| `local_content` | `SharedContent` | Private content of the agent invoking the tool |
| `agent_name` | `str` | Name of the agent that invoked the tool |
| `session_id` | `str` | Active session ID |

**Convenience methods:**

| Method | Description |
| --- | --- |
| `save_content(name, data, *, scope="shared")` | Saves content. `scope`: `"shared"` or `"local"` |
| `load_content(name, *, scope="shared")` | Loads content by name |

---

## SharedContent — Two Storage Levels

Content is stored in two independent scopes using the same `SharedContent` class:

| Scope | Access | Location | Typical use |
| --- | --- | --- | --- |
| **Session** (`shared_content`) | All agents in the session | `session.shared_content` | Final results, shared data, artifacts |
| **Agent** (`local_content`) | Only the owning agent | `agent.local_content` | Drafts, cache, private intermediate data |

```python
from nono.agent import Agent, Session

session = Session()

# ── Shared content (session) ──
session.shared_content.save("report", "Visible to all")

# ── Local content (agent) ──
agent_a = Agent(name="a", instruction="...", provider="google")
agent_b = Agent(name="b", instruction="...", provider="google")

agent_a.local_content.save("scratch", "Private to A")
agent_b.local_content.save("scratch", "Private to B")

# Isolation: A cannot see B's local content
assert agent_a.local_content.load("scratch").data == "Private to A"
assert agent_b.local_content.load("scratch").data == "Private to B"

# Both access shared content
assert session.shared_content.load("report").data == "Visible to all"
```

### From a Tool

```python
@tool(description="Processes data.")
def process(text: str, tool_context: ToolContext) -> str:
    tool_context.save_content("draft", text, scope="local")        # private
    tool_context.save_content("result", text.upper(), scope="shared")  # shared
    return "done"
```

### SharedContent API

| Method | Returns | Description |
| --- | --- | --- |
| `save(name, data, *, content_type, metadata, created_by)` | `ContentItem` | Saves content (overwrites if exists) |
| `load(name)` | `ContentItem \| None` | Loads by name |
| `names()` | `list[str]` | Stored keys |
| `delete(name)` | `bool` | Deletes content |
| `clear()` | `None` | Deletes all content |
| `name in store` | `bool` | Membership |
| `len(store)` | `int` | Number of keys |

### Capacity Limits

Each `SharedContent` store enforces two built-in limits:

| Limit | Default | Behaviour |
| --- | --- | --- |
| **Max items** | `200` | LRU eviction — the least-recently accessed item is removed when the limit is exceeded |
| **Max item size** | `10 MB` (10 485 760 bytes) | `ValueError` on `save()` if `len(data)` exceeds the threshold |

Override the item capacity at construction time:

```python
# Unlimited items
session = Session()
session.shared_content = SharedContent(max_items=0)

# Tight cache
session.shared_content = SharedContent(max_items=50)
```

---

## Session

A `Session` represents one conversation thread. It holds chronological events, mutable state, and two content stores.

### Constructor

```python
from nono.agent import Session

session = Session(
    session_id="conv-42",      # auto-generated UUID if omitted
    state={"lang": "es"},      # initial state dict
    memory=kim_instance,       # optional KeepInMind for persistent memory
    max_events=500,            # 0 = unlimited (default)
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `session_id` | `str \| None` | auto UUID | Unique identifier for the session |
| `state` | `dict[str, Any] \| None` | `{}` | Initial mutable state dict |
| `memory` | `Any \| None` | `None` | Optional `KeepInMind` instance — auto-loads prior history and commits new events |
| `max_events` | `int` | `0` | Maximum events to retain. `0` = unlimited. Oldest events are evicted first |

### Thread-Safe State Helpers

When multiple agents run concurrently (e.g., inside `ParallelAgent`), use the lock-protected helpers instead of accessing `session.state` directly:

```python
# ✅ Safe — uses internal lock
session.state_set("counter", 0)
session.state_update({"a": 1, "b": 2})
value = session.state_get("counter", default=0)

# ❌ Unsafe under concurrency — no lock
session.state["counter"] = 0
```

| Method | Signature | Description |
| --- | --- | --- |
| `state_set` | `(key: str, value: Any) → None` | Set one key under lock |
| `state_get` | `(key: str, default: Any = None) → Any` | Get one key under lock |
| `state_update` | `(mapping: dict[str, Any]) → None` | Merge multiple keys under lock |

### Key Properties

| Property | Type | Description |
| --- | --- | --- |
| `events` | `list[Event]` | Chronological event log |
| `state` | `dict[str, Any]` | Mutable state dict (use helpers for concurrency) |
| `shared_content` | `SharedContent` | Session-level content store (visible to all agents) |

---

## InvocationContext

Everything an agent needs to execute a single turn. Created per invocation and passed to `agent.run(ctx)` / `agent.run_async(ctx)`.

```python
from nono.agent import Session, InvocationContext

session = Session(state={"lang": "es"})
ctx = InvocationContext(
    session=session,
    user_message="Summarise the report",
)
response = agent.run(ctx)
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `session` | `Session` | *(required)* | The active session |
| `user_message` | `str` | `""` | Current user message |
| `parent_agent` | `BaseAgent \| None` | `None` | The agent that delegated (sub-agent scenarios) |
| `trace_collector` | `Any \| None` | `None` | Optional `TraceCollector` for observability |
| `transfer_depth` | `int` | `0` | Nested `transfer_to_agent` depth — auto-incremented; raises `RecursionError` at `MAX_TRANSFER_DEPTH` (10) |

---

## State Isolation Patterns

By default all agents in a session share the same `session.state` and `shared_content`. When you need isolation, choose the appropriate level:

### Level 1 — Private content (`local_content`)

Each agent has its own `local_content` store. Sub-agents **do not** inherit it.

```python
agent.local_content.save("scratch", "only I can see this")
```

### Level 2 — Filtered context (`ContextFilterAgent`)

Reduce noise without creating a new session. The sub-agent still shares state, but sees fewer events:

```python
from nono.agent import ContextFilterAgent

focused = ContextFilterAgent(
    name="focused",
    sub_agents=[analyst],
    max_history=10,
    exclude_sources=["debug_logger"],
)
```

### Level 3 — Full isolation (new `Session`)

Create a fresh `Session` for a sub-agent that must not read or write the parent's state:

```python
from nono.agent import Session, InvocationContext

# Parent session
parent_session = Session(state={"user": "Alice"})

# Isolated sub-agent — completely independent state
isolated_session = Session()
isolated_ctx = InvocationContext(
    session=isolated_session,
    user_message="Analyse this data independently",
)
response = sub_agent.run(isolated_ctx)

# Parent state is untouched
assert "user" in parent_session.state
assert isolated_session.state.get("user") is None
```

### Summary

| Level | Mechanism | Shares state | Shares events | Shares content |
| --- | --- | --- | --- | --- |
| Default | Same `Session` | Yes | Yes | Yes (`shared_content`) |
| `local_content` | Agent property | Yes | Yes | No (per-agent) |
| `ContextFilterAgent` | Filtered events | Yes | Partial | Yes |
| New `Session` | Separate instance | No | No | No |

---

## transfer_to_agent

LLM-driven dynamic task delegation between agents. When an `LlmAgent` has `sub_agents`, the framework auto-registers a `transfer_to_agent` tool that allows the LLM to decide when and to whom to delegate.

### How it works

1. A coordinator agent is configured with `sub_agents`
2. The framework automatically generates the `transfer_to_agent` tool with schema `{agent_name: str, message: str}`
3. The LLM receives the tool in its function declarations — the description includes the available sub-agents
4. When the LLM calls `transfer_to_agent`, the framework finds the sub-agent, creates a child `InvocationContext`, and runs it
5. The sub-agent's response is returned as a tool result to the coordinator
6. The coordinator can continue, delegate again, or provide the final response

### Example

```python
from nono.agent import Agent, Runner

# Specialist agents
math_agent = Agent(
    name="math_expert",
    description="Solves complex mathematical problems",
    instruction="You are a mathematics expert.",
    provider="google",
)

writer_agent = Agent(
    name="content_writer",
    description="Writes creative and technical content",
    instruction="You are an expert writer.",
    provider="google",
)

# Coordinator — transfer_to_agent is auto-registered
coordinator = Agent(
    name="coordinator",
    instruction="You are a coordinator. Delegate tasks to the appropriate specialist.",
    provider="google",
    sub_agents=[math_agent, writer_agent],
)

runner = Runner(agent=coordinator)
response = runner.run("What is the integral of x² dx?")
# → The LLM decides to delegate to math_expert automatically
```

### Generated events

| Event | Description |
| --- | --- |
| `AGENT_TRANSFER` | The coordinator initiates the transfer |
| `USER_MESSAGE` | The sub-agent receives the message |
| `AGENT_MESSAGE` | The sub-agent responds |
| `TOOL_RESULT` | The response is returned to the coordinator |

### Advantages vs. static orchestration

| | `transfer_to_agent` | `SequentialAgent` / `ParallelAgent` |
| --- | --- | --- |
| **Decision** | The LLM decides dynamically | Predefined flow in code |
| **Flexibility** | Adapts to user input | Always runs the same sequence |
| **Context** | The LLM formulates the message | Same context for all |
| **Cost** | Requires LLM call to decide | No LLM cost for routing |
| **Use case** | Intelligent routing, multi-domain chatbots | Deterministic pipelines, ETL |

---

## Orchestration Agents

Compose multiple agents into deterministic pipelines — no LLM needed for routing.

### SequentialAgent

Runs sub-agents one after another. Each agent's response is passed as `user_message` to the next.

```python
from nono.agent import SequentialAgent

pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer, reviewer],
)
# researcher → writer → reviewer (chained)
```

### ParallelAgent

Runs sub-agents concurrently. Uses `ThreadPoolExecutor` in sync and `asyncio.gather` in async for true parallelism. By default all sub-agents receive the same `user_message`. Use `message_map` to send a different message to specific agents. Use `result_key` to automatically collect all responses into `session.state`.

```python
from nono.agent import ParallelAgent

# Same message for all
gather = ParallelAgent(
    name="gather_info",
    sub_agents=[web_search, db_search, news_feed],
    max_workers=3,  # threads in sync
)

# Per-agent messages
gather = ParallelAgent(
    name="gather_info",
    sub_agents=[web_search, db_search],
    message_map={"web_search": "AI trends", "db_search": "Q1 sales data"},
)

# Collect all results into session.state
gather = ParallelAgent(
    name="gather_info",
    sub_agents=[web_search, db_search],
    result_key="parallel_results",
)
# After run: session.state["parallel_results"] == {"web_search": "...", "db_search": "..."}
```

`result_key` is the recommended way to feed parallel results into the next agent in a `SequentialAgent` pipeline. For other strategies see [Working with results](README_orchestration.md#working-with-results).

### LoopAgent

Repeats sub-agents until a condition is met or `max_iterations` is reached.

```python
from nono.agent import LoopAgent

loop = LoopAgent(
    name="refine",
    sub_agents=[improver],
    max_iterations=5,
    stop_condition=lambda state: state.get("quality", 0) > 0.9,
)
```

### MapReduceAgent

Fan-out to mappers in parallel, then reduce all results into a single answer via a dedicated `reduce_agent`.

```python
from nono.agent import MapReduceAgent

mapreduce = MapReduceAgent(
    name="summarise_all",
    sub_agents=[search_web, search_db, search_docs],
    reduce_agent=summariser,
    result_key="map_results",  # optional: store mapper outputs in session.state
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Mapper agents executed in parallel |
| `reduce_agent` | `BaseAgent` | required | Agent that combines all mapper outputs |
| `max_workers` | `int \| None` | `None` | Thread pool size (sync) |
| `message_map` | `dict[str, str] \| None` | `None` | Per-mapper custom messages |
| `result_key` | `str \| None` | `None` | Auto-collect mapper results into `session.state` |

### ConsensusAgent

Multiple agents answer the same question independently, then a `judge_agent` synthesises a single consensus response.

```python
from nono.agent import ConsensusAgent

consensus = ConsensusAgent(
    name="fact_check",
    sub_agents=[model_a, model_b, model_c],
    judge_agent=judge,
    result_key="votes",  # optional: store voter answers in session.state
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Voter agents that each produce an answer |
| `judge_agent` | `BaseAgent` | required | Agent that produces the consensus |
| `max_workers` | `int \| None` | `None` | Thread pool size (sync) |
| `result_key` | `str \| None` | `None` | Auto-collect voter answers into `session.state` |

### ProducerReviewerAgent

Iterative produce-then-review loop: a `producer` generates content, a `reviewer` evaluates it. Repeats until the reviewer approves or `max_iterations` is reached.

```python
from nono.agent import ProducerReviewerAgent

pr = ProducerReviewerAgent(
    name="blog_pipeline",
    producer=writer,
    reviewer=editor,
    max_iterations=3,
    approval_keyword="APPROVED",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `producer` | `BaseAgent` | required | Agent that generates/refines content |
| `reviewer` | `BaseAgent` | required | Agent that evaluates the output |
| `max_iterations` | `int` | `3` | Maximum produce-review cycles |
| `approval_keyword` | `str` | `"APPROVED"` | Substring in reviewer response that signals approval (case-insensitive) |

### DebateAgent

Adversarial debate — two agents argue in rounds, a judge renders the final verdict.

```python
from nono.agent import DebateAgent

debate = DebateAgent(
    name="policy_debate",
    agent_a=optimist,
    agent_b=pessimist,
    judge=arbiter,
    max_rounds=3,
    resolution_keyword="RESOLVED",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent_a` | `BaseAgent` | required | First debater |
| `agent_b` | `BaseAgent` | required | Second debater |
| `judge` | `BaseAgent` | required | Agent that renders the final verdict |
| `max_rounds` | `int` | `3` | Maximum debate rounds |
| `resolution_keyword` | `str` | `"RESOLVED"` | Substring in judge response that ends the debate early (case-insensitive) |

### EscalationAgent

Try agents in order, stop at the first success. Escalates from cheap/fast to expensive/powerful.

```python
from nono.agent import EscalationAgent

escalation = EscalationAgent(
    name="tiered_support",
    sub_agents=[fast_model, medium_model, powerful_model],
    failure_keyword="ESCALATE",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Agents tried in order (cheap → expensive) |
| `failure_keyword` | `str` | `"ESCALATE"` | Substring in response that triggers escalation to the next agent |
| `on_escalation` | `Callable \| None` | `None` | Optional callback invoked on each escalation |

### SupervisorAgent

LLM-powered supervisor that delegates tasks to sub-agents, evaluates their output, and can re-delegate. Unlike `RouterAgent` (routes once), the supervisor actively monitors and iterates.

```python
from nono.agent import SupervisorAgent

supervisor = SupervisorAgent(
    name="project_lead",
    sub_agents=[researcher, writer, reviewer],
    model="gemini-3-flash-preview",
    provider="google",
    supervisor_instruction="Delegate research first, then writing, then review. Re-delegate if quality is low.",
    max_iterations=5,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Available worker agents |
| `model` | `str \| None` | provider default | LLM model for the supervisor |
| `provider` | `str` | `"google"` | LLM provider for supervisor decisions |
| `supervisor_instruction` | `str` | `""` | Extra instructions for the supervisor prompt |
| `max_iterations` | `int` | `5` | Maximum delegate-evaluate cycles |

### VotingAgent

Majority-vote orchestration — N agents answer the same question in parallel, the most frequent normalised response wins. No LLM judge needed.

```python
from nono.agent import VotingAgent

voting = VotingAgent(
    name="ensemble",
    sub_agents=[model_a, model_b, model_c],
    max_workers=3,
    result_key="votes",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Voter agents that each produce an answer |
| `max_workers` | `int \| None` | `None` | Thread pool size (sync) |
| `normalize` | `Callable \| None` | `None` | Optional function to normalise responses before counting |
| `result_key` | `str \| None` | `None` | Auto-collect voter answers into `session.state` |

### HandoffAgent

Peer-to-peer handoff mesh — agents transfer **full control** to each other. Unlike `transfer_to_agent` (agent-as-tools, the caller retains control), handoff means the receiving agent takes full ownership of the conversation.

```python
from nono.agent import HandoffAgent

handoff = HandoffAgent(
    name="tutoring",
    entry_agent=triage,
    handoff_rules={
        "triage": [math_tutor, history_tutor],
        "math_tutor": [triage],
        "history_tutor": [triage],
    },
    max_handoffs=5,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `entry_agent` | `BaseAgent` | *required* | First agent to receive the user message |
| `handoff_rules` | `dict[str, list[BaseAgent]]` | `{}` | Map of agent name → list of allowed handoff targets |
| `max_handoffs` | `int` | `10` | Safety limit to prevent infinite handoff chains |
| `handoff_keyword` | `str` | `"HANDOFF:"` | Keyword prefix for handoff directives in agent output |

### GroupChatAgent

N-agent group chat with manager-controlled speaker selection. All agents see the full conversation history. Supports round-robin, LLM-based, and custom speaker selection strategies.

```python
from nono.agent import GroupChatAgent

chat = GroupChatAgent(
    name="creative_team",
    sub_agents=[writer, reviewer],
    speaker_selection="round_robin",
    max_rounds=6,
    termination_keyword="APPROVED",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Participant agents |
| `speaker_selection` | `str \| Callable` | `"round_robin"` | `"round_robin"`, `"llm"`, or custom `(messages, agents) → agent` |
| `max_rounds` | `int` | `10` | Maximum conversation rounds |
| `termination_condition` | `Callable \| None` | `None` | `(messages) → bool` to stop the chat |
| `termination_keyword` | `str \| None` | `None` | Keyword in last message that stops the chat |
| `model` | `str \| None` | `None` | LLM model (required for `"llm"` selection) |
| `provider` | `str` | `"google"` | LLM provider (for `"llm"` selection) |
| `result_key` | `str \| None` | `None` | Store full transcript in `session.state` |

### HierarchicalAgent

Multi-level hierarchical orchestration with LLM-powered manager. Unlike `SupervisorAgent` (flat pool of workers), `HierarchicalAgent` creates a tree-shaped command structure where each department head may itself be an orchestration agent with its own sub-agents. The manager sees the full org-chart and delegates across rounds, then synthesises a final answer.

```python
from nono.agent import Agent, SequentialAgent, HierarchicalAgent

backend = SequentialAgent(
    name="backend_team",
    sub_agents=[architect, developer],
)
qa = Agent(name="qa", instruction="Review code for bugs.", provider="google")

cto = HierarchicalAgent(
    name="cto",
    provider="google",
    sub_agents=[backend, qa],
    max_iterations=3,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Department-head agents (may have their own sub-agents) |
| `model` | `str \| None` | provider default | LLM model for manager decisions |
| `provider` | `str` | `"google"` | LLM provider for manager decisions |
| `manager_instruction` | `str` | `""` | Extra instructions for the manager prompt |
| `max_iterations` | `int` | `3` | Maximum delegation rounds |
| `synthesis_prompt` | `str` | `""` | Custom prompt for final synthesis |
| `result_key` | `str \| None` | `None` | Store synthesis in `session.state` |

### GuardrailAgent

Pre/post validation pipeline. Wraps a main agent with optional pre-validator (input check/transform) and post-validator (output check). If the post-validator's response contains `rejection_keyword`, the main agent retries up to `max_retries` times.

```python
from nono.agent import Agent, GuardrailAgent

safe = GuardrailAgent(
    name="safe_writer",
    main_agent=writer,
    post_validator=checker,
    rejection_keyword="REJECTED",
    max_retries=2,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `main_agent` | `BaseAgent` | *(required)* | The core agent whose output is validated |
| `pre_validator` | `BaseAgent \| None` | `None` | Checks/transforms input before main agent |
| `post_validator` | `BaseAgent \| None` | `None` | Validates output; rejection triggers retry |
| `rejection_keyword` | `str` | `"REJECTED"` | Keyword in post-validator output that triggers retry |
| `max_retries` | `int` | `1` | Max retries after post-validation rejection |
| `result_key` | `str \| None` | `None` | Store validated output in `session.state` |

### BestOfNAgent

Runs the same agent N times in parallel and picks the best response using a scoring function. Useful for creative tasks where quality varies between runs.

```python
from nono.agent import Agent, BestOfNAgent

best = BestOfNAgent(
    name="best_writer",
    agent=writer,
    n=3,
    score_fn=lambda r: float(len(r)),
    result_key="scoring",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | The agent to run N times |
| `n` | `int` | `3` | Number of parallel runs |
| `score_fn` | `Callable[[str], float] \| None` | `None` (uses `len`) | Scoring function to evaluate responses |
| `max_workers` | `int` | `4` | Max threads for parallel execution |
| `result_key` | `str \| None` | `None` | Store `best_index`, `best_score`, `all_scores` |

### BatchAgent

Processes a list of items through one agent with concurrency control. Items can be provided statically or resolved from `session.state` at runtime.

```python
from nono.agent import Agent, BatchAgent

batch = BatchAgent(
    name="batch_classify",
    agent=classifier,
    items=["text1", "text2", "text3"],
    template="Classify: {item}",
    max_workers=3,
    result_key="classifications",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | The agent to process each item |
| `items` | `list[str] \| None` | `None` | Static item list |
| `items_key` | `str \| None` | `None` | Key in `session.state` to read items from |
| `template` | `str \| None` | `None` | Template with `{item}` placeholder |
| `max_workers` | `int` | `4` | Max concurrent workers |
| `result_key` | `str \| None` | `None` | Store `{index: response}` dict |

### CascadeAgent

Progressive cascade of agents with quality threshold. Tries each stage sequentially, scoring the output. Stops when a stage meets the threshold, avoiding unnecessary expensive calls.

```python
from nono.agent import Agent, CascadeAgent

cascade = CascadeAgent(
    name="smart",
    sub_agents=[flash_agent, pro_agent],
    score_fn=lambda r: 1.0 if len(r) > 200 else 0.3,
    threshold=0.8,
    result_key="cascade_info",
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Stages in order (cheapest → most capable) |
| `score_fn` | `Callable[[str], float]` | *(required)* | Scores each stage's output (0.0–1.0) |
| `threshold` | `float` | `0.8` | Minimum score to accept a stage's output |
| `result_key` | `str \| None` | `None` | Store `stage`, `agent`, `score`, `met_threshold` |

### Full Composability

All orchestration parameters that accept an agent (`producer`, `reviewer`, `reduce_agent`, `judge_agent`, `sub_agents[]`) are typed as `BaseAgent`. Since **every** orchestration agent is a `BaseAgent` subclass, you can nest any pattern inside any other — the composition depth is unlimited.

| Parameter | Accepts | You can pass... |
| --- | --- | --- |
| `ProducerReviewerAgent.producer` | `BaseAgent` | `SequentialAgent`, `ParallelAgent`, `MapReduceAgent`... |
| `ProducerReviewerAgent.reviewer` | `BaseAgent` | `ConsensusAgent`, `LlmAgent`, `LoopAgent`... |
| `MapReduceAgent.reduce_agent` | `BaseAgent` | `ProducerReviewerAgent`, `ConsensusAgent`... |
| `ConsensusAgent.judge_agent` | `BaseAgent` | `SequentialAgent`, `ProducerReviewerAgent`... |
| `SequentialAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
| `ParallelAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |

```python
# Producer is a full pipeline, reviewer is a multi-model consensus
pr = ProducerReviewerAgent(
    name="quality_pipeline",
    producer=SequentialAgent(name="draft", sub_agents=[researcher, writer]),
    reviewer=ConsensusAgent(
        name="multi_review",
        sub_agents=[reviewer_gpt, reviewer_gemini],
        judge_agent=final_judge,
    ),
    max_iterations=3,
)
```

See [Composite Patterns](README_orchestration.md#step-11--composite-patterns) for more examples.

---

## RouterAgent

LLM-powered dynamic orchestrator. Makes a lightweight LLM call to decide **which agents** to use and **how** to execute them — picking from four modes: `single`, `sequential`, `parallel`, and `loop`. Internally composes `SequentialAgent`, `ParallelAgent`, and `LoopAgent` at runtime.

```python
from nono.agent import RouterAgent, Runner

router = RouterAgent(
    name="orchestrator",
    model="gemini-3-flash-preview",
    provider="google",
    sub_agents=[researcher, writer, reviewer, coder],
    routing_instruction="Use sequential for articles, single for quick questions.",
    max_iterations=3,  # default for loop mode
)

runner = Runner(agent=router)
response = runner.run("Research AI trends and write a blog post")
# LLM picks sequential mode → researcher then writer
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `str \| None` | provider default | LLM model for routing |
| `provider` | `str` | `"google"` | LLM provider for the routing call |
| `routing_instruction` | `str` | `""` | Extra rules for the routing prompt |
| `temperature` | `float` | `0.0` | LLM temperature (low = deterministic) |
| `max_iterations` | `int` | `3` | Default max iterations for loop mode |

See [README_orchestration.md](README_orchestration.md) for the complete orchestration guide with composite patterns, decision flowchart, and step-by-step examples.

See [README_agent_factory.md](README_agent_factory.md) for dynamic agent generation from natural language descriptions, including orchestration pattern selection.

---

## Tasker Integration

The `tasker_tool` and `json_task_tool` factories bridge `nono.tasker` with the agent framework — wrap a `TaskExecutor` as a `FunctionTool` that any agent can invoke.

### Why use it?

| Direct LLM call (`Agent`) | Via Tasker (`tasker_tool`) |
| --- | --- |
| Agent sends raw prompt to the LLM | Uses TaskExecutor with templates, schemas, batching |
| No output validation | `output_schema` validates JSON responses |
| Single provider/model per agent | Each tool can target a **different** provider/model |
| Agent's own system prompt | Task-specific `system_prompt` per tool |

### tasker_tool — inline configuration

Build a tool from explicit parameters:

```python
from nono.agent import Agent, Runner, tasker_tool

# Create a tool backed by TaskExecutor
summarise = tasker_tool(
    name="summarise",
    description="Summarise a document in 3 sentences.",
    provider="google",
    model="gemini-3-flash-preview",
    system_prompt="You are a professional summariser. Output exactly 3 sentences.",
    temperature=0.3,
    max_tokens=512,
)

agent = Agent(
    name="assistant",
    model="gemini-3-flash-preview",
    instruction="Use the summarise tool when the user asks for a summary.",
    tools=[summarise],
)

runner = Runner(agent=agent)
print(runner.run("Summarise the latest AI trends report."))
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | `"execute_task"` | Tool name visible to the agent |
| `description` | `str` | `"Execute an AI task…"` | Description for the LLM |
| `provider` | `str` | `"google"` | AI provider for the TaskExecutor |
| `model` | `str` | `"gemini-3-flash-preview"` | Model name |
| `api_key` | `str \| None` | `None` | API key (auto-resolved if `None`) |
| `temperature` | `float \| str` | `0.7` | Temperature (float or preset name) |
| `max_tokens` | `int` | `2048` | Maximum response tokens |
| `output_schema` | `dict \| None` | `None` | JSON schema for structured output |
| `system_prompt` | `str \| None` | `None` | System instruction prepended to each call |

### json_task_tool — from a JSON task file

Build a tool from a JSON task definition in `prompts/`:

```python
from nono.agent import Agent, Runner, json_task_tool

# Load task definition — name, description, provider, model, and schema
# are extracted automatically from the JSON file
classify = json_task_tool("nono/tasker/prompts/name_classifier.json")

agent = Agent(
    name="analyst",
    model="gemini-3-flash-preview",
    instruction="Use name_classifier when the user provides a list of names.",
    tools=[classify],
)

runner = Runner(agent=agent)
print(runner.run('["Alice Smith", "Acme Corp", "John Doe"]'))
```

**Parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `task_file` | `str` | required | Path to the JSON task file |
| `name` | `str` | from JSON `task.name` | Override tool name |
| `description` | `str` | from JSON `task.description` | Override description |
| `provider` | `str` | from JSON `genai.provider` | Override provider |
| `model` | `str` | from JSON `genai.model` | Override model |
| `api_key` | `str \| None` | `None` | API key override |

### Multi-provider agent

Combine tools targeting different providers in the same agent:

```python
from nono.agent import Agent, Runner, tasker_tool

fast_task = tasker_tool(
    name="quick_answer",
    description="Fast answer via Groq.",
    provider="groq",
    model="llama-3.3-70b-versatile",
)

deep_task = tasker_tool(
    name="deep_analysis",
    description="Deep analysis via OpenAI.",
    provider="openai",
    model="gpt-4o-mini",
    system_prompt="Provide thorough, detailed analysis.",
)

agent = Agent(
    name="multi_provider",
    model="gemini-3-flash-preview",
    instruction="Use quick_answer for simple questions, deep_analysis for complex ones.",
    tools=[fast_task, deep_task],
)
```

### Architecture

```
┌──────────────────────────────────┐
│          LlmAgent                │
│  (tool calling loop)             │
└───────┬──────────────┬───────────┘
        │              │
   ┌────▼────┐    ┌────▼──────────┐
   │  @tool  │    │  tasker_tool  │
   │ (custom)│    │ (TaskExecutor)│
   └─────────┘    └───────┬───────┘
                          │
                  ┌───────▼───────┐
                  │  TaskExecutor  │
                  │  (templates,   │
                  │   schemas,     │
                  │   batching)    │
                  └───────┬───────┘
                          │
                  ┌───────▼───────┐
                  │ connector_genai│
                  │ (14 providers) │
                  └───────────────┘
```

---

## Workflow Integration

Agents can be embedded as steps in a `Workflow` pipeline via the `agent_node()` factory. This is the reverse of `tasker_tool` — instead of using Tasker inside an agent, you use an Agent inside a workflow.

`agent_node()` wraps an agent in a `Callable[[dict], dict]` that plugs into `Workflow.step()`. Because all node types (plain functions, `tasker_node`, `agent_node`) share the same `step()` interface, they inherit every manipulation method (insert, replace, swap, branch, etc.).

```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node, tasker_node

writer = Agent(name="writer", instruction="Write a blog post.", provider="google")

flow = Workflow("article")
flow.step("gather", lambda s: {"research": f"Key findings about {s['topic']}"})
flow.step("summarise", tasker_node(
    system_prompt="Summarise in 3 sentences.",
    input_key="research", output_key="summary"))
flow.step("write", agent_node(writer, input_key="summary", output_key="draft"))
flow.connect("gather", "summarise", "write")

result = flow.run(topic="GenAI")
```

📖 [Workflow Documentation](../workflows/README_workflow.md) — full API reference for `agent_node()`, `tasker_node()`, and the Workflow engine.

---

## Runner

Convenient executor that manages the session automatically. Supports multi-turn (same session across calls).

```python
from nono.agent import Agent, Runner

agent = Agent(name="bot", provider="google", instruction="...")
runner = Runner(agent=agent)

# Simple execution
response = runner.run("Hello")

# Multi-turn (same session)
response = runner.run("And the population?")

# Inject state before execution
response = runner.run("Summarize", topic="AI", max_words=100)

# Event streaming
for event in runner.stream("Hello"):
    print(f"[{event.event_type.value}] {event.content}")

# History, reset, session
print(runner.history)                          # list[Event]
print(runner.session.state)                    # dict
print(runner.session.shared_content.names())   # list[str]
runner.reset()                                 # new session
```

**Methods:**

| Method | Returns | Description |
| --- | --- | --- |
| `run(msg, **state)` | `str` | Synchronous single-turn execution |
| `stream(msg, **state)` | `Iterator[Event]` | Synchronous event streaming |
| `run_async(msg, **state)` | `str` | Asynchronous execution |
| `astream(msg, **state)` | `AsyncIterator[Event]` | Asynchronous streaming |
| `reset()` | `None` | New session |
| `history` | `list[Event]` | Property — session events |

---

## Sync and Async

All agents implement **both** modes as abstract methods — no fallback or wrapper. Each agent defines its sync and async logic independently.

```python
import asyncio
from nono.agent import Agent, Runner, Session, InvocationContext

agent = Agent(name="bot", provider="google", instruction="...")

# --- Sync ---
session = Session()
ctx = InvocationContext(session=session, user_message="Hello")
response = agent.run(ctx)

# --- Async ---
async def main():
    session = Session()
    ctx = InvocationContext(session=session, user_message="Hello")
    response = await agent.run_async(ctx)
    print(response)

asyncio.run(main())

# --- Async with Runner ---
async def main():
    runner = Runner(agent=agent)
    response = await runner.run_async("Hello")
    async for event in runner.astream("And then?"):
        print(event.content)

asyncio.run(main())
```

`ParallelAgent` leverages async for truly concurrent execution with `asyncio.gather`.

---

## Callbacks

Lifecycle hooks to intercept execution without modifying core logic. Available on all agents.

```python
from nono.agent import Agent

def log_before(agent, ctx):
    print(f"[BEFORE] {agent.name} starts")
    return None  # None = continue; string = short-circuit with that response

def log_after(agent, ctx, response):
    print(f"[AFTER] {agent.name} → {response[:50]}")
    return None  # None = no changes; string = transform response

def log_tool_before(agent, tool_name, args):
    print(f"[TOOL] {tool_name}({args})")
    return None  # None = no changes; dict = replace args

def log_tool_after(agent, tool_name, args, result):
    print(f"[TOOL RESULT] {tool_name} → {result}")
    return None  # None = no changes; Any = replace result

agent = Agent(
    name="bot",
    instruction="...",
    before_agent_callback=log_before,
    after_agent_callback=log_after,
    before_tool_callback=log_tool_before,
    after_tool_callback=log_tool_after,
)
```

| Callback | Signature | Short-circuit |
| --- | --- | --- |
| `before_agent` | `(agent, ctx) → str \| None` | `str` = immediate response, skips execution |
| `after_agent` | `(agent, ctx, response) → str \| None` | `str` = replaces the response |
| `before_tool` | `(agent, tool_name, args) → dict \| None` | `dict` = replaces tool args |
| `after_tool` | `(agent, tool_name, args, result) → Any \| None` | `Any` = replaces tool result |

---

## What Nono Has That Other Frameworks Don't

Comparison with LangChain, LangGraph, CrewAI, AutoGen, and Google ADK (for reference).

### 1. Unified connector with 14 providers — switch provider in 1 line

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Provider switch | `provider="groq"` | Change class + import | LLM config | Change wrapper | Gemini only |
| Native providers | **14** | Multiple (via plugins) | ~5 | ~3 | 1 |

In Nono, switching from Google to Groq to Ollama is a parameter:

```python
# Exactly the same code, only provider changes
agent = Agent(name="bot", provider="google", ...)
agent = Agent(name="bot", provider="groq", ...)
agent = Agent(name="bot", provider="ollama", ...)
```

### 2. Dual content scope: Session + Agent

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Shared content (session) | `shared_content` | Memory modules | Shared memory | Chat history | Artifact Store |
| Private content (agent) | `local_content` | Not native | Not native | Not native | Not native |
| State isolation | 3 levels (local, filter, new Session) | Not native | Not native | Not native | Not native |
| Session with pluggable memory | `Session(memory=kim)` | External memory modules | No | No | No |

Nono is the **only** framework that natively offers two-level content storage with three isolation levels — tools access both content scopes with `scope="shared"` or `scope="local"`, and sub-agents can be fully isolated via a new `Session`.

### 3. ToolContext with automatic injection

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Context injection | Type hint → auto-inject | Manual / RunnableConfig | No | No | Yes (similar) |
| Excluded from LLM schema | Yes, automatic | N/A | N/A | N/A | Yes |
| Access to state + content | Both scopes | Runnable config only | No | No | Artifacts only |

```python
# Just declare the type hint — the framework does the rest
@tool(description="...")
def my_tool(query: str, tool_context: ToolContext) -> str:
    tool_context.state["count"] = tool_context.state.get("count", 0) + 1
    tool_context.save_content("result", query, scope="local")
    return "done"
```

### 4. First-class corporate SSL

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Configurable SSL | `SSLVerificationMode` (CERTIFI, CUSTOM, INSECURE) | Manual `os.environ` | No | No | No |
| Custom certificates | Yes, configurable path | Monkey-patching | No | No | No |

```python
from nono.connector import configure_ssl_verification, SSLVerificationMode
configure_ssl_verification(SSLVerificationMode.CUSTOM, custom_cert_path="corp.crt")
```

### 5. Built-in rate limiting and resilience

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Token Bucket | Yes, with CSV of per-model limits | Not native | No | No | No |
| Circuit Breaker | Yes (`api_manager`) | Not native | No | No | No |
| Retry with backoff | Yes (exponential, fibonacci, jitter) | Via external `tenacity` | No | No | No |
| Health monitoring | HEALTHY / DEGRADED / UNHEALTHY | No | No | No | No |

Nono includes `APIRateLimiter` with Token Bucket, Circuit Breaker, retry strategies (exponential, fibonacci), and API health monitoring — all built-in without external libraries.

### 6. Dual orchestration: deterministic + dynamic

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Deterministic orchestration | `Sequential`, `Parallel`, `Loop` | LangGraph (graph) | Process (sequential) | GroupChat | Yes (similar) |
| Dynamic delegation (LLM) | `transfer_to_agent` (auto-tool) | Not native | Manager agent | Speaker selection | `transfer_to_agent` |
| Both in the same system | **Yes** | Requires manual design | No | Partial | Yes |
| Inter-agent data passing | **3 mechanisms** (see below) | `RunnablePassthrough` + LCEL piping | Implicit via role context | Shared chat history | Via `state` dict |

Nono allows mixing static orchestration (predefined pipelines) and dynamic orchestration (LLM decides) in the same agent system, with the same API.

**Inter-agent data passing** — Nono provides three built-in mechanisms with zero boilerplate:

| Mechanism | Pattern | How it works |
| --- | --- | --- |
| `user_message` chaining | `SequentialAgent` | Automatic — agent A's response becomes agent B's input |
| `result_key` | `ParallelAgent`, `MapReduceAgent`, `ConsensusAgent`, … | Collects all outputs into `session.state[key]` as a dict |
| `message_map` | `ParallelAgent` | Sends a different custom message to each sub-agent |

Other frameworks require manual wiring (LangChain LCEL), rely on implicit chat history (AutoGen), or limit data flow to a shared state dict (Google ADK).

### 7. Zero agent framework dependencies

| | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Agent dependencies | **None** | langchain-core, etc. | crewai | pyautogen | google-adk |
| Framework size | Lightweight (~1500 lines) | Heavy | Medium | Medium | Medium |
| Lock-in | No | High | High | High | High |

Nono's Agent module is fully implemented in ~1500 lines of pure Python, with no external agent dependencies. It only uses Nono's connector layer (`nono.connector`).

### Summary

| Feature | Nono | LangChain | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Native providers (no plugins) | 14 | Many (plugins) | ~5 | ~3 | 1 |
| 1-line provider switch | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dual content (session + agent) | ✅ | ❌ | ❌ | ❌ | ❌ |
| State isolation (3 levels) | ✅ | ❌ | ❌ | ❌ | ❌ |
| ToolContext auto-inject | ✅ | ❌ | ❌ | ❌ | ✅ |
| Native corporate SSL | ✅ | ❌ | ❌ | ❌ | ❌ |
| Rate limit + Circuit Breaker | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM-powered router | ✅ `RouterAgent` | ❌ | Manager agent | Speaker selection | ❌ |
| Dual orchestration (static + LLM) | ✅ | Partial | ❌ | Partial | ✅ |
| Inter-agent data passing (3 mechanisms) | ✅ | Manual (LCEL) | Implicit | Chat history | `state` dict |
| Tasker-as-tool integration | ✅ `tasker_tool` | ❌ | ❌ | ❌ | ❌ |
| Abstract Sync + Async | ✅ | ✅ | ❌ | ✅ | ✅ |
| Zero agent dependencies | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## API Reference

### Main exports

```python
from nono.agent import (
    # Core
    Agent, LlmAgent, BaseAgent, Runner,
    Session, Event, EventType, InvocationContext,
    # Tools
    FunctionTool, ToolContext, tool,
    # Content
    SharedContent, ContentItem,
    # Orchestration
    SequentialAgent, ParallelAgent, LoopAgent,
    MapReduceAgent, ConsensusAgent, ProducerReviewerAgent,
    RouterAgent,
    # Tasker integration
    tasker_tool, json_task_tool,
    # Callbacks
    BeforeAgentCallback, AfterAgentCallback,
    BeforeToolCallback, AfterToolCallback,
)
```

### EventType

| Value | Description |
| --- | --- |
| `USER_MESSAGE` | User message |
| `AGENT_MESSAGE` | Agent response |
| `TOOL_CALL` | Tool invocation |
| `TOOL_RESULT` | Tool result |
| `STATE_UPDATE` | Session state change |
| `AGENT_TRANSFER` | Delegation to sub-agent via `transfer_to_agent` |
| `ERROR` | Error during execution |

### Supported providers

| Provider | Default model | Internal class |
| --- | --- | --- |
| `google` / `gemini` | `gemini-3-flash-preview` | `GeminiService` |
| `openai` | `gpt-4o-mini` | `OpenAIService` |
| `groq` | `llama-3.3-70b-versatile` | `GroqService` |
| `deepseek` | `deepseek-chat` | `DeepSeekService` |
| `xai` | `grok-3` | `XAIService` |
| `cerebras` | `llama-3.3-70b` | `CerebrasService` |
| `nvidia` | `meta/llama-3.3-70b-instruct` | `NvidiaService` |
| `perplexity` | `sonar` | `PerplexityService` |
| `github` | `openai/gpt-5` | `GitHubModelsService` |
| `openrouter` | `openrouter/auto` | `OpenRouterService` |
| `azure` | `openai/gpt-4o` | `AzureAIService` |
| `vercel` | `anthropic/claude-opus-4.5` | `VercelAIService` |
| `ollama` | `llama3` | `OllamaService` |
