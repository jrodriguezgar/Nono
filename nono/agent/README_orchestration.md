# Agent Orchestration Guide

> Complete guide to composing and coordinating multiple agents in Nono — from simple pipelines to LLM-powered dynamic routing.

## Overview

Orchestration is the process of coordinating multiple agents to complete a task that none of them could handle alone. Nono provides **forty-four orchestration strategies** that cover the full spectrum from fully deterministic to fully dynamic:

| Strategy | Routing decision | LLM cost | Best for |
| --- | --- | --- | --- |
| `SequentialAgent` | Fixed order in code | None | Pipelines, ETL, review chains |
| `ParallelAgent` | All at once | None | Data gathering, multi-source search |
| `LoopAgent` | Repeat until done | None | Iterative refinement, retries |
| `MapReduceAgent` | Parallel map + single reduce | None | Summarisation, aggregation, multi-source synthesis |
| `ConsensusAgent` | Parallel vote + single judge | None | Fact-checking, ensemble answers, diversity of opinion |
| `ProducerReviewerAgent` | Iterative produce → review | None | Content generation with quality gate |
| `DebateAgent` | Adversarial rounds + judge | None | Adversarial reasoning, policy review |
| `EscalationAgent` | Try in order, stop on success | None | Tiered support, cost-optimised inference |
| `SupervisorAgent` | LLM delegates + evaluates | 1 call per iteration | Active monitoring, complex delegation |
| `VotingAgent` | Parallel majority vote | None | Ensemble answers, classification consensus |
| `HandoffAgent` | Peer-to-peer handoff (keyword) | None | Triage routing, expert transfer, support desks |
| `GroupChatAgent` | Manager-controlled turns | Optional (LLM speaker selection) | Collaborative writing, multi-agent brainstorming |
| `HierarchicalAgent` | LLM manager delegates to departments | 1 call per round + 1 synthesis | Multi-level org structure, cross-department coordination |
| `GuardrailAgent` | Pre/post validation pipeline | None | Input sanitisation, output safety, content moderation |
| `BestOfNAgent` | Same agent N times, pick best | None | Creative tasks, quality sampling, stochastic outputs |
| `BatchAgent` | One agent processes item list | None | Bulk classification, batch translation, data labeling |
| `CascadeAgent` | Sequential stages with threshold | None | Cost-optimised quality, progressive refinement |
| `TreeOfThoughtsAgent` | BFS branching + evaluate + prune | None | Complex reasoning, multi-path exploration |
| `PlannerAgent` | LLM decomposes into steps | 1 plan + 1 synthesis | Task decomposition, dependency-aware execution |
| `SubQuestionAgent` | LLM decomposes questions | 1 decompose + 1 synthesis | Multi-faceted analysis, research questions |
| `ContextFilterAgent` | Per-agent event filtering | None | Noise reduction, focused delegation |
| `ReflexionAgent` | Generate → evaluate → reflect loop | None | Self-correction, iterative improvement |
| `SpeculativeAgent` | Race agents, first to pass threshold wins | None | Latency optimisation, model comparison |
| `CircuitBreakerAgent` | Track failures, switch to fallback | None | Resilience, fault tolerance |
| `TournamentAgent` | Bracket elimination with judge | None | Best-of-field selection, creative contests |
| `ShadowAgent` | Parallel stable + shadow, yield stable | None | A/B testing, safe model migration |
| `CompilerAgent` | Iterative prompt optimisation via examples | 1 LLM call per iteration | Prompt engineering, DSPy-style compilation |
| `CheckpointableAgent` | Sequential with checkpoint/resume | None | Long pipelines, failure recovery |
| `DynamicFanOutAgent` | LLM decomposes into work items, parallel workers | 1 decompose call | Dynamic task decomposition, research |
| `SwarmAgent` | Agent handoff via session state | None | Multi-agent routing, helpdesk flows |
| `MemoryConsolidationAgent` | Auto-summarise old events | None | Long conversations, context management |
| `PriorityQueueAgent` | Priority-grouped execution | None | Ordered processing, resource allocation |
| `MonteCarloAgent` | MCTS with UCT (explore/exploit) | None | Complex reasoning, search optimisation |
| `GraphOfThoughtsAgent` | DAG generate + aggregate + score | None | Multi-path reasoning with merge |
| `BlackboardAgent` | Expert activation on shared board | None | Collaborative problem solving |
| `MixtureOfExpertsAgent` | Gating + top-k weighted blend | None | Specialised multi-domain queries |
| `CoVeAgent` | Draft → verify → revise pipeline | None | Anti-hallucination, fact-checking |
| `SagaAgent` | Steps with compensating rollback | None | Multi-step transactions, error recovery |
| `LoadBalancerAgent` | Round-robin / random / least-used | None | Load distribution, model pooling |
| `EnsembleAgent` | Aggregate all outputs (concat/weighted) | None | Multi-model ensemble, diversity |
| `TimeoutAgent` | Deadline wrapper with fallback | None | Latency SLA, graceful degradation |
| `AdaptivePlannerAgent` | LLM re-plans after each step | 1 call per step | Emergent tasks, dynamic workflows |
| `HumanInputAgent` | Human decides at runtime | None | Approval gates, review checkpoints |
| `RouterAgent` | LLM picks agents + mode | 1 lightweight call | Dynamic orchestration, intent-based pipelines |
| `transfer_to_agent` | LLM decides mid-conversation | Part of main call | Open-ended delegation, chatbots |

```
Deterministic ◄────────────────────────────────────────────────────────► Dynamic

Sequential  Parallel  Loop  HandoffAgent         RouterAgent          transfer_to_agent
(fixed)     (all)     (repeat)  GroupChatAgent  (LLM picks agents + mode)  (LLM decides when)
```

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1 — Create Specialist Agents](#step-1--create-specialist-agents)
- [Step 2 — SequentialAgent (Pipeline)](#step-2--sequentialagent-pipeline)
- [Step 3 — ParallelAgent (Fan-out)](#step-3--parallelagent-fan-out)
- [Step 4 — LoopAgent (Iterative Refinement)](#step-4--loopagent-iterative-refinement)
- [Step 5 — MapReduceAgent (Fan-out + Reduce)](#step-5--mapreduceagent-fan-out--reduce)
- [Step 6 — ConsensusAgent (Vote + Judge)](#step-6--consensusagent-vote--judge)
- [Step 7 — ProducerReviewerAgent (Produce + Review Loop)](#step-7--producerrevieweragent-produce--review-loop)
- [Step 8 — DebateAgent (Adversarial Debate)](#step-8--debateagent-adversarial-debate)
- [Step 9 — EscalationAgent (Tiered Escalation)](#step-9--escalationagent-tiered-escalation)
- [Step 10 — SupervisorAgent (LLM Supervisor)](#step-10--supervisoragent-llm-supervisor)
- [Step 11 — VotingAgent (Majority Vote)](#step-11--votingagent-majority-vote)
- [Step 12 — HandoffAgent (Peer-to-Peer Handoff)](#step-12--handoffagent-peer-to-peer-handoff)
- [Step 13 — GroupChatAgent (N-Agent Group Chat)](#step-13--groupchatagent-n-agent-group-chat)
- [Step 14 — HierarchicalAgent (Multi-Level Hierarchy)](#step-14--hierarchicalagent-multi-level-hierarchy)
- [Step 15 — GuardrailAgent (Pre/Post Validation)](#step-15--guardrailagent-prepost-validation)
- [Step 16 — BestOfNAgent (Best-of-N Sampling)](#step-16--bestofnagent-best-of-n-sampling)
- [Step 17 — BatchAgent (Batch Processing)](#step-17--batchagent-batch-processing)
- [Step 18 — CascadeAgent (Progressive Cascade)](#step-18--cascadeagent-progressive-cascade)
- [Step 19 — TreeOfThoughtsAgent (Tree-of-Thoughts)](#step-19--treeofthoughtsagent-tree-of-thoughts)
- [Step 20 — PlannerAgent (Plan-and-Execute)](#step-20--planneragent-plan-and-execute)
- [Step 21 — SubQuestionAgent (Question Decomposition)](#step-21--subquestionagent-question-decomposition)
- [Step 22 — ContextFilterAgent (Context Filtering)](#step-22--contextfilteragent-context-filtering)
- [Step 23 — ReflexionAgent (Self-Improvement)](#step-23--reflexionagent-self-improvement)
- [Step 24 — RouterAgent (LLM-powered Routing)](#step-24--routeragent-llm-powered-routing)
- [Step 25 — SpeculativeAgent (Speculative Execution)](#step-25--speculativeagent-speculative-execution)
- [Step 26 — CircuitBreakerAgent (Failure Recovery)](#step-26--circuitbreakeragent-failure-recovery)
- [Step 27 — TournamentAgent (Bracket Elimination)](#step-27--tournamentagent-bracket-elimination)
- [Step 28 — ShadowAgent (Shadow Testing)](#step-28--shadowagent-shadow-testing)
- [Step 29 — CompilerAgent (Prompt Optimisation)](#step-29--compileragent-prompt-optimisation)
- [Step 30 — CheckpointableAgent (Checkpoint/Resume)](#step-30--checkpointableagent-checkpointresume)
- [Step 31 — DynamicFanOutAgent (LLM Decomposition)](#step-31--dynamicfanoutagent-llm-decomposition)
- [Step 32 — SwarmAgent (Agent Handoff Swarm)](#step-32--swarmagent-agent-handoff-swarm)
- [Step 33 — MemoryConsolidationAgent (History Summarisation)](#step-33--memoryconsolidationagent-history-summarisation)
- [Step 34 — PriorityQueueAgent (Priority Execution)](#step-34--priorityqueueagent-priority-execution)
- [Step 35 — MonteCarloAgent (MCTS Search)](#step-35--montecarloagent-mcts-search)
- [Step 36 — GraphOfThoughtsAgent (DAG Reasoning)](#step-36--graphofthoughtsagent-dag-reasoning)
- [Step 37 — BlackboardAgent (Expert Blackboard)](#step-37--blackboardagent-expert-blackboard)
- [Step 38 — MixtureOfExpertsAgent (Gated Experts)](#step-38--mixtureofexpertsagent-gated-experts)
- [Step 39 — CoVeAgent (Chain-of-Verification)](#step-39--coveagent-chain-of-verification)
- [Step 40 — SagaAgent (Compensating Transactions)](#step-40--sagaagent-compensating-transactions)
- [Step 41 — LoadBalancerAgent (Request Distribution)](#step-41--loadbalanceragent-request-distribution)
- [Step 42 — EnsembleAgent (Output Aggregation)](#step-42--ensembleagent-output-aggregation)
- [Step 43 — TimeoutAgent (Deadline Wrapper)](#step-43--timeoutagent-deadline-wrapper)
- [Step 44 — AdaptivePlannerAgent (Re-planning)](#step-44--adaptiveplanneragent-re-planning)
- [Step 45 — transfer\_to\_agent (Dynamic Delegation)](#step-45--transfer_to_agent-dynamic-delegation)
- [Step 46 — Composite Patterns](#step-46--composite-patterns)
- [Decision Guide](#decision-guide)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [API Reference](#api-reference)
- [Sync and Async Orchestration](#sync-and-async-orchestration)
- [Workflow vs Agent Orchestration](#workflow-vs-agent-orchestration)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Custom Orchestration Patterns](#custom-orchestration-patterns)

---

## Prerequisites

- Nono installed with at least one provider API key configured in `config.toml`
- Python 3.10+
- Basic familiarity with `Agent`, `Runner`, and `Session` (see [README_agent.md](README_agent.md))

```python
# All orchestration imports
from nono.agent import (
    Agent,
    Runner,
    Session,
    InvocationContext,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
    MapReduceAgent,
    ConsensusAgent,
    ProducerReviewerAgent,
    DebateAgent,
    EscalationAgent,
    SupervisorAgent,
    VotingAgent,
    HandoffAgent,
    GroupChatAgent,
    HierarchicalAgent,
    GuardrailAgent,
    BestOfNAgent,
    BatchAgent,
    CascadeAgent,
    TreeOfThoughtsAgent,
    PlannerAgent,
    SubQuestionAgent,
    ContextFilterAgent,
    ReflexionAgent,
    SpeculativeAgent,
    CircuitBreakerAgent,
    TournamentAgent,
    ShadowAgent,
    CompilerAgent,
    CheckpointableAgent,
    DynamicFanOutAgent,
    SwarmAgent,
    MemoryConsolidationAgent,
    PriorityQueueAgent,
    MonteCarloAgent,
    GraphOfThoughtsAgent,
    BlackboardAgent,
    MixtureOfExpertsAgent,
    CoVeAgent,
    SagaAgent,
    LoadBalancerAgent,
    EnsembleAgent,
    TimeoutAgent,
    AdaptivePlannerAgent,
    RouterAgent,
    tool,
)
```

---

## Step 1 — Create Specialist Agents

Every orchestration pattern starts with specialist agents. Each one has a clear, focused responsibility.

```python
researcher = Agent(
    name="researcher",
    description="Finds factual information on any topic.",
    instruction="You are a research assistant. Provide accurate data with sources.",
    provider="google",
    model="gemini-3-flash-preview",
)

writer = Agent(
    name="writer",
    description="Writes clear, engaging prose.",
    instruction="You are a professional writer. Write polished, engaging content.",
    provider="google",
    model="gemini-3-flash-preview",
)

reviewer = Agent(
    name="reviewer",
    description="Reviews text for clarity, accuracy, and style.",
    instruction=(
        "You are a senior editor. Review the text and suggest improvements. "
        "Set state['quality'] to a float 0-1 indicating overall quality."
    ),
    provider="google",
    model="gemini-3-flash-preview",
)

coder = Agent(
    name="coder",
    description="Writes and debugs Python code.",
    instruction="You are a Python expert. Write clean, well-documented code.",
    provider="google",
    model="gemini-3-flash-preview",
)
```

**Key rule**: always provide a meaningful `description` — it is used by `RouterAgent` and `transfer_to_agent` to decide which agent to pick.

---

## Step 2 — SequentialAgent (Pipeline)

Runs sub-agents **one after another** in declared order. Each agent's response becomes the `user_message` for the next one.

### When to use

- Fixed, predictable pipelines (research → write → review)
- Processing chains where each step depends on the previous output
- ETL-style data transformations

### Implementation

```python
from nono.agent import SequentialAgent, Runner

pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer, reviewer],
)

runner = Runner(agent=pipeline)
response = runner.run("Write an article about renewable energy in Spain")
print(response)
# researcher gathers facts → writer drafts article → reviewer polishes it
```

### How data flows

```
User message ──► researcher ──► writer ──► reviewer ──► Final response
                  "facts..."     "draft..."  "polished..."
```

Each agent receives the **full session** (all events) plus the previous agent's response as `user_message`. This means the writer sees the researcher's output, and the reviewer sees the writer's draft.

### Monitoring with streaming

```python
for event in runner.stream("Write about renewable energy"):
    if event.event_type.value == "agent_message":
        print(f"[{event.author}] {event.content[:80]}...")
```

---

## Step 3 — ParallelAgent (Fan-out)

Runs sub-agents **concurrently** and collects all results. Uses `ThreadPoolExecutor` in sync mode and `asyncio.gather` in async mode.

### When to use

- Gathering information from independent sources
- Running multiple analyses on the same data
- Any task where sub-agents don't depend on each other

### Implementation

```python
from nono.agent import ParallelAgent, Runner

web_search = Agent(
    name="web_search",
    description="Searches the web for information.",
    instruction="Search the web and summarize findings.",
    provider="google",
)

db_search = Agent(
    name="db_search",
    description="Queries internal databases.",
    instruction="Query our databases and return relevant records.",
    provider="google",
)

gather = ParallelAgent(
    name="gather_info",
    sub_agents=[web_search, db_search],
    max_workers=2,  # thread pool size (sync mode)
)

runner = Runner(agent=gather)
response = runner.run("Find information about customer churn rates")
```

### How data flows

```
                    ┌──► web_search ──┐
User message ──────┤                  ├──► All results collected
                    └──► db_search ──┘
```

All sub-agents receive the **same** user message by default. Use `message_map` to send a different message to specific agents. Results come back in completion order. If a sub-agent raises an exception, an `ERROR` event is yielded and other agents continue.

### Per-agent messages with `message_map`

When each sub-agent needs to work on a different input, pass a `message_map`:

```python
from nono.agent import ParallelAgent, Runner

gather = ParallelAgent(
    name="gather",
    sub_agents=[web_search, db_search],
    message_map={
        "web_search": "AI trends 2026",
        "db_search": "SELECT * FROM sales WHERE quarter = 'Q1'",
    },
)

runner = Runner(agent=gather)
response = runner.run("Go")  # ctx.user_message is ignored for mapped agents
```

Agents **not** in the map still receive the original `user_message`. This lets you mix shared and custom inputs:

```python
# Only web_search gets a custom message; db_search gets the user's question
gather = ParallelAgent(
    name="gather",
    sub_agents=[web_search, db_search],
    message_map={"web_search": "Search for recent AI papers"},
)
runner.run("What are the latest trends?")  # db_search receives this
```

`message_map` can be updated between runs:

```python
gather.message_map = {"web_search": "New topic"}
runner.run("Run again")
```

### Collecting all results with `result_key`

By default, when `ParallelAgent` runs inside a `SequentialAgent`, only the
**last** `AGENT_MESSAGE` is forwarded to the next agent — the other N-1
responses are lost for the chaining mechanism.  Set `result_key` to
automatically collect **every** sub-agent's response into `session.state`:

```python
gather = ParallelAgent(
    name="gather",
    sub_agents=[web_search_agent, db_search_agent],
    result_key="parallel_results",
)

runner = Runner(agent=gather)
runner.run("Customer churn analysis")

# All responses available:
print(runner.session.state["parallel_results"])
# {"web_search": "AI trends are...", "db_search": "Sales data shows..."}
```

`result_key` works with `message_map`:

```python
gather = ParallelAgent(
    name="gather",
    sub_agents=[web_search_agent, db_search_agent],
    message_map={"web_search": "AI papers", "db_search": "Q1 revenue"},
    result_key="parallel_results",
)
```

When a sub-agent emits multiple `AGENT_MESSAGE` events, only the **last**
content is stored in the dict.

This is the recommended approach for feeding parallel results into the next
agent in a `SequentialAgent` pipeline:

```python
summarizer = LlmAgent(
    name="summarizer",
    instruction=(
        "Read state['parallel_results'] and produce a unified summary. "
        "The dict maps agent names to their responses."
    ),
)

pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[gather, summarizer],
)
```

### Async for true parallelism

```python
import asyncio

async def main():
    runner = Runner(agent=gather)
    response = await runner.run_async("Customer churn analysis")
    print(response)

asyncio.run(main())
```

In async mode, `asyncio.gather` is used instead of threads — ideal when sub-agents make I/O-bound LLM calls.

### Working with results

`run()` returns only the **last** `AGENT_MESSAGE` (from whichever agent finishes last). To access **all** results, use one of these strategies:

#### Option A: `stream()` — real-time

```python
runner = Runner(agent=gather)
for event in runner.stream("Query"):
    if event.event_type.value == "agent_message":
        print(f"[{event.author}] {event.content}")
# [web_search] AI trends are...
# [db_search] Sales data shows...
```

#### Option B: `session.events` — post-execution

```python
runner = Runner(agent=gather)
runner.run("Query")

results = {
    e.author: e.content
    for e in runner.session.events
    if e.event_type.value == "agent_message"
}
print(results["web_search"])
print(results["db_search"])
```

#### Option C: `session.state` — structured data via tools

Each sub-agent writes its result to a named key in `state`:

```python
from nono.agent import tool, ToolContext

@tool(description="Save result to state.")
def save_result(text: str, tool_context: ToolContext) -> str:
    tool_context.state[tool_context.agent_name] = text
    return "saved"

web_search = Agent(name="web_search", tools=[save_result], ...)
db_search = Agent(name="db_search", tools=[save_result], ...)

gather = ParallelAgent(name="gather", sub_agents=[web_search, db_search])
runner = Runner(agent=gather)
runner.run("Query")

print(runner.session.state["web_search"])
print(runner.session.state["db_search"])
```

This is the best approach when `ParallelAgent` is inside a `SequentialAgent` — the next agent reads `state` to combine the parallel results.

#### Option D: `shared_content` — named artifacts

```python
@tool(description="Publish findings.")
def publish(text: str, tool_context: ToolContext) -> str:
    tool_context.save_content(tool_context.agent_name, text, scope="shared")
    return "published"

# After run:
web_data = runner.session.shared_content.load("web_search").data
db_data = runner.session.shared_content.load("db_search").data
```

#### Option E: `result_key` — automatic collection (recommended)

The simplest approach — no custom tools needed.  Set `result_key` on
`ParallelAgent` and all responses are collected automatically:

```python
gather = ParallelAgent(
    name="gather",
    sub_agents=[web_search_agent, db_search_agent],
    result_key="parallel_results",
)

runner = Runner(agent=gather)
runner.run("Query")

print(runner.session.state["parallel_results"])
# {"web_search": "...", "db_search": "..."}
```

See [Collecting all results with result_key](#collecting-all-results-with-result_key) for details.

#### Comparison

| Method | When | Access | Best for |
| --- | --- | --- | --- |
| `result_key` | After parallel run | `state[key]` → `{name: response}` | **Default choice** for Sequential pipelines |
| `stream()` | Real-time | `event.author` + `event.content` | Logging, progress display |
| `session.events` | Post-execution | Filter by `AGENT_MESSAGE` | Inspection, debugging |
| `session.state` | During or after | `state["agent_name"]` | Custom structured data via tools |
| `shared_content` | During or after | `load("name").data` | Versioned artifacts, files |

---

## Step 4 — LoopAgent (Iterative Refinement)

Repeats sub-agents **in a loop** until a stop condition is satisfied or the maximum number of iterations is reached.

### When to use

- Iterative quality improvement (write → review → improve → review)
- Retry patterns with quality gates
- Agentic loops that converge toward a goal

### Implementation

```python
from nono.agent import LoopAgent, Runner

improver = Agent(
    name="improver",
    description="Improves text quality.",
    instruction=(
        "Review the previous text and improve it. "
        "After improving, set state['quality'] to a float 0-1."
    ),
    provider="google",
)

refine_loop = LoopAgent(
    name="refine",
    sub_agents=[improver],
    max_iterations=5,
    stop_condition=lambda state: state.get("quality", 0) > 0.9,
)

runner = Runner(agent=refine_loop)
response = runner.run("Write a professional email to a client about a project delay")
```

### How data flows

```
              ┌─────────────────────────────────┐
              │                                 │
User msg ──► │  improver ──► check quality ──► │ ──► Final (quality > 0.9 or max 5)
              │       ▲              │          │
              │       └──────────────┘          │
              └─────────────────────────────────┘
```

### Multi-agent loops

You can put multiple agents inside the loop — they run sequentially each iteration:

```python
loop = LoopAgent(
    name="write_review_loop",
    sub_agents=[writer, reviewer],
    max_iterations=3,
    stop_condition=lambda state: state.get("approved", False),
)
# Iteration 1: writer → reviewer
# Iteration 2: writer → reviewer (if not approved)
# Iteration 3: writer → reviewer (final attempt)
```

### Stop condition

The `stop_condition` is a callable that receives the session `state` dict and returns `True` to stop:

```python
# Stop on quality threshold
stop_condition = lambda state: state.get("quality", 0) > 0.9

# Stop on explicit approval
stop_condition = lambda state: state.get("approved") is True

# Stop when count reaches target
stop_condition = lambda state: state.get("items_processed", 0) >= 100
```

If no `stop_condition` is provided, the loop always runs `max_iterations` times.

---

## Step 5 — MapReduceAgent (Fan-out + Reduce)

Splits a problem across multiple mapper agents running in **parallel**, then feeds all results into a single **reduce** agent that produces the final output.

### When to use

- Summarising information from multiple sources
- Aggregating diverse perspectives into one report
- Any task where you can "divide and conquer"

### Example

```python
from nono.agent import Agent, MapReduceAgent, Runner

web_search = Agent(name="web_search", description="Search the web.", ...)
db_search = Agent(name="db_search", description="Search the database.", ...)
news_feed = Agent(name="news_feed", description="Search news sources.", ...)
summariser = Agent(name="summariser", instruction="Combine all inputs into a concise summary.", ...)

mapreduce = MapReduceAgent(
    name="multi_source_summary",
    sub_agents=[web_search, db_search, news_feed],  # mappers (parallel)
    reduce_agent=summariser,                          # reducer (single)
    result_key="source_results",                      # optional: save mapper outputs
)

runner = Runner(agent=mapreduce)
result = runner.run("Latest AI trends in healthcare")
```

### With message_map

```python
mapreduce = MapReduceAgent(
    name="targeted_search",
    sub_agents=[web_search, db_search],
    reduce_agent=summariser,
    message_map={
        "web_search": "AI trends in healthcare 2026",
        "db_search": "SELECT * FROM reports WHERE topic='AI healthcare'",
    },
)
```

---

## Step 6 — ConsensusAgent (Vote + Judge)

Runs multiple agents on the **same input** in parallel, then a **judge** reviews all answers and produces a single consensus response.

### When to use

- Fact-checking with diverse models or prompts
- Getting multiple perspectives and reconciling them
- Ensemble answers for higher accuracy

### Example

```python
from nono.agent import Agent, ConsensusAgent, Runner

model_a = Agent(name="model_a", provider="google", ...)
model_b = Agent(name="model_b", provider="openai", ...)
model_c = Agent(name="model_c", provider="groq", ...)
judge = Agent(
    name="judge",
    instruction="Review all answers and produce a single authoritative response.",
    ...
)

consensus = ConsensusAgent(
    name="fact_check",
    sub_agents=[model_a, model_b, model_c],  # voters (parallel)
    judge_agent=judge,                         # synthesises consensus
    result_key="votes",                        # optional: save voter answers
)

runner = Runner(agent=consensus)
result = runner.run("What is the capital of Australia?")
```

---

## Step 7 — ProducerReviewerAgent (Produce + Review Loop)

An iterative loop where a **producer** generates content and a **reviewer** evaluates it. The loop repeats until the reviewer **approves** (by including a keyword in its response) or `max_iterations` is reached.

### When to use

- Content generation with quality gates
- Code generation with automated review
- Any task where output quality improves through feedback cycles

### Example

```python
from nono.agent import Agent, ProducerReviewerAgent, Runner

writer = Agent(
    name="writer",
    instruction="Write a blog post on the given topic. Incorporate any feedback provided.",
    ...
)
editor = Agent(
    name="editor",
    instruction=(
        "Review the text for clarity, accuracy, and style. "
        "If the quality is high enough, respond with 'APPROVED'. "
        "Otherwise, provide specific feedback for improvement."
    ),
    ...
)

pr = ProducerReviewerAgent(
    name="blog_pipeline",
    producer=writer,
    reviewer=editor,
    max_iterations=3,
    approval_keyword="APPROVED",
)

runner = Runner(agent=pr)
result = runner.run("Write a blog post about sustainable AI")
```

---

## Step 8 — DebateAgent (Adversarial Debate)

Adversarial debate — two agents argue in rounds, a **judge** renders the final verdict. Useful when you want contrasting viewpoints before a decision.

### When to use

- Policy review with pro/con arguments
- Adversarial reasoning to stress-test an idea
- Any task that benefits from dialectical exploration

### Example

```python
from nono.agent import Agent, DebateAgent, Runner

optimist = Agent(
    name="optimist",
    instruction="Argue in favour of the proposition. Be persuasive.",
    ...
)
pessimist = Agent(
    name="pessimist",
    instruction="Argue against the proposition. Be critical.",
    ...
)
arbiter = Agent(
    name="arbiter",
    instruction=(
        "Read both sides and render a verdict. "
        "Include 'RESOLVED' when you reach a conclusion."
    ),
    ...
)

debate = DebateAgent(
    name="policy_debate",
    agent_a=optimist,
    agent_b=pessimist,
    judge=arbiter,
    max_rounds=3,
    resolution_keyword="RESOLVED",
)

runner = Runner(agent=debate)
result = runner.run("Should companies adopt a 4-day work week?")
```

---

## Step 9 — EscalationAgent (Tiered Escalation)

Try agents in order, stop at the **first success**. Escalates from cheap/fast to expensive/powerful — useful for cost-optimised inference.

### When to use

- Tiered customer support (bot → junior → senior)
- Cost-optimised inference (small model → large model)
- Fallback chains where the first capable agent should answer

### Example

```python
from nono.agent import Agent, EscalationAgent, Runner

fast = Agent(name="fast", instruction="Answer concisely.", provider="groq", ...)
medium = Agent(name="medium", instruction="Answer in detail.", provider="openai", ...)
powerful = Agent(name="powerful", instruction="Provide expert analysis.", provider="google", ...)

escalation = EscalationAgent(
    name="tiered_support",
    sub_agents=[fast, medium, powerful],
    failure_keyword="ESCALATE",
)

runner = Runner(agent=escalation)
result = runner.run("Explain the P vs NP problem")
# fast tries first; if response contains "ESCALATE", medium tries, etc.
```

---

## Step 10 — SupervisorAgent (LLM Supervisor)

An LLM-powered supervisor that **delegates** tasks to sub-agents, **evaluates** their output, and can **re-delegate**. Unlike `RouterAgent` (which routes once), the supervisor actively monitors and iterates.

### When to use

- Complex tasks that require iterative delegation and quality control
- Workflows where the orchestrator must evaluate intermediate results
- Project-manager patterns with active monitoring

### Example

```python
from nono.agent import Agent, SupervisorAgent, Runner

researcher = Agent(name="researcher", instruction="Research the topic.", ...)
writer = Agent(name="writer", instruction="Write an article.", ...)
reviewer = Agent(name="reviewer", instruction="Review for quality.", ...)

supervisor = SupervisorAgent(
    name="project_lead",
    sub_agents=[researcher, writer, reviewer],
    model="gemini-3-flash-preview",
    provider="google",
    supervisor_instruction="Delegate research first, then writing, then review. Re-delegate if quality is low.",
    max_iterations=5,
)

runner = Runner(agent=supervisor)
result = runner.run("Create a comprehensive report on AI in healthcare")
```

---

## Step 11 — VotingAgent (Majority Vote)

Majority-vote orchestration — N agents answer the same question **in parallel**, the most frequent normalised response wins. No LLM judge needed — purely deterministic counting.

### When to use

- Ensemble classification (multiple models vote on a label)
- Fact-checking with diverse models
- Any scenario where the majority answer is likely correct

### Example

```python
from nono.agent import Agent, VotingAgent, Runner

model_a = Agent(name="model_a", provider="google", instruction="Classify the sentiment.", ...)
model_b = Agent(name="model_b", provider="openai", instruction="Classify the sentiment.", ...)
model_c = Agent(name="model_c", provider="groq", instruction="Classify the sentiment.", ...)

voting = VotingAgent(
    name="sentiment_ensemble",
    sub_agents=[model_a, model_b, model_c],
    max_workers=3,
    normalize=lambda s: s.strip().lower(),
    result_key="votes",
)

runner = Runner(agent=voting)
result = runner.run("I absolutely love this product!")
# All three vote in parallel; most frequent normalised answer wins
# session.state["votes"] == {"model_a": "positive", "model_b": "positive", "model_c": "positive"}
```

---

## Step 12 — HandoffAgent (Peer-to-Peer Handoff)

Agents transfer **full control** to each other via handoff directives. Unlike `transfer_to_agent` (agent-as-tools, the caller retains control), handoff means the receiving agent **takes full ownership** of the conversation.

### When to use

- Triage routing (a front-desk agent routes to specialists)
- Support desks where different agents handle different domains
- Expert transfer where the original agent should not remain in the loop

### Example

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
    handoff_rules={
        "triage": [math, history],
        "math_tutor": [triage],
        "history_tutor": [triage],
    },
    max_handoffs=5,
)

runner = Runner(agent=handoff)
result = runner.run("What year did the French Revolution start?")
# triage → HANDOFF: history_tutor → history_tutor answers
```

### How it works

1. The `entry_agent` receives the user message with the full conversation history.
2. If the agent's response contains `HANDOFF: <target_name>`, control transfers to that target.
3. The target receives the full conversation history (including the handoff directive).
4. The loop continues until an agent completes without handing off, or `max_handoffs` is reached.

### Custom handoff keyword

```python
handoff = HandoffAgent(
    name="support",
    entry_agent=triage,
    handoff_rules={"triage": [billing, tech]},
    handoff_keyword="TRANSFER:",  # agents write "TRANSFER: billing"
)
```

---

## Step 13 — GroupChatAgent (N-Agent Group Chat)

N-agent group chat with manager-controlled speaker selection. A central manager selects which agent speaks next each round. All agents see the full conversation history.

### When to use

- Collaborative writing (writer + reviewer take turns)
- Multi-agent brainstorming
- Iterative refinement where different specialists contribute in turns

### Example

```python
from nono.agent import Agent, GroupChatAgent, Runner

writer = Agent(
    name="writer",
    instruction="Write marketing copy.",
    provider="google",
)
reviewer = Agent(
    name="reviewer",
    instruction="Review copy for clarity and persuasion. Say APPROVED when satisfied.",
    provider="google",
)

chat = GroupChatAgent(
    name="creative_team",
    sub_agents=[writer, reviewer],
    speaker_selection="round_robin",
    max_rounds=6,
    termination_keyword="APPROVED",
)

runner = Runner(agent=chat)
result = runner.run("Create a tagline for an AI startup")
```

### Speaker selection strategies

| Strategy | Description |
|----------|-------------|
| `"round_robin"` | Cycle through agents in order (default) |
| `"llm"` | An LLM picks the best next speaker each round |
| `callable` | Custom function `(messages, agents) → agent` |

### LLM-based speaker selection

```python
chat = GroupChatAgent(
    name="creative_team",
    sub_agents=[writer, reviewer, designer],
    speaker_selection="llm",
    provider="google",
    model="gemini-3-flash-preview",
    max_rounds=6,
)
```

### Termination

The chat terminates when:
- `max_rounds` is reached
- A response contains `termination_keyword` (if set)
- `termination_condition(messages)` returns `True` (if set)

---

## Step 14 — HierarchicalAgent (Multi-Level Hierarchy)

A multi-level tree-shaped orchestration where an **LLM manager** delegates to department heads — which may themselves be orchestration agents (e.g. `SequentialAgent`, `SupervisorAgent`) with their own sub-agents.

Unlike `SupervisorAgent` (flat pool of workers), `HierarchicalAgent` sees the **full org-chart** and delegates across multiple rounds to different departments before synthesising a final answer.

```
HierarchicalAgent("cto")
├── SequentialAgent("backend_team")
│   ├── LlmAgent("architect")
│   └── LlmAgent("developer")
├── SupervisorAgent("qa_team")
│   ├── LlmAgent("tester")
│   └── LlmAgent("security")
└── LlmAgent("devops")
```

```python
from nono.agent import (
    Agent, SequentialAgent, HierarchicalAgent, Runner,
)

architect = Agent(name="architect", instruction="Design the system.", provider="google")
developer = Agent(name="developer", instruction="Implement the code.", provider="google")

backend = SequentialAgent(
    name="backend_team",
    description="Backend development pipeline.",
    sub_agents=[architect, developer],
)
qa = Agent(name="qa", description="Quality assurance.", instruction="Review code.", provider="google")

cto = HierarchicalAgent(
    name="cto",
    provider="google",
    sub_agents=[backend, qa],
    max_iterations=3,
    manager_instruction="Delegate to backend_team first, then QA.",
)

runner = Runner(cto)
result = runner.run("Build a REST API for user management")
```

### How it works

1. The **manager** sees the org-chart (including nested agents) and decides which department to delegate to.
2. The chosen department runs its own internal pipeline autonomously.
3. The manager evaluates the output and can delegate to another department or **synthesise** a final answer.
4. When the loop ends (either by explicit synthesis or `max_iterations`), the manager produces a cohesive synthesis from all collected department outputs.

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Department-head agents |
| `model` | `str \| None` | provider default | LLM for manager decisions |
| `provider` | `str` | `"google"` | LLM provider |
| `manager_instruction` | `str` | `""` | Extra instructions for the manager |
| `max_iterations` | `int` | `3` | Maximum delegation rounds |
| `synthesis_prompt` | `str` | `""` | Custom prompt for final synthesis |
| `result_key` | `str \| None` | `None` | Store synthesis in `session.state` |

---

## Step 15 — GuardrailAgent (Pre/Post Validation)

Wraps a main agent with optional **pre-validator** (input check/transform) and **post-validator** (output check). If the post-validator's response contains the `rejection_keyword`, the main agent retries automatically.

```
      ┌─────────────┐
      │ Pre-Validator│──ERROR──► STOP
      └──────┬──────┘
             │ (transformed input)
      ┌──────▼──────┐
      │  Main Agent  │◄─── retry
      └──────┬──────┘       │
             │               │
      ┌──────▼───────┐      │
      │ Post-Validator│──REJECTED?
      └──────┬───────┘
             │ (APPROVED)
          OUTPUT
```

```python
writer = Agent(name="writer", instruction="Write marketing copy.", provider="google")
checker = Agent(name="checker", instruction="Reply REJECTED if toxic, else APPROVED.", provider="google")

safe = GuardrailAgent(
    name="safe_writer",
    main_agent=writer,
    post_validator=checker,
    rejection_keyword="REJECTED",
    max_retries=2,
)

runner = Runner(safe)
result = runner.run("Write a tagline for our product")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `main_agent` | `BaseAgent` | *(required)* | The core agent whose output is validated |
| `pre_validator` | `BaseAgent \| None` | `None` | Checks/transforms input before main agent |
| `post_validator` | `BaseAgent \| None` | `None` | Validates output; rejection triggers retry |
| `rejection_keyword` | `str` | `"REJECTED"` | Keyword that triggers retry |
| `max_retries` | `int` | `1` | Max retries on rejection |
| `result_key` | `str \| None` | `None` | Store validated output |

---

## Step 16 — BestOfNAgent (Best-of-N Sampling)

Runs the same agent **N times in parallel** and picks the best response using a scoring function. Ideal for creative tasks where quality varies between runs.

```
      ┌──────────┐
      │  Agent×1  │──► score=0.4
      │  Agent×2  │──► score=0.9  ◄── BEST
      │  Agent×3  │──► score=0.6
      └──────────┘
```

```python
writer = Agent(name="writer", instruction="Write a headline.", provider="google")

best = BestOfNAgent(
    name="best_writer",
    agent=writer,
    n=3,
    score_fn=lambda r: float(len(r)),
    result_key="scoring",
)

runner = Runner(best)
result = runner.run("Headline for an AI conference")
# session.state["scoring"] → {"best_index": 2, "best_score": 47.0, "all_scores": [...]}
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | Agent to run N times |
| `n` | `int` | `3` | Number of parallel runs |
| `score_fn` | `Callable[[str], float] \| None` | `None` (uses `len`) | Scoring function |
| `max_workers` | `int` | `4` | Max threads for parallel execution |
| `result_key` | `str \| None` | `None` | Store scoring details |

---

## Step 17 — BatchAgent (Batch Processing)

Processes a **list of items** through one agent with concurrency control. Items can be provided statically or resolved from `session.state` at runtime.

```
items = ["text1", "text2", "text3"]
        ┌──────────┐
  ──────►│ Agent(1) │──► result[0]
  ──────►│ Agent(2) │──► result[1]
  ──────►│ Agent(3) │──► result[2]
        └──────────┘
```

```python
classifier = Agent(name="classifier", instruction="Classify sentiment.", provider="google")

batch = BatchAgent(
    name="batch_classify",
    agent=classifier,
    items=["I love this!", "Terrible.", "It's okay."],
    template="Classify: {item}",
    max_workers=3,
    result_key="classifications",
)

runner = Runner(batch)
result = runner.run("Classify all items")
# session.state["classifications"] → {0: "positive", 1: "negative", 2: "neutral"}
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | Agent to process each item |
| `items` | `list[str] \| None` | `None` | Static item list |
| `items_key` | `str \| None` | `None` | Key in `session.state` |
| `template` | `str \| None` | `None` | Template with `{item}` placeholder |
| `max_workers` | `int` | `4` | Max concurrent workers |
| `result_key` | `str \| None` | `None` | Store `{index: response}` dict |

---

## Step 18 — CascadeAgent (Progressive Cascade)

Tries progressively more capable (and expensive) agents in sequence, stopping when a **quality threshold** is met. This avoids unnecessary expensive calls when a cheap model suffices.

```
Stage 1 (flash) ──► score=0.3 < 0.8 → NEXT
Stage 2 (pro)   ──► score=0.9 ≥ 0.8 → STOP ✓
Stage 3 (opus)  ──► never reached
```

```python
flash = Agent(name="flash", instruction="Answer.", provider="google", model="gemini-3-flash-preview")
pro = Agent(name="pro", instruction="Answer thoroughly.", provider="google", model="gemini-2.5-pro-preview-06-05")

cascade = CascadeAgent(
    name="smart",
    sub_agents=[flash, pro],
    score_fn=lambda r: 1.0 if len(r) > 200 else 0.3,
    threshold=0.8,
    result_key="cascade_info",
)

runner = Runner(cascade)
result = runner.run("Explain quantum entanglement")
# If flash gives a long-enough answer, pro never runs
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Stages: cheapest → most capable |
| `score_fn` | `Callable[[str], float]` | *(required)* | Scores output (0.0–1.0) |
| `threshold` | `float` | `0.8` | Minimum score to accept |
| `result_key` | `str \| None` | `None` | Store `stage`, `agent`, `score`, `met_threshold` |

---

## Step 19 — TreeOfThoughtsAgent (Tree-of-Thoughts)

Explores multiple reasoning branches at each depth level, evaluates them, keeps only the top-k, and expands further — a BFS implementation of the Tree of Thoughts algorithm (Yao et al., 2023).

### When to use

- Complex reasoning where a single chain of thought may get stuck
- Math problems, planning tasks, puzzle solving
- When you want to explore multiple paths and pick the best

### Implementation

```python
from nono.agent import Agent, Runner, TreeOfThoughtsAgent

thinker = Agent(
    name="thinker",
    instruction="You are a creative problem solver.",
    provider="google",
)

tot = TreeOfThoughtsAgent(
    name="reasoner",
    agent=thinker,
    evaluate_fn=lambda r: 1.0 if "correct" in r.lower() else 0.3,
    n_branches=3,
    top_k=2,
    max_depth=3,
    threshold=0.9,
    result_key="tree_info",
)

runner = Runner(tot)
result = runner.run("What is the optimal strategy for the 24-game with 1, 5, 5, 5?")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | Agent that generates each thought branch |
| `evaluate_fn` | `Callable[[str], float]` | *(required)* | Scoring function (higher is better) |
| `n_branches` | `int` | `3` | Branches per frontier node per depth |
| `top_k` | `int` | `2` | How many branches survive pruning |
| `max_depth` | `int` | `3` | Maximum tree depth |
| `threshold` | `float` | `0.9` | Score for early acceptance |
| `max_workers` | `int` | `4` | Concurrent threads |
| `result_key` | `str \| None` | `None` | Store `depth`, `score`, `met_threshold`, `path` |

---

## Step 20 — PlannerAgent (Plan-and-Execute)

Uses an LLM call to decompose a task into steps with dependencies, executes them (parallel where possible), and synthesises a final answer — inspired by CrewAI's `planning=True` and LangGraph's plan-and-execute pattern.

### When to use

- Multi-step tasks where the plan depends on the question
- When steps have dependencies (A before B, C and D in parallel)
- Project management, research workflows, data processing pipelines

### Implementation

```python
from nono.agent import Agent, Runner, PlannerAgent

researcher = Agent(name="researcher", description="Finds facts", provider="google")
writer = Agent(name="writer", description="Writes content", provider="google")
reviewer = Agent(name="reviewer", description="Reviews quality", provider="google")

planner = PlannerAgent(
    name="project_manager",
    provider="google",
    sub_agents=[researcher, writer, reviewer],
    max_steps=5,
    result_key="plan_info",
)

runner = Runner(planner)
result = runner.run("Write a blog post about quantum computing")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Pool of available agents |
| `model` | `str \| None` | `None` | LLM model for planning |
| `provider` | `str` | `"google"` | LLM provider |
| `planning_instruction` | `str` | `""` | Extra instruction for the planner |
| `max_steps` | `int` | `5` | Maximum steps in plan |
| `synthesis_prompt` | `str` | `""` | Custom synthesis prompt |
| `result_key` | `str \| None` | `None` | Store `plan`, `step_results`, `synthesis` |

---

## Step 21 — SubQuestionAgent (Question Decomposition)

Breaks a complex question into targeted sub-questions, assigns each to the most appropriate agent, dispatches them in parallel, and synthesises the answers — inspired by LlamaIndex's `SubQuestionQueryEngine`.

### When to use

- Multi-faceted research questions that span different domains
- When one question needs multiple specialist perspectives
- Analytical reports requiring diverse data sources

### Implementation

```python
from nono.agent import Agent, Runner, SubQuestionAgent

market = Agent(name="market", description="Market analysis expert", provider="google")
tech = Agent(name="tech", description="Technology trends expert", provider="google")

analyst = SubQuestionAgent(
    name="analyst",
    provider="google",
    sub_agents=[market, tech],
    max_sub_questions=4,
    result_key="analysis",
)

runner = Runner(analyst)
result = runner.run("How does AI affect the global economy?")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Pool of specialist agents |
| `model` | `str \| None` | `None` | LLM model for decomposition |
| `provider` | `str` | `"google"` | LLM provider |
| `decomposition_instruction` | `str` | `""` | Extra instruction for decomposition |
| `max_sub_questions` | `int` | `5` | Maximum sub-questions |
| `max_workers` | `int` | `4` | Concurrent threads (sync) |
| `synthesis_prompt` | `str` | `""` | Custom synthesis prompt |
| `result_key` | `str \| None` | `None` | Store `sub_questions`, `synthesis` |

---

## Step 22 — ContextFilterAgent (Context Filtering)

Filters the accumulated event history before delegating to sub-agents — reducing noise and hallucination in long multi-agent workflows.  Inspired by AutoGen's `MessageFilterAgent`.

### When to use

- Long workflows where accumulated context is confusing agents
- When different agents need different context windows
- Whitelisting/blacklisting event sources per agent

### Implementation

```python
from nono.agent import Agent, Runner, ContextFilterAgent

analyst = Agent(name="analyst", instruction="Analyse the data.", provider="google")
writer = Agent(name="writer", instruction="Write the report.", provider="google")

focused = ContextFilterAgent(
    name="focused",
    sub_agents=[analyst, writer],
    max_history=10,
    exclude_sources=["debug_logger"],
    mode="sequential",
    result_key="filter_stats",
)

runner = Runner(focused)
result = runner.run("Summarise the quarterly results")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `sub_agents` | `list[BaseAgent]` | `[]` | Agents receiving filtered context |
| `filter_fn` | `Callable` | `None` | Custom `(agent, events) → events` filter |
| `max_history` | `int \| None` | `None` | Maximum recent events per agent |
| `include_sources` | `list[str] \| None` | `None` | Whitelist of event authors |
| `exclude_sources` | `list[str] \| None` | `None` | Blacklist of event authors |
| `mode` | `str` | `"sequential"` | `"sequential"` or `"parallel"` |
| `result_key` | `str \| None` | `None` | Store `filter_stats`, `original_events` |

### Custom filter function

When the built-in filters are not enough, pass a custom `filter_fn`. It receives the target agent and the full event list, and must return the filtered list:

```python
from nono.agent import ContextFilterAgent, BaseAgent, Event

def only_tool_results(agent: BaseAgent, events: list[Event]) -> list[Event]:
    """Keep only TOOL_RESULT events for the analyst; keep all for writer."""
    if agent.name == "analyst":
        return [e for e in events if e.event_type.value == "tool_result"]
    return events  # writer sees everything

focused = ContextFilterAgent(
    name="focused",
    sub_agents=[analyst, writer],
    filter_fn=only_tool_results,
)
```

### Combining with state isolation

`ContextFilterAgent` filters **events** but still shares `session.state`. To isolate state completely, create a new `Session` (see [State Isolation Patterns](README_agent.md#state-isolation-patterns)):

```python
from nono.agent import Session, InvocationContext

# Filtered events + shared state (default)
focused = ContextFilterAgent(name="f", sub_agents=[analyst], max_history=5)

# Full isolation — separate session
isolated = Session()
ctx = InvocationContext(session=isolated, user_message="Analyse independently")
response = analyst.run(ctx)
```

---

## Step 23 — ReflexionAgent (Self-Improvement)

Implements the Reflexion algorithm (Shinn et al., 2023): generate → evaluate → reflect → retry — with persistent memory of lessons learned across attempts.

### When to use

- Code generation where test feedback drives improvement
- When the agent should learn from its mistakes within a session
- Quality-critical tasks that benefit from self-correction

### Implementation

```python
from nono.agent import Agent, Runner, ReflexionAgent

coder = Agent(name="coder", instruction="Write Python code.", provider="google")
reviewer = Agent(name="reviewer", instruction="Review the code, reply PASS or FAIL with feedback.", provider="google")

learner = ReflexionAgent(
    name="learner",
    agent=coder,
    evaluator=reviewer,
    score_fn=lambda r: 1.0 if "PASS" in r else 0.3,
    threshold=0.8,
    max_attempts=3,
    result_key="reflexion_info",
)

runner = Runner(learner)
result = runner.run("Write a function to merge two sorted lists")
```

### Parameters

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `agent` | `BaseAgent` | *(required)* | Main agent that generates responses |
| `evaluator` | `BaseAgent` | *(required)* | Agent that evaluates and provides feedback |
| `score_fn` | `Callable[[str], float]` | *(required)* | Scores evaluator response |
| `threshold` | `float` | `0.8` | Score to accept |
| `max_attempts` | `int` | `3` | Maximum generation attempts |
| `memory_key` | `str` | `"reflexion_memory"` | Key in `session.state` for accumulated lessons |
| `result_key` | `str \| None` | `None` | Store `attempts`, `accepted_attempt`, `score` |

---

## Step 24 — RouterAgent (LLM-powered Routing)

Uses a **lightweight LLM call** to decide **which agents** to use and **how** to
execute them.  The LLM picks both the agents and the execution mode:

| Mode | LLM response | What happens |
| --- | --- | --- |
| `single` | `{"mode": "single", "agents": ["coder"]}` | Delegate to one agent |
| `sequential` | `{"mode": "sequential", "agents": ["researcher", "writer"]}` | Run agents in order (pipeline) |
| `parallel` | `{"mode": "parallel", "agents": ["web", "db"]}` | Run agents concurrently |
| `loop` | `{"mode": "loop", "agents": ["refiner"], "max_iterations": 3}` | Repeat one agent N times |

This makes `RouterAgent` a **true orchestrator** that composes `SequentialAgent`,
`ParallelAgent`, and `LoopAgent` dynamically at runtime.

### When to use

- Multi-domain systems where both **who** and **how** should be decided dynamically
- When you don't want to hardcode execution patterns
- Intent classification + automatic pipeline assembly
- When `transfer_to_agent` is too implicit (embedded in the main agent's conversation)

### How it works

```
                                        ┌──► single:     agent._run_impl()
                                        │
User message ──► RouterAgent ──► LLM ──►├──► sequential: SequentialAgent(agents)
                                        │
                                        ├──► parallel:   ParallelAgent(agents)
                                        │
                                        └──► loop:       LoopAgent(agent, max_iter)
```

1. The LLM receives all sub-agents (names + descriptions) and the 4 execution modes
2. It returns a JSON: `{"mode": "...", "agents": [...], "message": "...", "max_iterations": N}`
3. The router resolves the agent names and builds an ephemeral workflow agent
4. The workflow agent executes and its events flow back through the session
5. Fallback: unknown agents are skipped; if none resolve, the first sub-agent is used

### Implementation

```python
from nono.agent import RouterAgent, Runner

router = RouterAgent(
    name="orchestrator",
    model="gemini-3-flash-preview",
    provider="google",
    sub_agents=[researcher, writer, reviewer, coder],
    max_iterations=3,  # default for loop mode
)

runner = Runner(agent=router)

# LLM picks single mode → routes to "coder"
response = runner.run("Write a fibonacci function in Python")

# LLM picks sequential mode → researcher then writer
response = runner.run("Research AI trends and write a blog post about them")

# LLM picks parallel mode → researcher and coder in parallel
response = runner.run("Find Python best practices and generate a linter config")

# LLM picks loop mode → reviewer iterates 3 times
response = runner.run("Review and improve this draft until it's polished")
```

### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Orchestrator name |
| `sub_agents` | `list[BaseAgent]` | `[]` | Candidate agents |
| `model` | `str \| None` | provider default | LLM model for routing |
| `provider` | `str` | `"google"` | LLM provider for routing |
| `routing_instruction` | `str` | `""` | Extra rules appended to the routing prompt |
| `temperature` | `float` | `0.0` | LLM temperature (low = deterministic) |
| `max_iterations` | `int` | `3` | Default max iterations for loop mode |
| `api_key` | `str \| None` | `None` | API key override |

### Custom routing instructions

Add business rules to influence the routing and mode selection:

```python
router = RouterAgent(
    name="support",
    provider="google",
    sub_agents=[coder, writer, researcher],
    routing_instruction=(
        "If the user asks for an article, use sequential mode: researcher → writer.\n"
        "If the user mentions 'bug' or 'error', use single mode with 'coder'.\n"
        "If the user asks for comparison data, use parallel mode."
    ),
)
```

### LLM response format

```json
{
  "mode": "sequential",
  "agents": ["researcher", "writer"],
  "message": "optionally refined message",
  "max_iterations": 3
}
```

- `mode`: One of `single`, `sequential`, `parallel`, `loop`
- `agents`: List of agent names (1 for single/loop, 1+ for sequential/parallel)
- `message`: Optional — refines the user message before forwarding
- `max_iterations`: Optional — only for loop mode (defaults to `max_iterations` param)

**Backward compatible**: the legacy format `{"agent_name": "coder"}` still works and is treated as single mode.

### Events emitted

| Event | Description |
| --- | --- |
| `AGENT_TRANSFER` | After routing decision — includes `mode` in `data` |
| Sub-agent events | Whatever the orchestrated agents produce |

```python
for event in runner.stream("Research and write an article"):
    if event.event_type.value == "agent_transfer":
        print(f"Mode: {event.data['mode']}, Target: {event.data['target_agent']}")
```

### Async support

```python
import asyncio

async def main():
    runner = Runner(agent=router)
    response = await runner.run_async("Explain quantum computing")
    print(response)

asyncio.run(main())
```

---

## Step 25 — transfer_to_agent (Dynamic Delegation)

Unlike `RouterAgent` (which is a separate orchestration layer), `transfer_to_agent` is a **tool embedded inside an LlmAgent**. The LLM decides **when** to delegate as part of its normal conversation flow.

### When to use

- Open-ended conversations where the agent sometimes needs help
- Multi-turn chats where delegation happens mid-conversation
- When you want the coordinator to synthesize sub-agent responses

### Implementation

```python
from nono.agent import Agent, Runner

# Specialists
math = Agent(
    name="math_expert",
    description="Solves complex mathematical problems.",
    instruction="You are a math expert.",
    provider="google",
)

history = Agent(
    name="history_expert",
    description="Answers questions about historical events.",
    instruction="You are a history expert.",
    provider="google",
)

# Coordinator — transfer_to_agent is auto-registered
coordinator = Agent(
    name="coordinator",
    instruction="You are a general assistant. Delegate to specialists when needed.",
    provider="google",
    sub_agents=[math, history],
)

runner = Runner(agent=coordinator)

# Simple question: coordinator answers directly
response = runner.run("What time is it?")

# Math question: coordinator delegates to math_expert via transfer_to_agent
response = runner.run("What is the derivative of x³ + 2x?")

# History question: coordinator delegates to history_expert
response = runner.run("When did the Roman Empire fall?")
```

### How it works internally

1. `sub_agents` triggers auto-registration of the `transfer_to_agent` tool
2. The tool schema is `{agent_name: str, message: str}`
3. The tool description includes all available sub-agents
4. The system prompt includes a reminder about available agents
5. When the LLM calls the tool, the framework runs the sub-agent and returns the result
6. The coordinator can use the result, delegate again, or answer directly

### Key difference from RouterAgent

| | `RouterAgent` | `transfer_to_agent` |
| --- | --- | --- |
| **Architecture** | Separate orchestration layer | Tool inside an LlmAgent |
| **Decision point** | Before agent runs | During conversation |
| **LLM cost** | 1 cheap routing call + 1 agent call | 1 agent call (may include delegation) |
| **Multi-delegation** | One agent per request | Can delegate multiple times per turn |
| **Coordinator synthesis** | No (just forwards) | Yes (coordinator sees sub-agent results) |
| **Control** | Explicit, testable | Implicit (LLM decides) |

---

## Step 26 — Composite Patterns

The real power of Nono's orchestration is that all strategies are `BaseAgent` subclasses — they compose freely.

### Pattern 1: Router → Sequential Pipeline

Route to a specialized pipeline based on task type:

```python
# Code pipeline: generate → test → review
code_pipeline = SequentialAgent(
    name="code_pipeline",
    description="Writes, tests, and reviews Python code.",
    sub_agents=[coder, tester, reviewer],
)

# Writing pipeline: research → write → edit
writing_pipeline = SequentialAgent(
    name="writing_pipeline",
    description="Researches and writes polished articles.",
    sub_agents=[researcher, writer, reviewer],
)

# Router picks the right pipeline
router = RouterAgent(
    name="task_router",
    provider="google",
    sub_agents=[code_pipeline, writing_pipeline],
)

runner = Runner(agent=router)
response = runner.run("Write a Python web scraper for news sites")
# → Routes to code_pipeline → coder → tester → reviewer
```

### Pattern 2: Sequential with Parallel Fan-out

Research phase gathers data in parallel, then a writer synthesizes:

```python
# Parallel research
parallel_research = ParallelAgent(
    name="research_phase",
    description="Gathers information from multiple sources.",
    sub_agents=[web_search, db_search, news_agent],
)

# Sequential: parallel research → writer → reviewer
pipeline = SequentialAgent(
    name="full_pipeline",
    sub_agents=[parallel_research, writer, reviewer],
)
```

### Pattern 3: Router → Loop

Route to a refinement loop for quality-critical tasks:

```python
quality_loop = LoopAgent(
    name="quality_loop",
    description="Iteratively improves content until quality threshold is met.",
    sub_agents=[writer, reviewer],
    max_iterations=3,
    stop_condition=lambda state: state.get("quality", 0) > 0.9,
)

quick_answer = Agent(
    name="quick_answer",
    description="Gives brief factual answers.",
    instruction="Answer briefly and factually.",
    provider="google",
)

router = RouterAgent(
    name="smart_router",
    provider="google",
    sub_agents=[quality_loop, quick_answer],
    routing_instruction="Use quality_loop for content creation. Use quick_answer for factual questions.",
)
```

### Pattern 4: Coordinator with Tools + Sub-agents

A coordinator that has its own tools AND can delegate:

```python
@tool(description="Gets the current date and time.")
def get_datetime() -> str:
    from datetime import datetime
    return datetime.now().isoformat()

coordinator = Agent(
    name="coordinator",
    instruction="You are a project manager. Use tools and delegate to specialists.",
    provider="google",
    tools=[get_datetime],
    sub_agents=[coder, writer, researcher],
)
```

### Pattern 5: Nested Routers

For large systems, routers can route to other routers:

```python
# Technical router
tech_router = RouterAgent(
    name="tech_router",
    description="Handles technical questions (code, data, DevOps).",
    provider="google",
    sub_agents=[coder, data_analyst, devops_agent],
)

# Creative router
creative_router = RouterAgent(
    name="creative_router",
    description="Handles creative tasks (writing, design, marketing).",
    provider="google",
    sub_agents=[writer, designer, marketer],
)

# Top-level router
main_router = RouterAgent(
    name="main_router",
    provider="google",
    sub_agents=[tech_router, creative_router],
)
```

### Pattern 6: ProducerReviewer with Composite Agents

The `producer` and `reviewer` parameters accept any `BaseAgent` — including other orchestration agents. This means you can use a full pipeline as producer and a multi-model consensus as reviewer:

```python
# Producer = sequential pipeline (research → write)
research_then_write = SequentialAgent(
    name="research_write",
    sub_agents=[researcher, writer],
)

# Reviewer = consensus of 3 different models
multi_model_review = ConsensusAgent(
    name="multi_review",
    sub_agents=[reviewer_gpt, reviewer_gemini, reviewer_groq],
    judge_agent=final_judge,
)

# Compose: both producer and reviewer are orchestrations
pr = ProducerReviewerAgent(
    name="quality_pipeline",
    producer=research_then_write,       # SequentialAgent
    reviewer=multi_model_review,        # ConsensusAgent
    max_iterations=3,
)
```

### Pattern 7: MapReduce with ProducerReviewer as Reducer

The `reduce_agent` can be any orchestration. Here the reducer is a produce-review loop that iterates until the summary is approved:

```python
# Reducer = produce + review loop
summary_loop = ProducerReviewerAgent(
    name="summary_refine",
    producer=summariser,
    reviewer=quality_checker,
    max_iterations=2,
    approval_keyword="APPROVED",
)

# MapReduce: parallel search → iterative summary
mapreduce = MapReduceAgent(
    name="deep_analysis",
    sub_agents=[search_web, search_db, search_docs],
    reduce_agent=summary_loop,  # ProducerReviewerAgent
)
```

### Pattern 8: Consensus with Pipeline Voters

Each voter in a `ConsensusAgent` can itself be a pipeline:

```python
# Each voter is a research → analysis pipeline targeting a different domain
tech_voter = SequentialAgent(
    name="tech_perspective",
    sub_agents=[tech_researcher, tech_analyst],
)

business_voter = SequentialAgent(
    name="business_perspective",
    sub_agents=[business_researcher, business_analyst],
)

# Consensus across pipeline voters
consensus = ConsensusAgent(
    name="multi_perspective",
    sub_agents=[tech_voter, business_voter],
    judge_agent=executive_judge,
)
```

### Composability Rule

> **Every parameter typed as `BaseAgent` accepts any orchestration agent.** This applies to `producer`, `reviewer`, `reduce_agent`, `judge_agent`, and every element in `sub_agents`. The composition depth is unlimited because all orchestration agents are `BaseAgent` subclasses (Liskov substitution principle).

| Parameter | Typed as | Accepts any |
| --- | --- | --- |
| `ProducerReviewerAgent.producer` | `BaseAgent` | `SequentialAgent`, `ParallelAgent`, `MapReduceAgent`, `ConsensusAgent`... |
| `ProducerReviewerAgent.reviewer` | `BaseAgent` | `ConsensusAgent`, `LlmAgent`, `LoopAgent`, `MapReduceAgent`... |
| `MapReduceAgent.reduce_agent` | `BaseAgent` | `ProducerReviewerAgent`, `ConsensusAgent`, `SequentialAgent`... |
| `MapReduceAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
| `ConsensusAgent.judge_agent` | `BaseAgent` | `SequentialAgent`, `ProducerReviewerAgent`, `LlmAgent`... |
| `ConsensusAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
| `SequentialAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
| `ParallelAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
| `LoopAgent.sub_agents[]` | `BaseAgent` | Any agent or orchestration |
```

---

## Decision Guide

Use this flowchart to pick the right strategy:

```
Is the execution order fixed?
├── YES → Do agents depend on each other's output?
│         ├── YES → Is it a produce-then-review cycle?
│         │         ├── YES → ProducerReviewerAgent
│         │         └── NO  → SequentialAgent
│         └── NO  → Do you need to combine results?
│                   ├── YES → MapReduceAgent
│                   └── NO  → Do you need consensus from diverse answers?
│                             ├── YES → ConsensusAgent
│                             └── NO  → ParallelAgent
│
└── NO  → Should routing be explicit and testable?
          ├── YES → RouterAgent
          └── NO  → Is it a multi-turn conversation?
                    ├── YES → transfer_to_agent
                    └── NO  → Do you need iteration?
                              ├── YES → LoopAgent
                              └── NO  → RouterAgent
```

### Quick reference

| You want to... | Use |
| --- | --- |
| Run A then B then C | `SequentialAgent` |
| Run A, B, C at the same time | `ParallelAgent` |
| Repeat until quality is good enough | `LoopAgent` |
| Fan-out to many agents, then combine their results | `MapReduceAgent` |
| Get diverse answers and synthesise a consensus | `ConsensusAgent` |
| Generate content and iterate until reviewer approves | `ProducerReviewerAgent` |
| Stress-test an idea with adversarial arguments | `DebateAgent` |
| Try cheap models first, escalate on failure | `EscalationAgent` |
| Delegate, evaluate, and re-delegate with active monitoring | `SupervisorAgent` |
| Pick the most common answer from N models (no judge) | `VotingAgent` |
| Transfer full control to another agent (peer-to-peer) | `HandoffAgent` |
| Collaborative multi-agent chat with managed turns | `GroupChatAgent` |
| Orchestrate a multi-level hierarchy of departments | `HierarchicalAgent` |
| Validate input/output with automatic retry on rejection | `GuardrailAgent` |
| Run same agent N times and pick best by score | `BestOfNAgent` |
| Process a list of items through one agent in parallel | `BatchAgent` |
| Try cheap→expensive agents, stop when quality is met | `CascadeAgent` |
| Explore multiple reasoning branches and pick the best | `TreeOfThoughtsAgent` |
| LLM decomposes task into dependency-aware steps | `PlannerAgent` |
| Break a complex question into sub-questions for specialists | `SubQuestionAgent` |
| Filter context/history before delegating to agents | `ContextFilterAgent` |
| Self-correct with persistent memory of past failures | `ReflexionAgent` |
| Pause for human approval or feedback | `HumanInputAgent` |
| Let an LLM pick the best agents and execution mode | `RouterAgent` |
| Let an agent delegate during conversation | `transfer_to_agent` |
| LLM-built pipeline (research then write) | `RouterAgent` (sequential mode) |
| LLM-built fan-out (gather in parallel) | `RouterAgent` (parallel mode) |
| LLM-driven iterative refinement | `RouterAgent` (loop mode) |

---

## Comparison with Other Frameworks

| Feature | Nono | LangGraph | CrewAI | AutoGen | Google ADK |
| --- | --- | --- | --- | --- | --- |
| Sequential orchestration | `SequentialAgent` | Graph edges | Process (sequential) | Sequential chat | `SequentialAgent` |
| Parallel orchestration | `ParallelAgent` | Fan-out nodes | Not native | Not native | `ParallelAgent` |
| Loop orchestration | `LoopAgent` | Conditional edges | Not native | Not native | `LoopAgent` |
| Map-Reduce | `MapReduceAgent` | Manual (fan-out + merge) | Not native | Not native | Not native |
| Consensus / voting | `ConsensusAgent` | Not native | Not native | Not native | Not native |
| Producer-Reviewer | `ProducerReviewerAgent` | Manual (cycle edges) | Not native | Not native | Not native |
| Adversarial debate | `DebateAgent` | Not native | Not native | Not native | Not native |
| Tiered escalation | `EscalationAgent` | Not native | Not native | Not native | Not native |
| LLM supervisor | `SupervisorAgent` | Not native | Not native | Not native | Not native |
| Majority voting | `VotingAgent` | Not native | Not native | Not native | Not native |
| Peer-to-peer handoff | `HandoffAgent` | Not native | Not native | Not native | Not native |
| N-agent group chat | `GroupChatAgent` | Not native | Not native | `GroupChat` | Not native |
| Hierarchical orchestration | `HierarchicalAgent` | Not native | **Y** (Process) | Not native | Not native |
| Pre/post validation guardrails | `GuardrailAgent` | Not native | Not native | Not native | Not native |
| Best-of-N sampling | `BestOfNAgent` | Not native | Not native | Not native | Not native |
| Batch item processing | `BatchAgent` | Not native | Not native | Not native | Not native |
| Progressive cascade | `CascadeAgent` | Not native | Not native | Not native | Not native |
| Tree-of-Thoughts reasoning | `TreeOfThoughtsAgent` | Not native | Not native | Not native | Not native |
| Plan-and-execute | `PlannerAgent` | Plan-and-execute | `planning=True` | Not native | Not native |
| Sub-question decomposition | `SubQuestionAgent` | Not native | Not native | Not native | Not native |
| Context/message filtering | `ContextFilterAgent` | Not native | Not native | `MessageFilterAgent` | Not native |
| Reflexion (self-improvement) | `ReflexionAgent` | Not native | Not native | Not native | Not native |
| Human-in-the-Loop | `HumanInputAgent` | `interrupt_before/after` | `human_input=True` | `HumanProxyAgent` | Not native |
| LLM-powered routing | `RouterAgent` (4 modes) | Not native | Manager agent | Speaker selection | Not native |
| Dynamic mode selection | `RouterAgent` | Not native | Not native | Not native | Not native |
| Dynamic delegation (tool) | `transfer_to_agent` | Not native | Not native | Not native | `transfer_to_agent` |
| Composable nesting | All agents are `BaseAgent` | Subgraphs | Limited | Limited | Yes |
| Zero external dependencies | Yes | langchain-core | crewai | pyautogen | google-adk |
| Provider-agnostic | 14 providers, 1-line switch | Via langchain | Config-based | Config-based | Gemini only |

Nono is the only lightweight framework that provides **all twenty-five strategies** (sequential, parallel, loop, map-reduce, consensus, producer-reviewer, debate, escalation, supervisor, voting, handoff, group-chat, hierarchical, guardrail, best-of-n, batch, cascade, tree-of-thoughts, planner, sub-question, context-filter, reflexion, human-in-the-loop, LLM router, dynamic delegation) as composable `BaseAgent` subclasses with zero external agent dependencies.

---

## API Reference

### SequentialAgent

```python
SequentialAgent(
    name: str,
    sub_agents: list[BaseAgent],
    description: str = "",
)
```

Runs sub-agents in declared order. Each agent's `AGENT_MESSAGE` becomes the next agent's `user_message`.

### ParallelAgent

```python
ParallelAgent(
    name: str,
    sub_agents: list[BaseAgent],
    description: str = "",
    max_workers: int | None = None,   # thread pool size (sync); ignored in async
    message_map: dict[str, str] | None = None,  # per-agent custom messages
    result_key: str | None = None,    # auto-collect results into session.state
)
```

Runs all sub-agents concurrently. Sync uses `ThreadPoolExecutor`, async uses `asyncio.gather`. Errors are yielded as `ERROR` events without stopping other agents.

`message_map` maps agent names to custom messages. Agents not in the map receive the caller's `user_message`. The map can be updated between runs.

`result_key`, when set, collects all sub-agent responses into `session.state[result_key]` as `{agent_name: last_agent_message}`. This is the recommended way to feed parallel results into the next agent in a `SequentialAgent` pipeline.

### LoopAgent

```python
LoopAgent(
    name: str,
    sub_agents: list[BaseAgent],
    description: str = "",
    max_iterations: int = 3,
    stop_condition: Callable[[dict], bool] | None = None,
)
```

Repeats sub-agents sequentially each iteration. Stops when `stop_condition(session.state)` returns `True` or `max_iterations` is reached.

### MapReduceAgent

```python
MapReduceAgent(
    name: str,
    sub_agents: list[BaseAgent],       # mapper agents (parallel)
    reduce_agent: BaseAgent,           # agent that combines all mapper outputs
    description: str = "",
    max_workers: int | None = None,
    message_map: dict[str, str] | None = None,
    result_key: str | None = None,
)
```

Runs all `sub_agents` in parallel (map phase), collects their `AGENT_MESSAGE` outputs, then feeds the combined text into `reduce_agent` (reduce phase). `result_key` stores mapper results as `{agent_name: response}` in `session.state`.

### ConsensusAgent

```python
ConsensusAgent(
    name: str,
    sub_agents: list[BaseAgent],       # voter agents (parallel)
    judge_agent: BaseAgent,            # agent that synthesises the consensus
    description: str = "",
    max_workers: int | None = None,
    result_key: str | None = None,
)
```

All `sub_agents` receive the same message and answer independently (vote phase). The `judge_agent` receives all voter answers with the original question and produces a single consensus response. `result_key` stores voter answers in `session.state`.

### ProducerReviewerAgent

```python
ProducerReviewerAgent(
    name: str,
    producer: BaseAgent,               # generates / refines content
    reviewer: BaseAgent,               # evaluates the output
    description: str = "",
    max_iterations: int = 3,
    approval_keyword: str = "APPROVED",
)
```

Iterative loop: `producer` generates content, `reviewer` evaluates it. If the reviewer's response contains `approval_keyword` (case-insensitive), the loop stops. Otherwise, the reviewer's feedback is passed back to the producer for refinement. Stops after `max_iterations` regardless.

### DebateAgent

```python
DebateAgent(
    name: str,
    agent_a: BaseAgent,                # first debater
    agent_b: BaseAgent,                # second debater
    judge: BaseAgent,                  # renders the final verdict
    description: str = "",
    max_rounds: int = 3,
    resolution_keyword: str = "RESOLVED",
)
```

Two agents argue in alternating rounds. After each round (or when `max_rounds` is reached), the `judge` receives the full debate transcript and renders a verdict. If the judge's response contains `resolution_keyword` (case-insensitive), the debate ends early.

### EscalationAgent

```python
EscalationAgent(
    name: str,
    sub_agents: list[BaseAgent],       # agents tried in order (cheap → expensive)
    description: str = "",
    failure_keyword: str = "ESCALATE",
    on_escalation: Callable | None = None,
)
```

Tries `sub_agents` in declared order. If a response contains `failure_keyword` (case-insensitive), the next agent is tried. Stops at the first successful response or after all agents have been exhausted. `on_escalation` is called on each escalation.

### SupervisorAgent

```python
SupervisorAgent(
    name: str,
    sub_agents: list[BaseAgent],       # available worker agents
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    supervisor_instruction: str = "",
    max_iterations: int = 5,
)
```

LLM-powered supervisor that delegates tasks, evaluates results, and can re-delegate. Each iteration the supervisor LLM decides which sub-agent to call and what message to send. After receiving the result, the supervisor evaluates and either produces a final answer or delegates again. Stops after `max_iterations`.

### VotingAgent

```python
VotingAgent(
    name: str,
    sub_agents: list[BaseAgent],       # voter agents (parallel)
    description: str = "",
    max_workers: int | None = None,
    normalize: Callable | None = None,
    result_key: str | None = None,
)
```

All `sub_agents` receive the same message and answer in parallel. Responses are normalised via `normalize` (if provided), then the most frequent answer wins. No LLM judge is needed — purely deterministic counting. `result_key` stores voter answers in `session.state`.

### HandoffAgent

```python
HandoffAgent(
    name: str,
    entry_agent: BaseAgent,            # first agent to receive the message
    description: str = "",
    handoff_rules: dict[str, list[BaseAgent]] | None = None,
    max_handoffs: int = 10,
    handoff_keyword: str = "HANDOFF:",
)
```

Peer-to-peer handoff mesh. The `entry_agent` runs first. If its response contains `HANDOFF: <target_name>`, control transfers to that target (if allowed by `handoff_rules`). The loop continues until an agent completes without handing off, or `max_handoffs` is reached. All agents receive the full conversation history.

### GroupChatAgent

```python
GroupChatAgent(
    name: str,
    sub_agents: list[BaseAgent],       # participant agents
    description: str = "",
    speaker_selection: str | Callable = "round_robin",
    max_rounds: int = 10,
    termination_condition: Callable | None = None,
    termination_keyword: str | None = None,
    model: str | None = None,          # for "llm" speaker selection
    provider: str = "google",
    result_key: str | None = None,
)
```

N-agent group chat with manager-controlled speaker selection. Supports `"round_robin"` (default), `"llm"` (LLM picks next speaker), or a custom callable `(messages, agents) → agent`. The chat terminates on `max_rounds`, `termination_keyword` in the last message, or `termination_condition(messages)` returning `True`. `result_key` stores the full transcript in `session.state`.

### HierarchicalAgent

```python
HierarchicalAgent(
    name: str,
    sub_agents: list[BaseAgent],       # department-head agents
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    api_key: str | None = None,
    manager_instruction: str = "",
    temperature: float = 0.0,
    max_iterations: int = 3,
    synthesis_prompt: str = "",
    result_key: str | None = None,
)
```

Multi-level hierarchical orchestration with LLM-powered manager. Each `sub_agent` may itself be an orchestration agent with its own sub-agents, forming a tree-shaped command structure. The manager sees the full org-chart, delegates across rounds, and synthesises a final answer from collected department outputs.

### GuardrailAgent

```python
GuardrailAgent(
    name: str,
    main_agent: BaseAgent,
    description: str = "",
    pre_validator: BaseAgent | None = None,
    post_validator: BaseAgent | None = None,
    rejection_keyword: str = "REJECTED",
    max_retries: int = 1,
    result_key: str | None = None,
)
```

Wraps a main agent with optional pre-validator (input check/transform) and post-validator (output check). If the post-validator's response contains the `rejection_keyword`, the main agent retries up to `max_retries` times.

### BestOfNAgent

```python
BestOfNAgent(
    name: str,
    agent: BaseAgent,
    description: str = "",
    n: int = 3,
    score_fn: Callable[[str], float] | None = None,
    max_workers: int = 4,
    result_key: str | None = None,
)
```

Runs the same agent N times in parallel and selects the best response using `score_fn`. Default scoring uses response length. Stores `best_index`, `best_score`, and `all_scores` in `result_key`.

### BatchAgent

```python
BatchAgent(
    name: str,
    agent: BaseAgent,
    description: str = "",
    items: list[str] | None = None,
    items_key: str | None = None,
    template: str | None = None,
    max_workers: int = 4,
    result_key: str | None = None,
)
```

Processes a list of items through one agent with concurrency control. Items from `items` (static) or `items_key` (session.state). Template can format each item via `{item}` placeholder. Stores `{index: response}` in `result_key`.

### CascadeAgent

```python
CascadeAgent(
    name: str,
    sub_agents: list[BaseAgent],
    score_fn: Callable[[str], float],
    description: str = "",
    threshold: float = 0.8,
    result_key: str | None = None,
)
```

Tries each stage sequentially, scoring the output with `score_fn`. Stops when a stage meets the `threshold`. If no stage meets it, yields the last attempt with `met_threshold=False` in `result_key`.

### TreeOfThoughtsAgent

```python
TreeOfThoughtsAgent(
    name: str,
    agent: BaseAgent,
    evaluate_fn: Callable[[str], float],
    description: str = "",
    n_branches: int = 3,
    top_k: int = 2,
    max_depth: int = 3,
    threshold: float = 0.9,
    max_workers: int = 4,
    result_key: str | None = None,
)
```

BFS tree exploration: generates `n_branches` per frontier node, evaluates with `evaluate_fn`, prunes to `top_k`, deepens up to `max_depth`. Stops early if `threshold` is met. Stores `depth`, `score`, `met_threshold`, `path` in `result_key`.

### PlannerAgent

```python
PlannerAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    planning_instruction: str = "",
    max_steps: int = 5,
    synthesis_prompt: str = "",
    result_key: str | None = None,
)
```

LLM decomposes task into JSON steps with dependencies. Executes respecting dependencies — parallel where possible. LLM synthesises final answer. Stores `plan`, `step_results`, `synthesis` in `result_key`.

### SubQuestionAgent

```python
SubQuestionAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    decomposition_instruction: str = "",
    max_sub_questions: int = 5,
    max_workers: int = 4,
    synthesis_prompt: str = "",
    result_key: str | None = None,
)
```

LLM decomposes question into sub-questions assigned to agents, dispatches in parallel, LLM synthesises. Stores `sub_questions`, `synthesis` in `result_key`.

### ContextFilterAgent

```python
ContextFilterAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    filter_fn: Callable[[BaseAgent, list[Event]], list[Event]] | None = None,
    max_history: int | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    mode: str = "sequential",
    result_key: str | None = None,
)
```

Applies per-agent event filtering before delegation. Custom `filter_fn` or built-in rules (`max_history`, `include_sources`, `exclude_sources`). Modes: `sequential` or `parallel`. Stores `filter_stats`, `original_events` in `result_key`.

### ReflexionAgent

```python
ReflexionAgent(
    name: str,
    agent: BaseAgent,
    evaluator: BaseAgent,
    score_fn: Callable[[str], float],
    description: str = "",
    threshold: float = 0.8,
    max_attempts: int = 3,
    memory_key: str = "reflexion_memory",
    result_key: str | None = None,
)
```

Iterative self-improvement: generate → evaluate → score → if pass: return, else: store lesson → retry with enriched prompt. Persists memories in `session.state[memory_key]`. Stores `attempts`, `accepted_attempt`, `score` in `result_key`.

### SpeculativeAgent

```python
SpeculativeAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    evaluator_fn: Callable[[str], float] | None = None,
    min_confidence: float = 0.8,
    result_key: str | None = None,
)
```

Races all sub-agents in parallel. As soon as one agent's output passes `evaluator_fn` with score ≥ `min_confidence`, the winner is accepted and slower agents are abandoned. Unlike `BestOfNAgent` (waits for all), this optimises for **latency**. Stores `winner`, `score`, `early_stop` in `result_key`.

### CircuitBreakerAgent

```python
CircuitBreakerAgent(
    name: str,
    agent: BaseAgent,
    description: str = "",
    fallback_agent: BaseAgent | None = None,
    failure_detector: Callable[[str], bool] | None = None,
    failure_threshold: int = 3,
    window_size: int = 10,
    recovery_timeout: int = 60,
    result_key: str | None = None,
)
```

Monitors failures via `failure_detector` (default: empty responses). After `failure_threshold` failures within `window_size`, the circuit **opens** and routes to `fallback_agent`. After `recovery_timeout` seconds, one test request is sent to the primary (half-open). Stores `outcome`, `circuit_state` in `result_key`.

### TournamentAgent

```python
TournamentAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    judge_agent: BaseAgent | None = None,
    description: str = "",
    result_key: str | None = None,
)
```

Single-elimination bracket: agents compete in pairs, `judge_agent` picks the winner of each match. Handles bye for odd-numbered brackets. Continues until one agent remains. Stores `winner`, `rounds`, `bracket` in `result_key`.

### ShadowAgent

```python
ShadowAgent(
    name: str,
    stable_agent: BaseAgent,
    shadow_agent: BaseAgent,
    description: str = "",
    diff_logger: Callable[[str, str], None] | None = None,
    result_key: str | None = None,
)
```

Runs `stable_agent` and `shadow_agent` in parallel. Only the stable output is yielded to the caller. The `diff_logger` receives both outputs for comparison. Stores `stable_output`, `shadow_output`, `match` in `result_key`.

### CompilerAgent

```python
CompilerAgent(
    name: str,
    target_agent: BaseAgent,
    description: str = "",
    examples: list[dict] | None = None,
    metric_fn: Callable[[str, str], float] | None = None,
    max_iterations: int = 3,
    model: str | None = None,
    provider: str = "google",
    result_key: str | None = None,
)
```

DSPy-style prompt compilation. Runs `target_agent` on `examples`, scores with `metric_fn(output, expected)`, then uses an LLM to improve the target's `instruction`. Iterates up to `max_iterations` or until a perfect score. Stores `best_score`, `best_instruction`, `iterations` in `result_key`.

### CheckpointableAgent

```python
CheckpointableAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    checkpoint_key: str = "checkpoint",
    result_key: str | None = None,
)
```

Runs sub-agents sequentially, saving progress in `session.state[checkpoint_key]` after each step. On resume, skips already-completed steps. Stores `completed`, `total_steps`, `outputs` in `result_key`.

### DynamicFanOutAgent

```python
DynamicFanOutAgent(
    name: str,
    worker_agent: BaseAgent,
    reducer_agent: BaseAgent | None = None,
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    decomposition_prompt: str = "",
    max_items: int = 10,
    max_workers: int = 4,
    result_key: str | None = None,
)
```

Uses an LLM to decompose the user message into a JSON array of work items. Each item is dispatched to `worker_agent` in parallel. Results are optionally reduced by `reducer_agent`. Stores `count`, `items`, `reduced` in `result_key`.

### SwarmAgent

```python
SwarmAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    initial_agent: str = "",
    context_variables: dict | None = None,
    max_handoffs: int = 10,
    result_key: str | None = None,
)
```

OpenAI Swarm-style handoff. Starts with `initial_agent`. Each agent can set `session.state["__next_agent__"]` to hand off, or `session.state["__done__"] = True` to stop. `context_variables` are injected into messages. Stores `chain`, `handoffs` in `result_key`.

### MemoryConsolidationAgent

```python
MemoryConsolidationAgent(
    name: str,
    main_agent: BaseAgent,
    summarizer_agent: BaseAgent,
    description: str = "",
    event_threshold: int = 50,
    keep_recent: int = 10,
    memory_key: str = "memory_summary",
    result_key: str | None = None,
)
```

Before delegating to `main_agent`, checks if session events exceed `event_threshold`. If so, older events are summarised by `summarizer_agent` and stored in `session.state[memory_key]`. Keeps the `keep_recent` most recent events intact. Stores `consolidated`, `event_count` in `result_key`.

### PriorityQueueAgent

```python
PriorityQueueAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    priority_map: dict[str, int] | None = None,
    stop_condition: Callable[[dict], bool] | None = None,
    result_key: str | None = None,
)
```

Groups agents by priority level (lower number = higher priority). Executes groups sequentially; agents within a group run in parallel. `stop_condition(session.state)` can halt processing before lower priorities. Stores `execution_order`, `total` in `result_key`.

### MonteCarloAgent

```python
MonteCarloAgent(
    name: str,
    agent: BaseAgent | None = None,
    description: str = "",
    evaluate_fn: Callable[[str], float] | None = None,
    n_simulations: int = 20,
    max_depth: int = 3,
    exploration_weight: float | None = None,  # default √2
    result_key: str | None = None,
)
```

Monte Carlo Tree Search with UCT (Upper Confidence bound for Trees). Runs `n_simulations` rollouts: select a leaf via UCT, expand by generating a thought with `agent`, score via `evaluate_fn`, and backpropagate. Returns the best path from root to leaf. Stores `best_response`, `best_score`, `best_path` in `result_key`.

### GraphOfThoughtsAgent

```python
GraphOfThoughtsAgent(
    name: str,
    agent: BaseAgent | None = None,
    aggregate_agent: BaseAgent | None = None,
    description: str = "",
    score_fn: Callable[[str], float] | None = None,
    n_branches: int = 3,
    n_rounds: int = 2,
    result_key: str | None = None,
)
```

DAG-based thought orchestration with generate, aggregate, and score operations. Unlike `TreeOfThoughtsAgent` (tree-shaped BFS), GoT allows thoughts to merge and refine. Each round generates `n_branches` thoughts, optionally aggregates them via `aggregate_agent`, and scores all candidates. Stores `best_thought`, `best_score`, `total_thoughts` in `result_key`.

### BlackboardAgent

```python
BlackboardAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    controller_fn: Callable[[dict, list[BaseAgent]], BaseAgent] | None = None,
    termination_fn: Callable[[dict], bool] | None = None,
    max_iterations: int = 10,
    board_key: str = "blackboard",
    result_key: str | None = None,
)
```

Shared blackboard architecture. `controller_fn` selects the most relevant expert each iteration. Experts read the current board state and contribute outputs. Converges when `termination_fn(board)` returns `True` or `max_iterations` is reached. Default controller: round-robin.

### MixtureOfExpertsAgent

```python
MixtureOfExpertsAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    gating_fn: Callable[[str, list[BaseAgent]], dict[str, float]] | None = None,
    top_k: int = 2,
    combine_fn: Callable[[list[tuple[str, str, float]]], str] | None = None,
    result_key: str | None = None,
)
```

`gating_fn` assigns weights to each expert. The top-k experts are executed and their outputs combined via `combine_fn`. Default gating: uniform weights. Default combine: weighted concatenation. Stores `weights`, `selected`, `top_k` in `result_key`.

### CoVeAgent

```python
CoVeAgent(
    name: str,
    drafter: BaseAgent | None = None,
    planner: BaseAgent | None = None,
    verifier: BaseAgent | None = None,
    reviser: BaseAgent | None = None,
    description: str = "",
    max_questions: int = 5,
    result_key: str | None = None,
)
```

Chain-of-Verification: 4-phase anti-hallucination pipeline. (1) `drafter` generates an initial response, (2) `planner` produces verification questions, (3) `verifier` answers each question independently, (4) `reviser` produces a final verified response. Async path parallelises verification questions.

### SagaAgent

```python
SagaAgent(
    name: str,
    steps: list[dict[str, Any]] | None = None,
    description: str = "",
    failure_detector: Callable[[str], bool] | None = None,
    result_key: str | None = None,
)
```

Distributed transactions with compensating rollback. Each step has an `action` agent and an optional `compensate` agent. If step N fails (detected by `failure_detector`), compensators for steps N-1→0 are executed in reverse order. Stores `completed`, `failed_step`, `compensated`, `success` in `result_key`.

### LoadBalancerAgent

```python
LoadBalancerAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    strategy: str | Callable = "round_robin",  # "round_robin" | "random" | "least_used"
    result_key: str | None = None,
)
```

Distributes requests across equivalent agents. Picks **one** agent per request (unlike `ParallelAgent` which runs all). Maintains internal counters for `least_used` strategy. Stores `selected`, `usage` in `result_key`.

### EnsembleAgent

```python
EnsembleAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    aggregate_fn: str | Callable = "concat",  # "concat" | "weighted" | callable
    weights: dict[str, float] | None = None,
    max_workers: int | None = None,
    result_key: str | None = None,
)
```

Runs all sub-agents and aggregates their outputs. Unlike `VotingAgent` (majority) or `ConsensusAgent` (judge), EnsembleAgent combines all outputs via a configurable strategy. Sync uses `ThreadPoolExecutor`, async uses `asyncio.gather`. Stores `outputs`, `count` in `result_key`.

### TimeoutAgent

```python
TimeoutAgent(
    name: str,
    agent: BaseAgent | None = None,
    description: str = "",
    timeout_seconds: float = 30.0,
    fallback_message: str = "Operation timed out.",
    result_key: str | None = None,
)
```

Wraps any sub-agent with a deadline. If the agent exceeds `timeout_seconds`, the run is abandoned and `fallback_message` is returned. Sync uses `ThreadPoolExecutor` with timeout, async uses `asyncio.wait_for`. Stores `timed_out`, `elapsed_seconds`, `output` in `result_key`.

### AdaptivePlannerAgent

```python
AdaptivePlannerAgent(
    name: str,
    sub_agents: list[BaseAgent] | None = None,
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    api_key: str | None = None,
    planning_instruction: str = "",
    max_steps: int = 10,
    result_key: str | None = None,
)
```

Like `PlannerAgent` but re-plans after every step. The LLM evaluates intermediate results and can modify or extend the remaining plan. Handles emergent tasks the original plan didn't anticipate. Stops when the LLM returns an empty plan or `max_steps` is reached. Stores `steps`, `total_steps`, `completed` in `result_key`.

### RouterAgent

```python
RouterAgent(
    name: str,
    sub_agents: list[BaseAgent],
    description: str = "",
    model: str | None = None,
    provider: str = "google",
    api_key: str | None = None,
    routing_instruction: str = "",
    temperature: float = 0.0,
    max_iterations: int = 3,
)
```

Makes a lightweight LLM call to select both the agents and the execution mode. The LLM returns:

```json
{"mode": "single|sequential|parallel|loop", "agents": [...], "message": "...", "max_iterations": 3}
```

Internally builds an ephemeral `SequentialAgent`, `ParallelAgent`, or `LoopAgent` as needed. Falls back to the first sub-agent on errors. Backward-compatible with the legacy `{"agent_name": "..."}` format.

### HumanInputAgent

```python
from nono.agent import HumanInputAgent
from nono.hitl import HumanInputResponse

def handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    answer = input(f"[{step_name}] {prompt}: ")
    return HumanInputResponse(approved=answer.lower() != "no", message=answer)

HumanInputAgent(
    name: str,
    handler: HumanInputHandler | None = None,
    async_handler: AsyncHumanInputHandler | None = None,
    prompt: str = "Awaiting human input...",
    on_reject: str = "error",     # "error" raises HumanRejectError, "continue" proceeds
    before_human: BeforeHumanCallback | None = None,
    after_human: AfterHumanCallback | None = None,
    state_key: str = "human_input",
)
```

Pauses execution until a human responds. Emits `HUMAN_INPUT_REQUEST` and `HUMAN_INPUT_RESPONSE` events. Use as a sub-agent in `SequentialAgent`, `LoopAgent`, etc.

**Callbacks:**
- `before_human(agent, ctx) -> Optional[str]`: Return a string to skip the human interaction.
- `after_human(agent, ctx, response) -> Optional[str]`: Return a string to override the agent's message.

**Session state:** The response is stored in `session.state[state_key]` with keys `approved`, `message`, plus any extra `data`.

### transfer_to_agent (auto-tool)

Auto-registered on any `LlmAgent` that has `sub_agents`. Schema: `{agent_name: str, message: str}`. The LLM calls it as a regular tool during conversation.

### Common to all agents

| Method | Returns | Description |
| --- | --- | --- |
| `run(ctx)` | `str` | Synchronous execution |
| `run_async(ctx)` | `str` | Asynchronous execution |
| `find_sub_agent(name)` | `BaseAgent \| None` | Recursive sub-agent lookup |

All orchestration agents support lifecycle callbacks (`before_agent_callback`, `after_agent_callback`).

---

## Sync and Async Orchestration

All orchestration agents implement **both** `run()` (sync) and `run_async()` (async) as independent code paths — there is no automatic conversion between them. The mode you choose on the orchestrator propagates to every sub-agent.

### How it works

| Orchestrator call | Sub-agent call | Concurrency mechanism |
| --- | --- | --- |
| `SequentialAgent.run()` | `sub_agent._run_impl()` | None — serial |
| `SequentialAgent.run_async()` | `sub_agent._run_async_impl()` | `async for` — serial |
| `ParallelAgent.run()` | `sub_agent._run_impl()` | `ThreadPoolExecutor` |
| `ParallelAgent.run_async()` | `sub_agent._run_async_impl()` | `asyncio.gather` |
| `LoopAgent.run()` | `sub_agent._run_impl()` | None — serial per iteration |
| `LoopAgent.run_async()` | `sub_agent._run_async_impl()` | `async for` — serial per iteration |
| `MapReduceAgent.run()` | Mappers: `ThreadPoolExecutor`; Reducer: serial | `ThreadPoolExecutor` + serial |
| `MapReduceAgent.run_async()` | Mappers: `asyncio.gather`; Reducer: `async for` | `asyncio.gather` + serial |
| `ConsensusAgent.run()` | Voters: `ThreadPoolExecutor`; Judge: serial | `ThreadPoolExecutor` + serial |
| `ConsensusAgent.run_async()` | Voters: `asyncio.gather`; Judge: `async for` | `asyncio.gather` + serial |
| `ProducerReviewerAgent.run()` | Producer then reviewer (serial loop) | None — serial per iteration |
| `ProducerReviewerAgent.run_async()` | Producer then reviewer (`async for` loop) | `async for` — serial per iteration |
| `DebateAgent.run()` | Debaters alternate (serial rounds); Judge: serial | None — serial per round |
| `DebateAgent.run_async()` | Debaters alternate (`async for` rounds); Judge: `async for` | `async for` — serial per round |
| `EscalationAgent.run()` | `sub_agent._run_impl()` in order | None — serial (stops on success) |
| `EscalationAgent.run_async()` | `sub_agent._run_async_impl()` in order | `async for` — serial (stops on success) |
| `SupervisorAgent.run()` | Supervisor LLM + delegated `sub_agent._run_impl()` | None — serial per iteration |
| `SupervisorAgent.run_async()` | Supervisor LLM + delegated `sub_agent._run_async_impl()` | `async for` — serial per iteration |
| `VotingAgent.run()` | Voters: `ThreadPoolExecutor` | `ThreadPoolExecutor` |
| `VotingAgent.run_async()` | Voters: `asyncio.gather` | `asyncio.gather` |
| `HandoffAgent.run()` | Active agent: `_run_impl()` in chain | None — serial per handoff |
| `HandoffAgent.run_async()` | Active agent: `_run_async_impl()` in chain | `async for` — serial per handoff |
| `GroupChatAgent.run()` | Speaker: `_run_impl()` per round | None — serial per round |
| `GroupChatAgent.run_async()` | Speaker: `_run_async_impl()` per round | `async for` — serial per round |
| `HierarchicalAgent.run()` | Manager LLM + delegated `dept._run_impl()` + synthesis LLM | None — serial per round |
| `HierarchicalAgent.run_async()` | Manager LLM + delegated `dept._run_async_impl()` + synthesis LLM | `async for` — serial per round |
| `GuardrailAgent.run()` | Pre-validator + main agent + post-validator (with retry) | None — serial pipeline |
| `GuardrailAgent.run_async()` | Pre-validator + main agent + post-validator (with retry) | `async for` — serial pipeline |
| `BestOfNAgent.run()` | N × `agent._run_impl()` via `ThreadPoolExecutor` | `ThreadPoolExecutor(max_workers)` |
| `BestOfNAgent.run_async()` | N × `agent._run_async_impl()` via `asyncio.gather` | `asyncio.gather` — all parallel |
| `BatchAgent.run()` | Per-item `agent._run_impl()` via `ThreadPoolExecutor` | `ThreadPoolExecutor(max_workers)` |
| `BatchAgent.run_async()` | Per-item `agent._run_async_impl()` via `asyncio.Semaphore` | `asyncio.Semaphore(max_workers)` |
| `CascadeAgent.run()` | Sequential `agent._run_impl()` per stage, early-stop | None — serial per stage |
| `CascadeAgent.run_async()` | Sequential `agent._run_async_impl()` per stage, early-stop | `async for` — serial per stage |
| `TreeOfThoughtsAgent.run()` | Per-depth: N branches via `ThreadPoolExecutor` | `ThreadPoolExecutor(max_workers)` per depth |
| `TreeOfThoughtsAgent.run_async()` | Per-depth: N branches via `asyncio.gather` | `asyncio.gather` — all parallel per depth |
| `PlannerAgent.run()` | LLM plan + parallel-where-possible via `ThreadPoolExecutor` | `ThreadPoolExecutor` + serial for deps |
| `PlannerAgent.run_async()` | LLM plan + parallel-where-possible via `asyncio.gather` | `asyncio.gather` + serial for deps |
| `SubQuestionAgent.run()` | Sub-questions via `ThreadPoolExecutor` | `ThreadPoolExecutor(max_workers)` |
| `SubQuestionAgent.run_async()` | Sub-questions via `asyncio.gather` | `asyncio.gather` — all parallel |
| `ContextFilterAgent.run()` | Sequential or parallel (`ThreadPoolExecutor`) | Depends on `mode` |
| `ContextFilterAgent.run_async()` | Sequential or parallel (`asyncio.gather`) | Depends on `mode` |
| `ReflexionAgent.run()` | Generate + evaluate per attempt (serial loop) | None — serial per attempt |
| `ReflexionAgent.run_async()` | Generate + evaluate per attempt (`async for` loop) | `async for` — serial per attempt |
| `SpeculativeAgent.run()` | All sub-agents via `ThreadPoolExecutor`, early cancel | `ThreadPoolExecutor` — first to pass wins |
| `SpeculativeAgent.run_async()` | All sub-agents via `asyncio` tasks, early cancel | `asyncio` tasks — first to pass wins |
| `CircuitBreakerAgent.run()` | Primary `_run_impl()` or fallback `_run_impl()` | None — serial |
| `CircuitBreakerAgent.run_async()` | Primary `_run_async_impl()` or fallback `_run_async_impl()` | `async for` — serial |
| `TournamentAgent.run()` | Pairs via serial `_run_impl()` per round | None — serial per match |
| `TournamentAgent.run_async()` | Pairs via `_run_async_impl()` per round | `async for` — serial per match |
| `ShadowAgent.run()` | Stable + shadow via `ThreadPoolExecutor` | `ThreadPoolExecutor(2)` |
| `ShadowAgent.run_async()` | Stable + shadow via `asyncio.gather` | `asyncio.gather` — both parallel |
| `CompilerAgent.run()` | Target `_run_impl()` per example, LLM optimise per iteration | None — serial per iteration |
| `CompilerAgent.run_async()` | Target `_run_async_impl()` per example, LLM optimise per iteration | `async for` — serial per iteration |
| `CheckpointableAgent.run()` | Sub-agents `_run_impl()` in order, checkpoint after each | None — serial |
| `CheckpointableAgent.run_async()` | Sub-agents `_run_async_impl()` in order, checkpoint after each | `async for` — serial |
| `DynamicFanOutAgent.run()` | LLM decompose + workers via `ThreadPoolExecutor` + reducer serial | `ThreadPoolExecutor(max_workers)` |
| `DynamicFanOutAgent.run_async()` | LLM decompose + workers via `asyncio.gather` + reducer `async for` | `asyncio.gather` — all parallel |
| `SwarmAgent.run()` | Active agent `_run_impl()` in chain | None — serial per handoff |
| `SwarmAgent.run_async()` | Active agent `_run_async_impl()` in chain | `async for` — serial per handoff |
| `MemoryConsolidationAgent.run()` | Optional summariser `_run_impl()` + main `_run_impl()` | None — serial |
| `MemoryConsolidationAgent.run_async()` | Optional summariser `_run_async_impl()` + main `_run_async_impl()` | `async for` — serial |
| `PriorityQueueAgent.run()` | Groups serial; within group: `ThreadPoolExecutor` if >1 | `ThreadPoolExecutor` per group |
| `PriorityQueueAgent.run_async()` | Groups serial; within group: `asyncio.gather` if >1 | `asyncio.gather` per group |
| `MonteCarloAgent.run()` | Sequential simulations: select → expand → backprop | None — serial per simulation |
| `MonteCarloAgent.run_async()` | Sequential simulations via `_run_async_impl` | `async for` — serial per simulation |
| `GraphOfThoughtsAgent.run()` | Per-round: N branches serial + optional aggregate | None — serial per branch |
| `GraphOfThoughtsAgent.run_async()` | Per-round: N branches via `asyncio.gather` + aggregate | `asyncio.gather` — parallel per round |
| `BlackboardAgent.run()` | Iterative expert activation (serial loop) | None — serial per iteration |
| `BlackboardAgent.run_async()` | Iterative expert activation (`async for` loop) | `async for` — serial per iteration |
| `MixtureOfExpertsAgent.run()` | Top-k experts serial `_run_impl()` | None — serial per expert |
| `MixtureOfExpertsAgent.run_async()` | Top-k experts via `asyncio.gather` | `asyncio.gather` — all parallel |
| `CoVeAgent.run()` | Draft → plan → verify (serial) → revise | None — serial pipeline |
| `CoVeAgent.run_async()` | Draft → plan → verify (`asyncio.gather`) → revise | `asyncio.gather` for verifications |
| `SagaAgent.run()` | Steps serial; compensators serial (reverse) | None — serial |
| `SagaAgent.run_async()` | Steps serial; compensators serial (reverse) | `async for` — serial |
| `LoadBalancerAgent.run()` | Single agent `_run_impl()` | None — single agent |
| `LoadBalancerAgent.run_async()` | Single agent `_run_async_impl()` | `async for` — single agent |
| `EnsembleAgent.run()` | All agents via `ThreadPoolExecutor` | `ThreadPoolExecutor(max_workers)` |
| `EnsembleAgent.run_async()` | All agents via `asyncio.gather` | `asyncio.gather` — all parallel |
| `TimeoutAgent.run()` | Agent via `ThreadPoolExecutor` with timeout | `ThreadPoolExecutor(1)` + timeout |
| `TimeoutAgent.run_async()` | Agent via `asyncio.wait_for` | `asyncio.wait_for` + timeout |
| `AdaptivePlannerAgent.run()` | LLM plan + agent `_run_impl()` per step | None — serial per step |
| `AdaptivePlannerAgent.run_async()` | LLM plan via `to_thread` + agent `_run_async_impl()` per step | `async for` — serial per step |
| `HumanInputAgent.run()` | Calls handler (blocks) | None — blocks on handler |
| `HumanInputAgent.run_async()` | Calls async_handler or `to_thread(handler)` | `asyncio.to_thread` |
| `RouterAgent.run()` | Delegates to the composed orchestrator's `_run_impl()` | Depends on selected mode |
| `RouterAgent.run_async()` | Delegates to the composed orchestrator's `_run_async_impl()` | Depends on selected mode |

### Key rules

1. **No fallback**: `BaseAgent` is an ABC that requires every agent to implement both `_run_impl` (sync) and `_run_async_impl` (async). There is no shim that wraps one in the other.
2. **Mode propagation**: If you call `runner.run()` (sync), the entire tree runs sync. If you call `runner.run_async()`, the entire tree runs async. You don't mix modes within a single execution.
3. **LLM calls adapt automatically**: `LlmAgent._run_impl()` calls `connector.generate_completion()` (blocking). `LlmAgent._run_async_impl()` offloads the same call via `asyncio.to_thread`, so it doesn't block the event loop.

### Sync example

```python
from nono.agent import Agent, Runner, SequentialAgent, ParallelAgent

researcher = Agent(name="researcher", instruction="Research the topic.", provider="google")
writer = Agent(name="writer", instruction="Write an article.", provider="google")

pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer],
)

runner = Runner(agent=pipeline)
response = runner.run("AI trends 2026")
# Every sub-agent runs via _run_impl → blocking calls, one after another.
```

### Async example

```python
import asyncio
from nono.agent import Agent, Runner, ParallelAgent

web = Agent(name="web", instruction="Search the web.", provider="perplexity")
db = Agent(name="db", instruction="Query the database.", provider="openai")

gather = ParallelAgent(
    name="gather",
    sub_agents=[web, db],
)

runner = Runner(agent=gather)
response = asyncio.run(runner.run_async("Latest sales data"))
# Both sub-agents run via _run_async_impl → asyncio.gather → true concurrency.
```

### When to use each mode

| Scenario | Recommended mode | Why |
| --- | --- | --- |
| Script / CLI / notebook | `run()` (sync) | Simpler code, no event loop needed |
| Web server (FastAPI, etc.) | `run_async()` (async) | Non-blocking; `ParallelAgent` uses `asyncio.gather` for true concurrency |
| Many parallel sub-agents with LLM calls | `run_async()` (async) | Avoids spawning many threads; `asyncio.gather` is more efficient |
| Single agent, no parallelism | Either | Both work equally well |

### ParallelAgent: threads vs. coroutines

`ParallelAgent` is the orchestrator where the mode choice matters most:

- **Sync** (`run()`): launches a `ThreadPoolExecutor` with `max_workers` threads (defaults to `len(sub_agents)`). Each thread blocks on its sub-agent's LLM call. Good for small fan-outs, but threads are heavier than coroutines.
- **Async** (`run_async()`): uses `asyncio.gather`. Each sub-agent runs as a coroutine. LLM calls inside `LlmAgent` are offloaded via `asyncio.to_thread`, so the event loop stays free. Preferred for large fan-outs or server environments.

```python
# Sync: 3 threads
parallel = ParallelAgent(name="p", sub_agents=[a, b, c], max_workers=3)
Runner(agent=parallel).run("query")  # ThreadPoolExecutor(3)

# Async: 3 coroutines
await Runner(agent=parallel).run_async("query")  # asyncio.gather
```

---

## Workflow vs Agent Orchestration

Nono provides two complementary orchestration paradigms: the **Workflow** module (graph-based pipelines) and the **Agent** module (event-driven agents). They solve different problems and combine naturally.

### Comparison

| Aspect | Workflow (`nono/workflows/`) | Agent orchestration (`nono/agent/`) |
| --- | --- | --- |
| **Control** | Developer-defined (code) | Agent-defined (LLM at runtime) or developer-defined |
| **Topology** | Explicit DAG: `step()` → `connect()` → `branch()` | Implicit via agent nesting (Sequential, Parallel, Loop) |
| **Routing** | Deterministic: `branch()` / `branch_if()` evaluate state | RouterAgent: LLM picks mode + agents; or fixed nesting |
| **State model** | Shared `dict` — each step returns a dict that is merged | `Session` — events list + `state` dict + `shared_content` |
| **Unit of work** | Any `Callable[[dict], dict]` (plain function, coroutine, tasker_node, agent_node) | `BaseAgent` subclass (LlmAgent, SequentialAgent, etc.) |
| **Parallel execution** | Not native (single step at a time) | `ParallelAgent` with threads (sync) or `asyncio.gather` (async) |
| **Loop / iteration** | Re-entrant via `branch()` back to earlier step | `LoopAgent` with `max_iterations` + `stop_condition` |
| **Dynamic manipulation** | `insert_before`, `insert_after`, `remove_step`, `replace_step`, `swap_steps` | Not supported — topology is fixed at construction |
| **Streaming** | `stream()` yields `(step_name, state_snapshot)` | `_run_impl` yields `Event` objects |
| **LLM routing** | Not native (use `agent_node(RouterAgent)` to bridge) | `RouterAgent` with 4 modes: single, sequential, parallel, loop |

### How they connect

The two systems integrate through factory functions:

#### Agent inside a Workflow — `agent_node()`

Wrap any agent (including `RouterAgent`) as a Workflow step:

```python
from nono.workflows import Workflow, agent_node
from nono.agent import Agent, RouterAgent

researcher = Agent(name="researcher", instruction="...", provider="perplexity")
writer = Agent(name="writer", instruction="...", provider="openai")

# RouterAgent decides dynamically which sub-agents to use
router = RouterAgent(
    name="router",
    provider="google",
    sub_agents=[researcher, writer],
)

flow = Workflow("hybrid_pipeline")
flow.step("preprocess", lambda s: {"clean_text": s["raw_text"].strip()})
flow.step("ai_route", agent_node(router, input_key="clean_text", output_key="ai_result"))
flow.step("postprocess", lambda s: {"final": s["ai_result"].upper()})
flow.connect("preprocess", "ai_route", "postprocess")

result = flow.run(raw_text="  Analyse market trends  ")
# preprocess (deterministic) → router (LLM decides) → postprocess (deterministic)
```

#### Workflow step inside an Agent — via tools

An agent can invoke a Workflow inside a `FunctionTool`:

```python
from nono.agent import Agent, tool, ToolContext
from nono.workflows import Workflow, tasker_node

etl_flow = Workflow("etl")
etl_flow.step("extract", lambda s: {"data": fetch_db(s["query"])})
etl_flow.step("transform", tasker_node(
    system_prompt="Clean and normalise the data.",
    input_key="data", output_key="clean_data"))
etl_flow.connect("extract", "transform")

@tool(description="Run the ETL pipeline for a query.")
def run_etl(query: str, tool_context: ToolContext) -> str:
    result = etl_flow.run(query=query)
    return result["clean_data"]

analyst = Agent(
    name="analyst",
    instruction="Use the ETL tool to gather data, then answer the question.",
    tools=[run_etl],
    provider="google",
)
```

### When to use each

| Scenario | Use |
| --- | --- |
| Fixed pipeline with known steps (ETL, preprocess → classify → output) | **Workflow** |
| Dynamic decision about which agents to invoke based on input | **RouterAgent** |
| Fixed pipeline with one intelligent decision node | **Workflow** + `agent_node(RouterAgent)` |
| Agents that chain, parallelise, or iterate | **SequentialAgent / ParallelAgent / LoopAgent** |
| Conditional branching based on data (`score > 0.8 → publish`) | **Workflow** with `branch_if()` |
| Conditional branching based on semantics ("does this need research?") | **RouterAgent** |
| Pipeline that needs runtime manipulation (insert/remove/swap steps) | **Workflow** |
| Multi-turn conversation with agent hand-off | **transfer_to_agent** |

### Design principle

**Workflow** is *code-first*: the developer defines the entire topology before execution. State flows as a `dict` that each step transforms. Predictable and testable.

**Agent orchestration** is *agent-first*: the topology is either fixed by nesting (`SequentialAgent`) or decided at runtime by an LLM (`RouterAgent`). State flows as an `Event` stream through a `Session`. Flexible and autonomous.

They are complementary — Workflow provides structure and predictability, agents provide intelligence and autonomy. Combined via `agent_node()` and `FunctionTool`, they enable hybrid architectures where fixed business logic wraps dynamic AI decisions.

> See also: [Workflow README](../workflows/README_workflow.md#integration-with-nono-tasker-and-agent) for `tasker_node()` and `agent_node()` reference.

---

## FAQ / Troubleshooting

**Q: Can I mix sync and async sub-agents in the same orchestration?**
A: No. The mode is determined by the top-level call (`run()` or `run_async()`). All sub-agents in the tree use the same mode. Every agent must implement both `_run_impl` and `_run_async_impl` (enforced by `BaseAgent`), so both paths are always available. See [Sync and Async Orchestration](#sync-and-async-orchestration).

**Q: RouterAgent always picks the same agent.**
A: Ensure each sub-agent has a distinct `description`. The routing LLM reads descriptions to decide — vague or missing descriptions lead to poor routing.

**Q: LoopAgent runs max_iterations even though quality is high.**
A: The `stop_condition` reads from `session.state`. Make sure your agent actually writes to the state dict (e.g., via `ToolContext.state["quality"] = 0.95`).

**Q: ParallelAgent is slower than SequentialAgent.**
A: In sync mode, `max_workers` defaults to the number of sub-agents — check if your system can handle that many concurrent LLM calls. Also check rate limits.

**Q: How should I write to `session.state` from parallel agents?**
A: Use the thread-safe helpers `session.state_set(key, value)` and `session.state_get(key)` instead of direct dict access. These acquire an internal lock. See [Session — Thread-Safe State Helpers](README_agent.md#thread-safe-state-helpers).

**Q: How do I isolate a sub-agent's state from the parent?**
A: Three options: (1) `local_content` for private content, (2) `ContextFilterAgent` for filtered events, (3) create a new `Session` for full isolation. See [State Isolation Patterns](README_agent.md#state-isolation-patterns).

**Q: transfer_to_agent loops infinitely between agents.**
A: The tool-calling loop has a built-in limit of 10 iterations. If the coordinator keeps delegating, the loop stops and asks for a final answer.

**Q: Can I mix providers in the same orchestration?**
A: Yes. Each `Agent` has its own `provider` parameter:

```python
researcher = Agent(name="r", provider="perplexity", ...)
writer = Agent(name="w", provider="openai", ...)
router = RouterAgent(name="router", provider="google", sub_agents=[researcher, writer])
# Router uses Google for routing, Perplexity for research, OpenAI for writing
```

**Q: How do I see which agent was selected by RouterAgent?**
A: Check the `AGENT_TRANSFER` event in the session:

```python
runner = Runner(agent=router)
for event in runner.stream("My question"):
    if event.event_type.value == "agent_transfer":
        print(f"Routed to: {event.data['target_agent']}")
```

---

## Custom Orchestration Patterns

Nono ships with 15 built-in patterns, but the system is **fully extensible** via `OrchestrationRegistry`. You can register your own patterns so that:

- The `AgentFactory` can automatically select and instantiate them.
- The `OrchestrationSelector` LLM and keyword heuristics recognize them.
- They appear in `OrchestrationRegistry.catalog()` alongside built-in patterns.

### Quick Start — Register a Custom Pattern

```python
from nono.agent import BaseAgent, register_pattern
from nono.agent.agent_factory import BlueprintValidationError


class MyCustomAgent(BaseAgent):
    """A domain-specific orchestration pattern."""

    def __init__(self, *, sub_agents, name="custom", description="", **kwargs):
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self.extra_config = kwargs

    def _run_impl(self, ctx):
        # Your orchestration logic here
        results = []
        for agent in self.sub_agents:
            result = agent.run(ctx)
            results.append(result)
        return "\n".join(results)

    async def _run_async_impl(self, ctx):
        results = []
        for agent in self.sub_agents:
            result = await agent.run_async(ctx)
            results.append(result)
        return "\n".join(results)


# Factory function — receives standard keyword arguments
def my_custom_factory(
    *, name, description, sub_agents, provider, model, pattern_kwargs,
):
    if len(sub_agents) < 2:
        raise BlueprintValidationError(
            "MyCustomAgent requires at least 2 sub-agents."
        )
    return MyCustomAgent(
        sub_agents=sub_agents,
        name=name,
        description=description,
        threshold=float(pattern_kwargs.get("threshold", 0.8)),
    )


# Register it — now available everywhere
register_pattern(
    key="my_custom",
    class_name="MyCustomAgent",
    description="Domain-specific multi-agent orchestration",
    keyword_hints=["custom", "domain specific", "my workflow"],
    factory=my_custom_factory,
    min_sub_agents=2,
)
```

After registration the pattern is immediately available:

```python
from nono.agent import AgentFactory, OrchestrationRegistry

# Verify registration
assert OrchestrationRegistry.contains("my_custom")
print(OrchestrationRegistry.list_patterns())  # includes "my_custom"

# Use via AgentFactory
factory = AgentFactory()
orch_bp = factory.generate_orchestrated_blueprint(
    "Run my custom domain specific workflow.",
)
# The LLM or keyword heuristics can now recommend "my_custom"
```

### OrchestrationRegistry API

| Method | Description |
| --- | --- |
| `OrchestrationRegistry.register(key, class_name, description, *, keyword_hints, factory, min_sub_agents)` | Register a new pattern |
| `OrchestrationRegistry.unregister(key)` | Remove a registered pattern |
| `OrchestrationRegistry.get(key)` | Get the `PatternRegistration` for a key |
| `OrchestrationRegistry.contains(key)` | Check if a key is registered |
| `OrchestrationRegistry.catalog()` | Return `{key: (class_name, description)}` dict |
| `OrchestrationRegistry.keyword_hints()` | Return `{key: [keywords]}` dict |
| `OrchestrationRegistry.list_patterns()` | Return sorted list of all pattern keys |
| `register_pattern(...)` | Module-level convenience wrapper |

### Factory Function Signature

Every factory function receives these keyword arguments:

| Parameter | Type | Description |
| --- | --- | --- |
| `name` | `str` | Orchestrator name |
| `description` | `str` | Orchestrator description |
| `sub_agents` | `list[BaseAgent]` | Pre-built sub-agents |
| `provider` | `str` | LLM provider (e.g. `"google"`) |
| `model` | `str \| None` | Model name override |
| `pattern_kwargs` | `dict[str, Any]` | Extra parameters (e.g. `max_iterations`, `threshold`) |

The function must return a `BaseAgent` instance (typically your custom orchestrator).

### Overriding Built-in Patterns

You can replace a built-in pattern by registering with the same key:

```python
register_pattern(
    key="sequential",
    class_name="MyBetterSequentialAgent",
    description="Enhanced sequential with logging",
    keyword_hints=["then", "after", "step by step"],
    factory=my_better_sequential_factory,
)
```

> **Warning**: This replaces the built-in pattern globally. The original factory is lost.

### Best Practices

1. **Inherit from `BaseAgent`** — implement both `_run_impl` and `_run_async_impl`.
2. **Validate in the factory** — raise `BlueprintValidationError` for invalid configurations.
3. **Use `min_sub_agents`** — the registry enforces this before calling your factory.
4. **Add `keyword_hints`** — so the heuristic fallback can detect your pattern from task descriptions.
5. **Keep the factory stateless** — it may be called multiple times with different arguments.
6. **Register early** — do it at import time (module level) so the pattern is available before any `AgentFactory` call.
