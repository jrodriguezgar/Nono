# Orchestration Guide — Deterministic, LLM Routing, and Hybrid

> Step-by-step guide to building orchestration pipelines in Nono: from fully deterministic state machines to LLM-powered dynamic routing, and the combinations that unlock both predictability and intelligence.

## Table of Contents

- [Overview](#overview)
- [Three Paradigms at a Glance](#three-paradigms-at-a-glance)
- [Choosing the Right Level of Complexity](#choosing-the-right-level-of-complexity)
- [Part A — Deterministic Orchestration](#part-a--deterministic-orchestration)
  - [A1 — Linear Workflow](#a1--linear-workflow)
  - [A2 — Conditional Branching](#a2--conditional-branching)
  - [A3 — Predicate Branching (branch_if)](#a3--predicate-branching-branch_if)
  - [A4 — Loops via Re-entry](#a4--loops-via-re-entry)
  - [A4b — Parallel Step (parallel_step)](#a4b--parallel-step-parallel_step)
  - [A4c — Loop Step (loop_step)](#a4c--loop-step-loop_step)
  - [A4d — Join Node (join)](#a4d--join-node-join)
  - [A4e — Checkpointing and Resume](#a4e--checkpointing-and-resume)
  - [A4f — Declarative Workflows (JSON/YAML)](#a4f--declarative-workflows-jsonyaml)
  - [A5 — SequentialAgent](#a5--sequentialagent)
  - [A6 — ParallelAgent](#a6--parallelagent)
  - [A7 — LoopAgent](#a7--loopagent)
  - [A8 — MapReduceAgent](#a8--mapreduceagent)
  - [A9 — ConsensusAgent](#a9--consensusagent)
  - [A10 — ProducerReviewerAgent](#a10--producerrevieweragent)
  - [A11 — DebateAgent](#a11--debateagent)
  - [A12 — EscalationAgent](#a12--escalationagent)
  - [A13 — SupervisorAgent](#a13--supervisoragent)
  - [A14 — VotingAgent](#a14--votingagent)
  - [A15 — HandoffAgent](#a15--handoffagent)
  - [A16 — GroupChatAgent](#a16--groupchatagent)
  - [A17 — HierarchicalAgent](#a17--hierarchicalagent)
  - [A18 — GuardrailAgent](#a18--guardrailagent)
  - [A19 — BestOfNAgent](#a19--bestofnagent)
  - [A20 — BatchAgent](#a20--batchagent)
  - [A21 — CascadeAgent](#a21--cascadeagent)
  - [A22 — TreeOfThoughtsAgent](#a22--treeofthoughtsagent)
  - [A23 — PlannerAgent](#a23--planneragent)
  - [A24 — SubQuestionAgent](#a24--subquestionagent)
  - [A25 — ContextFilterAgent](#a25--contextfilteragent)
  - [A26 — ReflexionAgent](#a26--reflexionagent)
  - [A27 — SpeculativeAgent](#a27--speculativeagent)
  - [A28 — CircuitBreakerAgent](#a28--circuitbreakeragent)
  - [A29 — TournamentAgent](#a29--tournamentagent)
  - [A30 — ShadowAgent](#a30--shadowagent)
  - [A31 — CompilerAgent](#a31--compileragent)
  - [A32 — CheckpointableAgent](#a32--checkpointableagent)
  - [A33 — DynamicFanOutAgent](#a33--dynamicfanoutagent)
  - [A34 — SwarmAgent](#a34--swarmagent)
  - [A35 — MemoryConsolidationAgent](#a35--memoryconsolidationagent)
  - [A36 — PriorityQueueAgent](#a36--priorityqueueagent)
  - [A37 — MonteCarloAgent](#a37--montecarloagent)
  - [A38 — GraphOfThoughtsAgent](#a38--graphofthoughtsagent)
  - [A39 — BlackboardAgent](#a39--blackboardagent)
  - [A40 — MixtureOfExpertsAgent](#a40--mixtureofexpertsagent)
  - [A41 — CoVeAgent](#a41--coveagent)
  - [A42 — SagaAgent](#a42--sagaagent)
  - [A43 — LoadBalancerAgent](#a43--loadbalanceragent)
  - [A44 — EnsembleAgent](#a44--ensembleagent)
  - [A45 — TimeoutAgent](#a45--timeoutagent)
  - [A46 — AdaptivePlannerAgent](#a46--adaptiveplanneragent)
  - [A47 — SkeletonOfThoughtAgent](#a47--skeletonofthoughtagent)
  - [A48 — LeastToMostAgent](#a48--leasttomostagent)
  - [A49 — SelfDiscoverAgent](#a49--selfdiscoveragent)
  - [A50 — GeneticAlgorithmAgent](#a50--geneticalgorithmagent)
  - [A51 — MultiArmedBanditAgent](#a51--multiarmedbanditagent)
  - [A52 — SocraticAgent](#a52--socraticagent)
  - [A53 — MetaOrchestratorAgent](#a53--metaorchestratoragent)
  - [A54 — CacheAgent](#a54--cacheagent)
  - [A55 — BudgetAgent](#a55--budgetagent)
  - [A56 — CurriculumAgent](#a56--curriculumagent)
  - [A57 — SelfConsistencyAgent](#a57--selfconsistencyagent)
  - [A58 — MixtureOfAgentsAgent](#a58--mixtureofagentsagent)
  - [A59 — StepBackAgent](#a59--stepbackagent)
  - [A60 — OrchestratorWorkerAgent](#a60--orchestratorworkeragent)
  - [A61 — SelfRefineAgent](#a61--selfrefineagent)
  - [A62 — BacktrackingAgent](#a62--backtrackingagent)
  - [A63 — ChainOfDensityAgent](#a63--chainofdensityagent)
  - [A64 — MediatorAgent](#a64--mediatoragent)
  - [A65 — DivideAndConquerAgent](#a65--divideandconqueragent)
  - [A66 — BeamSearchAgent](#a66--beamsearchagent)
  - [A67 — RephraseAndRespondAgent](#a67--rephraseandrespondagent)
  - [A68 — CumulativeReasoningAgent](#a68--cumulativereasoningagent)
  - [A69 — MultiPersonaAgent](#a69--multipersonaagent)
  - [A70 — AntColonyAgent](#a70--antcolonyagent)
  - [A71 — PipelineParallelAgent](#a71--pipelineparallelagent)
  - [A72 — ContractNetAgent](#a72--contractnetagent)
  - [A73 — RedTeamAgent](#a73--redteamagent)
  - [A74 — FeedbackLoopAgent](#a74--feedbackloopagent)
  - [A75 — WinnowingAgent](#a75--winnowingagent)
  - [A76 — MixtureOfThoughtsAgent](#a76--mixtureofthoughtsagent)
  - [A77 — SimulatedAnnealingAgent](#a77--simulatedannealingagent)
  - [A78 — TabuSearchAgent](#a78--tabusearchagent)
  - [A79 — ParticleSwarmAgent](#a79--particleswarmagent)
  - [A80 — DifferentialEvolutionAgent](#a80--differentialevolutionagent)
  - [A81 — BayesianOptimizationAgent](#a81--bayesianoptimizationagent)
  - [A82 — AnalogicalReasoningAgent](#a82--analogicalreasoningagent)
  - [A83 — ThreadOfThoughtAgent](#a83--threadofthoughtagent)
  - [A84 — ExpertPromptingAgent](#a84--expertpromptingagent)
  - [A85 — BufferOfThoughtsAgent](#a85--bufferofthoughtsagent)
  - [A86 — ChainOfAbstractionAgent](#a86--chainofabstractionagent)
  - [A87 — VerifierAgent](#a87--verifieragent)
  - [A88 — ProgOfThoughtAgent](#a88--progofthoughtagent)
  - [A89 — InnerMonologueAgent](#a89--innermonologueagent)
  - [A90 — RolePlayingAgent](#a90--roleplayingagent)
  - [A91 — GossipProtocolAgent](#a91--gossipprotocolagent)
  - [A92 — AuctionAgent](#a92--auctionagent)
  - [A93 — DelphiMethodAgent](#a93--delphimethodagent)
  - [A94 — NominalGroupAgent](#a94--nominalgroupagent)
  - [A95 — ActiveRetrievalAgent](#a95--activeretrievalagent)
  - [A96 — IterativeRetrievalAgent](#a96--iterativeretrievalagent)
  - [A97 — PromptChainAgent](#a97--promptchainagent)
  - [A98 — HypothesisTestingAgent](#a98--hypothesistestingagent)
  - [A99 — SkillLibraryAgent](#a99--skilllibraryagent)
  - [A100 — RecursiveCriticAgent](#a100--recursivecriticagent)
  - [A101 — DemonstrateSearchPredictAgent](#a101--demonstratesearchpredictagent)
  - [A102 — DoubleLoopLearningAgent](#a102--doublelooplearningagent)
  - [A103 — AgendaAgent](#a103--agendaagent)
  - [A104 — Nesting Deterministic Agents](#a104--nesting-deterministic-agents)
- [Part B — LLM Routing](#part-b--llm-routing)
  - [B1 — RouterAgent (Single Mode)](#b1--routeragent-single-mode)
  - [B2 — RouterAgent (Sequential Mode)](#b2--routeragent-sequential-mode)
  - [B3 — RouterAgent (Parallel Mode)](#b3--routeragent-parallel-mode)
  - [B4 — RouterAgent (Loop Mode)](#b4--routeragent-loop-mode)
  - [B5 — Custom Routing Instructions](#b5--custom-routing-instructions)
  - [B6 — transfer_to_agent (Dynamic Delegation)](#b6--transfer_to_agent-dynamic-delegation)
  - [B7 — RouterAgent vs transfer_to_agent](#b7--routeragent-vs-transfer_to_agent)
- [Part C — Hybrid Orchestration](#part-c--hybrid-orchestration)
  - [C1 — Workflow + tasker_node (Deterministic + AI Steps)](#c1--workflow--tasker_node-deterministic--ai-steps)
  - [C2 — Workflow + agent_node (Deterministic + Agent Steps)](#c2--workflow--agent_node-deterministic--agent-steps)
  - [C3 — Workflow + RouterAgent (Deterministic Shell, Dynamic Core)](#c3--workflow--routeragent-deterministic-shell-dynamic-core)
  - [C4 — Workflow + Conditional Branch + AI Scoring](#c4--workflow--conditional-branch--ai-scoring)
  - [C5 — Router → Deterministic Pipelines](#c5--router--deterministic-pipelines)
  - [C6 — Agent Tool Wrapping a Workflow](#c6--agent-tool-wrapping-a-workflow)
  - [C7 — Full Production Pattern](#c7--full-production-pattern)
- [Orchestration Lifecycle Hooks](#orchestration-lifecycle-hooks)
  - [Available Hooks](#available-hooks)
  - [SequentialAgent Hooks](#sequentialagent-hooks)
  - [ParallelAgent Hooks](#parallelagent-hooks)
  - [LoopAgent Hooks](#loopagent-hooks)
  - [Comparison with Workflow Hooks](#comparison-with-workflow-hooks)
- [Decision Matrix](#decision-matrix)
- [Tracing Across Paradigms](#tracing-across-paradigms)
- [Visualization](#visualization)
- [FAQ](#faq)
- [See Also](#see-also)

---

## Overview

Nono provides two independent orchestration systems that combine naturally:

| System | Module | Unit of work | State model | Routing |
| --- | --- | --- | --- | --- |
| **Workflow** | `nono.workflows` | `Callable[[dict], dict]` | Shared `dict` | `branch()` / `branch_if()` — deterministic + `parallel_step` / `loop_step` / `join` |
| **Agent** | `nono.agent` | `BaseAgent` subclass | `Session` (events + state) | `RouterAgent` — LLM decides at runtime |

The first is **code-first**: the developer defines every step, edge, and condition before execution. The second is **agent-first**: an LLM decides which agents to invoke and how. Combining them yields hybrid architectures where fixed business logic wraps dynamic AI decisions.

```
                 DETERMINISTIC                              LLM ROUTING
        ┌─────────────────────────┐              ┌──────────────────────────┐
        │  Workflow (state dict)  │              │  RouterAgent (4 modes)   │
        │  SequentialAgent        │              │  SupervisorAgent         │
        │  ParallelAgent          │   ◄─────►   │  transfer_to_agent       │
        │  LoopAgent              │   HYBRID     │  HandoffAgent            │
        │  MapReduceAgent         │              │  GroupChatAgent          │
        │  ConsensusAgent         │              │  HierarchicalAgent       │
        │  ProducerReviewerAgent  │              │  LLM picks agents + how  │
        │  DebateAgent            │              └──────────────────────────┘
        │  EscalationAgent        │
        │  VotingAgent            │
        │  GuardrailAgent         │
        │  BestOfNAgent           │
        │  BatchAgent             │
        │  CascadeAgent           │
        │  SpeculativeAgent       │
        │  CircuitBreakerAgent    │
        │  TournamentAgent        │
        │  ShadowAgent            │
        │  CompilerAgent          │
        │  CheckpointableAgent    │
        │  DynamicFanOutAgent     │
        │  SwarmAgent             │
        │  MemoryConsolidation…   │
        │  PriorityQueueAgent     │
        │  MonteCarloAgent        │
        │  GraphOfThoughtsAgent   │
        │  BlackboardAgent        │
        │  MixtureOfExpertsAgent  │
        │  CoVeAgent              │
        │  SagaAgent              │
        │  LoadBalancerAgent      │
        │  EnsembleAgent          │
        │  TimeoutAgent           │
        │  AdaptivePlannerAgent   │
        │  branch() / branch_if() │
        └─────────────────────────┘
                          │                                    │
                          └─────── agent_node() / tasker_node() ──┘
                                    FunctionTool (workflow)
```

---

## Three Paradigms at a Glance

| Paradigm | Who decides? | LLM cost | Predictability | Best for |
| --- | --- | --- | --- | --- |
| **Deterministic** | Developer (code) | None for routing | 100% reproducible | ETL, compliance, fixed pipelines |
| **LLM Routing** | LLM at runtime | 1 lightweight call | Varies by prompt | Intent classification, multi-domain |
| **Hybrid** | Both | Minimal (only AI steps) | High (frame is fixed) | Production systems with intelligent nodes |

---

## Choosing the Right Level of Complexity

> **"Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short."**
>
> **"The most successful implementations weren't using complex frameworks. Instead, they were building with simple, composable patterns."**
>
> — *Anthropic, "Building effective agents" (2024)*

Nono offers 100+ orchestration patterns — but **you should almost never start with a complex one**. The most successful AI systems use the simplest approach that solves the problem. You should consider adding complexity **only when it demonstrably improves outcomes**.

### Workflows vs Agents — the key distinction

Before choosing a pattern, understand the two fundamental architectures:

> **"Workflows** are systems where LLMs and tools are orchestrated through **predefined code paths**. **Agents** are systems where LLMs **dynamically direct** their own processes and tool usage, maintaining control over how they accomplish tasks."
> — *Anthropic*

In Nono:

| Architecture | Who decides routing? | Nono module | Examples |
| --- | --- | --- | --- |
| **Workflow** | Developer (code) | `nono.workflows` — `Workflow`, `SequentialAgent`, `ParallelAgent`, `LoopAgent` | ETL, compliance, fixed pipelines |
| **Agent** | LLM (at runtime) | `nono.agent` — `RouterAgent`, `SupervisorAgent`, `transfer_to_agent` | Intent classification, open-ended Q&A |
| **Hybrid** | Both | `agent_node()`, `tasker_node()`, `FunctionTool` wrapping a `Workflow` | Production systems with intelligent gates |

**Start with workflows.** They are predictable, reproducible, and easy to debug. Switch to agent-driven routing only when the routing decision itself requires semantic understanding that code cannot express.

### The Complexity Ladder

Always start at the top row and move down only when the current level is insufficient:

| Level | What to use | When to move down |
| --- | --- | --- |
| **0 — Single LLM call** | `TaskExecutor.execute("prompt")` | Output quality is too low even after prompt engineering |
| **1 — Augmented LLM** | `Agent` with `@tool` / `FunctionTool` | Task requires external data, actions, or memory |
| **2 — Simple workflow** | `Workflow` (linear) or `SequentialAgent` | Task has distinct steps that benefit from decomposition |
| **3 — Branching workflow** | `branch_if()`, `parallel_step()`, `loop_step()` | Task requires conditional logic, parallelism, or iteration |
| **4 — LLM routing** | `RouterAgent`, `transfer_to_agent` | Routing decisions require semantic understanding |
| **5 — Advanced patterns** | `TreeOfThoughts`, `MonteCarlo`, `Saga`, etc. | Measurable quality gain over simpler patterns; domain demands it |

### Five Canonical Patterns

Before exploring the full catalog, know that most production systems are built from just **five** workflow shapes:

| Pattern | Shape | Nono equivalent | Example use case |
| --- | --- | --- | --- |
| **Prompt chaining** | A → gate → B → C | `Workflow.connect()` + `branch_if` | Generate copy → check quality → translate |
| **Routing** | Classify → specialist | `RouterAgent` / `Workflow.branch()` | Route support tickets to billing vs. tech |
| **Parallelization** | Fan-out → merge | `ParallelAgent` / `parallel_step` / `VotingAgent` | Multi-perspective analysis; majority-vote |
| **Orchestrator-workers** | LLM decomposes → N workers → merge | `PlannerAgent` / `DynamicFanOutAgent` | Complex code changes across multiple files |
| **Evaluator-optimizer** | Generate → evaluate → loop | `ProducerReviewerAgent` / `ReflexionAgent` | Iterative writing with editorial feedback |

If your task fits one of these, start there. The remaining 95+ patterns exist for specialized needs — reach for them when the five above are demonstrably insufficient.

### Design Principles

> **"Frameworks often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug."** — *Anthropic*

1. **Simplicity first.** Every level of orchestration adds latency, cost, and debugging surface. Keep your design as simple as possible. If a single well-crafted `TaskExecutor.execute()` call solves the problem, stop there.

2. **Measure before promoting.** Move to the next complexity level only with evidence (eval scores, latency budgets, failure rates) that the current level falls short. Never add a pattern because it *sounds* powerful — add it because the data says the simpler approach isn't good enough.

3. **Transparency.** Prioritize transparency by explicitly showing the agent's planning steps. Use `TraceCollector`, `StateTransition`, and `runner.stream()` to make every decision visible. An orchestration you can't observe is one you can't trust.

4. **Craft your tool interface.** Tool definitions should be given just as much prompt engineering attention as your overall prompts. Invest real effort in tool descriptions (docstrings, `FunctionTool.description`, parameter names). Think of it as writing documentation for a junior developer — if it's ambiguous to a human, it's ambiguous to the LLM.

5. **Composability over cleverness.** Prefer nesting simple, composable patterns (`SequentialAgent` → `ParallelAgent` → `LoopAgent`) over reaching for a single complex pattern that tries to do everything. Simple pieces you understand are always better than a monolithic pattern you can't debug.

6. **Minimize abstraction layers.** Understand the underlying prompts and responses at every level of your pipeline. When Nono provides a high-level agent (e.g., `PlannerAgent`), read the generated routing prompt before deploying — incorrect assumptions about what's under the hood are a common source of error.

> **Rule of thumb:** If you can solve it with a single well-crafted `TaskExecutor.execute()` call, do that. If you need orchestration, start with a linear `Workflow` or `SequentialAgent`. Reach for routers, planners, and advanced agents only when the simpler approach measurably fails.

---

## Part A — Deterministic Orchestration

In deterministic orchestration, **the developer defines every decision** at design time. The execution path is fully predictable and reproducible.

### A1 — Linear Workflow

The simplest pipeline: steps execute in registration order.

```python
from nono.workflows import Workflow

def extract(state: dict) -> dict:
    """Simulate data extraction."""
    return {"raw_data": f"Data about {state['topic']}"}

def transform(state: dict) -> dict:
    """Normalize and clean data."""
    return {"clean_data": state["raw_data"].upper()}

def load(state: dict) -> dict:
    """Store the processed result."""
    return {"stored": True, "output": state["clean_data"]}

etl = Workflow("etl_pipeline")
etl.step("extract", extract)
etl.step("transform", transform)
etl.step("load", load)
etl.connect("extract", "transform", "load")

result = etl.run(topic="healthcare AI")
print(result["output"])
# → "DATA ABOUT HEALTHCARE AI"
```

**How it works**: each step receives the full `state` dict and returns a dict with keys to merge. Steps execute in the order defined by `connect()`.

```
📋 etl_pipeline
├── ○ extract
├── ○ transform
└── ○ load
```

### A2 — Conditional Branching

Use `branch()` to route execution based on state values. The condition function receives the state and returns the next step name.

```python
from nono.workflows import Workflow, END

def review(state: dict) -> dict:
    """Score the content quality."""
    score = len(state.get("draft", "")) % 100  # simulated score
    return {"quality_score": score}

def publish(state: dict) -> dict:
    """Publish the approved content."""
    return {"status": "published", "final": state["draft"]}

def revise(state: dict) -> dict:
    """Send content back for revision."""
    return {"draft": state["draft"] + " [REVISED]", "revision_count": state.get("revision_count", 0) + 1}

flow = Workflow("review_pipeline")
flow.step("review", review)
flow.step("publish", publish)
flow.step("revise", revise)

# Route from review based on quality_score
flow.branch("review", lambda s: "publish" if s["quality_score"] >= 80 else "revise")

# After revision, go back to review (loop)
flow.connect("revise", "review")

result = flow.run(draft="Initial article about AI diagnostics")
print(result["status"])         # "published" or still revising
print(result.get("revision_count", 0))
```

```
📋 review_pipeline
├── ◆ review → publish | revise
├── ○ publish
└── ○ revise → review
```

**Key point**: `branch()` takes a callable `(state) -> step_name`. Return `END` to terminate the workflow.

### A3 — Predicate Branching (branch_if)

For simple boolean conditions, `branch_if()` is more readable:

```python
flow = Workflow("approval")
flow.step("score", review)
flow.step("approve", publish)
flow.step("reject", revise)

flow.branch_if(
    "score",
    lambda s: s["quality_score"] >= 80,
    then="approve",
    otherwise="reject",
)

result = flow.run(draft="AI in diagnostics...")
```

`branch_if()` supports dotted key access and comparison operators for even simpler cases:

```python
# These are equivalent:
flow.branch_if("score", lambda s: s["quality_score"] >= 80, then="approve", otherwise="reject")
```

### A4 — Loops via Re-entry

Workflows don't have a dedicated loop construct — loops are expressed as branches that point backward. The **branch step itself** must be the loop re-entry point (the cycle detector allows revisiting branch steps):

```python
flow = Workflow("refine_loop")

# Combined step: improve text and score it
flow.step("improve", lambda s: {
    "text": s.get("text", "initial") + " improved",
    "iteration": s.get("iteration", 0) + 1,
    "score": min((s.get("iteration", 0) + 1) * 30, 100),
})
flow.step("finalize", lambda s: {"final": s["text"]})

# Branch on 'improve' — it loops back to itself until score >= 90
flow.branch("improve", lambda s: "finalize" if s["score"] >= 90 else "improve")

result = flow.run()
print(result["iteration"])  # 3
print(result["final"])      # "initial improved improved improved"
```

```
📋 refine_loop
├── ◆ improve → finalize | improve
└── ○ finalize
```

**Key rule**: the step with the `branch()` can be revisited; other steps cannot. Place your loop logic in the branch step.

> **Tip**: for simple loops with a condition, prefer [`loop_step()`](#a4c--loop-step-loop_step) — it's more readable and doesn't require manual re-entry wiring.

### A4b — Parallel Step (parallel_step)

Run multiple functions **concurrently** within a single Workflow step. All functions receive the same state snapshot and their results are merged when all complete.

```python
from nono.workflows import Workflow

flow = Workflow("data_gathering")

# Step 1: prepare the query
flow.step("prepare", lambda s: {"query": s["topic"].strip().lower()})

# Step 2: fan-out to three concurrent data sources
flow.parallel_step("fetch", {
    "news":   lambda s: {"news": f"News about {s['query']}"},
    "papers": lambda s: {"papers": f"Papers about {s['query']}"},
    "social": lambda s: {"social": f"Social data about {s['query']}"},
}, max_workers=3)

# Step 3: combine results
flow.step("combine", lambda s: {
    "report": f"{s['news']}\n{s['papers']}\n{s['social']}"
})

flow.connect("prepare", "fetch", "combine")
result = flow.run(topic="AI diagnostics")
print(result["report"])
```

```
📋 data_gathering (Workflow, 3 steps)
├── ○ prepare
├── ⏸ fetch  (parallel: news, papers, social)
└── ○ combine
```

**Key parameters:**

| Parameter | Default | Description |
| --- | --- | --- |
| `fns` | (required) | `dict[str, Callable]` — sub-functions to run in parallel |
| `max_workers` | `len(fns)` | Thread pool size |

**Thread safety**: each function receives an **independent copy** of the state dict. Results are merged into the original state after all complete.

### A4c — Loop Step (loop_step)

Repeat a function while a condition holds, up to a maximum number of iterations. This is the **preferred** way to implement deterministic loops in Workflows.

```python
from nono.workflows import Workflow

flow = Workflow("quality_loop")
flow.step("draft", lambda s: {"text": "initial draft", "quality": 0.3})

flow.loop_step(
    "improve",
    lambda s: {"text": s["text"] + " [improved]", "quality": s["quality"] + 0.2},
    condition=lambda s: s["quality"] < 0.9,
    max_iterations=5,
)

flow.step("publish", lambda s: {"status": "published"})
flow.connect("draft", "improve", "publish")

result = flow.run()
print(result["quality"])              # >= 0.9
print(result["__loop_iterations__"])  # number of iterations executed
print(result["status"])               # "published"
```

```
📋 quality_loop (Workflow, 3 steps)
├── ○ draft
├── 🔁 improve  (loop max 5x)
└── ○ publish
```

**Key parameters:**

| Parameter | Default | Description |
| --- | --- | --- |
| `condition` | (required) | `Callable[[dict], bool]` — loop continues while `True` |
| `max_iterations` | `10` | Safety cap to prevent infinite loops |

**State key**: after execution, `state["__loop_iterations__"]` contains the number of iterations actually run.

**When to use `loop_step` vs re-entry loops (A4)**:

| Criterion | `loop_step()` | Re-entry (`branch → self`) |
| --- | --- | --- |
| Readability | Cleaner — one method call | Requires manual wiring |
| Condition | Python predicate | Branch callable |
| Max iterations | Built-in safety cap | Manual counter in state |
| Visibility | Shows `🔁` icon in visualization | Shows `◆` branch icon |
| Best for | Simple repeat-until patterns | Complex loops with multiple entry/exit points |

### A4d — Join Node (join)

An explicit **wait-for-all** barrier. Validates that all required predecessor steps have executed before continuing. An optional `reducer` can post-process the merged state.

```python
from nono.workflows import Workflow

flow = Workflow("merge_pipeline")

# Two independent branches
flow.step("branch_a", lambda s: {"analysis_a": "tech perspective"})
flow.step("branch_b", lambda s: {"analysis_b": "business perspective"})

# Join with reducer
flow.join("merge", wait_for=["branch_a", "branch_b"],
          reducer=lambda s: {"summary": f"{s['analysis_a']} + {s['analysis_b']}"})

flow.step("report", lambda s: {"output": f"Report: {s['summary']}"})
flow.connect("branch_a", "branch_b", "merge", "report")

result = flow.run()
print(result["summary"])  # "tech perspective + business perspective"
```

```
📋 merge_pipeline (Workflow, 4 steps)
├── ○ branch_a
├── ○ branch_b
├── ⏩ merge  (join: branch_a, branch_b)
└── ○ report
```

### A4e — Checkpointing and Resume

For long-running pipelines, enable **state persistence** so you can resume after a crash without re-executing completed steps.

```python
from nono.workflows import Workflow

flow = Workflow("long_pipeline")
flow.enable_checkpoints("./checkpoints")

flow.step("expensive_a", expensive_fn_a)
flow.step("expensive_b", expensive_fn_b)
flow.step("expensive_c", expensive_fn_c)
flow.connect("expensive_a", "expensive_b", "expensive_c")

# First run — saves state after each step
result = flow.run(data="input")

# If it crashed during expensive_b, resume from the checkpoint:
result = flow.run(resume=True, data="input")
# → skips expensive_a, resumes from expensive_b
```

Checkpoints are saved as JSON in the specified directory (atomic write-then-rename). Use `flow.resume()` to inspect the last checkpoint programmatically:

```python
state, last_step = flow.resume()
# state = {"data": "input", "result_a": ...}
# last_step = "expensive_a"
```

### A4f — Declarative Workflows (JSON/YAML)

Define pipelines in configuration files instead of Python code. Load them with `load_workflow()`:

**JSON format:**
```json
{
  "name": "review_pipeline",
  "steps": [
    {"name": "draft", "type": "passthrough"},
    {"name": "evaluate", "type": "tasker",
     "provider": "google", "system_prompt": "Score 0–100.",
     "input_key": "draft", "output_key": "score"},
    {"name": "publish"},
    {"name": "revise"}
  ],
  "edges": [["draft", "evaluate"]],
  "branches": [
    {"from": "evaluate", "condition": "score > 80",
     "then": "publish", "otherwise": "revise"}
  ],
  "checkpoint_dir": "./checkpoints"
}
```

**Loading:**
```python
from nono.workflows import load_workflow

flow = load_workflow("pipelines/review.json", step_registry={
    "draft": write_draft_fn,
    "publish": publish_fn,
    "revise": revise_fn,
})
result = flow.run()
```

Steps with `type: passthrough` are no-ops unless overridden by `step_registry`. Steps with `type: tasker` auto-create `tasker_node()` calls. YAML files (`.yaml`/`.yml`) are also supported (requires `pip install pyyaml`).

### A5 — SequentialAgent

**Pattern**: Pipeline / Chain of Responsibility — the simplest orchestration pattern where agents execute in a fixed linear order, each receiving the previous agent's output as its input.

Each agent's response is forwarded as the `user_message` to the next agent in the chain. The final agent's response becomes the pipeline output. This is the agent equivalent of `Workflow.connect("a", "b", "c")` but with LLM agents instead of pure functions.

**When to use**: content pipelines (research → write → review), data processing chains, any task that naturally decomposes into sequential stages where each stage depends on the previous one's output.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `Workflow.connect()` | Workflow uses shared state dict; SequentialAgent passes messages between agents |
| `PlannerAgent` | Planner lets an LLM decide the order; Sequential uses developer-defined order |
| `CheckpointableAgent` | CheckpointableAgent adds persistence; Sequential has no checkpoint support |

```python
from nono.agent import Agent, SequentialAgent, Runner

researcher = Agent(
    name="researcher",
    provider="google",
    instruction="Research the topic and provide factual data with sources.",
)
writer = Agent(
    name="writer",
    provider="google",
    instruction="Write a clear, engaging article based on the provided research.",
)
reviewer = Agent(
    name="reviewer",
    provider="google",
    instruction="Review the article for accuracy, clarity, and engagement.",
)

pipeline = SequentialAgent(
    name="article_pipeline",
    sub_agents=[researcher, writer, reviewer],
)

runner = Runner(pipeline)
result = runner.run("AI-powered diagnostics in healthcare for 2026")
```

```
🤖 article_pipeline (SequentialAgent)
├── 🤖 researcher
├── 🤖 writer
└── 🤖 reviewer
```

**Data flow**: `user_message → researcher → (response as new user_message) → writer → reviewer → final response`

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Ordered list of agents to execute sequentially |

### A6 — ParallelAgent

**Pattern**: Fork-Join / Scatter-Gather — runs independent agents concurrently and collects all results into a single state entry. This is the agent equivalent of `Workflow.parallel_step()`.

All agents receive the same user message (or customised messages via `message_map`) and execute in parallel using a thread pool (sync) or coroutines (async). Results are collected into `session.state[result_key]` as a `{agent_name: response}` dict. No agent sees another's output — they work in total isolation.

**When to use**: multi-perspective analysis, concurrent data gathering from different sources, independent evaluations that can run simultaneously, and any scenario where agents don't depend on each other's output.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential chains agents; Parallel runs them concurrently |
| `MapReduceAgent` | MapReduce adds a reduce/synthesis step; Parallel just collects |
| `EnsembleAgent` | Ensemble aggregates into one output; Parallel keeps outputs separate |
| `LoadBalancerAgent` | LoadBalancer picks *one* agent; Parallel runs *all* |

```python
from nono.agent import Agent, ParallelAgent, Runner

tech = Agent(name="tech_analyst", provider="google", instruction="Analyze from a technology perspective.")
biz = Agent(name="biz_analyst", provider="google", instruction="Analyze from a business perspective.")
ethics = Agent(name="ethics_analyst", provider="google", instruction="Analyze ethical implications.")

gather = ParallelAgent(
    name="multi_perspective",
    sub_agents=[tech, biz, ethics],
    result_key="perspectives",            # auto-collect all results
    message_map={                          # each agent gets a specific angle
        "tech_analyst": "AI diagnostic accuracy, speed, and EHR integration",
        "biz_analyst": "Market size, competition, reimbursement models",
        "ethics_analyst": "Bias in training data, liability, patient consent",
    },
)

runner = Runner(gather)
runner.run("analyze")

# All responses in one dict:
for name, analysis in runner.session.state["perspectives"].items():
    print(f"[{name}] {analysis[:100]}...")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents to run concurrently |
| `result_key` | `None` | Collects `{agent_name: response}` into `session.state[key]` |
| `message_map` | `None` | `dict[str, str]` — custom message per agent |
| `max_workers` | `len(sub_agents)` | Thread pool size in sync mode (ignored in async) |

### A7 — LoopAgent

**Pattern**: Iterative Refinement Loop — repeats a set of sub-agents until a condition is met or a maximum iteration count is reached. This is the agent equivalent of `Workflow.loop_step()`.

Each iteration executes all sub-agents sequentially (like a mini `SequentialAgent`), then evaluates `stop_condition`. If the condition returns `True`, the loop exits. If `max_iterations` is reached, the loop exits regardless. The session state accumulates across iterations, enabling progressive refinement.

**When to use**: iterative content improvement, agent chains that need multiple passes (write → review → revise), convergence-based algorithms, and any task where "good enough" requires multiple attempts.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `Workflow.loop_step()` | Workflow loops a single function; LoopAgent loops a sequence of agents |
| `ProducerReviewerAgent` | ProducerReviewer has a built-in produce/review protocol; LoopAgent is generic |
| `ReflexionAgent` | Reflexion accumulates failure reflections; LoopAgent has no built-in reflection |

```python
from nono.agent import Agent, LoopAgent, Runner

improver = Agent(
    name="improver",
    provider="google",
    instruction=(
        "Improve the article quality. After improving, "
        "set state['quality'] to a float 0.0-1.0."
    ),
)

refine = LoopAgent(
    name="quality_loop",
    sub_agents=[improver],
    max_iterations=5,
    stop_condition=lambda state: state.get("quality", 0) > 0.9,
)

runner = Runner(refine)
result = runner.run("Draft about AI diagnostics...")
```

**Multi-agent loops**: put multiple agents inside — they run sequentially each iteration:

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

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents that execute sequentially per iteration |
| `max_iterations` | `5` | Safety cap to prevent infinite loops |
| `stop_condition` | `None` | `Callable[[dict], bool]` — loop exits when `True` |

### A8 — MapReduceAgent

**Pattern**: Map-Reduce (Dean & Ghemawat, 2004) — the classic distributed computing paradigm adapted for agent orchestration. Multiple "mapper" agents process the input in parallel, then a single "reducer" agent synthesises all outputs into a final result.

The map phase runs all sub-agents concurrently (like `ParallelAgent`). The reduce phase feeds all mapper outputs to the `reduce_agent`, which produces the final consolidated response. Optional `message_map` allows sending customised prompts to each mapper.

**When to use**: multi-source information gathering (web + DB + docs → summary), parallel analysis from different angles with a synthesis step, and any scenario that follows the "gather independently, combine centrally" pattern.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ParallelAgent` | Parallel collects results; MapReduce adds a synthesis/reduce step |
| `ConsensusAgent` | Consensus uses a judge to arbitrate; MapReduce uses a reducer to combine |
| `DynamicFanOutAgent` | DynamicFanOut creates items at runtime; MapReduce has fixed sub-agents |
| `SubQuestionAgent` | SubQuestion decomposes questions; MapReduce fans out the same question to different agents |

```python
from nono.agent import Agent, MapReduceAgent, Runner

search_web = Agent(name="search_web", instruction="Search the web.", provider="google")
search_db = Agent(name="search_db", instruction="Search the database.", provider="google")
search_docs = Agent(name="search_docs", instruction="Search internal docs.", provider="google")
summariser = Agent(name="summariser", instruction="Combine all research into a summary.", provider="google")

mapreduce = MapReduceAgent(
    name="summarise_all",
    sub_agents=[search_web, search_db, search_docs],
    reduce_agent=summariser,
    result_key="raw_results",  # optional: store mapper results in state
)

runner = Runner(mapreduce)
result = runner.run("What do we know about quantum computing?")
```

**Custom per-agent messages** via `message_map`:

```python
mapreduce = MapReduceAgent(
    name="multi_source",
    sub_agents=[search_web, search_db, search_docs],
    reduce_agent=summariser,
    message_map={
        "search_web": "Find recent news about quantum computing",
        "search_db": "Query internal DB for quantum computing papers",
    },
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Mapper agents (run in parallel) |
| `reduce_agent` | (required) | Agent that synthesises all mapper outputs |
| `message_map` | `None` | Custom messages per mapper agent |
| `result_key` | `None` | Optional state key to store raw mapper results |

### A9 — ConsensusAgent

**Pattern**: Consensus / Jury — inspired by ensemble learning and the "wisdom of crowds" principle. Multiple agents independently answer the same question, then a dedicated judge agent reviews all responses and synthesises a single consensus answer.

Unlike `VotingAgent` (which picks the majority answer mechanically), `ConsensusAgent` uses an LLM judge that can reason about nuances, resolve contradictions, and synthesise the best elements from each response. This is more expensive (extra LLM call for the judge) but produces higher-quality synthesis.

**When to use**: fact-checking (cross-reference multiple LLMs), high-stakes decisions where agreement matters, multi-model quality assurance, and any scenario where you want an informed synthesis rather than a simple majority vote.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting picks majority mechanically; Consensus uses an LLM judge to synthesise |
| `EnsembleAgent` | Ensemble concatenates/weights outputs; Consensus produces a unified answer |
| `MapReduceAgent` | MapReduce has specialised mappers; Consensus has identical responders + a judge |
| `DebateAgent` | Debate is adversarial (argue for/against); Consensus is cooperative (agree on truth) |

```python
from nono.agent import Agent, ConsensusAgent, Runner

model_a = Agent(name="model_a", instruction="Answer the question.", provider="google")
model_b = Agent(name="model_b", instruction="Answer the question.", provider="openai")
model_c = Agent(name="model_c", instruction="Answer the question.", provider="deepseek")
judge = Agent(name="judge", instruction="Review all answers and produce a consensus.", provider="google")

consensus = ConsensusAgent(
    name="fact_check",
    sub_agents=[model_a, model_b, model_c],
    judge_agent=judge,
    result_key="votes",  # optional: store individual answers in state
)

runner = Runner(consensus)
result = runner.run("What is the capital of Australia?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Responder agents (run in parallel) |
| `judge_agent` | (required) | LLM judge that synthesises the consensus |
| `result_key` | `None` | Optional state key to store individual responses |

### A10 — ProducerReviewerAgent

**Pattern**: Producer-Consumer with Quality Gate — an iterative produce-then-review loop inspired by editorial workflows. The producer generates content, the reviewer evaluates it, and the loop repeats until the reviewer signals approval (via a keyword) or `max_iterations` is reached.

Unlike `LoopAgent` (which is generic), `ProducerReviewerAgent` has a built-in **approval protocol**: the reviewer's response is scanned for `approval_keyword`. If found, the loop exits successfully. If not, the reviewer's feedback is appended to the conversation and the producer retries with awareness of the critique.

**When to use**: content creation pipelines (write → edit), code generation with review (code → test), any workflow where a producer's output must pass a quality gate before proceeding.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `LoopAgent` | LoopAgent uses a generic `stop_condition`; ProducerReviewer has built-in approval keyword protocol |
| `ReflexionAgent` | Reflexion accumulates failure reflections in memory; ProducerReviewer passes feedback directly |
| `GuardrailAgent` | GuardrailAgent validates format/safety; ProducerReviewer validates quality/content |
| `CompilerAgent` | Compiler optimises the instruction; ProducerReviewer optimises the output |

```python
from nono.agent import Agent, ProducerReviewerAgent, Runner

writer = Agent(
    name="writer",
    instruction="Write a blog post. Incorporate any reviewer feedback.",
    provider="google",
)
editor = Agent(
    name="editor",
    instruction="Review the blog post. Say APPROVED if it meets quality standards.",
    provider="google",
)

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

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `producer` | (required) | Agent that generates content |
| `reviewer` | (required) | Agent that evaluates and provides feedback |
| `max_iterations` | `3` | Maximum produce-review cycles |
| `approval_keyword` | `"APPROVED"` | Keyword in reviewer's response that signals success |

### A11 — DebateAgent

**Pattern**: Adversarial Debate / Red Team-Blue Team — inspired by dialectical reasoning (thesis → antithesis → synthesis) and structured academic debate. Two agents argue opposing positions across multiple rounds, then a judge renders a verdict.

Each round, agent A presents its argument (seeing agent B's previous argument), then agent B responds. After `max_rounds` of debate, the judge receives the full transcript and produces a final verdict that weighs both perspectives. This surfaces stronger arguments than either agent would produce alone.

**When to use**: ethical analysis (pro vs con), risk assessment (optimistic vs pessimistic), decision-making where both sides must be heard, red-teaming (attacker vs defender), and any binary question that benefits from adversarial reasoning.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ConsensusAgent` | Consensus is cooperative (agents agree); Debate is adversarial (agents disagree) |
| `TournamentAgent` | Tournament eliminates losers; Debate synthesises both sides into a verdict |
| `ProducerReviewerAgent` | ProducerReviewer is collaborative (improve together); Debate is confrontational |

```python
from nono.agent import Agent, DebateAgent, Runner

optimist = Agent(name="optimist", instruction="Argue the positive case.", provider="google")
pessimist = Agent(name="pessimist", instruction="Argue the negative case.", provider="google")
moderator = Agent(name="moderator", instruction="Judge the debate and give a verdict.", provider="google")

debate = DebateAgent(
    name="ai_ethics_debate",
    agent_a=optimist,
    agent_b=pessimist,
    judge=moderator,
    max_rounds=3,
)

runner = Runner(debate)
result = runner.run("Should AI replace radiologists entirely?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent_a` | (required) | First debater (typically argues the positive case) |
| `agent_b` | (required) | Second debater (typically argues the negative case) |
| `judge` | (required) | Agent that renders the final verdict |
| `max_rounds` | `3` | Number of debate rounds before judging |

### A12 — EscalationAgent

**Pattern**: Escalation Chain / Fallback Ladder — a cost-optimisation pattern that tries agents sequentially from cheapest/fastest to most expensive/capable, stopping at the first success.

Each agent's response is checked against `failure_keyword`. If the keyword is detected (e.g. "I don't know"), the agent is considered unsuccessful and the next one in the chain is tried. This enables a cost-efficient architecture where simple questions are answered by cheap/fast models (Groq, Cerebras) and only complex questions escalate to expensive models (GPT-4, Claude).

**When to use**: cost-optimised multi-provider setups, customer support tiers (L1 → L2 → L3), progressive difficulty handling, and any scenario where you want to minimise cost while maintaining quality.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CascadeAgent` | Cascade uses quality scoring; Escalation uses keyword-based failure detection |
| `SpeculativeAgent` | Speculative races all in parallel; Escalation tries sequentially |
| `CircuitBreakerAgent` | CircuitBreaker remembers failure history; Escalation is stateless per request |
| `LoadBalancerAgent` | LoadBalancer distributes evenly; Escalation prefers cheaper agents |

```python
from nono.agent import Agent, EscalationAgent, Runner

fast = Agent(name="fast", instruction="Answer quickly. Say 'I don't know' if unsure.", provider="groq")
medium = Agent(name="medium", instruction="Answer carefully.", provider="google")
powerful = Agent(name="powerful", instruction="Answer thoroughly.", provider="openai")

escalation = EscalationAgent(
    name="smart_fallback",
    sub_agents=[fast, medium, powerful],
    failure_keyword="I don't know",
)

runner = Runner(escalation)
result = runner.run("Explain the Riemann hypothesis")
```

**Custom escalation callback**:

```python
escalation = EscalationAgent(
    name="smart_fallback",
    sub_agents=[fast, medium, powerful],
    failure_keyword="I don't know",
    on_escalation=lambda agent, resp: print(f"Escalated past {agent}"),
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents in escalation order (cheapest first) |
| `failure_keyword` | (required) | Keyword that triggers escalation to the next agent |
| `on_escalation` | `None` | `Callable[[str, str], None]` — callback on each escalation |

### A13 — SupervisorAgent

**Pattern**: Supervisor / Manager-Worker — inspired by organisational management hierarchies. An LLM-powered supervisor reads the user request, analyses the available workers (via their descriptions), delegates to the best one, evaluates the result, and can re-delegate if unsatisfied.

The supervisor uses an LLM to make delegation decisions (making this a **hybrid** pattern — deterministic loop with LLM routing inside). Each iteration: (1) the supervisor selects a worker, (2) the worker executes, (3) the supervisor evaluates the result, (4) if satisfied, returns; if not, re-delegates (possibly to a different worker). The loop continues until satisfied or `max_iterations` is reached.

**When to use**: complex tasks that require intelligent delegation, teams of specialised agents where the best assignment isn't obvious, quality-controlled workflows where a manager must approve output, and scenarios where re-work/re-delegation may be necessary.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `RouterAgent` | Router delegates once and returns; Supervisor evaluates and can re-delegate |
| `HierarchicalAgent` | Hierarchical has nested sub-teams; Supervisor has flat worker pool |
| `PlannerAgent` | Planner creates a multi-step plan; Supervisor delegates one task at a time |
| `AdaptivePlannerAgent` | AdaptivePlanner maintains a plan; Supervisor re-evaluates per delegation |

```python
from nono.agent import Agent, SupervisorAgent, Runner

coder = Agent(name="coder", description="Writes Python code.", instruction="Write clean Python.", provider="google")
writer = Agent(name="writer", description="Writes prose.", instruction="Write clear text.", provider="google")
researcher = Agent(name="researcher", description="Researches facts.", instruction="Research with sources.", provider="google")

supervisor = SupervisorAgent(
    name="manager",
    provider="google",
    model="gemini-3-flash-preview",
    sub_agents=[coder, writer, researcher],
    max_iterations=3,
    supervisor_instruction="Pick the best worker for each task. Re-delegate if the result is poor.",
)

runner = Runner(supervisor)
result = runner.run("Write a Python function to calculate BMI, with documentation")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Worker agents (must have `description` for routing) |
| `provider` / `model` | (required) | LLM for the supervisor's delegation decisions |
| `max_iterations` | `3` | Maximum delegation rounds |
| `supervisor_instruction` | `""` | Additional instructions for the supervisor LLM |

### A14 — VotingAgent

**Pattern**: Majority Vote / Self-Consistency (Wang et al., 2022) — N agents answer the same question independently and the most frequent response wins. This requires **no LLM judge** — the aggregation is purely mechanical.

Inspired by the self-consistency prompting technique where generating multiple answers and taking the majority reduces hallucination rates. Each agent runs in parallel, responses are normalised (via an optional `normalize` function), and a frequency count determines the winner. This works best for questions with discrete, verifiable answers (math, factual Q&A, classification).

**When to use**: math problems (multiple sampling for accuracy), factual questions with definitive answers, classification tasks, and any query where the "correct" answer should appear more frequently than hallucinated alternatives.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ConsensusAgent` | Consensus uses an LLM judge (intelligent synthesis); Voting uses frequency counting (mechanical) |
| `EnsembleAgent` | Ensemble combines all outputs; Voting picks the single most common one |
| `BestOfNAgent` | BestOfN scores with a function; Voting counts frequency |
| `TournamentAgent` | Tournament uses pairwise judging; Voting uses simple majority |

```python
from nono.agent import Agent, VotingAgent, Runner

model_a = Agent(name="model_a", instruction="Answer the math question.", provider="google")
model_b = Agent(name="model_b", instruction="Answer the math question.", provider="openai")
model_c = Agent(name="model_c", instruction="Answer the math question.", provider="deepseek")

voting = VotingAgent(
    name="majority_vote",
    sub_agents=[model_a, model_b, model_c],
    result_key="vote_details",  # optional: store vote breakdown in state
)

runner = Runner(voting)
result = runner.run("What is 7 * 8?")
```

**Custom normalization** (useful when responses have different formatting):

```python
voting = VotingAgent(
    name="majority_vote",
    sub_agents=[model_a, model_b, model_c],
    normalize=lambda s: s.strip().lower().rstrip("."),
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Voting agents (run in parallel) |
| `normalize` | `None` | `Callable[[str], str]` — normalises responses before counting |
| `result_key` | `None` | Optional state key to store vote breakdown |

### A15 — HandoffAgent

**Pattern**: Peer-to-Peer Handoff Mesh — inspired by call-centre transfer protocols where agents transfer **full control** of a conversation to a peer. Unlike `transfer_to_agent` (tool-based delegation where the caller retains control and sees the result), handoff transfers **ownership** — the receiving agent takes over the conversation completely.

The active agent runs with the full conversation history. If its response contains `HANDOFF: <target_name>`, the framework transfers control to that target agent. The target then runs with the accumulated history, and may itself hand off to another agent. The loop continues until an agent completes without handing off, or `max_handoffs` is reached.

`handoff_rules` defines the allowed transfer graph — which agents can hand off to which others. This prevents arbitrary transfers and ensures the conversation stays within valid paths.

**When to use**: tutoring systems (triage → math_tutor → science_tutor), customer support with specialist routing, multi-domain assistants where full context transfer is needed, and any scenario where agents must pass full ownership of a conversation.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `transfer_to_agent` | transfer_to_agent is a tool call (caller retains control); HandoffAgent transfers ownership |
| `SwarmAgent` | SwarmAgent uses session state for handoffs; HandoffAgent uses keyword detection in text |
| `RouterAgent` | Router makes one routing decision; HandoffAgent allows multi-hop transfers |
| `SupervisorAgent` | Supervisor delegates and evaluates; HandoffAgent is peer-to-peer (no manager) |

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

runner = Runner(handoff)
result = runner.run("What year did the French Revolution start?")
```

**How it works**: The active agent runs with the full conversation history. If its response contains `HANDOFF: <target_name>`, control transfers to that target. The loop continues until an agent completes without handing off, or `max_handoffs` is reached.

**Custom handoff keyword**:

```python
handoff = HandoffAgent(
    name="support",
    entry_agent=triage,
    handoff_rules={"triage": [billing, tech]},
    handoff_keyword="TRANSFER:",  # agents write "TRANSFER: billing"
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `entry_agent` | (required) | First agent to handle the conversation |
| `handoff_rules` | (required) | `dict[str, list[Agent]]` — allowed transfer graph |
| `max_handoffs` | `5` | Maximum agent-to-agent transfers |
| `handoff_keyword` | `"HANDOFF:"` | Keyword prefix that triggers transfer |

### A16 — GroupChatAgent

**Pattern**: Multi-Agent Group Chat (AutoGen-style) — N agents participate in a managed conversation where a speaker selection mechanism controls who talks next. All agents see the full conversation history, simulating a collaborative group discussion.

Each round, the speaker selector picks the next agent. That agent generates a response that's appended to the shared message history. The cycle continues until a termination condition is met (`termination_keyword` in an agent's response, `termination_condition` callable, or `max_rounds`). Speaker selection can be round-robin, LLM-based, or a custom callable.

**When to use**: collaborative content creation (writer + designer + reviewer), brainstorming sessions with diverse perspectives, simulated panel discussions, multi-expert consultation, and any scenario where agents should interact with each other's outputs in real time.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential runs each agent once in fixed order; GroupChat runs multiple rounds with dynamic selection |
| `DebateAgent` | Debate has exactly 2 adversarial agents; GroupChat supports N collaborative agents |
| `BlackboardAgent` | Blackboard uses shared state + controller; GroupChat uses message history + speaker selection |
| `HierarchicalAgent` | Hierarchical has a manager with subordinates; GroupChat has equal peers with a turn manager |

```python
from nono.agent import Agent, GroupChatAgent, Runner

writer = Agent(name="writer", instruction="Write marketing copy.", provider="google")
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

runner = Runner(chat)
result = runner.run("Create a tagline for an AI startup")
```

**Speaker selection strategies**:

| Strategy | Description |
|----------|-------------|
| `"round_robin"` | Cycle through agents in order (default) |
| `"llm"` | An LLM picks the best next speaker each round |
| `callable` | Custom function `(messages, agents) → agent` |

**LLM-based speaker selection**:

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

**Custom speaker selection**:

```python
def pick_reviewer_after_writer(messages, agents):
    if messages and messages[-1]["role"] == "writer":
        return next(a for a in agents if a.name == "reviewer")
    return agents[0]

chat = GroupChatAgent(
    name="creative_team",
    sub_agents=[writer, reviewer],
    speaker_selection=pick_reviewer_after_writer,
    max_rounds=6,
)
```

**Termination condition** (callable):

```python
chat = GroupChatAgent(
    name="research_team",
    sub_agents=[researcher, analyst],
    termination_condition=lambda msgs: len(msgs) >= 10,
    max_rounds=20,
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Participating agents |
| `speaker_selection` | `"round_robin"` | Strategy: `"round_robin"`, `"llm"`, or `Callable` |
| `max_rounds` | `6` | Maximum conversation rounds |
| `termination_keyword` | `None` | Keyword that ends the chat when detected in a response |
| `termination_condition` | `None` | `Callable[[list], bool]` — custom termination check |
| `provider` / `model` | `None` | Required when `speaker_selection="llm"` |

### A17 — HierarchicalAgent

**Pattern**: Hierarchical Delegation — inspired by corporate org charts and military command structures. An LLM manager sits at the top, delegates to department heads (which may themselves be orchestration agents like `SequentialAgent` or `ParallelAgent`), evaluates results, and synthesises a final answer.

This enables **multi-level** orchestration trees: the CTO delegates to the backend team (a `SequentialAgent` with architect + developer) and the QA team (a single agent). Sub-agents can be of any type, creating arbitrarily deep hierarchies.

**When to use**: complex projects that naturally decompose into departments/teams, multi-disciplinary tasks where sub-teams have their own internal workflows, organisational simulations, and any scenario where a manager delegates to teams rather than individuals.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SupervisorAgent` | Supervisor has a flat worker pool; Hierarchical supports nested sub-teams |
| `RouterAgent` | Router delegates once; Hierarchical evaluates results and can re-delegate (multi-iteration) |
| `PlannerAgent` | Planner creates a dependency graph; Hierarchical uses an LLM manager for ad hoc delegation |
| `GroupChatAgent` | GroupChat has equal peers; Hierarchical has a clear top-down command structure |

```python
from nono.agent import Agent, SequentialAgent, HierarchicalAgent, Runner

architect = Agent(name="architect", instruction="Design the system.", provider="google")
developer = Agent(name="developer", instruction="Implement the code.", provider="google")

backend = SequentialAgent(
    name="backend_team",
    description="Backend pipeline.",
    sub_agents=[architect, developer],
)
qa = Agent(name="qa", description="QA.", instruction="Review code.", provider="google")

cto = HierarchicalAgent(
    name="cto",
    provider="google",
    sub_agents=[backend, qa],
    max_iterations=3,
)

result = Runner(cto).run("Build a REST API")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Department heads / sub-teams (can be orchestration agents) |
| `provider` | (required) | AI provider for the manager LLM |
| `max_iterations` | `3` | Maximum delegation rounds |

### A18 — GuardrailAgent

**Pattern**: Guard / Validator Wrapper — a safety pattern that wraps a main agent with pre-execution and/or post-execution validators. If the post-validator rejects the output (via a keyword), the main agent automatically retries with awareness of the rejection.

This is the **safety-first** orchestration pattern: the main agent produces output, the validator checks for safety, compliance, format adherence, or toxicity, and rejection triggers re-generation. Unlike `ProducerReviewerAgent` (which iterates for quality), GuardrailAgent focuses on **policy compliance and safety**.

**When to use**: content moderation (reject toxic/offensive output), format validation (ensure JSON output is valid), compliance checking (PII detection, legal disclaimers), and any scenario where agent output must pass a safety gate before delivery.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ProducerReviewerAgent` | ProducerReviewer iterates for *quality*; GuardrailAgent validates for *safety/compliance* |
| `CoVeAgent` | CoVe validates *factual accuracy*; GuardrailAgent validates *format/safety/policy* |
| `CircuitBreakerAgent` | CircuitBreaker handles provider failures; GuardrailAgent handles content failures |

```python
from nono.agent import Agent, GuardrailAgent, Runner

writer = Agent(name="writer", instruction="Write marketing copy.", provider="google")
checker = Agent(name="checker", instruction="Reply REJECTED if toxic, else APPROVED.", provider="google")

safe = GuardrailAgent(
    name="safe_writer",
    main_agent=writer,
    post_validator=checker,
    rejection_keyword="REJECTED",
    max_retries=2,
)

result = Runner(safe).run("Write a tagline")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `main_agent` | (required) | Agent whose output is validated |
| `post_validator` | (required) | Agent that checks the output |
| `rejection_keyword` | `"REJECTED"` | Keyword that triggers re-generation |
| `max_retries` | `2` | Maximum re-generation attempts |

### A19 — BestOfNAgent

**Pattern**: Best-of-N Sampling — a generation strategy from language model decoding where the same prompt is run N times in parallel and the best result (by a scoring function) is selected. This exploits the stochastic nature of LLM responses: with temperature > 0, each generation may vary significantly in quality.

All N runs execute concurrently (thread pool in sync, coroutines in async). Each response is scored by `score_fn`, and the highest-scoring response is returned. The `result_key` option stores the full scoring breakdown for analysis.

**When to use**: creative tasks where quality varies (headlines, taglines, summaries), code generation where multiple attempts have different correctness rates, and any scenario where N cheap attempts are better than one expensive attempt.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting picks by *frequency*; BestOfN picks by *score* |
| `TournamentAgent` | Tournament uses pairwise judging; BestOfN uses independent scoring |
| `SpeculativeAgent` | Speculative early-exits on first good result; BestOfN waits for all N |
| `EnsembleAgent` | Ensemble combines all outputs; BestOfN picks the single best |

```python
from nono.agent import Agent, BestOfNAgent, Runner

writer = Agent(name="writer", instruction="Write a headline.", provider="google")

best = BestOfNAgent(
    name="best_writer",
    agent=writer,
    n=3,
    score_fn=lambda r: float(len(r)),
    result_key="scoring",
)

result = Runner(best).run("Headline for AI conference")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent to run N times |
| `n` | `3` | Number of parallel generations |
| `score_fn` | (required) | `Callable[[str], float]` — scores each response |
| `result_key` | `None` | Optional state key to store all scores |

### A20 — BatchAgent

**Pattern**: Batch Processing with Concurrency Control — processes a list of items through a single agent with configurable parallelism. Each item is formatted via a template and sent to the agent independently.

This is the **data-parallel** pattern: instead of fan-out to different agents (like `ParallelAgent`), BatchAgent clones the same agent for each item in the list and runs them concurrently up to `max_workers`. Results are collected into `session.state[result_key]` as an ordered list.

**When to use**: classifying a list of texts, generating descriptions for a product catalogue, translating a batch of strings, summarising multiple documents, and any task where the same agent must process N independent items.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ParallelAgent` | Parallel runs *different* agents on the same input; BatchAgent runs the *same* agent on different items |
| `DynamicFanOutAgent` | DynamicFanOut lets an LLM decompose the items; BatchAgent takes an explicit item list |
| `MapReduceAgent` | MapReduce has different mapper agents + a reducer; BatchAgent has one agent + no reducer |

```python
from nono.agent import Agent, BatchAgent, Runner

classifier = Agent(name="classifier", instruction="Classify sentiment.", provider="google")

batch = BatchAgent(
    name="batch_classify",
    agent=classifier,
    items=["I love this!", "Terrible.", "It's okay."],
    template="Classify: {item}",
    max_workers=3,
    result_key="classifications",
)

result = Runner(batch).run("Classify all items")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent to process each item |
| `items` | (required) | `list[str]` — items to process |
| `template` | `"{item}"` | Template string with `{item}` placeholder |
| `max_workers` | `3` | Maximum concurrent executions |
| `result_key` | `None` | State key to store ordered results |

### A21 — CascadeAgent

**Pattern**: Progressive Cascade / Cost-Aware Quality Escalation — tries agents in order from cheapest/fastest to most expensive/capable, stopping at the first one that meets a quality threshold.

Unlike `EscalationAgent` (which uses keyword-based failure detection), CascadeAgent uses a **scoring function** to objectively measure response quality. Each agent's output is scored by `score_fn`; if the score meets `threshold`, the cascade stops. If not, the next more-capable agent is tried. This enables fine-grained cost-quality tradeoffs.

**When to use**: multi-tier LLM deployments where you want to minimise cost (use Flash for easy queries, Pro for hard ones), quality-sensitive applications where a measurable metric exists, and any scenario where "cheap enough" is preferred over "always the best".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `EscalationAgent` | Escalation uses keyword detection; Cascade uses numeric quality scoring |
| `SpeculativeAgent` | Speculative races all in parallel; Cascade tries sequentially (saves cost) |
| `BestOfNAgent` | BestOfN runs the same agent N times; Cascade runs different agents in quality order |
| `CircuitBreakerAgent` | CircuitBreaker tracks historical failures; Cascade evaluates per-request quality |

```python
from nono.agent import Agent, CascadeAgent, Runner

flash = Agent(name="flash", instruction="Answer.", provider="google", model="gemini-3-flash-preview")
pro = Agent(name="pro", instruction="Answer thoroughly.", provider="google", model="gemini-2.5-pro-preview-06-05")

cascade = CascadeAgent(
    name="smart",
    sub_agents=[flash, pro],
    score_fn=lambda r: 1.0 if len(r) > 200 else 0.3,
    threshold=0.8,
)

result = Runner(cascade).run("Explain quantum entanglement")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents in cost order (cheapest first) |
| `score_fn` | (required) | `Callable[[str], float]` — scores each response 0.0–1.0 |
| `threshold` | `0.8` | Quality threshold — stop cascade when met |

### A22 — TreeOfThoughtsAgent

**Pattern**: Tree of Thoughts (ToT) — systematic exploration of reasoning paths via breadth-first search with beam pruning (Yao et al., 2023).

At each depth level the agent generates `n_branches` candidate continuations, scores every one with `evaluate_fn`, keeps only the `beam_width` best (beam pruning), and recurses deeper. This emulates how humans explore and backtrack through a problem space, unlike Chain-of-Thought which follows a single linear path.

**When to use**: problems that benefit from exploring multiple solution paths before committing — strategic planning, creative design, multi-step reasoning where the first idea isn't always the best.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `MonteCarloAgent` | ToT uses exhaustive BFS; MCTS uses stochastic sampling with UCT |
| `GraphOfThoughtsAgent` | ToT is a tree (no merges); GoT is a DAG (thoughts can aggregate) |
| `BestOfNAgent` | BestOfN is flat (N independent runs); ToT is hierarchical (explores depth) |

```python
from nono.agent import Agent, TreeOfThoughtsAgent, Runner

thinker = Agent(name="thinker", instruction="Propose a solution.", provider="google")

tot = TreeOfThoughtsAgent(
    name="tot",
    agent=thinker,
    evaluate_fn=lambda r: 1.0 if len(r) > 100 else 0.3,
    n_branches=3,
    beam_width=2,
    max_depth=3,
    result_key="tot_result",
)

result = Runner(tot).run("Design a caching strategy for a real-time API")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `n_branches` | `3` | Candidates generated per depth level |
| `beam_width` | `2` | Top candidates kept per level (beam pruning) |
| `max_depth` | `3` | Maximum tree depth |
| `evaluate_fn` | (required) | `Callable[[str], float]` — scores each candidate 0.0–1.0 |
| `result_key` | `None` | Optional state key to store the full exploration tree |

### A23 — PlannerAgent

**Pattern**: Plan-and-Execute — the classic two-phase approach where an LLM generates a dependency-aware execution plan *before* any work begins, then a deterministic executor runs the plan respecting the dependency graph.

The planner LLM receives the user request and the list of available sub-agents with their descriptions. It produces a JSON plan: an ordered list of steps, each mapping to a sub-agent, with explicit dependency edges. The executor then runs independent steps in parallel and dependent steps sequentially, maximising throughput while respecting constraints.

**When to use**: complex tasks where the decomposition itself is non-trivial and benefits from LLM reasoning — project planning, multi-stage research, report generation with dependencies between sections.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `AdaptivePlannerAgent` | Planner creates the plan once; Adaptive re-plans after every step |
| `SequentialAgent` | Sequential runs agents in fixed order; Planner lets the LLM decide order and parallelism |
| `SubQuestionAgent` | SubQuestion decomposes a *question*; Planner decomposes a *task* into agent assignments |

```python
from nono.agent import Agent, PlannerAgent, Runner

researcher = Agent(name="researcher", instruction="Research the topic.", provider="google")
writer = Agent(name="writer", instruction="Write based on research.", provider="google")
reviewer = Agent(name="reviewer", instruction="Review and polish.", provider="google")

planner = PlannerAgent(
    name="planner",
    sub_agents=[researcher, writer, reviewer],
    model="gemini-3-flash-preview",
    provider="google",
    result_key="plan_result",
)

result = Runner(planner).run("Write a technical blog post about WebAssembly")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `model` | (required) | LLM model for plan generation |
| `provider` | (required) | AI provider for the planning LLM |
| `sub_agents` | (required) | Available agents the planner can assign |
| `result_key` | `None` | Optional state key to store the plan + execution results |

### A24 — SubQuestionAgent

**Pattern**: Question Decomposition — a RAG-inspired technique that breaks a complex question into independent sub-questions, answers each one in isolation, then synthesises the partial answers into a coherent final response.

The pipeline has three phases: (1) the **decomposer** LLM analyses the original question and generates N sub-questions, (2) the **worker** agent answers each sub-question independently (in parallel), (3) the **synthesizer** agent receives all sub-answers and produces a unified response. This prevents the "lost-in-the-middle" problem where LLMs struggle with multi-part questions.

**When to use**: complex multi-faceted questions, comparative analysis, questions that require facts from different domains, and any query where a single prompt would exceed the model's "attention budget".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `PlannerAgent` | Planner decomposes *tasks* into agent assignments; SubQuestion decomposes *questions* into sub-queries |
| `MapReduceAgent` | MapReduce fans out the *same* question to multiple agents; SubQuestion fans out *different* sub-questions |
| `DynamicFanOutAgent` | DynamicFanOut uses an LLM to decompose tasks; SubQuestion is specialised for question decomposition |

```python
from nono.agent import Agent, SubQuestionAgent, Runner

expert = Agent(name="expert", instruction="Answer factual questions.", provider="google")
synth = Agent(name="synthesiser", instruction="Combine answers into a coherent response.", provider="google")

sq = SubQuestionAgent(
    name="decomposer",
    decomposer_agent=expert,
    worker_agent=expert,
    synthesizer_agent=synth,
    model="gemini-3-flash-preview",
    provider="google",
    result_key="sq_result",
)

result = Runner(sq).run("Compare the economic impacts of AI in healthcare vs education")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `decomposer_agent` | (required) | Agent that generates sub-questions |
| `worker_agent` | (required) | Agent that answers each sub-question |
| `synthesizer_agent` | (required) | Agent that merges partial answers |
| `model` / `provider` | (required) | LLM for the decomposition step |
| `result_key` | `None` | Optional state key to store sub-answers + final |

### A25 — ContextFilterAgent

**Pattern**: Context Window Management — a middleware agent that filters, transforms, or truncates the event history before delegating to sub-agents, ensuring each receives only the relevant context.

As conversations grow, the accumulated event history can overwhelm sub-agents with irrelevant information, exceed token limits, or leak sensitive data between stages. `ContextFilterAgent` sits between the orchestrator and the workers, applying a custom `filter_fn` that receives the full event list and the target agent, and returns a trimmed subset. This enables privacy isolation (agent A never sees agent B's data), noise reduction (skip `STATE_UPDATE` events), and context compression (keep only the last N exchanges).

**When to use**: long multi-turn conversations where context grows unbounded, pipelines with privacy constraints between agents, and any scenario where sub-agents perform better with focused context rather than the full history.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `MemoryConsolidationAgent` | MemoryConsolidation *summarises* old events; ContextFilter *removes* them |
| `SequentialAgent` | Sequential passes full history; ContextFilter tailors history per agent |

```python
from nono.agent import Agent, ContextFilterAgent, Runner

writer = Agent(name="writer", instruction="Write content.", provider="google")
reviewer = Agent(name="reviewer", instruction="Review content.", provider="google")

filtered = ContextFilterAgent(
    name="ctx_filter",
    sub_agents=[writer, reviewer],
    filter_fn=lambda events, agent: [
        e for e in events if e.event_type.name != "STATE_UPDATE"
    ],
)

result = Runner(filtered).run("Write and review an article on AI safety")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `filter_fn` | (required) | `Callable[[list[Event], BaseAgent], list[Event]]` — per-agent event filter |
| `sub_agents` | (required) | Agents to run with filtered context |

### A26 — ReflexionAgent

**Pattern**: Reflexion (Shinn et al., 2023) — iterative self-improvement through verbal reinforcement learning. The agent generates an output, an evaluator scores it, and if the score falls below the threshold the agent receives both the evaluation *and* a self-reflection of what went wrong, then retries with accumulated lessons.

Unlike simple retry-until-approved loops (`ProducerReviewerAgent`), Reflexion maintains a **persistent memory of failures across attempts** — each retry sees the original prompt plus all previous attempts, evaluations, and reflections. This lets the agent learn from mistakes within a single session.

The pipeline per attempt: `generate → evaluate → (if below threshold) reflect → retry with reflection memory`.

**When to use**: code generation, mathematical proofs, logic puzzles, and any task where the agent's first attempt is often close but not quite right and where understanding *why* it failed helps the next attempt.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ProducerReviewerAgent` | ProducerReviewer has no reflection memory; each retry is independent |
| `LoopAgent` | LoopAgent repeats blindly; Reflexion accumulates failure analysis |
| `CompilerAgent` | Compiler optimises the *instruction*; Reflexion optimises the *output* |

```python
from nono.agent import Agent, ReflexionAgent, Runner

coder = Agent(name="coder", instruction="Write Python code.", provider="google")
critic = Agent(name="critic", instruction="Evaluate code. Rate 0-1.", provider="google")

reflexion = ReflexionAgent(
    name="reflexion",
    agent=coder,
    evaluator=critic,
    max_attempts=3,
    threshold=0.8,
    result_key="reflexion_result",
)

result = Runner(reflexion).run("Write a binary search implementation")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | The agent that generates outputs |
| `evaluator` | (required) | Agent that scores each attempt (returns 0.0–1.0) |
| `max_attempts` | `3` | Maximum retries before returning best attempt |
| `threshold` | `0.8` | Score threshold — stop early if met |
| `result_key` | `None` | Optional state key to store attempt history |

### A27 — SpeculativeAgent

**Pattern**: Speculative Execution — inspired by speculative decoding in LLM inference and branch prediction in CPUs. Multiple agents race on the same task in parallel; the first result that exceeds a confidence threshold is accepted and the remaining computations are discarded.

This is a **latency-optimisation** pattern: instead of trying agents sequentially (like `EscalationAgent` or `CascadeAgent`), you run all of them concurrently and early-exit as soon as one delivers a high-confidence answer. You trade compute cost for wall-clock time. A cheap-but-fast agent (e.g. Groq/Cerebras) may win on simple queries, while a powerful-but-slow agent (e.g. GPT-4) wins on hard ones.

**When to use**: latency-sensitive applications where multiple providers are available and you can afford the extra compute, real-time systems where response time matters more than cost, and fallback scenarios where speed and quality must be balanced dynamically.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `EscalationAgent` | Escalation tries agents *sequentially*; Speculative runs them *in parallel* |
| `CascadeAgent` | Cascade is sequential with cost escalation; Speculative races all at once |
| `BestOfNAgent` | BestOfN waits for *all* results; Speculative early-exits on the *first* good one |
| `ParallelAgent` | Parallel runs all and collects all; Speculative discards losers |

```python
from nono.agent import Agent, SpeculativeAgent, Runner

fast = Agent(name="fast", instruction="Quick answer.", provider="groq")
smart = Agent(name="smart", instruction="Thorough answer.", provider="google")

spec = SpeculativeAgent(
    name="racer",
    sub_agents=[fast, smart],
    evaluator_fn=lambda r: 0.9 if len(r) > 50 else 0.2,
    min_confidence=0.7,
    result_key="spec_result",
)

result = Runner(spec).run("What is gradient descent?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents to race in parallel |
| `evaluator_fn` | (required) | `Callable[[str], float]` — scores each response 0.0–1.0 |
| `min_confidence` | `0.7` | Threshold to accept an early result |
| `result_key` | `None` | Optional state key to store race details |

### A28 — CircuitBreakerAgent

**Pattern**: Circuit Breaker (Nygard, *Release It!*, 2007) — a resilience pattern from distributed systems that prevents cascading failures by monitoring error rates and short-circuiting to a fallback when a service becomes unreliable.

The agent tracks consecutive failures. It operates in three states:

| State | Behaviour |
|---|---|
| **Closed** (normal) | Requests pass through to the primary agent. Each failure increments the counter. |
| **Open** (tripped) | After `failure_threshold` consecutive failures, *all* requests go directly to the fallback agent without contacting the primary. |
| **Half-Open** (probe) | After `recovery_timeout` seconds, one request is allowed through to the primary. If it succeeds, the circuit closes; if it fails, it reopens. |

This protects against situations where an LLM provider goes down, hits rate limits, or returns garbage — instead of retrying indefinitely and burning tokens, the circuit breaks and the fallback provides a degraded-but-functional response.

**When to use**: production systems with multiple LLM providers, any scenario where API reliability is uncertain, cost protection against runaway retries, and graceful degradation requirements.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `EscalationAgent` | Escalation tries agents in order per request; CircuitBreaker remembers failure *history across requests* |
| `CascadeAgent` | Cascade always starts from the cheapest; CircuitBreaker stays on fallback until recovery |
| `TimeoutAgent` | TimeoutAgent guards a single call; CircuitBreaker guards across multiple calls |

```python
from nono.agent import Agent, CircuitBreakerAgent, Runner

primary = Agent(name="primary", instruction="Answer.", provider="google")
fallback = Agent(name="fallback", instruction="Give a safe default answer.", provider="google")

cb = CircuitBreakerAgent(
    name="breaker",
    agent=primary,
    fallback_agent=fallback,
    failure_threshold=3,
    recovery_timeout=60.0,
    result_key="cb_result",
)

result = Runner(cb).run("Answer this question")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Primary agent |
| `fallback_agent` | (required) | Agent used when circuit is open |
| `failure_threshold` | `3` | Consecutive failures before tripping |
| `recovery_timeout` | `60.0` | Seconds before attempting half-open probe |
| `result_key` | `None` | Optional state key to store circuit state |

### A29 — TournamentAgent

**Pattern**: Tournament Selection — inspired by bracket-elimination tournaments in competitive sports and selection pressure in genetic algorithms.

N agents generate responses in parallel (first round). They are paired into matches; an LLM judge evaluates each pair and selects the winner. Losers are eliminated. Winners advance to the next round. This continues until one champion remains. For odd-numbered brackets, one agent gets a bye (auto-advances).

This is more robust than `VotingAgent` (simple majority) because the judge does **pairwise comparison** with full reasoning, which produces higher-quality rankings. It's also more efficient than `BestOfNAgent` when you need a judge: instead of scoring N responses independently, only `N-1` comparisons are needed.

**When to use**: creative competitions (best tagline, headline, design), quality selection where relative ranking matters more than absolute scoring, and multi-model shootouts.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting uses majority vote (no judge); Tournament uses pairwise LLM judging |
| `BestOfNAgent` | BestOfN scores independently (needs a metric function); Tournament compares head-to-head |
| `ConsensusAgent` | Consensus synthesises all answers; Tournament eliminates and picks one winner |

```python
from nono.agent import Agent, TournamentAgent, Runner

a = Agent(name="a", instruction="Generate a tagline.", provider="google")
b = Agent(name="b", instruction="Generate a tagline.", provider="google")
c = Agent(name="c", instruction="Generate a tagline.", provider="google")
d = Agent(name="d", instruction="Generate a tagline.", provider="google")
judge = Agent(name="judge", instruction="Pick the better tagline.", provider="google")

tournament = TournamentAgent(
    name="bracket",
    sub_agents=[a, b, c, d],
    judge_agent=judge,
    result_key="tournament_result",
)

result = Runner(tournament).run("Best tagline for an AI startup")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Competing agents (minimum 2) |
| `judge_agent` | (required) | LLM judge for pairwise comparisons |
| `result_key` | `None` | Optional state key to store bracket + results |

### A30 — ShadowAgent

**Pattern**: Shadow / Dark Launch — a deployment pattern from site reliability engineering where a new (shadow) model runs alongside the stable production model, but its output is **never shown to users**. Only the stable output is returned; the shadow output is logged for offline comparison.

Both agents run in parallel on the same input. The stable agent's response is always returned to the caller. The `diff_logger` callback receives both responses, enabling automated comparison (BLEU scores, length ratios, semantic similarity) or simple side-by-side logging. This is the safest way to evaluate a new model in production without any user-facing risk.

**When to use**: A/B testing a new LLM model before promoting it, comparing providers (e.g. Gemini vs GPT), regression testing after fine-tuning, and any scenario where you want production-realistic evaluation without risk.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `BestOfNAgent` | BestOfN picks the best output to return; Shadow always returns the stable one |
| `EnsembleAgent` | Ensemble merges outputs; Shadow keeps them separate (one live, one logged) |
| `ParallelAgent` | Parallel returns all results; Shadow returns only the stable result |

```python
from nono.agent import Agent, ShadowAgent, Runner

stable = Agent(name="v1", instruction="Answer.", provider="google", model="gemini-3-flash-preview")
shadow = Agent(name="v2", instruction="Answer.", provider="google", model="gemini-2.5-pro-preview-06-05")

ab = ShadowAgent(
    name="shadow_test",
    stable_agent=stable,
    shadow_agent=shadow,
    diff_logger=lambda s, sh: print(f"Stable: {s[:80]}... Shadow: {sh[:80]}..."),
    result_key="shadow_result",
)

result = Runner(ab).run("Explain transformers in deep learning")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `stable_agent` | (required) | Production agent (its output is returned) |
| `shadow_agent` | (required) | Candidate agent (its output is logged only) |
| `diff_logger` | `None` | `Callable[[str, str], None]` — receives (stable_output, shadow_output) |
| `result_key` | `None` | Optional state key to store both outputs |

### A31 — CompilerAgent

**Pattern**: Prompt Compilation (DSPy-style) — meta-optimisation of agent instructions through iterative evaluation against a dataset of examples. Instead of hand-tuning prompts, the LLM itself rewrites the target agent's `instruction` to maximise a metric score.

The compilation loop: (1) run the target agent on each example, (2) compute `metric_fn(output, expected)` for each, (3) feed the scores + examples to the compiler LLM, (4) the compiler LLM proposes a new instruction, (5) repeat. After `max_iterations`, the best-scoring instruction is applied permanently to the target agent.

This is a form of **automated prompt engineering** — the LLM learns what works by seeing its own failures, similar to how DSPy compiles chain-of-thought prompts from demonstrations.

**When to use**: systematic prompt optimisation when you have a dataset of input/expected pairs, replacing manual prompt iteration, A/B testing different instruction strategies, and bootstrapping agents from examples.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ReflexionAgent` | Reflexion optimises a single *output*; Compiler optimises the *instruction* across a dataset |
| `ProducerReviewerAgent` | ProducerReviewer improves one response; Compiler improves the agent itself |
| `LoopAgent` | LoopAgent repeats execution; Compiler repeats *meta-optimisation* of the prompt |

```python
from nono.agent import Agent, CompilerAgent, Runner

target = Agent(name="writer", instruction="Write concisely.", provider="google")

compiler = CompilerAgent(
    name="compiler",
    target_agent=target,
    examples=[
        {"input": "Explain AI", "expected": "short clear answer"},
        {"input": "Explain ML", "expected": "short clear answer"},
    ],
    metric_fn=lambda output, expected: 1.0 if len(output) < 200 else 0.3,
    model="gemini-3-flash-preview",
    provider="google",
    max_iterations=3,
    result_key="compiler_result",
)

result = Runner(compiler).run("Optimise the writer agent")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `target_agent` | (required) | Agent whose instruction will be optimised |
| `examples` | (required) | `list[dict]` with `input` and `expected` keys |
| `metric_fn` | (required) | `Callable[[str, str], float]` — scores output vs expected |
| `model` / `provider` | (required) | LLM for the meta-optimisation step |
| `max_iterations` | `3` | Compilation rounds |
| `result_key` | `None` | Optional state key to store iteration history |

### A32 — CheckpointableAgent

**Pattern**: Checkpoint / Resume — a fault-tolerance pattern that persists intermediate state after each completed sub-agent, enabling the pipeline to resume from the last successful step after a crash or interruption.

Internally, `CheckpointableAgent` wraps a `SequentialAgent` with state tracking. After each sub-agent completes, the current step index and accumulated session state are saved to `session.state[checkpoint_key]`. On subsequent invocations with the same session, the agent reads the checkpoint and **skips already-completed steps**, resuming from where it left off.

This is the **agent-level equivalent** of `Workflow.enable_checkpoints()`. Use this when you need checkpoint/resume in agent pipelines (which don't have Workflow's built-in checkpointing).

**When to use**: long-running agent pipelines where each step involves expensive LLM calls, batch processing that may be interrupted, pipelines running in unreliable environments (spot instances, preemptible VMs), and any multi-step process where re-running completed steps wastes time and cost.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential has no persistence; CheckpointableAgent survives crashes |
| `Workflow.enable_checkpoints()` | Workflow checkpoints use file-based JSON; CheckpointableAgent uses session state |
| `SagaAgent` | Saga handles *rollback* on failure; Checkpointable handles *resume* after failure |

```python
from nono.agent import Agent, CheckpointableAgent, Runner

step1 = Agent(name="fetch", instruction="Fetch data.", provider="google")
step2 = Agent(name="process", instruction="Process data.", provider="google")
step3 = Agent(name="report", instruction="Generate report.", provider="google")

pipeline = CheckpointableAgent(
    name="pipeline",
    sub_agents=[step1, step2, step3],
    checkpoint_key="cp_state",
    result_key="pipeline_result",
)

result = Runner(pipeline).run("Run the data pipeline")
# On failure, re-run with the same session to resume from checkpoint
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Sequential pipeline of agents |
| `checkpoint_key` | `"checkpoint"` | State key where progress is persisted |
| `result_key` | `None` | Optional state key to store final results |

### A33 — DynamicFanOutAgent

**Pattern**: Dynamic Map-Reduce — an LLM-powered variant of `MapReduceAgent` where the decomposition of work items happens **at runtime** instead of at design time. The planner LLM analyses the user request, generates a list of work items, distributes them to parallel workers, and a reducer combines the results.

Unlike `MapReduceAgent` (where you pre-define the sub-agents and their messages), `DynamicFanOutAgent` lets the LLM decide *how many* items to create and *what* each one should contain. This makes it ideal for tasks where the decomposition itself requires intelligence.

The pipeline: (1) LLM planner generates up to `max_items` work items as JSON, (2) the `worker_agent` is cloned N times and runs in parallel, each receiving one work item, (3) the `reducer_agent` receives all worker outputs and synthesises the final response.

**When to use**: research tasks where the number of sub-topics is unknown upfront, content generation that requires dynamic decomposition, data gathering from variable numbers of sources, and any Map-Reduce scenario where the map keys emerge from the input.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `MapReduceAgent` | MapReduce has fixed sub-agents; DynamicFanOut creates work items at runtime |
| `PlannerAgent` | Planner assigns tasks to *different* agents; DynamicFanOut clones the *same* worker |
| `SubQuestionAgent` | SubQuestion decomposes questions; DynamicFanOut decomposes generic tasks |
| `ParallelAgent` | Parallel runs pre-defined agents; DynamicFanOut determines the fan-out dynamically |

```python
from nono.agent import Agent, DynamicFanOutAgent, Runner

worker = Agent(name="worker", instruction="Complete the assigned task.", provider="google")
reducer = Agent(name="reducer", instruction="Combine all results into one.", provider="google")

fanout = DynamicFanOutAgent(
    name="dynamic",
    worker_agent=worker,
    reducer_agent=reducer,
    model="gemini-3-flash-preview",
    provider="google",
    max_items=5,
    result_key="fanout_result",
)

result = Runner(fanout).run("Research the top 5 AI trends for 2026")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `worker_agent` | (required) | Agent cloned for each work item |
| `reducer_agent` | (required) | Agent that combines all worker outputs |
| `model` / `provider` | (required) | LLM for the decomposition/planning step |
| `max_items` | `5` | Maximum work items the planner can generate |
| `result_key` | `None` | Optional state key to store items + results |

### A34 — SwarmAgent

**Pattern**: Agent Swarm (OpenAI Swarm-style) — a decentralised handoff network where agents transfer control to each other through session state, maintaining shared context variables across the swarm.

Unlike `HandoffAgent` (which uses keyword detection in responses), `SwarmAgent` transfers control through **session state updates** — a more structured and reliable mechanism. Each agent can read and write to `context_variables` (global shared state), and the handoff decision is mediated by the framework rather than by parsing text output.

The execution loop: (1) the `initial_agent` runs first, (2) after each agent completes, the framework checks session state for a handoff directive, (3) if a handoff is requested, control transfers to the target agent with the updated context, (4) the loop continues until no handoff is requested or `max_handoffs` is reached.

**When to use**: customer support systems with triage → specialist routing, multi-stage workflows where agents need shared stateful context (e.g. customer tier, session ID), and any scenario modelled as a state machine with context-dependent transitions.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `HandoffAgent` | HandoffAgent parses keywords in text; SwarmAgent uses session state for handoffs |
| `RouterAgent` | RouterAgent is a centralised decision; SwarmAgent is decentralised (agents decide themselves) |
| `SupervisorAgent` | Supervisor delegates and evaluates; SwarmAgent is peer-to-peer with shared state |

```python
from nono.agent import Agent, SwarmAgent, Runner

triage = Agent(name="triage", instruction="Route to the right specialist.", provider="google")
billing = Agent(name="billing", instruction="Handle billing queries.", provider="google")
tech = Agent(name="tech", instruction="Handle technical issues.", provider="google")

swarm = SwarmAgent(
    name="helpdesk",
    sub_agents=[triage, billing, tech],
    initial_agent="triage",
    context_variables={"customer_tier": "premium"},
    max_handoffs=5,
    result_key="swarm_result",
)

result = Runner(swarm).run("I can't access my premium features")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | All agents in the swarm |
| `initial_agent` | (required) | Name of the first agent to run |
| `context_variables` | `{}` | Shared state dict accessible by all agents |
| `max_handoffs` | `5` | Maximum agent-to-agent transfers |
| `result_key` | `None` | Optional state key to store handoff trace |

### A35 — MemoryConsolidationAgent

**Pattern**: Memory Consolidation — inspired by how human memory compresses episodic memories into semantic summaries during sleep. When the event history grows beyond a configurable threshold, older events are **summarised** by a dedicated summarizer agent and replaced with a compact summary event, keeping the context window manageable.

This solves the "infinite conversation" problem: long-running agents accumulate thousands of events that eventually exceed the LLM's context window. Instead of hard-truncating (losing information), MemoryConsolidation produces a semantic summary that preserves key facts while dramatically reducing token count. The `keep_recent` parameter ensures the most recent events remain verbatim for immediate context.

The consolidation trigger: when `len(events) > event_threshold`, consolidation fires automatically before running the main agent.

**When to use**: long-running conversational agents, customer support bots that handle extended sessions, multi-step pipelines with many intermediate events, and any agent that runs for many turns.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ContextFilterAgent` | ContextFilter *removes* events; MemoryConsolidation *summarises* them (no information loss) |
| `SequentialAgent` | Sequential has no memory management; MemoryConsolidation actively compresses history |

```python
from nono.agent import Agent, MemoryConsolidationAgent, Runner

main = Agent(name="assistant", instruction="Help the user.", provider="google")
summariser = Agent(name="summariser", instruction="Summarise the conversation so far.", provider="google")

mc = MemoryConsolidationAgent(
    name="smart_assistant",
    main_agent=main,
    summarizer_agent=summariser,
    event_threshold=20,
    keep_recent=5,
    result_key="mc_result",
)

result = Runner(mc).run("Continue our long conversation")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `main_agent` | (required) | The primary agent to run |
| `summarizer_agent` | (required) | Agent that produces the summary |
| `event_threshold` | `20` | Event count that triggers consolidation |
| `keep_recent` | `5` | Number of recent events to keep verbatim |
| `result_key` | `None` | Optional state key to store consolidation metadata |

### A36 — PriorityQueueAgent

**Pattern**: Priority Queue Scheduling — a scheduling pattern from operating systems where tasks are grouped by priority class and executed in strict priority order. Within each priority level, agents run in parallel; across levels, execution is strictly sequential (all priority-0 agents must complete before priority-1 begins).

This models real-world scenarios where some operations are critical (security checks, input validation) and must complete before less urgent work (analytics, logging) begins. A priority map assigns each agent a numeric priority (lower = higher priority).

**When to use**: multi-agent pipelines with mixed criticality levels, systems where security/validation must run before business logic, batch processing with tiered urgency, and request processing where critical agents must not be blocked by background tasks.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential runs one-by-one in fixed order; PriorityQueue parallelises within priority levels |
| `ParallelAgent` | Parallel runs all at once; PriorityQueue enforces ordering between priority groups |
| `EscalationAgent` | Escalation stops at first success; PriorityQueue runs all agents, just in priority order |

```python
from nono.agent import Agent, PriorityQueueAgent, Runner

critical = Agent(name="security", instruction="Check security.", provider="google")
normal = Agent(name="analysis", instruction="Analyse data.", provider="google")
background = Agent(name="logging", instruction="Log the interaction.", provider="google")

pq = PriorityQueueAgent(
    name="processor",
    sub_agents=[critical, normal, background],
    priority_map={"security": 0, "analysis": 1, "logging": 2},
    result_key="pq_result",
)

result = Runner(pq).run("Process this request")
```

**Execution order for the example above**:

```
Round 1 (priority 0): security      ← must complete first
Round 2 (priority 1): analysis       ← waits for security
Round 3 (priority 2): logging        ← waits for analysis
```

If multiple agents share the same priority level, they run **in parallel** within that round.

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | All agents to schedule |
| `priority_map` | (required) | `dict[str, int]` — agent name → priority (0 = highest) |
| `result_key` | `None` | Optional state key to store execution order + results |

### A37 — MonteCarloAgent

**Pattern**: Monte Carlo Tree Search (MCTS) with UCT — the algorithm behind AlphaGo and AlphaZero, adapted for LLM reasoning. Instead of exhaustively exploring all paths (like `TreeOfThoughtsAgent`), MCTS uses **stochastic sampling** guided by Upper Confidence bound for Trees (UCT) to balance exploration of new ideas vs exploitation of promising ones.

The MCTS loop for each simulation: (1) **Select** a leaf node using UCT ($$UCT = \bar{X}_j + C \sqrt{\frac{\ln N}{n_j}}$$), (2) **Expand** by generating a new thought via the agent, (3) **Evaluate** with `evaluate_fn`, (4) **Backpropagate** the score up to the root. After `n_simulations` iterations, the highest-value path is returned.

This is more **sample-efficient** than ToT for large search spaces: instead of generating all `n_branches^max_depth` nodes, MCTS focuses compute on the most promising regions.

**When to use**: open-ended creative problems with large solution spaces, strategic planning where exploration-exploitation tradeoff matters, problem-solving where depth matters more than breadth, and any scenario where exhaustive BFS would be too expensive.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TreeOfThoughtsAgent` | ToT is exhaustive BFS with beam pruning; MCTS is stochastic with UCT guidance |
| `GraphOfThoughtsAgent` | GoT allows DAG merges; MCTS explores a tree with backpropagation |
| `BestOfNAgent` | BestOfN is flat sampling; MCTS is hierarchical with depth exploration |

```python
from nono.agent import Agent, MonteCarloAgent, Runner

thinker = Agent(name="thinker", instruction="Propose a creative solution.", provider="google")

mcts = MonteCarloAgent(
    name="mcts",
    agent=thinker,
    evaluate_fn=lambda r: 1.0 if "innovative" in r.lower() else 0.3,
    n_simulations=20,
    max_depth=3,
    result_key="mcts_result",
)

result = Runner(mcts).run("Find the best approach to reduce API latency")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates thoughts at each node |
| `evaluate_fn` | (required) | `Callable[[str], float]` — scores each thought 0.0–1.0 |
| `n_simulations` | `20` | Total MCTS simulations (select→expand→evaluate→backprop) |
| `max_depth` | `3` | Maximum tree depth per simulation |
| `exploration_weight` | `1.414` | UCT exploration constant $C$ (√2 by default) |
| `result_key` | `None` | Optional state key to store the full MCTS tree |

### A38 — GraphOfThoughtsAgent

**Pattern**: Graph of Thoughts (Besta et al., 2023) — DAG-based thought orchestration that extends Tree of Thoughts by allowing thoughts to **merge** (aggregate) before scoring. This models how humans combine ideas from different branches into novel hybrid solutions.

The GoT pipeline operates in rounds: (1) **Generate** — the agent produces `n_branches` candidate thoughts, (2) **Aggregate** — the `aggregate_agent` receives all candidates and merges them into combined proposals, (3) **Score** — `score_fn` evaluates each result (both original and merged), (4) the best candidates survive to the next round. After `n_rounds`, the highest-scoring thought is returned.

The key innovation over ToT is the **aggregate** step: where a tree can only fork (diverge), a DAG can also merge (converge). This enables the system to find solutions that combine the strengths of different branches.

**When to use**: complex design problems where the best solution combines multiple ideas, brainstorming sessions where cross-pollination between approaches is valuable, multi-perspective synthesis, and any problem where "the answer is a combination of partial solutions".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TreeOfThoughtsAgent` | ToT is a tree (fork only); GoT is a DAG (fork + merge) |
| `MonteCarloAgent` | MCTS samples stochastically; GoT generates all branches then aggregates |
| `ConsensusAgent` | Consensus merges *agent outputs*; GoT merges *thoughts within one agent* |
| `MapReduceAgent` | MapReduce uses independent workers; GoT's branches are aware of each other via aggregation |

```python
from nono.agent import Agent, GraphOfThoughtsAgent, Runner

gen = Agent(name="gen", instruction="Propose a design idea.", provider="google")
merger = Agent(name="merger", instruction="Merge these ideas into one cohesive design.", provider="google")

got = GraphOfThoughtsAgent(
    name="got",
    agent=gen,
    aggregate_agent=merger,
    score_fn=lambda r: 1.0 if len(r) > 100 else 0.4,
    n_branches=3,
    n_rounds=2,
    result_key="got_result",
)

result = Runner(got).run("Design a notification system for a social app")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates candidate thoughts |
| `aggregate_agent` | (required) | Agent that merges multiple thoughts into one |
| `score_fn` | (required) | `Callable[[str], float]` — scores each thought 0.0–1.0 |
| `n_branches` | `3` | Candidates generated per round |
| `n_rounds` | `2` | Number of generate→aggregate→score rounds |
| `result_key` | `None` | Optional state key to store the full DAG |

### A39 — BlackboardAgent

**Pattern**: Blackboard Architecture (Erman et al., 1980 — Hearsay-II speech recognition system) — a collaborative problem-solving pattern where multiple specialist agents ("knowledge sources") share a common data store (the "blackboard") and iteratively contribute partial solutions until the problem is solved.

The blackboard is a shared dict in session state. Each iteration, a **controller** evaluates the board state and selects the most relevant expert to contribute next. The selected expert reads the current board, performs its analysis, and writes its findings back. The cycle continues until the `termination_fn` returns `True` (problem solved) or `max_iterations` is reached.

This pattern excels at problems that require **incremental, multi-discipline reasoning** where no single agent can solve the problem alone but each can contribute domain-specific insights that build on others' work.

**When to use**: medical diagnosis (symptoms + labs + imaging converge on a diagnosis), complex troubleshooting (network + database + application logs), multi-disciplinary design reviews, and any problem requiring collaborative convergence from different expert perspectives.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `GroupChatAgent` | GroupChat uses round-robin/LLM speaker selection with messages; Blackboard uses shared state + controller |
| `ConsensusAgent` | Consensus runs all agents once then judges; Blackboard iterates until convergence |
| `SwarmAgent` | Swarm hands off control entirely; Blackboard experts contribute to shared state without taking control |
| `HierarchicalAgent` | Hierarchical delegates top-down; Blackboard converges bottom-up via shared data |

```python
from nono.agent import Agent, BlackboardAgent, Runner

symptom_agent = Agent(name="symptoms", instruction="Analyse symptoms from the board.", provider="google")
lab_agent = Agent(name="lab", instruction="Interpret lab results from the board.", provider="google")
radiology = Agent(name="radiology", instruction="Interpret imaging findings.", provider="google")

bb = BlackboardAgent(
    name="diagnosis",
    sub_agents=[symptom_agent, lab_agent, radiology],
    termination_fn=lambda board: board.get("diagnosis") is not None,
    max_iterations=6,
    result_key="bb_result",
)

result = Runner(bb).run("Patient: fatigue, joint pain, elevated CRP")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Expert agents (knowledge sources) |
| `termination_fn` | (required) | `Callable[[dict], bool]` — checks if the board is solved |
| `max_iterations` | `6` | Maximum expert consultation rounds |
| `result_key` | `None` | Optional state key to store final board state |

### A40 — MixtureOfExpertsAgent

**Pattern**: Mixture of Experts (MoE) — a neural architecture concept (Shazeer et al., 2017) adapted to agent orchestration. A **gating function** assigns relevance weights to each expert agent, the top-k experts by weight are activated, and their outputs are blended proportionally.

Unlike `RouterAgent` (which picks exactly one agent) or `VotingAgent` (which runs all and picks the majority), MoE provides a **weighted blend** of multiple expert opinions. The gating function can be a simple keyword-based heuristic, an embedding similarity measure, or even an LLM call. Only the top-k experts are executed, making it cost-efficient for large expert pools.

The pipeline: (1) the gating function receives the user message and returns weights per expert, (2) the top-k experts by weight are selected, (3) selected experts run in parallel, (4) outputs are combined using weight-proportional blending.

**When to use**: multi-domain queries that span several expertise areas ("calculate ROI and write a report"), systems with many specialist agents where only a subset is relevant per request, and scenarios where partial expertise from multiple agents produces a better answer than one expert alone.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `RouterAgent` | Router picks *one* agent; MoE runs top-k and blends |
| `VotingAgent` | Voting runs all and picks majority; MoE weights by relevance |
| `EnsembleAgent` | Ensemble runs all agents equally; MoE selects only top-k by gating |
| `ParallelAgent` | Parallel runs all without weighting; MoE gates + weights |

```python
from nono.agent import Agent, MixtureOfExpertsAgent, Runner

math = Agent(name="math", instruction="Solve math problems.", provider="google")
code = Agent(name="code", instruction="Write code.", provider="google")
writing = Agent(name="writing", instruction="Write prose.", provider="google")

moe = MixtureOfExpertsAgent(
    name="moe",
    sub_agents=[math, code, writing],
    gating_fn=lambda msg, agents: {
        "math": 0.8 if "calculate" in msg.lower() else 0.1,
        "code": 0.8 if "code" in msg.lower() else 0.1,
        "writing": 0.8 if "write" in msg.lower() else 0.1,
    },
    top_k=2,
    result_key="moe_result",
)

result = Runner(moe).run("Calculate the ROI and write a summary report")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | All expert agents |
| `gating_fn` | (required) | `Callable[[str, list], dict[str, float]]` — returns weights per agent |
| `top_k` | `2` | Number of experts to activate |
| `result_key` | `None` | Optional state key to store gating weights + outputs |

### A41 — CoVeAgent

**Pattern**: Chain-of-Verification (Dhuliawala et al., 2023) — a structured 4-phase anti-hallucination pipeline that dramatically reduces factual errors by forcing the model to **verify its own claims** before delivering the final answer.

The four phases:

| Phase | Agent | Purpose |
|---|---|---|
| 1. **Draft** | `drafter` | Generate an initial answer (may contain hallucinations) |
| 2. **Plan** | `planner` | Analyse the draft and generate verification questions targeting factual claims |
| 3. **Verify** | `verifier` | Answer each verification question independently (fresh context, no access to the draft) |
| 4. **Revise** | `reviser` | Receive the original draft + verified facts, cross-reference, and produce a corrected final answer |

The key insight is **phase 3**: by answering verification questions in isolation (without seeing the original draft), the verifier avoids confirmation bias. If the draft said "OpenAI was founded in 2014" and the verifier independently answers "2015", the reviser catches and corrects the error.

**When to use**: factual Q&A where accuracy is critical, knowledge-intensive tasks (dates, names, statistics), any customer-facing application where hallucinations carry reputational or legal risk, and generating content that will be published without human review.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ReflexionAgent` | Reflexion reflects on *quality*; CoVe verifies *factual claims* |
| `ProducerReviewerAgent` | ProducerReviewer has no structured verification; CoVe has explicit plan→verify→revise |
| `GuardrailAgent` | GuardrailAgent validates format/safety; CoVe validates factual accuracy |

```python
from nono.agent import Agent, CoVeAgent, Runner

drafter = Agent(name="drafter", instruction="Draft an answer.", provider="google")
planner = Agent(name="planner", instruction="Generate verification questions for the draft.", provider="google")
verifier = Agent(name="verifier", instruction="Answer the verification question factually.", provider="google")
reviser = Agent(name="reviser", instruction="Revise the draft using verified facts.", provider="google")

cove = CoVeAgent(
    name="verified_answer",
    drafter=drafter,
    planner=planner,
    verifier=verifier,
    reviser=reviser,
    max_questions=3,
    result_key="cove_result",
)

result = Runner(cove).run("List the founders and founding dates of the top 5 AI labs")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `drafter` | (required) | Agent that generates the initial draft |
| `planner` | (required) | Agent that generates verification questions |
| `verifier` | (required) | Agent that answers verification questions in isolation |
| `reviser` | (required) | Agent that cross-references and produces the corrected answer |
| `max_questions` | `3` | Maximum verification questions per draft |
| `result_key` | `None` | Optional state key to store all phases |

### A42 — SagaAgent

**Pattern**: Saga (Garcia-Molina & Salem, 1987) — a distributed transaction pattern that guarantees **eventual consistency** through compensating actions. Each step in the saga has a forward action and an optional compensator (undo). If step N fails, compensators execute in reverse order (N-1 → N-2 → ... → 0) to roll back all completed work.

This replaces traditional two-phase commit (2PC) in agent pipelines where atomicity isn't feasible — you can't "un-call" an LLM, but you can have a compensating agent undo the logical effect (cancel a reservation, issue a refund, revoke access).

The execution model: steps run sequentially. After each step, `failure_detector` evaluates the output. If a failure is detected, the saga enters **compensation mode** and runs compensators in reverse for every step that already completed successfully.

**When to use**: multi-step business processes where partial completion is harmful (e-commerce: reserve → charge → ship), agent pipelines that modify external systems (databases, APIs, files), and any workflow where "all-or-nothing" semantics are required but a single transaction boundary isn't available.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential has no rollback; Saga compensates on failure |
| `CheckpointableAgent` | Checkpointable *resumes* after failure; Saga *rolls back* |
| `EscalationAgent` | Escalation tries alternatives; Saga undoes completed work |
| `CircuitBreakerAgent` | CircuitBreaker prevents calls; Saga handles cleanup after failure |

```python
from nono.agent import Agent, SagaAgent, Runner

reserve = Agent(name="reserve", instruction="Reserve inventory.", provider="google")
charge = Agent(name="charge", instruction="Charge payment.", provider="google")
ship = Agent(name="ship", instruction="Ship the order.", provider="google")
release = Agent(name="release", instruction="Release reserved inventory.", provider="google")
refund = Agent(name="refund", instruction="Refund the payment.", provider="google")

saga = SagaAgent(
    name="order_saga",
    steps=[
        {"action": reserve, "compensate": release},
        {"action": charge, "compensate": refund},
        {"action": ship},
    ],
    failure_detector=lambda o: "ERROR" in o.upper(),
    result_key="saga_result",
)

result = Runner(saga).run("Process order #12345")
```

**Compensation flow on failure at step 2 (charge)**:

```
✅ reserve → ❌ charge (ERROR detected)
   ⬅️ compensate: release (undoes reserve)
   → saga returns failure with compensation log
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `steps` | (required) | `list[dict]` — each with `action` (required) and `compensate` (optional) agents |
| `failure_detector` | (required) | `Callable[[str], bool]` — returns `True` if the output indicates failure |
| `result_key` | `None` | Optional state key to store execution + compensation log |

### A43 — LoadBalancerAgent

**Pattern**: Load Balancer — a classic distributed systems pattern that distributes requests across N equivalent backends to maximise throughput, improve redundancy, and balance utilisation.

Unlike `ParallelAgent` (which runs **all** agents on every request), `LoadBalancerAgent` selects **exactly one** agent per request using a configurable strategy. This is useful when you have multiple equivalent LLM providers and want to spread the load, avoid rate limits, or rotate between them for cost balancing.

Supported strategies:

| Strategy | Description |
|---|---|
| `"round_robin"` | Cycle through agents in order (default) |
| `"random"` | Select randomly with uniform probability |
| `"least_used"` | Pick the agent with the fewest prior invocations |

**When to use**: multi-provider setups where several LLMs can handle the same task, rate-limit distribution across API keys, cost averaging between expensive and cheap providers, and high-availability configurations where one provider's downtime shouldn't halt the system.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ParallelAgent` | Parallel runs *all* agents; LoadBalancer picks *one* |
| `RouterAgent` | Router uses LLM intelligence to pick; LoadBalancer uses mechanical strategies |
| `EscalationAgent` | Escalation is serial (try each until success); LoadBalancer picks one upfront |
| `SpeculativeAgent` | Speculative races all; LoadBalancer routes to one |

```python
from nono.agent import Agent, LoadBalancerAgent, Runner

gemini = Agent(name="gemini", instruction="Answer.", provider="google")
gpt = Agent(name="gpt", instruction="Answer.", provider="openai")
groq = Agent(name="groq", instruction="Answer.", provider="groq")

lb = LoadBalancerAgent(
    name="lb",
    sub_agents=[gemini, gpt, groq],
    strategy="round_robin",
    result_key="lb_result",
)

# Each call routes to a different backend
for q in ["Q1?", "Q2?", "Q3?"]:
    print(Runner(lb).run(q))
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Equivalent backend agents |
| `strategy` | `"round_robin"` | Selection strategy: `round_robin`, `random`, or `least_used` |
| `result_key` | `None` | Optional state key to store routing decisions |

### A44 — EnsembleAgent

**Pattern**: Ensemble Methods — from machine learning, where combining multiple models' predictions produces better results than any single model. EnsembleAgent runs all sub-agents on the same input and aggregates their outputs using a configurable strategy.

Unlike `VotingAgent` (majority wins) or `ConsensusAgent` (judge synthesises), EnsembleAgent provides flexible aggregation: concatenation, weighted combination, or a fully custom reducer function. This is particularly useful when you want to hear every model's perspective rather than choosing one winner.

Supported aggregation strategies:

| Strategy | Description |
|---|---|
| `"concat"` | Concatenate all outputs with agent-name labels |
| `"weighted"` | Prepend weight percentages to each output (requires `weights`) |
| `callable` | Custom function `(dict[str, str]) -> str` for full control |

**When to use**: multi-model answers where you want a comprehensive combined view, situations where different models excel at different aspects (one is creative, another is factual), building training data by collecting multiple model outputs, and any scenario where "more perspectives = better".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting picks *one* answer by majority; Ensemble *combines all* |
| `ConsensusAgent` | Consensus uses an LLM judge to synthesise; Ensemble uses mechanical aggregation |
| `ParallelAgent` | Parallel collects results; Ensemble collects *and aggregates* into one output |
| `MixtureOfExpertsAgent` | MoE runs only top-k by gating; Ensemble runs *all* agents |

```python
from nono.agent import Agent, EnsembleAgent, Runner

model_a = Agent(name="gemini", instruction="Answer.", provider="google")
model_b = Agent(name="gpt", instruction="Answer.", provider="openai")

ens = EnsembleAgent(
    name="ensemble",
    sub_agents=[model_a, model_b],
    aggregate_fn="weighted",
    weights={"gemini": 0.6, "gpt": 0.4},
    result_key="ens_result",
)

result = Runner(ens).run("Explain quantum computing in simple terms")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents to ensemble |
| `aggregate_fn` | `"concat"` | Aggregation strategy: `"concat"`, `"weighted"`, or `Callable` |
| `weights` | `None` | `dict[str, float]` — per-agent weights (required for `"weighted"`) |
| `result_key` | `None` | Optional state key to store individual + aggregated outputs |

### A45 — TimeoutAgent

**Pattern**: Timeout / Deadline — a fundamental resilience pattern from distributed systems (Nygard, *Release It!*) that wraps any agent with a hard time limit. If the agent doesn't respond within `timeout_seconds`, execution is aborted and a fallback message is returned.

Sync mode uses `ThreadPoolExecutor` with a `concurrent.futures.wait(timeout=...)` call. Async mode uses `asyncio.wait_for()`. In both cases, if the deadline expires, the agent's computation is abandoned (not killed — the thread/coroutine may continue but its result is ignored).

This is a **safety net** for production systems where latency SLAs must be met regardless of LLM response times, which can vary wildly depending on provider load, model complexity, and prompt length.

**When to use**: user-facing applications with latency SLAs, wrapping expensive models (e.g. GPT-4) that occasionally take 30+ seconds, API endpoints with response time guarantees, and any scenario where "no answer" is better than "a slow answer".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CircuitBreakerAgent` | CircuitBreaker tracks failure *history*; TimeoutAgent guards *single calls* |
| `SpeculativeAgent` | Speculative races agents for the *best*; TimeoutAgent guards against *slowness* |
| `EscalationAgent` | Escalation retries on failure; TimeoutAgent returns fallback immediately |

```python
from nono.agent import Agent, TimeoutAgent, Runner

thinker = Agent(name="deep_thinker", instruction="Think very carefully.", provider="google")

guarded = TimeoutAgent(
    name="guarded",
    agent=thinker,
    timeout_seconds=10.0,
    fallback_message="Request timed out. Please try a simpler question.",
    result_key="timeout_result",
)

result = Runner(guarded).run("Solve this complex optimisation problem")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | The agent to wrap with a deadline |
| `timeout_seconds` | (required) | Maximum time in seconds before fallback |
| `fallback_message` | `"Timeout"` | Message returned when the deadline expires |
| `result_key` | `None` | Optional state key to store timeout status |

### A46 — AdaptivePlannerAgent

**Pattern**: Adaptive Plan-and-Execute — an evolution of `PlannerAgent` (A23) that incorporates **online re-planning**. Instead of creating a plan once and executing it blindly, the adaptive planner evaluates intermediate results after each step and can modify, extend, reorder, or shorten the remaining plan based on what it discovers.

This models how experienced professionals work: they start with a plan, but adjust it as new information emerges. If step 1 (research) reveals an unexpected subtopic, the planner can insert an additional research step that wasn't in the original plan. If step 2 (writing) already covers review points, the planner can skip the review step.

The execution loop: (1) LLM generates an initial plan, (2) execute the first step, (3) LLM receives the intermediate result and decides: continue with the current plan, modify it, or declare completion, (4) repeat until all steps complete or `max_steps` is reached.

**When to use**: exploratory tasks where the full scope isn't known upfront, research-heavy workflows where findings affect the strategy, dynamic environments where conditions change during execution, and any complex task where a static plan is too rigid.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `PlannerAgent` | Planner creates the plan once; AdaptivePlanner re-plans after every step |
| `LoopAgent` | LoopAgent repeats the same agents; AdaptivePlanner can change which agents run |
| `SupervisorAgent` | Supervisor delegates one task at a time; AdaptivePlanner maintains a full plan with dependencies |
| `DynamicFanOutAgent` | DynamicFanOut decomposes then executes in parallel; AdaptivePlanner is sequential with re-planning |

```python
from nono.agent import Agent, AdaptivePlannerAgent, Runner

researcher = Agent(name="researcher", instruction="Research the topic.", provider="google")
writer = Agent(name="writer", instruction="Write based on research.", provider="google")
reviewer = Agent(name="reviewer", instruction="Review and suggest improvements.", provider="google")

adaptive = AdaptivePlannerAgent(
    name="adaptive",
    sub_agents=[researcher, writer, reviewer],
    model="gemini-3-flash-preview",
    provider="google",
    max_steps=6,
    result_key="adaptive_result",
)

result = Runner(adaptive).run("Write a comprehensive report on quantum computing trends")
```

**Re-planning example**:

```
Original plan: researcher → writer → reviewer
After step 1:  researcher found 3 sub-topics
Revised plan:  researcher(topic2) → researcher(topic3) → writer → reviewer
                ↑ new steps inserted dynamically
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Available agents for the planner to assign |
| `model` / `provider` | (required) | LLM for plan generation and re-planning |
| `max_steps` | `6` | Maximum total steps (safety cap for re-planning) |
| `result_key` | `None` | Optional state key to store plan evolution + results |

### A47 — SkeletonOfThoughtAgent

**Pattern**: Skeleton-of-Thought (Ning et al., 2023) — a latency-optimisation technique that parallelises the **elaboration** phase of long-form generation. Instead of producing a complete answer sequentially, the agent first generates a skeleton (N key points), then N workers elaborate each point **concurrently**, and an assembler stitches the elaborated sections into a coherent document.

The pipeline has three phases: (1) **Skeleton** — the skeleton agent analyses the question and produces a numbered list of key points, (2) **Elaborate** — each point is sent to a worker agent in parallel (ThreadPoolExecutor sync, asyncio.gather async), (3) **Assemble** — the assembler receives all elaborated sections with the original skeleton and produces the final merged document.

This pattern dramatically reduces wall-clock time for long-form content: if elaboration of each point takes T seconds, sequential generation takes N×T while Skeleton-of-Thought takes approximately T (the parallelisation overhead is minimal).

**When to use**: long-form content generation (reports, articles, documentation), any task where the output has natural structure (sections, chapters, bullet points), latency-sensitive applications where parallel elaboration beats serial generation, and educational content creation.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential runs agents in series; SkeletonOfThought parallelises the elaboration phase |
| `MapReduceAgent` | MapReduce sends the same input to different agents; SkeletonOfThought divides the *output structure* |
| `DynamicFanOutAgent` | DynamicFanOut decomposes tasks generically; SkeletonOfThought decomposes the *answer structure* |
| `PlannerAgent` | Planner creates a dependency graph; SkeletonOfThought creates a flat outline for parallel work |

```python
from nono.agent import Agent, SkeletonOfThoughtAgent, Runner

outliner = Agent(name="outliner", instruction="Create a numbered outline.", provider="google")
writer = Agent(name="writer", instruction="Elaborate on the given point.", provider="google")
assembler = Agent(name="assembler", instruction="Merge sections into coherent text.", provider="google")

sot = SkeletonOfThoughtAgent(
    name="sot",
    skeleton_agent=outliner,
    worker_agent=writer,
    assembler_agent=assembler,
    max_points=5,
    result_key="sot_result",
)

result = Runner(sot).run("Write a comprehensive guide to microservices architecture")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `skeleton_agent` | (required) | Agent that generates the numbered outline |
| `worker_agent` | (required) | Agent that elaborates each point (cloned per point) |
| `assembler_agent` | (required) | Agent that merges elaborated sections |
| `max_points` | `7` | Maximum outline points to extract |
| `result_key` | `None` | Optional state key to store skeleton + elaborations + final |

### A48 — LeastToMostAgent

**Pattern**: Least-to-Most Prompting (Zhou et al., 2022) — a reasoning strategy that decomposes a complex problem into sub-problems ordered **from easiest to hardest**, solves each sequentially, and accumulates all previous solutions as context for the next. Each step builds on the knowledge gained from simpler sub-problems.

Unlike `SubQuestionAgent` (which solves sub-questions independently in parallel), LeastToMost enforces a **strict difficulty order** and **cumulative context**: the solver for sub-problem 3 sees the solutions to sub-problems 1 and 2. This mimics how curriculum learning works — you master fundamentals before attempting advanced material.

The pipeline: (1) **Decompose** — the decomposer agent breaks the problem into ordered sub-problems, (2) **Solve sequentially** — each sub-problem is solved with all prior solutions as context, (3) **Synthesise** — the synthesizer combines all partial solutions into a coherent final answer.

**When to use**: multi-step reasoning where each step builds on previous ones, mathematical problem-solving (solve simpler cases first), programming challenges (implement helpers before the main function), and any domain where foundational knowledge enables advanced reasoning.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SubQuestionAgent` | SubQuestion solves sub-queries in parallel (independent); LeastToMost solves sequentially (cumulative) |
| `PlannerAgent` | Planner creates a dependency graph; LeastToMost imposes strict difficulty ordering |
| `AdaptivePlannerAgent` | AdaptivePlanner re-plans after each step; LeastToMost follows the difficulty order fixed at decomposition |
| `SequentialAgent` | Sequential runs fixed agents; LeastToMost dynamically decomposes and accumulates context |

```python
from nono.agent import Agent, LeastToMostAgent, Runner

decomposer = Agent(name="decomposer", instruction="Break into easy-to-hard sub-problems.", provider="google")
solver = Agent(name="solver", instruction="Solve using previous solutions as context.", provider="google")
synth = Agent(name="synth", instruction="Combine all solutions into a final answer.", provider="google")

l2m = LeastToMostAgent(
    name="l2m",
    decomposer_agent=decomposer,
    solver_agent=solver,
    synthesizer_agent=synth,
    max_subproblems=5,
    result_key="l2m_result",
)

result = Runner(l2m).run("Implement a balanced binary search tree in Python")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `decomposer_agent` | (required) | Agent that breaks the problem into ordered sub-problems |
| `solver_agent` | (required) | Agent that solves each sub-problem with cumulative context |
| `synthesizer_agent` | (required) | Agent that produces the final consolidated answer |
| `max_subproblems` | `5` | Maximum sub-problems to generate |
| `result_key` | `None` | Optional state key to store sub-problems + solutions + final |

### A49 — SelfDiscoverAgent

**Pattern**: Self-Discover (Zhou et al., 2024) — a meta-reasoning framework where the LLM **composes its own reasoning structure** before solving a task. Instead of following a fixed prompt template, the agent dynamically selects, adapts, and implements reasoning modules from a pool of strategies.

The four phases: (1) **SELECT** — given the task and a pool of reasoning modules (e.g. "Critical Thinking", "Decomposition", "Analogy"), the agent selects the most relevant ones, (2) **ADAPT** — tailor each selected module to the specific task, (3) **IMPLEMENT** — combine the adapted modules into a step-by-step reasoning structure, (4) **EXECUTE** — follow the composed structure to produce the answer.

This outperforms fixed Chain-of-Thought prompting because the reasoning strategy is **tailored per task** — a math problem gets different modules than a creative writing task.

**When to use**: complex tasks where the optimal reasoning strategy is unknown upfront, benchmarks and evaluations that test diverse reasoning skills, tasks that span multiple cognitive domains, and any scenario where "think about how to think" improves quality.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TreeOfThoughtsAgent` | ToT explores reasoning *paths*; SelfDiscover composes a reasoning *structure* before starting |
| `CompilerAgent` | Compiler optimises instructions against a dataset; SelfDiscover composes reasoning modules per-query |
| `ReflexionAgent` | Reflexion improves output iteratively; SelfDiscover improves the *method* upfront |
| `PlannerAgent` | Planner decomposes into agent tasks; SelfDiscover decomposes into cognitive strategies |

```python
from nono.agent import Agent, SelfDiscoverAgent, Runner

thinker = Agent(name="thinker", instruction="Follow the reasoning structure.", provider="google")

sd = SelfDiscoverAgent(
    name="discovery",
    agent=thinker,
    reasoning_modules=["Critical Thinking", "Step-by-Step", "Analogy", "First Principles"],
    result_key="sd_result",
)

result = Runner(sd).run("Why do some startups succeed while others fail?")
```

**Default reasoning modules** (used when none specified):

| Module | Description |
|--------|-------------|
| Critical Thinking | Evaluate assumptions and evidence |
| Step-by-Step Analysis | Break into sequential steps |
| Analogical Reasoning | Draw parallels to known domains |
| Decomposition | Divide into smaller parts |
| Abstraction & Generalisation | Extract general principles |
| Cause-and-Effect Analysis | Trace causal chains |
| Hypothesis Testing | Form and test hypotheses |
| Constraint Satisfaction | Identify and work within constraints |
| Pattern Recognition | Identify recurring patterns |
| First Principles Reasoning | Reason from fundamental truths |

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | LLM agent used for all four phases |
| `reasoning_modules` | (10 defaults) | Pool of reasoning strategies to select from |
| `result_key` | `None` | Optional state key to store all four phases |

### A50 — GeneticAlgorithmAgent

**Pattern**: Genetic Algorithm / Evolutionary Optimisation (Holland, 1975; Mada Mani et al., 2023 — EvoPrompting) — maintains a **population** of candidate solutions that evolve through selection, crossover, and mutation over multiple generations.

Each generation: (1) **Evaluate** — `fitness_fn` scores every candidate, (2) **Select** — top candidates become parents (elite carry-over ensures the best survive), (3) **Crossover** — a crossover agent combines pairs of parents into offspring, (4) **Mutate** — a mutation agent introduces creative variation (probability controlled by `mutation_rate`). After `n_generations`, the highest-fitness individual is returned.

This is more powerful than `BestOfNAgent` (single generation) because it compounds improvements: each generation's offspring are better starting points for the next round of crossover and mutation.

**When to use**: creative optimisation where quality varies widely (marketing copy, code, designs), prompt optimisation (evolving better prompts), multi-objective optimisation, and any scenario where iterative recombination of good solutions produces better ones.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `BestOfNAgent` | BestOfN is a single generation; Genetic evolves across multiple generations |
| `MonteCarloAgent` | MCTS explores a tree; Genetic evolves a flat population |
| `TournamentAgent` | Tournament selects a winner; Genetic creates *new* solutions via crossover |
| `CompilerAgent` | Compiler evolves instructions; Genetic evolves the actual output |

```python
from nono.agent import Agent, GeneticAlgorithmAgent, Runner

generator = Agent(name="gen", instruction="Generate a creative tagline.", provider="google")
combiner = Agent(name="cross", instruction="Combine the best parts of both taglines.", provider="google")
mutator = Agent(name="mut", instruction="Add a creative twist to the tagline.", provider="google")

ga = GeneticAlgorithmAgent(
    name="ga",
    agent=generator,
    crossover_agent=combiner,
    mutation_agent=mutator,
    fitness_fn=lambda r: min(len(r) / 50, 1.0),
    population_size=6,
    n_generations=3,
    mutation_rate=0.3,
    elite_count=1,
    result_key="ga_result",
)

result = Runner(ga).run("Create a tagline for an AI-powered writing assistant")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates initial candidates |
| `crossover_agent` | (required) | Agent that combines two parents into offspring |
| `mutation_agent` | (required) | Agent that introduces variation |
| `fitness_fn` | (required) | `Callable[[str], float]` — scores each candidate |
| `population_size` | `6` | Candidates per generation |
| `n_generations` | `3` | Number of evolutionary cycles |
| `mutation_rate` | `0.3` | Probability of mutation per offspring |
| `elite_count` | `1` | Top candidates preserved unchanged |
| `result_key` | `None` | Optional state key to store evolution log |

### A51 — MultiArmedBanditAgent

**Pattern**: Multi-Armed Bandit (Robbins, 1952; Auer et al., 2002 — UCB) — an **online learning** router that maintains per-agent performance statistics and uses them to balance exploration (trying underused agents) vs. exploitation (using the best-performing agent).

Unlike `RouterAgent` (which uses an LLM to decide each time, with no memory) or `LoadBalancerAgent` (which distributes mechanically), the bandit **learns from results** over time. After each request, `reward_fn` scores the output and updates the selected agent's statistics. Over many requests, the bandit converges on the best-performing agent while still occasionally exploring alternatives.

Three strategies are supported:

| Strategy | Description |
|----------|-------------|
| `"epsilon_greedy"` | Exploit best agent with probability 1−ε, explore randomly with probability ε |
| `"ucb1"` | Upper Confidence Bound — balance average reward + exploration bonus |
| `"thompson"` | Thompson Sampling — sample from Beta distributions of each agent's reward |

**When to use**: multi-provider setups where you want to automatically discover the best model for a task, A/B testing where you want to converge on the winner, systems where agent performance varies over time (model updates, load changes), and cost optimisation where cheaper agents might be "good enough".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `RouterAgent` | Router uses LLM intelligence per-request; Bandit learns from historical rewards |
| `LoadBalancerAgent` | LoadBalancer distributes mechanically; Bandit distributes intelligently |
| `CascadeAgent` | Cascade tries in fixed order; Bandit adapts order based on experience |
| `SpeculativeAgent` | Speculative races all agents; Bandit picks one based on learned stats |

```python
from nono.agent import Agent, MultiArmedBanditAgent, Runner

fast = Agent(name="fast", instruction="Quick answer.", provider="groq")
smart = Agent(name="smart", instruction="Thorough answer.", provider="google")
cheap = Agent(name="cheap", instruction="Answer.", provider="deepseek")

bandit = MultiArmedBanditAgent(
    name="bandit",
    sub_agents=[fast, smart, cheap],
    reward_fn=lambda r: 1.0 if len(r) > 100 else 0.3,
    strategy="ucb1",
    result_key="bandit_result",
)

# Each call updates stats; over time, the best agent is favoured
for q in ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]:
    result = Runner(bandit).run(q)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Pool of candidate agents |
| `reward_fn` | (required) | `Callable[[str], float]` — reward in [0, 1] per response |
| `strategy` | `"epsilon_greedy"` | Selection strategy: `epsilon_greedy`, `ucb1`, or `thompson` |
| `epsilon` | `0.1` | Exploration rate (epsilon_greedy only) |
| `stats_key` | `"_bandit_stats"` | Session state key for persisting arm statistics |
| `result_key` | `None` | Optional state key to store selection log |

### A52 — SocraticAgent

**Pattern**: Socratic Method (Plato; Chang et al., 2023 — Socratic Models) — a cooperative exploration pattern where a **questioner** agent asks increasingly probing questions and a **respondent** agent answers them. The questioner analyses each answer, identifies gaps or unexplored aspects, and formulates deeper follow-up questions. The loop continues until the topic is exhaustively explored or `max_rounds` is reached.

Unlike `DebateAgent` (adversarial: attack vs. defend), the Socratic pattern is **cooperative**: the questioner doesn't challenge — it guides deeper exploration. Unlike `GroupChatAgent` (many equal peers), the Socratic pattern has **asymmetric roles**: teacher (questioner) and student (respondent).

The dialogue accumulates context: each round, both agents see the full history of questions and answers, enabling progressively deeper inquiry.

**When to use**: knowledge exploration where depth is more important than breadth, tutoring systems that need to probe understanding, research assistance where iterative questioning reveals nuances, and interview simulations.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `DebateAgent` | Debate is adversarial (attack/defend); Socratic is cooperative (question/deepen) |
| `GroupChatAgent` | GroupChat has N equal peers; Socratic has asymmetric teacher/student roles |
| `LoopAgent` | Loop repeats the same agents blindly; Socratic evolves questions based on answers |
| `ReflexionAgent` | Reflexion reflects on its own output; Socratic uses external questioning |

```python
from nono.agent import Agent, SocraticAgent, Runner

questioner = Agent(
    name="questioner",
    instruction="Ask probing questions. Reply EXPLORATION_COMPLETE when done.",
    provider="google",
)
respondent = Agent(name="respondent", instruction="Answer thoroughly.", provider="google")
synthesizer = Agent(name="synth", instruction="Synthesise all insights.", provider="google")

socratic = SocraticAgent(
    name="socratic",
    questioner_agent=questioner,
    respondent_agent=respondent,
    synthesizer_agent=synthesizer,
    max_rounds=5,
    result_key="socratic_result",
)

result = Runner(socratic).run("Explore the ethics of autonomous AI agents")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `questioner_agent` | (required) | Agent that generates probing questions |
| `respondent_agent` | (required) | Agent that answers each question |
| `synthesizer_agent` | `None` | Optional agent that produces a final summary |
| `max_rounds` | `5` | Maximum question-answer rounds |
| `completion_keyword` | `"EXPLORATION_COMPLETE"` | Keyword that ends the dialogue early |
| `result_key` | `None` | Optional state key to store full dialogue |

### A53 — MetaOrchestratorAgent

**Pattern**: Meta-Orchestration — an LLM doesn't choose an **agent** — it chooses the **orchestration pattern**. Given a task, the meta-orchestrator analyses its nature and selects the best strategy (Sequential, Parallel, Debate, Cascade, Consensus, MapReduce, etc.) from a registry of available patterns, then constructs and executes the chosen pattern with the supplied worker agents.

This is **orchestration of the orchestration** — a higher-order pattern that automates the architectural decision that normally a developer makes at design time. With a patterns registry, the system becomes self-configuring: new orchestration strategies can be registered at runtime.

**When to use**: general-purpose AI assistants where the optimal strategy varies by task type, platforms that serve diverse task categories, research systems that want to compare orchestration strategies automatically, and any scenario where "which pattern?" is a non-trivial decision.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `RouterAgent` | Router chooses an *agent*; MetaOrchestrator chooses a *pattern* |
| `PlannerAgent` | Planner creates a plan of steps; MetaOrchestrator selects the entire topology |
| `AdaptivePlannerAgent` | AdaptivePlanner re-plans steps; MetaOrchestrator selects the orchestration strategy |
| `SupervisorAgent` | Supervisor delegates within a flat pool; MetaOrchestrator composes the delegation structure |

```python
from nono.agent import (
    Agent, MetaOrchestratorAgent, SequentialAgent, ParallelAgent,
    DebateAgent, Runner,
)

researcher = Agent(name="researcher", description="Research topics.", instruction="Research.", provider="google")
writer = Agent(name="writer", description="Write content.", instruction="Write.", provider="google")
reviewer = Agent(name="reviewer", description="Review content.", instruction="Review.", provider="google")

meta = MetaOrchestratorAgent(
    name="meta",
    sub_agents=[researcher, writer, reviewer],
    patterns={
        "SequentialAgent": lambda *, name, sub_agents: SequentialAgent(name=name, sub_agents=sub_agents),
        "ParallelAgent": lambda *, name, sub_agents: ParallelAgent(name=name, sub_agents=sub_agents),
    },
    model="gemini-3-flash-preview",
    provider="google",
    result_key="meta_result",
)

result = Runner(meta).run("Write a technical blog post about WebAssembly")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Worker agents available for any pattern |
| `patterns` | `{}` | `dict[str, Callable]` — pattern name → factory function |
| `model` / `provider` | (required) | LLM for pattern selection |
| `result_key` | `None` | Optional state key to store selected pattern + output |

### A54 — CacheAgent

**Pattern**: Semantic Caching / Memoization (GPTCache, Bang et al., 2023) — a wrapper that intercepts requests and checks a cache before calling the wrapped agent. If the input matches a cached entry (exact hash match or semantic similarity above threshold), the cached response is returned instantly without an LLM call.

The cache operates in session state with configurable TTL (time-to-live) and LRU eviction when `max_entries` is exceeded. For semantic matching, an optional `similarity_fn` compares the new query against cached queries — enabling "fuzzy" cache hits where slightly rephrased questions return the same answer.

This pattern can reduce LLM costs by 50-90% for applications with repetitive queries (FAQ bots, customer support, batch processing of similar items).

**When to use**: FAQ-style bots where the same questions recur, batch processing where many items have similar prompts, cost reduction for high-volume applications, and latency optimisation for repeated queries.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TimeoutAgent` | Timeout guards against slowness; Cache eliminates redundant calls |
| `LoadBalancerAgent` | LoadBalancer distributes calls; Cache avoids calls entirely |
| `CheckpointableAgent` | Checkpointable persists pipeline progress; Cache persists individual responses |
| `BudgetAgent` | Budget caps total cost; Cache reduces cost by reusing results |

```python
from nono.agent import Agent, CacheAgent, Runner

expensive = Agent(name="gpt4", instruction="Answer thoroughly.", provider="openai", model="gpt-4o")

cached = CacheAgent(
    name="cached_gpt4",
    agent=expensive,
    ttl_seconds=3600,      # 1 hour TTL
    max_entries=500,
    result_key="cache_result",
)

# First call: cache MISS → calls GPT-4
result1 = Runner(cached).run("What is quantum computing?")

# Second identical call: cache HIT → instant response
result2 = Runner(cached).run("What is quantum computing?")
```

**Semantic caching** (fuzzy matching):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_sim(a: str, b: str) -> float:
    embs = model.encode([a, b])
    return float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])))

cached = CacheAgent(
    name="semantic_cache",
    agent=expensive,
    similarity_fn=cosine_sim,
    similarity_threshold=0.92,
)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | The agent to wrap with caching |
| `similarity_fn` | `None` | `Callable[[str, str], float]` — semantic similarity function |
| `similarity_threshold` | `0.95` | Minimum similarity for a semantic cache hit |
| `ttl_seconds` | `0` | Time-to-live in seconds (0 = no expiry) |
| `max_entries` | `100` | Maximum cache entries before LRU eviction |
| `cache_key` | `"_cache_store"` | Session state key for the cache |
| `result_key` | `None` | Optional state key to store hit/miss info |

### A55 — BudgetAgent

**Pattern**: Cost-Aware / Frugal AI (Chen et al., 2023) — a wrapper that enforces a **token or cost budget** across a pipeline of sub-agents. After each agent completes, a `cost_fn` estimates the cost of its output. When the cumulative cost exceeds the budget, the agent can stop, switch to a cheaper fallback, or continue with a warning.

This is the **financial guardrail** pattern: in production, unbounded LLM pipelines can generate unexpected costs. `BudgetAgent` ensures a hard cap. The `cost_fn` can estimate tokens (e.g. `len(output) / 4`), use actual API billing data, or apply any custom cost model.

**When to use**: production deployments with cost SLAs, multi-step pipelines where total cost must be capped, educational/demo environments with limited API credits, and any scenario where "runaway costs" are a risk.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CascadeAgent` | Cascade escalates to better models; Budget degrades to cheaper models when budget runs out |
| `TimeoutAgent` | Timeout caps *time*; Budget caps *cost* |
| `CircuitBreakerAgent` | CircuitBreaker reacts to failures; Budget reacts to consumption |
| `CacheAgent` | Cache reduces cost by reusing results; Budget caps cost by stopping/degrading |

```python
from nono.agent import Agent, BudgetAgent, Runner

researcher = Agent(name="researcher", instruction="Research.", provider="google")
writer = Agent(name="writer", instruction="Write.", provider="google")
reviewer = Agent(name="reviewer", instruction="Review.", provider="google")
fallback = Agent(name="cheap", instruction="Quick summary.", provider="groq")

budget = BudgetAgent(
    name="budget_pipeline",
    sub_agents=[researcher, writer, reviewer],
    cost_fn=lambda r: len(r) / 4,  # approximate token count
    budget=2000,                    # max 2000 tokens total
    fallback_agent=fallback,
    on_exhausted="fallback",
    result_key="budget_result",
)

result = Runner(budget).run("Write a research report on AI safety")
```

**Exhaustion strategies**:

| Strategy | Behaviour |
|----------|-----------|
| `"stop"` | Stop the pipeline immediately (default) |
| `"fallback"` | Switch to `fallback_agent` for remaining work |
| `"warn"` | Log a warning but continue executing |

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Pipeline of agents to run within budget |
| `cost_fn` | `len(r)/4` | `Callable[[str], float]` — estimates cost per output |
| `budget` | `1000.0` | Maximum total cost |
| `fallback_agent` | `None` | Agent used when budget is exhausted (with `"fallback"`) |
| `on_exhausted` | `"stop"` | Strategy: `"stop"`, `"fallback"`, or `"warn"` |
| `result_key` | `None` | Optional state key to store cost log |

### A56 — CurriculumAgent

**Pattern**: Automatic Curriculum Learning (Wang et al., 2023 — Voyager) — an open-ended learning pattern where an agent progressively tackles harder tasks and stores successful solutions in a **skill library**. On future tasks, the library is consulted before attempting a fresh solution, enabling knowledge reuse and incremental capability building.

The loop: (1) **Propose** — the proposer agent generates the next task (progressively harder, building on acquired skills), (2) **Check library** — if a matching skill already exists, reuse it, (3) **Solve** — the solver agent attempts the task with the skill library as context, (4) **Evaluate** — `success_fn` determines if the task was solved, (5) **Store** — successful solutions are added to the library.

This pattern enables agents to **accumulate capabilities over sessions** — the skill library persists in session state and grows with each successful task.

**When to use**: code generation where building blocks accumulate (helper functions, modules), training simulations where progressive difficulty improves learning, research tasks where each discovery enables the next, and any open-ended scenario where an agent needs to build incrementally.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ReflexionAgent` | Reflexion improves a single output; Curriculum accumulates a reusable skill library |
| `CompilerAgent` | Compiler optimises one agent's instructions; Curriculum generates and stores complete solutions |
| `AdaptivePlannerAgent` | AdaptivePlanner re-plans within one task; Curriculum generates a *programme of tasks* |
| `LoopAgent` | Loop repeats the same workflow; Curriculum proposes *different* tasks of increasing difficulty |

```python
from nono.agent import Agent, CurriculumAgent, Runner

proposer = Agent(
    name="proposer",
    instruction="Propose progressively harder Python tasks.",
    provider="google",
)
solver = Agent(
    name="solver",
    instruction="Solve the task. Use available skills when possible.",
    provider="google",
)

curriculum = CurriculumAgent(
    name="curriculum",
    proposer_agent=proposer,
    solver_agent=solver,
    success_fn=lambda r: "def " in r or "class " in r,  # basic check
    max_tasks=5,
    result_key="cur_result",
)

result = Runner(curriculum).run("Learn to build a REST API in Python")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `proposer_agent` | (required) | Agent that generates progressively harder tasks |
| `solver_agent` | (required) | Agent that attempts each task |
| `success_fn` | (required) | `Callable[[str], bool]` — True if the task was solved |
| `max_tasks` | `5` | Maximum tasks to attempt |
| `library_key` | `"_skill_library"` | Session state key for the persistent skill library |
| `result_key` | `None` | Optional state key to store curriculum log |

### A57 — SelfConsistencyAgent

**Pattern**: Self-Consistency (Wang et al., 2022) — sample N reasoning paths from the **same agent** (with sampling diversity), extract the final answer from each path, and pick the answer with **majority vote**. The diversity comes from the stochastic nature of LLM sampling, not from different agents.

Unlike `VotingAgent` (which uses *different* agents) or `BestOfNAgent` (which uses an LLM judge to rank), `SelfConsistencyAgent` samples the *same* agent multiple times and applies a deterministic `extract_fn` + `Counter.most_common` vote. This makes it cheaper and more robust for tasks with a clear "correct answer" (math, classification, factual questions).

**When to use**: mathematical reasoning where multiple Chain-of-Thought paths may diverge, classification tasks where sampling variance reveals the most confident answer, factual Q&A where the most frequent answer is most likely correct, and any task where "ask the same expert N times and take the majority" outperforms "ask once".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting uses different agents; SelfConsistency samples the same agent N times |
| `BestOfNAgent` | BestOfN uses an LLM judge; SelfConsistency uses majority vote |
| `EnsembleAgent` | Ensemble aggregates with a custom function; SelfConsistency extracts + votes |
| `SelfRefineAgent` | SelfRefine iterates on one output; SelfConsistency generates N independent outputs |

```python
from nono.agent import Agent, SelfConsistencyAgent, Runner

solver = Agent(name="solver", instruction="Solve step by step.", provider="google")

sc = SelfConsistencyAgent(
    name="sc",
    agent=solver,
    n_samples=5,
    extract_fn=lambda s: s.strip().split("\\n")[-1],  # last line = answer
    result_key="sc_result",
)

result = Runner(sc).run("What is 17 × 23?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent to sample from (called N times) |
| `n_samples` | `5` | Number of independent reasoning paths |
| `extract_fn` | `strip().lower()` | Extracts the "answer" portion from full output |
| `max_workers` | `4` | Thread pool parallelism for sampling |
| `result_key` | `None` | Optional state key to store outputs + winner |

### A58 — MixtureOfAgentsAgent

**Pattern**: Mixture-of-Agents (Together AI / Wang et al., 2024) — a **multi-layer** architecture: Layer 1 runs N proposers in parallel, Layer 2 runs M aggregators that each see **all** Layer 1 outputs, repeat for K layers, then an optional final agent synthesises the result. Each layer refines the collective output.

This stacks refinement layers — unlike `EnsembleAgent` (single-layer aggregation), `MixtureOfExpertsAgent` (routes to ONE expert), or `DebateAgent` (adversarial). MoA is **collaborative multi-layer**: every aggregator has full visibility of all previous proposals, enabling emergent quality improvement.

**When to use**: complex tasks requiring multiple perspectives (reports, designs, code), scenarios where iterative refinement across diverse models improves quality, multi-model pipelines where each model contributes unique strengths, and any task where "N heads × K refinements > 1 shot".

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `EnsembleAgent` | Ensemble is single-layer; MoA stacks multiple refinement layers |
| `MixtureOfExpertsAgent` | MoE routes to ONE expert; MoA runs ALL and stacks |
| `DebateAgent` | Debate is adversarial; MoA is collaborative multi-layer |
| `ParallelAgent` | Parallel runs once; MoA repeats with refinement context |

```python
from nono.agent import Agent, MixtureOfAgentsAgent, Runner

p1 = Agent(name="creative", instruction="Creative approach.", provider="google")
p2 = Agent(name="analytical", instruction="Analytical approach.", provider="openai")
agg = Agent(name="aggregator", instruction="Synthesise all proposals.", provider="google")
final = Agent(name="final", instruction="Polish the final output.", provider="google")

moa = MixtureOfAgentsAgent(
    name="moa",
    proposer_agents=[p1, p2],
    aggregator_agents=[agg],
    final_agent=final,
    n_layers=3,
    result_key="moa_result",
)

result = Runner(moa).run("Design a microservices architecture for e-commerce")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `proposer_agents` | (required) | Layer 1 agents that generate initial proposals |
| `aggregator_agents` | `None` | Agents for layers 2-N (defaults to proposers) |
| `final_agent` | `None` | Optional single agent for final synthesis |
| `n_layers` | `2` | Number of aggregation layers |
| `max_workers` | `4` | Thread pool parallelism |
| `result_key` | `None` | Optional state key to store layer outputs |

### A59 — StepBackAgent

**Pattern**: Step-Back Prompting (Zheng et al., 2023) — a two-phase reasoning strategy: (1) **Abstraction** — generate a higher-level question about the underlying principles, (2) **Grounded reasoning** — answer the original question using the abstraction as context. Simple but powerful: it forces the LLM to reason from fundamentals.

Unlike `SelfDiscoverAgent` (which composes multiple reasoning modules) or `SubQuestionAgent` (which decomposes into sub-queries), `StepBackAgent` moves *up* one abstraction level before moving *down* to the answer.

**When to use**: scientific reasoning where principles underpin specific cases, debugging where understanding root causes precedes fixing symptoms, strategy questions where first-principles thinking beats pattern matching, and any domain where "why?" before "how?" improves quality.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SelfDiscoverAgent` | SelfDiscover composes reasoning modules; StepBack abstracts the problem |
| `SubQuestionAgent` | SubQuestion decomposes into sub-queries; StepBack "zooms out" |
| `LeastToMostAgent` | LeastToMost solves easy→hard; StepBack abstracts→grounds |
| `ReflexionAgent` | Reflexion reflects on output; StepBack reflects on the problem itself |

```python
from nono.agent import Agent, StepBackAgent, Runner

abstractor = Agent(name="abstractor", instruction="What general principle applies?", provider="google")
reasoner = Agent(name="reasoner", instruction="Answer using the abstraction.", provider="google")

sb = StepBackAgent(
    name="stepback",
    abstractor_agent=abstractor,
    reasoner_agent=reasoner,
    result_key="sb_result",
)

result = Runner(sb).run("Why does ice float on water?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `abstractor_agent` | (required) | Agent that generates the higher-level abstraction |
| `reasoner_agent` | (required) | Agent that answers with the abstraction as context |
| `result_key` | `None` | Optional state key to store abstraction + answer |

### A60 — OrchestratorWorkerAgent

**Pattern**: Orchestrator-Worker (Anthropic MCP / OpenAI Agents SDK canonical pattern) — an LLM that **iteratively** plans, delegates to workers, evaluates results, and re-plans. Unlike `SupervisorAgent` (single-round delegation) or `PlannerAgent` (static DAG), the orchestrator maintains an interactive conversation loop until it declares the task complete.

The LLM outputs JSON delegations: `{"worker": "name", "task": "sub-task"}`. After each worker completes, the result is fed back to the orchestrator, which decides what's next. This continues until `completion_keyword` is detected or `max_rounds` is reached.

**When to use**: complex multi-step tasks where the plan depends on intermediate results, agentic assistants that need to decompose and execute dynamically, scenarios where fixed plans fail because each step's output changes the strategy, and any "manager + workers" pattern.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SupervisorAgent` | Supervisor delegates once per round; Orchestrator iterates interactively |
| `PlannerAgent` | Planner creates a static DAG; Orchestrator re-plans after each delegation |
| `AdaptivePlannerAgent` | AdaptivePlanner re-plans steps; Orchestrator reasons about worker results |
| `MetaOrchestratorAgent` | Meta selects the pattern; Orchestrator IS the interactive pattern |

```python
from nono.agent import Agent, OrchestratorWorkerAgent, Runner

researcher = Agent(name="researcher", description="Research topics.", instruction="Research.", provider="google")
coder = Agent(name="coder", description="Write code.", instruction="Code.", provider="google")
reviewer = Agent(name="reviewer", description="Review work.", instruction="Review.", provider="google")

ow = OrchestratorWorkerAgent(
    name="orchestrator",
    sub_agents=[researcher, coder, reviewer],
    model="gemini-3-flash-preview",
    provider="google",
    max_rounds=10,
    result_key="ow_result",
)

result = Runner(ow).run("Build a REST API with authentication and tests")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Worker agents to delegate to |
| `model` / `provider` | (required) | LLM for orchestration decisions |
| `max_rounds` | `10` | Maximum plan→delegate→evaluate cycles |
| `completion_keyword` | `"TASK_COMPLETE"` | Keyword that signals task completion |
| `result_key` | `None` | Optional state key to store rounds log |

### A61 — SelfRefineAgent

**Pattern**: Self-Refine (Madaan et al., 2023) — iterative loop of **generate → self-critique → refine** using a single agent. The critique step produces structured feedback (what failed, what to improve), and the refine step uses that feedback to produce an improved version. Repeats until `stop_phrase` appears in the critique or `max_iterations` is reached.

Unlike `ReflexionAgent` (which persists lessons across tasks) or `ProducerReviewerAgent` (which uses two different agents), `SelfRefineAgent` uses the **same** agent for both critique and refinement, and the feedback is **immediate** (no persistent memory).

**When to use**: writing tasks where iterative editing improves quality, code generation where self-review catches bugs, any creative task where "write → critique → rewrite" beats single-shot generation, and quality-sensitive outputs where one pass isn't enough.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ReflexionAgent` | Reflexion persists lessons across tasks; SelfRefine is immediate per-query |
| `ProducerReviewerAgent` | PR uses two agents for one round; SelfRefine uses one agent for N rounds |
| `LoopAgent` | Loop repeats blindly; SelfRefine generates structured critique between iterations |
| `CompilerAgent` | Compiler optimises instructions; SelfRefine optimises the actual output |

```python
from nono.agent import Agent, SelfRefineAgent, Runner

writer = Agent(name="writer", instruction="Write and critique clearly.", provider="google")

sr = SelfRefineAgent(
    name="self_refine",
    agent=writer,
    max_iterations=3,
    stop_phrase="NO_ISSUES_FOUND",
    result_key="sr_result",
)

result = Runner(sr).run("Write a technical blog post about WebAssembly")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for generate, critique, and refine |
| `max_iterations` | `3` | Maximum refinement rounds |
| `stop_phrase` | `"NO_ISSUES_FOUND"` | Phrase in critique that stops the loop |
| `critique_instruction` | (default) | System prompt for the critique step |
| `refine_instruction` | (default) | System prompt for the refine step |
| `result_key` | `None` | Optional state key to store iterations log |

### A62 — BacktrackingAgent

**Pattern**: Backtracking Search (inspired by LATS / Zhou et al., 2023) — runs a pipeline of `sub_agents` sequentially with `validate_fn` after each step. If validation fails, the agent **rewinds** to a previous step (`backtrack_to`) and retries with knowledge of failed attempts. After `max_retries` backtrack cycles, it gives up.

Unlike `SagaAgent` (which compensates forward), `CircuitBreakerAgent` (which stops entirely), or `LoopAgent` (which restarts from scratch), `BacktrackingAgent` maintains a **history of failures** and injects that context into retries, preventing repeated mistakes.

**When to use**: multi-step pipelines where intermediate results need validation, code generation where compilation/tests must pass between steps, content pipelines with quality gates, and any workflow where "try, validate, retry differently" is the strategy.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SagaAgent` | Saga compensates forward on failure; Backtracking rewinds backward |
| `CircuitBreakerAgent` | CircuitBreaker fails open; Backtracking retries with different strategy |
| `LoopAgent` | Loop restarts from scratch; Backtracking rewinds to a specific step |
| `CheckpointableAgent` | Checkpoint persists state; Backtracking uses failure history for smarter retries |

```python
from nono.agent import Agent, BacktrackingAgent, Runner

planner = Agent(name="planner", instruction="Plan the solution.", provider="google")
coder = Agent(name="coder", instruction="Implement the plan.", provider="google")
tester = Agent(name="tester", instruction="Write tests.", provider="google")

bt = BacktrackingAgent(
    name="backtrack",
    sub_agents=[planner, coder, tester],
    validate_fn=lambda output: "error" not in output.lower(),
    backtrack_to=1,  # rewind to coder on failure
    max_retries=3,
    result_key="bt_result",
)

result = Runner(bt).run("Implement a binary search algorithm")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Ordered pipeline of agents |
| `validate_fn` | `lambda s: True` | `Callable[[str], bool]` — True if step output is valid |
| `backtrack_to` | `0` | Index of agent to rewind to (0-based) |
| `max_retries` | `3` | Maximum backtrack cycles |
| `result_key` | `None` | Optional state key to store failed attempts log |

### A63 — ChainOfDensityAgent

**Pattern**: Chain of Density (Adams et al., 2023) — iterative content densification. Each round: (1) identify missing entities/information in the current text, (2) rewrite to incorporate them **without increasing length**. Continues for `n_rounds` or until the agent signals `stop_phrase`.

The output gets progressively **denser in information** while maintaining the same conciseness — like a document that becomes more precise with each edit. This is not about making text longer but about making every word carry more meaning.

**When to use**: summarisation where initial drafts are sparse, executive summaries that need maximum information density, abstracts and blurbs where space is limited, and any content compression task.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CompilerAgent` | Compiler optimises instructions against a dataset; CoD densifies content iteratively |
| `ReflexionAgent` | Reflexion corrects errors; CoD adds missing information |
| `SelfRefineAgent` | SelfRefine is generic improvement; CoD has a specific densification objective |
| `LoopAgent` | Loop repeats blindly; CoD tracks information density as the driving metric |

```python
from nono.agent import Agent, ChainOfDensityAgent, Runner

writer = Agent(name="writer", instruction="Write informatively.", provider="google")

cod = ChainOfDensityAgent(
    name="cod",
    agent=writer,
    n_rounds=3,
    stop_phrase="FULLY_DENSE",
    result_key="cod_result",
)

result = Runner(cod).run("Summarise the history of quantum computing")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent used for all densification rounds |
| `n_rounds` | `3` | Number of densification iterations |
| `stop_phrase` | `"FULLY_DENSE"` | Phrase that signals "fully dense" |
| `result_key` | `None` | Optional state key to store rounds log |

### A64 — MediatorAgent

**Pattern**: Mediation / Conflict Resolution (Abdelnabi et al., 2023) — runs N agents in parallel to produce competing proposals, then a neutral **mediator** agent: (1) analyses points of agreement and disagreement, (2) synthesises a **compromise** that incorporates the best of each. Unlike a debate judge (who picks a winner), the mediator creates something new.

**When to use**: design decisions where multiple valid approaches exist, strategy discussions where different stakeholders have conflicting priorities, code review where different team members suggest different solutions, and any multi-perspective task where compromise beats selection.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `DebateAgent` | Debate judge picks a winner; Mediator creates a compromise |
| `ConsensusAgent` | Consensus iterates until peers converge; Mediator intervenes actively |
| `VotingAgent` | Voting picks by majority; Mediator fuses positions |
| `EnsembleAgent` | Ensemble aggregates mechanically; Mediator reasons about differences |

```python
from nono.agent import Agent, MediatorAgent, Runner

optimist = Agent(name="optimist", instruction="Propose an aggressive strategy.", provider="google")
pessimist = Agent(name="pessimist", instruction="Propose a conservative strategy.", provider="google")
mediator = Agent(name="mediator", instruction="Find the best compromise.", provider="google")

med = MediatorAgent(
    name="mediation",
    sub_agents=[optimist, pessimist],
    mediator_agent=mediator,
    result_key="med_result",
)

result = Runner(med).run("Should we launch the new product this quarter?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Agents that produce competing proposals |
| `mediator_agent` | (required) | Neutral agent that synthesises the compromise |
| `max_workers` | `4` | Thread pool parallelism for proposals |
| `result_key` | `None` | Optional state key to store proposals + compromise |

### A65 — DivideAndConquerAgent

**Pattern**: Recursive Divide-and-Conquer / Decomposed Prompting (Khot et al., 2022) — splits a problem into sub-problems **recursively**. If a sub-problem is simple (`is_base_case`), solve it directly. If complex, split again. Results merge bottom-up through the `merger_agent`. The key difference from flat decomposition is the **recursive nesting** with hierarchical merge.

Unlike `MapReduceAgent` (flat, single-level), `SubQuestionAgent` (independent sub-queries), or `LeastToMostAgent` (sequential ordering), `DivideAndConquerAgent` creates a **tree of sub-problems** and merges results hierarchically.

**When to use**: large documents that need section-by-section processing with hierarchical summary, complex coding tasks where top-level functions decompose into helper functions, research questions that have nested sub-questions, and any problem with natural recursive structure.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `MapReduceAgent` | MapReduce is flat (1 level); D&C is recursive (N levels) |
| `SubQuestionAgent` | SubQuestion generates independent queries; D&C nests recursively |
| `LeastToMostAgent` | LeastToMost solves sequentially; D&C solves in a tree with hierarchical merge |
| `SkeletonOfThoughtAgent` | Skeleton divides the output structure; D&C divides the problem |

```python
from nono.agent import Agent, DivideAndConquerAgent, Runner

splitter = Agent(name="splitter", instruction="Decompose into 2-4 sub-problems.", provider="google")
solver = Agent(name="solver", instruction="Solve this sub-problem.", provider="google")
merger = Agent(name="merger", instruction="Merge sub-results into one answer.", provider="google")

dc = DivideAndConquerAgent(
    name="dac",
    splitter_agent=splitter,
    solver_agent=solver,
    merger_agent=merger,
    is_base_case=lambda p: len(p) < 100,
    max_depth=3,
    result_key="dac_result",
)

result = Runner(dc).run("Explain the entire TCP/IP protocol stack")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `splitter_agent` | (required) | Agent that decomposes problems into sub-problems |
| `solver_agent` | (required) | Agent that solves base-case sub-problems |
| `merger_agent` | (required) | Agent that merges sub-results bottom-up |
| `is_base_case` | `len(p) < 100` | `Callable[[str], bool]` — True if the problem is simple enough |
| `max_depth` | `3` | Maximum recursion depth |
| `result_key` | `None` | Optional state key to store result |

### A66 — BeamSearchAgent

**Pattern**: Beam Search (Jurafsky & Martin; Xie et al., 2023 — Self-Evaluation Guided Beam Search) — maintains `beam_width` candidate reasoning paths in parallel. Each step: (1) expand each beam into `n_expansions` candidates, (2) score all candidates with `score_fn`, (3) prune to the top `beam_width`. After `n_steps`, the highest-scoring beam is returned.

Unlike `TreeOfThoughtsAgent` (BFS/DFS exploration with pruning), `MonteCarloAgent` (stochastic with UCB), or `GeneticAlgorithmAgent` (crossover + mutation), `BeamSearchAgent` is a classic **breadth-limited search** with deterministic scoring and fixed pruning at each step.

**When to use**: multi-step reasoning where maintaining K "hypotheses" improves quality, translation or paraphrasing where exploring alternatives beats greedy, step-by-step problem solving where partial progress is scorable, and any task where "parallel exploration with pruning" outperforms sequential.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TreeOfThoughtsAgent` | ToT explores with BFS/DFS and variable pruning; Beam keeps fixed K paths |
| `MonteCarloAgent` | MCTS is stochastic with UCB; Beam is deterministic with scoring |
| `GeneticAlgorithmAgent` | GA mixes solutions via crossover; Beam expands and prunes independently |
| `BestOfNAgent` | BestOfN generates N independently; Beam expands iteratively from prior best |

```python
from nono.agent import Agent, BeamSearchAgent, Runner

reasoner = Agent(name="reasoner", instruction="Continue the reasoning.", provider="google")

bs = BeamSearchAgent(
    name="beam",
    agent=reasoner,
    score_fn=lambda s: len(s) / 100.0,  # prefer longer, more detailed reasoning
    beam_width=3,
    n_expansions=2,
    n_steps=3,
    result_key="beam_result",
)

result = Runner(bs).run("Prove that the square root of 2 is irrational")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that expands each beam candidate |
| `score_fn` | (required) | `Callable[[str], float]` — scores each candidate |
| `beam_width` | `3` | Number of beams to keep after pruning |
| `n_expansions` | `2` | Expansions per beam per step |
| `n_steps` | `3` | Number of expand-score-prune steps |
| `max_workers` | `4` | Thread pool parallelism |
| `result_key` | `None` | Optional state key to store best score + result |

### A67 — RephraseAndRespondAgent

**Pattern**: Rephrase and Respond (Deng et al., 2023) — a two-phase self-improvement strategy. Phase 1: the agent rephrases the user's question to improve clarity and precision. Phase 2: the same agent solves the rephrased version. The insight is that LLMs often understand questions better when they rephrase them first.

**When to use**: ambiguous or poorly worded inputs, customer support where user phrasing is inconsistent, any task where clarifying the question before answering improves accuracy.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SelfRefineAgent` | SelfRefine iterates on the *answer*; RaR improves the *question* first |
| `ReflexionAgent` | Reflexion accumulates memory across attempts; RaR is a single rephrase+solve |
| `StepBackAgent` | StepBack abstracts to a general principle; RaR rephrases without abstraction |

```python
from nono.agent import Agent, RephraseAndRespondAgent, Runner

solver = Agent(name="solver", instruction="Answer the question.", provider="google")

rar = RephraseAndRespondAgent(
    name="rar",
    agent=solver,
    result_key="rar_result",
)

result = Runner(rar).run("What's the thing with electrons and clouds?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent used for both rephrase and solve phases |
| `result_key` | `None` | Optional state key for final answer |

### A68 — CumulativeReasoningAgent

**Pattern**: Cumulative Reasoning (Zhang et al., 2024) — a three-role cyclic pipeline. Each round the proposer generates a hypothesis, the verifier accepts or rejects it, and accepted facts accumulate. After N rounds the reporter synthesises all verified facts into a final answer. Knowledge grows incrementally.

**When to use**: complex questions requiring incremental knowledge building, multi-fact research, any scenario where partial evidence needs verification before aggregation.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CoVeAgent` | CoVe verifies fixed claims; CumulativeReasoning grows the knowledge base |
| `ProducerReviewerAgent` | Producer-Reviewer iterates on one answer; Cumulative adds new facts |
| `ReflexionAgent` | Reflexion stores lessons; Cumulative stores verified propositions |

```python
from nono.agent import Agent, CumulativeReasoningAgent, Runner

proposer = Agent(name="proposer", instruction="Propose a new relevant fact.", provider="google")
verifier = Agent(name="verifier", instruction="ACCEPT or REJECT the fact.", provider="google")
reporter = Agent(name="reporter", instruction="Synthesise all facts.", provider="google")

cr = CumulativeReasoningAgent(
    name="cr",
    proposer_agent=proposer,
    verifier_agent=verifier,
    reporter_agent=reporter,
    n_rounds=3,
)

result = Runner(cr).run("What caused the 2008 financial crisis?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `proposer_agent` | (required) | Generates hypotheses each round |
| `verifier_agent` | (required) | Accepts or rejects proposals |
| `reporter_agent` | (required) | Synthesises verified facts |
| `n_rounds` | `3` | Number of proposal rounds |
| `result_key` | `None` | Optional state key |

### A69 — MultiPersonaAgent

**Pattern**: Multi-Persona / Solo Performance Prompting (Wang et al., 2024 SPP) — a single agent adopts multiple personas sequentially. For each persona, the agent receives a tailored instruction and produces a perspective. A final synthesis step aggregates all perspectives.

**When to use**: brainstorming from multiple viewpoints, design reviews, any task benefiting from diverse perspectives without multiple separate agents.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `DebateAgent` | Debate uses separate agents arguing; MultiPersona uses one agent in roles |
| `ConsensusAgent` | Consensus votes among N agents; MultiPersona aggregates one agent's personas |
| `RolePlayingAgent` | RolePlaying has instructor+assistant; MultiPersona cycles through N identities |

```python
from nono.agent import Agent, MultiPersonaAgent, Runner

thinker = Agent(name="thinker", instruction="Provide your perspective.", provider="google")

mp = MultiPersonaAgent(
    name="multi",
    agent=thinker,
    personas=["scientist", "artist", "economist"],
    result_key="perspectives",
)

result = Runner(mp).run("How should cities handle urban flooding?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | The agent that adopts each persona |
| `personas` | (required) | List of persona descriptions |
| `result_key` | `None` | Optional state key |

### A70 — AntColonyAgent

**Pattern**: Ant Colony Optimisation (Dorigo & Stützle, 2004) — pheromone-guided parallel exploration. Multiple "ants" (agent calls) explore solutions in parallel. Each solution is scored, and pheromone values are updated to bias future exploration toward better paths. Over iterations, the colony converges on high-quality solutions.

**When to use**: solution spaces where good partial solutions can guide future attempts, creative exploration with progressive refinement, any optimisation where collective experience improves results.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `GeneticAlgorithmAgent` | GA mixes solutions via crossover; ACO uses pheromone-guided exploration |
| `ParticleSwarmAgent` | PSO tracks personal/global best positions; ACO tracks path frequencies |
| `SimulatedAnnealingAgent` | SA perturbs single solution with temperature; ACO runs parallel colony |

```python
from nono.agent import Agent, AntColonyAgent, Runner

explorer = Agent(name="ant", instruction="Propose a solution.", provider="google")

aco = AntColonyAgent(
    name="colony",
    agent=explorer,
    score_fn=lambda r: float(len(r)),
    n_ants=5,
    n_iterations=3,
    result_key="best_ant",
)

result = Runner(aco).run("Design a caching strategy for a web app")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates candidate solutions |
| `score_fn` | (required) | `Callable[[str], float]` — scores each solution |
| `n_ants` | `5` | Number of parallel ants per iteration |
| `n_iterations` | `3` | Number of pheromone update iterations |
| `evaporation` | `0.3` | Pheromone evaporation rate |
| `result_key` | `None` | Optional state key |

### A71 — PipelineParallelAgent

**Pattern**: Pipeline Parallelism — assembly-line stages. Items from `session.state["items"]` flow through a sequence of stage agents. Each item passes through all stages sequentially, while different items can be in different stages concurrently.

**When to use**: data processing pipelines, ETL workflows, any multi-stage processing of item lists.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential runs entire agents in order; Pipeline processes items through stages |
| `BatchAgent` | Batch processes items through one agent; Pipeline chains multiple stages |
| `MapReduceAgent` | MapReduce fans out then reduces; Pipeline is strictly sequential per item |

```python
from nono.agent import Agent, PipelineParallelAgent, Runner, Session

extract = Agent(name="extract", instruction="Extract key facts.", provider="google")
transform = Agent(name="transform", instruction="Rewrite concisely.", provider="google")

pipeline = PipelineParallelAgent(
    name="etl",
    stages=[extract, transform],
)

session = Session()
session.state["items"] = ["Document 1 text...", "Document 2 text..."]
result = Runner(pipeline).run("Process documents", session=session)
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `stages` | (required) | Ordered list of stage agents |
| `result_key` | `None` | Optional state key |

### A72 — ContractNetAgent

**Pattern**: Contract Net Protocol (Smith, 1980) — competitive bidding. The manager announces a task, all bidder agents submit proposals, and a `bid_fn` scores each bid. The highest bidder wins and executes the full task.

**When to use**: specialist selection, task delegation where agents have different strengths, resource allocation.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `AuctionAgent` | Auction bids on self-suitability; ContractNet bids on task proposals |
| `RouterAgent` | Router uses LLM classification; ContractNet uses competitive scoring |
| `MixtureOfExpertsAgent` | MoE blends all experts; ContractNet picks one winner |

```python
from nono.agent import Agent, ContractNetAgent, Runner

bidder1 = Agent(name="fast", instruction="I specialise in speed.", provider="google")
bidder2 = Agent(name="deep", instruction="I specialise in depth.", provider="google")

cn = ContractNetAgent(
    name="contract",
    sub_agents=[bidder1, bidder2],
    bid_fn=lambda name, resp: float(len(resp)),
    result_key="winner",
)

result = Runner(cn).run("Summarise this legal contract")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Bidding agents |
| `bid_fn` | `len(response)` | `Callable[[str, str], float]` — scores bids |
| `result_key` | `None` | Optional state key |

### A73 — RedTeamAgent

**Pattern**: Red Teaming (Perez et al., 2022) — adversarial hardening. A defender agent produces a response, then an attacker agent tries to find weaknesses. The defender revises iteratively until the attacker cannot find flaws or `max_rounds` is reached.

**When to use**: security testing of prompts, robustness hardening, policy compliance checking, any scenario where adversarial probing improves quality.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `DebateAgent` | Debate has two sides + judge; RedTeam has attacker probing defender |
| `ProducerReviewerAgent` | Producer-Reviewer cooperates; RedTeam adversarially attacks |
| `RecursiveCriticAgent` | Critic nests criticism deeply; RedTeam attacks from external adversary |

```python
from nono.agent import Agent, RedTeamAgent, Runner

defender = Agent(name="defender", instruction="Write a safe AI policy.", provider="google")
attacker = Agent(name="attacker", instruction="Find loopholes.", provider="google")

rt = RedTeamAgent(
    name="red_team",
    defender_agent=defender,
    attacker_agent=attacker,
    max_rounds=3,
)

result = Runner(rt).run("Draft content moderation policy")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `defender_agent` | (required) | Agent that produces and revises |
| `attacker_agent` | (required) | Agent that probes for weaknesses |
| `max_rounds` | `3` | Maximum attack-defend rounds |
| `result_key` | `None` | Optional state key |

### A74 — FeedbackLoopAgent

**Pattern**: Feedback Loop — circular chain with convergence detection. Agents in a chain pass output to the next, cycling until the output stabilises (measured by Jaccard similarity exceeding a threshold) or `max_rounds` is reached.

**When to use**: iterative refinement where multiple agents each improve a different aspect, collaborative editing, any circular workflow needing convergence.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `LoopAgent` | Loop repeats one agent; FeedbackLoop cycles through a chain |
| `SelfRefineAgent` | SelfRefine has generate+critique; FeedbackLoop chains N agents |
| `ProducerReviewerAgent` | Producer-Reviewer is 2 agents; FeedbackLoop supports N in a ring |

```python
from nono.agent import Agent, FeedbackLoopAgent, Runner

a = Agent(name="editor", instruction="Improve clarity.", provider="google")
b = Agent(name="checker", instruction="Check factual accuracy.", provider="google")

fb = FeedbackLoopAgent(
    name="feedback",
    chain=[a, b],
    max_rounds=5,
    threshold=0.9,
)

result = Runner(fb).run("Draft an article about quantum computing")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `chain` | (required) | Ordered list of agents forming the cycle |
| `max_rounds` | `5` | Maximum cycles |
| `threshold` | `0.9` | Jaccard similarity for convergence |
| `result_key` | `None` | Optional state key |

### A75 — WinnowingAgent

**Pattern**: Winnowing — progressive elimination by global ranking. Starts with N candidates generated in parallel. Each round, an evaluator agent scores all survivors and the bottom fraction is eliminated. Repeats until one candidate remains.

**When to use**: large candidate pools where iterative elimination is more efficient than pairwise tournament, progressive quality filtering, any scenario where global ranking beats bracket elimination.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `TournamentAgent` | Tournament uses pair brackets; Winnowing uses global ranking |
| `BestOfNAgent` | BestOfN is single-round; Winnowing iterates |
| `CascadeAgent` | Cascade tests one candidate at a time; Winnowing eliminates in bulk |

```python
from nono.agent import Agent, WinnowingAgent, Runner

gen = Agent(name="gen", instruction="Write a solution.", provider="google")
judge = Agent(name="judge", instruction="Score the solution 1-10.", provider="google")

win = WinnowingAgent(
    name="winnow",
    agent=gen,
    evaluator_agent=judge,
    n_candidates=6,
    cull_fraction=0.5,
    score_fn=lambda r: float(r.strip().split()[-1]) if r.strip()[-1].isdigit() else 5.0,
)

result = Runner(win).run("Solve this optimisation problem")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Generates candidates |
| `evaluator_agent` | (required) | Scores each candidate |
| `n_candidates` | `6` | Initial candidate count |
| `cull_fraction` | `0.5` | Fraction eliminated per round |
| `score_fn` | `len(r)` | `Callable[[str], float]` — numeric extractor |
| `result_key` | `None` | Optional state key |

### A76 — MixtureOfThoughtsAgent

**Pattern**: Mixture of Thoughts — parallel reasoning strategies. Runs the same agent with four different prompting strategies simultaneously (Chain-of-Thought, direct, step-back, devil's advocate), then a selector agent picks the best response.

**When to use**: uncertain tasks where the best prompting approach isn't known, high-stakes decisions benefiting from multiple reasoning styles, any scenario where strategy diversity improves quality.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `BestOfNAgent` | BestOfN repeats same prompt; MoT uses different strategies |
| `MixtureOfExpertsAgent` | MoE blends weighted experts; MoT selects one best strategy |
| `EnsembleAgent` | Ensemble aggregates all; MoT picks via selector agent |

```python
from nono.agent import Agent, MixtureOfThoughtsAgent, Runner

thinker = Agent(name="thinker", instruction="Answer the question.", provider="google")
selector = Agent(name="selector", instruction="Pick the best answer.", provider="google")

mot = MixtureOfThoughtsAgent(
    name="mot",
    agent=thinker,
    selector_agent=selector,
    result_key="best_thought",
)

result = Runner(mot).run("Should we adopt microservices?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates under different strategies |
| `selector_agent` | (required) | Agent that picks the best response |
| `result_key` | `None` | Optional state key |

### A77 — SimulatedAnnealingAgent

**Pattern**: Simulated Annealing (Kirkpatrick et al., 1983) — temperature-based acceptance of neighbouring solutions. Starts with a high temperature (accepting worse solutions to escape local optima) and gradually cools, eventually converging on an optimal or near-optimal solution.

**When to use**: optimisation with many local optima, creative exploration where initial freedom improves final quality, any task where escaping early good-but-not-great solutions matters.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `GeneticAlgorithmAgent` | GA uses population + crossover; SA perturbs single solution |
| `TabuSearchAgent` | Tabu forbids recent moves; SA probabilistically accepts worse moves |
| `MonteCarloAgent` | MCTS uses tree + UCB; SA uses temperature decay |

```python
from nono.agent import Agent, SimulatedAnnealingAgent, Runner

solver = Agent(name="solver", instruction="Improve the solution.", provider="google")

sa = SimulatedAnnealingAgent(
    name="sa",
    agent=solver,
    score_fn=lambda r: float(len(r)),
    max_iterations=10,
    initial_temp=1.0,
    cooling_rate=0.9,
)

result = Runner(sa).run("Optimise this API design")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that proposes neighbouring solutions |
| `score_fn` | (required) | `Callable[[str], float]` — scores solutions |
| `max_iterations` | `10` | Number of iterations |
| `initial_temp` | `1.0` | Starting temperature |
| `cooling_rate` | `0.9` | Temperature decay multiplier |
| `result_key` | `None` | Optional state key |

### A78 — TabuSearchAgent

**Pattern**: Tabu Search (Glover, 1986) — local search with memory. Keeps a tabu list of recently visited solutions (by hash) to prevent cycling. Each iteration generates a new neighbour; if it's already been visited, it's discarded. Tracks the global best across all iterations.

**When to use**: combinatorial optimisation, design exploration where revisiting old solutions wastes resources, any iterative search benefiting from visit memory.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SimulatedAnnealingAgent` | SA accepts worse moves probabilistically; Tabu forbids recent moves |
| `ReflexionAgent` | Reflexion stores lessons in text; Tabu stores solution hashes |
| `CompilerAgent` | Compiler optimises prompts; Tabu optimises solution space |

```python
from nono.agent import Agent, TabuSearchAgent, Runner

solver = Agent(name="solver", instruction="Propose a variant.", provider="google")

ts = TabuSearchAgent(
    name="tabu",
    agent=solver,
    score_fn=lambda r: float(len(r)),
    max_iterations=10,
    tabu_tenure=5,
)

result = Runner(ts).run("Design a load-balancing algorithm")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that proposes neighbours |
| `score_fn` | (required) | `Callable[[str], float]` — scores solutions |
| `max_iterations` | `10` | Number of search iterations |
| `tabu_tenure` | `5` | How long a solution stays tabu |
| `result_key` | `None` | Optional state key |

### A79 — ParticleSwarmAgent

**Pattern**: Particle Swarm Optimisation (Kennedy & Eberhart, 1995) — swarm intelligence. Multiple "particles" (agent calls) explore the solution space. Each particle tracks its personal best and is influenced by the global best. Velocities (prompt adjustments) are updated each iteration to balance exploration and exploitation.

**When to use**: continuous optimisation problems, hyperparameter tuning, any multi-agent search where personal and collective experience guide exploration.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `AntColonyAgent` | ACO uses pheromone trails; PSO uses personal/global best |
| `GeneticAlgorithmAgent` | GA uses crossover/mutation; PSO uses velocity updates |
| `DifferentialEvolutionAgent` | DE uses donor vectors; PSO uses personal/global attractors |

```python
from nono.agent import Agent, ParticleSwarmAgent, Runner

solver = Agent(name="particle", instruction="Propose a solution.", provider="google")

pso = ParticleSwarmAgent(
    name="pso",
    agent=solver,
    score_fn=lambda r: float(len(r)),
    n_particles=5,
    n_iterations=3,
)

result = Runner(pso).run("Find optimal database indexing strategy")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent per particle |
| `score_fn` | (required) | `Callable[[str], float]` — scores solutions |
| `n_particles` | `5` | Swarm size |
| `n_iterations` | `3` | Number of iterations |
| `result_key` | `None` | Optional state key |

### A80 — DifferentialEvolutionAgent

**Pattern**: Differential Evolution (Storn & Price, 1997) — population-based optimisation. Each generation: for each member, pick three others, create a donor by combining them, and replace the current member if the donor scores higher. Population converges over generations.

**When to use**: parameter tuning, solution refinement where combining existing solutions may produce better ones, any optimisation benefiting from population diversity.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `GeneticAlgorithmAgent` | GA uses random crossover/mutation; DE uses structured donor vectors |
| `ParticleSwarmAgent` | PSO tracks velocities; DE creates donors from population differences |
| `SimulatedAnnealingAgent` | SA perturbs one solution; DE maintains and evolves a population |

```python
from nono.agent import Agent, DifferentialEvolutionAgent, Runner

solver = Agent(name="evolve", instruction="Combine and improve.", provider="google")

de = DifferentialEvolutionAgent(
    name="de",
    agent=solver,
    score_fn=lambda r: float(len(r)),
    population_size=6,
    n_generations=3,
)

result = Runner(de).run("Optimise a recommendation algorithm")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that creates donor solutions |
| `score_fn` | (required) | `Callable[[str], float]` — scores solutions |
| `population_size` | `6` | Number of population members |
| `n_generations` | `3` | Number of generations |
| `result_key` | `None` | Optional state key |

### A81 — BayesianOptimizationAgent

**Pattern**: Bayesian Optimisation (Snoek et al., 2012) — sample-efficient optimisation using history to guide exploration. Each iteration, the agent reviews all previous attempts and their scores, then proposes the next most promising solution. An acquisition prompt biases toward unexplored regions.

**When to use**: expensive evaluations where each attempt must count, hyperparameter search, any scenario where past results should inform next steps.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SimulatedAnnealingAgent` | SA uses temperature; BO uses informed acquisition based on history |
| `CompilerAgent` | Compiler optimises prompts; BO optimises any measurable objective |
| `MultiArmedBanditAgent` | MAB selects from fixed arms; BO generates new candidates each step |

```python
from nono.agent import Agent, BayesianOptimizationAgent, Runner

proposer = Agent(name="bo", instruction="Suggest the next experiment.", provider="google")

bo = BayesianOptimizationAgent(
    name="bayesian",
    agent=proposer,
    score_fn=lambda r: float(len(r)),
    n_iterations=5,
)

result = Runner(bo).run("Find optimal learning rate for training")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that proposes next candidate |
| `score_fn` | (required) | `Callable[[str], float]` — scores candidates |
| `n_iterations` | `5` | Number of iterations |
| `result_key` | `None` | Optional state key |

### A82 — AnalogicalReasoningAgent

**Pattern**: Analogical Reasoning (Yasunaga et al., 2023) — the agent first self-generates relevant analogies or similar solved problems, then uses those analogies as context to solve the actual task. This mirrors how humans leverage past experience for novel problems.

**When to use**: novel problems where drawing parallels to known domains helps, creative problem solving, educational contexts, any task where analogies improve understanding.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `StepBackAgent` | StepBack abstracts to general principles; Analogical finds similar cases |
| `BufferOfThoughtsAgent` | Buffer stores reusable templates; Analogical generates fresh analogies |
| `SkillLibraryAgent` | SkillLibrary reuses cached solutions; Analogical draws structural parallels |

```python
from nono.agent import Agent, AnalogicalReasoningAgent, Runner

solver = Agent(name="solver", instruction="Solve using the analogies.", provider="google")

ar = AnalogicalReasoningAgent(
    name="analogy",
    agent=solver,
    result_key="analogy_result",
)

result = Runner(ar).run("How should we design fault tolerance for a satellite network?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for analogy generation and final solving |
| `n_analogies` | `3` | Number of analogies to generate |
| `result_key` | `None` | Optional state key |

### A83 — ThreadOfThoughtAgent

**Pattern**: Thread of Thought (Zhou et al., 2023) — the agent walks through the input segment by segment, building understanding incrementally, then distills the analysis into a final concise answer. Useful for long or complex inputs that benefit from systematic reading.

**When to use**: long document analysis, multi-paragraph comprehension, any task where systematic segment-by-segment processing improves accuracy.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `LeastToMostAgent` | LeastToMost decomposes by difficulty; Thread walks through by position |
| `SkeletonOfThoughtAgent` | Skeleton outlines then expands; Thread reads then distills |
| `ChainOfDensityAgent` | ChainOfDensity densifies summaries; Thread analyses then synthesises |

```python
from nono.agent import Agent, ThreadOfThoughtAgent, Runner

reader = Agent(name="reader", instruction="Analyse carefully.", provider="google")

tot = ThreadOfThoughtAgent(
    name="thread",
    agent=reader,
    result_key="thread_result",
)

result = Runner(tot).run("Analyse this 10-page legal contract: ...")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for segment walk-through and distillation |
| `result_key` | `None` | Optional state key |

### A84 — ExpertPromptingAgent

**Pattern**: Expert Prompting (Xu et al., 2023) — the agent auto-generates an expert identity description tailored to the task, then answers as that expert. This leverages the finding that LLMs perform better when given an appropriate expert persona.

**When to use**: domain-specific questions, any task where signalling expertise in the prompt improves quality, automatic persona construction.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `MultiPersonaAgent` | MultiPersona cycles through given personas; ExpertPrompting generates one |
| `RolePlayingAgent` | RolePlaying has instructor+assistant; ExpertPrompting constructs expert identity |
| `SelfDiscoverAgent` | SelfDiscover composes reasoning modules; ExpertPrompting constructs persona |

```python
from nono.agent import Agent, ExpertPromptingAgent, Runner

solver = Agent(name="solver", instruction="Answer as the expert.", provider="google")

ep = ExpertPromptingAgent(
    name="expert",
    agent=solver,
    result_key="expert_answer",
)

result = Runner(ep).run("What are the risks of quantum computing for RSA encryption?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates expert identity and answer |
| `result_key` | `None` | Optional state key |

### A85 — BufferOfThoughtsAgent

**Pattern**: Buffer of Thoughts (Yang et al., 2024) — maintains a reusable buffer of thought-templates from prior tasks. For each new task, retrieves the most relevant template, instantiates it, and solves. If no template matches, solves from scratch and distills a new template for the buffer.

**When to use**: recurring task types, any scenario where past reasoning patterns can be reused, knowledge accumulation across tasks.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SkillLibraryAgent` | SkillLibrary caches full solutions; Buffer stores abstract templates |
| `CacheAgent` | Cache stores exact answers; Buffer stores reasoning patterns |
| `ReflexionAgent` | Reflexion stores failure lessons; Buffer stores success templates |

```python
from nono.agent import Agent, BufferOfThoughtsAgent, Runner

solver = Agent(name="solver", instruction="Use the template to solve.", provider="google")

bot = BufferOfThoughtsAgent(
    name="buffer",
    agent=solver,
    thought_buffer=[],
    result_key="buffer_result",
)

result = Runner(bot).run("Calculate compound interest for 5 years at 7%")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that retrieves, instantiates, and solves |
| `thought_buffer` | `[]` | List of stored thought-templates |
| `result_key` | `None` | Optional state key |

### A86 — ChainOfAbstractionAgent

**Pattern**: Chain of Abstraction (Gao et al., 2024) — two-phase reasoning. Phase 1: the agent reasons about the problem using abstract placeholders instead of specific values. Phase 2: the agent grounds the abstract reasoning with specific facts and data. This separates reasoning structure from factual content.

**When to use**: complex reasoning tasks where separating logic from facts improves accuracy, multi-step problems with many specific values, any task where abstract planning before grounding helps.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `StepBackAgent` | StepBack abstracts to general principles; CoA uses placeholder tokens |
| `PlannerAgent` | Planner decomposes into steps; CoA reasons then grounds |
| `DemonstrateSearchPredictAgent` | DSP demonstrates+searches; CoA abstracts+grounds |

```python
from nono.agent import Agent, ChainOfAbstractionAgent, Runner

reasoner = Agent(name="reasoner", instruction="Reason and ground.", provider="google")

coa = ChainOfAbstractionAgent(
    name="coa",
    agent=reasoner,
    result_key="coa_result",
)

result = Runner(coa).run("Compare GDP growth rates of Japan and Germany 2020-2024")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for abstract reasoning and grounding |
| `result_key` | `None` | Optional state key |

### A87 — VerifierAgent

**Pattern**: Verifier / Solution Selection (Cobbe et al., 2021) — generates N candidate solutions in parallel, scores each with a verifier agent and `score_fn`, and returns the highest-scoring solution. Combines generation diversity with verification accuracy.

**When to use**: mathematical reasoning, code generation, any task where multiple attempts with verification outperform a single best-effort.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `BestOfNAgent` | BestOfN uses score_fn only; Verifier uses a separate verifier agent |
| `SelfConsistencyAgent` | SelfConsistency votes on frequency; Verifier scores on quality |
| `WinnowingAgent` | Winnowing eliminates iteratively; Verifier is single-round |

```python
from nono.agent import Agent, VerifierAgent, Runner

gen = Agent(name="gen", instruction="Solve the math problem.", provider="google")
ver = Agent(name="ver", instruction="Rate the solution 1-10.", provider="google")

v = VerifierAgent(
    name="verifier",
    generator_agent=gen,
    verifier_agent=ver,
    n_candidates=5,
    score_fn=lambda r: float(r.strip().split()[-1]) if r.strip()[-1].isdigit() else 5.0,
    result_key="verified",
)

result = Runner(v).run("What is 23 × 47?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `generator_agent` | (required) | Agent that generates candidate solutions |
| `verifier_agent` | (required) | Agent that evaluates each candidate |
| `n_candidates` | `5` | Number of parallel candidates |
| `score_fn` | `len(r)` | `Callable[[str], float]` — extracts score from verifier output |
| `result_key` | `None` | Optional state key to store scores |

### A88 — ProgOfThoughtAgent

**Pattern**: Program of Thought (Chen et al., 2023) — the agent generates executable Python code to solve the problem, then executes it in a sandboxed environment. The code output becomes the final answer. This separates reasoning (code generation) from computation (execution).

**When to use**: mathematical calculations, data transformations, any task where generating and running code is more reliable than verbal reasoning.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SelfRefineAgent` | SelfRefine iterates on text answers; PoT generates and runs code |
| `VerifierAgent` | Verifier scores text solutions; PoT executes code for ground truth |
| `PlannerAgent` | Planner decomposes into steps; PoT generates a program |

```python
from nono.agent import Agent, ProgOfThoughtAgent, Runner

coder = Agent(name="coder", instruction="Write Python code to solve.", provider="google")

pot = ProgOfThoughtAgent(
    name="pot",
    agent=coder,
    result_key="code_result",
)

result = Runner(pot).run("What is the sum of primes below 100?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates Python code |
| `result_key` | `None` | Optional state key |

### A89 — InnerMonologueAgent

**Pattern**: Inner Monologue (Huang et al., 2022) — closed-loop execution with environment feedback. The agent acts, receives feedback from a `feedback_fn`, and continues iterating until a success signal (containing `done_token`) or `max_steps` is reached. Models embodied agents and interactive systems.

**When to use**: interactive environments, robotics-style planning, any task requiring iterative action with external feedback, step-by-step debugging.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `LoopAgent` | Loop repeats unconditionally; InnerMonologue uses external feedback |
| `ReflexionAgent` | Reflexion stores lessons; InnerMonologue responds to real-time feedback |
| `AdaptivePlannerAgent` | Adaptive re-plans based on results; InnerMonologue reacts to environment |

```python
from nono.agent import Agent, InnerMonologueAgent, Runner

actor = Agent(name="actor", instruction="Take the next action.", provider="google")

im = InnerMonologueAgent(
    name="monologue",
    agent=actor,
    feedback_fn=lambda action: "DONE" if "correct" in action else "Try again",
    done_token="DONE",
    max_steps=5,
)

result = Runner(im).run("Navigate to the goal")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates actions |
| `feedback_fn` | (required) | `Callable[[str], str]` — returns environment feedback |
| `done_token` | `"DONE"` | String indicating task completion |
| `max_steps` | `10` | Maximum action steps |
| `result_key` | `None` | Optional state key |

### A90 — RolePlayingAgent

**Pattern**: Role-Playing / CAMEL (Li et al., 2023) — two agents alternate: an instructor gives directions and an assistant follows them. The conversation continues for N turns, building a collaborative dialog. Inspired by the CAMEL framework for cooperative AI.

**When to use**: collaborative task solving, tutoring, paired brainstorming, any scenario where structured instructor-assistant dialog produces better results.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `DebateAgent` | Debate is adversarial; RolePlaying is cooperative |
| `GroupChatAgent` | GroupChat has N agents + manager; RolePlaying has exactly 2 roles |
| `MultiPersonaAgent` | MultiPersona uses one agent in roles; RolePlaying uses two separate agents |

```python
from nono.agent import Agent, RolePlayingAgent, Runner

instructor = Agent(name="teacher", instruction="Guide the student.", provider="google")
assistant = Agent(name="student", instruction="Follow instructions.", provider="google")

rp = RolePlayingAgent(
    name="camel",
    instructor_agent=instructor,
    assistant_agent=assistant,
    n_turns=3,
)

result = Runner(rp).run("Design a REST API for a bookstore")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `instructor_agent` | (required) | Agent giving instructions |
| `assistant_agent` | (required) | Agent following instructions |
| `n_turns` | `3` | Number of conversation turns |
| `result_key` | `None` | Optional state key |

### A91 — GossipProtocolAgent

**Pattern**: Gossip Protocol (Demers et al., 1987) — epidemic information spreading. Each agent starts with its own knowledge (response to the task). In each round, agents randomly exchange and merge information with peers. Over rounds, knowledge propagates through the entire group until convergence.

**When to use**: distributed consensus without centralised coordination, information aggregation from independent sources, any scenario where organic information spread is natural.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ConsensusAgent` | Consensus uses a central judge; Gossip spreads peer-to-peer |
| `GroupChatAgent` | GroupChat has managed turns; Gossip has random pairwise exchanges |
| `DelphiMethodAgent` | Delphi uses a facilitator; Gossip is fully decentralised |

```python
from nono.agent import Agent, GossipProtocolAgent, Runner

a1 = Agent(name="node1", instruction="Share your knowledge.", provider="google")
a2 = Agent(name="node2", instruction="Share your knowledge.", provider="google")
a3 = Agent(name="node3", instruction="Share your knowledge.", provider="google")

gossip = GossipProtocolAgent(
    name="gossip",
    sub_agents=[a1, a2, a3],
    n_rounds=3,
)

result = Runner(gossip).run("What are the key risks of AI?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required, ≥2) | Participating agents |
| `n_rounds` | `3` | Number of gossip rounds |
| `result_key` | `None` | Optional state key |

### A92 — AuctionAgent

**Pattern**: Auction (Vickrey, 1961) — competitive self-selection. Each sub-agent receives the task and produces a bid explaining its suitability. A `bid_fn` scores each bid. The highest bidder wins and executes the full task.

**When to use**: multi-specialist routing where agents self-assess fitness, dynamic task allocation, any competitive selection scenario.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ContractNetAgent` | ContractNet bids on proposals; Auction bids on self-suitability |
| `RouterAgent` | Router uses LLM classification; Auction uses self-assessed bidding |
| `MixtureOfExpertsAgent` | MoE uses gating weights; Auction uses competitive bids |

```python
from nono.agent import Agent, AuctionAgent, Runner

fast = Agent(name="fast", instruction="I specialise in speed.", provider="google")
deep = Agent(name="deep", instruction="I specialise in thorough analysis.", provider="google")

auc = AuctionAgent(
    name="auction",
    sub_agents=[fast, deep],
    bid_fn=lambda name, resp: float(len(resp)),
    result_key="auction_result",
)

result = Runner(auc).run("Analyse this dataset")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Bidding agents |
| `bid_fn` | `len(response)` | `Callable[[str, str], float]` scores bids |
| `result_key` | `None` | Optional state key |

### A93 — DelphiMethodAgent

**Pattern**: Delphi Method (Dalkey & Helmer, 1963) — anonymous iterative expert consultation. Multiple expert agents answer independently per round, then a facilitator agent synthesises the anonymised opinions. This repeats for N rounds, guiding convergence toward informed consensus.

**When to use**: forecasting, policy decisions, any scenario where independent expert opinions need structured convergence without bias.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ConsensusAgent` | Consensus has a single round; Delphi iterates |
| `NominalGroupAgent` | NGT uses voting/ranking; Delphi uses facilitator synthesis |
| `GossipProtocolAgent` | Gossip is decentralised; Delphi has a central facilitator |

```python
from nono.agent import Agent, DelphiMethodAgent, Runner

e1 = Agent(name="expert1", instruction="Give your expert opinion.", provider="google")
e2 = Agent(name="expert2", instruction="Give your expert opinion.", provider="google")
fac = Agent(name="facilitator", instruction="Synthesise the opinions.", provider="google")

delphi = DelphiMethodAgent(
    name="delphi",
    experts=[e1, e2],
    facilitator=fac,
    n_rounds=3,
)

result = Runner(delphi).run("What will AI adoption look like in 5 years?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `experts` | (required) | List of expert agents |
| `facilitator` | (required) | Agent that synthesises each round |
| `n_rounds` | `3` | Number of Delphi rounds |
| `result_key` | `None` | Optional state key |

### A94 — NominalGroupAgent

**Pattern**: Nominal Group Technique (Delbecq et al., 1971) — structured group decision-making. Phase 1: agents silently generate ideas independently. Phase 2: ideas are shared in round-robin. Phase 3: agents rank all ideas. Phase 4: rankings are tallied to determine the group's priority order.

**When to use**: brainstorming with prioritisation, team decision-making, any scenario where structured voting prevents groupthink.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `VotingAgent` | Voting picks most frequent; NGT ranks and tallies all options |
| `DelphiMethodAgent` | Delphi iterates with a facilitator; NGT is a single structured cycle |
| `ConsensusAgent` | Consensus uses a judge; NGT uses mathematical rank aggregation |

```python
from nono.agent import Agent, NominalGroupAgent, Runner

a1 = Agent(name="member1", instruction="Generate ideas.", provider="google")
a2 = Agent(name="member2", instruction="Generate ideas.", provider="google")

ngt = NominalGroupAgent(
    name="ngt",
    sub_agents=[a1, a2],
    result_key="ngt_result",
)

result = Runner(ngt).run("What features should we build next quarter?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `sub_agents` | (required) | Participating agents |
| `result_key` | `None` | Optional state key |

### A95 — ActiveRetrievalAgent

**Pattern**: Active Retrieval / FLARE (Jiang et al., 2023) — the agent generates sentence by sentence; when confidence in a sentence drops below a threshold (`confidence_threshold`), a `retrieve_fn` is called to fetch supporting information before continuing. This ensures retrieval happens only when needed.

**When to use**: long-form generation where facts must be verified on-demand, knowledge-intensive QA, any task where selective retrieval is more efficient than always-retrieve.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `IterativeRetrievalAgent` | Iterative retrieves every step; Active only retrieves when unsure |
| `SubQuestionAgent` | SubQuestion decomposes and retrieves per sub-question; Active monitors confidence |
| `DemonstrateSearchPredictAgent` | DSP has fixed 3-stage pipeline; Active is adaptive |

```python
from nono.agent import Agent, ActiveRetrievalAgent, Runner

writer = Agent(name="writer", instruction="Write with inline retrieval.", provider="google")

ar = ActiveRetrievalAgent(
    name="flare",
    agent=writer,
    retrieve_fn=lambda q: f"Retrieved context for: {q}",
    confidence_threshold=0.5,
    max_steps=5,
)

result = Runner(ar).run("Write about the history of quantum computing")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that generates and signals confidence |
| `retrieve_fn` | (required) | `Callable[[str], str]` — retrieval function |
| `confidence_threshold` | `0.5` | Below this triggers retrieval |
| `max_steps` | `5` | Maximum generation steps |
| `result_key` | `None` | Optional state key |

### A96 — IterativeRetrievalAgent

**Pattern**: Iterative Retrieval / IRCoT (Trivedi et al., 2023) — interleaves Chain-of-Thought reasoning with retrieval at every step. Each iteration: retrieve relevant documents, then reason one step using the retrieved context. Continues for N steps.

**When to use**: multi-hop QA, complex research questions requiring multiple retrieval steps, any task where reasoning and retrieval must alternate.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ActiveRetrievalAgent` | Active retrieves selectively; Iterative retrieves every step |
| `SubQuestionAgent` | SubQuestion decomposes upfront; Iterative retrieves incrementally |
| `CumulativeReasoningAgent` | Cumulative proposes/verifies facts; Iterative retrieves external docs |

```python
from nono.agent import Agent, IterativeRetrievalAgent, Runner

reasoner = Agent(name="reasoner", instruction="Reason one step.", provider="google")

ir = IterativeRetrievalAgent(
    name="ircot",
    agent=reasoner,
    retrieve_fn=lambda q: f"Retrieved docs for: {q}",
    n_steps=3,
)

result = Runner(ir).run("Who influenced the person who invented the telephone?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for each reasoning step |
| `retrieve_fn` | (required) | `Callable[[str], str]` — retrieval function |
| `n_steps` | `3` | Number of retrieve-reason iterations |
| `result_key` | `None` | Optional state key |

### A97 — PromptChainAgent

**Pattern**: Prompt Chaining (Wu et al., 2022) — sequential prompt templates where each step's output feeds into the next step's `{previous}` variable. Define a list of prompt templates with `{input}` and `{previous}` placeholders.

**When to use**: multi-step transformations with explicit templates, translation pipelines, any sequential prompt series with known structure.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `SequentialAgent` | Sequential passes session state; PromptChain uses template variables |
| `PipelineParallelAgent` | Pipeline processes items through stages; PromptChain transforms one input |
| `FeedbackLoopAgent` | FeedbackLoop cycles; PromptChain is strictly sequential |

```python
from nono.agent import Agent, PromptChainAgent, Runner

worker = Agent(name="worker", instruction="Follow the template.", provider="google")

pc = PromptChainAgent(
    name="chain",
    agent=worker,
    templates=[
        "Summarise: {input}",
        "Translate to French: {previous}",
        "Make it formal: {previous}",
    ],
)

result = Runner(pc).run("The quick brown fox jumps over the lazy dog.")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that executes each template |
| `templates` | (required) | List of prompt templates with `{input}` and `{previous}` |
| `result_key` | `None` | Optional state key |

### A98 — HypothesisTestingAgent

**Pattern**: Hypothesis Testing (Popper, 1959 — falsificationism) — iterative scientific method. The agent generates hypotheses, designs experiments to test them (via `test_fn`), analyses results, and refines. Hypotheses that survive testing are kept; falsified ones are revised. Continues for N rounds.

**When to use**: scientific reasoning, root cause analysis, any task where generating and falsifying hypotheses converges on truth.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `CumulativeReasoningAgent` | Cumulative proposes+verifies facts; HypothesisTesting falsifies theories |
| `RedTeamAgent` | RedTeam attacks externally; HypothesisTesting self-falsifies |
| `SelfRefineAgent` | SelfRefine iterates on quality; HypothesisTesting iterates on truth |

```python
from nono.agent import Agent, HypothesisTestingAgent, Runner

scientist = Agent(name="sci", instruction="Generate and test hypotheses.", provider="google")

ht = HypothesisTestingAgent(
    name="hypothesis",
    agent=scientist,
    test_fn=lambda h: f"Experiment result for: {h}",
    n_rounds=3,
)

result = Runner(ht).run("Why is the server response time increasing?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for hypothesis generation and analysis |
| `test_fn` | (required) | `Callable[[str], str]` — runs experiments |
| `n_rounds` | `3` | Number of hypothesis-test cycles |
| `result_key` | `None` | Optional state key |

### A99 — SkillLibraryAgent

**Pattern**: Skill Library (Wang et al., 2023 — Voyager) — reusable skill accumulation. When a task matches a stored skill, the agent reuses it. When no match is found, the agent solves from scratch and distills the solution into a new reusable skill added to the library.

**When to use**: long-running agents that encounter recurring task types, knowledge accumulation, any scenario where learning from past tasks improves efficiency.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `BufferOfThoughtsAgent` | Buffer stores templates; SkillLibrary stores complete solutions |
| `CacheAgent` | Cache stores exact Q/A pairs; SkillLibrary stores transferable skills |
| `ReflexionAgent` | Reflexion stores failure lessons; SkillLibrary stores success solutions |

```python
from nono.agent import Agent, SkillLibraryAgent, Runner

solver = Agent(name="solver", instruction="Solve and learn.", provider="google")

sl = SkillLibraryAgent(
    name="skills",
    agent=solver,
    skill_library={},
    result_key="skill_result",
)

result = Runner(sl).run("Write a function to sort a list")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that solves and distills skills |
| `skill_library` | `{}` | Dictionary of stored skills |
| `result_key` | `None` | Optional state key |

### A100 — RecursiveCriticAgent

**Pattern**: Recursive Critic — multi-depth nested critique with revision. The agent produces a draft, a critic agent reviews it, and the agent revises. This process recurses to the specified depth, with each level providing deeper critique. The final revision benefits from progressively refined feedback.

**When to use**: high-stakes writing, legal/medical review, any content requiring multiple layers of quality control.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ProducerReviewerAgent` | Producer-Reviewer has flat iterations; RecursiveCritic nests deeply |
| `SelfRefineAgent` | SelfRefine uses self-critique; RecursiveCritic uses external critic |
| `RedTeamAgent` | RedTeam attacks adversarially; RecursiveCritic provides constructive critique |

```python
from nono.agent import Agent, RecursiveCriticAgent, Runner

writer = Agent(name="writer", instruction="Write and revise.", provider="google")
critic = Agent(name="critic", instruction="Provide detailed critique.", provider="google")

rc = RecursiveCriticAgent(
    name="deep_review",
    agent=writer,
    critic_agent=critic,
    depth=3,
)

result = Runner(rc).run("Draft a security policy document")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that writes and revises |
| `critic_agent` | (required) | Agent that critiques at each depth |
| `depth` | `2` | Number of critique layers |
| `result_key` | `None` | Optional state key |

### A101 — DemonstrateSearchPredictAgent

**Pattern**: Demonstrate-Search-Predict / DSP (Khattab et al., 2022) — a three-stage pipeline. Stage 1 (Demonstrate): generate few-shot examples. Stage 2 (Search): retrieve supporting passages using `retrieve_fn`. Stage 3 (Predict): answer using demonstrations + retrieved evidence.

**When to use**: knowledge-intensive QA, open-domain tasks where few-shot + retrieval improves accuracy, any task following a demo→search→predict pattern.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ActiveRetrievalAgent` | Active retrieves on low confidence; DSP always retrieves |
| `IterativeRetrievalAgent` | Iterative interleaves CoT+retrieval; DSP is 3 fixed stages |
| `SubQuestionAgent` | SubQuestion decomposes; DSP demonstrates then retrieves |

```python
from nono.agent import Agent, DemonstrateSearchPredictAgent, Runner

solver = Agent(name="dsp", instruction="Answer using demos and evidence.", provider="google")

dsp = DemonstrateSearchPredictAgent(
    name="dsp",
    agent=solver,
    retrieve_fn=lambda q: f"Evidence for: {q}",
    result_key="dsp_result",
)

result = Runner(dsp).run("When was the Eiffel Tower built?")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent for all three stages |
| `retrieve_fn` | (required) | `Callable[[str], str]` — retrieval function |
| `result_key` | `None` | Optional state key |

### A102 — DoubleLoopLearningAgent

**Pattern**: Double-Loop Learning (Argyris & Schön, 1978) — on failure, instead of just adjusting actions (single-loop), the agent questions and revises its underlying assumptions and mental model. If the initial answer passes the `quality_fn` threshold, it's returned. Otherwise, the agent re-examines its assumptions and tries again.

**When to use**: complex problem-solving where failures indicate wrong assumptions, strategic planning, any scenario where "trying harder" isn't enough and the approach itself needs revision.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `ReflexionAgent` | Reflexion accumulates lessons; DoubleLoop revises mental models |
| `SelfRefineAgent` | SelfRefine improves the answer; DoubleLoop questions the approach |
| `AdaptivePlannerAgent` | Adaptive re-plans steps; DoubleLoop re-examines assumptions |

```python
from nono.agent import Agent, DoubleLoopLearningAgent, Runner

thinker = Agent(name="thinker", instruction="Solve and reflect.", provider="google")

dl = DoubleLoopLearningAgent(
    name="double_loop",
    agent=thinker,
    quality_fn=lambda r: 0.8 if "good" in r else 0.3,
    threshold=0.7,
    max_loops=3,
)

result = Runner(dl).run("Design a scalable notification system")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that solves and reflects |
| `quality_fn` | (required) | `Callable[[str], float]` — evaluates answer quality |
| `threshold` | `0.7` | Quality threshold for acceptance |
| `max_loops` | `3` | Maximum assumption-revision cycles |
| `result_key` | `None` | Optional state key |

### A103 — AgendaAgent

**Pattern**: Agenda-based Planning (Allen & Ferguson, 1994) — priority-queue sub-goal decomposition. The agent decomposes the task into prioritised sub-goals (agenda items), then processes them in priority order. Each sub-goal can spawn new sub-goals. Processing continues until the agenda is empty or `max_steps` is reached.

**When to use**: complex tasks with many interdependent sub-goals, project planning, any hierarchical goal decomposition benefiting from prioritisation.

**Key differences from similar agents**:

| vs | Difference |
|---|---|
| `PlannerAgent` | Planner creates a fixed plan upfront; Agenda evolves dynamically |
| `PriorityQueueAgent` | PriorityQueue dispatches to sub-agents; Agenda decomposes into sub-goals |
| `AdaptivePlannerAgent` | Adaptive re-plans after each step; Agenda manages a priority queue |

```python
from nono.agent import Agent, AgendaAgent, Runner

planner = Agent(name="planner", instruction="Decompose and solve sub-goals.", provider="google")

agenda = AgendaAgent(
    name="agenda",
    agent=planner,
    max_steps=10,
    result_key="agenda_result",
)

result = Runner(agenda).run("Build a complete CI/CD pipeline for a Python project")
```

**Key parameters**:

| Parameter | Default | Description |
|---|---|---|
| `agent` | (required) | Agent that decomposes and solves sub-goals |
| `max_steps` | `10` | Maximum sub-goal processing steps |
| `result_key` | `None` | Optional state key |

### A104 — Nesting Deterministic Agents

All orchestration agents are `BaseAgent` subclasses — they compose freely:

```python
from nono.agent import SequentialAgent, ParallelAgent, LoopAgent, Runner

# Stage 1: gather perspectives in parallel
research = ParallelAgent(
    name="research",
    sub_agents=[tech, biz, ethics],
    result_key="perspectives",
)

# Stage 2: write + review in a loop
write_review = SequentialAgent(name="write_review", sub_agents=[writer, reviewer])
refine = LoopAgent(
    name="refine",
    sub_agents=[write_review],
    max_iterations=2,
)

# Complete pipeline: parallel research → iterative writing
full = SequentialAgent(
    name="full_pipeline",
    sub_agents=[research, refine],
)

runner = Runner(full)
result = runner.run("AI in healthcare 2026")
```

```
🤖 full_pipeline (SequentialAgent)
├── 🤖 research (ParallelAgent)
│   ├── 🤖 tech_analyst
│   ├── 🤖 biz_analyst
│   └── 🤖 ethics_analyst
└── 🤖 refine (LoopAgent, max=2)
    └── 🤖 write_review (SequentialAgent)
        ├── 🤖 writer
        └── 🤖 reviewer
```

---

## Part B — LLM Routing

In LLM routing, **an AI model decides** which agents to invoke and how. This shifts orchestration from design-time to runtime.

### B1 — RouterAgent (Single Mode)

The simplest case: the LLM picks **one** agent to handle the request.

```python
from nono.agent import Agent, RouterAgent, Runner

researcher = Agent(
    name="researcher",
    description="Finds factual information on any topic.",
    instruction="Research the topic and provide data with sources.",
    provider="google",
)
writer = Agent(
    name="writer",
    description="Writes clear, engaging prose.",
    instruction="Write polished, engaging content.",
    provider="google",
)
coder = Agent(
    name="coder",
    description="Writes and debugs Python code.",
    instruction="Write clean, well-documented Python code.",
    provider="google",
)

router = RouterAgent(
    name="assistant",
    provider="google",
    model="gemini-3-flash-preview",
    sub_agents=[researcher, writer, coder],
)

runner = Runner(router)

# LLM selects "researcher":
print(runner.run("What are the latest AI diagnostic tools for radiology?"))

# LLM selects "coder":
print(runner.run("Write a Python function to calculate BMI"))

# LLM selects "writer":
print(runner.run("Write an intro paragraph about healthcare AI"))
```

**How it works**: RouterAgent sends the user message + agent descriptions to the LLM. The LLM returns:

```json
{"mode": "single", "agents": ["researcher"]}
```

The router then delegates to that agent.

> **Critical**: always provide a meaningful `description` on each sub-agent. The routing LLM reads descriptions to make its decision.

### B2 — RouterAgent (Sequential Mode)

The LLM decides to run **multiple agents in sequence** — building a pipeline dynamically:

```python
runner = Runner(router)

# LLM returns: {"mode": "sequential", "agents": ["researcher", "writer"]}
result = runner.run("Research AI trends and write a blog post about them")
# → researcher gathers data → writer produces the article
```

The router internally builds an ephemeral `SequentialAgent` with the selected agents and runs it. The output of each agent feeds the next as `user_message`.

### B3 — RouterAgent (Parallel Mode)

The LLM decides to run agents **concurrently** — useful for gathering independent data:

```python
runner = Runner(router)

# LLM returns: {"mode": "parallel", "agents": ["researcher", "coder"]}
result = runner.run("Find Python best practices AND generate a linter configuration")
```

Internally builds an ephemeral `ParallelAgent`. All selected agents receive the same user message and run in threads (sync) or coroutines (async).

### B4 — RouterAgent (Loop Mode)

The LLM decides to **iterate** one agent for quality improvement:

```python
runner = Runner(router)

# LLM returns: {"mode": "loop", "agents": ["writer"], "max_iterations": 3}
result = runner.run("Write a perfect introduction — iterate until it shines")
```

Internally builds an ephemeral `LoopAgent` with the selected agent and the specified iteration count.

### B5 — Custom Routing Instructions

Add business rules to influence the LLM's routing decisions:

```python
router = RouterAgent(
    name="editorial_router",
    provider="google",
    sub_agents=[researcher, writer, coder, reviewer],
    routing_instruction=(
        "Rules:\n"
        "- For article requests: use sequential mode with [researcher, writer].\n"
        "- For code requests: use single mode with coder.\n"
        "- For review requests: use loop mode with reviewer (max 2 iterations).\n"
        "- For comparison research: use parallel mode with [researcher, coder]."
    ),
    temperature=0.0,  # low temperature = more deterministic routing
)
```

| Parameter | Type | Default | Purpose |
| --- | --- | --- | --- |
| `routing_instruction` | `str` | `""` | Business rules appended to the routing prompt |
| `temperature` | `float` | `0.0` | Controls LLM creativity in routing (lower = more predictable) |
| `max_iterations` | `int` | `3` | Default iteration count for loop mode |

### B6 — transfer_to_agent (Dynamic Delegation)

Unlike `RouterAgent` (separate orchestration layer), `transfer_to_agent` is a **tool embedded inside an LlmAgent**. The LLM decides **when** to delegate as part of its normal conversation:

```python
from nono.agent import Agent, Runner

# Specialists
diagnostics = Agent(
    name="diagnostics_expert",
    description="Expert in AI diagnostic tools and medical imaging.",
    instruction="Answer questions about AI diagnostic systems.",
    provider="google",
)
regulations = Agent(
    name="regulations_expert",
    description="Expert in healthcare regulations and compliance.",
    instruction="Answer questions about healthcare regulations.",
    provider="google",
)

# Coordinator — transfer_to_agent auto-registered because of sub_agents
coordinator = Agent(
    name="healthcare_assistant",
    instruction=(
        "You are a healthcare AI assistant. Answer general questions directly. "
        "Delegate to specialists for domain-specific questions."
    ),
    provider="google",
    sub_agents=[diagnostics, regulations],
)

runner = Runner(coordinator)

# Coordinator answers directly (general question):
print(runner.run("What is AI?"))

# Coordinator delegates to diagnostics_expert:
print(runner.run("How accurate are AI systems in detecting lung nodules on CT scans?"))

# Coordinator delegates to regulations_expert:
print(runner.run("What are the FDA requirements for AI medical devices?"))
```

**How it works internally**:

1. `sub_agents` triggers auto-registration of the `transfer_to_agent` tool
2. The LLM sees the tool schema: `{agent_name: str, message: str}`
3. The LLM calls the tool when it decides a specialist should handle the question
4. The framework runs the sub-agent and returns the result to the coordinator
5. The coordinator can use the result, delegate again, or answer directly

### B7 — RouterAgent vs transfer_to_agent

| Aspect | RouterAgent | transfer_to_agent |
| --- | --- | --- |
| **Architecture** | Separate orchestration layer | Tool inside an LlmAgent |
| **Decision point** | Before agent runs (pre-route) | During conversation (mid-stream) |
| **LLM cost** | 1 routing call + N agent calls | 1 coordinator call (includes tool calls) |
| **Multi-delegation** | One decision per request | Can delegate multiple times per turn |
| **Coordinator synthesis** | No — forwards directly | Yes — coordinator sees and uses results |
| **Mode selection** | 4 modes (single/seq/parallel/loop) | Always single delegation |
| **Control** | Explicit, testable, logging | Implicit (LLM decides) |

**Rule of thumb**:
- Use `RouterAgent` when you want **explicit, logged routing** with mode selection
- Use `transfer_to_agent` when you want **conversational delegation** where the coordinator synthesizes results

---

## Part C — Hybrid Orchestration

Hybrid patterns combine the **predictability** of deterministic pipelines with the **intelligence** of LLM routing. The developer controls the frame; the AI makes decisions within it.

### C1 — Workflow + tasker_node (Deterministic + AI Steps)

Add AI-powered steps inside a deterministic pipeline using `tasker_node()`:

```python
from nono.workflows import Workflow, tasker_node

flow = Workflow("content_pipeline")

# Step 1: deterministic preprocessing
flow.step("prepare", lambda s: {
    "clean_topic": s["topic"].strip().lower(),
    "timestamp": "2026-03-30",
})

# Step 2: AI classification (via TaskExecutor)
flow.step("classify", tasker_node(
    system_prompt="Classify the topic into one of: technology, business, ethics, policy. Return only the category.",
    input_key="clean_topic",
    output_key="category",
    provider="google",
    model="gemini-3-flash-preview",
))

# Step 3: AI content generation
flow.step("generate", tasker_node(
    system_prompt="Write a 200-word summary about this topic for a healthcare newsletter.",
    input_key="clean_topic",
    output_key="summary",
))

# Step 4: deterministic formatting
flow.step("format", lambda s: {
    "output": f"[{s['category'].upper()}] {s['summary']}"
})

flow.connect("prepare", "classify", "generate", "format")

result = flow.run(topic="  AI Diagnostics in Radiology  ")
print(result["output"][:200])
```

**Pattern**: deterministic → AI → AI → deterministic. The pipeline structure is fixed; only the AI steps produce dynamic output.

### C2 — Workflow + agent_node (Deterministic + Agent Steps)

Use `agent_node()` to embed full agents (with tools, memory, delegation) inside a workflow step:

```python
from nono.agent import Agent
from nono.workflows import Workflow, agent_node

researcher = Agent(
    name="researcher",
    provider="perplexity",    # use Perplexity for web search
    instruction="Research the topic and provide factual data with sources.",
)
writer = Agent(
    name="writer",
    provider="openai",        # use OpenAI for writing
    model="gpt-4o-mini",
    instruction="Write a clear article based on the provided research notes.",
)

flow = Workflow("multi_provider_pipeline")
flow.step("research", agent_node(researcher, input_key="topic", output_key="notes"))
flow.step("write", agent_node(writer, input_key="notes", output_key="article"))
flow.step("postprocess", lambda s: {"final": f"# {s['topic']}\n\n{s['article']}"})
flow.connect("research", "write", "postprocess")

result = flow.run(topic="AI in healthcare diagnostics 2026")
print(result["final"][:300])
```

**Key advantage**: each agent can use a **different provider** — Perplexity for research (web search), OpenAI for writing. The Workflow coordinates them deterministically.

### C3 — Workflow + RouterAgent (Deterministic Shell, Dynamic Core)

The most powerful hybrid: a deterministic Workflow wraps a `RouterAgent` that makes the intelligent decision:

```python
from nono.agent import Agent, RouterAgent
from nono.workflows import Workflow, agent_node, tasker_node

# Specialist agents
researcher = Agent(name="researcher", description="Finds factual information.", provider="google", instruction="Research the topic.")
writer = Agent(name="writer", description="Writes articles.", provider="google", instruction="Write an article.")
coder = Agent(name="coder", description="Writes Python code.", provider="google", instruction="Write Python code.")

# RouterAgent decides which specialist to use
router = RouterAgent(
    name="smart_router",
    provider="google",
    sub_agents=[researcher, writer, coder],
)

flow = Workflow("smart_pipeline")

# Step 1: deterministic preprocessing
flow.step("preprocess", lambda s: {"clean_input": s["request"].strip()})

# Step 2: LLM-powered routing (RouterAgent wrapped as a workflow step)
flow.step("ai_route", agent_node(router, input_key="clean_input", output_key="ai_result"))

# Step 3: deterministic quality check
def quality_gate(state: dict) -> dict:
    result = state.get("ai_result", "")
    return {
        "quality_score": min(len(result) // 10, 100),
        "passed": len(result) > 50,
    }

flow.step("quality_check", quality_gate)

# Step 4: deterministic formatting
flow.step("format", lambda s: {"output": f"[Score: {s['quality_score']}] {s['ai_result'][:500]}"})

flow.connect("preprocess", "ai_route", "quality_check", "format")

result = flow.run(request="Research AI diagnostic tools and write a summary")
print(result["output"][:200])
```

```
📋 smart_pipeline
├── ○ preprocess
├── ○ ai_route          ← RouterAgent decides internally
├── ○ quality_check
└── ○ format
```

**Pattern**: preprocess (deterministic) → route (LLM) → quality gate (deterministic) → format (deterministic). The developer controls the overall structure; the LLM controls which specialist runs.

### C4 — Workflow + Conditional Branch + AI Scoring

Combine `branch_if()` with AI-powered quality scoring for a deterministic review loop with an intelligent gate. The **branch step** is the loop re-entry point:

```python
from nono.workflows import Workflow, tasker_node, END

flow = Workflow("review_loop")

# Step 1: generate initial draft
flow.step("draft", tasker_node(
    system_prompt="Write a 200-word article about the given topic.",
    input_key="topic",
    output_key="article",
))

# Step 2: score + decide (this step has the branch, so it can loop)
flow.step("evaluate", tasker_node(
    system_prompt="Rate this article from 0 to 100 for quality. Return ONLY the number.",
    input_key="article",
    output_key="raw_score",
))

# Step 3: parse and route (branch step — can be revisited)
flow.step("route", lambda s: {"score": int(s.get("raw_score", "0").strip())})

# Step 4: finalize
flow.step("finalize", lambda s: {"status": "approved", "final": s["article"]})

# Step 5: revise (AI-powered improvement)
flow.step("revise", tasker_node(
    system_prompt="Improve this article. Make it more engaging and accurate.",
    input_key="article",
    output_key="article",  # overwrites the previous draft
))

# Flow: draft → evaluate → route, then branch
flow.connect("draft", "evaluate", "route")
flow.branch_if("route", lambda s: s["score"] >= 80, then="finalize", otherwise="revise")

# After revision, loop back to the branch step (route can be revisited)
flow.connect("revise", "route")

result = flow.run(topic="AI diagnostics in healthcare")
print(f"Status: {result['status']}, Score: {result.get('score')}")
```

```
📋 review_loop
├── ○ draft
├── ○ evaluate
├── ◆ route → finalize | revise
├── ○ finalize
└── ○ revise → route
```

**Pattern**: the loop structure is deterministic (branch + re-entry on the branch step), but the scoring and revision steps use AI. This gives you predictable control flow with intelligent evaluation.

### C5 — Router → Deterministic Pipelines

The `RouterAgent` selects between pre-built deterministic pipelines:

```python
from nono.agent import Agent, SequentialAgent, ParallelAgent, RouterAgent, Runner

# Pipeline A: code workflow
code_pipeline = SequentialAgent(
    name="code_pipeline",
    description="Writes, tests, and reviews Python code.",
    sub_agents=[coder, tester, reviewer],
)

# Pipeline B: article workflow
article_pipeline = SequentialAgent(
    name="article_pipeline",
    description="Researches, writes, and reviews articles.",
    sub_agents=[researcher, writer, reviewer],
)

# Pipeline C: multi-perspective analysis
analysis_pipeline = ParallelAgent(
    name="analysis_pipeline",
    description="Analyzes a topic from technology, business, and ethics angles.",
    sub_agents=[tech, biz, ethics],
    result_key="analyses",
)

# Router picks the right pipeline
router = RouterAgent(
    name="project_router",
    provider="google",
    sub_agents=[code_pipeline, article_pipeline, analysis_pipeline],
    routing_instruction=(
        "Route code requests to code_pipeline.\n"
        "Route writing requests to article_pipeline.\n"
        "Route analysis requests to analysis_pipeline."
    ),
)

runner = Runner(router)

# Routes to code_pipeline:
runner.run("Write a Python web scraper for healthcare news")

# Routes to article_pipeline:
runner.run("Write an article about AI in radiology")

# Routes to analysis_pipeline:
runner.run("Analyze the market for AI diagnostic tools")
```

```
🤖 project_router (RouterAgent)
├── 🤖 code_pipeline (SequentialAgent)
│   ├── 🤖 coder
│   ├── 🤖 tester
│   └── 🤖 reviewer
├── 🤖 article_pipeline (SequentialAgent)
│   ├── 🤖 researcher
│   ├── 🤖 writer
│   └── 🤖 reviewer
└── 🤖 analysis_pipeline (ParallelAgent)
    ├── 🤖 tech
    ├── 🤖 biz
    └── 🤖 ethics
```

### C6 — Agent Tool Wrapping a Workflow

An agent can invoke an entire Workflow through a `FunctionTool`:

```python
from nono.agent import Agent, Runner, FunctionTool
from nono.workflows import Workflow, tasker_node

# Build a review workflow
review_flow = Workflow("review")
review_flow.step("score", tasker_node(
    system_prompt="Rate this text 0-100 for quality. Return only the number.",
    input_key="text",
    output_key="score",
))
review_flow.step("classify", tasker_node(
    system_prompt="Classify the sentiment: positive, negative, neutral. Return one word.",
    input_key="text",
    output_key="sentiment",
))
review_flow.connect("score", "classify")

# Wrap as a tool
def review_article(article_text: str) -> str:
    """Run quality scoring and sentiment classification on an article."""
    result = review_flow.run(text=article_text)
    return f"Score: {result.get('score', 'N/A')}/100, Sentiment: {result.get('sentiment', 'N/A')}"

review_tool = FunctionTool(review_article, description="Review and classify an article.")

editor = Agent(
    name="editor",
    provider="google",
    instruction="You are an editor. Use the review tool to evaluate articles before publishing.",
    tools=[review_tool],
)

runner = Runner(editor)
print(runner.run("Review this article: AI is transforming healthcare diagnostics..."))
```

**Pattern**: the agent decides **when** to invoke the workflow; the workflow provides a **structured, deterministic** review process.

### C7 — Full Production Pattern

A realistic production pipeline that combines all three paradigms:

```python
from nono.agent import Agent, SequentialAgent, ParallelAgent, RouterAgent, Runner
from nono.workflows import Workflow, agent_node, tasker_node, END

# ── Layer 1: Specialist agents ──

researcher = Agent(name="researcher", description="Finds factual data.", provider="perplexity", instruction="Research the topic.")
writer = Agent(name="writer", description="Writes articles.", provider="openai", model="gpt-4o-mini", instruction="Write a polished article.")
reviewer = Agent(name="reviewer", description="Reviews for accuracy and style.", provider="google", instruction="Review and suggest improvements.")

# ── Layer 2: Deterministic agent pipeline ──

write_review = SequentialAgent(
    name="write_review",
    description="Writes and reviews content.",
    sub_agents=[writer, reviewer],
)

# ── Layer 3: Workflow with hybrid steps ──

flow = Workflow("production_pipeline")

# Deterministic: validate input
flow.step("validate", lambda s: {
    "valid": bool(s.get("topic", "").strip()),
    "topic": s.get("topic", "").strip(),
})

# LLM: research via agent
flow.step("research", agent_node(researcher, input_key="topic", output_key="notes"))

# LLM: write + review via agent pipeline
flow.step("write_review", agent_node(write_review, input_key="notes", output_key="article"))

# LLM: quality score
flow.step("score", tasker_node(
    system_prompt="Rate this article 0-100. Return only the number.",
    input_key="article",
    output_key="raw_score",
))

# Deterministic: parse score
flow.step("parse_score", lambda s: {"score": int(s.get("raw_score", "0").strip())})

# Deterministic: format output
flow.step("publish", lambda s: {"status": "published", "output": f"[{s['score']}/100] {s['article']}"})
flow.step("needs_work", lambda s: {"status": "needs_revision", "output": s["article"]})

# Routing
flow.connect("validate", "research", "write_review", "score", "parse_score")
flow.branch_if("parse_score", lambda s: s["score"] >= 75, then="publish", otherwise="needs_work")

result = flow.run(topic="AI-powered diagnostics in radiology 2026")
print(f"Status: {result['status']}, Score: {result.get('score')}")
```

```
📋 production_pipeline
├── ○ validate
├── ○ research            ← Perplexity agent
├── ○ write_review        ← OpenAI writer → Google reviewer (SequentialAgent)
├── ○ score               ← AI quality scoring
├── ◆ parse_score → publish | needs_work
├── ○ publish
└── ○ needs_work
```

**Architecture summary**:
- **Frame**: Workflow (deterministic steps, branching, state management)
- **Intelligence**: Agents (research, writing, review), TaskExecutor (scoring)
- **Routing**: `branch_if()` for deterministic quality gate
- **Multi-provider**: Perplexity for research, OpenAI for writing, Google for review and scoring

---

## Orchestration Lifecycle Hooks

Orchestration agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) support **lifecycle hooks** — callbacks that fire at key moments during multi-agent execution. These are the agent-level equivalent of Workflow's `on_start`, `on_end`, `on_before_step`, `on_after_step`, and `on_between_steps` hooks.

Hooks are registered via **fluent API** methods on any `BaseAgent` subclass. They enable logging, metrics, debugging, early termination, and external integrations without modifying agent logic.

### Available Hooks

| Hook | Signature | Fires when |
|---|---|---|
| `on_start` | `(orchestrator_name, session) -> None` | The orchestrator begins execution, before the first sub-agent |
| `on_end` | `(orchestrator_name, session, agents_executed) -> None` | The orchestrator finishes (normal or halted). `agents_executed` = count of sub-agents that ran |
| `on_between_agents` | `(completed_agent, next_agent \| None, session) -> Optional[bool]` | Between two sequential sub-agents. Return `False` to halt early. `next_agent` is `None` when the pipeline is about to end |
| `on_agent_start` | `(sub_agent_name, session) -> None` | Right before a sub-agent begins execution |
| `on_agent_end` | `(sub_agent_name, session, error) -> None` | Right after a sub-agent finishes. `error` is `None` on success, or the error message on failure |

All hooks are optional. Multiple hooks can be combined on the same agent.

### SequentialAgent Hooks

`SequentialAgent` fires **all five hooks**. This is the most complete scenario: sub-agents run one after another, so there is always a "between" moment.

```python
from nono.agent import Agent, SequentialAgent, Runner

log = []

step1 = Agent(name="fetch", instruction="Fetch data.", provider="google")
step2 = Agent(name="process", instruction="Process data.", provider="google")
step3 = Agent(name="report", instruction="Generate report.", provider="google")

pipeline = SequentialAgent(
    name="etl",
    sub_agents=[step1, step2, step3],
)

# Register all hooks via fluent API
pipeline \
    .on_start(lambda name, sess: log.append(f"START {name}")) \
    .on_end(lambda name, sess, n: log.append(f"END {name} ({n} agents)")) \
    .on_between_agents(lambda done, nxt, sess: log.append(f"{done} -> {nxt}")) \
    .on_agent_start(lambda name, sess: log.append(f"  >> {name}")) \
    .on_agent_end(lambda name, sess, err: log.append(f"  << {name} {'OK' if not err else err}"))

result = Runner(pipeline).run("Run ETL pipeline")

# log contains:
# ["START etl", "  >> fetch", "  << fetch OK", "fetch -> process",
#  "  >> process", "  << process OK", "process -> report",
#  "  >> report", "  << report OK", "END etl (3 agents)"]
```

**Early termination** — return `False` from `on_between_agents` to stop the pipeline:

```python
pipeline.on_between_agents(
    lambda done, nxt, sess: False if sess.state.get("abort") else None
)
```

### ParallelAgent Hooks

`ParallelAgent` fires **four hooks**: `on_start`, `on_end`, `on_agent_start`, and `on_agent_end`. It does **not** fire `on_between_agents` because agents run concurrently — there is no "between" moment.

```python
from nono.agent import Agent, ParallelAgent, Runner

metrics = {"started": 0, "completed": 0}

analyzer = Agent(name="analyzer", instruction="Analyse data.", provider="google")
summarizer = Agent(name="summarizer", instruction="Summarise data.", provider="google")

parallel = ParallelAgent(
    name="dual",
    sub_agents=[analyzer, summarizer],
)

parallel \
    .on_start(lambda name, sess: print(f"Parallel {name} starting")) \
    .on_agent_start(lambda name, sess: metrics.update(started=metrics["started"] + 1)) \
    .on_agent_end(lambda name, sess, err: metrics.update(completed=metrics["completed"] + 1)) \
    .on_end(lambda name, sess, n: print(f"All done: {n} agents"))

result = Runner(parallel).run("Analyse and summarise")
```

### LoopAgent Hooks

`LoopAgent` fires **all five hooks**, including `on_between_agents` between iterations. It is useful for tracking iteration progress and implementing dynamic exit conditions.

```python
from nono.agent import Agent, LoopAgent, Runner

iteration_count = 0

refiner = Agent(name="refiner", instruction="Refine the answer.", provider="google")

loop = LoopAgent(
    name="iterative",
    sub_agents=[refiner],
    max_iterations=5,
)

loop \
    .on_start(lambda name, sess: print("Loop starting")) \
    .on_between_agents(lambda done, nxt, sess: (
        # Stop after 3 iterations based on session state
        False if sess.state.get("quality_score", 0) > 0.9 else None
    )) \
    .on_end(lambda name, sess, n: print(f"Loop finished after {n} iterations"))

result = Runner(loop).run("Improve this answer iteratively")
```

### Comparison with Workflow Hooks

| Workflow Hook | Agent Hook | Notes |
|---|---|---|
| `on_start(state)` | `on_start(name, session)` | Same purpose. Agent version receives orchestrator name + session |
| `on_end(state)` | `on_end(name, session, count)` | Agent version also receives the count of executed agents |
| `on_between_steps(prev, next, state, attempt)` | `on_between_agents(done, next, session)` | Both support `return False` to halt. Agent version has no `attempt` (agents don't have built-in retry) |
| `on_before_step(name, state, attempt)` | `on_agent_start(name, session)` | Same purpose. Agent version has no `attempt` counter |
| `on_after_step(name, state, attempt)` | `on_agent_end(name, session, error)` | Agent version includes error information |
| `on_error(name, error, state, attempt)` | *(via `on_agent_end` error param)* | Agent errors are reported through `on_agent_end` rather than a separate hook |
| `on_retry(name, state, attempt)` | *(not applicable)* | Agents don't have built-in step-level retry |

---

## Decision Matrix

| You want to... | Paradigm | Implementation |
| --- | --- | --- |
| Fixed pipeline (A → B → C) | Deterministic | `Workflow.connect()` or `SequentialAgent` |
| Conditional routing on data | Deterministic | `Workflow.branch_if()` |
| Fan-out / gather (functions) | Deterministic | `Workflow.parallel_step()` |
| Fan-out / gather (agents) | Deterministic | `ParallelAgent` |
| Repeat while condition | Deterministic | `Workflow.loop_step()` |
| Iterative refinement (agents) | Deterministic | `LoopAgent` or Workflow loop via `branch()` |
| Fan-out + reduce (agents) | Deterministic | `MapReduceAgent` |
| Multi-agent consensus + judge | Deterministic | `ConsensusAgent` |
| Produce-review loop until approval | Deterministic | `ProducerReviewerAgent` |
| Adversarial debate + verdict | Deterministic | `DebateAgent` |
| Cost-efficient fallback chain | Deterministic | `EscalationAgent` |
| LLM supervisor (delegate + evaluate) | LLM Routing | `SupervisorAgent` |
| Majority vote (no LLM judge) | Deterministic | `VotingAgent` |
| Peer-to-peer handoff (full control transfer) | Deterministic | `HandoffAgent` |
| N-agent group chat (managed turns) | Deterministic | `GroupChatAgent` |
| Multi-level hierarchy (LLM manager + departments) | LLM Routing | `HierarchicalAgent` |
| Pre/post validation with retry | Deterministic | `GuardrailAgent` |
| Best-of-N sampling (pick best by score) | Deterministic | `BestOfNAgent` |
| Batch item processing (concurrent) | Deterministic | `BatchAgent` |
| Progressive cascade (cost-aware quality) | Deterministic | `CascadeAgent` |
| BFS multi-path reasoning (tree exploration) | Deterministic | `TreeOfThoughtsAgent` |
| LLM decomposes task into dependency-aware steps | LLM Routing | `PlannerAgent` |
| Break complex question into sub-questions | LLM Routing | `SubQuestionAgent` |
| Filter context/history before delegation | Deterministic | `ContextFilterAgent` |
| Self-correct with persistent memory | Deterministic | `ReflexionAgent` |
| Wait-for-all barrier | Deterministic | `Workflow.join()` |
| State persistence / resume | Deterministic | `Workflow.enable_checkpoints()` + `run(resume=True)` |
| Pipeline from config file | Deterministic | `load_workflow("file.yaml")` |
| LLM picks which agent | LLM Routing | `RouterAgent` (single mode) |
| LLM builds a pipeline | LLM Routing | `RouterAgent` (sequential/parallel/loop modes) |
| Mid-conversation delegation | LLM Routing | `transfer_to_agent` |
| Fixed pipeline + one smart step | Hybrid | `Workflow` + `tasker_node()` or `agent_node()` |
| Fixed frame + dynamic routing inside | Hybrid | `Workflow` + `agent_node(RouterAgent)` |
| Pre-built pipelines selected by LLM | Hybrid | `RouterAgent` → `SequentialAgent` sub-agents |
| Agent invokes structured process | Hybrid | Agent + `FunctionTool` wrapping a `Workflow` |
| Production system | Hybrid | Full pattern (C7) |

### When to use Workflow vs Agent orchestration

| Criterion | Workflow | Agent orchestration |
| --- | --- | --- |
| **Topology** | Explicit DAG (steps + edges + branches) | Implicit via nesting |
| **State** | Shared `dict` — each step returns merge updates | `Session` — events + state + shared_content |
| **Routing** | `branch()` / `branch_if()` (deterministic) | `RouterAgent` (LLM) |
| **Parallelism** | `parallel_step()` (threads) | `ParallelAgent` (threads or asyncio) |
| **Loops** | `loop_step()` (condition + max) | `LoopAgent` |
| **Join / barrier** | `join()` (explicit wait-for-all) | Implicit in `ParallelAgent` |
| **Checkpointing** | `enable_checkpoints()` / `run(resume=True)` | Not built-in |
| **Declarative** | `load_workflow()` (JSON/YAML) | Python only |
| **Dynamic manipulation** | `insert_before`, `remove_step`, `swap_steps` | Not supported |
| **Streaming** | `stream()` yields `(step_name, state)` | `Runner.stream()` yields `Event` objects |

**Best practice**: use Workflow when you need explicit branching, parallel steps, loops, checkpointing, or a clear state-machine structure. Use Agent orchestration when you need LLM routing, multi-turn conversations, or composable agent nesting.

---

## Tracing Across Paradigms

All three paradigms support the same `TraceCollector` for unified observability:

```python
from nono import TraceCollector
from nono.agent import Runner
from nono.workflows import Workflow, tasker_node

collector = TraceCollector()

# Deterministic workflow with AI steps
flow = Workflow("traced_pipeline")
flow.step("classify", tasker_node(
    system_prompt="Classify the topic.",
    input_key="topic",
    output_key="category",
))
result = flow.run(trace_collector=collector, topic="AI diagnostics")

# Agent execution
runner = Runner(pipeline, trace_collector=collector)
runner.run("AI in healthcare")

# Unified summary across all modules
collector.print_summary()
```

| Module | trace_collector parameter |
| --- | --- |
| `Workflow.run()` | `trace_collector=collector` |
| `Workflow.run_async()` | `trace_collector=collector` |
| `Workflow.stream()` | `trace_collector=collector` |
| `Runner(agent, trace_collector=)` | `trace_collector=collector` |
| `TaskExecutor.execute()` | `trace_collector=collector` |

> See [README_events_tracking.md](../agent/README_events_tracking.md) for detailed tracing API.

---

## Visualization

Both Workflow and Agent trees render as ASCII diagrams:

```python
from nono import draw, draw_workflow, draw_agent

# Workflow visualization
flow = Workflow("pipeline")
flow.step("extract", extract)
flow.step("transform", transform)
flow.step("load", load)
flow.connect("extract", "transform", "load")
print(draw_workflow(flow))
# 📋 pipeline
# ├── ○ extract
# ├── ○ transform
# └── ○ load

# Agent visualization
from nono.agent import SequentialAgent, ParallelAgent
pipeline = SequentialAgent(name="pipeline", sub_agents=[researcher, writer])
print(draw_agent(pipeline))
# 🤖 pipeline (SequentialAgent)
# ├── 🤖 researcher
# └── 🤖 writer

# Auto-detect (works with both)
print(draw(flow))
print(draw(pipeline))
```

> See [Part 6 of the Step-by-Step Guide](README_guide.md) for more visualization examples.

---

## FAQ

**Q: Can I mix Workflow and Agent orchestration in the same pipeline?**
A: Yes. Use `agent_node()` to embed agents inside workflows, or wrap workflows in `FunctionTool` for use inside agents. See Part C.

**Q: When should I use `branch_if()` vs `RouterAgent`?**
A: Use `branch_if()` when the condition can be computed from state data (scores, flags, counts). Use `RouterAgent` when the condition requires semantic understanding ("is this about code or writing?").

**Q: Can RouterAgent select between SequentialAgent sub-agents?**
A: Yes. Any `BaseAgent` subclass works as a RouterAgent sub-agent, including `SequentialAgent`, `ParallelAgent`, `LoopAgent`, and other `RouterAgent` instances.

**Q: How do I debug routing decisions?**
A: Use `runner.stream()` and watch for `AGENT_TRANSFER` events:

```python
for event in runner.stream("My request"):
    if event.event_type.value == "agent_transfer":
        print(f"Mode: {event.data['mode']}, Target: {event.data['target_agent']}")
```

**Q: Can each agent use a different provider?**
A: Yes. Each `Agent` has its own `provider` and `model` parameters. A Workflow using `agent_node()` can mix Perplexity (research), OpenAI (writing), and Google (routing).

**Q: What happens if RouterAgent can't parse the LLM response?**
A: It falls back to single mode with the first sub-agent. Set `temperature=0.0` for more reliable routing.

---

## See Also

- [README_guide.md](../README_guide.md) — Step-by-step guide (Parts 1–7)
- [README_orchestration.md](../agent/README_orchestration.md) — Agent orchestration API reference
- [README_events_tracking.md](../agent/README_events_tracking.md) — Tracing and observability
- [README_workflow.md](README_workflow.md) — Workflow API reference (if available)
