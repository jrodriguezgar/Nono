"""
Decision Wizard — choose the right orchestration pattern.

Interactive and programmatic helper that guides users through the Complexity
Ladder (Level 0–5) by asking a short series of yes/no questions.  Returns a
concrete recommendation with the Nono class to use and a code snippet.

Usage (programmatic):
    from nono.wizard import recommend, recommend_interactive

    # Non-interactive — pass answers as a dict
    rec = recommend({
        "single_call_enough": False,
        "needs_tools": True,
        "needs_multiple_steps": True,
        "needs_branching": False,
        "needs_semantic_routing": False,
    })
    print(rec.summary)
    print(rec.snippet)

    # Interactive — asks questions on stdin
    rec = recommend_interactive()

Usage (CLI):
    nono wizard            # interactive mode
    nono wizard --json     # interactive, JSON output

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("Nono.Wizard")


__all__ = [
    "Recommendation",
    "Question",
    "QUESTIONS",
    "recommend",
    "recommend_interactive",
    "suggest_next",
    "suggest_simpler",
    "complexity_for_agent",
    "ComplexityBudget",
    "audit_agent_tree",
]


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass(frozen=True)
class Recommendation:
    """Result of the decision wizard.

    Attributes:
        level: Complexity Ladder level (0–5).
        level_name: Human-readable level label.
        pattern: Recommended pattern or class name.
        summary: One-paragraph explanation of why this level fits.
        snippet: Ready-to-run Python code example.
        next_if: Hint on when to escalate to the next level.
    """

    level: int
    level_name: str
    pattern: str
    summary: str
    snippet: str
    next_if: str

    def as_dict(self) -> Dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "level": self.level,
            "level_name": self.level_name,
            "pattern": self.pattern,
            "summary": self.summary,
            "snippet": self.snippet,
            "next_if": self.next_if,
        }

    def as_json(self) -> str:
        """Serialise to indented JSON."""
        return json.dumps(self.as_dict(), indent=2)


@dataclass(frozen=True)
class Question:
    """A single yes/no question in the wizard flow.

    Attributes:
        key: Identifier used in the answers dict.
        text: The question shown to the user.
        hint: Optional help text shown after the question.
    """

    key: str
    text: str
    hint: str = ""


# ============================================================================
# QUESTIONS
# ============================================================================

QUESTIONS: List[Question] = [
    Question(
        key="single_call_enough",
        text="Can a single, well-crafted LLM prompt solve your task?",
        hint=(
            "Most tasks can be solved with a single call after "
            "good prompt engineering. Try it first."
        ),
    ),
    Question(
        key="needs_tools",
        text="Does the task require external tools, APIs, or data retrieval?",
        hint=(
            "If the LLM needs to call functions, search databases, "
            "or access live data, you need an augmented LLM (Level 1)."
        ),
    ),
    Question(
        key="needs_multiple_steps",
        text="Does the task have distinct steps that should execute in sequence?",
        hint=(
            "Examples: research → write → review, or extract → transform → load. "
            "If yes, you need a workflow or SequentialAgent (Level 2)."
        ),
    ),
    Question(
        key="needs_branching",
        text="Do you need conditional logic, parallelism, or iteration?",
        hint=(
            "Examples: if score > 80 then publish else revise; "
            "fan-out to 3 sources in parallel; retry until quality > 0.9."
        ),
    ),
    Question(
        key="needs_semantic_routing",
        text="Do routing decisions require the LLM to understand intent?",
        hint=(
            "If a human could write an if/else for the routing, use branch_if(). "
            "If routing requires understanding \"is this about billing or tech support?\", "
            "you need LLM routing (Level 4)."
        ),
    ),
]


# ============================================================================
# RECOMMENDATIONS DATABASE
# ============================================================================

_RECOMMENDATIONS: Dict[int, Recommendation] = {
    0: Recommendation(
        level=0,
        level_name="Single LLM call",
        pattern="TaskExecutor.execute()",
        summary=(
            "Your task can be solved with a single, well-crafted prompt. "
            "This is the simplest, cheapest, and fastest approach. "
            "Optimise the prompt and add few-shot examples before considering "
            "anything more complex."
        ),
        snippet="""\
from nono.tasker import TaskExecutor

executor = TaskExecutor(provider="google")
result = executor.execute(
    "Summarise this text in 3 bullet points: ...",
    system_prompt="You are a concise summariser.",
)
print(result)""",
        next_if=(
            "Escalate to Level 1 if the LLM needs to call external tools, "
            "fetch live data, or perform actions beyond text generation."
        ),
    ),
    1: Recommendation(
        level=1,
        level_name="Augmented LLM",
        pattern="Agent + @tool / FunctionTool",
        summary=(
            "Your task needs an LLM augmented with tools — function calling, "
            "data retrieval, or external APIs. A single Agent with tools "
            "handles this without orchestration overhead."
        ),
        snippet="""\
from nono.agent import Agent, Runner, tool

@tool
def search_db(query: str) -> str:
    \"\"\"Search the internal database.\"\"\"
    return f"Results for: {query}"

agent = Agent(
    name="assistant",
    provider="google",
    instruction="Help the user. Use search_db when you need data.",
    tools=[search_db],
)
result = Runner(agent).run("Find sales data for Q1 2026")
print(result)""",
        next_if=(
            "Escalate to Level 2 if the task has distinct sequential steps "
            "where each step's output feeds the next."
        ),
    ),
    2: Recommendation(
        level=2,
        level_name="Simple workflow",
        pattern="Workflow (linear) / SequentialAgent",
        summary=(
            "Your task decomposes into distinct sequential steps. "
            "Use a linear Workflow for function-based pipelines or "
            "SequentialAgent for agent-based pipelines. Both are fully "
            "deterministic and easy to debug."
        ),
        snippet="""\
from nono.workflows import Workflow

def research(state: dict) -> dict:
    return {"notes": f"Research about {state['topic']}"}

def write(state: dict) -> dict:
    return {"article": f"Article based on: {state['notes']}"}

flow = Workflow("pipeline")
flow.step("research", research)
flow.step("write", write)
flow.connect("research", "write")

result = flow.run(topic="AI in healthcare")
print(result["article"])""",
        next_if=(
            "Escalate to Level 3 if you need conditional branching, "
            "parallel execution, or loop-until-condition logic."
        ),
    ),
    3: Recommendation(
        level=3,
        level_name="Branching workflow",
        pattern="branch_if() / parallel_step() / loop_step()",
        summary=(
            "Your task needs conditional routing, parallel fan-out, or "
            "iterative refinement — but the decisions can be expressed as "
            "code (not semantic understanding). Use Workflow branching "
            "primitives or deterministic orchestration agents."
        ),
        snippet="""\
from nono.workflows import Workflow

flow = Workflow("review_loop")
flow.step("draft", lambda s: {"text": "draft", "score": 60})
flow.step("publish", lambda s: {"status": "published"})
flow.step("revise", lambda s: {"text": s["text"] + " [revised]", "score": 85})

flow.connect("draft")
flow.score_gate("draft", "score", 80, then="publish", otherwise="revise")
flow.connect("revise", "draft")  # loop back

result = flow.run()
print(result["status"])""",
        next_if=(
            "Escalate to Level 4 if routing decisions require the LLM "
            "to understand user intent or content semantics."
        ),
    ),
    4: Recommendation(
        level=4,
        level_name="LLM routing",
        pattern="RouterAgent / transfer_to_agent",
        summary=(
            "Your routing decisions require semantic understanding — "
            "the LLM must read the input and decide which specialist to "
            "invoke. Use RouterAgent for explicit mode selection or "
            "transfer_to_agent for conversational delegation."
        ),
        snippet="""\
from nono.agent import Agent, RouterAgent, Runner

researcher = Agent(name="researcher", description="Finds facts.",
                   instruction="Research the topic.", provider="google")
coder = Agent(name="coder", description="Writes Python.",
              instruction="Write clean code.", provider="google")

router = RouterAgent(
    name="router",
    provider="google",
    sub_agents=[researcher, coder],
)

result = Runner(router).run("Write a Python function to calculate BMI")
print(result)""",
        next_if=(
            "Escalate to Level 5 only if you have measured evidence that "
            "these patterns are insufficient — e.g., you need MCTS search, "
            "evolutionary optimisation, or distributed saga transactions."
        ),
    ),
    5: Recommendation(
        level=5,
        level_name="Advanced pattern",
        pattern="TreeOfThoughts / MonteCarlo / Saga / etc.",
        summary=(
            "Your task requires a specialised orchestration pattern. "
            "Before using this level, verify with measurements that "
            "Levels 0–4 are insufficient. Advanced patterns add "
            "significant latency, cost, and debugging complexity."
        ),
        snippet="""\
# Example: TreeOfThoughtsAgent for multi-path reasoning
from nono.agent import Agent, TreeOfThoughtsAgent, Runner

thinker = Agent(name="thinker", instruction="Propose a solution.",
                provider="google")

tot = TreeOfThoughtsAgent(
    name="tot",
    agent=thinker,
    evaluate_fn=lambda r: 1.0 if len(r) > 100 else 0.3,
    n_branches=3,
    beam_width=2,
    max_depth=3,
)

result = Runner(tot).run("Design a caching strategy for a real-time API")
print(result)""",
        next_if=(
            "You are at the highest level. If this still doesn't work, "
            "consider combining patterns (Hybrid orchestration, Part C) "
            "or revisiting the problem decomposition."
        ),
    ),
}


# ============================================================================
# DECISION ENGINE
# ============================================================================


def recommend(answers: Dict[str, bool]) -> Recommendation:
    """Return a recommendation based on pre-supplied answers.

    The answers dict maps question keys to boolean values.  Questions
    are evaluated in order; the first ``True`` that moves down the
    ladder determines the final level.

    Args:
        answers: Mapping of ``Question.key`` → ``bool``.

    Returns:
        The recommended level and pattern.

    Example:
        >>> rec = recommend({"single_call_enough": True})
        >>> rec.level
        0
    """
    if answers.get("single_call_enough", False):
        return _RECOMMENDATIONS[0]

    if not answers.get("needs_tools", False) \
       and not answers.get("needs_multiple_steps", False):
        return _RECOMMENDATIONS[0]

    if answers.get("needs_tools", False) \
       and not answers.get("needs_multiple_steps", False):
        return _RECOMMENDATIONS[1]

    if not answers.get("needs_branching", False):
        return _RECOMMENDATIONS[2]

    if not answers.get("needs_semantic_routing", False):
        return _RECOMMENDATIONS[3]

    return _RECOMMENDATIONS[4]


def recommend_interactive(
    *,
    output: Callable[[str], None] = lambda s: print(s),
    input_fn: Callable[[str], str] = input,
) -> Recommendation:
    """Run the wizard interactively on stdin/stdout.

    Args:
        output: Callable to display text (default: ``print``).
        input_fn: Callable to read user input (default: ``input``).

    Returns:
        The recommended level and pattern.
    """
    output("")
    output("=" * 60)
    output("  Nono Decision Wizard")
    output("  Find the right orchestration level for your task")
    output("=" * 60)
    output("")

    answers: Dict[str, bool] = {}

    for q in QUESTIONS:
        output(f"  {q.text}")

        if q.hint:
            output(f"  \033[90m{q.hint}\033[0m")

        while True:
            reply = input_fn("  → [y/N]: ").strip().lower()

            if reply in ("", "n", "no"):
                answers[q.key] = False
                break

            if reply in ("y", "yes", "s", "si"):
                answers[q.key] = True
                break

            output("  Please answer y or n.")

        output("")

        # Short-circuit: if they said yes to single_call_enough, stop
        if q.key == "single_call_enough" and answers[q.key]:
            break

        # Short-circuit: no tools and no multi-step → level 0
        if q.key == "needs_multiple_steps" \
           and not answers.get("needs_tools", False) \
           and not answers.get("needs_multiple_steps", False):
            break

        # Short-circuit: tools but no multi-step → level 1
        if q.key == "needs_multiple_steps" \
           and answers.get("needs_tools", False) \
           and not answers.get("needs_multiple_steps", False):
            break

        # Short-circuit: multi-step but no branching → level 2
        if q.key == "needs_branching" and not answers.get("needs_branching", False):
            break

        # Short-circuit: branching but no semantic routing → level 3
        if q.key == "needs_semantic_routing" \
           and not answers.get("needs_semantic_routing", False):
            break

    rec = recommend(answers)

    output("-" * 60)
    output(f"  ✓ Recommendation: Level {rec.level} — {rec.level_name}")
    output(f"    Pattern: {rec.pattern}")
    output("")
    output(f"  {rec.summary}")
    output("")
    output("  Example:")
    for line in rec.snippet.splitlines():
        output(f"    {line}")
    output("")
    output(f"  Next level if: {rec.next_if}")
    output("-" * 60)

    return rec


def get_recommendation(level: int) -> Optional[Recommendation]:
    """Return the recommendation for a specific complexity level.

    Args:
        level: Complexity Ladder level (0–5).

    Returns:
        Recommendation or ``None`` if *level* is out of range.
    """
    return _RECOMMENDATIONS.get(level)


def list_levels() -> List[Recommendation]:
    """Return all complexity levels in order.

    Returns:
        List of all 6 recommendations (levels 0–5).
    """
    return [_RECOMMENDATIONS[i] for i in range(6)]


# ============================================================================
# AGENT COMPLEXITY MAP
# ============================================================================

# Maps every agent class name to its Complexity Ladder level.
# Level 0 = single call, Level 5 = advanced specialist pattern.
_AGENT_COMPLEXITY: Dict[str, int] = {
    # Level 0 — no orchestration
    "TaskExecutor": 0,
    # Level 1 — augmented LLM
    "Agent": 1,
    "LlmAgent": 1,
    # Level 2 — simple workflow / sequential
    "SequentialAgent": 2,
    "EscalationAgent": 2,
    "BatchAgent": 2,
    "TimeoutAgent": 2,
    "CacheAgent": 2,
    "BudgetAgent": 2,
    # Level 3 — branching / parallel / loop
    "ParallelAgent": 3,
    "LoopAgent": 3,
    "MapReduceAgent": 3,
    "PriorityQueueAgent": 3,
    "LoadBalancerAgent": 3,
    "EnsembleAgent": 3,
    "ProducerReviewerAgent": 3,
    "GuardrailAgent": 3,
    "BestOfNAgent": 3,
    "CascadeAgent": 3,
    "ContextFilterAgent": 3,
    "SkeletonOfThoughtAgent": 3,
    "LeastToMostAgent": 3,
    "CheckpointableAgent": 3,
    # Level 4 — LLM routing / semantic decisions
    "RouterAgent": 4,
    "HandoffAgent": 4,
    "GroupChatAgent": 4,
    "SupervisorAgent": 4,
    "HierarchicalAgent": 4,
    "VotingAgent": 4,
    "ConsensusAgent": 4,
    "DebateAgent": 4,
    "DynamicFanOutAgent": 4,
    "SwarmAgent": 4,
    "SubQuestionAgent": 4,
    "PlannerAgent": 4,
    "AdaptivePlannerAgent": 4,
    "ReflexionAgent": 4,
    "CoVeAgent": 4,
    "MemoryConsolidationAgent": 4,
    "SocraticAgent": 4,
    "MetaOrchestratorAgent": 4,
    "CurriculumAgent": 4,
    "ShadowAgent": 4,
    "HumanInputAgent": 4,
    # Level 5 — advanced / specialist
    "TreeOfThoughtsAgent": 5,
    "MonteCarloAgent": 5,
    "GraphOfThoughtsAgent": 5,
    "BlackboardAgent": 5,
    "MixtureOfExpertsAgent": 5,
    "SagaAgent": 5,
    "GeneticAlgorithmAgent": 5,
    "MultiArmedBanditAgent": 5,
    "SelfDiscoverAgent": 5,
    "SpeculativeAgent": 5,
    "CircuitBreakerAgent": 5,
    "TournamentAgent": 5,
    "CompilerAgent": 5,
}


# ============================================================================
# START HERE, SCALE THUS
# ============================================================================


def complexity_for_agent(agent_class_name: str) -> int:
    """Return the Complexity Ladder level for an agent class.

    Args:
        agent_class_name: Class name (e.g. ``"SequentialAgent"``).
            Also accepts an agent instance or class — the class name
            is extracted automatically.

    Returns:
        Level 0–5, or ``-1`` if the class is unrecognised.

    Example:
        >>> complexity_for_agent("MonteCarloAgent")
        5
        >>> complexity_for_agent("SequentialAgent")
        2
    """
    if not isinstance(agent_class_name, str):
        agent_class_name = type(agent_class_name).__name__

    return _AGENT_COMPLEXITY.get(agent_class_name, -1)


def suggest_next(current_level: int) -> Optional[Recommendation]:
    """Suggest the next level up from the current one.

    Returns ``None`` when already at Level 5.  Includes a concrete
    description of *when* to escalate (via ``next_if``).

    Args:
        current_level: Your current Complexity Ladder level (0–5).

    Returns:
        The next-level ``Recommendation``, or ``None`` at Level 5.

    Example:
        >>> nxt = suggest_next(2)
        >>> nxt.level
        3
        >>> nxt.level_name
        'Branching workflow'
    """
    if current_level >= 5:
        return None

    return _RECOMMENDATIONS.get(current_level + 1)


def suggest_simpler(current_level: int) -> Optional[Recommendation]:
    """Suggest a simpler alternative to the current level.

    Returns ``None`` when already at Level 0.  Use this when you
    suspect over-engineering.

    Args:
        current_level: Your current Complexity Ladder level (0–5).

    Returns:
        The lower-level ``Recommendation``, or ``None`` at Level 0.

    Example:
        >>> simpler = suggest_simpler(4)
        >>> simpler.level
        3
        >>> simpler.level_name
        'Branching workflow'
    """
    if current_level <= 0:
        return None

    return _RECOMMENDATIONS.get(current_level - 1)


# ============================================================================
# COMPLEXITY BUDGET
# ============================================================================


@dataclass
class ComplexityBudget:
    """Track and cap the total complexity of an agent composition.

    Assigns a score to each agent in a tree and warns (or raises) when
    the total exceeds a budget.  Encourages the Anthropic principle:
    *"Start with the simplest solution, and increase complexity only
    when needed."*

    Scoring: each agent's score equals its Complexity Ladder level
    (0–5).  A ``SequentialAgent`` with 3 ``LlmAgent`` sub-agents
    scores 2 + 1 + 1 + 1 = 5.

    Args:
        max_score: Maximum allowed cumulative complexity score.
            Default 10 — roughly a Level-3 pipeline with a few sub-agents.

    Example:
        >>> budget = ComplexityBudget(max_score=10)
        >>> report = budget.audit(my_agent)
        >>> print(report.total_score)
        >>> if report.over_budget:
        ...     print(report.suggestion)
    """

    max_score: int = 10

    def audit(self, agent: Any) -> "BudgetReport":
        """Walk an agent tree and produce a complexity report.

        Args:
            agent: Root ``BaseAgent`` instance to audit.

        Returns:
            A ``BudgetReport`` with the breakdown and verdict.
        """
        entries: List[BudgetEntry] = []
        self._walk(agent, entries, depth=0)
        total = sum(e.score for e in entries)
        over = total > self.max_score
        suggestion = ""

        if over:
            # Find the highest-level agent contributing
            entries_sorted = sorted(entries, key=lambda e: e.score, reverse=True)
            top = entries_sorted[0]
            simpler = suggest_simpler(top.score)
            suggestion = (
                f"Total complexity {total} exceeds budget {self.max_score}. "
                f"The most complex component is '{top.agent_name}' "
                f"({top.agent_type}, level {top.score}). "
            )

            if simpler:
                suggestion += (
                    f"Consider replacing it with a Level {simpler.level} "
                    f"pattern ({simpler.pattern})."
                )

        return BudgetReport(
            entries=entries,
            total_score=total,
            max_score=self.max_score,
            over_budget=over,
            suggestion=suggestion,
        )

    def _walk(
        self,
        agent: Any,
        entries: List["BudgetEntry"],
        depth: int,
    ) -> None:
        """Recursively walk the agent tree."""
        cls_name = type(agent).__name__
        score = _AGENT_COMPLEXITY.get(cls_name, 0)
        entries.append(BudgetEntry(
            agent_name=agent.name if hasattr(agent, "name") else cls_name,
            agent_type=cls_name,
            score=score,
            depth=depth,
        ))

        sub = getattr(agent, "sub_agents", None) or []

        for child in sub:
            self._walk(child, entries, depth + 1)


@dataclass(frozen=True)
class BudgetEntry:
    """A single agent's contribution to the complexity budget.

    Attributes:
        agent_name: Instance name of the agent.
        agent_type: Class name (e.g. ``"SequentialAgent"``).
        score: Complexity Ladder level (0–5).
        depth: Nesting depth in the tree (0 = root).
    """

    agent_name: str
    agent_type: str
    score: int
    depth: int


@dataclass(frozen=True)
class BudgetReport:
    """Result of a complexity budget audit.

    Attributes:
        entries: Per-agent breakdown.
        total_score: Sum of all agent scores.
        max_score: The budget cap.
        over_budget: ``True`` if *total_score* > *max_score*.
        suggestion: Human-readable advice when over budget.
    """

    entries: List[BudgetEntry]
    total_score: int
    max_score: int
    over_budget: bool
    suggestion: str

    def summary_table(self) -> str:
        """Format the audit as a readable table.

        Returns:
            Multi-line string with agent tree indented by depth.
        """
        lines = [f"Complexity Budget: {self.total_score}/{self.max_score}"]
        status = "OVER BUDGET" if self.over_budget else "OK"
        lines.append(f"Status: {status}")
        lines.append("")
        lines.append(f"{'Agent':<30} {'Type':<28} {'Level':>5}")
        lines.append("-" * 65)

        for e in self.entries:
            indent = "  " * e.depth
            name = f"{indent}{e.agent_name}"
            lines.append(f"{name:<30} {e.agent_type:<28} {e.score:>5}")

        if self.suggestion:
            lines.append("")
            lines.append(self.suggestion)

        return "\n".join(lines)


def audit_agent_tree(
    agent: Any,
    *,
    max_score: int = 10,
    warn: bool = True,
) -> BudgetReport:
    """Convenience function: audit an agent tree against a complexity budget.

    Args:
        agent: Root ``BaseAgent`` instance.
        max_score: Budget cap (default 10).
        warn: If ``True``, emit a ``logging.WARNING`` when over budget.

    Returns:
        A ``BudgetReport``.

    Example:
        >>> report = audit_agent_tree(my_pipeline, max_score=8)
        >>> if report.over_budget:
        ...     print(report.suggestion)
    """
    budget = ComplexityBudget(max_score=max_score)
    report = budget.audit(agent)

    if warn and report.over_budget:
        logger.warning(
            "Complexity budget exceeded: %d/%d. %s",
            report.total_score,
            report.max_score,
            report.suggestion,
        )

    return report
