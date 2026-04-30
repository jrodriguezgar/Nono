"""
ASCII visualization for Workflow pipelines and Agent orchestration trees.

Renders human-readable tree diagrams directly in the terminal.
Works with both ``Workflow`` graphs and ``BaseAgent`` hierarchies.

Usage — Workflow:
    from nono.workflows import Workflow
    from nono.visualize import draw_workflow

    flow = Workflow("pipeline")
    flow.step("fetch", fetch_fn)
    flow.step("process", process_fn)
    flow.connect("fetch", "process")

    print(draw_workflow(flow))

Usage — Agent orchestration:
    from nono.agent import Agent, SequentialAgent
    from nono.visualize import draw_agent

    pipeline = SequentialAgent(name="pipeline", sub_agents=[agent_a, agent_b])
    print(draw_agent(pipeline))
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("Nono.Visualize")

# ── Box drawing characters ───────────────────────────────────────────────

_H = "─"
_V = "│"
_TL = "┌"
_TR = "┐"
_BL = "└"
_BR = "┘"
_ARROW_DOWN = "▼"
_ARROW_RIGHT = "►"
_TEE_DOWN = "┬"
_TEE_RIGHT = "├"
_TEE_LEFT = "┤"
_BRANCH = "◆"
_LOOP = "↻"
_PARALLEL = "═"

# ── Workflow icons ───────────────────────────────────────────────────────

_WORKFLOW_ICON = "📋"
_STEP_ICON = "○"
_BRANCH_ICON = "◆"
_LOOP_REF_ICON = "↻"
_PARALLEL_ICON = "⏸"
_LOOP_ICON = "🔁"
_JOIN_ICON = "⏩"

# ── Workflow drawing ─────────────────────────────────────────────────────


def draw_workflow(workflow: Any, *, title: bool = True) -> str:
    """Render a Workflow as an ASCII tree diagram.

    Uses the same tree connector style as :func:`draw_agent` —
    ``├──``, ``└──``, ``│`` — for a consistent visual language.

    Branch steps (``branch`` / ``branch_if``) show their targets
    as nested children under the branch node.

    Args:
        workflow: A ``nono.workflows.Workflow`` instance.
        title: Whether to include the workflow root label.

    Returns:
        Multi-line ASCII string.

    Example:
        >>> from nono.workflows import Workflow
        >>> flow = Workflow("demo")
        >>> flow.step("a", lambda s: s)
        >>> flow.step("b", lambda s: s)
        >>> flow.connect("a", "b")
        >>> print(draw_workflow(flow))
        📋 demo (Workflow, 2 steps)
        ├── ○ a
        └── ○ b
    """
    steps: list[str] = workflow.steps
    if not steps:
        return "(empty workflow)"

    edges: list[tuple[str, str]] = list(workflow._edges)
    branches: dict[str, Callable] = dict(workflow._branches)
    parallel_groups: dict[str, list[str]] = dict(getattr(workflow, "_parallel_groups", {}))
    loop_defs: dict[str, Any] = dict(getattr(workflow, "_loop_defs", {}))
    join_defs: dict[str, Any] = dict(getattr(workflow, "_join_defs", {}))

    # Build adjacency for non-branching edges
    forward: dict[str, list[str]] = {}
    for src, dst in edges:
        forward.setdefault(src, []).append(dst)

    # Determine the execution order
    order = workflow._resolve_execution_order()

    # Determine which steps are branch targets (shown as children, not top-level)
    branch_targets: set[str] = set()
    for step_name in branches:
        for target in forward.get(step_name, []):
            branch_targets.add(target)

    # Top-level steps: everything in order except branch targets
    top_level = [s for s in order if s not in branch_targets]

    lines: list[str] = []
    drawn: set[str] = set()

    if title:
        n = len(steps)
        root_label = (
            f"{_WORKFLOW_ICON} {workflow.name} "
            f"(Workflow, {n} step{'s' if n != 1 else ''})"
        )
        lines.append(root_label)

    for i, step_name in enumerate(top_level):
        if step_name in drawn:
            continue
        drawn.add(step_name)

        is_last = i == len(top_level) - 1
        is_branch = step_name in branches
        is_parallel = step_name in parallel_groups
        is_loop = step_name in loop_defs
        is_join = step_name in join_defs

        if is_parallel:
            icon = _PARALLEL_ICON
        elif is_loop:
            icon = _LOOP_ICON
        elif is_join:
            icon = _JOIN_ICON
        elif is_branch:
            icon = _BRANCH_ICON
        else:
            icon = _STEP_ICON

        if title:
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
        else:
            connector = ""
            child_prefix = ""

        # Annotate step label
        annotations: list[str] = []

        if is_parallel:
            subs = parallel_groups[step_name]
            annotations.append(f"parallel: {', '.join(subs)}")

        if is_loop:
            ld = loop_defs[step_name]
            max_iter = getattr(ld, "max_iterations", "?")
            annotations.append(f"loop max {max_iter}x")

        if is_join:
            jd = join_defs[step_name]
            wait_for = getattr(jd, "wait_for", [])
            annotations.append(f"join: {', '.join(wait_for)}")

        suffix = f"  ({'; '.join(annotations)})" if annotations else ""
        lines.append(f"{connector}{icon} {step_name}{suffix}")

        # Branch step → show targets as nested children
        if is_branch:
            targets = forward.get(step_name, [])
            for j, target in enumerate(targets):
                drawn.add(target)
                t_is_last = j == len(targets) - 1
                t_connector = "└── " if t_is_last else "├── "
                t_icon = _STEP_ICON
                lines.append(
                    f"{child_prefix}{t_connector}{t_icon} {target}"
                )

    return "\n".join(lines)


# ── Agent tree drawing ───────────────────────────────────────────────────

_AGENT_ICONS: dict[str, str] = {
    "LlmAgent": "🤖",
    "Agent": "🤖",
    "SequentialAgent": "⏩",
    "ParallelAgent": "⏸",
    "LoopAgent": "🔁",
    "RouterAgent": "🔀",
    "MapReduceAgent": "🗺️",
    "ConsensusAgent": "🤝",
    "ProducerReviewerAgent": "📝",
    "DebateAgent": "⚔️",
    "EscalationAgent": "📶",
    "SupervisorAgent": "👔",
    "VotingAgent": "🗳️",
    "HandoffAgent": "🤝",
    "GroupChatAgent": "💬",
    "HierarchicalAgent": "🏛️",
    "GuardrailAgent": "🛡️",
    "BestOfNAgent": "🏆",
    "BatchAgent": "📦",
    "CascadeAgent": "🌊",
    "TreeOfThoughtsAgent": "🌳",
    "PlannerAgent": "📋",
    "SubQuestionAgent": "❓",
    "ContextFilterAgent": "🔍",
    "ReflexionAgent": "🪞",
    "SpeculativeAgent": "⚡",
    "CircuitBreakerAgent": "🔌",
    "TournamentAgent": "🏅",
    "ShadowAgent": "👻",
    "CompilerAgent": "⚙️",
    "CheckpointableAgent": "💾",
    "DynamicFanOutAgent": "🌐",
    "SwarmAgent": "🐝",
    "MemoryConsolidationAgent": "🧠",
    "PriorityQueueAgent": "📊",
    "MonteCarloAgent": "🎲",
    "GraphOfThoughtsAgent": "🕸️",
    "BlackboardAgent": "📋",
    "MixtureOfExpertsAgent": "🎯",
    "CoVeAgent": "✅",
    "SagaAgent": "🔄",
    "LoadBalancerAgent": "⚖️",
    "EnsembleAgent": "🎼",
    "TimeoutAgent": "⏱️",
    "AdaptivePlannerAgent": "📐",
    "SkeletonOfThoughtAgent": "🦴",
    "LeastToMostAgent": "📈",
    "SelfDiscoverAgent": "🔮",
    "GeneticAlgorithmAgent": "🧬",
    "MultiArmedBanditAgent": "🎰",
    "SocraticAgent": "🏛️",
    "MetaOrchestratorAgent": "🪆",
    "CacheAgent": "💿",
    "BudgetAgent": "💰",
    "CurriculumAgent": "🎓",
    "SelfConsistencyAgent": "🗳️",
    "MixtureOfAgentsAgent": "🧪",
    "StepBackAgent": "⏪",
    "OrchestratorWorkerAgent": "🎭",
    "SelfRefineAgent": "✏️",
    "BacktrackingAgent": "↩️",
    "ChainOfDensityAgent": "📝",
    "MediatorAgent": "⚖️",
    "DivideAndConquerAgent": "🪓",
    "BeamSearchAgent": "🔦",
    "RephraseAndRespondAgent": "🔄",
    "CumulativeReasoningAgent": "📚",
    "MultiPersonaAgent": "🎭",
    "AntColonyAgent": "🐜",
    "PipelineParallelAgent": "🏭",
    "ContractNetAgent": "📝",
    "RedTeamAgent": "🔴",
    "FeedbackLoopAgent": "♻️",
    "WinnowingAgent": "🌾",
    "MixtureOfThoughtsAgent": "💭",
    "SimulatedAnnealingAgent": "🌡️",
    "TabuSearchAgent": "🚫",
    "ParticleSwarmAgent": "🫧",
    "DifferentialEvolutionAgent": "🧬",
    "BayesianOptimizationAgent": "📉",
    "AnalogicalReasoningAgent": "🔗",
    "ThreadOfThoughtAgent": "🧵",
    "ExpertPromptingAgent": "🎓",
    "BufferOfThoughtsAgent": "📋",
    "ChainOfAbstractionAgent": "🔲",
    "VerifierAgent": "✅",
    "ProgOfThoughtAgent": "💻",
    "InnerMonologueAgent": "🗣️",
    "RolePlayingAgent": "🎬",
    "GossipProtocolAgent": "📡",
    "AuctionAgent": "🔨",
    "DelphiMethodAgent": "🏛️",
    "NominalGroupAgent": "📊",
    "ActiveRetrievalAgent": "🎣",
    "IterativeRetrievalAgent": "🔁",
    "PromptChainAgent": "⛓️",
    "HypothesisTestingAgent": "🔬",
    "SkillLibraryAgent": "📖",
    "RecursiveCriticAgent": "🔍",
    "DemonstrateSearchPredictAgent": "🧭",
    "DoubleLoopLearningAgent": "🔄",
    "AgendaAgent": "📅",
}


def _agent_label(agent: Any) -> str:
    """Build a display label for an agent.

    Args:
        agent: A ``BaseAgent`` instance.

    Returns:
        Label string like ``"🤖 assistant (LlmAgent, google/gemini-3-flash)"``
    """
    cls_name = type(agent).__name__
    icon = _AGENT_ICONS.get(cls_name, "○")
    label = f"{icon} {agent.name}"

    parts: list[str] = [cls_name]

    provider = getattr(agent, "_provider", "")
    model = getattr(agent, "_model", "") or ""
    if provider:
        model_short = model.split("/")[-1] if model else ""
        if model_short:
            parts.append(f"{provider}/{model_short}")
        else:
            parts.append(provider)

    tools = getattr(agent, "tools", [])
    if tools:
        parts.append(f"{len(tools)} tools")

    max_iter = getattr(agent, "max_iterations", 0)
    if cls_name == "LoopAgent" and max_iter:
        parts.append(f"max {max_iter}x")

    max_workers = getattr(agent, "max_workers", None)
    if cls_name == "ParallelAgent" and max_workers:
        parts.append(f"{max_workers} workers")

    label += f" ({', '.join(parts)})"
    return label


_MAX_DRAW_DEPTH: int = 50
"""Safety limit for ``draw_agent`` recursion depth."""


def draw_agent(
    agent: Any,
    *,
    _prefix: str = "",
    _child_prefix: str = "",
    _visited: frozenset[int] | None = None,
    _depth: int = 0,
) -> str:
    """Render an Agent hierarchy as an ASCII tree.

    Recursively draws the agent and all its sub-agents with tree
    connectors, icons, and metadata annotations.  Includes cycle
    detection and a maximum depth guard (``_MAX_DRAW_DEPTH``) to
    prevent ``RecursionError`` on circular topologies.

    Args:
        agent: A ``BaseAgent`` instance (or any object with ``name``
            and ``sub_agents``).

    Returns:
        Multi-line ASCII string.

    Example:
        >>> from nono.agent import Agent, SequentialAgent
        >>> a = Agent(name="a", provider="google", instruction="Hi.")
        >>> b = Agent(name="b", provider="google", instruction="Hi.")
        >>> seq = SequentialAgent(name="pipe", sub_agents=[a, b])
        >>> print(draw_agent(seq))
        ⏩ pipe (SequentialAgent)
        ├── 🤖 a (LlmAgent, google)
        └── 🤖 b (LlmAgent, google)
    """
    visited = _visited or frozenset()
    agent_id = id(agent)

    lines: list[str] = []
    label = _agent_label(agent)

    if agent_id in visited:
        lines.append(f"{_prefix}{label} ↺ (cycle)")
        return "\n".join(lines)

    if _depth >= _MAX_DRAW_DEPTH:
        lines.append(f"{_prefix}{label} … (max depth)")
        return "\n".join(lines)

    lines.append(f"{_prefix}{label}")

    visited = visited | {agent_id}

    sub_agents = getattr(agent, "sub_agents", []) or []
    tools = getattr(agent, "tools", [])

    # Collect all children: tools first, then sub-agents
    children: list[tuple[str, Any]] = []
    for t in tools:
        t_name = getattr(t, "name", str(t))
        children.append(("tool", t_name))
    for sa in sub_agents:
        children.append(("agent", sa))

    for i, (kind, child) in enumerate(children):
        is_last = (i == len(children) - 1)
        branch = "└── " if is_last else "├── "
        next_prefix = _child_prefix + ("    " if is_last else "│   ")

        if kind == "tool":
            lines.append(f"{_child_prefix}{branch}🔧 {child}")
        else:
            child_text = draw_agent(
                child,
                _prefix=f"{_child_prefix}{branch}",
                _child_prefix=next_prefix,
                _visited=visited,
                _depth=_depth + 1,
            )
            lines.append(child_text)

    return "\n".join(lines)


# ── Unified draw() ───────────────────────────────────────────────────────

# ── Routine icons ────────────────────────────────────────────────────────

_ROUTINE_ICON = "⚙️"
_TRIGGER_ICONS: dict[str, str] = {
    "schedule": "⏰",
    "event": "📡",
    "webhook": "🌐",
    "manual": "👆",
}


def draw_routine(routine: Any, *, title: bool = True) -> str:
    """Render a Routine as an ASCII tree diagram.

    Shows the routine name, triggers, configuration, and executable.

    Args:
        routine: A ``nono.routines.Routine`` instance.
        title: Whether to include the routine root label.

    Returns:
        Multi-line ASCII string.

    Example:
        >>> from nono.routines import Routine, ScheduleTrigger
        >>> r = Routine(name="nightly", description="Nightly review")
        >>> print(draw_routine(r))
        ⚙️  nightly (Routine, idle)
        │   "Nightly review"
        ├── Triggers
        │   └── (none)
        ├── Config
        │   ├── timeout: 300s
        │   └── retries: 0
        └── Executable: (none)
    """
    lines: list[str] = []

    if title:
        status = routine.status.value if hasattr(routine.status, "value") else str(routine.status)
        lines.append(f"{_ROUTINE_ICON}  {routine.name} (Routine, {status})")
        if routine.description:
            lines.append(f"│   \"{routine.description}\"")

    # Triggers section
    triggers = routine.triggers or []
    has_exe = routine.executable is not None
    lines.append("├── Triggers")

    if triggers:
        for i, trigger in enumerate(triggers):
            is_last_trigger = i == len(triggers) - 1
            t_connector = "└── " if is_last_trigger else "├── "
            t_type = trigger.trigger_type.value if hasattr(trigger, "trigger_type") else "unknown"
            t_icon = _TRIGGER_ICONS.get(t_type, "?")

            label = f"{t_icon} {t_type}"
            if hasattr(trigger, "cron") and trigger.cron:
                label += f" [{trigger.cron}]"
            elif hasattr(trigger, "interval_seconds") and trigger.interval_seconds:
                label += f" [every {trigger.interval_seconds}s]"
            elif hasattr(trigger, "event_name") and trigger.event_name:
                label += f" [{trigger.event_name}]"
            elif hasattr(trigger, "event_pattern") and trigger.event_pattern:
                label += f" [/{trigger.event_pattern}/]"
            elif hasattr(trigger, "path") and trigger.path:
                label += f" [{trigger.path}]"

            if hasattr(trigger, "description") and trigger.description:
                label += f"  — {trigger.description}"

            lines.append(f"│   {t_connector}{label}")
    else:
        lines.append("│   └── (none)")

    # Config section
    config = routine.config
    lines.append("├── Config")
    lines.append(f"│   ├── timeout: {config.timeout_seconds}s")
    lines.append(f"│   └── retries: {config.max_retries}")

    # Executable
    exe = routine.executable
    if exe is not None:
        exe_type = type(exe).__name__
        exe_name = getattr(exe, "name", exe_type)
        lines.append(f"└── Executable: {exe_name} ({exe_type})")
    else:
        lines.append("└── Executable: (none)")

    return "\n".join(lines)


def draw_runner(runner: Any, *, title: bool = True) -> str:
    """Render a RoutineRunner status as an ASCII diagram.

    Shows all registered routines with their triggers and status.

    Args:
        runner: A ``nono.routines.RoutineRunner`` instance.
        title: Whether to include the runner header.

    Returns:
        Multi-line ASCII string.
    """
    lines: list[str] = []

    routines = runner.list_routines()
    is_running = runner.is_running

    if title:
        state = "running" if is_running else "stopped"
        lines.append(
            f"🏭 RoutineRunner ({state}, {len(routines)} routine"
            f"{'s' if len(routines) != 1 else ''})"
        )

    if not routines:
        lines.append("└── (no routines registered)")
        return "\n".join(lines)

    for i, routine in enumerate(routines):
        is_last = i == len(routines) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "

        status = routine.status.value if hasattr(routine.status, "value") else str(routine.status)
        triggers_str = ", ".join(
            t.trigger_type.value if hasattr(t, "trigger_type") else "?"
            for t in routine.triggers
        ) or "manual"

        lines.append(f"{connector}{_ROUTINE_ICON}  {routine.name}  [{status}]  ({triggers_str})")

        if routine.description:
            lines.append(f"{child_prefix}   \"{routine.description}\"")

    return "\n".join(lines)


def draw(obj: Any, **kwargs: Any) -> str:
    """Auto-detect object type and draw the appropriate ASCII diagram.

    Accepts either a ``Workflow`` or a ``BaseAgent`` instance.

    Args:
        obj: A ``Workflow`` or ``BaseAgent`` to visualize.
        **kwargs: Extra keyword arguments forwarded to the specific drawer.

    Returns:
        Multi-line ASCII string.

    Raises:
        TypeError: If the object type is not recognized.

    Example:
        >>> from nono.visualize import draw
        >>> print(draw(my_workflow))
        >>> print(draw(my_agent))
    """
    cls_name = type(obj).__name__

    # Check for Workflow
    if hasattr(obj, "_step_order") and hasattr(obj, "_edges"):
        return draw_workflow(obj, **kwargs)

    # Check for BaseAgent (has name + sub_agents)
    if hasattr(obj, "name") and hasattr(obj, "sub_agents"):
        return draw_agent(obj, **kwargs)

    # Check for Routine (has triggers + executable + config)
    if hasattr(obj, "triggers") and hasattr(obj, "executable") and hasattr(obj, "config"):
        return draw_routine(obj, **kwargs)

    # Check for RoutineRunner (has _routines + list_routines)
    if hasattr(obj, "_routines") and hasattr(obj, "list_routines"):
        return draw_runner(obj, **kwargs)

    raise TypeError(
        f"Cannot draw object of type {cls_name!r}. "
        "Expected a Workflow, BaseAgent, Routine, or RoutineRunner instance."
    )
