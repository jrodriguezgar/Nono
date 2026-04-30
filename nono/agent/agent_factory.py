"""
Dynamic agent factory — generate agents from natural language descriptions.

Produces ``AgentBlueprint`` specs from a user prompt that can be reviewed
before instantiation.  The factory is **modular**: system-prompt generation,
tool selection, orchestration selection, and final configuration are handled
by independent, replaceable components.

When the task is complex enough, the factory can recommend an **orchestration
pattern** (Sequential, Parallel, Planner, etc.) and generate a full
multi-agent ``OrchestrationBlueprint`` with sub-agent specs.

Security:
    - Gated by ``agent.allow_dynamic_creation`` in config.toml (default OFF).
    - Tool allowlist restricts which tools a generated agent may use.
    - System prompts are sanitised against common injection patterns.
    - Blueprint review step allows human-in-the-loop before instantiation.

Usage::

    from nono.agent.agent_factory import AgentFactory, create_agent_from_prompt

    # Quick — one-liner with defaults
    agent = create_agent_from_prompt(
        "An agent that summarises PDFs and extracts key entities.",
        available_tools=[pdf_reader, entity_extractor],
    )

    # Orchestrated — the factory picks the best pattern
    factory = AgentFactory()
    orch_bp = factory.generate_orchestrated_blueprint(
        "Research a topic, then write an article, then review it.",
        available_tools=[web_search, summarise],
    )
    print(orch_bp)           # review orchestration + sub-agents
    agent = factory.build_orchestrated(orch_bp, available_tools=[...])

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .base import BaseAgent
from .tool import FunctionTool

logger = logging.getLogger("Nono.Agent.Factory")

__all__ = [
    "AgentBlueprint",
    "AgentFactory",
    "BlueprintValidationError",
    "DynamicCreationDisabledError",
    "OrchestrationBlueprint",
    "OrchestrationFactory",
    "OrchestrationRegistry",
    "OrchestrationSelector",
    "PatternRegistration",
    "SystemPromptGenerator",
    "ToolSelector",
    "AgentConfigurator",
    "create_agent_from_prompt",
    "register_pattern",
]


# ── Exceptions ────────────────────────────────────────────────────────────────

class DynamicCreationDisabledError(RuntimeError):
    """Raised when dynamic agent creation is attempted but disabled in config."""


class BlueprintValidationError(ValueError):
    """Raised when a generated blueprint fails security or schema validation."""


# ── Configuration helpers ─────────────────────────────────────────────────────

def _load_factory_config() -> dict[str, Any]:
    """Load ``[agent.factory]`` settings from config.toml.

    Returns:
        Dict with factory settings. Missing keys fall back to safe defaults.
    """
    defaults: dict[str, Any] = {
        "allow_dynamic_creation": False,
        "max_tools_per_agent": 10,
        "max_instruction_length": 4000,
        "default_provider": "google",
        "default_model": None,
        "allowed_providers": [],
        "tool_allowlist": [],
    }

    try:
        from nono.config import load_config
        cfg = load_config()

        for key in defaults:
            val = cfg.get(f"agent.factory.{key}")
            if val is not None:
                defaults[key] = val
    except Exception:
        pass

    return defaults


# ── Prompt sanitisation ───────────────────────────────────────────────────────

# Patterns that indicate prompt-injection or role-hijacking attempts
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\|?system\|?>", re.I),
    re.compile(r"```\s*(system|assistant)\b", re.I),
    re.compile(r"do\s+not\s+follow\s+(any|your)\s+(rules|instructions|guidelines)", re.I),
    re.compile(r"override\s+(all\s+)?(safety|security|restriction|guardrail)", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system|initial|original)\s+prompt", re.I),
]


def sanitise_instruction(text: str, *, max_length: int = 4000) -> str:
    """Remove injection patterns and enforce length limits.

    Args:
        text: Raw instruction text.
        max_length: Maximum allowed character length.

    Returns:
        Sanitised instruction string.

    Raises:
        BlueprintValidationError: If the text contains injection patterns.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise BlueprintValidationError(
                f"Instruction contains a disallowed pattern: {pattern.pattern!r}"
            )

    if len(text) > max_length:
        logger.warning(
            "Instruction truncated from %d to %d characters.",
            len(text), max_length,
        )
        text = text[:max_length]

    return text.strip()


# ── AgentBlueprint ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentBlueprint:
    """Immutable specification for a dynamically generated agent.

    Produced by ``AgentFactory.generate_blueprint()`` and consumed by
    ``AgentFactory.build()``.  Inspect before building to review what the
    LLM decided.

    Args:
        name: Agent name (lowercase, alphanumeric + underscores).
        description: Human-readable purpose.
        instruction: System prompt for the agent.
        provider: LLM provider name.
        model: Model name (or ``None`` for provider default).
        temperature: LLM temperature.
        tool_names: Names of tools selected from the available pool.
        output_format: ``"text"`` or ``"json"``.
        metadata: Arbitrary extra data from the generation step.
    """
    name: str
    description: str
    instruction: str
    provider: str = "google"
    model: str | None = None
    temperature: float = 0.7
    tool_names: tuple[str, ...] = ()
    output_format: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the blueprint to a plain dict.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "instruction": self.instruction,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "tool_names": list(self.tool_names),
            "output_format": self.output_format,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AgentBlueprint:
        """Deserialise a blueprint from a dict.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            A new ``AgentBlueprint``.
        """
        return AgentBlueprint(
            name=data["name"],
            description=data.get("description", ""),
            instruction=data.get("instruction", ""),
            provider=data.get("provider", "google"),
            model=data.get("model"),
            temperature=float(data.get("temperature", 0.7)),
            tool_names=tuple(data.get("tool_names", ())),
            output_format=data.get("output_format", "text"),
            metadata=data.get("metadata", {}),
        )


# ── Orchestration pattern registry ─────────────────────────────────────────────

# Type alias for orchestration factory functions.
# A factory receives the standard keyword arguments and returns a BaseAgent.
OrchestrationFactory = Callable[..., BaseAgent]
"""Signature: ``(*, name, description, sub_agents, provider, model, pattern_kwargs) -> BaseAgent``."""


@dataclass(frozen=True)
class PatternRegistration:
    """Immutable specification for a registered orchestration pattern.

    Args:
        key: Unique pattern key (snake_case).
        class_name: Name of the workflow-agent class (for catalogue display).
        description: Human-readable description of the pattern.
        keyword_hints: Trigger words for heuristic pattern selection.
        factory: Callable that instantiates the orchestration agent.
        min_sub_agents: Minimum number of sub-agents required.
    """

    key: str
    class_name: str
    description: str
    keyword_hints: tuple[str, ...] = ()
    factory: OrchestrationFactory | None = None
    min_sub_agents: int = 1


class OrchestrationRegistry:
    """Extensible registry for orchestration patterns.

    All built-in patterns are registered at module load.  Third-party code
    can register additional patterns at any time using :meth:`register` or
    the module-level :func:`register_pattern` helper.

    The registry is the **single source of truth** for the set of available
    orchestration patterns.  Keyword hints and instantiation logic all
    read from here.

    Example — register a custom pattern::

        from nono.agent.agent_factory import register_pattern

        def my_factory(*, name, description, sub_agents, provider, model, pattern_kwargs):
            return MyCustomAgent(sub_agents=sub_agents, name=name, ...)

        register_pattern(
            key="my_custom",
            class_name="MyCustomAgent",
            description="My domain-specific orchestration",
            keyword_hints=["custom", "special workflow"],
            factory=my_factory,
            min_sub_agents=2,
        )
    """

    _patterns: dict[str, PatternRegistration] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(
        cls,
        key: str,
        class_name: str,
        description: str,
        *,
        keyword_hints: Sequence[str] = (),
        factory: OrchestrationFactory | None = None,
        min_sub_agents: int = 1,
    ) -> None:
        """Register a new orchestration pattern.

        Args:
            key: Unique pattern key (snake_case).
            class_name: Workflow-agent class name.
            description: Human-readable description.
            keyword_hints: Trigger words for heuristic selection.
            factory: Callable to instantiate the pattern. Receives keyword
                arguments ``name``, ``description``, ``sub_agents``,
                ``provider``, ``model``, and ``pattern_kwargs``.
            min_sub_agents: Minimum required sub-agents.

        Raises:
            ValueError: If *key* is empty or *class_name* is empty.
        """
        if not key or not key.strip():
            raise ValueError("Pattern key must not be empty.")
        if not class_name or not class_name.strip():
            raise ValueError("class_name must not be empty.")

        entry = PatternRegistration(
            key=key,
            class_name=class_name,
            description=description,
            keyword_hints=tuple(keyword_hints),
            factory=factory,
            min_sub_agents=min_sub_agents,
        )
        with cls._lock:
            if key in cls._patterns:
                logger.info("Overwriting existing pattern %r.", key)
            cls._patterns[key] = entry

    @classmethod
    def unregister(cls, key: str) -> None:
        """Remove a pattern from the registry.

        Args:
            key: Pattern key to remove.

        Raises:
            KeyError: If the key is not registered.
        """
        with cls._lock:
            if key not in cls._patterns:
                raise KeyError(f"Pattern {key!r} is not registered.")
            del cls._patterns[key]

    @classmethod
    def get(cls, key: str) -> PatternRegistration:
        """Retrieve a pattern registration.

        Args:
            key: Pattern key.

        Returns:
            The ``PatternRegistration`` for *key*.

        Raises:
            KeyError: If the key is not registered.
        """
        try:
            return cls._patterns[key]
        except KeyError:
            raise KeyError(
                f"Pattern {key!r} not registered. "
                f"Available: {sorted(cls._patterns.keys())}"
            ) from None

    @classmethod
    def contains(cls, key: str) -> bool:
        """Check whether a pattern key is registered."""
        return key in cls._patterns

    @classmethod
    def catalog(cls) -> dict[str, tuple[str, str]]:
        """Return the catalogue in legacy format: ``{key: (class_name, description)}``."""
        return {
            k: (p.class_name, p.description)
            for k, p in cls._patterns.items()
        }

    @classmethod
    def keyword_hints(cls) -> dict[str, list[str]]:
        """Return keyword hints for all patterns that have them."""
        return {
            k: list(p.keyword_hints)
            for k, p in cls._patterns.items()
            if p.keyword_hints
        }

    @classmethod
    def list_patterns(cls) -> list[str]:
        """Return sorted list of registered pattern keys."""
        return sorted(cls._patterns.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all patterns (for testing only)."""
        with cls._lock:
            cls._patterns.clear()


def register_pattern(
    key: str,
    class_name: str,
    description: str,
    *,
    keyword_hints: Sequence[str] = (),
    factory: OrchestrationFactory | None = None,
    min_sub_agents: int = 1,
) -> None:
    """Module-level convenience wrapper for :meth:`OrchestrationRegistry.register`.

    Args:
        key: Unique pattern key (snake_case).
        class_name: Workflow-agent class name.
        description: Human-readable description.
        keyword_hints: Trigger words for heuristic selection.
        factory: Callable to instantiate the pattern.
        min_sub_agents: Minimum required sub-agents.
    """
    OrchestrationRegistry.register(
        key,
        class_name,
        description,
        keyword_hints=keyword_hints,
        factory=factory,
        min_sub_agents=min_sub_agents,
    )


# ── Built-in pattern factories ────────────────────────────────────────────────
# Each factory lazily imports from workflow_agents to avoid circular deps.

def _factory_sequential(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.SequentialAgent(sub_agents=sub_agents, name=name, description=description)


def _factory_parallel(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.ParallelAgent(sub_agents=sub_agents, name=name, description=description)


def _factory_loop(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.LoopAgent(
        sub_agents=sub_agents,
        max_iterations=int(pattern_kwargs.get("max_iterations", 3)),
        name=name, description=description,
    )


def _factory_router(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.RouterAgent(
        sub_agents=sub_agents, provider=provider, model=model,
        name=name, description=description,
    )


def _factory_planner(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.PlannerAgent(
        sub_agents=sub_agents, provider=provider, model=model,
        max_steps=int(pattern_kwargs.get("max_steps", 5)),
        name=name, description=description,
    )


def _factory_map_reduce(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    if len(sub_agents) < 2:
        raise BlueprintValidationError(
            "MapReduceAgent requires at least 2 sub-agents (mappers + 1 reducer)."
        )
    return wa.MapReduceAgent(
        sub_agents=sub_agents[:-1], reduce_agent=sub_agents[-1],
        name=name, description=description,
    )


def _factory_supervisor(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.SupervisorAgent(
        sub_agents=sub_agents, provider=provider, model=model,
        max_iterations=int(pattern_kwargs.get("max_iterations", 3)),
        name=name, description=description,
    )


def _factory_producer_reviewer(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    if len(sub_agents) < 2:
        raise BlueprintValidationError(
            "ProducerReviewerAgent requires at least 2 sub-agents (producer + reviewer)."
        )
    return wa.ProducerReviewerAgent(
        producer=sub_agents[0], reviewer=sub_agents[1],
        max_iterations=int(pattern_kwargs.get("max_iterations", 3)),
        name=name, description=description,
    )


def _factory_debate(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    if len(sub_agents) < 3:
        raise BlueprintValidationError(
            "DebateAgent requires at least 3 sub-agents (2 debaters + 1 judge)."
        )
    return wa.DebateAgent(sub_agents=sub_agents, name=name, description=description)


def _factory_pipeline_parallel(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.PipelineParallelAgent(stages=sub_agents, name=name, description=description)


def _factory_dynamic_fan_out(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    if len(sub_agents) < 2:
        raise BlueprintValidationError(
            "DynamicFanOutAgent requires at least 2 sub-agents (worker + reducer)."
        )
    return wa.DynamicFanOutAgent(
        worker_agent=sub_agents[0], reducer_agent=sub_agents[-1],
        provider=provider, model=model,
        max_items=int(pattern_kwargs.get("max_items", 10)),
        name=name, description=description,
    )


def _factory_hierarchical(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.HierarchicalAgent(
        sub_agents=sub_agents, provider=provider, model=model,
        name=name, description=description,
    )


def _factory_guardrail(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    if not sub_agents:
        raise BlueprintValidationError("GuardrailAgent requires at least one sub-agent.")
    return wa.GuardrailAgent(sub_agents=sub_agents, name=name, description=description)


def _factory_escalation(
    *, name: str, description: str, sub_agents: list[BaseAgent],
    provider: str, model: str | None, pattern_kwargs: dict[str, Any],
) -> BaseAgent:
    from . import workflow_agents as wa
    return wa.EscalationAgent(sub_agents=sub_agents, name=name, description=description)


# ── Register built-in patterns ────────────────────────────────────────────────

def _register_builtin_patterns() -> None:
    """Register all built-in orchestration patterns in the registry."""
    _builtins: list[tuple[str, str, str, tuple[str, ...], OrchestrationFactory | None, int]] = [
        # (key, class_name, description, keyword_hints, factory, min_sub_agents)
        ("none", "LlmAgent", "Single agent, no orchestration needed", (), None, 1),
        ("sequential", "SequentialAgent", "Run sub-agents one after another in order",
         ("then", "after", "followed by", "step by step", "first", "next", "finally"),
         _factory_sequential, 1),
        ("parallel", "ParallelAgent", "Run sub-agents concurrently and merge results",
         ("simultaneously", "concurrently", "at the same time", "in parallel", "all at once"),
         _factory_parallel, 1),
        ("loop", "LoopAgent", "Repeat sub-agents until a condition is met",
         ("iterate", "repeat", "until", "refine", "improve iteratively"),
         _factory_loop, 1),
        ("router", "RouterAgent", "LLM dynamically routes to the best sub-agent",
         ("route", "classify first", "depending on", "if it's about"),
         _factory_router, 1),
        ("planner", "PlannerAgent", "LLM decomposes into dependency-aware steps",
         ("plan", "decompose", "break down", "complex task", "dependencies"),
         _factory_planner, 1),
        ("map_reduce", "MapReduceAgent", "Fan-out to mappers, then reduce results",
         ("each item", "for every", "aggregate", "map and reduce", "batch process"),
         _factory_map_reduce, 2),
        ("supervisor", "SupervisorAgent", "LLM supervisor delegates and evaluates",
         ("supervise", "delegate", "evaluate", "manager", "oversee"),
         _factory_supervisor, 1),
        ("producer_reviewer", "ProducerReviewerAgent", "Iterative produce → review loop",
         ("review", "critique", "revise", "draft and review", "produce and review"),
         _factory_producer_reviewer, 2),
        ("debate", "DebateAgent", "Adversarial debate with two agents and a judge",
         ("debate", "argue", "pros and cons", "adversarial", "opposing views"),
         _factory_debate, 3),
        ("pipeline_parallel", "PipelineParallelAgent", "Assembly-line stages for item lists",
         (), _factory_pipeline_parallel, 1),
        ("dynamic_fan_out", "DynamicFanOutAgent", "LLM determines parallel work items",
         (), _factory_dynamic_fan_out, 2),
        ("hierarchical", "HierarchicalAgent", "Multi-level tree with LLM manager",
         ("departments", "hierarchy", "multiple teams", "levels"),
         _factory_hierarchical, 1),
        ("guardrail", "GuardrailAgent", "Pre/post validation wrapper with retry",
         (), _factory_guardrail, 1),
        ("escalation", "EscalationAgent", "Try agents in order, stop at first success",
         ("try first", "fallback", "escalate", "if fails"),
         _factory_escalation, 1),
    ]

    for key, cls_name, desc, hints, factory, min_sa in _builtins:
        OrchestrationRegistry.register(
            key, cls_name, desc,
            keyword_hints=hints,
            factory=factory,
            min_sub_agents=min_sa,
        )


_register_builtin_patterns()


@dataclass(frozen=True)
class OrchestrationBlueprint:
    """Immutable specification for a dynamically generated orchestrated pipeline.

    Extends the single-agent ``AgentBlueprint`` concept to multi-agent
    systems.  Contains the orchestration pattern, parameters, and a
    blueprint for each sub-agent.

    Args:
        name: Orchestrator name.
        description: High-level purpose of the pipeline.
        pattern: Orchestration pattern key (from ``OrchestrationRegistry``).
        sub_agent_blueprints: Ordered list of sub-agent specifications.
        pattern_kwargs: Extra keyword arguments for the orchestration agent
            constructor (e.g. ``max_iterations``, ``max_steps``).
        provider: Default LLM provider for orchestrator calls.
        model: Default model for orchestrator calls.
        metadata: Arbitrary extra data.
    """
    name: str
    description: str
    pattern: str
    sub_agent_blueprints: tuple[AgentBlueprint, ...] = ()
    pattern_kwargs: dict[str, Any] = field(default_factory=dict)
    provider: str = "google"
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the orchestration blueprint to a plain dict.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "sub_agent_blueprints": [bp.to_dict() for bp in self.sub_agent_blueprints],
            "pattern_kwargs": self.pattern_kwargs,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OrchestrationBlueprint:
        """Deserialise from a dict.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            A new ``OrchestrationBlueprint``.
        """
        return OrchestrationBlueprint(
            name=data["name"],
            description=data.get("description", ""),
            pattern=data.get("pattern", "none"),
            sub_agent_blueprints=tuple(
                AgentBlueprint.from_dict(bp)
                for bp in data.get("sub_agent_blueprints", [])
            ),
            pattern_kwargs=data.get("pattern_kwargs", {}),
            provider=data.get("provider", "google"),
            model=data.get("model"),
            metadata=data.get("metadata", {}),
        )


# ── Modular components ────────────────────────────────────────────────────────

class SystemPromptGenerator:
    """Generate a system prompt from a natural-language agent description.

    Uses an LLM call to transform a high-level description into a concrete,
    well-structured system prompt.

    Args:
        provider: LLM provider for the generation call.
        model: Model name override.
    """

    _META_SYSTEM_PROMPT: str = (
        "You are an expert AI prompt engineer. Given a description of an AI "
        "agent's purpose, produce a precise system prompt for that agent.\n\n"
        "Rules:\n"
        "- Write in second person (\"You are…\").\n"
        "- Be specific about the agent's domain and constraints.\n"
        "- Include output format expectations when relevant.\n"
        "- Keep it concise (under 2000 characters).\n"
        "- Do NOT include any tool-calling instructions — tools are injected "
        "  separately by the framework.\n"
        "- Respond ONLY with the system prompt text, no explanations.\n"
    )

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._service: Any = None
        self._lock = threading.Lock()

    @property
    def _svc(self) -> Any:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model,
                    )
        return self._service

    def generate(self, description: str) -> str:
        """Produce a system prompt from a description.

        Args:
            description: Natural-language agent purpose.

        Returns:
            Generated system prompt string.
        """
        messages = [
            {"role": "system", "content": self._META_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        response = self._svc.generate_completion(
            messages=messages, temperature=0.4,
        )
        return response.strip()


class ToolSelector:
    """Select appropriate tools from a pool based on agent description.

    Can operate in two modes:
    - **LLM-assisted**: an LLM picks the best tools from descriptions.
    - **Manual / keyword**: match tool names against the description (no LLM).

    Args:
        provider: LLM provider for assisted selection.
        model: Model name override.
        max_tools: Maximum number of tools to select.
    """

    _META_SYSTEM_PROMPT: str = (
        "You are a tool-selection assistant. Given an agent description and a "
        "list of available tools (name + description), select the tools the "
        "agent needs.\n\n"
        "Respond with a JSON array of tool names (strings). "
        "Only include tools that are clearly relevant. "
        "Do not invent tool names — pick from the list.\n"
        "Respond ONLY with the JSON array, no markdown fences.\n"
    )

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        max_tools: int = 10,
    ) -> None:
        self._provider = provider
        self._model = model
        self.max_tools = max_tools
        self._service: Any = None
        self._lock = threading.Lock()

    @property
    def _svc(self) -> Any:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model,
                    )
        return self._service

    def select(
        self,
        description: str,
        available_tools: Sequence[FunctionTool],
        *,
        use_llm: bool = True,
    ) -> list[str]:
        """Select tools matching the agent description.

        Args:
            description: Agent purpose description.
            available_tools: Pool of tools to pick from.
            use_llm: If ``True``, use an LLM call; otherwise keyword match.

        Returns:
            List of selected tool names.
        """
        if not available_tools:
            return []

        if not use_llm:
            return self._keyword_select(description, available_tools)

        tool_catalog = "\n".join(
            f"- {t.name}: {t.description}" for t in available_tools
        )
        user_msg = (
            f"Agent description:\n{description}\n\n"
            f"Available tools:\n{tool_catalog}"
        )
        messages = [
            {"role": "system", "content": self._META_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._svc.generate_completion(
                messages=messages, temperature=0.0,
            )
            selected = json.loads(raw.strip().strip("`").strip())
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("LLM tool selection failed (%s), falling back to keyword.", exc)
            return self._keyword_select(description, available_tools)

        valid_names = {t.name for t in available_tools}
        result = [n for n in selected if n in valid_names]
        return result[: self.max_tools]

    def _keyword_select(
        self,
        description: str,
        available_tools: Sequence[FunctionTool],
    ) -> list[str]:
        """Fallback: select tools whose name/description overlaps the query."""
        desc_lower = description.lower()
        scored: list[tuple[int, str]] = []

        for t in available_tools:
            score = 0
            for word in t.name.replace("_", " ").split():
                if word.lower() in desc_lower:
                    score += 2
            for word in (t.description or "").lower().split():
                if len(word) > 3 and word in desc_lower:
                    score += 1
            if score > 0:
                scored.append((score, t.name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[: self.max_tools]]


class OrchestrationSelector:
    """Analyse a task description and recommend an orchestration pattern.

    Uses an LLM call to determine:
    1. Whether the task needs orchestration or a single agent suffices.
    2. Which pattern from ``OrchestrationRegistry`` fits best.
    3. How to decompose the task into sub-agent roles.

    Can also operate **without** LLM via keyword heuristics (fallback).

    Args:
        provider: LLM provider for the analysis call.
        model: Model name override.
    """

    _META_SYSTEM_PROMPT: str = (
        "You are an expert multi-agent system architect. Given a task "
        "description, decide the best orchestration pattern and sub-agent "
        "decomposition.\n\n"
        "Available patterns:\n{catalog}\n\n"
        "Respond with a JSON object (no markdown fences):\n"
        "{{\n"
        '  "pattern": "pattern_key",\n'
        '  "reasoning": "one sentence why",\n'
        '  "sub_agents": [\n'
        "    {{\n"
        '      "name": "agent_name",\n'
        '      "description": "what this agent does",\n'
        '      "instruction": "concise system prompt"\n'
        "    }}\n"
        "  ],\n"
        '  "pattern_kwargs": {{}}\n'
        "}}\n\n"
        "Rules:\n"
        '- Use "none" if a single agent can handle the task.\n'
        "- Sub-agent names must be snake_case.\n"
        "- Keep sub-agent instructions under 500 characters.\n"
        "- pattern_kwargs may include max_iterations, max_steps, etc.\n"
        "- Do NOT invent pattern names — pick from the list.\n"
    )

    # Keyword hints are now served by OrchestrationRegistry.keyword_hints().
    # Kept as a read-only property for backward compatibility.

    @staticmethod
    def _get_keyword_hints() -> dict[str, list[str]]:
        """Return keyword hints from the registry (includes custom patterns)."""
        return OrchestrationRegistry.keyword_hints()

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._service: Any = None
        self._lock = threading.Lock()

    @property
    def _svc(self) -> Any:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    from .llm_agent import _create_service
                    self._service = _create_service(
                        self._provider, self._model,
                    )
        return self._service

    def select(
        self,
        description: str,
        *,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """Analyse the task and recommend an orchestration strategy.

        Args:
            description: Natural-language task description.
            use_llm: If ``True``, use an LLM call; otherwise keyword heuristics.

        Returns:
            Dict with keys ``pattern``, ``sub_agents``, ``pattern_kwargs``,
            and ``reasoning``.
        """
        if not use_llm:
            return self._keyword_select(description)

        live_catalog = OrchestrationRegistry.catalog()
        catalog_text = "\n".join(
            f"- {key}: {desc}" for key, (_, desc) in live_catalog.items()
        )
        system_prompt = self._META_SYSTEM_PROMPT.format(catalog=catalog_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description},
        ]

        try:
            raw = self._svc.generate_completion(
                messages=messages, temperature=0.0,
            )
            result = json.loads(raw.strip().strip("`").strip())
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "LLM orchestration selection failed (%s), falling back to heuristics.",
                exc,
            )
            return self._keyword_select(description)

        # Validate pattern
        pattern = result.get("pattern", "none")
        if not OrchestrationRegistry.contains(pattern):
            logger.warning("Unknown pattern %r from LLM, falling back to 'none'.", pattern)
            pattern = "none"
            result["pattern"] = pattern

        return {
            "pattern": pattern,
            "sub_agents": result.get("sub_agents", []),
            "pattern_kwargs": result.get("pattern_kwargs", {}),
            "reasoning": result.get("reasoning", ""),
        }

    def _keyword_select(self, description: str) -> dict[str, Any]:
        """Heuristic pattern selection based on keyword matching.

        Args:
            description: Task description.

        Returns:
            Orchestration recommendation dict.
        """
        desc_lower = description.lower()
        best_pattern = "none"
        best_score = 0

        for pattern, keywords in self._get_keyword_hints().items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_pattern = pattern

        # If no strong signal, stay with single agent
        if best_score < 1:
            return {
                "pattern": "none",
                "sub_agents": [],
                "pattern_kwargs": {},
                "reasoning": "No orchestration signals detected.",
            }

        return {
            "pattern": best_pattern,
            "sub_agents": [],  # keyword mode cannot decompose sub-agents
            "pattern_kwargs": {},
            "reasoning": f"Keyword heuristic matched pattern '{best_pattern}'.",
        }


class AgentConfigurator:
    """Assemble an ``AgentBlueprint`` from the generated components.

    Applies security constraints: tool allowlist, provider restrictions,
    instruction sanitisation.

    Args:
        config: Factory configuration dict (from ``_load_factory_config``).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or _load_factory_config()

    def configure(
        self,
        *,
        name: str,
        description: str,
        instruction: str,
        tool_names: list[str],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        output_format: str = "text",
    ) -> AgentBlueprint:
        """Build a validated ``AgentBlueprint``.

        Args:
            name: Agent name.
            description: Agent purpose.
            instruction: Generated system prompt.
            tool_names: Selected tool names.
            provider: LLM provider.
            model: Model name.
            temperature: LLM temperature.
            output_format: ``"text"`` or ``"json"``.

        Returns:
            A validated ``AgentBlueprint``.

        Raises:
            BlueprintValidationError: If constraints are violated.
        """
        # Sanitise name
        safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
        safe_name = safe_name.strip("_")
        if not safe_name:
            safe_name = "dynamic_agent"

        # Sanitise instruction
        max_len = int(self.config.get("max_instruction_length", 4000))
        instruction = sanitise_instruction(instruction, max_length=max_len)

        # Enforce provider restrictions
        provider = provider or self.config.get("default_provider", "google")
        allowed_providers = self.config.get("allowed_providers", [])
        if allowed_providers and provider not in allowed_providers:
            raise BlueprintValidationError(
                f"Provider {provider!r} not in allowed list: {allowed_providers}"
            )

        # Enforce tool allowlist
        tool_allowlist: list[str] = self.config.get("tool_allowlist", [])
        if tool_allowlist:
            rejected = [t for t in tool_names if t not in tool_allowlist]
            if rejected:
                logger.warning(
                    "Tools not in allowlist removed: %s", rejected,
                )
                tool_names = [t for t in tool_names if t in tool_allowlist]

        # Enforce max tools
        max_tools = int(self.config.get("max_tools_per_agent", 10))
        if len(tool_names) > max_tools:
            logger.warning(
                "Too many tools (%d > %d), truncating.", len(tool_names), max_tools,
            )
            tool_names = tool_names[:max_tools]

        return AgentBlueprint(
            name=safe_name,
            description=description,
            instruction=instruction,
            provider=provider,
            model=model or self.config.get("default_model"),
            temperature=temperature,
            tool_names=tuple(tool_names),
            output_format=output_format,
        )


# ── AgentFactory ──────────────────────────────────────────────────────────────

class AgentFactory:
    """Dynamic agent factory — generate and build agents from descriptions.

    Orchestrates the modular pipeline:
    ``SystemPromptGenerator`` → ``ToolSelector`` → ``OrchestrationSelector``
    → ``AgentConfigurator`` → ``AgentBlueprint`` / ``OrchestrationBlueprint``
    → ``LlmAgent`` or workflow agent.

    Args:
        prompt_generator: Custom system-prompt generator (or ``None`` for default).
        tool_selector: Custom tool selector (or ``None`` for default).
        orchestration_selector: Custom orchestration selector (or ``None``).
        configurator: Custom configurator (or ``None`` for default).
        config: Override factory config dict.

    Example::

        factory = AgentFactory()

        # Single agent
        blueprint = factory.generate_blueprint("Summarise documents.")
        agent = factory.build(blueprint)

        # Orchestrated multi-agent
        orch_bp = factory.generate_orchestrated_blueprint(
            "Research a topic, then write an article, then review it.",
        )
        agent = factory.build_orchestrated(orch_bp)
    """

    def __init__(
        self,
        *,
        prompt_generator: SystemPromptGenerator | None = None,
        tool_selector: ToolSelector | None = None,
        orchestration_selector: OrchestrationSelector | None = None,
        configurator: AgentConfigurator | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or _load_factory_config()
        self._prompt_gen = prompt_generator
        self._tool_sel = tool_selector
        self._orch_sel = orchestration_selector
        self._configurator = configurator or AgentConfigurator(self._config)

    def _ensure_enabled(self) -> None:
        """Raise if dynamic creation is disabled in config."""
        if not self._config.get("allow_dynamic_creation", False):
            raise DynamicCreationDisabledError(
                "Dynamic agent creation is disabled. "
                "Set agent.factory.allow_dynamic_creation = true in config.toml"
            )

    def _get_prompt_generator(self) -> SystemPromptGenerator:
        if self._prompt_gen is None:
            provider = self._config.get("default_provider", "google")
            model = self._config.get("default_model")
            self._prompt_gen = SystemPromptGenerator(
                provider=provider, model=model,
            )
        return self._prompt_gen

    def _get_tool_selector(self) -> ToolSelector:
        if self._tool_sel is None:
            provider = self._config.get("default_provider", "google")
            model = self._config.get("default_model")
            max_tools = int(self._config.get("max_tools_per_agent", 10))
            self._tool_sel = ToolSelector(
                provider=provider, model=model, max_tools=max_tools,
            )
        return self._tool_sel

    def _get_orchestration_selector(self) -> OrchestrationSelector:
        if self._orch_sel is None:
            provider = self._config.get("default_provider", "google")
            model = self._config.get("default_model")
            self._orch_sel = OrchestrationSelector(
                provider=provider, model=model,
            )
        return self._orch_sel

    def generate_blueprint(
        self,
        description: str,
        *,
        available_tools: Sequence[FunctionTool] | None = None,
        name: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        output_format: str = "text",
        use_llm_for_tools: bool = True,
        instruction_override: str | None = None,
    ) -> AgentBlueprint:
        """Generate an agent blueprint from a natural-language description.

        This is the main entry point.  It calls the three modular components
        in sequence and returns a reviewable blueprint.

        Args:
            description: Natural-language purpose of the agent.
            available_tools: Pool of tools the agent may use.
            name: Agent name (auto-generated from description if omitted).
            provider: LLM provider override.
            model: Model name override.
            temperature: LLM temperature for the generated agent.
            output_format: ``"text"`` or ``"json"``.
            use_llm_for_tools: Use LLM for tool selection (vs keyword match).
            instruction_override: Skip LLM generation, use this instruction.

        Returns:
            An ``AgentBlueprint`` ready for review and building.

        Raises:
            DynamicCreationDisabledError: If the feature is disabled.
            BlueprintValidationError: If validation fails.
        """
        self._ensure_enabled()
        logger.info("Generating blueprint for: %s", description[:80])

        # 1) System prompt
        if instruction_override:
            instruction = instruction_override
        else:
            gen = self._get_prompt_generator()
            instruction = gen.generate(description)

        # 2) Tool selection
        tool_names: list[str] = []
        if available_tools:
            sel = self._get_tool_selector()
            tool_names = sel.select(
                description, available_tools, use_llm=use_llm_for_tools,
            )

        # 3) Name generation
        if not name:
            name = self._derive_name(description)

        # 4) Assemble and validate
        blueprint = self._configurator.configure(
            name=name,
            description=description,
            instruction=instruction,
            tool_names=tool_names,
            provider=provider,
            model=model,
            temperature=temperature,
            output_format=output_format,
        )

        logger.info(
            "Blueprint generated: name=%s, tools=%s",
            blueprint.name, blueprint.tool_names,
        )
        return blueprint

    def build(
        self,
        blueprint: AgentBlueprint,
        *,
        available_tools: Sequence[FunctionTool] | None = None,
        sub_agents: list[BaseAgent] | None = None,
    ) -> BaseAgent:
        """Instantiate an ``LlmAgent`` from a blueprint.

        Args:
            blueprint: The validated blueprint.
            available_tools: Full pool of tools (filtered by ``blueprint.tool_names``).
            sub_agents: Optional sub-agents to attach.

        Returns:
            A fully configured ``LlmAgent`` instance.

        Raises:
            DynamicCreationDisabledError: If the feature is disabled.
        """
        self._ensure_enabled()

        from .llm_agent import LlmAgent

        # Resolve tools by name
        tools: list[FunctionTool] = []
        if available_tools and blueprint.tool_names:
            tool_map = {t.name: t for t in available_tools}
            tools = [tool_map[n] for n in blueprint.tool_names if n in tool_map]

        agent = LlmAgent(
            name=blueprint.name,
            model=blueprint.model,
            provider=blueprint.provider,
            instruction=blueprint.instruction,
            description=blueprint.description,
            tools=tools,
            temperature=blueprint.temperature,
            output_format=blueprint.output_format,
            sub_agents=sub_agents,
        )

        logger.info("Agent built from blueprint: %s", agent.name)
        return agent

    # ── Orchestrated generation ────────────────────────────────────────

    def generate_orchestrated_blueprint(
        self,
        description: str,
        *,
        available_tools: Sequence[FunctionTool] | None = None,
        name: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        use_llm: bool = True,
    ) -> OrchestrationBlueprint:
        """Generate a multi-agent orchestration blueprint from a description.

        Analyses the task to determine the best orchestration pattern, then
        generates an ``AgentBlueprint`` for each sub-agent.

        If the task is simple enough, the selected pattern will be ``"none"``
        and only one sub-agent blueprint is produced (equivalent to a single
        ``AgentBlueprint``).

        Args:
            description: Natural-language task description.
            available_tools: Pool of tools sub-agents may use.
            name: Orchestrator name (auto-generated if omitted).
            provider: LLM provider override.
            model: Model name override.
            temperature: LLM temperature for generated agents.
            use_llm: Use LLM for orchestration and tool selection.

        Returns:
            An ``OrchestrationBlueprint`` ready for review and building.

        Raises:
            DynamicCreationDisabledError: If the feature is disabled.
            BlueprintValidationError: If validation fails.
        """
        self._ensure_enabled()
        logger.info("Generating orchestrated blueprint for: %s", description[:80])

        # 1) Orchestration pattern selection
        orch = self._get_orchestration_selector()
        recommendation = orch.select(description, use_llm=use_llm)
        pattern = recommendation["pattern"]
        pattern_kwargs = recommendation.get("pattern_kwargs", {})
        llm_sub_agents: list[dict[str, Any]] = recommendation.get("sub_agents", [])

        logger.info(
            "Orchestration selected: pattern=%s, reasoning=%s",
            pattern, recommendation.get("reasoning", ""),
        )

        # 2) Build sub-agent blueprints
        provider = provider or self._config.get("default_provider", "google")
        sub_blueprints: list[AgentBlueprint] = []

        if pattern == "none":
            # Single agent — generate a standard blueprint
            bp = self.generate_blueprint(
                description,
                available_tools=available_tools,
                name=name,
                provider=provider,
                model=model,
                temperature=temperature,
                use_llm_for_tools=use_llm,
            )
            sub_blueprints.append(bp)
        elif llm_sub_agents:
            # LLM provided sub-agent decomposition
            for sa in llm_sub_agents:
                sa_name = sa.get("name", "sub_agent")
                sa_desc = sa.get("description", "")
                sa_instruction = sa.get("instruction", "")

                # Select tools for this sub-agent
                tool_names: list[str] = []
                if available_tools:
                    sel = self._get_tool_selector()
                    sa_full_desc = f"{sa_desc}. {sa_instruction}"
                    tool_names = sel.select(
                        sa_full_desc, available_tools, use_llm=False,
                    )

                bp = self._configurator.configure(
                    name=sa_name,
                    description=sa_desc,
                    instruction=sa_instruction or f"You are a {sa_desc}.",
                    tool_names=tool_names,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                )
                sub_blueprints.append(bp)
        else:
            # Keyword mode: no sub-agent decomposition from LLM.
            # Create a single worker agent.
            bp = self.generate_blueprint(
                description,
                available_tools=available_tools,
                name=f"{name or 'worker'}_agent",
                provider=provider,
                model=model,
                temperature=temperature,
                use_llm_for_tools=use_llm,
            )
            sub_blueprints.append(bp)

        # 3) Orchestrator name
        if not name:
            name = self._derive_name(description) + "_orchestrator"

        orch_bp = OrchestrationBlueprint(
            name=re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_") or "orchestrator",
            description=description,
            pattern=pattern,
            sub_agent_blueprints=tuple(sub_blueprints),
            pattern_kwargs=pattern_kwargs,
            provider=provider,
            model=model,
            metadata={"reasoning": recommendation.get("reasoning", "")},
        )

        logger.info(
            "Orchestration blueprint: pattern=%s, sub_agents=%d",
            orch_bp.pattern, len(orch_bp.sub_agent_blueprints),
        )
        return orch_bp

    def build_orchestrated(
        self,
        blueprint: OrchestrationBlueprint,
        *,
        available_tools: Sequence[FunctionTool] | None = None,
    ) -> BaseAgent:
        """Instantiate a workflow agent from an orchestration blueprint.

        Args:
            blueprint: The validated orchestration blueprint.
            available_tools: Full pool of tools for sub-agents.

        Returns:
            A workflow agent (``SequentialAgent``, ``PlannerAgent``, etc.)
            wrapping the generated sub-agents.  If the pattern is ``"none"``,
            returns a single ``LlmAgent``.

        Raises:
            DynamicCreationDisabledError: If the feature is disabled.
            BlueprintValidationError: If the pattern is unknown.
        """
        self._ensure_enabled()

        # Build sub-agents from blueprints
        sub_agents: list[BaseAgent] = []
        for sa_bp in blueprint.sub_agent_blueprints:
            agent = self.build(sa_bp, available_tools=available_tools)
            sub_agents.append(agent)

        pattern = blueprint.pattern

        # Single agent — no orchestration
        if pattern == "none":
            if sub_agents:
                return sub_agents[0]
            raise BlueprintValidationError("Pattern 'none' requires at least one sub-agent.")

        # Validate pattern
        if not OrchestrationRegistry.contains(pattern):
            raise BlueprintValidationError(
                f"Unknown orchestration pattern {pattern!r}. "
                f"Available: {OrchestrationRegistry.list_patterns()}"
            )

        return self._instantiate_orchestrator(
            pattern=pattern,
            name=blueprint.name,
            description=blueprint.description,
            sub_agents=sub_agents,
            provider=blueprint.provider,
            model=blueprint.model,
            pattern_kwargs=blueprint.pattern_kwargs,
        )

    @staticmethod
    def _instantiate_orchestrator(
        *,
        pattern: str,
        name: str,
        description: str,
        sub_agents: list[BaseAgent],
        provider: str,
        model: str | None,
        pattern_kwargs: dict[str, Any],
    ) -> BaseAgent:
        """Create the appropriate workflow agent for the pattern.

        Looks up the pattern in :class:`OrchestrationRegistry` and calls
        the registered factory function.  Custom patterns registered via
        :func:`register_pattern` are handled automatically.

        Args:
            pattern: Orchestration pattern key.
            name: Orchestrator name.
            description: Orchestrator description.
            sub_agents: Pre-built sub-agents.
            provider: LLM provider.
            model: Model name.
            pattern_kwargs: Extra kwargs for the constructor.

        Returns:
            An orchestration agent instance.

        Raises:
            BlueprintValidationError: If the pattern has no factory or
                sub-agent constraints are not met.
        """
        registration = OrchestrationRegistry.get(pattern)

        # Validate minimum sub-agent count
        if len(sub_agents) < registration.min_sub_agents:
            raise BlueprintValidationError(
                f"Pattern {pattern!r} ({registration.class_name}) requires "
                f"at least {registration.min_sub_agents} sub-agent(s), "
                f"got {len(sub_agents)}."
            )

        factory = registration.factory
        if factory is None:
            raise BlueprintValidationError(
                f"Pattern {pattern!r} has no factory function registered. "
                f"Register one via register_pattern(key={pattern!r}, factory=...)."
            )

        return factory(
            name=name,
            description=description,
            sub_agents=sub_agents,
            provider=provider,
            model=model,
            pattern_kwargs=pattern_kwargs,
        )

    @staticmethod
    def _derive_name(description: str) -> str:
        """Derive a snake_case agent name from a description.

        Args:
            description: Agent description text.

        Returns:
            A short, safe agent name.
        """
        words = re.findall(r"[a-zA-Z]+", description)
        name_words = [w.lower() for w in words[:4]]
        name = "_".join(name_words) if name_words else "dynamic_agent"
        # Ensure reasonable length
        return name[:40]


# ── Convenience function ──────────────────────────────────────────────────────

def create_agent_from_prompt(
    description: str,
    *,
    available_tools: Sequence[FunctionTool] | None = None,
    name: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    output_format: str = "text",
    review_callback: Optional[Any] = None,
    config: dict[str, Any] | None = None,
) -> BaseAgent:
    """One-liner: generate and build an agent from a description.

    Optionally pass a ``review_callback(blueprint) -> bool`` to inspect the
    blueprint before building.  If the callback returns ``False`` the agent
    is not created and ``BlueprintValidationError`` is raised.

    Args:
        description: Natural-language agent purpose.
        available_tools: Tools the agent may use.
        name: Agent name (auto-generated if omitted).
        provider: LLM provider.
        model: Model name.
        temperature: LLM temperature.
        output_format: ``"text"`` or ``"json"``.
        review_callback: Optional ``(AgentBlueprint) -> bool`` for HITL review.
        config: Override factory config dict.

    Returns:
        A configured ``LlmAgent`` instance.

    Raises:
        DynamicCreationDisabledError: If dynamic creation is disabled.
        BlueprintValidationError: If review rejects the blueprint.

    Example::

        agent = create_agent_from_prompt(
            "A research agent that searches the web and summarises findings.",
            available_tools=[web_search, summarise],
        )
    """
    factory = AgentFactory(config=config)
    blueprint = factory.generate_blueprint(
        description,
        available_tools=available_tools,
        name=name,
        provider=provider,
        model=model,
        temperature=temperature,
        output_format=output_format,
    )

    if review_callback is not None:
        if not review_callback(blueprint):
            raise BlueprintValidationError(
                "Blueprint rejected by review callback."
            )

    return factory.build(blueprint, available_tools=available_tools)
