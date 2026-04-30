"""
Skill system for Nono agents.

A **Skill** is a reusable, composable unit of AI capability that bundles
an agent (or pipeline), domain-specific tools, and a state contract into
a single object.  Skills can be:

- **Attached** to an ``LlmAgent`` via its ``skills`` parameter — they are
  auto-converted to function-calling tools.
- **Used standalone** via ``skill.run("message")``.
- **Composed** into pipelines by extracting their inner agent with
  ``skill.build_agent()``.
- **Discovered** via the global ``registry``.

Architecture::

    ┌──────────────────────────────────────────────────┐
    │  BaseSkill                                       │
    │  ├── descriptor   → SkillDescriptor (metadata)   │
    │  ├── build_agent  → BaseAgent (execution)        │
    │  ├── build_tools  → list[FunctionTool] (tools)   │
    │  ├── as_tool()    → FunctionTool (for LLM)       │
    │  └── run()        → str (standalone execution)   │
    └──────────────────────────────────────────────────┘

Usage::

    from nono.agent.skill import BaseSkill, SkillDescriptor
    from nono.agent.skills import SummarizeSkill

    # Standalone
    skill = SummarizeSkill()
    result = skill.run("Long text to summarize...")

    # As a tool inside an agent
    from nono.agent import Agent
    agent = Agent(
        name="analyst",
        instruction="You analyze data. Use your skills when needed.",
        skills=[SummarizeSkill(), ClassifySkill()],
    )

    # Via registry
    from nono.agent.skill import registry
    for desc in registry.list_skills():
        print(f"{desc.name}: {desc.description}")

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent, Session
from .tool import FunctionTool

logger = logging.getLogger("Nono.Agent.Skill")

__all__ = [
    "BaseSkill",
    "SkillDescriptor",
    "SkillRegistry",
    "registry",
    "skill_from_agent",
]


# ── Descriptor ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SkillDescriptor:
    """Immutable metadata describing a skill's capabilities.

    Follows the `Agent Skills <https://agentskills.io/specification>`_
    open standard.

    Args:
        name: Unique skill identifier (used in registry, CLI, API).
        description: Human-readable description of what the skill does.
        version: Semantic version string.
        tags: Classification tags for discovery (e.g. ``("text", "analysis")``).
        input_keys: State keys the skill reads from ``session.state``.
        output_keys: State keys the skill writes to ``session.state``.
        license: License name or reference to a bundled license file.
        compatibility: Environment requirements (intended product,
            system packages, network access, etc.).  Max 500 chars.
        metadata: Arbitrary key-value map for additional properties
            not defined by the core spec.
        allowed_tools: Pre-approved tools the skill may use
            (space-delimited in YAML, stored as a tuple).
    """

    name: str
    description: str
    version: str = "1.0.0"
    tags: tuple[str, ...] = ()
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()


# ── BaseSkill ─────────────────────────────────────────────────────────────────


def _inject_tools(agent: BaseAgent, tools: list[FunctionTool]) -> None:
    """Inject domain-specific tools into an agent when it supports them.

    Only ``LlmAgent`` (or subclasses with a ``tools`` list attribute) can
    receive tools.  Other agent types silently ignore the call.

    Args:
        agent: The agent returned by ``build_agent()``.
        tools: Tools returned by ``build_tools()``.
    """
    if not tools:
        return

    agent_tools = getattr(agent, "tools", None)

    if isinstance(agent_tools, list):
        # Avoid duplicates — check by tool name
        existing = {t.name for t in agent_tools}
        for tool in tools:
            if tool.name not in existing:
                agent_tools.append(tool)


class BaseSkill(ABC):
    """Abstract base for all Nono skills.

    Subclasses must implement :meth:`descriptor` and :meth:`build_agent`.
    """

    @property
    @abstractmethod
    def descriptor(self) -> SkillDescriptor:
        """Return the skill's metadata descriptor."""

    @abstractmethod
    def build_agent(self, **overrides: Any) -> BaseAgent:
        """Create the agent (or pipeline) that executes this skill.

        Args:
            **overrides: Optional overrides for provider, model, etc.

        Returns:
            A configured ``BaseAgent`` ready to run.
        """

    def build_tools(self) -> list[FunctionTool]:
        """Return domain-specific tools for the skill's agent.

        Override in subclasses that need tools beyond what the LLM
        provides natively.  Default returns an empty list.

        Returns:
            List of ``FunctionTool`` instances.
        """
        return []

    def as_tool(self) -> FunctionTool:
        """Convert this skill into a ``FunctionTool`` for LLM function-calling.

        The generated tool accepts a single ``input`` string parameter,
        runs the skill's agent via ``Runner``, and returns the result.
        If the skill defines ``build_tools()``, those tools are injected
        into the inner agent before execution.

        Returns:
            A ``FunctionTool`` wrapping this skill.
        """
        desc = self.descriptor

        def _invoke(input: str) -> str:  # noqa: A002
            from .runner import Runner

            agent = self.build_agent()
            _inject_tools(agent, self.build_tools())
            return Runner(agent).run(input)

        return FunctionTool(
            fn=_invoke,
            name=desc.name,
            description=desc.description,
        )

    def run(
        self,
        user_message: str,
        *,
        session: Session | None = None,
        **overrides: Any,
    ) -> str:
        """Execute the skill standalone.

        Args:
            user_message: Input text for the skill.
            session: Optional session to share state.
            **overrides: Forwarded to :meth:`build_agent`.

        Returns:
            The agent's final text response.
        """
        from .runner import Runner

        agent = self.build_agent(**overrides)
        _inject_tools(agent, self.build_tools())
        return Runner(agent, session=session).run(user_message)

    def __repr__(self) -> str:
        desc = self.descriptor
        return f"{self.__class__.__name__}(name={desc.name!r})"


# ── SkillRegistry ─────────────────────────────────────────────────────────────

class SkillRegistry:
    """Global registry for skill discovery.

    Skills register themselves (or are registered externally) so that
    the CLI, API, and other agents can discover and invoke them by name.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(
        self,
        skill_or_cls: BaseSkill | type[BaseSkill],
    ) -> BaseSkill | type[BaseSkill]:
        """Register a skill instance or class.

        Can be used as a decorator on a class::

            @registry.register
            class MySkill(BaseSkill):
                ...

        Or called directly with an instance::

            registry.register(SummarizeSkill())

        Args:
            skill_or_cls: A ``BaseSkill`` instance or subclass.

        Returns:
            The same object (passthrough for decorator usage).
        """
        if isinstance(skill_or_cls, type):
            instance = skill_or_cls()
        else:
            instance = skill_or_cls

        name = instance.descriptor.name
        self._skills[name] = instance
        logger.debug("Registered skill %r", name)
        return skill_or_cls

    def get(self, name: str) -> BaseSkill | None:
        """Look up a skill by name.

        Args:
            name: Skill name (from ``descriptor.name``).

        Returns:
            The skill instance, or ``None`` if not found.
        """
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDescriptor]:
        """Return descriptors for all registered skills.

        Returns:
            Sorted list of ``SkillDescriptor``.
        """
        return sorted(
            (s.descriptor for s in self._skills.values()),
            key=lambda d: d.name,
        )

    def find_by_tag(self, tag: str) -> list[SkillDescriptor]:
        """Find skills that have a specific tag.

        Args:
            tag: Tag string to search for.

        Returns:
            List of matching ``SkillDescriptor``.
        """
        return [
            d for d in self.list_skills() if tag in d.tags
        ]

    @property
    def names(self) -> list[str]:
        """Sorted list of registered skill names."""
        return sorted(self._skills.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={self.names})"


# ── Singleton registry ────────────────────────────────────────────────────────

registry = SkillRegistry()
"""Module-level singleton registry.  Import and use directly::

    from nono.agent.skill import registry
    registry.register(MySkill())
"""


# ── Convenience factory ──────────────────────────────────────────────────────

def skill_from_agent(
    *,
    name: str,
    description: str,
    agent_factory: Any,
    tags: tuple[str, ...] = (),
    input_keys: tuple[str, ...] = (),
    output_keys: tuple[str, ...] = (),
    version: str = "1.0.0",
    register: bool = False,
) -> BaseSkill:
    """Create a skill from an existing agent factory function.

    Wraps any callable that returns a ``BaseAgent`` (like the functions
    in ``nono.agent.templates``) as a ``BaseSkill``.

    Args:
        name: Skill name.
        description: Human-readable description.
        agent_factory: Callable that returns a ``BaseAgent``.
        tags: Discovery tags.
        input_keys: State keys the skill reads.
        output_keys: State keys the skill writes.
        version: Skill version.
        register: If ``True``, auto-register in the global ``registry``.

    Returns:
        A ``BaseSkill`` wrapping the factory.

    Example::

        from nono.agent.templates import summarizer_agent
        from nono.agent.skill import skill_from_agent

        summarize = skill_from_agent(
            name="summarize",
            description="Summarize text.",
            agent_factory=summarizer_agent,
        )
        result = summarize.run("Long text here...")
    """
    desc = SkillDescriptor(
        name=name,
        description=description,
        version=version,
        tags=tags,
        input_keys=input_keys,
        output_keys=output_keys,
    )

    class _WrappedSkill(BaseSkill):
        @property
        def descriptor(self) -> SkillDescriptor:
            return desc

        def build_agent(self, **overrides: Any) -> BaseAgent:
            return agent_factory(**overrides)

    skill = _WrappedSkill()
    if register:
        registry.register(skill)
    return skill
