"""Tests for the Nono skill system."""

from __future__ import annotations

import pytest
from typing import Any, Iterator

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.skill import (
    BaseSkill,
    SkillDescriptor,
    SkillRegistry,
    registry,
    skill_from_agent,
)
from nono.agent.tool import FunctionTool
from nono.agent.runner import Runner


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _EchoAgent(BaseAgent):
    """Minimal agent that echoes the user message."""

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo: {ctx.user_message}")

    async def _run_async_impl(self, ctx: InvocationContext):
        yield Event(EventType.AGENT_MESSAGE, self.name, f"echo: {ctx.user_message}")


class _EchoSkill(BaseSkill):
    """Minimal skill wrapping an echo agent."""

    def __init__(self, name: str = "echo", tags: tuple[str, ...] = ("test",)):
        self._name = name
        self._tags = tags

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name=self._name,
            description="Echoes input back.",
            version="1.0.0",
            tags=self._tags,
            input_keys=("input",),
            output_keys=("output",),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        return _EchoAgent(name=f"{self._name}_agent")


class _ToolEchoAgent(BaseAgent):
    """Echo agent that also lists its tools in the output."""

    def __init__(self, name: str, tools: list[FunctionTool] | None = None):
        super().__init__(name=name)
        self.tools: list[FunctionTool] = tools or []

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        tool_names = [t.name for t in self.tools]
        yield Event(
            EventType.AGENT_MESSAGE,
            self.name,
            f"echo: {ctx.user_message} tools={tool_names}",
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        tool_names = [t.name for t in self.tools]
        yield Event(
            EventType.AGENT_MESSAGE,
            self.name,
            f"echo: {ctx.user_message} tools={tool_names}",
        )


class _SkillWithTools(BaseSkill):
    """Skill that provides domain-specific tools via build_tools()."""

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="tooled",
            description="Skill with custom tools.",
            tags=("test",),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:
        return _ToolEchoAgent(name="tooled_agent")

    def build_tools(self) -> list[FunctionTool]:
        def lookup(word: str) -> str:
            return f"found: {word}"

        return [
            FunctionTool(fn=lookup, name="lexicon", description="Look up a word."),
        ]


class _StatefulSkill(BaseSkill):
    """Skill that reads and writes session state."""

    @property
    def descriptor(self) -> SkillDescriptor:
        return SkillDescriptor(
            name="stateful",
            description="Reads count from state, increments it.",
            input_keys=("count",),
            output_keys=("count",),
        )

    def build_agent(self, **overrides: Any) -> BaseAgent:

        class _Agent(BaseAgent):
            def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
                n = ctx.session.state.get("count", 0)
                ctx.session.state["count"] = n + 1
                yield Event(EventType.AGENT_MESSAGE, self.name, f"count={n + 1}")

            async def _run_async_impl(self, ctx: InvocationContext):
                for evt in self._run_impl(ctx):
                    yield evt

        return _Agent(name="stateful_agent")


# ── SkillDescriptor ───────────────────────────────────────────────────────────

class TestSkillDescriptor:

    def test_defaults(self):
        d = SkillDescriptor(name="test", description="A test skill")
        assert d.name == "test"
        assert d.description == "A test skill"
        assert d.version == "1.0.0"
        assert d.tags == ()
        assert d.input_keys == ()
        assert d.output_keys == ()

    def test_with_tags(self):
        d = SkillDescriptor(name="x", description="y", tags=("a", "b"))
        assert d.tags == ("a", "b")

    def test_frozen(self):
        d = SkillDescriptor(name="x", description="y")
        with pytest.raises(AttributeError):
            d.name = "z"  # type: ignore[misc]


# ── BaseSkill ─────────────────────────────────────────────────────────────────

class TestBaseSkill:

    def test_run_standalone(self):
        skill = _EchoSkill()
        result = skill.run("hello")
        assert result == "echo: hello"

    def test_run_with_session(self):
        skill = _StatefulSkill()
        session = Session()
        skill.run("go", session=session)
        assert session.state["count"] == 1
        skill.run("go", session=session)
        assert session.state["count"] == 2

    def test_as_tool(self):
        skill = _EchoSkill()
        tool = skill.as_tool()
        assert isinstance(tool, FunctionTool)
        assert tool.name == "echo"
        assert "Echoes input" in tool.description
        result = tool.invoke({"input": "test"})
        assert result == "echo: test"

    def test_build_tools_default(self):
        skill = _EchoSkill()
        assert skill.build_tools() == []

    def test_descriptor_property(self):
        skill = _EchoSkill()
        d = skill.descriptor
        assert d.name == "echo"
        assert d.tags == ("test",)

    def test_repr(self):
        skill = _EchoSkill()
        assert "EchoSkill" in repr(skill)
        assert "echo" in repr(skill)


# ── SkillRegistry ─────────────────────────────────────────────────────────────

class TestSkillRegistry:

    def test_register_instance(self):
        reg = SkillRegistry()
        skill = _EchoSkill()
        reg.register(skill)
        assert "echo" in reg
        assert reg.get("echo") is skill

    def test_register_class(self):
        reg = SkillRegistry()
        reg.register(_EchoSkill)
        assert "echo" in reg
        assert isinstance(reg.get("echo"), _EchoSkill)

    def test_register_as_decorator(self):
        reg = SkillRegistry()

        @reg.register
        class MySkill(_EchoSkill):
            pass

        assert "echo" in reg

    def test_get_nonexistent(self):
        reg = SkillRegistry()
        assert reg.get("nonexistent") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(_EchoSkill(name="alpha"))
        reg.register(_EchoSkill(name="beta"))
        descriptors = reg.list_skills()
        assert len(descriptors) == 2
        assert descriptors[0].name == "alpha"
        assert descriptors[1].name == "beta"

    def test_find_by_tag(self):
        reg = SkillRegistry()
        reg.register(_EchoSkill(name="a", tags=("text", "analysis")))
        reg.register(_EchoSkill(name="b", tags=("code",)))
        reg.register(_EchoSkill(name="c", tags=("text",)))

        text_skills = reg.find_by_tag("text")
        assert len(text_skills) == 2
        names = {d.name for d in text_skills}
        assert names == {"a", "c"}

    def test_names_property(self):
        reg = SkillRegistry()
        reg.register(_EchoSkill(name="z"))
        reg.register(_EchoSkill(name="a"))
        assert reg.names == ["a", "z"]

    def test_len(self):
        reg = SkillRegistry()
        assert len(reg) == 0
        reg.register(_EchoSkill())
        assert len(reg) == 1

    def test_contains(self):
        reg = SkillRegistry()
        assert "echo" not in reg
        reg.register(_EchoSkill())
        assert "echo" in reg

    def test_repr(self):
        reg = SkillRegistry()
        reg.register(_EchoSkill())
        assert "echo" in repr(reg)


# ── skill_from_agent factory ─────────────────────────────────────────────────

class TestSkillFromAgent:

    def test_wraps_factory(self):
        def my_factory(**kwargs) -> BaseAgent:
            return _EchoAgent(name="wrapped")

        skill = skill_from_agent(
            name="my_echo",
            description="Test wrapper",
            agent_factory=my_factory,
            tags=("test",),
        )

        assert skill.descriptor.name == "my_echo"
        assert skill.descriptor.tags == ("test",)
        result = skill.run("hello")
        assert result == "echo: hello"

    def test_auto_register(self):
        reg_before = len(registry)

        def factory(**kw) -> BaseAgent:
            return _EchoAgent(name="auto")

        skill_from_agent(
            name="auto_registered_test",
            description="Auto",
            agent_factory=factory,
            register=True,
        )
        assert len(registry) > reg_before
        assert "auto_registered_test" in registry

    def test_as_tool_from_factory(self):
        def factory(**kw) -> BaseAgent:
            return _EchoAgent(name="tool_test")

        skill = skill_from_agent(
            name="tool_echo",
            description="Tool test",
            agent_factory=factory,
        )
        tool = skill.as_tool()
        assert tool.name == "tool_echo"
        result = tool.invoke({"input": "hi"})
        assert result == "echo: hi"


# ── Global registry built-in skills ──────────────────────────────────────────

class TestBuiltInSkills:

    def test_built_in_skills_registered(self):
        """Importing nono.agent.skills registers built-in skills."""
        import nono.agent.skills  # noqa: F401

        assert "summarize" in registry
        assert "classify" in registry
        assert "extract" in registry
        assert "code_review" in registry
        assert "translate" in registry

    def test_built_in_descriptors(self):
        import nono.agent.skills  # noqa: F401

        descs = registry.list_skills()
        names = {d.name for d in descs}
        assert names >= {"summarize", "classify", "extract", "code_review", "translate"}

    def test_summarize_descriptor(self):
        skill = registry.get("summarize")
        assert skill is not None
        d = skill.descriptor
        assert "summariz" in d.description.lower()
        assert "text" in d.tags

    def test_classify_descriptor(self):
        skill = registry.get("classify")
        assert skill is not None
        d = skill.descriptor
        assert "classif" in d.description.lower()

    def test_extract_descriptor(self):
        skill = registry.get("extract")
        assert skill is not None
        assert "extract" in skill.descriptor.description.lower()

    def test_code_review_descriptor(self):
        skill = registry.get("code_review")
        assert skill is not None
        assert "code" in skill.descriptor.tags

    def test_translate_descriptor(self):
        skill = registry.get("translate")
        assert skill is not None
        assert "translat" in skill.descriptor.description.lower()

    def test_find_text_skills(self):
        import nono.agent.skills  # noqa: F401
        text_skills = registry.find_by_tag("text")
        assert len(text_skills) >= 3  # summarize, classify, extract, translate

    def test_build_agent_returns_base_agent(self):
        for name in ("summarize", "classify", "extract", "code_review", "translate"):
            skill = registry.get(name)
            assert skill is not None
            agent = skill.build_agent()
            assert isinstance(agent, BaseAgent)
            assert agent.name  # has a name

    def test_as_tool_returns_function_tool(self):
        for name in ("summarize", "classify", "extract", "code_review", "translate"):
            skill = registry.get(name)
            assert skill is not None
            tool = skill.as_tool()
            assert isinstance(tool, FunctionTool)
            assert tool.name == name


# ── LlmAgent skills integration ──────────────────────────────────────────────

class TestLlmAgentSkills:

    def test_skills_appear_in_all_tools(self):
        from nono.agent.llm_agent import LlmAgent

        skill = _EchoSkill()
        agent = LlmAgent(
            name="test",
            instruction="test",
            skills=[skill],
        )
        tools = agent._all_tools
        tool_names = [t.name for t in tools]
        assert "echo" in tool_names

    def test_skills_and_tools_combined(self):
        from nono.agent.llm_agent import LlmAgent

        def my_fn(x: str) -> str:
            return x

        explicit_tool = FunctionTool(fn=my_fn, name="my_tool", description="A tool")
        skill = _EchoSkill()

        agent = LlmAgent(
            name="test",
            instruction="test",
            tools=[explicit_tool],
            skills=[skill],
        )
        tools = agent._all_tools
        tool_names = [t.name for t in tools]
        assert "my_tool" in tool_names
        assert "echo" in tool_names

    def test_empty_skills_no_extra_tools(self):
        from nono.agent.llm_agent import LlmAgent

        agent = LlmAgent(name="test", instruction="test")
        assert agent.skills == []
        assert agent._all_tools == []


# ── build_tools injection ─────────────────────────────────────────────────────

class TestBuildToolsInjection:
    """Verify that build_tools() are wired into as_tool() and run()."""

    def test_run_injects_tools(self):
        """run() injects build_tools() into the agent returned by build_agent()."""
        skill = _SkillWithTools()
        result = skill.run("hello")
        assert "lexicon" in result  # tool name appears in echo output

    def test_as_tool_injects_tools(self):
        """as_tool() wraps run, which injects build_tools()."""
        skill = _SkillWithTools()
        tool = skill.as_tool()
        result = tool.invoke({"input": "hello"})
        assert "lexicon" in result

    def test_no_tools_no_injection(self):
        """Skill without build_tools() produces no extra tools on the agent."""
        skill = _EchoSkill()
        result = skill.run("hello")
        assert result == "echo: hello"

    def test_build_tools_not_duplicated(self):
        """Calling run() twice doesn't duplicate tools."""
        skill = _SkillWithTools()
        # Run twice
        r1 = skill.run("a")
        r2 = skill.run("b")
        # Each creates a fresh agent, so no accumulation
        assert r1.count("lexicon") == 1
        assert r2.count("lexicon") == 1
