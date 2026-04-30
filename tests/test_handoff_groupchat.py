"""Tests for HandoffAgent and GroupChatAgent.

Verifies:
- HandoffAgent transfers control based on handoff directives.
- HandoffAgent respects handoff rules (disallowed targets blocked).
- HandoffAgent stops when no handoff directive in output.
- HandoffAgent respects max_handoffs safety limit.
- GroupChatAgent cycles through agents with round-robin.
- GroupChatAgent terminates on keyword.
- GroupChatAgent terminates on termination_condition.
- GroupChatAgent supports custom speaker selection.
- GroupChatAgent stores transcript in result_key.
- Both sync and async paths work correctly.

Run:
    python -m pytest tests/test_handoff_groupchat.py -v
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import AsyncIterator, Iterator

from nono.agent.base import (
    BaseAgent,
    Event,
    EventType,
    InvocationContext,
    Session,
)
from nono.agent.workflow_agents import (
    GroupChatAgent,
    HandoffAgent,
)


# ── Stub agents ───────────────────────────────────────────────────────────────


class StubAgent(BaseAgent):
    """Returns a fixed response."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description=f"stub-{name}")
        self._response = response

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class CountingAgent(BaseAgent):
    """Tracks how many times it was called."""

    def __init__(self, *, name: str, response: str = "ok") -> None:
        super().__init__(name=name, description=f"counting-{name}")
        self._response = response
        self.call_count = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self.call_count += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        self.call_count += 1
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


class HandoffTriageAgent(BaseAgent):
    """Simulates triage: hands off to a target based on content."""

    def __init__(self, *, name: str, target: str) -> None:
        super().__init__(name=name, description=f"triage-{name}")
        self._target = target

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"Routing to specialist.\nHANDOFF: {self._target}",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"Routing to specialist.\nHANDOFF: {self._target}",
        )


def _make_ctx(user_message: str = "test") -> InvocationContext:
    """Create a minimal invocation context."""
    return InvocationContext(
        session=Session(),
        user_message=user_message,
    )


def _collect_events(events: Iterator[Event]) -> list[Event]:
    """Collect all events from an iterator."""
    return list(events)


def _collect_async_events(events: AsyncIterator[Event]) -> list[Event]:
    """Collect all events from an async iterator."""
    async def _gather() -> list[Event]:
        return [event async for event in events]
    return asyncio.run(_gather())


# ══════════════════════════════════════════════════════════════════════════════
# HandoffAgent Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHandoffAgent:
    """Tests for HandoffAgent."""

    def test_basic_handoff(self) -> None:
        """Triage hands off to math_tutor, which completes the task."""
        triage = HandoffTriageAgent(name="triage", target="math_tutor")
        math = StubAgent(name="math_tutor", response="The answer is 42.")
        history = StubAgent(name="history_tutor", response="WWII ended in 1945.")

        agent = HandoffAgent(
            name="tutoring",
            entry_agent=triage,
            handoff_rules={
                "triage": [math, history],
                "math_tutor": [triage],
                "history_tutor": [triage],
            },
        )

        ctx = _make_ctx("What is 6 * 7?")
        events = _collect_events(agent._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]

        assert len(messages) == 2  # triage + math_tutor
        assert len(transfers) == 1
        assert transfers[0].data["from"] == "triage"
        assert transfers[0].data["to"] == "math_tutor"
        assert "42" in messages[-1].content

    def test_no_handoff_completes(self) -> None:
        """Agent that doesn't issue handoff completes immediately."""
        entry = StubAgent(name="entry", response="Direct answer.")

        agent = HandoffAgent(
            name="simple",
            entry_agent=entry,
        )

        ctx = _make_ctx("Hello")
        events = _collect_events(agent._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]

        assert len(messages) == 1
        assert len(transfers) == 0
        assert messages[0].content == "Direct answer."

    def test_disallowed_handoff_blocked(self) -> None:
        """Handoff to an agent not in the rules is silently blocked."""
        triage = HandoffTriageAgent(name="triage", target="secret_agent")
        math = StubAgent(name="math_tutor", response="Math answer.")

        agent = HandoffAgent(
            name="restricted",
            entry_agent=triage,
            handoff_rules={
                "triage": [math],
            },
        )

        ctx = _make_ctx("Hack the system")
        events = _collect_events(agent._run_impl_traced(ctx))

        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]
        assert len(transfers) == 0  # blocked

    def test_max_handoffs_respected(self) -> None:
        """Handoff chain stops at max_handoffs."""
        # Both agents keep handing off to each other
        a = HandoffTriageAgent(name="agent_a", target="agent_b")
        b = HandoffTriageAgent(name="agent_b", target="agent_a")

        agent = HandoffAgent(
            name="ping_pong",
            entry_agent=a,
            handoff_rules={
                "agent_a": [b],
                "agent_b": [a],
            },
            max_handoffs=3,
        )

        ctx = _make_ctx("Start")
        events = _collect_events(agent._run_impl_traced(ctx))

        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]
        # max_handoffs=3 means 3 transfers (entry + 3 handoffs = 4 agent runs)
        assert len(transfers) == 3

    def test_chain_handoff(self) -> None:
        """Multi-hop: triage → specialist → back to triage → different specialist."""
        triage = HandoffTriageAgent(name="triage", target="math")
        math = HandoffTriageAgent(name="math", target="triage")
        # When triage runs second time, it will try to hand off to math again
        # but we only care about the chain happening

        agent = HandoffAgent(
            name="chain",
            entry_agent=triage,
            handoff_rules={
                "triage": [math],
                "math": [triage],
            },
            max_handoffs=2,
        )

        ctx = _make_ctx("Complex question")
        events = _collect_events(agent._run_impl_traced(ctx))

        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]
        assert len(transfers) == 2
        assert transfers[0].data["from"] == "triage"
        assert transfers[0].data["to"] == "math"
        assert transfers[1].data["from"] == "math"
        assert transfers[1].data["to"] == "triage"

    def test_custom_handoff_keyword(self) -> None:
        """Custom handoff keyword is respected."""
        class CustomHandoffAgent(BaseAgent):
            def __init__(self, *, name: str, target: str) -> None:
                super().__init__(name=name)
                self._target = target

            def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
                yield Event(
                    EventType.AGENT_MESSAGE, self.name,
                    f"TRANSFER_TO: {self._target}",
                )

            async def _run_async_impl(
                self, ctx: InvocationContext,
            ) -> AsyncIterator[Event]:
                yield Event(
                    EventType.AGENT_MESSAGE, self.name,
                    f"TRANSFER_TO: {self._target}",
                )

        triage = CustomHandoffAgent(name="triage", target="specialist")
        specialist = StubAgent(name="specialist", response="Done.")

        agent = HandoffAgent(
            name="custom_kw",
            entry_agent=triage,
            handoff_rules={"triage": [specialist]},
            handoff_keyword="TRANSFER_TO:",
        )

        ctx = _make_ctx("test")
        events = _collect_events(agent._run_impl_traced(ctx))

        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]
        assert len(transfers) == 1

    def test_async_handoff(self) -> None:
        """Async handoff works correctly."""
        triage = HandoffTriageAgent(name="triage", target="math_tutor")
        math = StubAgent(name="math_tutor", response="Async answer: 42.")

        agent = HandoffAgent(
            name="async_tutoring",
            entry_agent=triage,
            handoff_rules={"triage": [math]},
        )

        ctx = _make_ctx("Async: what is 6*7?")
        events = _collect_async_events(agent._run_async_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        transfers = [e for e in events if e.event_type == EventType.AGENT_TRANSFER]

        assert len(messages) == 2
        assert len(transfers) == 1
        assert "42" in messages[-1].content

    def test_sub_agents_collected(self) -> None:
        """All agents from handoff_rules are registered as sub_agents."""
        a = StubAgent(name="a")
        b = StubAgent(name="b")
        c = StubAgent(name="c")

        agent = HandoffAgent(
            name="test",
            entry_agent=a,
            handoff_rules={
                "a": [b, c],
                "b": [a],
            },
        )

        names = {sa.name for sa in agent.sub_agents}
        assert names == {"a", "b", "c"}


# ══════════════════════════════════════════════════════════════════════════════
# GroupChatAgent Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGroupChatAgent:
    """Tests for GroupChatAgent."""

    def test_round_robin_basic(self) -> None:
        """Round-robin cycles through agents in order."""
        a = CountingAgent(name="agent_a", response="A speaks")
        b = CountingAgent(name="agent_b", response="B speaks")

        chat = GroupChatAgent(
            name="chat",
            sub_agents=[a, b],
            speaker_selection="round_robin",
            max_rounds=4,
        )

        ctx = _make_ctx("Start the discussion")
        events = _collect_events(chat._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(messages) == 4  # 4 rounds
        assert a.call_count == 2
        assert b.call_count == 2

    def test_termination_keyword(self) -> None:
        """Group chat terminates when keyword detected in output."""
        writer = StubAgent(name="writer", response="Here is my draft.")
        reviewer = StubAgent(name="reviewer", response="Looks great. APPROVED!")

        chat = GroupChatAgent(
            name="creative",
            sub_agents=[writer, reviewer],
            speaker_selection="round_robin",
            max_rounds=10,
            termination_keyword="APPROVED",
        )

        ctx = _make_ctx("Write a slogan")
        events = _collect_events(chat._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        # Should stop after round 2 (reviewer says APPROVED)
        assert len(messages) == 2

    def test_termination_condition(self) -> None:
        """Group chat terminates when condition returns True."""
        a = CountingAgent(name="a", response="Response A")
        b = CountingAgent(name="b", response="Response B")

        def stop_after_3(messages: list[dict]) -> bool:
            return len(messages) > 3  # user msg + 3 agent messages = 4

        chat = GroupChatAgent(
            name="conditional",
            sub_agents=[a, b],
            speaker_selection="round_robin",
            max_rounds=100,
            termination_condition=stop_after_3,
        )

        ctx = _make_ctx("Start")
        events = _collect_events(chat._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        # Should stop at 3 agent messages (user + 3 = 4 total)
        assert len(messages) == 3

    def test_custom_speaker_selection(self) -> None:
        """Custom callable controls speaker selection."""
        a = CountingAgent(name="agent_a", response="A response")
        b = CountingAgent(name="agent_b", response="B response")

        # Always select agent_b
        def always_b(messages, agents):
            return agents[1]

        chat = GroupChatAgent(
            name="custom_select",
            sub_agents=[a, b],
            speaker_selection=always_b,
            max_rounds=3,
        )

        ctx = _make_ctx("Start")
        events = _collect_events(chat._run_impl_traced(ctx))

        assert a.call_count == 0
        assert b.call_count == 3

    def test_result_key_stores_transcript(self) -> None:
        """result_key captures the full message transcript."""
        a = StubAgent(name="a", response="Hello from A")
        b = StubAgent(name="b", response="Hello from B")

        chat = GroupChatAgent(
            name="transcript",
            sub_agents=[a, b],
            speaker_selection="round_robin",
            max_rounds=2,
            result_key="chat_log",
        )

        ctx = _make_ctx("Greetings")
        _collect_events(chat._run_impl_traced(ctx))

        transcript = ctx.session.state.get("chat_log")
        assert transcript is not None
        assert len(transcript) == 3  # 1 user + 2 agent
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Greetings"

    def test_max_rounds_respected(self) -> None:
        """Chat stops at max_rounds even without termination."""
        a = CountingAgent(name="a", response="Going on...")
        b = CountingAgent(name="b", response="Still going...")

        chat = GroupChatAgent(
            name="limited",
            sub_agents=[a, b],
            max_rounds=5,
        )

        ctx = _make_ctx("Start")
        events = _collect_events(chat._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(messages) == 5

    def test_empty_sub_agents(self) -> None:
        """No sub-agents produces no events."""
        chat = GroupChatAgent(
            name="empty",
            sub_agents=[],
            max_rounds=5,
        )

        ctx = _make_ctx("Hello")
        events = _collect_events(chat._run_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        assert len(messages) == 0

    def test_state_update_events(self) -> None:
        """Each round emits a STATE_UPDATE event with speaker info."""
        a = StubAgent(name="a", response="Hi")
        b = StubAgent(name="b", response="Hey")

        chat = GroupChatAgent(
            name="events_test",
            sub_agents=[a, b],
            max_rounds=2,
        )

        ctx = _make_ctx("Start")
        events = _collect_events(chat._run_impl_traced(ctx))

        state_updates = [
            e for e in events if e.event_type == EventType.STATE_UPDATE
        ]
        assert len(state_updates) == 2
        assert state_updates[0].data["speaker"] == "a"
        assert state_updates[1].data["speaker"] == "b"

    def test_async_group_chat(self) -> None:
        """Async group chat works correctly."""
        a = CountingAgent(name="a", response="Async A")
        b = CountingAgent(name="b", response="DONE")

        chat = GroupChatAgent(
            name="async_chat",
            sub_agents=[a, b],
            max_rounds=4,
            termination_keyword="DONE",
        )

        ctx = _make_ctx("Async start")
        events = _collect_async_events(chat._run_async_impl_traced(ctx))

        messages = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        # a speaks round 1, b speaks round 2 with DONE → terminates
        assert len(messages) == 2
