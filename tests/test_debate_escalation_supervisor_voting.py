"""Tests for DebateAgent, EscalationAgent, SupervisorAgent, and VotingAgent.

Verifies:
- DebateAgent runs debate rounds and judge phase.
- EscalationAgent tries agents in order, stops at first success.
- SupervisorAgent delegates via LLM decisions (mocked).
- VotingAgent picks the most frequent response.
- Both sync and async paths work correctly.

Run:
    python -m pytest tests/test_debate_escalation_supervisor_voting.py -v
"""

import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

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
    DebateAgent,
    EscalationAgent,
    SupervisorAgent,
    VotingAgent,
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


class EchoAgent(BaseAgent):
    """Echoes back the user message with a prefix."""

    def __init__(self, *, name: str, prefix: str = "") -> None:
        super().__init__(name=name, description=f"echo-{name}")
        self._prefix = prefix

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{self._prefix}{ctx.user_message}",
        )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        yield Event(
            EventType.AGENT_MESSAGE, self.name,
            f"{self._prefix}{ctx.user_message}",
        )


class FailThenSucceedAgent(BaseAgent):
    """Returns failure_keyword until call N, then succeeds."""

    def __init__(
        self, *, name: str, succeed_on: int = 2,
        failure_keyword: str = "I don't know",
    ) -> None:
        super().__init__(name=name, description=f"fail-then-{name}")
        self._succeed_on = succeed_on
        self._failure_keyword = failure_keyword
        self._call_count = 0

    def _run_impl(self, ctx: InvocationContext) -> Iterator[Event]:
        self._call_count += 1
        if self._call_count >= self._succeed_on:
            yield Event(EventType.AGENT_MESSAGE, self.name, "Got the answer!")
        else:
            yield Event(
                EventType.AGENT_MESSAGE, self.name,
                f"{self._failure_keyword}: too hard",
            )

    async def _run_async_impl(
        self, ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        self._call_count += 1
        if self._call_count >= self._succeed_on:
            yield Event(EventType.AGENT_MESSAGE, self.name, "Got the answer!")
        else:
            yield Event(
                EventType.AGENT_MESSAGE, self.name,
                f"{self._failure_keyword}: too hard",
            )


def _ctx(msg: str = "hello") -> InvocationContext:
    return InvocationContext(session=Session(), user_message=msg)


# ── DebateAgent tests ─────────────────────────────────────────────────────────


def test_debate_sync_runs_rounds_and_judge():
    """DebateAgent runs debate rounds then a judge phase."""
    agent_a = StubAgent(name="optimist", response="AI is great")
    agent_b = StubAgent(name="pessimist", response="AI is risky")
    judge = StubAgent(name="judge", response="RESOLVED: balanced view")

    debate = DebateAgent(
        name="debate",
        agent_a=agent_a,
        agent_b=agent_b,
        judge=judge,
        max_rounds=2,
    )

    events = list(debate._run_impl(_ctx("AI ethics")))

    # Expect: 2 rounds × (STATE_UPDATE + agent_a msg + agent_b msg) + judge msg
    state_events = [e for e in events if e.event_type == EventType.STATE_UPDATE]
    assert len(state_events) == 2

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # 2 rounds × 2 debaters + 1 judge = 5 agent messages
    assert len(agent_msgs) == 5

    # Last message is from judge
    assert agent_msgs[-1].author == "judge"
    assert "RESOLVED" in agent_msgs[-1].content


def test_debate_async_runs_rounds_and_judge():
    """DebateAgent async path runs debate rounds then a judge phase."""
    agent_a = StubAgent(name="pro", response="yes")
    agent_b = StubAgent(name="con", response="no")
    judge = StubAgent(name="moderator", response="verdict")

    debate = DebateAgent(
        name="async_debate",
        agent_a=agent_a,
        agent_b=agent_b,
        judge=judge,
        max_rounds=1,
    )

    async def _run():
        return [event async for event in debate._run_async_impl(_ctx("topic"))]

    events = asyncio.run(_run())

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # 1 round × 2 debaters + 1 judge = 3
    assert len(agent_msgs) == 3
    assert agent_msgs[-1].author == "moderator"


def test_debate_sub_agents_are_registered():
    """DebateAgent registers agent_a, agent_b, and judge as sub_agents."""
    a = StubAgent(name="a")
    b = StubAgent(name="b")
    j = StubAgent(name="j")

    debate = DebateAgent(name="d", agent_a=a, agent_b=b, judge=j)
    assert len(debate.sub_agents) == 3
    assert debate.find_sub_agent("a") is a
    assert debate.find_sub_agent("j") is j


# ── EscalationAgent tests ────────────────────────────────────────────────────


def test_escalation_sync_stops_at_first_success():
    """EscalationAgent stops at the first agent that succeeds."""
    fail1 = StubAgent(name="cheap", response="I don't know anything")
    success = StubAgent(name="medium", response="Here's the answer")
    expensive = StubAgent(name="expensive", response="Overkill answer")

    escalation = EscalationAgent(
        name="fallback",
        sub_agents=[fail1, success, expensive],
        failure_keyword="I don't know",
    )

    events = list(escalation._run_impl(_ctx("question")))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    # Only the successful agent's message should be yielded (after failures)
    assert any("Here's the answer" in e.content for e in agent_msgs)
    # expensive should never be called
    assert not any(e.author == "expensive" for e in agent_msgs)


def test_escalation_sync_yields_last_on_all_fail():
    """If all agents fail, EscalationAgent yields the last agent's events."""
    fail1 = StubAgent(name="a", response="I don't know #1")
    fail2 = StubAgent(name="b", response="I don't know #2")

    escalation = EscalationAgent(
        name="all_fail",
        sub_agents=[fail1, fail2],
        failure_keyword="I don't know",
    )

    events = list(escalation._run_impl(_ctx("hard question")))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert len(agent_msgs) == 1
    assert "I don't know #2" in agent_msgs[0].content


def test_escalation_async_stops_at_first_success():
    """EscalationAgent async path stops at first success."""
    fail = StubAgent(name="weak", response="I don't know")
    ok = StubAgent(name="strong", response="Solved it")

    escalation = EscalationAgent(
        name="async_fallback",
        sub_agents=[fail, ok],
        failure_keyword="I don't know",
    )

    async def _run():
        return [event async for event in escalation._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("Solved it" in e.content for e in agent_msgs)


def test_escalation_on_escalation_callback():
    """EscalationAgent invokes on_escalation callback on failure."""
    fail = StubAgent(name="tier1", response="I don't know")
    ok = StubAgent(name="tier2", response="answer")

    callback_log: list[tuple[str, str]] = []

    escalation = EscalationAgent(
        name="cb_test",
        sub_agents=[fail, ok],
        failure_keyword="I don't know",
        on_escalation=lambda name, resp: callback_log.append((name, resp)),
    )

    list(escalation._run_impl(_ctx()))

    assert len(callback_log) == 1
    assert callback_log[0][0] == "tier1"


# ── SupervisorAgent tests ────────────────────────────────────────────────────


def test_supervisor_sync_delegates_and_accepts():
    """SupervisorAgent delegates to a worker then accepts."""
    worker = StubAgent(name="coder", response="def hello(): pass")

    supervisor = SupervisorAgent(
        name="boss",
        sub_agents=[worker],
        max_iterations=2,
    )

    # Mock the service to return delegate then accept
    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "coder", "message": "write code", "reason": "best fit"}',
        '{"action": "accept", "agent": "", "message": "", "reason": "good work"}',
    ]
    supervisor._service = mock_service

    events = list(supervisor._run_impl(_ctx("write hello")))

    transfer_events = [
        e for e in events if e.event_type == EventType.AGENT_TRANSFER
    ]
    assert len(transfer_events) == 1
    assert transfer_events[0].data["worker"] == "coder"

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("def hello" in e.content for e in agent_msgs)


def test_supervisor_async_delegates_and_accepts():
    """SupervisorAgent async path delegates then accepts."""
    worker = StubAgent(name="writer", response="A great story")

    supervisor = SupervisorAgent(
        name="async_boss",
        sub_agents=[worker],
        max_iterations=2,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.side_effect = [
        '{"action": "delegate", "agent": "writer", "message": "write story", "reason": "creative"}',
        '{"action": "accept", "agent": "", "message": "", "reason": "excellent"}',
    ]
    supervisor._service = mock_service

    async def _run():
        return [
            event async for event in supervisor._run_async_impl(_ctx("story"))
        ]

    events = asyncio.run(_run())

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("great story" in e.content for e in agent_msgs)


def test_supervisor_no_workers_yields_error():
    """SupervisorAgent with no workers yields an error event."""
    supervisor = SupervisorAgent(name="empty_boss", sub_agents=[])

    events = list(supervisor._run_impl(_ctx()))

    error_events = [e for e in events if e.event_type == EventType.ERROR]
    assert len(error_events) == 1
    assert "No workers" in error_events[0].content


def test_supervisor_unknown_worker_falls_back_to_first():
    """SupervisorAgent falls back to first worker on unknown agent name."""
    w1 = StubAgent(name="alpha", response="alpha-result")
    w2 = StubAgent(name="beta", response="beta-result")

    supervisor = SupervisorAgent(
        name="fallback_boss",
        sub_agents=[w1, w2],
        max_iterations=1,
    )

    mock_service = MagicMock()
    mock_service.generate_completion.return_value = (
        '{"action": "delegate", "agent": "nonexistent", "reason": "oops"}'
    )
    supervisor._service = mock_service

    events = list(supervisor._run_impl(_ctx()))

    agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
    assert any("alpha-result" in e.content for e in agent_msgs)


# ── VotingAgent tests ─────────────────────────────────────────────────────────


def test_voting_sync_picks_majority():
    """VotingAgent picks the most frequent response."""
    a = StubAgent(name="voter_a", response="Paris")
    b = StubAgent(name="voter_b", response="Paris")
    c = StubAgent(name="voter_c", response="London")

    voting = VotingAgent(
        name="vote",
        sub_agents=[a, b, c],
    )

    events = list(voting._run_impl(_ctx("capital of France?")))

    # The final AGENT_MESSAGE from VotingAgent itself should be "Paris"
    vote_msgs = [
        e for e in events
        if e.event_type == EventType.AGENT_MESSAGE and e.author == "vote"
    ]
    assert len(vote_msgs) == 1
    assert vote_msgs[0].content.strip().lower() == "paris"


def test_voting_async_picks_majority():
    """VotingAgent async path picks majority."""
    a = StubAgent(name="v1", response="42")
    b = StubAgent(name="v2", response="42")
    c = StubAgent(name="v3", response="99")

    voting = VotingAgent(name="async_vote", sub_agents=[a, b, c])

    async def _run():
        return [event async for event in voting._run_async_impl(_ctx())]

    events = asyncio.run(_run())

    vote_msgs = [
        e for e in events
        if e.event_type == EventType.AGENT_MESSAGE and e.author == "async_vote"
    ]
    assert len(vote_msgs) == 1
    assert "42" in vote_msgs[0].content


def test_voting_stores_result_in_session():
    """VotingAgent stores vote details in session.state when result_key set."""
    a = StubAgent(name="v1", response="Yes")
    b = StubAgent(name="v2", response="Yes")
    c = StubAgent(name="v3", response="No")

    session = Session()
    voting = VotingAgent(
        name="stored_vote",
        sub_agents=[a, b, c],
        result_key="vote_result",
    )

    ctx = InvocationContext(session=session, user_message="agree?")
    list(voting._run_impl(ctx))

    assert "vote_result" in session.state
    result = session.state["vote_result"]
    assert result["winner"] in ("v1", "v2")
    assert result["response"].strip().lower() == "yes"


def test_voting_custom_normalize():
    """VotingAgent uses custom normalize function."""
    a = StubAgent(name="v1", response="  YES  ")
    b = StubAgent(name="v2", response="yes!")
    c = StubAgent(name="v3", response="NO")

    # Custom normalizer that strips everything except alpha chars
    voting = VotingAgent(
        name="custom_norm",
        sub_agents=[a, b, c],
        normalize=lambda s: "".join(c for c in s if c.isalpha()).lower(),
    )

    events = list(voting._run_impl(_ctx()))

    vote_msgs = [
        e for e in events
        if e.event_type == EventType.AGENT_MESSAGE and e.author == "custom_norm"
    ]
    assert len(vote_msgs) == 1
    # "YES" and "yes!" both normalize to "yes" → majority
    assert "yes" in vote_msgs[0].content.strip().lower()


def test_voting_empty_sub_agents():
    """VotingAgent with no sub_agents produces no events."""
    voting = VotingAgent(name="empty_vote", sub_agents=[])

    events = list(voting._run_impl(_ctx()))
    assert len(events) == 0
