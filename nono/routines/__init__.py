"""
Routines — Autonomous execution infrastructure for Nono.

A routine is a saved configuration: a prompt, an executable (agent, workflow,
or task), a set of tools/connectors, and one or more triggers — packaged once
and executed automatically.

Inspired by Claude Code Routines, adapted for Nono's local/programmatic
paradigm.  Routines run in-process (threads) or can be exposed via API.

Core components:
    - **Routine**: Immutable definition (what to run + how to trigger).
    - **RoutineResult**: Outcome of a single execution.
    - **RoutineStatus**: Lifecycle state (idle, running, paused, error).
    - **RoutineTrigger**: Base class for trigger types.
    - **ScheduleTrigger**: Cron / interval-based recurring execution.
    - **EventTrigger**: Fires on custom application events.
    - **WebhookTrigger**: HTTP POST endpoint trigger.
    - **ManualTrigger**: Explicit ``fire()`` call.
    - **RoutineRunner**: Manages registration, lifecycle, and execution.
    - **RoutineStore**: JSON-based persistence for routine definitions.

Usage:
    from nono.routines import (
        Routine, RoutineRunner, ScheduleTrigger, EventTrigger,
    )
    from nono.agent import Agent

    agent = Agent(name="reviewer", provider="google", instruction="Review code.")

    routine = Routine(
        name="nightly_review",
        description="Run code review every night at 2 AM",
        executable=agent,
        triggers=[ScheduleTrigger(cron="0 2 * * *")],
    )

    runner = RoutineRunner()
    runner.register(routine)
    runner.start()  # begins scheduler loop

    # Manual fire
    result = runner.fire("nightly_review", context={"repo": "myproject"})

    # Stop all routines
    runner.stop()
"""

from .routine import (
    Routine,
    RoutineConfig,
    RoutineResult,
    RoutineStatus,
    RoutineRunRecord,
)
from .triggers import (
    RoutineTrigger,
    ScheduleTrigger,
    EventTrigger,
    WebhookTrigger,
    ManualTrigger,
    TriggerType,
)
from .runner import RoutineRunner
from .store import RoutineStore

__all__ = [
    # Core
    "Routine",
    "RoutineConfig",
    "RoutineResult",
    "RoutineStatus",
    "RoutineRunRecord",
    # Triggers
    "RoutineTrigger",
    "ScheduleTrigger",
    "EventTrigger",
    "WebhookTrigger",
    "ManualTrigger",
    "TriggerType",
    # Runner
    "RoutineRunner",
    # Persistence
    "RoutineStore",
]
