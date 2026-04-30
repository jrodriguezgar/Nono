"""
Routine definition and result types.

Defines the core ``Routine`` dataclass — the unit of autonomous execution.
A routine bundles an executable (agent, workflow, task, or callable) with
metadata, configuration, and one or more triggers.

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union


# ── Status ────────────────────────────────────────────────────────────────────

class RoutineStatus(Enum):
    """Lifecycle state of a routine.

    States:
        IDLE: Registered but not actively scheduled.
        ACTIVE: Triggers are armed and the routine will fire when matched.
        RUNNING: Currently executing.
        PAUSED: Temporarily suspended — triggers ignored.
        ERROR: Last execution failed; awaiting manual intervention.
        DISABLED: Permanently deactivated.
    """

    IDLE = "idle"
    ACTIVE = "active"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    DISABLED = "disabled"


# ── Result ────────────────────────────────────────────────────────────────────

class ResultStatus(Enum):
    """Outcome of a single routine execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class RoutineResult:
    """Captures the outcome of a single routine execution.

    Args:
        routine_name: Name of the routine that was executed.
        status: Execution outcome.
        output: Text or structured output from the executable.
        data: Arbitrary structured data returned by the executable.
        error: Error message if the execution failed.
        started_at: UTC timestamp when execution began.
        finished_at: UTC timestamp when execution ended.
        duration_seconds: Wall-clock duration.
        trigger_type: Which trigger caused this execution.
        context: Input context passed to the execution.
        run_id: Unique identifier for this run.
    """

    routine_name: str
    status: ResultStatus = ResultStatus.SUCCESS
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    trigger_type: str = "manual"
    context: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        if self.started_at:
            d["started_at"] = self.started_at.isoformat()
        if self.finished_at:
            d["finished_at"] = self.finished_at.isoformat()
        return d


# ── Run Record ────────────────────────────────────────────────────────────────

@dataclass
class RoutineRunRecord:
    """Persistent record of a routine execution for history / audit.

    Args:
        run_id: Unique run identifier.
        routine_name: Name of the routine.
        trigger_type: Which trigger initiated the run.
        status: Outcome status.
        started_at: Start timestamp.
        finished_at: End timestamp.
        duration_seconds: Duration in seconds.
        output_preview: First N characters of the output.
        error: Error message if failed.
    """

    run_id: str
    routine_name: str
    trigger_type: str = "manual"
    status: str = "success"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output_preview: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        d = asdict(self)
        if self.started_at:
            d["started_at"] = self.started_at.isoformat()
        if self.finished_at:
            d["finished_at"] = self.finished_at.isoformat()
        return d


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RoutineConfig:
    """Execution configuration for a routine.

    Args:
        timeout_seconds: Maximum execution time (0 = no limit).
        max_retries: Number of retries on transient failure.
        retry_delay_seconds: Delay between retries.
        max_history: Maximum run records to keep per routine.
        environment: Key-value pairs injected as context.
        tags: Arbitrary labels for filtering / grouping.
    """

    timeout_seconds: int = 300
    max_retries: int = 0
    retry_delay_seconds: float = 5.0
    max_history: int = 100
    environment: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return asdict(self)


# ── Routine ───────────────────────────────────────────────────────────────────

# Forward reference — triggers are defined in triggers.py
RoutineTrigger = Any  # resolved at runtime via __init__.py imports

# Executable can be an Agent, Workflow, or any callable
Executable = Union[Any, Callable[..., Any]]


@dataclass
class Routine:
    """A saved, autonomous execution unit.

    Bundles an executable (agent, workflow, task, or callable) with one
    or more triggers and configuration.  The ``RoutineRunner`` uses this
    definition to arm triggers and execute the routine when they fire.

    Args:
        name: Unique identifier for this routine.
        description: Human-readable description of what the routine does.
        executable: The agent, workflow, task function, or callable to run.
        triggers: List of trigger definitions.
        config: Execution configuration.
        instruction: System prompt / instruction passed to the executable.
        input_template: Template for the user message (supports ``{variable}``).
        tools: List of tools to attach (for agent executables).
        status: Current lifecycle status.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        routine_id: Unique UUID.
        metadata: Arbitrary key-value metadata.
    """

    name: str
    description: str = ""
    executable: Optional[Executable] = None
    triggers: list[Any] = field(default_factory=list)
    config: RoutineConfig = field(default_factory=RoutineConfig)
    instruction: str = ""
    input_template: str = ""
    tools: list[Any] = field(default_factory=list)
    status: RoutineStatus = RoutineStatus.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    routine_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise routine definition to a JSON-safe dictionary.

        Excludes the ``executable`` and ``tools`` fields (not serialisable).
        """
        return {
            "name": self.name,
            "description": self.description,
            "routine_id": self.routine_id,
            "instruction": self.instruction,
            "input_template": self.input_template,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "triggers": [
                t.to_dict() if hasattr(t, "to_dict") else str(t)
                for t in self.triggers
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        triggers_str = ", ".join(
            t.__class__.__name__ for t in self.triggers
        )
        return (
            f"Routine(name={self.name!r}, status={self.status.value}, "
            f"triggers=[{triggers_str}])"
        )
