"""Agent Execution Model — Claude Code-inspired enhancements.

Implements typed task packets, worker state machine, failure taxonomy,
executable policies, green-ness verification contract, worktree isolation,
stale-branch detection, conversation-level checkpoints, and plan mode.

Based on the design in ``ToDo/agent_execution_schema.md``.

Modules:
    - **TaskPacket**: Typed alternative to natural-language prompts.
    - **WorkerState / WorkerStateMachine**: 8-state lifecycle for workers.
    - **FailureCategory / FailureClassifier**: 10-class failure taxonomy.
    - **PolicyRule / PolicyEngine**: Machine-enforced executable policies.
    - **VerificationLevel / VerificationContract**: 4-level green-ness.
    - **WorktreeManager**: Git worktree isolation per session.
    - **StaleBranchDetector**: Detect and auto-rebase stale branches.
    - **ConversationCheckpoint / ConversationCheckpointManager**: Snapshots.
    - **PlanModeAgent**: Read-only exploration agent.
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Optional

logger = logging.getLogger("Nono.Agent.Execution")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TYPED TASK PACKETS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EscalationPolicy:
    """Escalation rules when auto-recovery fails.

    Args:
        retry: Number of automatic retry attempts before escalating.
        then: Action after retries exhausted (``"notify_human"``, ``"abort"``).
    """

    retry: int = 1
    then: str = "notify_human"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EscalationPolicy:
        """Deserialise from dictionary."""
        return cls(retry=data.get("retry", 1), then=data.get("then", "notify_human"))


@dataclass
class ReportingContract:
    """Contract for what the agent must report on completion.

    Args:
        format: Output format (``"json"``, ``"text"``, ``"markdown"``).
        fields: List of required report fields.
    """

    format: str = "json"
    fields: list[str] = field(default_factory=lambda: [
        "files_changed", "tests_run", "diff_summary",
    ])

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportingContract:
        """Deserialise from dictionary."""
        return cls(
            format=data.get("format", "json"),
            fields=data.get("fields", ["files_changed", "tests_run", "diff_summary"]),
        )


@dataclass
class TaskPacket:
    """Typed alternative to natural-language prompts for autonomous execution.

    Eliminates ambiguity — can be logged, retried, transformed, and
    audited programmatically.

    Args:
        objective: What needs to be accomplished.
        scope: Scope constraint (e.g. ``"module:src/auth/"``).
        worktree: Path to the git worktree for isolation.
        branch_policy: Branch creation/management policy string.
        acceptance_tests: Commands that must pass for success.
        commit_policy: How to commit (``"squash_on_green"``, ``"atomic"``, etc.).
        reporting_contract: What to report on completion.
        escalation_policy: When and how to escalate to a human.
        metadata: Arbitrary extra data.
        packet_id: Unique identifier (auto-generated).
        created_at: UTC timestamp.
    """

    objective: str
    scope: str = ""
    worktree: str = ""
    branch_policy: str = ""
    acceptance_tests: list[str] = field(default_factory=list)
    commit_policy: str = "squash_on_green"
    reporting_contract: ReportingContract = field(default_factory=ReportingContract)
    escalation_policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    packet_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the task packet.
        """
        return {
            "objective": self.objective,
            "scope": self.scope,
            "worktree": self.worktree,
            "branch_policy": self.branch_policy,
            "acceptance_tests": list(self.acceptance_tests),
            "commit_policy": self.commit_policy,
            "reporting_contract": self.reporting_contract.to_dict(),
            "escalation_policy": self.escalation_policy.to_dict(),
            "metadata": dict(self.metadata),
            "packet_id": self.packet_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPacket:
        """Deserialise from a dictionary.

        Args:
            data: Dictionary with task packet fields.

        Returns:
            New ``TaskPacket`` instance.
        """
        rc = data.get("reporting_contract", {})
        ep = data.get("escalation_policy", {})
        return cls(
            objective=data["objective"],
            scope=data.get("scope", ""),
            worktree=data.get("worktree", ""),
            branch_policy=data.get("branch_policy", ""),
            acceptance_tests=data.get("acceptance_tests", []),
            commit_policy=data.get("commit_policy", "squash_on_green"),
            reporting_contract=ReportingContract.from_dict(rc) if rc else ReportingContract(),
            escalation_policy=EscalationPolicy.from_dict(ep) if ep else EscalationPolicy(),
            metadata=data.get("metadata", {}),
            packet_id=data.get("packet_id", uuid.uuid4().hex[:12]),
        )

    def to_json(self) -> str:
        """Serialise to JSON string.

        Returns:
            JSON representation.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> TaskPacket:
        """Deserialise from JSON string.

        Args:
            text: JSON string.

        Returns:
            New ``TaskPacket`` instance.
        """
        return cls.from_dict(json.loads(text))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WORKER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════


class WorkerState(Enum):
    """Lifecycle states for an agent worker session.

    Each transition emits a typed event — no log scraping.
    """

    SPAWNING = "spawning"
    TRUST_REQUIRED = "trust_required"
    READY_FOR_PROMPT = "ready_for_prompt"
    PROMPT_ACCEPTED = "prompt_accepted"
    RUNNING = "running"
    BLOCKED = "blocked"
    FINISHED = "finished"
    FAILED = "failed"


# Valid transitions: current_state → set of allowed next states
_TRANSITIONS: dict[WorkerState, set[WorkerState]] = {
    WorkerState.SPAWNING: {WorkerState.TRUST_REQUIRED, WorkerState.READY_FOR_PROMPT},
    WorkerState.TRUST_REQUIRED: {WorkerState.READY_FOR_PROMPT, WorkerState.FAILED},
    WorkerState.READY_FOR_PROMPT: {WorkerState.PROMPT_ACCEPTED, WorkerState.FAILED},
    WorkerState.PROMPT_ACCEPTED: {WorkerState.RUNNING, WorkerState.FAILED},
    WorkerState.RUNNING: {WorkerState.BLOCKED, WorkerState.FINISHED, WorkerState.FAILED},
    WorkerState.BLOCKED: {WorkerState.RUNNING, WorkerState.FAILED},
    WorkerState.FINISHED: set(),
    WorkerState.FAILED: set(),
}


@dataclass(frozen=True)
class WorkerTransition:
    """Record of a state transition.

    Args:
        from_state: Previous state.
        to_state: New state.
        reason: Human-readable reason for the transition.
        timestamp: UTC timestamp.
        transition_id: Unique identifier.
    """

    from_state: WorkerState
    to_state: WorkerState
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""


class WorkerStateMachine:
    """State machine managing the lifecycle of a worker session.

    Thread-safe — all state mutations are guarded by a lock.

    Args:
        worker_id: Unique identifier for this worker.
        initial_state: Starting state (default ``SPAWNING``).

    Example:
        >>> sm = WorkerStateMachine("worker-1")
        >>> sm.transition(WorkerState.READY_FOR_PROMPT, "Trust auto-resolved")
        >>> sm.state
        <WorkerState.READY_FOR_PROMPT: 'ready_for_prompt'>
    """

    def __init__(
        self,
        worker_id: str = "",
        initial_state: WorkerState = WorkerState.SPAWNING,
    ) -> None:
        self._worker_id = worker_id or uuid.uuid4().hex[:8]
        self._state = initial_state
        self._lock = threading.Lock()
        self._history: list[WorkerTransition] = []
        self._listeners: list[Callable[[WorkerTransition], None]] = []

    @property
    def worker_id(self) -> str:
        """Unique worker identifier."""
        return self._worker_id

    @property
    def state(self) -> WorkerState:
        """Current state."""
        with self._lock:
            return self._state

    @property
    def history(self) -> list[WorkerTransition]:
        """Immutable copy of transition history."""
        with self._lock:
            return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """Whether the worker is in a terminal state."""
        return self.state in {WorkerState.FINISHED, WorkerState.FAILED}

    def on_transition(self, listener: Callable[[WorkerTransition], None]) -> None:
        """Register a callback for state transitions.

        Args:
            listener: Called with the ``WorkerTransition`` after each change.
        """
        self._listeners.append(listener)

    def transition(self, to_state: WorkerState, reason: str = "") -> WorkerTransition:
        """Transition to a new state.

        Args:
            to_state: Target state.
            reason: Explanation for the transition.

        Returns:
            The recorded transition.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        with self._lock:
            allowed = _TRANSITIONS.get(self._state, set())

            if to_state not in allowed:
                raise InvalidTransitionError(
                    f"Cannot transition from {self._state.value} to "
                    f"{to_state.value}. Allowed: "
                    f"{[s.value for s in allowed]}"
                )

            transition = WorkerTransition(
                from_state=self._state,
                to_state=to_state,
                reason=reason,
            )
            self._state = to_state
            self._history.append(transition)

        logger.info(
            "Worker %s: %s → %s (%s)",
            self._worker_id, transition.from_state.value,
            transition.to_state.value, reason or "no reason",
        )

        for listener in self._listeners:
            try:
                listener(transition)
            except Exception:
                logger.exception("Transition listener error")

        return transition

    def can_transition(self, to_state: WorkerState) -> bool:
        """Check if a transition is valid without performing it.

        Args:
            to_state: Target state to check.

        Returns:
            ``True`` if the transition is allowed.
        """
        with self._lock:
            return to_state in _TRANSITIONS.get(self._state, set())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAILURE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


class FailureCategory(Enum):
    """10-class failure taxonomy.

    Classify failures **before** acting on them. Each category has a
    default auto-recovery recipe.
    """

    PROMPT_DELIVERY = "prompt_delivery"
    TRUST_GATE = "trust_gate"
    BRANCH_DIVERGENCE = "branch_divergence"
    COMPILE = "compile"
    TEST = "test"
    PLUGIN_STARTUP = "plugin_startup"
    MCP_STARTUP = "mcp_startup"
    MCP_HANDSHAKE = "mcp_handshake"
    TOOL_RUNTIME = "tool_runtime"
    INFRA = "infra"


@dataclass(frozen=True)
class RecoveryRecipe:
    """Auto-recovery instructions for a failure category.

    Args:
        description: Human-readable recovery strategy.
        max_attempts: Maximum automatic recovery attempts.
        action: Machine-readable action (``"retry"``, ``"rebuild"``, ``"degrade"``, etc.).
        backoff_seconds: Seconds to wait between retries.
    """

    description: str
    max_attempts: int = 1
    action: str = "retry"
    backoff_seconds: float = 0.0


# Default recovery recipes per failure category
_DEFAULT_RECIPES: dict[FailureCategory, RecoveryRecipe] = {
    FailureCategory.PROMPT_DELIVERY: RecoveryRecipe(
        description="Retry prompt, detect misdelivery",
        action="retry_prompt",
    ),
    FailureCategory.TRUST_GATE: RecoveryRecipe(
        description="Auto-resolve if repo is allowlisted",
        action="auto_trust",
    ),
    FailureCategory.BRANCH_DIVERGENCE: RecoveryRecipe(
        description="Auto rebase / merge-forward",
        action="rebase",
    ),
    FailureCategory.COMPILE: RecoveryRecipe(
        description="Incremental rebuild",
        action="rebuild",
    ),
    FailureCategory.TEST: RecoveryRecipe(
        description="Diagnose, fix, re-run",
        action="diagnose_fix",
        max_attempts=2,
    ),
    FailureCategory.PLUGIN_STARTUP: RecoveryRecipe(
        description="Retry init, continue degraded",
        action="retry_or_degrade",
    ),
    FailureCategory.MCP_STARTUP: RecoveryRecipe(
        description="Retry with backoff",
        action="retry",
        backoff_seconds=2.0,
    ),
    FailureCategory.MCP_HANDSHAKE: RecoveryRecipe(
        description="Retry, degraded mode",
        action="retry_or_degrade",
        backoff_seconds=1.0,
    ),
    FailureCategory.TOOL_RUNTIME: RecoveryRecipe(
        description="Retry or alternative tool",
        action="retry_or_alternative",
    ),
    FailureCategory.INFRA: RecoveryRecipe(
        description="Wait + retry",
        action="wait_retry",
        backoff_seconds=5.0,
        max_attempts=3,
    ),
}


@dataclass(frozen=True)
class ClassifiedFailure:
    """A failure classified with category and recovery plan.

    Args:
        category: The failure category.
        message: Error message or description.
        recipe: Applicable recovery recipe.
        context: Additional context data.
        failure_id: Unique identifier.
        timestamp: UTC timestamp.
    """

    category: FailureCategory
    message: str
    recipe: RecoveryRecipe
    context: dict[str, Any] = field(default_factory=dict)
    failure_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FailureClassifier:
    """Classify errors into the 10-class failure taxonomy.

    Uses keyword-based heuristics.  Can be extended with custom classifiers
    via :meth:`register_classifier`.

    Example:
        >>> fc = FailureClassifier()
        >>> result = fc.classify(RuntimeError("compilation failed: syntax error"))
        >>> result.category
        <FailureCategory.COMPILE: 'compile'>
    """

    def __init__(self) -> None:
        self._custom_classifiers: list[
            Callable[[Exception, str], FailureCategory | None]
        ] = []

    def register_classifier(
        self,
        fn: Callable[[Exception, str], FailureCategory | None],
    ) -> None:
        """Register a custom classifier function.

        Args:
            fn: Callable that receives ``(exception, message)`` and returns
                a ``FailureCategory`` or ``None`` to defer.
        """
        self._custom_classifiers.append(fn)

    def classify(
        self,
        error: Exception | None = None,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> ClassifiedFailure:
        """Classify a failure into a category with a recovery recipe.

        Args:
            error: The exception, if any.
            message: Error message string (used if no exception).
            context: Additional context for the failure.

        Returns:
            A ``ClassifiedFailure`` with category and recovery recipe.
        """
        msg = message or (str(error) if error else "Unknown error")

        # Try custom classifiers first
        for fn in self._custom_classifiers:
            try:
                cat = fn(error, msg) if error else fn(None, msg)

                if cat is not None:
                    return ClassifiedFailure(
                        category=cat,
                        message=msg,
                        recipe=_DEFAULT_RECIPES.get(cat, RecoveryRecipe(description="Unknown")),
                        context=context or {},
                    )
            except Exception:
                logger.exception("Custom classifier error")

        # Keyword-based heuristics
        category = self._heuristic_classify(msg)
        return ClassifiedFailure(
            category=category,
            message=msg,
            recipe=_DEFAULT_RECIPES[category],
            context=context or {},
        )

    @staticmethod
    def _heuristic_classify(msg: str) -> FailureCategory:
        """Classify by keyword matching.

        Args:
            msg: Error message to classify.

        Returns:
            Best-matching ``FailureCategory``.
        """
        lower = msg.lower()

        patterns: list[tuple[list[str], FailureCategory]] = [
            (["prompt", "delivery", "misdeliver"], FailureCategory.PROMPT_DELIVERY),
            (["trust", "permission", "not authorized", "allowlist"], FailureCategory.TRUST_GATE),
            (["branch", "diverge", "stale", "behind main", "merge conflict"], FailureCategory.BRANCH_DIVERGENCE),
            (["compil", "syntax error", "build fail", "cannot resolve"], FailureCategory.COMPILE),
            (["test fail", "assertion", "assert ", "pytest", "unittest"], FailureCategory.TEST),
            (["plugin", "extension", "startup fail"], FailureCategory.PLUGIN_STARTUP),
            (["mcp", "server start", "mcp_startup"], FailureCategory.MCP_STARTUP),
            (["handshake", "mcp_handshake", "negotiation"], FailureCategory.MCP_HANDSHAKE),
            (["tool", "function call", "tool_runtime"], FailureCategory.TOOL_RUNTIME),
            (["network", "timeout", "connection", "dns", "infra", "503", "502"], FailureCategory.INFRA),
        ]

        for keywords, category in patterns:
            if any(kw in lower for kw in keywords):
                return category

        return FailureCategory.INFRA  # default fallback

    @staticmethod
    def get_recipe(category: FailureCategory) -> RecoveryRecipe:
        """Get the default recovery recipe for a failure category.

        Args:
            category: The failure category.

        Returns:
            The associated recovery recipe.
        """
        return _DEFAULT_RECIPES.get(category, RecoveryRecipe(description="Unknown"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXECUTABLE POLICIES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PolicyResult:
    """Result of a policy evaluation.

    Args:
        triggered: Whether the policy condition was met.
        action: The action to take.
        reason: Human-readable explanation.
        policy_name: Name of the policy that produced this result.
    """

    triggered: bool
    action: str
    reason: str = ""
    policy_name: str = ""


class PolicyRule(ABC):
    """Abstract base for machine-enforced policy rules.

    Subclass and implement :meth:`evaluate` to create custom policies.
    Follows the same extensibility pattern as ``CompactionStrategy``:
    subclass for full control, or use ``CallablePolicy`` for quick
    inline definitions.

    Attributes:
        enabled: Whether this policy is active.  Disabled policies are
            skipped during evaluation without removing them.
        priority: Evaluation order (lower = earlier).  Policies with equal
            priority are evaluated in registration order.
    """

    enabled: bool = True
    priority: int = 100

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this policy rule."""

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate the policy against the given context.

        Args:
            context: Execution context data (workspace state, test results, etc.).

        Returns:
            A ``PolicyResult`` indicating whether the policy was triggered.
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialise policy metadata to a dictionary.

        Override in subclasses to persist custom parameters.

        Returns:
            Dictionary with ``name``, ``enabled``, ``priority``, and
            ``type`` (the class name).
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority,
        }


class CallablePolicy(PolicyRule):
    """Adapter that wraps a plain function as a policy rule.

    Mirrors ``CallableStrategy`` from the compaction module — lets users
    define policies without subclassing::

        def my_policy(ctx: dict) -> PolicyResult:
            if ctx.get("disk_full"):
                return PolicyResult(True, "pause_writes", "Disk full")
            return PolicyResult(False, "proceed")

        engine.register(CallablePolicy("disk_check", my_policy, priority=10))

    The callable signature must be::

        (context: dict[str, Any]) -> PolicyResult

    Args:
        policy_name: Unique name for this policy.
        fn: Callable that receives a context dict and returns a
            ``PolicyResult``.
        enabled: Whether the policy starts enabled.
        priority: Evaluation priority (lower = earlier).
    """

    def __init__(
        self,
        policy_name: str,
        fn: Callable[[dict[str, Any]], PolicyResult],
        *,
        enabled: bool = True,
        priority: int = 100,
    ) -> None:
        self._name = policy_name
        self._fn = fn
        self.enabled = enabled
        self.priority = priority

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Delegate to the wrapped callable.

        Args:
            context: Execution context data.

        Returns:
            The ``PolicyResult`` returned by the callable.
        """
        return self._fn(context)


class AutoMergePolicy(PolicyRule):
    """POLICY 1: Auto-merge when green + scoped + reviewed.

    Context keys:
        - ``workspace_green``: bool
        - ``diff_scoped``: bool
        - ``review_passed``: bool
        - ``target_branch``: str (default ``"dev"``)
    """

    @property
    def name(self) -> str:
        return "auto_merge"

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate auto-merge conditions.

        Args:
            context: Must contain ``workspace_green``, ``diff_scoped``, ``review_passed``.

        Returns:
            Policy result with ``auto_merge`` action if all conditions met.
        """
        green = context.get("workspace_green", False)
        scoped = context.get("diff_scoped", False)
        reviewed = context.get("review_passed", False)
        target = context.get("target_branch", "dev")

        if green and scoped and reviewed:
            return PolicyResult(
                triggered=True,
                action=f"auto_merge({target})",
                reason="Workspace green, diff scoped, review passed",
            )
        missing = [
            k for k, v in [("green", green), ("scoped", scoped), ("reviewed", reviewed)]
            if not v
        ]
        return PolicyResult(
            triggered=False,
            action="wait",
            reason=f"Missing conditions: {', '.join(missing)}",
        )


class StaleBranchPolicy(PolicyRule):
    """POLICY 2: Merge-forward before broad tests if branch is stale."""

    @property
    def name(self) -> str:
        return "stale_branch_merge_forward"

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether the branch is stale against main.

        Args:
            context: Must contain ``branch_stale_against_main``.

        Returns:
            Policy result with ``merge_forward`` action if stale.
        """
        if context.get("branch_stale_against_main", False):
            return PolicyResult(
                triggered=True,
                action="merge_forward",
                reason="Branch is stale against main — merge-forward before tests",
            )
        return PolicyResult(triggered=False, action="proceed", reason="Branch is up-to-date")


class StartupRecoveryPolicy(PolicyRule):
    """POLICY 3: Recover once, then escalate on startup failures."""

    @property
    def name(self) -> str:
        return "startup_recovery"

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate startup recovery.

        Args:
            context: Must contain ``startup_blocked``, optionally ``recovery_attempts``.

        Returns:
            Policy result with ``recover`` or ``escalate`` action.
        """
        if not context.get("startup_blocked", False):
            return PolicyResult(triggered=False, action="proceed", reason="Startup OK")

        attempts = context.get("recovery_attempts", 0)

        if attempts < 1:
            return PolicyResult(
                triggered=True,
                action="recover",
                reason=f"Startup blocked — attempt recovery ({attempts + 1}/1)",
            )

        return PolicyResult(
            triggered=True,
            action="escalate_human",
            reason=f"Startup blocked after {attempts} recovery attempt(s)",
        )


class LaneCompletionPolicy(PolicyRule):
    """POLICY 4: Emit closeout + cleanup when a lane completes."""

    @property
    def name(self) -> str:
        return "lane_completion"

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate lane completion.

        Args:
            context: Must contain ``lane_completed``.

        Returns:
            Policy result with ``emit_closeout`` action if completed.
        """
        if context.get("lane_completed", False):
            return PolicyResult(
                triggered=True,
                action="emit_closeout_and_cleanup",
                reason="Lane completed — emit closeout and cleanup session",
            )
        return PolicyResult(triggered=False, action="continue", reason="Lane still active")


class DegradedModePolicy(PolicyRule):
    """POLICY 5: Continue in degraded mode if one MCP fails but others are healthy."""

    @property
    def name(self) -> str:
        return "degraded_mode"

    def evaluate(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate degraded mode.

        Args:
            context: Must contain ``mcp_server_failed`` and ``others_healthy``.

        Returns:
            Policy result with ``degraded_mode`` action if applicable.
        """
        failed = context.get("mcp_server_failed", False)
        others_ok = context.get("others_healthy", True)

        if failed and others_ok:
            return PolicyResult(
                triggered=True,
                action="degraded_mode_continue",
                reason="MCP server failed but others healthy — continuing degraded",
            )

        if failed and not others_ok:
            return PolicyResult(
                triggered=True,
                action="escalate_human",
                reason="Multiple MCP servers failed",
            )

        return PolicyResult(triggered=False, action="proceed", reason="All MCP servers healthy")


# ── Register built-in policies in the type registry ───────────────────────

_BUILTIN_POLICIES: list[type[PolicyRule]] = [
    AutoMergePolicy,
    StaleBranchPolicy,
    StartupRecoveryPolicy,
    LaneCompletionPolicy,
    DegradedModePolicy,
]


class PolicyEngine:
    """Machine-enforced policy evaluation engine.

    Evaluates registered policies against a context and returns results.
    Policies are evaluated in ``priority`` order (lower = first); ties
    preserve registration order.

    Follows the same extensibility pattern as the compaction module:

    * **Subclass** ``PolicyRule`` for full control.
    * **Use** ``CallablePolicy`` for quick inline definitions.
    * **Import** custom policies from any module and ``register()`` them.
    * **Toggle** policies on/off via ``enable()`` / ``disable()``.
    * **Replace** or **remove** policies by name at runtime.

    Example — subclass::

        class CostCapPolicy(PolicyRule):
            name = "cost_cap"
            priority = 10
            def evaluate(self, ctx):
                if ctx.get("total_cost", 0) > 100:
                    return PolicyResult(True, "pause", "Budget exceeded")
                return PolicyResult(False, "proceed")

        engine = PolicyEngine()
        engine.register(CostCapPolicy())

    Example — callable shorthand::

        engine.register(CallablePolicy(
            "disk_check",
            lambda ctx: PolicyResult(True, "pause", "Disk full")
                        if ctx.get("disk_full") else
                        PolicyResult(False, "proceed"),
            priority=5,
        ))

    Example — from plain dict config::

        engine = PolicyEngine.from_config([
            {"name": "auto_merge",   "type": "AutoMergePolicy",   "enabled": True},
            {"name": "stale_branch", "type": "StaleBranchPolicy", "enabled": False},
        ])
    """

    # Built-in policy class registry for config-driven construction
    _POLICY_REGISTRY: dict[str, type[PolicyRule]] = {}

    def __init__(self) -> None:
        self._policies: list[PolicyRule] = []

    # ── Class-level policy type registry ──────────────────────────────────

    @classmethod
    def register_type(cls, policy_cls: type[PolicyRule]) -> type[PolicyRule]:
        """Register a policy class so it can be instantiated by name.

        Can be used as a decorator::

            @PolicyEngine.register_type
            class MyCustomPolicy(PolicyRule):
                ...

        Args:
            policy_cls: The ``PolicyRule`` subclass to register.

        Returns:
            The same class (unmodified), for decorator compatibility.
        """
        cls._POLICY_REGISTRY[policy_cls.__name__] = policy_cls
        return policy_cls

    @classmethod
    def registered_types(cls) -> dict[str, type[PolicyRule]]:
        """Return a copy of the policy type registry.

        Returns:
            Mapping of class name → class.
        """
        return dict(cls._POLICY_REGISTRY)

    # ── Instance-level registration ───────────────────────────────────────

    def register(self, policy: PolicyRule) -> PolicyEngine:
        """Register a policy rule.

        Args:
            policy: The policy rule to add.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If a policy with the same name is already registered.
        """
        for existing in self._policies:
            if existing.name == policy.name:
                raise ValueError(
                    f"Policy '{policy.name}' already registered.  "
                    f"Use replace() to update it."
                )

        self._policies.append(policy)
        return self

    def register_callable(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], PolicyResult],
        *,
        enabled: bool = True,
        priority: int = 100,
    ) -> PolicyEngine:
        """Convenience: register a plain function as a policy.

        Args:
            name: Unique policy name.
            fn: ``(context) -> PolicyResult`` callable.
            enabled: Whether the policy starts enabled.
            priority: Evaluation priority.

        Returns:
            Self for chaining.
        """
        return self.register(CallablePolicy(name, fn, enabled=enabled, priority=priority))

    def unregister(self, name: str) -> PolicyRule:
        """Remove a policy by name.

        Args:
            name: Name of the policy to remove.

        Returns:
            The removed policy.

        Raises:
            KeyError: If no policy with that name exists.
        """
        for i, p in enumerate(self._policies):
            if p.name == name:
                return self._policies.pop(i)

        raise KeyError(f"No policy named '{name}'")

    def replace(self, policy: PolicyRule) -> PolicyRule | None:
        """Replace an existing policy with the same name, or add if new.

        Args:
            policy: The policy rule to insert.

        Returns:
            The previously registered policy, or ``None`` if this is new.
        """
        for i, existing in enumerate(self._policies):
            if existing.name == policy.name:
                self._policies[i] = policy
                return existing

        self._policies.append(policy)
        return None

    def enable(self, name: str) -> None:
        """Enable a policy by name.

        Args:
            name: Policy name.

        Raises:
            KeyError: If no policy with that name exists.
        """
        self._get(name).enabled = True

    def disable(self, name: str) -> None:
        """Disable a policy by name (skipped during evaluation).

        Args:
            name: Policy name.

        Raises:
            KeyError: If no policy with that name exists.
        """
        self._get(name).enabled = False

    def get(self, name: str) -> PolicyRule | None:
        """Get a policy by name.

        Args:
            name: Policy name.

        Returns:
            The policy, or ``None``.
        """
        for p in self._policies:
            if p.name == name:
                return p

        return None

    @property
    def policies(self) -> list[PolicyRule]:
        """All registered policies (in registration order)."""
        return list(self._policies)

    @property
    def names(self) -> list[str]:
        """Names of all registered policies."""
        return [p.name for p in self._policies]

    def _get(self, name: str) -> PolicyRule:
        """Get a policy by name or raise.

        Args:
            name: Policy name.

        Returns:
            The policy rule.

        Raises:
            KeyError: If not found.
        """
        for p in self._policies:
            if p.name == name:
                return p

        raise KeyError(f"No policy named '{name}'")

    def _sorted_policies(self) -> list[PolicyRule]:
        """Return enabled policies sorted by priority.

        Returns:
            Sorted list of active policies.
        """
        return sorted(
            (p for p in self._policies if p.enabled),
            key=lambda p: p.priority,
        )

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, context: dict[str, Any]) -> list[PolicyResult]:
        """Evaluate all enabled policies (by priority) and return results.

        Args:
            context: Execution context data.

        Returns:
            List of ``PolicyResult`` objects for all enabled policies.
        """
        results: list[PolicyResult] = []

        for policy in self._sorted_policies():
            try:
                result = policy.evaluate(context)
                # Attach policy name if not already set
                if not result.policy_name:
                    result = PolicyResult(
                        triggered=result.triggered,
                        action=result.action,
                        reason=result.reason,
                        policy_name=policy.name,
                    )
                results.append(result)
            except Exception:
                logger.exception("Policy '%s' evaluation error", policy.name)
                results.append(PolicyResult(
                    triggered=False,
                    action="error",
                    reason=f"Policy '{policy.name}' raised an exception",
                    policy_name=policy.name,
                ))

        return results

    def triggered(self, context: dict[str, Any]) -> list[PolicyResult]:
        """Return only the triggered policy results.

        Args:
            context: Execution context data.

        Returns:
            List of triggered ``PolicyResult`` objects.
        """
        return [r for r in self.evaluate(context) if r.triggered]

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_config(self) -> list[dict[str, Any]]:
        """Serialise all policies to a config-friendly list.

        Returns:
            List of policy dictionaries.
        """
        return [p.to_dict() for p in self._policies]

    # ── Factory methods ───────────────────────────────────────────────────

    @classmethod
    def default(cls) -> PolicyEngine:
        """Create an engine pre-loaded with all 5 default policies.

        Returns:
            A ``PolicyEngine`` with standard policies registered.
        """
        engine = cls()
        engine.register(AutoMergePolicy())
        engine.register(StaleBranchPolicy())
        engine.register(StartupRecoveryPolicy())
        engine.register(LaneCompletionPolicy())
        engine.register(DegradedModePolicy())
        return engine

    @classmethod
    def from_policies(cls, policies: list[PolicyRule]) -> PolicyEngine:
        """Create an engine from an explicit list of policies.

        Args:
            policies: Policies to register.

        Returns:
            A new ``PolicyEngine``.
        """
        engine = cls()

        for p in policies:
            engine.register(p)

        return engine

    @classmethod
    def from_callables(
        cls,
        callables: dict[str, Callable[[dict[str, Any]], PolicyResult]],
    ) -> PolicyEngine:
        """Create an engine entirely from plain functions.

        Args:
            callables: Mapping of ``policy_name → callable``.

        Returns:
            A new ``PolicyEngine`` with ``CallablePolicy`` wrappers.

        Example:
            >>> engine = PolicyEngine.from_callables({
            ...     "disk_check": lambda ctx: PolicyResult(
            ...         ctx.get("disk_full", False), "pause", "Disk full"
            ...     ),
            ... })
        """
        engine = cls()

        for name, fn in callables.items():
            engine.register(CallablePolicy(name, fn))

        return engine

    @classmethod
    def from_config(
        cls,
        config: list[dict[str, Any]],
        *,
        extra_types: dict[str, type[PolicyRule]] | None = None,
    ) -> PolicyEngine:
        """Create an engine from a serialised config list.

        Each entry must have at least ``"type"`` (class name) and
        optionally ``"enabled"`` and ``"priority"``.  The class is
        looked up in the built-in registry plus any ``extra_types``
        mapping.

        Args:
            config: List of policy config dicts.
            extra_types: Additional ``class_name → class`` mapping.

        Returns:
            A new ``PolicyEngine``.

        Raises:
            ValueError: If a policy type is not found in the registry.

        Example:
            >>> engine = PolicyEngine.from_config([
            ...     {"type": "AutoMergePolicy", "enabled": True},
            ...     {"type": "StaleBranchPolicy", "enabled": False},
            ... ])
        """
        registry = {**cls._POLICY_REGISTRY, **(extra_types or {})}
        engine = cls()

        for entry in config:
            type_name = entry.get("type", "")
            policy_cls = registry.get(type_name)

            if policy_cls is None:
                raise ValueError(
                    f"Unknown policy type '{type_name}'.  "
                    f"Available: {list(registry.keys())}.  "
                    f"Register with PolicyEngine.register_type()."
                )

            policy = policy_cls()
            policy.enabled = entry.get("enabled", True)
            policy.priority = entry.get("priority", 100)
            engine.register(policy)

        return engine


# Auto-register built-in policy types for config-driven construction
for _pcls in _BUILTIN_POLICIES:
    PolicyEngine.register_type(_pcls)
PolicyEngine.register_type(CallablePolicy)
del _pcls


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GREEN-NESS CONTRACT (4-LEVEL VERIFICATION)
# ═══════════════════════════════════════════════════════════════════════════════


class VerificationLevel(Enum):
    """Four levels of green-ness verification.

    Each level implies all previous levels have passed.
    """

    TARGETED_TEST = 1
    MODULE_GREEN = 2
    WORKSPACE_GREEN = 3
    MERGE_READY = 4


@dataclass(frozen=True)
class VerificationResult:
    """Result of a single verification check.

    Args:
        level: Which verification level was checked.
        passed: Whether the check passed.
        details: Human-readable details.
        command: Command that was executed (if any).
        duration_seconds: Time taken to run the check.
    """

    level: VerificationLevel
    passed: bool
    details: str = ""
    command: str = ""
    duration_seconds: float = 0.0


class VerificationContract:
    """4-level green-ness verification contract.

    Progressively verifies that changes are safe, from targeted tests
    up to merge-readiness.

    Args:
        commands: Mapping of ``VerificationLevel`` to shell commands.
            If not provided, uses default ``pytest`` commands.
        working_dir: Directory to run commands in.

    Example:
        >>> vc = VerificationContract(commands={
        ...     VerificationLevel.TARGETED_TEST: "pytest tests/test_auth.py -x",
        ...     VerificationLevel.MODULE_GREEN: "pytest tests/ -x",
        ...     VerificationLevel.WORKSPACE_GREEN: "pytest --tb=short",
        ...     VerificationLevel.MERGE_READY: "pytest && ruff check .",
        ... })
        >>> results = vc.verify_up_to(VerificationLevel.MODULE_GREEN)
    """

    def __init__(
        self,
        *,
        commands: dict[VerificationLevel, str] | None = None,
        working_dir: str = ".",
    ) -> None:
        self._commands: dict[VerificationLevel, str] = commands or {
            VerificationLevel.TARGETED_TEST: "pytest -x --tb=short -q",
            VerificationLevel.MODULE_GREEN: "pytest -x --tb=short",
            VerificationLevel.WORKSPACE_GREEN: "pytest --tb=short",
            VerificationLevel.MERGE_READY: "pytest && ruff check . --quiet",
        }
        self._working_dir = working_dir
        self._results: list[VerificationResult] = []

    @property
    def results(self) -> list[VerificationResult]:
        """All verification results recorded so far."""
        return list(self._results)

    @property
    def highest_passed(self) -> VerificationLevel | None:
        """Highest verification level that passed, or ``None``."""
        passed = [r.level for r in self._results if r.passed]

        if not passed:
            return None

        return max(passed, key=lambda l: l.value)

    def verify_level(self, level: VerificationLevel) -> VerificationResult:
        """Run verification for a specific level.

        Args:
            level: The verification level to check.

        Returns:
            The verification result.
        """
        command = self._commands.get(level, "")

        if not command:
            result = VerificationResult(
                level=level, passed=True,
                details="No command configured — auto-pass",
            )
            self._results.append(result)
            return result

        start = time.monotonic()

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._working_dir,
                timeout=300,
            )
            elapsed = time.monotonic() - start
            passed = proc.returncode == 0
            details = proc.stdout[-500:] if passed else proc.stderr[-500:]
            result = VerificationResult(
                level=level, passed=passed,
                details=details, command=command,
                duration_seconds=round(elapsed, 2),
            )
        except subprocess.TimeoutExpired:
            result = VerificationResult(
                level=level, passed=False,
                details="Command timed out (300s)", command=command,
                duration_seconds=300.0,
            )
        except Exception as exc:
            result = VerificationResult(
                level=level, passed=False,
                details=f"Execution error: {exc}", command=command,
                duration_seconds=time.monotonic() - start,
            )

        self._results.append(result)
        logger.info(
            "Verification %s: %s (%s)",
            level.name, "PASS" if result.passed else "FAIL",
            result.details[:80],
        )
        return result

    def verify_up_to(self, target: VerificationLevel) -> list[VerificationResult]:
        """Run all verification levels up to and including ``target``.

        Stops on the first failure.

        Args:
            target: Highest level to verify.

        Returns:
            List of verification results.
        """
        results: list[VerificationResult] = []

        for level in VerificationLevel:
            result = self.verify_level(level)
            results.append(result)

            if not result.passed or level == target:
                break

        return results

    def is_green(self, level: VerificationLevel) -> bool:
        """Check if a specific level has been verified as green.

        Args:
            level: The level to check.

        Returns:
            ``True`` if that level passed.
        """
        return any(r.level == level and r.passed for r in self._results)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WORKTREE ISOLATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WorktreeInfo:
    """Information about a git worktree.

    Args:
        path: Absolute path to the worktree.
        branch: Branch checked out in this worktree.
        session_id: Associated session identifier.
        created_at: UTC timestamp of creation.
    """

    path: str
    branch: str
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorktreeError(Exception):
    """Error during worktree operations."""


class WorkspaceMismatchError(WorktreeError):
    """CWD does not match the session's assigned worktree."""


class WorktreeManager:
    """Git worktree manager for parallel session isolation.

    Each session gets its own git worktree to prevent phantom completions
    (writes to wrong worktree reported as success).

    Args:
        repo_root: Path to the main git repository.

    Example:
        >>> wm = WorktreeManager("/path/to/repo")
        >>> info = wm.create("session-1", "fix/auth-bug")
        >>> wm.validate_cwd("session-1", info.path)  # OK
        >>> wm.remove("session-1")
    """

    def __init__(self, repo_root: str) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._worktrees: dict[str, WorktreeInfo] = {}
        self._lock = threading.Lock()

    @property
    def repo_root(self) -> Path:
        """Root of the main git repository."""
        return self._repo_root

    @property
    def active_worktrees(self) -> dict[str, WorktreeInfo]:
        """Copy of active worktrees by session ID."""
        with self._lock:
            return dict(self._worktrees)

    def create(
        self,
        session_id: str,
        branch: str,
        base_dir: str | None = None,
    ) -> WorktreeInfo:
        """Create an isolated git worktree for a session.

        Args:
            session_id: Unique session identifier.
            branch: Branch to check out (created if it does not exist).
            base_dir: Base directory for worktrees. Defaults to
                ``<repo_root>/.worktrees/``.

        Returns:
            ``WorktreeInfo`` with the path and metadata.

        Raises:
            WorktreeError: If git commands fail.
        """
        base = Path(base_dir) if base_dir else self._repo_root / ".worktrees"
        base.mkdir(parents=True, exist_ok=True)
        wt_path = base / f"wt-{session_id}"

        try:
            # Check if branch exists
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch],
                capture_output=True, text=True,
                cwd=str(self._repo_root),
            )

            if result.returncode != 0:
                # Create branch from HEAD
                subprocess.run(
                    ["git", "branch", branch],
                    capture_output=True, text=True,
                    cwd=str(self._repo_root),
                    check=True,
                )

            subprocess.run(
                ["git", "worktree", "add", str(wt_path), branch],
                capture_output=True, text=True,
                cwd=str(self._repo_root),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise WorktreeError(
                f"Failed to create worktree for session '{session_id}': {exc.stderr}"
            ) from exc

        info = WorktreeInfo(
            path=str(wt_path),
            branch=branch,
            session_id=session_id,
        )

        with self._lock:
            self._worktrees[session_id] = info

        logger.info("Created worktree %s at %s (branch: %s)", session_id, wt_path, branch)
        return info

    def remove(self, session_id: str) -> None:
        """Remove the worktree associated with a session.

        Args:
            session_id: Session whose worktree to remove.

        Raises:
            WorktreeError: If the session has no worktree or removal fails.
        """
        with self._lock:
            info = self._worktrees.pop(session_id, None)

        if info is None:
            raise WorktreeError(f"No worktree for session '{session_id}'")

        try:
            subprocess.run(
                ["git", "worktree", "remove", info.path, "--force"],
                capture_output=True, text=True,
                cwd=str(self._repo_root),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("Worktree removal warning: %s", exc.stderr)

        logger.info("Removed worktree for session %s", session_id)

    def validate_cwd(self, session_id: str, cwd: str) -> None:
        """Validate that a CWD matches the session's worktree.

        Args:
            session_id: Session identifier.
            cwd: Current working directory to validate.

        Raises:
            WorkspaceMismatchError: If CWD does not match the worktree.
        """
        with self._lock:
            info = self._worktrees.get(session_id)

        if info is None:
            return  # No worktree assigned — no validation needed

        wt_path = Path(info.path).resolve()
        cwd_path = Path(cwd).resolve()

        if not (cwd_path == wt_path or wt_path in cwd_path.parents):
            raise WorkspaceMismatchError(
                f"CWD '{cwd}' does not match worktree '{info.path}' "
                f"for session '{session_id}'"
            )

    def get_worktree(self, session_id: str) -> WorktreeInfo | None:
        """Get worktree info for a session.

        Args:
            session_id: Session identifier.

        Returns:
            ``WorktreeInfo`` or ``None`` if no worktree exists.
        """
        with self._lock:
            return self._worktrees.get(session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. STALE-BRANCH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BranchStatus:
    """Status of a branch relative to a reference branch.

    Args:
        branch: Name of the branch.
        reference: Reference branch compared against.
        is_stale: Whether the branch is behind the reference.
        commits_behind: Number of commits behind.
        commits_ahead: Number of commits ahead.
        needs_rebase: Whether a rebase/merge-forward is recommended.
    """

    branch: str
    reference: str
    is_stale: bool
    commits_behind: int = 0
    commits_ahead: int = 0
    needs_rebase: bool = False


class StaleBranchDetector:
    """Detect and auto-rebase stale branches before running broad tests.

    Prevents misclassifying stale failures as new regressions.

    Args:
        repo_path: Path to the git repository.
        reference_branch: Branch to compare against (default ``"main"``).

    Example:
        >>> detector = StaleBranchDetector("/path/to/repo")
        >>> status = detector.check("feature/auth-fix")
        >>> if status.is_stale:
        ...     detector.merge_forward("feature/auth-fix")
    """

    def __init__(
        self,
        repo_path: str,
        reference_branch: str = "main",
    ) -> None:
        self._repo_path = Path(repo_path).resolve()
        self._reference = reference_branch

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command.

        Args:
            *args: Git command arguments.

        Returns:
            Completed process result.
        """
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True,
            cwd=str(self._repo_path),
        )

    def check(self, branch: str | None = None) -> BranchStatus:
        """Check if a branch is stale relative to the reference.

        Args:
            branch: Branch to check. ``None`` uses the current branch.

        Returns:
            ``BranchStatus`` with staleness information.
        """
        if branch is None:
            result = self._git("rev-parse", "--abbrev-ref", "HEAD")
            branch = result.stdout.strip() or "HEAD"

        # Fetch latest to ensure accurate comparison
        self._git("fetch", "origin", self._reference)

        # Count commits behind and ahead
        result = self._git(
            "rev-list", "--left-right", "--count",
            f"origin/{self._reference}...{branch}",
        )

        behind, ahead = 0, 0

        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()

            if len(parts) == 2:
                behind = int(parts[0])
                ahead = int(parts[1])

        is_stale = behind > 0
        return BranchStatus(
            branch=branch,
            reference=self._reference,
            is_stale=is_stale,
            commits_behind=behind,
            commits_ahead=ahead,
            needs_rebase=is_stale,
        )

    def merge_forward(self, branch: str | None = None) -> bool:
        """Merge the reference branch into the target branch.

        Args:
            branch: Branch to update. ``None`` uses the current branch.

        Returns:
            ``True`` if merge succeeded, ``False`` on conflict.
        """
        if branch is None:
            result = self._git("rev-parse", "--abbrev-ref", "HEAD")
            branch = result.stdout.strip()

        # Checkout the branch
        self._git("checkout", branch)

        # Merge reference
        result = self._git("merge", f"origin/{self._reference}", "--no-edit")

        if result.returncode != 0:
            logger.warning(
                "Merge-forward failed for %s: %s",
                branch, result.stderr[:200],
            )
            # Abort the merge
            self._git("merge", "--abort")
            return False

        logger.info("Merge-forward: %s updated from origin/%s", branch, self._reference)
        return True

    def rebase(self, branch: str | None = None) -> bool:
        """Rebase the target branch onto the reference branch.

        Args:
            branch: Branch to rebase. ``None`` uses the current branch.

        Returns:
            ``True`` if rebase succeeded, ``False`` on conflict.
        """
        if branch is None:
            result = self._git("rev-parse", "--abbrev-ref", "HEAD")
            branch = result.stdout.strip()

        self._git("checkout", branch)
        result = self._git("rebase", f"origin/{self._reference}")

        if result.returncode != 0:
            logger.warning("Rebase failed for %s: %s", branch, result.stderr[:200])
            self._git("rebase", "--abort")
            return False

        logger.info("Rebased %s onto origin/%s", branch, self._reference)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONVERSATION-LEVEL CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConversationCheckpoint:
    """Snapshot of an agent conversation at a point in time.

    Captures the full session state including events, state dict, and
    message history — enabling rewind to any previous point.

    Args:
        checkpoint_id: Unique identifier.
        session_id: Associated session.
        events_snapshot: Serialised copy of all events at checkpoint time.
        state_snapshot: Deep copy of the session state dict.
        messages_snapshot: Copy of the message history.
        description: Human-readable description.
        created_at: UTC timestamp.
    """

    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    session_id: str = ""
    events_snapshot: list[dict[str, Any]] = field(default_factory=list)
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    messages_snapshot: list[dict[str, str]] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "events_snapshot": self.events_snapshot,
            "state_snapshot": self.state_snapshot,
            "messages_snapshot": self.messages_snapshot,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationCheckpoint:
        """Deserialise from dictionary.

        Args:
            data: Dictionary with checkpoint fields.

        Returns:
            New ``ConversationCheckpoint`` instance.
        """
        return cls(
            checkpoint_id=data.get("checkpoint_id", uuid.uuid4().hex[:10]),
            session_id=data.get("session_id", ""),
            events_snapshot=data.get("events_snapshot", []),
            state_snapshot=data.get("state_snapshot", {}),
            messages_snapshot=data.get("messages_snapshot", []),
            description=data.get("description", ""),
        )

    def to_json(self) -> str:
        """Serialise to JSON.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> ConversationCheckpoint:
        """Deserialise from JSON.

        Args:
            text: JSON string.

        Returns:
            New ``ConversationCheckpoint`` instance.
        """
        return cls.from_dict(json.loads(text))


class ConversationCheckpointManager:
    """Manage conversation-level checkpoints for agent sessions.

    Provides save, restore, and rewind for full conversation state —
    including events, state dict, and message history.

    Args:
        max_checkpoints: Maximum checkpoints to keep per session (0 = unlimited).

    Example:
        >>> mgr = ConversationCheckpointManager()
        >>> mgr.save(session, messages, description="Before refactor")
        >>> # ... agent works ...
        >>> mgr.rewind(session, messages_ref, checkpoint_id="abc123")
    """

    def __init__(self, max_checkpoints: int = 50) -> None:
        self._checkpoints: dict[str, list[ConversationCheckpoint]] = {}
        self._max = max_checkpoints
        self._lock = threading.Lock()

    def save(
        self,
        session: Any,
        messages: list[dict[str, str]] | None = None,
        description: str = "",
    ) -> ConversationCheckpoint:
        """Create a checkpoint of the current conversation state.

        Args:
            session: A ``Session`` object (from ``nono.agent.base``).
            messages: Current LLM message history.
            description: Human-readable description.

        Returns:
            The created ``ConversationCheckpoint``.
        """
        sid = getattr(session, "session_id", "unknown")
        events = getattr(session, "events", [])
        state = getattr(session, "state", {})

        # Serialise events
        events_data: list[dict[str, Any]] = []

        for ev in events:
            ev_dict: dict[str, Any] = {
                "event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type),
                "author": getattr(ev, "author", ""),
                "content": getattr(ev, "content", ""),
                "data": getattr(ev, "data", {}),
                "event_id": getattr(ev, "event_id", ""),
            }
            events_data.append(ev_dict)

        checkpoint = ConversationCheckpoint(
            session_id=sid,
            events_snapshot=events_data,
            state_snapshot=copy.deepcopy(state),
            messages_snapshot=copy.deepcopy(messages or []),
            description=description,
        )

        with self._lock:
            session_ckpts = self._checkpoints.setdefault(sid, [])
            session_ckpts.append(checkpoint)

            if self._max > 0 and len(session_ckpts) > self._max:
                session_ckpts.pop(0)

        logger.info(
            "Saved conversation checkpoint '%s' for session %s (%d events, %d messages)",
            checkpoint.checkpoint_id, sid, len(events_data),
            len(messages or []),
        )
        return checkpoint

    def list_checkpoints(self, session_id: str) -> list[ConversationCheckpoint]:
        """List all checkpoints for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of checkpoints in chronological order.
        """
        with self._lock:
            return list(self._checkpoints.get(session_id, []))

    def get_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> ConversationCheckpoint | None:
        """Retrieve a specific checkpoint.

        Args:
            session_id: Session identifier.
            checkpoint_id: Checkpoint identifier.

        Returns:
            The checkpoint, or ``None`` if not found.
        """
        with self._lock:
            for ckpt in self._checkpoints.get(session_id, []):
                if ckpt.checkpoint_id == checkpoint_id:
                    return ckpt

        return None

    def rewind(
        self,
        session: Any,
        messages_ref: list[dict[str, str]],
        checkpoint_id: str,
    ) -> ConversationCheckpoint | None:
        """Rewind a session to a previous checkpoint.

        Restores the session's state dict and returns the checkpoint
        data.  The caller is responsible for replacing the message list
        content using the returned snapshot.

        Args:
            session: A ``Session`` object.
            messages_ref: Mutable reference to the message list (will be
                cleared and repopulated).
            checkpoint_id: ID of the checkpoint to rewind to.

        Returns:
            The checkpoint that was restored, or ``None`` if not found.
        """
        sid = getattr(session, "session_id", "unknown")
        ckpt = self.get_checkpoint(sid, checkpoint_id)

        if ckpt is None:
            logger.warning("Checkpoint '%s' not found for session %s", checkpoint_id, sid)
            return None

        # Restore state
        if hasattr(session, "state") and isinstance(session.state, dict):
            session.state.clear()
            session.state.update(copy.deepcopy(ckpt.state_snapshot))

        # Restore messages
        messages_ref.clear()
        messages_ref.extend(copy.deepcopy(ckpt.messages_snapshot))

        # Remove checkpoints after the rewound one
        with self._lock:
            session_ckpts = self._checkpoints.get(sid, [])
            idx = next(
                (i for i, c in enumerate(session_ckpts) if c.checkpoint_id == checkpoint_id),
                None,
            )

            if idx is not None:
                self._checkpoints[sid] = session_ckpts[:idx + 1]

        logger.info(
            "Rewound session %s to checkpoint '%s' (%s)",
            sid, checkpoint_id, ckpt.description,
        )
        return ckpt

    def clear(self, session_id: str) -> int:
        """Remove all checkpoints for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Number of checkpoints removed.
        """
        with self._lock:
            removed = len(self._checkpoints.pop(session_id, []))

        logger.info("Cleared %d checkpoints for session %s", removed, session_id)
        return removed


# ═══════════════════════════════════════════════════════════════════════════════
# 9. PLAN MODE (READ-ONLY EXPLORATION)
# ═══════════════════════════════════════════════════════════════════════════════

# Import base agent lazily to avoid circular imports
_BASE_AGENT_IMPORTED = False
_BaseAgent: Any = None
_Event: Any = None
_EventType: Any = None
_InvocationContext: Any = None
_FunctionTool: Any = None


def _lazy_import() -> None:
    """Import agent base classes on first use to avoid circular imports."""
    global _BASE_AGENT_IMPORTED, _BaseAgent, _Event, _EventType  # noqa: PLW0603
    global _InvocationContext, _FunctionTool  # noqa: PLW0603

    if _BASE_AGENT_IMPORTED:
        return

    from .base import BaseAgent as BA
    from .base import Event as Ev
    from .base import EventType as ET
    from .base import InvocationContext as IC
    from .tool import FunctionTool as FT

    _BaseAgent = BA
    _Event = Ev
    _EventType = ET
    _InvocationContext = IC
    _FunctionTool = FT
    _BASE_AGENT_IMPORTED = True


def _is_read_only_tool(tool: Any) -> bool:
    """Determine if a tool is read-only (safe for plan mode).

    Args:
        tool: A ``FunctionTool`` instance.

    Returns:
        ``True`` if the tool is considered read-only.
    """
    name = getattr(tool, "name", "").lower()
    description = getattr(tool, "description", "").lower()

    # Explicit read-only markers
    read_markers = [
        "read", "get", "list", "search", "find", "fetch", "query",
        "inspect", "check", "status", "info", "describe", "show",
        "count", "exists", "is_", "has_", "lookup", "browse",
    ]
    write_markers = [
        "write", "create", "delete", "remove", "update", "set",
        "modify", "edit", "insert", "drop", "execute", "run",
        "send", "push", "deploy", "install", "move", "rename",
    ]

    # Check tool name first (most reliable)
    for marker in write_markers:
        if marker in name:
            return False

    for marker in read_markers:
        if marker in name:
            return True

    # Fall back to description
    for marker in write_markers:
        if marker in description:
            return False

    return True


class PlanModeAgent:
    """Read-only exploration agent that explores before executing.

    Wraps an existing agent and restricts it to read-only tools.
    Produces a plan (list of proposed actions) without making changes.

    Args:
        agent: The underlying agent to wrap.
        allowed_tool_names: Explicit set of tool names allowed.
            If ``None``, uses heuristic-based read-only detection.

    Example:
        >>> plan_agent = PlanModeAgent(my_agent)
        >>> result = plan_agent.run(session, "Analyse the auth module")
        >>> print(result.plan)  # Proposed actions without execution
    """

    def __init__(
        self,
        agent: Any,
        *,
        allowed_tool_names: set[str] | None = None,
    ) -> None:
        _lazy_import()
        self._agent = agent
        self._allowed_names = allowed_tool_names
        self._plan_items: list[str] = []

    @property
    def plan(self) -> list[str]:
        """Collected plan items from the last run."""
        return list(self._plan_items)

    def _filter_tools(self, tools: list[Any]) -> list[Any]:
        """Filter tools to read-only subset.

        Args:
            tools: Original tool list.

        Returns:
            Filtered list of read-only tools.
        """
        if self._allowed_names is not None:
            return [t for t in tools if getattr(t, "name", "") in self._allowed_names]

        return [t for t in tools if _is_read_only_tool(t)]

    def _inject_plan_instruction(self, instruction: str) -> str:
        """Add plan-mode prefix to the agent instruction.

        Args:
            instruction: Original instruction.

        Returns:
            Modified instruction with plan-mode prefix.
        """
        prefix = (
            "You are in PLAN MODE (read-only). Do NOT make any changes. "
            "Instead, explore the codebase and produce a detailed plan of "
            "proposed actions. List each action as a numbered step. "
            "Describe WHAT you would do and WHY, but do not execute.\n\n"
        )
        return prefix + (instruction or "")

    def run(self, session: Any, user_message: str) -> PlanResult:
        """Run the agent in plan mode (read-only).

        Args:
            session: A ``Session`` object.
            user_message: The user's request.

        Returns:
            A ``PlanResult`` with the exploration output and proposed plan.
        """
        _lazy_import()

        # Save original tools and instruction
        original_tools = list(getattr(self._agent, "tools", []))
        original_instruction = getattr(self._agent, "instruction", "")

        try:
            # Restrict to read-only tools
            filtered = self._filter_tools(original_tools)

            if hasattr(self._agent, "tools"):
                self._agent.tools = filtered

            # Inject plan-mode instruction
            if hasattr(self._agent, "instruction"):
                self._agent.instruction = self._inject_plan_instruction(original_instruction)

            # Run via Runner or direct
            output = ""
            events: list[Any] = []

            if hasattr(self._agent, "run"):
                from .base import InvocationContext as IC
                ctx = IC(session=session, user_message=user_message)

                for event in self._agent._run_impl(ctx):
                    events.append(event)

                    if hasattr(event, "event_type") and hasattr(event, "content"):
                        if event.event_type.value == "agent_message":
                            output = event.content

            # Parse plan items from output
            self._plan_items = self._extract_plan(output)

            return PlanResult(
                output=output,
                plan=self._plan_items,
                tools_available=[getattr(t, "name", str(t)) for t in filtered],
                tools_blocked=[
                    getattr(t, "name", str(t))
                    for t in original_tools if t not in filtered
                ],
                events=events,
            )
        finally:
            # Restore original state
            if hasattr(self._agent, "tools"):
                self._agent.tools = original_tools

            if hasattr(self._agent, "instruction"):
                self._agent.instruction = original_instruction

    @staticmethod
    def _extract_plan(output: str) -> list[str]:
        """Extract numbered plan items from agent output.

        Args:
            output: Agent output text.

        Returns:
            List of plan step descriptions.
        """
        import re

        items: list[str] = []
        pattern = re.compile(r"^\s*(\d+)[.)]\s+(.+)", re.MULTILINE)

        for match in pattern.finditer(output):
            items.append(match.group(2).strip())

        return items


@dataclass
class PlanResult:
    """Result of a plan-mode execution.

    Args:
        output: Full text output from the agent.
        plan: Extracted plan steps.
        tools_available: Names of read-only tools that were available.
        tools_blocked: Names of write tools that were blocked.
        events: Raw events from the agent run.
    """

    output: str = ""
    plan: list[str] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    tools_blocked: list[str] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
