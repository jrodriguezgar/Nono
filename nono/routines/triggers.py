"""
Trigger types for routine execution.

Each trigger defines *when* a routine should fire.  A routine can combine
multiple triggers — any match starts a new execution.

Trigger types:
    - **ScheduleTrigger**: Cron expression or fixed interval.
    - **EventTrigger**: Application-level custom events.
    - **WebhookTrigger**: HTTP POST endpoint.
    - **ManualTrigger**: Explicit ``fire()`` call (always present implicitly).

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ── Trigger Types ─────────────────────────────────────────────────────────────

class TriggerType(Enum):
    """Supported trigger categories."""

    SCHEDULE = "schedule"
    EVENT = "event"
    WEBHOOK = "webhook"
    MANUAL = "manual"


# ── Base ──────────────────────────────────────────────────────────────────────

@dataclass
class RoutineTrigger(ABC):
    """Abstract base for all trigger types.

    Args:
        enabled: Whether this trigger is currently active.
        description: Human-readable description.
    """

    enabled: bool = True
    description: str = ""

    @property
    @abstractmethod
    def trigger_type(self) -> TriggerType:
        """Return the trigger category."""

    @abstractmethod
    def should_fire(self, now: datetime | None = None) -> bool:
        """Evaluate whether the trigger condition is met.

        Args:
            now: Current UTC timestamp (auto-generated if omitted).

        Returns:
            ``True`` if the routine should execute.
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialise trigger to a JSON-safe dictionary."""
        d = asdict(self)
        d["type"] = self.trigger_type.value
        return d


# ── Schedule Trigger ──────────────────────────────────────────────────────────

def _parse_cron_field(field_str: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integers.

    Supports: ``*``, ``N``, ``N-M``, ``*/N``, ``N-M/S``, comma-separated.

    Args:
        field_str: The cron field string.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Set of integers that match the field expression.

    Raises:
        ValueError: If the field cannot be parsed.
    """
    values: set[int] = set()

    for part in field_str.split(","):
        part = part.strip()

        # */N — every N from min_val
        if match := re.match(r"^\*/(\d+)$", part):
            step = int(match.group(1))
            if step == 0:
                raise ValueError(f"Step cannot be zero in cron field: {field_str}")
            values.update(range(min_val, max_val + 1, step))

        # N-M/S — range with step
        elif match := re.match(r"^(\d+)-(\d+)/(\d+)$", part):
            start, end, step = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if step == 0:
                raise ValueError(f"Step cannot be zero in cron field: {field_str}")
            values.update(range(start, end + 1, step))

        # N-M — range
        elif match := re.match(r"^(\d+)-(\d+)$", part):
            start, end = int(match.group(1)), int(match.group(2))
            values.update(range(start, end + 1))

        # * — wildcard
        elif part == "*":
            values.update(range(min_val, max_val + 1))

        # N — exact value
        elif re.match(r"^\d+$", part):
            values.add(int(part))

        else:
            raise ValueError(f"Cannot parse cron field component: {part!r}")

    return values


def _cron_matches(cron: str, dt: datetime) -> bool:
    """Check whether a cron expression matches a given datetime.

    Standard 5-field cron: ``minute hour day_of_month month day_of_week``.

    Args:
        cron: Cron expression string.
        dt: Datetime to test.

    Returns:
        ``True`` if the expression matches.

    Raises:
        ValueError: If the cron expression is malformed.
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Cron expression must have 5 fields (minute hour dom month dow), "
            f"got {len(parts)}: {cron!r}"
        )

    minute_f, hour_f, dom_f, month_f, dow_f = parts

    minutes = _parse_cron_field(minute_f, 0, 59)
    hours = _parse_cron_field(hour_f, 0, 23)
    doms = _parse_cron_field(dom_f, 1, 31)
    months = _parse_cron_field(month_f, 1, 12)
    dows = _parse_cron_field(dow_f, 0, 6)  # 0 = Monday

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in doms
        and dt.month in months
        and dt.weekday() in dows
    )


@dataclass
class ScheduleTrigger(RoutineTrigger):
    """Cron-based or interval-based schedule trigger.

    Supports two modes:
        1. **Cron expression**: Standard 5-field (``minute hour dom month dow``).
        2. **Interval**: Fixed delay in seconds between executions.

    At least one of ``cron`` or ``interval_seconds`` must be set.

    Args:
        cron: Cron expression (5 fields). ``None`` to use interval mode.
        interval_seconds: Seconds between executions. ``0`` to use cron mode.
        last_fired: Timestamp of the last execution (for interval calculation).
        timezone_name: IANA timezone name (informational, execution uses UTC).

    Example:
        >>> ScheduleTrigger(cron="0 2 * * *")          # daily at 2 AM
        >>> ScheduleTrigger(cron="*/15 * * * *")        # every 15 minutes
        >>> ScheduleTrigger(cron="0 9 * * 1-5")        # weekdays at 9 AM
        >>> ScheduleTrigger(interval_seconds=3600)      # every hour
    """

    cron: Optional[str] = None
    interval_seconds: int = 0
    last_fired: Optional[datetime] = None
    timezone_name: str = "UTC"

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.SCHEDULE

    def should_fire(self, now: datetime | None = None) -> bool:
        """Check if the schedule matches the current time.

        Args:
            now: Current UTC time. Auto-generated if ``None``.

        Returns:
            ``True`` if the routine should fire.
        """
        if not self.enabled:
            return False

        now = now or datetime.now(timezone.utc)

        if self.cron:
            return _cron_matches(self.cron, now)

        if self.interval_seconds > 0:
            if self.last_fired is None:
                return True
            elapsed = (now - self.last_fired).total_seconds()
            return elapsed >= self.interval_seconds

        return False

    def mark_fired(self, at: datetime | None = None) -> None:
        """Record that the trigger has fired.

        Args:
            at: Timestamp of the execution.  Defaults to UTC now.
        """
        self.last_fired = at or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.last_fired:
            d["last_fired"] = self.last_fired.isoformat()
        return d


# ── Event Trigger ─────────────────────────────────────────────────────────────

@dataclass
class EventTrigger(RoutineTrigger):
    """Fires when a matching application event is emitted.

    Events are matched by ``event_name`` with optional ``filter_fn`` for
    payload filtering.

    Args:
        event_name: Event name to listen for (exact match).
        filter_fn: Optional callable ``(event_data) -> bool`` for filtering.
        event_pattern: Regex pattern for matching event names (alternative to exact match).

    Example:
        >>> EventTrigger(event_name="pr.opened")
        >>> EventTrigger(event_name="alert.fired", filter_fn=lambda d: d.get("severity") == "critical")
        >>> EventTrigger(event_pattern=r"deploy\\..*")
    """

    event_name: str = ""
    filter_fn: Optional[Any] = field(default=None, repr=False)
    event_pattern: str = ""

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.EVENT

    def should_fire(self, now: datetime | None = None) -> bool:
        """Always returns ``False`` — event triggers are push-based.

        Use ``matches_event()`` to check if a specific event matches.
        """
        return False

    def matches_event(self, event_name: str, event_data: dict[str, Any] | None = None) -> bool:
        """Check whether an incoming event matches this trigger.

        Args:
            event_name: Name of the emitted event.
            event_data: Event payload for filter evaluation.

        Returns:
            ``True`` if the event should cause the routine to fire.
        """
        if not self.enabled:
            return False

        # Exact match
        if self.event_name and self.event_name != event_name:
            return False

        # Pattern match
        if self.event_pattern:
            if not re.match(self.event_pattern, event_name):
                return False

        # Payload filter
        if self.filter_fn and event_data:
            try:
                if not self.filter_fn(event_data):
                    return False
            except Exception:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.trigger_type.value,
            "enabled": self.enabled,
            "description": self.description,
            "event_name": self.event_name,
            "event_pattern": self.event_pattern,
        }
        return d


# ── Webhook Trigger ───────────────────────────────────────────────────────────

@dataclass
class WebhookTrigger(RoutineTrigger):
    """HTTP POST endpoint trigger.

    When registered with the ``RoutineRunner``, this trigger gets an
    auto-generated path (``/routines/{routine_name}/fire``) that accepts
    POST requests.  An optional ``secret`` enables HMAC-SHA256 validation
    of the ``X-Routine-Signature`` header.

    Args:
        path: Custom URL path (auto-generated if empty).
        secret: Shared secret for HMAC signature verification.
        allowed_ips: IP whitelist (empty = allow all).

    Example:
        >>> WebhookTrigger()                                    # auto-path, no auth
        >>> WebhookTrigger(secret="sk-my-secret-token")         # with HMAC validation
    """

    path: str = ""
    secret: str = ""
    allowed_ips: list[str] = field(default_factory=list)

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.WEBHOOK

    def should_fire(self, now: datetime | None = None) -> bool:
        """Always returns ``False`` — webhook triggers are push-based."""
        return False

    def validate_request(
        self,
        body: bytes,
        signature: str = "",
        remote_ip: str = "",
    ) -> bool:
        """Validate an incoming webhook request.

        Args:
            body: Raw request body.
            signature: Value of the ``X-Routine-Signature`` header.
            remote_ip: Client IP address.

        Returns:
            ``True`` if the request is valid.
        """
        # IP whitelist
        if self.allowed_ips and remote_ip not in self.allowed_ips:
            return False

        # HMAC verification
        if self.secret:
            import hashlib
            import hmac

            expected = hmac.new(
                self.secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, signature):
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.trigger_type.value,
            "enabled": self.enabled,
            "description": self.description,
            "path": self.path,
            "has_secret": bool(self.secret),
            "allowed_ips": self.allowed_ips,
        }
        return d


# ── Manual Trigger ────────────────────────────────────────────────────────────

@dataclass
class ManualTrigger(RoutineTrigger):
    """Explicit fire-on-demand trigger.

    Every routine implicitly supports manual firing via
    ``RoutineRunner.fire()``.  Attach a ``ManualTrigger`` explicitly
    only if you want to document or configure manual execution.

    Example:
        >>> ManualTrigger(description="Run via CLI: nono routine fire nightly_review")
    """

    @property
    def trigger_type(self) -> TriggerType:
        return TriggerType.MANUAL

    def should_fire(self, now: datetime | None = None) -> bool:
        """Always returns ``False`` — manual triggers are push-based."""
        return False
