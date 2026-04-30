"""
RoutineStore — JSON-based persistence for routine definitions.

Saves and loads routine definitions to/from a JSON file.  The store
handles serialisation of triggers, config, and metadata.  Executables
(agents, workflows, callables) are referenced by name and must be
re-bound at load time.

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .routine import Routine, RoutineConfig, RoutineStatus
from .triggers import (
    EventTrigger,
    ManualTrigger,
    RoutineTrigger,
    ScheduleTrigger,
    TriggerType,
    WebhookTrigger,
)

logger = logging.getLogger("Nono.Routines.Store")


def _trigger_from_dict(data: dict[str, Any]) -> RoutineTrigger:
    """Deserialise a trigger from its dictionary representation.

    Args:
        data: Trigger dictionary with a ``"type"`` key.

    Returns:
        A ``RoutineTrigger`` subclass instance.

    Raises:
        ValueError: If the trigger type is unknown.
    """
    ttype = data.get("type", "manual")

    if ttype == TriggerType.SCHEDULE.value:
        last_fired = None
        if data.get("last_fired"):
            try:
                last_fired = datetime.fromisoformat(data["last_fired"])
            except (ValueError, TypeError):
                pass

        return ScheduleTrigger(
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            cron=data.get("cron"),
            interval_seconds=data.get("interval_seconds", 0),
            last_fired=last_fired,
            timezone_name=data.get("timezone_name", "UTC"),
        )

    if ttype == TriggerType.EVENT.value:
        return EventTrigger(
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            event_name=data.get("event_name", ""),
            event_pattern=data.get("event_pattern", ""),
        )

    if ttype == TriggerType.WEBHOOK.value:
        return WebhookTrigger(
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            path=data.get("path", ""),
            secret=data.get("secret", ""),
            allowed_ips=data.get("allowed_ips", []),
        )

    if ttype == TriggerType.MANUAL.value:
        return ManualTrigger(
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
        )

    raise ValueError(f"Unknown trigger type: {ttype!r}")


def _routine_from_dict(data: dict[str, Any]) -> Routine:
    """Deserialise a routine from its dictionary representation.

    The ``executable`` field is **not** restored — it must be re-bound
    by the caller (e.g., via an executable registry or factory function).

    Args:
        data: Routine dictionary.

    Returns:
        A ``Routine`` instance (without executable).
    """
    config_data = data.get("config", {})
    config = RoutineConfig(
        timeout_seconds=config_data.get("timeout_seconds", 300),
        max_retries=config_data.get("max_retries", 0),
        retry_delay_seconds=config_data.get("retry_delay_seconds", 5.0),
        max_history=config_data.get("max_history", 100),
        environment=config_data.get("environment", {}),
        tags=config_data.get("tags", []),
    )

    triggers = [
        _trigger_from_dict(t) if isinstance(t, dict) else ManualTrigger()
        for t in data.get("triggers", [])
    ]

    status_str = data.get("status", "idle")
    try:
        status = RoutineStatus(status_str)
    except ValueError:
        status = RoutineStatus.IDLE

    created_at = datetime.now(timezone.utc)
    if data.get("created_at"):
        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except (ValueError, TypeError):
            pass

    updated_at = datetime.now(timezone.utc)
    if data.get("updated_at"):
        try:
            updated_at = datetime.fromisoformat(data["updated_at"])
        except (ValueError, TypeError):
            pass

    return Routine(
        name=data["name"],
        description=data.get("description", ""),
        executable=None,  # must be re-bound
        triggers=triggers,
        config=config,
        instruction=data.get("instruction", ""),
        input_template=data.get("input_template", ""),
        tools=[],
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        routine_id=data.get("routine_id", ""),
        metadata=data.get("metadata", {}),
    )


class RoutineStore:
    """JSON file-based persistence for routine definitions.

    Args:
        path: Path to the JSON file.  Created on first ``save()``.

    Example:
        >>> store = RoutineStore("routines.json")
        >>> store.save([routine1, routine2])
        >>> loaded = store.load()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, routines: list[Routine]) -> None:
        """Persist routine definitions to disk.

        Args:
            routines: List of routines to save.
        """
        data = {
            "version": "1.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "routines": [r.to_dict() for r in routines],
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved %d routines to %s", len(routines), self.path)

    def load(self) -> list[Routine]:
        """Load routine definitions from disk.

        Returns:
            List of routines (executables will be ``None``).

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Routine store not found: {self.path}")

        text = self.path.read_text(encoding="utf-8")
        data = json.loads(text)

        routines_data = data.get("routines", [])
        routines = []
        for rd in routines_data:
            try:
                routines.append(_routine_from_dict(rd))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed routine entry: %s", exc,
                )

        logger.info("Loaded %d routines from %s", len(routines), self.path)
        return routines

    def exists(self) -> bool:
        """Check if the store file exists."""
        return self.path.exists()

    def delete(self) -> None:
        """Delete the store file."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Deleted routine store %s", self.path)
