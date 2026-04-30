"""
Human-in-the-Loop (HITL) shared types.

Defines the data structures and callback protocols used by both the normal
Workflow (``nono.workflows``) and the agentic workflow
(``nono.agent.HumanInputAgent``) to pause execution and wait for human input.

The human can **approve**, **reject**, or provide a **custom prompt/message**
that feeds back into the pipeline.

Usage:
    from nono.hitl import HumanInputHandler, HumanInputResponse

    def console_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        print(f"[{step_name}] {prompt}")
        print(f"  State keys: {list(state.keys())}")
        user_input = input("Approve? (y/n) or type a message: ")
        if user_input.lower() in ("y", "yes"):
            return HumanInputResponse(approved=True)
        if user_input.lower() in ("n", "no"):
            return HumanInputResponse(approved=False, message="Rejected by user")
        return HumanInputResponse(approved=True, message=user_input)

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union


__all__ = [
    "AsyncHumanInputHandler",
    "HumanInputHandler",
    "HumanInputResponse",
    "HumanRejectError",
    "console_handler",
    "format_state_for_review",
    "make_auto_handler",
]


# ── Response dataclass ───────────────────────────────────────────────────────

@dataclass
class HumanInputResponse:
    """Result returned by a human-input handler.

    Args:
        approved: Whether the human approved the current step/state.
        message: Free-text feedback, correction, or prompt from the human.
        data: Arbitrary key-value pairs to merge into the workflow state
            or agent session state.

    Example:
        >>> HumanInputResponse(approved=True)
        >>> HumanInputResponse(approved=False, message="Needs more detail")
        >>> HumanInputResponse(approved=True, data={"revised_topic": "AI safety"})
    """
    approved: bool = True
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ── Handler type aliases ─────────────────────────────────────────────────────

HumanInputHandler = Callable[[str, dict, str], HumanInputResponse]
"""Sync handler: ``(step_name, current_state, prompt) -> HumanInputResponse``.

Called when the workflow or agent reaches a human-input checkpoint.  The
handler **blocks** until the human provides a response."""

AsyncHumanInputHandler = Callable[
    [str, dict, str], Awaitable[HumanInputResponse]
]
"""Async handler: ``(step_name, current_state, prompt) -> Awaitable[HumanInputResponse]``.

Same contract as ``HumanInputHandler`` but returns an awaitable — useful
when the human input comes from a web API, WebSocket, or async queue."""


# ── Exception ────────────────────────────────────────────────────────────────

class HumanRejectError(Exception):
    """Raised when a human rejects a step and no reject-branch is configured.

    Args:
        step_name: The step where rejection occurred.
        message: The human's feedback message.
    """

    def __init__(self, step_name: str, message: str = "") -> None:
        self.step_name = step_name
        self.human_message = message
        super().__init__(
            f"Human rejected step {step_name!r}"
            + (f": {message}" if message else ""),
        )


# ── Review formatting ─────────────────────────────────────────────────────────

_MAX_VALUE_LEN = 500
"""Maximum characters displayed per state value before truncation."""


def format_state_for_review(
    state: dict[str, Any],
    display_keys: list[str] | None = None,
    *,
    max_value_len: int = _MAX_VALUE_LEN,
) -> str:
    """Format workflow state values into a human-readable review block.

    When *display_keys* is given, only those keys are shown (in order).
    Otherwise all keys are shown in insertion order.

    Long values are truncated to *max_value_len* characters with an
    ellipsis ``…`` appended.

    Args:
        state: Current workflow / session state dict.
        display_keys: Subset of keys to include.  ``None`` means all.
        max_value_len: Maximum characters per value.

    Returns:
        Multi-line formatted string ready for console / UI display.
    """
    keys = display_keys if display_keys is not None else list(state.keys())
    lines: list[str] = []

    for key in keys:
        if key not in state:
            continue
        value = state[key]
        text = str(value)

        if len(text) > max_value_len:
            text = text[:max_value_len] + "…"

        # Indent multi-line values for readability
        if "\n" in text:
            indented = text.replace("\n", "\n           ")
            lines.append(f"  ▸ {key}:\n           {indented}")
        else:
            lines.append(f"  ▸ {key}: {text}")

    return "\n".join(lines) if lines else "  (empty state)"


# ── Built-in handlers ────────────────────────────────────────────────────────

def console_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Interactive console handler — prompts the user via stdin/stdout.

    Displays the step name, prompt, and current state **values** so the
    human can review the content before approving or rejecting.  The user
    types ``y`` / ``yes`` to approve, ``n`` / ``no`` to reject, or any
    other text to approve with that text as a message.

    Designed for CLI and local development usage.

    Args:
        step_name: Name of the step requesting input.
        state: Current workflow/session state (read-only copy).
        prompt: Message to display to the human.

    Returns:
        HumanInputResponse based on user input.

    Example:
        >>> from nono.hitl import console_handler
        >>> from nono.workflows import Workflow
        >>> flow = Workflow("review")
        >>> flow.human_step("review", handler=console_handler, prompt="Approve?")
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "console_handler requires an interactive terminal (stdin is not a TTY). "
            "Use make_auto_handler() for non-interactive environments."
        )
    print(f"\n{'─'*60}")
    print(f"  ⏸  Human input required: {step_name}")
    print(f"{'─'*60}")
    print(f"  Prompt:  {prompt}")
    if state:
        print()
        print("  Content to review:")
        print(format_state_for_review(state))
    print()
    user_input = input("  Approve? (y/n) or type a message: ").strip()

    if user_input.lower() in ("y", "yes", ""):
        return HumanInputResponse(approved=True, message="Approved")
    if user_input.lower() in ("n", "no"):
        reason = input("  Reason (optional): ").strip()
        return HumanInputResponse(approved=False, message=reason or "Rejected by user")
    return HumanInputResponse(approved=True, message=user_input)


def make_auto_handler(
    responses: dict[str, dict[str, Any]] | None = None,
    *,
    default_approved: bool = True,
    default_message: str = "",
) -> HumanInputHandler:
    """Create a handler that returns pre-configured responses per step.

    Useful for testing, API integration, and CI/CD pipelines where
    human responses are known in advance.

    Args:
        responses: Mapping of ``step_name`` to response fields::

            {
                "review": {"approved": True, "message": "LGTM"},
                "final_check": {"approved": False, "message": "Needs work"},
            }

        default_approved: Default approval when a step has no entry.
        default_message: Default message when a step has no entry.

    Returns:
        A ``HumanInputHandler`` callable.

    Example:
        >>> from nono.hitl import make_auto_handler
        >>> handler = make_auto_handler({
        ...     "review": {"approved": True, "message": "Looks good"},
        ... })
        >>> resp = handler("review", {}, "Approve?")
        >>> resp.approved
        True
    """
    _responses = responses or {}

    def _handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        cfg = _responses.get(step_name, {})
        return HumanInputResponse(
            approved=cfg.get("approved", default_approved),
            message=cfg.get("message", default_message),
            data=cfg.get("data", {}),
        )

    return _handler
