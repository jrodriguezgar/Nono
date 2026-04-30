"""Content pipeline with human review — HITL-enabled workflow template.

Steps:
    1. draft — Generate a draft from the topic.
    2. review — Human approves or rejects the draft.
    3. publish — Publish the approved draft.
    4. revise — (on reject) Revise the draft based on human feedback.

The ``review`` step pauses execution and calls the configured handler.
If no handler is provided, the workflow uses a default auto-approve
handler suitable for non-interactive environments (API, CI).

Usage::

    from nono.workflows.templates import build_content_review_pipeline
    from nono.hitl import console_handler

    # Interactive (CLI)
    flow = build_content_review_pipeline(handler=console_handler)
    result = flow.run(topic="AI safety in 2026")

    # Non-interactive (API / testing)
    from nono.hitl import make_auto_handler
    handler = make_auto_handler({"review": {"approved": True, "message": "LGTM"}})
    flow = build_content_review_pipeline(handler=handler)
    result = flow.run(topic="AI safety in 2026")

Author: DatamanEdge
License: MIT
"""

from __future__ import annotations

from typing import Any

from nono.hitl import HumanInputHandler, HumanInputResponse, make_auto_handler
from nono.workflows import Workflow


def build_content_review_pipeline(
    handler: HumanInputHandler | None = None,
) -> Workflow:
    """Build a content pipeline with a human review gate.

    Args:
        handler: HITL handler for the review step.  When ``None`` a
            default auto-approve handler is used.

    Returns:
        Configured ``Workflow`` with draft → review → publish | revise.
    """
    effective_handler = handler or make_auto_handler(default_approved=True)

    flow = Workflow("content_review_pipeline")

    def draft(state: dict[str, Any]) -> dict[str, Any]:
        topic = state.get("topic", "general topic")
        return {"draft": f"Draft article about {topic}"}

    def publish(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "published": True,
            "final": state.get("draft", ""),
            "review_message": state.get("human_input", {}).get("message", ""),
        }

    def revise(state: dict[str, Any]) -> dict[str, Any]:
        feedback = state.get("human_input", {}).get("message", "")
        original = state.get("draft", "")
        return {
            "draft": f"{original} [revised: {feedback}]",
            "revised": True,
        }

    flow.step("draft", draft)
    flow.human_step(
        "review",
        handler=effective_handler,
        prompt="Review and approve the draft before publishing.",
        on_reject="revise",
    )
    flow.step("publish", publish)
    flow.step("revise", revise)

    flow.connect("draft", "review")
    flow.connect("review", "publish")

    return flow
