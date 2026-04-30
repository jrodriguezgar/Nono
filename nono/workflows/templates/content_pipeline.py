"""Content generation pipeline workflow.

Steps:
    1. outline — Generate a structured outline from the topic.
    2. draft — Write a full draft based on the outline.
    3. review — Review the draft for quality and suggest improvements.
    4. Polish — Apply review suggestions and produce final content.

Usage::

    flow = build_content_pipeline()
    result = flow.run(input="Write a blog post about async Python patterns.")
    # result["final"]  ->  "# Async Python Patterns ..."
"""

from __future__ import annotations

from nono.workflows import Workflow, tasker_node


def build_content_pipeline() -> Workflow:
    """Build the content generation pipeline.

    Returns:
        Configured ``Workflow`` with four sequential steps.
    """
    flow = Workflow("content_pipeline")

    flow.step(
        "outline",
        tasker_node(
            system_prompt=(
                "You are a content strategist. Given a topic, create a "
                "structured outline with: title, 4-6 sections (each with "
                "a heading and 2-3 bullet points), and a conclusion. "
                "Return the outline as Markdown."
            ),
            input_key="input",
            output_key="outline",
            temperature=0.5,
        ),
    )

    flow.step(
        "draft",
        tasker_node(
            system_prompt=(
                "You are a technical writer. Given an outline in Markdown, "
                "write a complete blog post (~800 words). Use clear headings, "
                "code examples where relevant, and a professional tone. "
                "Return the full Markdown article."
            ),
            input_key="outline",
            output_key="draft",
            temperature=0.7,
        ),
    )

    flow.step(
        "review",
        tasker_node(
            system_prompt=(
                "You are an editor. Review the draft and return a JSON object "
                "with: overall_score (1-10), strengths (list), improvements "
                "(list of specific suggestions), and revised_sections (dict "
                "mapping section heading to improved text, only for sections "
                "that need changes)."
            ),
            input_key="draft",
            output_key="review",
            temperature=0.3,
        ),
    )

    flow.step(
        "polish",
        tasker_node(
            system_prompt=(
                "You are a copy editor. Given the original draft and the "
                "review feedback, produce the final polished article. Apply "
                "all suggested improvements while keeping the author's voice. "
                "Return only the final Markdown article."
            ),
            input_key="review",
            output_key="final",
            temperature=0.4,
        ),
    )

    flow.connect("outline", "draft", "review", "polish")
    return flow
