"""Sentiment analysis pipeline workflow.

Steps:
    1. classify — Classify the input text sentiment and category.
    2. score — Assign a numerical confidence score.
    3. summarise — Produce a one-line human-readable summary.

Usage::

    flow = build_sentiment_pipeline()
    result = flow.run(input="The new update is amazing but login is broken.")
    # result["summary"]  ->  "Mixed sentiment: positive (feature) + negative (bug)"
"""

from __future__ import annotations

from nono.workflows import Workflow, tasker_node


def build_sentiment_pipeline() -> Workflow:
    """Build the sentiment analysis pipeline.

    Returns:
        Configured ``Workflow`` with three sequential steps.
    """
    flow = Workflow("sentiment_pipeline")

    flow.step(
        "classify",
        tasker_node(
            system_prompt=(
                "You are a sentiment classifier. Analyse the input text and "
                "return a JSON object with: sentiment (positive, negative, "
                "neutral, mixed), category (product, service, support, general), "
                "and key_phrases (list of strings)."
            ),
            input_key="input",
            output_key="classification",
            temperature=0.1,
        ),
    )

    flow.step(
        "score",
        tasker_node(
            system_prompt=(
                "You are a confidence scorer. Given a sentiment classification "
                "(in JSON), assign a confidence score from 0.0 to 1.0 for the "
                "sentiment label. Return JSON: {\"confidence\": <float>, "
                "\"reasoning\": \"<one sentence>\"}."
            ),
            input_key="classification",
            output_key="score",
            temperature=0.0,
        ),
    )

    flow.step(
        "summarise",
        tasker_node(
            system_prompt=(
                "You are a report summariser. Given a sentiment classification "
                "and confidence score, produce a single-line human-readable "
                "summary. Example: 'Positive sentiment (0.92) — user praises "
                "dark mode feature'. Return only the summary string."
            ),
            input_key="score",
            output_key="summary",
            temperature=0.3,
        ),
    )

    flow.connect("classify", "score", "summarise")
    return flow
