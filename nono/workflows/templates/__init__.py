"""
Workflow templates — pre-configured Workflow instances ready to run.

Each module exports a builder function that returns a fully configured
``Workflow``.  The function name follows the pattern ``build_<name>()``.

Usage::

    from nono.workflows.templates import (
        build_sentiment_pipeline,
        build_content_pipeline,
        build_data_enrichment,
    )

    flow = build_sentiment_pipeline()
    result = flow.run(input="I love this product!")
"""

from .sentiment_pipeline import build_sentiment_pipeline
from .content_pipeline import build_content_pipeline
from .data_enrichment import build_data_enrichment
from .content_review_pipeline import build_content_review_pipeline

__all__ = [
    "build_sentiment_pipeline",
    "build_content_pipeline",
    "build_data_enrichment",
    "build_content_review_pipeline",
]
