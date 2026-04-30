"""
Built-in skills — ready-to-use AI capabilities.

Each skill can be used standalone, as a tool inside an agent, or
composed into pipelines.

Available skills:
    - **SummarizeSkill**: Text summarization into key points.
    - **ClassifySkill**: Input classification and labelling.
    - **ExtractSkill**: Structured data extraction from text.
    - **CodeReviewSkill**: Code quality review and suggestions.
    - **TranslateSkill**: Text translation between languages.

All built-in skills are auto-registered in the global ``registry``.

Usage::

    from nono.agent.skills import SummarizeSkill
    result = SummarizeSkill().run("Long text to summarize...")

    # Or via registry
    from nono.agent.skill import registry
    skill = registry.get("summarize")
    result = skill.run("Long text...")
"""

from .summarize import SummarizeSkill
from .classify import ClassifySkill
from .extract import ExtractSkill
from .code_review import CodeReviewSkill
from .translate import TranslateSkill

__all__ = [
    "SummarizeSkill",
    "ClassifySkill",
    "ExtractSkill",
    "CodeReviewSkill",
    "TranslateSkill",
]
