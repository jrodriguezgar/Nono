"""
Example: Data Processing Pipeline
==================================

End-to-end project that reads input data, processes it with agents
(using tasker_tool for structured tasks and @tool for custom logic),
and produces a final output file.

Scenario
--------
A company receives raw customer feedback (CSV-like text). The pipeline:

1. **Extractor** (tasker_tool) — pulls structured fields from raw text.
2. **Classifier** (tasker_tool) — labels each item by category and sentiment.
3. **Analyst** (Agent + custom tools) — analyses patterns, computes stats,
   and writes a summary report.
4. **Guardrail** (tasker_tool) — redacts PII before the report is saved.

Architecture::

    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
    │ Raw data │──▶│ Extractor  │──▶│Classifier│──▶│ Analyst  │
    │  (input) │    │(tasker_tool)│   │(tasker_tool)│  │(Agent)   │
    └──────────┘    └───────────┘    └──────────┘    └────┬─────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │ Guardrail  │
                                                    │(tasker_tool)│
                                                    └─────┬──────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │   Output   │
                                                    │  (report)  │
                                                    └────────────┘

Usage::

    python -m nono.examples.example_data_pipeline

    # Or with custom input:
    python -m nono.examples.example_data_pipeline --input feedback.txt

    # With tracing:
    python -m nono.examples.example_data_pipeline --trace
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Nono imports ─────────────────────────────────────────────────
from nono.agent import (
    Agent,
    Runner,
    SequentialAgent,
    ToolContext,
    TraceCollector,
    tool,
    tasker_tool,
)

# ── Sample input data ───────────────────────────────────────────

SAMPLE_FEEDBACK = """\
1. "The app crashes every time I open settings. Very frustrating!" - John Smith, john.smith@example.com, 2026-03-15
2. "Love the new dark mode feature, great job!" - María García, 2026-03-16
3. "Billing charged me twice for March. Please refund ASAP. My card ends in 4242." - Bob Jones, bob@corp.net, 2026-03-17
4. "Search is slow when I type more than 3 words." - Alice W., 2026-03-18
5. "The onboarding tutorial was very helpful. Smooth experience overall." - Chen Wei, chen.wei@mail.cn, 2026-03-19
"""

# ── Tools (tasker-based) ────────────────────────────────────────

extract_tool = tasker_tool(
    name="extract_feedback",
    description="Extract structured fields (user, message, date, contact) "
                "from raw customer feedback text.",
    system_prompt=(
        "You are a data extraction specialist. Extract structured data from "
        "raw customer feedback. Return a JSON array of objects with fields: "
        "id, user, message, contact_info, date. "
        "Use null for missing fields. Do NOT invent data."
    ),
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "user": {"type": "string"},
                "message": {"type": "string"},
                "contact_info": {"type": "string"},
                "date": {"type": "string"},
            },
        },
    },
    temperature=0.1,
)

classify_tool = tasker_tool(
    name="classify_feedback",
    description="Classify each feedback item by category and sentiment.",
    system_prompt=(
        "You are a feedback classifier. For each item, assign:\n"
        "- category: one of bug, feature_request, billing, performance, praise, other\n"
        "- sentiment: positive, negative, or neutral\n"
        "- priority: critical, high, medium, low\n"
        "Return a JSON array with id, category, sentiment, priority."
    ),
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "category": {"type": "string"},
                "sentiment": {"type": "string"},
                "priority": {"type": "string"},
            },
        },
    },
    temperature=0.1,
)

redact_tool = tasker_tool(
    name="redact_pii",
    description="Remove PII (emails, names, card numbers) from text.",
    system_prompt=(
        "You are a PII redaction specialist. Replace all personally "
        "identifiable information with placeholders: [NAME], [EMAIL], "
        "[CARD], [PHONE]. Return the cleaned text only."
    ),
    temperature=0.0,
)

# ── Custom tools (@tool) ────────────────────────────────────────


@tool(description="Compute statistics from classified feedback data.")
def compute_stats(classified_json: str, tool_context: ToolContext) -> str:
    """Parse classified feedback JSON and compute summary statistics."""
    try:
        items = json.loads(classified_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON input"

    total = len(items)
    categories: dict[str, int] = {}
    sentiments: dict[str, int] = {}
    priorities: dict[str, int] = {}

    for item in items:
        cat = item.get("category", "unknown")
        sent = item.get("sentiment", "unknown")
        pri = item.get("priority", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        sentiments[sent] = sentiments.get(sent, 0) + 1
        priorities[pri] = priorities.get(pri, 0) + 1

    stats = {
        "total_items": total,
        "by_category": categories,
        "by_sentiment": sentiments,
        "by_priority": priorities,
    }

    # Store in shared content for other agents to use
    tool_context.save_content("feedback_stats", stats, scope="shared")

    return json.dumps(stats, indent=2)


@tool(description="Save a report to the output file.")
def save_report(report_text: str, tool_context: ToolContext) -> str:
    """Save the final report text. Returns confirmation."""
    tool_context.save_content(
        "final_report",
        report_text,
        content_type="text/markdown",
        scope="shared",
    )
    return "Report saved successfully."


# ── Agent assembly ──────────────────────────────────────────────

def build_pipeline() -> SequentialAgent:
    """Build the full data processing pipeline.

    Returns:
        A SequentialAgent with 4 stages: extract → classify → analyse → redact.
    """
    extractor = Agent(
        name="extractor",
        description="Extracts structured data from raw feedback.",
        instruction=(
            "You receive raw customer feedback text. "
            "Use the extract_feedback tool to parse it into structured JSON. "
            "Return ONLY the JSON result from the tool."
        ),
        tools=[extract_tool],
        temperature=0.1,
        output_format="json",
    )

    classifier = Agent(
        name="classifier",
        description="Classifies feedback by category, sentiment, and priority.",
        instruction=(
            "You receive structured feedback JSON from the previous step. "
            "Use the classify_feedback tool to classify each item. "
            "Return ONLY the JSON result from the tool."
        ),
        tools=[classify_tool],
        temperature=0.1,
        output_format="json",
    )

    analyst = Agent(
        name="analyst",
        description="Analyses feedback patterns and writes a summary report.",
        instruction=(
            "You are a customer feedback analyst. You receive classified "
            "feedback data.\n\n"
            "Steps:\n"
            "1. Use compute_stats to get summary statistics.\n"
            "2. Write a concise Markdown report with:\n"
            "   - Executive summary (2-3 sentences)\n"
            "   - Key metrics table\n"
            "   - Top issues (critical/high priority)\n"
            "   - Positive highlights\n"
            "   - Recommendations (3 bullet points)\n"
            "3. Use save_report to save the final report.\n\n"
            "Return the report text."
        ),
        tools=[compute_stats, save_report],
        temperature=0.5,
    )

    guardrail = Agent(
        name="guardrail",
        description="Redacts PII from the final report.",
        instruction=(
            "You receive a report that may contain PII. "
            "Use the redact_pii tool to clean it. "
            "Return the redacted report."
        ),
        tools=[redact_tool],
        temperature=0.0,
    )

    return SequentialAgent(
        name="feedback_pipeline",
        description="Full pipeline: extract → classify → analyse → redact.",
        sub_agents=[extractor, classifier, analyst, guardrail],
    )


# ── Execution ───────────────────────────────────────────────────

def run_pipeline(
    input_data: str,
    *,
    trace: bool = False,
    output_file: str | None = None,
) -> str:
    """Execute the full pipeline on input data.

    Args:
        input_data: Raw feedback text.
        trace: Whether to enable tracing.
        output_file: Path to write the final report (optional).

    Returns:
        The final redacted report.
    """
    pipeline = build_pipeline()
    collector = TraceCollector() if trace else None

    runner = Runner(agent=pipeline, trace_collector=collector)
    result = runner.run(input_data)

    # Save output if requested
    if output_file:
        Path(output_file).write_text(result, encoding="utf-8")
        print(f"\nReport saved to: {output_file}")

    # Print trace summary if enabled
    if collector:
        print("\n" + "=" * 60)
        print("TRACE SUMMARY")
        print("=" * 60)
        collector.print_summary()

    return result


def run_pipeline_streamed(
    input_data: str,
    *,
    trace: bool = False,
) -> str:
    """Execute the pipeline with streaming events.

    Args:
        input_data: Raw feedback text.
        trace: Whether to enable tracing.

    Returns:
        The final report text.
    """
    pipeline = build_pipeline()
    collector = TraceCollector() if trace else None
    runner = Runner(agent=pipeline, trace_collector=collector)

    last_content = ""
    for event in runner.stream(input_data):
        prefix = f"[{event.author}]"
        etype = event.event_type.value
        snippet = event.content[:100] if event.content else ""
        print(f"  {prefix:<15} {etype:<16} {snippet}")
        if event.content:
            last_content = event.content

    if collector:
        print("\n" + "=" * 60)
        collector.print_summary()

    return last_content


# ── CLI ─────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the data pipeline example."""
    parser = argparse.ArgumentParser(
        description="Nono Data Pipeline — Feedback Analysis Example",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a text file with raw feedback. Uses built-in sample if omitted.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the final report.",
    )
    parser.add_argument(
        "--trace", "-t",
        action="store_true",
        help="Enable tracing and print summary.",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Stream events instead of batch execution.",
    )
    args = parser.parse_args()

    # Load input
    if args.input:
        input_data = Path(args.input).read_text(encoding="utf-8")
    else:
        input_data = SAMPLE_FEEDBACK
        print("Using built-in sample feedback data.\n")

    print("=" * 60)
    print("NONO DATA PIPELINE — Customer Feedback Analysis")
    print("=" * 60)
    print(f"Input: {len(input_data)} chars, "
          f"{input_data.count(chr(10)) + 1} lines")
    print(f"Trace: {'ON' if args.trace else 'OFF'}")
    print(f"Mode:  {'streaming' if args.stream else 'batch'}")
    print("=" * 60 + "\n")

    # Run
    if args.stream:
        result = run_pipeline_streamed(input_data, trace=args.trace)
    else:
        result = run_pipeline(
            input_data,
            trace=args.trace,
            output_file=args.output,
        )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
