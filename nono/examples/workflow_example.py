"""
Example: Workflow - Multi-step AI pipeline with conditional branching.
Usage: python workflow_example.py

This example demonstrates:
1. Creating a multi-step workflow with the fluent API
2. Connecting steps with explicit edges
3. Conditional branching based on state values
4. Dynamic step manipulation (insert, remove, swap)
5. Streaming execution with per-step state snapshots
6. Integration with Nono's AI connector (optional)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.workflows import Workflow, END


# ── Example 1: Simple linear pipeline ────────────────────────────────────────

def example_linear_pipeline():
    """Basic three-step pipeline without AI calls."""
    print(f"\n{'='*60}")
    print("  Example 1: Linear Pipeline")
    print(f"{'='*60}")

    def gather(state: dict) -> dict:
        topic = state.get("topic", "AI")
        return {"notes": f"Research notes about {topic}."}

    def draft(state: dict) -> dict:
        return {"draft": f"Article based on: {state['notes']}"}

    def polish(state: dict) -> dict:
        return {"final": state["draft"].replace("based on", "crafted from")}

    flow = Workflow("article_pipeline")
    flow.step("gather", gather)
    flow.step("draft", draft)
    flow.step("polish", polish)
    flow.connect("gather", "draft", "polish")

    result = flow.run(topic="Generative AI in 2026")

    print(f"\n📋 Steps: {flow.steps}")
    print(f"📝 Final: {result['final']}")
    print(f"✅ Done.")


# ── Example 2: Conditional branching ─────────────────────────────────────────

def example_branching():
    """Workflow with conditional routing based on a quality score."""
    print(f"\n{'='*60}")
    print("  Example 2: Conditional Branching")
    print(f"{'='*60}")

    def analyze(state: dict) -> dict:
        text = state.get("text", "")
        score = len(text) / 100  # simple heuristic
        return {"score": min(score, 1.0)}

    def approve(state: dict) -> dict:
        return {"decision": "approved", "reason": "Quality score is high."}

    def revise(state: dict) -> dict:
        return {"decision": "needs_revision", "reason": "Quality score is low."}

    flow = Workflow("review_pipeline")
    flow.step("analyze", analyze)
    flow.step("approve", approve)
    flow.step("revise", revise)

    # Route based on score
    flow.branch_if("analyze", "score > 0.5", then="approve", otherwise="revise")

    # Test with short text (score < 0.5 -> revise)
    result = flow.run(text="Short.")
    print(f"\n📊 Score: {result['score']:.2f} -> {result['decision']}")

    # Test with long text (score > 0.5 -> approve)
    result = flow.run(text="A" * 80)
    print(f"📊 Score: {result['score']:.2f} -> {result['decision']}")


# ── Example 3: Dynamic step manipulation ─────────────────────────────────────

def example_dynamic_manipulation():
    """Insert, remove, and swap steps at runtime."""
    print(f"\n{'='*60}")
    print("  Example 3: Dynamic Step Manipulation")
    print(f"{'='*60}")

    flow = Workflow("dynamic")
    flow.step("step_a", lambda s: {"log": s.get("log", []) + ["A"]})
    flow.step("step_b", lambda s: {"log": s.get("log", []) + ["B"]})
    flow.step("step_c", lambda s: {"log": s.get("log", []) + ["C"]})

    print(f"  Initial:        {flow.steps}")

    flow.insert_before("step_b", "step_x", lambda s: {"log": s.get("log", []) + ["X"]})
    print(f"  After insert_before(B, X): {flow.steps}")

    flow.swap_steps("step_a", "step_c")
    print(f"  After swap(A, C):          {flow.steps}")

    flow.remove_step("step_x")
    print(f"  After remove(X):           {flow.steps}")

    result = flow.run()
    print(f"  Execution log:             {result['log']}")


# ── Example 4: Streaming execution ───────────────────────────────────────────

def example_streaming():
    """Stream state updates as each step completes."""
    print(f"\n{'='*60}")
    print("  Example 4: Streaming Execution")
    print(f"{'='*60}")

    flow = Workflow("streamed")
    flow.step("init", lambda s: {"counter": 0})
    flow.step("increment", lambda s: {"counter": s["counter"] + 1})
    flow.step("double", lambda s: {"counter": s["counter"] * 2})
    flow.connect("init", "increment", "double")

    for step_name, snapshot in flow.stream():
        print(f"  [{step_name}] counter = {snapshot['counter']}")


# ── Example 5: AI-powered workflow (requires connector) ──────────────────────

def example_ai_workflow():
    """Multi-step AI pipeline using Nono's connector.

    Requires a valid API key in config.toml or environment.
    """
    print(f"\n{'='*60}")
    print("  Example 5: AI-Powered Workflow (optional)")
    print(f"{'='*60}")

    try:
        from nono.tasker import TaskExecutor
    except ImportError:
        print("  [skipped] nono.tasker not available.")
        return

    PROVIDER = "google"
    MODEL = "gemini-3-flash-preview"

    def summarize(state: dict) -> dict:
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": f"Summarize in one sentence: {state['text']}"},
        ]
        summary = executor.execute_prompt(messages)
        return {"summary": summary}

    def classify(state: dict) -> dict:
        executor = TaskExecutor(provider=PROVIDER, model=MODEL)
        messages = [
            {"role": "system", "content": "Classify the topic as: tech, science, or other."},
            {"role": "user", "content": state["summary"]},
        ]
        category = executor.execute_prompt(messages)
        return {"category": category}

    flow = Workflow("ai_pipeline")
    flow.step("summarize", summarize)
    flow.step("classify", classify)
    flow.connect("summarize", "classify")

    if input("\n  Execute with AI? (y/n): ").lower() != "y":
        print("  Skipped.")
        return

    result = flow.run(
        text="Large Language Models are transforming how developers write code, "
             "enabling AI-assisted programming tools that suggest completions, "
             "refactor code, and generate tests automatically."
    )
    print(f"\n  Summary:  {result.get('summary', 'N/A')}")
    print(f"  Category: {result.get('category', 'N/A')}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_linear_pipeline()
    example_branching()
    example_dynamic_manipulation()
    example_streaming()
    example_ai_workflow()

    print(f"\n{'='*60}")
    print("  All examples completed.")
    print(f"{'='*60}")
