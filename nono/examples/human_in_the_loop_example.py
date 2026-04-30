"""
Example: Human-in-the-Loop (HITL) — conditional intervention in loops.

Usage:
    python human_in_the_loop_example.py

Demonstrates:
1. Workflow with human_step() — basic approval gate
2. Workflow with human_step() and reject branch
3. LoopAgent with conditional human intervention (before_human callback)
4. Workflow human_step() inside a loop with conditional branching
5. Data injection — human modifies workflow state

All examples use a simulated handler (no real terminal input).
Replace the handler with a real one (console, API, WebSocket) for production.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nono.agent import (
    Runner,
    Session,
    Event,
    EventType,
    InvocationContext,
)
from nono.agent.base import BaseAgent
from nono.agent.human_input import HumanInputAgent
from nono.agent.workflow_agents import LoopAgent, SequentialAgent
from nono.hitl import HumanInputResponse, HumanRejectError
from nono.workflows import Workflow, END


# ── Simulated handlers ───────────────────────────────────────────────────────

def auto_approve_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Simulates a human that always approves."""
    print(f"    [HUMAN] {step_name}: {prompt} → Approved")
    return HumanInputResponse(approved=True, message="Looks good!")


def auto_reject_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Simulates a human that always rejects with feedback."""
    print(f"    [HUMAN] {step_name}: {prompt} → Rejected")
    return HumanInputResponse(approved=False, message="Needs more detail on methodology")


def quality_based_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Simulates a human that approves when quality is above 0.7."""
    quality = state.get("quality", 0)

    if quality >= 0.7:
        print(f"    [HUMAN] {step_name}: quality={quality:.1f} → Approved")
        return HumanInputResponse(approved=True, message="Quality acceptable")

    print(f"    [HUMAN] {step_name}: quality={quality:.1f} → Rejected, requesting improvements")
    return HumanInputResponse(
        approved=False,
        message="Please improve the draft, focus on clarity",
    )


def editor_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
    """Simulates an editor that injects corrections into the state."""
    print(f"    [HUMAN] {step_name}: {prompt} → Approved with edits")
    return HumanInputResponse(
        approved=True,
        message="Approved with corrections",
        data={"corrections": "Fix paragraph 2, add conclusion section"},
    )


# ── Minimal agent for examples ──────────────────────────────────────────────

class SimpleAgent(BaseAgent):
    """Minimal agent that returns a fixed response and optionally updates state."""

    def __init__(self, name: str, response: str, state_update: dict | None = None) -> None:
        super().__init__(name=name, description=f"Simple agent: {name}")
        self._response = response
        self._state_update = state_update or {}

    def _run_impl(self, ctx: InvocationContext):
        if self._state_update:
            ctx.session.state.update(self._state_update)
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)

    async def _run_async_impl(self, ctx: InvocationContext):
        if self._state_update:
            ctx.session.state.update(self._state_update)
        yield Event(EventType.AGENT_MESSAGE, self.name, self._response)


# ═════════════════════════════════════════════════════════════════════════════
# Example 1: Workflow — basic human approval gate
# ═════════════════════════════════════════════════════════════════════════════

def example_1_basic_approval():
    """Simple draft → human review → publish pipeline."""
    print(f"\n{'='*60}")
    print("  Example 1: Basic human approval gate (Workflow)")
    print(f"{'='*60}\n")

    flow = Workflow("review_pipeline")
    flow.step("draft", lambda s: {"draft": f"Article about {s['topic']}"})
    flow.human_step("review", handler=auto_approve_handler, prompt="Approve the draft?")
    flow.step("publish", lambda s: {"published": True, "final": s["draft"]})

    result = flow.run(topic="AI trends 2026")

    print(f"    Published: {result['published']}")
    print(f"    Human said: {result['human_input']['message']}")


# ═════════════════════════════════════════════════════════════════════════════
# Example 2: Workflow — reject with automatic redirect
# ═════════════════════════════════════════════════════════════════════════════

def example_2_reject_with_branch():
    """Human rejects → flow redirects to a revision step."""
    print(f"\n{'='*60}")
    print("  Example 2: Reject with branch (Workflow)")
    print(f"{'='*60}\n")

    flow = Workflow("review_pipeline")
    flow.step("draft", lambda s: {"draft": f"Article about {s['topic']}"})
    flow.human_step(
        "review",
        handler=auto_reject_handler,
        prompt="Approve the draft?",
        on_reject="revise",
    )
    flow.step("publish", lambda s: {"published": True})
    flow.step("revise", lambda s: {"revised": True, "revision_note": s["human_input"]["message"]})

    flow.connect("draft", "review")
    flow.connect("review", "publish")

    result = flow.run(topic="AI trends 2026")

    print(f"    Published: {result.get('published', False)}")
    print(f"    Revised: {result.get('revised', False)}")
    print(f"    Revision note: {result.get('revision_note', 'N/A')}")


# ═════════════════════════════════════════════════════════════════════════════
# Example 3: LoopAgent — conditional human intervention
# ═════════════════════════════════════════════════════════════════════════════

def example_3_conditional_loop():
    """Human only intervenes when quality is in the "uncertain zone" (0.5–0.8).

    - quality < 0.5  → auto-reject (no human needed)
    - quality 0.5–0.8 → ask human
    - quality > 0.8  → auto-approve (no human needed)
    """
    print(f"\n{'='*60}")
    print("  Example 3: Conditional HITL in LoopAgent")
    print(f"{'='*60}\n")

    iteration_count = 0

    def before_human_conditional(agent, ctx):
        """Skip human interaction based on quality score."""
        quality = ctx.session.state.get("quality", 0)

        if quality > 0.8:
            print(f"    [AUTO] quality={quality:.1f} → Auto-approved (above threshold)")
            return "Auto-approved: quality exceeds threshold"

        if quality < 0.5:
            print(f"    [AUTO] quality={quality:.1f} → Auto-rejected (below minimum)")
            ctx.session.state["human_rejected"] = True
            return "Auto-rejected: quality below minimum"

        # 0.5 ≤ quality ≤ 0.8 → ask human
        print(f"    [CONDITIONAL] quality={quality:.1f} → Asking human...")
        return None  # proceed to human handler

    writer = SimpleAgent("writer", "Draft v{n}", state_update={})
    reviewer = SimpleAgent("reviewer", "Reviewed", state_update={})

    # Quality increases each iteration to demonstrate progression
    qualities = [0.3, 0.6, 0.85]

    class QualityScorerAgent(BaseAgent):
        """Simulates a quality scorer that increases quality each iteration."""

        def __init__(self):
            super().__init__(name="scorer", description="Score draft quality")
            self._call = 0

        def _run_impl(self, ctx: InvocationContext):
            q = qualities[min(self._call, len(qualities) - 1)]
            self._call += 1
            ctx.session.state["quality"] = q
            print(f"    [SCORER] Iteration {self._call}: quality = {q:.1f}")
            yield Event(EventType.AGENT_MESSAGE, self.name, f"Quality: {q}")

        async def _run_async_impl(self, ctx: InvocationContext):
            for event in self._run_impl(ctx):
                yield event

    human_review = HumanInputAgent(
        name="human_review",
        handler=quality_based_handler,
        prompt="Review the current draft quality?",
        on_reject="continue",  # don't raise — let the loop continue
        before_human=before_human_conditional,
    )

    loop = LoopAgent(
        name="refinement_loop",
        sub_agents=[QualityScorerAgent(), human_review],
        max_iterations=3,
        stop_condition=lambda state: state.get("quality", 0) > 0.8,
    )

    session = Session()
    runner = Runner(loop, session=session)
    result = runner.run("Improve the article")

    final_q = session.state.get("quality", 0)
    print(f"\n    Final quality: {final_q:.1f}")
    print(f"    Stopped: {'quality > 0.8 (auto-approved)' if final_q > 0.8 else 'max iterations reached'}")


# ═════════════════════════════════════════════════════════════════════════════
# Example 4: Workflow loop with conditional human gate
# ═════════════════════════════════════════════════════════════════════════════

def example_4_workflow_loop_with_human():
    """Workflow that loops: write → score → conditional human review → branch.

    The human is only asked on every 2nd iteration (even iterations).
    """
    print(f"\n{'='*60}")
    print("  Example 4: Workflow loop with conditional human gate")
    print(f"{'='*60}\n")

    call_count = {"n": 0}

    def write(state: dict) -> dict:
        call_count["n"] += 1
        n = call_count["n"]
        quality = min(0.3 + n * 0.25, 0.95)
        print(f"    [WRITE] Iteration {n}: quality={quality:.2f}")
        return {"draft": f"Draft v{n}", "quality": quality, "iteration": n}

    def conditional_human_handler(step_name: str, state: dict, prompt: str) -> HumanInputResponse:
        """Only actually asks human on even iterations."""
        iteration = state.get("iteration", 1)
        quality = state.get("quality", 0)

        if iteration % 2 != 0:
            # Odd iterations: auto-approve, skip human
            print(f"    [HUMAN] Iteration {iteration}: SKIPPED (odd iteration)")
            return HumanInputResponse(approved=True, message="Auto-skip (odd iteration)")

        # Even iterations: human decides based on quality
        if quality >= 0.8:
            print(f"    [HUMAN] Iteration {iteration}: quality={quality:.2f} → Approved")
            return HumanInputResponse(approved=True, message="Approved")
        else:
            print(f"    [HUMAN] Iteration {iteration}: quality={quality:.2f} → Needs work")
            return HumanInputResponse(approved=False, message="Improve clarity")

    def publish(state: dict) -> dict:
        print(f"    [PUBLISH] Final draft: {state['draft']}")
        return {"published": True}

    def revise(state: dict) -> dict:
        feedback = state.get("human_input", {}).get("message", "")
        print(f"    [REVISE] Applying feedback: {feedback}")
        return {"revision_applied": True}

    flow = Workflow("iterative_review")
    flow.step("write", write)
    flow.human_step(
        "review",
        handler=conditional_human_handler,
        prompt="Review the draft?",
        on_reject="write",  # loop back to write on rejection
    )
    flow.step("publish", publish)

    flow.connect("write", "review")
    flow.connect("review", "publish")

    result = flow.run()

    print(f"\n    Published: {result.get('published', False)}")
    print(f"    Total iterations: {call_count['n']}")


# ═════════════════════════════════════════════════════════════════════════════
# Example 5: Human data injection — editor modifies state
# ═════════════════════════════════════════════════════════════════════════════

def example_5_data_injection():
    """Human injects corrections that the next step picks up."""
    print(f"\n{'='*60}")
    print("  Example 5: Human data injection (Workflow)")
    print(f"{'='*60}\n")

    flow = Workflow("editor_pipeline")
    flow.step("draft", lambda s: {"draft": f"Article about {s.get('topic', 'AI')}"})
    flow.human_step("editor_review", handler=editor_handler, prompt="Review and annotate?")
    flow.step("apply_edits", lambda s: {
        "final": f"{s['draft']} [edits: {s.get('corrections', 'none')}]",
    })

    result = flow.run(topic="Neural Networks")

    print(f"    Final: {result['final']}")
    print(f"    Corrections applied: {result.get('corrections', 'N/A')}")


# ═════════════════════════════════════════════════════════════════════════════
# Example 6: SequentialAgent pipeline with conditional HITL
# ═════════════════════════════════════════════════════════════════════════════

def example_6_sequential_conditional():
    """SequentialAgent: research → conditional human gate → writer.

    Human only intervenes if session state has `needs_review=True`.
    """
    print(f"\n{'='*60}")
    print("  Example 6: SequentialAgent with conditional HITL")
    print(f"{'='*60}\n")

    def before_human_check(agent, ctx):
        """Only ask human when needs_review is True."""
        needs_review = ctx.session.state.get("needs_review", False)

        if not needs_review:
            print("    [CONDITIONAL] needs_review=False → Skipping human")
            return "Auto-approved: no review needed"

        print("    [CONDITIONAL] needs_review=True → Asking human")
        return None

    research = SimpleAgent(
        "researcher",
        "Research complete",
        state_update={"notes": "AI research notes", "needs_review": True},
    )

    human_gate = HumanInputAgent(
        name="review_gate",
        handler=auto_approve_handler,
        prompt="Approve the research before writing?",
        before_human=before_human_check,
    )

    writer = SimpleAgent("writer", "Article written based on approved research")

    pipeline = SequentialAgent(
        name="pipeline",
        sub_agents=[research, human_gate, writer],
    )

    print("  --- Run 1: needs_review=True (human is asked) ---")
    session1 = Session()
    runner1 = Runner(pipeline, session=session1)
    runner1.run("Write about AI")

    hitl_requests = [e for e in session1.events if e.event_type == EventType.HUMAN_INPUT_REQUEST]
    hitl_responses = [e for e in session1.events if e.event_type == EventType.HUMAN_INPUT_RESPONSE]
    print(f"    HUMAN_INPUT_REQUEST events: {len(hitl_requests)}")
    print(f"    HUMAN_INPUT_RESPONSE events: {len(hitl_responses)}")

    print("\n  --- Run 2: needs_review=False (human is skipped) ---")

    research_no_review = SimpleAgent(
        "researcher",
        "Research complete",
        state_update={"notes": "Simple topic", "needs_review": False},
    )

    pipeline2 = SequentialAgent(
        name="pipeline",
        sub_agents=[research_no_review, human_gate, writer],
    )

    session2 = Session()
    runner2 = Runner(pipeline2, session=session2)
    runner2.run("Write a greeting")

    hitl_requests2 = [e for e in session2.events if e.event_type == EventType.HUMAN_INPUT_REQUEST]
    hitl_responses2 = [e for e in session2.events if e.event_type == EventType.HUMAN_INPUT_RESPONSE]
    print(f"    HUMAN_INPUT_REQUEST events: {len(hitl_requests2)} (skipped by before_human)")
    print(f"    HUMAN_INPUT_RESPONSE events: {len(hitl_responses2)}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*60}")
    print("  Human-in-the-Loop (HITL) Examples")
    print(f"{'#'*60}")

    example_1_basic_approval()
    example_2_reject_with_branch()
    example_3_conditional_loop()
    example_4_workflow_loop_with_human()
    example_5_data_injection()
    example_6_sequential_conditional()

    print(f"\n{'#'*60}")
    print("  All examples completed!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
