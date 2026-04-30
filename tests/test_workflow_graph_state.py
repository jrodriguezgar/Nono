"""Tests for graph state machine features: error recovery, retry, audit trail,
and state schema.

Run:
    uv run pytest tests/test_workflow_graph_state.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nono.workflows import (
    END,
    DEFAULT_STEP_RETRIES,
    Workflow,
    WorkflowError,
)
from nono.workflows.workflow import StateSchema, StateTransition


# ── Helpers ─────────────────────────────────────────────────────────────────


def _increment(state: dict) -> dict:
    return {"value": state.get("value", 0) + 1}


def _noop(state: dict) -> dict:
    return {}


def _fail_always(state: dict) -> dict:
    raise RuntimeError("boom")


# ──────────────────────────────────────────────────────────────────────────────
# Per-step error recovery (on_error routing)
# ──────────────────────────────────────────────────────────────────────────────


class TestOnErrorRecovery:
    """step(on_error=...) routes to a fallback step instead of raising."""

    def test_on_error_routes_to_fallback(self) -> None:
        flow = Workflow("recovery")
        flow.step("risky", _fail_always, on_error="fallback")
        flow.step("fallback", lambda s: {"recovered": True})
        result = flow.run(value=0)

        assert result["recovered"] is True
        assert result["__error__"]["step"] == "risky"
        assert result["__error__"]["type"] == "RuntimeError"
        assert result["__error__"]["message"] == "boom"

    def test_on_error_preserves_state(self) -> None:
        flow = Workflow("preserve")
        flow.step("init", lambda s: {"x": 42})
        flow.step("risky", _fail_always, on_error="fallback")
        flow.step("fallback", lambda s: {"y": s.get("x", 0) + 1})
        flow.connect("init", "risky")
        result = flow.run()

        assert result["x"] == 42
        assert result["y"] == 43

    def test_without_on_error_raises(self) -> None:
        flow = Workflow("no_recovery")
        flow.step("risky", _fail_always)

        with pytest.raises(RuntimeError, match="boom"):
            flow.run()

    def test_on_error_in_stream(self) -> None:
        flow = Workflow("stream_recovery")
        flow.step("risky", _fail_always, on_error="fallback")
        flow.step("fallback", lambda s: {"ok": True})
        steps = list(flow.stream())

        assert any(s[1].get("ok") for s in steps)


# ──────────────────────────────────────────────────────────────────────────────
# Per-step retry
# ──────────────────────────────────────────────────────────────────────────────


class TestStepRetry:
    """step(retry=N) retries the step N times before failing."""

    def test_retry_succeeds_on_later_attempt(self) -> None:
        call_count = {"n": 0}

        def flaky(state: dict) -> dict:
            call_count["n"] += 1

            if call_count["n"] < 3:
                raise RuntimeError("transient")

            return {"ok": True}

        flow = Workflow("retry")
        flow.step("flaky", flaky, retry=3)
        result = flow.run()

        assert result["ok"] is True
        assert call_count["n"] == 3

    def test_retry_exhausted_raises(self) -> None:
        flow = Workflow("retry_fail")
        flow.step("always_fail", _fail_always, retry=2)

        with pytest.raises(RuntimeError, match="boom"):
            flow.run()

    def test_retry_exhausted_with_on_error(self) -> None:
        flow = Workflow("retry_recovery")
        flow.step("always_fail", _fail_always, retry=2, on_error="fallback")
        flow.step("fallback", lambda s: {"recovered": True})
        result = flow.run()

        assert result["recovered"] is True

    def test_retry_records_retries_in_transition(self) -> None:
        call_count = {"n": 0}

        def flaky(state: dict) -> dict:
            call_count["n"] += 1

            if call_count["n"] < 2:
                raise RuntimeError("transient")

            return {"ok": True}

        flow = Workflow("retry_audit")
        flow.step("flaky", flaky, retry=3)
        flow.run()

        assert len(flow.transitions) == 1
        assert flow.transitions[0].retries == 1  # succeeded on 2nd attempt


# ──────────────────────────────────────────────────────────────────────────────
# State transition audit trail
# ──────────────────────────────────────────────────────────────────────────────


class TestAuditTrail:
    """Workflow.transitions records a StateTransition per step."""

    def test_audit_trail_recorded(self) -> None:
        flow = Workflow("audit")
        flow.step("a", lambda s: {"x": 1})
        flow.step("b", lambda s: {"y": 2})
        flow.connect("a", "b")
        flow.run()

        assert len(flow.transitions) == 2
        assert flow.transitions[0].step == "a"
        assert flow.transitions[1].step == "b"

    def test_keys_changed_tracked(self) -> None:
        flow = Workflow("keys")
        flow.step("init", lambda s: {"alpha": 1, "beta": 2})
        flow.run()

        t = flow.transitions[0]
        assert "alpha" in t.keys_changed
        assert "beta" in t.keys_changed

    def test_duration_positive(self) -> None:
        flow = Workflow("dur")
        flow.step("a", _noop)
        flow.run()

        assert flow.transitions[0].duration_ms >= 0

    def test_branch_taken_recorded(self) -> None:
        flow = Workflow("br")
        flow.step("check", lambda s: {"score": 0.9})
        flow.step("good", _noop)
        flow.step("bad", _noop)
        flow.branch("check", lambda s: "good" if s["score"] > 0.5 else "bad")
        flow.run()

        assert flow.transitions[0].branch_taken == "good"

    def test_error_recorded_in_transition(self) -> None:
        flow = Workflow("err_audit")
        flow.step("bad", _fail_always, on_error="fix")
        flow.step("fix", _noop)
        flow.run()

        assert flow.transitions[0].step == "bad"
        assert flow.transitions[0].error == "boom"

    def test_audit_trail_reset_between_runs(self) -> None:
        flow = Workflow("reset")
        flow.step("a", _noop)
        flow.run()

        assert len(flow.transitions) == 1

        flow.run()

        assert len(flow.transitions) == 1  # reset, not appended

    def test_transitions_immutable(self) -> None:
        flow = Workflow("immutable")
        flow.step("a", _noop)
        flow.run()

        t = flow.transitions[0]
        assert isinstance(t, StateTransition)
        assert isinstance(t.keys_changed, frozenset)


# ──────────────────────────────────────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────────────────────────────────────


class TestStateSchema:
    """StateSchema validates types and applies reducers."""

    def test_schema_validates_types(self) -> None:
        schema = StateSchema(fields={"count": int, "name": str})
        errors = schema.validate({"count": 42, "name": "ok"})

        assert errors == []

    def test_schema_reports_type_mismatch(self) -> None:
        schema = StateSchema(fields={"count": int})
        errors = schema.validate({"count": "not_a_number"})

        assert len(errors) == 1
        assert "count" in errors[0]

    def test_schema_ignores_missing_keys(self) -> None:
        schema = StateSchema(fields={"count": int, "name": str})
        errors = schema.validate({"count": 42})

        assert errors == []

    def test_schema_ignores_none_values(self) -> None:
        schema = StateSchema(fields={"count": int})
        errors = schema.validate({"count": None})

        assert errors == []

    def test_reducer_appends(self) -> None:
        schema = StateSchema(
            fields={"notes": list},
            reducers={"notes": lambda old, new: (old or []) + new},
        )
        state = {"notes": ["a"]}
        schema.apply_reducers(state, {"notes": ["b", "c"]})

        assert state["notes"] == ["a", "b", "c"]

    def test_reducer_not_applied_for_new_key(self) -> None:
        schema = StateSchema(
            fields={"notes": list},
            reducers={"notes": lambda old, new: old + new},
        )
        state = {}
        schema.apply_reducers(state, {"notes": ["first"]})

        assert state["notes"] == ["first"]

    def test_workflow_with_schema_validates(self) -> None:
        schema = StateSchema(fields={"value": int})
        flow = Workflow("typed", schema=schema)
        flow.step("init", lambda s: {"value": 42})
        result = flow.run()

        assert result["value"] == 42

    def test_workflow_schema_with_reducer(self) -> None:
        schema = StateSchema(
            fields={"items": list},
            reducers={"items": lambda old, new: (old or []) + new},
        )
        flow = Workflow("reduce", schema=schema)
        flow.step("a", lambda s: {"items": ["x"]})
        flow.step("b", lambda s: {"items": ["y"]})
        flow.connect("a", "b")
        result = flow.run()

        assert result["items"] == ["x", "y"]

    def test_schema_property(self) -> None:
        schema = StateSchema(fields={"x": int})
        flow = Workflow("prop", schema=schema)

        assert flow.schema is schema

    def test_no_schema_by_default(self) -> None:
        flow = Workflow("def")

        assert flow.schema is None


# ──────────────────────────────────────────────────────────────────────────────
# Integration: combined features
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """Combined scenarios using retry + on_error + audit + schema."""

    def test_retry_then_recovery_then_audit(self) -> None:
        call_count = {"n": 0}

        def always_fail(state: dict) -> dict:
            call_count["n"] += 1
            raise ValueError("fail")

        schema = StateSchema(fields={"recovered": bool})
        flow = Workflow("combo", schema=schema)
        flow.step("risky", always_fail, retry=1, on_error="fix")
        flow.step("fix", lambda s: {"recovered": True})
        result = flow.run()

        assert result["recovered"] is True
        assert call_count["n"] == 2  # 1 original + 1 retry

        assert len(flow.transitions) == 2
        assert flow.transitions[0].error is not None
        assert flow.transitions[0].retries == 1
        assert flow.transitions[1].error is None

    def test_branching_with_audit_trail(self) -> None:
        flow = Workflow("branch_audit")
        flow.step("score", lambda s: {"quality": 0.9})
        flow.step("accept", lambda s: {"result": "accepted"})
        flow.step("reject", lambda s: {"result": "rejected"})
        flow.branch("score", lambda s: "accept" if s["quality"] > 0.5 else "reject")
        result = flow.run()

        assert result["result"] == "accepted"
        assert flow.transitions[0].branch_taken == "accept"
