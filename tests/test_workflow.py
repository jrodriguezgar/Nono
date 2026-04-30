"""Tests for nono.workflows — Workflow pipeline, node factories, dynamic ops.

Covers:
- Core pipeline (step, connect, run, stream)
- Conditional branching (branch, branch_if)
- Dynamic manipulation (insert_before, insert_after, remove_step, etc.)
- tasker_node() factory
- agent_node() factory
- Unified step() proving that plain functions, tasker_node, and agent_node
  all share the same manipulation API.

Run:
    python tests/test_workflow.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — allow running as ``python tests/test_workflow.py``
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mock jinjapromptpy if not installed (prevents import error in genai_tasker)
# ---------------------------------------------------------------------------
if "jinjapromptpy" not in sys.modules:
    _jpp_mock = MagicMock()
    sys.modules["jinjapromptpy"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_generator"] = _jpp_mock
    sys.modules["jinjapromptpy.prompt_template"] = _jpp_mock
    sys.modules["jinjapromptpy.batch_generator"] = _jpp_mock

from nono.workflows import (
    END,
    DuplicateStepError,
    StepNotFoundError,
    Workflow,
    WorkflowError,
    agent_node,
    tasker_node,
)

# ── Pass/Fail counters (same pattern as other Nono test files) ───────────

_pass = 0
_fail = 0


def _ok(label: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def _ko(label: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    msg = f"  FAIL  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _assert(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        _ok(label)
    else:
        _ko(label, detail)


# ── Helpers ──────────────────────────────────────────────────────────────


def _upper(state: dict) -> dict:
    return {"text": state["text"].upper()}


def _exclaim(state: dict) -> dict:
    return {"text": state["text"] + "!"}


def _score(state: dict) -> dict:
    return {"score": len(state.get("text", "")) / 100}


# ── Core pipeline tests ─────────────────────────────────────────────────


def test_linear_pipeline() -> None:
    flow = Workflow("t1")
    flow.step("upper", _upper)
    flow.step("exclaim", _exclaim)
    result = flow.run(text="hello")
    _assert(result["text"] == "HELLO!", "linear_pipeline")


def test_explicit_edges() -> None:
    flow = Workflow("t2")
    flow.step("upper", _upper)
    flow.step("exclaim", _exclaim)
    flow.connect("upper", "exclaim")
    result = flow.run(text="hello")
    _assert(result["text"] == "HELLO!", "explicit_edges")


def test_branch_callable() -> None:
    flow = Workflow("t3")
    flow.step("score", _score)
    flow.step("good", lambda s: {"label": "good"})
    flow.step("bad", lambda s: {"label": "bad"})
    flow.branch("score", lambda s: "good" if s["score"] > 0.5 else "bad")
    result = flow.run(text="a" * 80)
    _assert(result["label"] == "good", "branch_callable")


def test_branch_if_predicate() -> None:
    flow = Workflow("t4")
    flow.step("score", _score)
    flow.step("pass_step", lambda s: {"result": "pass"})
    flow.step("fail_step", lambda s: {"result": "fail"})
    flow.branch_if(
        "score", lambda s: s["score"] > 0.5,
        then="pass_step", otherwise="fail_step",
    )
    result = flow.run(text="short")
    _assert(result["result"] == "fail", "branch_if_predicate")


def test_branch_if_high_score() -> None:
    flow = Workflow()
    flow.step("init", lambda s: {"level": 10})
    flow.step("high", lambda s: {"tag": "high"})
    flow.step("low", lambda s: {"tag": "low"})
    flow.branch_if("init", lambda s: s["level"] >= 5, then="high", otherwise="low")
    result = flow.run()
    _assert(result["tag"] == "high", "branch_if_high_score")


def test_stream() -> None:
    flow = Workflow("st")
    flow.step("upper", _upper)
    flow.step("exclaim", _exclaim)
    updates = list(flow.stream(text="hi"))
    _assert(len(updates) == 2, "stream_yields_two_updates")
    _assert(updates[0][0] == "upper", "stream_first_step_name")
    _assert(updates[1][1]["text"] == "HI!", "stream_final_text")


def test_async_run() -> None:
    async def async_upper(state: dict) -> dict:
        return {"text": state["text"].upper()}

    flow = Workflow("async")
    flow.step("upper", async_upper)
    flow.step("exclaim", _exclaim)
    result = asyncio.run(flow.run_async(text="yo"))
    _assert(result["text"] == "YO!", "async_run")


def test_duplicate_step_raises() -> None:
    flow = Workflow()
    flow.step("a", _upper)
    raised = False
    try:
        flow.step("a", _exclaim)
    except DuplicateStepError:
        raised = True
    _assert(raised, "duplicate_step_raises_DuplicateStepError")


def test_end_sentinel_stops() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"v": 1})
    flow.step("b", lambda s: {"v": 2})
    flow.connect("a", END)
    result = flow.run()
    _assert(result["v"] == 1, "end_sentinel_stops_at_first_step")


# ── Dynamic manipulation tests ───────────────────────────────────────────


def test_insert_before() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"log": ["a"]})
    flow.step("b", lambda s: {"log": s["log"] + ["b"]})
    flow.insert_before("b", "mid", lambda s: {"log": s["log"] + ["mid"]})
    result = flow.run()
    _assert(result["log"] == ["a", "mid", "b"], "insert_before")


def test_insert_after() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"log": ["a"]})
    flow.step("b", lambda s: {"log": s["log"] + ["b"]})
    flow.insert_after("a", "mid", lambda s: {"log": s["log"] + ["mid"]})
    result = flow.run()
    _assert(result["log"] == ["a", "mid", "b"], "insert_after")


def test_remove_step() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"v": 1})
    flow.step("b", lambda s: {"v": 2})
    flow.step("c", lambda s: {"v": 3})
    flow.remove_step("b")
    result = flow.run()
    _assert(result["v"] == 3, "remove_step")


def test_replace_step() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"v": "old"})
    flow.replace_step("a", lambda s: {"v": "new"})
    result = flow.run()
    _assert(result["v"] == "new", "replace_step")


def test_swap_steps() -> None:
    flow = Workflow()
    flow.step("a", lambda s: {"log": s.get("log", []) + ["a"]})
    flow.step("b", lambda s: {"log": s.get("log", []) + ["b"]})
    flow.swap_steps("a", "b")
    result = flow.run()
    _assert(result["log"] == ["b", "a"], "swap_steps")


# ── tasker_node() factory tests ──────────────────────────────────────────


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_inline(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "positive"
    mock_cls.return_value = mock_exec

    flow = Workflow("tn-inline")
    flow.step("classify", tasker_node(
        provider="google",
        model="gemini-3-flash-preview",
        system_prompt="Classify sentiment.",
        input_key="text",
        output_key="sentiment",
    ))
    result = flow.run(text="I love this!")
    _assert(result["sentiment"] == "positive", "tasker_node_inline")
    _assert(mock_exec.execute.called, "tasker_node_inline_calls_execute")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_json_task(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.run_json_task.return_value = {"category": "tech"}
    mock_cls.return_value = mock_exec

    flow = Workflow("tn-json")
    flow.step("categorize", tasker_node(
        task_file="tasks/categorize.json",
        input_key="article",
        output_key="category",
    ))
    result = flow.run(article="AI is transforming industry")
    _assert(result["category"] == {"category": "tech"}, "tasker_node_json_task")
    _assert(mock_exec.run_json_task.called, "tasker_node_json_task_calls_run_json_task")


def test_tasker_node_is_regular_callable() -> None:
    fn = tasker_node(system_prompt="test")
    _assert(callable(fn), "tasker_node_returns_callable")
    flow = Workflow()
    flow.step("step_a", fn)
    _assert("step_a" in flow.steps, "tasker_node_callable_registered_as_step")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_default_keys(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "result"
    mock_cls.return_value = mock_exec

    fn = tasker_node(system_prompt="Do stuff.")
    out = fn({"input": "hello"})
    _assert(out == {"output": "result"}, "tasker_node_default_keys")


# ── agent_node() factory tests ──────────────────────────────────────────


@patch("nono.agent.runner.Runner")
def test_agent_node_basic(mock_runner_cls: MagicMock) -> None:
    mock_runner = MagicMock()
    mock_runner.run.return_value = "Agent says hello"
    mock_runner_cls.return_value = mock_runner

    fake_agent = MagicMock()
    flow = Workflow("an-basic")
    flow.step("greet", agent_node(
        fake_agent, input_key="message", output_key="reply",
    ))
    result = flow.run(message="Hi there")
    _assert(result["reply"] == "Agent says hello", "agent_node_basic")
    _assert(mock_runner.run.called, "agent_node_basic_calls_run")


@patch("nono.agent.runner.Runner")
def test_agent_node_with_state_keys(mock_runner_cls: MagicMock) -> None:
    mock_runner = MagicMock()
    mock_runner.run.return_value = "summary"
    mock_runner_cls.return_value = mock_runner

    fake_agent = MagicMock()
    flow = Workflow("an-state")
    flow.step("summarize", agent_node(
        fake_agent,
        input_key="document",
        output_key="summary",
        state_keys={"context": "background"},
    ))
    result = flow.run(document="Long text...", background="tech paper")
    _assert(result["summary"] == "summary", "agent_node_with_state_keys")
    call_kwargs = mock_runner.run.call_args
    _assert("context" in call_kwargs.kwargs, "agent_node_state_keys_forwarded")


def test_agent_node_is_regular_callable() -> None:
    fake_agent = MagicMock()
    fn = agent_node(fake_agent)
    _assert(callable(fn), "agent_node_returns_callable")
    flow = Workflow()
    flow.step("agent_step", fn)
    _assert("agent_step" in flow.steps, "agent_node_callable_registered_as_step")


@patch("nono.agent.runner.Runner")
def test_agent_node_default_keys(mock_runner_cls: MagicMock) -> None:
    mock_runner = MagicMock()
    mock_runner.run.return_value = "result"
    mock_runner_cls.return_value = mock_runner

    fake_agent = MagicMock()
    fn = agent_node(fake_agent)
    out = fn({"input": "test"})
    _assert(out == {"output": "result"}, "agent_node_default_keys")


# ── Unified step() — all node types share manipulation ───────────────────


@patch("nono.agent.runner.Runner")
@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_mixed_pipeline(mock_task_cls: MagicMock, mock_runner_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "classified"
    mock_task_cls.return_value = mock_exec

    mock_runner = MagicMock()
    mock_runner.run.return_value = "polished"
    mock_runner_cls.return_value = mock_runner

    flow = Workflow("mixed")
    flow.step("preprocess", lambda s: {"text": s["raw"].strip()})
    flow.step("classify", tasker_node(
        system_prompt="Classify.", input_key="text", output_key="label",
    ))
    flow.step("polish", agent_node(
        MagicMock(), input_key="label", output_key="final",
    ))
    result = flow.run(raw="  hello  ")
    _assert(result["text"] == "hello", "mixed_pipeline_preprocess")
    _assert(result["label"] == "classified", "mixed_pipeline_tasker")
    _assert(result["final"] == "polished", "mixed_pipeline_agent")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_insert_before(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "enriched"
    mock_cls.return_value = mock_exec

    flow = Workflow("dyn")
    flow.step("a", lambda s: {"log": ["a"]})
    flow.step("b", lambda s: {"log": s["log"] + ["b"]})
    flow.insert_before("b", "enrich", tasker_node(
        system_prompt="Enrich.", input_key="log", output_key="enriched",
    ))
    _assert(flow.steps == ["a", "enrich", "b"], "tasker_node_insert_before_order")


@patch("nono.agent.runner.Runner")
def test_agent_node_replace_step(mock_runner_cls: MagicMock) -> None:
    mock_runner = MagicMock()
    mock_runner.run.return_value = "v2"
    mock_runner_cls.return_value = mock_runner

    flow = Workflow("replace")
    flow.step("process", lambda s: {"v": "old"})
    flow.replace_step("process", agent_node(
        MagicMock(), input_key="input", output_key="v",
    ))
    result = flow.run(input="go")
    _assert(result["v"] == "v2", "agent_node_replace_step")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_swap_steps(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "tasker_out"
    mock_cls.return_value = mock_exec

    flow = Workflow("swap")
    flow.step("plain", lambda s: {"log": s.get("log", []) + ["plain"]})
    flow.step("tasker", tasker_node(
        system_prompt="X", input_key="input", output_key="tasker_out",
    ))
    flow.swap_steps("plain", "tasker")
    _assert(flow.steps[0] == "tasker", "tasker_node_swap_first")
    _assert(flow.steps[1] == "plain", "tasker_node_swap_second")


@patch("nono.agent.runner.Runner")
def test_agent_node_insert_after(mock_runner_cls: MagicMock) -> None:
    mock_runner = MagicMock()
    mock_runner.run.return_value = "refined"
    mock_runner_cls.return_value = mock_runner

    flow = Workflow("ins-after")
    flow.step("a", lambda s: {"log": ["a"]})
    flow.step("c", lambda s: {"log": s["log"] + ["c"]})
    flow.insert_after("a", "refine", agent_node(
        MagicMock(), input_key="log", output_key="refined",
    ))
    _assert(flow.steps == ["a", "refine", "c"], "agent_node_insert_after_order")


@patch("nono.tasker.genai_tasker.TaskExecutor")
def test_tasker_node_remove_step(mock_cls: MagicMock) -> None:
    mock_exec = MagicMock()
    mock_exec.execute.return_value = "x"
    mock_cls.return_value = mock_exec

    flow = Workflow("rm")
    flow.step("a", lambda s: {"v": 1})
    flow.step("tasker", tasker_node(system_prompt="X"))
    flow.step("c", lambda s: {"v": 3})
    flow.remove_step("tasker")
    _assert("tasker" not in flow.steps, "tasker_node_remove_step")
    _assert(flow.steps == ["a", "c"], "tasker_node_remove_step_order")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Workflow pipeline + node factory tests")
    print("=" * 60)
    print()

    tests = [
        # Core pipeline
        test_linear_pipeline,
        test_explicit_edges,
        test_branch_callable,
        test_branch_if_predicate,
        test_branch_if_high_score,
        test_stream,
        test_async_run,
        test_duplicate_step_raises,
        test_end_sentinel_stops,
        # Dynamic manipulation
        test_insert_before,
        test_insert_after,
        test_remove_step,
        test_replace_step,
        test_swap_steps,
        # tasker_node factory
        test_tasker_node_inline,
        test_tasker_node_json_task,
        test_tasker_node_is_regular_callable,
        test_tasker_node_default_keys,
        # agent_node factory
        test_agent_node_basic,
        test_agent_node_with_state_keys,
        test_agent_node_is_regular_callable,
        test_agent_node_default_keys,
        # Unified step — all types share manipulation
        test_mixed_pipeline,
        test_tasker_node_insert_before,
        test_agent_node_replace_step,
        test_tasker_node_swap_steps,
        test_agent_node_insert_after,
        test_tasker_node_remove_step,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            _ko(test_fn.__name__, str(exc))

    print()
    total = _pass + _fail
    print(f"Results: {_pass}/{total} passed, {_fail} failed")
    if _fail:
        sys.exit(1)
    print("All tests passed!")
