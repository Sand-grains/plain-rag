"""unit：obs/failure_attribution 归因器五类判定 + finalize_traces 归因并入 trace.jsonl。

按序判定 rerank_fallback → empty_recall → recall_absent → rerank_drop → generation_error;
多 expected any-miss 语义, 池截断归 rerank_drop 并注明 pool_truncation, evidence 记 eval_diagnosis。
"""
import json

import pytest

from obs import trace as trace_module
from obs.trace_lifecycle import finalize_traces
from obs.failure_attribution import classify_failure
from obs.trace import RagTrace, clear_session_traces, trace_scope, trace_var


@pytest.fixture(autouse=True)
def _clean_session():
    clear_session_traces()
    yield
    clear_session_traces()


class _FakeJudge:
    def __init__(self, verdict="pass", judge_error=None, generator_error=None):
        self.verdict = verdict
        self.judge_error = judge_error
        self.generator_error = generator_error


def _trace(**overrides) -> RagTrace:
    """构造带默认检索现场(recalled r1/r2, retrieved r1)的 RagTrace, 供归因判定。"""
    trace = RagTrace(query_id="Q1", query="q")
    trace.recalled_ids = ["r1", "r2"]
    trace.retrieved_chunks = ["r1"]
    trace.candidates_before_rerank = ["r1", "r2"]
    trace.candidates_after_rerank = ["r1"]
    for key, value in overrides.items():
        setattr(trace, key, value)
    return trace


class TestClassifyFailure:
    def test_rerank_fallback_wins_over_other_judgements(self):
        result = classify_failure(_trace(fallback_required=True), ["missing"])
        assert result["failure_type"] == "rerank_fallback"

    def test_empty_recall(self):
        result = classify_failure(_trace(retrieved_chunks=[], recalled_ids=[]), ["r1"])
        assert result["failure_type"] == "empty_recall"
        assert result["evidence"]["retrieved_chunks_empty"] is True

    def test_empty_recall_priority_over_recall_absent(self):
        result = classify_failure(_trace(retrieved_chunks=[], recalled_ids=["r1"]), ["r1"])
        assert result["failure_type"] == "empty_recall"

    def test_recall_absent_missing_expected_with_hit_ratio(self):
        result = classify_failure(_trace(recalled_ids=["r1", "r2"]), ["r1", "missing"])
        assert result["failure_type"] == "recall_absent"
        assert result["evidence"]["missing_expected"] == ["missing"]
        assert result["evidence"]["expected_hit_ratio"] == 0.5

    def test_recall_absent_any_miss_semantics(self):
        result = classify_failure(_trace(recalled_ids=["r1", "r2"]), ["r1", "r2", "missing"])
        assert result["failure_type"] == "recall_absent"

    def test_rerank_drop_expected_not_in_after(self):
        result = classify_failure(_trace(), ["r2"])
        assert result["failure_type"] == "rerank_drop"
        assert result["evidence"]["dropped_by_rerank"] == ["r2"]
        assert "pool_truncation" not in result["evidence"]

    def test_rerank_drop_pool_truncation_evidence(self):
        trace = _trace(recalled_ids=["r1", "r2", "r3"],
                       candidates_before_rerank=["r1", "r2"], candidates_after_rerank=["r1"])
        result = classify_failure(trace, ["r3"])
        assert result["failure_type"] == "rerank_drop"
        assert result["evidence"]["pool_truncation"] == ["r3"]

    def test_generation_error_on_verdict_error(self):
        result = classify_failure(_trace(), ["r1"], _FakeJudge(verdict="error"))
        assert result["failure_type"] == "generation_error"

    def test_generation_error_on_judge_error(self):
        result = classify_failure(_trace(), ["r1"], _FakeJudge(verdict="fail", judge_error="Judge call failed"))
        assert result["failure_type"] == "generation_error"
        assert result["evidence"]["judge_error"] == "Judge call failed"

    def test_generation_error_on_generator_error(self):
        result = classify_failure(_trace(), ["r1"], _FakeJudge(generator_error="boom"))
        assert result["failure_type"] == "generation_error"
        assert result["evidence"]["generator_error"] == "boom"

    def test_no_failure_returns_none(self):
        assert classify_failure(_trace(), ["r1"], _FakeJudge(verdict="pass")) is None

    def test_none_trace_returns_none(self):
        assert classify_failure(None, ["r1"]) is None


class TestFinalizeTracesWithAttribution:
    def test_finalize_writes_attribution_with_eval_diagnosis(self, tmp_path):
        class _Item:
            query_id = "Q1"
            expected_parent_ids = ["r1", "absent"]

        class _Layer1:
            query_id = "Q1"
            diagnosis = "recall_miss"

        with trace_scope("Q1", "q"):
            current = trace_var.get()
            current.retrieved_chunks = ["r1"]
            current.recalled_ids = ["r1"]
        trace_path = finalize_traces([_Item()], [], layer1_results=[_Layer1()], out_dir=tmp_path)
        assert trace_path is not None
        line = json.loads(trace_path.read_text(encoding="utf-8").strip())
        attribution = line["failure_attribution"]
        assert attribution["failure_type"] == "recall_absent"
        assert attribution["evidence"]["eval_diagnosis"] == "recall_miss"

    def test_finalize_no_attribution_for_passed_trace(self, tmp_path):
        class _Item:
            query_id = "Q1"
            expected_parent_ids = ["r1"]

        with trace_scope("Q1", "q"):
            current = trace_var.get()
            current.retrieved_chunks = ["r1"]
            current.recalled_ids = ["r1"]
            current.candidates_after_rerank = ["r1"]
        trace_path = finalize_traces([_Item()], [], out_dir=tmp_path)
        line = json.loads(trace_path.read_text(encoding="utf-8").strip())
        assert "failure_attribution" not in line
