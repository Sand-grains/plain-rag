"""unit: obs 逐阶段证据覆盖(F25)。

验证 compute_evidence_coverage: expected 空/全中/部分中, lost_stage 判定(recalled<expected→recall 优先,
after_rerank<recalled→rerank), candidates_after_rerank 空退 retrieved_chunks; finalize_traces 归因循环
并入 failure_attribution.evidence(eval 路径), agent 路径(None)不并入。不触真 benchmark/检索。
"""
import json

import pytest

from obs import trace as trace_module
from obs.evidence import compute_evidence_coverage
from obs.trace_lifecycle import finalize_traces
from obs.trace import RagTrace, clear_session_traces, request_id_var, trace_scope


@pytest.fixture(autouse=True)
def _clean_state():
    """每测试清空 session 收集器并复位 request_id, 防跨测试污染。"""
    clear_session_traces()
    request_id_var.set("-")
    yield
    clear_session_traces()


class _Item:
    """benchmark 条目鸭子类型: 只读 .query_id/.expected_parent_ids。"""

    def __init__(self, query_id, expected_parent_ids):
        self.query_id = query_id
        self.expected_parent_ids = expected_parent_ids


def _trace(query_id="Q1", recalled=None, after=None, retrieved=None):
    trace = RagTrace(query_id=query_id, query="q")
    trace.recalled_ids = recalled or []
    trace.candidates_after_rerank = after or []
    trace.retrieved_chunks = retrieved or []
    return trace


class TestComputeCoverage:
    def test_expected_empty_returns_none(self):
        assert compute_evidence_coverage(_trace(recalled=["r1"]), []) is None

    def test_full_coverage_no_loss(self):
        result = compute_evidence_coverage(_trace(recalled=["r1", "r2"], after=["r1", "r2"]), ["r1", "r2"])
        assert result == {"expected": 2, "recalled": 2, "after_rerank": 2, "lost_stage": None}

    def test_loss_in_recall_stage(self):
        result = compute_evidence_coverage(_trace(recalled=["r1"], after=["r1"]), ["r1", "r2"])
        assert result["lost_stage"] == "recall"
        assert result == {"expected": 2, "recalled": 1, "after_rerank": 1, "lost_stage": "recall"}

    def test_loss_in_rerank_stage(self):
        result = compute_evidence_coverage(_trace(recalled=["r1", "r2"], after=["r1"]), ["r1", "r2"])
        assert result["lost_stage"] == "rerank"
        assert result["after_rerank"] == 1

    def test_after_rerank_falls_back_to_retrieved_chunks(self):
        result = compute_evidence_coverage(_trace(recalled=["r1", "r2"], after=[], retrieved=["r1"]), ["r1", "r2"])
        assert result["after_rerank"] == 1
        assert result["lost_stage"] == "rerank"

    def test_recall_loss_takes_precedence_over_rerank(self):
        # recalled 也丢且 rerank 也丢: 先定位上游 recall(判序: 上游 > 下游)
        result = compute_evidence_coverage(_trace(recalled=["r1", "r2"], after=["r1"]), ["r1", "r2", "r3"])
        assert result == {"expected": 3, "recalled": 2, "after_rerank": 1, "lost_stage": "recall"}


class TestFinalizeMerge:
    def test_recall_absent_merges_recall_coverage(self, tmp_path):
        with trace_scope("Q1", "q1"):
            trace_module.trace_var.get().recalled_ids = ["r1"]
            trace_module.trace_var.get().candidates_after_rerank = ["r1"]
            trace_module.trace_var.get().retrieved_chunks = ["r1"]
        trace_path = finalize_traces([_Item("Q1", ["r1", "r2"])], [], out_dir=tmp_path)
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        attribution = payload["failure_attribution"]
        assert attribution["failure_type"] == "recall_absent"
        assert attribution["evidence"]["evidence_coverage"] == {
            "expected": 2, "recalled": 1, "after_rerank": 1, "lost_stage": "recall",
        }

    def test_rerank_drop_merges_rerank_coverage(self, tmp_path):
        with trace_scope("Q2", "q2"):
            trace_module.trace_var.get().recalled_ids = ["r1", "r2"]
            trace_module.trace_var.get().candidates_after_rerank = ["r1"]
            trace_module.trace_var.get().retrieved_chunks = ["r1"]
        trace_path = finalize_traces([_Item("Q2", ["r1", "r2"])], [], out_dir=tmp_path)
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        attribution = payload["failure_attribution"]
        assert attribution["failure_type"] == "rerank_drop"
        assert attribution["evidence"]["evidence_coverage"]["lost_stage"] == "rerank"

    def test_agent_path_no_attribution_no_coverage(self, tmp_path):
        with trace_scope("1", "你好", trace_type="agent"):
            trace_module.trace_var.get().recalled_ids = ["r1"]
        trace_path = finalize_traces(out_dir=tmp_path)  # benchmark_items=None → 跳过归因
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        assert "failure_attribution" not in payload

    def test_succeeded_query_no_attribution_no_coverage(self, tmp_path):
        # 全中无失败: classify_failure 返 None, 无 failure_attribution 可并入
        with trace_scope("Q3", "q3"):
            trace_module.trace_var.get().recalled_ids = ["r1", "r2"]
            trace_module.trace_var.get().candidates_after_rerank = ["r1", "r2"]
            trace_module.trace_var.get().retrieved_chunks = ["r1", "r2"]
        trace_path = finalize_traces([_Item("Q3", ["r1", "r2"])], [], out_dir=tmp_path)
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        assert "failure_attribution" not in payload
