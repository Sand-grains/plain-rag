"""unit：obs/metrics_sink.compare_runs 跨 run 对齐与显著性。

验证 signatures 交集配对/稳定子集、matched<6 或全不匹配退化、corpus 可对比性、
Wilcoxon 配对 + Holm 校正、delta-only 指标(aggregate 全量)、跨模式字段缺失不崩。
"""
import pytest

from obs import metrics_sink


def _pq(query_id, recall=0.5, mrr=0.5, **extra) -> dict:
    """构造单条 query 的纯指标记录(默认 retrieval 模式字段, 无 layer2)。"""
    row = {
        "recall_at_k": recall,
        "precision_at_k": 0.5,
        "hit_at_k": 1,
        "mrr": mrr,
        "map_at_k": 0.4,
        "ndcg_at_k": 0.6,
        "child_hit_at_k": 0,
        "child_recall_at_k": 0.0,
        "retrieve_ms": 100.0,
        "generate_ms": 0.0,
        "diagnosis": "accept",
    }
    row.update(extra)
    return row


def _make_run(run_id, per_query, signatures=None, corpus="corpus-x",
              summary=None, attribution=None) -> dict:
    """构造 run 记录; signatures 缺省按 expected=[query_id] 算(query 级签名与 query 一一对应)。"""
    return {
        "run_id": run_id,
        "corpus_signature": corpus,
        "test_mode": "retrieval",
        "signatures": signatures or {
            query_id: metrics_sink.query_signature(query_id, [query_id])
            for query_id in per_query
        },
        "per_query": per_query,
        "summary": summary or {"aggregate": {}, "cost": {}, "layer2": {}},
        "attribution": attribution or {},
    }


def _query_range(count, start=1):
    return {f"Q{index}": _pq(f"Q{index}") for index in range(start, start + count)}


class TestAlignment:
    def test_matching_subset_aligns(self):
        run_a = _make_run("a", _query_range(8))
        run_b = _make_run("b", _query_range(8))
        result = metrics_sink.compare_runs(run_a, run_b)
        assert result["matched"] == 8
        assert result["total_a"] == 8 and result["total_b"] == 8
        assert result["alignable"] is True
        assert result["degenerated"] is False
        assert result["comparable"] is True

    def test_reannotated_queries_excluded(self):
        per_query = _query_range(6)
        sig_a = {query_id: metrics_sink.query_signature(query_id, [query_id]) for query_id in per_query}
        sig_b = {query_id: metrics_sink.query_signature(query_id, [query_id, "NEW"]) for query_id in per_query}
        result = metrics_sink.compare_runs(
            _make_run("a", per_query, signatures=sig_a),
            _make_run("b", per_query, signatures=sig_b),
        )
        assert result["matched"] == 0  # 全部重标注, 签名全不匹配
        assert result["alignable"] is False
        assert result["degenerated"] is True

    def test_new_queries_do_not_degenerate(self):
        run_a = _make_run("a", _query_range(8))
        run_b = _make_run("b", _query_range(10))  # 追加 Q9/Q10, 旧 8 条标注不变
        result = metrics_sink.compare_runs(run_a, run_b)
        assert result["matched"] == 8  # 稳定子集照常配对, 不因整体 size 不同而退化
        assert result["total_a"] == 8 and result["total_b"] == 10
        assert result["degenerated"] is False

    def test_matched_below_six_degenerates(self):
        run_a = _make_run("a", _query_range(5))
        run_b = _make_run("b", _query_range(5))
        result = metrics_sink.compare_runs(run_a, run_b)
        assert result["matched"] == 5
        assert result["degenerated"] is True
        paired = [metric for metric in result["metrics"] if metric["mode"] == "paired"]
        assert paired  # 仍有均值/delta
        assert all(metric["p_value"] is None for metric in paired)  # 样本过少, 全部降纯 delta

    def test_no_match_degenerates(self):
        run_a = _make_run("a", _query_range(6))
        run_b = _make_run("b", _query_range(6))
        sig_a = {query_id: metrics_sink.query_signature(query_id, ["a"]) for query_id in run_a["per_query"]}
        sig_b = {query_id: metrics_sink.query_signature(query_id, ["b"]) for query_id in run_b["per_query"]}
        result = metrics_sink.compare_runs(
            _make_run("a", run_a["per_query"], signatures=sig_a),
            _make_run("b", run_b["per_query"], signatures=sig_b),
        )
        assert result["matched"] == 0
        assert result["degenerated"] is True


class TestComparability:
    def test_corpus_change_non_comparable(self):
        run_a = _make_run("a", _query_range(6), corpus="corpus-1")
        run_b = _make_run("b", _query_range(6), corpus="corpus-2")
        result = metrics_sink.compare_runs(run_a, run_b)
        assert result["comparable"] is False
        paired = [metric for metric in result["metrics"] if metric["mode"] == "paired"]
        assert paired  # 仍出均值/delta
        assert all(metric["p_value"] is None for metric in paired)  # 语料已变, 不报显著性

    def test_same_corpus_comparable(self):
        run_a = _make_run("a", _query_range(6))
        run_b = _make_run("b", _query_range(6))
        result = metrics_sink.compare_runs(run_a, run_b)
        assert result["comparable"] is True


class TestPairedStats:
    def test_zero_diff_p_is_one(self):
        run_a = _make_run("a", _query_range(6))
        run_b = _make_run("b", _query_range(6))  # 全部字段相同, 全 0 diff
        result = metrics_sink.compare_runs(run_a, run_b)
        recall_row = next(metric for metric in result["metrics"] if metric["name"] == "recall_at_k")
        assert recall_row["p_value"] == 1.0  # 无差异, 不崩
        assert recall_row["significant"] is False
        assert recall_row["n"] == 6
        assert recall_row["delta"] == 0.0

    def test_significant_improvement(self):
        run_a = _make_run("a", {f"Q{index}": _pq(f"Q{index}", recall=0.3, mrr=0.2)
                                for index in range(1, 11)})
        run_b = _make_run("b", {f"Q{index}": _pq(f"Q{index}", recall=0.9, mrr=0.8)
                                for index in range(1, 11)})
        result = metrics_sink.compare_runs(run_a, run_b)
        recall_row = next(metric for metric in result["metrics"] if metric["name"] == "recall_at_k")
        assert recall_row["delta"] > 0
        assert recall_row["n"] == 10
        assert recall_row["p_value"] < 0.05  # Holm 校正后仍显著
        assert recall_row["significant"] is True

    def test_cross_mode_layer2_missing_does_not_crash(self):
        # run_a 有 layer2(full 模式), run_b 无(retrieval 模式): 跨模式配对按字段存在性, 缺则自然缺行
        run_a = _make_run("a", {f"Q{index}": _pq(f"Q{index}", faithfulness=0.8, verdict="pass")
                                for index in range(1, 7)})
        run_b = _make_run("b", _query_range(6))
        result = metrics_sink.compare_runs(run_a, run_b)
        names = [metric["name"] for metric in result["metrics"]]
        assert "faithfulness" not in names  # 对方无字段, 不进配对, 不报 0 污染
        assert "recall_at_k" in names  # 公共字段照常对比


class TestDeltaOnly:
    def _full_summary(self):
        return {
            "aggregate": {"hit_at_k": 0.8, "child_hit_at_k": 0.5,
                          "diagnosis_distribution": {"accept": 8, "recall_miss": 0}},
            "cost": {"estimated_cost": 0.0012, "total_input_tokens": 100},
            "layer2": {"verdict_distribution": {"pass": 5, "fail": 1}},
        }

    def test_delta_metrics_take_full_aggregate(self):
        per_query = _query_range(6)
        run_a = _make_run("a", per_query, summary=self._full_summary(),
                          attribution={"Q1": {"failure_type": "rerank_drop", "evidence": {}}})
        run_b = _make_run("b", per_query, summary=self._full_summary())
        result = metrics_sink.compare_runs(run_a, run_b)
        hit_row = next(metric for metric in result["metrics"] if metric["name"] == "hit_at_k")
        assert hit_row["mode"] == "delta"
        assert hit_row["p_value"] is None
        assert hit_row["a"] == 0.8 and hit_row["b"] == 0.8 and hit_row["delta"] == 0.0

    def test_cost_verdict_diagnosis_attribution_metrics(self):
        run_a = _make_run("a", _query_range(6), summary=self._full_summary(),
                          attribution={"Q1": {"failure_type": "rerank_drop", "evidence": {}}})
        run_b = _make_run("b", _query_range(6), summary=self._full_summary(),
                          attribution={"Q1": {"failure_type": "rerank_drop", "evidence": {}},
                                       "Q2": {"failure_type": "empty_recall", "evidence": {}}})
        result = metrics_sink.compare_runs(run_a, run_b)
        names = {metric["name"]: metric for metric in result["metrics"]}
        assert names["cost.estimated_cost"]["delta"] == 0.0
        assert names["cost.total_input_tokens"]["delta"] == 0.0
        assert names["verdict.pass"]["delta"] == 0.0
        assert names["diagnosis.accept"]["delta"] == 0.0
        # 归因计数: 双 run 都有的 rerank_drop 照常对比; 单侧缺失的 empty_recall 不报(防 0 污染)
        assert names["attribution.rerank_drop"]["a"] == 1.0 and names["attribution.rerank_drop"]["b"] == 1.0
        assert "attribution.empty_recall" not in names

    def test_missing_field_not_reported(self):
        summary_a = {"aggregate": {"hit_at_k": 0.8}, "cost": {"estimated_cost": 0.001}}
        summary_b = {"aggregate": {"hit_at_k": 0.8}}  # 无 cost
        result = metrics_sink.compare_runs(
            _make_run("a", _query_range(6), summary=summary_a),
            _make_run("b", _query_range(6), summary=summary_b),
        )
        names = [metric["name"] for metric in result["metrics"]]
        assert "hit_at_k" in names
        assert "cost.estimated_cost" not in names  # 一侧缺失不报, 防 0 污染


class TestHolm:
    def test_holm_correction_formula(self):
        corrected = metrics_sink._holm_correction([0.01, 0.04, 0.9])
        # 升序 [0.01, 0.04, 0.9]: 0.01*3=0.03, 0.04*2=0.08, 0.9*1=0.9
        assert corrected == pytest.approx([0.03, 0.08, 0.9])

    def test_empty_holm_returns_empty(self):
        assert metrics_sink._holm_correction([]) == []


class TestStructure:
    def test_returns_full_schema(self):
        result = metrics_sink.compare_runs(_make_run("a", _query_range(6)),
                                           _make_run("b", _query_range(6)))
        assert set(result) == {"run_a", "run_b", "alignable", "comparable", "matched",
                               "total_a", "total_b", "degenerated", "metrics"}
        assert result["run_a"] == "a" and result["run_b"] == "b"
        assert all(metric["mode"] in ("paired", "delta") for metric in result["metrics"])
        for metric in result["metrics"]:
            assert set(metric) == {"name", "a", "b", "delta", "p_value", "significant", "mode", "n"}
