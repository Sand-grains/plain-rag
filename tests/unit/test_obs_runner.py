"""unit：eval/runner 指标库收尾接线(metrics_sink record_run)。

验证 _record_run_metrics 组装与落库、build_metrics_per_query 白名单瘦身(丢弃正文)、
_resolve_corpus_signature 复用/派生、_load_previous_run 改读指标库、落库失败 log-only 不打断。
fake 层由 conftest autouse 安装。
"""
import hashlib
import logging
from types import SimpleNamespace

import pytest

import config
import eval.runner as runner
from eval.core.benchmark import BenchmarkItem
from eval.serialization import build_metrics_per_query
from eval.core.llm_as_judge.judge import JudgeResult
from eval.core.retrieval.retrieval_layer import LayerOutput, RetrievalEvalResult
from obs import metrics_sink
from obs.monitor_metrics import MonitorMetrics


@pytest.fixture(autouse=True)
def _isolated_metrics_dir(tmp_path, monkeypatch):
    """每个测试独立的指标库目录, 防跨测试污染。"""
    monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setattr(config, "OBS_METRICS_MAX_RUNS", 500)


def _make_result(query_id="Q1") -> RetrievalEvalResult:
    return RetrievalEvalResult(
        query_id=query_id, query="问题", category="cat", difficulty="easy",
        recall_at_k=0.5, precision_at_k=0.4, hit_at_k=1, mrr=0.5,
        map_at_k=0.4, ndcg_at_k=0.6, diagnosis="accept",
        final_chunk_ids=["c1"], candidate_chunk_ids=["c1", "c2"],
        retrieved_files=["doc.md"], child_hit_at_k=0, child_recall_at_k=0.0,
        query_rewritten="问题", query_rewritten_flag="原", final_context_text="上下文正文",
    )


def _make_output(results) -> LayerOutput:
    return LayerOutput(
        results=results,
        aggregate={"num_queries": len(results), "recall_at_k": 0.5},
        by_category={},
        by_difficulty={},
    )


def _make_judge(query_id="Q1") -> JudgeResult:
    return JudgeResult(
        query_id=query_id, faithfulness=0.8, answer_relevancy=0.7,
        context_precision=0.6, context_recall=0.9, answer_correctness=0.75,
        verdict="pass",
    )


def _make_item(query_id="Q1") -> BenchmarkItem:
    return BenchmarkItem(
        query_id=query_id, query="问题", reference_facts="", source_doc="",
        category="cat", difficulty="easy", expected_parent_ids=["c1"],
        relevance={"c1": 3}, expected_files=[], expected_pages=[],
    )


class _WithSignature:
    _corpus_signature = "content-hash"


class _WithoutSignature:
    index_store = SimpleNamespace(chunk_ids={"c1", "c2"})


class TestBuildMetricsPerQuery:
    def test_whitelist_keeps_metrics_drops_body(self):
        output = _make_output([_make_result("Q1")])
        per_query = build_metrics_per_query(output, [_make_judge("Q1")])
        record = per_query["Q1"]
        assert record["recall_at_k"] == 0.5
        assert record["faithfulness"] == 0.8
        assert record["verdict"] == "pass"
        assert record["retrieve_ms"] is None
        assert record["diagnosis"] == "accept"
        # 正文/候选/文件/改写一律丢弃(回放归 trace.jsonl)
        for dropped in ("query", "final_context_text", "candidate_chunk_ids",
                        "retrieved_files", "query_rewritten"):
            assert dropped not in record

    def test_retrieval_mode_has_no_layer2(self):
        output = _make_output([_make_result("Q1")])
        per_query = build_metrics_per_query(output, [])
        record = per_query["Q1"]
        assert "faithfulness" not in record
        assert "verdict" not in record


class TestResolveCorpusSignature:
    def test_prefers_retriever_signature(self):
        assert runner._resolve_corpus_signature(_WithSignature()) == "content-hash"

    def test_fallback_from_chunk_ids(self):
        sig = runner._resolve_corpus_signature(_WithoutSignature())
        assert sig == hashlib.sha256("c1|c2".encode("utf-8")).hexdigest()[:12]


class TestRecordRunMetrics:
    def test_records_full_run_to_sink(self, tmp_path):
        metrics = MonitorMetrics()
        runner._record_run_metrics(
            run_id="2026-08-18_120000", benchmark_path="benchmark/x.json",
            retriever=_WithSignature(), items=[_make_item("Q1")],
            output=_make_output([_make_result("Q1")]), judge_results=[_make_judge("Q1")],
            metrics=metrics, attribution={"Q1": {"failure_type": "rerank_drop", "evidence": {}}},
            test_mode="full",
        )
        records = metrics_sink.load_all()
        assert len(records) == 1
        record = records[0]
        assert record["run_id"] == "2026-08-18_120000"
        assert record["test_mode"] == "full"
        assert record["benchmark"] == "benchmark/x.json"
        assert record["num_queries"] == 1
        assert record["corpus_signature"] == "content-hash"
        assert record["per_query"]["Q1"]["faithfulness"] == 0.8
        assert record["attribution"]["Q1"]["failure_type"] == "rerank_drop"
        assert record["signatures"]["Q1"] == metrics_sink.query_signature("Q1", ["c1"])
        assert "config" in record and record["config"]["top_k"] == config.TOP_K

    def test_records_retrieval_run_without_layer2(self):
        metrics = MonitorMetrics()
        runner._record_run_metrics(
            run_id="2026-08-18_120001", benchmark_path="benchmark/x.json",
            retriever=_WithSignature(), items=[_make_item("Q1")],
            output=_make_output([_make_result("Q1")]), judge_results=[], metrics=metrics,
            attribution={}, test_mode="retrieval",
        )
        record = metrics_sink.load_all()[0]
        assert record["test_mode"] == "retrieval"
        assert "faithfulness" not in record["per_query"]["Q1"]

    def test_failure_logs_warning_not_raise(self, monkeypatch, tmp_path, caplog):
        def _boom(**kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(metrics_sink, "record_run", _boom)
        metrics = MonitorMetrics()
        with caplog.at_level(logging.WARNING):
            runner._record_run_metrics(
                run_id="2026-08-18_120002", benchmark_path="benchmark/x.json",
                retriever=_WithSignature(), items=[_make_item("Q1")],
                output=_make_output([_make_result("Q1")]), judge_results=[], metrics=metrics,
                attribution={}, test_mode="retrieval",
            )
        assert "指标库落库失败" in caplog.text  # log-only, 不打断
        assert metrics_sink.load_all() == []


class TestLoadPreviousRun:
    def test_loads_most_recent_faithfulness_run(self):
        # 先写一个 retrieval 无 faithfulness, 再写一个 full 有 faithfulness
        metrics_sink.record_run(
            run_id="older", benchmark="b", config={}, corpus_signature="c", test_mode="retrieval",
            num_queries=1, expected_parent_ids_by_query={"Q1": ["c1"]},
            summary={}, per_query={"Q1": {"recall_at_k": 0.5, "diagnosis": "accept"}},
            attribution={}, timestamp="2026-08-18T10:00:00",
        )
        metrics_sink.record_run(
            run_id="newer", benchmark="b", config={}, corpus_signature="c", test_mode="full",
            num_queries=1, expected_parent_ids_by_query={"Q1": ["c1"]},
            summary={}, per_query={"Q1": {"recall_at_k": 0.5, "faithfulness": 0.8, "diagnosis": "accept"}},
            attribution={}, timestamp="2026-08-18T11:00:00",
        )
        previous = runner._load_previous_run()
        assert previous is not None
        assert previous["Q1"]["faithfulness"] == 0.8  # 取最近含 faithfulness 的 run

    def test_none_when_no_faithfulness(self):
        metrics_sink.record_run(
            run_id="r", benchmark="b", config={}, corpus_signature="c", test_mode="retrieval",
            num_queries=1, expected_parent_ids_by_query={"Q1": ["c1"]},
            summary={}, per_query={"Q1": {"recall_at_k": 0.5}}, attribution={},
        )
        assert runner._load_previous_run() is None

    def test_none_when_sink_empty(self):
        assert runner._load_previous_run() is None
