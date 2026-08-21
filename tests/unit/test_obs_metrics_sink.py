"""unit：obs/metrics_sink 跨 run 指标库。

验证 query_signature 确定性/顺序无关、record_run 落库往返 + 同秒后缀 + MAX_RUNS 截尾、
list_runs 新到旧、get_run 命中/未命中、load_all 坏行跳过。纯文件 I/O, 不触 eval/检索/LLM。
"""
import json

import pytest

import config
from obs import metrics_sink


@pytest.fixture(autouse=True)
def _isolated_metrics_dir(tmp_path, monkeypatch):
    """每个测试独立的指标库目录与默认上限, 防跨测试污染。"""
    monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
    monkeypatch.setattr(config, "OBS_METRICS_MAX_RUNS", 500)


def _record_args(**overrides) -> dict:
    """构造 record_run 的典型参数; overrides 可覆盖任意字段。"""
    args = {
        "run_id": "2026-08-18_120000",
        "benchmark": "benchmark/private_v6.json",
        "config": {"chunk_size": 300, "chunk_overlap": 50, "top_k": 5,
                   "model_id": "deepseek", "git_commit": "abc1234"},
        "corpus_signature": "corpus-sig-1",
        "test_mode": "retrieval",
        "num_queries": 2,
        "expected_parent_ids_by_query": {"Q1": ["c1", "c2"], "Q2": ["c3"]},
        "summary": {"aggregate": {"recall_at_k": 0.5}, "by_category": {}, "by_difficulty": {}},
        "per_query": {"Q1": {"recall_at_k": 0.5, "diagnosis": "accept"},
                      "Q2": {"recall_at_k": 1.0, "diagnosis": "accept"}},
        "attribution": {"Q1": {"failure_type": "rerank_drop", "evidence": {}}},
        "timestamp": "2026-08-18T12:00:00",
    }
    args.update(overrides)
    return args


class TestQuerySignature:
    def test_deterministic_and_order_independent(self):
        sig_a = metrics_sink.query_signature("Q1", ["c2", "c1"])
        sig_b = metrics_sink.query_signature("Q1", ["c1", "c2"])
        assert sig_a == sig_b  # 同一标注集合, 顺序无关
        assert len(sig_a) == 16  # 16 位 hex

    def test_different_expected_produces_different_signature(self):
        sig_a = metrics_sink.query_signature("Q1", ["c1"])
        sig_b = metrics_sink.query_signature("Q1", ["c1", "c2"])
        assert sig_a != sig_b

    def test_different_query_produces_different_signature(self):
        sig_a = metrics_sink.query_signature("Q1", ["c1"])
        sig_b = metrics_sink.query_signature("Q2", ["c1"])
        assert sig_a != sig_b


class TestRecordRun:
    def test_record_then_load_roundtrip(self):
        path = metrics_sink.record_run(**_record_args())
        assert path.exists()
        records = metrics_sink.load_all()
        assert len(records) == 1
        record = records[0]
        assert record["run_id"] == "2026-08-18_120000"
        assert record["test_mode"] == "retrieval"
        assert record["benchmark"] == "benchmark/private_v6.json"
        assert record["corpus_signature"] == "corpus-sig-1"
        assert record["num_queries"] == 2
        assert record["config"]["model_id"] == "deepseek"
        assert record["summary"]["aggregate"]["recall_at_k"] == 0.5
        assert record["per_query"]["Q1"]["diagnosis"] == "accept"
        assert record["attribution"]["Q1"]["failure_type"] == "rerank_drop"
        # signatures 由 expected_parent_ids 计算, 不是直传
        assert record["signatures"]["Q1"] == metrics_sink.query_signature("Q1", ["c1", "c2"])

    def test_creates_metrics_dir_automatically(self):
        path = metrics_sink.record_run(**_record_args())
        assert path.parent.exists()
        assert path.parent.name == "metrics"

    def test_same_second_run_id_gets_monotonic_suffix(self):
        metrics_sink.record_run(**_record_args())
        metrics_sink.record_run(**_record_args())
        run_ids = [record["run_id"] for record in metrics_sink.load_all()]
        assert run_ids == ["2026-08-18_120000", "2026-08-18_120000-1"]

    def test_truncates_beyond_max_runs(self, monkeypatch):
        monkeypatch.setattr(config, "OBS_METRICS_MAX_RUNS", 2)
        for run_id in ("r1", "r2", "r3"):
            metrics_sink.record_run(**_record_args(run_id=run_id))
        records = metrics_sink.load_all()
        assert [record["run_id"] for record in records] == ["r2", "r3"]  # 只留最近 2, 最早被截尾


class TestListAndGet:
    def test_list_runs_newest_first(self):
        metrics_sink.record_run(**_record_args(run_id="older"))
        metrics_sink.record_run(**_record_args(run_id="newer"))
        assert [record["run_id"] for record in metrics_sink.list_runs()] == ["newer", "older"]

    def test_get_run_hit_and_miss(self):
        metrics_sink.record_run(**_record_args(run_id="target"))
        assert metrics_sink.get_run("target") is not None
        assert metrics_sink.get_run("absent") is None

    def test_load_all_no_file_returns_empty(self):
        assert metrics_sink.load_all() == []
        assert metrics_sink.list_runs() == []
        assert metrics_sink.get_run("x") is None

    def test_load_all_skips_bad_lines(self):
        metrics_sink.record_run(**_record_args(run_id="good"))
        path = metrics_sink._metrics_path()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("{ not valid json }\n")
        records = metrics_sink.load_all()
        assert [record["run_id"] for record in records] == ["good"]  # 坏行跳过, 好行保留
