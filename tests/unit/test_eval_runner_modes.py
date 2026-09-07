"""unit：eval.runner 新模式 (precheck / no-report / smoke)。

收口流程第 1 步的验收：--precheck 前置自检四种分支、--mode full --smoke 冒烟门禁
（限条数 / error 退 3 / 不写 timeline）。test_eval_runner_modes.py 是 F08 门禁的一部分。

fake 层由 tests/unit/conftest.py autouse 安装; smoke 测试 patch _prepare_eval/_evaluate_one
隔离真实检索与 LLM, 只测驱动循环本身。
"""
import json
import logging

import pytest

import config
import eval.runner as runner
from eval.core.benchmark import BenchmarkItem
from eval.core.llm_as_judge.judge import JudgeResult
from obs import metrics_sink


def _benchmark(tmp_path, items) -> str:
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return str(path)


class _FakeStore:
    def __init__(self, chunk_ids=()):
        self._ids = set(chunk_ids)

    @property
    def chunk_ids(self) -> set[str]:
        return self._ids


def _make_item(query_id: str) -> BenchmarkItem:
    return BenchmarkItem(
        query_id=query_id, query="q", reference_facts="", source_doc="",
        category="", difficulty="", expected_parent_ids=[], relevance={},
        expected_files=[], expected_pages=[],
    )


def _patch_index(monkeypatch, store) -> None:
    class _Stub:
        @staticmethod
        def vector_restore():
            return store
    monkeypatch.setattr(runner, "IndexStore", _Stub)


class TestPrecheck:
    def test_index_missing_exits_nonzero(self, monkeypatch):
        _patch_index(monkeypatch, None)
        with pytest.raises(SystemExit) as exc:
            runner.run_precheck("benchmark/private_v6.json")
        assert exc.value.code == 1

    def test_annotation_invalid_exits_nonzero(self, monkeypatch, tmp_path):
        _patch_index(monkeypatch, _FakeStore(chunk_ids={"c1"}))
        bench = _benchmark(tmp_path, [{"query_id": "Q1", "query": "q",
                                       "expected_parent_ids": ["MISSING"]}])
        with pytest.raises(SystemExit) as exc:
            runner.run_precheck(bench)
        assert exc.value.code == 1

    def test_empty_items_exits_nonzero(self, monkeypatch, tmp_path):
        _patch_index(monkeypatch, _FakeStore(chunk_ids=set()))
        bench = _benchmark(tmp_path, [])
        with pytest.raises(SystemExit) as exc:
            runner.run_precheck(bench)
        assert exc.value.code == 1

    def test_ok_exits_zero(self, monkeypatch, tmp_path):
        _patch_index(monkeypatch, _FakeStore(chunk_ids={"c1"}))
        bench = _benchmark(tmp_path, [{"query_id": "Q1", "query": "q",
                                       "expected_parent_ids": ["c1"]}])
        runner.run_precheck(bench)  # 正常返回即退 0


class TestSmokeMode:
    def _patch_runner(self, monkeypatch, items, evaluate) -> None:
        monkeypatch.setattr(runner, "_prepare_eval", lambda benchmark_path: (object(), items))
        monkeypatch.setattr(runner, "_evaluate_one", evaluate)

    def test_runs_only_limit_items(self, monkeypatch):
        items = [_make_item(f"Q{i}") for i in range(3)]
        calls = []

        def evaluate(item, retriever, generator):
            calls.append(item.query_id)
            return JudgeResult(query_id=item.query_id, verdict="pass")

        self._patch_runner(monkeypatch, items, evaluate)
        runner.run_smoke_mode("dummy.json", limit=2)
        assert calls == ["Q0", "Q1"]

    def test_no_error_exits_zero(self, monkeypatch):
        items = [_make_item(f"Q{i}") for i in range(2)]

        def evaluate(item, retriever, generator):
            return JudgeResult(query_id=item.query_id, verdict="pass")

        self._patch_runner(monkeypatch, items, evaluate)
        runner.run_smoke_mode("dummy.json", limit=5)  # 正常返回即退 0

    def test_error_exits_three(self, monkeypatch):
        items = [_make_item(f"Q{i}") for i in range(2)]

        def evaluate(item, retriever, generator):
            verdict = "error" if item.query_id == "Q1" else "pass"
            return JudgeResult(query_id=item.query_id, verdict=verdict)

        self._patch_runner(monkeypatch, items, evaluate)
        with pytest.raises(SystemExit) as exc:
            runner.run_smoke_mode("dummy.json", limit=5)
        assert exc.value.code == 3

    def test_generator_error_counts_as_failure(self, monkeypatch):
        items = [_make_item("Q0")]

        def evaluate(item, retriever, generator):
            return JudgeResult(query_id=item.query_id, generator_error="boom")

        self._patch_runner(monkeypatch, items, evaluate)
        with pytest.raises(SystemExit) as exc:
            runner.run_smoke_mode("dummy.json", limit=5)
        assert exc.value.code == 3

    def test_smoke_writes_no_timeline(self, monkeypatch):
        items = [_make_item("Q0")]

        def evaluate(item, retriever, generator):
            return JudgeResult(query_id=item.query_id, verdict="pass")

        self._patch_runner(monkeypatch, items, evaluate)
        timeline = runner._PROJECT_ROOT / "eval" / "results" / "timeline"
        before = set(timeline.iterdir()) if timeline.exists() else set()
        runner.run_smoke_mode("dummy.json", limit=5)
        after = set(timeline.iterdir()) if timeline.exists() else set()
        assert before == after


class TestMultiSubset:
    """011-0: 多子集加载/分列统计 + 缺失即 skip。"""

    def test_prepare_subsets_skips_missing_crawler(self, monkeypatch, tmp_path):
        import config as config_mod
        _patch_index(monkeypatch, _FakeStore(chunk_ids={"c1"}))
        bench = _benchmark(tmp_path, [{"query_id": "Q1", "query": "q", "expected_parent_ids": ["c1"]}])
        # private_crawler.json 已存在, 用 tmp 下真正缺失的文件 + 注入 skip 集合验证"缺失即 skip"
        missing = str(tmp_path / "missing_crawler.json")
        monkeypatch.setattr(config_mod, "SKIP_IF_MISSING_BENCHMARKS", {missing})
        subsets = runner._prepare_subsets([bench, missing])
        assert len(subsets) == 2
        assert subsets[0].skipped is False
        assert subsets[0].kind == "private"
        assert subsets[1].skipped is True
        assert subsets[1].retriever is None

    def test_prepare_subsets_aborts_on_invalid(self, monkeypatch, tmp_path):
        _patch_index(monkeypatch, _FakeStore(chunk_ids={"c1"}))
        bench = _benchmark(tmp_path, [{"query_id": "Q1", "query": "q", "expected_parent_ids": ["MISSING"]}])
        with pytest.raises(SystemExit) as exc:
            runner._prepare_subsets([bench])
        assert exc.value.code == 1

    def test_retrieval_mode_runs_each_subset(self, monkeypatch, tmp_path):
        from eval.core.retrieval import retrieval_layer

        class _FakeOutput:
            def __init__(self, query_id):
                self.results = [type("R", (), {"query_id": query_id})()]
                self.aggregate = {"recall_at_k": 0.5, "hit_at_k": 1.0, "mrr": 0.5, "ndcg_at_k": 0.5}

        def fake_prepare(paths):
            return [
                type("S", (), {"path": "a.json", "kind": "private", "retriever": object(),
                               "items": [_make_item("Q0")], "skipped": False})(),
                type("S", (), {"path": "b.json", "kind": "public", "retriever": object(),
                               "items": [_make_item("Q1")], "skipped": False})(),
            ]

        calls = []
        recorded = []

        def fake_eval(retriever, items, per_query_ctx=None):
            calls.append(items[0].query_id)
            return _FakeOutput(items[0].query_id)

        def fake_record(**kwargs):
            recorded.append(kwargs["benchmark_path"])

        monkeypatch.setattr(runner, "_prepare_subsets", fake_prepare)
        monkeypatch.setattr(retrieval_layer, "run_retrieval_eval", fake_eval)
        monkeypatch.setattr(runner, "_record_run_metrics", fake_record)
        runner.run_retrieval_mode(["a.json", "b.json"], no_report=True)
        assert calls == ["Q0", "Q1"]
        assert recorded == ["a.json", "b.json"]


class TestRunCompare:
    """F20: run_compare 改读指标库(metrics_sink), 参数语义为 run_id。"""

    def _seed_runs(self, metrics_dir) -> None:
        per_query = {}
        for index in range(1, 7):
            per_query[f"Q{index}"] = {"recall_at_k": 0.5, "hit_at_k": 1, "diagnosis": "accept"}
        expected_by_query = {query_id: [query_id] for query_id in per_query}
        for run_id in ("run-a", "run-b"):
            metrics_sink.record_run(
                run_id=run_id, benchmark="b", config={}, corpus_signature="corpus-x",
                test_mode="retrieval", num_queries=6, expected_parent_ids_by_query=expected_by_query,
                summary={"aggregate": {"hit_at_k": 0.8, "diagnosis_distribution": {"accept": 6}}},
                per_query=per_query, attribution={},
            )

    def test_compare_renders_sections(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
        self._seed_runs(tmp_path / "metrics")
        with caplog.at_level(logging.INFO):
            runner.run_compare("run-a", "run-b")
        assert "对比 run-a vs run-b" in caplog.text
        assert "[Layer1]" in caplog.text
        assert "hit_at_k" in caplog.text

    def test_invalid_run_id_exits_nonzero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
        with pytest.raises(SystemExit) as exc:
            runner.run_compare("absent-a", "absent-b")
        assert exc.value.code == 1

    def test_legacy_run_reports_hint(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
        monkeypatch.setattr(runner, "_PROJECT_ROOT", tmp_path)  # 让 timeline 目录判定落在 tmp_path
        (tmp_path / "eval" / "results" / "timeline" / "old-run").mkdir(parents=True)
        with pytest.raises(SystemExit) as exc:
            runner.run_compare("old-run", "absent-run")
        assert exc.value.code == 1
        assert "legacy timeline run" in caplog.text
