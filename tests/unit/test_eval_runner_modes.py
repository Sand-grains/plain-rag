"""unit：eval.runner 新模式 (precheck / no-report / smoke)。

收口流程第 1 步的验收：--precheck 前置自检四种分支、--mode full --smoke 冒烟门禁
（限条数 / error 退 3 / 不写 timeline）。test_eval_runner_modes.py 是 F08 门禁的一部分。

fake 层由 tests/unit/conftest.py autouse 安装; smoke 测试 patch _prepare_eval/_evaluate_one
隔离真实检索与 LLM, 只测驱动循环本身。
"""
import json

import pytest

import eval.runner as runner
from eval.core.benchmark import BenchmarkItem
from eval.core.llm_as_judge.judge import JudgeResult


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
