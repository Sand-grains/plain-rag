"""unit: scripts/parse_baseline 多格式解析质量基线(011-6)——指标聚合/报告落盘/CLI。"""
import json
from pathlib import Path

import pytest

import scripts.parse_baseline as pb


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测指标聚合/报告逻辑, 无需重依赖。"""
    yield


def _make_corpus(root: Path) -> None:
    """构造最小转制语料: pdf/ 与 html/ 各一篇 + gold。"""
    (root / "corpus" / "pdf").mkdir(parents=True)
    (root / "corpus" / "html").mkdir(parents=True)
    (root / "corpus" / "pdf" / "doc.pdf").write_text("pdf", encoding="utf-8")
    (root / "corpus" / "html" / "doc.html").write_text("html", encoding="utf-8")
    (root / "gold").mkdir(parents=True)
    (root / "gold" / "doc.md").write_text("# 标题\n正文", encoding="utf-8")


class TestComputeSampleMetrics:
    def test_identical(self):
        metrics = pb.compute_sample_metrics("# 标题\n正文", "# 标题\n正文")
        assert metrics["text_recall"] == 1.0
        assert metrics["text_precision"] == 1.0
        assert metrics["structure"]["overall"] == 1.0


class TestAggregateReport:
    def test_groups_by_format_and_morphology(self):
        per_sample = [
            {"format": "pdf", "morphology": "plain", "text_recall": 0.8, "text_precision": 0.9,
             "structure": {"overall": 0.7}},
            {"format": "pdf", "morphology": "plain", "text_recall": 0.6, "text_precision": 0.7,
             "structure": {"overall": 0.5}},
            {"format": "html", "morphology": "code", "text_recall": 1.0, "text_precision": 1.0,
             "structure": {"overall": 1.0}},
        ]
        report = pb.aggregate_report(per_sample)
        assert report["overall"]["num_samples"] == 3
        assert report["by_format"]["pdf"]["num_samples"] == 2
        assert report["by_format"]["html"]["num_samples"] == 1
        assert report["by_morphology"]["plain"]["num_samples"] == 2
        assert report["by_morphology"]["code"]["num_samples"] == 1


class TestRunBaseline:
    def test_runs_with_injected_parser(self, tmp_path):
        _make_corpus(tmp_path)
        report = pb.run_baseline(str(tmp_path / "corpus"), str(tmp_path / "gold"),
                                 parse_fn=lambda path: "# 标题\n正文")
        assert report["overall"]["num_samples"] == 2
        assert set(report["by_format"].keys()) == {"pdf", "html"}
        assert report["header"]["backend"] == "lightweight"

    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError):
            pb.run_baseline(str(tmp_path / "corpus"), str(tmp_path / "gold"), backend="bogus")

    def test_failing_parser_records_failure_without_crash(self, tmp_path):
        _make_corpus(tmp_path)

        def _boom(path):
            raise RuntimeError("后端失败")

        report = pb.run_baseline(str(tmp_path / "corpus"), str(tmp_path / "gold"),
                                 parse_fn=_boom)
        assert report["overall"]["num_samples"] == 0
        assert report["header"]["num_failed"] == 2
        assert report["header"]["success_rate"] == 0.0
        assert len(report["failures"]) == 2
        assert "后端失败" in report["failures"][0]["error"]


class TestMakeParseFn:
    def test_maps_heavy_chain_to_callable(self):
        fn = pb._make_parse_fn("heavy-chain")
        assert callable(fn)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            pb._make_parse_fn("bogus")


class TestMorphology:
    def test_classifies_gold(self):
        assert pb._morphology("| a |\n|---|") == "table"
        assert pb._morphology("```\nx\n```") == "code"
        assert pb._morphology("正文") == "plain"


class TestCli:
    def test_main_writes_report(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        monkeypatch.setattr(pb, "run_baseline",
                            lambda corpus, gold, backend="lightweight": {
                                "overall": {"num_samples": 2, "text_recall": 0.8,
                                            "text_precision": 0.9, "structure_overall": 0.7},
                                "by_format": {}, "by_morphology": {},
                                "header": {"backend": backend},
                            })
        out = tmp_path / "baseline.json"
        assert pb.main(["--corpus", str(tmp_path / "corpus"), "--gold", str(tmp_path / "gold"),
                        "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["overall"]["num_samples"] == 2
