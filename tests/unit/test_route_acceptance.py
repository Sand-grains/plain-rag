"""unit: scripts/route_acceptance 路由验收(012-1 2.7)——四类决策映射/指标/报告/CLI。"""
import json
from pathlib import Path

import pytest

import scripts.route_acceptance as ra
from preprocess.format_precheck import DispatchDecision


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测路由验收逻辑, 无需重依赖。"""
    yield


class TestRouteLabel:
    def test_maps_text_decisions(self):
        assert ra._route_label(DispatchDecision.HTML_TEXT) == "text"
        assert ra._route_label(DispatchDecision.WHOLE_TEXT_PIPELINE) == "text"

    def test_maps_vlm_and_skip(self):
        assert ra._route_label(DispatchDecision.VLM_TEXT_PIPELINE) == "vlm"
        assert ra._route_label(DispatchDecision.SKIP_TEXT_PIPELINE) == "skip"

    def test_colpali_triggered_priority(self):
        # colpali_triggered 优先于文本/VLM 决策
        assert ra._route_label(DispatchDecision.HTML_TEXT, colpali_triggered=True) == "colpali"
        assert ra._route_label(DispatchDecision.VLM_TEXT_PIPELINE, colpali_triggered=True) == "colpali"


class TestMetrics:
    def test_perfect(self):
        m = ra._metrics(["text", "text"], ["text", "text"])
        assert m["accuracy"] == 1.0 and m["misroute_cost"] == 0.0
        assert m["per_class"]["text"]["precision"] == 1.0

    def test_fn_cost_higher_than_fp(self):
        # 1 个 text 被误判为 skip(FN) 代价 = w_fn=2; 1 个 skip 误判为 text(FP) 代价 = w_fp=1
        m = ra._metrics(["text", "skip"], ["skip", "text"])
        assert m["per_class"]["text"]["fn"] == 1
        assert m["per_class"]["skip"]["fp"] == 1
        assert m["misroute_cost"] == ra._W_FN + ra._W_FP

    def test_four_class_metrics(self):
        m = ra._metrics(["text", "vlm", "colpali", "skip"],
                        ["text", "vlm", "colpali", "skip"])
        assert m["accuracy"] == 1.0
        assert set(m["per_class"]) == {"text", "vlm", "colpali", "skip"}


class TestRouteAcceptance:
    def test_runs_with_mocked_precheck_and_gold(self, tmp_path, monkeypatch):
        (tmp_path / "a.html").write_text("<html>a</html>", encoding="utf-8")
        (tmp_path / "b.html").write_text("<html>b</html>", encoding="utf-8")
        (tmp_path / "a_files").mkdir()  # 资源子目录应被忽略

        class _Result:
            doc_decision = DispatchDecision.HTML_TEXT
            colpali_triggered = False
            degraded_flags = []
            sampling_format_stats = {"main_text_chars": 100, "image_count": 0}

        monkeypatch.setattr(ra, "precheck", lambda path: _Result())
        report = ra.route_acceptance(str(tmp_path), {"a.html": "text", "b.html": "text"})
        assert report["num_samples"] == 2
        assert report["gold_source"] == "provided"
        assert report["metrics"]["accuracy"] == 1.0

    def test_missing_gold_raises(self, tmp_path):
        (tmp_path / "a.html").write_text("<html>a</html>", encoding="utf-8")
        with pytest.raises(ValueError, match="需要人工标注 gold"):
            ra.route_acceptance(str(tmp_path), {})

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="无顶层 HTML"):
            ra.route_acceptance(str(tmp_path), {"a.html": "text"})


class TestCli:
    def test_main_writes_report(self, tmp_path, monkeypatch):
        (tmp_path / "a.html").write_text("<html>a</html>", encoding="utf-8")
        gold = tmp_path / "gold.json"
        gold.write_text(json.dumps({"a.html": "text"}), encoding="utf-8")
        monkeypatch.setattr(ra, "route_acceptance",
                            lambda sample_dir, gold: {
                                "sample_dir": sample_dir, "num_samples": 1,
                                "gold_source": "provided",
                                "metrics": {"accuracy": 1.0, "misroute_cost": 0.0,
                                            "per_class": {"text": {"precision": 1.0,
                                                                   "recall": 1.0, "f1": 1.0}}},
                                "per_sample": [],
                            })
        out = tmp_path / "route.json"
        assert ra.main(["--sample-dir", str(tmp_path), "--gold", str(gold), "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["num_samples"] == 1
