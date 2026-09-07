"""unit: scripts/parse_diff 重型链差分报告(012-1)——版本头/聚合/阈值判定/CLI。"""
import json
from pathlib import Path

import pytest

import scripts.parse_diff as pd


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测差分报告逻辑, 无需重依赖。"""
    yield


def _make_corpus(root: Path) -> None:
    (root / "corpus" / "html").mkdir(parents=True)
    (root / "corpus" / "html" / "doc.html").write_text("<html>正文</html>", encoding="utf-8")
    (root / "gold").mkdir(parents=True)
    (root / "gold" / "doc.md").write_text("# 标题\n正文", encoding="utf-8")
    (root / "scan").mkdir(parents=True)
    (root / "scan" / "doc.png").write_bytes(b"png")
    (root / "scan" / "manifest.json").write_text(
        json.dumps({"samples": [{"stem": "doc", "scan": "doc.png", "gold": "doc.md"}]}),
        encoding="utf-8")
    (root / "baseline.json").write_text(
        json.dumps({"overall": {"text_recall": 0.9, "text_precision": 0.9,
                                "structure_overall": 0.8, "num_samples": 1}}),
        encoding="utf-8")


class TestBackendVersions:
    def test_returns_all_keys(self):
        versions = pd._backend_versions()
        assert {"docling", "mineru", "markitdown", "pymupdf", "trafilatura"} <= set(versions)


class TestCorpusSha:
    def test_reads_version(self, tmp_path):
        m = tmp_path / "corpus_manifest.json"
        m.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
        assert pd._corpus_sha(str(m)) == "v1"

    def test_unknown_when_missing(self, tmp_path):
        assert pd._corpus_sha(str(tmp_path / "nope.json")) == "unknown"


class TestRunCorpusPerSample:
    def test_returns_metrics(self, tmp_path):
        _make_corpus(tmp_path)
        rows = pd._run_corpus_per_sample(str(tmp_path / "corpus"), str(tmp_path / "gold"),
                                         lambda path: "# 标题\n正文")
        assert len(rows) == 1
        assert rows[0]["format"] == "html"
        assert rows[0]["text_recall"] == 1.0

    def test_records_failure(self, tmp_path):
        _make_corpus(tmp_path)
        rows = pd._run_corpus_per_sample(str(tmp_path / "corpus"), str(tmp_path / "gold"),
                                         lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
        assert rows[0]["status"] == "failed"


class TestRunScanPerSample:
    def test_returns_metrics(self, tmp_path):
        _make_corpus(tmp_path)
        rows = pd._run_scan_per_sample(str(tmp_path / "scan"), str(tmp_path / "gold"),
                                       lambda path: "# 标题\n正文")
        assert len(rows) == 1
        assert rows[0]["format"] == "scan"


class TestWilcoxon:
    def test_none_when_insufficient(self):
        assert pd._wilcoxon_p([0.5, 0.6], [0.5, 0.6]) is None

    def test_returns_p_when_enough(self):
        heavy = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        light = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        p = pd._wilcoxon_p(heavy, light)
        assert p is not None and p < 0.05


class TestRunDiff:
    def test_report_structure_and_verdict(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        import scripts.parse_baseline as pb
        monkeypatch.setattr(pb, "_lightweight_parse", lambda path: "# 标题\n正文")
        report = pd.run_diff(str(tmp_path / "corpus"), str(tmp_path / "gold"),
                             str(tmp_path / "scan"), str(tmp_path / "baseline.json"),
                             parse_fn=lambda path: "# 标题\n正文")
        assert report["header"]["backend"] == "heavy-chain"
        assert "versions" in report["header"]
        assert "thresholds" in report["header"]
        assert report["judgement"]["verdict"] in ("pass", "pause")
        assert report["heavy"]["scan"]["overall"]["num_samples"] == 1


class TestCli:
    def test_main_writes_report(self, tmp_path, monkeypatch):
        _make_corpus(tmp_path)
        monkeypatch.setattr(pd, "run_diff", lambda *a, **k: {
            "header": {"backend": "heavy-chain"},
            "judgement": {"verdict": "pass", "main_layer_recall_drop_pt": 0.0,
                          "docling_success_rate": 1.0, "markitdown_ratio": 0.0},
        })
        out = tmp_path / "diff.json"
        assert pd.main(["--corpus", str(tmp_path / "corpus"), "--gold", str(tmp_path / "gold"),
                        "--scan", str(tmp_path / "scan"), "--baseline", str(tmp_path / "baseline.json"),
                        "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["judgement"]["verdict"] == "pass"
