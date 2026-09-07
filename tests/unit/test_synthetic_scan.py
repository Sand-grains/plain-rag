"""unit: scripts/synthetic_scan 合成扫描件 gold 集(012-1)——渲染/加噪/生成/CLI。"""
import json
from pathlib import Path

import pytest

import scripts.synthetic_scan as ss


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测扫描生成逻辑, 无需重依赖。"""
    yield


def _make_pdf(path: Path) -> None:
    """用 pymupdf 生成一页含文本的迷你 pdf。"""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello scan")
    doc.save(str(path))
    doc.close()


class TestRenderPdfPageToPng:
    def test_renders_png(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        _make_pdf(pdf)
        out = tmp_path / "doc.png"
        ss.render_pdf_page_to_png(str(pdf), str(out))
        assert out.exists() and out.stat().st_size > 0


class TestAddScanNoise:
    def test_outputs_png(self, tmp_path):
        from PIL import Image
        src = tmp_path / "src.png"
        Image.new("L", (40, 40), 200).save(src)
        out = tmp_path / "noisy.png"
        ss.add_scan_noise(str(src), str(out), noise=0.05, skew=1.0)
        assert out.exists() and out.stat().st_size > 0


class TestBuildScanGold:
    def test_generates_manifest_and_pngs(self, tmp_path, monkeypatch):
        gold = tmp_path / "gold"
        pdf = tmp_path / "pdf"
        out = tmp_path / "scan"
        gold.mkdir(); pdf.mkdir()
        (gold / "a.md").write_text("# A", encoding="utf-8")
        (gold / "b.md").write_text("# B", encoding="utf-8")
        (pdf / "a.pdf").write_bytes(b"pdf-a")
        (pdf / "b.pdf").write_bytes(b"pdf-b")
        (pdf / "orphan.pdf").write_bytes(b"orphan")  # 无对应 gold, 应被忽略

        monkeypatch.setattr(ss, "render_pdf_page_to_png", lambda *a, **k: None)
        monkeypatch.setattr(ss, "add_scan_noise", lambda *a, **k: None)
        manifest = ss.build_scan_gold(str(gold), str(pdf), str(out))
        assert manifest["num_samples"] == 2
        assert {s["stem"] for s in manifest["samples"]} == {"a", "b"}
        assert (out / "manifest.json").exists()

    def test_limit(self, tmp_path, monkeypatch):
        gold = tmp_path / "gold"
        pdf = tmp_path / "pdf"
        out = tmp_path / "scan"
        gold.mkdir(); pdf.mkdir()
        for i in range(3):
            (gold / f"d{i}.md").write_text("# x", encoding="utf-8")
            (pdf / f"d{i}.pdf").write_bytes(b"x")
        monkeypatch.setattr(ss, "render_pdf_page_to_png", lambda *a, **k: None)
        monkeypatch.setattr(ss, "add_scan_noise", lambda *a, **k: None)
        manifest = ss.build_scan_gold(str(gold), str(pdf), str(out), limit=2)
        assert manifest["num_samples"] == 2


class TestCli:
    def test_main_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ss, "build_scan_gold",
                            lambda *a, **k: {"num_samples": 2, "samples": [], "params": {}})
        assert ss.main(["--gold", str(tmp_path), "--pdf", str(tmp_path),
                        "--out", str(tmp_path / "scan")]) == 0
