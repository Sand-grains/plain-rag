"""unit: scripts/corpus_convert 多格式解析语料转制(011-4)——内容形态分类/分层样本/gold/pptx/快照/pandoc 检测。"""
import json
from pathlib import Path

import pytest

import scripts.corpus_convert as cc


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测文件/转制逻辑, 无需重依赖(避免 ~18s 导入成本)。"""
    yield


def _make_data(root: Path) -> None:
    """构造最小 data/: 纯文本/表格/代码/混合 各一篇(放在 data/ 子目录, 与真实布局一致)。"""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "plain.md").write_text("# 纯文本\n正文", encoding="utf-8")
    (data / "table.md").write_text("# 表格\n| a | b |\n|---|---|\n| 1 | 2 |", encoding="utf-8")
    (data / "code.md").write_text("# 代码\n```python\nprint(1)\n```", encoding="utf-8")
    (data / "mixed.md").write_text("# 混合\n| a |\n|---|\n| 1 |\n```\nx\n```", encoding="utf-8")


class TestClassifyMd:
    def test_plain(self):
        assert cc.classify_md("# 标题\n正文") == "plain"

    def test_table(self):
        assert cc.classify_md("| a | b |\n|---|---|") == "table"

    def test_code(self):
        assert cc.classify_md("```python\nx\n```") == "code"

    def test_mixed(self):
        assert cc.classify_md("| a |\n|---|\n```\nx\n```") == "mixed"


class TestSelectSamples:
    def test_stratified_selection(self, tmp_path):
        _make_data(tmp_path)
        selection = cc.select_samples(str(tmp_path / "data"), plain_n=10, code_n=10)
        assert "plain.md" in selection.plain
        assert "table.md" in selection.table
        assert "code.md" in selection.code
        assert "mixed.md" in selection.mixed
        assert len(selection.all_paths()) == 4

    def test_table_all_keeps_all(self, tmp_path):
        _make_data(tmp_path)
        selection = cc.select_samples(str(tmp_path / "data"), plain_n=1, code_n=1)
        assert "table.md" in selection.table  # 表格全量


class TestGenerateGold:
    def test_returns_md_content(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# 标题\n正文", encoding="utf-8")
        assert cc.generate_gold(str(path)) == "# 标题\n正文"


class TestConvertPptx:
    def test_creates_pptx(self, tmp_path):
        md_path = tmp_path / "doc.md"
        md_path.write_text("# 标题\n正文\n```\ncode\n```", encoding="utf-8")
        out = tmp_path / "doc.pptx"
        cc.convert_md_to_pptx(str(md_path), str(out))
        assert out.exists()
        assert out.stat().st_size > 0


class TestPandocDetection:
    def _clear_pandoc_path(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "PANDOC_PATH", "")

    def test_require_pandoc_raises_when_missing(self, monkeypatch):
        self._clear_pandoc_path(monkeypatch)
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError):
            cc._require_pandoc()

    def test_require_pandoc_uses_explicit_path(self, monkeypatch, tmp_path):
        import config
        fake = tmp_path / "pandoc.exe"
        fake.write_bytes(b"")
        monkeypatch.setattr(config, "PANDOC_PATH", str(fake))
        assert cc._require_pandoc() == str(fake)

    def test_convert_html_raises_when_pandoc_missing(self, monkeypatch, tmp_path):
        self._clear_pandoc_path(monkeypatch)
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError):
            cc.convert_md_to_html(str(tmp_path / "a.md"), str(tmp_path / "a.html"))

    def test_convert_docx_raises_when_pandoc_missing(self, monkeypatch, tmp_path):
        self._clear_pandoc_path(monkeypatch)
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError):
            cc.convert_md_to_docx(str(tmp_path / "a.md"), str(tmp_path / "a.docx"))

    def test_convert_pdf_raises_when_pandoc_missing(self, monkeypatch, tmp_path):
        self._clear_pandoc_path(monkeypatch)
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError):
            cc.convert_md_to_pdf(str(tmp_path / "a.md"), str(tmp_path / "a.pdf"))


class TestFindBrowser:
    def test_returns_none_when_no_browser(self, monkeypatch):
        monkeypatch.setattr(cc, "_EDGE_CANDIDATES", [])
        monkeypatch.setattr(cc.shutil, "which", lambda name: None)
        assert cc._find_browser() is None

    def test_returns_candidate_when_exists(self, monkeypatch, tmp_path):
        fake = tmp_path / "msedge.exe"
        fake.write_bytes(b"")
        monkeypatch.setattr(cc, "_EDGE_CANDIDATES", [str(fake)])
        assert cc._find_browser() == str(fake)


class TestSnapshotConverted:
    def test_creates_manifest(self, tmp_path):
        (tmp_path / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.pdf").write_bytes(b"pdf")
        manifest = cc.snapshot_converted(str(tmp_path), str(tmp_path / "manifest.json"))
        assert "a.md" in manifest["files"]
        assert "sub/b.pdf" in manifest["files"]
        assert len(manifest["files"]) == 2
        assert (tmp_path / "manifest.json").exists()


class TestCli:
    def test_select_cli(self, tmp_path, monkeypatch):
        _make_data(tmp_path)
        monkeypatch.setattr(cc, "_PROJECT_ROOT", tmp_path)
        out = tmp_path / "samples.json"
        assert cc.main(["select", "--data", str(tmp_path / "data"), "--out", str(out)]) == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "plain.md" in data["plain"]

    def test_snapshot_cli(self, tmp_path):
        (tmp_path / "a.md").write_text("x", encoding="utf-8")
        out = tmp_path / "manifest.json"
        assert cc.main(["snapshot", "--corpus", str(tmp_path), "--out", str(out)]) == 0
        assert out.exists()

    def test_convert_cli(self, tmp_path, monkeypatch):
        _make_data(tmp_path)
        monkeypatch.setattr(cc, "_PROJECT_ROOT", tmp_path)
        samples = tmp_path / "samples.json"
        samples.write_text(json.dumps({"plain": ["plain.md"], "table": [], "code": [], "mixed": []}),
                           encoding="utf-8")
        calls = []
        for name in ("convert_md_to_html", "convert_md_to_pdf", "convert_md_to_docx", "convert_md_to_pptx"):
            monkeypatch.setattr(cc, name, lambda md, out, _n=name: calls.append(_n))
        out_dir = tmp_path / "out"
        gold_dir = tmp_path / "gold"
        assert cc.main(["convert", "--samples", str(samples), "--out", str(out_dir),
                        "--gold", str(gold_dir)]) == 0
        assert calls.count("convert_md_to_html") == 1
        assert calls.count("convert_md_to_pptx") == 1
        # gold 已生成(原 md 内容)
        assert (gold_dir / "plain.md").read_text(encoding="utf-8") == "# 纯文本\n正文"
