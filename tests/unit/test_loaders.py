"""unit：indexing/loaders 四格式归一化（008-2）+ 012-2 F41 重型路由挂点。

覆盖：轻量 loader 归一化（PDF/HTML/DOCX/PPTX 经 load_multiformat 直接调）、
VLM/SKIP 路由、012-2 挂点（text 走重型链, 重型关 -> 失败清单不落轻量）、
.md/.txt 路径不受影响。
"""
from pathlib import Path

import pytest

from indexing.loader import load
from indexing.loaders import load_multiformat
from preprocess.format_precheck import DispatchDecision, PrecheckResult
from tests._doc_samples import (
    make_docx,
    make_html,
    make_pdf,
    make_pptx,
)


@pytest.fixture(autouse=True)
def _reset_failure_list():
    """隔离失败清单单例(012-2 F41): 每个测试从空开始。"""
    from indexing.parse_backends.failure_list import reset_failure_list
    reset_failure_list()
    yield
    reset_failure_list()


def _text_precheck() -> PrecheckResult:
    return PrecheckResult(doc_decision=DispatchDecision.WHOLE_TEXT_PIPELINE)


class TestVlmRouting:
    def test_vlm_decision_returns_empty_with_route_decision(self, tmp_path):
        # VLM_TEXT_PIPELINE 文档不落轻量: 返回空 MD + route_decision=vlm
        path = make_docx([], with_image=True, tmp_path=tmp_path)
        precheck_result = PrecheckResult(
            doc_decision=DispatchDecision.VLM_TEXT_PIPELINE,
            vlm_candidate_count=1,
        )
        markdown, meta = load_multiformat(path, precheck_result)
        assert markdown == ""
        assert meta["route_decision"] == "vlm"
        assert meta["skipped"] is True


class TestVlmRenderPages:
    """H1: VLM 路由先把文档渲染为图像(单图直通 / PDF 逐页 / 其余进失败清单)。"""

    def test_image_passthrough(self, tmp_path):
        from indexing.loader import _vlm_render_pages
        img = tmp_path / "a.png"
        img.write_bytes(b"x")
        assert _vlm_render_pages(img, str(tmp_path)) == [str(img)]

    def test_pdf_renders_each_page(self, tmp_path, monkeypatch):
        from indexing.loader import _vlm_render_pages
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"x")
        monkeypatch.setattr(
            "indexing.parse_backends.paged_pipeline.split_pdf_pages",
            lambda p, w: [str(tmp_path / "p0.pdf"), str(tmp_path / "p1.pdf")])
        monkeypatch.setattr(
            "indexing.parse_backends.paged_pipeline.render_pdf_page",
            lambda p, w: str(Path(p).with_suffix(".png")))
        images = _vlm_render_pages(pdf, str(tmp_path))
        assert len(images) == 2
        assert all(img.endswith(".png") for img in images)

    def test_pdf_no_renderable_page_raises(self, tmp_path, monkeypatch):
        from indexing.loader import _vlm_render_pages
        from indexing.parse_backends.vlm import VlmUnavailableError
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"x")
        monkeypatch.setattr(
            "indexing.parse_backends.paged_pipeline.split_pdf_pages",
            lambda p, w: [str(tmp_path / "p0.pdf")])
        monkeypatch.setattr(
            "indexing.parse_backends.paged_pipeline.render_pdf_page",
            lambda p, w: None)
        with pytest.raises(VlmUnavailableError, match="无法渲染"):
            _vlm_render_pages(pdf, str(tmp_path))

    def test_non_renderable_raises(self, tmp_path):
        from indexing.loader import _vlm_render_pages
        from indexing.parse_backends.vlm import VlmUnavailableError
        docx = tmp_path / "a.docx"
        docx.write_bytes(b"x")
        with pytest.raises(VlmUnavailableError, match="无法渲染"):
            _vlm_render_pages(docx, str(tmp_path))


class TestHeavyRouting:
    """012-2 F41 挂点: text 走重型链, 重型关 -> 失败清单不落轻量。"""

    def test_text_pdf_heavy_off_goes_failure_list(self, tmp_path, monkeypatch):
        from indexing.parse_backends import _BACKEND_GATES
        from indexing.parse_backends.failure_list import failure_list
        # 重型文本链 + VLM 全关(012-2 2.6 默认开, 此处显式模拟"无管线可用")
        for name in ("docling", "mineru", "markitdown", "vlm"):
            monkeypatch.setitem(_BACKEND_GATES, name, False)
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello World sample content. " * 20))
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []  # 不落轻量
        assert len(failure_list) == 1
        assert failure_list._records[0]["route_decision"] == "text"

    def test_text_html_heavy_off_goes_failure_list(self, tmp_path, monkeypatch):
        from indexing.parse_backends import _BACKEND_GATES
        from indexing.parse_backends.failure_list import failure_list
        for name in ("docling", "mineru", "markitdown", "vlm"):
            monkeypatch.setitem(_BACKEND_GATES, name, False)
        html = "<html><body><h1>大标题</h1><p>正文段落内容。" * 20 + "</p></body></html>"
        path = tmp_path / "a.html"
        path.write_text(html, encoding="utf-8")
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []
        assert len(failure_list) == 1

    def test_skip_returns_empty_no_failure(self, tmp_path):
        from indexing.parse_backends.failure_list import failure_list
        path = tmp_path / "b.pdf"
        path.write_bytes(make_pdf("hi"))
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []
        assert len(failure_list) == 0  # skip 不进失败清单


class TestHeavyRoutingSuccess:
    """012-2 F41 挂点成功路径: text/VLM 管线产出 Chunk + colpali 触发标记。"""

    def test_text_pipeline_success_produces_chunk(self, monkeypatch, tmp_path):
        from indexing.parse_backends import ParseResult
        class _Fake:
            def extract(self, path):
                return ParseResult(markdown="# 标题\n正文", format_meta={"backend": "docling"})
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello World sample content. " * 20))
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "# 标题" in chunks[0].content
        meta = chunks[0].metadata["format_meta"]
        assert meta["route_decision"] == "text"
        assert meta["colpali_triggered"] is False

    def test_vlm_pipeline_success_produces_chunk(self, monkeypatch, tmp_path):
        from preprocess.format_precheck import PrecheckResult, DispatchDecision
        # VLM 先渲染为图像, 再逐图走 _run_backend(子进程隔离 + 熔断 + 统计); 测试注入渲染与返回值
        monkeypatch.setattr(
            "indexing.parse_backends._run_backend",
            lambda name, path: ("# 扫描件\n正文", {"backend": "vlm"}))
        monkeypatch.setattr(
            "indexing.loader._vlm_render_pages",
            lambda path, workdir: [str(tmp_path / "page_000.png")])
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("x"))
        precheck_result = PrecheckResult(
            doc_decision=DispatchDecision.VLM_TEXT_PIPELINE, vlm_candidate_count=1)
        monkeypatch.setattr("preprocess.format_precheck.precheck", lambda p: precheck_result)
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "# 扫描件" in chunks[0].content
        assert chunks[0].metadata["format_meta"]["route_decision"] == "vlm"
        assert chunks[0].metadata["format_meta"]["vlm_pages"] == 1

    def test_colpali_triggered_unavailable_records_failure(self, monkeypatch, tmp_path):
        from indexing.parse_backends import ParseResult
        from indexing.parse_backends.failure_list import failure_list
        class _Fake:
            def extract(self, path):
                return ParseResult(markdown="# 标题\n正文", format_meta={"backend": "docling"})
        monkeypatch.setattr("indexing.parse_backends.resolve", lambda name: _Fake())
        monkeypatch.setattr("indexing.parse_backends.colpali.is_available", lambda: False)
        # 构造 colpali_triggered 的 precheck: 直接调 _load_multiformat 内部路径
        import indexing.loader as loader_mod
        from preprocess.format_precheck import PrecheckResult, DispatchDecision
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello World sample content. " * 20))
        precheck_result = PrecheckResult(
            doc_decision=DispatchDecision.WHOLE_TEXT_PIPELINE, colpali_triggered=True)
        monkeypatch.setattr("preprocess.format_precheck.precheck", lambda p: precheck_result)
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1  # 文本仍产出
        assert chunks[0].metadata["format_meta"]["colpali_triggered"] is True
        assert any(r["route_decision"] == "colpali" for r in failure_list._records)


class TestPdfLoader:
    def test_text_pdf_normalized(self, tmp_path):
        # 轻量 loader 归一化(直接调 load_multiformat, 008-2 路径)
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello World sample content. " * 20))
        markdown, meta = load_multiformat(path, _text_precheck())
        assert "Hello World" in markdown
        assert meta["doc_type"] == ".pdf"


class TestHtmlLoader:
    def test_html_normalized_heading_and_table(self, tmp_path):
        html = (
            "<html><body><h1>大标题</h1><p>正文段落内容。" * 20 + "</p>"
            "<table><tr><th>列A</th><th>列B</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table></body></html>"
        )
        path = tmp_path / "a.html"
        path.write_text(html, encoding="utf-8")
        markdown, meta = load_multiformat(path, _text_precheck())
        assert "# 大标题" in markdown
        assert "| 列A | 列B |" in markdown

    def test_image_only_html_skipped(self, tmp_path):
        path = tmp_path / "b.html"
        path.write_text(make_html("标题", "x", images=2), encoding="utf-8")
        precheck_result = PrecheckResult(
            doc_decision=DispatchDecision.SKIP_TEXT_PIPELINE, vlm_candidate_count=2)
        markdown, meta = load_multiformat(path, precheck_result)
        assert markdown == ""
        assert meta["skipped"] is True


class TestDocxLoader:
    def test_docx_normalized_heading_table(self, tmp_path):
        import docx
        document = docx.Document()
        document.add_heading("章节标题", level=1)
        document.add_paragraph("正文段落。" * 60)
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "A"
        table.cell(0, 1).text = "B"
        table.cell(1, 0).text = "1"
        table.cell(1, 1).text = "2"
        path = tmp_path / "doc.docx"
        document.save(str(path))
        markdown, meta = load_multiformat(path, _text_precheck())
        assert "# 章节标题" in markdown
        assert "| A | B |" in markdown

    def test_empty_docx_skipped(self, tmp_path):
        path = make_docx([], tmp_path=tmp_path)
        precheck_result = PrecheckResult(doc_decision=DispatchDecision.SKIP_TEXT_PIPELINE)
        markdown, meta = load_multiformat(path, precheck_result)
        assert markdown == ""


class TestPptxLoader:
    def test_text_pptx_has_slide_boundaries(self, tmp_path):
        path = make_pptx([["第一页标题", "第一页内容。" * 20],
                          ["第二页标题", "第二页内容。" * 20]], tmp_path=tmp_path)
        markdown, meta = load_multiformat(path, _text_precheck())
        assert markdown.count("## ") == 2  # 两个 slide 边界

    def test_pptx_self_classify_image_slide_skipped(self, tmp_path):
        # 一页文本 slide + 一页纯图 slide → 只归一化文本页, 图文页单跳过记 vlm
        path = make_pptx([["标题", "正文内容" * 20]], with_image_slide=True, tmp_path=tmp_path)
        markdown, meta = load_multiformat(path, _text_precheck())
        assert markdown.count("## ") == 1
        assert meta.get("vlm_candidate_count", 0) >= 1


class TestPlainTextUnaffected:
    def test_md_still_loads(self, tmp_path):
        path = tmp_path / "a.md"
        path.write_text("# 标题\n\n正文。", encoding="utf-8")
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "# 标题" in chunks[0].content
        assert chunks[0].origin_metadata.protect_tables is False

    def test_clean_plain_text_wired(self, monkeypatch, tmp_path):
        import indexing.loader as loader_mod
        monkeypatch.setattr(loader_mod, "CLEAN_PLAIN_TEXT", True)
        path = tmp_path / "b.md"
        path.write_text("关注公众号：x\n正文内容。", encoding="utf-8")
        chunks = load(str(path), base_dir=str(tmp_path))
        assert "关注公众号" not in chunks[0].content
        assert "正文内容。" in chunks[0].content


class TestLoaderSwitch:
    def test_loader_disabled_skips_format(self, monkeypatch, tmp_path):
        import indexing.loaders as loaders_mod
        monkeypatch.setattr(loaders_mod, "_FORMAT_LOADER_GATES", {**loaders_mod._FORMAT_LOADER_GATES, "pdf": False})
        from indexing.loaders import load_multiformat
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello content. " * 20))
        markdown, format_meta = load_multiformat(path, None)
        assert markdown == ""
        assert format_meta.get("disabled") is True


class TestPptxDedupAndTable:
    def test_pptx_title_not_duplicated(self, tmp_path):
        path = make_pptx([["唯一标题", "正文内容" * 20]], tmp_path=tmp_path)
        markdown, meta = load_multiformat(path, _text_precheck())
        assert markdown.count("唯一标题") == 1

    def test_text_box_not_counted_as_image(self, tmp_path):
        # 纯文本 slide 图占比应为 0（文本框/占位符不算图）
        path = make_pptx([["标题", "正文内容" * 20]], tmp_path=tmp_path)
        from pptx import Presentation
        from preprocess.format_precheck.shared import pptx_image_area_ratio
        prs = Presentation(str(path))
        slide_area = (prs.slide_width or 0) * (prs.slide_height or 0)
        assert pptx_image_area_ratio(prs.slides[0], slide_area) == 0.0

    def test_pptx_table_normalized_to_pipe(self, tmp_path):
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide.shapes.add_textbox(100000, 100000, 5000000, 400000)
        box.text = "表页标题"
        table_shape = slide.shapes.add_table(6, 3, Inches(1), Inches(2), Inches(4), Inches(2))
        for r in range(6):
            for c in range(3):
                table_shape.table.cell(r, c).text = f"值{r}{c}"
        path = tmp_path / "tbl.pptx"
        prs.save(str(path))
        markdown, meta = load_multiformat(path, _text_precheck())
        assert "## 表页标题" in markdown
        assert "| 值00 | 值01 | 值02 |" in markdown
