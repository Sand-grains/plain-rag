"""unit：indexing/loaders 四格式归一化（008-2）。

覆盖：PDF/HTML/DOCX/PPTX 经 load() 归一化为 Markdown + format_meta、
loader 逐页/逐 slide 自判（图文单元单跳过）、PPTX ## 边界、SKIP_TEXT_PIPELINE 返回空、
.md/.txt 路径不受影响。
"""
from indexing.loader import load
from tests._doc_samples import (
    make_docx,
    make_html,
    make_pdf,
    make_pptx,
)


class TestPdfLoader:
    def test_text_pdf_normalized(self, tmp_path):
        # 手写测试 PDF 用 Helvetica, 仅可靠提取 ASCII（真实 PDF 含 ToUnicode 可正常提中文）
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("Hello World sample content. " * 20))
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "Hello World" in chunks[0].content
        assert chunks[0].origin_metadata.doc_type == ".pdf"
        assert chunks[0].origin_metadata.protect_tables is True

    def test_short_pdf_skipped(self, tmp_path):
        path = tmp_path / "b.pdf"
        path.write_bytes(make_pdf("hi"))
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []


class TestHtmlLoader:
    def test_html_normalized_heading_and_table(self, tmp_path):
        html = (
            "<html><body><h1>大标题</h1><p>正文段落内容。" * 20 + "</p>"
            "<table><tr><th>列A</th><th>列B</th></tr>"
            "<tr><td>1</td><td>2</td></tr></table></body></html>"
        )
        path = tmp_path / "a.html"
        path.write_text(html, encoding="utf-8")
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "# 大标题" in chunks[0].content
        assert "| 列A | 列B |" in chunks[0].content
        assert chunks[0].origin_metadata.doc_type == ".html"

    def test_image_only_html_skipped(self, tmp_path):
        path = tmp_path / "b.html"
        path.write_text(make_html("标题", "x", images=2), encoding="utf-8")
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []


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
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "# 章节标题" in chunks[0].content
        assert "| A | B |" in chunks[0].content

    def test_empty_docx_skipped(self, tmp_path):
        path = make_docx([], tmp_path=tmp_path)
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks == []


class TestPptxLoader:
    def test_text_pptx_has_slide_boundaries(self, tmp_path):
        path = make_pptx([["第一页标题", "第一页内容。" * 20],
                          ["第二页标题", "第二页内容。" * 20]], tmp_path=tmp_path)
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert chunks[0].content.count("## ") == 2  # 两个 slide 边界

    def test_pptx_self_classify_image_slide_skipped(self, tmp_path):
        # 一页文本 slide + 一页纯图 slide → 只归一化文本页, 图文页单跳过记 vlm
        path = make_pptx([["标题", "正文内容" * 20]], with_image_slide=True, tmp_path=tmp_path)
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert chunks[0].content.count("## ") == 1
        format_meta = chunks[0].metadata.get("format_meta", {})
        assert format_meta.get("vlm_candidate_count", 0) >= 1


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
        chunks = load(str(path), base_dir=str(tmp_path))
        assert chunks[0].content.count("唯一标题") == 1

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
        chunks = load(str(path), base_dir=str(tmp_path))
        assert len(chunks) == 1
        assert "## 表页标题" in chunks[0].content
        assert "| 值00 | 值01 | 值02 |" in chunks[0].content
