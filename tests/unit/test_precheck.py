"""unit：preprocess/format_precheck 预检分流（008-1）。

覆盖四格式采样统计与分流决策（WHOLE_TEXT_PIPELINE / HTML_TEXT / SKIP_TEXT_PIPELINE）、
质量档位 degraded_flags（text_page_ratio_below_threshold / multi_column /
low_text_tag_ratio）与 vlm_candidate 聚合计数。
"""
import pytest

from preprocess.format_precheck import DispatchDecision, PrecheckResult, precheck
from tests._doc_samples import (
    make_docx,
    make_html,
    make_pdf,
    make_pptx,
    multi_column_content_stream,
)

TEXT_PDF = "这是正文内容。" * 40  # 40*7=280 字符


class TestPdfPrecheck:
    def test_text_pdf_whole_text(self, tmp_path):
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf(TEXT_PDF))
        result = precheck(path)
        assert isinstance(result, PrecheckResult)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE
        assert "text_page_ratio_below_threshold" not in result.degraded_flags

    def test_short_pdf_skip(self, tmp_path):
        path = tmp_path / "b.pdf"
        path.write_bytes(make_pdf("hi"))
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE

    def test_empty_pdf_skip(self, tmp_path):
        path = tmp_path / "c.pdf"
        path.write_bytes(make_pdf(""))
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.empty_page_ratio == 1.0

    def test_multi_column_flag(self, tmp_path):
        path = tmp_path / "d.pdf"
        path.write_bytes(make_pdf("", content_streams=[multi_column_content_stream()]))
        result = precheck(path)
        assert "multi_column" in result.degraded_flags

    def test_text_page_ratio_below_threshold(self, tmp_path):
        # 5 页: 1 页文本 + 4 页空 → 采样占比 0.2 < 0.9 → 质量档位, 仍 WHOLE_TEXT_PIPELINE
        path = tmp_path / "e.pdf"
        path.write_bytes(make_pdf([TEXT_PDF, "", "", "", ""]))
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE
        assert "text_page_ratio_below_threshold" in result.degraded_flags
        assert result.vlm_candidate_count > 0


class TestHtmlPrecheck:
    def test_normal_html_html_text(self, tmp_path):
        path = tmp_path / "a.html"
        path.write_text(make_html("标题", "正文内容。" * 30), encoding="utf-8")
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.HTML_TEXT

    def test_image_only_html_skip_vlm(self, tmp_path):
        path = tmp_path / "b.html"
        path.write_text(make_html("标题", "x", images=3), encoding="utf-8")
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.vlm_candidate_count == 3

    def test_empty_html_skip(self, tmp_path):
        path = tmp_path / "c.html"
        path.write_text(make_html("", ""), encoding="utf-8")
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.vlm_candidate_count == 0

    def test_low_text_tag_ratio_flag(self, tmp_path):
        # 大量标签、少量文字 → text_tag_ratio 低 → 低质量标记
        body = "<div>" * 200 + "短。" + "</div>" * 200
        path = tmp_path / "d.html"
        path.write_text(make_html("标题", body), encoding="utf-8")
        result = precheck(path)
        assert "low_text_tag_ratio" in result.degraded_flags


class TestDocxPrecheck:
    def test_text_docx_whole_text(self, tmp_path):
        path = make_docx(["段落一。", "段落二。" * 40], tmp_path=tmp_path)
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE

    def test_image_only_docx_skip_vlm(self, tmp_path):
        path = make_docx([], with_image=True, tmp_path=tmp_path)
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.vlm_candidate_count > 0

    def test_empty_docx_skip(self, tmp_path):
        path = make_docx([], tmp_path=tmp_path)
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.vlm_candidate_count == 0


class TestPptxPrecheck:
    def test_text_pptx_whole_text(self, tmp_path):
        path = make_pptx([["标题", "正文内容" * 30]], tmp_path=tmp_path)
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE

    def test_image_only_pptx_skip_vlm(self, tmp_path):
        path = make_pptx([[]], with_image_slide=True, tmp_path=tmp_path)
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE
        assert result.vlm_candidate_count > 0


class TestDispatcher:
    def test_unsupported_suffix_raises(self, tmp_path):
        path = tmp_path / "x.md"
        path.write_text("# hi", encoding="utf-8")
        with pytest.raises(ValueError):
            precheck(path)


class TestPrecheckSwitchFallback:
    def test_master_precheck_disabled_falls_back_whole_text(self, monkeypatch, tmp_path):
        import preprocess.format_precheck as precheck_mod
        monkeypatch.setattr(precheck_mod, "PRECHECK_ENABLED", False)
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("hi"))
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE
        assert "precheck_disabled" in result.degraded_flags

    def test_master_precheck_disabled_html_falls_back_html_text(self, monkeypatch, tmp_path):
        import preprocess.format_precheck as precheck_mod
        monkeypatch.setattr(precheck_mod, "PRECHECK_ENABLED", False)
        path = tmp_path / "a.html"
        path.write_text(make_html("标题", "x"), encoding="utf-8")
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.HTML_TEXT

    def test_format_precheck_gate_off_falls_back(self, monkeypatch, tmp_path):
        import preprocess.format_precheck as precheck_mod
        monkeypatch.setattr(precheck_mod, "_FORMAT_PRECHECK_GATES", {**precheck_mod._FORMAT_PRECHECK_GATES, "pdf": False})
        path = tmp_path / "a.pdf"
        path.write_bytes(make_pdf("hi"))
        result = precheck(path)
        assert result.doc_decision is DispatchDecision.WHOLE_TEXT_PIPELINE
