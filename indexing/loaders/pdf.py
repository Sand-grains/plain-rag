"""PDF loader：按 precheck 决策抽取，loader 全文逐页自判图文分类。

WHOLE_TEXT_PIPELINE: 逐页 extract_text，文本页拼入归一化 MD，图文/纯图页单页跳过记 vlm_candidate。
SKIP_TEXT_PIPELINE: 返回空，format_meta 记 vlm_candidate。
"""
from pathlib import Path

from config import PDF_IMAGE_AREA_RATIO, PDF_TEXT_THRESHOLD
from preprocess.format_precheck import DispatchDecision, PrecheckResult
from preprocess.format_precheck.shared import pdf_image_area_ratio
from indexing.loaders import skip_result
from ._md import count_degraded_tables


def pdf_loader(path: Path, precheck: PrecheckResult) -> tuple[str, dict]:
    """把 PDF 归一化为 Markdown + format_meta。

    Args:
        path: PDF 文件路径。
        precheck: precheck 产出的分流决策。

    Returns:
        (markdown, format_meta)：归一化 MD（SKIP_TEXT_PIPELINE 或空文档时为空串）+ 格式元数据。
    """
    import pdfplumber

    format_meta: dict = {"doc_type": ".pdf"}

    if precheck.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE:
        return skip_result(".pdf", precheck.vlm_candidate_count)

    blocks: list[str] = []
    vlm_candidates = 0
    total_pages = 0
    empty_pages = 0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            if not text:
                empty_pages += 1
            if len(text) >= PDF_TEXT_THRESHOLD and pdf_image_area_ratio(page) <= PDF_IMAGE_AREA_RATIO:
                blocks.append(text)
            else:
                vlm_candidates += 1

    markdown = "\n\n".join(blocks)
    format_meta.update({
        "page_count": total_pages,
        "empty_page_count": empty_pages,
        "vlm_candidate_count": vlm_candidates,
        "degraded_table_count": count_degraded_tables(markdown),
    })
    return markdown, format_meta
