"""PDF 采样预检：摄入 PDF 文件路径，产出 PrecheckResult（分流决策 + 采样统计[样本页数/文本页数/空页率/图占比] + 质量降级标记[multi_column / text_page_ratio_below_threshold]）。

决策域: {WHOLE_TEXT_PIPELINE, VLM_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}
  - 采样页全为文本(text>=150 且无图) 或 多栏 -> WHOLE_TEXT_PIPELINE(Docling 布局)
  - 采样页全为视觉(text<150 且有图) -> VLM_TEXT_PIPELINE
  - 采样页全为空(text<150 且无图) -> SKIP_TEXT_PIPELINE
  - 混合 -> WHOLE_TEXT_PIPELINE(图文混排, vlm 页记 vlm_candidate)
图表 bbox 面积 >= 页面 30% -> colpali_triggered(并行触发, 非互斥)。
"""
from pathlib import Path

from config import (
    IMAGE_AREA_RATIO,
    PDF_IMAGE_AREA_RATIO,
    PDF_MULTI_COLUMN_GAP,
    PDF_SAMPLE_PAGES,
    PDF_TEXT_PAGE_RATIO,
    PDF_TEXT_THRESHOLD,
)
from .result import DispatchDecision, PrecheckResult
from .shared import pdf_image_area_ratio


def _sample_indices(total_pages: int, sample_n: int) -> list[int]:
    """首/中/尾均匀采样索引：从 0 到 total-1 等距取 n 个（含首尾），小文档全采。"""
    if total_pages <= 0:
        return []
    n = min(sample_n, total_pages)
    if n == 1:
        return [0]
    return sorted(round(i * (total_pages - 1) / (n - 1)) for i in range(n))


def _is_multi_column(page) -> bool:
    """词 x 坐标聚列检测：存在 > 阈值的 x 空隙把词分成 2+ 簇视为多栏。"""
    words = page.extract_words() or []
    if len(words) < 5:
        return False
    x_positions = sorted(word["x0"] for word in words)
    clusters = 1
    previous = x_positions[0]
    for x in x_positions[1:]:
        if x - previous > PDF_MULTI_COLUMN_GAP:
            clusters += 1
        previous = x
    return clusters >= 2


def precheck_pdf(path: Path) -> PrecheckResult:
    """对 PDF 做采样预检，产出 PrecheckResult（doc_decision + 质量信号）。

    Args:
        path: PDF 文件路径。

    Returns:
        PrecheckResult：采样窗口内的分流决策与质量统计。
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        sample_indices = _sample_indices(total_pages, PDF_SAMPLE_PAGES)
        text_pages = 0
        vlm_pages = 0
        skip_pages = 0
        empty_pages = 0
        vlm_candidates = 0
        sampled = 0
        multi_column = False
        colpali_triggered = False

        for index in sample_indices:
            page = pdf.pages[index]
            text = (page.extract_text() or "").strip()
            sampled += 1
            if not text:
                empty_pages += 1
            area_ratio = pdf_image_area_ratio(page)
            has_text = len(text) >= PDF_TEXT_THRESHOLD
            has_image = area_ratio > 0
            if has_text and not has_image:
                text_pages += 1
            elif not has_text and has_image:
                vlm_pages += 1
            elif not has_text and not has_image:
                skip_pages += 1
            else:  # 图文混排: 文本管线处理, 图页记 vlm_candidate
                text_pages += 1
                vlm_candidates += 1
            if _is_multi_column(page):
                multi_column = True
            if area_ratio >= IMAGE_AREA_RATIO:
                colpali_triggered = True

        ratio = text_pages / sampled if sampled else 0.0
        degraded_flags: list[str] = []
        if multi_column:
            degraded_flags.append("multi_column")
        if 0 < ratio < PDF_TEXT_PAGE_RATIO:
            degraded_flags.append("text_page_ratio_below_threshold")

        if multi_column or text_pages > 0:
            decision = DispatchDecision.WHOLE_TEXT_PIPELINE
        elif vlm_pages > 0:
            decision = DispatchDecision.VLM_TEXT_PIPELINE
        else:
            decision = DispatchDecision.SKIP_TEXT_PIPELINE

        return PrecheckResult(
            doc_decision=decision,
            empty_page_ratio=(empty_pages / sampled) if sampled else 0.0,
            vlm_candidate_count=vlm_candidates,
            colpali_triggered=colpali_triggered,
            degraded_flags=degraded_flags,
            sampling_format_stats={
                "total_pages": total_pages,
                "sampled_pages": sampled,
                "sample_indices": sample_indices,
                "text_pages": text_pages,
                "vlm_pages": vlm_pages,
                "skip_pages": skip_pages,
            },
        )
