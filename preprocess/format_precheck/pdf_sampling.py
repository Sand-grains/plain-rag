"""PDF 采样预检：摄入 PDF 文件路径，产出 PrecheckResult（分流决策 + 采样统计[样本页数/文本页数/空页率/图占比] + 质量降级标记[multi_column / text_page_ratio_below_threshold]）。

决策域: {WHOLE_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}

采样窗口内存在文本页 -> WHOLE_TEXT_PIPELINE；无文本页 -> SKIP_TEXT_PIPELINE。
文本页占比不足(0 < 占比 < 0.9)不是决策，是 WHOLE_TEXT_PIPELINE 的质量档位，
记 degraded_flag("text_page_ratio_below_threshold") + vlm_candidate_count。
多栏检测只标记 degraded_flag("multi_column")，本期不处理阅读顺序。
"""
from pathlib import Path

from config import (
    PDF_IMAGE_AREA_RATIO,
    PDF_MULTI_COLUMN_GAP,
    PDF_SAMPLE_PAGES,
    PDF_TEXT_PAGE_RATIO,
    PDF_TEXT_THRESHOLD,
)
from .result import DispatchDecision, PrecheckResult


def _sample_indices(total_pages: int, sample_n: int) -> list[int]:
    """首/中/尾均匀采样索引：从 0 到 total-1 等距取 n 个（含首尾），小文档全采。"""
    if total_pages <= 0:
        return []
    n = min(sample_n, total_pages)
    if n == 1:
        return [0]
    return sorted(round(i * (total_pages - 1) / (n - 1)) for i in range(n))


def image_area_ratio(page) -> float:
    """页面图片面积占比：图片 bbox 面积和 / 页面积（0.0 ~ 1.0）。"""
    page_area = (page.width or 0) * (page.height or 0)
    if not page_area:
        return 0.0
    total_area = 0.0
    for image in page.images:
        width = (image.get("x1", 0) or 0) - (image.get("x0", 0) or 0)
        height = (image.get("bottom", 0) or 0) - (image.get("top", 0) or 0)
        total_area += max(width, 0) * max(height, 0)
    return total_area / page_area


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
        empty_pages = 0
        vlm_candidates = 0
        sampled = 0
        multi_column = False

        for index in sample_indices:
            page = pdf.pages[index]
            text = (page.extract_text() or "").strip()
            sampled += 1
            if not text:
                empty_pages += 1
            area_ratio = image_area_ratio(page)
            if len(text) >= PDF_TEXT_THRESHOLD and area_ratio <= PDF_IMAGE_AREA_RATIO:
                text_pages += 1
            else:
                vlm_candidates += 1
            if _is_multi_column(page):
                multi_column = True

        ratio = text_pages / sampled if sampled else 0.0
        degraded_flags: list[str] = []
        if multi_column:
            degraded_flags.append("multi_column")
        if 0 < ratio < PDF_TEXT_PAGE_RATIO:
            degraded_flags.append("text_page_ratio_below_threshold")

        return PrecheckResult(
            doc_decision=DispatchDecision.WHOLE_TEXT_PIPELINE if text_pages > 0 else DispatchDecision.SKIP_TEXT_PIPELINE,
            empty_page_ratio=(empty_pages / sampled) if sampled else 0.0,
            vlm_candidate_count=vlm_candidates,
            degraded_flags=degraded_flags,
            sampling_format_stats={
                "total_pages": total_pages,
                "sampled_pages": sampled,
                "sample_indices": sample_indices,
                "text_pages": text_pages,
            },
        )
