"""PPTX 采样预检：摄入 PPTX 文件路径，产出 PrecheckResult（分流决策 + slide 统计[文本 slide 数/图占比]）。

决策域: {WHOLE_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}

文本 slide 定义：text_per_slide >= PPTX_SLIDE_TEXT_THRESHOLD 且图占比 <= 50%。
整份存在文本 slide -> WHOLE_TEXT_PIPELINE；整份无文本 slide -> SKIP_TEXT_PIPELINE。
逐 slide 图文分类由 pptx_loader 自行判定（precheck 只聚合，B7 消费一致）。
"""
from pathlib import Path

from config import PPTX_SLIDE_TEXT_THRESHOLD
from .result import DispatchDecision, PrecheckResult

_PICTURE_TYPES = {13}  # MSO_SHAPE_TYPE.PICTURE; 仅真图片计入图占比(文本框/占位符/表格是文本内容, 不计)


def slide_text(slide) -> str:
    """聚合 slide 内所有文本形状的字符。"""
    parts = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            parts.append(shape.text_frame.text)
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
    return "".join(parts)


def image_area_ratio(slide, slide_area: float) -> float:
    """slide 内图/图表形状面积和 / 版面积（0.0 ~ 1.0）。

    只统计真图片(MSO_SHAPE_TYPE.PICTURE=13)与图表(has_chart)；
    文本框/占位符/表格是文本内容，不占图占比（避免大文本框 slide 被误判"图主导"）。
    """
    if not slide_area:
        return 0.0
    total_area = 0.0
    for shape in slide.shapes:
        is_visual = (
            getattr(shape, "shape_type", None) in _PICTURE_TYPES
            or getattr(shape, "has_chart", False)
        )
        if is_visual:
            total_area += (shape.width or 0) * (shape.height or 0)
    return total_area / slide_area


def precheck_pptx(path: Path) -> PrecheckResult:
    """对 PPTX 做预检：逐 slide 统计文本与图占比并产出分流决策。

    Args:
        path: PPTX 文件路径。

    Returns:
        PrecheckResult：WHOLE_TEXT_PIPELINE 或 SKIP_TEXT_PIPELINE 决策与质量信号。
    """
    from pptx import Presentation

    presentation = Presentation(str(path))
    slide_width = presentation.slide_width or 0
    slide_height = presentation.slide_height or 0
    slide_area = slide_width * slide_height

    text_slides = 0
    vlm_candidates = 0
    total = len(presentation.slides)

    for slide in presentation.slides:
        text = slide_text(slide)
        area_ratio = image_area_ratio(slide, slide_area)
        if len(text) >= PPTX_SLIDE_TEXT_THRESHOLD and area_ratio <= 0.5:
            text_slides += 1
        else:
            vlm_candidates += 1

    return PrecheckResult(
        doc_decision=DispatchDecision.WHOLE_TEXT_PIPELINE if text_slides > 0 else DispatchDecision.SKIP_TEXT_PIPELINE,
        empty_page_ratio=0.0,
        vlm_candidate_count=vlm_candidates,
        degraded_flags=[],
        sampling_format_stats={
            "total_slides": total,
            "text_slides": text_slides,
            "slide_area": round(slide_area, 2),
        },
    )
