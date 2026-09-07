"""PPTX 采样预检：摄入 PPTX 文件路径，产出 PrecheckResult（分流决策 + slide 统计[文本 slide 数/图占比]）。

决策域: {WHOLE_TEXT_PIPELINE, VLM_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}
  - 存在文本 slide(text>=50 且图占比<=50%) -> WHOLE_TEXT_PIPELINE
  - 无文本 slide 但存在视觉 slide(text<50 且图占比>50%) -> VLM_TEXT_PIPELINE
  - 其余 -> SKIP_TEXT_PIPELINE
存在 chart 形状 -> colpali_triggered(并行触发)。
逐 slide 图文分类由 pptx_loader 自行判定（precheck 只聚合，B7 消费一致）。
"""
from pathlib import Path

from config import IMAGE_AREA_RATIO, PPTX_SLIDE_TEXT_THRESHOLD
from .result import DispatchDecision, PrecheckResult
from .shared import pptx_image_area_ratio, slide_text


def _slide_has_chart(slide) -> bool:
    """slide 内是否存在 chart 形状。"""
    return any(getattr(shape, "has_chart", False) for shape in slide.shapes)


def precheck_pptx(path: Path) -> PrecheckResult:
    """对 PPTX 做预检：逐 slide 统计文本与图占比并产出分流决策。

    Args:
        path: PPTX 文件路径。

    Returns:
        PrecheckResult：WHOLE_TEXT_PIPELINE / VLM_TEXT_PIPELINE / SKIP_TEXT_PIPELINE 决策与质量信号。
    """
    from pptx import Presentation

    presentation = Presentation(str(path))
    slide_width = presentation.slide_width or 0
    slide_height = presentation.slide_height or 0
    slide_area = slide_width * slide_height

    text_slides = 0
    vlm_slides = 0
    vlm_candidates = 0
    colpali_triggered = False
    total = len(presentation.slides)

    for slide in presentation.slides:
        text = slide_text(slide)
        area_ratio = pptx_image_area_ratio(slide, slide_area)
        has_text = len(text) >= PPTX_SLIDE_TEXT_THRESHOLD
        has_visual = area_ratio > 0.5
        if has_text and not has_visual:
            text_slides += 1
        elif not has_text and has_visual:
            vlm_slides += 1
            vlm_candidates += 1
        elif has_text:
            text_slides += 1
        else:
            vlm_candidates += 1
        if _slide_has_chart(slide) or area_ratio >= IMAGE_AREA_RATIO:
            colpali_triggered = True

    if text_slides > 0:
        decision = DispatchDecision.WHOLE_TEXT_PIPELINE
    elif vlm_slides > 0:
        decision = DispatchDecision.VLM_TEXT_PIPELINE
    else:
        decision = DispatchDecision.SKIP_TEXT_PIPELINE

    return PrecheckResult(
        doc_decision=decision,
        empty_page_ratio=0.0,
        vlm_candidate_count=vlm_candidates,
        colpali_triggered=colpali_triggered,
        degraded_flags=[],
        sampling_format_stats={
            "total_slides": total,
            "text_slides": text_slides,
            "vlm_slides": vlm_slides,
            "slide_area": round(slide_area, 2),
        },
    )
