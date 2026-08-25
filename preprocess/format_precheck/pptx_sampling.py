"""PPTX 采样预检：摄入 PPTX 文件路径，产出 PrecheckResult（分流决策 + slide 统计[文本 slide 数/图占比]）。

决策域: {WHOLE_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}

文本 slide 定义：text_per_slide >= PPTX_SLIDE_TEXT_THRESHOLD 且图占比 <= 50%。
整份存在文本 slide -> WHOLE_TEXT_PIPELINE；整份无文本 slide -> SKIP_TEXT_PIPELINE。
逐 slide 图文分类由 pptx_loader 自行判定（precheck 只聚合，B7 消费一致）。
"""
from pathlib import Path

from config import PPTX_SLIDE_TEXT_THRESHOLD
from .result import DispatchDecision, PrecheckResult
from .shared import pptx_image_area_ratio, slide_text


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
        area_ratio = pptx_image_area_ratio(slide, slide_area)
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
