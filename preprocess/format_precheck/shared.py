"""format_precheck 与 indexing/loaders 公用的图文判定逻辑（B7 消费一致）。

供采样预检 (xxx_sampling.py) 与 loader 共用, 保证"图文页/图文 slide"判定口径两边一致。
这里只放无状态纯函数，不掺格式解析逻辑。

- pdf_image_area_ratio: PDF 页图片面积占比
- pptx_image_area_ratio: slide 图/图表面积占比
- slide_text: 聚合 slide 内文本
"""
from typing import Any

# PPTX 仅真图片计入图占比（文本框/占位符/表格是文本内容, 不计）
_PICTURE_TYPES = {13}  # MSO_SHAPE_TYPE.PICTURE


def pdf_image_area_ratio(page: Any) -> float:
    """PDF 页图片面积占比：图片 bbox 面积和 / 页面积（0.0 ~ 1.0）。

    Args:
        page: pdfplumber Page 实例。

    Returns:
        float: 图片面积占整页的比例。
    """
    page_area = (page.width or 0) * (page.height or 0)
    if not page_area:
        return 0.0
    total_area = 0.0
    for image in page.images:
        width = (image.get("x1", 0) or 0) - (image.get("x0", 0) or 0)
        height = (image.get("bottom", 0) or 0) - (image.get("top", 0) or 0)
        total_area += max(width, 0) * max(height, 0)
    return total_area / page_area


def pptx_image_area_ratio(slide: Any, slide_area: float) -> float:
    """slide 内图/图表形状面积和 / 版面积（0.0 ~ 1.0）。

    只统计真图片(MSO_SHAPE_TYPE.PICTURE=13)与图表(has_chart)；
    文本框/占位符/表格是文本内容，不占图占比（避免大文本框 slide 被误判"图主导"）。

    Args:
        slide: python-pptx Slide 实例。
        slide_area: 版面积（宽 x 高）。

    Returns:
        图片/图表面积占版面的比例。
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


def slide_text(slide: Any) -> str:
    """聚合 slide 内所有文本形状的字符。

    Args:
        slide: python-pptx Slide 实例。

    Returns:
        str: 所有文本框 + 表格单元格文本拼接。
    """
    parts = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            parts.append(shape.text_frame.text)
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
    return "".join(parts)
