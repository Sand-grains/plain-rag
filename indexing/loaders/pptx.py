"""PPTX loader：按 precheck 决策抽取，loader 逐 slide 自判图文分类。

每个文本 slide 输出 `## <标题>` 边界标记（切分器可见 slide 边界，且不横跨），
正文不含重复标题行；表格/图表 shape 转 pipe-table 或仅计数；图文/纯图 slide 单跳过记 vlm。
"""
from pathlib import Path

from config import PPTX_SLIDE_TEXT_THRESHOLD
from preprocess.format_precheck import DispatchDecision, PrecheckResult
from preprocess.format_precheck.shared import pptx_image_area_ratio, slide_text
from indexing.loaders import skip_result, vlm_result
from ._md import count_degraded_tables, to_pipe_table


def _slide_blocks(slide) -> list[str]:
    """把 slide 内文本帧与表格归一化为块列表。

    首文本帧的首行即标题（由 _slide_title 取用），故在正文块里跳过该行，
    避免 ## 标题重复出现；其余文本帧整段保留，表格转 pipe-table。
    """
    blocks: list[str] = []
    first_text_done = False
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            text = shape.text_frame.text.strip()
            if not text:
                continue
            if not first_text_done:
                body = "\n".join(text.splitlines()[1:]).strip()
                if body:
                    blocks.append(body)
                first_text_done = True
            else:
                blocks.append(text)
        if getattr(shape, "has_table", False):
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            table_md = to_pipe_table(rows)
            if table_md:
                blocks.append(table_md)
    return blocks


def _slide_title(slide) -> str:
    """取 slide 首个文本帧的首行作标题（去重：正文不再重复该行）。"""
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            first = shape.text_frame.text.strip().splitlines()
            if first:
                return first[0][:50]
    return ""


def pptx_loader(path: Path, precheck: PrecheckResult) -> tuple[str, dict]:
    """把 PPTX 归一化为 Markdown + format_meta。

    Args:
        path: PPTX 文件路径。
        precheck: precheck 产出的分流决策。

    Returns:
        (markdown, format_meta)：归一化 MD（SKIP_TEXT_PIPELINE 时为空串）+ 格式元数据。
    """
    from pptx import Presentation

    format_meta: dict = {"doc_type": ".pptx"}

    if precheck.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE:
        return skip_result(".pptx", precheck.vlm_candidate_count)
    if precheck.doc_decision is DispatchDecision.VLM_TEXT_PIPELINE:
        return vlm_result(".pptx", precheck.vlm_candidate_count)

    presentation = Presentation(str(path))
    slide_width = presentation.slide_width or 0
    slide_height = presentation.slide_height or 0
    slide_area = slide_width * slide_height

    lines: list[str] = []
    vlm_candidates = 0

    for index, slide in enumerate(presentation.slides, 1):
        text = slide_text(slide).strip()
        area_ratio = pptx_image_area_ratio(slide, slide_area)
        if len(text) >= PPTX_SLIDE_TEXT_THRESHOLD and area_ratio <= 0.5:
            title = _slide_title(slide) or f"第{index}页"
            lines.append(f"## {title}")
            lines.extend(_slide_blocks(slide))
        else:
            vlm_candidates += 1

    markdown = "\n\n".join(lines)
    format_meta.update({
        "slide_count": len(presentation.slides),
        "vlm_candidate_count": vlm_candidates,
        "degraded_table_count": count_degraded_tables(markdown),
    })
    return markdown, format_meta
