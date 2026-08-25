"""DOCX loader：按 precheck 决策抽取。

WHOLE_TEXT_PIPELINE: 段落(heading->#)、表格(->pipe-table)、内嵌图(仅计数)；
SKIP_TEXT_PIPELINE: 返回空。
"""

import re
from collections.abc import Iterator
from pathlib import Path

from preprocess.format_precheck import DispatchDecision, PrecheckResult
from indexing.loaders import skip_result
from ._md import count_degraded_tables, heading_md, to_pipe_table

_HEADING_STYLE_RE = re.compile(r"^Heading\s*([1-6])$", re.IGNORECASE)


def _heading_level(paragraph) -> int | None:
    """从段落样式名解析标题级别（Heading 1~6）, 非标题返回 None。"""
    style_name = (paragraph.style.name if paragraph.style else "") or ""
    match = _HEADING_STYLE_RE.match(style_name)
    if match:
        return int(match.group(1))
    return None


def _document_body_blocks(document) -> Iterator[object]:
    """按文档顺序产出段落/表格块（用于正文与表格交错归一化）。

    Args:
        document: python-docx Document 实例。

    Yields:
        段落（docx.text.paragraph.Paragraph）或表格（docx.table.Table）对象。
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _table_to_pipe(table) -> str:
    """把 python-docx 表格转成归一化 pipe-table Markdown。

    Args:
        table: python-docx Table 实例。

    Returns:
        str：pipe-table 块；空表返回空串。
    """
    rows = []
    for row in table.rows:
        rows.append([cell.text for cell in row.cells])
    return to_pipe_table(rows)


def docx_loader(path: Path, precheck: PrecheckResult) -> tuple[str, dict]:
    """把 DOCX 归一化为 Markdown + format_meta。

    Args:
        path: DOCX 文件路径。
        precheck: precheck 产出的分流决策。

    Returns:
        (markdown, format_meta)：归一化 MD（SKIP_TEXT_PIPELINE 时为空串）+ 格式元数据。
    """
    from docx import Document
    from docx.table import Table

    format_meta: dict = {"doc_type": ".docx"}

    if precheck.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE:
        return skip_result(".docx", precheck.vlm_candidate_count)

    document = Document(str(path))
    lines: list[str] = []
    image_count = 0

    for block in _document_body_blocks(document):
        if isinstance(block, Table):
            table_md = _table_to_pipe(block)
            if table_md:
                lines.append(table_md)
            continue
        level = _heading_level(block)
        text = block.text.strip()
        if not text:
            continue
        if level is not None:
            lines.append(heading_md(level, text))
        else:
            lines.append(text)

    # 图片仅计数（python-docx 无直接 API，用 inline_shapes）
    try:
        image_count = len(document.inline_shapes)
    except Exception:
        image_count = 0

    markdown = "\n\n".join(lines)
    format_meta.update({
        "paragraph_count": len(document.paragraphs),
        "image_count": image_count,
        "degraded_table_count": count_degraded_tables(markdown),
    })
    return markdown, format_meta
