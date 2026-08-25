"""loaders 共用的"归一化 Markdown 构建工具", 负责把 loaders 的表格/标题转成统一的 Markdown 写法, 并统计被降级的表格数。

统一输出契约:
  标题 -> # 层级
  表格 -> pipe-table
  代码 -> fenced block
"""

import re

from config import EMBEDDING_MODEL_TOKEN_CONSTRAINT, TABLE_ATOMIC_MAX_CHARS
from indexing.splitter.utils import find_protected_table_ranges

_TABLE_ROW_SPLIT_RE = re.compile(r"\s*\|\s*")


def count_degraded_tables(markdown: str) -> int:
    """统计归一化 MD 中超限降级（按行切分）的表格数，写进 format_meta。

    Args:
        markdown: 归一化后的 pipe-table 文本。

    Returns:
        int: 降级表数（超 TABLE_ATOMIC_MAX_CHARS 或受保护后仍超 token 预算的表）。
    """
    _, degraded_tables = find_protected_table_ranges(
        markdown, TABLE_ATOMIC_MAX_CHARS, EMBEDDING_MODEL_TOKEN_CONSTRAINT
    )
    return degraded_tables


def to_pipe_table(rows: list[list[str]]) -> str:
    """把二维行列表（list[list[str]]）转成 pipe-table（含表头分隔行）

    Args:
        rows: 表格行，每行是一列字符串列表。

    Returns:
        str：归一化 pipe-table Markdown 块。
    """
    if not rows:
        return ""
    lines = []
    for row in rows:
        cells = [str(cell).replace("\n", " ").strip() for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 1:
        # 第二行插入分隔行（若只有表头则补一行空分隔）
        sep = "| " + " | ".join(["---"] * max(len(rows[0]), 1)) + " |"
        lines.insert(1, sep)
    return "\n".join(lines)


def heading_md(level: int, text: str) -> str:
    """把标题级别映射为 # 层级（clamp 1~6）。"""
    level = max(1, min(6, level))
    return "#" * level + " " + text.strip()
