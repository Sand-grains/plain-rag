"""页级管线的 跨页合并
它把页级重试/VLM 修正后得到的逐页 Markdown 文本, 合并回整篇, 并避免表格拆断, 列表编号断裂

锚点规则:
  - 表格续页: 检测"续表"/重复表头; 前后表格列数一致且满足 max_page_gap 时合并, 去掉重复表头
  - 有序列表续页: 下一页首项编号 = 上一页末项编号 + 1 时合并
  - 无序列表续页: 前后 bullet 符号一致时合并

合并后 metadata 记录 cross_page_merged / cross_page_merge_kind / cross_page_merge_pages。
"""
from __future__ import annotations

import re

from indexing.parse_backends._normalize import PIPE_TABLE_LINE_RE, TABLE_SEPARATOR_RE

_ORDERED_ITEM_RE = re.compile(r"^(\d+)\.\s+")
_UNORDERED_BULLET_RE = re.compile(r"^([-*])\s+")


def _table_columns(line: str) -> int:
    """pipe-table 行列数(非空单元格数)。"""
    return len([cell for cell in line.split("|") if cell.strip()])


def _trailing_table(lines: list[str]) -> list[str]:
    """取 lines 末尾的连续 pipe-table 行(含表头/分隔/数据)。"""
    result: list[str] = []
    for line in reversed(lines):
        if PIPE_TABLE_LINE_RE.match(line):
            result.append(line)
        elif result:
            break
    return list(reversed(result))


def _leading_table(lines: list[str]) -> list[str]:
    """取 lines 开头的连续 pipe-table 行。"""
    result: list[str] = []
    for line in lines:
        if PIPE_TABLE_LINE_RE.match(line):
            result.append(line)
        else:
            break
    return result


def _merge_table(merged: list[str], page_lines: list[str]) -> bool:
    """尝试把 page_lines 开头的表格续接到 merged 末尾的表格(列数一致, 去重复表头)。

    Args:
        merged: 已累积的输出行。
        page_lines: 当前页行。

    Returns:
        bool: 是否完成表格合并(True 时 page_lines 已并入 merged)。
    """
    trailing = _trailing_table(merged)
    leading = _leading_table(page_lines)
    if not trailing or not leading:
        return False
    if _table_columns(trailing[-1]) != _table_columns(leading[0]):
        return False
    # 续页表格若含表头+分隔行(重复表头), 去掉表头与分隔行, 只留数据行
    body = list(leading)
    if len(body) >= 2 and TABLE_SEPARATOR_RE.match(body[1]):
        body = body[2:]
    merged.extend(body)
    return True


def _merge_list(merged: list[str], page_lines: list[str]) -> bool:
    """尝试把 page_lines 开头的列表续接到 merged 末尾的列表(有序编号+1 / 无序 bullet 一致)。

    Args:
        merged: 已累积的输出行。
        page_lines: 当前页行。

    Returns:
        bool: 是否完成列表合并(True 时 page_lines 首项已并入 merged)。
    """
    last = next((line for line in reversed(merged) if line.strip()), "")
    first = next((line for line in page_lines if line.strip()), "")
    if not last or not first:
        return False
    last_ordered = _ORDERED_ITEM_RE.match(last)
    first_ordered = _ORDERED_ITEM_RE.match(first)
    if last_ordered and first_ordered:
        if int(first_ordered.group(1)) == int(last_ordered.group(1)) + 1:
            merged.append(first)
            return True
        return False
    last_bullet = _UNORDERED_BULLET_RE.match(last)
    first_bullet = _UNORDERED_BULLET_RE.match(first)
    if last_bullet and first_bullet and last_bullet.group(1) == first_bullet.group(1):
        merged.append(first)
        return True
    return False


def merge_cross_page(pages: list[str]) -> tuple[str, dict]:
    """把多页 Markdown 合并为整篇, 应用跨页表格/列表合并锚点。

    Args:
        pages: 每页归一化 Markdown 列表(按页序)。

    Returns:
        tuple[str, dict]: (合并后的整篇 Markdown, 合并元信息
            {cross_page_merged, cross_page_merge_kind, cross_page_merge_pages})。
    """
    merged_lines: list[str] = []
    meta: dict = {"cross_page_merged": False, "cross_page_merge_kind": [], "cross_page_merge_pages": []}
    for index, page in enumerate(pages):
        lines = page.splitlines()
        if not merged_lines:
            merged_lines = lines
            continue
        if _merge_table(merged_lines, lines):
            meta["cross_page_merged"] = True
            meta["cross_page_merge_kind"].append("table")
            meta["cross_page_merge_pages"].append(index)
            continue
        if _merge_list(merged_lines, lines):
            meta["cross_page_merged"] = True
            meta["cross_page_merge_kind"].append("list")
            meta["cross_page_merge_pages"].append(index)
            continue
        merged_lines.append("")
        merged_lines.extend(lines)
    return "\n".join(merged_lines), meta
