"""解析保真度信号: 用可计算的质量判定, 驱动文本管线降级/页级重试/是否进 VLM
(即这份解析结果有没有忠实保留原文档的内容和结构, 是"有没有明显丢东西"

它本身只是判定, 实际动手分给 parse_backends/__init__.py _run_whole_doc_chain, paged_pipeline.py _parse_page)
任意后端输出经 quality_ok 判定, 能稳定区分"达标/不达标", 驱动降级链正确换后端或进 VLM 修正

重型后端(MinerU/MarkItDown)多数无原生 confidence, 用代理信号来近似质量判定:
  - 空产出: 归一化 MD strip 后为空 -> 判失败
  - 表格丢失: 当前输出表格数 / 原始表格数 < 0.8 -> 判失败(原始不可得时退化相对前一级, proxy_basis=previous)
  - 代码 fence 保留率: 当前 fenced code 块数 / 预期块数 < 0.8 -> 判失败
  - 标题树连续性: 标题层级跳级(如 # 直接到 ###) -> 判失败(沿用六维诊断字段)
"""
from __future__ import annotations

import re

from indexing.parse_backends._normalize import (
    FENCED_CODE_RE,
    PIPE_TABLE_LINE_RE,
    TABLE_SEPARATOR_RE,
)

# 代理信号阈值
MIN_TABLE_RATIO_RETAINED = 0.8
MIN_CODE_FENCE_RETAINED = 0.8

_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+")


def is_empty(markdown: str) -> bool:
    """空产出判定: 归一化 MD strip 后为空。"""
    return not markdown.strip()


def table_count(markdown: str) -> int:
    """统计 pipe-table 张数(连续 pipe 行视为一张表)。"""
    tables = 0
    in_table = False
    for line in markdown.splitlines():
        if PIPE_TABLE_LINE_RE.match(line):
            if not in_table:
                tables += 1
                in_table = True
        else:
            in_table = False
    return tables


def code_fence_count(markdown: str) -> int:
    """统计 fenced code 块数。"""
    return len(FENCED_CODE_RE.findall(markdown))


def heading_continuous(markdown: str) -> bool:
    """标题树连续性: 标题层级不跳级(如 # 直接到 ### 视为不连续)。

    Args:
        markdown: 归一化 MD。

    Returns:
        bool: 标题层级连续(无标题或层级递增不超过 1)返回 True。
    """
    levels = [len(m.group(1)) for m in _HEADING_RE.finditer(markdown)]
    for previous, current in zip(levels, levels[1:]):
        if current > previous + 1:
            return False
    return True


def table_retention(parsed: str, original_count: int | None) -> float | None:
    """表格保留率: 当前输出表格数 / 原始表格数。

    Args:
        parsed: 解析出的归一化 MD。
        original_count: 原始表格数(precheck 阶段统计); None 表示不可得。

    Returns:
        float | None: 保留率; original_count 为 None 或 0 时返回 None(不可判)。
    """
    if original_count is None or original_count <= 0:
        return None
    return table_count(parsed) / original_count


def code_fence_retention(parsed: str, expected_count: int | None) -> float | None:
    """代码 fence 保留率: 当前 fenced code 块数 / 预期块数。

    Args:
        parsed: 解析出的归一化 MD。
        expected_count: 预期代码块数; None 表示不可得。

    Returns:
        float | None: 保留率; expected_count 为 None 或 0 时返回 None(不可判)。
    """
    if expected_count is None or expected_count <= 0:
        return None
    return code_fence_count(parsed) / expected_count


def quality_ok(markdown: str, original_table_count: int | None = None,
                expected_code_blocks: int | None = None) -> tuple[bool, list[str], dict]:
    """质量代理判定: 空产出/表格丢失/代码 fence 丢失/标题不连续。

    Args:
        markdown: 解析出的归一化 MD。
        original_table_count: 原始表格数(可判时传入)。
        expected_code_blocks: 预期代码块数(可判时传入)。

    Returns:
        tuple[bool, list[str], dict]: (是否达标, 未达标原因列表, 信号字典)。
    """
    reasons: list[str] = []
    if is_empty(markdown):
        reasons.append("empty")
    tr = table_retention(markdown, original_table_count)
    if tr is not None and tr < MIN_TABLE_RATIO_RETAINED:
        reasons.append("table_loss")
    cr = code_fence_retention(markdown, expected_code_blocks)
    if cr is not None and cr < MIN_CODE_FENCE_RETAINED:
        reasons.append("code_fence_loss")
    if not heading_continuous(markdown):
        reasons.append("heading_discontinuous")
    signals = {
        "table_retention": tr,
        "code_fence_retention": cr,
        "heading_continuous": heading_continuous(markdown),
    }
    return (not reasons, reasons, signals)
