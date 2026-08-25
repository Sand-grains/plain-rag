"""Splitter 共用工具：token 估算、fenced code block 范围检测、表格范围检测、标题正则。"""
import re

_HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_REGEX = re.compile(r"^```", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def find_fenced_block_ranges(text: str) -> list[tuple[int, int]]:
    """返回 fenced code block 的 [开始索引start, 结束索引end): 未闭合则延伸到文末"""

    ranges = [] # 返回列表, 每个元素表示文档中一个代码块的字符区间
    in_code = False # 表示"当前扫描位置是否在某个代码块内部"
    start = 0 # 暂存"当前正在扫描的代码块的起始索引"
    for match in _FENCE_REGEX.finditer(text): # m是finditer返回的迭代器，每次迭代返回一个match对象
        if in_code:
            ranges.append((start, match.end()))
            in_code = False
        else:
            start = match.start()
            in_code = True
    if in_code:
        ranges.append((start, len(text)))
    return ranges


def inside_code(pos: int, ranges: list[tuple[int, int]]) -> bool:
    # 判断单个位置 pos 是否落在任一代码块区间内
    return any(s <= pos < e for s, e in ranges)


def overlaps_code(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    # 判断区间 [a, b) 是否与任一代码块区间有交集
    return any(a < e and b > s for s, e in ranges)


def token_estimate(text: str) -> int:
    """token 估算(伪精确, 仅用于诊断而非监控数据)"""
    return len(text) // 2


def find_table_ranges(text: str) -> list[tuple[int, int]]:
    """返回 pipe-table 的 [开始索引start, 结束索引end) 区间列表。

    归一化 Markdown 表格是连续以 | 开头的行（含表头/分隔行/数据行）构成的块；
    只聚连续表格行，非表格行断开。
    """
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    position = 0
    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        if _TABLE_ROW_RE.match(content):
            if start is None:
                start = position
        elif start is not None:
            ranges.append((start, position))
            start = None
        position += len(raw_line)
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def find_protected_table_ranges(text: str, max_chars: int, token_budget: int
                                ) -> tuple[list[tuple[int, int]], int]:
    """按大小与 token 预算筛出"应受保护"的表格区间，并统计降级表数。

    小表(<= max_chars 且整块 token_estimate <= token_budget)进保护列表；
    超限大表或受保护后整块仍超 token 预算的表不保护（按行切分），计入降级数。

    Returns:
        (protected_ranges, degraded_table_count)：受保护表格区间 + 降级表数。
    """
    protected: list[tuple[int, int]] = []
    degraded = 0
    for start, end in find_table_ranges(text):
        block = text[start:end]
        if len(block) <= max_chars and token_estimate(block) <= token_budget:
            protected.append((start, end))
        else:
            degraded += 1
    return protected, degraded