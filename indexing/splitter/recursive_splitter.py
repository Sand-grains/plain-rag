"""递归字符切分器：迭代分隔符栈、code fence 保护、tail-rune overlap、孤儿标题合并。

本模块实现 BaseSplitter 抽象契约下的通用字符切分算子
RecursiveCharacterTextSplitter：按优先级从高到低的分隔符栈，把文本迭代
切分到不超过 chunk_size 的片段，全程不切断 fenced code block；切不动时
逐级降级到更细的分隔符，最终按字符硬切兜底。同时提供 overlap 与孤儿标题
合并两个后处理阶段，可被上层（Router / ParentChildMappingWrapper）分别
用作父块与子块的切分器。
"""
from typing import Sequence

from config import (
    EMBEDDING_MODEL_TOKEN_CONSTRAINT,
    PROTECT_TABLES,
    SEPARATORS,
    TABLE_ATOMIC_MAX_CHARS,
)
from indexing.chunk import Chunk
from .base import BaseSplitter
from .utils import (
    _HEADING_REGEX,
    find_fenced_block_ranges,
    find_protected_table_ranges,
    overlaps_code,
)


class RecursiveCharacterTextSplitter(BaseSplitter):
    """按分隔符栈递归切分, 并使分块大小符合 chunk_size（字符）, 代码块保持完整，支持 overlap。"""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0,
                 separators: Sequence[str] | None = None, chunk_level: str = "child"):
        super().__init__()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = list(separators) if separators is not None else SEPARATORS
        self._chunk_level = chunk_level

    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """对整段文本执行完整切分流水线，产出 Chunk 列表。

        Args:
            text: 待切分的整段文档文本。
            metadata: 文档元数据，须含 doc_id（用于生成 chunk_id 前缀），
                其余键会并入每个 chunk 的 metadata。

        Returns:
            list[Chunk]：按文档顺序的切分结果。chunk_id 形如
                f"{doc_id}:{i}"（i 为顺序下标），chunk_level 写入
                origin_metadata。
        """
        code_ranges = self._protected_ranges(text, metadata)
        pieces = self._split_text(text, code_ranges)
        pieces = self._merge_isolated_headings(pieces)
        pieces = self._merge_code_adjacent_micro_ws(pieces, code_ranges)
        pieces = self._apply_overlap(pieces, code_ranges)
        chunks = []
        for i, (content, start) in enumerate(pieces):
            chunk = self._package_to_chunk(content, metadata, start_char_index=start)
            chunk.chunk_id = f"{metadata['doc_id']}:{i}"
            chunk.origin_metadata.chunk_level = self._chunk_level
            chunks.append(chunk)
        return chunks

    # ---- 核心切分逻辑 ----

    def _protected_ranges(self, text: str, metadata: dict) -> list[tuple[int, int]]:
        """构建不可切分区间：fenced code block（恒保护）+ 表格（仅 protect_tables 开启）。

        表格保护只对新格式归一化 MD 开启（doc_meta.protect_tables=True）；
        .md/.txt 默认 False → 仅 code block 保护，分块字节级不变。
        超限大表 / 受保护后仍超 token 预算的表由 find_protected_table_ranges 剔除（按行切分）。
        """
        code_ranges = find_fenced_block_ranges(text) # 代码块, 恒保护
        doc_meta = metadata.get("doc_meta")
        if PROTECT_TABLES and doc_meta is not None and getattr(doc_meta, "protect_tables", False): # 表格: 仅新格式开
            protected_tables, _ = find_protected_table_ranges(
                text, TABLE_ATOMIC_MAX_CHARS, EMBEDDING_MODEL_TOKEN_CONSTRAINT
            )
            return code_ranges + protected_tables
        return code_ranges # .md/.txt 只保护代码块

    def _split_text(self, text: str, code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
        """用分隔符栈把长文本迭代切分到不超过 chunk_size 的片段。

        从最高优先级分隔符开始切，切不动（分隔符存在但切分点全落在代码块内，
        或分隔符栈耗尽）就降级到更细的分隔符，最终兜底按字符硬切，全程不切断代码块
        相邻的合格小片先攒进 good_pieces, 由 _merge_good
        合并以逼近 chunk_size，减少碎片。
        返回前按全局起始偏移排序，恢复因 LIFO 栈打乱的文档顺序。

        Args:
            text: 待切分的整段文本。
            code_ranges: fenced code block 的 [(start, end), ...] 区间列表，
                落在此区间内的切分点会被跳过。

        Returns:
            list[(content, start)]：切分后的片段及其在原始文档中的全局起始
            字符偏移。此阶段未做 overlap 与孤儿标题合并，由调用方 split()
            在后续后处理中完成。
        """
        final_result: list[tuple[str, int]] = []
        separator_stack = [(text, 0, list(self._separators))]
        while separator_stack:
            segment, segment_start, rest_separators = separator_stack.pop()
            if not segment:
                continue
            if len(segment) <= self._chunk_size:
                final_result.append((segment, segment_start))
                continue
            separator, next_separators = self._pick_separator(segment, rest_separators)
            if separator == "":
                final_result.extend(self._hard_split(segment, segment_start, code_ranges))
                continue
            offsets = self._separator_offsets(segment, separator, code_ranges, segment_start)
            if not offsets:
                # 分隔符存在但切分点全落在代码块内 → 降级到下一级分隔符
                separator_stack.append((segment, segment_start, next_separators))
                continue
            splits = []
            cursor = 0 # 切分游标
            for position in offsets:
                splits.append((segment[cursor:position + len(separator)], segment_start + cursor))
                cursor = position + len(separator)
            splits.append((segment[cursor:], segment_start + cursor))
            good_pieces: list[tuple[str, int]] = []
            for split_piece, split_piece_start in splits:
                if len(split_piece) <= self._chunk_size:
                    good_pieces.append((split_piece, split_piece_start))
                else:
                    final_result.extend(self._merge_good(good_pieces))
                    good_pieces = []
                    if next_separators:
                        separator_stack.append((split_piece, split_piece_start, next_separators))
                    else:
                        final_result.extend(self._hard_split(split_piece, split_piece_start, code_ranges))
            final_result.extend(self._merge_good(good_pieces))
        final_result.sort(key=lambda p: p[1])  # LIFO 栈按 start 重排，恢复文档顺序（overlap/孤儿合并/编号依赖顺序）
        return final_result

    def _pick_separator(self, segment: str, rest_separators: list[str]) -> tuple[str, list[str]]:
        """选当前段可用的最高优先级分隔符，并返回降级后的剩余分隔符栈。

        Args:
            segment: 当前待切分的文本段。
            rest_separators: 尚未尝试的分隔符列表（按优先级从高到低）。

        Returns:
            (separator, next_separators)：第一个出现在 segment 中的分隔符，
            及其之后（更低优先级）的剩余分隔符；若栈耗尽或全部未命中，
            返回 ("", [])，由调用方转入字符硬切。
        """
        for index, separator_candidate in enumerate(rest_separators):
            if separator_candidate == "":
                return "", []
            if separator_candidate in segment:
                return separator_candidate, rest_separators[index + 1:]
        return "", []

    def _separator_offsets(self, segment: str, separator: str, code_ranges: list[tuple[int, int]],
                           segment_start: int) -> list[int]:
        """返回分隔符在段内所有"不落在代码块内"的出现位置（相对偏移）。

        Args:
            segment: 当前待切分的文本段。
            separator: 本轮选中的分隔符。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。
            segment_start: 段的全局起始偏移，用于把相对位置换算为全局位置，
                以判断是否与代码块重叠。

        Returns:
            list[int]：分隔符在 segment 中的相对偏移列表，已过滤掉落在
            code_ranges 内的位置；未命中时为空列表。
        """
        offsets = []
        search_from = 0
        while True:
            idx = segment.find(separator, search_from)
            if idx == -1:
                break
            if not overlaps_code(segment_start + idx, segment_start + idx + len(separator), code_ranges):
                offsets.append(idx)
            search_from = idx + len(separator)
        return offsets

    def _hard_split(self, segment: str, segment_start: int, code_ranges: list[tuple[int, int]]
                    ) -> list[tuple[str, int]]:
        """按 chunk_size 等宽硬切，切分边界避开 fenced code block。

        Args:
            segment: 当前待切分的文本段。
            segment_start: 段的全局起始偏移。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。

        Returns:
            list[(content, start)]：硬切后的片段及其全局起始偏移。若切分
            边界落在代码块内，则把边界拉到代码块结尾并吞掉其后连续的
            换行，保证代码块完整且尾部 \n 不成为孤儿微块；
            若代码块过长导致边界无法推进，退化为 chunk_size 边界兜底。
        """
        pieces = []
        length = len(segment) # 当前段的字符总数
        cursor_index = 0
        while cursor_index < length:
            end_index = min(cursor_index + self._chunk_size, length)
            for code_range_start, code_range_end in code_ranges:
                if code_range_start <= segment_start + end_index < code_range_end:
                    end_index = min(code_range_end, segment_start + length) - segment_start
                    while end_index < length and segment[end_index] == "\n":
                        end_index += 1   # 吞尾随换行：孤儿 \n 在硬切源头消失
                    break
            if end_index <= cursor_index:
                end_index = min(cursor_index + self._chunk_size, length)
            pieces.append((segment[cursor_index:end_index], segment_start + cursor_index))
            cursor_index = end_index
        return pieces

    def _merge_good(self, good_pieces: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """贪心合并相邻的合格小片，逼近 chunk_size 以减少碎片。

        Args:
            good_pieces: 连续的合格片段（每片长度 ≤ chunk_size）列表。

        Returns:
            list[(content, start)]：合并后的片段及其全局起始偏移（取自首片）。
            末尾残留内容会被兜底刷出。
        """
        merged_chunks = []
        current_chunk = ""
        current_chunk_start = None
        for text_piece, start_offset in good_pieces:
            if current_chunk and len(current_chunk) + len(text_piece) > self._chunk_size:
                merged_chunks.append((current_chunk, current_chunk_start))
                current_chunk = text_piece
                current_chunk_start = start_offset
            else:
                if current_chunk_start is None:
                    current_chunk_start = start_offset
                current_chunk += text_piece
        if current_chunk:
            merged_chunks.append((current_chunk, current_chunk_start))
        return merged_chunks

    # ---- 后处理 ----

    def _apply_overlap(self, pieces: list[tuple[str, int]],
                       code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
        """为相邻片段应用重叠，但不跨代码块边界回填。

        Args:
            pieces: 切分后的 (content, start) 片段列表。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。
                前一块尾部窗口与任一代码区间重叠时跳过 overlap，防止代码
                尾巴 + 闭合围栏污染下一块（代码→正文）。

        Returns:
            list[(content, start)]：重叠后的片段列表。窗口起点
            start - min(overlap, len(prev)) 保证不越过 prev 起点（无越界
            误判）；start 保持原值（指向片段自身内容起点，不含 overlap 前缀）。
        """
        if self._chunk_overlap <= 0:
            return pieces
        result = []
        for content, start in pieces:
            if result:
                prev_content, _ = result[-1]
                window_start = start - min(self._chunk_overlap, len(prev_content))
                if not overlaps_code(window_start, start, code_ranges):
                    content = prev_content[-self._chunk_overlap:] + content
            result.append((content, start))
        return result

    @staticmethod
    def _merge_code_adjacent_micro_ws(pieces: list[tuple[str, int]],
                                      code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
        """把"紧跟代码块"的纯空白微块（ ≤2 字符）并入邻片。

        Args:
            pieces: 切分后的 (content, start) 片段列表。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。

        Returns:
            list[(content, start)]：微块被并入邻片后的片段列表。微块是超长
            代码块后的 \n\n 经 \n 级切分留下的 1 字符孤儿（start 距某
            code_range_end ≤ 2）；纯文本无代码块 → 无命中 → 逐字节不变。
            首片微块并入后一片。
        """
        result = list(pieces)
        index = 0
        while index < len(result):
            content, start = result[index]
            adjacent = any(0 <= start - code_range_end <= 2 for _, code_range_end in code_ranges)
            if content.strip() == "" and len(content) <= 2 and adjacent:
                if index > 0:
                    prev_content, prev_start = result[index - 1]
                    result[index - 1] = (prev_content + content, prev_start)
                elif index + 1 < len(result):
                    next_content, next_start = result[index + 1]
                    result[index + 1] = (content + next_content, next_start)
                result.pop(index)
            else:
                index += 1
        return result

    def _merge_isolated_headings(self, pieces: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """把孤立的标题行合并进前一个片段。

        Args:
            pieces: 切分后的 (content, start) 片段列表。

        Returns:
            list[(content, start)]：孤立标题（单行且匹配标题正则的片段）
            并入前一片段（以换行连接），其余片段原样保留。
        """
        result = []
        for content, start in pieces:
            if self._is_isolated_heading(content) and result:
                prev_content, prev_start = result[-1]
                result[-1] = (prev_content + "\n" + content, prev_start)
            else:
                result.append((content, start))
        return result

    @staticmethod
    def _is_isolated_heading(content: str) -> bool:
        """判断片段是否为"孤立标题"：单行内容且整行匹配标题正则。

        Args:
            content: 单个片段的内容。

        Returns:
            bool：内容不含换行且整行是标题（如 "# Title"）时为 True。
        """
        return content.count("\n") == 0 and bool(_HEADING_REGEX.match(content))
