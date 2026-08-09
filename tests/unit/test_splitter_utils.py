"""unit：splitter 共用工具（indexing/splitter/utils.py）。"""
from indexing.splitter.utils import (
    find_fenced_block_ranges, inside_code, overlaps_code, token_estimate,
)


class TestFindFencedBlockRanges:
    def test_paired_blocks(self):
        text = "aa\n```\ncode\n```\nbb\n```\ncode2\n```\ncc"
        ranges = find_fenced_block_ranges(text)
        assert len(ranges) == 2
        first_start, first_end = ranges[0]
        assert text[first_start:first_start + 3] == "```"
        # 闭合区间 end 落在闭合围栏末尾
        assert text[first_end - 3:first_end] == "```"

    def test_unclosed_extends_to_end(self):
        text = "aa\n```\ncode never closes"
        ranges = find_fenced_block_ranges(text)
        assert ranges == [(3, len(text))]

    def test_no_fence_returns_empty(self):
        assert find_fenced_block_ranges("纯文本") == []


class TestInsideCode:
    def test_position_inside(self):
        ranges = [(5, 20)]
        assert inside_code(10, ranges) is True

    def test_position_at_end_boundary_not_inside(self):
        ranges = [(5, 20)]
        assert inside_code(20, ranges) is False

    def test_position_outside(self):
        ranges = [(5, 20)]
        assert inside_code(25, ranges) is False


class TestOverlapsCode:
    def test_overlapping_interval(self):
        assert overlaps_code(8, 15, [(5, 20)]) is True

    def test_adjacent_does_not_overlap(self):
        # [20,25) 与 [5,20) 恰好相接，无交集
        assert overlaps_code(20, 25, [(5, 20)]) is False

    def test_touching_start_boundary(self):
        assert overlaps_code(0, 5, [(5, 20)]) is False


class TestTokenEstimate:
    def test_floor_half(self):
        assert token_estimate("abcde") == 2

    def test_empty(self):
        assert token_estimate("") == 0
