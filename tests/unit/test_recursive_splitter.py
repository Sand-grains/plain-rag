"""unit：递归字符切分器（indexing/splitter/recursive_splitter.py）。"""
from indexing.splitter.recursive_splitter import RecursiveCharacterTextSplitter
from indexing.splitter.utils import find_fenced_block_ranges


class TestSplit:
    def test_short_text_single_chunk(self):
        splitter = RecursiveCharacterTextSplitter(100, 0)
        chunks = splitter.split("hello world", {"doc_id": "doc"})
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc:0"
        assert chunks[0].content == "hello world"
        assert chunks[0].metadata["start_char_index"] == 0
        assert chunks[0].origin_metadata.chunk_level == "child"

    def test_empty_text_returns_empty(self):
        splitter = RecursiveCharacterTextSplitter(100)
        assert splitter.split("", {"doc_id": "doc"}) == []

    def test_long_text_sequential_ids_and_ordered_starts(self):
        splitter = RecursiveCharacterTextSplitter(20, 0, separators=["\n"])
        text = "aaaaa\nbbbbb\nccccc\nddddd\neeeee"
        chunks = splitter.split(text, {"doc_id": "doc"})
        assert [chunk.chunk_id for chunk in chunks] == [f"doc:{index}" for index in range(len(chunks))]
        starts = [chunk.metadata["start_char_index"] for chunk in chunks]
        assert starts == sorted(starts)
        # 无 overlap 时每块 ≤ chunk_size
        assert all(len(chunk.content) <= 20 for chunk in chunks)
        # 内容按序拼接回原文
        assert "".join(chunk.content for chunk in chunks) == text

    def test_chunk_level_parent(self):
        splitter = RecursiveCharacterTextSplitter(100, 0, chunk_level="parent")
        chunks = splitter.split("abc", {"doc_id": "doc"})
        assert chunks[0].origin_metadata.chunk_level == "parent"


class TestSplitText:
    def test_code_block_not_split_across_pieces(self):
        splitter = RecursiveCharacterTextSplitter(100, 0, separators=["\n"])
        text = "before\n```python\n" + "x" * 300 + "\n```\nafter"
        code_ranges = find_fenced_block_ranges(text)
        pieces = splitter._split_text(text, code_ranges)
        code_start, code_end = code_ranges[0]
        code_content = text[code_start:code_end]
        # 代码块内容完整出现在某个 piece 中（未被切断）
        assert any(code_content in content for content, _ in pieces)


class TestPickSeparator:
    def test_highest_priority_selected(self):
        splitter = RecursiveCharacterTextSplitter(100)
        separator, rest = splitter._pick_separator("a\n\nb\nc", ["\n\n", "\n"])
        assert separator == "\n\n"
        assert rest == ["\n"]

    def test_not_found_returns_empty(self):
        splitter = RecursiveCharacterTextSplitter(100)
        separator, rest = splitter._pick_separator("abc", ["\n\n"])
        assert separator == ""
        assert rest == []

    def test_empty_string_terminates_stack(self):
        splitter = RecursiveCharacterTextSplitter(100)
        separator, rest = splitter._pick_separator("abc", ["", "x"])
        assert separator == ""
        assert rest == []


class TestSeparatorOffsets:
    def test_skips_code_overlapping_offsets(self):
        splitter = RecursiveCharacterTextSplitter(100)
        # 段 "aaaa\nbbbb\ncccc" 的 "\n" 在 idx 4 与 idx 9；[2,6) 只覆盖 idx 4
        offsets = splitter._separator_offsets("aaaa\nbbbb\ncccc", "\n", [(2, 6)], 0)
        assert offsets == [9]


class TestHardSplit:
    def test_boundary_pulled_out_of_code_block(self):
        splitter = RecursiveCharacterTextSplitter(6)
        # chunk_size=6 的边界落点 6 在代码块 [4,12) 内 → 拉到代码块尾 12
        pieces = splitter._hard_split("aaaa```bbbb", 0, [(4, 12)])
        assert pieces[0][0] == "aaaa```bbbb"
        assert pieces[0][1] == 0

    def test_swallows_trailing_newlines_after_code(self):
        splitter = RecursiveCharacterTextSplitter(6)
        pieces = splitter._hard_split("aaaa```bbbb\n\n", 0, [(4, 12)])
        assert pieces[0][0] == "aaaa```bbbb\n\n"

    def test_plain_equal_width(self):
        splitter = RecursiveCharacterTextSplitter(4)
        pieces = splitter._hard_split("abcdefgh", 0, [])
        assert [content for content, _ in pieces] == ["abcd", "efgh"]
        assert [start for _, start in pieces] == [0, 4]


class TestMergeGood:
    def test_greedy_merge_until_chunk_size(self):
        splitter = RecursiveCharacterTextSplitter(3)
        pieces = splitter._merge_good([("a", 0), ("b", 1), ("c", 2), ("d", 3)])
        assert pieces == [("abc", 0), ("d", 3)]


class TestApplyOverlap:
    def test_prepends_prev_tail_keeps_start(self):
        splitter = RecursiveCharacterTextSplitter(100, chunk_overlap=2)
        pieces = splitter._apply_overlap([("AAAA", 0), ("BBBB", 4), ("CCCC", 8)], [])
        assert pieces == [("AAAA", 0), ("AABBBB", 4), ("BBCCCC", 8)]

    def test_skips_overlap_across_code_block(self):
        splitter = RecursiveCharacterTextSplitter(100, chunk_overlap=2)
        pieces = splitter._apply_overlap([("AAAA", 0), ("BBBB", 4)], [(2, 6)])
        assert pieces == [("AAAA", 0), ("BBBB", 4)]


class TestMergeCodeAdjacentMicroWs:
    def test_micro_ws_after_code_merged_into_previous(self):
        splitter = RecursiveCharacterTextSplitter(100)
        pieces = splitter._merge_code_adjacent_micro_ws(
            [("aaaa", 0), ("\n", 12), ("bbbb", 13)], [(0, 12)]
        )
        assert pieces == [("aaaa\n", 0), ("bbbb", 13)]

    def test_plain_text_unchanged(self):
        splitter = RecursiveCharacterTextSplitter(100)
        pieces = splitter._merge_code_adjacent_micro_ws([("a", 0), ("b", 1)], [])
        assert pieces == [("a", 0), ("b", 1)]


class TestMergeIsolatedHeadings:
    def test_heading_merged_into_previous(self):
        splitter = RecursiveCharacterTextSplitter(100)
        pieces = splitter._merge_isolated_headings([("para", 0), ("# Title", 10), ("body", 20)])
        assert pieces == [("para\n# Title", 0), ("body", 20)]

    def test_is_isolated_heading(self):
        splitter = RecursiveCharacterTextSplitter(100)
        assert splitter._is_isolated_heading("# T") is True
        assert splitter._is_isolated_heading("multi\n# T") is False
        assert splitter._is_isolated_heading("plain") is False
