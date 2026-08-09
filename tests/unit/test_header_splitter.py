"""unit：标题切分器（indexing/splitter/header_splitter.py）。

F3：超长 section 由内部 RecursiveCharacterTextSplitter 切分，id 为 {doc_id}:{i}（非 p{i}）。
"""
import re

from indexing.splitter.header_splitter import HeadingSplitter
from indexing.splitter.utils import find_fenced_block_ranges


class TestParseHeadingsWithPaths:
    def test_hierarchy_stack(self):
        splitter = HeadingSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], 1000)
        text = "# A\n\n## B\n\n### C\n\n# D"
        items = splitter._parse_headings_with_paths(text, find_fenced_block_ranges(text))
        assert items == [(["A"], 0), (["A", "B"], 5), (["A", "B", "C"], 11), (["D"], 18)]

    def test_skips_code_block_hashes(self):
        splitter = HeadingSplitter([("#", "h1"), ("##", "h2")], 1000)
        text = "# Real\n\n```\n# fake\n```\n\n## Real2"
        paths = [path for path, _ in splitter._parse_headings_with_paths(
            text, find_fenced_block_ranges(text))]
        assert paths == [["Real"], ["Real", "Real2"]]


class TestBuildSections:
    def test_preamble_and_orphan_heading(self):
        splitter = HeadingSplitter([("#", "h1")], 1000)
        text = "preamble\n\n# A\n\ncontent\n\n# B"
        sections = splitter._build_sections(text, find_fenced_block_ranges(text))
        paths = [path for path, _, _ in sections]
        # preamble（空路径）+ section A（孤立标题 B 并入 A）
        assert paths == [[], ["A"]]
        assert "preamble" in sections[0][1]
        assert "# B" in sections[1][1]


class TestSplit:
    def test_regular_sections_use_p_ids(self):
        splitter = HeadingSplitter([("#", "h1"), ("##", "h2")], 1000)
        text = "# A\n\nbody a\n\n## B\n\nbody b"
        chunks = splitter.split(text, {"doc_id": "doc"})
        assert [chunk.chunk_id for chunk in chunks] == ["doc:p0", "doc:p1"]
        assert all(chunk.origin_metadata.chunk_level == "parent" for chunk in chunks)
        assert chunks[0].metadata["section_path"] == ["A"]
        assert chunks[1].metadata["section_path"] == ["A", "B"]

    def test_overflow_section_uses_colon_index_ids(self):
        # F3：超长 section 内部 _overflow_splitter 切分 → id 形如 {doc_id}:{i}
        splitter = HeadingSplitter([("#", "h1")], 500)
        text = "# A\n\n" + "x" * 2000
        chunks = splitter.split(text, {"doc_id": "doc"})
        assert len(chunks) > 1
        assert chunks[0].chunk_id == "doc:0"
        for chunk in chunks:
            assert re.match(r"^doc:\d+$", chunk.chunk_id)
            assert chunk.origin_metadata.chunk_level == "parent"
