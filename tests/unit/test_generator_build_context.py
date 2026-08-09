"""unit：Generator 上下文拼接（retrieval/generator.py _build_context）。"""
from indexing.chunk import Chunk, DocMetadata
from retrieval.generator import _build_context


def _chunk(content, title="", source=""):
    return Chunk(content=content, origin_metadata=DocMetadata(title=title, source=source))


class TestBuildContext:
    def test_single_chunk_with_title(self):
        assert _build_context([_chunk("body", title="DocA")]) == "[来源1] 文档: DocA\nbody"

    def test_title_falls_back_to_source(self):
        assert _build_context([_chunk("body", source="/path/doc.md")]) == \
            "[来源1] 文档: /path/doc.md\nbody"

    def test_unknown_source_fallback(self):
        assert "[未知来源]" in _build_context([_chunk("body")])

    def test_multiple_chunks_joined_with_separator(self):
        context = _build_context([_chunk("aaa", title="A"), _chunk("bbb", title="B")])
        assert context == "[来源1] 文档: A\naaa\n\n---\n\n[来源2] 文档: B\nbbb"

    def test_empty_list(self):
        assert _build_context([]) == ""
