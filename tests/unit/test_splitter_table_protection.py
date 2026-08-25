"""unit：splitter 表格 protected_ranges（008-2 条例四）。

小表保护（切分点不落表内）、超限大表降级按行切分、受保护表整块估算 token 超预算也降级、
.md/.txt（protect_tables=False）走原路径不保护。
"""
from indexing.chunk import DocMetadata
from indexing.splitter.recursive_splitter import RecursiveCharacterTextSplitter
from indexing.splitter.utils import find_protected_table_ranges, find_table_ranges


def _table(row_count: int = 20) -> str:
    lines = ["| 列1 | 列2 |", "| --- | --- |"]
    for i in range(row_count):
        lines.append(f"| x{i} | y{i} |")
    return "\n".join(lines)


def _split(text: str, protect_tables: bool, chunk_size: int = 150):
    splitter = RecursiveCharacterTextSplitter(chunk_size, 0)
    meta = {"doc_id": "t", "doc_meta": DocMetadata(protect_tables=protect_tables)}
    return splitter.split(text, meta)


class TestFindTableRanges:
    def test_detects_contiguous_table(self):
        text = "前文。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后文。"
        ranges = find_table_ranges(text)
        assert len(ranges) == 1
        start, end = ranges[0]
        block = text[start:end]
        assert block.startswith("| a | b |")
        assert block.rstrip("\n").endswith("| 1 | 2 |")

    def test_ignores_non_table_lines(self):
        text = "普通行\n| a | b |\n普通行\n| c | d |"
        ranges = find_table_ranges(text)
        assert len(ranges) == 2  # 被非表格行断开


class TestProtectedTable:
    def test_small_table_protected_never_cut(self):
        table = _table(20)
        text = "前言。\n\n" + table + "\n\n" + "后文内容。" * 40
        chunks = _split(text, protect_tables=True)
        assert "".join(c.content for c in chunks) == text  # 无内容丢失
        assert any(table in c.content for c in chunks)     # 整表落在一个块内

    def test_plain_text_table_not_protected(self):
        # .md/.txt 默认 protect_tables=False → 大表被切分, 不再整表独块
        table = _table(20)
        text = "前言。\n\n" + table + "\n\n" + "后文内容。" * 40
        chunks = _split(text, protect_tables=False)
        assert not any(table in c.content for c in chunks)

    def test_global_protect_tables_gate(self, monkeypatch):
        # 全局 PROTECT_TABLES=False 时即使 doc_meta.protect_tables=True 也不保护
        import indexing.splitter.recursive_splitter as rs_mod
        monkeypatch.setattr(rs_mod, "PROTECT_TABLES", False)
        table = _table(20)
        text = "前言。\n\n" + table + "\n\n" + "后文内容。" * 40
        chunks = _split(text, protect_tables=True)
        assert not any(table in c.content for c in chunks)


class TestDegradation:
    def test_oversized_table_degrades(self):
        table = _table(50)  # 字符数超过默认 TABLE_ATOMIC_MAX_CHARS
        text = table
        protected, degraded = find_protected_table_ranges(text, max_chars=100, token_budget=8192)
        assert protected == []
        assert degraded == 1

    def test_token_budget_degradation(self):
        # 字符未超 max_chars 但整块 token_estimate(len//2) 超 token_budget → 仍降级
        table = "| " + ("x" * 20000) + " |"  # len > 16384 → token > 8192
        protected, degraded = find_protected_table_ranges(table, max_chars=25000, token_budget=8192)
        assert protected == []
        assert degraded == 1
