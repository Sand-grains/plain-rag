"""unit：chunk_id 序号解析（indexing/index_store.py _chunk_seq）。"""
from indexing.index_store import _chunk_seq


class TestChunkSeq:
    def test_parent_seq(self):
        assert _chunk_seq("doc:p3") == 3

    def test_child_seq(self):
        assert _chunk_seq("doc:p1:c2") == 2

    def test_colon_index_seq(self):
        assert _chunk_seq("doc:5") == 5

    def test_no_digits_returns_zero(self):
        assert _chunk_seq("doc:abc") == 0

    def test_multiple_digit_groups_take_tail(self):
        assert _chunk_seq("doc:p10:c25") == 25
