"""unit：retriever 纯助手（retrieval/retriever.py）——_parent_key / _corpus_signature / _apply_auto_merge。"""
import pytest

from indexing.chunk import Chunk
from retrieval.retriever import _parent_key, _corpus_signature, _apply_auto_merge


def _chunk(chunk_id, parent_id=None, dense_score=None, content="x"):
    metadata = {}
    if parent_id:
        metadata["parent_id"] = parent_id
    if dense_score is not None:
        metadata["dense_score"] = dense_score
    return Chunk(chunk_id=chunk_id, content=content, metadata=metadata)


class TestParentKey:
    def test_child_uses_parent_id(self):
        assert _parent_key(_chunk("doc:p0:c1", parent_id="doc:p0")) == "doc:p0"

    def test_flat_uses_own_id(self):
        assert _parent_key(_chunk("doc:p0")) == "doc:p0"


class _FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks


class TestCorpusSignature:
    def test_stable_regardless_of_insertion_order(self):
        a = Chunk(chunk_id="p2", content="B")
        b = Chunk(chunk_id="p1", content="A")
        signature_a = _corpus_signature(_FakeStore([a, b]))
        signature_b = _corpus_signature(_FakeStore([b, a]))
        assert signature_a == signature_b
        assert len(signature_a) == 12

    def test_changes_when_content_changes(self):
        signature_a = _corpus_signature(_FakeStore([Chunk(chunk_id="p1", content="A")]))
        signature_b = _corpus_signature(_FakeStore([Chunk(chunk_id="p1", content="B")]))
        assert signature_a != signature_b


class TestApplyAutoMerge:
    def test_non_adjacent_multihit_boosted(self):
        rrf_scores = {"doc:p1": 2.0, "doc:p2": 1.0}
        dense = [
            _chunk("doc:p1:c0", parent_id="doc:p1", dense_score=0.9),
            _chunk("doc:p1:c2", parent_id="doc:p1", dense_score=0.8),
        ]
        _apply_auto_merge(rrf_scores, dense, boost=1.5, min_hits=2)
        assert rrf_scores["doc:p1"] == pytest.approx(3.0)
        assert rrf_scores["doc:p2"] == 1.0

    def test_adjacent_hits_filtered_below_min_hits(self):
        rrf_scores = {"doc:p1": 2.0}
        dense = [
            _chunk("doc:p1:c0", parent_id="doc:p1", dense_score=0.9),
            _chunk("doc:p1:c1", parent_id="doc:p1", dense_score=0.8),
        ]
        _apply_auto_merge(rrf_scores, dense, boost=1.5, min_hits=2)
        assert rrf_scores["doc:p1"] == 2.0  # 相邻被过滤 → kept=1 < min_hits → 不 boost

    def test_only_pool_internal_parents_affected(self):
        rrf_scores = {"doc:p1": 2.0}
        dense = [
            _chunk("doc:p3:c0", parent_id="doc:p3", dense_score=0.9),
            _chunk("doc:p3:c2", parent_id="doc:p3", dense_score=0.8),
        ]
        _apply_auto_merge(rrf_scores, dense, boost=1.5, min_hits=2)
        assert rrf_scores == {"doc:p1": 2.0}  # p3 不在池内，不推进

    def test_boost_one_is_noop(self):
        rrf_scores = {"doc:p1": 2.0}
        dense = [
            _chunk("doc:p1:c0", parent_id="doc:p1", dense_score=0.9),
            _chunk("doc:p1:c2", parent_id="doc:p1", dense_score=0.8),
        ]
        _apply_auto_merge(rrf_scores, dense, boost=1.0, min_hits=2)
        assert rrf_scores["doc:p1"] == 2.0
