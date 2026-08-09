"""integration：memory 后端 IndexStore 全链路（Router→split→batch_add→检索→持久化回环）。"""
import pickle

import numpy
import pytest

from indexing.chunk import Chunk, DocMetadata
from indexing.index_store import IndexStore


class TestIngestAndSearch:
    def test_ingest_mini_corpus_adds_parents(self, ingest_docs, memory_store, mini_corpus):
        added = ingest_docs(mini_corpus)
        assert added  # 至少一个父块
        assert memory_store.chunks  # 父块已入库

    def test_search_dense_returns_children_sorted_desc(self, semantic_fakes, ingest_docs, memory_store):
        semantic_fakes({"rag": ["RAG"], "code": ["Python"]})
        ingest_docs({"doc_rag": "RAG 是检索增强生成", "doc_code": "Python 列表与字典"})
        from retrieval.embedding import embed
        query_vector = embed(["RAG 查询"])[0]
        results = memory_store.search_dense(query_vector, top_k=5)
        assert results
        assert all("dense_score" in chunk.metadata for chunk in results)
        scores = [chunk.metadata["dense_score"] for chunk in results]
        assert scores == sorted(scores, reverse=True)
        # rag 组子块相似度最高
        assert results[0].metadata["dense_score"] > results[-1].metadata["dense_score"]

    def test_search_sparse_returns_parents(self, ingest_docs, memory_store):
        ingest_docs({"doc_a": "你好世界"})
        results = memory_store.search_sparse("你好", top_k=5)
        assert results
        assert all(chunk.chunk_id.startswith("doc_a:") for chunk in results)

    def test_get_parents_preserves_order(self, ingest_docs, memory_store):
        # 用 h1 文档强制 structured 路径 → 父块 id 为 {doc_id}:p{i}
        ingest_docs({"doc_a": "# 甲\n\n内容", "doc_b": "# 乙\n\n内容"})
        ids = ["doc_a:p0", "doc_b:p0"]
        parents = memory_store.get_parents(ids)
        assert [parent.chunk_id for parent in parents] == ids

    def test_get_children_sorted_by_seq(self, semantic_fakes, memory_store):
        parent = Chunk(chunk_id="doc:p0", content="parent", origin_metadata=DocMetadata(), metadata={})
        children = [
            Chunk(chunk_id="doc:p0:c2", content="c2", metadata={"parent_id": "doc:p0"}),
            Chunk(chunk_id="doc:p0:c0", content="c0", metadata={"parent_id": "doc:p0"}),
            Chunk(chunk_id="doc:p0:c1", content="c1", metadata={"parent_id": "doc:p0"}),
        ]
        from retrieval.embedding import embed
        memory_store.batch_add([parent], children, embed([c.content for c in children]))
        got = memory_store.get_children("doc:p0")
        assert [chunk.chunk_id for chunk in got] == ["doc:p0:c0", "doc:p0:c1", "doc:p0:c2"]

    def test_empty_batch_is_noop(self, memory_store):
        memory_store.batch_add([], [], [])
        assert memory_store.chunks == []
        assert memory_store.search_dense([0.0] * 8, top_k=3) == []
        assert memory_store.search_sparse("x", top_k=3) == []


class TestPersistence:
    def test_persistence_restore_roundtrip(self, semantic_fakes, ingest_docs, memory_store, tmp_path):
        semantic_fakes({"rag": ["RAG"]})
        ingest_docs({"doc_rag": "RAG 是检索增强生成"})
        cache_dir = str(tmp_path / "cache")
        memory_store.vector_persistence(cache_dir)
        restored = IndexStore.vector_restore(cache_dir)
        assert restored is not None
        assert restored.chunk_ids == memory_store.chunk_ids
        # 恢复后可检索
        assert restored.search_sparse("RAG", top_k=3)

    def test_restore_missing_cache_returns_none(self, tmp_path):
        assert IndexStore.vector_restore(str(tmp_path / "nope")) is None

    def test_restore_version_mismatch_returns_none(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "chunks.pkl").write_bytes(pickle.dumps((99, [], [])))  # 版本不符
        numpy.save(cache_dir / "vectors.npy", numpy.array([[]]))
        assert IndexStore.vector_restore(str(cache_dir)) is None
