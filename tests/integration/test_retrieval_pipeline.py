"""integration：检索 + 生成链路（语义分组 + Fake 全链）——命中断言、双路返回值、rerank 关闭、MERGE_BOOST 覆盖、FakeGenerator。"""
import config
from indexing.chunk import Chunk, DocMetadata
from retrieval.retriever import Retriever


class TestSemanticRetrieval:
    def test_retrieve_hits_semantic_group(self, semantic_fakes, ingest_docs, memory_store):
        semantic_fakes({"rag": ["RAG"], "code": ["Python"]})
        ingest_docs({
            "doc_rag": "RAG 是检索增强生成",
            "doc_code": "Python 列表与字典",
            "doc_plain": "纯文本笔记",
        })
        retriever = Retriever(memory_store, rerank_enabled=False)
        chunks = retriever.retrieve("RAG 是什么", top_k=2)
        assert any(chunk.chunk_id.startswith("doc_rag:") for chunk in chunks)

    def test_retrieve_returns_parents_not_children(self, semantic_fakes, ingest_docs, memory_store):
        semantic_fakes({"rag": ["RAG"]})
        ingest_docs({"doc_rag": "RAG 是检索增强生成"})
        retriever = Retriever(memory_store, rerank_enabled=False)
        chunks = retriever.retrieve("RAG 是什么", top_k=2)
        assert all(chunk.chunk_id.count(":c") == 0 for chunk in chunks)  # 父块

    def test_retrieve_with_dense_child_returns_both(self, semantic_fakes, ingest_docs, memory_store):
        semantic_fakes({"rag": ["RAG"]})
        ingest_docs({"doc_rag": "RAG 是检索增强生成"})
        retriever = Retriever(memory_store, rerank_enabled=False)
        parents, dense_children = retriever.retrieve_with_dense_child("RAG 是什么", top_k=2)
        assert parents
        assert dense_children
        assert all("dense_score" in chunk.metadata for chunk in dense_children)

    def test_rerank_disabled_path_no_error(self, semantic_fakes, ingest_docs, memory_store):
        semantic_fakes({"rag": ["RAG"]})
        ingest_docs({"doc_rag": "RAG 是检索增强生成"})
        retriever = Retriever(memory_store, rerank_enabled=False)
        assert retriever.retrieve("RAG 是什么", top_k=2)  # 不抛错即通过


class TestAutoMergeConfigOverride:
    def test_merge_boost_override_read_dynamically(self, monkeypatch, semantic_fakes, memory_store):
        """F1：config.MERGE_BOOST 进程内覆盖生效（retriever 调用时动态读）。"""
        semantic_fakes({"rag": ["RAG"]})
        parent = Chunk(chunk_id="doc:p0", content="RAG 内容",
                       origin_metadata=DocMetadata(), metadata={})
        children = [
            Chunk(chunk_id="doc:p0:c0", content="RAG 检索", metadata={"parent_id": "doc:p0"}),
            Chunk(chunk_id="doc:p0:c1", content="无关内容", metadata={"parent_id": "doc:p0"}),
            Chunk(chunk_id="doc:p0:c2", content="RAG 增强", metadata={"parent_id": "doc:p0"}),
        ]
        from retrieval.embedding import embed
        memory_store.batch_add([parent], children, embed([c.content for c in children]))

        retriever = Retriever(memory_store, rerank_enabled=True)
        # 覆盖前后各跑一次：动态读 config，不因 import 缓存旧值
        monkeypatch.setattr(config, "MERGE_BOOST", 0.0)
        parents_zero, _ = retriever.retrieve_with_dense_child("RAG 是什么", top_k=1)
        monkeypatch.setattr(config, "MERGE_BOOST", 2.0)
        parents_boosted, _ = retriever.retrieve_with_dense_child("RAG 是什么", top_k=1)
        assert parents_zero and parents_boosted
        assert parents_boosted[0].chunk_id == "doc:p0"
        monkeypatch.setattr(config, "MERGE_BOOST", 0.0)  # 复位，防污染


class TestFakeGenerator:
    def test_generate_returns_fixed_answer(self, semantic_fakes):
        from retrieval.generator import Generator
        generator = Generator()  # install_all_fakes 已替换为 FakeGenerator
        assert generator.generate("什么是 RAG", []) == "固定回答：什么是 RAG"
