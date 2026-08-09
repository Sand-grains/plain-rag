"""integration：external 后端（G6 明示缺口）——默认排除，手动三服务 + SLIGHT_RAG_EXTERNAL=1 才跑。

- `@pytest.mark.external` + `skipif(not SLIGHT_RAG_EXTERNAL)` 双重门控
- fixture 三端口探活（PgSQL/ES/Milvus），任一未开 → skip 而非 fail
- 用例写真实三库（幂等，写前 rollback_doc 清理），测试 doc_id 用 `ext_test_` 前缀便于排查
- 不满足环境时整个模块 skip，不影响默认 `uv run pytest -q`（addopts 已排除 external）
"""
import os
import socket
import time

import pytest
from indexing.chunk import Chunk, DocMetadata

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        not os.getenv("SLIGHT_RAG_EXTERNAL"),
        reason="external 回归为明示缺口：需 SLIGHT_RAG_EXTERNAL=1 且三服务在线",
    ),
]

_EXTERNAL_PORTS = [5432, 9200, 19530]  # PgSQL / ES / Milvus
_DIM = 1024  # Milvus collection 维度（BGE-M3）


@pytest.fixture(scope="module")
def external_services_up():
    """三端口探活：任一端口未开放 → skip（而非 fail）。"""
    for port in _EXTERNAL_PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(("localhost", port))
        except OSError:
            sock.close()
            pytest.skip(f"端口 {port} 未开放（external 三服务未启动，先跑 scripts/infra_check.py）")
        sock.close()


def _vector(hot_index=0):
    vector = [0.0] * _DIM
    vector[hot_index] = 1.0
    return vector


def _wait_dense_hit(store, query_vector, expected_chunk_id, top_k=3, timeout=10.0):
    """轮询 dense 检索直到命中预期子块：Milvus 最终一致性，刚插入向量未必立即可搜（auto-flush ~1s）。

    Returns:
        list[Chunk]：最后一次检索结果（命中的是"到点即返回"的确定性行为，未命中则返回超时前的最后结果）。
    """
    deadline = time.monotonic() + timeout
    children = []
    while time.monotonic() < deadline:
        children = store.search_dense(query_vector, top_k=top_k)
        if any(chunk.chunk_id == expected_chunk_id for chunk in children):
            return children
        time.sleep(0.5)
    return children


@pytest.fixture
def external_store(monkeypatch, external_services_up):
    """强制 external 后端构造 IndexStore（覆盖 autouse 的 memory patch）。"""
    from indexing import index_store as index_store_module
    from indexing.index_store import IndexStore
    monkeypatch.setattr(index_store_module, "STORAGE_BACKEND", "external")
    store = IndexStore()  # _init_external：连接三库 + ensure schema + cleanup_suspending
    yield store


class TestExternalBackend:
    def test_init_external_connects(self, external_store):
        assert hasattr(external_store, "_pgsql")
        assert hasattr(external_store, "_es")
        assert hasattr(external_store, "_milvus")
        # _init_external 会把 PgSQL 已 indexed 父块回填 _chunks_cache（真实语料非空，断言"是 dict"而非"空"）
        assert isinstance(external_store._chunks_cache, dict)

    def test_batch_add_and_search_external(self, external_store):
        doc_id = "ext_test_doc"
        parent = Chunk(
            chunk_id=f"{doc_id}:p0", doc_id=doc_id, content="外部模式测试内容 RAG 检索增强",
            origin_metadata=DocMetadata(title=doc_id, doc_type=".md"), metadata={},
        )
        child = Chunk(
            chunk_id=f"{doc_id}:p0:c0", doc_id=doc_id, content="外部模式测试内容 RAG 检索增强",
            origin_metadata=DocMetadata(title=doc_id, doc_type=".md"),
            metadata={"parent_id": f"{doc_id}:p0"},
        )
        try:
            external_store.batch_add([parent], [child], [_vector()])
            # ES BM25 → 返父块
            parents = external_store.search_sparse("外部模式", top_k=3)
            assert any(chunk.chunk_id == f"{doc_id}:p0" for chunk in parents)
            # Milvus 稠密 → 返子块（轮询等最终一致性，见 _wait_dense_hit）
            children = _wait_dense_hit(external_store, _vector(), f"{doc_id}:p0:c0")
            assert any(chunk.chunk_id == f"{doc_id}:p0:c0" for chunk in children)
            # get_children_external
            got = external_store.get_children(f"{doc_id}:p0")
            assert any(chunk.chunk_id == f"{doc_id}:p0:c0" for chunk in got)
        finally:
            # 清理测试数据，防污染真实语料
            from indexing.chunk_ingest_ex import rollback_doc
            rollback_doc(doc_id, external_store._pgsql, external_store._es, external_store._milvus)

    def test_cleanup_suspending_idempotent(self, external_store):
        from indexing.chunk_ingest_ex import cleanup_suspending
        cleanup_suspending(external_store._pgsql, external_store._es, external_store._milvus)
        cleanup_suspending(external_store._pgsql, external_store._es, external_store._milvus)
