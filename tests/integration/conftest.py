"""integration 层 conftest：autouse 装 install_all_fakes + ingest_docs 入库助手。

memory_store / mini_corpus / semantic_fakes 共享 fixture 在根 conftest（tests/conftest.py）。
"""
import pytest

from tests._fakes import install_all_fakes


@pytest.fixture(autouse=True)
def auto_install_fakes(monkeypatch):
    """每个 integration 测试前置安装 Fake 层（含 STORAGE_BACKEND → memory）。"""
    install_all_fakes(monkeypatch)


@pytest.fixture
def ingest_docs(memory_store):
    """把语料经 Router→split→batch_add 入库（embed 在调用时取 patch 后的 Fake）。

    Returns:
        callable(corpus: dict[str, str]) -> set[str]：入库后返回全部父块 chunk_id 集合。
    """
    from preprocess.md_diagnosis import diagnose
    from indexing.router import Router
    from indexing.chunk import DocMetadata

    def _ingest(corpus):
        from retrieval.embedding import embed  # 调用时导入 → 取 install_all_fakes 后的 Fake
        router = Router()
        added_chunk_ids = set()
        for doc_id, content in corpus.items():
            report = diagnose(content)
            splitter = router.route(report)
            base_meta = {"doc_id": doc_id, "doc_meta": DocMetadata(title=doc_id, doc_type=".md")}
            result = splitter.split(content, base_meta)
            parents, children = result if isinstance(result, tuple) else (result, result)
            if not parents:
                continue
            vectors = embed([child.content for child in children])
            memory_store.batch_add(parents, children, vectors)
            added_chunk_ids.update(parent.chunk_id for parent in parents)
        return added_chunk_ids

    return _ingest
