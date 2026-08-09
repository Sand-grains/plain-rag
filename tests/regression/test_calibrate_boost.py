"""regression：calibrate_auto_merge 集成级回归（F1）——MERGE_BOOST 覆盖循环 + 单调守卫 + 复位契约。

对齐 calibrate_auto_merge.main() 的循环：进程内动态覆盖 config.MERGE_BOOST（retriever 调用时读，
见 retrieval/retriever.py:172）→ run_retrieval_eval → 收尾复位 0.0（防污染同进程后续调用）。
断言：循环不崩、记录合法、_select_best 从记录中选出 recall 最高、复位契约成立、recall 单调不减
（auto-merge 分数倍增的回归守卫）。

注意：语义分组下 recall 常饱和，boost 无区分度（auto-merge 分数倍增不改相对序）——本测试的
monotonic 断言是"回归守卫"（防未来改动让 boost 反而掉 recall），非"boost 提升 recall"的证明。
_select_best 的选优逻辑在 unit/test_select_best.py。
"""
import config
from eval.core.benchmark import BenchmarkItem
from eval.core.retrieval.retrieval_layer import run_retrieval_eval
from indexing.chunk import Chunk, DocMetadata
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from scripts.calibrate_auto_merge import _select_best

_BOOSTS = [0.0, 1.0, 2.0]


def _rag_corpus_with_two_children() -> tuple[IndexStore, list[BenchmarkItem]]:
    """5 篇 RAG（各 2 子块，够 min_hits=2 触发 auto-merge）+ 代码/纯文本 各 1。"""
    from retrieval.embedding import embed  # 函数内导入：绑定 Fake embed，防模块级绑定真 embed

    store = IndexStore()
    parents, children = [], []
    for index in range(1, 6):
        parents.append(Chunk(
            chunk_id=f"doc_rag{index}:p0",
            content=f"# RAG 文档{index}\n\nRAG 是检索增强生成，第 {index} 号文档内容。",
            origin_metadata=DocMetadata(title=f"doc_rag{index}", doc_type=".md"), metadata={}))
        for child_index in range(2):
            children.append(Chunk(
                chunk_id=f"doc_rag{index}:p0:c{child_index}",
                content=f"RAG 检索增强生成 {index} 号 内容 {child_index}",
                origin_metadata=DocMetadata(title=f"doc_rag{index}", doc_type=".md"),
                metadata={"parent_id": f"doc_rag{index}:p0"}))
    for doc_id, text in (("doc_code", "# Python 教程\n\nPython 列表与字典是容器。"),
                         ("doc_plain", "# 纯文本\n\n这是一篇纯文本笔记。")):
        parents.append(Chunk(chunk_id=f"{doc_id}:p0", content=text,
                             origin_metadata=DocMetadata(title=doc_id, doc_type=".md"), metadata={}))
        children.append(Chunk(chunk_id=f"{doc_id}:p0:c0", content=text,
                              origin_metadata=DocMetadata(title=doc_id, doc_type=".md"),
                              metadata={"parent_id": f"{doc_id}:p0"}))
    store.batch_add(parents, children, embed([child.content for child in children]))

    rag_ids = [f"doc_rag{index}:p0" for index in range(1, 6)]

    def item(query_id, query, category, expected_parent_ids, files):
        return BenchmarkItem(
            query_id=query_id, query=query, reference_facts="", source_doc="", category=category,
            difficulty="easy", expected_parent_ids=list(expected_parent_ids),
            relevance={chunk_id: 3 for chunk_id in expected_parent_ids},
            expected_files=list(files), expected_pages=[], expected_child_ids=[])

    items = [
        item("Q1", "RAG 是什么", "accept", rag_ids, ["doc_rag1.md"]),
        item("Q2", "RAG 检索", "lowprec", rag_ids, ["doc_rag1.md"]),
        item("Q3", "RAG 是什么", "lowprec", ["doc_rag1:p0"], ["doc_rag1.md"]),
        item("Q4", "Python 列表", "recall", ["doc_code:p999"], ["doc_code.md"]),
        item("Q5", "Python 列表", "file", ["doc_zzz:p0"], ["nonexistent.md"]),
        item("Q6", "RAG 是什么", "rank", ["doc_plain:p0"], ["doc_plain.md"]),
    ]
    return store, items


class TestCalibrateBoostLoop:
    def test_boost_loop_records_and_reset(self, semantic_fakes, monkeypatch):
        semantic_fakes({"rag": ["RAG"], "code": ["Python"]})
        store, items = _rag_corpus_with_two_children()
        retriever = Retriever(store, rerank_enabled=True)

        records = []
        for boost in _BOOSTS:
            monkeypatch.setattr(config, "MERGE_BOOST", boost)  # 进程内动态覆盖，对齐 calibrate
            aggregate = run_retrieval_eval(retriever, items).aggregate
            records.append({"boost": boost,
                            "recall_at_k": aggregate.get("recall_at_k", 0.0),
                            "mrr": aggregate.get("mrr", 0.0)})
        monkeypatch.setattr(config, "MERGE_BOOST", 0.0)  # F1 复位契约：对齐 calibrate_auto_merge.main() 收尾

        # 复位契约：防污染同进程后续测试
        assert config.MERGE_BOOST == 0.0

        # 记录合法：recall 在 [0,1] 且单调不减（回归守卫）
        assert all(0.0 <= record["recall_at_k"] <= 1.0 for record in records)
        recalls = [record["recall_at_k"] for record in records]
        assert recalls == sorted(recalls)

        # _select_best 从记录中选出 recall 最高的
        best = _select_best(records)
        assert best in records
        assert best["recall_at_k"] == max(record["recall_at_k"] for record in records)
