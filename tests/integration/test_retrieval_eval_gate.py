"""integration：评测 gate（G2，评测计算=被测对象）——语义分组 + 手工 benchmark 构造五类 diagnosis。

run_retrieval_eval 是纯计算（不触 runner daemon/panel/计费、不写 eval/results），正是要被断言的被测对象。
五类 diagnosis（retrieval_layer.py:264）：accept / recall_miss / file_miss / ranking_miss / low_precision。
"""
import pytest

from eval.core.benchmark import BenchmarkItem
from eval.core.retrieval.retrieval_layer import run_retrieval_eval
from retrieval.retriever import Retriever

_RAG_IDS = [f"doc_rag{index}:p0" for index in range(1, 6)]


def _corpus():
    corpus = {}
    for index in range(1, 6):
        corpus[f"doc_rag{index}"] = f"# RAG 文档{index}\n\nRAG 是检索增强生成，第 {index} 号文档内容。"
    corpus["doc_code"] = "# Python 教程\n\nPython 列表与字典是容器。"
    corpus["doc_plain"] = "# 纯文本\n\n这是一篇纯文本笔记。"
    return corpus


def _item(query_id, query, category, expected_parent_ids, files, child_ids=()):
    return BenchmarkItem(
        query_id=query_id, query=query, reference_facts="", source_doc="",
        category=category, difficulty="easy",
        expected_parent_ids=list(expected_parent_ids),
        relevance={chunk_id: 3 for chunk_id in expected_parent_ids},
        expected_files=list(files), expected_pages=[],
        expected_child_ids=list(child_ids),
    )


@pytest.fixture
def eval_store(semantic_fakes, ingest_docs, memory_store):
    semantic_fakes({"rag": ["RAG"], "code": ["Python"]})
    ingest_docs(_corpus())
    return memory_store


class TestDiagnosisFiveClasses:
    def test_distribution_has_all_five_classes(self, eval_store):
        items = [
            # accept：rag 查询命中全部 5 个 rag 父块，precision=1.0；且标注证据子块
            _item("Q1", "RAG 是什么", "accept", _RAG_IDS, ["doc_rag1.md"],
                  child_ids=["doc_rag1:p0:c0"]),
            # lowprec 组基线：干净命中（noise=0）
            _item("Q2", "RAG 检索", "lowprec", _RAG_IDS, ["doc_rag1.md"]),
            # low_precision：同一 rag 查询但只期望 1 个 → noise=0.8 > 0.4*1.5
            _item("Q3", "RAG 是什么", "lowprec", ["doc_rag1:p0"], ["doc_rag1.md"]),
            # recall_miss：期望不存在的子块 id，但文档本身被检索（file stem 命中）
            _item("Q4", "Python 列表", "recall", ["doc_code:p999"], ["doc_code.md"]),
            # file_miss：期望父块/文件完全不在检索结果中（stem 不重叠）
            _item("Q5", "Python 列表", "file", ["doc_zzz:p0"], ["nonexistent.md"]),
            # ranking_miss：doc_plain 在候选池（candidate_k=10）但不在 final top5
            _item("Q6", "RAG 是什么", "rank", ["doc_plain:p0"], ["doc_plain.md"]),
        ]
        retriever = Retriever(eval_store, rerank_enabled=False)
        output = run_retrieval_eval(retriever, items)

        # aggregate 全键
        for key in ("num_queries", "recall_at_k", "precision_at_k", "hit_at_k", "mrr",
                    "map_at_k", "ndcg_at_k", "num_child_annotated", "child_hit_at_k",
                    "child_recall_at_k", "diagnosis_distribution"):
            assert key in output.aggregate, f"aggregate 缺键: {key}"

        assert output.aggregate["num_queries"] == 6

        # 五类诊断齐全
        distribution = output.aggregate["diagnosis_distribution"]
        assert distribution["accept"] == 2        # Q1 + Q2
        assert distribution["low_precision"] == 1  # Q3
        assert distribution["recall_miss"] == 1    # Q4
        assert distribution["file_miss"] == 1      # Q5
        assert distribution["ranking_miss"] == 1   # Q6

        # child 指标仅 annotated query（Q1）参与
        assert output.aggregate["num_child_annotated"] == 1
        assert output.aggregate["child_hit_at_k"] == 1.0
        assert output.aggregate["child_recall_at_k"] == 1.0

    def test_grouped_outputs_present(self, eval_store):
        items = [_item("Q1", "RAG 是什么", "accept", _RAG_IDS, ["doc_rag1.md"])]
        retriever = Retriever(eval_store, rerank_enabled=False)
        output = run_retrieval_eval(retriever, items)
        assert "accept" in output.by_category
        assert "easy" in output.by_difficulty
        assert len(output.results) == 1
        assert output.results[0].diagnosis == "accept"

    def test_retrieval_eval_pure_computation(self, eval_store):
        """G2：run_retrieval_eval 只做纯计算返回 LayerOutput，不触达 runner daemon/计费。"""
        items = [_item("Q1", "RAG 是什么", "accept", _RAG_IDS, ["doc_rag1.md"])]
        retriever = Retriever(eval_store, rerank_enabled=False)
        output = run_retrieval_eval(retriever, items)
        assert output.results[0].final_context_text  # 上下文已构建（纯文本，无 LLM）
