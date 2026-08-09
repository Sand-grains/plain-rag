"""Layer 1 检索评估编排：两轮评估 + 诊断分类，纯数学指标，不调 LLM。

核心特性：
    - 第一轮：计算 6 项 IR 指标 + 诊断分类（accept / recall_miss / file_miss / ranking_miss / low_precision）
    - 第二轮：对 low_precision 样本追加检测（判定是检索问题还是标注问题）
    - CANDIDATE_K = TOP_K * 2，候选池应比最终输出大，用于 ranking_miss 检测
    - 父子双轨: 父块指标 + dense 子块指标（child_hit_at_k / child_recall_at_k，仅对标注了 expected_child_ids 的 query 参与）
    - 聚合输出 LayerOutput（逐 query 结果 + 聚合指标 + 按 category/difficulty 分组）

用法示例::

    from eval.core.retrieval.retrieval_layer import run_retrieval_eval
    output = run_retrieval_eval(retriever, benchmark_items)
    print(output.aggregate["recall_at_k"])  # → 0.8256

公共接口：
    - RetrievalEvalResult: 单条 query 的检索评估结果
    - LayerOutput: 完整 Layer 1 输出（results + aggregate + by_category + by_difficulty）
    - run_retrieval_eval: 执行完整两轮检索评估
"""

import statistics
from dataclasses import dataclass, field

from eval.core.benchmark import BenchmarkItem
from eval.core.calculator.metrics import (
    recall_at_k, precision_at_k, hit_at_k,
    mrr, avg_precision, ndcg_at_k,
)
from retrieval.retriever import Retriever
from retrieval.generator import _build_context
from config import TOP_K
from eval.utils import stem


@dataclass
class RetrievalEvalResult:
    """单条 query 的 Layer 1 评估结果。"""
    query_id: str
    query: str
    category: str
    difficulty: str
    recall_at_k: float
    precision_at_k: float
    hit_at_k: int
    mrr: float
    map_at_k: float
    ndcg_at_k: float
    diagnosis: str  # recall_miss / file_miss / ranking_miss / low_precision / accept
    final_chunk_ids: list[str] = field(default_factory=list)
    candidate_chunk_ids: list[str] = field(default_factory=list)
    retrieved_files: list[str] = field(default_factory=list)
    child_hit_at_k: int = 0        # dense 子块候选是否命中任一证据子块
    child_recall_at_k: float = 0.0  # 证据子块中被 dense 候选命中的比例
    child_annotated: bool = False   # 该 query 是否标注了 expected_child_ids
    query_rewritten: str = ""
    query_rewritten_flag: str = "原"  # "原" 或 "rewritten"
    final_context_text: str = ""


@dataclass
class LayerOutput:
    """Layer 1 完整输出。"""
    results: list[RetrievalEvalResult]
    aggregate: dict  # 全局聚合指标
    by_category: dict[str, dict]  # category → 聚合指标
    by_difficulty: dict[str, dict]  # difficulty → 聚合指标


CANDIDATE_K = TOP_K * 2


def run_retrieval_eval(retriever: Retriever, items: list[BenchmarkItem]) -> LayerOutput:
    """执行 Layer 1 检索评估，含两轮 low_precision 判定。

    Args:
        retriever: 检索器，提供 retrieve_with_dense_child 双路召回。
        items: benchmark 条目列表。

    Returns:
        LayerOutput: 包含逐 query 结果、聚合指标、分组统计。
    """
    results = _first_pass(retriever, items)
    results = _second_pass_low_precision(results)
    aggregate = _aggregate(results)
    by_category = _group_by(results, key=lambda result: result.category)
    by_difficulty = _group_by(results, key=lambda result: result.difficulty)
    return LayerOutput(
        results=results,
        aggregate=aggregate,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


def _first_pass(retriever: Retriever, items: list[BenchmarkItem]) -> list[RetrievalEvalResult]:
    """第一轮：计算指标。ranking_miss 通过 candidate 级检索判定。

    Args:
        retriever: 检索器，提供 retrieve_with_dense_child 双路召回。
        items: benchmark 条目列表。

    Returns:
        list[RetrievalEvalResult]：逐 query 的检索评估结果，诊断含 pending（待第二轮判定）。
    """
    results = []
    for item in items:
        # 用 top_k * 2 做候选检索，前 TOP_K 为 final，剩余为 candidate-only
        all_parents, dense_children = retriever.retrieve_with_dense_child(item.query, top_k=CANDIDATE_K)
        final_chunks = all_parents[:TOP_K]
        candidate_only = all_parents[TOP_K:]

        relevant_ids = set(item.expected_parent_ids)
        retrieved_ids = [chunk.chunk_id for chunk in final_chunks]
        candidate_ids = [chunk.chunk_id for chunk in all_parents]
        # child 层指标输入切片到 CANDIDATE_K * 2 : reranker 开时 dense 候选扩到 40, 切片恢复 child 口径可比
        dense_child_ids = [chunk.chunk_id for chunk in dense_children[:CANDIDATE_K * 2]]
        expected_child_ids = set(item.expected_child_ids)
        child_annotated = bool(expected_child_ids)
        retrieved_files = list({chunk.origin_metadata.title + chunk.origin_metadata.doc_type for chunk in final_chunks})

        has_intersection = any(retrieved_id in relevant_ids for retrieved_id in retrieved_ids)
        has_candidate_intersection = any(retrieved_id in relevant_ids for retrieved_id in candidate_ids)
        expected_files_stems = {stem(filepath) for filepath in item.expected_files}
        retrieved_files_stems = {stem(filepath) for filepath in retrieved_files}
        file_miss_flag = not expected_files_stems & retrieved_files_stems if expected_files_stems else False

        if not has_intersection:
            if has_candidate_intersection:
                diagnosis = "ranking_miss"
            elif file_miss_flag:
                diagnosis = "file_miss"
            else:
                diagnosis = "recall_miss"
        else:
            diagnosis = "pending"  # 等第二轮 low_precision 判定

        results.append(RetrievalEvalResult(
            query_id=item.query_id,
            query=item.query,
            category=item.category,
            difficulty=item.difficulty,
            recall_at_k=recall_at_k(retrieved_ids, relevant_ids, TOP_K),
            precision_at_k=precision_at_k(retrieved_ids, relevant_ids, TOP_K),
            hit_at_k=hit_at_k(retrieved_ids, relevant_ids, TOP_K),
            mrr=mrr(retrieved_ids, relevant_ids),
            map_at_k=avg_precision(retrieved_ids, relevant_ids, TOP_K),
            ndcg_at_k=ndcg_at_k(retrieved_ids, item.relevance, TOP_K),
            diagnosis=diagnosis,
            final_chunk_ids=retrieved_ids,
            candidate_chunk_ids=candidate_ids,
            retrieved_files=retrieved_files,
            child_hit_at_k=hit_at_k(dense_child_ids, expected_child_ids, len(dense_child_ids)),
            child_recall_at_k=recall_at_k(dense_child_ids, expected_child_ids, len(dense_child_ids)),
            child_annotated=child_annotated,
            query_rewritten=item.query,
            query_rewritten_flag="原",
            final_context_text=_build_context(final_chunks),
        ))
    return results


def _second_pass_low_precision(results: list[RetrievalEvalResult]) -> list[RetrievalEvalResult]:
    """第二轮：按 category 分组计算噪声并比较基线，标记 low_precision。

    Args:
        results: 第一轮的检索评估结果（含 pending 诊断）。

    Returns:
        list[RetrievalEvalResult]：pending 条目被判定为 accept / low_precision / recall_miss。
    """
    # 按 category 分组收集噪声比（1 - precision）
    by_category: dict[str, list[float]] = {}
    for result in results:
        noise_ratio = 1.0 - result.precision_at_k
        by_category.setdefault(result.category, []).append(noise_ratio)

    # 计算每组的噪声比中位数作为基线
    baseline = {
        category: statistics.median(values) if values else 0.0
        for category, values in by_category.items()
    }

    for result in results:
        if result.diagnosis != "pending":
            continue
        category_baseline = baseline.get(result.category, 0.0)
        noise_ratio = 1.0 - result.precision_at_k
        if category_baseline > 0 and noise_ratio > category_baseline * 1.5:
            result.diagnosis = "low_precision"
        elif result.hit_at_k:
            result.diagnosis = "accept"
        else:
            result.diagnosis = "recall_miss"

    return results


def _aggregate(results: list[RetrievalEvalResult]) -> dict:
    """计算全局聚合指标（均值）。

    Args:
        results: 全部 query 的检索评估结果。

    Returns:
        dict：含 query 数、各 IR 指标均值、child 指标均值、诊断分布。
    """
    if not results:
        return {}
    return {
        "num_queries": len(results),
        "recall_at_k": statistics.mean(result.recall_at_k for result in results),
        "precision_at_k": statistics.mean(result.precision_at_k for result in results),
        "hit_at_k": statistics.mean(result.hit_at_k for result in results),
        "mrr": statistics.mean(result.mrr for result in results),
        "map_at_k": statistics.mean(result.map_at_k for result in results),
        "ndcg_at_k": statistics.mean(result.ndcg_at_k for result in results),
        "num_child_annotated": sum(1 for result in results if result.child_annotated),
        "child_hit_at_k": _avg_child(results, "child_hit_at_k"),
        "child_recall_at_k": _avg_child(results, "child_recall_at_k"),
        "diagnosis_distribution": _diagnosis_distribution(results),
    }


def _avg_child(results: list[RetrievalEvalResult], attribute: str) -> float:
    """child 层指标均值：仅对标注了 expected_child_ids 的 query 求均值，其余不参与。

    Args:
        results: 全部 query 的检索评估结果。
        attribute: 取值的属性名（child_hit_at_k 或 child_recall_at_k）。

    Returns:
        float：标注了证据子块的 query 的指标均值；无标注样本时返回 0.0。
    """
    values = [getattr(result, attribute) for result in results if result.child_annotated]
    return statistics.mean(values) if values else 0.0


def _group_by(results: list[RetrievalEvalResult], key) -> dict[str, dict]:
    """按给定 key 函数分组并计算每组聚合指标。

    Args:
        results: 全部 query 的检索评估结果。
        key: 取分组键的函数（如 result.category）。

    Returns:
        dict[str, dict]：分组键 → 该组的聚合指标。
    """
    groups: dict[str, list[RetrievalEvalResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)
    return {category: _aggregate(values) for category, values in groups.items()}


def _diagnosis_distribution(results: list[RetrievalEvalResult]) -> dict[str, int]:
    """统计五种诊断分类的计数。

    Args:
        results: 全部 query 的检索评估结果。

    Returns:
        dict[str, int]：诊断名 → 计数（五种分类齐全，缺失分类计 0）。
    """
    dist = {"recall_miss": 0, "file_miss": 0, "ranking_miss": 0, "low_precision": 0, "accept": 0}
    for result in results:
        if result.diagnosis in dist:
            dist[result.diagnosis] += 1
    return dist


