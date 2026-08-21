"""Eval 结果序列化与指标提取: 把 eval 对象转成可写 JSON 的纯 dict。

核心特性：
    - serialize_result()/serialize_judge_result(): Layer 1 / Layer 2 单条结果 → dict
    - build_summary(): 聚合(aggregate + 分组 + Layer2(可选) + cost(可选)) → summary dict
    - build_run_info(): 配置快照(git commit + 分块/模型参数) → run_info dict
    - build_metrics_per_query(): 按 _PER_QUERY_WHITELIST 白名单瘦身 → 纯指标 per_query dict
    - 供 eval/reporter.py(文件落盘) 与 eval/runner.py(指标库收尾) 复用, 避免 runner 直接依赖 reporter 的序列化实现

与上层的关系: 纯 dict 消费, 不 import obs(保持 obs -> eval 分层单向); 序列化/聚合逻辑集中于此, reporter 退化为纯文件 I/O。
"""

from datetime import datetime

from eval.core.retrieval.retrieval_layer import LayerOutput, RetrievalEvalResult
from eval.utils import get_git_commit


# per_query 落指标库的白名单: 只留 数值指标/verdict/延迟/诊断, 显式丢弃 final_context_text/query 等正文(回放归 trace.jsonl)
_PER_QUERY_WHITELIST = (
    "recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k",
    "child_hit_at_k", "child_recall_at_k",
    "faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness",
    "verdict", "retrieve_ms", "generate_ms", "diagnosis",
)


def serialize_result(result: RetrievalEvalResult) -> dict:
    """将单条 Layer 1 结果序列化为可写入 JSON 的 dict。"""
    return {
        "query_id": result.query_id,
        "query": result.query,
        "category": result.category,
        "difficulty": result.difficulty,
        "recall_at_k": result.recall_at_k,
        "precision_at_k": result.precision_at_k,
        "hit_at_k": result.hit_at_k,
        "mrr": result.mrr,
        "map_at_k": result.map_at_k,
        "ndcg_at_k": result.ndcg_at_k,
        "child_hit_at_k": result.child_hit_at_k,
        "child_recall_at_k": result.child_recall_at_k,
        "child_annotated": result.child_annotated,
        "diagnosis": result.diagnosis,
        "final_chunk_ids": result.final_chunk_ids,
        "candidate_chunk_ids": result.candidate_chunk_ids,
        "retrieved_files": result.retrieved_files,
        "query_rewritten": result.query_rewritten,
        "query_rewritten_flag": result.query_rewritten_flag,
        "final_context_text": result.final_context_text,
    }


def serialize_judge_result(judge_result) -> dict:
    """序列化 JudgeResult 中需写入 per_query.json 的字段。"""
    return {
        "faithfulness": judge_result.faithfulness,
        "answer_relevancy": judge_result.answer_relevancy,
        "context_precision": judge_result.context_precision,
        "context_recall": judge_result.context_recall,
        "answer_correctness": judge_result.answer_correctness,
        "verdict": judge_result.verdict,
        "parse_error": judge_result.parse_error,
        "judge_error": judge_result.judge_error,
        "generator_error": judge_result.generator_error,
        "retrieve_ms": judge_result.retrieve_ms,
        "generate_ms": judge_result.generate_ms,
        "judge_faithfulness_ms": judge_result.judge_faithfulness_ms,
        "judge_quality_ms": judge_result.judge_quality_ms,
    }


def build_summary(output: LayerOutput, judge_results: list | None = None,
                  metrics_summary: dict | None = None) -> dict:
    """构建 summary dict(聚合 + 分组 + Layer2(可选) + cost(可选)), 供 generate_report 落盘与 metrics_sink 复用。

    Args:
        output: Layer 1 评估完整输出。
        judge_results: Layer 2 JudgeResult 列表(可选)。
        metrics_summary: MonitorMetrics.summary_dict()(可选)。

    Returns:
        dict: 含 aggregate/by_category/by_difficulty, 按传入追加 layer2/cost。
    """
    summary = {
        "aggregate": output.aggregate,
        "by_category": output.by_category,
        "by_difficulty": output.by_difficulty,
    }
    if judge_results is not None:
        summary["layer2"] = _build_layer2_summary(judge_results)
    if metrics_summary is not None:
        summary["cost"] = metrics_summary
    return summary


def _build_layer2_summary(judge_results: list) -> dict:
    """计算 Layer 2 聚合：5 项均值 + verdict 分布 + judge/generator errors。"""
    valid = [judge_result for judge_result in judge_results if judge_result.faithfulness is not None]
    count = len(valid) if valid else 0
    return {
        "num_valid": count,
        "num_total": len(judge_results),
        "faithfulness_avg": sum(judge_result.faithfulness or 0 for judge_result in valid) / count if count else None,
        "answer_relevancy_avg": sum(judge_result.answer_relevancy or 0 for judge_result in valid) / count if count else None,
        "context_precision_avg": sum(judge_result.context_precision or 0 for judge_result in valid) / count if count else None,
        "context_recall_avg": sum(judge_result.context_recall or 0 for judge_result in valid) / count if count else None,
        "answer_correctness_avg": sum(judge_result.answer_correctness or 0 for judge_result in valid) / count if count else None,
        "verdict_distribution": _count_verdicts(judge_results),
        "judge_errors": sum(1 for judge_result in judge_results if judge_result.judge_error),
        "generator_errors": sum(1 for judge_result in judge_results if judge_result.generator_error),
    }


def _count_verdicts(judge_results: list) -> dict:
    """统计 pass / partial / fail / error 各 verdict 数量。"""
    counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    for judge_result in judge_results:
        verdict = judge_result.verdict if judge_result.verdict in counts else "error"
        counts[verdict] += 1
    return counts


def build_run_info(benchmark_path: str, run_mode: str = "retrieval") -> dict:
    """构建 run_info 配置快照。

    Args:
        benchmark_path: benchmark 文件路径。
        run_mode: 运行模式（retrieval / full）。

    Returns:
        dict：含分块参数、模型 id、git commit、benchmark 路径的配置快照。
    """
    from config import CHILD_CHUNK_SIZE, CHILD_OVERLAP, TOP_K, LLM_MODEL_ID
    return {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": CHILD_CHUNK_SIZE,
        "chunk_overlap": CHILD_OVERLAP,
        "top_k": TOP_K,
        "model_id": LLM_MODEL_ID,
        "git_commit": get_git_commit(),
        "benchmark": benchmark_path,
        "test_mode": run_mode,
    }


def build_metrics_per_query(output: LayerOutput, judge_results: list | None) -> dict:
    """per_query 白名单瘦身: 复用 serialize_result/serialize_judge_result 后按 _PER_QUERY_WHITELIST 提取, 只留纯指标。

    Args:
        output: Layer 1 评估输出(含逐 query 结果)。
        judge_results: Layer 2 JudgeResult 列表(full 模式), 按 query_id 合并进对应记录。

    Returns:
        dict: query_id -> 纯指标记录(白名单子集)。
    """
    judge_by_query = {result.query_id: result for result in (judge_results or [])}
    per_query = {}
    for result in output.results:
        record = serialize_result(result)
        judge = judge_by_query.get(result.query_id)
        if judge is not None:
            record.update(serialize_judge_result(judge))
        per_query[result.query_id] = {
            key: record[key] for key in _PER_QUERY_WHITELIST if key in record
        }
    return per_query
