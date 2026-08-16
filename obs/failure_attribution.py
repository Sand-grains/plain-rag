"""失败归因器: 按序判定五类失败, 产出 failure_type 标签
注: 查完整原因看 trace 的 stage_errors, 看分布才看 failure_type

判定顺序(命中即停): rerank_fallback → empty_recall → recall_absent → rerank_drop → generation_error。
降级与空召回先于召回/排序判定, 避免在残缺数据上误归因。

为什么是命中即停? 因为归因的目标是找 最先断掉的环节, 指导"先修哪一环"(优先级: 上游 > 下游)
为什么判定的优先级上游 > 下游? 因为如果上游断了(没召回、被排掉), 生成器拿到的上下文本身就是残缺的, 再直接把原因指向最底层没有意义且是错误的。

可能的多原因信息缺失也有补偿机制, 并没有真的丢:
1. evidence 留痕: 命中的那类会记录关键数据——missing_expected、dropped_by_rerank、pool_truncation、expected_hit_ratio。
这些是"主因"的诊断浓缩。
2. RagTrace 全量信息保留: 归因(一行 failure_type)只是附加在 trace 上的一个聚合标签, 完整 trace 仍在
  trace.jsonl——stage_errors 列表可以同时装多条不同阶段的错误(embed 挂 + rerank 降级 + generate 失败各一条
  StageError)。
  "多种原因同时存在"这个事实在 trace 里分毫未丢, 归因只是给它打一个"主因"标记(因为一个环节出问题往往会连带其他部分, 这就是可能造成排查迷惑的点)
3. 单标签便于统计: failure_type 的用途是按类型计数(看失败分布), 如果返回多标签, 聚合维度上就麻烦了。这是"主因标签 vs
  完整根因"的取舍——要查完整原因看 trace 的 stage_errors, 要看分布用 failure_type。

与 eval diagnosis 的边界: obs failure_type 是排障归因口径(基于完整召回集 recalled_ids),
retrieval_layer.diagnosis 是 eval 层1评分口径(基于 rerank 输出池), 同 query 两套判定可能不一致,
属设计使然, 不可互换。归因 evidence 记 eval_diagnosis 供交叉引用对账。

obs 不 import eval: judge_result 以鸭子类型消费(只读 .verdict/.judge_error/.generator_error)。
"""

from __future__ import annotations

from obs.trace import RagTrace

RERANK_FALLBACK = "rerank_fallback"
EMPTY_RECALL = "empty_recall"
RECALL_ABSENT = "recall_absent"
RERANK_DROP = "rerank_drop"
GENERATION_ERROR = "generation_error"

FAILURE_TYPES = (RERANK_FALLBACK, EMPTY_RECALL, RECALL_ABSENT, RERANK_DROP, GENERATION_ERROR)


def classify_failure(trace: RagTrace | None, expected_parent_ids: list[str],
                     judge_result=None) -> dict | None:
    """按序判定 trace 的失败类型;
      无失败或无可判定数据时返回 None。

    Args:
        trace: 完整链路数据; None 时直接返回 None(无 trace 不误报)。
        expected_parent_ids: benchmark 中标注的 expected 父块 id 列表(偶尔可能为空)。
        judge_result: JudgeResult 鸭子类型(只读 .verdict/.judge_error/.generator_error), 仅 full 模式提供。

    Returns:
        dict | None: {"query_id", "failure_type", "evidence"}；无失败返回 None。
    """
    if trace is None:
        return None

    # 1. 是否触发 rerank 降级(即精排打分失败时, 退回原 RRF), 有则不做召回/排序判定
    # 降级后, 所谓的 candidates_after_rerank 并没有真正经过重排(重排结果不可信, 在这种"重排"数据上再做召回/排序判定会误归因)
    if trace.fallback_required:
        return _result(trace.query_id, RERANK_FALLBACK, {"fallback_required": True})

    # 2. 空召回: 检索无结果
    if not trace.retrieved_chunks:
        return _result(trace.query_id, EMPTY_RECALL, {"retrieved_chunks_empty": True})

    recalled_set = set(trace.recalled_ids)

    # 3. 召回缺失: 完整召回集缺失任一 expected_parent_id (any miss)
    missing = [chunk_id for chunk_id in expected_parent_ids if chunk_id not in recalled_set]
    if missing:
        return _result(
            trace.query_id, RECALL_ABSENT,
            {
                "missing_expected": missing,
                "expected_hit_ratio": _hit_ratio(expected_parent_ids, recalled_set),
            },
        )

    # 4. 有效结果被 rerank 筛掉: 全部召回但存在单个或部分 expected_parent_id 未进最终 top_k(重排排掉或池截断)
    after_set = set(trace.candidates_after_rerank) or set(trace.retrieved_chunks)
    dropped = [chunk_id for chunk_id in expected_parent_ids if chunk_id not in after_set]
    if dropped:
        before_set = set(trace.candidates_before_rerank)
        evidence = {
            "dropped_by_rerank": dropped,
            "expected_hit_ratio": _hit_ratio(expected_parent_ids, after_set),
        }
        pool_truncated = [chunk_id for chunk_id in dropped if chunk_id not in before_set]
        if pool_truncated:
            evidence["pool_truncation"] = pool_truncated  # 池截断(在 recalled 但未进 rerank 池)归 rerank_drop
        return _result(trace.query_id, RERANK_DROP, evidence)

    # 5. 生成/Judge 失败(仅服务 full 模式, retrieval模式直接跳过)
    if judge_result is not None:
        verdict = getattr(judge_result, "verdict", "")
        judge_error = getattr(judge_result, "judge_error", None)
        generator_error = getattr(judge_result, "generator_error", None)
        if verdict == "error" or judge_error or generator_error: # 有答案, 但被判错 / 根本没产出答案(生成器调用抛异常) / 评不了分(Judge调用抛异常)
            evidence = {}
            if generator_error:
                evidence["generator_error"] = generator_error
            if judge_error:
                evidence["judge_error"] = judge_error
            return _result(trace.query_id, GENERATION_ERROR, evidence)

    return None


def _hit_ratio(expected_parent_ids: list[str], matched_set: set[str]) -> float:
    """expected 中被 matched_set 命中的比例(any-miss 语义, 命中即部分归因)。"""
    if not expected_parent_ids:
        return 0.0
    return len([chunk_id for chunk_id in expected_parent_ids if chunk_id in matched_set]) / len(expected_parent_ids)


def _result(query_id: str, failure_type: str, evidence: dict) -> dict:
    """组装归因结果结构。"""
    return {"query_id": query_id, "failure_type": failure_type, "evidence": evidence}
