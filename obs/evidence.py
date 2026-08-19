"""丢失环节定位（阶段证据保留计数返回是附加项）
输入: 一条 trace 的三视图 + benchmark expected_parent_ids
输出: ① 两关(召回/重排)各保住几个 expected → 计数画像
     ② lost_stage: 证据先丢在哪一环节 → 定位器(核心产出)
     计数是支撑定位的原始数据, lost_stage本身是结论

只读消费 trace 的 recalled_ids/candidates_after_rerank/retrieved_chunks 三视图, 不改链路。
与 failure_attribution 互补: failure_attribution 答"哪类失败"(上游优先),
                            覆盖答"证据丢在哪个阶段"(recall / rerank)。

产出 evidence_coverage 由 lifecycle.py: finalize_traces 并入 failure_attribution.evidence。
"""

from __future__ import annotations

from obs.trace import RagTrace


def compute_evidence_coverage(trace: RagTrace, expected_parent_ids: list[str]) -> dict | None:
    """计算证据覆盖: expected_parent_ids 在召回集与 rerank 后池的命中数, 并作丢失阶段定位。

    lost_stage 判定顺序(先上游后下游): recalled_count < expected_count -> lost_stage = "recall" (embed/dense/sparse 召回阶段丢);
                                    after_rerank_count < recalled_count -> lost_stage = "rerank" (rerank 筛掉或池截断);
    否则无丢失: lost_stage=None。

    Args:
        trace: 只读消费 .recalled_ids/.candidates_after_rerank/.retrieved_chunks。
        expected_parent_ids: benchmark 标注的 expected 父块 id 列表; 空则无可测, 返回 None。

    Returns:
        dict | None: {"expected", "recalled", "after_rerank", "lost_stage"}; expected 为空返回 None。
    """
    if not expected_parent_ids:
        return None
    expected_set = set(expected_parent_ids)
    recalled_set = set(trace.recalled_ids)
    # after_rerank 语义对齐 failure_attribution 的 rerank_drop 判据: candidates_after_rerank 为空时退最终返回集
    after_set = set(trace.candidates_after_rerank) or set(trace.retrieved_chunks)
    recalled_count = len(expected_set & recalled_set)
    after_rerank_count = len(expected_set & after_set)

    if recalled_count < len(expected_parent_ids):
        lost_stage = "recall"
    elif after_rerank_count < recalled_count:
        lost_stage = "rerank"
    else:
        lost_stage = None
    return {
        "expected": len(expected_parent_ids),
        "recalled": recalled_count,
        "after_rerank": after_rerank_count,
        "lost_stage": lost_stage,
    }
