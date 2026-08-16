"""管理 Eval session(一次 eval run 的 trace 收集会话) 生命周期管理
- 诞生(reset_traces)     → run 开头清空收集器, 新 session 从干净起点开始
- 过程(trace_scope)      → 各 query 开/关, trace 一条条 append 进 session 收集器(accumulate)
- 终结(finalize_traces)  → run 结束, 取全部 trace → 归因 → 落盘 → 清空(consume + 释放)

eval/runner 作为边界只允许两行观测调用:
    - run 开头 reset_traces()(与 reset_metrics 并列)
    - 收尾 finalize_traces(benchmark_items, judge_results, layer1_results)
    (两模式均传 run_retrieval_eval 返回的 layer1.results)

trace 落盘独立于质量报告: 写入 logs/traces/trace-<ts>.jsonl(无论 --no-report 与否都写)
不并入 per_query.json(需对照时按 query_id join)。F13 在写前并入归因结果。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import config
from obs import trace as trace_module
from obs.failure_attribution import classify_failure
from obs.retention import prune_logs

logger = logging.getLogger(__name__)

# 已知敏感字段名(落盘脱敏用): 字段级 strip, 不动 query/chunk 正文
_SENSITIVE_KEY_PATTERNS = (
    "authorization", "api_key", "apikey", "token", "password", "secret", "credential",
)


def reset_traces() -> None:
    """在 eval run 开头清空 session 收集器(与 reset_metrics 并列)。
    注意: run_retrieval_mode 里 finalize_traces 没有任何 try/finally 保护,
    因此依然需要 reset_traces() 作防御式清场兜底,
    防止 run 在 finalize_traces 之前异常中断(如 benchmark 加载/检索崩溃), 残留 trace 混入下一次 run 的落盘

    """
    trace_module.clear_session_traces()


def finalize_traces(benchmark_items, judge_results, layer1_results=None, out_dir=None) -> Path | None:
    """收尾: 从 session收集器 取全部 trace → 归因 → 写 trace.jsonl → 清空 session。

    Args:
        benchmark_items: benchmark 条目列表(供 expected_parent_ids 归因用)。
        judge_results: JudgeResult 列表(full 模式提供; retrieval 模式传空), 归因 generation_error 用。
        layer1_results: Layer1 结果(鸭子类型, 只读 .query_id/.diagnosis), 供 evidence.eval_diagnosis 交叉引用。
        out_dir: 落盘目录, 默认 config.OBS_LOG_DIR/traces(测试可显式传入临时目录)。

    Returns:
        Path | None: 落盘的 trace.jsonl 路径; 无 trace 时返回 None。
    """
    traces = trace_module.session_traces()
    if not traces:
        trace_module.clear_session_traces()
        return None
    expected_by_query = {
        getattr(item, "query_id", ""): list(getattr(item, "expected_parent_ids", []))
        for item in benchmark_items
    }
    judge_by_query = {
        getattr(result, "query_id", ""): result for result in judge_results
    }
    diagnosis_by_query = {
        getattr(result, "query_id", ""): getattr(result, "diagnosis", None)
        for result in (layer1_results or [])
    }
    payloads = []
    for trace in traces:
        payload = trace.to_dict()
        attribution = classify_failure(
            trace,
            expected_by_query.get(trace.query_id, []),
            judge_by_query.get(trace.query_id),
        )
        if attribution is not None:
            diagnosis = diagnosis_by_query.get(trace.query_id)
            if diagnosis is not None:
                attribution["evidence"]["eval_diagnosis"] = diagnosis
            payload["failure_attribution"] = attribution
        payloads.append(payload)
    if config.OBS_SANITIZE:
        payloads = [_sanitize(payload) for payload in payloads]
    logs_root = Path(config.OBS_LOG_DIR if out_dir is None else out_dir)
    day_dir = logs_root / "traces" / datetime.now().strftime("%Y-%m-%d")  # 按天分包
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    trace_path = day_dir / f"trace-{timestamp}.jsonl"
    with open(trace_path, "w", encoding="utf-8") as file_handle:
        for payload in payloads:
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    trace_module.clear_session_traces()
    prune_logs(logs_dir=logs_root, max_traces=None, max_total=200)  # 任何时候总日志文件数硬上限
    logger.info("trace 落盘: %s (%d 条)", trace_path, len(payloads))
    return trace_path


def _sanitize(value):
    """脱敏
    落盘前做 字段脱敏(仅配置 OBS_SANITIZE 为开时): 递归 strip 已知敏感字段名. 不动 query/chunk 正文。"""
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if not any(pattern in str(key).lower() for pattern in _SENSITIVE_KEY_PATTERNS)
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value
