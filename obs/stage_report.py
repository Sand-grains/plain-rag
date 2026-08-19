"""run 级阶段报告
把这次 run 所有 trace 的 durations/fallback_required/stage_errors 跨 query 聚合,落成一份 step_timing-<ts>.jsonl

stages — 每阶段跑了多少次 + 耗时分布(count/avg/p50/p95)

retry — 降级损失统计
- rerank_fallback_count: 触发 rerank 降级(fallback_required=True)的 query 数
- retry_wasted_ms

error_codes — 错误码聚合

报告结构示例:
{
  "stages": {
    "embed":  {"count": 110, "avg_ms": 42.1, "p50_ms": 38.0, "p95_ms": 61.2},
    "dense":  {"count": 110, "avg_ms": 18.4, ...},
    "rerank": {"count": 110, "avg_ms": 34.2, "p50_ms": 28.0, "p95_ms": 71.5}
    // generate 不出现在 retrieval 模式: 0 耗时被跳过
  },
  "retry": {
    "rerank_fallback_count": 4,
    "retry_wasted_ms": 210.0
  },
  "error_codes": {
    "ConnectionError": {"count": 3, "query_ids": ["Q1", "Q5"]}
  }
}

调用时机: eval runner 收尾在 finalize_traces 之前调 write_step_report(session_traces(), trace_type="eval")
(把 agent trace 排除在外,防 agent 的检索/生成阶段污染 eval 统计)

优势在可回放、带错误码、带降级损失

eval 层面(以 query 为粒度手动打点)MonitorMetrics 优势在端到端 + judge 阶段
trace 层面(从装饰器埋点反推) stage_report 优势在可回放、带错误码、带降级损失

"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import config
from obs._stats import avg_of, p95, percentile
from obs.trace import STAGE_DURATION_KEYS

logger = logging.getLogger(__name__)


def aggregate_stage_report(traces, trace_type: str | None = None) -> dict:
    """跨 query 聚合阶段计时: 每阶段 count/avg/p50/p95 + retry 差额 + error_code 统计。

    跳过 0 耗时(阶段未跑/不适用, 如 retrieval 模式的 generate), 避免把缺失阶段计入延迟分布;
    trace_type 过滤按 F23 判别, 防 agent 检索/生成阶段污染 eval 阶段统计。

    Args:
        traces: 只读消费 .durations/.fallback_required/.stage_errors/.trace_type/.query_id 的 trace 集合。
        trace_type: 只统计该类型的 trace("eval"/"agent"); None 统计全部。

    Returns:
        dict: {"stages": {stage: {count, avg_ms, p50_ms, p95_ms}},
               "retry": {"rerank_fallback_count", "retry_wasted_ms"},
               "error_codes": {error_code: {count, query_ids}}}(error_codes 无错误时不产出)。
    """
    selected = [trace for trace in traces
                if trace_type is None or getattr(trace, "trace_type", "eval") == trace_type]
    stages: dict[str, dict] = {}
    for stage in STAGE_DURATION_KEYS:
        values = [trace.durations.get(stage, 0.0) for trace in selected if trace.durations.get(stage, 0.0) > 0]
        if not values:
            continue
        stages[stage] = {
            "count": len(values),
            "avg_ms": _round1(avg_of(values)),
            "p50_ms": _round1(percentile(values, 50)),
            "p95_ms": _round1(p95(values)),
        }
    fallback_traces = [trace for trace in selected if getattr(trace, "fallback_required", False)]
    report = {
        "stages": stages,
        "retry": {
            "rerank_fallback_count": len(fallback_traces),
            "retry_wasted_ms": _round1(sum(trace.durations.get("rerank", 0.0) for trace in fallback_traces)),
        },
    }
    error_codes = _aggregate_error_codes(selected)
    if error_codes:
        report["error_codes"] = error_codes
    return report


def write_step_report(traces, out_dir: str | Path | None = None, trace_type: str | None = None) -> Path | None:
    """聚合并落盘 step_timing-<ts>.jsonl; 无 trace 时返回 None。

    Args:
        traces: 见 aggregate_stage_report。
        out_dir: 落盘根目录(内部追加 traces/日期子目录), 默认 config.OBS_LOG_DIR(测试可显式传临时目录)。
        trace_type: 见 aggregate_stage_report。

    Returns:
        Path | None: 落盘的 step_timing 路径; 无 trace 不产出返回 None。
    """
    if not traces:
        return None
    report = aggregate_stage_report(traces, trace_type=trace_type)
    logs_root = Path(config.OBS_LOG_DIR if out_dir is None else out_dir)
    day_dir = logs_root / "traces" / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    report_path = day_dir / f"step_timing-{timestamp}.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    logger.info("step_timing 落盘: %s", report_path)
    return report_path


def _aggregate_error_codes(traces) -> dict:
    """StageError.error_code 按错误码统计: 出现次数 + 关联 query_id(去重)。按次数降序、码字典序。"""
    by_code: dict[str, dict] = {}
    for trace in traces:
        for error in getattr(trace, "stage_errors", []):
            code = getattr(error, "error_code", "") or "unknown"
            entry = by_code.setdefault(code, {"count": 0, "query_ids": []})
            entry["count"] += 1
            if trace.query_id not in entry["query_ids"]:
                entry["query_ids"].append(trace.query_id)
    return {code: entry for code, entry in sorted(by_code.items(), key=lambda kv: (-kv[1]["count"], kv[0]))}


def _round1(value: float | None) -> float:
    """保留一位小数; None(avg_of 全空)按 0.0。"""
    if value is None:
        return 0.0
    return round(value, 1)
