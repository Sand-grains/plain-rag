"""纯文件落盘器, 只负责落盘: 把单次 run 的评估结果写成 4 个 JSON 文件
序列化/聚合下沉到 serialization, 跨 run 留存移交 metrics_sink

核心特性：
    - generate_report() 一站式写入：per_query.json / failures.json / summary.json / run_info.json
      (跨 run 指标留存已由 obs/metrics_sink 承担, history.jsonl 废止)
    - per_query.json 自动合并 Layer 1（检索指标 + 诊断）+ Layer 2（Judge 分数 + verdict + 阶段延迟）

终端输出职责已移交 MonitorPanel, 序列化/聚合逻辑已抽至 eval/serialization.py
reporter 仅做文件 I/O

用法示例::

    from eval.reporter import generate_report
    from eval.serialization import build_run_info
    run_info = build_run_info("bench.json", run_mode="full")
    generate_report(layer1_output, results_dir, run_info, judge_results=judge_results, metrics_summary=metrics.summary_dict())

唯一公共接口：
    - generate_report: 生成完整 eval 报告（4个文件）
"""

import os

from eval.core.retrieval.retrieval_layer import LayerOutput
from eval.serialization import build_summary, serialize_judge_result, serialize_result
from eval.utils import write_json


def generate_report(output: LayerOutput, results_dir: str, run_info: dict,
                    judge_results: list | None = None,
                    metrics_summary: dict | None = None) -> str:
    """生成完整 eval 报告：写入 summary.json / per_query.json / failures.json / run_info.json。

    Args:
        output: Layer 1 评估完整输出。
        results_dir: 结果输出目录。
        run_info: 运行配置快照。
        judge_results: Layer 2 JudgeResult 列表（可选，full mode 时传入）。
        metrics_summary: MonitorMetrics.summary_dict()（可选，full mode 时传入）。

    Returns:
        str：结果输出目录。
    """
    os.makedirs(results_dir, exist_ok=True)

    # per_query.json: query_id → 完整 trace
    per_query = {result.query_id: serialize_result(result) for result in output.results}
    write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # failures.json: diagnosis != "accept" 的子集
    failures = {result.query_id: serialize_result(result) for result in output.results if result.diagnosis != "accept"}
    write_json(os.path.join(results_dir, "failures.json"), failures)

    # summary.json: 聚合 + 分组 + Layer 2 + cost
    summary = build_summary(output, judge_results, metrics_summary)
    write_json(os.path.join(results_dir, "summary.json"), summary)

    # run_info.json: 配置快照
    write_json(os.path.join(results_dir, "run_info.json"), run_info)

    # per_query.json: 合并 Layer 2 数据
    if judge_results is not None:
        for judge_result in judge_results:
            if judge_result.query_id in per_query:
                per_query[judge_result.query_id].update(serialize_judge_result(judge_result))
        write_json(os.path.join(results_dir, "per_query.json"), per_query)

    return results_dir
