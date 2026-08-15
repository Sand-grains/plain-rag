"""Eval CLI 主入口：retrieval / full 两种评估模式 + --precheck / --smoke / --no-report / --compare。

核心特性：
    - --mode retrieval：仅 Layer 1 检索评估（不调 LLM，免费），MonitorPanel.final_report() 输出终端报告
    - --mode full：Layer 1 + Layer 2 完整评估（调 Judge LLM，计费），MonitorPanel daemon 实时面板 + 最终报告
    - --precheck：独立前置自检（索引存在 + benchmark 标注有效 + 条目非空）, 返回 0/1
    - --mode full --smoke：生成链路冒烟（限前 N 条完整 retrieve->generate->judge，有 error 退 3, 不写 eval/results 报告、不起 MonitorPanel daemon，供 harness e2e 门禁）
    - --no-report：跳过 generate_report 与 timeline 落盘，保留终端报告（e2e 一律携带）
    - --compare RUN_A RUN_B：对比两次运行的指标差异
    - 外层 ThreadPoolExecutor（max_workers=EVAL_THREADPOOL_WORKERS）控 query 级并发，_evaluate_one 做 per-query 隔离
    - _load_previous_run() 加载最近一次 per_query.json 作为 Delta 基线

默认读 benchmark/private_v6.json，可用 --benchmark 指定其他文件。
结果写入 eval/results/timeline/<timestamp>/。

用法示例::

    uv run python -m eval.runner --mode retrieval
    uv run python -m eval.runner --mode full
    uv run python -m eval.runner --precheck
    uv run python -m eval.runner --mode full --smoke --limit 5
    uv run python -m eval.runner --compare 2026-07-26_120000 2026-07-26_150000

公共接口：
    - run_retrieval_mode: Layer 1 检索评估
    - run_full_mode: Layer 1 + Layer 2 完整评估（含 MonitorPanel）
    - run_precheck: 前置自检（索引/标注/条目）
    - run_smoke_mode: 生成链路冒烟（限条数，error 退 3）
    - run_compare: 对比两次运行
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import TOP_K, LLM_MODEL_ID, EVAL_LLM_MODEL_ID, EVAL_THREADPOOL_WORKERS, GENERATOR_TEMPERATURE, STORAGE_BACKEND
from eval.utils import read_json
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator

if TYPE_CHECKING:
    from eval.core.benchmark import BenchmarkItem, BenchmarkLoadResult
    from eval.core.llm_as_judge.judge import JudgeResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # eval/runner.py → eval/ → 项目根


# ---- 模块级状态 ----

_OUTER_POOL: ThreadPoolExecutor | None = None


# ---- 工具函数 ----

def _get_outer_pool() -> ThreadPoolExecutor:
    """返回模块级外层线程池单例，限制同时运行的 query 数。

    Returns:
        ThreadPoolExecutor：供 run_full_mode 提交 per-query 任务的线程池。
    """
    global _OUTER_POOL
    if _OUTER_POOL is None:
        _OUTER_POOL = ThreadPoolExecutor(
            max_workers=EVAL_THREADPOOL_WORKERS,
            thread_name_prefix="eval-outer",
        )
    return _OUTER_POOL


def _load_previous_run() -> dict[str, dict] | None:
    """加载最近一次运行的 per_query.json（排除当前这次），按 query_id 索引。

    Returns:
        dict[str, dict] | None：最近一次含 Layer 2 结果（faithfulness）的 per_query 数据；无则 None。
    """
    timeline = _PROJECT_ROOT / "eval" / "results" / "timeline"
    if not timeline.exists():
        return None
    runs = sorted(
        [directory for directory in timeline.iterdir() if directory.is_dir()],
        reverse=True,
    )
    for run_dir in runs:
        per_query_path = run_dir / "per_query.json"
        if per_query_path.exists():
            data = read_json(per_query_path)
            if data is None:
                continue
            if data and any(
                "faithfulness" in value for value in data.values()
                if isinstance(value, dict)
            ):
                return data
    return None


# ---- 单条评估 ----

def _evaluate_one(item: BenchmarkItem, retriever: Retriever, generator: Generator) -> JudgeResult:
    """单条 query 的完整 eval：retrieve → Generator 缓存/生成 → Judge。

    内含 Generator 缓存查/写 + 阶段耗时打点; 外层 try/except 隔离，异常时返回 error 判定的 JudgeResult。

    Args:
        item: 单条 benchmark 条目。
        retriever: 检索器，负责双路召回。
        generator: 生成器，负责产出回答（缓存 miss 时调用）。

    Returns:
        JudgeResult：含各阶段耗时与判定结果；异常时 verdict="error"。
    """
    from eval.core.llm_as_judge.judge import run_judge, JudgeResult
    from eval.core.llm_as_judge.judge_formatter import get_formatter, build_judge_context
    from eval.core.llm_as_judge.judge_cache import _cache_generator_key
    from eval.monitor import get_metrics
    from eval.monitor import get_panel
    from infra.cache import get_cache as get_cache_backend
    from infra.config import REDIS_DEFAULT_TTL

    try:
        # Stage: retrieve
        retrieve_start = time_module.time()
        chunks = retriever.retrieve(item.query, top_k=TOP_K)
        retrieve_ms = (time_module.time() - retrieve_start) * 1000

        # Generator cache
        context_str = build_judge_context(chunks)
        generator_cache_key = _cache_generator_key(item.query_id, context_str)
        cache_backend = get_cache_backend()
        generator_cached = cache_backend.get(generator_cache_key)

        metrics = get_metrics()

        # Stage: generate
        if generator_cached is not None:
            metrics.record_generator_cache_hit()
            answer = generator_cached
            generate_ms = 0.0
        else:
            metrics.record_generator_cache_miss()
            generate_start = time_module.time()
            answer = generator.generate(item.query, chunks, temperature=GENERATOR_TEMPERATURE)
            generate_ms = (time_module.time() - generate_start) * 1000
            metrics.record_llm_call("generator")
            cache_backend.set(generator_cache_key, answer, ttl_seconds=REDIS_DEFAULT_TTL)

        # Judge（temperature=0 保证确定性）
        result = run_judge(item.query_id, item.query, chunks, answer,
                          item.reference_facts, formatter=get_formatter(),
                          temperature=0.0)
        result.retrieve_ms = retrieve_ms
        result.generate_ms = generate_ms

        return result
    except Exception as error:
        panel = get_panel()
        if panel:
            panel.alert(item.query_id, f"Generator {type(error).__name__}: {error}")
        judge_result = JudgeResult(query_id=item.query_id)
        judge_result.generator_error = f"evaluate_one failed: {error}"
        judge_result.verdict = "error"
        return judge_result


# ---- 评估模式 ----

def _prepare_eval(benchmark_path: str):
    """前置自检 + 就绪: 索引存在 + benchmark 标注有效 + 条目非空; 未满足时打印原因并退出 1。

    抽取自 run_retrieval_mode/run_full_mode 的公共前奏, 供 precheck / smoke 复用;
    三者的前置判定与打印逐字一致, 保证 precheck 的"前置未满足"与 mode 运行的报错语义相同。

    Args:
        benchmark_path: benchmark 文件路径。

    Returns:
        tuple[Retriever, list[BenchmarkItem]]: 就绪的检索器与有效条目。

    Raises:
        SystemExit: 索引缺失 / benchmark 标注失效 / 条目为空时退出 1。
    """
    from eval.core.benchmark import load_benchmark

    print(f"加载索引（{STORAGE_BACKEND} 模式）...")
    store = IndexStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在（memory 模式下需先运行 agent_pipeline.py 入库）")
        sys.exit(1)
    retriever = Retriever(store)

    print(f"加载 benchmark: {benchmark_path}")
    result = load_benchmark(benchmark_path, valid_chunk_ids=store.chunk_ids)
    _abort_if_invalid_benchmark(result)
    items = result.valid_items
    total = len(items)
    print(f"  条目: {total}")
    if total == 0:
        print("错误: benchmark 无有效条目，中止")
        sys.exit(1)
    return retriever, items


def _abort_if_invalid_benchmark(result: BenchmarkLoadResult) -> None:
    """基准校验：expected_chunk_ids 与当前索引不一致 → 打印醒目警告并中止。

    分块策略变更后旧标注整体失效，继续跑会产出全零假数据，拒绝执行。

    Args:
        result: benchmark 加载结果，含 invalid_chunk_ids 校验信息。
    """
    if not result.invalid_chunk_ids:
        return
    print("=" * 64)
    print("⚠⚠  benchmark 含无效 expected_parent_ids（与当前索引的父块 chunk_id 集合不符）")
    print(f"     共 {len(result.invalid_chunk_ids)} 条含缺失 id：")
    for index, chunk_ids in list(result.invalid_chunk_ids.items())[:5]:
        shown = ", ".join(chunk_ids[:5]) + ("..." if len(chunk_ids) > 5 else "")
        print(f"       条目 #{index + 1}: 缺失 {shown}")
    print("     分块策略已变更（父子块），旧标注整体失效。请先重标注：")
    print("       uv run python benchmark/anno_tool.py --output benchmark/private_v6.json")
    print("     已中止本次评估——拒绝产出全零假数据。")
    print("=" * 64)
    sys.exit(1)


def run_retrieval_mode(benchmark_path: str, no_report: bool = False) -> str | None:
    """仅 Layer 1 检索评估（不调 LLM，免费），输出终端报告并落盘结果。

    Args:
        benchmark_path: benchmark 文件路径。
        no_report: True 时跳过 generate_report 与 timeline 落盘，保留终端报告。

    Returns:
        str | None：本次运行结果目录（eval/results/timeline/<ts>）；no_report 时返回 None。
    """
    from eval.core.retrieval.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info
    from eval.monitor import get_metrics, reset_metrics
    from eval.monitor import MonitorPanel

    reset_metrics()
    metrics = get_metrics()

    retriever, items = _prepare_eval(benchmark_path)
    total = len(items)

    # MonitorPanel（仅 final_report, 无 daemon 线程）
    previous_per_query = _load_previous_run()
    panel = MonitorPanel(metrics, previous_per_query)
    panel.set_total(total)
    panel.set_meta(benchmark_name=benchmark_path, eval_mode="retrieval")

    print("执行 Layer 1 检索评估...")
    output = run_retrieval_eval(retriever, items)
    metrics.layer1_results = output.results

    panel.query_count = total
    panel.final_report()

    if no_report:
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / timestamp)
    run_info = build_run_info(benchmark_path)
    generate_report(output, results_dir, run_info)
    return results_dir


def run_full_mode(benchmark_path: str, no_report: bool = False) -> str | None:
    """Layer 1 + Layer 2 完整评估（调 Judge LLM，计费），实时面板 + 落盘报告。

    Args:
        benchmark_path: benchmark 文件路径。
        no_report: True 时跳过 generate_report 与 timeline 落盘，保留终端报告。

    Returns:
        str | None：本次运行结果目录（eval/results/timeline/<ts>）；no_report 时返回 None。
    """
    from eval.core.retrieval.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info
    from eval.core.llm_as_judge.judge import _get_client
    from eval.monitor import get_metrics, reset_metrics
    from eval.monitor import MonitorPanel, set_panel

    reset_metrics()
    metrics = get_metrics()

    retriever, items = _prepare_eval(benchmark_path)
    total = len(items)
    generator = Generator(model=LLM_MODEL_ID)

    # 主线程预初始化 OpenAI client，避免并行区域多线程竞争
    _get_client()

    # 加载上次运行（Delta 基线）
    previous_per_query = _load_previous_run()

    # 模型预热：触达 embedding 模型加载（吸收 tqdm 进度条），避免打乱面板输出
    retriever.retrieve(items[0].query, top_k=TOP_K)

    # 创建 MonitorPanel（所有 print 需在 start() 前完成，否则被 ANSI 清屏覆盖）
    print("启动监控面板...")
    panel = MonitorPanel(metrics, previous_per_query)
    set_panel(panel)
    panel.set_total(total)
    panel.set_meta(
        benchmark_name=benchmark_path,
        generator_model=LLM_MODEL_ID or "unknown",
        judge_model=EVAL_LLM_MODEL_ID,
    )
    panel.start()

    layer1 = run_retrieval_eval(retriever, items)
    metrics.layer1_results = layer1.results

    judge_results = []
    pool = _get_outer_pool()
    futures = {}
    for item in items:
        future = pool.submit(_evaluate_one, item, retriever, generator)
        futures[future] = item

    try:
        for future in as_completed(futures):
            judge_result = future.result()
            judge_results.append(judge_result)

            # 推入唯一真源
            metrics.layer2_results.append(judge_result)
            if judge_result.retrieve_ms is not None:
                metrics.record_stage("retrieve", judge_result.retrieve_ms)
            if judge_result.generate_ms is not None:
                metrics.record_stage("generate", judge_result.generate_ms)
            if judge_result.judge_faithfulness_ms is not None:
                metrics.record_stage("judge_faithfulness", judge_result.judge_faithfulness_ms)
            if judge_result.judge_quality_ms is not None:
                metrics.record_stage("judge_quality", judge_result.judge_quality_ms)

            # 端到端延迟 = retrieve + generate + max(faith, qual)，逐 query 计算后取百分位
            if judge_result.retrieve_ms is not None and judge_result.generate_ms is not None:
                end_to_end = judge_result.retrieve_ms + judge_result.generate_ms + max(
                    judge_result.judge_faithfulness_ms or 0,
                    judge_result.judge_quality_ms or 0,
                )
                metrics.record_stage("end_to_end", end_to_end)

            panel.query_done()
    finally:
        panel.stop()  # 保证幂等

    # 最终报告（终端）
    panel.final_report()

    # 写入文件报告
    if no_report:
        return None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / timestamp)
    run_info = build_run_info(benchmark_path, run_mode="full")
    generate_report(layer1, results_dir, run_info,
                    judge_results=judge_results,
                    metrics_summary=metrics.summary_dict())

    print(f"\n报告: {results_dir}")
    return results_dir


SMOKE_DEFAULT_LIMIT = 5


def run_precheck(benchmark_path: str) -> None:
    """独立前置自检: 索引存在 + benchmark 标注有效 + 条目非空, 不跑 eval。

    供 harness e2e 的 precheck 字段引用; 未满足时打印原因并退出 1（与 mode 运行报错语义一致）。

    Args:
        benchmark_path: benchmark 文件路径。

    Raises:
        SystemExit: 前置未满足时退出 1。
    """
    _prepare_eval(benchmark_path)
    print("前置自检通过: 索引存在 + benchmark 标注有效 + 条目非空")


def run_smoke_mode(benchmark_path: str, limit: int = SMOKE_DEFAULT_LIMIT) -> None:
    """生成链路冒烟: 只跑前 limit 条完整 retrieve->generate->judge 路径。

    e2e 是冒烟门不是质量门: 有 verdict=error 或 generator_error 的条目 → 退 3, 否则退 0。
    独立驱动循环（不复用 run_full_mode 的 as_completed 循环, 其无条件调 panel.query_done）,
    不写 eval/results 质量报告（不建 timeline、不调 generate_report）、不起 MonitorPanel daemon;
    内部 _evaluate_one 的 get_panel 有 None 守卫, 无面板安全。

    Args:
        benchmark_path: benchmark 文件路径。
        limit: 只评估前 N 条（query 级）, 默认 SMOKE_DEFAULT_LIMIT。

    Raises:
        SystemExit: 有异常条目时退出 3。
    """
    from eval.core.llm_as_judge.judge import _get_client
    from eval.monitor import reset_metrics

    reset_metrics()
    retriever, items = _prepare_eval(benchmark_path)
    generator = Generator(model=LLM_MODEL_ID)
    _get_client()

    selected = items[:limit]
    print(f"[smoke] 冒烟评估前 {len(selected)}/{len(items)} 条 (limit={limit})...")
    error_count = 0
    for item in selected:
        result = _evaluate_one(item, retriever, generator)
        is_error = result.verdict == "error" or bool(result.generator_error)
        if is_error:
            error_count += 1
        print(f"  {item.query_id}: verdict={result.verdict or 'n/a'}"
              + ("  [error]" if is_error else ""))
    if error_count:
        print(f"[smoke] 结束: {error_count} 条异常 (verdict=error 或 generator_error), 冒烟未通过")
        sys.exit(3)
    print(f"[smoke] 结束: 无异常, 冒烟通过")


def run_compare(run_a: str, run_b: str) -> None:
    """对比两次运行的指标差异（终端表格输出）。

    Args:
        run_a: 第一次运行的 timeline 目录名。
        run_b: 第二次运行的 timeline 目录名。
    """
    base = str(_PROJECT_ROOT / "eval" / "results" / "timeline")
    dir_a = os.path.join(base, run_a)
    dir_b = os.path.join(base, run_b)

    summary_a = read_json(os.path.join(dir_a, "summary.json"))
    summary_b = read_json(os.path.join(dir_b, "summary.json"))

    if not summary_a or not summary_b:
        print("错误: 找不到 summary.json")
        sys.exit(1)

    agg_a = summary_a["aggregate"]
    agg_b = summary_b["aggregate"]

    metrics_list = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k",
                    "child_hit_at_k", "child_recall_at_k"]
    print(f"\n{'Metric':<18} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
    print("-" * 50)
    for metric_name in metrics_list:
        value_a = agg_a.get(metric_name, 0)
        value_b = agg_b.get(metric_name, 0)
        delta = value_b - value_a
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else " ")
        print(f"{metric_name:<18} {value_a:>10.4f} {value_b:>10.4f} {arrow} {abs(delta):>8.4f}")


# ---- CLI 入口 ----

def main() -> None:
    """CLI 入口：路由 --mode / --compare 到对应执行函数。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "openai", "jieba", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="plain-rag eval runner")
    parser.add_argument("--mode", choices=["retrieval", "full"], help="评估模式")
    parser.add_argument("--benchmark", default="benchmark/private_v6.json", help="benchmark 文件路径")
    parser.add_argument("--precheck", action="store_true", help="前置自检(索引/标注/条目), 不跑 eval")
    parser.add_argument("--smoke", action="store_true", help="生成链路冒烟(只跑前 limit 条, 隐含 --no-report)")
    parser.add_argument("--limit", type=int, default=SMOKE_DEFAULT_LIMIT, help="smoke 只评估前 N 条")
    parser.add_argument("--no-report", action="store_true", help="不写 eval/results 质量报告, 保留终端报告")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="对比两次运行")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.precheck:
        run_precheck(args.benchmark)
    elif args.mode == "retrieval":
        run_retrieval_mode(args.benchmark, no_report=args.no_report)
    elif args.mode == "full":
        if args.smoke:
            run_smoke_mode(args.benchmark, limit=args.limit)
        else:
            run_full_mode(args.benchmark, no_report=args.no_report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
