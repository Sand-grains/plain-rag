"""Eval CLI 主入口: 评估 + 指标库收尾接线的薄编排器
把一次 run 的编排、落盘、跨 run 指标持久化、指标对比 四件事串起来
序列化下沉到 serialization, 持久化下沉到 metrics_sink, 文件落盘下沉到 reporter

CLI 入口与模式路由(main)
评估编排（run_retrieval_mode / run_full_mode / run_smoke_mode / _prepare_eval）
指标库收尾接线（_record_run_metrics + _resolve_corpus_signature）
Delta 基线加载（_load_previous_run）
阶段计时 + 归因（收尾段）
compare（run_compare + _render_compare + _compare_section + _report_missing_run）

包含 retrieval / full 两种评估模式,
  --precheck / --smoke / --no-report / --compare 命令后缀作可选特性叠加

核心特性：
    - --mode retrieval：仅 Layer 1 检索评估（不调 LLM，免费），MonitorPanel.final_report() 输出终端报告
    - --mode full：Layer 1 + Layer 2 完整评估（调 Judge LLM，计费），MonitorPanel daemon 实时面板 + 最终报告
    - --precheck：独立前置自检（索引存在 + benchmark 标注有效 + 条目非空）, 返回 0/1
    - --mode full --smoke：生成链路冒烟（限前 N 条完整 retrieve->generate->judge，有 error 退 3, 不写 eval/results 报告、不起 MonitorPanel daemon，供 harness e2e 门禁）
    - --no-report：跳过 generate_report 与 timeline 落盘，保留终端报告（e2e 一律携带）
    - --compare RUN_A RUN_B：对比两次运行的指标差异
    - 外层 ThreadPoolExecutor（max_workers=EVAL_THREADPOOL_WORKERS）控 query 级并发，_evaluate_one 做 per-query 隔离
    - _load_previous_run() 从指标库加载最近一次 per_query 作为 Delta 基线

默认读 benchmark/private_v6.json (可用 --benchmark 指定其他文件)

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
import hashlib
import logging
import os
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import TOP_K, LLM_MODEL_ID, EVAL_LLM_MODEL_ID, EVAL_THREADPOOL_WORKERS, GENERATOR_TEMPERATURE, STORAGE_BACKEND, DEFAULT_BENCHMARK
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator

if TYPE_CHECKING:
    from eval.core.benchmark import BenchmarkItem, BenchmarkLoadResult
    from eval.core.llm_as_judge.judge import JudgeResult
    from eval.core.retrieval.retrieval_layer import LayerOutput
    from eval.core.subsets import BenchmarkSubset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # eval/runner.py → eval/ → 项目根

logger = logging.getLogger(__name__)


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
    """从指标库加载最近一次含 Layer 2 结果（faithfulness）的 run 的 per_query，按 query_id 索引。

    指标库是唯一跨 run 数据源; record_run 在收尾才写, 当前 run 天然不在库中, 无需排除本次。

    Returns:
        dict[str, dict] | None：最近一次含 faithfulness 的 per_query 数据；无则 None。
    """
    from obs.metrics_sink import list_runs
    for record in list_runs():
        per_query = record.get("per_query", {}) or {}
        if per_query and any(
            isinstance(value, dict) and "faithfulness" in value
            for value in per_query.values()
        ):
            return per_query
    return None


# ---- 指标库收尾(metrics_sink record_run) ----

def _resolve_corpus_signature(retriever) -> str:
    """语料签名: 优先复用 retriever 懒计算的 _corpus_signature(内容强哈希), 否则由 chunk_ids 派生。

    两条路径算法不同(内容 vs id 集合), 但 rerank 开关切换的跨模式比较本就该降 delta, 保守可接受。
    """
    signature = getattr(retriever, "_corpus_signature", None)
    if signature:
        return signature
    store = getattr(retriever, "index_store", None)
    chunk_ids = sorted(store.chunk_ids) if store is not None else []
    material = "|".join(chunk_ids)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def _record_run_metrics(*, run_id: str, benchmark_path: str, retriever, items, output,
                        judge_results, metrics, attribution: dict, test_mode: str) -> None:
    """收尾落指标库: 组装 summary/per_query/signatures 后 record_run。

    指标库是旁路: 任何组装/落库异常只 log-only 警告, 不打断 eval 主流程; 与 --no-report 解耦恒写。

    Args:
        run_id: 共享时间戳(%Y-%m-%d_%H%M%S, 与 timeline 目录同名)。
        benchmark_path: benchmark 文件路径。
        retriever: 检索器(供语料签名)。
        items: benchmark 条目列表(供 expected_parent_ids / num_queries)。
        output: Layer 1 评估输出。
        judge_results: Layer 2 JudgeResult 列表。
        metrics: MonitorMetrics 单例(供 summary_dict 成本/token/缓存快照)。
        attribution: finalize_traces 返回的归因表(query_id -> {failure_type, evidence})。
        test_mode: retrieval / full。
    """
    from eval.serialization import build_metrics_per_query, build_run_info, build_summary
    from obs.metrics_sink import record_run
    try:
        record_run(
            run_id=run_id,
            benchmark=benchmark_path,
            config=build_run_info(benchmark_path, run_mode=test_mode),
            corpus_signature=_resolve_corpus_signature(retriever),
            test_mode=test_mode,
            num_queries=len(items),
            expected_parent_ids_by_query={
                item.query_id: list(item.expected_parent_ids) for item in items
            },
            summary=build_summary(output, judge_results, metrics.summary_dict()),
            per_query=build_metrics_per_query(output, judge_results),
            attribution=attribution,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as error:
        logger.warning("指标库落库失败(旁路, 不打断): %s", error)


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
    from obs import get_metrics
    from obs import get_panel
    from obs.trace import trace_scope, trace_var
    from infra.cache import get_cache as get_cache_backend
    from infra.config import REDIS_DEFAULT_TTL

    try:
        with trace_scope(item.query_id, item.query):  # per-query trace: worker 内开, 包全链路
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

            # generate 计时旁路(不重构 generator.py): 现有 generate_ms 计时写入 trace
            current_trace = trace_var.get()
            if current_trace is not None:
                current_trace.durations["generate"] = generate_ms

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

def _restore_store_or_exit() -> IndexStore:
    """恢复本地索引 store; 若缺失, 打印原因并退出 1 ( _prepare_eval/_prepare_subsets 共用)。

    Returns:
        IndexStore: 恢复成功的本地 store。

    Raises:
        SystemExit: 索引缓存不存在时退出 1。
    """
    logger.info("加载索引（%s 模式）...", STORAGE_BACKEND)
    store = IndexStore.vector_restore()
    if store is None:
        logger.error("索引缓存不存在（memory 模式下需先运行 agent_pipeline.py 入库）")
        sys.exit(1)
    return store


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

    store = _restore_store_or_exit()
    retriever = Retriever(store)

    logger.info("加载 benchmark: %s", benchmark_path)
    result = load_benchmark(benchmark_path, valid_chunk_ids=store.chunk_ids)
    _abort_if_invalid_benchmark(result)
    items = result.valid_items
    total = len(items)
    logger.info("  条目: %d", total)
    if total == 0:
        logger.error("benchmark 无有效条目，中止")
        sys.exit(1)
    return retriever, items


def _abort_if_invalid_benchmark(result: BenchmarkLoadResult) -> None:
    """基准校验：expected_chunk_ids 与当前索引不一致 → 打印醒目警告并中止。

    分块策略变更后旧标注整体失效，继续跑会产出假数据，拒绝执行。

    Args:
        result: benchmark 加载结果，含 invalid_chunk_ids 校验信息。
    """
    if not result.invalid_chunk_ids:
        return
    logger.warning("=" * 64)
    logger.warning("⚠⚠  benchmark 含无效 expected_parent_ids（与当前索引的父块 chunk_id 集合不符）")
    logger.warning("     共 %d 条含缺失 id：", len(result.invalid_chunk_ids))
    for index, chunk_ids in list(result.invalid_chunk_ids.items())[:5]:
        shown = ", ".join(chunk_ids[:5]) + ("..." if len(chunk_ids) > 5 else "")
        logger.warning("       条目 #%d: 缺失 %s", index + 1, shown)
    logger.warning("     分块策略已变更（父子块），旧标注整体失效。请先重标注：")
    logger.warning("       uv run python benchmark/anno_tool.py --output benchmark/private_v6.json")
    logger.warning("     为拒绝产出全量假数据, 已终止本次评估")
    logger.warning("=" * 64)
    sys.exit(1)


def _prepare_subsets(benchmark_paths: list[str]) -> list[BenchmarkSubset]:
    """多子集前置自检 + 就绪: 加载本地索引 + 加载多个 benchmark 子集(私有/公开), 缺失则 skip。

    私有子集校验本地 IndexStore; 公开子集装配独立 memory store + 交集断言(见 eval/core/subsets);
    任一非跳过子集标注失效或条目为空时打印原因并退出 1。

    Args:
        benchmark_paths: benchmark 文件路径列表。

    Returns:
        list[BenchmarkSubset]: 加载后的子集列表(含 skipped 标记)。

    Raises:
        SystemExit: 索引缺失 / 任一子集标注失效 / 条目为空时退出 1。
    """
    from eval.core.subsets import load_subsets
    from config import PUBLIC_BENCHMARK_PATHS, PUBLIC_STORE_CACHE_DIR, SKIP_IF_MISSING_BENCHMARKS

    store = _restore_store_or_exit()

    subsets = load_subsets(
        benchmark_paths,
        local_store=store,
        public_paths=set(PUBLIC_BENCHMARK_PATHS),
        public_cache_dir=PUBLIC_STORE_CACHE_DIR,
        skip_if_missing=set(SKIP_IF_MISSING_BENCHMARKS),
    )
    for subset in subsets:
        if subset.skipped:
            logger.info("  跳过子集(缺失): %s", subset.path)
            continue
        _abort_if_invalid_benchmark(subset.load_result)
        total = len(subset.items)
        logger.info("  子集 %s (%s): %d 条", subset.path, subset.kind, total)
        if total == 0:
            logger.error("benchmark 无有效条目，中止: %s", subset.path)
            sys.exit(1)
    return subsets


def run_retrieval_mode(benchmark_paths: list[str], no_report: bool = False) -> str | None:
    """仅 Layer 1 检索评估（不调 LLM，免费），多子集分列统计 + 终端报告 + 落盘结果。

    Args:
        benchmark_paths: benchmark 文件路径列表(多子集, 私有/公开独立统计, 缺失即 skip)。
        no_report: True 时跳过 generate_report 与 timeline 落盘，保留终端报告。

    Returns:
        str | None：本次运行结果目录（eval/results/timeline/<ts>）；no_report 时返回 None。
    """
    from eval.core.retrieval.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report
    from eval.serialization import build_run_info
    from obs import get_metrics, reset_metrics
    from obs import MonitorPanel
    from obs.trace_lifecycle import reset_traces, finalize_traces
    from obs.trace_stage_report import write_step_report
    from obs.trace import session_traces, trace_scope

    reset_metrics()
    reset_traces()
    metrics = get_metrics()

    subsets = _prepare_subsets(benchmark_paths)
    active_subsets = [subset for subset in subsets if not subset.skipped]
    total = sum(len(subset.items) for subset in active_subsets)

    # MonitorPanel（仅 final_report, 无 daemon 线程）
    previous_per_query = _load_previous_run()
    panel = MonitorPanel(metrics, previous_per_query)
    panel.set_total(total)
    panel.set_meta(benchmark_name=",".join(benchmark_paths), eval_mode="retrieval")

    logger.info("执行 Layer 1 检索评估...")
    per_subset_outputs: list[tuple[BenchmarkSubset, LayerOutput]] = []
    all_results = []
    all_items = []
    for subset in active_subsets:
        output = run_retrieval_eval(subset.retriever, subset.items,
                                    per_query_ctx=lambda item: trace_scope(item.query_id, item.query))
        per_subset_outputs.append((subset, output))
        all_results.extend(output.results)
        all_items.extend(subset.items)
    metrics.layer1_results = all_results

    panel.query_count = total
    panel.final_report()

    # 分列统计(终端): 每子集独立指标, 合计为总规模, 不互相污染口径
    for subset, output in per_subset_outputs:
        aggregate = output.aggregate
        logger.info("  [%s] %s: recall@K=%.4f hit@K=%.4f MRR=%.4f NDCG@K=%.4f (%d query)",
                    subset.kind, subset.path, aggregate.get("recall_at_k", 0.0),
                    aggregate.get("hit_at_k", 0.0), aggregate.get("mrr", 0.0),
                    aggregate.get("ndcg_at_k", 0.0), len(subset.items))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # 共享 run_id: 同时喂 timeline 目录与指标库
    write_step_report(session_traces(), trace_type="eval")  # 阶段计时报告(finalize 消费收集器前只读聚合)
    _, attribution = finalize_traces(all_items, [], layer1_results=all_results, return_attribution=True)
    # 指标库按子集分列记录(每子集一条 run, 语料签名各自独立, 不互相污染)
    # M1: 归因表按子集 query_id 过滤, 避免把其他子集的归因条目写进本子集记录
    for subset, output in per_subset_outputs:
        subset_query_ids = {item.query_id for item in subset.items}
        subset_attribution = {
            query_id: att for query_id, att in attribution.items() if query_id in subset_query_ids
        }
        _record_run_metrics(run_id=f"{timestamp}-{subset.kind}-{Path(subset.path).stem}",
                            benchmark_path=subset.path, retriever=subset.retriever,
                            items=subset.items, output=output, judge_results=[], metrics=metrics,
                            attribution=subset_attribution, test_mode="retrieval")

    if no_report:
        return None
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / timestamp)
    for subset, output in per_subset_outputs:
        subset_dir = str(Path(results_dir) / f"{subset.kind}-{Path(subset.path).stem}")
        run_info = build_run_info(subset.path)
        generate_report(output, subset_dir, run_info)
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
    from eval.reporter import generate_report
    from eval.serialization import build_run_info
    from eval.core.llm_as_judge.judge import _get_client
    from obs import get_metrics, reset_metrics
    from obs import MonitorPanel, set_panel
    from obs.trace_lifecycle import reset_traces, finalize_traces
    from obs.trace_stage_report import write_step_report
    from obs.trace import session_traces

    reset_metrics()
    reset_traces()
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
    logger.info("启动监控面板...")
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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # 共享 run_id: 同时喂 timeline 目录与指标库
    write_step_report(session_traces(), trace_type="eval")  # 阶段计时报告(finalize 消费收集器前只读聚合)
    _, attribution = finalize_traces(items, judge_results, layer1_results=layer1.results, return_attribution=True)
    _record_run_metrics(run_id=timestamp, benchmark_path=benchmark_path, retriever=retriever,
                        items=items, output=layer1, judge_results=judge_results, metrics=metrics,
                        attribution=attribution, test_mode="full")

    # 写入文件报告
    if no_report:
        return None
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / timestamp)
    run_info = build_run_info(benchmark_path, run_mode="full")
    generate_report(layer1, results_dir, run_info,
                    judge_results=judge_results,
                    metrics_summary=metrics.summary_dict())

    logger.info("报告: %s", results_dir)
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
    logger.info("前置自检通过: 索引存在 + benchmark 标注有效 + 条目非空")


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
    from obs import reset_metrics

    reset_metrics()
    retriever, items = _prepare_eval(benchmark_path)
    generator = Generator(model=LLM_MODEL_ID)
    _get_client()

    selected = items[:limit]
    logger.info("[smoke] 冒烟评估前 %d/%d 条 (limit=%d)...", len(selected), len(items), limit)
    error_count = 0
    for item in selected:
        result = _evaluate_one(item, retriever, generator)
        is_error = result.verdict == "error" or bool(result.generator_error)
        if is_error:
            error_count += 1
        logger.info("  %s: verdict=%s%s", item.query_id, result.verdict or "n/a",
                    "  [error]" if is_error else "")
    if error_count:
        logger.error("[smoke] 结束: %d 条异常 (verdict=error 或 generator_error), 冒烟未通过", error_count)
        sys.exit(3)
    logger.info("[smoke] 结束: 无异常, 冒烟通过")


def run_compare(run_a: str, run_b: str) -> None:
    """对比两次运行的指标差异（终端分节表格, 数据源为指标库 metrics_sink）。

    参数语义为 run_id(指标库记录); 查不到时区分"非法 run_id"与"legacy timeline run"
    (存量 run 无 metrics 行, 升级后不可比, 提示需重跑 record_run 入库)。

    Args:
        run_a: 第一次运行的 run_id。
        run_b: 第二次运行的 run_id。
    """
    from obs.metrics_sink import compare_runs, get_run

    record_a = get_run(run_a)
    record_b = get_run(run_b)
    if record_a is None or record_b is None:
        _report_missing_run(run_a, record_a)
        _report_missing_run(run_b, record_b)
        sys.exit(1)

    result = compare_runs(record_a, record_b)
    _render_compare(result)


_LAYER2_METRICS = {"faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"}
_SECTION_ORDER = ("Layer1", "Layer2", "延迟", "成本", "归因")


def _report_missing_run(run_id: str, record) -> None:
    """报错区分非法 run_id 与 legacy timeline run(存量 run 无 metrics 行, 升级后不可比)。"""
    if record is not None:
        return
    legacy_dir = _PROJECT_ROOT / "eval" / "results" / "timeline" / run_id
    if legacy_dir.exists():
        logger.error("run_id %s 是 legacy timeline run(存量 run 无 metrics 行, 升级后不可比); 请重跑 record_run 入库", run_id)
    else:
        logger.error("run_id %s 不存在于指标库(非法 run_id)", run_id)


def _compare_section(metric_name: str) -> str:
    """指标归到输出分节: 归因/成本/Layer2/延迟/Layer1(诊断分布并入 Layer1)。"""
    if metric_name.startswith("attribution."):
        return "归因"
    if metric_name.startswith("cost."):
        return "成本"
    if metric_name.startswith("verdict.") or metric_name in _LAYER2_METRICS:
        return "Layer2"
    if metric_name in ("retrieve_ms", "generate_ms") or metric_name.startswith("stage_"):
        return "延迟"
    return "Layer1"


def _render_compare(result: dict) -> None:
    """渲染 compare_runs 结果: 表头对齐信息 + 分节表格, paired 行附校正后 p 值 + "**"(p<0.05)。"""
    logger.info("\n对比 %s vs %s", result["run_a"], result["run_b"])
    logger.info("  对齐: %d/%d-%d matched | 可对比: %s | 退化: %s",
                result["matched"], result["total_a"], result["total_b"],
                result["comparable"], result["degenerated"])
    if not result["comparable"]:
        logger.warning("  语料签名不一致, 不报显著性(降 delta)")
    grouped: dict[str, list[dict]] = {}
    for metric in result["metrics"]:
        grouped.setdefault(_compare_section(metric["name"]), []).append(metric)
    for section in _SECTION_ORDER:
        rows = grouped.get(section)
        if not rows:
            continue
        logger.info("\n[%s]", section)
        logger.info("%s", f"  {'Metric':<24}{'Run A':>10}{'Run B':>10}{'Delta':>12}  {'校正后 p':>11}")
        for metric in rows:
            p_label = "-"
            if metric["p_value"] is not None:
                p_label = f"{metric['p_value']:.4f}" + (" **" if metric["significant"] else "")
            logger.info("%s", f"  {metric['name']:<24}{metric['a']:>10.4f}{metric['b']:>10.4f}{metric['delta']:>+12.4f}  {p_label:>11}")


# ---- CLI 入口 ----

def main() -> None:
    """CLI 入口：路由 --mode / --compare 到对应执行函数。"""
    from obs.logging_setup import setup_logging
    from obs.trace_exit_guard import register_exit_guard
    setup_logging()
    register_exit_guard()  # atexit 兜底清空 trace 收集器 + SIGINT/SIGTERM 收尾
    for noisy in ("httpx", "openai", "jieba", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="plain-rag eval runner")
    parser.add_argument("--mode", choices=["retrieval", "full"], help="评估模式")
    parser.add_argument("--benchmark", nargs="+", default=[DEFAULT_BENCHMARK],
                        help="benchmark 文件路径(可多个, 多子集分列统计; 缺省 DEFAULT_BENCHMARK)")
    parser.add_argument("--precheck", action="store_true", help="前置自检(索引/标注/条目), 不跑 eval")
    parser.add_argument("--smoke", action="store_true", help="生成链路冒烟(只跑前 limit 条, 隐含 --no-report)")
    parser.add_argument("--limit", type=int, default=SMOKE_DEFAULT_LIMIT, help="smoke 只评估前 N 条")
    parser.add_argument("--no-report", action="store_true", help="不写 eval/results 质量报告, 保留终端报告")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="对比两次运行")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.precheck:
        run_precheck(args.benchmark[0])
    elif args.mode == "retrieval":
        run_retrieval_mode(args.benchmark, no_report=args.no_report)
    elif args.mode == "full":
        if args.smoke:
            run_smoke_mode(args.benchmark[0], limit=args.limit)
        else:
            run_full_mode(args.benchmark[0], no_report=args.no_report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
