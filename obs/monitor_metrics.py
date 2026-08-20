"""Eval 进程运行时监控指标采集容器(run 进行中, 属于跑完即失的进程内存)
MonitorMetrics 为唯一真源(自 eval/monitor 迁入, 逻辑与类名不变)。

- token 明细: total_input/output_tokens + 每次调用的列表
- 阶段延迟列表: retrieve / generate / judge_faithfulness / judge_quality / end_to_end 各一份 ms 列表
- 缓存计数: Generator/Judge 各自的 hits/misses
- 错误/重试: error_count / error_types(dict)/ retry_count / parse_error_count
- LLM 调用计数: generator / judge_faithfulness / judge_quality 三路
- per-query 原始结果对象: layer1_results / layer2_results(对象列表, 唯一真源)
- 成本单价

核心特性：
    - 阶段延迟 ×4：stage_retrieve_ms / stage_generate_ms / stage_judge_faithfulness_ms / stage_judge_quality_ms
    - 缓存拆分：Generator 缓存（hits/misses）+ Judge 缓存（hits/misses）独立计数
    - LLM 调用按类型拆分：generator_llm_calls / judge_faithfulness_calls / judge_quality_calls
    - Per-query 原始结果：layer1_results / layer2_results（MonitorPanel 和 reporter 的唯一数据来源）
    - 聚合方法内部浅拷贝后遍历（list(results)），线程安全
    - 模块级单例 get_metrics() / reset_metrics()，每次 run_full_mode 开头重置

用法示例::

    from obs.monitor_metrics import get_metrics, reset_metrics
    reset_metrics()
    metrics = get_metrics()
    metrics.record_stage("retrieve", 120.5)
    metrics.record_llm_call("generator")
    print(metrics.layer1_avgs())       # → {"recall_at_k": 0.8256, ...}
    print(metrics.stage_percentiles())  # → {"retrieve": {"p50": 120.5, ...}, ...}

公共接口：
    - MonitorMetrics: 运行时指标容器（记录方法 + 聚合方法 + 计算属性）
    - get_metrics: 模块级单例获取
    - reset_metrics: 重置单例（每次评测运行开头调用）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from obs._stats import avg_of, percentile, p95


@dataclass
class MonitorMetrics:
    # ---- Token ----
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens_list: list[int] = field(default_factory=list)
    output_tokens_list: list[int] = field(default_factory=list)

    # ---- 阶段延迟（per-query 毫秒列表）----
    stage_retrieve_ms: list[float] = field(default_factory=list)
    stage_generate_ms: list[float] = field(default_factory=list)
    stage_judge_faithfulness_ms: list[float] = field(default_factory=list)
    stage_judge_quality_ms: list[float] = field(default_factory=list)
    stage_end_to_end_ms: list[float] = field(default_factory=list)

    # ---- 缓存（拆分为 Generator / Judge 独立计数）----
    generator_cache_hits: int = 0
    generator_cache_misses: int = 0
    judge_cache_hits: int = 0
    judge_cache_misses: int = 0

    # ---- 错误与重试 ----
    error_count: int = 0
    error_types: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    parse_error_count: int = 0

    # ---- LLM 调用计数（按类型拆分）----
    generator_llm_calls: int = 0
    judge_faithfulness_calls: int = 0
    judge_quality_calls: int = 0

    # ---- Per-query 原始结果（唯一真源，裸 list 类型避免 import 依赖）----
    layer1_results: list = field(default_factory=list)   # list[RetrievalEvalResult]
    layer2_results: list = field(default_factory=list)   # list[JudgeResult]

    # ---- 成本 ----
    input_price_per_1k: float = 0.0003
    output_price_per_1k: float = 0.0012

    # ---- 记录方法 ----
    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """记录一次 LLM 调用的 token 消耗。"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.input_tokens_list.append(input_tokens)
        self.output_tokens_list.append(output_tokens)

    def record_stage(self, stage: str, ms: float) -> None:
        """记录某阶段耗时（stage 对应 stage_{stage}_ms 字段名）。"""
        getattr(self, f"stage_{stage}_ms").append(ms)

    def record_generator_cache_hit(self) -> None:
        self.generator_cache_hits += 1

    def record_generator_cache_miss(self) -> None:
        self.generator_cache_misses += 1

    def record_judge_cache_hit(self) -> None:
        self.judge_cache_hits += 1

    def record_judge_cache_miss(self) -> None:
        self.judge_cache_misses += 1

    def record_llm_call(self, call_type: str) -> None:
        """记录一次 LLM 调用。

        Args:
            call_type: 'generator' | 'judge_faithfulness' | 'judge_quality'。
        """
        if call_type == "generator":
            self.generator_llm_calls += 1
        elif call_type == "judge_faithfulness":
            self.judge_faithfulness_calls += 1
        elif call_type == "judge_quality":
            self.judge_quality_calls += 1

    def record_retry(self) -> None:
        self.retry_count += 1

    def record_parse_error(self) -> None:
        self.parse_error_count += 1

    def record_error(self, error_type: str) -> None:
        self.error_count += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    # ---- 计算属性 ----
    @property
    def total_llm_calls(self) -> int:
        return self.generator_llm_calls + self.judge_faithfulness_calls + self.judge_quality_calls

    @property
    def generator_cache_hit_rate(self) -> float:
        total = self.generator_cache_hits + self.generator_cache_misses
        return self.generator_cache_hits / total if total > 0 else 0.0

    @property
    def judge_cache_hit_rate(self) -> float:
        total = self.judge_cache_hits + self.judge_cache_misses
        return self.judge_cache_hits / total if total > 0 else 0.0

    @property
    def rerank_cache_hit_rate(self) -> float | None:
        """rerank 缓存命中率；无任何 rerank 调用时返回 None（面板/报告显示 —）。"""
        from retrieval.reranker import get_rerank_cache_statistics
        stats = get_rerank_cache_statistics()
        total = stats["hits"] + stats["misses"]
        return stats["hits"] / total if total > 0 else None

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_llm_calls if self.total_llm_calls > 0 else 0.0

    @property
    def estimated_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost

    @property
    def input_tokens_p95(self) -> float:
        return p95(self.input_tokens_list)

    @property
    def output_tokens_p95(self) -> float:
        return p95(self.output_tokens_list)

    # ---- 聚合方法（MonitorPanel 面板和 render_final 共用）----
    def layer1_avgs(self) -> dict[str, float]:
        results = list(self.layer1_results)
        if not results:
            return {}
        count = len(results)
        return {
            "recall_at_k": sum(result.recall_at_k for result in results) / count,
            "precision_at_k": sum(result.precision_at_k for result in results) / count,
            "hit_at_k": sum(result.hit_at_k for result in results) / count,
            "mrr": sum(result.mrr for result in results) / count,
            "map_at_k": sum(result.map_at_k for result in results) / count,
            "ndcg_at_k": sum(result.ndcg_at_k for result in results) / count,
        }

    def layer2_avgs(self) -> dict[str, float | None]:
        results = list(self.layer2_results)
        valid_results = [result for result in results if result.faithfulness is not None]
        if not valid_results:
            return {}
        return {
            "faithfulness": avg_of([result.faithfulness for result in valid_results]),
            "answer_relevancy": avg_of([result.answer_relevancy for result in valid_results]),
            "context_precision": avg_of([result.context_precision for result in valid_results]),
            "context_recall": avg_of([result.context_recall for result in valid_results]),
            "answer_correctness": avg_of([result.answer_correctness for result in valid_results]),
        }

    def verdict_distribution(self) -> dict[str, int]:
        counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        for result in list(self.layer2_results):
            verdict = result.verdict if result.verdict in counts else "error"
            counts[verdict] += 1
        return counts

    def stage_percentiles(self) -> dict[str, dict[str, float]]:
        stages = {
            "retrieve": list(self.stage_retrieve_ms),
            "generate": list(self.stage_generate_ms),
            "judge_faithfulness": list(self.stage_judge_faithfulness_ms),
            "judge_quality": list(self.stage_judge_quality_ms),
            "end_to_end": list(self.stage_end_to_end_ms),
        }
        result = {}
        for name, values in stages.items():
            result[name] = {
                "p50": percentile(values, 50),
                "p75": percentile(values, 75),
                "p95": percentile(values, 95),
            }
        return result

    # ---- 序列化 ----
    def summary_dict(self) -> dict:
        stage_percentiles_map = self.stage_percentiles()
        rerank_hit_rate = self.rerank_cache_hit_rate
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_tokens_p95": self.input_tokens_p95,
            "output_tokens_p95": self.output_tokens_p95,
            "estimated_cost": round(self.estimated_cost, 6),
            "generator_cache_hit_rate": round(self.generator_cache_hit_rate, 4),
            "judge_cache_hit_rate": round(self.judge_cache_hit_rate, 4),
            "rerank_cache_hit_rate": round(rerank_hit_rate, 4) if rerank_hit_rate is not None else None,
            "error_rate": round(self.error_rate, 4),
            "errors_by_type": dict(self.error_types),
            "retry_count": self.retry_count,
            "parse_error_count": self.parse_error_count,
            "generator_llm_calls": self.generator_llm_calls,
            "judge_faithfulness_calls": self.judge_faithfulness_calls,
            "judge_quality_calls": self.judge_quality_calls,
            "stage_retrieve_p50": stage_percentiles_map.get("retrieve", {}).get("p50", 0.0),
            "stage_retrieve_p75": stage_percentiles_map.get("retrieve", {}).get("p75", 0.0),
            "stage_retrieve_p95": stage_percentiles_map.get("retrieve", {}).get("p95", 0.0),
            "stage_generate_p50": stage_percentiles_map.get("generate", {}).get("p50", 0.0),
            "stage_generate_p75": stage_percentiles_map.get("generate", {}).get("p75", 0.0),
            "stage_generate_p95": stage_percentiles_map.get("generate", {}).get("p95", 0.0),
            "stage_judge_faithfulness_p50": stage_percentiles_map.get("judge_faithfulness", {}).get("p50", 0.0),
            "stage_judge_faithfulness_p75": stage_percentiles_map.get("judge_faithfulness", {}).get("p75", 0.0),
            "stage_judge_faithfulness_p95": stage_percentiles_map.get("judge_faithfulness", {}).get("p95", 0.0),
            "stage_judge_quality_p50": stage_percentiles_map.get("judge_quality", {}).get("p50", 0.0),
            "stage_judge_quality_p75": stage_percentiles_map.get("judge_quality", {}).get("p75", 0.0),
            "stage_judge_quality_p95": stage_percentiles_map.get("judge_quality", {}).get("p95", 0.0),
            "stage_end_to_end_p50": stage_percentiles_map.get("end_to_end", {}).get("p50", 0.0),
            "stage_end_to_end_p75": stage_percentiles_map.get("end_to_end", {}).get("p75", 0.0),
            "stage_end_to_end_p95": stage_percentiles_map.get("end_to_end", {}).get("p95", 0.0),
        }

    def print_summary(self) -> None:
        logging.info(
            "cost: $%.6f | Gen cache: %.0f%% | Judge cache: %.0f%% | errors: %d | "
            "retries: %d | parse err: %d | input P95: %d tok | output P95: %d tok",
            self.estimated_cost,
            self.generator_cache_hit_rate * 100,
            self.judge_cache_hit_rate * 100,
            self.error_count,
            self.retry_count,
            self.parse_error_count,
            int(self.input_tokens_p95),
            int(self.output_tokens_p95),
        )


_metrics: MonitorMetrics | None = None


def get_metrics() -> MonitorMetrics:
    """模块级单例。"""
    global _metrics
    if _metrics is None:
        from config import COST_INPUT_1K_PRICE, COST_OUTPUT_1K_PRICE
        _metrics = MonitorMetrics(
            input_price_per_1k=COST_INPUT_1K_PRICE,
            output_price_per_1k=COST_OUTPUT_1K_PRICE,
        )
    return _metrics


def reset_metrics() -> None:
    """重置指标（每次 run_full_mode 开头调用）。"""
    global _metrics
    _metrics = None
