"""阶段装饰器: 业务模块经装饰器埋点
装饰器只读 trace.py 的 ContextVar, 不 import 业务模块。

观测只发生在装饰器边界与 per-call 诊断槽: 业务模块自身不出现 trace.append/durations.update
语句, 只多一个默认 None 的可选 _diagnostics 参数(降级/召回信号经槽传出)。

零开销旁路: 读 trace_var 为 None 时直接跳过, 无 trace 上下文(agent 路径/单元测试)观测为零开销。
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from config import TOP_K
from obs.trace import RagTrace, StageError, trace_var


def _current_trace() -> RagTrace | None:
    """读当前 trace; 无 trace 上下文时返回 None(装饰器据此跳过)。"""
    return trace_var.get()


def _elapsed_ms(start: float) -> float:
    """perf_counter 起止耗时转毫秒。"""
    return (time.perf_counter() - start) * 1000


def observe_stage(stage: str) -> Callable:
    """阶段装饰器
    记录: 该阶段耗时 + 该阶段异常
    捕获裸抛异常写 stage_errors 并 re-raise。

    Args:
        stage: 阶段名(embed/dense/sparse), 对应 RagTrace.durations 键与 stage_errors.stage。

    Returns:
        Callable: 装饰器。
        捕获异常后必须 re-raise(吞异常会改变业务语义: 失败本应让 query 进 _evaluate_one 的 except 得 error 判定)。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace()
            if trace is None:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                trace.durations[stage] = _elapsed_ms(start)
                trace.stage_errors.append(StageError(
                    stage=stage,
                    error=str(error),
                    error_code=type(error).__name__,
                    fallback_to=None,
                    severity_level="error",
                ))
                raise
            trace.durations[stage] = _elapsed_ms(start)
            return result

        return wrapper

    return decorator


def observe_retrieval_pipeline(func: Callable) -> Callable:
    """检索编排层的输出汇总 (retrieve_with_dense_child)
    包住整段 "混合检索 → RRF 融合 → [auto-merge → rerank]" 流程, 调用成功后从返回值 + 诊断槽汇总四样:
     - durations["retrieve_total"] = 整个检索编排耗时
     - retrieved_chunks = 最终返回父块的 chunk_id 列表
     - context_chars = 父块文本总长(进入生成的上下文量)
     - recalled_ids = 诊断槽传出的 RRF 融合全量

    不捕获异常, 只做成功后汇总。如有异常则向上抛出
    完整召回键集(recalled_ids)由检索器内部经 _diagnostics 槽传出(RRF 融合全量 key),
    供后续 归因 区分"真没召回"(recall_absent)与"池截断"(rerank_drop)。
    """

    @functools.wraps(func)
    def wrapper(self, query: str, top_k: int = TOP_K, _diagnostics: dict | None = None):
        trace = _current_trace()
        if trace is None:
            return func(self, query, top_k=top_k, _diagnostics=_diagnostics)
        diagnostics = {} if _diagnostics is None else _diagnostics
        start = time.perf_counter()
        parents, dense_children = func(self, query, top_k=top_k, _diagnostics=diagnostics)
        trace.durations["retrieve_total"] = _elapsed_ms(start)
        trace.retrieved_chunks = [chunk.chunk_id for chunk in parents]
        trace.context_chars = sum(len(chunk.content) for chunk in parents)
        recalled_ids = diagnostics.get("recalled_ids")
        if recalled_ids is not None:
            trace.recalled_ids = recalled_ids
        return parents, dense_children

    return wrapper


def trace_rerank(func: Callable) -> Callable:
    """Reranker.rerank 装饰器: 记录 rerank 前后候选三视图 + rerank 计时 + 缓存命中 + 降级标记。
    - 调用前: candidates_before_rerank = 入 rerank 的候选 (RRF 池截断集)
    - 调用后: candidates_after_rerank = Rerank后的 top_k; durations["rerank"]; is_rerank_cache_hit = 诊断槽信号
    - 降级判定: 若诊断槽有 fallback_required → fallback_required=True + 追加降级 stage_errors(severity_level="warning",
  fallback_to="rrf_order")

    降级不靠边界推断: rerank 内部经 _diagnostics 槽写 fallback 标记与异常类型(区分CrossEncoder 降级与缓存读失败),
    装饰器读入 fallback_required 与 stage_errors。
    """

    @functools.wraps(func)
    def wrapper(self, query: str, parent_chunks: list, top_k: int,
                corpus_signature: str = "", skip_cache: bool = False,
                _diagnostics: dict | None = None):
        trace = _current_trace()
        if trace is None:
            return func(self, query, parent_chunks, top_k,
                        corpus_signature=corpus_signature, skip_cache=skip_cache,
                        _diagnostics=_diagnostics)
        trace.candidates_before_rerank = [chunk.chunk_id for chunk in parent_chunks]
        diagnostics = {} if _diagnostics is None else _diagnostics
        start = time.perf_counter()
        result = func(self, query, parent_chunks, top_k,
                      corpus_signature=corpus_signature, skip_cache=skip_cache,
                      _diagnostics=diagnostics)
        trace.durations["rerank"] = _elapsed_ms(start)
        trace.candidates_after_rerank = [chunk.chunk_id for chunk in result]
        trace.is_rerank_cache_hit = diagnostics.get("is_rerank_cache_hit")
        if diagnostics.get("fallback_required"):
            trace.fallback_required = True
            trace.stage_errors.append(StageError(
                stage="rerank",
                error=str(diagnostics.get("error", "")),
                error_code=str(diagnostics.get("error_code", "")),
                fallback_to=str(diagnostics.get("fallback_to", "")),
                severity_level="warning",
            ))
        return result

    return wrapper

