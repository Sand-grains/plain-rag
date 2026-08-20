"""Trace 链路数据模型与 ContextVar 协议(一期字段, 对齐 009-1)。
实际包含两个抽象层级: query 级模型(RagTrace) + session 级收集器 (一次 eval.runner --mode retrieval/full 进程跑完 = 一个 session)


本模块定义三个 ContextVar, 用以支撑 eval 并发观测:
(也是三个全局唯一的 ContextVar 对象(一个eval进程独一份), 但值按线程一份)
    - request_id_var: 请求级 id, 形如 eval-<query_id> / agent-<seq>, 供日志 Filter 注入日志行
    - trace_id_var: uuid4, 单条 trace 链路唯一标识
    - trace_var: 指向当前 RagTrace 实例容器(可为 None, 装饰器据此零开销跳过)

trace_scope(query_id, query) 上下文管理器:
    - 进入: 生成新 request_id/trace_id + new 一个空 RagTrace, 写入三个 var
    - 退出: RagTrace 自动 append 进 session 收集器(异常路径也 append, 幂等一次), 并还原三个 var

线程隔离: contextvars 不依赖线程局部存储
trace_scope 必须开在 _evaluate_one worker 内部(per-query),
trace_scope 若开在线程池 submit 循环外面, 所有任务拿到的是同一份提交时快照, 快照里只有一个 trace_var

"""

from __future__ import annotations

import contextvars
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Iterator

# 定义全局 ContextVar 对象
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
trace_var: contextvars.ContextVar["RagTrace | None"] = contextvars.ContextVar("trace", default=None)

# durations 键固定为 6 个 (retrieval 模式无 generate, 缺失阶段读作 0, 结构稳定)
STAGE_DURATION_KEYS = ("embed", "dense", "sparse", "retrieve_total", "rerank", "generate")


@dataclass
class StageError:
    """结构化阶段错误: 供归因统计按 error_code 聚合。"""
    stage: str        # 出错的阶段: embed / dense / sparse / rerank / generate / pipeline
    error: str        # 错误信息
    error_code: str   # 错误码(异常类名或降级目标字符串, 非独立枚举)
    fallback_to: str | None = None  # 降级去向(如 "rrf_order")
    severity_level: str = "error"   # "warning" / "error"


@dataclass
class RagTrace:
    """单条 query 的完整链路数据集 (三视图 + 完整召回集 + 计时 + 阶段错误)。

    trace_type 判别 eval/agent: 009 只记 eval, 010-2 F23 agent 路径复用同一收集器与
    trace.jsonl, 类型判别供阶段聚合(010-1 F24)按类型过滤, 防 agent 检索/生成阶段污染 eval 阶段统计。
    """
    query_id: str = ""                       # 查询标识(agent 路径为会话内序列号)
    query: str = ""                          # 查询文本
    trace_type: str = "eval"                 # 链路类型判别: "eval" | "agent"(加法式, 默认兼容既有)
    candidates_before_rerank: list[str] = field(default_factory=list)  # rerank 前候选(RRF 池截断集)
    candidates_after_rerank: list[str] = field(default_factory=list)   # rerank 后候选
    recalled_ids: list[str] = field(default_factory=list)              # 完整召回键集(RRF 全量 key, 含被池截断)
    retrieved_chunks: list[str] = field(default_factory=list)          # 最终返回父块
    context_chars: int = 0                                             # 进入生成的上下文总字符数
    durations: dict[str, float] = field(default_factory=dict)            # [stage, 耗时(ms)]
    stage_errors: list[StageError] = field(default_factory=list)   # 结构化阶段错误
    fallback_required: bool = False                                    # 是否触发降级路径
    is_rerank_cache_hit: bool | None = None                            # rerank 缓存命中(无 rerank 为 None)

    def __post_init__(self) -> None:
        """durations 键固定为 6 个并置 0, 缺失阶段读作 0, 序列化结构稳定。"""
        self.durations = {stage: 0.0 for stage in STAGE_DURATION_KEYS}

    def to_dict(self) -> dict:
        """序列化为 JSON 原生类型 dict(全部 list/dict/str/float/bool), 供 trace.jsonl 落盘。"""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "trace_type": self.trace_type,
            "candidates_before_rerank": self.candidates_before_rerank,
            "candidates_after_rerank": self.candidates_after_rerank,
            "recalled_ids": self.recalled_ids,
            "retrieved_chunks": self.retrieved_chunks,
            "context_chars": self.context_chars,
            "durations": self.durations,
            "stage_errors": [asdict(error) for error in self.stage_errors],
            "fallback_required": self.fallback_required,
            "is_rerank_cache_hit": self.is_rerank_cache_hit,
        }


# ---- session 级收集器(worker 汇聚通道) ----

_session_traces: list[RagTrace] = []
_collector_lock = threading.Lock()


def _append_session_trace(trace: RagTrace) -> None:
    """线程安全 append 进 session 收集器( full 模式多 worker 并发汇聚, 异常路径也 append)。"""
    with _collector_lock:
        _session_traces.append(trace)


def session_traces() -> list[RagTrace]:
    """读收集器快照(收尾 finalize_traces 消费用, 不落并发锁)."""
    return list(_session_traces)


def clear_session_traces() -> None:
    """清空收集器(run 开头 reset_traces / 收尾 finalize_traces 消费后各一次)。"""
    _session_traces.clear()


# ---- trace_scope ----

@contextmanager
def trace_scope(query_id: str, query: str, trace_type: str = "eval") -> Iterator[RagTrace]:
    """上下文管理器: 单条 query 的观测作用域
    进入并生成新 request_id/trace_id/RagTrace 并写三个 var, 退出自动 append 进收集器。

    Args:
        query_id: 查询标识(agent 路径传会话内序列号)。
        query: 查询文本。
        trace_type: 链路类型 "eval" | "agent", 决定 request_id 前缀与阶段聚合的过滤维度。

    Yields:
        RagTrace: 当前链路的 trace 容器(装饰器自动消费, 调用方一般不直接使用)。
    """
    token_request_id = request_id_var.set(f"{trace_type}-{query_id}")
    token_trace_id = trace_id_var.set(str(uuid.uuid4()))
    trace = RagTrace(query_id=query_id, query=query, trace_type=trace_type)
    token_trace = trace_var.set(trace)
    try:
        yield trace
    finally:
        _append_session_trace(trace)
        trace_var.reset(token_trace)
        trace_id_var.reset(token_trace_id)
        request_id_var.reset(token_request_id)
