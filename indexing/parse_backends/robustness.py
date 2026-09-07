"""robustness.py：运行时健壮性——失败分类/熔断/统计/warm_up。

- 失败分类: 把异常映射到 8 类
(import_error/model_load_error/parse_error/timeout/empty/quality_fail/vlm_unavailable/colpali_unavailable), 供失败清单与统计使用。
- 熔断: 连续 3 次失败/超时后禁用对应后端; import_error/model_load_error 立即禁用。
- 统计: run 级 backend 分布/成功率/p50/p95/降级原因 top5/ColPali 触发率。
- warm_up: 启动探活, 失败则禁用对应管线并 warning。
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 失败分类
CATEGORY_IMPORT_ERROR = "import_error"
CATEGORY_MODEL_LOAD_ERROR = "model_load_error"
CATEGORY_PARSE_ERROR = "parse_error"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_EMPTY = "empty"
CATEGORY_QUALITY_FAIL = "quality_fail"
CATEGORY_VLM_UNAVAILABLE = "vlm_unavailable"
CATEGORY_COLPALI_UNAVAILABLE = "colpali_unavailable"

# 立即熔断的分类: 环境/模型加载问题, 重试无意义
_IMMEDIATE_OPEN = {CATEGORY_IMPORT_ERROR, CATEGORY_MODEL_LOAD_ERROR}


def classify_error(exc: BaseException, backend_name: str = "") -> str:
    """把异常映射到失败分类(规则式, 供失败清单/统计使用)。

    Args:
        exc: 捕获的异常。
        backend_name: 后端名(用于区分 vlm/colpali 不可用)。

    Returns:
        str: 失败分类。
    """
    if isinstance(exc, TimeoutError):
        return CATEGORY_TIMEOUT
    msg = str(exc)
    if backend_name == "vlm" and "不可用" in msg:
        return CATEGORY_VLM_UNAVAILABLE
    if backend_name == "colpali" and "不可用" in msg:
        return CATEGORY_COLPALI_UNAVAILABLE
    if "未安装" in msg or "不可用" in msg:
        return CATEGORY_IMPORT_ERROR
    if "模型" in msg or "model" in msg.lower():
        return CATEGORY_MODEL_LOAD_ERROR
    if "空" in msg or "empty" in msg.lower():
        return CATEGORY_EMPTY
    if "质量" in msg:
        return CATEGORY_QUALITY_FAIL
    return CATEGORY_PARSE_ERROR


@dataclass
class CircuitBreaker:
    """按后端熔断: 连续失败达阈值或立即熔断分类时禁用该后端(本次运行)。"""

    max_failures: int = 3
    _failures: dict[str, int] = field(default_factory=dict)
    _open: set[str] = field(default_factory=set)

    def record_success(self, name: str) -> None:
        """成功后清零该后端连续失败计数。"""
        self._failures[name] = 0

    def record_failure(self, name: str, category: str) -> None:
        """记录一次失败; 达阈值或立即熔断分类时打开熔断。"""
        if category in _IMMEDIATE_OPEN:
            self._open.add(name)
            return
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= self.max_failures:
            self._open.add(name)
            logger.warning("熔断: 后端 %s 连续 %d 次失败, 本次运行禁用", name, self.max_failures)

    def is_open(self, name: str) -> bool:
        """该后端是否已熔断(禁用)。"""
        return name in self._open

    def reset(self) -> None:
        """清空熔断状态(新 run 开始时调用)。"""
        self._failures.clear()
        self._open.clear()


@dataclass
class RunStats:
    """run 级统计: backend 分布/成功率/p50/p95/降级原因 top5/ColPali 触发率。"""

    backend_counts: Counter = field(default_factory=Counter)
    durations: dict[str, list[float]] = field(default_factory=dict)
    degradation_reasons: Counter = field(default_factory=Counter)
    colpali_triggered: int = 0
    total: int = 0
    success: int = 0

    def record(self, backend: str, duration: float, ok: bool,
               reason: str | None = None) -> None:
        """记录一次后端执行结果。

        Args:
            backend: 后端名。
            duration: 耗时(秒)。
            ok: 是否成功。
            reason: 失败分类(失败时)。
        """
        self.total += 1
        if ok:
            self.success += 1
            self.backend_counts[backend] += 1
        elif reason:
            self.degradation_reasons[reason] += 1
        self.durations.setdefault(backend, []).append(duration)

    def record_colpali_trigger(self) -> None:
        """记录一次 ColPali 触发(loader 在 precheck.colpali_triggered 时调用, L6 接线)。"""
        self.colpali_triggered += 1

    def reset(self) -> None:
        """清空 run 级统计(新 run 开始时调用, 替代 __init__ 重置)。"""
        self.backend_counts.clear()
        self.durations.clear()
        self.degradation_reasons.clear()
        self.colpali_triggered = 0
        self.total = 0
        self.success = 0

    def summary(self) -> dict[str, Any]:
        """汇总为可序列化 dict(供报告/日志)。"""
        p50_p95: dict[str, dict[str, float]] = {}
        for name, durs in self.durations.items():
            if not durs:
                continue
            s = sorted(durs)
            p50_p95[name] = {
                "p50": s[len(s) // 2],
                "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
            }
        return {
            "total": self.total,
            "success": self.success,
            "success_rate": (self.success / self.total) if self.total else 0.0,
            "backend_distribution": dict(self.backend_counts),
            "p50_p95": p50_p95,
            "degradation_reasons_top5": self.degradation_reasons.most_common(5),
            "colpali_triggered": self.colpali_triggered,
        }


def warm_up(enabled_backends: list[str], probe: Callable[[str], str],
            timeout_s: float) -> dict[str, bool]:
    """启动探活: 对每个启用后端跑一次最小探针(import/版本), 失败则禁用并 warning。

    Args:
        enabled_backends: 门控开启的后端名列表。
        probe: 探针函数(输入后端名, 返回版本串; 失败抛异常)。
        timeout_s: 单次探针超时。

    Returns:
        dict[str, bool]: 后端名 -> 是否可用(False 表示已禁用)。
    """
    result: dict[str, bool] = {}
    for name in enabled_backends:
        try:
            probe(name)
            result[name] = True
        except Exception as exc:  # noqa: BLE001 - 探活失败禁用对应管线
            result[name] = False
            logger.warning("warm_up 失败, 禁用后端 %s: %s", name, exc)
    return result
