"""obs 观测进程退出时作兜底收尾, 保证中断/异常退出时, 未 finalize 的 trace 不残留且日志不丢
处理两类进程事件: 进程退出事件(atexit), 中断信号(SIGINT/SIGTERM)
atexit 兜底收尾 + SIGINT/SIGTERM 转 KeyboardInterrupt, 补异常退出兜底(010-2 F22)。

register_exit_guard() 被 eval/runner 和 agent_pipeline 的 main() 各调一次(幂等), 做两件事:
    ├─ atexit.register(_fallback_clear_session)      ← 进程退出时跑(正常/异常都触发)
    └─ signal.signal(SIGINT, _signal_to_keyboard_interrupt)   ← Ctrl+C
       signal.signal(SIGTERM, _signal_to_keyboard_interrupt)   ← kill 默认信号

核心特性:
    - atexit 兜底: 进程退出时若 session 收集器有残留(未 finalize 的 trace), 清空防混入进程内下一次 run
    - 日志 flush: 进程退出时 flush root logger 全部 handler(setup_logging 可选注册)
    - signal handler 最小化: SIGINT/SIGTERM 只 raise KeyboardInterrupt, 不做重活
      (审查 A3: signal handler 在主线程字节码间隙运行, 写文件/线程协调有死锁或半写风险, 重活放 finally/atexit)
    - 幂等: 重复 register 安全; 收集器已空则清空 no-op(异常路径 + atexit 双重 finalize 不冲突)

与上层的关系: eval/runner main() 调 register_exit_guard() 一次; 不加 daemon 线程(全局硬约束 8)。
"""
from __future__ import annotations

import atexit
import logging
import signal
import types

from obs.trace import clear_session_traces, session_traces

logger = logging.getLogger(__name__)

_registered: bool = False


def register_exit_guard() -> None:
    """注册进程级退出钩子(幂等): atexit 兜底清空 + SIGINT/SIGTERM 转 KeyboardInterrupt。

    重活(清空收集器/flush 日志)在 atexit 上下文执行, signal handler 只 raise,
    保证中断/异常退出时 finally/atexit 的正常收尾逻辑照常运行(对齐审查 A3)。
    """
    global _registered
    if _registered:
        return
    _registered = True
    atexit.register(_fallback_clear_session)
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _signal_to_keyboard_interrupt)
        except ValueError:
            logger.warning("无法注册 %s 处理器(仅主线程可注册)", sig_name)


def _fallback_clear_session() -> None:
    """atexit 兜底: session 收集器有残留(未 finalize)则清空, 并 flush 日志。

    幂等: 正常完成时 finalize_traces 已消费收集器, 此处天然为空;
    中断/崩溃未走到 finalize 时清空残留, 防混入进程内下一次 run。
    """
    remaining = session_traces()
    if remaining:
        logger.warning("进程退出时 session 收集器残留 %d 条 trace(未 finalize), 兜底清空", len(remaining))
        clear_session_traces()
    for handler in logging.getLogger().handlers:
        handler.flush()


def _signal_to_keyboard_interrupt(signum: int, frame: types.FrameType | None) -> None:
    """最小化 signal handler: 转 KeyboardInterrupt 触发正常收尾, 不在 signal 上下文做重活。

    Args:
        signum: 触发信号编号(SIGINT/SIGTERM)。
        frame: 中断时当前栈帧(signal 模块注入, 本函数不使用)。

    Raises:
        KeyboardInterrupt: 恒抛, 由 Python 正常 unwinding 进 finally/atexit。
    """
    raise KeyboardInterrupt
