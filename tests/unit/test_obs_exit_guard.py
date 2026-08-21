"""unit: obs/exit_guard 进程事件。

验证 atexit 兜底清空 session 收集器(residue 清空 + 空收集器 no-op)、
SIGINT/SIGTERM 最小化转 KeyboardInterrupt、register_exit_guard 注册与幂等、
非主线程 signal 注册失败降级警告。不触发真进程退出。
"""
import atexit
import logging
import signal

import pytest

from obs import exit_guard
from obs.trace import clear_session_traces, trace_scope


@pytest.fixture(autouse=True)
def _clean_session_collector():
    """每测试清空 session 收集器, 防残留跨测试污染。"""
    clear_session_traces()
    yield
    clear_session_traces()


@pytest.fixture
def _isolated_registration(monkeypatch):
    """隔离 register_exit_guard 副作用: 重置 _registered + 恢复信号/atexit。"""
    original = {
        name: signal.getsignal(getattr(signal, name))
        for name in ("SIGINT", "SIGTERM")
    }
    monkeypatch.setattr(exit_guard, "_registered", False)
    yield
    atexit.unregister(exit_guard._fallback_clear_session)
    for name, handler in original.items():
        try:
            signal.signal(getattr(signal, name), handler)
        except ValueError:
            pass
    monkeypatch.setattr(exit_guard, "_registered", False)


class TestFallbackClearSession:
    def test_residue_cleared_with_warning(self, caplog):
        with trace_scope("q1", "query1"):
            pass  # 退出 trace_scope 时 append 进 session 收集器
        assert exit_guard.session_traces()  # 有残留(未 finalize)
        with caplog.at_level(logging.WARNING):
            exit_guard._fallback_clear_session()
        assert exit_guard.session_traces() == []
        assert "残留" in caplog.text

    def test_empty_collector_is_noop(self):
        exit_guard._fallback_clear_session()  # 空收集器不崩、无警告


class TestSignalHandler:
    def test_raises_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt):
            exit_guard._signal_to_keyboard_interrupt(signal.SIGINT, None)


class TestRegisterExitGuard:
    def test_registers_signal_handlers(self, _isolated_registration):
        exit_guard.register_exit_guard()
        assert signal.getsignal(signal.SIGINT) == exit_guard._signal_to_keyboard_interrupt
        assert signal.getsignal(signal.SIGTERM) == exit_guard._signal_to_keyboard_interrupt

    def test_idempotent_does_not_stack_atexit(self, _isolated_registration, monkeypatch):
        calls = []
        original_register = atexit.register

        def _spy(func):
            calls.append(func)
            original_register(func)

        monkeypatch.setattr(atexit, "register", _spy)
        exit_guard.register_exit_guard()
        exit_guard.register_exit_guard()
        assert len(calls) == 1  # 二次调用幂等, atexit 只注册一次

    def test_signal_register_value_error_degrades_with_warning(self, _isolated_registration,
                                                               monkeypatch, caplog):
        def _boom(sig, handler):
            raise ValueError("只能在主线程调用")

        monkeypatch.setattr(signal, "signal", _boom)
        with caplog.at_level(logging.WARNING):
            exit_guard.register_exit_guard()  # 不崩, 降级警告
        assert "无法注册" in caplog.text
