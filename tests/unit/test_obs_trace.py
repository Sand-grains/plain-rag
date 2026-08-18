"""unit：obs/trace RagTrace 数据模型 + ContextVar 协议 + trace_scope 收集器。

验证 ContextVar 桶 set/reset/子线程隔离, trace_scope 退出自动入 session 收集器(异常路径也入),
RagTrace.to_dict() 全 JSON 原生类型且 durations 六键结构稳定。
"""
import json
import threading

import pytest

import config
from obs import trace as trace_module
from obs.lifecycle import reset_traces, finalize_traces
from obs.trace import RagTrace, clear_session_traces, request_id_var, trace_id_var, trace_scope, trace_var


@pytest.fixture(autouse=True)
def _clean_state():
    """每个测试前后清空 session 收集器并复位三个 ContextVar, 防跨测试污染。"""
    clear_session_traces()
    request_id_var.set("-")
    trace_id_var.set("-")
    trace_var.set(None)
    yield
    clear_session_traces()


class TestRagTrace:
    def test_to_dict_all_json_native_with_six_durations(self):
        trace = RagTrace(query_id="Q1", query="你好")
        data = trace.to_dict()
        assert data["query_id"] == "Q1"
        assert data["query"] == "你好"
        assert set(data["durations"]) == {"embed", "dense", "sparse", "retrieve_total", "rerank", "generate"}
        assert data["durations"]["generate"] == 0.0  # retrieval 模式无 generate, 缺失读作 0, 结构稳定
        json.dumps(data, ensure_ascii=False)  # 全 JSON 原生类型可直接序列化

    def test_stage_errors_serialize_to_plain_dicts(self):
        trace = RagTrace(query_id="Q1", query="q")
        trace.stage_errors.append(trace_module.StageError(
            stage="embed", error="boom", error_code="RuntimeError", fallback_to=None, severity_level="error"))
        assert trace.to_dict()["stage_errors"] == [{
            "stage": "embed", "error": "boom", "error_code": "RuntimeError",
            "fallback_to": None, "severity_level": "error"}]


class TestTraceScope:
    def test_scope_sets_vars_and_appends_to_session(self):
        with trace_scope("Q1", "你好"):
            assert request_id_var.get() == "eval-Q1"
            assert trace_id_var.get() != "-"
            assert trace_var.get() is not None
            trace_var.get().durations["embed"] = 1.0
        traces = trace_module.session_traces()
        assert len(traces) == 1
        assert traces[0].query_id == "Q1"
        assert traces[0].durations["embed"] == 1.0

    def test_scope_resets_vars_on_exit(self):
        with trace_scope("Q1", "q"):
            pass
        assert trace_var.get() is None
        assert request_id_var.get() == "-"
        assert trace_id_var.get() == "-"

    def test_scope_appends_on_exception(self):
        with pytest.raises(RuntimeError):
            with trace_scope("Q1", "q"):
                raise RuntimeError("boom")
        assert len(trace_module.session_traces()) == 1

    def test_thread_isolation(self):
        results = []

        def worker(query_id):
            with trace_scope(query_id, "q"):
                results.append((query_id, trace_id_var.get(), trace_var.get() is not None))

        threads = [threading.Thread(target=worker, args=(f"Q{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len({trace_id for _, trace_id, _ in results}) == 4  # 每 worker 独立 trace_id, 不串扰
        assert all(has_trace for _, _, has_trace in results)
        assert len(trace_module.session_traces()) == 4


class TestContextVarDefaults:
    def test_no_scope_reads_defaults(self):
        assert request_id_var.get() == "-"
        assert trace_id_var.get() == "-"
        assert trace_var.get() is None


class TestFinalizeTraces:
    def test_writes_trace_jsonl_and_clears_collector(self, tmp_path):
        with trace_scope("Q1", "q1"):
            trace_var.get().durations["embed"] = 0.5
        with trace_scope("Q2", "q2"):
            pass
        trace_path = finalize_traces([], [], out_dir=tmp_path)
        assert trace_path is not None
        assert trace_path.exists()
        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["query_id"] == "Q1"
        assert first["durations"]["embed"] == 0.5
        assert trace_module.session_traces() == []  # 消费后清空

    def test_no_traces_returns_none(self, tmp_path):
        assert finalize_traces([], [], out_dir=tmp_path) is None

    def test_reset_traces_clears_collector(self):
        with trace_scope("Q1", "q"):
            pass
        reset_traces()
        assert trace_module.session_traces() == []

    def test_sanitize_applied_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_SANITIZE", True)
        with trace_scope("Q1", "q"):
            pass
        assert finalize_traces([], [], out_dir=tmp_path) is not None

    def test_sanitize_strips_sensitive_field_names(self):
        from obs.lifecycle import _sanitize
        payload = {
            "query": "q", "api_key": "secret",
            "nested": {"authorization": "Bearer x", "ok": 1},
            "items": [{"token": "t", "name": "n"}],
        }
        assert _sanitize(payload) == {
            "query": "q", "nested": {"ok": 1}, "items": [{"name": "n"}]}
