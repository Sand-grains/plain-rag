"""unit: obs agent 路径 trace(F23)。

验证 trace_type 判别(eval 默认 / agent 显式 + request_id 前缀)、agent 阶段键经装饰器写入 durations、
finalize_traces 无 benchmark(None)时跳过归因直接落盘(不误判 empty_recall/recall_absent)、
eval 归因路径语义不回退。不触真 agent/LLM。
"""
import json

import pytest

from obs import trace as trace_module
from obs.lifecycle import finalize_traces
from obs.trace import RagTrace, clear_session_traces, request_id_var, trace_scope


@pytest.fixture(autouse=True)
def _clean_state():
    """每测试清空 session 收集器并复位 request_id, 防跨测试污染。"""
    clear_session_traces()
    request_id_var.set("-")
    yield
    clear_session_traces()


class TestTraceTypeDiscrimination:
    def test_default_trace_type_is_eval(self):
        trace = RagTrace(query_id="Q1", query="q")
        assert trace.trace_type == "eval"
        assert trace.to_dict()["trace_type"] == "eval"

    def test_agent_trace_scope_sets_type_and_request_id(self):
        with trace_scope("1", "你好", trace_type="agent"):
            assert request_id_var.get() == "agent-1"
            assert trace_module.trace_var.get().trace_type == "agent"
        traces = trace_module.session_traces()
        assert len(traces) == 1
        assert traces[0].trace_type == "agent"

    def test_eval_trace_scope_request_id_prefix_unchanged(self):
        with trace_scope("Q1", "q"):
            assert request_id_var.get() == "eval-Q1"
        assert trace_module.session_traces()[0].trace_type == "eval"


class TestAgentStageKeys:
    def test_observe_stage_writes_agent_stage_duration(self):
        from obs.trace_decorators import observe_stage

        @observe_stage("agent_generate")
        def _fake_generate():
            return "answer"

        with trace_scope("1", "q", trace_type="agent"):
            _fake_generate()
        trace = trace_module.session_traces()[0]
        assert trace.durations["agent_generate"] > 0  # agent 阶段键写入 durations(纯 dict, 不锁死六键)


class TestFinalizeNoBenchmark:
    def test_skip_attribution_writes_without_failure_attribution(self, tmp_path):
        with trace_scope("1", "你好", trace_type="agent"):
            trace_module.trace_var.get().durations["agent_generate"] = 12.5
        trace_path = finalize_traces(out_dir=tmp_path)  # benchmark_items=None → 跳过归因
        assert trace_path is not None
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        assert payload["trace_type"] == "agent"
        assert payload["durations"]["agent_generate"] == 12.5
        assert "failure_attribution" not in payload  # 无 benchmark 不硬归因, 防误判 empty_recall/recall_absent
        assert trace_module.session_traces() == []  # 消费后清空

    def test_eval_path_still_attributes(self, tmp_path):
        with trace_scope("Q1", "q1"):
            pass
        trace_path, attribution = finalize_traces([], [], out_dir=tmp_path, return_attribution=True)
        assert trace_path is not None
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        assert "failure_attribution" in payload  # eval 归因语义不回退
        assert attribution["Q1"]["failure_type"] == "empty_recall"

    def test_no_traces_returns_none(self, tmp_path):
        assert finalize_traces(out_dir=tmp_path) is None
