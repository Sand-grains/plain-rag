"""unit：obs/trace_decorators 阶段边界装饰器——durations/三视图/降级标记/召回集。

装饰器用 dummy 函数验证行为(业务函数真实接线由检索 e2e 覆盖): 零开销跳过、
durations 记录、异常写 stage_errors 并 re-raise、trace_rerank 三视图与诊断槽降级信号、
observe_retrieval_pipeline 召回集与 context_chars。
"""
import pytest

from obs import trace as trace_module
from obs.trace import clear_session_traces, trace_scope, trace_var
from obs.trace_decorators import observe_retrieval_pipeline, observe_stage, trace_rerank


@pytest.fixture(autouse=True)
def _clean_state():
    """每个测试前后清空 session 收集器并复位 trace_var, 防跨测试污染。"""
    clear_session_traces()
    trace_var.set(None)
    yield
    clear_session_traces()


class _Chunk:
    def __init__(self, chunk_id, content=""):
        self.chunk_id = chunk_id
        self.content = content


class TestObserveStage:
    def test_records_timing_and_returns_result(self):
        @observe_stage("embed")
        def dummy(texts):
            return ["vec"]

        with trace_scope("Q1", "q"):
            assert dummy(["t"]) == ["vec"]
        assert trace_module.session_traces()[0].durations["embed"] >= 0.0

    def test_records_stage_error_and_reraises(self):
        @observe_stage("embed")
        def dummy(texts):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            with trace_scope("Q1", "q"):
                dummy(["t"])
        error = trace_module.session_traces()[0].stage_errors[0]
        assert error.stage == "embed"
        assert error.error_code == "RuntimeError"
        assert error.severity_level == "error"

    def test_skips_zero_overhead_without_trace(self):
        @observe_stage("embed")
        def dummy(texts):
            return ["vec"]

        assert dummy(["t"]) == ["vec"]  # 无 trace 上下文直接透传


class TestTraceRerank:
    def test_records_candidates_and_cache_hit(self):
        class DummyReranker:
            @trace_rerank
            def rerank(self, query, parent_chunks, top_k, corpus_signature="", skip_cache=False,
                       _diagnostics=None):
                if _diagnostics is not None:
                    _diagnostics["is_rerank_cache_hit"] = True
                return parent_chunks[:top_k]

        chunks = [_Chunk("c1"), _Chunk("c2")]
        with trace_scope("Q1", "q"):
            result = DummyReranker().rerank("q", chunks, top_k=1)
        assert [chunk.chunk_id for chunk in result] == ["c1"]
        trace = trace_module.session_traces()[0]
        assert trace.candidates_before_rerank == ["c1", "c2"]
        assert trace.candidates_after_rerank == ["c1"]
        assert trace.is_rerank_cache_hit is True
        assert trace.durations["rerank"] >= 0.0

    def test_records_fallback_via_diagnostics_slot(self):
        class DummyReranker:
            @trace_rerank
            def rerank(self, query, parent_chunks, top_k, corpus_signature="", skip_cache=False,
                       _diagnostics=None):
                if _diagnostics is not None:
                    _diagnostics["fallback_required"] = True
                    _diagnostics["error"] = "model crashed"
                    _diagnostics["error_code"] = "RuntimeError"
                    _diagnostics["fallback_to"] = "rrf_order"
                return parent_chunks[:top_k]

        with trace_scope("Q1", "q"):
            DummyReranker().rerank("q", [_Chunk("c1")], top_k=1)
        trace = trace_module.session_traces()[0]
        assert trace.fallback_required is True
        assert trace.stage_errors[0].fallback_to == "rrf_order"
        assert trace.stage_errors[0].severity_level == "warning"


class TestObservePipeline:
    def test_records_retrieved_chunks_context_and_recalled_ids(self):
        class DummyRetriever:
            @observe_retrieval_pipeline
            def retrieve_with_dense_child(self, query, top_k=5, _diagnostics=None):
                if _diagnostics is not None:
                    _diagnostics["recalled_ids"] = ["r1", "r2", "r3"]
                return [_Chunk("r1", "text one")], [_Chunk("c1")]

        with trace_scope("Q1", "q"):
            parents, dense_children = DummyRetriever().retrieve_with_dense_child("q", top_k=1)
        assert [chunk.chunk_id for chunk in parents] == ["r1"]
        trace = trace_module.session_traces()[0]
        assert trace.retrieved_chunks == ["r1"]
        assert trace.context_chars == len("text one")
        assert trace.recalled_ids == ["r1", "r2", "r3"]
        assert trace.durations["retrieve_total"] >= 0.0

    def test_skips_zero_overhead_without_trace(self):
        class DummyRetriever:
            @observe_retrieval_pipeline
            def retrieve_with_dense_child(self, query, top_k=5, _diagnostics=None):
                if _diagnostics is not None:
                    _diagnostics["recalled_ids"] = ["r1"]
                return [_Chunk("r1")], []

        parents, dense = DummyRetriever().retrieve_with_dense_child("q")
        assert [chunk.chunk_id for chunk in parents] == ["r1"]
        assert trace_module.session_traces() == []  # 无 trace 时不产出
