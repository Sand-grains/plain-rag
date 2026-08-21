"""unit: obs 阶段计时聚合(F24)。

验证 aggregate_stage_report: 跨 query 每阶段 count/avg/p50/p95(跳过 0 耗时)、retry 差额
(fallback_required→rerank 求和)、error_code 聚合(出现次数 + query_id 去重)、trace_type 过滤;
write_step_report: 落盘路径与空收集器不产出。只读消费, 不触真 eval/检索。
"""
import json
from datetime import datetime
from pathlib import Path

from obs._stats import percentile
from obs.stage_report import _round1, aggregate_stage_report, write_step_report
from obs.trace import RagTrace, StageError, STAGE_DURATION_KEYS


def _trace(query_id, durations=None, fallback=False, errors=None, trace_type="eval"):
    trace = RagTrace(query_id=query_id, query="q", trace_type=trace_type)
    for stage, value in (durations or {}).items():
        trace.durations[stage] = value
    trace.fallback_required = fallback
    trace.stage_errors = errors or []
    return trace


def _error(code, stage="retrieval"):
    return StageError(stage=stage, error=f"{code} happened", error_code=code)


class TestAggregateBasic:
    def test_single_trace_each_stage_count_avg(self):
        trace = _trace("Q1", {"embed": 10.0, "dense": 20.0, "sparse": 30.0, "retrieve_total": 60.0, "rerank": 40.0})
        report = aggregate_stage_report([trace])
        assert report["stages"]["embed"] == {"count": 1, "avg_ms": 10.0, "p50_ms": 10.0, "p95_ms": 10.0}
        assert report["stages"]["rerank"]["count"] == 1
        assert report["stages"]["rerank"]["avg_ms"] == 40.0

    def test_multiple_traces_aggregate_stats(self):
        traces = [_trace("Q1", {"rerank": 10.0, "generate": 100.0}),
                  _trace("Q2", {"rerank": 30.0, "generate": 300.0}),
                  _trace("Q3", {"rerank": 20.0, "generate": 200.0})]
        report = aggregate_stage_report(traces)
        rerank = report["stages"]["rerank"]
        assert rerank["count"] == 3
        assert rerank["avg_ms"] == 20.0
        assert rerank["p50_ms"] == 20.0  # [10,20,30] 中位
        assert rerank["p95_ms"] == round(percentile([10.0, 30.0, 20.0], 95), 1)
        assert report["stages"]["generate"]["avg_ms"] == 200.0

    def test_empty_traces_empty_stages_zero_retry(self):
        report = aggregate_stage_report([])
        assert report == {"stages": {}, "retry": {"rerank_fallback_count": 0, "retry_wasted_ms": 0.0}}

    def test_zero_durations_skipped(self):
        # retrieval 模式: generate 恒 0, 视为阶段未跑, 不进入延迟分布
        traces = [_trace("Q1", {"embed": 5.0, "rerank": 8.0}), _trace("Q2", {"embed": 7.0, "rerank": 9.0})]
        report = aggregate_stage_report(traces)
        assert "generate" not in report["stages"]
        assert report["stages"]["embed"]["count"] == 2
        assert set(report["stages"]) <= set(STAGE_DURATION_KEYS)


class TestRetryDelta:
    def test_rerank_fallback_count_and_wasted(self):
        traces = [_trace("Q1", {"rerank": 50.0}, fallback=True),
                  _trace("Q2", {"rerank": 70.0}, fallback=True),
                  _trace("Q3", {"rerank": 30.0}, fallback=False)]
        report = aggregate_stage_report(traces)
        assert report["retry"] == {"rerank_fallback_count": 2, "retry_wasted_ms": 120.0}

    def test_no_fallback_zero_wasted(self):
        traces = [_trace("Q1", {"rerank": 30.0})]
        assert aggregate_stage_report(traces)["retry"] == {"rerank_fallback_count": 0, "retry_wasted_ms": 0.0}


class TestTraceTypeFilter:
    def test_filter_eval_excludes_agent(self):
        traces = [_trace("Q1", {"embed": 10.0}, trace_type="eval"),
                  _trace("1", {"embed": 100.0}, trace_type="agent")]
        report = aggregate_stage_report(traces, trace_type="eval")
        assert report["stages"]["embed"]["count"] == 1
        assert report["stages"]["embed"]["avg_ms"] == 10.0

    def test_filter_agent_only(self):
        traces = [_trace("Q1", {"embed": 10.0}, trace_type="eval"),
                  _trace("1", {"embed": 100.0}, trace_type="agent")]
        report = aggregate_stage_report(traces, trace_type="agent")
        assert report["stages"]["embed"]["avg_ms"] == 100.0

    def test_no_filter_counts_all(self):
        traces = [_trace("Q1", {"embed": 10.0}, trace_type="eval"),
                  _trace("1", {"embed": 100.0}, trace_type="agent")]
        assert aggregate_stage_report(traces)["stages"]["embed"]["count"] == 2


class TestErrorCodes:
    def test_counts_and_query_ids_dedup(self):
        traces = [_trace("Q1", errors=[_error("ConnectionError"), _error("ConnectionError")]),
                  _trace("Q2", errors=[_error("TimeoutError")]),
                  _trace("Q3", errors=[_error("ConnectionError")])]
        codes = aggregate_stage_report(traces)["error_codes"]
        assert codes["ConnectionError"] == {"count": 3, "query_ids": ["Q1", "Q3"]}
        assert codes["TimeoutError"] == {"count": 1, "query_ids": ["Q2"]}

    def test_sorted_by_count_desc_then_code(self):
        traces = [_trace("Q1", errors=[_error("A")]),
                  _trace("Q2", errors=[_error("B"), _error("B")])]
        assert list(aggregate_stage_report(traces)["error_codes"]) == ["B", "A"]

    def test_no_errors_key_omitted(self):
        assert "error_codes" not in aggregate_stage_report([_trace("Q1")])

    def test_error_codes_respect_trace_type_filter(self):
        traces = [_trace("Q1", trace_type="eval", errors=[_error("EvalCode")]),
                  _trace("1", trace_type="agent", errors=[_error("AgentCode")])]
        report = aggregate_stage_report(traces, trace_type="eval")
        assert list(report["error_codes"]) == ["EvalCode"]


class TestWriteStepReport:
    def test_writes_jsonl_under_day_dir(self, tmp_path):
        traces = [_trace("Q1", {"embed": 10.0, "rerank": 40.0})]
        path = write_step_report(traces, out_dir=tmp_path)
        assert path is not None
        assert path.is_file()
        assert path.name.startswith("step_timing-")
        assert path.name.endswith(".jsonl")
        assert path.parent.parent.parent == tmp_path  # tmp/traces/<today>/step_timing-*
        assert path.parent.parent.name == "traces"
        assert path.parent.name == datetime.now().strftime("%Y-%m-%d")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["stages"]["embed"]["avg_ms"] == 10.0
        assert "retry" in payload

    def test_empty_traces_returns_none(self, tmp_path):
        assert write_step_report([], out_dir=tmp_path) is None
        assert list(Path(tmp_path).rglob("step_timing-*.jsonl")) == []

    def test_round1(self):
        assert _round1(34.24) == 34.2
        assert _round1(812.0) == 812.0
        assert _round1(None) == 0.0
