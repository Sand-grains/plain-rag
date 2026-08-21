"""unit：eval/reporter 报告生成 + eval/serialization 序列化/聚合(F19 抽取)。

验证 build_summary 抽取(aggregate/groups + layer2/cost 可选)、serialize_result/serialize_judge_result
字段完整、generate_report 一站式写 4 文件(per_query/failures/summary/run_info, history 已废止)。
"""
import json

import eval.reporter as reporter
import eval.serialization as serialization
from eval.core.llm_as_judge.judge import JudgeResult
from eval.core.retrieval.retrieval_layer import LayerOutput, RetrievalEvalResult


def _make_result(query_id="Q1") -> RetrievalEvalResult:
    return RetrievalEvalResult(
        query_id=query_id, query="问题", category="cat", difficulty="easy",
        recall_at_k=0.5, precision_at_k=0.4, hit_at_k=1, mrr=0.5,
        map_at_k=0.4, ndcg_at_k=0.6, diagnosis="accept",
        final_chunk_ids=["c1"], candidate_chunk_ids=["c1", "c2"],
        retrieved_files=["doc.md"], child_hit_at_k=0, child_recall_at_k=0.0,
        query_rewritten="问题", query_rewritten_flag="原", final_context_text="正文",
    )


def _make_output(results) -> LayerOutput:
    return LayerOutput(
        results=results,
        aggregate={"num_queries": len(results), "recall_at_k": 0.5},
        by_category={"cat": {"recall_at_k": 0.5}},
        by_difficulty={"easy": {"recall_at_k": 0.5}},
    )


def _make_judge(query_id="Q1") -> JudgeResult:
    return JudgeResult(
        query_id=query_id, faithfulness=0.8, answer_relevancy=0.7,
        context_precision=0.6, context_recall=0.9, answer_correctness=0.75, verdict="pass",
    )


class TestBuildSummary:
    def test_aggregate_and_groups_present(self):
        summary = serialization.build_summary(_make_output([_make_result()]))
        assert set(summary) == {"aggregate", "by_category", "by_difficulty"}
        assert summary["aggregate"]["recall_at_k"] == 0.5

    def test_layer2_and_cost_optional(self):
        summary = serialization.build_summary(
            _make_output([_make_result()]),
            judge_results=[_make_judge()],
            metrics_summary={"estimated_cost": 0.001},
        )
        assert "layer2" in summary
        assert summary["layer2"]["faithfulness_avg"] == 0.8
        assert summary["cost"]["estimated_cost"] == 0.001

    def test_omit_optional_when_none(self):
        summary = serialization.build_summary(_make_output([_make_result()]))
        assert "layer2" not in summary
        assert "cost" not in summary


class TestSerialize:
    def test_serialize_result_full_fields(self):
        data = serialization.serialize_result(_make_result())
        assert data["query_id"] == "Q1"
        assert data["recall_at_k"] == 0.5
        assert data["final_context_text"] == "正文"

    def test_serialize_judge_result(self):
        data = serialization.serialize_judge_result(_make_judge())
        assert data["faithfulness"] == 0.8
        assert data["verdict"] == "pass"
        assert data["retrieve_ms"] is None


class TestGenerateReport:
    def test_writes_four_report_files(self, tmp_path):
        output = _make_output([_make_result()])
        results_dir = reporter.generate_report(
            output, str(tmp_path / "run"), {"test_mode": "full", "benchmark": "b"},
            judge_results=[_make_judge()],
            metrics_summary={"estimated_cost": 0.001},
        )
        assert results_dir == str(tmp_path / "run")
        for name in ("per_query.json", "failures.json", "summary.json", "run_info.json"):
            assert (tmp_path / "run" / name).exists()
        assert not (tmp_path / "run" / "history.jsonl").exists()  # history 已废止
        summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
        assert summary["aggregate"]["recall_at_k"] == 0.5
        assert "layer2" in summary and "cost" in summary
        per_query = json.loads((tmp_path / "run" / "per_query.json").read_text(encoding="utf-8"))
        assert per_query["Q1"]["faithfulness"] == 0.8  # layer2 已合并进 per_query
