"""test_anno_llm_validation.py：F51 validation.py 单元测试(双门控 + Q4 + 健康阈值 + 人工检查点)。

覆盖 anno_llm.md §6 的门控 1 宽松口径 / 门控 2 pilot / Q4 坏 query 处置 / 健康阈值 /
前置验证人工检查点, 以及制副本+清洗(堵泄漏)。
"""
import json
from types import SimpleNamespace

import pytest

from benchmark.anno_llm.validation import (
    attribution_summary,
    baseline_diagnosis,
    checkpoint_status,
    gate1_hit_rate,
    gate2_pilot_summary,
    health_review_ratio,
    make_sanitized_copy,
    precision_metrics,
    precision_sampling,
    preflight,
    q4_disposition_accuracy,
    sanitize_entries,
    scheme_b_decision,
)


def _chunk(chunk_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, content=content)


DOC_INDEX = {
    "Crawler/0": [
        _chunk("Crawler/0:p0", "设计模式分为三大类 创建型 结构型 行为型 单例工厂"),
        _chunk("Crawler/0:p1", "食堂饭菜口味 很不错"),
        _chunk("Crawler/0:p2", "模式 分为 创建 结构 行为 设计 分类"),
    ],
}

GOLD = [
    {"query_id": "crawler-0000", "source_doc": "Crawler/0",
     "expected_parent_ids": ["Crawler/0:p0"], "reference_facts": "设计模式分为三大类"},
]


def _passed(ids, qid="crawler-0000"):
    return {"query_id": qid, "source_doc": "Crawler/0", "expected_parent_ids": ids}


# ---- sanitize_entries / make_sanitized_copy ----

def test_sanitize_entries_clears_gold_and_metadata():
    """清洗: 只留元数据, 清空 gold 字段, 删除 expected_child_ids(堵泄漏)。"""
    items = [{
        "query_id": "Q0001", "query": "q", "source_doc": "Crawler/0", "category": "crawler",
        "expected_files": ["Crawler/0"], "expected_pages": [],
        "expected_parent_ids": ["Crawler/0:p0"], "expected_child_ids": ["Crawler/0:p0:c0"],
        "relevance": {"Crawler/0:p0": 3}, "difficulty": "single_chunk",
        "question_type": "factual", "reference_facts": "正确gold",
    }]
    cleaned = sanitize_entries(items)
    assert cleaned[0]["query_id"] == "Q0001"
    assert "expected_child_ids" not in cleaned[0]
    assert "expected_parent_ids" not in cleaned[0]
    assert "reference_facts" not in cleaned[0]
    assert "difficulty" not in cleaned[0]
    assert "relevance" not in cleaned[0]
    assert cleaned[0]["query"] == "q"


def test_make_sanitized_copy_writes(tmp_path):
    """制清洗副本并返回条目数。"""
    src = tmp_path / "src.json"
    src.write_text('[{"query_id": "Q1", "query": "q", "source_doc": "Crawler/0", '
                   '"category": "crawler", "expected_files": ["Crawler/0"], "expected_pages": [], '
                   '"expected_parent_ids": ["x"], "reference_facts": "gold"}]', encoding="utf-8")
    dst = tmp_path / "anno_test.json"
    count = make_sanitized_copy(str(src), str(dst))
    import json
    data = json.loads(dst.read_text(encoding="utf-8"))
    assert count == 1
    assert "expected_parent_ids" not in data[0]
    assert "reference_facts" not in data[0]


# ---- gate1_hit_rate ----

def test_gate1_exact_hit():
    """自动标注块 ∈ 人工 gold → 命中。"""
    result = gate1_hit_rate([_passed(["Crawler/0:p0"])], GOLD, DOC_INDEX)
    assert result["total"] == 1
    assert result["hits"] == 1
    assert result["rate"] == 1.0


def test_gate1_equivalence_hit():
    """自动标注块 ∈ 等价集(同 doc 术语重叠≥0.3) → 命中(宽松口径)。"""
    result = gate1_hit_rate([_passed(["Crawler/0:p2"])], GOLD, DOC_INDEX)
    assert result["hits"] == 1
    assert result["rate"] == 1.0


def test_gate1_miss():
    """自动标注块既不在 gold 也不在等价集 → 未命中。"""
    result = gate1_hit_rate([_passed(["Crawler/0:p1"])], GOLD, DOC_INDEX)
    assert result["hits"] == 0
    assert result["rate"] == 0.0


def test_gate1_excludes_unmatched():
    """passed 与 gold 无法按 query_id 对齐的条目不计入门控。"""
    result = gate1_hit_rate([_passed([], qid="unmatched")], GOLD, DOC_INDEX)
    assert result["total"] == 0
    assert result["rate"] == 0.0


# ---- gate2_pilot_summary ----

def test_gate2_semantic_rate_and_dispositions():
    """门控2: 语义相关率 + 噪音处置量化(该删/该改/可留)。"""
    results = [
        {"query_id": "a", "semantic_relevant": True, "disposition": "keep"},
        {"query_id": "b", "semantic_relevant": True, "disposition": "fix"},
        {"query_id": "c", "semantic_relevant": False, "disposition": "delete"},
    ]
    summary = gate2_pilot_summary(results)
    assert summary["semantic_rate"] == pytest.approx(2 / 3)
    assert summary["disposition_counts"] == {"keep": 1, "fix": 1, "delete": 1}
    assert summary["passed"] is False  # < 0.8


def test_gate2_passed_at_threshold():
    """语义相关率 ≥ 0.8 → passed(主门控)。"""
    results = [{"query_id": f"crawler-{i:04d}", "semantic_relevant": True, "disposition": "keep"}
               for i in range(8)]
    results[1] = {"query_id": "b", "semantic_relevant": True, "disposition": "keep"}
    assert gate2_pilot_summary(results)["passed"] is True


def test_gate2_empty():
    """空 pilot → 不 passed。"""
    assert gate2_pilot_summary([])["passed"] is False


# ---- q4_disposition_accuracy ----

def test_q4_accuracy():
    """坏 query 处置正确率: should_delete 与 did_delete 一致才计对; 阈值留待 pilot 校准。"""
    results = [
        {"query_id": "a", "should_delete": True, "did_delete": True},
        {"query_id": "b", "should_delete": True, "did_delete": False},  # 该删未删
        {"query_id": "c", "should_delete": False, "did_delete": False},
    ]
    result = q4_disposition_accuracy(results)
    assert result["correct"] == 2
    assert result["total"] == 3
    assert result["rate"] == pytest.approx(2 / 3)
    assert result["threshold_pending"] is True  # 不硬断 pass, 由 pilot 校准


def test_q4_empty():
    result = q4_disposition_accuracy([])
    assert result["correct"] == 0
    assert result["threshold_pending"] is True


# ---- health_review_ratio ----

def test_health_review_ratio_issue_when_over_limit():
    """review > 35% ⇒ 计划有问题(非阻断)。"""
    issue = health_review_ratio(review_count=40, total=100)
    assert issue["ratio"] == 0.4
    assert issue["plan_issue"] is True


def test_health_review_ratio_ok():
    assert health_review_ratio(review_count=30, total=100)["plan_issue"] is False


def test_health_review_ratio_empty_total():
    assert health_review_ratio(0, 0)["plan_issue"] is False


# ---- checkpoint_status ----

def test_checkpoint_requires_manual_pause():
    """前置验证人工检查点: 产出后停下等人工, 不自动放行。"""
    status = checkpoint_status({"hits": 10, "total": 12})
    assert status["requires_manual_checkpoint"] is True
    assert status["approved"] is False  # 必须人工逐条校对后才放行
    assert "停下" in status["message"]


def test_preflight_orchestration(tmp_path):
    """§6 前置验证端到端编排: 制副本→gate1→checkpoint 停下(非 LLM 部分)。"""
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps([{
        "query_id": "crawler-0000", "query": "设计模式分几类？", "source_doc": "Crawler/0",
        "category": "crawler", "expected_files": ["Crawler/0"], "expected_pages": [],
        "expected_parent_ids": ["Crawler/0:p0"], "reference_facts": "设计模式分为三大类",
    }]), encoding="utf-8")
    doc_index = {
        "Crawler/0": [
            SimpleNamespace(chunk_id="Crawler/0:p0",
                            content="设计模式分为三大类 创建型 结构型 行为型"),
            SimpleNamespace(chunk_id="Crawler/0:p1", content="食堂饭菜 味道 不错"),
        ],
    }
    passed_entries = [{"query_id": "crawler-0000", "source_doc": "Crawler/0",
                     "expected_parent_ids": ["Crawler/0:p1"]}]
    target = tmp_path / "anno_test.json"
    report = preflight(json.loads(gold_path.read_text(encoding="utf-8")), passed_entries,
                       doc_index, str(gold_path), str(target))
    assert report["sanitized_count"] == 1
    assert report["checkpoint"]["approved"] is False  # 必须停下等人工
    # 清洗副本不含 gold 字段(堵泄漏)
    cleaned = json.loads(target.read_text(encoding="utf-8"))
    assert "reference_facts" not in cleaned[0]
    assert "expected_parent_ids" not in cleaned[0]


# ---- F52: 精度向防全标指标 ----

def test_precision_metrics_overlap_and_block_count():
    """精度向防全标: 重叠率 + 平均块数对比(全标模型块数显著偏多)。"""
    gold = [
        {"query_id": "Q1", "expected_parent_ids": ["a:p0", "a:p1"]},
        {"query_id": "Q2", "expected_parent_ids": ["b:p0"]},
    ]
    passed = [
        {"query_id": "Q1", "expected_parent_ids": ["a:p0", "a:p1", "a:p2", "a:p3"]},  # 全标: 块数偏多
        {"query_id": "Q2", "expected_parent_ids": ["b:p0"]},
    ]
    result = precision_metrics(passed, gold)
    assert result["total"] == 2
    # Q1 重叠 2/2=1.0, Q2 重叠 1/1=1.0 → 平均 1.0
    assert result["overlap_rate"] == pytest.approx(1.0)
    # 平均块数: passed (4+1)/2=2.5 vs gold (2+1)/2=1.5 → 全标偏多
    assert result["avg_passed_blocks"] == pytest.approx(2.5)
    assert result["avg_gold_blocks"] == pytest.approx(1.5)


def test_precision_metrics_low_overlap_detects_all_label():
    """全标模型: 重叠率低(标了无关块) → 指标暴露。"""
    gold = [{"query_id": "Q1", "expected_parent_ids": ["a:p0"]}]
    passed = [{"query_id": "Q1", "expected_parent_ids": ["a:p0", "a:p9"]}]  # 多标一个无关块
    result = precision_metrics(passed, gold)
    assert result["overlap_rate"] == pytest.approx(1.0)  # gold 全被覆盖
    assert result["avg_passed_blocks"] == 2.0
    assert result["avg_gold_blocks"] == 1.0


def test_precision_metrics_excludes_unmatched():
    """无法按 query_id 对齐的条目不计入精度指标。"""
    gold = [{"query_id": "Q1", "expected_parent_ids": ["a:p0"]}]
    passed = [{"query_id": "unmatched", "expected_parent_ids": ["a:p0"]}]
    result = precision_metrics(passed, gold)
    assert result["total"] == 0
    assert result["overlap_rate"] == 0.0


# ---- F52: 前置基线诊断 ----

def test_baseline_diagnosis_proceed_when_gate1_high():
    """gate1 ≥ 0.8 → proceed=True, 可进入阈值校准。"""
    result = baseline_diagnosis({"rate": 0.85, "hits": 17, "total": 20})
    assert result["proceed"] is True
    assert "可进入阈值校准" in result["message"]


def test_baseline_diagnosis_block_when_gate1_low():
    """gate1 < 0.8 → proceed=False, 先修 prompt/机制, 不进入校准。"""
    result = baseline_diagnosis({"rate": 0.6, "hits": 12, "total": 20})
    assert result["proceed"] is False
    assert "先修 prompt" in result["message"]


# ---- 009: 方案乙精度抽样(部分命中即正) ----

def test_precision_sampling_partial_hit_is_positive():
    """精度抽样: 部分命中(交集非空)即正(与 gold 集合包含关系)。"""
    gold = [
        {"query_id": "Q1", "expected_parent_ids": ["a:p0", "a:p1"]},
        {"query_id": "Q2", "expected_parent_ids": ["b:p0"]},
    ]
    passed = [
        {"query_id": "Q1", "expected_parent_ids": ["a:p0", "a:p9"]},  # 部分命中 p0 → 正
        {"query_id": "Q2", "expected_parent_ids": ["b:p7"]},          # 未命中 → 负
    ]
    result = precision_sampling(passed, gold)
    assert result["total"] == 2
    assert result["hits"] == 1
    assert result["rate"] == pytest.approx(0.5)


def test_precision_sampling_sample_size_limits():
    """精度抽样: sample_size 限制抽样条数(取前 N 条对齐条目)。"""
    gold = [{"query_id": f"Q{index}", "expected_parent_ids": [f"a:p{index}"]} for index in range(5)]
    passed = [{"query_id": f"Q{index}", "expected_parent_ids": [f"a:p{index}"]} for index in range(5)]
    result = precision_sampling(passed, gold, sample_size=3)
    assert result["total"] == 3
    assert result["hits"] == 3
    assert result["rate"] == 1.0


def test_precision_sampling_excludes_unmatched():
    """无法按 query_id 对齐的条目不计入精度抽样。"""
    gold = [{"query_id": "Q1", "expected_parent_ids": ["a:p0"]}]
    passed = [{"query_id": "unmatched", "expected_parent_ids": ["a:p0"]}]
    result = precision_sampling(passed, gold)
    assert result["total"] == 0
    assert result["rate"] == 0.0


# ---- 009: 方案乙决策规则 ----

def test_scheme_b_decision_proceed_full():
    """乙下 review 率 < 35% 且精度抽样 ≥ 0.8 → 全量。"""
    result = scheme_b_decision(review_ratio=0.2, precision=0.9)
    assert result["proceed_full"] is True
    assert "可全量" in result["message"]


def test_scheme_b_decision_fallback_when_review_high():
    """乙下 review 率 ≥ 35% → 回退甲/丙。"""
    result = scheme_b_decision(review_ratio=0.4, precision=0.9)
    assert result["proceed_full"] is False
    assert "回退甲/丙" in result["message"]


def test_scheme_b_decision_fallback_when_precision_low():
    """乙下精度抽样 < 0.8 → 回退甲/丙。"""
    result = scheme_b_decision(review_ratio=0.2, precision=0.6)
    assert result["proceed_full"] is False


# ---- 009: 归因表汇总 ----

def test_attribution_summary_counts_by_reason():
    """归因表汇总: 按 reject 子原因统计条数(一条可含多个子原因)。"""
    attribution = [
        {"query_id": "Q1", "reject_reasons": ["cos", "term_overlap"]},
        {"query_id": "Q2", "reject_reasons": ["cos"]},
        {"query_id": "Q3", "reject_reasons": ["quote_absent"]},
    ]
    summary = attribution_summary(attribution)
    assert summary["total"] == 3
    assert summary["by_reason"] == {"cos": 2, "term_overlap": 1, "quote_absent": 1}


def test_attribution_summary_empty():
    """空归因表 → total=0, by_reason 空。"""
    summary = attribution_summary([])
    assert summary["total"] == 0
    assert summary["by_reason"] == {}
