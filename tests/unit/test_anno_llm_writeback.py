"""test_anno_llm_writeback.py：F49 writeback.py 单元测试(重拼 gold / 分流 / 删除写回 / 报告 / fail-fast)。

覆盖 anno_llm.md §7 的 rebuild_reference_facts / confidence_split / apply_delete_list /
report_grounding_rate 用例, 以及 §4.5 fail-fast 健壮性。
"""
from types import SimpleNamespace

import pytest

from benchmark.anno_llm.writeback import (
    IndexCoverageError,
    apply_delete_list,
    assert_index_coverable,
    confidence_split,
    rebuild_reference_facts,
    report_grounding_rate,
    write_benchmark,
)


# ---- rebuild_reference_facts ----

def test_rebuild_reference_facts_concatenates_quotes_in_order():
    """按 expected_parent_ids 次序逐字拼接 evidence_quote(顺序稳定)。"""
    annotation = {
        "expected_parent_ids": ["Crawler/0000:p0", "Crawler/0000:p2"],
        "evidence_quote": {
            "Crawler/0000:p2": "第二部分引用",
            "Crawler/0000:p0": "第一部分引用",
        },
    }
    assert rebuild_reference_facts(annotation) == "第一部分引用; 第二部分引用"


def test_rebuild_reference_facts_skips_empty_quotes():
    """空引用不参与拼接。"""
    annotation = {"expected_parent_ids": ["a:p0", "a:p1"],
                  "evidence_quote": {"a:p0": "内容", "a:p1": "  "}}
    assert rebuild_reference_facts(annotation) == "内容"


def test_rebuild_reference_facts_empty_when_no_quote():
    """无任何引用 → 空串。"""
    assert rebuild_reference_facts({"expected_parent_ids": [], "evidence_quote": {}}) == ""


# ---- confidence_split ----

def test_confidence_split_threshold():
    """confidence >= 阈值进自动采纳, 否则进 review。"""
    annotations = [
        {"query_id": "a", "confidence": 0.9},
        {"query_id": "b", "confidence": 0.8},
        {"query_id": "c", "confidence": 0.5},
    ]
    auto, review = confidence_split(annotations, threshold=0.8)
    assert [item["query_id"] for item in auto] == ["a", "b"]
    assert [item["query_id"] for item in review] == ["c"]


# ---- apply_delete_list ----

def test_apply_delete_list_removes_and_renumbers():
    """自动剔除删除项并重排 query_id 为 crawler-<idx:04d>。"""
    items = [
        {"query_id": "crawler-0000", "query": "q0"},
        {"query_id": "crawler-0001", "query": "q1"},
        {"query_id": "crawler-0002", "query": "q2"},
    ]
    remaining = apply_delete_list(items, ["crawler-0001"])
    assert [item["query_id"] for item in remaining] == ["crawler-0000", "crawler-0001"]
    # 重排后 index 被复用: crawler-0001 现在是 q2(被删的 q1 已不在)
    assert [item["query"] for item in remaining] == ["q0", "q2"]


def test_apply_delete_list_custom_prefix():
    """query_id 重排前缀可参数化(P2-5), 不硬编码 crawler-。"""
    remaining = apply_delete_list([{"query_id": "crawler-0000", "query": "q"}], [], prefix="custom-")
    assert remaining[0]["query_id"] == "custom-0000"


# ---- report_grounding_rate ----

def test_report_grounding_rate():
    """grounding 率 = grounded 条目占比。"""
    results = [{"query_id": "a", "grounded": True},
               {"query_id": "b", "grounded": True},
               {"query_id": "c", "grounded": False}]
    report = report_grounding_rate(results)
    assert report["total"] == 3
    assert report["grounded"] == 2
    assert report["rate"] == pytest.approx(2 / 3)


def test_report_grounding_rate_empty():
    """空结果 rate=0.0 不除零。"""
    assert report_grounding_rate([])["rate"] == 0.0


# ---- write_benchmark ----

def test_write_benchmark_atomic(tmp_path):
    """write_benchmark 复用 anno_tool.save_benchmark 原子写回。"""
    target = tmp_path / "out.json"
    write_benchmark([{"query_id": "crawler-0000"}], str(target))
    import json
    assert json.loads(target.read_text(encoding="utf-8")) == [{"query_id": "crawler-0000"}]


# ---- assert_index_coverable (fail-fast) ----

def _chunk(chunk_id: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, content="内容")


def test_assert_index_coverable_ok():
    """全部 source_doc 可解析 → 不抛错。"""
    doc_index = {"Crawler/0000": [_chunk("Crawler/0000:p0")]}
    items = [{"query_id": "crawler-0000", "source_doc": "Crawler/0000"}]
    assert_index_coverable(doc_index, items)  # 不抛即通过


def test_assert_index_coverable_missing_raises():
    """存在缺失 source_doc → fail-fast 抛 IndexCoverageError。"""
    doc_index = {"Crawler/0000": [_chunk("Crawler/0000:p0")]}
    items = [{"query_id": "crawler-0001", "source_doc": "Crawler/9999"},
             {"query_id": "crawler-0000", "source_doc": "Crawler/0000"}]
    with pytest.raises(IndexCoverageError):
        assert_index_coverable(doc_index, items)


def test_assert_index_coverable_fuzzy_match_counts_resolved():
    """子串模糊命中算可解析(不留 fail-fast 误报)。"""
    doc_index = {"Crawler/0000": [_chunk("Crawler/0000:p0")]}
    items = [{"query_id": "crawler-0000", "source_doc": "Crawler/0000/子目录"}]
    # Crawler/0000 是 Crawler/0000/子目录 的子串 → 命中
    assert_index_coverable(doc_index, items)
