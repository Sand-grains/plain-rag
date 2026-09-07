"""test_anno_tool_only.py：F50 anno_tool --only 纯函数单元测试。

覆盖 anno_llm.md §4.4/§11 的向后兼容 --only 参数:
    不传则行为不变; 传逗号分隔 query_id 列表只复核指定条目; 删除清单自动写回、--only 不消费。
filter_by_only 为纯函数, 直接测试, 不触 anno_tool 交互主流程。
"""
from benchmark.anno_tool import filter_by_only


def _items() -> list[dict]:
    return [
        {"query_id": "Q0001", "query": "q1"},
        {"query_id": "Q0002", "query": "q2"},
        {"query_id": "Q0003", "query": "q3"},
    ]


def test_no_only_returns_original():
    """不传 --only(None) → 返回原列表, 行为不变(向后兼容)。"""
    items = _items()
    assert filter_by_only(items, None) is items


def test_only_filters_to_matching_ids():
    """传逗号分隔 query_id → 只返回指定条目(保持原顺序)。"""
    result = filter_by_only(_items(), "Q0001,Q0003")
    assert [item["query_id"] for item in result] == ["Q0001", "Q0003"]


def test_only_trims_whitespace():
    """逗号与空白被容忍。"""
    result = filter_by_only(_items(), " Q0001 , Q0003 ")
    assert [item["query_id"] for item in result] == ["Q0001", "Q0003"]


def test_only_unknown_id_returns_empty():
    """全部不匹配 → 空列表。"""
    assert filter_by_only(_items(), "Q9999") == []


def test_only_empty_string_returns_original():
    """--only 传空串视为不过滤。"""
    items = _items()
    assert filter_by_only(items, "") is items
