"""unit: 共享工具函数(本重构配套测试), md_escape 与 sorted_items。"""
from harness.core.models import FeatureItem
from harness.core.utils import md_escape, sorted_items


def _item(item_id: str) -> FeatureItem:
    return FeatureItem(id=item_id, behavior=f"行为-{item_id}", gate="pytest -q")


class TestMdEscape:
    def test_escapes_pipe(self):
        assert md_escape("a|b") == "a\\|b"

    def test_newline_to_space(self):
        assert md_escape("a\nb") == "a b"

    def test_plain_text_unchanged(self):
        assert md_escape("普通文本") == "普通文本"

    def test_empty_string(self):
        assert md_escape("") == ""


class TestSortedItems:
    def test_sorts_by_id(self):
        items = [_item("F03"), _item("F01"), _item("F02")]
        assert [item.id for item in sorted_items(items)] == ["F01", "F02", "F03"]

    def test_empty_list(self):
        assert sorted_items([]) == []

    def test_does_not_mutate_input(self):
        items = [_item("F02"), _item("F01")]
        original = [item.id for item in items]
        sorted_items(items)
        assert [item.id for item in items] == original
