"""unit: eval/core/parse_metrics 多格式解析质量指标(011-6)——文本召回率/精确率 + 结构保真度。"""
import pytest

from eval.core.parse_metrics import (
    text_recall, text_precision, heading_fidelity, table_fidelity,
    code_fidelity, structure_fidelity,
)


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测纯文本指标, 无需重依赖。"""
    yield


class TestTextRecall:
    def test_identical_is_one(self):
        assert text_recall("hello world", "hello world") == 1.0

    def test_disjoint_is_zero(self):
        assert text_recall("abc", "xyz") == 0.0

    def test_partial_overlap(self):
        recall = text_recall("hello world foo", "hello world bar")
        assert 0.0 < recall < 1.0

    def test_whitespace_normalized(self):
        assert text_recall("hello\nworld", "hello world") == 1.0

    def test_empty_gold(self):
        assert text_recall("", "") == 1.0
        assert text_recall("abc", "") == 0.0


class TestTextPrecision:
    def test_identical_is_one(self):
        assert text_precision("hello world", "hello world") == 1.0

    def test_extra_text_lowers_precision(self):
        precision = text_precision("hello world extra noise", "hello world")
        assert 0.0 < precision < 1.0

    def test_empty_parsed_is_one(self):
        assert text_precision("", "hello") == 1.0


class TestHeadingFidelity:
    def test_identical_headings(self):
        parsed = "# 标题A\n## 标题B"
        gold = "# 标题A\n## 标题B"
        assert heading_fidelity(parsed, gold) == 1.0

    def test_missing_heading_lowers(self):
        parsed = "# 标题A"
        gold = "# 标题A\n## 标题B"
        assert heading_fidelity(parsed, gold) == 0.5

    def test_no_gold_heading_is_one(self):
        assert heading_fidelity("正文", "正文") == 1.0


class TestTableFidelity:
    def test_identical_table(self):
        parsed = "| a | b |\n|---|---|\n| 1 | 2 |"
        gold = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert table_fidelity(parsed, gold) == 1.0

    def test_different_structure_lowers(self):
        parsed = "| a |\n|---|\n| 1 |"
        gold = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert table_fidelity(parsed, gold) == 0.0

    def test_no_gold_table_is_one(self):
        assert table_fidelity("正文", "正文") == 1.0


class TestCodeFidelity:
    def test_identical_code(self):
        parsed = "```python\nprint(1)\n```"
        gold = "```python\nprint(1)\n```"
        assert code_fidelity(parsed, gold) == 1.0

    def test_different_code_lowers(self):
        parsed = "```python\nprint(2)\n```"
        gold = "```python\nprint(1)\n```"
        assert code_fidelity(parsed, gold) == 0.0

    def test_no_gold_code_is_one(self):
        assert code_fidelity("正文", "正文") == 1.0


class TestStructureFidelity:
    def test_aggregate(self):
        parsed = "# 标题A\n| a | b |\n|---|---|\n| 1 | 2 |\n```python\nprint(1)\n```"
        gold = "# 标题A\n| a | b |\n|---|---|\n| 1 | 2 |\n```python\nprint(1)\n```"
        result = structure_fidelity(parsed, gold)
        assert result["heading"] == 1.0
        assert result["table"] == 1.0
        assert result["code"] == 1.0
        assert result["overall"] == 1.0
