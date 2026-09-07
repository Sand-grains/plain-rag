"""unit: indexing/parse_backends 质量代理信号(012-2 2.3)——空产出/表格/代码/标题连续性。"""
import pytest

from indexing.parse_backends.quality import (
    code_fence_count,
    code_fence_retention,
    heading_continuous,
    is_empty,
    quality_ok,
    table_count,
    table_retention,
)


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测质量信号, 无需重依赖。"""
    yield


class TestIsEmpty:
    def test_empty(self):
        assert is_empty("   \n  ") is True

    def test_non_empty(self):
        assert is_empty("正文") is False


class TestTableCount:
    def test_counts_tables(self):
        md = "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |"
        assert table_count(md) == 2

    def test_zero(self):
        assert table_count("无表格") == 0


class TestCodeFenceCount:
    def test_counts_fences(self):
        md = "```py\nx=1\n```\n\n```\ny=2\n```"
        assert code_fence_count(md) == 2


class TestHeadingContinuous:
    def test_continuous(self):
        assert heading_continuous("# a\n## b\n### c") is True

    def test_jump_discontinuous(self):
        assert heading_continuous("# a\n### c") is False

    def test_no_heading(self):
        assert heading_continuous("正文") is True


class TestRetention:
    def test_table_retention(self):
        assert table_retention("| a |\n|---|\n| 1 |", 2) == 0.5
        assert table_retention("| a |\n|---|\n| 1 |", None) is None
        assert table_retention("| a |\n|---|\n| 1 |", 0) is None

    def test_code_fence_retention(self):
        assert code_fence_retention("```\nx\n```", 2) == 0.5
        assert code_fence_retention("```\nx\n```", None) is None


class TestQualityOk:
    def test_ok_when_all_good(self):
        md = "# 标题\n| a |\n|---|\n| 1 |\n```\nx\n```"
        ok, reasons, signals = quality_ok(md, original_table_count=1, expected_code_blocks=1)
        assert ok is True and reasons == []

    def test_empty_fails(self):
        ok, reasons, _ = quality_ok("   ")
        assert ok is False and "empty" in reasons

    def test_table_loss_fails(self):
        ok, reasons, _ = quality_ok("正文", original_table_count=2)
        assert ok is False and "table_loss" in reasons

    def test_code_fence_loss_fails(self):
        ok, reasons, _ = quality_ok("正文", expected_code_blocks=2)
        assert ok is False and "code_fence_loss" in reasons

    def test_heading_discontinuous_fails(self):
        ok, reasons, _ = quality_ok("# a\n### c")
        assert ok is False and "heading_discontinuous" in reasons
