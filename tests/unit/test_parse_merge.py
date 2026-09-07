"""unit: indexing/parse_backends 跨页合并(012-2 2.4)——表格续页/有序列表/无序列表。"""
import pytest

from indexing.parse_backends.merge import merge_cross_page


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测跨页合并, 无需重依赖。"""
    yield


class TestMergeCrossPage:
    def test_table_continuation_drops_repeated_header(self):
        page1 = "| 列A | 列B |\n|---|---|\n| 1 | 2 |"
        page2 = "| 列A | 列B |\n|---|---|\n| 3 | 4 |"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is True
        assert "table" in meta["cross_page_merge_kind"]
        # 重复表头被去掉, 只留数据行
        assert merged.count("| 列A | 列B |") == 1
        assert "| 3 | 4 |" in merged

    def test_table_columns_mismatch_no_merge(self):
        page1 = "| 列A | 列B |\n|---|---|\n| 1 | 2 |"
        page2 = "| 列A |\n|---|\n| 3 |"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is False

    def test_ordered_list_continuation(self):
        page1 = "1. 第一项\n2. 第二项"
        page2 = "3. 第三项\n4. 第四项"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is True
        assert "list" in meta["cross_page_merge_kind"]
        assert "3. 第三项" in merged

    def test_ordered_list_gap_no_merge(self):
        page1 = "1. 第一项"
        page2 = "5. 第五项"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is False

    def test_unordered_list_continuation(self):
        page1 = "- 甲\n- 乙"
        page2 = "- 丙"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is True
        assert "- 丙" in merged

    def test_plain_pages_joined_with_separator(self):
        page1 = "第一页正文"
        page2 = "第二页正文"
        merged, meta = merge_cross_page([page1, page2])
        assert meta["cross_page_merged"] is False
        assert "第一页正文\n\n第二页正文" in merged
