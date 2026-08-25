"""unit：preprocess/cleaner.py 确定性清洗（008-3）。

覆盖：幂等性、样板行去除、URL/邮箱归一化、HTML 残留清理、零宽/控制字符清理，
以及"不去除正常重复内容"（重复标题/法律条款不误删）。
"""
from preprocess.cleaner import clean


class TestIdempotent:
    def test_clean_clean_equals_clean(self):
        samples = [
            "普通正文，没有特殊内容。",
            "版权所有 © 2026 示例公司。\n正文段落。\n关注公众号获取更多。",
            "参考 https://example.com/a?page=2&x=1#top 的内容。\n正文。",
            "残留 <b>加粗</b> 与 <div class=\"x\">块</div>。\n\u200b零宽\u00ad字符\ufeff。",
        ]
        for sample in samples:
            once = clean(sample)
            assert clean(once) == once


class TestBoilerplate:
    def test_removes_boilerplate_lines(self):
        text = "版权声明\n正文第一段。\n关注公众号：xxx\nread more\n正文第二段。\n免责声明：仅供参考"
        result = clean(text)
        assert "版权声明" not in result
        assert "关注公众号" not in result
        assert "read more" not in result
        assert "免责声明" not in result
        assert "正文第一段。" in result
        assert "正文第二段。" in result

    def test_keeps_normal_repeated_content(self):
        # 重复标题/法律条款不是样板行，不误删
        text = "## 引言\n重复标题。\n## 引言\n法律条款如下：\n1. 甲方\n2. 乙方\n第一条 双方应遵守。"
        result = clean(text)
        assert "## 引言" in result
        assert "法律条款如下" in result
        assert "第一条" in result
        assert "1. 甲方" in result


class TestUrlEmail:
    def test_url_truncates_query_and_fragment(self):
        result = clean("见 https://example.com/path?a=1&b=2#section 说明。")
        assert "https://example.com/path" in result
        assert "a=1" not in result
        assert "#section" not in result

    def test_email_lowercased(self):
        result = clean("联系 Foo.Bar@Example.COM。")
        assert "foo.bar@example.com" in result
        assert "Foo.Bar@Example.COM" not in result


class TestHtmlAndControl:
    def test_removes_html_tags(self):
        result = clean("正文 <b>粗体</b> <span style=\"x\">内联</span> 结束。")
        assert "<b>" not in result
        assert "<span" not in result
        assert "粗体" in result
        assert "内联" in result

    def test_removes_zero_width_and_control_chars(self):
        result = clean("a\u200bb\u00adc\ufeffd\u0000e")
        assert result == "abcde"
