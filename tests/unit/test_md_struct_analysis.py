"""unit：Markdown 结构静态分析（preprocess/md_struct_analysis.py）。"""
from preprocess.md_struct_analysis import (
    parse_headings, check_heading_continuity, heading_density_ok,
    section_token_statistics, text_ratio, detect_encoding_issues, Heading,
)
from indexing.splitter.utils import token_estimate


class TestParseHeadings:
    def test_skips_code_block_and_records_position(self):
        text = "# Title\n\n## Section\n\n```\n# fake\n```\n\n### Sub"
        headings = parse_headings(text)
        assert [heading.text for heading in headings] == ["Title", "Section", "Sub"]
        assert [heading.level for heading in headings] == [1, 2, 3]
        assert headings[0].line_number == 1
        assert headings[0].char_start == 0

    def test_no_headings(self):
        assert parse_headings("纯文本") == []


class TestCheckHeadingContinuity:
    def test_continuous_ok(self):
        headings = [Heading(1, "a", 0, 0, 1), Heading(2, "b", 1, 1, 2), Heading(3, "c", 2, 2, 3)]
        assert check_heading_continuity(headings) is True

    def test_skip_level_intercepted(self):
        headings = [Heading(1, "a", 0, 0, 1), Heading(3, "b", 1, 1, 2)]
        assert check_heading_continuity(headings) is False

    def test_regression_from_deeper_allowed(self):
        headings = [Heading(4, "a", 0, 0, 1), Heading(2, "b", 1, 1, 2)]
        assert check_heading_continuity(headings) is True

    def test_empty_returns_false(self):
        assert check_heading_continuity([]) is False


class TestHeadingDensityOk:
    def test_empty_returns_false(self):
        assert heading_density_ok([], "text") is False

    def test_h1_exempt_with_min_heading_count(self):
        headings = [Heading(1, "a", index, index, index) for index in range(3)]
        assert heading_density_ok(headings, "x") is True

    def test_density_ratio_controls_without_h1(self):
        headings = [Heading(2, "a", index, index, index) for index in range(3)]
        assert heading_density_ok(headings, "x" * 3000) is True   # 3000/3=1000 ≤ 2000
        assert heading_density_ok(headings, "x" * 9000) is False  # 9000/3=3000 > 2000


class TestSectionTokenStatistics:
    def test_no_headings_whole_doc(self):
        text = "hello world"
        max_tokens, median_tokens = section_token_statistics(text, [])
        assert max_tokens == token_estimate(text)
        assert median_tokens == token_estimate(text)

    def test_with_sections_max_ge_median(self):
        text = "# A\n\naaaa\n\n## B\n\nbbbbbbbb"
        headings = parse_headings(text)
        max_tokens, median_tokens = section_token_statistics(text, headings)
        assert max_tokens >= median_tokens > 0


class TestTextRatio:
    def test_empty_returns_zero(self):
        assert text_ratio("") == 0.0

    def test_pure_text_full(self):
        assert text_ratio("helloworld") == 1.0

    def test_whitespace_lowers_ratio(self):
        assert 0.0 < text_ratio("hello world") < 1.0

    def test_code_excluded(self):
        text = "hello\n```\ncode\n```\nworld"  # 非空白纯文本 10 字符 / 总 24
        assert text_ratio(text) == 10 / 24


class TestDetectEncodingIssues:
    def test_replacement_char_detected(self):
        assert detect_encoding_issues("bad � char") is True

    def test_clean_text(self):
        assert detect_encoding_issues("normal text") is False
