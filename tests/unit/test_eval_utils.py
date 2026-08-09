"""unit：eval 通用工具（eval/utils.py）——I/O 回环、统计、字符串、LLM 解析、运维。"""
import json

import pytest

from eval.utils import (
    read_json, write_json, append_jsonl, avg_of, percentile, p95,
    stem, fill, format_time, clamp_score, extract_json, get_git_commit,
)


class TestReadWriteJson:
    def test_write_then_read_roundtrip(self, tmp_path):
        path = tmp_path / "data.json"
        write_json(path, {"key": "中文值"})
        assert read_json(path) == {"key": "中文值"}

    def test_read_missing_returns_none(self, tmp_path):
        assert read_json(tmp_path / "missing.json") is None

    def test_read_corrupted_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert read_json(path) is None

    def test_read_returns_list(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text('[{"a": 1}]', encoding="utf-8")
        assert read_json(path) == [{"a": 1}]


class TestAppendJsonl:
    def test_appends_one_line_per_record(self, tmp_path):
        path = tmp_path / "history.jsonl"
        append_jsonl(path, {"run_id": "r1"})
        append_jsonl(path, {"run_id": "r2"})
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line)["run_id"] for line in lines] == ["r1", "r2"]


class TestStats:
    def test_avg_of_ignores_none(self):
        assert avg_of([1.0, 2.0, 3.0, None]) == pytest.approx(2.0)

    def test_avg_of_all_none_returns_none(self):
        assert avg_of([None, None]) is None

    def test_percentile_median_even(self):
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_percentile_linear_interp(self):
        # numpy.percentile 默认线性插值：[1..100] 的 p95 = 95.05
        assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)

    def test_percentile_edges(self):
        assert percentile([1, 2, 3, 4], 0) == pytest.approx(1.0)
        assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)

    def test_percentile_empty_returns_zero(self):
        assert percentile([], 95) == 0.0

    def test_p95_empty_returns_zero(self):
        assert p95([]) == 0.0


class TestStringPath:
    def test_stem_lowercases_and_drops_extension(self):
        assert stem("D:/docs/My_File.MD") == "my_file"

    def test_stem_keeps_hidden_prefix(self):
        assert stem("/data/.hidden.txt") == ".hidden"

    def test_fill_replaces_placeholders(self):
        assert fill("{a} 与 {b}", a="甲", b="乙") == "甲 与 乙"

    def test_format_time(self):
        assert format_time(3661) == "01:01:01"


class TestClampScore:
    def test_in_range_unchanged(self):
        assert clamp_score(0.5) == 0.5

    def test_below_lo(self):
        assert clamp_score(-1.0) == 0.0

    def test_above_hi(self):
        assert clamp_score(2.0) == 1.0

    def test_custom_bounds(self):
        assert clamp_score(5.0, lo=1.0, hi=3.0) == 3.0


class TestExtractJson:
    def test_direct_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        text = "说明文字\n```json\n{\"a\": 1}\n```"
        assert extract_json(text) == {"a": 1}

    def test_fence_without_json_tag(self):
        text = "```\n{\"a\": 2}\n```"
        assert extract_json(text) == {"a": 2}

    def test_brace_scan_with_prefix(self):
        assert extract_json('result: {"a": {"b": 3}}') == {"a": {"b": 3}}

    def test_no_brace_raises(self):
        with pytest.raises(ValueError):
            extract_json("没有花括号的文本")

    def test_unmatched_brace_raises(self):
        with pytest.raises(ValueError):
            extract_json('{"a": 1')


class TestGetGitCommit:
    def test_returns_short_hash_or_unknown(self):
        value = get_git_commit()
        assert len(value) == 7 or value == "unknown"
