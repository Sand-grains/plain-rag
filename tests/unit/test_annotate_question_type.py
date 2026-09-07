"""unit: scripts/annotate_question_type 启发式 question_type 标注(011-1)。"""
import json
from pathlib import Path

import pytest

from scripts import annotate_question_type as aqt


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测启发式逻辑, 无需重依赖。"""
    yield


class TestInfer:
    def test_multi_chunk_is_multi_hop(self):
        assert aqt.infer_question_type("任何问题", "multi_chunk") == "multi_hop"

    def test_comparison_markers(self):
        assert aqt.infer_question_type("A 和 B 有什么区别？", "single_chunk") == "comparison"
        assert aqt.infer_question_type("对比 X 与 Y 的优劣", "single_chunk") == "comparison"

    def test_conditional_markers(self):
        assert aqt.infer_question_type("如果用户指定了文件，第一步做什么？", "single_chunk") == "conditional"
        assert aqt.infer_question_type("什么情况下会触发熔断？", "single_chunk") == "conditional"

    def test_factual_fallback(self):
        assert aqt.infer_question_type("MCP 协议是什么？", "single_chunk") == "factual"


class TestAnnotate:
    def test_adds_field(self):
        items = [
            {"query": "A 和 B 的区别", "difficulty": "single_chunk"},
            {"query": "如果 X 会怎样", "difficulty": "single_chunk"},
            {"query": "MCP 是什么", "difficulty": "single_chunk"},
            {"query": "跨块问题", "difficulty": "multi_chunk"},
        ]
        aqt.annotate(items)
        assert [i["question_type"] for i in items] == [
            "comparison", "conditional", "factual", "multi_hop"]


class TestMain:
    def test_writes_back(self, tmp_path):
        path = tmp_path / "private_builtin.json"
        path.write_text(json.dumps([
            {"query": "A 和 B 的区别", "difficulty": "single_chunk"},
            {"query": "MCP 是什么", "difficulty": "single_chunk"},
        ], ensure_ascii=False), encoding="utf-8")
        assert aqt.main(["--in", str(path)]) == 0
        items = json.loads(path.read_text(encoding="utf-8"))
        assert items[0]["question_type"] == "comparison"
        assert items[1]["question_type"] == "factual"
