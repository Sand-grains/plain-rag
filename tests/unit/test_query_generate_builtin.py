"""unit: scripts/query_generate_builtin 011-1——110 现有 + 新增补到 target, schema 对齐 private_v6。"""
import json
from pathlib import Path

import pytest

from scripts import query_generate_builtin as qgb


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测生成逻辑, 无需重依赖。"""
    yield


def _make_base(tmp_path: Path) -> Path:
    base = tmp_path / "private_v6.json"
    base.write_text(json.dumps([
        {"query_id": "Q0001", "query": "q1", "reference_facts": "rf1",
         "source_doc": "Agent/a", "category": "builtin", "difficulty": "single_chunk",
         "expected_parent_ids": ["Agent/a:p0"], "expected_child_ids": [],
         "relevance": {"Agent/a:p0": 3}, "expected_files": ["Agent/a"], "expected_pages": []},
    ], ensure_ascii=False), encoding="utf-8")
    return base


class TestDeriveDocId:
    def test_relative_path(self, tmp_path):
        assert qgb._derive_doc_id(tmp_path / "Agent" / "a.md", tmp_path) == "Agent/a"


class TestBuildItem:
    def test_schema_matches_private_v6(self):
        item = qgb._build_builtin_item("Q0002", "q", "rf", "Agent/b",
                                       ["Agent/b:p0"], "single_chunk")
        assert item["query_id"] == "Q0002"
        assert item["difficulty"] == "single_chunk"
        assert item["expected_parent_ids"] == ["Agent/b:p0"]
        assert item["expected_child_ids"] == []
        assert "question_type" not in item  # private_v6 无此字段


class TestMain:
    def test_seeds_base_and_appends_new(self, tmp_path, monkeypatch):
        base = _make_base(tmp_path)
        corpus = tmp_path / "data"
        (corpus / "Agent").mkdir(parents=True)
        (corpus / "Agent" / "a.md").write_text("# 标题\n\n正文内容。" * 30, encoding="utf-8")
        (corpus / "Agent" / "b.md").write_text(
            "# 标题\n\n" + "正文内容。" * 200 + "\n\n## 小节\n\n" + "更多内容。" * 200,
            encoding="utf-8")
        (corpus / "Crawler").mkdir()
        (corpus / "Crawler" / "c.md").write_text("# 爬取\n\n内容。" * 30, encoding="utf-8")  # 应被排除
        out = tmp_path / "private_builtin.json"
        monkeypatch.setattr(qgb, "call_llm", lambda p, c, max_retries=1: {"questions": [
            {"query": "nq", "reference_facts": "nrf", "question_type": "factual"}]})
        rc = qgb.main(["--base", str(base), "--corpus", str(corpus), "--out", str(out),
                       "--target", "5"])
        assert rc == 0
        items = json.loads(out.read_text(encoding="utf-8"))
        # 1 条 base + 新增(单篇 1 single + 2+父块 1 multi)
        assert len(items) >= 2
        assert items[0]["query_id"] == "Q0001"
        assert items[1]["query_id"] == "Q0002"
        # Crawler 被排除
        assert all("Crawler" not in i["source_doc"] for i in items)
        # 新增 source_doc 为相对 data/ 路径
        assert any(i["source_doc"] == "Agent/a" for i in items)
        assert any(i["source_doc"] == "Agent/b" for i in items)
        # 难度混合(single + multi)
        diffs = {i["difficulty"] for i in items}
        assert "single_chunk" in diffs
