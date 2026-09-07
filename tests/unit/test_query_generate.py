"""unit: scripts/query_generate 011-2——切分/typed 问题解析/single+multi 生成(mock LLM)。"""
import json

import pytest

from scripts import query_generate as qg


@pytest.fixture
def auto_install_fakes():
    """覆盖 conftest 的 autouse fake 安装: 本模块只测生成逻辑, 无需重依赖。"""
    yield


class TestSplitArticle:
    def test_returns_parents_with_chunk_ids(self):
        text = "# 标题\n\n正文内容。" * 30
        parents = qg.split_article(text, "Crawler/abc")
        assert parents
        assert all(p.chunk_id.startswith("Crawler/abc:p") for p in parents)

    def test_short_text_returns_parents(self):
        parents = qg.split_article("短文本", "Crawler/x")
        assert parents


class TestParseQuestions:
    def test_normalizes_questions(self):
        payload = {"questions": [
            {"query": "q1", "reference_facts": "rf1", "question_type": "comparison"},
            {"question": "q2", "expected_answer": "rf2", "question_type": "multi_hop"},
            {"query": "q3", "question_type": "badtype"},  # 非法类型回退 factual
            {"question_type": "factual"},  # 空 query 跳过
        ]}
        rows = qg.parse_questions(payload)
        assert len(rows) == 3
        assert rows[0]["question_type"] == "comparison"
        assert rows[1]["question_type"] == "multi_hop"
        assert rows[2]["question_type"] == "factual"
        assert all(r["query"] for r in rows)

    def test_empty_payload(self):
        assert qg.parse_questions({}) == []


class TestBuildItem:
    def test_fields_aligned(self):
        item = qg.build_item("crawler-0001", "q", "rf", "Crawler/a",
                             ["Crawler/a:p0"], "single_chunk", "comparison")
        assert item["query_id"] == "crawler-0001"
        assert item["expected_parent_ids"] == ["Crawler/a:p0"]
        assert item["relevance"] == {"Crawler/a:p0": 3}
        assert item["difficulty"] == "single_chunk"
        assert item["question_type"] == "comparison"
        assert item["category"] == "crawler"


class TestGenerateSingle:
    def test_generates_typed_queries_per_parent(self):
        parents = qg.split_article("# 标题\n\n正文内容。" * 30, "Crawler/a")
        fake_llm = lambda p, c: {"questions": [
            {"query": "q-a", "reference_facts": "rf", "question_type": "factual"},
            {"query": "q-b", "reference_facts": "rf", "question_type": "comparison"},
        ]}
        items = qg.generate_single(parents, "Crawler/a", 0, fake_llm, per_chunk=2)
        assert len(items) == len(parents) * 2
        assert all(i["difficulty"] == "single_chunk" for i in items)
        assert all(len(i["expected_parent_ids"]) == 1 for i in items)
        assert {i["question_type"] for i in items} == {"factual", "comparison"}


class TestGenerateMulti:
    def test_pairs_span_two_parents_and_multi_hop(self):
        # 长文本确保产出 2+ 父块
        text = ("# 标题\n\n" + "正文内容。" * 200 + "\n\n## 小节\n\n" + "更多内容。" * 200)
        parents = qg.split_article(text, "Crawler/a")
        assert len(parents) >= 2
        fake_llm = lambda p, c: {"questions": [
            {"query": "mh", "reference_facts": "rf", "question_type": "multi_hop"}]}
        items = qg.generate_multi(parents, "Crawler/a", 0, fake_llm, pairs=2)
        assert items
        assert all(i["difficulty"] == "multi_chunk" for i in items)
        assert all(i["question_type"] == "multi_hop" for i in items)
        assert all(len(i["expected_parent_ids"]) == 2 for i in items)


class TestCallLlm:
    def test_parses_json(self, monkeypatch):
        class _Choice:
            message = type("M", (), {"content":
                '{"questions": [{"query": "q", "reference_facts": "rf", "question_type": "factual"}]}'})()
        class _Resp:
            choices = [_Choice()]
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        return _Resp()
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        payload = qg.call_llm("prompt", "content")
        assert payload["questions"][0]["query"] == "q"

    def test_repairs_malformed_json(self, monkeypatch):
        # LLM 产出未转义引号的坏 JSON, json_repair 兜底修复
        bad = '{"questions": [{"query": "q", "reference_facts": "含"-"引号", "question_type": "factual"}]}'
        class _Choice:
            message = type("M", (), {"content": bad})()
        class _Resp:
            choices = [_Choice()]
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        return _Resp()
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        payload = qg.call_llm("prompt", "content")
        assert payload["questions"][0]["query"] == "q"

    def test_unparseable_raises(self, monkeypatch):
        class _Choice:
            message = type("M", (), {"content": "not json"})()
        class _Resp:
            choices = [_Choice()]
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        return _Resp()
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        with pytest.raises(RuntimeError, match="不可解析"):
            qg.call_llm("prompt", "content")

    def test_api_failure_raises(self, monkeypatch):
        class _FakeOpenAI:
            def __init__(self, *a, **k): pass
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **k):
                        raise RuntimeError("boom")
        monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
        with pytest.raises(RuntimeError, match="LLM 调用失败"):
            qg.call_llm("prompt", "content")


class TestRunMultiOnly:
    def test_appends_multi_hop_to_existing(self, tmp_path, monkeypatch):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # 长文本确保 2+ 父块, generate_multi 才能产出
        (corpus / "a.md").write_text(
            "# 标题\n\n" + "正文内容。" * 200 + "\n\n## 小节\n\n" + "更多内容。" * 200,
            encoding="utf-8")
        out = tmp_path / "private_crawler.json"
        existing = [{"query_id": "crawler-0000", "query": "q", "reference_facts": "rf",
                     "source_doc": "Crawler/a", "difficulty": "single_chunk",
                     "question_type": "factual", "expected_parent_ids": ["Crawler/a:p0"]}]
        out.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(qg, "call_llm", lambda p, c, max_retries=1: {"questions": [
            {"query": "mh", "reference_facts": "rf", "question_type": "multi_hop"}]})
        llm_fn = __import__("functools").partial(qg.call_llm, max_retries=1)
        rc = qg._run_multi_only(str(corpus), out, llm_fn, multi_pairs=2)
        assert rc == 0
        items = json.loads(out.read_text(encoding="utf-8"))
        assert len(items) == 2
        assert items[1]["question_type"] == "multi_hop"
        assert items[1]["difficulty"] == "multi_chunk"
        assert items[1]["query_id"] == "crawler-0001"


class TestMain:
    def test_generates_and_writes(self, tmp_path, monkeypatch):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("# 标题\n\n正文内容。" * 30, encoding="utf-8")
        out = tmp_path / "private_crawler.json"
        monkeypatch.setattr(qg, "call_llm", lambda p, c, max_retries=1: {"questions": [
            {"query": "q", "reference_facts": "rf", "question_type": "factual"}]})
        rc = qg.main(["--corpus", str(corpus), "--out", str(out), "--target", "5",
                      "--multi-pairs", "1", "--per-chunk", "1"])
        assert rc == 0
        items = json.loads(out.read_text(encoding="utf-8"))
        assert items
        assert all(i["query_id"].startswith("crawler-") for i in items)
        assert all("question_type" in i for i in items)

    def test_empty_corpus_returns_1(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        out = tmp_path / "private_crawler.json"
        assert qg.main(["--corpus", str(corpus), "--out", str(out)]) == 1
