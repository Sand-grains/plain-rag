"""test_anno_llm_grounding.py：F47 grounding.py 单元测试(引用存在性 + 语义弱校验 + 一致性)。

覆盖 anno_llm.md §7 的 verify_grounding / verify_semantic 用例:
    引用在/不在 chunk 内两种情况 + 归一化边界(大小写/全角空格/标点) / 语义低线判定。
"""
from types import SimpleNamespace

from benchmark.anno_llm.grounding import (
    MAX_EMBED_CHARS,
    enforce_consistency,
    normalize,
    quote_present,
    term_overlap,
    verify_grounding,
    verify_semantic,
)


def _chunk(chunk_id: str = "Crawler/0000:p0", content: str = "") -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, content=content)


# ---- normalize: 归一化边界 ----

def test_normalize_fold_fullwidth_lowercase_whitespace():
    """全角标点/全角字母/大小写/空白折叠为同一归一化形态。"""
    raw = "Hello，World！  RAG系统"
    assert normalize(raw) == normalize("hello, world! rag系统")


def test_normalize_fullwidth_space_and_punct():
    """全角空格与标点参与归一化, 子串匹配不依赖原始形态。"""
    assert normalize("a　b") == "a b"
    assert normalize("。").strip() in normalize("a。b")


# ---- verify_grounding: 引用存在性 ----

def test_grounding_true_when_quote_substring():
    """quote 是 chunk 内容子串(归一化后)→ 通过。"""
    annotation = {"evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"}}
    chunk = _chunk(content="本文介绍设计模式，设计模式分为三大类：创建型、结构型、行为型。")
    assert verify_grounding(annotation, chunk) is True


def test_grounding_true_with_fullwidth_diff():
    """归一化子串匹配允许全角/标点差异。"""
    annotation = {"evidence_quote": {"Crawler/0000:p0": "Design Patterns：GoF"}}
    chunk = _chunk(content="Design Patterns: GoF 是经典著作。")
    assert verify_grounding(annotation, chunk) is True


def test_grounding_false_when_quote_absent():
    """quote 不在 chunk 内容中 → 失败(引用不在文档)。"""
    annotation = {"evidence_quote": {"Crawler/0000:p0": "完全无关的句子"}}
    chunk = _chunk(content="本文只讲设计模式分类。")
    assert verify_grounding(annotation, chunk) is False


def test_grounding_false_when_no_quote_for_chunk():
    """该 chunk 没有 evidence_quote → 失败。"""
    annotation = {"evidence_quote": {}}
    assert verify_grounding(annotation, _chunk(content="内容")) is False


def test_quote_present_empty_quote_false():
    """空引用判为未 grounded。"""
    assert quote_present("", "内容") is False


# ---- term_overlap ----

def test_term_overlap_high_when_identical():
    """query 与 quote 术语高度重合。"""
    ratio = term_overlap("设计模式分为三大类", "设计模式分为三大类：创建型结构型行为型")
    assert ratio >= 0.8


def test_term_overlap_low_when_unrelated():
    """不相关文本重叠率低于 0.3 低线。"""
    ratio = term_overlap("AgentScope 的架构层次", "食堂饭菜口味非常不错")
    assert ratio < 0.3


# ---- verify_semantic: 语义弱校验 ----

def _fake_embed(similarity: float):
    """构造伪造 embed_fn: 返回两条归一化向量, 点积恒等于 similarity。"""
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [similarity, (1 - similarity ** 2) ** 0.5]]
    return embed_fn


def test_semantic_single_chunk_uses_chunk_content():
    """single_chunk(1 块): 度量对象 = chunk.content, effective_source='chunk'。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"}}
    chunk = _chunk(content="设计模式分为三大类 创建型 结构型 行为型")
    result = verify_semantic(annotation, "设计模式分为三大类", chunk, embed_fn=_fake_embed(0.9))
    assert result["effective_source"] == "chunk"
    assert result["cos_sim"] == 0.9
    assert result["passed"] is True


def test_semantic_multi_chunk_uses_concat_quotes():
    """multi_chunk(≥2 块): 度量对象 = concat(evidence_quote) 按 expected_parent_ids 次序。"""
    annotation = {
        "expected_parent_ids": ["Crawler/0000:p0", "Crawler/0000:p1"],
        "evidence_quote": {"Crawler/0000:p0": "第一块答案", "Crawler/0000:p1": "第二块答案"},
    }
    chunk = _chunk(content="无关内容")  # multi_chunk 不用 chunk.content
    result = verify_semantic(annotation, "第一块答案 第二块答案", chunk, embed_fn=_fake_embed(0.9))
    assert result["effective_source"] == "concat_quotes"
    assert result["cos_sim"] == 0.9
    assert result["passed"] is True


def test_semantic_passed_when_cos_high():
    """余弦>0.85 且词面重叠≥0.1 → passed。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"}}
    chunk = _chunk(content="设计模式分为三大类")
    result = verify_semantic(annotation, "设计模式分为三大类", chunk, embed_fn=_fake_embed(0.9))
    assert result["passed"] is True
    assert result["cos_sim"] == 0.9


def test_semantic_fails_when_cos_low():
    """余弦 ≤0.85 → 不通过(假绿拦截)。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"}}
    chunk = _chunk(content="设计模式分为三大类")
    result = verify_semantic(annotation, "设计模式分为三大类", chunk, embed_fn=_fake_embed(0.5))
    assert result["passed"] is False


def test_semantic_fails_when_no_lexical_overlap():
    """完全无词面重叠(term_overlap < 0.1) → 不通过(极低 floor 拦完全无重叠)。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "AgentScope 架构"}}
    chunk = _chunk(content="AgentScope 架构")
    result = verify_semantic(annotation, "食堂饭菜口味", chunk, embed_fn=_fake_embed(0.9))
    assert result["passed"] is False
    assert result["term_overlap"] < 0.1


def test_semantic_term_overlap_floor_configurable():
    """术语重叠极低 floor 可配置: 传 term_overlap_low=0.0 即关(仅余弦主判)。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "AgentScope 架构"}}
    chunk = _chunk(content="AgentScope 架构")
    # 默认 floor 0.1: 完全无重叠 → 不通过
    default = verify_semantic(annotation, "食堂饭菜口味", chunk, embed_fn=_fake_embed(0.9))
    assert default["passed"] is False
    # floor 0.0(关): 重叠 0 >= 0 且余弦 0.9 > 0.85 → 通过
    calibrated = verify_semantic(annotation, "食堂饭菜口味", chunk,
                                 embed_fn=_fake_embed(0.9), term_overlap_low=0.0)
    assert calibrated["passed"] is True


def test_semantic_cos_threshold_configurable():
    """余弦阈值可配置: 默认 0.85 判不过的 0.7, 传 cos_threshold=0.6 后通过(校准入口)。"""
    annotation = {"expected_parent_ids": ["Crawler/0000:p0"],
                  "evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"}}
    chunk = _chunk(content="设计模式分为三大类")
    # 默认阈值 0.85: 余弦 0.7 不通过
    default = verify_semantic(annotation, "设计模式分为三大类", chunk, embed_fn=_fake_embed(0.7))
    assert default["passed"] is False
    # 校准阈值 0.6: 余弦 0.7 通过
    calibrated = verify_semantic(annotation, "设计模式分为三大类", chunk,
                                 embed_fn=_fake_embed(0.7), cos_threshold=0.6)
    assert calibrated["passed"] is True


def test_semantic_truncates_concat_quotes():
    """multi_chunk 拼接文本超 MAX_EMBED_CHARS 时按字符保守截断(不超 BGE-M3 上限)。"""
    long_quote = "答案" * (MAX_EMBED_CHARS + 100)
    annotation = {
        "expected_parent_ids": ["Crawler/0000:p0", "Crawler/0000:p1"],
        "evidence_quote": {"Crawler/0000:p0": long_quote, "Crawler/0000:p1": "第二块"},
    }
    chunk = _chunk(content="无关")
    result = verify_semantic(annotation, "答案", chunk, embed_fn=_fake_embed(0.9))
    assert result["effective_source"] == "concat_quotes"
    assert result["passed"] is True


# ---- enforce_consistency ----

def test_consistency_derives_difficulty_from_block_count():
    """difficulty 由选块数确定性派生(>=2 → multi_chunk)。"""
    annotation, review = enforce_consistency(
        {"expected_parent_ids": ["a:p0", "a:p1"], "question_type": "factual", "difficulty": "single_chunk"})
    assert annotation["difficulty"] == "multi_chunk"
    assert review is False


def test_consistency_single_block():
    """单块 → single_chunk, 覆盖 LLM 标注值。"""
    annotation, _ = enforce_consistency(
        {"expected_parent_ids": ["a:p0"], "question_type": "factual", "difficulty": "multi_chunk"})
    assert annotation["difficulty"] == "single_chunk"


def test_consistency_multi_hop_requires_multi_chunk():
    """multi_hop 却只有单块 → 矛盾, review_needed=True。"""
    _, review = enforce_consistency(
        {"expected_parent_ids": ["a:p0"], "question_type": "multi_hop"})
    assert review is True


def test_consistency_multi_hop_with_multi_chunk_ok():
    """multi_hop + 多块 → 一致, 不需 review。"""
    _, review = enforce_consistency(
        {"expected_parent_ids": ["a:p0", "a:p1"], "question_type": "multi_hop"})
    assert review is False


def test_consistency_caps_to_max_chunks():
    """4-chunk 上限: expected_parent_ids >4 时 cap 到 ≤4, 同步裁剪 relevance/evidence_quote,
    并标记进 review(不静默丢块)。"""
    ids = [f"a:p{index}" for index in range(6)]
    annotation, review = enforce_consistency({
        "expected_parent_ids": ids,
        "relevance": {chunk_id: 3 for chunk_id in ids},
        "evidence_quote": {chunk_id: f"quote{index}" for index, chunk_id in enumerate(ids)},
        "question_type": "factual", "difficulty": "multi_chunk"})
    assert len(annotation["expected_parent_ids"]) == 4
    assert annotation["expected_parent_ids"] == ids[:4]
    assert set(annotation["relevance"].keys()) == set(ids[:4])
    assert set(annotation["evidence_quote"].keys()) == set(ids[:4])
    assert annotation.get("_capped_to_max_chunks") is True
    assert review is True  # 超上限 → 进 review, 不静默接受


def test_consistency_no_cap_when_within_max():
    """expected_parent_ids ≤4 时不触发 cap, 不标记 review。"""
    annotation, review = enforce_consistency({
        "expected_parent_ids": ["a:p0", "a:p1"],
        "relevance": {"a:p0": 3, "a:p1": 3},
        "evidence_quote": {"a:p0": "q0", "a:p1": "q1"},
        "question_type": "factual", "difficulty": "multi_chunk"})
    assert annotation.get("_capped_to_max_chunks") is None
    assert review is False
