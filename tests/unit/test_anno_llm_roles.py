"""test_anno_llm_roles.py：F48 roles.py 单元测试(提议者 + 校验者两轮 + 删除核对 + 负样本注入)。

覆盖 anno_llm.md §7 的 propose / verify_two_pass / verify_delete 用例:
    提议者结构化输出解析 / 两轮对拍(一致/⊇/不一致→review) / 按 hint_location 确无→确认删除。
call_llm 一律 monkeypatch(mock API), 不触真实计费调用。
"""
from types import SimpleNamespace

import pytest

import benchmark.anno_llm.roles as roles_mod
from benchmark.anno_llm.prompt import ChunkIdReviewError
from benchmark.anno_llm.roles import (
    NEGATIVE_PROBE_PREFIX,
    negative_inject,
    propose,
    verify_delete,
    verify_two_pass,
)


def _chunk(chunk_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, content=content)


def _entry(query: str = "设计模式分几类？") -> dict:
    return {"query_id": "crawler-0000", "query": query, "source_doc": "Crawler/0000",
            "category": "crawler", "expected_files": ["Crawler/0000"], "expected_pages": []}


def _sequenced(*responses):
    """构造按调用次序依次返回的 mock call_llm(校验者两轮按 round1→round2 顺序)。"""
    state = {"index": 0}

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        response = responses[state["index"]]
        state["index"] += 1
        return response

    return fake_call_llm


FULL_DOC = [
    _chunk("Crawler/0000:p0", "设计模式分为三大类：创建型、结构型、行为型。"),
    _chunk("Crawler/0000:p1", "单例模式属于创建型模式。"),
]

PROPOSAL = {
    "query_id": "crawler-0000",
    "query": "设计模式分几类？",
    "expected_parent_ids": ["Crawler/0000:p0"],
    "relevance": {"Crawler/0000:p0": 3},
    "evidence_quote": {"Crawler/0000:p0": "设计模式分为三大类"},
    "mark_for_delete": False,
    "hint_location": None,
}


# ---- propose ----

def test_propose_uses_proposer_call(monkeypatch):
    """propose 用提议者模板组织 prompt, 返回解析后的结构化标注。"""
    captured: dict = {}

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        captured["prompt"] = prompt
        return dict(PROPOSAL)

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    result = propose(_entry(), FULL_DOC)
    assert result["expected_parent_ids"] == ["Crawler/0000:p0"]
    assert result["evidence_quote"]["Crawler/0000:p0"] == "设计模式分为三大类"
    assert "设计模式分几类？" in captured["prompt"]
    assert "[CHUNK 1] Crawler/0000:p0" in captured["prompt"]
    # 提议者不喂 reference_facts(gold)
    assert "reference_facts" not in captured["prompt"] or "幻觉" not in captured["prompt"]


def test_propose_delete_branch_returns_hint_location(monkeypatch):
    """无解分支: 提议者输出空数组 + mark_for_delete + hint_location。"""
    delete_proposal = {
        "query_id": "crawler-0000", "query": "无解query",
        "expected_parent_ids": [], "relevance": {}, "evidence_quote": {},
        "mark_for_delete": True, "hint_location": "Crawler/0000:p1",
    }
    monkeypatch.setattr(roles_mod, "call_llm", lambda prompt, **kw: dict(delete_proposal))
    result = propose(_entry("无解query"), FULL_DOC)
    assert result["mark_for_delete"] is True
    assert result["hint_location"] == "Crawler/0000:p1"
    assert result["expected_parent_ids"] == []


def test_propose_passes_timeout_to_call_llm(monkeypatch):
    """单次超时透传: propose 把 timeout 传给 call_llm(供 OpenAI create 使用)。"""
    captured: dict = {}

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        captured["timeout"] = timeout
        return dict(PROPOSAL)

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    propose(_entry(), FULL_DOC, timeout=42.0)
    assert captured["timeout"] == 42.0


# ---- 提议者 chunk_id 格式校验/纠正 ----

def _proposal_with_ids(expected_ids, relevance_ids=None, quote_ids=None):
    """构造提议者输出(可指定 expected_parent_ids / relevance / evidence_quote 的 id)。"""
    relevance_ids = relevance_ids or expected_ids
    quote_ids = quote_ids or expected_ids
    return {
        "query_id": "crawler-0000", "query": "设计模式分几类？",
        "expected_parent_ids": list(expected_ids),
        "relevance": {chunk_id: 3 for chunk_id in relevance_ids},
        "evidence_quote": {chunk_id: f"quote-{chunk_id}" for chunk_id in quote_ids},
        "mark_for_delete": False, "hint_location": None,
        "confidence": 0.9, "question_type": "factual",
    }


def test_propose_valid_chunk_ids_no_correction(monkeypatch):
    """合法 chunk_id 直接通过, 不触发纠正(仅调一次 call_llm)。"""
    calls = {"n": 0}
    valid = _proposal_with_ids(["Crawler/0000:p0"])

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        calls["n"] += 1
        return dict(valid)

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    result = propose(_entry(), FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})
    assert calls["n"] == 1
    assert result["expected_parent_ids"] == ["Crawler/0000:p0"]


def test_propose_corrects_invalid_chunk_ids(monkeypatch):
    """doc:xxx:p0 非法 → 触发纠正提示词重试 → 输出合法 id 后通过。"""
    invalid = _proposal_with_ids(["doc:xxx:p0"])
    valid = _proposal_with_ids(["Crawler/0000:p0"])
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(invalid, valid))
    result = propose(_entry(), FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})
    assert result["expected_parent_ids"] == ["Crawler/0000:p0"]
    assert result["relevance"].keys() == {"Crawler/0000:p0"}
    assert result["evidence_quote"].keys() == {"Crawler/0000:p0"}


def test_propose_still_invalid_after_retry_raises(monkeypatch):
    """CHUNK n 编号非法, 纠正重试一次仍非法 → 抛 ChunkIdReviewError(进 review)。"""
    invalid1 = _proposal_with_ids(["doc:xxx:p0"])
    invalid2 = _proposal_with_ids(["CHUNK 1"])
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(invalid1, invalid2))
    with pytest.raises(ChunkIdReviewError):
        propose(_entry(), FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})


# ---- verify_two_pass ----

def test_verify_two_pass_consistent_when_full_superset(monkeypatch):
    """两轮一致或全读 ⊇ 第一轮 → 通过(review_needed=False)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True},
                            "Crawler/0000:p1": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["consistent"] is True
    assert result["agrees_with_round1"] is True
    assert result["review_needed"] is False


def test_verify_two_pass_review_when_disagree(monkeypatch):
    """两轮不一致(ROUND1 有块在 ROUND2 不可答/多余) → review_needed=True。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": False}},
              "agrees_with_round1": False, "full_superset": False}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["consistent"] is False
    assert result["review_needed"] is True


def test_verify_two_pass_rejects_negative_block(monkeypatch):
    """块级负样本被校验者拒绝(grounded=false) → negative_rejected=True。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True},
                            "Other/0000:p9": {"grounded": False}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    negative = [_chunk("Other/0000:p9", "无关文档内容")]
    result = verify_two_pass(PROPOSAL, FULL_DOC, negative_blocks=negative)
    assert result["negative_rejected"] is True
    assert result["review_needed"] is False


def test_verify_two_pass_negative_not_rejected_triggers_review(monkeypatch):
    """负样本块被误判 grounded=true(全标倾向) → review_needed=True。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Other/0000:p9": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    negative = [_chunk("Other/0000:p9", "无关文档内容")]
    result = verify_two_pass(PROPOSAL, FULL_DOC, negative_blocks=negative)
    assert result["negative_rejected"] is False
    assert result["review_needed"] is True


def test_verify_two_pass_programmatic_superset_overrides_model_self_report(monkeypatch):
    """程序化对拍: ROUND2.grounded ⊇ ROUND1.grounded 时, 即便模型自报 full_superset=false 也判一致。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True},
                            "Crawler/0000:p1": {"grounded": True}},
              "agrees_with_round1": False, "full_superset": False}  # 模型自报矛盾
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["consistent"] is True
    assert result["review_needed"] is False


def test_verify_two_pass_programmatic_round1_extra_triggers_review(monkeypatch):
    """程序化对拍: ROUND1 有块在 ROUND2 不可答(多余) → 即便模型自报 full_superset=true 也判不一致。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True},
                            "Crawler/0000:p1": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}  # 模型自报矛盾
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["consistent"] is False
    assert result["review_needed"] is True


def test_verify_two_pass_string_booleans_parsed(monkeypatch):
    """布尔兼容字符串 true/false: per_block.grounded 与 agrees/full_superset 为字符串也能正确判定。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": "true"},
                            "Other/0000:p9": {"grounded": "false"}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": "true"}},
              "agrees_with_round1": "true", "full_superset": "true"}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    negative = [_chunk("Other/0000:p9", "无关文档内容")]
    result = verify_two_pass(PROPOSAL, FULL_DOC, negative_blocks=negative)
    assert result["consistent"] is True
    assert result["negative_rejected"] is True
    assert result["agrees_with_round1"] is True
    assert result["full_superset"] is True
    assert result["review_needed"] is False


def test_verify_two_pass_sufficiency_sufficient_passes(monkeypatch):
    """ROUND2 找到 >4 块, 第三者判 sufficient → 通过, 最终用 ≤4 候选集(ROUND1 提议集)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {f"Crawler/0000:p{i}": {"grounded": True} for i in range(5)},
              "agrees_with_round1": True, "full_superset": True}
    judge = {"sufficiency": "sufficient", "reason": "这些块足以回答"}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2, judge))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["sufficiency_grade"] == "sufficient"
    assert result["sufficiency_review"] is False
    assert result["sufficiency_candidate_ids"] == ["Crawler/0000:p0"]
    assert result["review_needed"] is False


def test_verify_two_pass_sufficiency_partial_review(monkeypatch):
    """ROUND2 找到 >4 块, 第三者判 partial → 进 review(sufficiency_review=True)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {f"Crawler/0000:p{i}": {"grounded": True} for i in range(5)},
              "agrees_with_round1": True, "full_superset": True}
    judge = {"sufficiency": "partial", "reason": "只能部分回答"}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2, judge))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["sufficiency_grade"] == "partial"
    assert result["sufficiency_review"] is True
    assert result["review_needed"] is True


def test_verify_two_pass_sufficiency_insufficient_review(monkeypatch):
    """ROUND2 找到 >4 块, 第三者判 insufficient → 进 review(sufficiency_review=True)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {f"Crawler/0000:p{i}": {"grounded": True} for i in range(5)},
              "agrees_with_round1": True, "full_superset": True}
    judge = {"sufficiency": "insufficient", "reason": "不足以回答"}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2, judge))
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert result["sufficiency_grade"] == "insufficient"
    assert result["sufficiency_review"] is True
    assert result["review_needed"] is True


def test_verify_two_pass_no_sufficiency_when_within_max(monkeypatch):
    """ROUND2 找到 ≤4 块 → 不触发第三者充分性判定(不额外调 LLM)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True},
                            "Crawler/0000:p1": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    calls = {"n": 0}

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        calls["n"] += 1
        return [round1, round2][calls["n"] - 1]

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    result = verify_two_pass(PROPOSAL, FULL_DOC)
    assert calls["n"] == 2  # 只调 round1+round2, 无第三者第三次
    assert result["sufficiency_grade"] is None
    assert result["sufficiency_review"] is False
    assert result["review_needed"] is False


def test_verify_two_pass_valid_chunk_ids_no_review(monkeypatch):
    """两轮 per_block key 均为真实 id → chunk_id_review=False(不额外调纠正)。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    calls = {"n": 0}

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        calls["n"] += 1
        return [round1, round2][calls["n"] - 1]

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    result = verify_two_pass(PROPOSAL, FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})
    assert calls["n"] == 2  # 均合法, 无纠正调用
    assert result["chunk_id_review"] is False
    assert result["review_needed"] is False


def test_verify_two_pass_corrects_invalid_round1_key(monkeypatch):
    """ROUND1 per_block key 非法 → 纠正重试合法 → chunk_id_review=False。"""
    round1_bad = {"round": "ROUND1_ONLY_PROPOSED",
                  "per_block": {"doc:xxx:p0": {"grounded": True}},
                  "sufficient": True}
    round1_ok = {"round": "ROUND1_ONLY_PROPOSED",
                 "per_block": {"Crawler/0000:p0": {"grounded": True}},
                 "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1_bad, round1_ok, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})
    assert result["chunk_id_review"] is False
    assert result["review_needed"] is False


def test_verify_two_pass_still_invalid_key_review(monkeypatch):
    """ROUND1 per_block key 纠正后仍非法 → chunk_id_review=True, 进 review。"""
    round1_bad = {"round": "ROUND1_ONLY_PROPOSED",
                  "per_block": {"doc:xxx:p0": {"grounded": True}},
                  "sufficient": True}
    round1_still_bad = {"round": "ROUND1_ONLY_PROPOSED",
                        "per_block": {"CHUNK 1": {"grounded": True}},
                        "sufficient": True}
    round2 = {"round": "ROUND2_FULL",
              "per_block": {"Crawler/0000:p0": {"grounded": True}},
              "agrees_with_round1": True, "full_superset": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1_bad, round1_still_bad, round2))
    result = verify_two_pass(PROPOSAL, FULL_DOC, known_ids={"Crawler/0000:p0", "Crawler/0000:p1"})
    assert result["chunk_id_review"] is True
    assert result["review_needed"] is True


# ---- verify_delete ----

def test_verify_delete_confirmed_when_no_solution(monkeypatch):
    """按 hint_location 读原文确无 → delete_confirmed=True。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED"}
    round2 = {"round": "ROUND2_FULL", "delete_confirmed": True}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_delete(PROPOSAL, hint="Crawler/0000:p1", full_doc=FULL_DOC)
    assert result["delete_confirmed"] is True


def test_verify_delete_not_confirmed(monkeypatch):
    """文档中实际存在可答块(确有解) → delete_confirmed=False。"""
    round1 = {"round": "ROUND1_ONLY_PROPOSED"}
    round2 = {"round": "ROUND2_FULL", "delete_confirmed": False}
    monkeypatch.setattr(roles_mod, "call_llm", _sequenced(round1, round2))
    result = verify_delete(PROPOSAL, hint="Crawler/0000:p1", full_doc=FULL_DOC)
    assert result["delete_confirmed"] is False


def test_verify_delete_fuzzy_hint_matches_chunk(monkeypatch):
    """区域型/近似 hint 未精确命中时按子串匹配落到候选块(P2-7), ROUND1 不空转。"""
    prompts: list[str] = []
    responses = iter([{"round": "ROUND1_ONLY_PROPOSED"}, {"round": "ROUND2_FULL", "delete_confirmed": True}])

    def fake_call_llm(prompt, client=None, model=None, timeout=None):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(roles_mod, "call_llm", fake_call_llm)
    result = verify_delete(PROPOSAL, hint="Crawler/0000", full_doc=FULL_DOC)  # 前缀型 hint
    assert result["delete_confirmed"] is True
    # ROUND1 输入经模糊匹配含 Crawler/0000:p0
    assert "Crawler/0000:p0" in prompts[0]


# ---- negative_inject ----

def test_negative_inject_ratio_and_marker():
    """按 ratio 注入无解探针, 带哨兵前缀与标记, seed 确定性。"""
    items = [{"query_id": f"crawler-{index:04d}", "query": f"q{index}", "source_doc": "Crawler/0"}
             for index in range(20)]
    probes_a, indexes_a = negative_inject(items, ratio=0.1, seed=42)
    probes_b, indexes_b = negative_inject(items, ratio=0.1, seed=42)
    assert indexes_a == indexes_b  # seed 确定性
    assert len(indexes_a) == 2  # 20 * 0.1
    for probe in probes_a:
        assert probe["_negative_probe"] is True
        assert probe["query"].startswith(NEGATIVE_PROBE_PREFIX)
        assert probe["query_id"].endswith("-neg")


def test_negative_inject_small_list():
    """条目太少时注入 1 条兜底。"""
    items = [{"query_id": "x", "query": "q"}]
    probes, indexes = negative_inject(items, ratio=0.1, seed=1)
    assert len(probes) == 1
    assert indexes == [0]
