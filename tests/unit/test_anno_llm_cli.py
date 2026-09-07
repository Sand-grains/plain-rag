"""test_anno_llm_cli.py：F51 cli.run_pipeline 管线级单元测试(钉死 reviewer P0-1/2/3 + P1-2)。

P0-1: 条目级 10% 无解探针真实 append 进工作集并走删除路径(不再只打印计数)。
P0-2: 块级负样本(2 个无关块)真实传给 verify_two_pass。
P0-3: grounding 率按"标记块引用可验证+语义通过比例"记录(含校验失败分支), 非恒 1.0/0.0。
P1-2: passed_adopted 保留原 query_id(不重排), 供 gate1 对拍。
call_llm 依赖经由 propose/verify_two_pass/verify_delete monkeypatch 隔离, 不触真实计费。
"""
from types import SimpleNamespace

import pytest

import benchmark.anno_llm.cli as cli_mod
from benchmark.anno_llm.prompt import ChunkIdReviewError, LLMParseError


def _chunk(chunk_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, content=content)


def _doc_index() -> dict:
    return {
        "Crawler/0": [
            _chunk("Crawler/0:p0", "设计模式分为三大类 创建型 结构型 行为型"),
            _chunk("Crawler/0:p1", "食堂饭菜 味道 不错"),
            _chunk("Crawler/0:p2", "足球比赛 比分 结果"),
        ],
        "Other/1": [_chunk("Other/1:p0", "无关文档内容")],
    }


def _entry(query_id: str = "crawler-0000", query: str = "设计模式分几类？",
           source_doc: str = "Crawler/0") -> dict:
    return {"query_id": query_id, "query": query, "source_doc": source_doc,
            "category": "crawler", "expected_files": ["Crawler/0"], "expected_pages": []}


def _normal_proposal(query_id: str = "crawler-0000") -> dict:
    return {"query_id": query_id, "query": "设计模式分几类？",
            "expected_parent_ids": ["Crawler/0:p0"],
            "relevance": {"Crawler/0:p0": 3},
            "evidence_quote": {"Crawler/0:p0": "设计模式分为三大类"},
            "mark_for_delete": False, "hint_location": None,
            "confidence": 0.9, "question_type": "factual"}


# ---- P0-1: 探针真实走删除路径 ----

def test_p0_1_negative_probe_appended_and_walks_delete_path(monkeypatch):
    """--negative-inject 探针被 append 进工作集, 走 propose→verify_delete→删除清单。"""
    items = [_entry("crawler-0000")]
    calls = {"propose": [], "verify_delete": []}

    def fake_propose(entry, chunks, known_ids=None, timeout=None):
        calls["propose"].append(entry["query_id"])
        if entry["query_id"].endswith("-neg"):
            return {"query_id": entry["query_id"], "query": entry["query"],
                    "expected_parent_ids": [], "mark_for_delete": True,
                    "hint_location": "Crawler/0:p0", "evidence_quote": {}, "relevance": {},
                    "confidence": 0.9, "question_type": "factual"}
        return _normal_proposal(entry["query_id"])

    def fake_verify_delete(proposal, hint, chunks, timeout=None):
        calls["verify_delete"].append(proposal["query_id"])
        return {"round1": {}, "round2": {"delete_confirmed": True}, "delete_confirmed": True}

    monkeypatch.setattr(cli_mod, "propose", fake_propose)
    monkeypatch.setattr(cli_mod, "verify_delete", fake_verify_delete)
    # 正常条目走 verify_two_pass(非删除), 这里仅验证探针计数/删除路径
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    result = cli_mod.run_pipeline(_doc_index(), items, inject_negative_probes=True, seed=1)
    assert result["probes_count"] == 1
    assert len(calls["propose"]) == 2  # 正常条目 + 探针
    probe_ids = [entry["query_id"] for entry in result["delete_list"] if entry["is_negative_probe"]]
    assert probe_ids, "探针应真实进入删除清单(P0-1 断点)"
    assert calls["verify_delete"], "探针应走 verify_delete 删除路径"


# ---- P0-2: 块级负样本注入校验者 ----

def test_p0_2_negative_blocks_passed_to_verify_two_pass(monkeypatch):
    """2 个无关块真实传给 verify_two_pass(校验者须拒绝), 来自本 source doc 且与 query 无关。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    captured: dict = {}

    def fake_verify_two_pass(proposal, chunks, negative_blocks=None, known_ids=None, timeout=None):
        captured["negative_blocks"] = negative_blocks or []
        return {"review_needed": False, "consistent": True, "negative_rejected": True,
                "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True}

    monkeypatch.setattr(cli_mod, "verify_two_pass", fake_verify_two_pass)
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    cli_mod.run_pipeline(_doc_index(), [_entry()], confidence_threshold=0.8)
    # 2 个无关块来自本 source doc Crawler/0(非 Other/), 且排除已提议的 p0
    assert len(captured["negative_blocks"]) == 2
    assert all(block.chunk_id.startswith("Crawler/0:") for block in captured["negative_blocks"])
    assert all(block.chunk_id != "Crawler/0:p0" for block in captured["negative_blocks"])


def test_sample_negative_blocks_from_same_doc(monkeypatch):
    """负样本从本 source doc 抽, 排除 proposed_ids, 且 term_overlap<0.3。"""
    from benchmark.anno_llm.grounding import TERM_OVERLAP_LOW, term_overlap

    doc_index = {
        "Crawler/0": [
            _chunk("Crawler/0:p0", "设计模式分为三大类 创建型 结构型 行为型"),
            _chunk("Crawler/0:p1", "食堂饭菜 味道 不错"),
            _chunk("Crawler/0:p2", "足球比赛 比分 结果"),
        ],
    }
    blocks = cli_mod._sample_negative_blocks(
        doc_index, "Crawler/0", "设计模式分几类？", ["Crawler/0:p0"], n=2, seed=1)
    assert len(blocks) == 2
    assert all(block.chunk_id.startswith("Crawler/0:") for block in blocks)
    assert all(block.chunk_id != "Crawler/0:p0" for block in blocks)
    # 与 query 无关(term_overlap < 0.3)
    assert all(term_overlap("设计模式分几类？", block.content) < TERM_OVERLAP_LOW for block in blocks)


def test_sample_negative_blocks_no_candidate_returns_empty():
    """本 source doc 无与 query 无关的候选块 → 返回 []。"""
    doc_index = {
        "Crawler/0": [_chunk("Crawler/0:p0", "设计模式分为三大类 创建型 结构型 行为型")],
    }
    blocks = cli_mod._sample_negative_blocks(
        doc_index, "Crawler/0", "设计模式分几类？", ["Crawler/0:p0"], n=2, seed=1)
    assert blocks == []


# ---- 诊断观测: 两轮不一致是 ROUND2 显式拒绝还是遗漏 ----

def test_build_review_diagnostics_rejected_verdict():
    """ROUND2 显式 grounded=false → round2_verdict='rejected'。"""
    verify = {
        "round1": {"per_block": {"Crawler/0:p0": {"grounded": True}, "Crawler/0:p1": {"grounded": True}}},
        "round2": {"per_block": {"Crawler/0:p0": {"grounded": True}, "Crawler/0:p1": {"grounded": False}}},
        "negative_rejected": True,
    }
    diag = cli_mod._build_review_diagnostics(
        "q1", "两轮不一致", verify, ["Crawler/0:p0", "Crawler/0:p1"], [])
    assert diag["round1_grounded"] == ["Crawler/0:p0", "Crawler/0:p1"]
    assert diag["round2_grounded"] == ["Crawler/0:p0"]
    # round1 - round2 = {p1}, 且 ROUND2 显式 grounded=false → rejected
    assert diag["round1_minus_round2"] == [{"chunk_id": "Crawler/0:p1", "round2_verdict": "rejected"}]
    assert diag["negative_rejected"] is True


def test_build_review_diagnostics_omitted_verdict():
    """ROUND2 per_block 未列出某提议块 → round2_verdict='omitted'。"""
    verify = {
        "round1": {"per_block": {"Crawler/0:p0": {"grounded": True}, "Crawler/0:p1": {"grounded": True}}},
        "round2": {"per_block": {"Crawler/0:p0": {"grounded": True}}},  # p1 未列出(遗漏)
        "negative_rejected": True,
    }
    diag = cli_mod._build_review_diagnostics(
        "q1", "两轮不一致", verify, ["Crawler/0:p0", "Crawler/0:p1"], [])
    assert diag["round1_grounded"] == ["Crawler/0:p0", "Crawler/0:p1"]
    assert diag["round2_grounded"] == ["Crawler/0:p0"]
    assert diag["round1_minus_round2"] == [{"chunk_id": "Crawler/0:p1", "round2_verdict": "omitted"}]


def test_run_pipeline_separates_obs_from_review(monkeypatch):
    """verify 判为需要 review 时, obs_list 收到诊断明细, review_list 只含 query_id/reason。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": True, "sufficiency_review": False, "consistent": False,
                            "negative_rejected": True,
                            "agrees_with_round1": False, "full_superset": False,
                            "round1": {"per_block": {"Crawler/0:p0": {"grounded": True}}},
                            "round2": {"per_block": {"Crawler/0:p0": {"grounded": False}}}})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    # review_list 基本字段(不混入诊断)
    assert result["review_list"] == [{"query_id": "crawler-0000", "reason": "两轮不一致"}]
    # obs_list 含完整诊断明细
    assert len(result["obs_list"]) == 1
    obs = result["obs_list"][0]
    assert obs["query_id"] == "crawler-0000"
    assert obs["round1_minus_round2"] == [{"chunk_id": "Crawler/0:p0", "round2_verdict": "rejected"}]
    assert "crawler-0000" not in result["passed_adopted"]


# ---- P0-3: grounding 率反映真实失败比例 ----

def test_p0_3_grounding_rate_reflects_failures(monkeypatch):
    """程序校验失败的条目也计入 grounding 率(非恒 1.0)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})

    semantic_by_id = {"crawler-0000": True, "crawler-0001": False}
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None, semantic_by_id=semantic_by_id: {
                            "passed": semantic_by_id[annotation["query_id"]],
                            "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    items = [_entry("crawler-0000", source_doc="Crawler/0"),
             _entry("crawler-0001", source_doc="Crawler/0")]
    result = cli_mod.run_pipeline(_doc_index(), items)
    # 两条都走到程序校验: 一条通过、一条失败 → grounding 率 0.5
    assert len(result["grounded_results"]) == 2
    assert sorted(entry["grounded"] for entry in result["grounded_results"]) == [False, True]
    assert result["review_list"], "校验失败条目应进 review"
    # 失败条目不应进 passed_adopted
    assert "crawler-0001" not in result["passed_adopted"]


# ---- P1-2: passed_adopted 保留原 query_id(不重排), 供 gate1 对拍 ----

def test_p1_2_passed_adopted_keeps_original_query_id(monkeypatch):
    """passed_adopted 以原 query_id 为 key(未重排), 保证 gate1 与 gold 按 query_id 对齐。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    items = [_entry("Q9001", source_doc="Crawler/0"), _entry("Q9002", source_doc="Crawler/0")]
    result = cli_mod.run_pipeline(_doc_index(), items)
    assert set(result["passed_adopted"].keys()) == {"Q9001", "Q9002"}
    assert result["passed_adopted"]["Q9001"]["query_id"] == "Q9001"


# ---- 删除分支不落入 passed/grounding ----

def test_delete_branch_not_in_passed_or_grounding(monkeypatch):
    """无解删除条目不进 passed_adopted 也不计入 grounding 率。"""
    def fake_propose(entry, chunks, known_ids=None, timeout=None):
        return {"query_id": entry["query_id"], "query": entry["query"], "expected_parent_ids": [],
                "mark_for_delete": True, "hint_location": "Crawler/0:p0",
                "evidence_quote": {}, "relevance": {}, "confidence": 0.9, "question_type": "factual"}

    monkeypatch.setattr(cli_mod, "propose", fake_propose)
    monkeypatch.setattr(cli_mod, "verify_delete",
                        lambda proposal, hint, chunks, timeout=None: {"delete_confirmed": True})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert result["delete_list"]
    assert "crawler-0000" not in result["passed_adopted"]
    assert result["grounded_results"] == []


def test_verify_parse_error_routes_to_review_not_crash(monkeypatch):
    """校验者/删除核对 LLM 解析失败 → 进 review + 继续, 不崩整批(实跑暴露的健壮性 bug)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: (_ for _ in ()).throw(
                            LLMParseError("返回内容不含 JSON 对象: ''")))
    # verify_two_pass 抛错 → 该条进 review, run_pipeline 不抛异常
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and "verify_error" in item["reason"]
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]


# ---- 4-chunk 上限: 第三者充分性判定 ----

def test_sufficiency_review_routes_to_review_with_reason(monkeypatch):
    """verify_two_pass 返回 sufficiency_review=True(partial/insufficient) → 进 review, reason=insufficient_blocks。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": True, "sufficiency_review": True, "consistent": True,
                            "negative_rejected": True, "round1": {}, "round2": {},
                            "agrees_with_round1": True, "full_superset": True,
                            "sufficiency_grade": "partial"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and item["reason"] == "insufficient_blocks"
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]


def test_sufficiency_sufficient_passes_with_candidate_set(monkeypatch):
    """第三者判 sufficient → 通过, 最终标注用 ≤4 候选集(ROUND1 提议集)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "sufficiency_review": False, "consistent": True,
                            "negative_rejected": True, "round1": {}, "round2": {},
                            "agrees_with_round1": True, "full_superset": True,
                            "sufficiency_grade": "sufficient",
                            "sufficiency_candidate_ids": ["Crawler/0:p0"]})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    adopted = result["passed_adopted"]["crawler-0000"]
    assert adopted["expected_parent_ids"] == ["Crawler/0:p0"]


def test_cap_to_max_chunks_routes_to_review(monkeypatch):
    """enforce_consistency 因 >4 块 cap 并标记 → 进 review, reason=over4_blocks(不静默丢块)。"""
    def over4_proposal(query_id="crawler-0000"):
        ids = [f"Crawler/0:p{index}" for index in range(5)]
        return {"query_id": query_id, "query": "设计模式分几类？",
                "expected_parent_ids": ids,
                "relevance": {chunk_id: 3 for chunk_id in ids},
                "evidence_quote": {chunk_id: f"quote{index}" for index, chunk_id in enumerate(ids)},
                "mark_for_delete": False, "hint_location": None,
                "confidence": 0.9, "question_type": "factual"}

    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: over4_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "sufficiency_review": False, "consistent": True,
                            "negative_rejected": True, "round1": {}, "round2": {},
                            "agrees_with_round1": True, "full_superset": True})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and item["reason"] == "over4_blocks"
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]


def test_propose_chunk_id_error_routes_to_review(monkeypatch):
    """提议者 chunk_id 经纠正仍非法(ChunkIdReviewError) → 进 review, reason=chunk_id_error。"""
    def fake_propose(entry, chunks, known_ids=None, timeout=None):
        raise ChunkIdReviewError("提议者输出 chunk_id 经纠正后仍不合法")

    monkeypatch.setattr(cli_mod, "propose", fake_propose)
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and item["reason"] == "chunk_id_error"
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]


def test_verify_chunk_id_review_routes_to_review(monkeypatch):
    """verify 返回 chunk_id_review=True → 进 review, reason=chunk_id_error。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": True, "chunk_id_review": True, "consistent": True,
                            "negative_rejected": True, "round1": {}, "round2": {},
                            "agrees_with_round1": True, "full_superset": True,
                            "sufficiency_review": False})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and item["reason"] == "chunk_id_error"
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]


# ---- 崩溃点 1: 程序校验(verify_grounding/verify_semantic)抛错降级进 review ----

def test_grounding_error_routes_to_review_not_crash(monkeypatch):
    """程序校验(verify_semantic 惰性加载 BGE-M3)抛错 → 进 review(grounding_error), 不崩整批。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: (_ for _ in ()).throw(
                            RuntimeError("BGE-M3 加载失败")))

    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert any(item["query_id"] == "crawler-0000" and "grounding_error" in item["reason"]
               for item in result["review_list"])
    assert "crawler-0000" not in result["passed_adopted"]
    # 程序校验抛错未真正评估 → 不计入 grounding 率
    assert result["grounded_results"] == []


# ---- 崩溃点 3: checkpoint 文件损坏按空集处理 ----

def test_load_checkpoint_corrupt_returns_empty(monkeypatch, capsys, tmp_path):
    """checkpoint 文件存在但内容非法 → 按空集处理 + warning, 不抛 JSONDecodeError。"""
    bad = tmp_path / "checkpoint.json"
    bad.write_text("{ 这不是合法 JSON", encoding="utf-8")
    result = cli_mod._load_checkpoint(str(bad))
    assert result == set()
    out = capsys.readouterr().out
    assert "checkpoint 文件损坏" in out


# ---- 崩溃点 4: --preflight 内部抛非 RuntimeError 被 main 兜住 ----

def test_preflight_non_runtime_error_caught(monkeypatch, capsys):
    """--preflight 内部抛非 RuntimeError(如 ValueError) → main 兜住转可读错误, 返回 1 不崩。"""
    import indexing.index_store as index_store_mod
    import benchmark.anno_tool as anno_tool_mod

    # main 内为局部 import, 需 patch 实际模块(而非 cli 模块属性)。
    monkeypatch.setattr(index_store_mod.IndexStore, "vector_restore",
                        classmethod(lambda cls, cache_dir: object()))
    monkeypatch.setattr(anno_tool_mod, "build_doc_index", lambda store: {})
    monkeypatch.setattr(anno_tool_mod, "load_benchmark", lambda path: [])
    monkeypatch.setattr(cli_mod, "_run_preflight",
                        lambda args, doc_index, gold_items: (_ for _ in ()).throw(ValueError("boom")))

    code = cli_mod.main(["--source", "benchmark/private_builtin.json",
                         "--output", "benchmark/private_builtin_anno_test.json", "--preflight"])
    assert code == 1
    out = capsys.readouterr().out
    assert "boom" in out


# ---- 诊断盲区修复: preflight 落盘 review/delete/diag ----

def test_preflight_writes_review_delete_diag(monkeypatch, capsys):
    """preflight 落盘 review/delete/diag(与主路径 main() 对齐), 能看到每条 review 原因。"""
    from types import SimpleNamespace
    import benchmark.anno_tool as anno_tool_mod

    args = SimpleNamespace(
        source="benchmark/private_builtin.json",
        limit=None, checkpoint=None, negative_inject=False,
        confidence_threshold=0.8, seed=None, timeout=None, total_timeout=None,
        no_negative_blocks=False, no_semantic_gate=False,
        semantic_mode="hard", semantic_threshold=None, workers=1,
        passed_file="p.json", log_file=None,
        review_file="r.json", delete_file="d.json", diag_file="diag.json",
    )
    monkeypatch.setattr(anno_tool_mod, "load_benchmark", lambda path: [])
    monkeypatch.setattr(cli_mod, "make_sanitized_copy", lambda src, dst: 0)
    monkeypatch.setattr(cli_mod, "run_pipeline", lambda *a, **k: {
        "passed_adopted": {},
        "review_list": [{"query_id": "Q1", "reason": "两轮不一致"}],
        "obs_list": [{"query_id": "Q1", "round1_minus_round2": []}],
        "semantic_diag": [{"query_id": "Q1", "cos_sim": 0.5}],
        "delete_list": [{"query_id": "Q2"}],
        "grounded_results": [], "processed": set(), "fuzzy_mappings": [],
        "attribution": [{"query_id": "Q1", "reject_reasons": ["cos"]}],
        "probes_count": 0, "timed_out": False, "elapsed": 0.0})
    monkeypatch.setattr(cli_mod, "gate1_hit_rate", lambda *a, **k: {"rate": 0.9, "hits": 9, "total": 10})
    monkeypatch.setattr(cli_mod, "checkpoint_status", lambda report: {"message": "ok"})
    writes: dict = {}
    monkeypatch.setattr(cli_mod, "_write_json",
                        lambda path, payload: writes.__setitem__(path, payload) or path)

    cli_mod._run_preflight(args, {}, [])
    assert writes["r.json"] == [{"query_id": "Q1", "reason": "两轮不一致"}]
    assert writes["d.json"] == [{"query_id": "Q2"}]
    assert writes["diag.json"] == {
        "obs": [{"query_id": "Q1", "round1_minus_round2": []}],
        "semantic_diag": [{"query_id": "Q1", "cos_sim": 0.5}],
        "attribution": [{"query_id": "Q1", "reject_reasons": ["cos"]}],
    }
    out = capsys.readouterr().out
    assert "review 落盘" in out
    assert "delete 落盘" in out
    assert "diag 落盘" in out


# ---- 超时控制: 单次透传 + 整批总超时 ----

def test_per_call_timeout_passed_to_propose(monkeypatch):
    """单次超时透传: run_pipeline 把 timeout 传给 propose(供 OpenAI create 使用)。"""
    captured: dict = {}

    def fake_propose(entry, chunks, known_ids=None, timeout=None):
        captured["timeout"] = timeout
        return _normal_proposal(entry["query_id"])

    monkeypatch.setattr(cli_mod, "propose", fake_propose)
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    cli_mod.run_pipeline(_doc_index(), [_entry()], timeout=30.0)
    assert captured["timeout"] == 30.0


def test_total_timeout_stops_early(monkeypatch):
    """整批总超时: 达到 --total-timeout 后在条目间中断, timed_out=True, 不处理全部条目。"""
    calls = {"propose": 0}

    def fake_propose(entry, chunks, known_ids=None, timeout=None):
        calls["propose"] += 1
        return _normal_proposal(entry["query_id"])

    monkeypatch.setattr(cli_mod, "propose", fake_propose)
    items = [_entry("crawler-0000"), _entry("crawler-0001")]
    result = cli_mod.run_pipeline(_doc_index(), items, total_timeout=0.0)
    assert result["timed_out"] is True
    # 协作式: 首条前 elapsed≈0 不触发, 处理 1 条后超时中断, 未处理全部 2 条
    assert calls["propose"] < len(items)
    assert result["elapsed"] >= 0.0


def test_progress_every_prints_incremental_progress(monkeypatch, capsys):
    """增量进度: progress_every=N 时每 N 条输出一行 [进度], 供后台长跑观测。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    items = [_entry(f"crawler-{index:04d}") for index in range(3)]
    cli_mod.run_pipeline(_doc_index(), items, progress_every=1)
    out = capsys.readouterr().out
    assert "[进度]" in out
    assert "passed" in out


# ---- 即时写盘: on_entry 每条处理完回调一次 ----

def test_on_entry_called_after_each_entry(monkeypatch):
    """即时写盘: on_entry 每条处理完回调一次, 快照反映当前累计状态(非仅末尾)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})

    snapshots: list[dict] = []
    items = [_entry("crawler-0000", source_doc="Crawler/0"),
             _entry("crawler-0001", source_doc="Crawler/0")]
    cli_mod.run_pipeline(_doc_index(), items, on_entry=snapshots.append)
    # 每条处理完回调一次(2 条 → 2 次), 而非仅末尾 1 次
    assert len(snapshots) == 2
    # 快照反映当前累计: 第 1 次后 1 条 passed, 第 2 次后 2 条 passed
    assert set(snapshots[0]["passed_adopted"].keys()) == {"crawler-0000"}
    assert set(snapshots[1]["passed_adopted"].keys()) == {"crawler-0000", "crawler-0001"}
    assert snapshots[1]["processed"] == {"crawler-0000", "crawler-0001"}


def test_main_passes_on_entry_to_run_pipeline(monkeypatch):
    """即时写盘接线: main 把 on_entry 回调传给 run_pipeline(全量非 dry-run)。"""
    import indexing.index_store as index_store_mod
    import benchmark.anno_tool as anno_tool_mod

    monkeypatch.setattr(index_store_mod.IndexStore, "vector_restore",
                        classmethod(lambda cls, cache_dir: object()))
    monkeypatch.setattr(anno_tool_mod, "build_doc_index", lambda store: {})
    monkeypatch.setattr(anno_tool_mod, "load_benchmark", lambda path: [
        {"query_id": "crawler-0000", "query": "q", "source_doc": "Crawler/0"}])
    monkeypatch.setattr(anno_tool_mod, "check_index_coverage", lambda doc_index, items: [])

    captured: dict = {}

    def fake_run_pipeline(doc_index, items, **kwargs):
        captured["on_entry"] = kwargs.get("on_entry")
        return {"passed_adopted": {}, "review_list": [], "obs_list": [], "semantic_diag": [],
                "delete_list": [], "grounded_results": [], "processed": set(), "fuzzy_mappings": [],
                "attribution": [], "probes_count": 0, "timed_out": False, "elapsed": 0.0}

    monkeypatch.setattr(cli_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli_mod, "write_benchmark", lambda items, path: None)
    monkeypatch.setattr(cli_mod, "_write_json", lambda path, payload: str(path))
    monkeypatch.setattr(cli_mod, "_save_checkpoint", lambda path, done: None)

    code = cli_mod.main(["--source", "benchmark/private_builtin.json",
                         "--output", "benchmark/private_builtin_anno_test.json"])
    assert code == 0
    assert callable(captured["on_entry"]), "main 应把即时写盘回调传给 run_pipeline"


# ---- 报告路径收敛: 默认输出按运行时间戳分包到 benchmark/anno_llm/reports/<时间戳>/ ----

def test_report_defaults_fill_into_timestamped_run_dir():
    """未显式指定的报告路径自动填为 reports/<YYYYMMDD-HHMMSS>/ 下的默认文件名。"""
    from types import SimpleNamespace

    args = SimpleNamespace(passed_file=None, review_file=None, delete_file=None, diag_file=None)
    cli_mod._resolve_report_paths(args)
    for attr, name in (("passed_file", "anno_llm_passed.json"),
                       ("review_file", "anno_llm_review.json"),
                       ("delete_file", "anno_llm_delete.json"),
                       ("diag_file", "anno_llm_obs.json")):
        path = getattr(args, attr)
        assert path.startswith("benchmark/anno_llm/reports/")
        assert path.endswith(name)
        # 时间戳子目录: reports/<8位日期>-<6位时间>/
        assert __import__("re").search(r"reports/\d{8}-\d{6}/", path), path


def test_report_defaults_keep_explicit_override():
    """显式传入的 --passed-file 等原样保留, 其余未传的仍填时间戳子目录。"""
    from types import SimpleNamespace

    args = SimpleNamespace(passed_file="custom/passed.json", review_file=None,
                           delete_file=None, diag_file=None)
    cli_mod._resolve_report_paths(args)
    assert args.passed_file == "custom/passed.json"
    assert args.review_file.startswith("benchmark/anno_llm/reports/")
    assert args.delete_file.startswith("benchmark/anno_llm/reports/")
    assert args.diag_file.startswith("benchmark/anno_llm/reports/")


# ---- 语义余弦诊断: 每条走到程序校验的条目记录 cos_sim/term_overlap/effective_source ----

def test_semantic_diag_records_cos_sim_for_all_entries(monkeypatch):
    """程序校验为每条记录 cos_sim/term_overlap/effective_source(整答案一次), 通过/失败都记。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    semantic_by_id = {"crawler-0000": (0.95, 0.9), "crawler-0001": (0.6, 0.1)}
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None, semantic_by_id=semantic_by_id: {
                            "passed": semantic_by_id[annotation["query_id"]][0] > 0.85,
                            "term_overlap": semantic_by_id[annotation["query_id"]][1],
                            "cos_sim": semantic_by_id[annotation["query_id"]][0],
                            "effective_source": "chunk"})

    items = [_entry("crawler-0000", source_doc="Crawler/0"),
             _entry("crawler-0001", source_doc="Crawler/0")]
    result = cli_mod.run_pipeline(_doc_index(), items)
    # 两条都走到程序校验, 各记录 1 条语义诊断(整答案一次, 通过/失败都记)
    assert len(result["semantic_diag"]) == 2
    for diag in result["semantic_diag"]:
        assert diag["chunk_ids"] == ["Crawler/0:p0"]
        assert "cos_sim" in diag and "term_overlap" in diag and "effective_source" in diag
    by_id = {d["query_id"]: d for d in result["semantic_diag"]}
    assert by_id["crawler-0000"]["cos_sim"] == 0.95
    assert by_id["crawler-0001"]["cos_sim"] == 0.6
    # _normal_proposal 单块 → effective_source=chunk
    assert all(d["effective_source"] == "chunk" for d in result["semantic_diag"])


# ---- F52: 负样本开关 + 语义门开关 ----

def test_no_negative_blocks_switch(monkeypatch):
    """inject_negative_blocks=False 时不注入块级负样本(纯对拍 gold)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    captured: dict = {}

    def fake_verify_two_pass(proposal, chunks, negative_blocks=None, known_ids=None, timeout=None):
        captured["negative_blocks"] = negative_blocks or []
        return {"review_needed": False, "consistent": True, "negative_rejected": True,
                "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True}

    monkeypatch.setattr(cli_mod, "verify_two_pass", fake_verify_two_pass)
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9,
                                                          "cos_sim": 0.9, "effective_source": "chunk"})
    cli_mod.run_pipeline(_doc_index(), [_entry()], inject_negative_blocks=False)
    assert captured["negative_blocks"] == []


def test_semantic_gate_off_skips_verify_semantic(monkeypatch):
    """semantic_gate=False 时不调用 verify_semantic(前置基线诊断测原始质量)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    calls = {"semantic": 0}
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None, calls=calls: calls.__setitem__("semantic", calls["semantic"] + 1) or {
                            "passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry()], semantic_gate=False)
    assert calls["semantic"] == 0
    assert result["semantic_diag"] == []
    assert "crawler-0000" in result["passed_adopted"]


# ---- 009: 归因分解(逐条 reject 子原因) ----

def test_attribution_records_reject_sub_reasons(monkeypatch):
    """009 归因分解: 引用/语义校验不达标时记录逐条 reject 子原因(cos/term_overlap/quote_absent)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    # 引用存在性失败(quote 不在块内) + 语义 cos 低 + term_overlap 低
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: False)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": False, "term_overlap": 0.0,
                                                          "cos_sim": 0.3, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert result["review_list"][0]["reason"] == "引用/语义校验不达标"
    reasons = result["review_list"][0]["reject_reasons"]
    # _normal_proposal 有 quote 但 verify_grounding False → quote_absent; cos/term_overlap 均低
    assert "quote_absent" in reasons
    assert "cos" in reasons
    assert "term_overlap" in reasons
    assert len(result["attribution"]) == 1
    attr = result["attribution"][0]
    assert attr["query_id"] == "crawler-0000"
    assert attr["reject_reasons"] == reasons
    assert attr["cos_sim"] == 0.3
    assert attr["semantic_gate_mode"] == "hard"


def test_attribution_quote_missing_when_no_quote(monkeypatch):
    """009 归因分解: 该块无 evidence_quote(空) → quote_missing(quote 回退)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: {
                            "query_id": entry["query_id"], "query": entry["query"],
                            "expected_parent_ids": ["Crawler/0:p0"],
                            "relevance": {"Crawler/0:p0": 3},
                            "evidence_quote": {"Crawler/0:p0": ""},
                            "mark_for_delete": False, "hint_location": None,
                            "confidence": 0.9, "question_type": "factual"})
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: False)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": True, "term_overlap": 0.9,
                                                          "cos_sim": 0.9, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")])
    assert "quote_missing" in result["attribution"][0]["reject_reasons"]


# ---- 009: 三方案可切换代码路径 ----

def test_semantic_gate_params_modes():
    """_semantic_gate_params 按三方案返回 (cos_threshold, term_overlap_low)。"""
    assert cli_mod._semantic_gate_params("hard", None) == (None, None)
    assert cli_mod._semantic_gate_params("weak", None) == (0.4, 0.0)
    assert cli_mod._semantic_gate_params("remove", None) == (None, None)
    assert cli_mod._semantic_gate_params("data", 0.65) == (0.65, 0.0)
    with pytest.raises(ValueError):
        cli_mod._semantic_gate_params("data", None)


def test_semantic_mode_weak_uses_weak_floor(monkeypatch):
    """009 方案甲(weak): verify_semantic 用 cos_threshold=0.4, term_overlap_low=0.0。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    captured: dict = {}
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None, captured=captured: (
                            captured.__setitem__("cos_threshold", cos_threshold),
                            captured.__setitem__("term_overlap_low", term_overlap_low)) or {
                            "passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    cli_mod.run_pipeline(_doc_index(), [_entry()], semantic_gate_mode="weak")
    assert captured["cos_threshold"] == 0.4
    assert captured["term_overlap_low"] == 0.0


def test_semantic_mode_remove_cos_only_diag(monkeypatch):
    """009 方案乙(remove): cos 只写 semantic_diag, 不参与采纳; all_grounded 只做引用存在性。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    # cos 极低(0.1)但乙下不参与采纳 → 仍 passed
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {"passed": False, "term_overlap": 0.0,
                                                          "cos_sim": 0.1, "effective_source": "chunk"})
    result = cli_mod.run_pipeline(_doc_index(), [_entry("crawler-0000")], semantic_gate_mode="remove")
    assert "crawler-0000" in result["passed_adopted"]
    assert result["review_list"] == []
    assert len(result["semantic_diag"]) == 1
    assert result["semantic_diag"][0]["semantic_gate_mode"] == "remove"
    assert result["attribution"] == []


def test_semantic_mode_data_uses_calibrated_threshold(monkeypatch):
    """009 方案丙(data): verify_semantic 用 cos_threshold=显式校准阈值, term_overlap_low=0.0。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    captured: dict = {}
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None, captured=captured: (
                            captured.__setitem__("cos_threshold", cos_threshold),
                            captured.__setitem__("term_overlap_low", term_overlap_low)) or {
                            "passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    cli_mod.run_pipeline(_doc_index(), [_entry()], semantic_gate_mode="data", semantic_threshold=0.65)
    assert captured["cos_threshold"] == 0.65
    assert captured["term_overlap_low"] == 0.0


# ---- 008 §3.1: 并发加速(线程池并行化) ----

def test_parallel_workers_processes_all_entries(monkeypatch):
    """008 §3.1 并行化: workers>1 线程池并发处理全部条目, 结果与串行一致。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {
                            "passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    items = [_entry(f"crawler-{index:04d}", source_doc="Crawler/0") for index in range(6)]
    result = cli_mod.run_pipeline(_doc_index(), items, workers=4)
    assert set(result["passed_adopted"].keys()) == {f"crawler-{index:04d}" for index in range(6)}
    assert result["review_list"] == []
    assert len(result["processed"]) == 6


def test_parallel_on_entry_called(monkeypatch):
    """008 §3.1 并行化: workers>1 时 on_entry 每条处理完回调一次(即时写盘仍成立)。"""
    monkeypatch.setattr(cli_mod, "propose",
                        lambda entry, chunks, known_ids=None, timeout=None: _normal_proposal(entry["query_id"]))
    monkeypatch.setattr(cli_mod, "verify_two_pass",
                        lambda proposal, chunks, negative_blocks=None, known_ids=None, timeout=None: {
                            "review_needed": False, "consistent": True, "negative_rejected": True,
                            "round1": {}, "round2": {}, "agrees_with_round1": True, "full_superset": True})
    monkeypatch.setattr(cli_mod, "verify_grounding", lambda annotation, chunk: True)
    monkeypatch.setattr(cli_mod, "verify_semantic",
                        lambda annotation, query, chunk, cos_threshold=None, term_overlap_low=None: {
                            "passed": True, "term_overlap": 0.9, "cos_sim": 0.9, "effective_source": "chunk"})
    snapshots: list[dict] = []
    items = [_entry("crawler-0000", source_doc="Crawler/0"),
             _entry("crawler-0001", source_doc="Crawler/0")]
    cli_mod.run_pipeline(_doc_index(), items, workers=2, on_entry=snapshots.append)
    assert len(snapshots) == 2
    # 最终快照含全部条目(即时写盘累计)
    assert set(snapshots[-1]["passed_adopted"].keys()) == {"crawler-0000", "crawler-0001"}
