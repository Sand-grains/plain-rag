"""test_anno_llm_prompt.py：F46 prompt.py 单元测试(提示词组装 + LLM 调用)。

覆盖 anno_llm.md §7 的 build_prompt / call_llm 对应用例：
    输入隔离 / 编号 / 不含 reference_facts / 多块与空输出删除分支(hint_location) /
    mock API 解析结构化输出(含空输出/删除/多块形态)。
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark.anno_llm.prompt import (
    CHUNK_ID_CORRECTION_PROMPT_TEMPLATE,
    LLMCallError,
    LLMParseError,
    PROPOSER_PROMPT_TEMPLATE,
    SUFFICIENCY_JUDGE_PROMPT_TEMPLATE,
    VERIFIER_PROMPT_TEMPLATE,
    build_prompt,
    call_llm,
    format_chunk_index,
    parse_json_response,
)


def _chunk(chunk_id: str, content: str) -> SimpleNamespace:
    """构造最小父块对象(anno_tool 复用侧只消费 chunk_id/content)。"""
    return SimpleNamespace(chunk_id=chunk_id, content=content)


def _entry(query_id: str = "crawler-0000", source_doc: str = "Crawler/0000",
           query: str = "设计模式分几类？",
           reference_facts: str = "幻觉 gold: 绝不能出现在 prompt 里") -> dict:
    """构造单条 entry(带 reference_facts 验证输入隔离)。"""
    return {
        "query_id": query_id,
        "query": query,
        "source_doc": source_doc,
        "category": "crawler",
        "reference_facts": reference_facts,
        "expected_files": ["Crawler/0000"],
        "expected_pages": [],
    }


# ---- build_prompt: 输入隔离 / 编号 / 多块 ----

def test_build_prompt_isolates_reference_facts():
    """reference_facts 绝不进入 prompt(输入隔离, 防把幻觉 gold 喂回 LLM)。"""
    entry = _entry()
    chunks = [_chunk("Crawler/0000:p0", "第一块内容"), _chunk("Crawler/0000:p1", "第二块内容")]
    prompt = build_prompt(entry, chunks, PROPOSER_PROMPT_TEMPLATE)
    # 输入隔离 = 绝不喂 reference_facts 的 gold 值; 模板中"reference_facts"仅出现在
    # 输出说明("程序从 evidence_quote 拼接, 你勿自行填写"), 非作为输入线索喂给 LLM。
    assert entry["reference_facts"] not in prompt
    assert entry["query"] in prompt


def test_build_prompt_numbers_chunks_with_ids():
    """父块编号 [CHUNK n] + doc id, 支撑标注精确指到块。"""
    chunks = [_chunk("Crawler/0000:p0", "块A"), _chunk("Crawler/0000:p1", "块B")]
    block_text = format_chunk_index(chunks)
    assert "[CHUNK 1] Crawler/0000:p0" in block_text
    assert "[CHUNK 2] Crawler/0000:p1" in block_text
    assert "块A" in block_text and "块B" in block_text


def test_build_prompt_backfills_entry_meta_without_gold():
    """条目元数据原样回填, 但绝不回填 gold 字段(expected_parent_ids/relevance 等)。"""
    entry = _entry()
    chunks = [_chunk("Crawler/0000:p0", "内容")]
    prompt = build_prompt(entry, chunks, PROPOSER_PROMPT_TEMPLATE)
    assert "crawler-0000" in prompt
    assert "Crawler/0000" in prompt
    assert '"category": "crawler"' in prompt
    # 原 query 出现且不被改写
    assert "设计模式分几类？" in prompt


def test_build_prompt_supports_delete_branch_template():
    """删除分支硬规则(无解/问偏→空数组+mark_for_delete+hint_location)写入模板。"""
    assert "mark_for_delete" in PROPOSER_PROMPT_TEMPLATE


def test_templates_have_no_escaped_double_braces():
    """防回归(P0 实跑 bug): 模板不得含 {{/}} —— 双花括号会泄漏进 prompt, LLM 照抄成
    {{ 导致 JSON 解析失败。build_prompt 用 .replace() 填充, 示例 JSON 必须单花括号。"""
    assert "{{" not in PROPOSER_PROMPT_TEMPLATE
    assert "}}" not in PROPOSER_PROMPT_TEMPLATE
    assert "{{" not in VERIFIER_PROMPT_TEMPLATE
    assert "}}" not in VERIFIER_PROMPT_TEMPLATE
    assert "hint_location" in PROPOSER_PROMPT_TEMPLATE
    assert "expected_parent_ids" in PROPOSER_PROMPT_TEMPLATE


def test_build_prompt_verifier_two_round_placeholders():
    """校验者模板含 ROUND1/ROUND2 两轮占位符与一致性字段。"""
    assert "{VERIFY_ROUND}" in VERIFIER_PROMPT_TEMPLATE
    assert "{VERIFY_INPUT_BLOCKS}" in VERIFIER_PROMPT_TEMPLATE
    assert "agrees_with_round1" in VERIFIER_PROMPT_TEMPLATE
    assert "full_superset" in VERIFIER_PROMPT_TEMPLATE
    assert "delete_confirmed" in VERIFIER_PROMPT_TEMPLATE


def test_verifier_round2_prompt_no_contradiction():
    """回归(根因1): ROUND2 提示词不得再自相矛盾——"第一轮遗漏(第二轮找到更多)"必须判为
    通过(full_superset=true), 而非"标记不一致"。"""
    # 新表述: 明确"第一轮遗漏 = 通过(superset=true)"
    assert "第一轮遗漏" in VERIFIER_PROMPT_TEMPLATE
    assert "full_superset=true" in VERIFIER_PROMPT_TEMPLATE
    # 旧冲突表述(把"遗漏"与"标记不一致"绑定)必须移除
    assert "若第二轮发现第一轮遗漏/多余/矛盾，标记不一致" not in VERIFIER_PROMPT_TEMPLATE
    # 仅"第二轮 ⊉ 第一轮(多余/矛盾)"才判不一致
    assert "第二轮 ⊉ 第一轮" in VERIFIER_PROMPT_TEMPLATE


def test_proposer_template_has_max_chunks_constraint():
    """提议者模板含 4-chunk 上限硬规则(最小答案闭环)。"""
    assert "最多 4 个块" in PROPOSER_PROMPT_TEMPLATE
    assert "总体不超过 4 个" in PROPOSER_PROMPT_TEMPLATE


def test_verifier_template_has_max_chunks_constraint():
    """校验者模板 ROUND2 含 4-chunk 上限约束(即使找到更多也不能多于 4)。"""
    assert "总体也不能多于 4 个" in VERIFIER_PROMPT_TEMPLATE
    assert "≤4 个块内即可达到最小答案闭环" in VERIFIER_PROMPT_TEMPLATE


def test_sufficiency_judge_template_exists():
    """存在第三者充分性判定模板(sufficient/partial/insufficient 分级)。"""
    assert "sufficient" in SUFFICIENCY_JUDGE_PROMPT_TEMPLATE
    assert "partial" in SUFFICIENCY_JUDGE_PROMPT_TEMPLATE
    assert "insufficient" in SUFFICIENCY_JUDGE_PROMPT_TEMPLATE
    assert "sufficiency" in SUFFICIENCY_JUDGE_PROMPT_TEMPLATE
    # 第三者只读候选块 + query, 严禁自身知识
    assert "严禁使用自身知识" in SUFFICIENCY_JUDGE_PROMPT_TEMPLATE


def test_chunkid_correction_template_exists():
    """存在 chunk_id 格式纠正模板(提醒 source_doc:块序号, 禁 doc:xxx/CHUNK n/doc:) 。"""
    assert "格式纠正" in CHUNK_ID_CORRECTION_PROMPT_TEMPLATE
    assert "{INVALID_IDS}" in CHUNK_ID_CORRECTION_PROMPT_TEMPLATE
    assert "{ORIGINAL_PROMPT}" in CHUNK_ID_CORRECTION_PROMPT_TEMPLATE
    assert "严禁使用 doc:xxx:p0" in CHUNK_ID_CORRECTION_PROMPT_TEMPLATE


def test_proposer_template_has_chunk_id_format_rule():
    """提议者模板含 chunk_id 统一口径硬规则(真实 id, 禁 doc:xxx/CHUNK n/doc: 前缀)。"""
    assert "真实 id" in PROPOSER_PROMPT_TEMPLATE
    assert "严禁编造 doc:xxx:p0" in PROPOSER_PROMPT_TEMPLATE
    assert "source_doc:块序号" in PROPOSER_PROMPT_TEMPLATE


def test_verifier_template_has_per_block_key_rule():
    """校验者模板含 per_block key 必须用真实 chunk_id 的硬规则。"""
    assert "真实 chunk_id" in VERIFIER_PROMPT_TEMPLATE
    assert "严禁编造 doc:xxx:p0" in VERIFIER_PROMPT_TEMPLATE


def test_build_prompt_unknown_placeholder_raises():
    """模板含未知占位符时应报错(防模板漂移带病跑 LLM)。"""
    chunks = [_chunk("Crawler/0000:p0", "内容")]
    with pytest.raises(KeyError):
        build_prompt(_entry(), chunks, "未知占位符 {UNKNOWN_KEY} 在此")


def test_build_prompt_doc_content_braces_not_placeholder():
    """回归: 文档正文含 {HNSW}/{IVF}(Milvus 索引类型)不得被误判为未填充占位符。
    实跑崩溃 bug: 填充后检查把文档内容里的 {HNSW}/{IVF} 当模板占位符抛 KeyError。"""
    chunks = [_chunk("Crawler/0000:p0", "Milvus 支持 {HNSW} 与 {IVF} 两种索引类型")]
    prompt = build_prompt(_entry(), chunks, PROPOSER_PROMPT_TEMPLATE)
    assert "{HNSW}" in prompt
    assert "{IVF}" in prompt


# ---- parse_json_response ----

def test_parse_json_strips_code_fence():
    """剥离 ```json 代码围栏后仍能解析。"""
    content = "```json\n{\"query_id\": \"crawler-0000\"}\n```"
    assert parse_json_response(content) == {"query_id": "crawler-0000"}


def test_parse_json_finds_object_amid_noise():
    """从带前后文字的输出中定位 JSON 对象。"""
    content = "好的，这是标注：{\"query_id\": \"crawler-0000\", \"mark_for_delete\": false} 结束"
    parsed = parse_json_response(content)
    assert parsed["query_id"] == "crawler-0000"
    assert parsed["mark_for_delete"] is False


def test_parse_json_non_object_raises():
    """非对象(数组/标量)抛 LLMParseError。"""
    with pytest.raises(LLMParseError):
        parse_json_response("[1, 2, 3]")


def test_parse_json_missing_raises():
    """无 JSON 对象时抛 LLMParseError。"""
    with pytest.raises(LLMParseError):
        parse_json_response("没有任何对象")


# ---- call_llm ----

class _FakeCompletions:
    """最小 OpenAI 兼容 completions mock(只实现 create)。"""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> Mock:
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        message = Mock()
        message.content = content
        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        return response


class _FakeClient:
    """最小 OpenAI 兼容客户端 mock(chat.completions.create)。"""

    def __init__(self, responses: list[str]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))

    @property
    def completions(self) -> _FakeCompletions:
        return self.chat.completions


def test_call_llm_parses_structured_json():
    """mock API 返回结构化 JSON(多块/删除形态)能被正确解析。"""
    payload = '{"query_id": "crawler-0000", "expected_parent_ids": ["Crawler/0000:p0"], ' \
              '"expected_files": ["Crawler/0000"], "mark_for_delete": false, "confidence": 0.9}'
    client = _FakeClient([payload])
    result = call_llm("prompt", client=client, model="deepseek-v4-flash")
    assert result["query_id"] == "crawler-0000"
    assert result["expected_parent_ids"] == ["Crawler/0000:p0"]
    assert result["confidence"] == 0.9


def test_call_llm_temperature_zero_default():
    """默认 temperature=0(anno_llm.md 硬约束), 且使用传入 model。"""
    client = _FakeClient(['{"query_id": "x"}'])
    call_llm("prompt", client=client, model="anno-model")
    kwargs = client.completions.calls[0]
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == "anno-model"
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_call_llm_passes_timeout_to_create():
    """单次超时透传: timeout 传给 OpenAI create(缺省 None 不设超时)。"""
    client = _FakeClient(['{"query_id": "x"}'])
    call_llm("prompt", client=client, model="anno-model", timeout=60.0)
    assert client.completions.calls[0]["timeout"] == 60.0
    # 缺省不传 timeout → create 收到 None(用 SDK 默认)
    client2 = _FakeClient(['{"query_id": "x"}'])
    call_llm("prompt", client=client2, model="anno-model")
    assert client2.completions.calls[0]["timeout"] is None


def test_call_llm_retries_once_on_parse_failure_then_success():
    """首次输出非 JSON → 重试一次; 第二次成功返回。"""
    client = _FakeClient(["乱码输出", '{"query_id": "ok"}'])
    result = call_llm("prompt", client=client)
    assert result == {"query_id": "ok"}
    assert len(client.completions.calls) == 2


def test_call_llm_two_parse_failures_raise():
    """连续两次解析失败抛 LLMParseError。"""
    client = _FakeClient(["乱码", "还是乱码"])
    with pytest.raises(LLMParseError):
        call_llm("prompt", client=client)
    assert len(client.completions.calls) == 2


def test_call_llm_api_error_raises():
    """API 调用抛异常转 LLMCallError。"""

    class _Boom:
        def create(self, **kwargs):
            raise RuntimeError("网络抖动")

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Boom()))
    with pytest.raises(LLMCallError):
        call_llm("prompt", client=client)


def test_call_llm_empty_choices_raises_llm_call_error():
    """API 返回空 choices → 转 LLMCallError(不抛 IndexError 崩批)。"""

    class _EmptyCompletions:
        def create(self, **kwargs):
            response = Mock()
            response.choices = []
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=_EmptyCompletions()))
    with pytest.raises(LLMCallError):
        call_llm("prompt", client=client)


# ---- F52: 瞬时错误重试/退避 ----

def test_is_transient_error_classification():
    """瞬时/永久错误分类: 5xx/429/timeout/connection 瞬时, auth/400/404/422 永久。"""
    from benchmark.anno_llm.prompt import _is_transient_error

    timeout = RuntimeError("timeout")
    timeout.status_code = 504
    assert _is_transient_error(timeout) is True
    rate = RuntimeError("rate")
    rate.status_code = 429
    assert _is_transient_error(rate) is True
    bad = RuntimeError("bad")
    bad.status_code = 400
    assert _is_transient_error(bad) is False
    auth = RuntimeError("auth")
    auth.status_code = 401
    assert _is_transient_error(auth) is False
    conn = RuntimeError("APIConnectionError")
    assert _is_transient_error(conn) is True


def test_call_llm_retries_transient_error_then_success(monkeypatch):
    """瞬时错误(5xx)退避重试后成功。"""
    import benchmark.anno_llm.prompt as prompt_mod

    monkeypatch.setattr(prompt_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(prompt_mod.random, "uniform", lambda low, high: 0.0)
    calls = {"n": 0}

    class _TransientThenOk:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                exc = RuntimeError("timeout")
                exc.status_code = 504
                raise exc
            message = Mock()
            message.content = '{"query_id": "ok"}'
            choice = Mock()
            choice.message = message
            response = Mock()
            response.choices = [choice]
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=_TransientThenOk()))
    result = call_llm("prompt", client=client)
    assert result == {"query_id": "ok"}
    assert calls["n"] == 2  # 初始 + 1 次重试


def test_call_llm_no_retry_on_permanent_error(monkeypatch):
    """永久错误(auth/400/404/422)不重试, 直接转 LLMCallError。"""
    calls = {"n": 0}

    class _Permanent:
        def create(self, **kwargs):
            calls["n"] += 1
            exc = RuntimeError("bad request")
            exc.status_code = 400
            raise exc

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Permanent()))
    with pytest.raises(LLMCallError):
        call_llm("prompt", client=client)
    assert calls["n"] == 1  # 不重试


def test_call_llm_transient_retry_exhausted(monkeypatch):
    """瞬时错误重试耗尽后仍失败 → LLMCallError(初始 + 2 次重试 = 3 次)。"""
    import benchmark.anno_llm.prompt as prompt_mod

    monkeypatch.setattr(prompt_mod.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(prompt_mod.random, "uniform", lambda low, high: 0.0)
    calls = {"n": 0}

    class _AlwaysTransient:
        def create(self, **kwargs):
            calls["n"] += 1
            exc = RuntimeError("timeout")
            exc.status_code = 504
            raise exc

    client = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysTransient()))
    with pytest.raises(LLMCallError):
        call_llm("prompt", client=client, max_retries=3)
    assert calls["n"] == 3


# ---- F52: 提议者 prompt 摘核心答案句 ----

def test_proposer_template_has_core_answer_sentence_rule():
    """提议者模板含摘核心答案句硬规则(F52, single_chunk 配套)。"""
    assert "核心答案句" in PROPOSER_PROMPT_TEMPLATE
    assert "不要摘弱句" in PROPOSER_PROMPT_TEMPLATE
    assert "宁可摘完整句也不摘残缺片段" in PROPOSER_PROMPT_TEMPLATE
