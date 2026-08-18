"""Layer 2 LLM-as-Judge：faithfulness + quality 双调用并行评估。

核心特性：
    - judge_faithfulness：评估生成答案对检索上下文的忠实度（5 级离散锚点评分 + grounded_claims 提取）
    - judge_quality：评估 answer_relevancy / context_precision / context_recall / answer_correctness 四个质量维度
    - 内层 ThreadPoolExecutor(max_workers=2) 并行提交两个 Judge 调用
    - _judge_with_retry：独立 1-worker 池 + future.result(timeout=deadline) + 指数退避（base_delay * 2^attempt * jitter）
    - _call_llm：调用后 extract_json 三层兜底（json.loads → markdown fence → regex brace），解析失败 record_parse_error + alert
    - execute_verdict 纯函数：有值维度中 ≥ 0.75 的比例 → pass / partial / fail
    - eval 场景 temperature=0（透传），确保 Judge 评分可复现
    - JudgeResult 含 4 个阶段延迟字段，向前兼容旧缓存（新字段默认 None）

用法示例::

    from eval.core.llm_as_judge.judge import run_judge, JudgeResult, execute_verdict
    result = run_judge("Q001", query, chunks, answer, reference_facts, temperature=0.0)
    print(result.faithfulness, result.verdict)

公共接口：
    - JudgeResult: 单条 query 的完整评估结果（5 项分数 + verdict + 异常信息 + 阶段延迟）
    - run_judge: 完整双调用 Judge 流程（含缓存查/写 + 并行提交 + verdict 计算）
    - execute_verdict: 纯函数，分数 dict → pass/partial/fail
    - JudgeFailedError: Judge LLM 调用重试耗尽后抛出的异常
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

import httpx
import openai
from openai import OpenAI

from config import (LLM_API_KEY, LLM_BASE_URL, EVAL_LLM_MODEL_ID,
                    JUDGE_MAX_RETRY, JUDGE_BASE_DELAY, JUDGE_DEADLINE, EVAL_THREADPOOL_WORKERS)
from eval.utils import clamp_score, extract_json
from eval.core.llm_as_judge.judge_formatter import Formatter, build_judge_context, get_formatter
from eval.core.llm_as_judge.judge_cache import _cache_judge_key, get_judge_cache, set_judge_cache
from obs import get_panel

if TYPE_CHECKING:
    from indexing.chunk import Chunk


# ---- 异常与结果类型 ----

class JudgeFailedError(Exception):
    """Judge LLM 调用重试耗尽后抛出。"""


RETRYABLE_ERRORS = (
    openai.RateLimitError,       # 429
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class JudgeResult:
    """单条 query 的 Layer 2 评估结果。"""
    query_id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    grounded_claims: list[dict] = field(default_factory=list)
    verdict: str = ""
    parse_error: str | None = None
    judge_error: str | None = None
    generator_error: str | None = None
    # 阶段延迟（由 runner / run_judge 填入）
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    judge_faithfulness_ms: float | None = None
    judge_quality_ms: float | None = None

    def to_json(self) -> str:
        """序列化为 JSON 字符串（供缓存存储）。"""
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> JudgeResult:
        """从 JSON 字符串反序列化 JudgeResult。"""
        return cls(**json.loads(json_str))


# ---- LLM 调用 ----

def _call_llm(client: OpenAI, model: str, prompt: str,
             temperature: float = 0.0, query_id: str = "",
             call_type: str = "") -> dict:
    """调用 LLM 并解析 LLM 返回的 JSON 输出（经 extract_json 三层兜底）。

    Args:
        client: OpenAI 兼容客户端。
        model: 模型 id。
        prompt: 发送给模型的完整 prompt。
        temperature: 采样温度（eval 下为 0，保证可复现）。
        query_id: 当前 query_id（解析失败时用于面板告警）。
        call_type: 调用类型标记（judge_faithfulness / judge_quality），用于指标打点。

    Returns:
        dict：解析成功的 JSON；解析失败返回 {"error": ..., "raw": ...}。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", "content": prompt
            }
        ],
        temperature=temperature,
    )
    try:
        from obs import get_metrics
        if call_type:
            get_metrics().record_llm_call(call_type)
        if response.usage:
            get_metrics().record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
    except ImportError:
        pass
    text = response.choices[0].message.content
    if text is None:
        return {"error": "LLM returned empty content", "raw": ""}
    try:
        return extract_json(text)
    except (json.JSONDecodeError, ValueError) as error:
        from obs import get_metrics
        try:
            get_metrics().record_parse_error()
        except ImportError:
            pass
        if query_id:
            panel = get_panel()
            if panel:
                panel.alert(query_id, f"Judge {call_type} parse error: {error}")
        return {"error": str(error), "raw": text[:500]}


def _judge_with_retry(
    judge_fn: Callable,
    judge_args: tuple,
    max_retries: int = 3,
    base_delay: float = 1.0,
    deadline: float = 120.0,
    query_id: str = "",
    judge_type: str = "",
) -> tuple:
    """对 judge_fn 做指数退避重试 + deadline 超时。

    pool 在循环外创建，finally shutdown(wait=False) 避免阻塞等孤儿线程。

    Args:
        judge_fn: 待重试的 Judge 调用函数。
        judge_args: 传给 judge_fn 的位置参数。
        max_retries: 最大重试次数。
        base_delay: 退避基准延迟（秒），实际延迟乘随机 jitter 0.5~1.5。
        deadline: 单次调用的超时阈值（秒）。
        query_id: 当前 query_id（告警用）。
        judge_type: 调用类型（faithfulness / quality）。 告警用。

    Returns:
        tuple：judge_fn 的返回值（分数元组）。

    Raises:
        JudgeFailedError: 重试耗尽后抛出。
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-retry")
    try:
        last_error = None
        for attempt in range(max_retries):
            future = pool.submit(judge_fn, *judge_args)
            try:
                return future.result(timeout=deadline)
            except TimeoutError:
                future.cancel()
                last_error = TimeoutError(f"Judge call timed out after {deadline}s")
            except RETRYABLE_ERRORS as error:
                from obs import get_metrics
                try:
                    get_metrics().record_retry()
                except ImportError:
                    pass
                if isinstance(error, openai.RateLimitError):
                    if query_id:
                        panel = get_panel()
                        if panel:
                            panel.alert(query_id,
                                f"Judge {judge_type} 429 限流 第{attempt + 1}/{max_retries}次重试")
                    logging.warning(
                        "DeepSeek 限流 (429)，第 %d/%d 次尝试",
                        attempt + 1, max_retries,
                    )
                last_error = error
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * random.uniform(0.5, 1.5)
                time.sleep(delay)
        if query_id:
            panel = get_panel()
            if panel:
                panel.alert(query_id,
                    f"Judge {judge_type} 重试耗尽: {type(last_error).__name__}")
        raise JudgeFailedError(
            f"Judge call failed after {max_retries} retries: {last_error}"
        )
    finally:
        pool.shutdown(wait=False)


# ---- 两个 Judge 调用 ----

def judge_faithfulness(
    client: OpenAI, model: str, formatter: Formatter,
    query: str, context_str: str, answer: str,
    temperature: float = 0.0, query_id: str = "",
) -> tuple[float | None, list[dict], str | None]:
    """调用 1：评估 faithfulness。

    Args:
        client: OpenAI 兼容客户端。
        model: 模型 id。
        formatter: prompt 格式化器。
        query: 用户问题。
        context_str: 结构化检索上下文。
        answer: 生成器回答。
        temperature: 采样温度。
        query_id: 当前 query_id。

    Returns:
        tuple[float | None, list[dict], str | None]：(分数, grounded_claims, 解析错误描述)。
    """
    prompt = formatter.build_faithfulness_prompt(query, context_str, answer)
    result = _call_llm(client, model, prompt, temperature=temperature,
                       query_id=query_id, call_type="judge_faithfulness")

    if "error" in result:
        return None, [], result.get("error")

    score = result.get("faithfulness")
    if score is not None:
        try:
            score = clamp_score(float(score))
        except (ValueError, TypeError):
            score = None
    return score, result.get("grounded_claims", []), None


def judge_quality(
    client: OpenAI,
    model: str,
    formatter: Formatter,
    query: str,
    context_str: str,
    answer: str,
    reference_facts: str,
    temperature: float = 0.0,
    query_id: str = "", ) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """调用 2：评估四个质量维度。

    Args:
        client: OpenAI 兼容客户端。
        model: 模型 id。
        formatter: prompt 格式化器。
        query: 用户问题。
        context_str: 结构化检索上下文。
        answer: 生成器回答。
        reference_facts: 参考答案事实。
        temperature: 采样温度。
        query_id: 当前 query_id。

    Returns:
        tuple[float | None, float | None, float | None, float | None, str | None]：
        (answer_relevancy, context_precision, context_recall, answer_correctness, 解析错误描述)。
    """
    prompt = formatter.build_quality_prompt(query, context_str, answer, reference_facts)
    result = _call_llm(client, model, prompt, temperature=temperature,
                       query_id=query_id, call_type="judge_quality")

    if "error" in result:
        return None, None, None, None, result.get("error")

    def _get(key: str) -> float | None:
        value = result.get(key)
        if value is None:
            return None
        try:
            return clamp_score(float(value))
        except (ValueError, TypeError):
            return None

    return (
        _get("answer_relevancy"),
        _get("context_precision"),
        _get("context_recall"),
        _get("answer_correctness"),
        None,
    )


# ---- verdict 计算 ----

def execute_verdict(scores: dict[str, float | None]) -> str:
    """纯函数：有值维度中 >= 0.75 的比例 → pass/partial/fail。

    >>> execute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.75, "context_precision": 0.50, "context_recall": 0.75, "answer_correctness": 0.75})
    'pass'
    >>> execute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.50, "context_precision": 0.75, "context_recall": 0.50, "answer_correctness": None})
    'partial'
    >>> execute_verdict({"faithfulness": 0.25, "answer_relevancy": 0.25, "context_precision": 0.25, "context_recall": None, "answer_correctness": 0.25})
    'fail'

    Args:
        scores: 各评估维度的分数（None 表示缺失，不参与计算）。

    Returns:
        str：pass / partial / fail。
    """
    valid = [value for value in scores.values() if value is not None]
    if not valid:
        return "fail"
    ratio = sum(1 for value in valid if value >= 0.75) / len(valid)
    if ratio >= 0.75:
        return "pass"
    elif ratio >= 0.50:
        return "partial"
    else:
        return "fail"


# ---- client 单例 ----

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """模块级 OpenAI client 单例，避免每次 run_judge 重复创建连接池。

    Returns:
        OpenAI：全局唯一客户端实例。
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_keepalive_connections=max(20, EVAL_THREADPOOL_WORKERS * 4),
                    max_connections=max(24, EVAL_THREADPOOL_WORKERS * 5),
                ),
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=JUDGE_DEADLINE + 10,
                    write=30.0,
                    pool=5.0,
                ),
            ),
        )
    return _client


# ---- 编排入口 ----

def _timed_judge_with_retry(
    judge_fn: Callable,
    judge_args: tuple,
    max_retries: int,
    base_delay: float,
    deadline: float,
    query_id: str,
    judge_type: str,
) -> tuple[tuple, float]:
    """带计时的 _judge_with_retry 包装器。

    每个 future 内部独立计时——消除并行调用间因 future.result() 串行等待导致的计时污染。

    Returns:
        tuple[tuple, float]：(judge_fn 返回值元组, 耗时毫秒)。
    """
    start_time = time.time()
    result = _judge_with_retry(judge_fn, judge_args, max_retries, base_delay, deadline, query_id, judge_type)
    return result, (time.time() - start_time) * 1000


def run_judge(
    query_id: str, query: str, chunks: list[Chunk], answer: str, reference_facts: str,
    client: OpenAI | None = None,
    model: str | None = None,
    formatter: Formatter | None = None,
    temperature: float = 0.0,
    skip_cache: bool = False,
) -> JudgeResult:
    """执行完整的双调用 Judge 流程，返回 JudgeResult。

    skip_cache=True 时绕过缓存读写——校准路径同 query+context 评不同答案，
    缓存键不含 answer 会串台，须强制绕过。

    Args:
        query_id: benchmark 条目 query_id。
        query: 用户问题。
        chunks: 检索返回的 chunk 列表（作为判定上下文）。
        answer: 生成器回答。
        reference_facts: 参考答案事实。
        client: OpenAI 客户端（默认模块级单例）。
        model: Judge 模型 id（默认 EVAL_LLM_MODEL_ID）。
        formatter: prompt 格式化器（默认模块级单例）。
        temperature: 采样温度（eval 下为 0）。
        skip_cache: 是否绕过缓存读写。

    Returns:
        JudgeResult：五项分数 + verdict + 异常信息 + 阶段延迟。
    """
    if client is None:
        client = _get_client()
    if model is None:
        model = EVAL_LLM_MODEL_ID
    if formatter is None:
        formatter = get_formatter()

    context_str = build_judge_context(chunks)
    answer_str = answer or "（模型未生成回答）"

    # 查缓存（v5后不再包含 answer_hash）
    prompt_version_hash = formatter.prompt_version_hash
    cache_key = _cache_judge_key(query_id, context_str, prompt_version_hash, model)
    if not skip_cache:
        cached = get_judge_cache(cache_key)
        if cached is not None:
            return cached

    result = JudgeResult(query_id=query_id)

    # 内层 2-worker 池：faithfulness + quality 并行（各自独立计时）
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="judge") as pool:
        faith_future = pool.submit(
            _timed_judge_with_retry, judge_faithfulness,
            (client, model, formatter, query, context_str, answer_str,
             temperature, query_id),
            JUDGE_MAX_RETRY, JUDGE_BASE_DELAY, JUDGE_DEADLINE,
            query_id, "faithfulness",
        )
        quality_future = pool.submit(
            _timed_judge_with_retry, judge_quality,
            (client, model, formatter, query, context_str, answer_str,
             reference_facts, temperature, query_id),
            JUDGE_MAX_RETRY, JUDGE_BASE_DELAY, JUDGE_DEADLINE,
            query_id, "quality",
        )

        try:
            (faithfulness, grounded_claims, faithfulness_error), faithfulness_ms = faith_future.result()
        except JudgeFailedError as error:
            faithfulness, grounded_claims, faithfulness_error, faithfulness_ms = None, [], str(error), 0.0
        result.judge_faithfulness_ms = faithfulness_ms

        try:
            (answer_relevancy, context_precision, context_recall, answer_correctness, quality_error), quality_ms = quality_future.result()
        except JudgeFailedError as error:
            answer_relevancy, context_precision, context_recall, answer_correctness = None, None, None, None
            quality_error, quality_ms = str(error), 0.0
        result.judge_quality_ms = quality_ms

    result.faithfulness = faithfulness
    result.grounded_claims = grounded_claims
    result.answer_relevancy = answer_relevancy
    result.context_precision = context_precision
    result.context_recall = context_recall
    result.answer_correctness = answer_correctness

    # 区分 parse_error 和 judge_error：JudgeFailedError → judge_error；其余 → parse_error
    judge_errors = []
    for label, error_value in [("faithfulness", faithfulness_error), ("quality", quality_error)]:
        if not error_value:
            continue
        if "Judge call failed" in error_value:
            judge_errors.append(f"{label}: {error_value}")
        else:
            if result.parse_error:
                result.parse_error += f"; {label}: {error_value}"
            else:
                result.parse_error = f"{label}: {error_value}"
    if judge_errors:
        result.judge_error = "; ".join(judge_errors)

    # verdict
    scores = {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
        "answer_correctness": result.answer_correctness,
    }
    result.verdict = execute_verdict(scores)

    # 写缓存（skip_cache=True 不写，避免校准结果污染主评测键空间）
    if not skip_cache:
        set_judge_cache(cache_key, result)

    return result
