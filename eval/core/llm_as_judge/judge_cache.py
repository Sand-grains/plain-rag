"""双层缓存键构造 + Judge 缓存 get/set。

核心特性：
    - Generator 缓存键：generator:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}
    - Judge 缓存键：judge:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}:{prompt_version}:{model_id}
    - GENERATOR_CONFIG_HASH 为模块级常量（config.py），捕获 Generator 全部配置变更
    - 废弃 answer_hash（temp > 0 时措辞抖动导致缓存永久失效）
    - Generator 缓存在 runner.py 的 _evaluate_one 中直接操作 Redis，Judge 缓存通过 get_judge_cache/set_judge_cache 编排

用法示例::

    from eval.core.llm_as_judge.judge_cache import _cache_judge_key, _cache_generator_key, get_judge_cache, set_judge_cache
    key = _cache_judge_key("Q001", context_str, "v1", "deepseek-chat")
    cached = get_judge_cache(key)  # → JudgeResult | None

公共接口：
    - _cache_judge_key: 构造 Judge 缓存键（4 参数，不含 answer）
    - _cache_generator_key: 构造 Generator 缓存键
    - get_judge_cache: 查 Judge 缓存并记录命中/未命中
    - set_judge_cache: 写 Judge 缓存
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from config import GENERATOR_CONFIG_HASH
from infra.cache import get_cache
from infra.config import REDIS_DEFAULT_TTL

if TYPE_CHECKING:
    from eval.core.llm_as_judge.judge import JudgeResult


def _cache_judge_key(query_id: str, context_str: str,
                     prompt_version: str, model_id: str) -> str:
    """构造 Judge 缓存键（不含 answer_hash，改用 GENERATOR_CONFIG_HASH 作为 Generator 配置变更信号）。

    Args:
        query_id: benchmark 条目 query_id。
        context_str: 检索上下文（build_judge_context 产出）。
        prompt_version: Judge prompt 版本号（模板内容哈希）。
        model_id: Judge 模型 id。

    Returns:
        str：judge: 前缀的完整缓存键。
    """
    context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    return f"judge:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}:{prompt_version}:{model_id}"


def _cache_generator_key(query_id: str, context_str: str) -> str:
    """构造 Generator 缓存键。

    Args:
        query_id: benchmark 条目 query_id。
        context_str: 检索上下文（build_judge_context 产出）。

    Returns:
        str：generator: 前缀的完整缓存键。
    """
    context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    return f"generator:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}"


def get_judge_cache(cache_key: str) -> JudgeResult | None:
    """查 Judge 缓存，命中则记录 metric 并返回反序列化的 JudgeResult。

    Args:
        cache_key: Judge 缓存键（_cache_judge_key 产出）。

    Returns:
        JudgeResult | None：命中返回反序列化结果，否则 None。
    """
    from eval.core.llm_as_judge.judge import JudgeResult
    from obs import get_metrics
    cache = get_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        get_metrics().record_judge_cache_hit()
        return JudgeResult.from_json(cached)
    get_metrics().record_judge_cache_miss()
    return None


def set_judge_cache(cache_key: str, result: JudgeResult) -> None:
    """将 JudgeResult 写入缓存。

    Args:
        cache_key: Judge 缓存键（_cache_judge_key 产出）。
        result: 待写入的 Judge 判定结果。
    """
    cache = get_cache()
    cache.set(cache_key, result.to_json(), ttl_seconds=REDIS_DEFAULT_TTL)
