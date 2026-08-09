"""unit：Judge/Generator 缓存键构造（eval/core/llm_as_judge/judge_cache.py）。"""
from config import GENERATOR_CONFIG_HASH
from eval.core.llm_as_judge.judge_cache import _cache_judge_key, _cache_generator_key


class TestCacheJudgeKey:
    def test_components_present(self):
        key = _cache_judge_key("Q001", "ctx", "v1", "deepseek")
        assert key.startswith("judge:Q001:")
        assert "v1" in key
        assert "deepseek" in key
        assert GENERATOR_CONFIG_HASH in key

    def test_deterministic(self):
        assert _cache_judge_key("Q1", "ctx", "v1", "m") == _cache_judge_key("Q1", "ctx", "v1", "m")

    def test_context_change_invalidates(self):
        assert _cache_judge_key("Q1", "ctxA", "v1", "m") != _cache_judge_key("Q1", "ctxB", "v1", "m")

    def test_model_change_invalidates(self):
        assert _cache_judge_key("Q1", "ctx", "v1", "m1") != _cache_judge_key("Q1", "ctx", "v1", "m2")


class TestCacheGeneratorKey:
    def test_components_present(self):
        key = _cache_generator_key("Q001", "ctx")
        assert key.startswith("generator:Q001:")
        assert GENERATOR_CONFIG_HASH in key

    def test_context_change_invalidates(self):
        assert _cache_generator_key("Q1", "ctxA") != _cache_generator_key("Q1", "ctxB")
