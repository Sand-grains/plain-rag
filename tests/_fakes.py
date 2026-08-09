"""集中 Fake 组件层：隔离重依赖（Embedding 模型 / CrossEncoder / Redis / LLM），供 unit / integration / regression 共用。

核心特性：
    - FakeEmbedding：确定性 hash 向量（8 维 L2 归一化，点积即余弦）；语义分组模式
      （groups: dict[str, list[str]] 子串匹配归组，组间正交 one-hot 向量，组数 ≤ 8）
    - FakeReranker：按 embed 相似度降序取 top_k，永不加载 CrossEncoder
    - FakeCache：内存 dict 实现 infra/cache/backend.py 的 CacheBackend ABC
    - FakeGenerator：固定回答，不触 OpenAI
    - install_all_fakes(monkeypatch, groups=None)：一键替换全部 patch 点位，
      内含 G3 接口自检——getattr 存在性 + inspect.signature 对照真函数，漂移即 AssertionError

patch 点位（关键：按命名空间，非按定义处）：
    - retrieval.embedding.embed + retrieval.retriever.embed（retriever 是绑定导入，须各打各的）
    - retrieval.retriever.Reranker（避免 __init__ 触模型）
    - retrieval.reranker.get_cache / eval.core.llm_as_judge.judge_cache.get_cache / infra.cache.get_cache
    - retrieval.generator.Generator
    - indexing.index_store.STORAGE_BACKEND → "memory"（.env 为 external，性命攸关，必须 autouse）

用法示例::

    from tests._fakes import install_all_fakes

    def test_xxx(monkeypatch):
        install_all_fakes(monkeypatch)                 # 默认确定性 hash
        install_all_fakes(monkeypatch, groups={"rag": ["RAG 检索增强生成"]})  # 语义分组

公共接口：
    - FakeEmbedding / FakeReranker / FakeCache / FakeGenerator
    - install_all_fakes: 一键替换全部重依赖（含接口自检）
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import math

from infra.cache.backend import CacheBackend

_DIM = 8  # 向量维度（组向量为 one-hot，组数须 ≤ _DIM 才能保持正交）


class FakeEmbedding:
    """确定性向量化：默认 hash 8 维 L2 归一化；语义分组模式组间正交。"""

    def __init__(self, groups: dict[str, list[str]] | None = None):
        self._groups = groups or {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本列表 → 归一化向量列表（与真实 embed 同签名语义）。"""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """单文本向量化：先查语义分组（子串匹配），未命中回退确定性 hash。"""
        group_vector = self._match_group(text)
        if group_vector is not None:
            return group_vector
        return self._hash_vector(text)

    def _match_group(self, text: str) -> list[float] | None:
        """子串匹配语义分组：命中某组关键词 → 该组 one-hot 正交向量。

        Args:
            text: 待向量化文本。

        Returns:
            list[float] | None：命中组返回该组单位向量；未命中返回 None。
        """
        for group_index, keywords in enumerate(self._groups.values()):
            if any(keyword in text for keyword in keywords):
                vector = [0.0] * _DIM
                vector[group_index % _DIM] = 1.0
                return vector
        return None

    def _hash_vector(self, text: str) -> list[float]:
        """确定性 hash 向量：sha256 取 8 字节 → 归一化，同文本恒同向量。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [float(digest[index % len(digest)]) for index in range(_DIM)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector = [1.0] + [0.0] * (_DIM - 1)
            norm = 1.0
        return [value / norm for value in vector]


class FakeReranker:
    """重排器 Fake：按 query 与父块内容的 embed 相似度降序取 top_k，不加载 CrossEncoder。"""

    def __init__(self, embedding: FakeEmbedding | None = None):
        self._embedding = embedding if embedding is not None else FakeEmbedding()

    def rerank(self, query: str, parent_chunks: list, top_k: int,
               corpus_signature: str = "", skip_cache: bool = False) -> list:
        """按相似度降序重排，返回前 top_k 块（签名对齐真实 Reranker.rerank）。"""
        if len(parent_chunks) <= 1:
            return parent_chunks[:top_k]
        query_vector = self._embedding.embed([query])[0]
        scored = []
        for chunk in parent_chunks:
            chunk_vector = self._embedding.embed([chunk.content])[0]
            score = sum(query_value * chunk_value
                        for query_value, chunk_value in zip(query_vector, chunk_vector))
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class FakeCache(CacheBackend):
    """内存缓存后端：实现 infra/cache/backend.py 的 CacheBackend ABC，替代 Redis/Noop。"""

    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._data[key] = value  # ttl 忽略（测试生命周期内不过期）

    def delete_by_pattern(self, pattern: str) -> int:
        prefix = pattern.split("*")[0]
        matched = [key for key in self._data if key.startswith(prefix)]
        for key in matched:
            del self._data[key]
        return len(matched)

    def clear(self) -> None:
        """测试辅助：清空全部缓存条目。"""
        self._data.clear()


class FakeGenerator:
    """生成器 Fake：固定回答（不触 OpenAI）。"""

    def generate(self, query: str, context_chunks: list, temperature: float = 0.0) -> str:
        """返回固定回答（含 query，便于断言）。"""
        return f"固定回答：{query}"


def _make_get_cache(cache: FakeCache):
    """构造返回指定 FakeCache 实例的 get_cache 闭包（签名无参，对齐真实 get_cache）。"""
    def _get_cache():
        return cache
    return _get_cache


_REAL_CACHE: dict[tuple[str, str], object] = {}


def _real(module_path: str, attribute: str):
    """取真接口引用（模块级缓存，防二次 install 读到 patch 后的值）。"""
    key = (module_path, attribute)
    if key not in _REAL_CACHE:
        try:
            _REAL_CACHE[key] = getattr(importlib.import_module(module_path), attribute)
        except AttributeError as error:
            raise AssertionError(
                f"[_fakes] 接口漂移：生产代码 {module_path}.{attribute} 不存在（Fake 层需同步更新）"
            ) from error
    return _REAL_CACHE[key]


def _check_signature(name: str, real_callable, fake_callable) -> None:
    """G3 接口自检：inspect.signature 对照真函数（参数名/数量），漂移即 AssertionError。"""
    real_params = list(inspect.signature(real_callable).parameters.values())
    fake_params = list(inspect.signature(fake_callable).parameters.values())
    if len(real_params) != len(fake_params):
        raise AssertionError(
            f"[_fakes] 接口漂移 {name}: 参数数量 {len(real_params)} != {len(fake_params)}")
    for real_param, fake_param in zip(real_params, fake_params):
        if real_param.name != fake_param.name:
            raise AssertionError(
                f"[_fakes] 接口漂移 {name}: 参数名 {real_param.name} != {fake_param.name}")


def install_all_fakes(monkeypatch, groups: dict[str, list[str]] | None = None) -> None:
    """一键替换全部重依赖（含 G3 接口自检），monkeypatch teardown 自动 restore。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        groups: 语义分组（dict[str, list[str]]），None 为默认确定性 hash 模式。
    """
    embedding = FakeEmbedding(groups=groups)
    cache = FakeCache()
    reranker = FakeReranker(embedding)
    generator = FakeGenerator()

    config_module = importlib.import_module("config")
    embedding_module = importlib.import_module("retrieval.embedding")
    retriever_module = importlib.import_module("retrieval.retriever")
    reranker_module = importlib.import_module("retrieval.reranker")
    generator_module = importlib.import_module("retrieval.generator")
    judge_cache_module = importlib.import_module("eval.core.llm_as_judge.judge_cache")
    infra_cache_module = importlib.import_module("infra.cache")
    index_store_module = importlib.import_module("indexing.index_store")

    # ---- G3 接口自检：存在性 + 签名（对照真接口缓存，非 patch 后的值）----
    real_embed = _real("retrieval.embedding", "embed")
    real_retriever_embed = _real("retrieval.retriever", "embed")
    real_reranker_class = _real("retrieval.retriever", "Reranker")
    real_rerank_get_cache = _real("retrieval.reranker", "get_cache")
    real_judge_get_cache = _real("eval.core.llm_as_judge.judge_cache", "get_cache")
    real_infra_get_cache = _real("infra.cache", "get_cache")
    real_generator_class = _real("retrieval.generator", "Generator")
    _real("indexing.index_store", "STORAGE_BACKEND")  # 存在性自检

    _check_signature("retrieval.embedding.embed", real_embed, embedding.embed)
    _check_signature("retrieval.retriever.embed", real_retriever_embed, embedding.embed)
    _check_signature("Reranker.rerank", real_reranker_class.rerank, FakeReranker.rerank)
    _check_signature("get_cache", real_rerank_get_cache, _make_get_cache(cache))
    _check_signature("get_cache", real_judge_get_cache, _make_get_cache(cache))
    _check_signature("get_cache", real_infra_get_cache, _make_get_cache(cache))
    _check_signature("Generator.generate", real_generator_class.generate, FakeGenerator.generate)

    # ---- 安装（按命名空间）----
    monkeypatch.setattr(embedding_module, "embed", embedding.embed)
    monkeypatch.setattr(retriever_module, "embed", embedding.embed)
    monkeypatch.setattr(retriever_module, "Reranker", lambda: reranker)
    monkeypatch.setattr(reranker_module, "get_cache", _make_get_cache(cache))
    monkeypatch.setattr(judge_cache_module, "get_cache", _make_get_cache(cache))
    monkeypatch.setattr(infra_cache_module, "get_cache", _make_get_cache(cache))
    monkeypatch.setattr(generator_module, "Generator", lambda: generator)
    # STORAGE_BACKEND 性命攸关：.env 为 external，IndexStore/vector_restore 均读此处，须强制 memory
    monkeypatch.setattr(index_store_module, "STORAGE_BACKEND", "memory")
    monkeypatch.setattr(config_module, "STORAGE_BACKEND", "memory")
