"""CrossEncoder 重排器: 对 RRF 候选父块 pair-wise 打分重排。

核心特性：
    - 模块级懒加载单例 _model（仿 retrieval/embedding.py），进程内只加载一次模型
    - 降级保序：模型加载 / predict / 缓存反序列化任一异常 → 返回原 RRF 序（parent_chunks[:top_k]），
      检索主链路永不因 rerank 崩溃；缓存 set 失败静默吞掉（best-effort）
    - 缓存 content-addressable：key = query 哈希 + 模型版本指纹 + 语料签名 + enrichment 变体，
      不含 top_k / candidate_ids（候选池与 top_k 无关，key 去 top_k 缓存全池 score_map、调用按 k 切片）
    - 缓存 hit 边界：只对当前候选集合查分（缺分置 0.0 排尾部），集合外旧条目忽略、不回流 top_k（防 stale chunk_id 泄漏）
    - skip_cache：跳过查/写缓存直接 predict，供 auto-merge 校准消除跨 boost 候选集漂移
    - 并发安全：模块级 _model_lock 守护模块级单例 _model（锁与被保护对象同级）
    - 模块级计数 _rerank_cache_statistics，供 eval MonitorMetrics 聚合（保住"MonitorMetrics 唯一真源"约定）

用法示例::

    from retrieval.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query, candidate_chunks, top_k=5, corpus_signature="ab12cd34")

公共接口：
    - Reranker: 重排器（rerank 方法，构造廉价、模型懒加载）
    - get_rerank_cache_statistics: 模块级缓存计数读取（hits / misses）
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from config import (
    RERANKER_MODEL_PATH, RERANKER_CACHE_TTL, RERANKER_ENRICHMENT,
    RERANKER_BATCH_SIZE, RERANKER_FP16, RERANKER_MAX_LENGTH,
)
from sentence_transformers import CrossEncoder
import torch
from infra.cache import get_cache
from indexing.chunk import Chunk

_model: CrossEncoder | None = None
_model_lock = threading.Lock()  # 模块级锁守护模块级单例 _model（锁与被保护对象同级）

_rerank_cache_statistics = {"hits": 0, "misses": 0}

_model_version_cache: str | None = None  # 进程内 memoize，避免同进程多次流式读 2.27GB


def _get_model() -> CrossEncoder:
    """懒加载 CrossEncoder 模型：首次调用时加载，后续复用（模块级单例）。

    Returns:
        CrossEncoder：进程内唯一的重排模型实例。
    """
    global _model
    if _model is None:
        model_kwargs = {"torch_dtype": torch.float16} if (RERANKER_FP16 and torch.cuda.is_available()) else {}
        _model = CrossEncoder(RERANKER_MODEL_PATH, max_length=RERANKER_MAX_LENGTH, model_kwargs=model_kwargs)
    return _model


def _sha256_head(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 sha256 前 8 位（分块读，避免整文件载入内存）。

    Args:
        path: 待计算的文件路径。
        chunk_size: 读块大小（字节），默认 1MB。

    Returns:
        str：8 位 sha256 前缀。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


def _model_version() -> str:
    """模型版本指纹：model.safetensors sha256 前 8 位（缓存 key 组件，模型更换即全量失效）。

    首次计算后写 model.safetensors.sha256 落盘于模型旁，后续进程直接读（免每进程流式读 2.27GB）；
    写盘 try/except（模型目录在项目外 D:/Model/... 可能不可写），失败仅进程内 memoize，退回每进程流式读。

    Returns:
        str：8 位 sha256 前缀；safetensors 缺失时返回 "missing"。
    """
    global _model_version_cache
    if _model_version_cache is not None:
        return _model_version_cache
    model_dir = Path(RERANKER_MODEL_PATH)
    safetensors = model_dir / "model.safetensors"
    fingerprint_file = model_dir / "model.safetensors.sha256"
    if not safetensors.exists():
        _model_version_cache = "missing"
        return _model_version_cache
    if fingerprint_file.exists():
        # 换模型须删 fingerprint_file——指纹文件缓存了旧模型版本，不删则缓存键永不过期（旧分数持续命中）
        _model_version_cache = fingerprint_file.read_text(encoding="utf-8").strip()[:8]
        return _model_version_cache
    version = _sha256_head(safetensors)
    try:
        fingerprint_file.write_text(version, encoding="utf-8")
    except OSError:
        logging.warning("无法写模型指纹落盘（%s），本进程退回流式读 2.27GB", fingerprint_file)
    _model_version_cache = version
    return _model_version_cache


def _build_pair(query: str, chunk: Chunk) -> tuple[str, str]:
    """构造 CrossEncoder 打分对。

    RERANKER_ENRICHMENT=1 时拼 [{title}][{section_path}] {content}（title 取 origin_metadata.title，
    section_path 取 metadata.get("section_path")，均缺省回退 chunk.doc_id）; =0 时维持 (query, chunk.content)。
    enrichment 只改 rerank 输入，不改 embedding / 检索文本。

    Args:
        query: 查询文本。
        chunk: 待打分的父块。

    Returns:
        tuple[str, str]: (query, 构造后的文档文本) pair。
    """
    if not RERANKER_ENRICHMENT:
        return query, chunk.content
    title = chunk.origin_metadata.title or chunk.doc_id
    section_path = chunk.metadata.get("section_path") or chunk.doc_id
    return query, f"[{title}][{section_path}] {chunk.content}"


def _cache_key(query: str, corpus_signature: str) -> str:
    """构造 rerank 缓存键。

    候选池与 top_k 无关（candidate_k=RERANKER_TOP_K 恒定），key 去 top_k 后 k=5/k=10 共享同一缓存；
    不含 candidate_ids（容忍 HNSW 候选波动）。`e` 标记 enrichment 变体，防基线与 A/B 串缓存。

    Args:
        query: 查询文本。
        corpus_signature: 语料签名（当前 store 里全部可检索父块内容哈希前 12 位）。

    Returns:
        str：完整缓存键（不含 Redis key_prefix 命名空间）。
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    enrichment_flag = "e" if RERANKER_ENRICHMENT else ""
    return f"rerank:{query_hash}:{_model_version()}:{corpus_signature}:{enrichment_flag}"


def get_rerank_cache_statistics() -> dict[str, int]:
    """读取模块级 rerank 缓存计数（供 eval MonitorMetrics 聚合 rerank_cache_hit_rate）。

    Returns:
        dict[str, int]：{"hits": N, "misses": N}。
    """
    return dict(_rerank_cache_statistics)


class Reranker:
    """CrossEncoder 重排器：对候选父块 pair 打分 → 按分降序取 top_k。

    模型懒加载(构造廉价), 进程内可安全持有多个实例共享同一模块级 _model（锁加在模块级）。
    """

    def rerank(self, query: str, parent_chunks: list[Chunk], top_k: int,
               corpus_signature: str = "", skip_cache: bool = False) -> list[Chunk]:
        """对候选父块重排：返回按 CrossEncoder 分数降序的前 top_k 块。

        Args:
            query: 查询文本。
            parent_chunks: RRF 融合后的候选父块列表（传入序即原 RRF 序）。
            top_k: 返回的最大条数。
            corpus_signature: 语料签名（作为缓存 key 组件, skip_cache 时不使用）。
            skip_cache: True 时跳过查/写缓存直接 predict（calibrate 专用）。

        Returns:
            list[Chunk]：按重排分数降序的前 top_k 块；异常时降级返回 parent_chunks[:top_k]（原 RRF 序）。
        """
        if len(parent_chunks) <= 1:  # 空/单候选直接返回（无可排性）
            return parent_chunks[:top_k]
        try:
            score_map = self._score_map(query, parent_chunks, corpus_signature, skip_cache)
        except Exception as exception:
            # 降级保序：任一异常 → 直接返回原 RRF 后的parent_chunks(检索主链路不因 reranker 而崩溃)
            logging.warning("reranker 异常，降级为原 RRF :%s", exception)
            return parent_chunks[:top_k]
        reranked_chunks = sorted(parent_chunks, key=lambda chunk: score_map.get(chunk.chunk_id, 0.0), reverse=True)
        return reranked_chunks[:top_k]

    def _score_map(self, query: str, parent_chunks: list[Chunk],
                   corpus_signature: str, skip_cache: bool) -> dict[str, float]:
        """取当前候选集的 chunk_id 映射为一个分数表。 chunk_id与打分分数一一对应

        skip_cache 时直接 predict: 否则先查缓存，命中则只对当前候选集合查分（集合外旧条目忽略），
        miss 则 predict 后写缓存。

        Args:
            query: 查询文本。
            parent_chunks: 当前候选父块列表。
            corpus_signature: 语料签名。
            skip_cache: 是否跳过缓存。

        Returns:
            dict[str, float]：候选 chunk_id → 重排分数（当前候选缺分时调用方按 0.0 排尾部）。
        """
        if skip_cache: # 如果跳过缓存, 直接全部重新打分
            return self._score_uncached(query, parent_chunks)
        key = _cache_key(query, corpus_signature)
        cached = self._read_cache(key)
        if cached is not None:
            _rerank_cache_statistics["hits"] += 1
            candidate_ids = {chunk.chunk_id for chunk in parent_chunks}
            return {chunk_id: score for chunk_id, score in cached.items() if chunk_id in candidate_ids}
        _rerank_cache_statistics["misses"] += 1
        score_map = self._score_uncached(query, parent_chunks)
        self._write_cache(key, score_map)
        return score_map

    def _score_uncached(self, query: str, parent_chunks: list[Chunk]) -> dict[str, float]:
        """批量 pair 打分（模块级锁串行化，守护共享 _model）。

        Args:
            query: 查询文本。
            parent_chunks: 候选父块列表。

        Returns:
            dict[str, float]：chunk_id → CrossEncoder 分数。
        """
        pairs = [_build_pair(query, chunk) for chunk in parent_chunks]
        with _model_lock:  # 模块级锁, 守护模块级单例 _model
            scores = _get_model().predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
        return {chunk.chunk_id: float(score) for chunk, score in zip(parent_chunks, scores)}

    def _read_cache(self, key: str) -> dict[str, float] | None:
        """读缓存 score_map；反序列化失败抛异常 → 外层降级保序（缓存腐坏不崩主链路）。

        Args:
            key: 缓存键。

        Returns:
            dict[str, float] | None：解析出的 score_map；键不存在时返回 None。
        """
        raw = get_cache().get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def _write_cache(self, key: str, score_map: dict[str, float]) -> None:
        """写缓存（best-effort）：失败静默吞掉，不影响主链路。

        Args:
            key: 缓存键。
            score_map: 待写入的 chunk_id → 分数表。
        """
        try:
            get_cache().set(key, json.dumps(score_map), ttl_seconds=RERANKER_CACHE_TTL)
        except Exception:
            pass  # best-effort：缓存写失败静默吞掉
