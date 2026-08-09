"""双路检索 + RRF 融合（父块层）+ 精排（CrossEncoder 重排 + auto-merge 后处理）。

核心特性：
    - 稠密检索：BGE-M3 embedding → 余弦相似度，返回**子块**（含 parent_id 与 dense_score）
    - 稀疏检索：jieba/IK 分词 → BM25 打分，返回**父块**
    - RRF (Reciprocal Rank Fusion, k=60) 在**父块层**融合：key = parent_id or chunk_id
    - 融合后 index_store.get_parents(top_ids) 统一落到父块（检索/benchmark 单元）
    - 候选池 candidate_k = RERANKER_POOL_K if > 0 else top_k * 2（池大小与 rerank 开关解耦，控制组可比）
    - 重排：RRF top N → CrossEncoder 打分 → top_k（rerank_enabled 实例级门控，agent 关闭）
    - auto-merge：同一父块多个**非相邻**子块命中 → RRF 分数 *= MERGE_BOOST（动态读 config，calibrate 可覆盖）
    - 降级保序：rerank 任何异常 → 返回原 RRF 序，检索主链路永不因 rerank 崩溃
    - skip_cache 实例属性：calibrate 传 rerank_skip_cache=True 贯穿到 rerank()，消除跨 boost 候选集漂移

用法示例::

    from retrieval import Retriever
    from indexing.index_store import IndexStore
    index_store = IndexStore.vector_restore()
    retriever = Retriever(index_store)
    chunks = retriever.retrieve("什么是 RAG", top_k=5)

公共接口：
    - Retriever: 双路检索器，持有 IndexStore 引用
"""
import hashlib

import config
from config import RRF_K, TOP_K
from indexing.chunk import Chunk
from indexing.index_store import IndexStore, _chunk_seq
from retrieval.embedding import embed
from retrieval.reranker import Reranker


def _parent_key(chunk: Chunk) -> str:
    """RRF 融合键：子块按 parent_id 归并到父块，flat_simple 无 parent_id → 自父。

    Args:
        chunk: 待归并的检索结果 Chunk（子块或父块）。

    Returns:
        str：父块 chunk_id。子块取 metadata.parent_id；无 parent_id（flat_simple）
        时取自身 chunk_id。
    """
    return chunk.metadata.get("parent_id") or chunk.chunk_id


def _corpus_signature(store: IndexStore) -> str:
    """一次性父块内容强哈希（rerank 缓存 key 的语料签名）。

    取 store.chunks 中全部父块 (chunk_id, content)，按 chunk_id 排序拼 f"{cid}:{content}" 用 | 连接 →
    sha256 前 12 位。必须按 chunk_id 排序否则依赖 dict 插入序、签名不稳定；任何父块文本变化 → 全量失效，
    匹配"重索引是唯一语料变更事件"。懒计算（首个 rerank 调用才调用，见 retrieve_with_dense_child）。

    为什么不放进reranker.py中? 因为reranker收到的候选是别人喂给它的, reranker只作为工具, 不知道、也不该知道"语料怎么样, 语料有多大"
    另外, retriever 持有 index_store 作为实际父块数据源

    Args:
        store: 索引存储门面，枚举父块用。

    Returns:
        str：12 位 sha256 前缀。
    """
    pairs = sorted((chunk.chunk_id, chunk.content) for chunk in store.chunks)
    payload = "|".join(f"{chunk_id}:{content}" for chunk_id, content in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _apply_auto_merge(rrf_scores: dict[str, float], dense_results: list[Chunk],
                      boost: float, min_hits: int) -> dict[str, float]:
    """auto-merge 后处理：若同一父块多个非相邻子块命中 → 该父块 RRF 分数 *= boost。

    按 _parent_key 分组**完整** dense_results（保留 auto-merge 作用面）→ 组内按 dense_score 降序 →
    依次保留"与所有已保留块 _chunk_seq 差 ≠ 1"的子块（相邻且分低的丢弃）→ 保留数 ≥ min_hits 则该父块 *= boost。
    只读不修改 dense_results（eval 子块层指标信号源不受影响）。不加 score 下限
    （dense top-k 命中分数全 ≥0.47，下限无约束力）。只作用于 RRF 池内父块（池外无法推进）。

    Args:
        rrf_scores: RRF 融合分数表（父块 key → 分数），原地修改。
        dense_results: dense 检索子块列表（含 metadata.dense_score 与 parent_id）。
        boost: 达标父块的分数倍增系数。
        min_hits: 触发所需的最小非相邻子块命中数。

    Returns:
        dict[str, float]：boost 应用后的 rrf_scores（原对象）。
    """
    by_parent: dict[str, list[Chunk]] = {}
    for chunk in dense_results:
        parent_id = _parent_key(chunk)
        by_parent.setdefault(parent_id, []).append(chunk)
    for parent_id, hits in by_parent.items():
        if parent_id not in rrf_scores:  # 只作用于 RRF 池内父块，池外不推进
            continue
        hits.sort(key=lambda chunk: chunk.metadata.get("dense_score", 0.0), reverse=True)
        kept: list[Chunk] = []
        for hit in hits:
            hit_seq = _chunk_seq(hit.chunk_id)
            if all(abs(hit_seq - _chunk_seq(kept_chunk.chunk_id)) != 1 for kept_chunk in kept):
                kept.append(hit)
        if len(kept) >= min_hits:
            rrf_scores[parent_id] *= boost
    return rrf_scores


class Retriever:
    """检索层：双路检索（稠密子块 + 稀疏父块） → 父块层 RRF 融合 → [auto-merge → CrossEncoder 重排] → top_k 父块。"""

    def __init__(self, store: IndexStore, rerank_enabled: bool = config.RERANKER_ENABLED,
                 rerank_skip_cache: bool = False):
        """构造检索器。

        Args:
            store: 索引存储门面。
            rerank_enabled: 是否启用 Reranker(CrossEncoder) 重排。
            rerank_skip_cache: 是否跳过 rerank 缓存（calibrate 传 True，消除跨 boost 候选集漂移）。
        """
        self.index_store = store
        self._rerank_enabled = rerank_enabled
        self._rerank_skip_cache = rerank_skip_cache
        self._reranker = Reranker()  # 构造廉价，模型懒加载
        self._corpus_signature: str | None = None  # 懒计算：首个 rerank 调用才哈希（rerank 关闭零开销）

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """单次检索：返回 top_k 个父块 Chunk（RRF 融合 + 可选重排后的最终结果）。

        Args:
            query: 查询文本。
            top_k: 返回的父块最大条数。

        Returns:
            list[Chunk]：按分数降序的父块列表。
        """
        parents, _ = self.retrieve_with_dense_child(query, top_k)
        return parents

    def retrieve_with_dense_child(self, query: str, top_k: int = TOP_K) -> tuple[list[Chunk], list[Chunk]]:
        """混合检索 → RRF 融合 → [auto-merge → CrossEncoder 重排] → 返回(父块 top_k, dense 子块候选)。

        dense 子块候选供 child 层指标/诊断使用（auto-merge 只读信号源，不修改）。

        这个方法主要用于eval模块,
        保留dense_child是因为测评时要同时衡量两套不同粒度的指标, 评测的子块层指标必须要这份RRF融合前数据

        Args:
            query: 查询文本。
            top_k: 返回的父块最大条数。

        Returns:
            tuple[list[Chunk], list[Chunk]]：(父块 top_k, dense 子块候选列表)。
            子块候选长度 = candidate_k（rerank 开时 RERANKER_POOL_K，关时 top_k*2）。
        """
        query_vector = embed([query])[0]
        candidate_k = config.RERANKER_POOL_K if config.RERANKER_POOL_K > 0 else top_k * 2  # 池大小与 rerank 开关解耦

        dense_results = self.index_store.search_dense(query_vector, top_k=candidate_k)
        sparse_results = self.index_store.search_sparse(query, top_k=candidate_k)

        # RRF 融合：score(id) = 1/(k + rank_dense) + 1/(k + rank_sparse)，key 统一到父块
        rrf_scores = {}
        for rank, chunk in enumerate(dense_results, start=1):
            key = _parent_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk in enumerate(sparse_results, start=1):
            key = _parent_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

        if not self._rerank_enabled:  # 基线/agent/控制组路径，原样返回（实例级门控）
            sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
            return self.index_store.get_parents(sorted_ids), dense_results

        # auto-merge：动态读 config.MERGE_BOOST（calibrate 进程内覆盖生效）
        if config.MERGE_BOOST > 0.0:
            rrf_scores = _apply_auto_merge(rrf_scores, dense_results,
                                           config.MERGE_BOOST, config.MERGE_MIN_CHILD_HITS)

        # 选 RRF top N → CrossEncoder 重排 → top_k
        rerank_n = min(config.RERANKER_TOP_K, len(rrf_scores))
        candidate_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:rerank_n]
        candidate_chunks = self.index_store.get_parents(candidate_ids)
        if self._corpus_signature is None:  # 懒计算：rerank 关闭路径从不触达（agent/基线/控制组零开销）
            self._corpus_signature = _corpus_signature(self.index_store)  # 并发 double-compute 幂等无害
        reranked = self._reranker.rerank(query, candidate_chunks, top_k,
                                         corpus_signature=self._corpus_signature,
                                         skip_cache=self._rerank_skip_cache)  # skip_cache 贯穿 calibrate → retriever → rerank
        return reranked, dense_results
