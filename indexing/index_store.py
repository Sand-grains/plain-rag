"""索引存储: 按 STORAGE_BACKEND 分流 memory 或 external（PgSQL+ES+Milvus）两种后端。
索引 父/子块本体 + 稠密向量矩阵 + BM25倒排结构(对应_parents / _children / _vectors / _bm25这些索引数据)

对外暴露统一接口（batch_add / search_dense / search_sparse / get_parents 等），
上层 retriever / agent_pipeline / eval.runner 不感知后端差异。

父子语义：检索/benchmark 单元为父块。
- 父块存 PgSQL/ES（全文），子块存 Milvus（向量 + parent_id）
- search_dense 返回子块（含 metadata.parent_id）
- search_sparse 返回父块
- get_parents 按 parent_id 取父块（RRF 融合后统一落到父块）
"""
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import jieba
import numpy
from rank_bm25 import BM25Okapi

from config import STORAGE_BACKEND, TOP_K, VECTOR_CACHE_DIR
from indexing.chunk import Chunk, DocMetadata
from indexing.chunk_ingest_ex import cleanup_suspending, ingest_doc
from obs.trace_decorators import observe_stage

logger = logging.getLogger(__name__)

_MEMORY_FORMAT_VERSION = 1

def _chunk_seq(chunk_id: str) -> int:
    """从 chunk_id 末段解析序号（:p{i}→i / :{i}→i / :c{j}→j），用于稳定排序。

    Args:
        chunk_id: 分块 ID。形如 {doc_id}:p{i} / {doc_id}:p{i}:c{j} / {doc_id}:{i}。

    Returns:
        int：末段拼接出的数字序号；末段无数字时返回 0。
    """
    tail = chunk_id.rsplit(":", 1)[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return int(digits) if digits else 0


class IndexStore:
    """向量存储对外统一接口
    按 STORAGE_BACKEND 分流 memory/external 两种实现, 对上层屏蔽后端差异"""

    def __init__(self):
        self._backend = STORAGE_BACKEND
        if self._backend == "external":
            self._init_external()
        else:
            self._init_memory()

    # ---- memory 模式 ----

    def _init_memory(self) -> None:
        """初始化 memory 后端的全部容器：父块 / 子块 / 向量矩阵 / BM25 索引。"""
        self._parents: list[Chunk] = []                 # 父块（检索/benchmark 单元，BM25 对齐）
        self._parent_ids: dict[str, Chunk] = {}         # chunk_id → 父块（按 id 快速取）
        self._children: list[Chunk] = []                # 子块（稠密向量对齐 numpy）
        self._vectors: numpy.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_tokenized: list[list[str]] = []

    # ---- external 模式 ----

    def _init_external(self) -> None:
        """连接三库并确保 schema 就绪
        进程启动时，把 PgSQL 里已 indexed 的父块重新读回内存的 _chunks_cache 字典 (回填父块缓存)
        启动即调用一次 cleanup_suspending 清理历史残留的 suspending 记录。
        """
        from infra.db.postgres import PgSQLClient
        from infra.search.es import ESClient
        from infra.vector.milvus import MilvusClientWrapper
        self._pgsql = PgSQLClient()
        self._es = ESClient()
        self._milvus = MilvusClientWrapper()
        self._pgsql.ensure_table()
        self._es.create_index()
        self._milvus.ensure_collection()
        cleanup_suspending(self._pgsql, self._es, self._milvus)
        self._chunks_cache: dict[str, Chunk] = {}       # 父块 id → Chunk
        self._load_cache_from_pgsql()

    def _load_cache_from_pgsql(self) -> None:
        """从 PgSQL 回填父块缓存，含 origin_metadata（doc_title/doc_type/chunk_level）。

        父块元数据分两段落回 Chunk：文档级字段进 origin_metadata，其余进 metadata。
        """
        cursor = self._pgsql._conn.cursor()
        cursor.execute(
            "SELECT chunk_id, doc_id, content, metadata FROM parent_chunks WHERE status = 'indexed'"
        )
        for chunk_id, doc_id, content, metadata in cursor.fetchall():
            metadata = metadata or {}
            origin_metadata = DocMetadata(
                title=metadata.get("doc_title", ""),
                doc_type=metadata.get("doc_type", ""),
                chunk_level=metadata.get("chunk_level", "parent"),  # parent_chunks 表只存父块，缺省按父块处理
            )
            chunk_metadata = {
                key: value for key, value in metadata.items() if key not in ("doc_title", "doc_type", "chunk_level")
            }
            self._chunks_cache[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                origin_metadata=origin_metadata,
                metadata=chunk_metadata,
            )
        cursor.close()

    # ---- 公共属性（父块集合） ----

    @property
    def chunk_ids(self) -> set[str]:
        """当前已索引的父块 chunk_id 集合（benchmark 校验用）。"""
        if self._backend == "external":
            return set(self._chunks_cache.keys())
        return {chunk.chunk_id for chunk in self._parents}

    @property
    def chunks(self) -> list[Chunk]:
        """当前已索引的父块列表（benchmark / anno_tool 枚举父块用）。"""
        if self._backend == "external":
            return list(self._chunks_cache.values())
        return list(self._parents)

    # ---- 公共方法 ----

    def batch_add(self, parents: list[Chunk], children: list[Chunk], child_vectors: list[list[float]]) -> None:
        """批量摄入一批父块 + 子块 + 子块向量，按后端分流写入。

        Args:
            parents: 父块列表（external 写入 PgSQL/ES，memory 直接进内存）。
            children: 子块列表（external 写入 Milvus）。
            child_vectors: 与 children 一一对应的稠密向量。
        """
        if self._backend == "external":
            self._batch_add_external(parents, children, child_vectors)
        else:
            self._batch_add_memory(parents, children, child_vectors)

    def get_parents(self, parent_ids: list[str]) -> list[Chunk]:
        """按 parent_id 取父块，保持传入顺序。

        Args:
            parent_ids: 待取的父块 chunk_id 列表（RRF 融合后的 top 结果）。

        Returns:
            list[Chunk]：命中缓存的父块，未命中 id 直接跳过。
        """
        if self._backend == "external":
            return [self._chunks_cache[chunk_id] for chunk_id in parent_ids if chunk_id in self._chunks_cache]
        return [self._parent_ids[chunk_id] for chunk_id in parent_ids if chunk_id in self._parent_ids]

    def get_children(self, parent_id: str) -> list[Chunk]:
        """按 parent_id 取子块（anno_tool 证据子块标注用），按子块序号排序。

        flat_simple 分块无 parent_id 映射 → 返回空列表。

        Args:
            parent_id: 父块 chunk_id。

        Returns:
            list[Chunk]：该父块下的全部子块，按序号升序；无子块时为空列表。
        """
        if self._backend == "external":
            children = self._get_children_external(parent_id)
        else:
            children = [chunk for chunk in self._children if chunk.metadata.get("parent_id") == parent_id]
        return sorted(children, key=lambda chunk: _chunk_seq(chunk.chunk_id))

    @observe_stage("dense")
    def search_dense(self, query_vector: list[float], top_k: int = TOP_K) -> list[Chunk]:
        """稠密检索：返回子块 Chunk（含 metadata.parent_id）。

        Args:
            query_vector: 查询文本的稠密向量（BGE-M3 产出）。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：按相似度降序的子块列表。
        """
        if self._backend == "external":
            return self._search_dense_external(query_vector, top_k)
        return self._search_dense_memory(query_vector, top_k)

    @observe_stage("sparse")
    def search_sparse(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """稀疏检索：返回父块 Chunk。

        Args:
            query: 查询文本（memory 用 jieba 分词，external 用 IK 分词器）。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：按 BM25 分数降序的父块列表。
        """
        if self._backend == "external":
            return self._search_sparse_external(query, top_k)
        return self._search_sparse_memory(query, top_k)

    def vector_persistence(self, cache_dir: str = VECTOR_CACHE_DIR) -> None:
        """持久化 memory 后端数据到磁盘（external 模式无操作）。

        Args:
            cache_dir: 缓存目录路径。
        """
        if self._backend == "external":
            return  # 外部模式无需 pickle/npy 持久化，数据已在数据库中
        self._persistence_memory(cache_dir)

    @classmethod
    def vector_restore(cls, cache_dir: str = VECTOR_CACHE_DIR) -> "IndexStore | None":
        """从缓存恢复 index_store 实例。

        Args:
            cache_dir: 缓存目录路径。

        Returns:
            IndexStore | None：memory 模式缓存缺失/损坏时返回 None（触发重索引）；
            external 模式直接连接三库返回新实例。
        """
        if STORAGE_BACKEND == "external":
            return cls()  # __init__ 自动连接外部服务
        return cls._restore_memory(cache_dir)

    # ---- memory 实现 ----

    def _batch_add_memory(self, parents: list[Chunk], children: list[Chunk], child_vectors: list[list[float]]) -> None:
        """内存追加父块/子块，重建向量矩阵与 BM25 索引。

        Args:
            parents: 父块列表。
            children: 子块列表。
            child_vectors: 与 children 一一对应的稠密向量。
        """
        if not parents or not children:
            return  # 空文档分块为零，跳过（numpy 空数组会让 vstack 维度塌缩）
        self._parents.extend(parents)
        for parent in parents:
            self._parent_ids[parent.chunk_id] = parent
        self._children.extend(children)
        new_vectors = numpy.array(child_vectors)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = numpy.vstack([self._vectors, new_vectors])
        self._bm25_tokenized = [list(jieba.cut(chunk.content)) for chunk in self._parents]
        self._bm25 = BM25Okapi(self._bm25_tokenized)

    def _search_dense_memory(self, query_vector: list[float], top_k: int = TOP_K) -> list[Chunk]:
        """Memory模式稠密检索：numpy 点积 + argpartition 取 top_k 子块。

        Args:
            query_vector: 查询文本的稠密向量。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：按相似度降序的子块列表；向量未初始化时为空列表。
        """
        if self._vectors is None:
            return []
        query = numpy.array(query_vector)
        scores = numpy.dot(self._vectors, query)
        top_k = min(top_k, len(scores))
        kth_index = len(scores) - top_k
        partitioned = numpy.argpartition(scores, kth_index)
        top_indices = partitioned[kth_index:]
        top_scores = scores[top_indices]
        sorted_order = numpy.argsort(top_scores)
        top_indices = top_indices[numpy.flip(sorted_order)]
        return [
            replace(self._children[index], metadata={**self._children[index].metadata, "dense_score": float(scores[index])})
            for index in top_indices
        ]

    def _search_sparse_memory(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """Memory模式稀疏检索：jieba 分词 → BM25Okapi 打分取 top_k 父块。

        Args:
            query: 查询文本。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：按 BM25 分数降序的父块列表；BM25 未构建时为空列表。
        """
        if self._bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_k = min(top_k, len(scores))
        top_indices = numpy.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[numpy.argsort(scores[top_indices])[::-1]]
        return [self._parents[index] for index in top_indices]

    def _persistence_memory(self, cache_dir: str = VECTOR_CACHE_DIR) -> None:
        """将子块/父块序列化为 pickle、向量存为 npy，供下次启动恢复。

        Args:
            cache_dir: 缓存目录路径。
        """
        path = Path(cache_dir)
        path.mkdir(exist_ok=True)
        with open(path / "chunks.pkl", "wb") as file_handle:
            pickle.dump((_MEMORY_FORMAT_VERSION, self._children, self._parents), file_handle)
        numpy.save(path / "vectors.npy", self._vectors)

    @classmethod
    def _restore_memory(cls, cache_dir: str = VECTOR_CACHE_DIR) -> "IndexStore | None":
        """从缓存目录恢复 memory 后端数据。

        Args:
            cache_dir: 缓存目录路径。

        Returns:
            IndexStore | None：恢复成功的 index_store 实例；缓存缺失或格式不符
            或解析失败时返回 None，由调用方触发重索引。
        """
        path = Path(cache_dir)
        chunks_file = path / "chunks.pkl"
        vectors_file = path / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            return None
        try:
            with open(chunks_file, "rb") as file_handle:
                payload = pickle.load(file_handle)
            if not isinstance(payload, tuple) or len(payload) != 3 or payload[0] != _MEMORY_FORMAT_VERSION:
                logger.warning("缓存格式版本不符（期望 v%s），触发重索引：%s", _MEMORY_FORMAT_VERSION, chunks_file)
                return None
            _, children, parents = payload
            store = cls()
            store._children = children
            store._parents = parents
            store._parent_ids = {chunk.chunk_id: chunk for chunk in parents}
            store._vectors = numpy.load(vectors_file)
            store._bm25_tokenized = [list(jieba.cut(chunk.content)) for chunk in parents]
            store._bm25 = BM25Okapi(store._bm25_tokenized)
            return store
        except Exception as exception:
            logger.warning("缓存恢复失败（%s），触发重索引", exception)
            return None

    # ---- external 实现 ----

    def _batch_add_external(self, parents: list[Chunk], children: list[Chunk], child_vectors: list[list[float]]) -> None:
        """三库摄入一批父块/子块（幂等），并把父块写入内存缓存。

        Args:
            parents: 父块列表（写入 PgSQL/ES）。
            children: 子块列表（写入 Milvus）。
            child_vectors: 与 children 一一对应的稠密向量。
        """
        ingest_doc(parents, children, numpy.array(child_vectors), self._pgsql, self._es, self._milvus)
        for parent in parents:
            self._chunks_cache[parent.chunk_id] = parent

    def _search_dense_external(self, query_vector: list[float], top_k: int = TOP_K) -> list[Chunk]:
        """Milvus 稠密检索：把命中子块拼成带父块元数据的 Chunk。

        Args:
            query_vector: 查询文本的稠密向量。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：子块列表，metadata 含 parent_id；doc_id / origin_metadata
            取自父块缓存，父块不在缓存时降级为空。
        """
        results = self._milvus.search_dense(query_vector, top_k)  # [(chunk_id, parent_id, content, score)]
        children = []
        for chunk_id, parent_id, content, score in results:
            parent = self._chunks_cache.get(parent_id)
            children.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=parent.doc_id if parent else "",
                    content=content,
                    origin_metadata=parent.origin_metadata if parent else DocMetadata(),
                    metadata={"parent_id": parent_id, "dense_score": float(score)} if parent_id else {}, # 子块携带显式的 dense_score 元数据
                ))
        return children

    def _search_sparse_external(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """ES BM25 检索：把命中父块 id 落到缓存返回父块 Chunk。

        Args:
            query: 查询文本（external 走 IK 分词器 ik_smart 粗粒度）。
            top_k: 返回的最大条数。

        Returns:
            list[Chunk]：按 ES 分数降序的父块列表；未命中缓存 id 直接跳过。
        """
        results = self._es.search_bm25(query, top_k)
        return [self._chunks_cache[chunk_id] for chunk_id, _ in results if chunk_id in self._chunks_cache]

    def _get_children_external(self, parent_id: str) -> list[Chunk]:
        """Milvus 按 parent_id 查子块，拼成带父块元数据的 Chunk 列表。

        Args:
            parent_id: 父块 chunk_id。

        Returns:
            list[Chunk]：该父块下的子块列表；父块不在缓存时元数据降级为空。
        """
        rows = self._milvus.query_by_parent_id(parent_id)
        parent = self._chunks_cache.get(parent_id)
        return [
            Chunk(
                chunk_id=chunk_id,
                doc_id=parent.doc_id if parent else "",
                content=content,
                origin_metadata=parent.origin_metadata if parent else DocMetadata(),
                metadata={"parent_id": parent_id},
            )
            for chunk_id, content in rows
        ]
