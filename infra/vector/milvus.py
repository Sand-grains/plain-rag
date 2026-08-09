"""Milvus 客户端：collection 管理 + Dense 向量检索（IP 度量 = 余弦相似度）。

collection 存子块（rag_child_chunks），行含 parent_id 关联父块。
子块是稠密检索单元，检索结果由 IndexStore 拼成带父块元数据的 Chunk。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from infra.config import MILVUS_COLLECTION, MILVUS_CONNECTION_URI, MILVUS_HNSW_EF

if TYPE_CHECKING:
    from indexing.chunk import Chunk

_EMBEDDING_DIM = 1024  # BGE-M3
_CONTENT_MAX_BYTES = 65535  # Milvus VARCHAR(65535) 上限（字节）


def _fit_varchar(text: str, max_bytes: int = _CONTENT_MAX_BYTES) -> str:
    """按字节截断到 VARCHAR 上限，避免超长 fenced code block 整块入库失败。

    Args:
        text: 待入库的子块内容。
        max_bytes: 字节数上限（Milvus VARCHAR 最大长度）。

    Returns:
        str：截断后不超过 max_bytes 字节的文本。
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    byte_size = 0
    kept_count = 0
    for char in text:
        byte_size += len(char.encode("utf-8"))
        if byte_size > max_bytes:
            break
        kept_count += 1
    return text[:kept_count]


class MilvusClientWrapper:
    """Milvus 客户端：子块 collection 管理 + Dense 向量检索（IP 度量）"""

    def __init__(self, uri: str = MILVUS_CONNECTION_URI, collection_name: str = MILVUS_COLLECTION):
        self._client = MilvusClient(uri=uri)
        self._collection = collection_name

    def ensure_collection(self) -> None:
        """确保 collection 存在：存在则加载，不存在则按子块 schema 创建并加载。"""
        if self._client.has_collection(self._collection):
            self._client.load_collection(self._collection)
            return
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=_EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, description="rag child chunks")
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_hnsw",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(self._collection)

    def drop_collection(self) -> None:
        """删除 collection（用于重建/清理）。"""
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)

    def insert_batch(self, children: list[Chunk], vectors: numpy.ndarray) -> None:
        """批量插入子块及向量，content 按 VARCHAR 上限截断。

        Args:
            children: 子块列表（稠密检索单元）。
            vectors: 与 children 一一对应的稠密向量矩阵（二维 ndarray）。
        """
        data = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "parent_id": chunk.metadata.get("parent_id") or "",
                "content": _fit_varchar(chunk.content),
                "vector": vectors[index].tolist(),
            }
            for index, chunk in enumerate(children)
        ]
        self._client.insert(collection_name=self._collection, data=data)

    def search_dense(self, query_vector: list[float] | numpy.ndarray, top_k: int = 10) -> list[tuple[str, str, str, float]]:
        """Dense 向量检索，返回命中子块及父块关联信息。

        Args:
            query_vector: 查询向量（list 或 ndarray）。
            top_k: 返回的最大条数。

        Returns:
            list[(chunk_id, parent_id, content, score)]：score 为 IP 距离（越大越相似）。
        """
        if isinstance(query_vector, numpy.ndarray):
            query_vector = query_vector.tolist()
        results = self._client.search(
            collection_name=self._collection,
            data=[query_vector],
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"ef": MILVUS_HNSW_EF}},
            limit=top_k,
            output_fields=["chunk_id", "parent_id", "content"],
        )
        return [
            (
                hit["entity"]["chunk_id"],
                hit["entity"].get("parent_id") or "",
                hit["entity"].get("content", ""),
                hit["distance"],
            )
            for hit in results[0]
        ]

    def query_by_parent_id(self, parent_id: str) -> list[tuple[str, str]]:
        """按 parent_id 查询子块。

        Args:
            parent_id: 父块 chunk_id。

        Returns:
            list[(chunk_id, content)]：该父块下的子块列表。
        """
        results = self._client.query(
            collection_name=self._collection,
            filter=f'parent_id == "{parent_id}"',
            output_fields=["chunk_id", "content"],
        )
        return [(record["chunk_id"], record.get("content", "")) for record in results]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按 doc_id 删除子块（rollback_doc 的一环）。"""
        self._client.delete(collection_name=self._collection, filter=f'doc_id == "{doc_id}"')

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """按 chunk_id 列表删除子块。"""
        filter_expression = "chunk_id in [" + ",".join(f'"{chunk_id}"' for chunk_id in chunk_ids) + "]"
        self._client.delete(collection_name=self._collection, filter=filter_expression)
