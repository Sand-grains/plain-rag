"""PostgreSQL 客户端：进行父块存储 + status 状态机 + 回滚清理 若干数据库操作

表名为 parent_chunks（父块集合，检索/benchmark 单元），
子块不入库（Milvus 存向量，metadata.parent_id 关联父块）。
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import psycopg2
import psycopg2.extras
from infra.config import PG_PENDING_CLEANUP_MINUTES, POSTGRES_CONNECTION_URI

if TYPE_CHECKING:
    from indexing.chunk import Chunk


class PgSQLClient:
    """PostgreSQL 客户端：parent_chunks 父块表读写 + suspending 状态机推进。"""

    def __init__(self, conn_url: str = POSTGRES_CONNECTION_URI):
        self._url = conn_url
        self._conn = psycopg2.connect(conn_url)
        self._conn.autocommit = True
        # 注册 jsonb typecaster：读回 JSONB 列自动反序列化为 dict
        psycopg2.extras.register_json(self._conn, loads=json.loads)

    def ensure_table(self) -> None:
        """建 parent_chunks 表（含 doc_id / status 索引），已存在则跳过。"""
        cursor = self._conn.cursor() # SQL执行句柄, 负责把任意 SQL 发给服务器并接收结果
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parent_chunks (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   JSONB DEFAULT '{}',
                status     TEXT DEFAULT 'suspending',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON parent_chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON parent_chunks(status)")
        cursor.close()

    def drop_table(self) -> None:
        """删除 parent_chunks 表（用于重建/清理）。"""
        cursor = self._conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS parent_chunks")
        cursor.close()

    def drop_legacy_tables(self) -> None:
        """迁移用：删除遗留的 chunks 孤儿表。"""
        cursor = self._conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS chunks")
        cursor.close()

    def insert_batch(self, parents: list[Chunk]) -> int:
        """批量插入父块，status 初始为 suspending。

        Args:
            parents: 父块列表。chunk_id 为主键，重复摄入同一文档前须先
                rollback_doc 清理旧数据，否则主键冲突。

        Returns:
            int：插入的行数。
        """
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.content,
                json.dumps(
                    {
                        **chunk.metadata,
                        "doc_title": chunk.origin_metadata.title,
                        "doc_type": chunk.origin_metadata.doc_type,
                        "chunk_level": chunk.origin_metadata.chunk_level,
                    },
                    ensure_ascii=False,
                ),
                "suspending",
            )
            for chunk in parents
        ]
        cursor = self._conn.cursor()
        psycopg2.extras.execute_values(
            cursor,
            "INSERT INTO parent_chunks (chunk_id, doc_id, content, metadata, status) VALUES %s",
            rows,
        )
        cursor.close()
        return len(rows)

    def update_status(self, chunk_ids: list[str], status: str = "indexed") -> None:
        """批量更新父块 status（suspending → indexed 状态机推进）。

        Args:
            chunk_ids: 待更新的 chunk_id 列表。
            status: 目标状态，默认 "indexed"。
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE parent_chunks SET status = %s WHERE chunk_id = ANY(%s::text[])",
            (status, list(chunk_ids)),
        )
        cursor.close()

    def get_by_ids(self, chunk_ids: list[str]) -> list[tuple[str, str, str, dict]]:
        """按 chunk_id 批量查询父块，返回原始行（由 IndexStore 组装为 Chunk）。

        Args:
            chunk_ids: 待查询的 chunk_id 列表。

        Returns:
            list[(chunk_id, doc_id, content, metadata_dict)]：查询命中的父块原始行。
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT chunk_id, doc_id, content, metadata FROM parent_chunks WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows  # [(chunk_id, doc_id, content, metadata_dict), ...]

    def cleanup_suspending(self, ttl_minutes: int = PG_PENDING_CLEANUP_MINUTES) -> list[tuple[str, str]]:
        """扫描超时未完成的 suspending 记录。

        Args:
            ttl_minutes: 超时阈值（分钟），超过该时长仍为 suspending 的记录视为残留。

        Returns:
            list[(chunk_id, doc_id)]：本次扫描到的残留记录，供上层按 doc_id 回滚三端。
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT chunk_id, doc_id FROM parent_chunks WHERE status = 'suspending'"
            " AND created_at < NOW() - INTERVAL '%s minutes'",
            (ttl_minutes,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return rows  # [(chunk_id, doc_id), ...]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除指定 doc_id 的全部父块（rollback_doc 的一环）。"""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM parent_chunks WHERE doc_id = %s", (doc_id,))
        cursor.close()

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """按 chunk_id 列表删除父块。"""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM parent_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
        cursor.close()
