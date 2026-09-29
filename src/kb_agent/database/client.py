"""PostgreSQL + pgvector 异步数据库客户端。

将原 MongoDB 版本迁移到 PostgreSQL：
- 向量存储 / 检索：pgvector（HNSW 索引，余弦相似度）
- 全文检索：tsvector + GIN 索引，ts_rank 打分
- 混合检索：RRF 融合（见 rrf.py）
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import asyncpg

from kb_agent.database.rrf import rrf_fuse
from kb_agent.database.utils import (
    build_tsquery,
    decode_metadata,
    quote_ident,
    vector_to_str,
)

_VECTOR_OPS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}


class PostgresClient:
    """基于 asyncpg + pgvector 的异步数据库客户端。"""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
            await self._pool.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def ensure_connected(self) -> None:
        if self._pool is None:
            await self.connect()

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    async def table_exists(self, table: str) -> bool:
        row = await self._pool.fetchrow(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = $1",
            quote_ident(table),
        )
        return row is not None

    async def list_tables(self) -> list[str]:
        rows = await self._pool.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        return [r["table_name"] for r in rows]

    async def create_tables(self, documents_table: str, vectors_table: str, dim: int) -> None:
        """创建文档元数据表与向量表（含 pgvector 向量列）。"""
        await self._pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quote_ident(documents_table)} (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                folder TEXT,
                relative_path TEXT,
                content TEXT,
                size INTEGER,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        await self._pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quote_ident(vectors_table)} (
                id SERIAL PRIMARY KEY,
                document_id INTEGER,
                document_name TEXT,
                content TEXT NOT NULL,
                embedding vector({int(dim)}) NOT NULL,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_insert(document: dict[str, Any]) -> tuple[list[str], list[str], list[Any]]:
        """把 dict 文档编码为 (列名, 占位符, 值)。

        - list/tuple 视为向量 → `::vector`
        - dict 视为元数据 → `::jsonb`
        """
        cols: list[str] = []
        placeholders: list[str] = []
        values: list[Any] = []
        for i, (col, val) in enumerate(document.items(), start=1):
            cols.append(quote_ident(col))
            if isinstance(val, (list, tuple)):
                placeholders.append(f"${i}::vector")
                values.append(vector_to_str(list(val)))
            elif isinstance(val, dict):
                placeholders.append(f"${i}::jsonb")
                values.append(json.dumps(val, ensure_ascii=False))
            else:
                placeholders.append(f"${i}")
                values.append(val)
        return cols, placeholders, values

    async def insert_document(self, table: str, document: dict[str, Any]) -> int:
        cols, placeholders, values = self._encode_insert(document)
        sql = (
            f"INSERT INTO {quote_ident(table)} ({', '.join(cols)}) "
            f"VALUES ({', '.join(placeholders)}) RETURNING id"
        )
        row = await self._pool.fetchrow(sql, *values)
        return row["id"]

    async def insert_documents(self, table: str, documents: list[dict[str, Any]]) -> list[int]:
        ids: list[int] = []
        for doc in documents:
            ids.append(await self.insert_document(table, doc))
        return ids

    async def delete_documents(
        self, table: str, where: Optional[dict[str, Any]] = None
    ) -> int:
        """删除行，返回删除数量。where 为空时删除全表。"""
        if where:
            conditions = []
            values = []
            for i, (col, val) in enumerate(where.items(), start=1):
                conditions.append(f"{quote_ident(col)} = ${i}")
                values.append(val)
            sql = f"DELETE FROM {quote_ident(table)} WHERE {' AND '.join(conditions)}"
            tag = await self._pool.execute(sql, *values)
        else:
            tag = await self._pool.execute(f"DELETE FROM {quote_ident(table)}")
        m = re.search(r"\d+$", tag)
        return int(m.group()) if m else 0

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    async def create_vector_index(
        self,
        table: str,
        index_name: str,
        vector_field: str,
        similarity: str = "cosine",
    ) -> None:
        ops = _VECTOR_OPS.get(similarity, "vector_cosine_ops")
        await self._pool.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} "
            f"ON {quote_ident(table)} USING hnsw ({quote_ident(vector_field)} {ops})"
        )

    async def create_text_index(
        self,
        table: str,
        index_name: str,
        text_field: str,
        language: str = "english",
    ) -> None:
        if not re.match(r"^[a-z]+$", language):
            raise ValueError(f"非法 language: {language!r}")
        await self._pool.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} "
            f"ON {quote_ident(table)} USING gin (to_tsvector('{language}', {quote_ident(text_field)}))"
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_doc(row: asyncpg.Record) -> dict[str, Any]:
        doc = dict(row)
        doc["metadata"] = decode_metadata(doc.get("metadata"))
        return doc

    async def vector_search(
        self,
        table: str,
        vector_field: str,
        query_vector: list[float],
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        vec = vector_to_str(query_vector)
        field = quote_ident(vector_field)
        sql = (
            f"SELECT id, document_id, document_name, content, metadata, "
            f"1 - ({field} <=> $1::vector) AS score "
            f"FROM {quote_ident(table)} ORDER BY {field} <=> $1::vector LIMIT $2"
        )
        rows = await self._pool.fetch(sql, vec, limit)
        results = [self._row_to_doc(r) for r in rows]
        if min_score is not None:
            results = [r for r in results if r["score"] >= min_score]
        return results

    async def text_search(
        self,
        table: str,
        text_field: str,
        query_text: str,
        limit: int = 10,
        language: str = "english",
        min_score: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if not re.match(r"^[a-z]+$", language):
            raise ValueError(f"非法 language: {language!r}")
        tsquery = build_tsquery(query_text)
        if not tsquery:
            return []
        field = quote_ident(text_field)
        sql = (
            f"SELECT id, document_id, document_name, content, metadata, "
            f"ts_rank(to_tsvector('{language}', {field}), to_tsquery('{language}', $1)) AS text_score "
            f"FROM {quote_ident(table)} "
            f"WHERE to_tsvector('{language}', {field}) @@ to_tsquery('{language}', $1) "
            f"ORDER BY text_score DESC LIMIT $2"
        )
        rows = await self._pool.fetch(sql, tsquery, limit)
        results = [self._row_to_doc(r) for r in rows]
        if min_score is not None:
            results = [r for r in results if r["text_score"] >= min_score]
        return results

    async def hybrid_search(
        self,
        table: str,
        vector_field: str,
        text_field: str,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        rrf_k: int = 60,
        min_vector_score: Optional[float] = None,
        min_text_score: Optional[float] = None,
        min_rrf_score: Optional[float] = None,
        language: str = "english",
    ) -> list[dict[str, Any]]:
        """向量检索 + 全文检索，用 RRF 融合后返回 top-k。"""
        vector_results = await self.vector_search(
            table=table,
            vector_field=vector_field,
            query_vector=query_vector,
            limit=limit * 2,
            min_score=min_vector_score,
        )
        text_results = await self.text_search(
            table=table,
            text_field=text_field,
            query_text=query_text,
            limit=limit * 2,
            language=language,
            min_score=min_text_score,
        )

        fused = rrf_fuse(
            vector_results=vector_results,
            text_results=text_results,
            vector_weight=semantic_weight,
            text_weight=1.0 - semantic_weight,
            k=rrf_k,
        )

        final: list[dict[str, Any]] = []
        for item in fused:
            if min_rrf_score is not None and item["rrf_score"] < min_rrf_score:
                continue
            doc = dict(item["document"])
            doc["rrf_score"] = item["rrf_score"]
            doc["vector_rank"] = item["vector_rank"]
            doc["text_rank"] = item["text_rank"]
            if item["vector_score"] is not None:
                doc["vector_score"] = item["vector_score"]
            if item["text_score"] is not None:
                doc["text_score"] = item["text_score"]
            final.append(doc)
            if len(final) >= limit:
                break
        return final
