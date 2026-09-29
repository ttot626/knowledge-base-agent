"""混合检索高层接口：封装嵌入生成 + 数据库检索 + 结果格式化。"""

from __future__ import annotations

from typing import Any, Optional

from kb_agent.database.client import PostgresClient
from kb_agent.embeddings.embedding_generator import EmbeddingGenerator


class HybridSearch:
    """提供索引与检索（向量 / 全文 / 混合）的统一入口。"""

    def __init__(
        self,
        db: PostgresClient,
        embedder: EmbeddingGenerator,
        documents_table: str = "documents",
        vectors_table: str = "vectors",
        vector_field: str = "embedding",
        text_field: str = "content",
    ):
        self._db = db
        self._embedder = embedder
        self._documents_table = documents_table
        self._vectors_table = vectors_table
        self._vector_field = vector_field
        self._text_field = text_field

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    async def index_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        table: Optional[str] = None,
    ) -> int:
        table = table or self._vectors_table
        embedding = await self._embedder.generate(content)
        doc = {
            self._text_field: content,
            self._vector_field: embedding,
            "metadata": metadata or {},
        }
        return await self._db.insert_document(table, doc)

    async def index_documents(
        self,
        documents: list[dict[str, Any]],
        table: Optional[str] = None,
    ) -> list[int]:
        table = table or self._vectors_table
        contents = [d.get("content", "") for d in documents]
        embeddings = await self._embedder.generate_batch(contents)
        ids: list[int] = []
        for doc, emb in zip(documents, embeddings):
            row = {
                self._text_field: doc.get("content", ""),
                self._vector_field: emb,
                "metadata": doc.get("metadata", {}),
            }
            ids.append(await self._db.insert_document(table, row))
        return ids

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = 10,
        semantic_weight: float = 0.7,
        rrf_k: int = 60,
        min_vector_score: Optional[float] = None,
        min_text_score: Optional[float] = None,
        min_rrf_score: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        query_vector = await self._embedder.generate(query)
        results = await self._db.hybrid_search(
            table=self._vectors_table,
            vector_field=self._vector_field,
            text_field=self._text_field,
            query_vector=query_vector,
            query_text=query,
            limit=limit,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
            min_vector_score=min_vector_score,
            min_text_score=min_text_score,
            min_rrf_score=min_rrf_score,
        )
        return [self._format(r) for r in results]

    async def vector_search(
        self, query: str, limit: int = 10, min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        query_vector = await self._embedder.generate(query)
        results = await self._db.vector_search(
            table=self._vectors_table,
            vector_field=self._vector_field,
            query_vector=query_vector,
            limit=limit,
            min_score=min_score,
        )
        return [self._format(r) for r in results]

    async def text_search(
        self, query: str, limit: int = 10, min_score: Optional[float] = None
    ) -> list[dict[str, Any]]:
        results = await self._db.text_search(
            table=self._vectors_table,
            text_field=self._text_field,
            query_text=query,
            limit=limit,
            min_score=min_score,
        )
        return [self._format(r) for r in results]

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    async def setup_indexes(
        self,
        vector_index_name: str,
        text_index_name: str,
        similarity: str = "cosine",
    ) -> None:
        await self._db.create_vector_index(
            self._vectors_table, vector_index_name, self._vector_field, similarity
        )
        await self._db.create_text_index(
            self._vectors_table, text_index_name, self._text_field
        )

    @staticmethod
    def _format(doc: dict[str, Any]) -> dict[str, Any]:
        """去掉向量字段、把 id 转成字符串，避免返回内容过大。"""
        doc = dict(doc)
        doc.pop("embedding", None)
        doc["id"] = str(doc.get("id", ""))
        return doc
