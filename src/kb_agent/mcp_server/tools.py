"""检索工具：把混合检索封装成可被 Agent / MCP 调用的函数。"""

import logging

from kb_agent.core.config import config
from kb_agent.core.log_setup import setup_logging
from kb_agent.database.client import PostgresClient
from kb_agent.embeddings.embedding_generator import EmbeddingGenerator
from kb_agent.embeddings.hybrid_search import HybridSearch

setup_logging()
logger = logging.getLogger("Retriever")

# 模块级单例；数据库连接在首次调用时惰性建立
_db = PostgresClient(config.db_url)
_embedder = EmbeddingGenerator(
    backend=config.embedding_backend,
    model=config.embedding_model,
    dimensions=config.embedding_dimension,
    api_key=config.embedding_api_key,
    base_url=config.embedding_base_url,
)
_hybrid = HybridSearch(
    _db, _embedder, config.db_documents_table, config.db_vectors_table
)


async def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """在知识库中做混合检索（向量 + 全文 + RRF 融合）。

    Args:
        query: 检索的问题文本。
        top_k: 返回的最大文档数，默认 3。

    Returns:
        按 RRF 分数降序的相关文档列表，每项含 content / metadata / rrf_score 等。
    """
    await _db.ensure_connected()
    logger.info("Hybrid search started: %s", query)

    results = await _hybrid.search(
        query=query,
        limit=top_k,
        semantic_weight=config.semantic_weight,
        rrf_k=config.rrf_k,
    )

    logger.info("Found %d relevant documents", len(results))
    return results
