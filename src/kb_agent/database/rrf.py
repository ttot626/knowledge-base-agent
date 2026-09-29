"""Reciprocal Rank Fusion（RRF）融合算法，纯函数、可单测。

RRF 是一种把多路检索结果按"排名"而非"原始分数"融合的经典方法，对
向量检索（余弦相似度）与全文检索（ts_rank）这种分数尺度不一致的场景尤其鲁棒。

公式：RRF_score(doc) = Σ weight_i / (k + rank_i)
"""

from __future__ import annotations

from typing import Any, Iterable


def rrf_fuse(
    vector_results: Iterable[dict[str, Any]],
    text_results: Iterable[dict[str, Any]],
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    k: int = 60,
) -> list[dict[str, Any]]:
    """按 RRF 公式融合向量检索与全文检索结果，返回按 rrf_score 降序的列表。

    每个结果条目包含：
        document    原始文档 dict
        rrf_score   融合后的 RRF 分数
        vector_rank 在向量结果中的排名（未命中为 None）
        text_rank   在全文结果中的排名（未命中为 None）
        vector_score / text_score 原始分数（未命中为 None）
    """
    fused: dict[Any, dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]
        fused[doc_id] = {
            "document": doc,
            "rrf_score": vector_weight / (k + rank),
            "vector_rank": rank,
            "vector_score": doc.get("score"),
            "text_rank": None,
            "text_score": None,
        }

    for rank, doc in enumerate(text_results, start=1):
        doc_id = doc["id"]
        score = text_weight / (k + rank)
        if doc_id in fused:
            fused[doc_id]["rrf_score"] += score
            fused[doc_id]["text_rank"] = rank
            fused[doc_id]["text_score"] = doc.get("text_score")
        else:
            fused[doc_id] = {
                "document": doc,
                "rrf_score": score,
                "vector_rank": None,
                "vector_score": None,
                "text_rank": rank,
                "text_score": doc.get("text_score"),
            }

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
