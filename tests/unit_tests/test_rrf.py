"""RRF 融合算法的单元测试。"""

from kb_agent.database.rrf import rrf_fuse


def _doc(doc_id, score=None):
    d = {"id": doc_id, "content": f"doc{doc_id}"}
    if score is not None:
        d["score"] = score
    return d


def _text_doc(doc_id, text_score=None):
    d = {"id": doc_id, "content": f"doc{doc_id}"}
    if text_score is not None:
        d["text_score"] = text_score
    return d


def test_combined_doc_gets_both_weights():
    vector = [_doc(1, 0.9), _doc(2, 0.8)]
    text = [_doc(2, 3.0), _doc(3, 2.0)]
    result = rrf_fuse(vector, text, vector_weight=0.7, text_weight=0.3, k=60)

    assert result[0]["document"]["id"] == 2  # 出现在两路，分数最高
    assert result[0]["vector_rank"] == 2
    assert result[0]["text_rank"] == 1
    # rrf = 0.7/(60+2) + 0.3/(60+1)
    expected = 0.7 / 62 + 0.3 / 61
    assert abs(result[0]["rrf_score"] - expected) < 1e-9


def test_ordering_and_rank_fields():
    vector = [_doc(1, 0.9)]
    text = [_doc(2, 3.0)]
    result = rrf_fuse(vector, text, vector_weight=0.7, text_weight=0.3, k=60)

    # 0.7/61 vs 0.3/61 → vector 结果排第一
    assert result[0]["document"]["id"] == 1
    assert result[0]["vector_rank"] == 1
    assert result[0]["text_rank"] is None
    assert result[1]["document"]["id"] == 2
    assert result[1]["text_rank"] == 1
    assert result[1]["vector_rank"] is None


def test_empty_inputs():
    assert rrf_fuse([], []) == []
