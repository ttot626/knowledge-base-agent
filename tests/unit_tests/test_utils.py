"""database.utils 纯函数的单元测试。"""

from kb_agent.database.utils import build_tsquery, decode_metadata, quote_ident, vector_to_str


def test_vector_to_str():
    assert vector_to_str([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"
    assert vector_to_str([1, 2, 3]) == "[1.0,2.0,3.0]"


def test_build_tsquery_ors_tokens():
    assert build_tsquery("annual leave days") == "annual | leave | days"


def test_build_tsquery_lowercases():
    assert build_tsquery("Remote Work") == "remote | work"


def test_build_tsquery_empty():
    assert build_tsquery("") is None
    assert build_tsquery("!!!") is None


def test_quote_ident_ok():
    assert quote_ident("vectors") == "vectors"
    assert quote_ident("_documents_2") == "_documents_2"


def test_quote_ident_rejects_injection():
    for bad in ["vectors; DROP TABLE users", "1abc", "a-b", "a b"]:
        try:
            quote_ident(bad)
            assert False, f"should have rejected {bad!r}"
        except ValueError:
            pass


def test_decode_metadata():
    assert decode_metadata(None) == {}
    assert decode_metadata({"a": 1}) == {"a": 1}
    assert decode_metadata('{"a": 1}') == {"a": 1}
    assert decode_metadata("not-json") == {}
