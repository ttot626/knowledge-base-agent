"""数据库层的纯工具函数（无第三方依赖，方便单测）。"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def quote_ident(name: str) -> str:
    """校验并返回安全的 SQL 标识符（表名/列名/索引名）。

    只允许字母/数字/下划线开头，防止 SQL 注入。
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"非法标识符: {name!r}")
    return name


def vector_to_str(vector: list[float]) -> str:
    """把浮点向量转成 pgvector 可识别的字符串 '[1.0,2.0,...]'。"""
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def build_tsquery(text: str) -> Optional[str]:
    """把自然语言查询拆词后用 OR 连接成 tsquery 字符串。

    使用 OR（而不是 PostgreSQL 默认的 AND）是为了避免"查询里所有词都必须命中"
    导致召回过低；停用词（如 how/the/of）会被 to_tsquery 自动丢弃。

    返回 None 表示没有任何可用词元。
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return None
    return " | ".join(tokens)


def decode_metadata(value: Any) -> dict:
    """把数据库返回的 jsonb（可能是 JSON 字符串）解码为 dict。"""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}
