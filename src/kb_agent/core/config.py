"""集中配置：PostgreSQL、DeepSeek LLM、嵌入后端、MCP 与检索参数。

所有配置项都提供默认值，确保在没有 .env 的情况下也能正常 import；
真正运行前请复制 `.env.example` 为 `.env` 并按需修改。
"""

import os

from dotenv import load_dotenv

# 先加载 .env（若存在），再读取环境变量
load_dotenv(override=True)


class Config:
    """kb_agent 的全部运行配置。"""

    # ---- 应用 ----
    app_name: str = "Knowledge Base Agent"
    app_version: str = "0.1.0"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ---- PostgreSQL ----
    db_url: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/kb_agent",
    )
    db_documents_table: str = os.getenv("DB_DOCUMENTS_TABLE", "documents")
    db_vectors_table: str = os.getenv("DB_VECTORS_TABLE", "vectors")
    db_vector_index_name: str = os.getenv("DB_VECTOR_INDEX_NAME", "vectors_embedding_hnsw")
    db_text_index_name: str = os.getenv("DB_TEXT_INDEX_NAME", "vectors_content_gin")

    # ---- LLM（DeepSeek，OpenAI 兼容接口）----
    model_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    model_base_url: str = os.getenv("MODEL_BASE_URL", "https://api.deepseek.com")
    text_model: str = os.getenv("TEXT_MODEL_NAME", "deepseek-chat")
    text_generation_kwargs: dict = {
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 0.1,
    }
    evaluation_model: str = os.getenv("EVALUATION_MODEL_NAME", "deepseek-chat")

    # ---- 嵌入后端 ----
    # local：本地 fastembed + BGE 模型（离线、零成本、默认）
    # openai：任意 OpenAI 兼容嵌入 API（需 EMBEDDING_API_KEY / EMBEDDING_BASE_URL）
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")

    # ---- MCP 服务 ----
    mcp_name: str = os.getenv("MCP_SERVER_NAME", "knowledge-base-search")
    mcp_host: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    mcp_port: int = int(os.getenv("MCP_SERVER_PORT", "8000"))

    # ---- 检索 ----
    # 语义 / 关键词权重：0.7 = 70% 语义 + 30% 关键词
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # ---- 特性开关 ----
    ff_mcp_server: bool = (
        os.getenv("FEATURE_FLAG_MCPSERVER_ENABLED", "false").lower() == "true"
    )
    ff_web_search: bool = (
        os.getenv("FEATURE_FLAG_WEBSEARCH_ENABLED", "false").lower() == "true"
    )

    # ---- 路径 ----
    ingested_doc_dir: str = os.getenv("INGESTED_DOC_DIRECTORY", "./data/ingested_documents")
    evaluation_doc_dir: str = os.getenv("EVALUATION_DOC_DIRECTORY", "./data/evaluation_documents")


config = Config()
