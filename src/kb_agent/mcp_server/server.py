"""FastMCP server：把 search_documents 暴露为 MCP 工具。

用法：
    mcp dev src/kb_agent/mcp_server/server.py   # 调试
    python -m kb_agent.mcp_server.server        # 以 stdio 运行
"""

import logging

from mcp.server.fastmcp import FastMCP

from kb_agent.core.config import config
from kb_agent.core.log_setup import setup_logging
from kb_agent.mcp_server.tools import search_documents

setup_logging()
logger = logging.getLogger("MCP Server")

mcp = FastMCP(
    name=config.mcp_name,
    host=config.mcp_host,
    port=config.mcp_port,
)

# 注册检索工具
mcp.tool()(search_documents)


@mcp.prompt()
def grounded_qa_prompt() -> str:
    """提供"严格基于检索文档回答"的提示词模板。"""
    return (
        "You answer ONLY using the provided documents. "
        "If information is missing, say you don't know."
    )


if __name__ == "__main__":
    logger.info("Starting MCP server (stdio)...")
    mcp.run(transport="stdio")
