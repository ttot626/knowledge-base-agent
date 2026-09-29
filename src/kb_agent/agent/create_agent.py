"""Agent 入口：构建 Agent 并提供交互式问答。

用法：
    python -m kb_agent.agent.create_agent
"""

import asyncio
import logging

from kb_agent.agent.prompts import system_prompt
from kb_agent.core.config import config
from kb_agent.core.log_setup import setup_logging

setup_logging()
logger = logging.getLogger("Agent")


async def build_agent():
    """按配置构建 Agent。

    FEATURE_FLAG_MCPSERVER_ENABLED=true 时走 MCP server 加载工具，
    否则直接用 LangChain StructuredTool 包装 search_documents。
    """
    if config.ff_mcp_server:
        from kb_agent.agent.utils.mcp_rag_agent_creator import create_mcp_rag_agent

        return await create_mcp_rag_agent(system_prompt=system_prompt, config=config)

    from pydantic import BaseModel, Field

    from kb_agent.agent.utils.rag_agent_creator import create_rag_agent
    from kb_agent.mcp_server.tools import search_documents
    from langchain_core.tools import StructuredTool

    class SearchDocumentsInput(BaseModel):
        """search_knowledge_base 工具的入参 schema。"""

        query: str = Field(description="要检索的问题文本")
        top_k: int = Field(default=3, ge=1, le=10, description="返回的最大文档数")

    tool = StructuredTool(
        name="search_knowledge_base",
        description=(
            "在知识库中检索与问题最相关的文档，用于支撑最终回答。"
        ),
        func=lambda query, top_k=3: asyncio.run(search_documents(query, top_k)),
        coroutine=search_documents,
        args_schema=SearchDocumentsInput,
    )

    return await create_rag_agent(system_prompt=system_prompt, tools=[tool], config=config)


async def ask(agent, question: str) -> str:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].content


async def main() -> None:
    agent = await build_agent()
    print("知识库问答 Agent 已就绪，输入问题开始（输入 exit 退出）")
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            break
        print("Agent: ", await ask(agent, question))


if __name__ == "__main__":
    asyncio.run(main())
