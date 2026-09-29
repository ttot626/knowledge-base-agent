"""MCP RAG Agent 工厂：通过 MCP server（stdio）加载检索工具。"""

import logging
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain_openai import ChatOpenAI

from kb_agent.core.config import Config, config
from kb_agent.core.log_setup import setup_logging

setup_logging()
logger = logging.getLogger("Agent Creator")

MCP_SERVER_NAME = config.mcp_name

# 通过 stdio 启动 MCP server 子进程
MCP_CONNECTIONS = {
    MCP_SERVER_NAME: {
        "command": "python",
        "args": ["-m", "kb_agent.mcp_server.server"],
        "transport": "stdio",
    }
}


async def _load_mcp_tools_and_prompt() -> tuple[List[BaseTool], str]:
    client = MultiServerMCPClient(MCP_CONNECTIONS)
    tools = await client.get_tools(server_name=MCP_SERVER_NAME)

    async with client.session(MCP_SERVER_NAME) as session:
        prompt_messages = await load_mcp_prompt(
            session, name="grounded_qa_prompt"
        )

    grounded_prompt = " ".join(m.content for m in prompt_messages)
    return tools, grounded_prompt


async def create_mcp_rag_agent(system_prompt: str, config: Config):
    """返回通过 MCP 加载工具、由 grounded_qa_prompt 约束的 LangGraph ReAct agent。"""
    tools, grounded_prompt = await _load_mcp_tools_and_prompt()

    full_system_prompt = system_prompt + "\n" + grounded_prompt

    model = ChatOpenAI(
        api_key=config.model_api_key,
        base_url=config.model_base_url,
        model=config.text_model,
        **config.text_generation_kwargs,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SystemMessage(content=full_system_prompt),
    )
    return agent
