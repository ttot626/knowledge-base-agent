"""普通 RAG Agent 工厂：LangChain 工具 + LangGraph ReAct。"""

import logging
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from kb_agent.core.config import Config
from kb_agent.core.log_setup import setup_logging

setup_logging()
logger = logging.getLogger("Agent Creator")


async def create_rag_agent(system_prompt: str, tools: List[BaseTool], config: Config):
    """返回一个 LangGraph ReAct agent。

    - 使用 ChatOpenAI + DeepSeek（OpenAI 兼容接口）
    - 通过 Tool Calling 调用检索工具（如 search_knowledge_base）
    """
    model = ChatOpenAI(
        api_key=config.model_api_key,
        base_url=config.model_base_url,
        model=config.text_model,
        **config.text_generation_kwargs,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SystemMessage(content=system_prompt),
        debug=config.debug,
    )
    return agent
