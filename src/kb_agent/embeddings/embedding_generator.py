"""嵌入生成：默认本地 fastembed + BGE，也支持 OpenAI 兼容嵌入 API。"""

from __future__ import annotations

import asyncio
from typing import Optional


class EmbeddingGenerator:
    """生成文本向量。

    backend = "local"：本地 fastembed + BGE 模型（离线、零成本、默认，懒加载）
    backend = "openai"：任意 OpenAI 兼容嵌入 API
    """

    def __init__(
        self,
        backend: str = "local",
        model: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._backend = backend
        self._model = model
        self._dimensions = dimensions
        self._local_model = None
        self._client = None

        if backend == "openai":
            import openai

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._backend

    def _ensure_local(self):
        if self._local_model is None:
            from fastembed import TextEmbedding

            self._local_model = TextEmbedding(model_name=self._model)
        return self._local_model

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_local()
        return [list(vec) for vec in model.embed(texts)]

    async def generate(self, text: str) -> list[float]:
        batch = await self.generate_batch([text])
        return batch[0]

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._backend == "openai":
            kwargs: dict = {"model": self._model, "input": texts}
            # 仅 OpenAI text-embedding-3 系列支持显式指定维度
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dimensions
            resp = await self._client.embeddings.create(**kwargs)
            data = sorted(resp.data, key=lambda x: x.index)
            return [list(d.embedding) for d in data]
        # 本地 fastembed 是同步的，放进线程池避免阻塞事件循环
        return await asyncio.to_thread(self._embed_local, texts)
