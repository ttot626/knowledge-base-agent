# 知识库问答 Agent（RAG + MCP + 混合检索 + 自动评测）

一个面向「内部知识库问答」场景的 **RAG Agent 后端**，把**混合检索**、**MCP 工具化**、**自动评测**三件事串成一个可运行的闭环。

- 基于开源项目 [mcp-rag-agent](https://github.com/luisrodriguesphd/mcp-rag-agent) 二次开发（MIT 协议）
- **主要改动**：数据库从 MongoDB Atlas → **PostgreSQL + pgvector**；LLM 从 OpenAI → **DeepSeek**（OpenAI 兼容接口）；嵌入从 OpenAI API → **本地 BGE 模型**（离线、零成本）

---

## 架构

```
                        ┌────────────────────────────┐
                        │        LangGraph ReAct      │
                        │        Agent (DeepSeek)     │
                        └──────────────┬─────────────┘
                                       │ Tool Calling
                        ┌──────────────▼─────────────┐
                        │     MCP Server (FastMCP)    │
                        │     search_documents        │
                        └──────────────┬─────────────┘
                                       │
                        ┌──────────────▼─────────────┐
                        │  混合检索 (Hybrid Search)    │
                        │  向量检索 + 全文检索 + RRF    │
                        └──────┬──────────────┬───────┘
                               │              │
                     ┌─────────▼───┐   ┌──────▼─────────┐
                     │ pgvector    │   │ tsvector       │
                     │ (HNSW 余弦) │   │ (GIN 全文)     │
                     └──────┬──────┘   └──────┬─────────┘
                            └────────┬────────┘
                                     │
                            PostgreSQL（vectors 表）
```

三个工作流：

1. **索引流**：文档 → 本地 BGE 嵌入 → 写入 `documents`（元数据）+ `vectors`（向量）
2. **问答流**：问题 → Agent 决策调用检索工具 → 混合检索召回相关文档 → 基于上下文生成回答
3. **评测流**：用测试集生成答案 → RAGAS 计算相关性/相似度/正确性 → 输出指标

---

## 技术栈

| 层 | 技术 |
|---|---|
| Agent | LangGraph（ReAct）、LangChain、Tool Calling |
| 协议 | MCP（FastMCP Server） |
| 检索 | 向量检索（pgvector / HNSW / 余弦）+ 全文检索（tsvector / GIN）+ RRF 融合 |
| 存储 | PostgreSQL + pgvector |
| 嵌入 | fastembed + BGE（本地），可选 OpenAI 兼容 API |
| LLM | DeepSeek（OpenAI 兼容接口） |
| 评测 | RAGAS（answer relevancy / similarity / correctness） |

---

## 快速开始

### 1. 启动数据库

```bash
docker compose up -d
```

> 使用 `pgvector/pgvector:pg16` 镜像，默认账号 `postgres/postgres`，库名 `kb_agent`。
> 若不用 Docker，请自行安装 PostgreSQL 并执行 `CREATE EXTENSION vector;`。

### 2. 安装依赖

Windows：双击 `start.cmd`；Linux/macOS：`bash start.sh`。
或手动：

```bash
python -m venv venv
# Windows: venv\Scripts\activate   Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
```

### 3. 配置

```bash
cp .env.example .env   # 然后填 DEEPSEEK_API_KEY
```

### 4. 索引文档

```bash
python -m kb_agent.embeddings.index_documents
```

文档放在 `data/ingested_documents/`，默认已内置 5 份示例（远程办公/报销/年假/IT 安全/可持续）。

### 5. 问答

```bash
python -m kb_agent.agent.create_agent
```

### 6. （可选）自动评测

```bash
pip install -e ".[eval]"
python -m evaluation.main
```

---

## 关键设计说明

### 为什么用混合检索（RRF）

- **向量检索**擅长语义相近、措辞不同的匹配（"远程办公" ≈ "在家工作"）
- **全文检索**擅长精确关键词/专有名词（政策编号、产品名）
- 两者的分数尺度不同（余弦相似度 vs ts_rank），直接相加不可靠
- **RRF** 用「排名」而非「原始分数」融合：`RRF(d) = Σ weightᵢ / (k + rankᵢ)`，对尺度差异鲁棒
- `semantic_weight` 可调语义/关键词权重（默认 0.7）

### 为什么把 OpenAI 换成 DeepSeek + 本地嵌入

- DeepSeek 提供 OpenAI 兼容接口，`ChatOpenAI(base_url=...)` 即可接入，迁移成本低
- DeepSeek 不提供嵌入 API，因此嵌入改用**本地 BGE 模型**：离线可用、零 API 成本、保护隐私

### 为什么用 PostgreSQL + pgvector

- 与已有技术栈统一（避免同时维护 MongoDB 与 PostgreSQL）
- pgvector 支持 HNSW 索引，余弦/内积/L2 多种距离
- 全文检索用原生 tsvector，无需额外搜索引擎

---

## 目录结构

```
knowledge-base-agent/
├── src/kb_agent/
│   ├── core/            # 配置、日志
│   ├── database/        # PostgresClient（asyncpg+pgvector）、RRF、工具函数
│   ├── embeddings/      # EmbeddingGenerator、HybridSearch、index_documents
│   ├── agent/           # Agent 入口、提示词、工厂
│   └── mcp_server/      # FastMCP server、search_documents 工具
├── evaluation/          # RAGAS 自动评测
├── tests/               # 单元测试（RRF、工具函数等纯逻辑）
├── data/                # 示例文档与评测集
├── docker-compose.yml   # PostgreSQL + pgvector
└── pyproject.toml
```

---

## 局限与后续方向

- 全文检索默认英文分词（`english`）；中文需引入 `pg_jieba`/`zhparser` 等分词扩展
- 当前按整文档索引，未做切块（chunking）；长文档应切块 + 元数据过滤
- 会话历史、Checkpoint 持久化可作为下一步（可复用本项目一的 PostgreSQL 会话方案）

---

## 许可

MIT。基于 [mcp-rag-agent](https://github.com/luisrodriguesphd/mcp-rag-agent) 二次开发，保留原作者署名。
