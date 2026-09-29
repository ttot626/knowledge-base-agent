"""把 data/ingested_documents 下的文档索引进 PostgreSQL（元数据表 + 向量表）。

用法：
    python -m kb_agent.embeddings.index_documents [--clear]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from kb_agent.core.config import config
from kb_agent.database.client import PostgresClient
from kb_agent.embeddings.embedding_generator import EmbeddingGenerator
from kb_agent.embeddings.hybrid_search import HybridSearch


async def index_documents_from_folder(
    folder_path: str,
    db: PostgresClient,
    hybrid: HybridSearch,
    documents_table: str,
    vectors_table: str,
) -> None:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误：目录 {folder_path} 不存在。")
        return

    files = sorted(folder.rglob("*.txt"))
    if not files:
        print(f"在 {folder_path} 下没有找到 .txt 文件。")
        return

    print(f"\n找到 {len(files)} 个文档待索引")
    print("=" * 60)

    ok = 0
    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8").strip()
            if not content:
                print(f"⚠️  跳过空文件: {fp.name}")
                continue

            rel = fp.relative_to(folder)
            folder_name = rel.parent.name if rel.parent != Path(".") else "root"
            now = datetime.now(timezone.utc)

            # 1) 写入文档元数据表
            doc_id = await db.insert_document(documents_table, {
                "name": fp.name,
                "folder": folder_name,
                "relative_path": str(rel),
                "content": content,
                "size": len(content),
                "created_at": now,
            })

            # 2) 生成嵌入并写入向量表
            await hybrid.index_document(
                content=content,
                metadata={
                    "document_id": doc_id,
                    "document_name": fp.name,
                    "source": folder_name,
                },
                table=vectors_table,
            )
            print(f"✅ 已索引: {fp.name}（{len(content)} 字符）")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"❌ 处理 {fp.name} 失败: {e}")

    print("=" * 60)
    print(f"✨ 完成：成功索引 {ok}/{len(files)} 个文档")


async def main(clear: bool = False) -> None:
    db = PostgresClient(config.db_url)
    await db.connect()

    embedder = EmbeddingGenerator(
        backend=config.embedding_backend,
        model=config.embedding_model,
        dimensions=config.embedding_dimension,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
    )
    hybrid = HybridSearch(
        db, embedder, config.db_documents_table, config.db_vectors_table
    )

    try:
        if clear:
            await db.delete_documents(config.db_documents_table)
            await db.delete_documents(config.db_vectors_table)
            print("🗑️  已清空旧数据")

        await db.create_tables(
            config.db_documents_table,
            config.db_vectors_table,
            config.embedding_dimension,
        )
        await hybrid.setup_indexes(
            config.db_vector_index_name, config.db_text_index_name
        )
        await index_documents_from_folder(
            config.ingested_doc_dir,
            db,
            hybrid,
            config.db_documents_table,
            config.db_vectors_table,
        )
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="索引知识库文档")
    parser.add_argument("--clear", action="store_true", help="索引前清空已有数据")
    args = parser.parse_args()
    asyncio.run(main(clear=args.clear))
