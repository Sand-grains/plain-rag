"""Agent 管线入口脚本：索引 → 检索工具 → SimpleAgent → 交互式问答。

核心特性：
    - 使用 hello-agents 框架（HelloAgentsLLM + SimpleAgent + ToolRegistry）
    - 将 RAGSearchTool 注册为 agent 工具
    - 支持交互式终端问答循环
    - __main__ 守卫: import 不触发交互循环, 入口才 setup_logging 并跑管线

用法示例::

    uv run python agent_pipeline.py

公共接口：
    - main: 管线编排入口
"""

import logging
import sys
from pathlib import Path

from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, STORAGE_BACKEND, RERANKER_AGENT_ENABLED  # 先加载 .env，确保后续导入的库能读到环境变量
from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from indexing.loader import load
from indexing.router import Router
from preprocess import diagnose
from retrieval.embedding import embed
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from agent.tools import RAGSearchTool
from obs.logging_setup import setup_logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
## Role
你是一个专业的知识库问答助手, 你的任务是严格根据【参考文档】回答用户的问题

## 工作流程
1.当用户提问时, 首先调用 search_knowledge_base 工具检索相关文档片段
2.严格根据工具返回的上下文进行回答

## Rules(关键)
1.必须**仅依赖**工具返回的【参考文档】进行回答, 不要使用你内部的训练知识
2.如果检索结果中没有包含回答问题所需的信息, 请直接回答: "知识库中未找到相关信息". **严禁编造**
3.回答需要简洁, 逻辑清晰, 准确, 有条理, 分点描述
4.引用来源时标注[来源X], 在回答的末尾注明引用的文档名称
"""


def main() -> None:
    """Agent 管线编排入口：setup_logging → 索引 → 检索工具 → SimpleAgent → 交互式问答。"""
    setup_logging()
    data_dir = Path("data")
    docs = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in (".txt", ".md"):
            docs.extend(load(str(file_path)))

    if not docs:
        logger.warning("data/ 目录下没有找到 .txt 或 .md 文档，请放入测试文档后重试")
        sys.exit(1)

    logger.info("已加载 %d 篇文档", len(docs))

    store = IndexStore()
    router = Router()
    total_parents = 0
    total_children = 0
    for doc in docs:
        diagnosed_doc = diagnose(doc.content)
        splitter = router.route(diagnosed_doc)
        base_meta = {"doc_id": doc.doc_id, "doc_meta": doc.origin_metadata}
        result = splitter.split(doc.content, base_meta)
        parents, children = result if isinstance(result, tuple) else (result, result)
        if not parents:
            logger.warning(" [SKIP] %s: 空文档，跳过", doc.doc_id)
            continue
        child_vectors = embed([c.content for c in children])
        store.batch_add(parents, children, child_vectors)
        total_parents += len(parents)
        total_children += len(children)
        logger.info("  [OK] %s: %d 父块 / %d 子块（%s）", doc.doc_id, len(parents), len(children), type(splitter).__name__)

    store.vector_persistence()
    logger.info("入库完成（%s 模式）：%d 父块 / %d 子块", STORAGE_BACKEND, total_parents, total_children)

    # ==================== Agent 层 ====================

    llm = HelloAgentsLLM(
        model=LLM_MODEL_ID,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        provider="custom"
    )

    registry = ToolRegistry()
    retriever = Retriever(store, rerank_enabled=RERANKER_AGENT_ENABLED)   # agent 交互路径默认关 rerank（不常驻重排模型）
    registry.register_tool(RAGSearchTool(retriever))         # 将检索工具注册到工具注册表

    agent = SimpleAgent(
        name="专业知识库问答助手",
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        tool_registry=registry
    )

    # ==================== 交互循环 ====================
    logger.info("\n%s\nAgent 已就绪，输入问题开始对话（输入 exit 退出）\n%s\n", "=" * 50, "=" * 50)

    while True:
        query = input(">>> ")
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query.strip():
            continue
        answer = agent.run(query)
        logger.info("\n%s\n", answer)


if __name__ == "__main__":
    main()
