"""Agent 工具注册。

旧 Agent 工具链会从这里拿到 rag_summarize 工具。
新版普通聊天默认直连 RAG，但保留该工具用于兼容 Agent 模式和报告生成场景。
"""

from langchain_core.tools import tool

from core.rag.services.rag_service import RagSummarizeService

rag = RagSummarizeService()

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """从向量库检索参考资料，并返回给 Agent 作为上下文。"""

    return rag.rag_summarize(query)
