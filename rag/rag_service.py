"""兼容旧导入路径的 RAG 检索服务入口。

真实实现已迁移到 ``rag.services.rag_service``，保留本模块是为了让历史代码和测试可以平滑过渡。
"""

from rag.services.rag_service import RagSummarizeService, RetrievalQuality

__all__ = ["RagSummarizeService", "RetrievalQuality"]

