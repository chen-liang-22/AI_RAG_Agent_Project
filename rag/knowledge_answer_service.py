"""兼容旧导入路径的知识直答服务入口。

真实实现已迁移到 ``rag.services.knowledge_answer_service``，保留本模块是为了减少一次性改动范围。
"""

from rag.services.knowledge_answer_service import KnowledgeAnswerService

__all__ = ["KnowledgeAnswerService"]

