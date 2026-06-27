"""兼容旧导入路径的向量库服务入口。

真实实现已迁移到 ``infrastructure.vector_store_service``，保留本模块是为了让旧代码可以继续运行。
"""

from app_v2.infrastructure.vector_store_service import VectorStoreService

__all__ = ["VectorStoreService"]
