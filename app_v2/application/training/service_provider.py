"""销售训练服务提供器。"""

from functools import lru_cache

from rag.knowledge_store import KnowledgeStore
from training.services.sales_training_service import SalesTrainingService


@lru_cache(maxsize=1)
def get_sales_training_service() -> SalesTrainingService:
    """获取销售训练旧服务单例。

    这是过渡期外观：V2 模块先依赖这个提供器，后续再把旧大服务逐步拆成真正的小服务。
    """

    return SalesTrainingService()


def get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据存储实例，用于读取系统字典。"""

    return KnowledgeStore()
