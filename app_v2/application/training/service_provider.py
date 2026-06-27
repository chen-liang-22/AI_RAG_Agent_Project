"""销售训练服务提供器。"""

from functools import lru_cache

from app_v2.application.training.sales_training_core import V2SalesTrainingCoreService


@lru_cache(maxsize=1)
def get_training_core_service() -> V2SalesTrainingCoreService:
    """获取 V2 销售训练核心服务单例。

    这是拆分期的核心实现入口。各 V2 小服务先延迟访问它，避免构造阶段继续创建旧大类。
    """

    return V2SalesTrainingCoreService()
