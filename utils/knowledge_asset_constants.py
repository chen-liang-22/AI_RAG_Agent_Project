"""知识资产相关常量。

该模块只放跨层共享的稳定常量，避免仓储层和服务层互相 import 造成循环依赖。
"""

TRAINING_COLLECTION_NAMES = frozenset({
    "sales_training_cases",
    "sales_training_cases_staging",
})
