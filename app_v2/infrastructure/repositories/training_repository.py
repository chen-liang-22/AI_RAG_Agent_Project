"""V2 销售训练仓储。

第一阶段继承旧仓储，先稳定接口；后续把自动建表和业务 SQL 拆到本模块。
"""

from training.repository import TrainingRepository


class V2TrainingRepository(TrainingRepository):
    """销售训练仓储适配器。

    这个类让 V2 应用服务只依赖 V2 路径，后面替换底层实现时不影响路由层。
    """
