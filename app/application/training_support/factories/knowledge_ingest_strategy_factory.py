"""训练资料入库策略工厂。

上传训练资料时，应用服务只传 source_type。
工厂负责选择 LMS 案例策略或通用策略，避免主流程出现一堆 if/else。
"""

from app.application.training_support.strategies.knowledge_ingest_strategy import (
    GenericTrainingIngestStrategy,
    KnowledgeIngestStrategy,
    LmsCaseIngestStrategy,
)


class KnowledgeIngestStrategyFactory:
    """训练知识入库策略工厂。

    使用工厂方法是为了让上传接口只关心 source_type，
    具体切片规则交给对应策略对象。
    """

    @staticmethod
    def create(source_type: str) -> KnowledgeIngestStrategy:
        """根据来源类型创建对应切片策略。

        这里是工厂方法模式：
        - 调用方不直接 new LmsCaseIngestStrategy；
        - 调用方只传 source_type；
        - 工厂负责决定返回哪个策略对象。
        """

        normalized_type = (source_type or "").strip().lower()
        if normalized_type == "lms_case":
            return LmsCaseIngestStrategy()
        return GenericTrainingIngestStrategy()
