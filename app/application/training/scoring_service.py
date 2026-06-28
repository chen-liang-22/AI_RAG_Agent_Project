"""销售训练评分应用服务。"""

from app.application.training_support.schemas import TrainingScoreResponse
from core.utils.logger_handler import logger

from .service_provider import get_training_core_service


class TrainingScoringApplicationService:
    """训练评分外观服务。"""

    def __init__(self, core_service=None):
        """初始化训练评分服务。

        评分依赖完整训练会话和训练目标，所以这里只转发到核心服务统一处理。
        """

        self._core_service = core_service
        self.service = None

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def final_score(self, session_id: str, *, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。"""

        logger.info("[V2销售训练-评分] 生成最终评分 会话编号=%s 模型模式=%s", session_id, model_mode)
        return self.core_service.final_score(session_id, model_mode=model_mode)
