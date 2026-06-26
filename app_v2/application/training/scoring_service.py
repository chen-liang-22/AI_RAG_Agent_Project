"""销售训练评分应用服务。"""

from training.schemas import TrainingScoreResponse
from utils.logger_handler import logger

from .service_provider import get_sales_training_service


class TrainingScoringApplicationService:
    """训练评分外观服务。"""

    def __init__(self, service=None):
        self.service = service or get_sales_training_service()

    def final_score(self, session_id: str, *, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。"""

        logger.info("[V2销售训练-评分] 生成最终评分 会话编号=%s 模型模式=%s", session_id, model_mode)
        return self.service.final_score(session_id, model_mode=model_mode)
