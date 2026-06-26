"""销售训练会话应用服务。"""

from collections.abc import Iterator

from training.schemas import (
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)
from utils.logger_handler import logger

from .service_provider import get_sales_training_service


class TrainingSessionApplicationService:
    """训练会话外观服务。"""

    def __init__(self, service=None):
        self.service = service or get_sales_training_service()

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始训练会话。"""

        logger.info("[V2销售训练-会话] 开始训练会话 方案编号=%s 学员编号=%s", request.plan_id, request.trainee_id)
        return self.service.start_session(request)

    def list_sessions(self, *, page: int, page_size: int, trainee_id: str | None = None) -> TrainingSessionListResponse:
        """分页查询训练会话。"""

        return self.service.list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练会话复盘详情。"""

        return self.service.get_session_detail(session_id)

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回答。"""

        logger.info("[V2销售训练-会话] 提交训练回复 会话编号=%s 回复模式=一次性", session_id)
        return self.service.submit_turn(session_id, request)

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并流式返回 AI 客户回答。"""

        logger.info("[V2销售训练-会话] 提交训练回复 会话编号=%s 回复模式=流式", session_id)
        return self.service.stream_turn(session_id, request)
