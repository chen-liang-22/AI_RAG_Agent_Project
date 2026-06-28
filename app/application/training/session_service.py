"""销售训练会话应用服务。"""

from collections.abc import Iterator

from app.application.training_support.schemas import (
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)
from core.utils.logger_handler import logger

from .service_provider import get_training_core_service


class TrainingSessionApplicationService:
    """训练会话外观服务。"""

    def __init__(self, core_service=None):
        """初始化训练会话服务。

        会话支持一次性和流式两种回复模式，具体业务仍交给核心服务。
        """

        self._core_service = core_service
        self.service = None

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始训练会话。"""

        logger.info(
            "[V2销售训练-会话] 开始训练会话 角色编号=%s 设置编号=%s 学员编号=%s",
            request.profile_id,
            request.setting_id,
            request.trainee_id,
        )
        return self.core_service.start_session(request)

    def list_sessions(self, *, page: int, page_size: int, trainee_id: str | None = None) -> TrainingSessionListResponse:
        """分页查询训练会话。"""

        logger.info("[V2销售训练-会话] 查询训练会话列表 页码=%s 每页数量=%s 学员编号=%s", page, page_size, trainee_id)
        return self.core_service.list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练会话复盘详情。"""

        logger.info("[V2销售训练-会话] 查询训练会话详情 会话编号=%s", session_id)
        return self.core_service.get_session_detail(session_id)

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回答。"""

        logger.info("[V2销售训练-会话] 提交训练回复 会话编号=%s 回复模式=一次性", session_id)
        return self.core_service.submit_turn(session_id, request)

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并流式返回 AI 客户回答。"""

        logger.info("[V2销售训练-会话] 提交训练回复 会话编号=%s 回复模式=流式", session_id)
        return self.core_service.stream_turn(session_id, request)
