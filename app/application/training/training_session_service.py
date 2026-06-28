"""销售训练会话聚合服务。"""

from __future__ import annotations

from collections.abc import Iterator

from app.application.training.training_session_basic_service import TrainingSessionBasicService
from app.application.training.training_session_scoring_service import TrainingSessionScoringService
from app.application.training.training_session_turn_service import TrainingSessionTurnService
from app.application.training_support.schemas import (
    TrainingScoreResponse,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)


class TrainingSessionService:
    """销售训练会话聚合服务。

    这里使用外观模式，把开始训练、一次性回复、流式回复、轮次保存、会话状态和评分入口协调到一个服务。
    """

    def __init__(
            self,
            *,
            basic_service: TrainingSessionBasicService,
            turn_service: TrainingSessionTurnService,
            scoring_service: TrainingSessionScoringService,
    ):
        """初始化训练会话聚合服务。"""

        self.basic_service = basic_service
        self.turn_service = turn_service
        self.scoring_service = scoring_service

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始训练会话。"""

        return self.basic_service.start_session(request)

    def list_sessions(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            trainee_id: str | None = None,
    ) -> TrainingSessionListResponse:
        """分页查询训练历史。"""

        return self.basic_service.list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练复盘详情。"""

        return self.basic_service.get_session_detail(session_id)

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回复。"""

        return self.turn_service.submit_turn(session_id, request)

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并返回 SSE 流。"""

        yield from self.turn_service.stream_turn(session_id, request)

    def final_score(self, session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。"""

        return self.scoring_service.final_score(session_id, model_mode=model_mode)
