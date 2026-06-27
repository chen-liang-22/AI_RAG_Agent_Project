"""销售训练目标应用服务。"""

from training.schemas import GoalSettingGenerateRequest, GoalSettingResponse
from utils.logger_handler import logger

from .service_provider import get_training_core_service


class TrainingGoalApplicationService:
    """训练目标外观服务。"""

    def __init__(self, core_service=None):
        self._core_service = core_service
        self.service = None

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def generate_goal_setting(self, profile_id: str, request: GoalSettingGenerateRequest) -> GoalSettingResponse:
        """生成开放式训练目标和动态轮数。"""

        logger.info("[V2销售训练-目标] 生成训练目标 画像编号=%s 训练方式=%s", profile_id, request.training_mode)
        return self.core_service.generate_goal_setting(
            profile_id=profile_id,
            trainee_id=request.trainee_id,
            training_mode=request.training_mode,
            plan_id=request.plan_id,
            model_mode=request.model_mode,
        )
