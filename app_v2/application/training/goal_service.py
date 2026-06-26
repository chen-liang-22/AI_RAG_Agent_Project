"""销售训练目标应用服务。"""

from training.schemas import GoalSettingGenerateRequest, GoalSettingResponse
from utils.logger_handler import logger

from .service_provider import get_sales_training_service


class TrainingGoalApplicationService:
    """训练目标外观服务。"""

    def __init__(self, service=None):
        self.service = service or get_sales_training_service()

    def generate_goal_setting(self, profile_id: str, request: GoalSettingGenerateRequest) -> GoalSettingResponse:
        """生成开放式训练目标和动态轮数。"""

        logger.info("[V2销售训练-目标] 生成训练目标 画像编号=%s 训练方式=%s", profile_id, request.training_mode)
        return self.service.generate_goal_setting(
            profile_id=profile_id,
            trainee_id=request.trainee_id,
            training_mode=request.training_mode,
            plan_id=request.plan_id,
            model_mode=request.model_mode,
        )
