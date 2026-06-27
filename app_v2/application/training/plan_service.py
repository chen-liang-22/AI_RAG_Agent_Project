"""销售训练方案应用服务。"""

from app_v2.application.training_support.schemas import TrainingPlanCreateRequest, TrainingPlanDeleteResponse, TrainingPlanDetailResponse, TrainingPlanListResponse, TrainingPlanUpdateRequest
from core.utils.logger_handler import logger

from .service_provider import get_training_core_service


class TrainingPlanApplicationService:
    """训练方案外观服务。"""

    def __init__(self, core_service=None):
        """初始化训练方案服务。

        训练方案本身是销售陪练的主线对象，这里只做代理，不直接访问数据库。
        """

        self._core_service = core_service
        self.service = None

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def create_plan(self, request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
        """创建训练方案。"""

        logger.info("[V2销售训练-方案] 创建训练方案 名称=%s", request.plan_name)
        return self.core_service.create_plan(request)

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None) -> TrainingPlanListResponse:
        """分页查询训练方案。"""

        return self.core_service.list_plans(page=page, page_size=page_size, keyword=keyword)

    def get_plan_detail(self, plan_id: str) -> TrainingPlanDetailResponse:
        """查看训练方案详情。"""

        return self.core_service.get_plan_detail(plan_id)

    def delete_plan(self, plan_id: str) -> TrainingPlanDeleteResponse:
        """删除训练方案。"""

        logger.info("[V2销售训练 方案] 删除训练方案 方案编号=%s", plan_id)
        return self.core_service.delete_plan(plan_id)

    def update_plan(self, plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
        """修改训练方案。"""

        logger.info("[V2销售训练-方案] 修改训练方案 方案编号=%s", plan_id)
        return self.core_service.update_plan(plan_id, request)
