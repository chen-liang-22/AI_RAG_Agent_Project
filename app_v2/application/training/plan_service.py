"""销售训练方案应用服务。"""

from training.schemas import TrainingPlanCreateRequest, TrainingPlanDeleteResponse, TrainingPlanDetailResponse, TrainingPlanListResponse, TrainingPlanUpdateRequest
from utils.logger_handler import logger

from .service_provider import get_sales_training_service


class TrainingPlanApplicationService:
    """训练方案外观服务。"""

    def __init__(self, service=None):
        self.service = service or get_sales_training_service()

    def create_plan(self, request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
        """创建训练方案。"""

        logger.info("[V2销售训练-方案] 创建训练方案 名称=%s", request.plan_name)
        return self.service.create_plan(request)

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None) -> TrainingPlanListResponse:
        """分页查询训练方案。"""

        return self.service.list_plans(page=page, page_size=page_size, keyword=keyword)

    def get_plan_detail(self, plan_id: str) -> TrainingPlanDetailResponse:
        """查看训练方案详情。"""

        return self.service.get_plan_detail(plan_id)

    def delete_plan(self, plan_id: str) -> TrainingPlanDeleteResponse:
        """删除训练方案。"""

        logger.info("[V2销售训练 方案] 删除训练方案 方案编号=%s", plan_id)
        return self.service.delete_plan(plan_id)

    def update_plan(self, plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
        """修改训练方案。"""

        logger.info("[V2销售训练-方案] 修改训练方案 方案编号=%s", plan_id)
        return self.service.update_plan(plan_id, request)
