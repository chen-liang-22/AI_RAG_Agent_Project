"""V2 首页驾驶舱接口。"""

from fastapi import APIRouter

from app.application.dashboard_service import DashboardApplicationService
from app.domain.schemas import DashboardOverviewResponse

router = APIRouter(prefix="/dashboard", tags=["V2 首页"])


@router.get("/overview", response_model=DashboardOverviewResponse)
def overview() -> DashboardOverviewResponse:
    """聚合首页需要的状态、知识库、训练和最近会话数据。"""

    return DashboardApplicationService().overview()
