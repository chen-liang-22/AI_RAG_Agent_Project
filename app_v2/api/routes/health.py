"""V2 健康检查接口。"""

from fastapi import APIRouter

from api.schemas import HealthResponse
from app_v2.application.dashboard_service import DashboardApplicationService

router = APIRouter(tags=["V2 健康检查"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """返回系统依赖状态。"""

    return DashboardApplicationService().health()
