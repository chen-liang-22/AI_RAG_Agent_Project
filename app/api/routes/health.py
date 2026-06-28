"""V2 健康检查接口。"""

from fastapi import APIRouter

from api.schemas import HealthDependenciesResponse, HealthResponse
from app.application.observability.health_service import HealthDependencyService
from app.application.dashboard_service import DashboardApplicationService

router = APIRouter(tags=["V2 健康检查"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """返回系统依赖状态。"""

    return DashboardApplicationService().health()


@router.get("/health/dependencies", response_model=HealthDependenciesResponse)
def health_dependencies() -> HealthDependenciesResponse:
    """返回系统关键依赖的健康检查明细。"""

    return HealthDependencyService().check_dependencies()
