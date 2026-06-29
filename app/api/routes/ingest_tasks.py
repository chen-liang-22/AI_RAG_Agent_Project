"""异步入库任务接口。"""

from fastapi import APIRouter

from api.schemas import IngestTaskResponse
from app.application.ingest_task_service import IngestTaskService


router = APIRouter(prefix="/ingest-tasks", tags=["V2 异步入库任务"])


def _service() -> IngestTaskService:
    """创建异步入库任务服务。"""

    return IngestTaskService()


@router.get("/{task_id}", response_model=IngestTaskResponse)
def get_ingest_task(task_id: str) -> IngestTaskResponse:
    """查询异步入库任务状态。"""

    return IngestTaskResponse(**_service().get_task(task_id))


@router.post("/{task_id}/retry", response_model=IngestTaskResponse)
def retry_ingest_task(task_id: str) -> IngestTaskResponse:
    """重试失败的异步入库任务。"""

    return IngestTaskResponse(**_service().retry_task(task_id))

