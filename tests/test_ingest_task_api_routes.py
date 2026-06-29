"""异步入库任务接口测试。"""

from app.api.routes import ingest_tasks


class FakeIngestTaskService:
    """接口测试任务服务替身。"""

    def get_task(self, task_id: str):
        return {
            "task_id": task_id,
            "task_type": "document_ingest",
            "business_scene": "knowledge",
            "document_id": "doc_1",
            "batch_id": None,
            "task_status": "queued",
            "status": "queued",
            "current_step": "queued",
            "progress": 5,
            "attempt_count": 0,
            "max_attempts": 3,
            "error_message": None,
            "metadata": {},
            "created_at": None,
            "updated_at": None,
            "started_at": None,
            "finished_at": None,
        }

    def retry_task(self, task_id: str):
        task = self.get_task(task_id)
        task["task_status"] = "running"
        task["status"] = "running"
        return task


def test_get_ingest_task_route_returns_task_snapshot(monkeypatch):
    """任务查询路由应返回统一任务状态结构。"""

    monkeypatch.setattr(ingest_tasks, "_service", lambda: FakeIngestTaskService())

    response = ingest_tasks.get_ingest_task("task_1")

    assert response.task_id == "task_1"
    assert response.task_status == "queued"
    assert response.progress == 5


def test_retry_ingest_task_route_returns_updated_snapshot(monkeypatch):
    """任务重试路由应返回新的任务状态。"""

    monkeypatch.setattr(ingest_tasks, "_service", lambda: FakeIngestTaskService())

    response = ingest_tasks.retry_ingest_task("task_1")

    assert response.task_status == "running"

