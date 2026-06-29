"""异步入库任务服务测试。"""

import pytest
from fastapi import HTTPException

from app.application.ingest_task_service import IngestTaskService


class FakeTaskRepository:
    """记录异步任务状态变更的测试仓储。"""

    def __init__(self):
        self.created = []
        self.tasks = {}
        self.status_updates = []
        self.progress_updates = []

    def create_task(self, **values):
        task = {
            "task_id": values.get("task_id") or "task_1",
            "task_type": values["task_type"],
            "business_scene": values.get("business_scene"),
            "document_id": values.get("document_id"),
            "batch_id": values.get("batch_id"),
            "status": "queued",
            "current_step": "queued",
            "progress": 5,
            "attempt_count": 0,
            "max_attempts": values.get("max_attempts", 3),
            "error_message": None,
            "metadata_json": values.get("metadata_json"),
        }
        self.created.append(task)
        self.tasks[task["task_id"]] = task
        return task

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def list_tasks_by_status(self, statuses):
        return [task for task in self.tasks.values() if task["status"] in statuses]

    def mark_running(self, task_id):
        self.tasks[task_id]["status"] = "running"
        self.tasks[task_id]["attempt_count"] += 1
        self.status_updates.append((task_id, "running"))

    def update_progress(self, task_id, *, current_step, progress):
        self.tasks[task_id]["current_step"] = current_step
        self.tasks[task_id]["progress"] = progress
        self.progress_updates.append((task_id, current_step, progress))

    def mark_succeeded(self, task_id):
        self.tasks[task_id]["status"] = "succeeded"
        self.tasks[task_id]["current_step"] = "succeeded"
        self.tasks[task_id]["progress"] = 100
        self.status_updates.append((task_id, "succeeded"))

    def mark_failed(self, task_id, error_message):
        self.tasks[task_id]["status"] = "failed"
        self.tasks[task_id]["current_step"] = "failed"
        self.tasks[task_id]["progress"] = 100
        self.tasks[task_id]["error_message"] = error_message
        self.status_updates.append((task_id, "failed", error_message))

    def reset_for_retry(self, task_id):
        self.tasks[task_id]["status"] = "queued"
        self.tasks[task_id]["current_step"] = "queued"
        self.tasks[task_id]["progress"] = 5
        self.tasks[task_id]["error_message"] = None


def test_create_document_ingest_task_returns_queued_snapshot():
    """创建入库任务时应返回可供前端展示的排队状态。"""

    repository = FakeTaskRepository()
    service = IngestTaskService(task_repository=repository, auto_run=False)

    task = service.create_document_ingest_task(
        document_id="doc_1",
        collection_name="agent",
        document_type="text",
        split_strategy="recursive",
    )

    assert task["task_id"] == "task_1"
    assert task["task_status"] == "queued"
    assert task["current_step"] == "queued"
    assert task["progress"] == 5
    assert repository.created[0]["task_type"] == "document_ingest"


def test_run_task_marks_success_and_reports_progress():
    """任务执行成功时应记录运行、进度和成功状态。"""

    repository = FakeTaskRepository()
    calls = []

    def processor(task, reporter):
        calls.append(task["task_id"])
        reporter("parsing", 25)
        reporter("indexing", 90)

    service = IngestTaskService(
        task_repository=repository,
        processors={"document_ingest": processor},
        auto_run=False,
    )
    task = service.create_document_ingest_task(
        document_id="doc_1",
        collection_name="agent",
        document_type="text",
        split_strategy="recursive",
    )

    result = service.run_task(task["task_id"])

    assert calls == ["task_1"]
    assert result["task_status"] == "succeeded"
    assert result["progress"] == 100
    assert repository.status_updates == [("task_1", "running"), ("task_1", "succeeded")]
    assert repository.progress_updates == [("task_1", "parsing", 25), ("task_1", "indexing", 90)]


def test_run_task_marks_failed_when_processor_raises():
    """任务处理失败时应记录失败原因，供页面展示和重试。"""

    repository = FakeTaskRepository()

    def processor(task, reporter):
        reporter("parsing", 25)
        raise RuntimeError("解析失败")

    service = IngestTaskService(
        task_repository=repository,
        processors={"document_ingest": processor},
        auto_run=False,
    )
    task = service.create_document_ingest_task(
        document_id="doc_1",
        collection_name="agent",
        document_type="text",
        split_strategy="recursive",
    )

    result = service.run_task(task["task_id"])

    assert result["task_status"] == "failed"
    assert result["current_step"] == "failed"
    assert result["error_message"] == "解析失败"


def test_retry_task_rejects_queued_task():
    """手动重试只允许失败任务，排队任务不能重复提交后台线程。"""

    repository = FakeTaskRepository()
    task = repository.create_task(task_id="task_queued", task_type="document_ingest", business_scene="knowledge")
    service = IngestTaskService(task_repository=repository, auto_run=False)

    with pytest.raises(HTTPException) as exc_info:
        service.retry_task(task["task_id"])

    assert exc_info.value.status_code == 409
    assert repository.tasks["task_queued"]["status"] == "queued"


def test_resume_pending_tasks_marks_orphan_running_failed_and_submits_queued(monkeypatch):
    """服务重启后应恢复排队任务，并把遗留 running 任务标记为失败供人工重试。"""

    repository = FakeTaskRepository()
    repository.create_task(task_id="task_running", task_type="document_ingest", business_scene="knowledge")
    repository.tasks["task_running"]["status"] = "running"
    repository.tasks["task_running"]["current_step"] = "chunking"
    repository.create_task(task_id="task_queued", task_type="document_ingest", business_scene="knowledge")
    submitted = []

    monkeypatch.setattr("app.application.ingest_task_service._EXECUTOR.submit", lambda callback, task_id: submitted.append(task_id))
    service = IngestTaskService(task_repository=repository, auto_run=False)

    result = service.resume_pending_tasks()

    assert result == {"failed_orphan_running": 1, "submitted_queued": 1}
    assert repository.tasks["task_running"]["status"] == "failed"
    assert "服务重启" in repository.tasks["task_running"]["error_message"]
    assert submitted == ["task_queued"]
