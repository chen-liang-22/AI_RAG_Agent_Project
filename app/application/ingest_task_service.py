"""异步入库任务应用服务。"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import HTTPException

from app.infrastructure.repositories.ingest_task_repository import IngestTaskRepository
from core.utils.logger_handler import logger


TaskProcessor = Callable[[dict[str, Any], Callable[[str, int], None]], None]

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest-task")


class IngestTaskService:
    """异步入库任务外观服务。

    这里使用外观模式统一创建、执行、查询和重试任务；使用策略模式按 task_type
    分派具体处理器，避免上传服务直接关心每种入库细节。
    """

    def __init__(
            self,
            *,
            task_repository: Any | None = None,
            processors: dict[str, TaskProcessor] | None = None,
            auto_run: bool = True,
    ):
        """初始化任务服务。"""

        self.task_repository = task_repository or IngestTaskRepository()
        self.processors = processors or {}
        self.auto_run = auto_run

    @staticmethod
    def task_snapshot(task: dict[str, Any]) -> dict[str, Any]:
        """把任务实体或字典转换为接口可用的状态快照。"""

        metadata = task.get("metadata_json")
        if isinstance(metadata, str) and metadata.strip():
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        elif metadata is None:
            metadata = {}
        return {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "business_scene": task.get("business_scene"),
            "document_id": task.get("document_id"),
            "batch_id": task.get("batch_id"),
            "task_status": task.get("status"),
            "status": task.get("status"),
            "current_step": task.get("current_step"),
            "progress": int(task.get("progress") or 0),
            "attempt_count": int(task.get("attempt_count") or 0),
            "max_attempts": int(task.get("max_attempts") or 3),
            "error_message": task.get("error_message"),
            "metadata": metadata if isinstance(metadata, dict) else {},
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
        }

    def create_document_ingest_task(
            self,
            *,
            document_id: str,
            collection_name: str,
            document_type: str,
            split_strategy: str,
            auto_run: bool | None = None,
    ) -> dict[str, Any]:
        """创建普通知识库文件入库任务。"""

        task = self.task_repository.create_task(
            task_type="document_ingest",
            business_scene="knowledge",
            document_id=document_id,
            metadata={
                "collection_name": collection_name,
                "document_type": document_type,
                "split_strategy": split_strategy,
            },
        )
        snapshot = self.task_snapshot(task)
        self._submit_if_needed(snapshot["task_id"], auto_run=auto_run)
        return snapshot

    def create_training_ingest_task(
            self,
            *,
            document_id: str,
            batch_id: str,
            source_type: str,
            model_mode: str | None,
            auto_run: bool | None = None,
    ) -> dict[str, Any]:
        """创建销售训练资料入库任务。"""

        task = self.task_repository.create_task(
            task_type="training_ingest",
            business_scene="training",
            document_id=document_id,
            batch_id=batch_id,
            metadata={
                "source_type": source_type,
                "model_mode": model_mode,
            },
        )
        snapshot = self.task_snapshot(task)
        self._submit_if_needed(snapshot["task_id"], auto_run=auto_run)
        return snapshot

    def create_training_reparse_task(
            self,
            *,
            document_id: str,
            batch_id: str,
            source_type: str,
            use_llm_fallback: bool,
            model_mode: str | None,
            auto_run: bool | None = None,
    ) -> dict[str, Any]:
        """创建销售训练资料重新切分任务。"""

        task = self.task_repository.create_task(
            task_type="training_reparse",
            business_scene="training",
            document_id=document_id,
            batch_id=batch_id,
            metadata={
                "source_type": source_type,
                "use_llm_fallback": use_llm_fallback,
                "model_mode": model_mode,
            },
        )
        snapshot = self.task_snapshot(task)
        self._submit_if_needed(snapshot["task_id"], auto_run=auto_run)
        return snapshot

    def get_task(self, task_id: str) -> dict[str, Any]:
        """查询任务状态。"""

        task = self.task_repository.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"入库任务不存在：{task_id}")
        return self.task_snapshot(task)

    def retry_task(self, task_id: str) -> dict[str, Any]:
        """重试失败任务。"""

        task = self.task_repository.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"入库任务不存在：{task_id}")
        if task.get("status") not in {"failed", "queued"}:
            raise HTTPException(status_code=409, detail=f"当前任务状态不允许重试：{task.get('status')}")
        self.task_repository.reset_for_retry(task_id)
        self._submit_if_needed(task_id, auto_run=True)
        return self.get_task(task_id)

    def resume_pending_tasks(self) -> dict[str, int]:
        """恢复服务重启后遗留的异步任务。"""

        if not hasattr(self.task_repository, "list_tasks_by_status"):
            logger.warning("[异步入库] 当前任务仓储不支持启动恢复")
            return {"failed_orphan_running": 0, "submitted_queued": 0}

        orphan_running = self.task_repository.list_tasks_by_status({"running"})
        for task in orphan_running:
            self.task_repository.mark_failed(task["task_id"], "服务重启导致任务中断，请手动重试")

        queued_tasks = self.task_repository.list_tasks_by_status({"queued"})
        for task in queued_tasks:
            self._submit_if_needed(task["task_id"], auto_run=True)

        result = {
            "failed_orphan_running": len(orphan_running),
            "submitted_queued": len(queued_tasks),
        }
        logger.info(
            "[异步入库] 启动恢复完成 遗留运行任务=%s 已提交排队任务=%s",
            result["failed_orphan_running"],
            result["submitted_queued"],
        )
        return result

    def run_task(self, task_id: str) -> dict[str, Any]:
        """同步执行一个任务，主要给后台线程和测试调用。"""

        task = self.task_repository.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"入库任务不存在：{task_id}")
        task_type = task["task_type"]
        processor = self.processors.get(task_type)
        if processor is None:
            from app.application.ingest_task_processors import build_ingest_task_processors

            processor = build_ingest_task_processors().get(task_type)
        if processor is None:
            raise HTTPException(status_code=400, detail=f"未配置入库任务处理器：{task_type}")

        logger.info("[异步入库] 任务开始 任务编号=%s 任务类型=%s", task_id, task_type)
        self.task_repository.mark_running(task_id)

        def reporter(current_step: str, progress: int) -> None:
            self.task_repository.update_progress(task_id, current_step=current_step, progress=progress)

        try:
            processor(self.task_snapshot(task), reporter)
            self.task_repository.mark_succeeded(task_id)
            logger.info("[异步入库] 任务完成 任务编号=%s 任务类型=%s", task_id, task_type)
        except Exception as exc:
            message = str(getattr(exc, "detail", None) or exc)
            self.task_repository.mark_failed(task_id, message)
            logger.error("[异步入库] 任务失败 任务编号=%s 错误=%s", task_id, message, exc_info=True)
        return self.get_task(task_id)

    def _submit_if_needed(self, task_id: str, *, auto_run: bool | None) -> None:
        """根据配置决定是否提交后台线程。"""

        should_auto_run = self.auto_run if auto_run is None else auto_run
        if not should_auto_run:
            return
        _EXECUTOR.submit(self.run_task, task_id)
