"""异步入库任务仓储。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import inspect, select, text

from app.application.training_support.repository import utc_now
from app.domain.entities import IngestTaskEntity
from app.infrastructure.id_generator import new_id
from app.infrastructure.orm_session import orm_session_context


class IngestTaskRepository:
    """封装 ingest_tasks 表访问。"""

    def __init__(self):
        """初始化任务仓储并确保任务表存在。"""

        self.ensure_table()

    @staticmethod
    def ensure_table() -> None:
        """确保 ingest_tasks 表存在。

        项目主初始化仍以 SQL 文件为准；这里做轻量兜底，避免旧库未执行新 SQL 时上传直接失败。
        """

        with orm_session_context() as session:
            inspector = inspect(session.get_bind())
            if "ingest_tasks" in inspector.get_table_names():
                return
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS ingest_tasks (
                  task_id VARCHAR(64) NOT NULL COMMENT '入库任务编号',
                  task_type VARCHAR(64) NOT NULL COMMENT '任务类型',
                  business_scene VARCHAR(64) NULL COMMENT '业务场景',
                  document_id VARCHAR(64) NULL COMMENT '关联 documents.document_id',
                  batch_id VARCHAR(64) NULL COMMENT '关联训练资料批次编号',
                  status VARCHAR(32) NOT NULL COMMENT '任务状态',
                  current_step VARCHAR(64) NOT NULL COMMENT '当前处理步骤',
                  progress INT NOT NULL DEFAULT 5 COMMENT '处理进度，0到100',
                  attempt_count INT NOT NULL DEFAULT 0 COMMENT '已尝试次数',
                  max_attempts INT NOT NULL DEFAULT 3 COMMENT '最大尝试次数',
                  error_message TEXT NULL COMMENT '失败原因',
                  metadata_json JSON NULL COMMENT '任务扩展参数 JSON',
                  started_at DATETIME NULL COMMENT '开始处理时间',
                  finished_at DATETIME NULL COMMENT '处理完成时间',
                  created_at DATETIME NOT NULL COMMENT '创建时间',
                  updated_at DATETIME NOT NULL COMMENT '更新时间',
                  PRIMARY KEY (task_id),
                  KEY idx_ingest_tasks_document (document_id, created_at),
                  KEY idx_ingest_tasks_batch (batch_id, created_at),
                  KEY idx_ingest_tasks_status (status, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='异步入库任务表'
            """))

    @staticmethod
    def _json(data: Any) -> str | None:
        """序列化任务扩展参数。"""

        if data is None:
            return None
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False)

    def create_task(self, **values: Any) -> IngestTaskEntity:
        """创建排队中的入库任务。"""

        now = utc_now()
        task_id = values.get("task_id") or new_id()
        task = IngestTaskEntity(
            task_id=task_id,
            task_type=values["task_type"],
            business_scene=values.get("business_scene"),
            document_id=values.get("document_id"),
            batch_id=values.get("batch_id"),
            status="queued",
            current_step="queued",
            progress=5,
            attempt_count=0,
            max_attempts=int(values.get("max_attempts") or 3),
            error_message=None,
            metadata_json=self._json(values.get("metadata") or values.get("metadata_json")),
            started_at=None,
            finished_at=None,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(task)
        created = self.get_task(task_id)
        if created is None:
            raise RuntimeError(f"入库任务创建失败：{task_id}")
        return created

    def get_task(self, task_id: str) -> IngestTaskEntity | None:
        """按任务编号查询任务。"""

        with orm_session_context() as session:
            return session.get(IngestTaskEntity, task_id)

    def get_latest_task_for_document(self, document_id: str) -> IngestTaskEntity | None:
        """查询指定文件最近一次入库任务。"""

        statement = (
            select(IngestTaskEntity)
            .where(IngestTaskEntity.document_id == document_id)
            .order_by(IngestTaskEntity.created_at.desc())
            .limit(1)
        )
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def get_latest_task_for_batch(self, batch_id: str) -> IngestTaskEntity | None:
        """查询指定训练批次最近一次入库任务。"""

        statement = (
            select(IngestTaskEntity)
            .where(IngestTaskEntity.batch_id == batch_id)
            .order_by(IngestTaskEntity.created_at.desc())
            .limit(1)
        )
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def list_tasks_by_status(self, statuses: set[str]) -> list[IngestTaskEntity]:
        """按状态查询任务列表，用于服务重启后的任务恢复。"""

        if not statuses:
            return []
        statement = (
            select(IngestTaskEntity)
            .where(IngestTaskEntity.status.in_(sorted(statuses)))
            .order_by(IngestTaskEntity.created_at.asc())
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def mark_running(self, task_id: str) -> None:
        """标记任务开始运行。"""

        with orm_session_context() as session:
            task = session.get(IngestTaskEntity, task_id)
            if task is None:
                raise ValueError(f"入库任务不存在：{task_id}")
            task.status = "running"
            task.current_step = "running"
            task.progress = max(int(task.progress or 0), 10)
            task.attempt_count = int(task.attempt_count or 0) + 1
            task.error_message = None
            task.started_at = utc_now()
            task.finished_at = None
            task.updated_at = utc_now()

    def update_progress(self, task_id: str, *, current_step: str, progress: int) -> None:
        """更新任务进度。"""

        safe_progress = max(0, min(100, int(progress)))
        with orm_session_context() as session:
            task = session.get(IngestTaskEntity, task_id)
            if task is None:
                raise ValueError(f"入库任务不存在：{task_id}")
            task.current_step = current_step
            task.progress = safe_progress
            task.updated_at = utc_now()

    def mark_succeeded(self, task_id: str) -> None:
        """标记任务成功。"""

        with orm_session_context() as session:
            task = session.get(IngestTaskEntity, task_id)
            if task is None:
                raise ValueError(f"入库任务不存在：{task_id}")
            task.status = "succeeded"
            task.current_step = "succeeded"
            task.progress = 100
            task.error_message = None
            task.finished_at = utc_now()
            task.updated_at = utc_now()

    def mark_failed(self, task_id: str, error_message: str) -> None:
        """标记任务失败并保存错误原因。"""

        with orm_session_context() as session:
            task = session.get(IngestTaskEntity, task_id)
            if task is None:
                raise ValueError(f"入库任务不存在：{task_id}")
            task.status = "failed"
            task.current_step = "failed"
            task.progress = 100
            task.error_message = error_message
            task.finished_at = utc_now()
            task.updated_at = utc_now()

    def reset_for_retry(self, task_id: str) -> None:
        """把失败任务重置为排队状态。"""

        with orm_session_context() as session:
            task = session.get(IngestTaskEntity, task_id)
            if task is None:
                raise ValueError(f"入库任务不存在：{task_id}")
            task.status = "queued"
            task.current_step = "queued"
            task.progress = 5
            task.error_message = None
            task.started_at = None
            task.finished_at = None
            task.updated_at = utc_now()
