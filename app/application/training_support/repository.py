"""销售训练关系型数据仓储。

这个模块封装销售训练相关表的读写：
- 训练资料批次和版本；
- 训练切片元数据；
- 角色画像、训练设置、训练方案；
- 训练会话、轮次记录和评分报告。

应用服务只调用仓储方法，不直接写 SQL，方便后续把表结构和查询优化集中处理。
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import func, inspect, or_, select, text
from sqlalchemy.orm import Session

from app.domain.entities import (
    DocumentEntity,
    SalesTrainingScoreEntity,
    SalesTrainingSessionEntity,
    SalesTrainingTurnEntity,
    TrainingGoalSettingEntity,
    TrainingKnowledgeBatchEntity,
    TrainingPlanEntity,
    TrainingRoleProfileEntity,
)
from app.infrastructure.id_generator import new_id
from app.infrastructure.orm_session import orm_session_context
from core.utils.logger_handler import logger


def utc_now_text() -> str:
    """返回统一格式的 UTC 时间字符串。"""

    return datetime.utcnow().isoformat(timespec="seconds", sep=" ")


def utc_now() -> datetime:
    """返回去掉微秒的 UTC 时间，便于写入 MySQL DATETIME 字段。"""

    return datetime.utcnow().replace(microsecond=0)


def to_datetime(value: Any) -> datetime | None:
    """把 service 层传入的时间字符串转换为 ORM DateTime 可写入的对象。"""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(microsecond=0)
    if isinstance(value, str):
        clean_value = value.strip()
        if not clean_value:
            return None
        try:
            return datetime.fromisoformat(clean_value).replace(microsecond=0)
        except ValueError:
            return utc_now()
    return utc_now()


class TrainingRepository:
    """销售训练业务仓储。

    这个类是训练域访问 MySQL 的唯一入口，职责类似 Java 项目里的
    Service 依赖 Mapper/Repository。所有关系型数据库读写都通过
    SQLAlchemy ORM Entity 完成，不再手写 cursor SQL。
    """

    def __init__(self):
        """初始化训练仓储并确保增量字段存在。

        这里会做轻量级兼容 DDL，保证旧库升级后也能运行新版训练资料流程。
        """

        # 表结构主来源仍是 docs/mysql初始化建表和基础数据.sql。
        # 这里仅补齐轻量兼容字段，避免旧库没执行迁移 SQL 时一启动就因为 document_id 字段缺失失败。
        self.ensure_training_batch_document_columns()

    @staticmethod
    def ensure_training_batch_document_columns() -> None:
        """确保训练批次表具备关联 documents 的字段和索引。

        老库可能已经有 training_knowledge_batches，但没有 document_id。
        新代码的查询会 outer join documents；如果字段不存在，列表、上传、去重都会报 SQL 错。
        这里做幂等补齐，真正的历史数据回填仍由 scripts/migrate_local_files_to_minio.py 完成。
        """

        with orm_session_context() as session:
            bind = session.get_bind()
            inspector = inspect(bind)
            columns = {column["name"] for column in inspector.get_columns("training_knowledge_batches")}
            if "document_id" not in columns:
                session.execute(text(
                    "ALTER TABLE training_knowledge_batches "
                    "ADD COLUMN document_id VARCHAR(64) NULL "
                    "COMMENT '关联 documents.document_id，文件基础信息统一保存在 documents 表' AFTER batch_id"
                ))
                logger.info("[销售训练] training_knowledge_batches.document_id 字段已自动补齐")

            indexes = {index["name"] for index in inspector.get_indexes("training_knowledge_batches")}
            if "idx_training_batches_document" not in indexes:
                session.execute(text(
                    "CREATE INDEX idx_training_batches_document ON training_knowledge_batches(document_id)"
                ))
                logger.info("[销售训练] training_knowledge_batches.document_id 索引已自动补齐")

    @staticmethod
    def _json(data: Any) -> str:
        """把 Python 对象序列化为数据库 JSON 字符串。"""

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _enrich_batch_with_document(
            batch: TrainingKnowledgeBatchEntity,
            document: DocumentEntity | None,
    ) -> TrainingKnowledgeBatchEntity:
        """把 documents 表的文件信息挂到训练批次实体上。

        这些 document_* 字段不是 training_knowledge_batches 表字段，
        只是列表和预览接口的只读展示字段，所以作为实体临时属性附加。
        """

        batch.document_filename = document.filename if document else None
        batch.document_file_path = document.file_path if document else None
        batch.document_file_md5 = document.file_md5 if document else None
        batch.document_file_type = document.file_type if document else None
        batch.document_file_size = document.file_size if document else None
        batch.document_bucket_name = document.bucket_name if document else None
        batch.document_object_name = document.object_name if document else None
        batch.document_public_url = document.public_url if document else None
        batch.document_status = document.status if document else None
        return batch

    @staticmethod
    def _batch_select_statement():
        """构造训练批次和文档信息的 ORM join 查询。"""

        return select(TrainingKnowledgeBatchEntity, DocumentEntity).outerjoin(
            DocumentEntity,
            DocumentEntity.document_id == TrainingKnowledgeBatchEntity.document_id,
        )

    @classmethod
    def _scalars_to_batches(
            cls,
            rows: list[tuple[TrainingKnowledgeBatchEntity, DocumentEntity | None]],
    ) -> list[TrainingKnowledgeBatchEntity]:
        """把 ORM join 结果转换成带文件展示信息的训练批次实体列表。"""

        return [cls._enrich_batch_with_document(batch, document) for batch, document in rows]

    def create_batch(self, **values: Any) -> TrainingKnowledgeBatchEntity:
        """创建训练知识上传批次。"""

        now = utc_now()
        batch_id = values.get("batch_id") or new_id()
        batch = TrainingKnowledgeBatchEntity(
            batch_id=batch_id,
            document_id=values.get("document_id"),
            source_type=values.get("source_type"),
            source_file=values.get("source_file"),
            file_path=values.get("file_path"),
            file_md5=values.get("file_md5"),
            version_group_id=values.get("version_group_id") or batch_id,
            version_no=int(values.get("version_no") or 1),
            previous_batch_id=values.get("previous_batch_id"),
            is_current=int(bool(values.get("is_current"))),
            profile_type=values.get("profile_type"),
            task_type=values.get("task_type"),
            industry=values.get("industry"),
            difficulty=values.get("difficulty"),
            visibility_default=values.get("visibility_default"),
            status=values.get("status", "uploaded"),
            chunk_count=int(values.get("chunk_count") or 0),
            point_count=int(values.get("point_count") or 0),
            error_message=values.get("error_message"),
            quality_report_json=self._json(values.get("quality_report")) if values.get("quality_report") is not None else None,
            created_by=values.get("created_by"),
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(batch)
        created = self.get_batch(batch_id)
        if created is None:
            raise RuntimeError(f"训练资料批次创建失败：{batch_id}")
        return created

    def update_batch_status(
            self,
            batch_id: str,
            *,
            status: str,
            chunk_count: int | None = None,
            point_count: int | None = None,
            error_message: str | None = None,
            quality_report: dict[str, Any] | None = None,
            is_current: bool | None = None,
    ) -> None:
        """更新训练知识上传批次状态和统计信息。"""

        with orm_session_context() as session:
            batch = session.get(TrainingKnowledgeBatchEntity, batch_id)
            if batch is None:
                return
            batch.status = status
            if chunk_count is not None:
                batch.chunk_count = int(chunk_count)
            if point_count is not None:
                batch.point_count = int(point_count)
            batch.error_message = error_message
            if quality_report is not None:
                batch.quality_report_json = self._json(quality_report)
            if is_current is not None:
                batch.is_current = int(is_current)
            batch.updated_at = utc_now()

    def get_latest_batch_for_version(self, *, source_type: str, source_file: str) -> TrainingKnowledgeBatchEntity | None:
        """按资料类型和文件名查询最新版本批次。"""

        statement = (
            self._batch_select_statement()
            .where(
                TrainingKnowledgeBatchEntity.source_type == source_type,
                TrainingKnowledgeBatchEntity.source_file == source_file,
                TrainingKnowledgeBatchEntity.status != "deleted",
            )
            .order_by(
                TrainingKnowledgeBatchEntity.version_no.desc(),
                TrainingKnowledgeBatchEntity.updated_at.desc(),
                TrainingKnowledgeBatchEntity.created_at.desc(),
            )
            .limit(1)
        )
        with orm_session_context() as session:
            row = session.execute(statement).one_or_none()
        if row is None:
            return None
        return self._enrich_batch_with_document(row[0], row[1])

    def list_current_published_batch_ids(self) -> list[str]:
        """查询当前参与训练检索的已发布批次编号。"""

        statement = (
            select(TrainingKnowledgeBatchEntity.batch_id)
            .where(
                TrainingKnowledgeBatchEntity.status == "published",
                TrainingKnowledgeBatchEntity.is_current == 1,
            )
            .order_by(
                TrainingKnowledgeBatchEntity.updated_at.desc(),
                TrainingKnowledgeBatchEntity.created_at.desc(),
            )
        )
        with orm_session_context() as session:
            return [str(batch_id) for batch_id in session.scalars(statement).all()]

    def list_published_batches_in_version_group(
            self,
            version_group_id: str,
            *,
            exclude_batch_id: str | None = None,
    ) -> list[TrainingKnowledgeBatchEntity]:
        """查询同一版本组内已发布或已归档的批次。"""

        conditions = [
            TrainingKnowledgeBatchEntity.version_group_id == version_group_id,
            TrainingKnowledgeBatchEntity.status.in_(["published", "archived"]),
        ]
        if exclude_batch_id:
            conditions.append(TrainingKnowledgeBatchEntity.batch_id != exclude_batch_id)
        statement = (
            self._batch_select_statement()
            .where(*conditions)
            .order_by(
                TrainingKnowledgeBatchEntity.version_no.desc(),
                TrainingKnowledgeBatchEntity.updated_at.desc(),
            )
        )
        with orm_session_context() as session:
            rows = session.execute(statement).all()
        return self._scalars_to_batches(rows)

    def list_batches_in_version_group(self, version_group_id: str) -> list[TrainingKnowledgeBatchEntity]:
        """查询同一版本组内的全部未删除批次。"""

        statement = (
            self._batch_select_statement()
            .where(
                TrainingKnowledgeBatchEntity.version_group_id == version_group_id,
                TrainingKnowledgeBatchEntity.status != "deleted",
            )
            .order_by(
                TrainingKnowledgeBatchEntity.version_no.desc(),
                TrainingKnowledgeBatchEntity.updated_at.desc(),
                TrainingKnowledgeBatchEntity.created_at.desc(),
            )
        )
        with orm_session_context() as session:
            rows = session.execute(statement).all()
        return self._scalars_to_batches(rows)

    def archive_other_versions(self, *, version_group_id: str, current_batch_id: str) -> None:
        """把同版本组内非当前版本标记为归档。"""

        statement = select(TrainingKnowledgeBatchEntity).where(
            TrainingKnowledgeBatchEntity.version_group_id == version_group_id,
            TrainingKnowledgeBatchEntity.batch_id != current_batch_id,
            TrainingKnowledgeBatchEntity.status == "published",
        )
        with orm_session_context() as session:
            for batch in session.scalars(statement):
                batch.status = "archived"
                batch.is_current = 0
                batch.updated_at = utc_now()

    def get_published_batch_by_md5(self, file_md5: str) -> TrainingKnowledgeBatchEntity | None:
        """按文件 MD5 查询已经成功入库的训练资料。"""

        statement = (
            self._batch_select_statement()
            .where(
                TrainingKnowledgeBatchEntity.status == "published",
                or_(
                    DocumentEntity.file_md5 == file_md5,
                    TrainingKnowledgeBatchEntity.file_md5 == file_md5,
                ),
            )
            .order_by(
                TrainingKnowledgeBatchEntity.updated_at.desc(),
                TrainingKnowledgeBatchEntity.created_at.desc(),
            )
            .limit(1)
        )
        with orm_session_context() as session:
            row = session.execute(statement).one_or_none()
        if row is None:
            return None
        return self._enrich_batch_with_document(row[0], row[1])

    def get_existing_batch_by_md5(self, file_md5: str) -> TrainingKnowledgeBatchEntity | None:
        """按文件 MD5 查询任意未删除训练资料批次，用于上传前去重。"""

        statement = (
            self._batch_select_statement()
            .where(
                TrainingKnowledgeBatchEntity.status != "deleted",
                or_(
                    DocumentEntity.file_md5 == file_md5,
                    TrainingKnowledgeBatchEntity.file_md5 == file_md5,
                ),
            )
            .order_by(
                TrainingKnowledgeBatchEntity.updated_at.desc(),
                TrainingKnowledgeBatchEntity.created_at.desc(),
            )
            .limit(1)
        )
        with orm_session_context() as session:
            row = session.execute(statement).one_or_none()
        if row is None:
            return None
        return self._enrich_batch_with_document(row[0], row[1])

    def list_batches(
            self,
            *,
            page: int,
            page_size: int,
            keyword: str | None = None,
    ) -> tuple[list[TrainingKnowledgeBatchEntity], int]:
        """分页查询训练资料上传批次。

        keyword 只做文件名模糊查询，既兼容历史 source_file 字段，
        也兼容新文件台账 documents.filename 字段。
        """

        offset = (page - 1) * page_size
        clean_keyword = (keyword or "").strip()
        conditions = [TrainingKnowledgeBatchEntity.status != "deleted"]
        if clean_keyword:
            like_keyword = f"%{clean_keyword}%"
            conditions.append(or_(
                TrainingKnowledgeBatchEntity.source_file.like(like_keyword),
                DocumentEntity.filename.like(like_keyword),
            ))
        count_statement = (
            select(func.count())
            .select_from(TrainingKnowledgeBatchEntity)
            .outerjoin(
                DocumentEntity,
                DocumentEntity.document_id == TrainingKnowledgeBatchEntity.document_id,
            )
            .where(*conditions)
        )
        list_statement = (
            self._batch_select_statement()
            .where(*conditions)
            .order_by(
                TrainingKnowledgeBatchEntity.created_at.desc(),
                TrainingKnowledgeBatchEntity.updated_at.desc(),
            )
            .limit(page_size)
            .offset(offset)
        )
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.execute(list_statement).all()
        return self._scalars_to_batches(rows), total

    def list_batches_by_document_id(self, document_id: str) -> list[TrainingKnowledgeBatchEntity]:
        """按 document_id 查询训练资料批次。"""

        statement = self._batch_select_statement().where(
            TrainingKnowledgeBatchEntity.document_id == document_id
        )
        with orm_session_context() as session:
            rows = session.execute(statement).all()
        return self._scalars_to_batches(rows)

    def delete_batch(self, batch_id: str) -> bool:
        """物理删除单个训练资料批次。"""

        with orm_session_context() as session:
            batch = session.get(TrainingKnowledgeBatchEntity, batch_id)
            if batch is None:
                return False
            session.delete(batch)
            return True

    def delete_batches_by_document_id(self, document_id: str) -> int:
        """按 document_id 物理删除训练资料批次，返回删除数量。"""

        statement = select(TrainingKnowledgeBatchEntity).where(
            TrainingKnowledgeBatchEntity.document_id == document_id
        )
        with orm_session_context() as session:
            batches = list(session.scalars(statement).all())
            for batch in batches:
                session.delete(batch)
            return len(batches)

    def get_batch(self, batch_id: str) -> TrainingKnowledgeBatchEntity | None:
        """按批次编号查询训练资料批次。"""

        statement = self._batch_select_statement().where(TrainingKnowledgeBatchEntity.batch_id == batch_id)
        with orm_session_context() as session:
            row = session.execute(statement).one_or_none()
        if row is None:
            return None
        return self._enrich_batch_with_document(row[0], row[1])

    def create_plan(self, **values: Any) -> TrainingPlanEntity:
        """创建训练方案。"""

        now = utc_now()
        plan_id = values.get("plan_id") or new_id()
        trainee = values["trainee"]
        plan = TrainingPlanEntity(
            plan_id=plan_id,
            plan_name=values["plan_name"],
            trainee_id=trainee.get("trainee_id") or "",
            trainee_name=trainee.get("trainee_name") or "",
            profile_type=values.get("profile_type") or "",
            trainee_json=self._json(trainee),
            selected_fields_json=self._json(values.get("selected_fields") or {}),
            scenario_description=values.get("scenario_description") or "",
            extra_details=values.get("extra_details") or "",
            model_mode=values.get("model_mode"),
            active_profile_id=None,
            active_setting_id=None,
            role_status="pending",
            goal_status="pending",
            score_status="pending",
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(plan)
        created = self.get_plan(plan_id)
        if created is None:
            raise RuntimeError(f"训练方案创建失败：{plan_id}")
        return created

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None) -> tuple[list[TrainingPlanEntity], int]:
        """分页查询训练方案列表。"""

        offset = (page - 1) * page_size
        conditions = []
        if keyword and keyword.strip():
            conditions.append(TrainingPlanEntity.plan_name.like(f"%{keyword.strip()}%"))
        count_statement = select(func.count()).select_from(TrainingPlanEntity)
        list_statement = (
            select(TrainingPlanEntity)
            .order_by(TrainingPlanEntity.updated_at.desc(), TrainingPlanEntity.created_at.desc())
            .limit(page_size)
            .offset(offset)
        )
        if conditions:
            count_statement = count_statement.where(*conditions)
            list_statement = list_statement.where(*conditions)
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.scalars(list_statement).all()
        return rows, total

    def get_plan(self, plan_id: str) -> TrainingPlanEntity | None:
        """按 ID 查询训练方案。"""

        with orm_session_context() as session:
            return session.get(TrainingPlanEntity, plan_id)

    def delete_plan(self, plan_id: str) -> bool:
        """物理删除训练方案记录。

        这里只删除 training_plans 本身，不删除历史会话、AI 角色或阶段配置。
        这些数据可能仍被训练复盘引用，误删会导致历史记录无法打开。
        """

        with orm_session_context() as session:
            plan = session.get(TrainingPlanEntity, plan_id)
            if plan is None:
                return False
            session.delete(plan)
            return True

    def update_plan(self, plan_id: str, **values: Any) -> TrainingPlanEntity | None:
        """更新训练方案基础信息和状态。"""

        allowed_columns = {
            "plan_name",
            "trainee_id",
            "trainee_name",
            "profile_type",
            "trainee_json",
            "selected_fields_json",
            "scenario_description",
            "extra_details",
            "model_mode",
            "active_profile_id",
            "active_setting_id",
            "role_status",
            "goal_status",
            "score_status",
        }
        with orm_session_context() as session:
            plan = session.get(TrainingPlanEntity, plan_id)
            if plan is None:
                return None
            for key, value in values.items():
                if key in allowed_columns:
                    setattr(plan, key, value)
            plan.updated_at = utc_now()
        return self.get_plan(plan_id)

    def attach_role_to_plan(self, plan_id: str, profile_id: str) -> TrainingPlanEntity | None:
        """把生成好的 AI 角色关联到训练方案，并标记阶段/评分需要重新生成。"""

        return self.update_plan(
            plan_id,
            active_profile_id=profile_id,
            active_setting_id=None,
            role_status="generated",
            goal_status="stale",
            score_status="stale",
        )

    def attach_goal_to_plan(self, plan_id: str, setting_id: str) -> TrainingPlanEntity | None:
        """把生成好的训练阶段关联到训练方案，并标记评分规则已随阶段生成。"""

        return self.update_plan(
            plan_id,
            active_setting_id=setting_id,
            goal_status="generated",
            score_status="generated",
        )

    def save_role_profile(self, **values: Any) -> TrainingRoleProfileEntity:
        """保存一次 AI 陪练角色。"""

        now = utc_now()
        profile_id = values.get("profile_id") or new_id()
        role_profile = TrainingRoleProfileEntity(
            profile_id=profile_id,
            trainee_id=values["trainee_id"],
            plan_id=values.get("plan_id"),
            profile_type=values["profile_type"],
            visible_profile_json=self._json(values["visible_profile"]),
            hidden_profile_json=self._json(values["hidden_profile"]),
            role_profile_json=self._json(values["role_profile"]),
            role_confirm_card_json=self._json(values["role_confirm_card"]),
            selected_fields_json=self._json(values.get("selected_fields") or {}),
            scenario_description=values.get("scenario_description"),
            extra_details=values.get("extra_details"),
            retrieved_evidence_json=self._json(values.get("retrieved_evidence") or []),
            status=values.get("status", "confirmed"),
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(role_profile)
        created = self.get_role_profile(profile_id)
        if created is None:
            raise RuntimeError(f"AI 客户角色保存失败：{profile_id}")
        return created

    def get_role_profile(self, profile_id: str) -> TrainingRoleProfileEntity | None:
        """按 ID 查询 AI 客户角色。"""

        with orm_session_context() as session:
            return session.get(TrainingRoleProfileEntity, profile_id)

    def update_role_profile(
            self,
            profile_id: str,
            *,
            visible_profile: dict | None = None,
            hidden_profile: dict | None = None,
            role_profile: dict | None = None,
            role_confirm_card: dict | None = None,
    ) -> TrainingRoleProfileEntity | None:
        """人工修改 AI 客户角色的某些 JSON 字段。"""

        with orm_session_context() as session:
            profile = session.get(TrainingRoleProfileEntity, profile_id)
            if profile is None:
                return None
            if visible_profile is not None:
                profile.visible_profile_json = self._json(visible_profile)
            if hidden_profile is not None:
                profile.hidden_profile_json = self._json(hidden_profile)
            if role_profile is not None:
                profile.role_profile_json = self._json(role_profile)
            if role_confirm_card is not None:
                profile.role_confirm_card_json = self._json(role_confirm_card)
            profile.updated_at = utc_now()
        return self.get_role_profile(profile_id)

    def save_goal_setting(self, **values: Any) -> TrainingGoalSettingEntity:
        """保存开放式训练设置。"""

        now = utc_now()
        setting_id = values.get("setting_id") or new_id()
        setting = TrainingGoalSettingEntity(
            setting_id=setting_id,
            profile_id=values["profile_id"],
            plan_id=values.get("plan_id"),
            trainee_id=values["trainee_id"],
            training_mode=values.get("training_mode", "open"),
            training_purpose=values["training_purpose"],
            round_limit=int(values["round_limit"]),
            stages_json=self._json(values["stages"]),
            scoring_rules_json=self._json(values.get("scoring_rules") or {}),
            status=values.get("status", "confirmed"),
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(setting)
        created = self.get_goal_setting(setting_id)
        if created is None:
            raise RuntimeError(f"训练设置保存失败：{setting_id}")
        return created

    def get_goal_setting(self, setting_id: str) -> TrainingGoalSettingEntity | None:
        """按 ID 查询训练设置。"""

        with orm_session_context() as session:
            return session.get(TrainingGoalSettingEntity, setting_id)

    def update_goal_setting(
            self,
            setting_id: str,
            *,
            training_purpose: str | None = None,
            round_limit: int | None = None,
            stages: list[dict[str, Any]] | None = None,
            scoring_rules: dict | None = None,
    ) -> TrainingGoalSettingEntity | None:
        """人工修改训练宗旨、轮数、阶段或评分规则。"""

        with orm_session_context() as session:
            setting = session.get(TrainingGoalSettingEntity, setting_id)
            if setting is None:
                return None
            if training_purpose is not None:
                setting.training_purpose = training_purpose
            if round_limit is not None:
                setting.round_limit = int(round_limit)
            if stages is not None:
                setting.stages_json = self._json(stages)
            if scoring_rules is not None:
                setting.scoring_rules_json = self._json(scoring_rules)
            setting.updated_at = utc_now()
        return self.get_goal_setting(setting_id)

    def create_session(self, **values: Any) -> SalesTrainingSessionEntity:
        """创建一次开放式训练会话。"""

        now = utc_now()
        session_id = values.get("session_id") or new_id()
        training_session = SalesTrainingSessionEntity(
            session_id=session_id,
            profile_id=values["profile_id"],
            setting_id=values["setting_id"],
            trainee_id=values["trainee_id"],
            training_mode=values.get("training_mode", "open"),
            response_mode=values.get("response_mode", "stream"),
            current_stage_no=1,
            status=values.get("status", "active"),
            round_limit=int(values["round_limit"]),
            total_score=None,
            level=None,
            report_json=None,
            started_at=now,
            ended_at=None,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(training_session)
        created = self.get_session(session_id)
        if created is None:
            raise RuntimeError(f"训练会话创建失败：{session_id}")
        return created

    def get_session(self, session_id: str) -> SalesTrainingSessionEntity | None:
        """按 ID 查询训练会话。"""

        with orm_session_context() as session:
            return session.get(SalesTrainingSessionEntity, session_id)

    def list_sessions(
            self,
            *,
            page: int,
            page_size: int,
            trainee_id: str | None = None,
    ) -> tuple[list[SalesTrainingSessionEntity], int]:
        """分页查询训练会话历史，并统计学员已回答轮数。"""

        offset = (page - 1) * page_size
        conditions = []
        if trainee_id:
            conditions.append(SalesTrainingSessionEntity.trainee_id == trainee_id)
        answered_count_subquery = (
            select(
                SalesTrainingTurnEntity.session_id.label("session_id"),
                func.count(SalesTrainingTurnEntity.turn_id).label("answered_count"),
            )
            .where(SalesTrainingTurnEntity.role == "trainee")
            .group_by(SalesTrainingTurnEntity.session_id)
            .subquery()
        )
        count_statement = select(func.count()).select_from(SalesTrainingSessionEntity)
        list_statement = (
            select(SalesTrainingSessionEntity, answered_count_subquery.c.answered_count)
            .outerjoin(
                answered_count_subquery,
                answered_count_subquery.c.session_id == SalesTrainingSessionEntity.session_id,
            )
            .order_by(
                SalesTrainingSessionEntity.updated_at.desc(),
                SalesTrainingSessionEntity.created_at.desc(),
            )
            .limit(page_size)
            .offset(offset)
        )
        if conditions:
            count_statement = count_statement.where(*conditions)
            list_statement = list_statement.where(*conditions)
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.execute(list_statement).all()

        sessions: list[SalesTrainingSessionEntity] = []
        for training_session, answered_count in rows:
            training_session.answered_count = int(answered_count or 0)
            sessions.append(training_session)
        return sessions, total

    def update_session_status(
            self,
            session_id: str,
            *,
            status: str,
            total_score: int | None = None,
            level: str | None = None,
            report: dict | None = None,
    ) -> None:
        """更新训练会话状态，评分完成时同步写报告摘要。"""

        now = utc_now()
        with orm_session_context() as session:
            training_session = session.get(SalesTrainingSessionEntity, session_id)
            if training_session is None:
                return
            training_session.status = status
            if total_score is not None:
                training_session.total_score = int(total_score)
            if level is not None:
                training_session.level = level
            if report is not None:
                training_session.report_json = self._json(report)
            if status in {"completed", "failed"} and training_session.ended_at is None:
                training_session.ended_at = now
            training_session.updated_at = now

    def add_turn(self, **values: Any) -> SalesTrainingTurnEntity:
        """保存训练对话轮次。"""

        now = utc_now()
        turn_id = values.get("turn_id") or new_id()
        turn = SalesTrainingTurnEntity(
            turn_id=turn_id,
            session_id=values["session_id"],
            role=values["role"],
            content=values["content"],
            round_no=int(values["round_no"]),
            stage_no=int(values.get("stage_no", 1)),
            response_mode=values.get("response_mode"),
            started_at=to_datetime(values.get("started_at")),
            submitted_at=to_datetime(values.get("submitted_at")),
            response_seconds=values.get("response_seconds"),
            retrieved_chunk_ids_json=self._json(values.get("retrieved_chunk_ids") or []),
            retrieved_evidence_json=self._json(values.get("retrieved_evidence") or []),
            stage_decision_json=self._json(values.get("stage_decision") or {}),
            coach_analysis_json=self._json(values.get("coach_analysis") or {}),
            metadata_json=self._json(values.get("metadata") or {}),
            created_at=now,
        )
        with orm_session_context() as session:
            session.add(turn)
        created = self.get_turn(turn_id)
        if created is None:
            raise RuntimeError(f"训练轮次保存失败：{turn_id}")
        return created

    def get_turn(self, turn_id: str) -> SalesTrainingTurnEntity | None:
        """按 ID 查询训练轮次。"""

        with orm_session_context() as session:
            return session.get(SalesTrainingTurnEntity, turn_id)

    def list_turns(self, session_id: str) -> list[SalesTrainingTurnEntity]:
        """查询训练会话内的全部轮次。"""

        statement = (
            select(SalesTrainingTurnEntity)
            .where(SalesTrainingTurnEntity.session_id == session_id)
            .order_by(SalesTrainingTurnEntity.round_no.asc(), SalesTrainingTurnEntity.created_at.asc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

    def next_round_no(self, session_id: str) -> int:
        """计算下一轮学员回复轮次。"""

        statement = select(func.coalesce(func.max(SalesTrainingTurnEntity.round_no), 0)).where(
            SalesTrainingTurnEntity.session_id == session_id,
            SalesTrainingTurnEntity.role == "trainee",
        )
        with orm_session_context() as session:
            max_round = int(session.scalar(statement) or 0)
        return max_round + 1

    def save_score(self, **values: Any) -> SalesTrainingScoreEntity:
        """保存训练评分结果。"""

        now = utc_now()
        score_id = values.get("score_id") or new_id()
        score = SalesTrainingScoreEntity(
            score_id=score_id,
            session_id=values["session_id"],
            general_score=int(values["general_score"]),
            stage_score=int(values["stage_score"]),
            penalty_score=int(values.get("penalty_score", 0)),
            final_score=int(values["final_score"]),
            level=values["level"],
            is_passed=1 if values.get("is_passed") else 0,
            detail_json=self._json(values.get("detail") or {}),
            review_status=values.get("review_status", "confirmed"),
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(score)
        created = self.get_score(score_id)
        if created is None:
            raise RuntimeError(f"训练评分保存失败：{score_id}")
        return created

    def get_score(self, score_id: str) -> SalesTrainingScoreEntity | None:
        """按 ID 查询训练评分。"""

        with orm_session_context() as session:
            return session.get(SalesTrainingScoreEntity, score_id)

    def get_latest_score_by_session(self, session_id: str) -> SalesTrainingScoreEntity | None:
        """查询某个训练会话最新的一份评分报告。"""

        statement = (
            select(SalesTrainingScoreEntity)
            .where(SalesTrainingScoreEntity.session_id == session_id)
            .order_by(
                SalesTrainingScoreEntity.updated_at.desc(),
                SalesTrainingScoreEntity.created_at.desc(),
            )
            .limit(1)
        )
        with orm_session_context() as session:
            return session.scalars(statement).first()
