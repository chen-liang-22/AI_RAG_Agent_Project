"""把历史本地文件迁移到 MinIO。

脚本会自动补齐本次迁移依赖的轻量字段；如果你希望显式管理 DDL，
也可以先手工执行：
    docs/mysql迁移_documents文件改为MinIO存储.sql
    docs/mysql迁移_销售训练批次关联documents.sql

脚本逻辑：
1. 扫描 documents 表中未删除且 object_name 为空的记录；
2. 读取原 file_path 指向的本地文件；
3. 上传到 MinIO，并回填 documents 的 MinIO 字段；
4. 扫描历史销售训练批次，把旧 file_path 文件补进 documents 文件台账；
5. 回填 training_knowledge_batches.document_id，让新代码统一从 documents 读取文件信息。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sqlalchemy import or_, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import DocumentEntity, TrainingKnowledgeBatchEntity
from infrastructure.file_storage_service import get_file_storage_service
from infrastructure.orm_session import orm_session_context
from rag.knowledge_store import KnowledgeStore
from training.repository import TrainingRepository
from utils.logger_handler import logger


@dataclass(frozen=True)
class MigrationCandidate:
    """迁移候选记录。

    这个对象只保存预检查阶段需要展示的信息，不直接参与数据库写入。
    可以把 dataclass 理解成 Java 里的简单 DTO。
    """

    record_type: str
    record_id: str
    filename: str
    source_path: str
    resolved_path: Path | None
    can_migrate: bool


@dataclass(frozen=True)
class MigrationPrecheckResult:
    """迁移前检查结果，便于 dry-run 和正式迁移共用同一套统计口径。"""

    document_candidates: list[MigrationCandidate]
    training_batch_candidates: list[MigrationCandidate]

    @property
    def all_candidates(self) -> list[MigrationCandidate]:
        """合并普通文档和训练批次候选记录。"""

        return [*self.document_candidates, *self.training_batch_candidates]

    @property
    def total_count(self) -> int:
        """返回所有待处理记录数量。"""

        return len(self.all_candidates)

    @property
    def migratable_count(self) -> int:
        """返回本机可以找到原文件、理论上可以迁移的记录数量。"""

        return sum(1 for candidate in self.all_candidates if candidate.can_migrate)

    @property
    def missing_count(self) -> int:
        """返回本机找不到原文件、无法迁移的记录数量。"""

        return sum(1 for candidate in self.all_candidates if not candidate.can_migrate)


def resolve_local_path(value: str | None) -> Path | None:
    """把数据库中的历史文件路径解析为当前项目可访问的本地路径。

    历史数据可能保存绝对路径，也可能保存相对路径。这里统一先按原值判断，
    再按项目根目录拼接判断，避免另一台机器迁移过来的路径读不到。
    """

    raw_path = str(value or "").strip()
    if not raw_path or raw_path.startswith("minio://"):
        return None

    direct_path = Path(raw_path)
    if direct_path.is_file():
        return direct_path

    project_path = PROJECT_ROOT / raw_path
    if project_path.is_file():
        return project_path

    # 另一台电脑迁移过来的绝对路径，盘符或上级目录可能不同。
    # 只要路径里包含当前项目目录名，就取项目目录之后的相对路径重新定位。
    path_parts = list(direct_path.parts)
    project_name = PROJECT_ROOT.name
    if project_name in path_parts:
        relative_parts = path_parts[path_parts.index(project_name) + 1:]
        relocated_path = PROJECT_ROOT.joinpath(*relative_parts)
        if relocated_path.is_file():
            return relocated_path

    # 有些历史数据只需要从 uploads 目录重新定位，例如 D:\old\...\uploads\doc_xxx\a.txt。
    if "uploads" in path_parts:
        upload_relative_path = PROJECT_ROOT.joinpath(*path_parts[path_parts.index("uploads"):])
        if upload_relative_path.is_file():
            return upload_relative_path

    # 更早的初始化数据可能来自 data 目录，数据库后来被写成 uploads/doc_xxx/文件名。
    # 当前机器如果 uploads 缺文件，但 data 下有同名原始资料，就用 data 作为兜底来源。
    data_path = PROJECT_ROOT / "data" / direct_path.name
    if data_path.is_file():
        return data_path
    return direct_path


def ensure_minio_ready() -> str:
    """检查 MinIO 是否可用，并返回最终使用的桶名。

    正式迁移前先检查桶，是为了避免出现“前几条上传成功，后面才发现配置不对”的半截状态。
    当前项目配置里通常要求手工创建公共桶；如果桶不存在，这里会抛出明确中文错误。
    """

    bucket_name = get_file_storage_service().ensure_bucket_ready()
    logger.info("[文件迁移] MinIO 存储桶检查通过 桶名=%s", bucket_name)
    return bucket_name


def ensure_storage_columns() -> None:
    """迁移前自动补齐文件存储相关字段。

    这一步让脚本在旧环境里更耐用：即使忘了手工执行迁移 SQL，
    脚本也会先补齐本次迁移需要的轻量字段和索引。
    """

    with orm_session_context() as session:
        KnowledgeStore.ensure_document_storage_columns(session)
    TrainingRepository.ensure_training_batch_document_columns()


def _build_candidate(
        *,
        record_type: str,
        record_id: str,
        filename: str | None,
        source_path: str | None,
) -> MigrationCandidate:
    """把数据库记录转换成预检查候选对象。"""

    resolved_path = resolve_local_path(source_path)
    can_migrate = bool(resolved_path and resolved_path.is_file())
    return MigrationCandidate(
        record_type=record_type,
        record_id=record_id,
        filename=str(filename or ""),
        source_path=str(source_path or ""),
        resolved_path=resolved_path,
        can_migrate=can_migrate,
    )


def precheck_migration() -> MigrationPrecheckResult:
    """扫描数据库，统计哪些历史文件需要迁移、哪些文件在本机缺失。"""

    with orm_session_context() as session:
        documents = list(
            session.scalars(
                select(DocumentEntity).where(
                    DocumentEntity.status != "deleted",
                    (DocumentEntity.object_name.is_(None)) | (DocumentEntity.object_name == ""),
                )
            ).all()
        )
        batches = list(
            session.scalars(
                select(TrainingKnowledgeBatchEntity).where(
                    TrainingKnowledgeBatchEntity.status != "deleted",
                    or_(
                        TrainingKnowledgeBatchEntity.document_id.is_(None),
                        TrainingKnowledgeBatchEntity.document_id == "",
                    ),
                )
            ).all()
        )

    document_candidates = [
        _build_candidate(
            record_type="documents",
            record_id=document.document_id,
            filename=document.filename,
            source_path=document.file_path,
        )
        for document in documents
    ]
    training_batch_candidates = [
        _build_candidate(
            record_type="training_knowledge_batches",
            record_id=batch.batch_id,
            filename=batch.source_file,
            source_path=batch.file_path,
        )
        for batch in batches
    ]
    result = MigrationPrecheckResult(
        document_candidates=document_candidates,
        training_batch_candidates=training_batch_candidates,
    )
    logger.info(
        "[文件迁移] 预检查完成 待处理总数=%s 可迁移=%s 缺失文件=%s documents=%s 训练批次=%s",
        result.total_count,
        result.migratable_count,
        result.missing_count,
        len(document_candidates),
        len(training_batch_candidates),
    )
    for candidate in result.all_candidates:
        if candidate.can_migrate:
            logger.info(
                "[文件迁移] 可迁移记录 类型=%s 编号=%s 文件名=%s 本机路径=%s",
                candidate.record_type,
                candidate.record_id,
                candidate.filename,
                candidate.resolved_path,
            )
        else:
            logger.warning(
                "[文件迁移] 原文件缺失，无法迁移 类型=%s 编号=%s 文件名=%s 数据库路径=%s 解析路径=%s",
                candidate.record_type,
                candidate.record_id,
                candidate.filename,
                candidate.source_path,
                candidate.resolved_path,
            )
    return result


def migrate_documents_to_minio() -> tuple[int, int]:
    """迁移历史 documents 本地文件到 MinIO，返回成功数和失败数。"""

    success_count = 0
    failed_count = 0
    with orm_session_context() as session:
        documents = list(
            session.scalars(
                select(DocumentEntity).where(
                    DocumentEntity.status != "deleted",
                    (DocumentEntity.object_name.is_(None)) | (DocumentEntity.object_name == ""),
                )
            ).all()
        )
        for document in documents:
            local_path = resolve_local_path(str(document.file_path or ""))
            if local_path is None or not local_path.is_file():
                failed_count += 1
                logger.error(
                    "[文件迁移] 历史文件不存在，无法迁移 文档编号=%s 路径=%s",
                    document.document_id,
                    local_path,
                )
                continue
            try:
                stored_file = get_file_storage_service().save_local_file(
                    file_path=str(local_path),
                    filename=document.filename,
                    prefix="documents",
                    owner_id=document.document_id,
                )
                document.storage_type = "minio"
                document.bucket_name = stored_file.bucket_name
                document.object_name = stored_file.object_name
                document.public_url = stored_file.public_url
                document.file_path = stored_file.file_path
                success_count += 1
                logger.info(
                    "[文件迁移] 历史文件已上传 MinIO 文档编号=%s 对象名=%s",
                    document.document_id,
                    stored_file.object_name,
                )
            except Exception as exc:
                failed_count += 1
                logger.error(
                    "[文件迁移] 历史文件上传 MinIO 失败 文档编号=%s 错误=%s",
                    document.document_id,
                    exc,
                    exc_info=True,
                )
    return success_count, failed_count


def migrate_training_batches_to_documents() -> tuple[int, int, int]:
    """把历史销售训练批次原文件补进 documents 台账。

    返回值分别是：成功迁移数量、失败数量、跳过数量。
    新批次已经有 document_id 时会跳过；历史批次如果没有本地文件，也只记录失败，不影响其他批次。
    """

    success_count = 0
    failed_count = 0
    skipped_count = 0
    with orm_session_context() as session:
        batches = list(
            session.scalars(
                select(TrainingKnowledgeBatchEntity).where(
                    TrainingKnowledgeBatchEntity.status != "deleted",
                    or_(
                        TrainingKnowledgeBatchEntity.document_id.is_(None),
                        TrainingKnowledgeBatchEntity.document_id == "",
                    ),
                )
            ).all()
        )
        for batch in batches:
            local_path = resolve_local_path(batch.file_path)
            if local_path is None:
                skipped_count += 1
                logger.info(
                    "[文件迁移] 训练批次没有可迁移文件路径，跳过 批次编号=%s 文件名=%s",
                    batch.batch_id,
                    batch.source_file,
                )
                continue
            if not local_path.is_file():
                failed_count += 1
                logger.error(
                    "[文件迁移] 训练批次历史文件不存在 批次编号=%s 路径=%s",
                    batch.batch_id,
                    local_path,
                )
                continue

            try:
                document_id = f"doc_{uuid4().hex}"
                filename = Path(batch.source_file or local_path.name).name
                file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                stored_file = get_file_storage_service().save_local_file(
                    file_path=str(local_path),
                    filename=filename,
                    prefix="training",
                    owner_id=document_id,
                )
                document = DocumentEntity(
                    document_id=document_id,
                    filename=filename,
                    file_path=stored_file.file_path,
                    storage_type="minio",
                    bucket_name=stored_file.bucket_name,
                    object_name=stored_file.object_name,
                    public_url=stored_file.public_url,
                    file_type=file_type,
                    file_md5=stored_file.file_md5,
                    file_size=stored_file.file_size,
                    status="indexed" if batch.status in {"published", "archived"} else "uploaded",
                    version=int(batch.version_no or 1),
                    chunk_count=int(batch.chunk_count or 0),
                    collection_name="sales_training_cases",
                    document_type="text",
                    split_strategy="recursive",
                    created_at=batch.created_at,
                    updated_at=batch.updated_at,
                    error_message=batch.error_message,
                )
                session.add(document)
                batch.document_id = document_id
                success_count += 1
                logger.info(
                    "[文件迁移] 训练批次已补齐 documents 台账 批次编号=%s 文档编号=%s 对象名=%s",
                    batch.batch_id,
                    document_id,
                    stored_file.object_name,
                )
            except Exception as exc:
                failed_count += 1
                logger.error(
                    "[文件迁移] 训练批次补齐 documents 台账失败 批次编号=%s 错误=%s",
                    batch.batch_id,
                    exc,
                    exc_info=True,
                )
    return success_count, failed_count, skipped_count


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --dry-run 只做预检查，不上传 MinIO、不回填数据库。
    这适合在正式迁移前先确认文件是否都能在当前机器上找到。
    """

    parser = argparse.ArgumentParser(description="把历史本地文件迁移到 MinIO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只做预检查，不上传 MinIO，也不修改数据库",
    )
    parser.add_argument(
        "--skip-minio-check",
        action="store_true",
        help="跳过 MinIO 桶检查；一般只建议在 dry-run 时使用",
    )
    parser.add_argument(
        "--auto-create-bucket",
        action="store_true",
        help="迁移前如果 MinIO 桶不存在，则自动创建桶",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.auto_create_bucket:
        # 只影响本次脚本进程，不修改 config/minio.yml。
        # 这样既能适配新机器第一次迁移，也不会改变项目默认的保守配置。
        os.environ["MINIO_AUTO_CREATE_BUCKET"] = "true"

    ensure_storage_columns()
    precheck_result = precheck_migration()
    if args.dry_run:
        print(
            "预检查完成："
            f"待处理总数={precheck_result.total_count}，"
            f"可迁移={precheck_result.migratable_count}，"
            f"缺失文件={precheck_result.missing_count}"
        )
        sys.exit(0)

    try:
        if not args.skip_minio_check:
            ensure_minio_ready()
        document_success, document_failed = migrate_documents_to_minio()
        batch_success, batch_failed, batch_skipped = migrate_training_batches_to_documents()
        print(
            "迁移完成："
            f"documents成功={document_success}，documents失败={document_failed}；"
            f"训练批次成功={batch_success}，训练批次失败={batch_failed}，训练批次跳过={batch_skipped}"
        )
    except Exception as exc:
        logger.error("[文件迁移] 迁移中止 错误=%s", exc, exc_info=True)
        print(f"迁移中止：{exc}")
        sys.exit(1)
