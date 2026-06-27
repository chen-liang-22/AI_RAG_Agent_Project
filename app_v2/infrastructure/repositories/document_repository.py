"""文档资产仓储。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from domain.entities import DocumentEntity
from infrastructure.orm_session import orm_session_context
from training.repository import utc_now
from utils.knowledge_asset_constants import TRAINING_COLLECTION_NAMES
from utils.logger_handler import logger


class DocumentRepository:
    """封装 documents 文件资产表访问。

    这里使用仓储模式：应用服务只说“查文档列表、查详情、按 MD5 去重”，不直接写 SQL。
    `store` 参数暂时保留给旧测试和过渡代码传入，但真实读路径已经改为 ORM。
    """

    def __init__(self, store: Any | None = None):
        self.store = store

    @staticmethod
    def ensure_storage_columns(session: Session) -> None:
        """确保 documents 表具备 MinIO 存储字段和索引。"""

        bind = session.get_bind()
        inspector = inspect(bind)
        columns = {column["name"] for column in inspector.get_columns("documents")}
        ddl_statements: list[str] = []
        if "storage_type" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN storage_type VARCHAR(32) NOT NULL DEFAULT 'minio' "
                "COMMENT '文件存储类型：minio 表示对象存储' AFTER file_path"
            )
        if "bucket_name" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN bucket_name VARCHAR(128) NULL COMMENT 'MinIO 桶名' AFTER storage_type"
            )
        if "object_name" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN object_name VARCHAR(1024) NULL COMMENT 'MinIO 对象路径' AFTER bucket_name"
            )
        if "public_url" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN public_url VARCHAR(2048) NULL COMMENT 'MinIO 公共访问地址' AFTER object_name"
            )
        for ddl_statement in ddl_statements:
            session.execute(text(ddl_statement))
        if ddl_statements:
            logger.info("[V2文档仓储] documents 表 MinIO 存储字段已自动补齐 字段数量=%s", len(ddl_statements))

        indexes = {index["name"] for index in inspector.get_indexes("documents")}
        if "idx_documents_storage_object" not in indexes:
            dialect_name = bind.dialect.name
            object_index_column = "object_name(255)" if dialect_name == "mysql" else "object_name"
            session.execute(text(
                "CREATE INDEX idx_documents_storage_object "
                f"ON documents(storage_type, bucket_name, {object_index_column})"
            ))
            logger.info("[V2文档仓储] documents 表 MinIO 存储索引已自动补齐 索引名=idx_documents_storage_object")

    def list_documents(self, *, include_training: bool = False) -> list[DocumentEntity]:
        """查询未删除的文档资产列表。

        include_training=False 时隐藏销售训练资料，避免普通知识库页面混入训练专用文件。
        """

        conditions = [DocumentEntity.status != "deleted"]
        if not include_training:
            conditions.append(DocumentEntity.collection_name.not_in(TRAINING_COLLECTION_NAMES))
        statement = (
            select(DocumentEntity)
            .where(*conditions)
            .order_by(DocumentEntity.created_at.desc())
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def get_document(self, document_id: str) -> DocumentEntity | None:
        """按文档资产编号查询单个文件。"""

        with orm_session_context() as session:
            return session.get(DocumentEntity, document_id)

    def find_active_document_by_md5(
        self,
        file_md5: str,
        *,
        collection_name: str | None = None,
    ) -> DocumentEntity | None:
        """按文件 MD5 查找未删除文档，用于上传去重。

        传入 collection_name 时只在指定向量库 collection 内去重；不传时做全局去重。
        """

        conditions = [
            DocumentEntity.file_md5 == file_md5,
            DocumentEntity.status != "deleted",
        ]
        if collection_name:
            conditions.append(DocumentEntity.collection_name == collection_name)
        statement = (
            select(DocumentEntity)
            .where(*conditions)
            .order_by(DocumentEntity.created_at.desc())
            .limit(1)
        )
        with orm_session_context() as session:
            return session.scalars(statement).first()
    def create_document(
        self,
        *,
        document_id: str,
        filename: str,
        file_path: str,
        file_type: str,
        file_md5: str,
        file_size: int,
        storage_type: str = "minio",
        bucket_name: str | None = None,
        object_name: str | None = None,
        public_url: str | None = None,
        status: str = "uploaded",
        collection_name: str = "agent",
        document_type: str = "text",
        split_strategy: str = "recursive",
    ) -> DocumentEntity:
        """创建知识库文档元数据。"""

        now = utc_now()
        document = DocumentEntity(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            storage_type=storage_type,
            bucket_name=bucket_name,
            object_name=object_name,
            public_url=public_url,
            file_type=file_type,
            file_md5=file_md5,
            file_size=int(file_size),
            status=status,
            version=1,
            chunk_count=0,
            collection_name=collection_name,
            document_type=document_type,
            split_strategy=split_strategy,
            created_at=now,
            updated_at=now,
            error_message=None,
        )
        with orm_session_context() as session:
            session.add(document)
        created = self.get_document(document_id)
        if created is None:
            raise RuntimeError(f"Document {document_id} was not created")
        return created

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
        increment_version: bool = False,
        collection_name: str | None = None,
        document_type: str | None = None,
        split_strategy: str | None = None,
    ) -> None:
        """更新文档索引状态和索引元数据。"""

        with orm_session_context() as session:
            document = session.get(DocumentEntity, document_id)
            if document is None:
                raise ValueError(f"Document {document_id} does not exist")
            document.status = status
            document.chunk_count = int(document.chunk_count if chunk_count is None else chunk_count)
            document.error_message = error_message
            if increment_version:
                document.version = int(document.version) + 1
            document.collection_name = collection_name or document.collection_name or "agent"
            document.document_type = document_type or document.document_type or "text"
            document.split_strategy = split_strategy or document.split_strategy or "recursive"
            document.updated_at = utc_now()

    def mark_document_deleted(self, document_id: str) -> None:
        """把文档标记为删除，保留给软删除场景调用。"""

        self.update_document_status(document_id, "deleted")

    def delete_document(self, document_id: str) -> bool:
        """从 documents 表物理删除文件资产记录。"""

        with orm_session_context() as session:
            document = session.get(DocumentEntity, document_id)
            if document is None:
                return False
            session.delete(document)
            return True
