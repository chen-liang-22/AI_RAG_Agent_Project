"""把历史本地文件迁移到 MinIO。

执行前请先运行：
    docs/mysql迁移_documents文件改为MinIO存储.sql

脚本逻辑：
1. 扫描 documents 表中未删除且 object_name 为空的记录；
2. 读取原 file_path 指向的本地文件；
3. 上传到 MinIO 的 documents/{document_id}/{filename}；
4. 回填 storage_type、bucket_name、object_name、public_url、file_path。
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.entities import DocumentEntity
from infrastructure.file_storage_service import FileStorageService, get_file_storage_service
from infrastructure.orm_session import orm_session_context
from utils.logger_handler import logger


def migrate_documents_to_minio() -> tuple[int, int]:
    """迁移历史 documents 本地文件到 MinIO，返回成功数和失败数。"""

    file_storage = get_file_storage_service()
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
            local_path = Path(str(document.file_path or ""))
            if not local_path.is_file():
                failed_count += 1
                logger.error(
                    "[文件迁移] 历史文件不存在，无法迁移 文档编号=%s 路径=%s",
                    document.document_id,
                    local_path,
                )
                continue
            try:
                object_name = f"documents/{document.document_id}/{document.filename}"
                uploaded = file_storage.client.upload_file(str(local_path), object_name=object_name)
                document.storage_type = "minio"
                document.bucket_name = uploaded.bucket_name
                document.object_name = uploaded.object_name
                document.public_url = uploaded.public_url
                document.file_path = FileStorageService.build_storage_uri(uploaded.bucket_name, uploaded.object_name)
                success_count += 1
                logger.info(
                    "[文件迁移] 历史文件已上传 MinIO 文档编号=%s 对象名=%s",
                    document.document_id,
                    uploaded.object_name,
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


if __name__ == "__main__":
    success, failed = migrate_documents_to_minio()
    print(f"迁移完成：成功={success}，失败={failed}")
