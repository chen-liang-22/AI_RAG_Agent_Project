"""销售训练资料管理服务。

这个模块承接销售训练资料的查询、预览、删除、版本和切片展示逻辑。
上传、发布、回滚、重切这类重流程会在下一步继续迁移；本服务先把低风险、
边界清楚的资料管理能力从核心编排类中拆出来。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from fastapi import HTTPException
from langchain_core.documents import Document

from app.application.knowledge.document_asset_service import DocumentAssetService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import (
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeBatchResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeChunkResponse,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgeVersionListResponse,
)
from app.infrastructure.file_storage_service import get_file_storage_service
from app.infrastructure.repositories.document_repository import DocumentRepository
from core.rag.file_processors import FileProcessorFactory
from core.utils.logger_handler import logger


ALLOWED_TRAINING_FILE_TYPES = {"txt", "pdf", "docx"}


class TrainingVectorPort(Protocol):
    """训练资料服务需要的向量库最小接口。"""

    def delete_by_metadata(self, key: str, value: str) -> int | None:
        """按元数据删除向量点。"""

    def list_documents_by_metadata(self, key: str, value: str) -> list[Document]:
        """按元数据读取向量文档。"""


class TrainingKnowledgeService:
    """销售训练资料管理服务。"""

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            vector_service: TrainingVectorPort,
            staging_vector_service: TrainingVectorPort,
            document_repository: DocumentRepository,
            asset_service: DocumentAssetService | None = None,
    ):
        """初始化训练资料服务。

        repository 负责训练资料业务表，vector_service/staging_vector_service 分别对应正式和临时 Qdrant collection。
        document_repository 用来补齐 documents 表中的文件元数据，asset_service 用来执行统一全链路删除。
        """

        self.repository = repository
        self.vector_service = vector_service
        self.staging_vector_service = staging_vector_service
        self.document_repository = document_repository
        self.asset_service = asset_service or DocumentAssetService()

    def list_batches(self, *, page: int = 1, page_size: int = 10) -> TrainingKnowledgeBatchListResponse:
        """分页查询已经上传过的训练资料。"""

        safe_page = max(1, page)
        safe_page_size = max(1, min(50, page_size))
        rows, total = self.repository.list_batches(page=safe_page, page_size=safe_page_size)
        return TrainingKnowledgeBatchListResponse(
            items=[self.batch_response(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def preview_batch(self, batch_id: str, *, max_chars: int = 30000) -> TrainingKnowledgePreviewResponse:
        """返回训练资料上传文件的站内预览数据。"""

        batch = self.get_active_batch(batch_id)
        file_info = self.batch_file_info(batch)
        file_url = str(file_info.get("public_url") or "").strip()
        if not file_url:
            object_name = str(file_info.get("object_name") or "").strip()
            if not object_name:
                raise HTTPException(status_code=400, detail="训练资料缺少 MinIO 对象路径，请先完成历史文件迁移")
            file_url = get_file_storage_service().client.get_public_url(
                object_name,
                bucket_name=file_info.get("bucket_name"),
            )
        preview = self.build_batch_preview(batch, max_chars=max_chars)

        return TrainingKnowledgePreviewResponse(
            batch=self.batch_response(batch),
            preview_type=preview["preview_type"],
            content=preview["content"],
            truncated=preview["truncated"],
            file_url=file_url,
            charset=preview.get("charset"),
        )

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次，并通过统一文件资产服务清理全链路数据。"""

        batch = self.get_active_batch(batch_id)
        document_id = str(batch.get("document_id") or "").strip()
        if not document_id:
            return self.delete_legacy_batch_without_document(batch_id)

        self.asset_service.delete_document_asset(document_id)
        logger.info("[销售训练] 训练资料已删除 批次编号=%s 文档编号=%s", batch_id, document_id)
        return TrainingKnowledgeDeleteResponse(status="deleted", batch_id=batch_id)

    def delete_legacy_batch_without_document(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除没有 document_id 的历史训练批次。"""

        self.vector_service.delete_by_metadata("batch_id", batch_id)
        self.staging_vector_service.delete_by_metadata("batch_id", batch_id)
        deleted_batch = self.repository.delete_batch(batch_id)
        logger.warning(
            "[销售训练] 已按历史批次兼容方式删除训练资料 批次编号=%s 批次已删除=%s",
            batch_id,
            deleted_batch,
        )
        return TrainingKnowledgeDeleteResponse(status="deleted", batch_id=batch_id)

    def list_batch_versions(self, batch_id: str) -> TrainingKnowledgeVersionListResponse:
        """查询指定训练资料所在版本组的版本链。"""

        batch = self.get_active_batch(batch_id)
        version_group_id = str(batch.get("version_group_id") or batch["batch_id"])
        rows = self.repository.list_batches_in_version_group(version_group_id)
        return TrainingKnowledgeVersionListResponse(
            version_group_id=version_group_id,
            items=[self.batch_response(row) for row in rows],
        )

    def list_chunks(self, batch_id: str) -> TrainingKnowledgeChunkListResponse:
        """查询某个上传批次的训练知识切片。"""

        batch = self.get_active_batch(batch_id)
        chunks = []
        chunk_rows = self.list_batch_chunk_rows(batch)
        for row in chunk_rows:
            metadata = self.load_json(row.get("metadata_json"), {})
            chunks.append(
                TrainingKnowledgeChunkResponse(
                    chunk_id=row["chunk_id"],
                    batch_id=row["batch_id"],
                    case_part=row.get("case_part") or "",
                    visibility=row.get("visibility") or "",
                    chunk_text=row["chunk_text"],
                    metadata=metadata,
                )
            )
        return TrainingKnowledgeChunkListResponse(batch_id=batch_id, chunks=chunks)

    def list_batch_chunk_rows(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """根据批次状态读取正式或临时切片。"""

        batch_id = str(batch["batch_id"])
        if batch.get("status") in {"pending_review", "embedding", "publish_failed", "parsing_failed"}:
            return self.list_staging_chunk_rows(batch_id)
        return self.list_published_chunk_rows(batch_id)

    def list_staging_chunk_rows(self, batch_id: str) -> list[dict[str, Any]]:
        """从临时向量库读取待审核切片行。"""

        return self.documents_to_chunk_rows(
            self.staging_vector_service.list_documents_by_metadata("batch_id", batch_id),
            batch_id=batch_id,
        )

    def list_published_chunk_rows(self, batch_id: str) -> list[dict[str, Any]]:
        """从正式向量库读取已发布切片行。"""

        return self.documents_to_chunk_rows(
            self.vector_service.list_documents_by_metadata("batch_id", batch_id),
            batch_id=batch_id,
        )

    def documents_to_chunk_rows(self, documents: list[Document], *, batch_id: str) -> list[dict[str, Any]]:
        """把 Qdrant 返回的 Document 转成前端切片行。"""

        rows: list[dict[str, Any]] = []
        for document in documents:
            metadata = dict(document.metadata)
            rows.append(
                {
                    "chunk_id": str(metadata.get("chunk_id") or ""),
                    "batch_id": str(metadata.get("batch_id") or batch_id),
                    "qdrant_point_id": str(metadata.get("chunk_id") or ""),
                    "chunk_text": document.page_content,
                    "source_type": metadata.get("source_type"),
                    "case_part": metadata.get("case_part"),
                    "visibility": metadata.get("visibility"),
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                    "metadata": metadata,
                }
            )
        return sorted(rows, key=self.chunk_sort_key)

    def chunk_sort_key(self, row: dict[str, Any]) -> tuple[int, str, str]:
        """按案例序号、切片编号稳定排序。"""

        metadata = self.load_json(row.get("metadata_json"), {})
        try:
            case_index = int(metadata.get("case_index") or 0)
        except (TypeError, ValueError):
            case_index = 0
        return case_index, str(row.get("chunk_id") or ""), str(row.get("case_part") or "")

    def batch_file_info(self, row: dict[str, Any]) -> dict[str, Any]:
        """读取训练资料关联的文件基础信息。"""

        document_id = str(row.get("document_id") or "").strip()
        document = None
        joined_source_file = row.get("document_filename")
        joined_file_path = row.get("document_file_path")
        joined_file_md5 = row.get("document_file_md5")
        joined_bucket_name = row.get("document_bucket_name")
        joined_object_name = row.get("document_object_name")
        joined_public_url = row.get("document_public_url")
        if document_id and not any((joined_source_file, joined_file_path, joined_file_md5)):
            document = self.document_repository.get_document(document_id)
        return {
            "document_id": document_id or None,
            "source_file": joined_source_file or (document or {}).get("filename") or row.get("source_file"),
            "file_path": joined_file_path or (document or {}).get("file_path") or row.get("file_path"),
            "file_md5": joined_file_md5 or (document or {}).get("file_md5") or row.get("file_md5"),
            "bucket_name": joined_bucket_name or (document or {}).get("bucket_name"),
            "object_name": joined_object_name or (document or {}).get("object_name"),
            "public_url": joined_public_url or (document or {}).get("public_url"),
        }

    def batch_response(self, row: dict[str, Any]) -> TrainingKnowledgeBatchResponse:
        """把训练资料批次数据库行转换成前端响应。"""

        file_info = self.batch_file_info(row)
        return TrainingKnowledgeBatchResponse(
            batch_id=row["batch_id"],
            document_id=file_info.get("document_id"),
            source_type=row["source_type"],
            source_file=file_info.get("source_file") or row["source_file"],
            file_path=file_info.get("file_path"),
            file_md5=file_info.get("file_md5"),
            version_group_id=row.get("version_group_id") or row["batch_id"],
            version_no=int(row.get("version_no") or 1),
            previous_batch_id=row.get("previous_batch_id"),
            is_current=bool(row.get("is_current")),
            profile_type=row.get("profile_type"),
            task_type=row.get("task_type"),
            industry=row.get("industry"),
            difficulty=row.get("difficulty"),
            visibility_default=row.get("visibility_default"),
            status=row["status"],
            chunk_count=int(row.get("chunk_count") or 0),
            point_count=int(row.get("point_count") or 0),
            error_message=row.get("error_message"),
            quality_report=self.load_json(row.get("quality_report_json"), {}),
            created_by=row.get("created_by"),
            created_at=self.format_response_time(row["created_at"]),
            updated_at=self.format_response_time(row["updated_at"]),
        )

    def get_active_batch(self, batch_id: str) -> dict[str, Any]:
        """读取未删除的训练资料批次。"""

        batch = self.repository.get_batch(batch_id)
        if batch is None or batch.get("status") == "deleted":
            raise HTTPException(status_code=404, detail=f"训练资料不存在：{batch_id}")
        return batch

    def build_batch_preview(self, batch: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
        """根据训练资料文件类型生成站内弹窗预览数据。"""

        file_info = self.batch_file_info(batch)
        source_file = str(file_info.get("source_file") or batch.get("source_file") or "")
        file_type = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else ""
        if file_type not in ALLOWED_TRAINING_FILE_TYPES:
            return {"preview_type": "file_url", "content": "", "truncated": False, "charset": None}

        with self.download_batch_file(batch) as file_path:
            if file_type == "txt":
                content, charset = self.decode_text_bytes(Path(file_path).read_bytes())
            else:
                documents = FileProcessorFactory.load_documents(file_path)
                content = "\n\n".join(document.page_content.strip() for document in documents if document.page_content.strip())
                charset = "document-parser"

        safe_max_chars = max(500, min(100000, max_chars))
        truncated = len(content) > safe_max_chars
        return {
            "preview_type": "text",
            "content": content[:safe_max_chars],
            "truncated": truncated,
            "charset": charset,
        }

    @contextmanager
    def download_batch_file(self, batch: dict[str, Any]) -> Iterator[str]:
        """从 MinIO 下载训练资料原文件到临时路径。"""

        file_info = self.batch_file_info(batch)
        object_name = str(file_info.get("object_name") or "").strip()
        if not object_name:
            raise HTTPException(status_code=400, detail="训练资料缺少 MinIO 对象路径，请先完成历史文件迁移")
        with get_file_storage_service().downloaded_temp_file(
                bucket_name=file_info.get("bucket_name"),
                object_name=object_name,
                filename=str(file_info.get("source_file") or batch.get("source_file") or "training_file"),
        ) as file_path:
            yield file_path

    @classmethod
    def decode_text_bytes(cls, raw_data: bytes) -> tuple[str, str]:
        """按常见中文编码解码 TXT，避免浏览器直连 MinIO 时乱码。"""

        for charset in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                text = raw_data.decode(charset)
            except UnicodeDecodeError:
                continue
            if not cls.looks_mojibake(text):
                return text, charset
        return raw_data.decode("utf-8", errors="replace"), "utf-8-replace"

    @staticmethod
    def looks_mojibake(text: str) -> bool:
        """粗略判断文本是否存在明显乱码。"""

        if not text:
            return False
        bad_chars = text.count("\ufffd") + text.count("�")
        suspicious_chars = sum(1 for char in text if "\ue000" <= char <= "\uf8ff")
        return (bad_chars + suspicious_chars) / max(len(text), 1) > 0.01

    @staticmethod
    def load_json(value: Any, default: Any) -> Any:
        """读取 JSON 字段，兼容数据库字符串和已解析对象。"""

        if value is None or value == "":
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def format_response_time(value: object) -> str | None:
        """把数据库时间字段统一转换成接口响应字符串。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds", sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return str(value)
