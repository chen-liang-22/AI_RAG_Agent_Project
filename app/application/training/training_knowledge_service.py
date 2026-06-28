"""销售训练资料管理服务。

这个模块承接销售训练资料的上传、预览、发布、回滚、重切、删除、
版本和切片展示逻辑，让核心外观类只保留稳定入口。
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from app.application.knowledge.document_asset_service import DocumentAssetService
from app.application.training_support.factories.knowledge_ingest_strategy_factory import KnowledgeIngestStrategyFactory
from app.application.training_support.llm_ingest import TrainingLlmFallbackSplitter
from app.application.training_support.publish_validation import TrainingPublishValidator
from app.application.training_support.quality import TrainingIngestQualityEvaluator
from app.application.training_support.repository import TrainingRepository, utc_now_text
from app.application.training_support.schemas import (
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeBatchResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeChunkResponse,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeUploadResponse,
    TrainingKnowledgeVersionListResponse,
)
from app.infrastructure.file_storage_service import get_file_storage_service
from app.infrastructure.id_generator import new_id
from app.infrastructure.repositories.document_repository import DocumentRepository
from core.rag.file_processors import FileProcessorFactory
from core.utils.logger_handler import logger


DEFAULT_TRAINING_COLLECTION_NAME = "sales_training_cases"
DEFAULT_TRAINING_STAGING_COLLECTION_NAME = "sales_training_cases_staging"
ALLOWED_TRAINING_FILE_TYPES = {"txt", "pdf", "docx"}
DEFAULT_TRAINING_VISIBILITY = "visible"


class TrainingVectorPort(Protocol):
    """训练资料服务需要的向量库最小接口。"""

    vector_store: Any

    def delete_by_metadata(self, key: str, value: str) -> int | None:
        """按元数据删除向量点。"""

    def list_documents_by_metadata(self, key: str, value: str) -> list[Document]:
        """按元数据读取向量文档。"""

    def copy_points_by_metadata_to(
            self,
            target_service: Any,
            key: str,
            value: str,
            *,
            metadata_updates: dict[str, Any] | None = None,
    ) -> int:
        """按元数据把向量点复制到另一个 collection。"""

    def update_metadata_by_metadata(self, key: str, value: str, *, metadata_updates: dict[str, Any]) -> int:
        """按元数据批量更新向量点 payload。"""


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
            training_collection_name: str = DEFAULT_TRAINING_COLLECTION_NAME,
            staging_collection_name: str = DEFAULT_TRAINING_STAGING_COLLECTION_NAME,
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
        self.training_collection_name = training_collection_name
        self.staging_collection_name = staging_collection_name

    def upload_knowledge(
            self,
            *,
            file: UploadFile,
            source_type: str,
            created_by: str | None,
            model_mode: str | None = None,
    ) -> TrainingKnowledgeUploadResponse:
        """上传训练资料并生成待确认预览。"""

        filename = self.safe_filename(file.filename)
        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if file_type not in ALLOWED_TRAINING_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"训练知识暂不支持该文件类型：{file_type}")

        document_id = new_id()
        batch_id = new_id()
        stored_file = get_file_storage_service().save_upload_file(
            file=file,
            filename=filename,
            prefix="training",
            owner_id=document_id,
        )

        file_md5 = stored_file.file_md5
        existing_batch = self.repository.get_published_batch_by_md5(file_md5)
        if existing_batch:
            get_file_storage_service().delete_object(
                bucket_name=stored_file.bucket_name,
                object_name=stored_file.object_name,
            )
            logger.info(
                "[销售训练] 训练知识命中重复文件 已复用批次=%s 文件名=%s",
                existing_batch["batch_id"],
                filename,
            )
            return TrainingKnowledgeUploadResponse(
                batch_id=existing_batch["batch_id"],
                document_id=existing_batch.get("document_id"),
                status="duplicated",
                chunk_count=int(existing_batch.get("chunk_count") or 0),
                point_count=int(existing_batch.get("point_count") or 0),
                source_file=self.batch_file_info(existing_batch).get("source_file"),
                duplicate_of=existing_batch["batch_id"],
                quality_report=self.load_json(existing_batch.get("quality_report_json"), {}),
            )

        version_info = self.next_training_batch_version(source_type=source_type, source_file=filename)
        self.document_repository.create_document(
            document_id=document_id,
            filename=filename,
            file_path=stored_file.file_path,
            file_type=file_type,
            file_md5=file_md5,
            file_size=stored_file.file_size,
            storage_type="minio",
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
            public_url=stored_file.public_url,
            status="indexing",
            collection_name=self.training_collection_name,
            document_type="text",
            split_strategy="recursive",
        )
        batch = self.repository.create_batch(
            batch_id=batch_id,
            document_id=document_id,
            source_type=source_type,
            source_file=filename,
            file_path=None,
            file_md5=None,
            version_group_id=version_info["version_group_id"],
            version_no=version_info["version_no"],
            previous_batch_id=version_info.get("previous_batch_id"),
            is_current=False,
            visibility_default=DEFAULT_TRAINING_VISIBILITY,
            status="parsing",
            created_by=created_by,
        )
        logger.info(
            "[销售训练] 训练知识上传开始 批次编号=%s 文件名=%s 类型=%s 版本组=%s 版本号=%s",
            batch_id,
            filename,
            source_type,
            version_info["version_group_id"],
            version_info["version_no"],
        )

        try:
            with self.download_batch_file(batch) as file_path:
                logger.info("[销售训练] 训练知识解析开始 批次编号=%s 临时文件=%s", batch_id, file_path)
                chunks = self.parse_training_chunks(
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=filename,
                    source_type=source_type,
                )
                if not chunks:
                    raise ValueError("文件没有切出有效训练知识")
                logger.info("[销售训练] 训练知识规则切片完成 批次编号=%s 切片数量=%s", batch_id, len(chunks))

                chunks, quality_report = self.improve_training_chunks_if_needed(
                    chunks=chunks,
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=filename,
                    source_type=source_type,
                    model_mode=model_mode,
                )
            logger.info("[销售训练] 训练知识写入临时向量库开始 批次编号=%s 临时Collection=%s", batch_id, self.staging_collection_name)
            point_count = self.write_staging_chunks(batch=batch, chunks=chunks, source_type=source_type)
            self.repository.update_batch_status(
                batch_id,
                status="pending_review",
                chunk_count=len(chunks),
                point_count=point_count,
                quality_report=quality_report,
            )
            self.document_repository.update_document_status(
                document_id,
                "indexed",
                chunk_count=len(chunks),
                error_message=None,
                collection_name=self.training_collection_name,
                document_type="text",
                split_strategy="recursive",
            )
            logger.info(
                "[销售训练] 训练知识预览生成完成 批次编号=%s 临时向量库=%s 切片数量=%s 向量点数量=%s 质量分=%s 切分方式=%s",
                batch_id,
                self.staging_collection_name,
                len(chunks),
                point_count,
                quality_report.get("score"),
                quality_report.get("selected_splitter"),
            )
            return TrainingKnowledgeUploadResponse(
                batch_id=batch["batch_id"],
                document_id=document_id,
                status="pending_review",
                chunk_count=len(chunks),
                point_count=point_count,
                source_file=filename,
                quality_report=quality_report,
            )
        except Exception as exc:
            self.document_repository.update_document_status(
                document_id,
                "failed",
                error_message=str(exc),
                collection_name=self.training_collection_name,
                document_type="text",
                split_strategy="recursive",
            )
            self.repository.update_batch_status(batch_id, status="parsing_failed", error_message=str(exc))
            logger.error("[销售训练] 训练知识预览生成失败 批次编号=%s 错误=%s", batch_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"训练知识预览生成失败：{exc}") from exc

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

    def publish_batch(self, batch_id: str) -> TrainingKnowledgePublishResponse:
        """人工确认发布训练资料。"""

        batch = self.get_active_batch(batch_id)
        if batch["status"] == "published":
            logger.info("[销售训练] 训练资料已经发布，直接返回 批次编号=%s", batch_id)
            return TrainingKnowledgePublishResponse(
                batch_id=batch_id,
                status="published",
                chunk_count=int(batch.get("chunk_count") or 0),
                point_count=int(batch.get("point_count") or 0),
                quality_report=self.load_json(batch.get("quality_report_json"), {}),
            )
        if batch["status"] not in {"pending_review", "embedding", "publish_failed"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许发布：{batch['status']}")

        chunks = self.list_staging_chunk_rows(batch_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="临时向量库没有可发布的训练切片，请重新上传或重新切分")

        logger.info(
            "[销售训练] 训练资料发布开始 批次编号=%s 临时Collection=%s 正式Collection=%s 临时切片数=%s",
            batch_id,
            self.staging_collection_name,
            self.training_collection_name,
            len(chunks),
        )
        self.repository.update_batch_status(batch_id, status="embedding")
        try:
            copied_count = self.publish_staging_vectors(batch=batch)
            quality_report = self.load_json(batch.get("quality_report_json"), {})
            publish_validation = TrainingPublishValidator().validate(
                vector_service=self.vector_service,
                batch_id=batch_id,
                chunks=chunks,
            )
            quality_report["publish_validation"] = publish_validation
            self.archive_previous_training_versions(batch)
            self.repository.update_batch_status(
                batch_id,
                status="published",
                chunk_count=len(chunks),
                point_count=copied_count,
                quality_report=quality_report,
                is_current=True,
            )
            self.delete_staging_vectors(batch_id)
        except Exception as exc:
            self.repository.update_batch_status(batch_id, status="publish_failed", error_message=str(exc))
            logger.error("[销售训练] 训练资料发布失败 批次编号=%s 错误=%s", batch_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"训练资料发布失败：{exc}") from exc

        logger.info(
            "[销售训练] 训练资料发布完成 批次编号=%s 临时向量库=%s 正式向量库=%s 向量点数量=%s 抽样验证=%s",
            batch_id,
            self.staging_collection_name,
            self.training_collection_name,
            copied_count,
            quality_report.get("publish_validation", {}).get("summary", "未执行"),
        )
        return TrainingKnowledgePublishResponse(
            batch_id=batch_id,
            status="published",
            chunk_count=len(chunks),
            point_count=copied_count,
            quality_report=quality_report,
        )

    def rollback_batch(self, batch_id: str) -> TrainingKnowledgeRollbackResponse:
        """回滚训练资料到指定历史版本。"""

        batch = self.get_active_batch(batch_id)
        if batch["status"] not in {"published", "archived"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许回滚：{batch['status']}")
        chunks = self.list_published_chunk_rows(batch_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="该版本没有可回滚的训练切片，请重新上传资料")

        version_group_id = batch.get("version_group_id") or batch["batch_id"]
        try:
            self.mark_version_group_vectors_archived(version_group_id)
            point_count = self.mark_batch_vectors_current(batch=batch)
            quality_report = self.load_json(batch.get("quality_report_json"), {})
            quality_report["rollback"] = {
                "rolled_back": True,
                "summary": f"已回滚到版本 {int(batch.get('version_no') or 1)}。",
            }
            publish_validation = TrainingPublishValidator().validate(
                vector_service=self.vector_service,
                batch_id=batch_id,
                chunks=chunks,
            )
            quality_report["publish_validation"] = publish_validation
            self.repository.archive_other_versions(version_group_id=version_group_id, current_batch_id=batch_id)
            self.repository.update_batch_status(
                batch_id,
                status="published",
                chunk_count=len(chunks),
                point_count=point_count,
                quality_report=quality_report,
                is_current=True,
            )
        except Exception as exc:
            logger.error("[销售训练] 训练资料版本回滚失败 批次编号=%s 错误=%s", batch_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"训练资料版本回滚失败：{exc}") from exc

        logger.info(
            "[销售训练] 训练资料版本回滚完成 批次编号=%s 版本组=%s 版本号=%s 向量点数量=%s",
            batch_id,
            version_group_id,
            int(batch.get("version_no") or 1),
            point_count,
        )
        return TrainingKnowledgeRollbackResponse(
            batch_id=batch_id,
            status="published",
            version_group_id=version_group_id,
            version_no=int(batch.get("version_no") or 1),
            chunk_count=len(chunks),
            point_count=point_count,
            quality_report=quality_report,
        )

    def reparse_batch(
            self,
            batch_id: str,
            *,
            use_llm_fallback: bool = True,
            model_mode: str | None = None,
    ) -> TrainingKnowledgeReparseResponse:
        """重新切分未发布训练资料。"""

        batch = self.get_active_batch(batch_id)
        if batch["status"] not in {"pending_review", "parsing_failed", "publish_failed"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许重新切分：{batch['status']}")

        source_type = str(batch.get("source_type") or "lms_case")
        source_file = str(batch.get("source_file") or "")
        try:
            with self.download_batch_file(batch) as file_path:
                rule_chunks = self.parse_training_chunks(
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=source_file,
                    source_type=source_type,
                )
                if use_llm_fallback:
                    chunks, quality_report = self.force_llm_reparse_chunks(
                        rule_chunks=rule_chunks,
                        file_path=file_path,
                        batch_id=batch_id,
                        source_file=source_file,
                        source_type=source_type,
                        model_mode=model_mode,
                    )
                else:
                    evaluator = TrainingIngestQualityEvaluator()
                    chunks = rule_chunks
                    quality_report = evaluator.evaluate(chunks).to_dict()
                    quality_report["selected_splitter"] = "rule_config"
                    quality_report["llm_fallback_used"] = False
                    quality_report["rule_score"] = quality_report.get("score")

            if not chunks:
                raise ValueError("重新切分没有生成有效训练切片")
            point_count = self.write_staging_chunks(batch=batch, chunks=chunks, source_type=source_type)
            self.repository.update_batch_status(
                batch_id,
                status="pending_review",
                chunk_count=len(chunks),
                point_count=point_count,
                quality_report=quality_report,
                is_current=False,
            )
        except Exception as exc:
            self.repository.update_batch_status(batch_id, status="parsing_failed", error_message=str(exc))
            logger.error("[销售训练] 训练资料重新切分失败 批次编号=%s 错误=%s", batch_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"训练资料重新切分失败：{exc}") from exc

        logger.info(
            "[销售训练] 训练资料重新切分完成 批次编号=%s 切片数量=%s 质量分=%s 切分方式=%s",
            batch_id,
            len(chunks),
            quality_report.get("score"),
            quality_report.get("selected_splitter"),
        )
        return TrainingKnowledgeReparseResponse(
            batch_id=batch_id,
            status="pending_review",
            chunk_count=len(chunks),
            point_count=point_count,
            source_file=source_file,
            quality_report=quality_report,
        )

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

    def write_staging_chunks(self, *, batch: dict[str, Any], chunks: list[Any], source_type: str) -> int:
        """把待审核训练切片写入临时向量库。"""

        batch_id = str(batch["batch_id"])
        documents = [
            self.document_from_training_chunk(chunk, batch=batch, source_type=source_type, status="pending_review", is_current=False)
            for chunk in chunks
        ]
        self.delete_staging_vectors(batch_id)
        if documents:
            self.staging_vector_service.vector_store.add_documents(documents)
        logger.info(
            "[销售训练] 待审核切片已写入临时向量库 批次编号=%s 临时向量库=%s 切片数量=%s",
            batch_id,
            self.staging_collection_name,
            len(documents),
        )
        return len(documents)

    def publish_staging_vectors(self, *, batch: dict[str, Any]) -> int:
        """把临时向量库中的待审核切片复制到正式向量库。"""

        batch_id = str(batch["batch_id"])
        metadata_updates = {
            "status": "published",
            "is_current": True,
            "published_at": utc_now_text(),
        }
        self.vector_service.delete_by_metadata("batch_id", batch_id)
        copied_count = self.staging_vector_service.copy_points_by_metadata_to(
            self.vector_service,
            "batch_id",
            batch_id,
            metadata_updates=metadata_updates,
        )
        if copied_count <= 0:
            raise ValueError("临时向量库复制到正式向量库失败，没有复制任何向量点")
        logger.info(
            "[销售训练] 待审核切片已复制到正式向量库 批次编号=%s 临时向量库=%s 正式向量库=%s 向量点数量=%s",
            batch_id,
            self.staging_collection_name,
            self.training_collection_name,
            copied_count,
        )
        return copied_count

    def delete_staging_vectors(self, batch_id: str) -> None:
        """删除临时向量库中某个批次的待审核切片。"""

        self.staging_vector_service.delete_by_metadata("batch_id", batch_id)

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

    def document_from_training_chunk(
            self,
            chunk: Any,
            *,
            batch: dict[str, Any],
            source_type: str,
            status: str,
            is_current: bool,
    ) -> Document:
        """把内存训练切片转换成 Qdrant Document。"""

        metadata = self.compact_metadata({
            "batch_id": batch["batch_id"],
            "document_id": batch.get("document_id"),
            "chunk_id": chunk.chunk_id,
            "content_type": "sales_training_case",
            "source_type": source_type,
            "source_file": self.batch_file_info(batch).get("source_file"),
            "file_md5": self.batch_file_info(batch).get("file_md5"),
            "version_group_id": batch.get("version_group_id") or batch["batch_id"],
            "version_no": int(batch.get("version_no") or 1),
            "status": status,
            "is_current": is_current,
            "case_part": chunk.case_part,
            "visibility": chunk.visibility,
            **dict(chunk.metadata or {}),
        })
        return Document(page_content=chunk.text, metadata=metadata)

    def parse_training_chunks(
            self,
            *,
            file_path: str,
            batch_id: str,
            source_file: str,
            source_type: str,
    ) -> list[Any]:
        """按资料类型解析训练切片。"""

        context = {
            "batch_id": batch_id,
            "source_file": source_file,
            "source_type": source_type,
            "visibility_default": DEFAULT_TRAINING_VISIBILITY,
        }
        strategy = KnowledgeIngestStrategyFactory.create(source_type)
        return strategy.parse_chunks(file_path, context)

    def next_training_batch_version(self, *, source_type: str, source_file: str) -> dict[str, Any]:
        """计算本次上传资料的版本信息。"""

        latest_batch = self.repository.get_latest_batch_for_version(source_type=source_type, source_file=source_file)
        if not latest_batch:
            return {"version_group_id": None, "version_no": 1, "previous_batch_id": None}
        return {
            "version_group_id": latest_batch.get("version_group_id") or latest_batch["batch_id"],
            "version_no": int(latest_batch.get("version_no") or 1) + 1,
            "previous_batch_id": latest_batch["batch_id"],
        }

    def archive_previous_training_versions(self, batch: dict[str, Any]) -> None:
        """发布新版本后归档同版本组旧版本，并保留旧版本向量点用于回滚。"""

        version_group_id = batch.get("version_group_id") or batch["batch_id"]
        previous_versions = self.repository.list_published_batches_in_version_group(
            version_group_id,
            exclude_batch_id=batch["batch_id"],
        )
        for previous_batch in previous_versions:
            self.vector_service.update_metadata_by_metadata(
                "batch_id",
                previous_batch["batch_id"],
                metadata_updates={"status": "archived", "is_current": False},
            )
        self.repository.archive_other_versions(version_group_id=version_group_id, current_batch_id=batch["batch_id"])
        if previous_versions:
            logger.info(
                "[销售训练] 旧版本训练资料已归档并保留向量 版本组=%s 当前批次=%s 归档数量=%s",
                version_group_id,
                batch["batch_id"],
                len(previous_versions),
            )

    def mark_version_group_vectors_archived(self, version_group_id: str) -> None:
        """把同版本组内所有正式向量点标记为历史版本。"""

        version_batches = self.repository.list_published_batches_in_version_group(version_group_id)
        for version_batch in version_batches:
            self.vector_service.update_metadata_by_metadata(
                "batch_id",
                version_batch["batch_id"],
                metadata_updates={"status": "archived", "is_current": False},
            )

    def mark_batch_vectors_current(self, *, batch: dict[str, Any]) -> int:
        """把某个批次的正式向量点标记为当前发布版本。"""

        return self.vector_service.update_metadata_by_metadata(
            "batch_id",
            batch["batch_id"],
            metadata_updates={
                "status": "published",
                "is_current": True,
                "version_group_id": batch.get("version_group_id") or batch["batch_id"],
                "version_no": int(batch.get("version_no") or 1),
            },
        )

    def improve_training_chunks_if_needed(
            self,
            *,
            chunks: list[Any],
            file_path: str,
            batch_id: str,
            source_file: str,
            source_type: str,
            model_mode: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """根据质量门禁决定是否调用 LLM 兜底切分。"""

        evaluator = TrainingIngestQualityEvaluator()
        rule_report = evaluator.evaluate(chunks).to_dict()
        rule_report["selected_splitter"] = "rule_config"
        rule_report["llm_fallback_used"] = False
        rule_report["rule_score"] = rule_report.get("score")

        fallback_splitter = TrainingLlmFallbackSplitter()
        if not fallback_splitter.should_trigger(rule_report):
            return chunks, rule_report

        logger.info(
            "[销售训练][资料切分] 规则切分质量偏低，准备尝试LLM兜底 批次编号=%s 文件名=%s 规则质量分=%s",
            batch_id,
            source_file,
            rule_report.get("score"),
        )
        source_text = self.read_training_source_text(file_path)
        llm_chunks = fallback_splitter.split(
            source_text=source_text,
            batch_id=batch_id,
            source_file=source_file,
            source_type=source_type,
            visibility_default=DEFAULT_TRAINING_VISIBILITY,
            model_mode=model_mode,
        )
        if not llm_chunks:
            rule_report["llm_fallback_attempted"] = True
            rule_report["llm_fallback_used"] = False
            rule_report.setdefault("warnings", []).append("已尝试 LLM 兜底切分，但模型未返回可用切片，继续使用规则切分结果。")
            return chunks, rule_report

        llm_report = evaluator.evaluate(llm_chunks).to_dict()
        llm_score = int(llm_report.get("score") or 0)
        rule_score = int(rule_report.get("score") or 0)
        if llm_score > rule_score:
            llm_report["selected_splitter"] = "llm_fallback"
            llm_report["llm_fallback_attempted"] = True
            llm_report["llm_fallback_used"] = True
            llm_report["rule_score"] = rule_score
            llm_report["llm_score"] = llm_score
            logger.info(
                "[销售训练][资料切分] 已采用LLM兜底切分 批次编号=%s 规则质量分=%s LLM质量分=%s 切片数量=%s",
                batch_id,
                rule_score,
                llm_score,
                len(llm_chunks),
            )
            return llm_chunks, llm_report

        rule_report["llm_fallback_attempted"] = True
        rule_report["llm_fallback_used"] = False
        rule_report["llm_score"] = llm_score
        rule_report.setdefault("warnings", []).append("LLM 兜底切分未明显优于规则切分，已保留规则切分结果。")
        logger.info(
            "[销售训练][资料切分] LLM兜底未被采用 批次编号=%s 规则质量分=%s LLM质量分=%s",
            batch_id,
            rule_score,
            llm_score,
        )
        return chunks, rule_report

    def force_llm_reparse_chunks(
            self,
            *,
            rule_chunks: list[Any],
            file_path: str,
            batch_id: str,
            source_file: str,
            source_type: str,
            model_mode: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """人工触发 LLM 重新切分，并把结果和规则切分质量一起记录。"""

        evaluator = TrainingIngestQualityEvaluator()
        rule_report = evaluator.evaluate(rule_chunks).to_dict()
        source_text = self.read_training_source_text(file_path)
        fallback_splitter = TrainingLlmFallbackSplitter()
        fallback_splitter.config["enabled"] = True
        llm_chunks = fallback_splitter.split(
            source_text=source_text,
            batch_id=batch_id,
            source_file=source_file,
            source_type=source_type,
            visibility_default=DEFAULT_TRAINING_VISIBILITY,
            model_mode=model_mode,
        )
        if not llm_chunks:
            rule_report["selected_splitter"] = "rule_config"
            rule_report["llm_fallback_attempted"] = True
            rule_report["llm_fallback_used"] = False
            rule_report["rule_score"] = rule_report.get("score")
            rule_report.setdefault("warnings", []).append("人工触发 LLM 重新切分失败，已保留规则切分结果。")
            return rule_chunks, rule_report

        llm_report = evaluator.evaluate(llm_chunks).to_dict()
        llm_report["selected_splitter"] = "llm_fallback"
        llm_report["llm_fallback_attempted"] = True
        llm_report["llm_fallback_used"] = True
        llm_report["manual_reparse"] = True
        llm_report["rule_score"] = rule_report.get("score")
        llm_report["llm_score"] = llm_report.get("score")
        return llm_chunks, llm_report

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

    @staticmethod
    def read_training_source_text(file_path: str) -> str:
        """读取训练资料原文，供低质量 LLM 兜底切分使用。"""

        documents = FileProcessorFactory.load_documents(file_path)
        return "\n\n".join(document.page_content for document in documents).strip()

    @staticmethod
    def safe_filename(filename: str | None) -> str:
        """清理上传文件名，防止路径穿越。"""

        clean = os.path.basename(filename or "").strip()
        return clean or f"training_{uuid.uuid4().hex}.txt"

    @staticmethod
    def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """清理写入 Qdrant 的 metadata，避免空标签污染向量库 payload。"""

        compacted: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            compacted[key] = value
        return compacted

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
