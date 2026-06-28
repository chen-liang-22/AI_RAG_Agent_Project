"""销售训练核心应用服务。

这个文件承载销售陪练一期的主业务编排：
- 训练资料上传、切片、质量评估、发布到正式向量库；
- 学员画像和客户画像合成 AI 客户角色；
- 生成开放式训练目标、动态轮数和评分规则；
- 训练会话对话、每轮检索案例证据、最终评分报告。

这里使用外观模式把多个子系统收敛成一个稳定入口。
文件较大是因为一期先保证流程闭环，后续可以继续按资料、角色、会话、评分拆小。
"""

import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.knowledge.document_asset_service import DocumentAssetService
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.file_storage_service import get_file_storage_service
from app.infrastructure.id_generator import new_id
from app.infrastructure.vector_store_service import VectorStoreService
from core.model.factory import get_chat_model
from core.rag.file_processors import FileProcessorFactory
from app.application.training_support.factories.knowledge_ingest_strategy_factory import KnowledgeIngestStrategyFactory
from app.application.training_support.llm_ingest import TrainingLlmFallbackSplitter
from app.application.training_support.publish_validation import TrainingPublishValidator
from app.application.training_support.quality import TrainingIngestQualityEvaluator
from app.application.training_support.repository import TrainingRepository, utc_now_text
from app.application.training_support.schemas import (
    GoalSettingResponse,
    GoalStage,
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestion,
    SupplementQuestionGenerateResponse,
    TrainingPlanCreateRequest,
    TrainingPlanDeleteResponse,
    TrainingPlanDetailResponse,
    TrainingPlanListResponse,
    TrainingPlanSummaryResponse,
    TrainingPlanUpdateRequest,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeBatchResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeVersionListResponse,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionSummaryResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeChunkResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgeUploadResponse,
    TrainingScoreResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRecordResponse,
    TrainingTurnRequest,
    TrainingTurnResponse,
)
from app.application.training.training_query_service import TrainingQueryService
from app.application.training.training_goal_setting_service import TrainingGoalSettingService
from app.application.training.training_role_service import TrainingRoleService
from app.application.training.training_session_prompt_service import TrainingSessionPromptService
from app.application.training.training_score_service import TrainingScoreService
from core.utils.database_connection import DatabaseErrorTypes
from core.utils.logger_handler import logger
from core.utils.config_handler import training_conf
from core.utils.prompt_manager import prompt_manager


DEFAULT_TRAINING_COLLECTION_NAME = "sales_training_cases"
DEFAULT_TRAINING_STAGING_COLLECTION_NAME = "sales_training_cases_staging"
TRAINING_COLLECTION_NAME = DEFAULT_TRAINING_COLLECTION_NAME
ALLOWED_TRAINING_FILE_TYPES = {"txt", "pdf", "docx"}
DEFAULT_TRAINING_VISIBILITY = "visible"


def _load_training_collection_config() -> dict[str, str]:
    """读取销售训练正式库和临时库 collection 配置。"""

    config = {
        "published": DEFAULT_TRAINING_COLLECTION_NAME,
        "staging": DEFAULT_TRAINING_STAGING_COLLECTION_NAME,
    }
    collection_config = training_conf.get("collections") if isinstance(training_conf, dict) else {}
    if not isinstance(collection_config, dict):
        return config

    published_collection = str(collection_config.get("published") or "").strip()
    staging_collection = str(collection_config.get("staging") or "").strip()
    if published_collection:
        config["published"] = published_collection
    if staging_collection:
        config["staging"] = staging_collection
    return config


def _format_response_time(value: object) -> str | None:
    """把数据库时间字段统一转换成接口响应字符串。"""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class V2SalesTrainingCoreService:
    """销售训练一期外观服务。

    外观模式用于把文件解析、向量库、LLM、业务数据库这些子系统收拢成
    前端能理解的训练流程接口。一期流程较短，暂不引入 Graph。
    """

    def __init__(
            self,
            repository: TrainingRepository | None = None,
            knowledge_store=None,
            document_repository: DocumentRepository | None = None,
    ):
        """初始化销售训练核心服务。

        这里组合训练仓储、文件台账、正式向量库和临时向量库。
        knowledge_store 只保留给旧测试兼容，真实文件台账统一走 DocumentRepository。
        """

        # repository 支持注入，主要是为了单元测试或局部替换仓储实现。
        self.repository = repository or TrainingRepository()
        # 文件台账复用知识库 documents 表，统一写入 MySQL。
        self.document_repository = document_repository or DocumentRepository(store=knowledge_store)
        collection_config = _load_training_collection_config()
        self.training_collection_name = collection_config["published"]
        self.staging_collection_name = collection_config["staging"]
        # 正式训练知识使用独立 collection，避免和智能客服的普通知识库混在一起。
        self.vector_service = VectorStoreService(collection_name=self.training_collection_name)
        # 待人工审核的上传切片写入临时 collection，发布成功后再清理，避免关系型数据库保存正文切片。
        self.staging_vector_service = VectorStoreService(collection_name=self.staging_collection_name)
        # 训练证据召回独立成查询服务，核心服务只负责业务编排。
        self.query_service = TrainingQueryService(
            repository=self.repository,
            vector_service=self.vector_service,
            collection_name=self.training_collection_name,
        )
        # 角色生成相关的纯逻辑拆到独立服务，核心外观只负责编排数据库、向量库和 LLM 调用。
        self.role_service = TrainingRoleService()
        # 训练目标生成同样拆成纯逻辑服务，避免核心编排类继续堆积提示词和兜底模板。
        self.goal_setting_service = TrainingGoalSettingService()
        # 会话提示词和兜底话术拆到独立服务，核心外观继续负责仓库读写和流式编排。
        self.session_prompt_service = TrainingSessionPromptService()
        logger.info(
            "[销售训练] 核心服务初始化完成 正式Collection=%s 临时Collection=%s",
            self.training_collection_name,
            self.staging_collection_name,
        )

    def upload_knowledge(
            self,
            *,
            file: UploadFile,
            source_type: str,
            created_by: str | None,
            model_mode: str | None = None,
    ) -> TrainingKnowledgeUploadResponse:
        """上传训练资料并生成待确认预览。

        主流程：
        1. 保存上传文件到 MinIO；
        2. 创建上传批次记录，状态为 parsing；
        3. 根据 source_type 选择切片策略；
        4. 对切片结果做质量评估；
        5. 保存切片明细；
        6. 状态改为 pending_review，等待人工确认发布。
        """

        # 阶段 1：清洗文件名并校验扩展名，避免危险路径和不支持的文件类型进入后续流程。
        filename = self._safe_filename(file.filename)
        # rsplit(".", 1) 只从右边切一次，能正确处理 a.b.docx 这种文件名。
        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if file_type not in ALLOWED_TRAINING_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"训练知识暂不支持该文件类型：{file_type}")

        # 阶段 2：为本次上传生成 document_id 和 batch_id。
        # document_id 管文件台账，batch_id 管训练资料的审核、发布、回滚流程。
        document_id = new_id()
        batch_id = new_id()
        # 原始文件直接持久化到 MinIO，本地只在解析时使用临时下载文件。
        stored_file = get_file_storage_service().save_upload_file(
            file=file,
            filename=filename,
            prefix="training",
            owner_id=document_id,
        )

        # 阶段 3：计算文件 MD5 做内容级去重。
        # 只要文件内容完全相同，就直接复用已经 published 的历史批次。
        # 文件上传 MinIO 前已经计算过 MD5，MD5 可以理解为文件内容的唯一指纹。
        file_md5 = stored_file.file_md5
        # 用文件 MD5 查询是否已经存在发布成功的同内容批次，避免重复解析和重复写入向量库。
        existing_batch = self.repository.get_published_batch_by_md5(file_md5)
        # Python 中有值的对象会被当作 True；这里表示查到了重复文件批次，就进入复用逻辑。
        if existing_batch:
            # 文件内容完全一样时不重复写入向量库，直接返回已有批次。
            # 刚上传到 MinIO 的重复对象不再保留，避免对象存储堆积重复文件。
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
                source_file=self._batch_file_info(existing_batch).get("source_file"),
                duplicate_of=existing_batch["batch_id"],
                quality_report=self._load_json(existing_batch.get("quality_report_json"), {}),
            )

        version_info = self._next_training_batch_version(source_type=source_type, source_file=filename)
        # 阶段 4：先创建 documents 文件台账，再创建训练批次记录，状态置为 parsing。
        # 这样即使后续解析失败，也能在后台和日志里追踪到失败原因。
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
        # self.repository 是当前服务的仓储成员变量，专门负责读写训练相关的关系型数据。
        batch = self.repository.create_batch(
            # 本次上传批次的唯一编号，后续查切片、删向量、预览文件都靠它关联。
            batch_id=batch_id,
            # 文件基础信息统一保存在 documents 表，训练批次只保留关联 ID。
            document_id=document_id,
            # 资料来源类型，例如 lms_case，用来决定后续采用哪一种解析切片策略。
            source_type=source_type,
            # 用户上传的原始文件名，保存到数据库后用于列表展示和文件预览。
            source_file=filename,
            # 新链路不再把文件路径和 MD5 冗余写入训练批次表。
            file_path=None,
            file_md5=None,
            version_group_id=version_info["version_group_id"],
            version_no=version_info["version_no"],
            previous_batch_id=version_info.get("previous_batch_id"),
            is_current=False,
            # 默认可见性不再由前端传入，统一由后端作为兜底值维护。
            # visible 表示默认切片对学员可见；策略解析时也可以按片段覆盖成 hidden/scoring_only。
            visibility_default=DEFAULT_TRAINING_VISIBILITY,
            # 批次初始状态设为 parsing，表示已创建记录，正在解析和入库。 	解析中	文件已保存，正在解析、切片并写入向量库
            status="parsing",
            # 创建人标识，通常来自当前登录用户；为空时表示系统或匿名上传。
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
            # 阶段 5：解析文件并生成预览切片；待审核正文只写入临时向量库。
            with self._download_batch_file(batch) as file_path:
                logger.info("[销售训练] 训练知识解析开始 批次编号=%s 临时文件=%s", batch_id, file_path)
                chunks = self._parse_training_chunks(
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=filename,
                    source_type=source_type,
                )
                if not chunks:
                    raise ValueError("文件没有切出有效训练知识")
                logger.info("[销售训练] 训练知识规则切片完成 批次编号=%s 切片数量=%s", batch_id, len(chunks))

                # 阶段 6：计算切片质量报告；质量较低时才尝试 LLM 兜底切分。
                chunks, quality_report = self._improve_training_chunks_if_needed(
                    chunks=chunks,
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=filename,
                    source_type=source_type,
                    model_mode=model_mode,
                )
            # 阶段 7：把待审核切片写入临时向量库，前端预览也从临时库按 batch_id 读取。
            logger.info("[销售训练] 训练知识写入临时向量库开始 批次编号=%s 临时Collection=%s", batch_id, self.staging_collection_name)
            point_count = self._write_staging_chunks(batch=batch, chunks=chunks, source_type=source_type)
            # 阶段 8：上传预览完成，等待人工确认发布。
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
            # 任意阶段失败都把批次标记为 parsing_failed，并保留错误消息。
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
            items=[self._batch_response(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def preview_batch(self, batch_id: str, *, max_chars: int = 30000) -> TrainingKnowledgePreviewResponse:
        """返回训练资料上传文件的站内预览数据。

        查看切片已经有独立入口，所以这里优先预览原文件文本；
        DOCX/PDF 会解析为文本，避免浏览器把 Word 文件当成下载处理。
        """

        batch = self._get_active_batch(batch_id)
        file_info = self._batch_file_info(batch)
        file_url = str(file_info.get("public_url") or "").strip()
        if not file_url:
            object_name = str(file_info.get("object_name") or "").strip()
            if not object_name:
                raise HTTPException(status_code=400, detail="训练资料缺少 MinIO 对象路径，请先完成历史文件迁移")
            file_url = get_file_storage_service().client.get_public_url(
                object_name,
                bucket_name=file_info.get("bucket_name"),
            )
        preview = self._build_batch_preview(batch, max_chars=max_chars)

        return TrainingKnowledgePreviewResponse(
            batch=self._batch_response(batch),
            preview_type=preview["preview_type"],
            content=preview["content"],
            truncated=preview["truncated"],
            file_url=file_url,
            charset=preview.get("charset"),
        )

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次，并通过统一文件资产服务清理全链路数据。"""

        batch = self._get_active_batch(batch_id)
        document_id = str(batch.get("document_id") or "").strip()
        if not document_id:
            return self._delete_legacy_batch_without_document(batch_id)

        DocumentAssetService().delete_document_asset(document_id)
        logger.info("[销售训练] 训练资料已删除 批次编号=%s 文档编号=%s", batch_id, document_id)
        return TrainingKnowledgeDeleteResponse(status="deleted", batch_id=batch_id)

    def _delete_legacy_batch_without_document(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除没有 document_id 的历史训练批次。

        老数据只存在 training_knowledge_batches 和训练向量库里，无法走 documents 统一文件资产链路。
        因此这里按 batch_id 清理正式库、临时库和批次记录，保留历史数据兼容能力。
        """

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
        """人工确认发布训练资料。

        上传阶段已经把待审核切片写入临时 Qdrant collection。
        发布阶段只把临时向量点复制到正式 collection，成功后删除临时点。
        """

        batch = self._get_active_batch(batch_id)
        if batch["status"] == "published":
            logger.info("[销售训练] 训练资料已经发布，直接返回 批次编号=%s", batch_id)
            return TrainingKnowledgePublishResponse(
                batch_id=batch_id,
                status="published",
                chunk_count=int(batch.get("chunk_count") or 0),
                point_count=int(batch.get("point_count") or 0),
                quality_report=self._load_json(batch.get("quality_report_json"), {}),
            )
        if batch["status"] not in {"pending_review", "embedding", "publish_failed"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许发布：{batch['status']}")

        chunks = self._list_staging_chunk_rows(batch_id)
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
            copied_count = self._publish_staging_vectors(batch=batch)
            quality_report = self._load_json(batch.get("quality_report_json"), {})
            publish_validation = TrainingPublishValidator().validate(
                vector_service=self.vector_service,
                batch_id=batch_id,
                chunks=chunks,
            )
            quality_report["publish_validation"] = publish_validation
            self._archive_previous_training_versions(batch)
            self.repository.update_batch_status(
                batch_id,
                status="published",
                chunk_count=len(chunks),
                point_count=copied_count,
                quality_report=quality_report,
                is_current=True,
            )
            self._delete_staging_vectors(batch_id)
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
        """回滚训练资料到指定历史版本。

        历史版本的正式向量点会长期保留，回滚时只切换当前版本标记和业务数据库状态。
        """

        batch = self._get_active_batch(batch_id)
        if batch["status"] not in {"published", "archived"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许回滚：{batch['status']}")
        chunks = self._list_published_chunk_rows(batch_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="该版本没有可回滚的训练切片，请重新上传资料")

        version_group_id = batch.get("version_group_id") or batch["batch_id"]
        try:
            self._mark_version_group_vectors_archived(version_group_id)
            point_count = self._mark_batch_vectors_current(batch=batch)
            quality_report = self._load_json(batch.get("quality_report_json"), {})
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
        """重新切分未发布训练资料。

        该接口用于人工预览发现规则切分不理想时，主动触发 LLM 兜底切分。
        已发布版本不能直接重切，避免绕过人工确认并破坏临时库到正式库的发布边界。
        """

        batch = self._get_active_batch(batch_id)
        if batch["status"] not in {"pending_review", "parsing_failed", "publish_failed"}:
            raise HTTPException(status_code=400, detail=f"当前状态不允许重新切分：{batch['status']}")

        source_type = str(batch.get("source_type") or "lms_case")
        source_file = str(batch.get("source_file") or "")
        try:
            with self._download_batch_file(batch) as file_path:
                rule_chunks = self._parse_training_chunks(
                    file_path=file_path,
                    batch_id=batch_id,
                    source_file=source_file,
                    source_type=source_type,
                )
                if use_llm_fallback:
                    chunks, quality_report = self._force_llm_reparse_chunks(
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
            point_count = self._write_staging_chunks(batch=batch, chunks=chunks, source_type=source_type)
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

        batch = self._get_active_batch(batch_id)
        version_group_id = str(batch.get("version_group_id") or batch["batch_id"])
        rows = self.repository.list_batches_in_version_group(version_group_id)
        return TrainingKnowledgeVersionListResponse(
            version_group_id=version_group_id,
            items=[self._batch_response(row) for row in rows],
        )

    def list_chunks(self, batch_id: str) -> TrainingKnowledgeChunkListResponse:
        """查询某个上传批次的训练知识切片。"""

        batch = self._get_active_batch(batch_id)
        chunks = []
        chunk_rows = self._list_batch_chunk_rows(batch)
        for row in chunk_rows:
            metadata = self._load_json(row.get("metadata_json"), {})
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

    def create_plan(self, request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
        """创建训练方案。

        训练方案先保存“输入快照”，还不会自动生成角色。
        这样用户可以先命名、检查画像和场景，再按步骤生成后续内容。
        """

        plan = self.repository.create_plan(
            plan_name=request.plan_name.strip(),
            trainee=request.trainee.model_dump(),
            profile_type=request.profile_type,
            selected_fields=request.selected_fields,
            scenario_description=request.scenario_description.strip(),
            extra_details=request.extra_details.strip(),
            model_mode=request.model_mode,
        )
        logger.info("[销售训练] 训练方案创建完成 方案编号=%s 名称=%s", plan["plan_id"], plan["plan_name"])
        return self._plan_detail_response(plan)

    def list_plans(self, *, page: int = 1, page_size: int = 10, keyword: str | None = None) -> TrainingPlanListResponse:
        """分页查询训练方案列表。"""

        safe_page = max(1, page)
        safe_page_size = max(1, min(50, page_size))
        rows, total = self.repository.list_plans(page=safe_page, page_size=safe_page_size, keyword=keyword)
        return TrainingPlanListResponse(
            items=[self._plan_summary(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def get_plan_detail(self, plan_id: str) -> TrainingPlanDetailResponse:
        """查询训练方案完整详情。"""

        return self._plan_detail_response(self._require_plan(plan_id))

    def delete_plan(self, plan_id: str) -> TrainingPlanDeleteResponse:
        """删除训练方案。

        训练方案是销售陪练配置入口。删除它只会让方案从列表和详情里消失，
        不清理训练资料、向量库、MinIO 文件，也不删除历史训练会话依赖的角色和阶段配置。
        """

        plan = self._require_plan(plan_id)
        deleted = self.repository.delete_plan(plan_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        logger.info("[销售训练] 训练方案已删除 方案编号=%s 名称=%s", plan_id, plan.get("plan_name"))
        return TrainingPlanDeleteResponse(status="deleted", plan_id=plan_id)

    def update_plan(self, plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
        """修改训练方案。

        依赖关系说明：
        - 修改学员画像/客户画像/场景/补充细节：角色已不可信，阶段和评分也必须重新生成；
        - 修改角色扮演画像或隐藏画像：训练阶段和评分必须重新生成；
        - 修改训练阶段：评分规则需要重新确认；
        - 只修改评分规则：不影响前面的角色和阶段。
        """

        plan = self._require_plan(plan_id)
        updates: dict[str, Any] = {}
        role_input_changed = False
        role_content_changed = False
        goal_changed = False

        if request.plan_name is not None and request.plan_name.strip() != plan["plan_name"]:
            updates["plan_name"] = request.plan_name.strip()
        if request.trainee is not None:
            trainee_data = request.trainee.model_dump()
            if self._json_changed(self._load_json(plan.get("trainee_json"), {}), trainee_data):
                updates["trainee_json"] = self.repository._json(trainee_data)
                updates["trainee_id"] = trainee_data["trainee_id"]
                updates["trainee_name"] = trainee_data.get("trainee_name") or "销售学员"
                role_input_changed = True
        if request.profile_type is not None and request.profile_type != plan["profile_type"]:
            updates["profile_type"] = request.profile_type
            role_input_changed = True
        if request.selected_fields is not None:
            if self._json_changed(self._load_json(plan.get("selected_fields_json"), {}), request.selected_fields):
                updates["selected_fields_json"] = self.repository._json(request.selected_fields)
                role_input_changed = True
        if request.scenario_description is not None:
            scenario_description = request.scenario_description.strip()
            if scenario_description != (plan.get("scenario_description") or ""):
                updates["scenario_description"] = scenario_description
                role_input_changed = True
        if request.extra_details is not None:
            extra_details = request.extra_details.strip()
            if extra_details != (plan.get("extra_details") or ""):
                updates["extra_details"] = extra_details
                role_input_changed = True
        if request.model_mode is not None:
            updates["model_mode"] = request.model_mode

        if role_input_changed:
            updates.update({
                "active_profile_id": None,
                "active_setting_id": None,
                "role_status": "stale",
                "goal_status": "stale",
                "score_status": "stale",
            })

        active_profile_id = updates.get("active_profile_id", plan.get("active_profile_id"))
        if active_profile_id and any(value is not None for value in (
                request.role_confirm_card,
                request.visible_profile,
                request.hidden_profile,
                request.role_profile,
        )):
            self.repository.update_role_profile(
                active_profile_id,
                visible_profile=request.visible_profile,
                hidden_profile=request.hidden_profile,
                role_profile=request.role_profile,
                role_confirm_card=request.role_confirm_card,
            )
            role_content_changed = request.hidden_profile is not None or request.role_profile is not None
            if role_content_changed and not role_input_changed:
                updates.update({
                    "active_setting_id": None,
                    "goal_status": "stale",
                    "score_status": "stale",
                })

        active_setting_id = updates.get("active_setting_id", plan.get("active_setting_id"))
        if active_setting_id and (
                request.training_purpose is not None
                or request.round_limit is not None
                or request.stages is not None
        ):
            self.repository.update_goal_setting(
                active_setting_id,
                training_purpose=request.training_purpose.strip() if request.training_purpose is not None else None,
                round_limit=request.round_limit,
                stages=[item.model_dump() for item in request.stages] if request.stages is not None else None,
            )
            goal_changed = True
        if active_setting_id and request.scoring_rules is not None:
            self.repository.update_goal_setting(active_setting_id, scoring_rules=request.scoring_rules)
        if goal_changed and not role_input_changed and not role_content_changed:
            updates["score_status"] = "stale"

        try:
            updated = self.repository.update_plan(plan_id, **updates) if updates else self._require_plan(plan_id)
        except DatabaseErrorTypes as exc:
            logger.error("[销售训练] 训练方案保存数据库异常 方案编号=%s 错误=%s", plan_id, exc, exc_info=True)
            raise
        logger.info(
            "[销售训练] 训练方案已修改 方案编号=%s 角色输入变化=%s 角色内容变化=%s 阶段变化=%s",
            plan_id,
            role_input_changed,
            role_content_changed,
            goal_changed,
        )
        return self._plan_detail_response(updated)

    def generate_supplement_questions(self, request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
        """生成 AI 陪练角色前的补充问答题。

        这是角色生成的前置澄清步骤：先让管理员选择客户真实顾虑、价值判断、
        业务痛点等细节，再把答案并入 extra_details 生成更稳定的角色。
        """

        query = self._build_role_query(request)
        logger.info(
            "[销售训练][补充问答题] 开始生成 方案编号=%s 学员=%s 模型档位=%s 查询预览=%s",
            request.plan_id or "-",
            request.trainee.trainee_id,
            request.model_mode or "默认",
            self._short_text(query),
        )
        evidence = self._search_training_evidence(query, visibility=("visible", "hidden"), k=4)
        prompt = self._supplement_questions_prompt(request, evidence)
        fallback = {"questions": self._fallback_supplement_questions(request)}
        result = self._invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=fallback,
            task_name="补充问答题生成",
        )
        questions = self._normalize_supplement_questions(result.get("questions"), request)
        logger.info(
            "[销售训练][补充问答题] 生成完成 题目数=%s 学员=%s 证据数量=%s",
            len(questions),
            request.trainee.trainee_id,
            len(evidence),
        )
        return SupplementQuestionGenerateResponse(questions=questions)

    def polish_scenario(self, request: ScenarioPolishRequest) -> ScenarioPolishResponse:
        """根据客户画像字段润色训练场景描述。

        这是销售陪练服务的一个小外观方法：前端只关心“把场景润色好”，
        具体调用哪个模型、如何兜底，都收敛在服务层。
        """

        prompt = self._scenario_polish_prompt(request)
        fallback = {"polished_scenario": self._fallback_polished_scenario(request)}
        logger.info(
            "[销售训练][场景润色] 开始润色 画像类型=%s 模型档位=%s 原始长度=%s 选择字段数=%s",
            request.profile_type,
            request.model_mode or "默认",
            len(request.scenario_description),
            len(request.selected_fields or {}),
        )
        result = self._invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=fallback,
            task_name="场景描述润色",
        )
        polished_scenario = str(result.get("polished_scenario") or "").strip()
        if not polished_scenario:
            polished_scenario = self._fallback_polished_scenario(request)
        logger.info(
            "[销售训练] 场景描述AI润色完成 画像类型=%s 原始长度=%s 润色后长度=%s",
            request.profile_type,
            len(request.scenario_description),
            len(polished_scenario),
        )
        return ScenarioPolishResponse(
            polished_scenario=polished_scenario,
            original_scenario=request.scenario_description,
        )

    def generate_role(self, request: RoleGenerateRequest) -> RoleGenerateResponse:
        """生成 AI 陪练角色。

        角色生成不是单纯让 LLM 编故事，而是先从训练向量库召回案例证据，
        再把“学员画像 + 客户字段 + 场景 + 证据”一起交给模型。
        """

        if request.plan_id:
            self._require_plan(request.plan_id)
        query = self._build_role_query(request)
        logger.info(
            "[销售训练][角色生成] 开始生成 方案编号=%s 学员=%s 画像类型=%s 模型档位=%s 选择字段数=%s 场景长度=%s",
            request.plan_id or "-",
            request.trainee.trainee_id,
            request.profile_type,
            request.model_mode or "默认",
            len(request.selected_fields or {}),
            len(request.scenario_description or ""),
        )
        # 角色生成阶段允许使用 visible 和 hidden 知识：
        # - visible：学员也能看到的显性案例；
        # - hidden：只给 AI 客户使用的底层顾虑/隐性心理。
        evidence = self._search_training_evidence(query, visibility=("visible", "hidden"), k=6)
        logger.info(
            "[销售训练][角色生成] 证据召回完成 方案编号=%s 证据数量=%s 命中切片=%s",
            request.plan_id or "-",
            len(evidence),
            self._join_values(item.get("chunk_id") for item in evidence),
        )
        prompt = self._role_prompt(request, evidence)
        result = self._invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=self._fallback_role(request, evidence),
            task_name="AI客户角色生成",
        )

        visible_profile = result.get("visible_profile") or {}
        hidden_profile = result.get("hidden_profile") or {}
        role_profile = result.get("role_profile") or {}
        role_confirm_card = result.get("role_confirm_card") or visible_profile

        saved = self.repository.save_role_profile(
            plan_id=request.plan_id,
            trainee_id=request.trainee.trainee_id,
            profile_type=request.profile_type,
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            selected_fields=request.selected_fields,
            scenario_description=request.scenario_description,
            extra_details=request.extra_details,
            retrieved_evidence=evidence,
            status="confirmed",
        )
        if request.plan_id:
            self.repository.attach_role_to_plan(request.plan_id, saved["profile_id"])
        logger.info(
            "[销售训练][角色生成] 生成完成 角色编号=%s 学员=%s 证据数量=%s 角色字段=%s",
            saved["profile_id"],
            request.trainee.trainee_id,
            len(evidence),
            self._dict_key_text(role_profile),
        )
        return RoleGenerateResponse(
            profile_id=saved["profile_id"],
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            hidden_summary="已生成隐藏心理画像，学员不可见",
            retrieved_cases=evidence,
            knowledge_facts=[item["content"][:160] for item in evidence],
        )

    def generate_goal_setting(
            self,
            *,
            profile_id: str,
            trainee_id: str,
            training_mode: str,
            plan_id: str | None = None,
            model_mode: str | None = None,
    ) -> GoalSettingResponse:
        """生成一期开放式训练设置。

        一期只支持开放式训练，所以这里只生成一个阶段。
        round_limit 由 LLM 根据角色复杂度动态给出，后端再做 5-100 的安全边界。
        """

        if training_mode != "open":
            raise HTTPException(status_code=400, detail="流程式训练二期支持，一期只支持开放式")

        if plan_id:
            self._require_plan(plan_id)
        profile = self._require_role_profile(profile_id)
        prompt = self._goal_prompt(profile)
        logger.info(
            "[销售训练][训练设置] 开始生成 方案编号=%s 角色编号=%s 学员=%s 模式=%s 模型档位=%s",
            plan_id or "-",
            profile_id,
            trainee_id,
            training_mode,
            model_mode or "默认",
        )
        result = self._invoke_json(
            prompt,
            model_mode=model_mode,
            fallback=self._fallback_goal(profile),
            task_name="训练阶段和评分规则生成",
        )
        round_limit = self._normalize_round_limit(result.get("round_limit"))
        stages = result.get("stages") or []
        if not stages:
            stages = self._fallback_goal(profile)["stages"]
        scoring_rules = self._normalize_scoring_rules(result.get("scoring_rules"), stages[:1], profile)

        saved = self.repository.save_goal_setting(
            profile_id=profile_id,
            trainee_id=trainee_id,
            training_mode="open",
            training_purpose=str(result.get("training_purpose") or "开放式销售训练")[:20],
            round_limit=round_limit,
            stages=stages[:1],
            scoring_rules=scoring_rules,
            plan_id=plan_id,
            status="confirmed",
        )
        if plan_id:
            self._require_plan(plan_id)
            self.repository.attach_goal_to_plan(plan_id, saved["setting_id"])
        logger.info(
            "[销售训练][训练设置] 生成完成 设置编号=%s 轮数=%s 阶段数量=%s 评分维度=%s",
            saved["setting_id"],
            round_limit,
            len(stages[:1]),
            len(scoring_rules.get("dimensions") or []),
        )
        return self._goal_response(saved)

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始一次开放式训练。

        和普通聊天不同，训练一开始应该由 AI 客户先“在场”。
        所以创建会话后会立刻生成并保存 round_no=0 的客户开场白。
        """

        setting = self._require_goal_setting(request.setting_id)
        if setting["profile_id"] != request.profile_id:
            raise HTTPException(status_code=400, detail="训练设置和陪练角色不匹配")
        response_mode = self._normalize_response_mode(request.response_mode)
        logger.info(
            "[销售训练] 创建训练会话开始 角色编号=%s 设置编号=%s 学员编号=%s 回复模式=%s 轮数上限=%s",
            request.profile_id,
            request.setting_id,
            request.trainee_id,
            response_mode,
            setting["round_limit"],
        )
        session = self.repository.create_session(
            profile_id=request.profile_id,
            setting_id=request.setting_id,
            trainee_id=request.trainee_id,
            training_mode="open",
            response_mode=response_mode,
            round_limit=int(setting["round_limit"]),
            status="active",
        )
        opening_message = self._generate_opening_message(session, model_mode=request.model_mode)
        self.repository.add_turn(
            session_id=session["session_id"],
            role="customer",
            content=opening_message,
            round_no=0,
            response_mode=response_mode,
            stage_no=1,
            started_at=session["started_at"],
            submitted_at=utc_now_text(),
            metadata={"turn_type": "opening"},
        )
        logger.info("[销售训练] 训练会话开始 会话编号=%s 回复模式=%s 已生成开场白", session["session_id"], response_mode)
        return self._session_response(session, opening_message=opening_message)

    def list_sessions(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            trainee_id: str | None = None,
    ) -> TrainingSessionListResponse:
        """分页查询训练历史。"""

        safe_page = max(1, page)
        safe_page_size = max(1, min(50, page_size))
        rows, total = self.repository.list_sessions(page=safe_page, page_size=safe_page_size, trainee_id=trainee_id)
        return TrainingSessionListResponse(
            items=[self._session_summary(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练复盘详情。

        这个接口给前端“最近训练”使用：
        - session：会话摘要；
        - turns：完整对话；
        - role_profile：角色确认卡片；
        - goal_setting：训练目标；
        - score：已有评分报告。
        """

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")

        turns = self.repository.list_turns(session_id)
        profile = self._require_role_profile(session["profile_id"])
        setting = self._require_goal_setting(session["setting_id"])
        score = self.repository.get_latest_score_by_session(session_id)
        visible_profile = self._load_json(profile.get("visible_profile_json"), {})
        hidden_profile = self._load_json(profile.get("hidden_profile_json"), {})
        role_profile = self._load_json(profile.get("role_profile_json"), {})
        role_confirm_card = self._load_json(profile.get("role_confirm_card_json"), {})
        retrieved_evidence = self._load_json(profile.get("retrieved_evidence_json"), [])
        # {**session, "answered_count": ...} 是字典合并写法：
        # 先复制 session，再覆盖/新增 answered_count 字段。
        summary = self._session_summary({**session, "answered_count": sum(1 for item in turns if item["role"] == "trainee")})
        return TrainingSessionDetailResponse(
            session=summary,
            turns=[self._turn_record(item) for item in turns],
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            goal_setting={
                "setting_id": setting["setting_id"],
                "training_purpose": setting["training_purpose"],
                "round_limit": int(setting["round_limit"]),
                "stages": self._load_json(setting.get("stages_json"), []),
                "scoring_rules": self._load_json(setting.get("scoring_rules_json"), self._default_scoring_rules()),
            },
            knowledge_facts=[item["content"][:160] for item in retrieved_evidence if item.get("content")],
            score=self._score_response(score) if score else None,
        )

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回复。"""

        response = self._handle_turn(session_id, request, stream=False)
        return response

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并返回 SSE 流。

        Python 的 Iterator[str] + yield 是生成器写法。
        FastAPI StreamingResponse 会一边读取 yield 出来的字符串，一边推给浏览器。

        这里返回的是 SSE 文本事件：
        - retrieval_done：本轮检索完成；
        - customer_delta：AI 客户回复增量；
        - stage_decision：阶段/会话状态；
        - turn_done：本轮完成；
        - error：异常。
        """

        try:
            # perf_counter 适合做耗时统计，不受系统时间调整影响。
            start_perf = time.perf_counter()
            session = self._require_session(session_id)
            started_at = utc_now_text()
            round_no = self.repository.next_round_no(session_id)
            logger.info(
                "[销售训练][流式轮次] 开始处理 会话编号=%s 轮次=%s 模型档位=%s 学员输入长度=%s 输入预览=%s",
                session_id,
                round_no,
                request.model_mode or "默认",
                len(request.message or ""),
                self._short_text(request.message),
            )
            self.repository.add_turn(
                session_id=session_id,
                role="trainee",
                content=request.message,
                round_no=round_no,
                response_mode="stream",
                started_at=started_at,
                submitted_at=started_at,
            )
            evidence = self._turn_evidence(session, request.message)
            logger.info(
                "[销售训练][流式轮次] 本轮检索完成 会话编号=%s 轮次=%s 证据数量=%s 命中切片=%s",
                session_id,
                round_no,
                len(evidence),
                self._join_values(item.get("chunk_id") for item in evidence),
            )
            # 先把检索结果发给前端，方便页面显示“本轮命中了哪些切片”。
            yield self._sse("retrieval_done", {"retrieved_chunk_ids": [item["chunk_id"] for item in evidence], "evidence": evidence})

            chunks: list[str] = []
            # model.stream(...) 会逐块返回模型输出，前端就能看到“打字机效果”。
            for chunk in self._stream_customer_reply(session, request.message, evidence, model_mode=request.model_mode):
                chunks.append(chunk)
                yield self._sse("customer_delta", {"content": chunk})

            customer_reply = "".join(chunks).strip() or self._fallback_customer_reply(evidence)
            response = self._finish_customer_turn(
                session=session,
                round_no=round_no,
                customer_reply=customer_reply,
                response_mode="stream",
                evidence=evidence,
                started_at=started_at,
                start_perf=start_perf,
            )
            yield self._sse(
                "stage_decision",
                {"stage_status": response.stage_status, "session_status": response.session_status},
            )
            yield self._sse("turn_done", response.model_dump())
            logger.info(
                "[销售训练][流式轮次] 处理完成 会话编号=%s 轮次=%s 回复长度=%s 总耗时秒=%s",
                session_id,
                round_no,
                len(customer_reply),
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
        except Exception as exc:
            logger.error("[销售训练] 流式训练轮次失败 会话编号=%s 错误=%s", session_id, exc, exc_info=True)
            yield self._sse("error", {"error": str(exc)})

    def final_score(self, session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。

        注意这里先查 session，再查已有评分：
        已完成会话重复点击“生成评分”时直接返回已有报告，避免重复写入评分记录。
        """

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")
        existing_score = self.repository.get_latest_score_by_session(session_id)
        if existing_score and session["status"] == "completed":
            logger.info("[销售训练] 训练评分已存在，直接返回 会话编号=%s", session_id)
            return self._score_response(existing_score)
        if session["status"] not in {"active", "scoring"}:
            raise HTTPException(status_code=400, detail=f"当前训练状态不允许评分：{session['status']}")

        turns = self.repository.list_turns(session_id)
        if not any(turn["role"] == "trainee" for turn in turns):
            raise HTTPException(status_code=400, detail="没有学员回复，不能评分")

        setting = self._require_goal_setting(session["setting_id"])
        profile = self._require_role_profile(session["profile_id"])
        conversation_text = self._conversation_text(turns)
        logger.info(
            "[销售训练][评分] 开始生成评分 会话编号=%s 模型档位=%s 对话轮次=%s 对话长度=%s",
            session_id,
            model_mode or "默认",
            len(turns),
            len(conversation_text),
        )
        evidence = self._search_training_evidence(conversation_text, visibility=("visible", "scoring_only"), k=6)
        logger.info(
            "[销售训练][评分] 评分证据召回完成 会话编号=%s 证据数量=%s 命中切片=%s",
            session_id,
            len(evidence),
            self._join_values(item.get("chunk_id") for item in evidence),
        )
        result = self._invoke_json(
            self._score_prompt(profile, setting, turns, evidence),
            model_mode=model_mode,
            fallback=self._fallback_score(turns, evidence),
            task_name="训练评分报告生成",
        )

        general_score = int(max(0, min(40, result.get("general_score") or 32)))
        stage_score = int(max(0, min(60, result.get("stage_score") or 43)))
        penalty_score = int(max(0, min(20, result.get("penalty_score") or 0)))
        # 最终得分以后端公式为准，不直接相信 LLM 返回的 total_score。
        final_score = int(max(0, min(100, general_score + stage_score - penalty_score)))
        level = self._score_level(final_score)
        report = {
            "hit_points": result.get("hit_points") or [],
            "missing_points": result.get("missing_points") or [],
            "wrong_points": result.get("wrong_points") or [],
            "evidence_refs": result.get("evidence_refs") or [],
            "improvement_advice": result.get("improvement_advice") or "",
            "reference_script": result.get("reference_script") or "",
            "next_training_plan": result.get("next_training_plan") or [],
            "scoring_rules": self._load_json(setting.get("scoring_rules_json"), self._default_scoring_rules()),
        }
        score = self.repository.save_score(
            session_id=session_id,
            general_score=general_score,
            stage_score=stage_score,
            penalty_score=penalty_score,
            final_score=final_score,
            level=level,
            is_passed=final_score >= 75,
            detail=report,
            review_status="confirmed",
        )
        self.repository.update_session_status(session_id, status="completed", total_score=final_score, level=level, report=report)
        logger.info("[销售训练] 训练评分完成 会话编号=%s 得分=%s 等级=%s", session_id, final_score, level)
        return self._score_response(score)

    def _handle_turn(self, session_id: str, request: TrainingTurnRequest, *, stream: bool) -> TrainingTurnResponse:
        """处理一次训练轮次，非流式和流式共用同一套收尾逻辑。

        stream 参数目前只是保留语义，真正流式走 stream_turn。
        一次性接口会在这里完整生成 AI 客户回复后再返回 JSON。
        """

        session = self._require_session(session_id)
        start_perf = time.perf_counter()
        started_at = utc_now_text()
        round_no = self.repository.next_round_no(session_id)
        response_mode = self._normalize_response_mode(request.response_mode)
        logger.info(
            "[销售训练][一次性轮次] 开始处理 会话编号=%s 轮次=%s 回复模式=%s 模型档位=%s 学员输入长度=%s 输入预览=%s",
            session_id,
            round_no,
            response_mode,
            request.model_mode or "默认",
            len(request.message or ""),
            self._short_text(request.message),
        )
        self.repository.add_turn(
            session_id=session_id,
            role="trainee",
            content=request.message,
            round_no=round_no,
            response_mode=response_mode,
            started_at=started_at,
            submitted_at=started_at,
        )
        evidence = self._turn_evidence(session, request.message)
        logger.info(
            "[销售训练][一次性轮次] 本轮检索完成 会话编号=%s 轮次=%s 证据数量=%s 命中切片=%s",
            session_id,
            round_no,
            len(evidence),
            self._join_values(item.get("chunk_id") for item in evidence),
        )
        customer_reply = self._generate_customer_reply(session, request.message, evidence, model_mode=request.model_mode)
        return self._finish_customer_turn(
            session=session,
            round_no=round_no,
            customer_reply=customer_reply,
            response_mode=response_mode,
            evidence=evidence,
            started_at=started_at,
            start_perf=start_perf,
        )

    def _finish_customer_turn(
            self,
            *,
            session: dict[str, Any],
            round_no: int,
            customer_reply: str,
            response_mode: str,
            evidence: list[dict[str, Any]],
            started_at: str,
            start_perf: float,
    ) -> TrainingTurnResponse:
        """保存 AI 客户回复并更新本轮状态。

        一期只有开放式一个阶段，因此 current_stage_no 固定为 1。
        当学员轮次达到 round_limit 时，会话进入 scoring 状态，提示前端可以评分。
        """

        session_status = "active"
        stage_status = "active"
        if round_no >= int(session["round_limit"]):
            session_status = "scoring"
            stage_status = "round_limit_reached"
            self.repository.update_session_status(session["session_id"], status=session_status)

        now = utc_now_text()
        # 这里统计从接收学员回复到 AI 客户回复落库的端到端耗时，便于前端展示训练响应速度。
        response_seconds = round(max(0.0, time.perf_counter() - start_perf), 3)
        coach_analysis = self._build_turn_coach_analysis(session, round_no, evidence)
        self.repository.add_turn(
            session_id=session["session_id"],
            role="customer",
            content=customer_reply,
            round_no=round_no,
            response_mode=response_mode,
            stage_no=1,
            started_at=started_at,
            submitted_at=now,
            response_seconds=response_seconds,
            retrieved_chunk_ids=[item["chunk_id"] for item in evidence],
            retrieved_evidence=evidence,
            stage_decision={"stage_status": stage_status, "session_status": session_status},
            coach_analysis=coach_analysis,
        )
        logger.info(
            "[销售训练] 训练轮次完成 会话编号=%s 轮次=%s 状态=%s 回复模式=%s 回复长度=%s 证据数量=%s 耗时秒=%s",
            session["session_id"],
            round_no,
            session_status,
            response_mode,
            len(customer_reply or ""),
            len(evidence),
            response_seconds,
        )
        return TrainingTurnResponse(
            customer_reply=customer_reply,
            current_stage_no=1,
            stage_status=stage_status,
            session_status=session_status,
            retrieved_chunk_ids=[item["chunk_id"] for item in evidence],
            coach_analysis=coach_analysis,
            response_seconds=response_seconds,
        )

    def _turn_evidence(self, session: dict[str, Any], message: str) -> list[dict[str, Any]]:
        """为某一轮学员回复检索训练证据。"""

        profile = self._require_role_profile(session["profile_id"])
        # 查询文本不只用学员本轮话术，还拼上场景描述，召回更贴近当前训练情境。
        query = f"{message}\n{profile.get('scenario_description') or ''}"
        logger.info(
            "[销售训练][本轮证据] 构造检索文本 会话编号=%s 角色编号=%s 学员输入预览=%s 检索文本长度=%s",
            session["session_id"],
            session["profile_id"],
            self._short_text(message),
            len(query),
        )
        return self._search_training_evidence(query, visibility=("visible", "hidden"), k=5)

    def _search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict[str, Any]]:
        """检索训练证据库，并过滤学员不可直接看到的内容。"""

        return self.query_service.search_training_evidence(query, visibility=visibility, k=k)

    def _generate_customer_reply(
            self,
            session: dict[str, Any],
            trainee_message: str,
            evidence: list[dict[str, Any]],
            *,
            model_mode: str | None,
    ) -> str:
        """一次性生成 AI 客户回复。"""

        model = get_chat_model(model_mode)
        prompt = self._customer_prompt(session, trainee_message, evidence)
        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI客户回复] 一次性调用开始 会话编号=%s 模型档位=%s 提示词长度=%s 证据数量=%s 学员输入预览=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
            len(evidence),
            self._short_text(trainee_message),
        )
        try:
            response = model.invoke(self._messages(prompt_manager.get("training.ai_customer_system"), prompt))
            text = self._content_text(response.content).strip()
            if text:
                logger.info(
                    "[销售训练][AI客户回复] 一次性调用完成 会话编号=%s 回复长度=%s 耗时秒=%s 回复预览=%s",
                    session["session_id"],
                    len(text),
                    round(max(0.0, time.perf_counter() - start_perf), 3),
                    self._short_text(text),
                )
                return text
            logger.warning("[销售训练][AI客户回复] 模型返回为空，使用兜底回复 会话编号=%s", session["session_id"])
            return self._fallback_customer_reply(evidence)
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户回复生成失败，使用兜底回复 会话编号=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return self._fallback_customer_reply(evidence)

    def _stream_customer_reply(
            self,
            session: dict[str, Any],
            trainee_message: str,
            evidence: list[dict[str, Any]],
            *,
            model_mode: str | None,
    ) -> Iterator[str]:
        """流式生成 AI 客户回复。"""

        model = get_chat_model(model_mode)
        prompt = self._customer_prompt(session, trainee_message, evidence)
        start_perf = time.perf_counter()
        chunk_count = 0
        char_count = 0
        logger.info(
            "[销售训练][AI客户回复] 流式调用开始 会话编号=%s 模型档位=%s 提示词长度=%s 证据数量=%s 学员输入预览=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
            len(evidence),
            self._short_text(trainee_message),
        )
        try:
            for chunk in model.stream(self._messages(prompt_manager.get("training.ai_customer_system"), prompt)):
                text = self._content_text(chunk.content)
                if text:
                    chunk_count += 1
                    char_count += len(text)
                    yield text
            logger.info(
                "[销售训练][AI客户回复] 流式调用完成 会话编号=%s 分片数量=%s 回复累计长度=%s 耗时秒=%s",
                session["session_id"],
                chunk_count,
                char_count,
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户流式生成失败，使用兜底回复 会话编号=%s 已返回分片=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                chunk_count,
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            yield self._fallback_customer_reply(evidence)

    def _generate_opening_message(self, session: dict[str, Any], *, model_mode: str | None) -> str:
        """生成 AI 客户开场白，让训练会话一开始就像真实客户在场。"""

        prompt = self._opening_prompt(session)
        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI客户开场白] 调用开始 会话编号=%s 模型档位=%s 提示词长度=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
        )
        try:
            response = get_chat_model(model_mode).invoke(
                self._messages(prompt_manager.get("training.ai_customer_system"), prompt)
            )
            text = self._content_text(response.content).strip()
            if text:
                logger.info(
                    "[销售训练][AI客户开场白] 调用完成 会话编号=%s 回复长度=%s 耗时秒=%s 回复预览=%s",
                    session["session_id"],
                    len(text),
                    round(max(0.0, time.perf_counter() - start_perf), 3),
                    self._short_text(text),
                )
                return text
            logger.warning("[销售训练][AI客户开场白] 模型返回为空，使用兜底开场白 会话编号=%s", session["session_id"])
            return self._fallback_opening_message(session)
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户开场白生成失败，使用兜底开场白 会话编号=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return self._fallback_opening_message(session)

    @staticmethod
    def _messages(system: str, human: str) -> list:
        """构造 LangChain 聊天消息。

        Java 里常见做法是 new SystemMessage(...) + new HumanMessage(...)；
        Python 这里直接返回 list，交给模型 invoke/stream。
        """

        return [SystemMessage(content=system), HumanMessage(content=human)]

    @staticmethod
    def _content_text(content: Any) -> str:
        """把不同模型返回格式统一转成字符串。

        有些模型返回 str，有些模型返回 [{"text": "..."}] 这种结构。
        这里做兼容，避免上层业务关心模型供应商差异。
        """

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
        return str(content)

    @staticmethod
    def _short_text(value: Any, limit: int = 120) -> str:
        """把长文本压缩成日志预览，避免 PyCharm 控制台被完整提示词刷屏。"""

        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text or "-"
        return f"{text[:limit]}..."

    @staticmethod
    def _join_values(values: Any, limit: int = 6) -> str:
        """把列表、元组或生成器压成一行日志文本，方便查看命中的来源。"""

        if values is None:
            return "-"
        if isinstance(values, (str, int, float)):
            return str(values)
        result: list[str] = []
        for value in values:
            if value is None or value == "":
                continue
            text = str(value)
            if text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return "、".join(result) if result else "-"

    @staticmethod
    def _dict_key_text(value: Any) -> str:
        """只打印字典字段名，不打印完整内容，避免日志泄露隐藏画像细节。"""

        if not isinstance(value, dict) or not value:
            return "-"
        return "、".join(str(key) for key in value.keys())

    def _invoke_json(self, prompt: str, *, model_mode: str | None, fallback: dict, task_name: str = "JSON生成") -> dict:
        """调用 LLM 并解析 JSON，失败时使用可解释兜底。

        LLM 不一定总是严格输出 JSON，所以这里统一 try/except：
        - 成功：解析模型 JSON；
        - 失败：记录中文日志，并返回 fallback，保证页面流程不中断。
        """

        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI调用开始] 任务=%s 模型档位=%s 提示词长度=%s 兜底字段=%s",
            task_name,
            model_mode or "默认",
            len(prompt),
            self._dict_key_text(fallback),
        )
        try:
            response = get_chat_model(model_mode).invoke(
                self._messages(prompt_manager.get("training.json_only_system"), prompt)
            )
            text = self._content_text(response.content)
            parsed = self._parse_json_object(text)
            logger.info(
                "[销售训练][AI调用完成] 任务=%s 模型档位=%s 返回长度=%s JSON字段=%s 耗时秒=%s",
                task_name,
                model_mode or "默认",
                len(text),
                self._dict_key_text(parsed),
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
            return parsed
        except Exception as exc:
            logger.warning(
                "[销售训练] LLM JSON生成失败，使用兜底结构 任务=%s 模型档位=%s 提示词长度=%s 耗时秒=%s 错误=%s",
                task_name,
                model_mode or "默认",
                len(prompt),
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return fallback

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """从模型输出文本中提取 JSON 对象。

        re.DOTALL 让正则里的 . 可以匹配换行。
        这样模型即使输出多行 JSON，也能被提取。
        """

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("模型没有输出 JSON 对象")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return parsed

    @staticmethod
    def _safe_filename(filename: str | None) -> str:
        """清理上传文件名，防止路径穿越。

        os.path.basename 会去掉目录部分：
        - ../../a.docx -> a.docx
        - C:\\tmp\\a.docx -> a.docx
        """

        clean = os.path.basename(filename or "").strip()
        return clean or f"training_{uuid.uuid4().hex}.txt"

    @staticmethod
    def _load_json(value: Any, default: Any) -> Any:
        """安全读取 JSON 字段。

        关系型数据库里 JSON 读出来可能是 str，也可能已是 dict/list。
        但部分内部调用可能已经传入 dict/list，这里直接返回，减少重复 json.loads。
        """

        if not value:
            return default
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """清理写入 Qdrant 的 metadata，避免空标签污染向量库 payload。"""

        compacted: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            compacted[key] = value
        return compacted

    @staticmethod
    def _json_changed(old_value: Any, new_value: Any) -> bool:
        """比较两个 JSON 结构是否真正变化。

        前端编辑页会提交完整快照，不能因为字段存在就认为内容变化。
        这里先按中文稳定序列化再比较，避免字典顺序导致误判。
        """

        return json.dumps(old_value, ensure_ascii=False, sort_keys=True) != json.dumps(
            new_value,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _normalize_response_mode(response_mode: str | None) -> str:
        """统一响应模式枚举。"""

        return "blocking" if response_mode in {"blocking", "once"} else "stream"

    @staticmethod
    def _normalize_round_limit(value: Any) -> int:
        """把模型返回的轮数限制在合理范围。

        LLM 可能返回字符串、空值甚至异常内容，所以先 int() 尝试转换，
        再用 max/min 做 5-100 的边界保护。
        """

        try:
            round_limit = int(value)
        except (TypeError, ValueError):
            round_limit = 8
        return max(5, min(100, round_limit))

    def _require_role_profile(self, profile_id: str) -> dict[str, Any]:
        """查询 AI 角色画像，不存在时直接抛出 404。"""

        profile = self.repository.get_role_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="AI 陪练角色不存在")
        return profile

    def _require_goal_setting(self, setting_id: str) -> dict[str, Any]:
        """查询训练目标设置，不存在时直接抛出 404。"""

        setting = self.repository.get_goal_setting(setting_id)
        if not setting:
            raise HTTPException(status_code=404, detail="训练设置不存在")
        return setting

    def _require_plan(self, plan_id: str) -> dict[str, Any]:
        """查询训练方案，不存在时直接抛出 404。"""

        plan = self.repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        return plan

    def _require_session(self, session_id: str) -> dict[str, Any]:
        """查询可继续对话的训练会话。

        completed/deleted 等状态不能继续提交学员回复。
        """

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")
        if session["status"] not in {"active", "scoring"}:
            raise HTTPException(status_code=400, detail=f"当前训练状态不允许继续对话：{session['status']}")
        return session

    @staticmethod
    def _plan_summary(row: dict[str, Any]) -> TrainingPlanSummaryResponse:
        """把训练方案数据库行转换成列表摘要。"""

        return TrainingPlanSummaryResponse(
            plan_id=row["plan_id"],
            plan_name=row["plan_name"],
            trainee_id=row["trainee_id"],
            trainee_name=row["trainee_name"],
            profile_type=row["profile_type"],
            model_mode=row.get("model_mode"),
            role_status=row["role_status"],
            goal_status=row["goal_status"],
            score_status=row["score_status"],
            active_profile_id=row.get("active_profile_id"),
            active_setting_id=row.get("active_setting_id"),
            created_at=_format_response_time(row["created_at"]),
            updated_at=_format_response_time(row["updated_at"]),
        )

    def _plan_detail_response(self, row: dict[str, Any]) -> TrainingPlanDetailResponse:
        """把训练方案数据库行转换成完整详情。"""

        role_row = self.repository.get_role_profile(row["active_profile_id"]) if row.get("active_profile_id") else None
        setting_row = self.repository.get_goal_setting(row["active_setting_id"]) if row.get("active_setting_id") else None
        visible_profile = self._load_json(role_row.get("visible_profile_json"), {}) if role_row else {}
        hidden_profile = self._load_json(role_row.get("hidden_profile_json"), {}) if role_row else {}
        role_profile = self._load_json(role_row.get("role_profile_json"), {}) if role_row else {}
        role_confirm_card = self._load_json(role_row.get("role_confirm_card_json"), {}) if role_row else {}
        retrieved_cases = self._load_json(role_row.get("retrieved_evidence_json"), []) if role_row else []
        return TrainingPlanDetailResponse(
            plan=self._plan_summary(row),
            trainee=self._load_json(row.get("trainee_json"), {}),
            selected_fields=self._load_json(row.get("selected_fields_json"), {}),
            scenario_description=row.get("scenario_description") or "",
            extra_details=row.get("extra_details") or "",
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            retrieved_cases=retrieved_cases,
            goal_setting=self._goal_response(setting_row) if setting_row else None,
        )

    def _write_staging_chunks(self, *, batch: dict[str, Any], chunks: list[Any], source_type: str) -> int:
        """把待审核训练切片写入临时向量库。"""

        batch_id = str(batch["batch_id"])
        documents = [
            self._document_from_training_chunk(chunk, batch=batch, source_type=source_type, status="pending_review", is_current=False)
            for chunk in chunks
        ]
        self._delete_staging_vectors(batch_id)
        if documents:
            self.staging_vector_service.vector_store.add_documents(documents)
        logger.info(
            "[销售训练] 待审核切片已写入临时向量库 批次编号=%s 临时向量库=%s 切片数量=%s",
            batch_id,
            self.staging_collection_name,
            len(documents),
        )
        return len(documents)

    def _publish_staging_vectors(self, *, batch: dict[str, Any]) -> int:
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

    def _delete_staging_vectors(self, batch_id: str) -> None:
        """删除临时向量库中某个批次的待审核切片。"""

        self.staging_vector_service.delete_by_metadata("batch_id", batch_id)

    def _list_batch_chunk_rows(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """按批次状态从临时库或正式库读取切片行。"""

        batch_id = str(batch["batch_id"])
        if batch.get("status") in {"pending_review", "embedding", "publish_failed", "parsing_failed"}:
            return self._list_staging_chunk_rows(batch_id)
        return self._list_published_chunk_rows(batch_id)

    def _list_staging_chunk_rows(self, batch_id: str) -> list[dict[str, Any]]:
        """从临时向量库读取待审核切片行。"""

        return self._documents_to_chunk_rows(
            self.staging_vector_service.list_documents_by_metadata("batch_id", batch_id),
            batch_id=batch_id,
        )

    def _list_published_chunk_rows(self, batch_id: str) -> list[dict[str, Any]]:
        """从正式向量库读取已发布切片行。"""

        return self._documents_to_chunk_rows(
            self.vector_service.list_documents_by_metadata("batch_id", batch_id),
            batch_id=batch_id,
        )

    def _documents_to_chunk_rows(self, documents: list[Document], *, batch_id: str) -> list[dict[str, Any]]:
        """把 Qdrant Document 列表转换为前端切片行。"""

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
        return sorted(rows, key=self._chunk_sort_key)

    @staticmethod
    def _chunk_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
        """按案例序号、切片编号稳定排序。"""

        metadata = V2SalesTrainingCoreService._load_json(row.get("metadata_json"), {})
        try:
            case_index = int(metadata.get("case_index") or 0)
        except (TypeError, ValueError):
            case_index = 0
        return case_index, str(row.get("chunk_id") or ""), str(row.get("case_part") or "")

    def _document_from_training_chunk(
            self,
            chunk: Any,
            *,
            batch: dict[str, Any],
            source_type: str,
            status: str,
            is_current: bool,
    ) -> Document:
        """把内存训练切片转换成 Qdrant Document。"""

        metadata = self._compact_metadata({
            "batch_id": batch["batch_id"],
            "document_id": batch.get("document_id"),
            "chunk_id": chunk.chunk_id,
            "content_type": "sales_training_case",
            "source_type": source_type,
            "source_file": self._batch_file_info(batch).get("source_file"),
            "file_md5": self._batch_file_info(batch).get("file_md5"),
            "version_group_id": batch.get("version_group_id") or batch["batch_id"],
            "version_no": int(batch.get("version_no") or 1),
            "status": status,
            "is_current": is_current,
            "case_part": chunk.case_part,
            "visibility": chunk.visibility,
            **dict(chunk.metadata or {}),
        })
        return Document(page_content=chunk.text, metadata=metadata)

    def _parse_training_chunks(
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

    def _next_training_batch_version(self, *, source_type: str, source_file: str) -> dict[str, Any]:
        """计算本次上传资料的版本信息。"""

        latest_batch = self.repository.get_latest_batch_for_version(source_type=source_type, source_file=source_file)
        if not latest_batch:
            return {"version_group_id": None, "version_no": 1, "previous_batch_id": None}
        return {
            "version_group_id": latest_batch.get("version_group_id") or latest_batch["batch_id"],
            "version_no": int(latest_batch.get("version_no") or 1) + 1,
            "previous_batch_id": latest_batch["batch_id"],
        }

    def _archive_previous_training_versions(self, batch: dict[str, Any]) -> None:
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

    def _mark_version_group_vectors_archived(self, version_group_id: str) -> None:
        """把同版本组内所有正式向量点标记为历史版本。"""

        version_batches = self.repository.list_published_batches_in_version_group(version_group_id)
        for version_batch in version_batches:
            self.vector_service.update_metadata_by_metadata(
                "batch_id",
                version_batch["batch_id"],
                metadata_updates={"status": "archived", "is_current": False},
            )

    def _mark_batch_vectors_current(self, *, batch: dict[str, Any]) -> int:
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

    def _saved_chunk_preview(self, batch: dict[str, Any], *, safe_max_chars: int) -> dict[str, Any]:
        """读取已经写入向量库的切片预览，保证预览内容和后续发布内容一致。"""

        stored_chunks = self._list_batch_chunk_rows(batch)
        parts: list[str] = []
        total_chars = 0
        truncated = False
        for chunk in stored_chunks:
            part = f"{chunk.get('case_part') or 'unknown'}\n{str(chunk.get('chunk_text') or '').strip()}"
            if not part.strip():
                continue
            separator = "\n\n" if parts else ""
            candidate = f"{separator}{part}"
            remaining = safe_max_chars - total_chars
            if len(candidate) > remaining:
                parts.append(candidate[:remaining])
                truncated = True
                break
            parts.append(candidate)
            total_chars += len(candidate)
        return {"content": "".join(parts), "truncated": truncated}

    def _improve_training_chunks_if_needed(
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
        source_text = self._read_training_source_text(file_path)
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

    def _force_llm_reparse_chunks(
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
        source_text = self._read_training_source_text(file_path)
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

    @staticmethod
    def _read_training_source_text(file_path: str) -> str:
        """读取训练资料原文，供低质量 LLM 兜底切分使用。"""

        documents = FileProcessorFactory.load_documents(file_path)
        return "\n\n".join(document.page_content for document in documents).strip()

    @staticmethod
    def _looks_mojibake(text: str) -> bool:
        """粗略判断文本是否存在明显乱码。"""

        if not text:
            return False
        bad_chars = text.count("\ufffd") + text.count("�")
        suspicious_chars = sum(1 for char in text if "\ue000" <= char <= "\uf8ff")
        return (bad_chars + suspicious_chars) / max(len(text), 1) > 0.01

    @classmethod
    def _decode_text_bytes(cls, raw_data: bytes) -> tuple[str, str]:
        """按常见中文编码解码 TXT，避免浏览器直连 MinIO 时乱码。"""

        for charset in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                text = raw_data.decode(charset)
            except UnicodeDecodeError:
                continue
            if not cls._looks_mojibake(text):
                return text, charset
        return raw_data.decode("utf-8", errors="replace"), "utf-8-replace"

    def _build_batch_preview(self, batch: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
        """根据训练资料文件类型生成站内弹窗预览数据。"""

        file_info = self._batch_file_info(batch)
        source_file = str(file_info.get("source_file") or batch.get("source_file") or "")
        file_type = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else ""
        if file_type not in ALLOWED_TRAINING_FILE_TYPES:
            return {"preview_type": "file_url", "content": "", "truncated": False, "charset": None}

        with self._download_batch_file(batch) as file_path:
            if file_type == "txt":
                content, charset = self._decode_text_bytes(Path(file_path).read_bytes())
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

    def _batch_file_info(self, row: dict[str, Any]) -> dict[str, Any]:
        """读取训练资料关联的文件基础信息。

        新数据以 documents 表为准；历史批次可能没有 document_id，
        新数据以 documents 表为准；旧批次如果缺少 MinIO 对象路径，需要先执行历史文件迁移。
        """

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

    def _batch_response(self, row: dict[str, Any]) -> TrainingKnowledgeBatchResponse:
        """把训练资料批次数据库行转换成前端响应。"""

        file_info = self._batch_file_info(row)
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
            quality_report=V2SalesTrainingCoreService._load_json(row.get("quality_report_json"), {}),
            created_by=row.get("created_by"),
            created_at=_format_response_time(row["created_at"]),
            updated_at=_format_response_time(row["updated_at"]),
        )

    def _get_active_batch(self, batch_id: str) -> dict[str, Any]:
        """读取未删除的训练资料批次。"""

        batch = self.repository.get_batch(batch_id)
        if batch is None or batch.get("status") == "deleted":
            raise HTTPException(status_code=404, detail=f"训练资料不存在：{batch_id}")
        return batch

    @contextmanager
    def _download_batch_file(self, batch: dict[str, Any]) -> Iterator[str]:
        """从 MinIO 下载训练资料原文件到临时路径。"""

        file_info = self._batch_file_info(batch)
        object_name = str(file_info.get("object_name") or "").strip()
        if not object_name:
            raise HTTPException(status_code=400, detail="训练资料缺少 MinIO 对象路径，请先完成历史文件迁移")
        with get_file_storage_service().downloaded_temp_file(
                bucket_name=file_info.get("bucket_name"),
                object_name=object_name,
                filename=str(file_info.get("source_file") or batch.get("source_file") or "training_file"),
        ) as file_path:
            yield file_path

    def _goal_response(self, row: dict[str, Any]) -> GoalSettingResponse:
        """把数据库训练设置行转换成 Pydantic 响应。"""

        # GoalStage(**item) 是关键字参数解包：
        # item={"stage_no":1,"stage_name":"开放式",...}
        # 等价于 GoalStage(stage_no=1, stage_name="开放式", ...)
        stages = [GoalStage(**item) for item in self._load_json(row.get("stages_json"), [])]
        return GoalSettingResponse(
            setting_id=row["setting_id"],
            profile_id=row["profile_id"],
            training_mode=row["training_mode"],
            training_purpose=row["training_purpose"],
            round_limit=int(row["round_limit"]),
            stages=stages,
            scoring_rules=self._load_json(row.get("scoring_rules_json"), self._default_scoring_rules()),
            status=row["status"],
        )

    @staticmethod
    def _session_response(row: dict[str, Any], opening_message: str | None = None) -> TrainingSessionResponse:
        """把数据库训练会话行转换成接口响应对象。"""

        return TrainingSessionResponse(
            session_id=row["session_id"],
            profile_id=row["profile_id"],
            setting_id=row["setting_id"],
            trainee_id=row["trainee_id"],
            training_mode=row["training_mode"],
            response_mode=row["response_mode"],
            current_stage_no=int(row["current_stage_no"]),
            status=row["status"],
            round_limit=int(row["round_limit"]),
            opening_message=opening_message,
        )

    @staticmethod
    def _session_summary(row: dict[str, Any]) -> TrainingSessionSummaryResponse:
        """把数据库会话行转换成前端历史摘要。"""

        return TrainingSessionSummaryResponse(
            session_id=row["session_id"],
            trainee_id=row["trainee_id"],
            training_mode=row["training_mode"],
            response_mode=row["response_mode"],
            status=row["status"],
            round_limit=int(row["round_limit"]),
            answered_count=int(row.get("answered_count") or 0),
            total_score=row.get("total_score"),
            level=row.get("level"),
            started_at=_format_response_time(row["started_at"]),
            ended_at=_format_response_time(row.get("ended_at")),
            updated_at=_format_response_time(row["updated_at"]),
        )

    def _turn_record(self, row: dict[str, Any]) -> TrainingTurnRecordResponse:
        """把数据库轮次行转换成复盘消息。"""

        return TrainingTurnRecordResponse(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            round_no=int(row["round_no"]),
            stage_no=int(row["stage_no"]),
            response_mode=row.get("response_mode"),
            response_seconds=row.get("response_seconds"),
            retrieved_chunk_ids=self._load_json(row.get("retrieved_chunk_ids_json"), []),
            stage_decision=self._load_json(row.get("stage_decision_json"), {}),
            coach_analysis=self._load_json(row.get("coach_analysis_json"), {}),
            created_at=_format_response_time(row["created_at"]),
        )

    def _score_response(self, row: dict[str, Any]) -> TrainingScoreResponse:
        """把数据库评分行转换成评分响应。"""

        return TrainingScoreResponse(
            score_id=row["score_id"],
            session_id=row["session_id"],
            total_score=int(row["final_score"]),
            level=row["level"],
            is_passed=bool(row["is_passed"]),
            general_score=int(row["general_score"]),
            stage_score=int(row["stage_score"]),
            penalty_score=int(row["penalty_score"]),
            report=self._load_json(row.get("detail_json"), {}),
        )

    def _normalize_scoring_rules(
            self,
            raw_rules: Any,
            stages: list[dict[str, Any]],
            profile: dict[str, Any],
    ) -> dict[str, Any]:
        """归一化评分规则，保证总分始终是 100。

        规则结构：
        - 通用能力固定 40 分，不能被 LLM 改坏；
        - 阶段能力固定 60 分，但考核点由 LLM 根据角色和目标生成；
        - 如果 LLM 输出不完整，就用后端兜底规则。
        """

        return TrainingScoreService.normalize_scoring_rules(raw_rules, stages, profile)

    @classmethod
    def _default_scoring_rules(
            cls,
            *,
            stages: list[dict[str, Any]] | None = None,
            profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """默认评分规则。

        通用能力严格固定 40 分；阶段能力 60 分在 LLM 失败时按开放式训练兜底拆分。
        """

        return TrainingScoreService.default_scoring_rules(stages=stages, profile=profile)

    @staticmethod
    def _normalize_dimension_scores(dimensions: list[Any], *, total_score: int) -> list[dict[str, Any]]:
        """按总分归一化评分维度。

        LLM 可能给出 55 或 63 分，这里统一按比例缩放到目标总分。
        """

        return TrainingScoreService.normalize_dimension_scores(dimensions, total_score=total_score)

    def _build_turn_coach_analysis(
            self,
            session: dict[str, Any],
            round_no: int,
            evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """生成每轮即时教练分析。

        一期先用规则生成，优点是快且稳定；后续可替换为 LLM 专项分析接口。
        """

        turns = self.repository.list_turns(session["session_id"])
        trainee_turn = next((item for item in reversed(turns) if item["role"] == "trainee" and int(item["round_no"]) == round_no), None)
        trainee_text = str(trainee_turn.get("content") or "") if trainee_turn else ""
        has_question = "?" in trainee_text or "？" in trainee_text
        has_case = any(keyword in trainee_text for keyword in ("案例", "客户", "数据", "证据", "效果", "ROI", "试点"))
        has_next_step = any(keyword in trainee_text for keyword in ("下一步", "约", "试", "确认", "发您", "安排", "继续"))
        strengths: list[str] = []
        suggestions: list[str] = []
        if has_question:
            strengths.append("本轮有提问动作，能推动客户继续释放信息。")
        else:
            suggestions.append("建议先追问客户当前最核心的顾虑，避免直接进入方案介绍。")
        if has_case:
            strengths.append("本轮尝试使用证据或案例降低客户不确定感。")
        else:
            suggestions.append("可以补一句同类客户案例、数据或试点路径，让表达更可信。")
        if has_next_step:
            strengths.append("本轮有下一步推进意识。")
        else:
            suggestions.append("结尾建议给出轻量下一步，例如约 15 分钟确认需求或先做小范围验证。")
        if not strengths:
            strengths.append("表达已完成基础回应，但还需要增强销售推进动作。")
        return {
            "round_no": round_no,
            "summary": "本轮建议优先补强需求挖掘、证据化表达和下一步推进。",
            "strengths": strengths,
            "suggestions": suggestions,
            "retrieval_hint": f"本轮命中 {len(evidence)} 条训练知识，可结合命中切片补充案例化表达。",
            "next_reply_hint": "先承接客户顾虑，再追问影响范围，最后用案例或试点降低风险。",
        }

    @staticmethod
    def _sse(event: str, payload: dict) -> str:
        """把事件名和数据包装成 SSE 协议文本。

        SSE 格式要求：
            event: 事件名
            data: JSON字符串

        末尾两个换行表示一个事件结束。
        """

        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _build_role_query(self, request: RoleGenerateRequest) -> str:
        """构造角色生成前的向量检索查询文本。"""

        return self.role_service.build_role_query(request)

    def _role_prompt(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        """构造 AI 客户角色生成提示词。"""

        return self.role_service.role_prompt(request, evidence)

    def _scenario_polish_prompt(self, request: ScenarioPolishRequest) -> str:
        """构造场景描述润色提示词。"""

        return self.role_service.scenario_polish_prompt(request)

    def _supplement_questions_prompt(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        """构造补充问答生成提示词。"""

        return self.role_service.supplement_questions_prompt(request, evidence)

    def _normalize_supplement_questions(self, raw_questions: Any, request: RoleGenerateRequest) -> list[SupplementQuestion]:
        """把 LLM 输出规整成前端稳定可渲染的 1-5 道题。"""

        return self.role_service.normalize_supplement_questions(raw_questions, request)

    def _goal_prompt(self, profile: dict[str, Any]) -> str:
        """构造开放式训练目标和评分规则生成提示词。"""

        return self.goal_setting_service.goal_prompt(profile)

    def _opening_prompt(self, session: dict[str, Any]) -> str:
        """构造 AI 客户开场白提示词。"""

        profile = self._require_role_profile(session["profile_id"])
        setting = self._require_goal_setting(session["setting_id"])
        return self.session_prompt_service.opening_prompt(profile, setting)

    def _customer_prompt(self, session: dict[str, Any], trainee_message: str, evidence: list[dict[str, Any]]) -> str:
        """构造每轮 AI 客户回复提示词。"""

        profile = self._require_role_profile(session["profile_id"])
        setting = self._require_goal_setting(session["setting_id"])
        turns = self.repository.list_turns(session["session_id"])[-10:]
        return self.session_prompt_service.customer_prompt(
            profile,
            setting,
            turns=turns,
            trainee_message=trainee_message,
            evidence=evidence,
        )

    def _score_prompt(
            self,
            profile: dict[str, Any],
            setting: dict[str, Any],
            turns: list[dict[str, Any]],
            evidence: list[dict[str, Any]],
    ) -> str:
        """构造最终评分报告提示词。"""

        return self.session_prompt_service.score_prompt(profile, setting, turns=turns, evidence=evidence)

    @staticmethod
    def _conversation_text(turns: list[dict[str, Any]]) -> str:
        """把训练对话轮次拼成评分和证据检索使用的纯文本。"""

        return TrainingSessionPromptService.conversation_text(turns)

    @staticmethod
    def _fallback_polished_scenario(request: ScenarioPolishRequest) -> str:
        """模型润色失败时的本地兜底文案。

        兜底逻辑只做安全拼接，不虚构业务事实；这样即使 LLM 报错，
        前端也能得到一段可用的场景描述。
        """

        return TrainingRoleService.fallback_polished_scenario(request)

    def _fallback_supplement_questions(self, request: RoleGenerateRequest) -> list[dict[str, Any]]:
        """补充问题兜底模板，保证模型不可用时流程仍可继续。"""

        return self.role_service.fallback_supplement_questions(request)

    def _fallback_role(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> dict:
        """角色生成失败时的本地兜底结果。

        兜底只使用用户选择的画像字段和已召回证据，不凭空扩展业务事实。
        """

        return self.role_service.fallback_role(request, evidence)

    @staticmethod
    def _fallback_goal(profile: dict[str, Any]) -> dict:
        """训练目标生成失败时的本地兜底结果。"""

        return TrainingGoalSettingService.fallback_goal(profile)

    @staticmethod
    def _fallback_customer_reply(evidence: list[dict[str, Any]]) -> str:
        """AI 客户回复失败时的兜底话术。"""

        return TrainingSessionPromptService.fallback_customer_reply(evidence)

    def _fallback_opening_message(self, session: dict[str, Any]) -> str:
        """AI 客户开场白失败时的兜底话术。"""

        profile = self._require_role_profile(session["profile_id"])
        return self.session_prompt_service.fallback_opening_message(profile)

    @staticmethod
    def _fallback_score(turns: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict:
        """评分模型失败时的兜底评分。

        兜底分只用于保证流程闭环，真实评分仍应优先使用 LLM 按评分规则判断。
        """

        return TrainingScoreService.fallback_score(turns, evidence)

    @staticmethod
    def _score_level(score: int) -> str:
        """把最终得分转换成中文等级。"""

        return TrainingScoreService.score_level(score)

