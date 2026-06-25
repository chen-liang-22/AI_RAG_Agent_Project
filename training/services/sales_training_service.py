import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

import yaml
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from api.services.upload_cleanup_services import delete_upload_path
from infrastructure.vector_store_service import VectorStoreService
from model.factory import get_chat_model
from rag.file_processors import FileProcessorFactory
from rag.knowledge_store import KnowledgeStore
from training.factories.knowledge_ingest_strategy_factory import KnowledgeIngestStrategyFactory
from training.llm_ingest import TrainingLlmFallbackSplitter
from training.publish_validation import TrainingPublishValidator
from training.quality import TrainingIngestQualityEvaluator
from training.repository import TrainingRepository, utc_now_text
from training.schemas import (
    GoalSettingResponse,
    GoalStage,
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestion,
    SupplementQuestionGenerateResponse,
    SupplementQuestionOption,
    TrainingPlanCreateRequest,
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
from utils.database_connection import DatabaseErrorTypes
from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


TRAINING_INGEST_CONFIG_PATH = get_abs_path("config/training_ingest.yml")
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
    try:
        with open(TRAINING_INGEST_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except OSError as exc:
        logger.warning("[销售训练] 读取训练入库配置失败，使用默认 collection 配置 错误=%s", exc)
        return config
    except yaml.YAMLError as exc:
        logger.warning("[销售训练] 解析训练入库配置失败，使用默认 collection 配置 错误=%s", exc)
        return config

    collection_config = data.get("collections") if isinstance(data, dict) else {}
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


class SalesTrainingService:
    """销售训练一期外观服务。

    外观模式用于把文件解析、向量库、LLM、业务数据库这些子系统收拢成
    前端能理解的训练流程接口。一期流程较短，暂不引入 Graph。
    """

    def __init__(
            self,
            repository: TrainingRepository | None = None,
            knowledge_store: KnowledgeStore | None = None,
    ):
        # repository 支持注入，主要是为了单元测试或局部替换仓储实现。
        self.repository = repository or TrainingRepository()
        # 文件台账复用知识库 documents 表，统一写入 MySQL。
        self.knowledge_store = knowledge_store or KnowledgeStore()
        collection_config = _load_training_collection_config()
        self.training_collection_name = collection_config["published"]
        self.staging_collection_name = collection_config["staging"]
        # 正式训练知识使用独立 collection，避免和智能客服的普通知识库混在一起。
        self.vector_service = VectorStoreService(collection_name=self.training_collection_name)
        # 待人工审核的上传切片写入临时 collection，发布成功后再清理，避免关系型数据库保存正文切片。
        self.staging_vector_service = VectorStoreService(collection_name=self.staging_collection_name)

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
        1. 保存上传文件到 uploads/batch_xxx；
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
        document_id = f"doc_{uuid.uuid4().hex}"
        batch_id = f"batch_{uuid.uuid4().hex}"
        # 拼出本文件上传目录的绝对路径，例如 uploads/doc_xxx，用来按文件维度保存原始资料。
        upload_dir = get_abs_path(os.path.join("uploads", document_id))
        # 确保上传目录存在；exist_ok=True 表示目录已存在时不报错，便于重复执行或异常重试。
        os.makedirs(upload_dir, exist_ok=True)
        # 拼出原始文件最终保存路径，把清洗后的文件名放到本批次上传目录下。
        file_path = os.path.join(upload_dir, filename)
        # 以二进制写入模式打开本地目标文件：w 表示写入，b 表示二进制。
        # 如果文件不存在会自动创建；如果文件已存在会先清空，适配 txt、pdf、docx 等不同文件格式。
        with open(file_path, "wb") as target:
            # 把 FastAPI 上传文件流复制到本地文件中，完成原始文件落盘保存。
            shutil.copyfileobj(file.file, target)

        # 阶段 3：计算文件 MD5 做内容级去重。
        # 只要文件内容完全相同，就直接复用已经 published 的历史批次。
        # 根据本地保存后的原始文件计算 MD5，MD5 可以理解为文件内容的唯一指纹。
        file_md5 = get_file_md5_hex(file_path)
        # 用文件 MD5 查询是否已经存在发布成功的同内容批次，避免重复解析和重复写入向量库。
        existing_batch = self.repository.get_published_batch_by_md5(file_md5)
        # Python 中有值的对象会被当作 True；这里表示查到了重复文件批次，就进入复用逻辑。
        if existing_batch:
            # 文件内容完全一样时不重复写入向量库，直接返回已有批次。
            # 临时上传文件已经落到新目录里，这里删除它，避免 uploads 下堆重复文件。
            shutil.rmtree(upload_dir, ignore_errors=True)
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
        self.knowledge_store.create_document(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_md5=file_md5,
            file_size=os.path.getsize(file_path),
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
            chunks = self._parse_training_chunks(
                file_path=file_path,
                batch_id=batch_id,
                source_file=filename,
                source_type=source_type,
            )
            if not chunks:
                raise ValueError("文件没有切出有效训练知识")

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
            point_count = self._write_staging_chunks(batch=batch, chunks=chunks, source_type=source_type)
            # 阶段 8：上传预览完成，等待人工确认发布。
            self.repository.update_batch_status(
                batch_id,
                status="pending_review",
                chunk_count=len(chunks),
                point_count=point_count,
                quality_report=quality_report,
            )
            self.knowledge_store.update_document_status(
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
            self.knowledge_store.update_document_status(
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
        """预览训练资料原文件。

        预览读的是上传保存下来的原始文件，不读 Qdrant 分片。
        这样用户能确认“我到底上传了哪份资料”。
        """

        batch = self._get_active_batch(batch_id)
        file_path = self._resolve_batch_file_path(batch)
        source_file = str(self._batch_file_info(batch).get("source_file") or batch.get("source_file") or "")
        file_type = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else ""
        safe_max_chars = max(500, min(100000, max_chars))

        stored_preview = self._saved_chunk_preview(batch, safe_max_chars=safe_max_chars)
        if stored_preview["content"]:
            content = stored_preview["content"]
            truncated = stored_preview["truncated"]
            preview_type = "saved_chunks"
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read(safe_max_chars + 1)
            truncated = len(content) > safe_max_chars
            preview_type = "text"
        elif file_type in ALLOWED_TRAINING_FILE_TYPES:
            # 兼容旧批次：如果数据库没有切片，再临时解析原文件做只读预览。
            strategy = KnowledgeIngestStrategyFactory.create(batch.get("source_type") or "lms_case")
            chunks = strategy.parse_chunks(file_path, {
                "batch_id": batch_id,
                "source_file": source_file,
                "source_type": batch.get("source_type"),
                "visibility_default": batch.get("visibility_default") or DEFAULT_TRAINING_VISIBILITY,
            })
            content = "\n\n".join(f"{chunk.case_part}\n{chunk.text.strip()}" for chunk in chunks)[:safe_max_chars]
            truncated = len(content) >= safe_max_chars
            preview_type = "document_text"
        else:
            raise HTTPException(status_code=400, detail=f"当前训练资料类型不支持预览：{file_type}")

        return TrainingKnowledgePreviewResponse(
            batch=self._batch_response(batch),
            preview_type=preview_type,
            content=content,
            truncated=truncated,
        )

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次。

        删除动作包含两层：
        1. Qdrant 正式库和临时库中按 batch_id 删除本批次向量点；
        2. 业务数据库批次状态改成 deleted，并同步物理删除 uploads 原文件。
        """

        batch = self._get_active_batch(batch_id)
        file_info = self._batch_file_info(batch)
        try:
            # 正式库和临时库都按 batch_id 删除，兼容待审核、已发布两种状态。
            self.vector_service.delete_by_metadata("batch_id", batch_id)
            self.staging_vector_service.delete_by_metadata("batch_id", batch_id)
            deleted = self.repository.mark_batch_deleted(batch_id)
            if not deleted:
                raise HTTPException(status_code=404, detail=f"训练资料不存在：{batch_id}")
            # 文件基础信息统一保存在 documents 表，删除批次时同步软删除文件台账记录。
            document_id = str(batch.get("document_id") or "").strip()
            if document_id:
                self.knowledge_store.mark_document_deleted(document_id)
            delete_upload_path(file_info.get("file_path"), document_id=document_id or None)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("[销售训练] 训练资料删除失败 批次编号=%s 错误=%s", batch_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"训练资料删除失败：{exc}") from exc

        logger.info(
            "[销售训练] 训练资料已删除 批次编号=%s 正式向量库=%s 临时向量库=%s",
            batch_id,
            self.training_collection_name,
            self.staging_collection_name,
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

        file_path = self._resolve_batch_file_path(batch)
        source_type = str(batch.get("source_type") or "lms_case")
        source_file = str(batch.get("source_file") or "")
        try:
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

        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][训练证据检索] 开始检索 collection=%s 可见性=%s 返回数量=%s 查询长度=%s 查询预览=%s",
            self.training_collection_name,
            self._join_values(visibility),
            k,
            len(query or ""),
            self._short_text(query),
        )
        current_batch_ids = self.repository.list_current_published_batch_ids()
        if not current_batch_ids:
            logger.info(
                "[销售训练][训练证据检索] 没有当前发布版本，跳过向量检索 collection=%s 耗时秒=%s",
                self.training_collection_name,
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
            return []
        search_filters = {"batch_id": current_batch_ids}
        documents = self.vector_service.search_documents(query, k=k, filters=search_filters)
        evidence: list[dict[str, Any]] = []
        for document in documents:
            metadata = dict(document.metadata)
            item_visibility = str(metadata.get("visibility") or "visible")
            if item_visibility not in visibility:
                continue
            # 只截取前 800 字交给 LLM，避免单条证据过长挤占上下文窗口。
            evidence.append(
                {
                    "chunk_id": str(metadata.get("chunk_id") or metadata.get("batch_id") or ""),
                    "case_part": str(metadata.get("case_part") or ""),
                    "visibility": item_visibility,
                    "score": metadata.get("_vector_score"),
                    "content": document.page_content[:800],
                    "source_file": metadata.get("source_file"),
                }
            )
        logger.info(
            "[销售训练][训练证据检索] 检索完成 collection=%s 原始文档数=%s 过滤后证据数=%s 来源文件=%s 案例部分=%s 耗时秒=%s",
            self.training_collection_name,
            len(documents),
            len(evidence),
            self._join_values(item.get("source_file") for item in evidence),
            self._join_values(item.get("case_part") for item in evidence),
            round(max(0.0, time.perf_counter() - start_perf), 3),
        )
        return evidence

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
            response = model.invoke(self._messages("你是销售训练中的 AI 客户。", prompt))
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
            for chunk in model.stream(self._messages("你是销售训练中的 AI 客户。", prompt)):
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
            response = get_chat_model(model_mode).invoke(self._messages("你是销售训练中的 AI 客户。", prompt))
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
            response = get_chat_model(model_mode).invoke(self._messages("请只输出 JSON。", prompt))
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
        profile = self.repository.get_role_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="AI 陪练角色不存在")
        return profile

    def _require_goal_setting(self, setting_id: str) -> dict[str, Any]:
        setting = self.repository.get_goal_setting(setting_id)
        if not setting:
            raise HTTPException(status_code=404, detail="训练设置不存在")
        return setting

    def _require_plan(self, plan_id: str) -> dict[str, Any]:
        plan = self.repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        return plan

    def _require_session(self, session_id: str) -> dict[str, Any]:
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

        metadata = SalesTrainingService._load_json(row.get("metadata_json"), {})
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

    def _batch_file_info(self, row: dict[str, Any]) -> dict[str, Any]:
        """读取训练资料关联的文件基础信息。

        新数据以 documents 表为准；历史批次可能没有 document_id，
        因此仍保留 source_file、file_path、file_md5 旧字段兜底。
        """

        document_id = str(row.get("document_id") or "").strip()
        document = None
        joined_source_file = row.get("document_filename")
        joined_file_path = row.get("document_file_path")
        joined_file_md5 = row.get("document_file_md5")
        if document_id and not any((joined_source_file, joined_file_path, joined_file_md5)):
            document = self.knowledge_store.get_document(document_id)
        return {
            "document_id": document_id or None,
            "source_file": joined_source_file or (document or {}).get("filename") or row.get("source_file"),
            "file_path": joined_file_path or (document or {}).get("file_path") or row.get("file_path"),
            "file_md5": joined_file_md5 or (document or {}).get("file_md5") or row.get("file_md5"),
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
            quality_report=SalesTrainingService._load_json(row.get("quality_report_json"), {}),
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

    def _resolve_batch_file_path(self, batch: dict[str, Any]) -> str:
        """获取训练资料原文件路径。

        新数据从 documents 读取文件路径；旧批次可能没有 document_id。
        为了兼容已经上传过的老资料，这里按 uploads/{batch_id}/{source_file} 再兜底找一次。
        """

        file_info = self._batch_file_info(batch)
        file_path = str(file_info.get("file_path") or "").strip()
        if not file_path:
            file_path = get_abs_path(os.path.join("uploads", batch["batch_id"], batch["source_file"]))
        return self._validate_training_file_path(file_path)

    @staticmethod
    def _validate_training_file_path(file_path: str) -> str:
        """校验训练资料原文件路径是否允许读取。"""

        target_path = os.path.abspath(file_path)
        uploads_root = os.path.abspath(get_abs_path("uploads"))
        if not os.path.commonpath([uploads_root, target_path]) == uploads_root:
            raise HTTPException(status_code=403, detail="训练资料文件路径不允许预览")
        if not os.path.isfile(target_path):
            raise HTTPException(status_code=404, detail="训练资料原文件不存在")
        return target_path

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

        rules = raw_rules if isinstance(raw_rules, dict) else {}
        default_rules = self._default_scoring_rules(stages=stages, profile=profile)
        stage_dimensions = rules.get("stage_dimensions")
        if not isinstance(stage_dimensions, list) or not stage_dimensions:
            stage_dimensions = default_rules["stage_dimensions"]
        else:
            # 阶段能力要像通用能力一样有多个评分维度。
            # 如果 LLM 只给了 1 个大维度，页面会显得太单薄，这里直接回退到三维度兜底规则。
            valid_dimensions = [item for item in stage_dimensions if isinstance(item, dict)]
            has_enough_dimensions = len(valid_dimensions) >= 3
            has_enough_points = all(
                isinstance(item.get("points"), list) and len(item.get("points") or []) >= 3
                for item in valid_dimensions
            )
            if not has_enough_dimensions or not has_enough_points:
                stage_dimensions = default_rules["stage_dimensions"]
        normalized_stage_dimensions = self._normalize_dimension_scores(stage_dimensions, total_score=60)
        return {
            "total_score": 100,
            "general_score": 40,
            "stage_score": 60,
            "general_dimensions": default_rules["general_dimensions"],
            "stage_dimensions": normalized_stage_dimensions,
            "review_mode": "ai_auto",
            "formula": "总分 = 通用能力得分 + 阶段能力得分 - 扣分；一期暂不启用违规词扣分",
        }

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

        role_profile = cls._load_json((profile or {}).get("role_profile_json"), {}) if profile else {}
        stage = (stages or [{}])[0] if stages else {}
        stage_name = str(stage.get("stage_name") or "开放式沟通目标")
        core_goal = str(stage.get("core_goal") or "围绕客户痛点推进有效沟通")
        customer_focus = role_profile.get("业务痛点") or role_profile.get("business_pain_points") or []
        focus_text = "、".join(str(item) for item in customer_focus[:2]) if isinstance(customer_focus, list) else str(customer_focus)
        return {
            "total_score": 100,
            "general_score": 40,
            "stage_score": 60,
            "general_dimensions": [
                {
                    "dimension_name": "内容质量",
                    "score": 20,
                    "points": [
                        {"point_name": "信息准确性", "score": 10, "description": "回答不编造事实，能基于已知客户信息和训练知识表达。"},
                        {"point_name": "需求理解与回应", "score": 5, "description": "能承接客户问题，不答非所问。"},
                        {"point_name": "价值传递", "score": 5, "description": "能把方案价值和客户痛点连接起来。"},
                    ],
                },
                {
                    "dimension_name": "语言表达",
                    "score": 10,
                    "points": [
                        {"point_name": "流利度", "score": 4, "description": "表达自然顺畅。"},
                        {"point_name": "专业术语使用", "score": 3, "description": "术语准确，不过度堆砌。"},
                        {"point_name": "逻辑清晰度", "score": 3, "description": "先回应问题，再给理由和下一步。"},
                    ],
                },
                {
                    "dimension_name": "互动与态度",
                    "score": 10,
                    "points": [
                        {"point_name": "倾听与承接", "score": 4, "description": "能接住客户情绪和顾虑。"},
                        {"point_name": "礼貌与亲和力", "score": 3, "description": "沟通态度专业、尊重客户。"},
                        {"point_name": "主动引导", "score": 3, "description": "能用问题推进下一步沟通。"},
                    ],
                },
            ],
            "stage_dimensions": [
                {
                    "dimension_name": "需求挖掘与痛点确认",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "背景追问", "score": 7, "description": "能围绕行业、规模、现有流程等背景连续追问，而不是直接讲方案。"},
                        {"point_name": "痛点定位", "score": 7, "description": f"能识别并复述客户真实痛点，重点关注：{focus_text or '投入产出、风险和落地成本'}。"},
                        {"point_name": "需求确认", "score": 6, "description": "能向客户确认优先级、影响范围和是否愿意继续沟通。"},
                    ],
                },
                {
                    "dimension_name": "价值呈现与证据支撑",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "价值匹配", "score": 7, "description": "能把方案价值与客户已经表达的痛点建立清晰连接。"},
                        {"point_name": "证据提供", "score": 7, "description": "能引用案例、数据、流程或知识库事实支撑表达，避免空泛承诺。"},
                        {"point_name": "风险降低", "score": 6, "description": "能解释落地方式、验证路径或试点方式，降低客户决策顾虑。"},
                    ],
                },
                {
                    "dimension_name": "异议处理与推进动作",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "异议承接", "score": 7, "description": "面对价格、风险、交付等异议时先承接再回应，不回避客户质疑。"},
                        {"point_name": "针对回应", "score": 7, "description": "能根据客户具体异议给出对应解释，不使用模板化套话。"},
                        {"point_name": "下一步推进", "score": 6, "description": "能争取客户继续沟通、试点、提供资料或约定下一次联系。"},
                    ],
                },
            ],
            "review_mode": "ai_auto",
            "formula": "总分 = 通用能力得分 + 阶段能力得分 - 扣分；一期暂不启用违规词扣分",
        }

    @staticmethod
    def _normalize_dimension_scores(dimensions: list[Any], *, total_score: int) -> list[dict[str, Any]]:
        """按总分归一化评分维度。

        LLM 可能给出 55 或 63 分，这里统一按比例缩放到目标总分。
        """

        normalized: list[dict[str, Any]] = []
        source_dimensions = [item for item in dimensions if isinstance(item, dict)]
        if not source_dimensions:
            return []
        raw_total = sum(max(0, int(item.get("score") or 0)) for item in source_dimensions) or total_score
        allocated = 0
        for index, item in enumerate(source_dimensions):
            score = int(round(max(0, int(item.get("score") or 0)) * total_score / raw_total))
            if index == len(source_dimensions) - 1:
                score = total_score - allocated
            allocated += score
            points = item.get("points") if isinstance(item.get("points"), list) else []
            normalized.append({
                "dimension_name": str(item.get("dimension_name") or item.get("stage_name") or "阶段评分"),
                "score": max(0, score),
                "core_goal": item.get("core_goal") or "",
                "points": points,
            })
        return normalized

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

        return "\n".join(
            [
                request.profile_type,
                request.scenario_description,
                request.extra_details,
                " ".join(request.trainee.weakness_tags),
                json.dumps(request.selected_fields, ensure_ascii=False),
            ]
        )

    def _role_prompt(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        return f"""
请根据学员画像、客户画像字段、场景描述和训练知识，生成销售训练 AI 陪练角色。

要求：
1. 只输出 JSON 对象，不要输出 Markdown。
2. 必须输出 visible_profile、hidden_profile、role_profile、role_confirm_card 四个对象。
3. 只生成训练必要字段，不要生成大而全的客户画像详情。
4. 每个数组最多 3 条，句子要短，避免长篇解释。
5. hidden_profile 给 AI 客户内部使用，训练对话中不能原文暴露给学员。
6. 不要编造具体数字；具体产品、案例、竞品、周期、效果等事实优先来自训练知识，没有证据就写成“需要进一步确认”。

JSON 字段建议：
{{
  "visible_profile": {{
    "角色名称": "适合页面展示的客户名称",
    "性别": "男/女/未知",
    "年龄": "合理年龄，可为空",
    "职位": "根据行业和客户类型推断合理职位",
    "身份": "职位 + 行业",
    "性格特征": "一句话",
    "角色摘要": "2-3句，说明客户背景、当前阶段和主要关注点",
    "成本控制习惯": ["最多3条"],
    "业务痛点": ["最多3条"],
    "潜台词": ["最多3条"]
  }},
  "hidden_profile": {{
    "真实顾虑": ["最多3条"],
    "成交触发器": ["最多3条"],
    "追问策略": ["最多3条"]
  }},
  "role_profile": {{
    "职位": "AI 客户扮演身份",
    "角色简介": "2-3句，供 AI 客户扮演使用",
    "性格特征": "扮演时保持的性格",
    "成本控制习惯": ["最多3条"],
    "业务痛点": ["最多3条"],
    "潜台词": ["最多3条"],
    "挑战策略": ["最多3条，专门针对学员短板"],
    "异议示例": ["最多3条"],
    "不能直接透露": ["最多3条"]
  }},
  "role_confirm_card": {{
    "角色名称": "页面标题",
    "性别": "男/女/未知",
    "年龄": "合理年龄，可为空",
    "身份": "职位 + 行业",
    "性格特征": "一句话",
    "角色摘要": "2-3句，不能太短",
    "成本控制习惯": ["最多3条"],
    "业务痛点": ["最多3条"],
    "潜台词": ["最多3条"]
  }}
}}

学员画像：
{request.trainee.model_dump_json(indent=2)}

客户字段：
{json.dumps(request.selected_fields, ensure_ascii=False, indent=2)}

场景描述：
{request.scenario_description}

补充细节：
{request.extra_details}

训练知识：
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""

    def _scenario_polish_prompt(self, request: ScenarioPolishRequest) -> str:
        return f"""
请根据客户画像字段、原始场景描述和补充细节，润色销售陪练的场景描述。

要求：
1. 只输出 JSON 对象，不要输出 Markdown。
2. JSON 只有一个字段：polished_scenario。
3. polished_scenario 使用中文，控制在 80-160 字。
4. 要把客户身份、合作阶段、核心顾虑、沟通氛围写清楚。
5. 不要新增没有依据的具体数字、公司名、成交金额或产品参数。
6. 语气要适合训练配置页展示，清晰、具体、有业务现场感。

输出示例：
{{
  "polished_scenario": "客户正在评估新的合作方案，当前对交付稳定性、投入产出和团队执行压力仍有顾虑。学员需要先通过提问确认客户真实目标，再结合案例和价值点推动客户愿意继续沟通。"
}}

画像类型：
{request.profile_type}

客户画像字段：
{json.dumps(request.selected_fields, ensure_ascii=False, indent=2)}

原始场景描述：
{request.scenario_description}

补充细节：
{request.extra_details}
"""

    def _supplement_questions_prompt(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> str:
        return f"""
请根据客户画像、学员画像、场景描述和训练知识，生成“补充问答场景细节”的选择题。

要求：
1. 只输出 JSON 对象，不要输出 Markdown。
2. questions 最少 1 道，最多 5 道，优先输出 5 道。
3. 每道题必须有 4 个选项，选项编码固定为 A/B/C/D。
4. 题目要覆盖客户核心痛点、价格/成本、性格与沟通风格、业务流程卡点、成交顾虑。
5. 题干要像真实业务访谈，不要写成技术参数问卷。
6. 选项要能直接影响后续 AI 客户角色，不要泛泛而谈。

JSON 格式：
{{
  "questions": [
    {{
      "question_id": "q1",
      "question_no": 1,
      "dimension": "价格顾虑",
      "question": "如果客户确实认可方案价值，在决策前最可能担心什么？",
      "options": [
        {{"option_code": "A", "option_text": "先对比几家，确认价格是否合理"}},
        {{"option_code": "B", "option_text": "担心团队学习成本太高"}},
        {{"option_code": "C", "option_text": "需要看到同行案例和效果证明"}},
        {{"option_code": "D", "option_text": "希望先试用，确认节省时间再付费"}}
      ],
      "allow_other": true
    }}
  ]
}}

学员画像：
{request.trainee.model_dump_json(indent=2)}

客户字段：
{json.dumps(request.selected_fields, ensure_ascii=False, indent=2)}

场景描述：
{request.scenario_description}

已有补充细节：
{request.extra_details}

训练知识：
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""

    def _normalize_supplement_questions(self, raw_questions: Any, request: RoleGenerateRequest) -> list[SupplementQuestion]:
        """把 LLM 输出规整成前端稳定可渲染的 1-5 道题。"""

        questions: list[SupplementQuestion] = []
        source = raw_questions if isinstance(raw_questions, list) else []
        fallback = self._fallback_supplement_questions(request)
        if not source:
            source = fallback

        for index, item in enumerate(source[:5], start=1):
            if not isinstance(item, dict):
                continue
            raw_options = item.get("options") if isinstance(item.get("options"), list) else []
            options: list[SupplementQuestionOption] = []
            for option_index, option in enumerate(raw_options[:4]):
                option_code = chr(ord("A") + option_index)
                if isinstance(option, dict):
                    option_text = str(option.get("option_text") or option.get("text") or "").strip()
                    option_code = str(option.get("option_code") or option_code).strip()[:1].upper() or option_code
                else:
                    option_text = str(option or "").strip()
                if option_text:
                    options.append(SupplementQuestionOption(option_code=option_code, option_text=option_text))

            if len(options) < 4:
                fallback_item = fallback[min(index - 1, len(fallback) - 1)]
                fallback_options = fallback_item["options"]
                for option in fallback_options[len(options):4]:
                    options.append(SupplementQuestionOption(**option))

            question_text = str(item.get("question") or "").strip()
            if not question_text:
                question_text = str(fallback[min(index - 1, len(fallback) - 1)]["question"])

            questions.append(
                SupplementQuestion(
                    question_id=str(item.get("question_id") or f"q{index}"),
                    question_no=int(item.get("question_no") or index),
                    question=question_text,
                    options=options[:4],
                    allow_other=bool(item.get("allow_other", True)),
                    dimension=str(item.get("dimension") or ""),
                )
            )

        if not questions:
            return [SupplementQuestion(**item) for item in fallback]
        return questions[:5]

    def _goal_prompt(self, profile: dict[str, Any]) -> str:
        role_profile = self._load_json(profile.get("role_profile_json"), {})
        hidden_profile = self._load_json(profile.get("hidden_profile_json"), {})
        return f"""
请生成一期开放式销售训练设置，只输出 JSON。

字段要求：
- training_purpose：20字以内
- round_limit：5到100之间，必须动态判断，不要固定值
- stages：数组，只能有一个阶段，包含 stage_no、stage_name、core_goal、success_conditions、failure_conditions
- scoring_rules：评分规则对象。通用能力固定 40 分不用改；你只生成 stage_dimensions，合计 60 分。
- stage_dimensions 至少 3 个评分维度；每个评分维度至少 3 个考核点；不要只输出一个大维度。
- 每个考核点要能对应实际对话表现，描述要具体，不要写成空泛口号。

scoring_rules 格式：
{{
  "stage_dimensions": [
    {{
      "dimension_name": "需求挖掘与痛点确认",
      "score": 20,
      "core_goal": "和训练目标一致",
      "points": [
        {{"point_name": "背景追问", "score": 7, "description": "能追问客户背景和现有流程"}},
        {{"point_name": "痛点定位", "score": 7, "description": "能定位客户明确痛点"}},
        {{"point_name": "需求确认", "score": 6, "description": "能确认优先级和下一步意向"}}
      ]
    }},
    {{
      "dimension_name": "价值呈现与证据支撑",
      "score": 20,
      "core_goal": "和训练目标一致",
      "points": [
        {{"point_name": "价值匹配", "score": 7, "description": "能把方案价值和客户痛点连接"}},
        {{"point_name": "证据提供", "score": 7, "description": "能提供案例、数据或知识库事实"}},
        {{"point_name": "风险降低", "score": 6, "description": "能说明试点、交付或验证路径"}}
      ]
    }},
    {{
      "dimension_name": "异议处理与推进动作",
      "score": 20,
      "core_goal": "和训练目标一致",
      "points": [
        {{"point_name": "异议承接", "score": 7, "description": "能先承接客户异议"}},
        {{"point_name": "针对回应", "score": 7, "description": "能针对具体异议给出回应"}},
        {{"point_name": "下一步推进", "score": 6, "description": "能推进试点、资料交换或下次沟通"}}
      ]
    }}
  ]
}}

AI 陪练角色：
{json.dumps(role_profile, ensure_ascii=False, indent=2)}

隐藏顾虑：
{json.dumps(hidden_profile, ensure_ascii=False, indent=2)}
"""

    def _opening_prompt(self, session: dict[str, Any]) -> str:
        profile = self._require_role_profile(session["profile_id"])
        setting = self._require_goal_setting(session["setting_id"])
        role_profile = self._load_json(profile.get("role_profile_json"), {})
        hidden_profile = self._load_json(profile.get("hidden_profile_json"), {})
        stages = self._load_json(setting.get("stages_json"), [])
        return f"""
你正在扮演销售训练中的客户。请生成训练开场白，只输出客户说的话。

要求：
1. 不要暴露 hidden_profile 和评分规则。
2. 语气要符合客户身份和性格，不要像客服助手。
3. 主动抛出一个业务背景或顾虑，让学员可以接话。
4. 控制在 60-120 字。

角色设定：
{json.dumps(role_profile, ensure_ascii=False, indent=2)}

隐藏顾虑：
{json.dumps(hidden_profile, ensure_ascii=False, indent=2)}

训练目标：
{json.dumps(stages, ensure_ascii=False, indent=2)}
"""

    def _customer_prompt(self, session: dict[str, Any], trainee_message: str, evidence: list[dict[str, Any]]) -> str:
        profile = self._require_role_profile(session["profile_id"])
        setting = self._require_goal_setting(session["setting_id"])
        role_profile = self._load_json(profile.get("role_profile_json"), {})
        hidden_profile = self._load_json(profile.get("hidden_profile_json"), {})
        stages = self._load_json(setting.get("stages_json"), [])
        turns = self.repository.list_turns(session["session_id"])[-10:]
        return f"""
你正在扮演销售训练中的客户，不是客服助手。

规则：
1. 必须保持客户身份、性格和隐藏顾虑。
2. 不要直接说出 hidden_profile 原文。
3. 根据学员回复逐步释放信息、追问、质疑或给出继续沟通信号。
4. 回复要自然，控制在 80-180 字。

角色设定：
{json.dumps(role_profile, ensure_ascii=False, indent=2)}

隐藏顾虑：
{json.dumps(hidden_profile, ensure_ascii=False, indent=2)}

训练目标：
{json.dumps(stages, ensure_ascii=False, indent=2)}

最近对话：
{json.dumps([{"role": item["role"], "content": item["content"]} for item in turns], ensure_ascii=False, indent=2)}

本轮学员回复：
{trainee_message}

本轮检索证据：
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""

    def _score_prompt(
            self,
            profile: dict[str, Any],
            setting: dict[str, Any],
            turns: list[dict[str, Any]],
            evidence: list[dict[str, Any]],
    ) -> str:
        scoring_rules = self._load_json(setting.get("scoring_rules_json"), self._default_scoring_rules())
        return f"""
请作为销售训练考官，对本次开放式训练评分，只输出 JSON。

输出字段：
total_score、general_score、stage_score、penalty_score、hit_points、missing_points、wrong_points、
evidence_refs、improvement_advice、reference_script、next_training_plan。

评分规则：
必须严格按照 scoring_rules 评分。通用能力最高 40 分，阶段能力最高 60 分，附加扣分一期只包含文字响应时效。
评分必须引用对话轮次或知识库证据。

角色：
{profile.get("role_profile_json")}

训练设置：
{setting.get("stages_json")}

评分设置：
{json.dumps(scoring_rules, ensure_ascii=False, indent=2)}

完整对话：
{json.dumps([{"round_no": item["round_no"], "role": item["role"], "content": item["content"]} for item in turns], ensure_ascii=False, indent=2)}

知识证据：
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""

    @staticmethod
    def _conversation_text(turns: list[dict[str, Any]]) -> str:
        return "\n".join(f"{item['role']}：{item['content']}" for item in turns)

    @staticmethod
    def _fallback_polished_scenario(request: ScenarioPolishRequest) -> str:
        """模型润色失败时的本地兜底文案。

        兜底逻辑只做安全拼接，不虚构业务事实；这样即使 LLM 报错，
        前端也能得到一段可用的场景描述。
        """

        selected_fields = request.selected_fields or {}
        field_text = "；".join(
            f"{key}：{value}"
            for key, value in selected_fields.items()
            if str(value or "").strip()
        )
        parts = [
            f"当前客户画像为{request.profile_type}",
            f"画像信息包括{field_text}" if field_text else "",
            f"原始场景是：{request.scenario_description.strip()}",
            f"补充要求是：{request.extra_details.strip()}" if request.extra_details.strip() else "",
            "学员需要围绕客户真实顾虑展开提问，并用匹配的价值表达推动客户继续沟通。",
        ]
        return "。".join(part.strip("。") for part in parts if part).strip("。") + "。"

    def _fallback_supplement_questions(self, request: RoleGenerateRequest) -> list[dict[str, Any]]:
        """补充问题兜底模板，保证模型不可用时流程仍可继续。"""

        fields = request.selected_fields or {}
        profile_name = str(fields.get("画像类型") or request.profile_type or "客户")
        scenario = request.scenario_description.strip() or "当前业务增长方案"
        return [
            {
                "question_id": "q1",
                "question_no": 1,
                "dimension": "决策顾虑",
                "question": f"如果“{profile_name}”确实能解决当前问题，在客户决策前，还有哪些因素最可能让客户犹豫？",
                "options": [
                    {"option_code": "A", "option_text": "先对比几家供应商，确认价格和方案是否更合理"},
                    {"option_code": "B", "option_text": "担心操作复杂，团队学习和迁移成本太高"},
                    {"option_code": "C", "option_text": "需要看到同行案例和真实效果证明"},
                    {"option_code": "D", "option_text": "希望先试用或小范围验证，再决定是否投入"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q2",
                "question_no": 2,
                "dimension": "价值判断",
                "question": "如果客户现在要换一套新工具，除了价格之外，最看重它做到什么程度才算值？",
                "options": [
                    {"option_code": "A", "option_text": "能明显减少重复工作，把时间省出来"},
                    {"option_code": "B", "option_text": "能降低错误率，避免报价、跟进或交付出问题"},
                    {"option_code": "C", "option_text": "能自动记住客户历史偏好，方便长期复购"},
                    {"option_code": "D", "option_text": "稳定、好上手、不添乱，价格合理就好"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q3",
                "question_no": 3,
                "dimension": "业务卡点",
                "question": "客户跟进业务时，哪个环节最容易让他觉得“明明可以更快但就是快不起来”？",
                "options": [
                    {"option_code": "A", "option_text": "每次都要翻聊天记录、邮件或历史报价"},
                    {"option_code": "B", "option_text": "报价格式、汇率、利润核算反复调整"},
                    {"option_code": "C", "option_text": "不同客户不同价格，担心记错或报错"},
                    {"option_code": "D", "option_text": "客户问一句答一句，碎片沟通占用太多时间"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q4",
                "question_no": 4,
                "dimension": "沟通性格",
                "question": "这位客户在沟通里更可能呈现哪种风格？",
                "options": [
                    {"option_code": "A", "option_text": "务实直接，先问价格和投入产出"},
                    {"option_code": "B", "option_text": "谨慎保守，需要反复确认风险"},
                    {"option_code": "C", "option_text": "结果导向，愿意听方案但只认效果"},
                    {"option_code": "D", "option_text": "容易质疑，喜欢追问细节和边界条件"},
                ],
                "allow_other": True,
            },
            {
                "question_id": "q5",
                "question_no": 5,
                "dimension": "训练挑战",
                "question": f"结合当前场景“{scenario[:40]}”，你希望 AI 客户重点挑战学员哪一类能力？",
                "options": [
                    {"option_code": "A", "option_text": "需求挖掘，逼学员问出真实痛点"},
                    {"option_code": "B", "option_text": "价格异议，持续追问为什么值得投入"},
                    {"option_code": "C", "option_text": "案例证明，要求学员用证据而不是口号说服"},
                    {"option_code": "D", "option_text": "推进下一步，考察学员能否争取试用或继续沟通"},
                ],
                "allow_other": True,
            },
        ]

    def _fallback_role(self, request: RoleGenerateRequest, evidence: list[dict[str, Any]]) -> dict:
        selected_fields = request.selected_fields or {}

        def pick_field(*keys: str, default: str) -> str:
            """从前端选择字段中按多个候选名称取值。

            Python 的 dict.get(key) 类似 Java Map#get；取不到时返回 None。
            这里做多 key 兼容，是为了适配后续页面字段改名但语义不变的情况。
            """

            for key in keys:
                value = selected_fields.get(key)
                if value:
                    return str(value).strip()
            return default

        def compact_list(*items: str, min_size: int = 3) -> list[str]:
            """组装列表并去掉空字符串，避免前端展示空行。"""

            result = [str(item).strip() for item in items if str(item or "").strip()]
            while len(result) < min_size:
                result.append("需要在训练对话中进一步确认")
            return result

        industry = pick_field("行业", "客户行业", default="外贸企业")
        customer_type = pick_field("客户类型", "客户画像", default="谨慎型业务负责人")
        price_sensitivity = pick_field("价格敏感度", default="较高")
        personality = pick_field("性格特征", "客户性格", default="务实谨慎，关注风险和投入产出")
        cooperation_stage = pick_field("合作阶段", default="初次接触")
        scenario = request.scenario_description.strip()
        extra_details = request.extra_details.strip()
        pain_source = extra_details or scenario or "客户正在评估新的业务增长方案，但担心成本投入、交付风险和团队执行压力。"
        trainee_weakness = "、".join(request.trainee.weakness_tags) or "需求挖掘和异议处理"
        # evidence 是向量库召回结果；列表推导式相当于 Java stream().map(...).limit(...).collect(...)。
        knowledge_facts = [str(item.get("content") or "")[:160] for item in evidence if item.get("content")]
        if not knowledge_facts:
            knowledge_facts = [
                "训练知识库暂未召回明确事实，需要学员在对话中先确认客户背景。",
                "客户关注成本投入、交付风险和内部推进难度。",
                "学员需要用提问获取更多业务细节，再匹配方案价值。",
            ]

        return {
            "visible_profile": {
                "角色名称": f"{customer_type}客户",
                "性别": "男",
                "年龄": "33",
                "职位": "业务负责人",
                "身份": f"{industry}｜业务负责人",
                "性格特征": personality,
                "角色摘要": f"来自{industry}领域，处于{cooperation_stage}阶段，正在判断方案是否值得继续推进。客户关注实际收益、投入产出和交付风险。",
                "成本控制习惯": compact_list("日常运营严格审核支出", "优先选择高性价比方案", "对长期订阅费用递增较敏感"),
                "业务痛点": compact_list(pain_source[:120], "内部推动需要明确收益依据", "担心方案落地后额外增加团队负担"),
                "潜台词": compact_list("先证明你懂我的业务", "不要只讲功能，要讲对我有什么用", "风险说不清就不会继续推进"),
            },
            "hidden_profile": {
                "真实顾虑": compact_list("担心投入后没有效果", "担心需要新增人力或改变现有流程", "担心内部汇报时缺少可量化依据"),
                "成交触发器": compact_list("同类案例足够具体", "交付路径清晰", "能回答价格与效果的对应关系"),
                "追问策略": compact_list("先追问业务理解", "再追问效果证据", "最后追问落地成本和下一步安排"),
            },
            "role_profile": {
                "职位": "业务负责人",
                "角色简介": f"一位来自{industry}的{customer_type}，处于{cooperation_stage}阶段，正在判断方案是否值得继续推进。",
                "性格特征": personality,
                "成本控制习惯": compact_list("会把价格和效果绑定判断", "会追问是否增加团队执行成本", "倾向先小范围验证"),
                "业务痛点": compact_list(pain_source[:120], "缺少可靠案例支撑内部决策", "担心业务团队执行压力过大"),
                "潜台词": compact_list("你先证明你懂我的业务", "不要只讲功能，要讲对我有什么用", "如果风险说不清，我不会继续推进"),
                "挑战策略": compact_list(f"针对学员短板：{trainee_weakness}，持续追问证据", "对价格价值关系施压", "当回答空泛时要求举例"),
                "异议示例": compact_list("听起来不错，但我怎么判断不是概念包装", "预算不低，你们的价值怎么量化", "我们团队现在很忙，配合成本会不会很高"),
                "不能直接透露": compact_list("真实顾虑不要一次性说完", "不要主动告诉学员评分标准", "隐藏心理只能通过追问逐步体现"),
            },
            "role_confirm_card": {
                "角色名称": f"{customer_type}客户",
                "性别": "男",
                "年龄": "33",
                "身份": f"{industry}｜业务负责人",
                "性格特征": personality,
                "角色摘要": f"客户正在评估新的业务方案，关注{industry}场景下的实际效果、投入产出和交付风险。沟通中会先观察学员是否理解业务，再决定是否继续释放信息。",
                "成本控制习惯": compact_list("日常运营严格审核支出", "优先选择高性价比方案", "对长期订阅费用递增较敏感"),
                "业务痛点": compact_list(pain_source[:120], "内部推动需要明确收益依据", "担心落地成本超出预期"),
                "潜台词": compact_list("争取首年折扣或免费试用机会", "担心操作复杂且不能解决真实痛点", "对比效率提升与成本节省是否成正比"),
            },
        }

    @staticmethod
    def _fallback_goal(profile: dict[str, Any]) -> dict:
        role_profile = SalesTrainingService._load_json(profile.get("role_profile_json"), {})
        hidden_profile = SalesTrainingService._load_json(profile.get("hidden_profile_json"), {})
        evidence = SalesTrainingService._load_json(profile.get("retrieved_evidence_json"), [])
        scenario_text = str(profile.get("scenario_description") or "")
        role_complexity = len(role_profile.get("business_pain_points") or []) + len(role_profile.get("challenge_strategy") or [])
        concern_complexity = len(hidden_profile.get("real_concerns") or [])
        # LLM 正常时会直接给出 round_limit；兜底时按场景复杂度估算，避免一期训练轮数退化成固定值。
        estimated_round_limit = 6 + min(8, role_complexity + concern_complexity + len(evidence) // 2 + len(scenario_text) // 120)
        return {
            "training_purpose": "需求挖掘",
            "round_limit": estimated_round_limit,
            "stages": [
                {
                    "stage_no": 1,
                    "stage_name": "开放式需求挖掘",
                    "core_goal": "通过自然沟通获取客户痛点、顾虑和下一步意向。",
                    "success_conditions": ["客户说出至少一个具体痛点", "客户愿意继续了解方案"],
                    "failure_conditions": ["客户明确拒绝继续沟通", "学员连续多轮没有回应客户关切"],
                }
            ],
            "scoring_rules": SalesTrainingService._default_scoring_rules(
                stages=[
                    {
                        "stage_no": 1,
                        "stage_name": "开放式需求挖掘",
                        "core_goal": "通过自然沟通获取客户痛点、顾虑和下一步意向。",
                    }
                ],
                profile=profile,
            ),
        }

    @staticmethod
    def _fallback_customer_reply(evidence: list[dict[str, Any]]) -> str:
        if evidence:
            return "你说的方向我能理解，不过我更关心实际效果和投入风险。你能结合类似客户案例，具体说说为什么这个方案适合我吗？"
        return "我先听听你的思路，但我比较关注投入产出和落地风险，你别只讲概念。"

    def _fallback_opening_message(self, session: dict[str, Any]) -> str:
        profile = self._require_role_profile(session["profile_id"])
        role_profile = self._load_json(profile.get("role_profile_json"), {})
        position = role_profile.get("position") or "业务负责人"
        pain_points = role_profile.get("business_pain_points") or ["投入产出和落地风险"]
        first_pain = str(pain_points[0]) if pain_points else "投入产出和落地风险"
        return f"我是这边的{position}。你可以先简单讲讲方案，不过我更关心{first_pain}，如果只是概念性的介绍，可能很难推动内部继续评估。"

    @staticmethod
    def _fallback_score(turns: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict:
        trainee_turns = [item for item in turns if item["role"] == "trainee"]
        base_score = 72 + min(10, len(trainee_turns) * 2)
        return {
            "total_score": min(88, base_score),
            "general_score": 32,
            "stage_score": max(0, min(56, base_score - 32)),
            "penalty_score": 0,
            "hit_points": ["完成了基本沟通", "能围绕客户问题继续回应"],
            "missing_points": ["需要更多追问客户真实顾虑", "需要引用更具体案例"],
            "wrong_points": [],
            "evidence_refs": [{"type": "dialogue", "round_no": item["round_no"]} for item in trainee_turns[:3]],
            "improvement_advice": "下一次训练重点加强需求挖掘和案例化表达。",
            "reference_script": "可以先确认客户当前卡点，再用同类客户案例降低风险感。",
            "next_training_plan": ["需求挖掘专项", "异议处理专项"],
        }

    @staticmethod
    def _score_level(score: int) -> str:
        if score > 90:
            return "优秀"
        if score > 80:
            return "良好"
        if score >= 75:
            return "及格"
        if score >= 60:
            return "待观察"
        return "不及格"
