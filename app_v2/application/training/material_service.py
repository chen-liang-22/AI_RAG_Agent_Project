"""销售训练资料应用服务。"""

from fastapi import UploadFile

from training.schemas import (
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeUploadResponse,
    TrainingKnowledgeVersionListResponse,
)
from utils.logger_handler import logger

from .service_provider import get_sales_training_service


class TrainingMaterialApplicationService:
    """训练资料外观服务。

    这里使用外观模式，把上传、预览、发布、删除、重切等资料管理入口收敛在一起。
    """

    def __init__(self, service=None):
        self.service = service or get_sales_training_service()

    def upload(self, *, file: UploadFile, source_type: str, model_mode: str | None, created_by: str | None) -> TrainingKnowledgeUploadResponse:
        """上传销售训练资料并生成预览切片。"""

        logger.info("[V2销售训练-资料] 上传资料开始 文件名=%s 来源类型=%s 创建人=%s", file.filename, source_type, created_by)
        return self.service.upload_knowledge(file=file, source_type=source_type, created_by=created_by, model_mode=model_mode)

    def list_batches(self, *, page: int, page_size: int) -> TrainingKnowledgeBatchListResponse:
        """分页查询训练资料批次。"""

        return self.service.list_batches(page=page, page_size=page_size)

    def preview_batch(self, batch_id: str, *, max_chars: int) -> TrainingKnowledgePreviewResponse:
        """预览训练资料原文件。"""

        return self.service.preview_batch(batch_id, max_chars=max_chars)

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次。"""

        logger.info("[V2销售训练-资料] 删除训练资料 批次编号=%s", batch_id)
        return self.service.delete_batch(batch_id)

    def publish_batch(self, batch_id: str) -> TrainingKnowledgePublishResponse:
        """发布训练资料。"""

        logger.info("[V2销售训练-资料] 发布训练资料 批次编号=%s", batch_id)
        return self.service.publish_batch(batch_id)

    def rollback_batch(self, batch_id: str) -> TrainingKnowledgeRollbackResponse:
        """回滚训练资料版本。"""

        return self.service.rollback_batch(batch_id)

    def reparse_batch(self, batch_id: str, *, use_llm_fallback: bool, model_mode: str | None) -> TrainingKnowledgeReparseResponse:
        """重新切分未发布训练资料。"""

        logger.info("[V2销售训练-资料] 重新切分训练资料 批次编号=%s 是否使用LLM兜底=%s", batch_id, use_llm_fallback)
        return self.service.reparse_batch(batch_id, use_llm_fallback=use_llm_fallback, model_mode=model_mode)

    def list_versions(self, batch_id: str) -> TrainingKnowledgeVersionListResponse:
        """查询训练资料版本链。"""

        return self.service.list_batch_versions(batch_id)

    def list_chunks(self, batch_id: str) -> TrainingKnowledgeChunkListResponse:
        """查询训练资料切片。"""

        return self.service.list_chunks(batch_id)
