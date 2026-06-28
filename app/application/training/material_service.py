"""销售训练资料应用服务。"""

from fastapi import UploadFile

from app.application.training_support.schemas import (
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
from core.utils.logger_handler import logger

from .service_provider import get_training_core_service


class TrainingMaterialApplicationService:
    """训练资料外观服务。

    这里使用外观模式，把上传、预览、发布、删除、重切等资料管理入口收敛在一起。
    """

    def __init__(self, core_service=None):
        """初始化训练资料服务。

        core_service 支持注入；默认延迟创建，避免页面只查资料列表时提前加载重依赖。
        """

        self._core_service = core_service
        self.service = None

    @property
    def core_service(self):
        """延迟获取 V2 销售训练核心服务，避免构造阶段触发旧大类初始化。"""

        if self._core_service is None:
            self._core_service = get_training_core_service()
        return self._core_service

    def upload(self, *, file: UploadFile, source_type: str, model_mode: str | None, created_by: str | None) -> TrainingKnowledgeUploadResponse:
        """上传销售训练资料并生成预览切片。"""

        logger.info("[V2销售训练-资料] 上传资料开始 文件名=%s 来源类型=%s 创建人=%s", file.filename, source_type, created_by)
        return self.core_service.upload_knowledge(file=file, source_type=source_type, created_by=created_by, model_mode=model_mode)

    def list_batches(self, *, page: int, page_size: int) -> TrainingKnowledgeBatchListResponse:
        """分页查询训练资料批次。"""

        logger.info("[V2销售训练-资料] 查询资料批次列表 页码=%s 每页数量=%s", page, page_size)
        return self.core_service.list_batches(page=page, page_size=page_size)

    def preview_batch(self, batch_id: str, *, max_chars: int) -> TrainingKnowledgePreviewResponse:
        """预览训练资料原文件。"""

        logger.info("[V2销售训练-资料] 预览训练资料 批次编号=%s 最大字符数=%s", batch_id, max_chars)
        return self.core_service.preview_batch(batch_id, max_chars=max_chars)

    def delete_batch(self, batch_id: str) -> TrainingKnowledgeDeleteResponse:
        """删除训练资料批次。"""

        logger.info("[V2销售训练-资料] 删除训练资料 批次编号=%s", batch_id)
        return self.core_service.delete_batch(batch_id)

    def publish_batch(self, batch_id: str) -> TrainingKnowledgePublishResponse:
        """发布训练资料。"""

        logger.info("[V2销售训练-资料] 发布训练资料 批次编号=%s", batch_id)
        return self.core_service.publish_batch(batch_id)

    def rollback_batch(self, batch_id: str) -> TrainingKnowledgeRollbackResponse:
        """回滚训练资料版本。"""

        logger.info("[V2销售训练-资料] 回滚训练资料版本 批次编号=%s", batch_id)
        return self.core_service.rollback_batch(batch_id)

    def reparse_batch(self, batch_id: str, *, use_llm_fallback: bool, model_mode: str | None) -> TrainingKnowledgeReparseResponse:
        """重新切分未发布训练资料。"""

        logger.info("[V2销售训练-资料] 重新切分训练资料 批次编号=%s 是否使用LLM兜底=%s", batch_id, use_llm_fallback)
        return self.core_service.reparse_batch(batch_id, use_llm_fallback=use_llm_fallback, model_mode=model_mode)

    def list_versions(self, batch_id: str) -> TrainingKnowledgeVersionListResponse:
        """查询训练资料版本链。"""

        logger.info("[V2销售训练-资料] 查询训练资料版本 批次编号=%s", batch_id)
        return self.core_service.list_batch_versions(batch_id)

    def list_chunks(self, batch_id: str) -> TrainingKnowledgeChunkListResponse:
        """查询训练资料切片。"""

        logger.info("[V2销售训练-资料] 查询训练资料切片 批次编号=%s", batch_id)
        return self.core_service.list_chunks(batch_id)
