"""统一文件资产服务。

该服务使用外观模式，把文件删除涉及的 MySQL、Qdrant、MinIO 操作集中到一个入口。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import HTTPException

from infrastructure.file_storage_service import FileStorageService, get_file_storage_service
from infrastructure.vector_store_service import VectorStoreService
from rag.knowledge_store import KnowledgeStore
from training.repository import TrainingRepository
from utils.knowledge_asset_constants import TRAINING_COLLECTION_NAMES
from utils.logger_handler import logger


@dataclass(slots=True)
class DocumentAssetDeleteResult:
    """文件资产删除结果。"""

    document_id: str
    deleted_document: bool
    deleted_training_batches: int
    deleted_batch_ids: list[str] = field(default_factory=list)
    deleted_minio_object: bool = False


class DocumentAssetService:
    """统一文件资产服务。

    路由和业务服务只调用这个外观服务，不直接拼接 MySQL、Qdrant、MinIO 的删除步骤。
    """

    def __init__(
            self,
            *,
            knowledge_store: KnowledgeStore | None = None,
            training_repository: TrainingRepository | None = None,
            file_storage: FileStorageService | None = None,
            vector_service_factory: Callable[[str], VectorStoreService] | None = None,
            delete_document_vectors: Callable[[str, str | None], None] | None = None,
    ):
        self.knowledge_store = knowledge_store or KnowledgeStore()
        self.training_repository = training_repository or TrainingRepository()
        self.file_storage = file_storage or get_file_storage_service()
        self.vector_service_factory = vector_service_factory or (
            lambda collection_name: VectorStoreService(collection_name=collection_name)
        )
        self.delete_document_vectors = delete_document_vectors or VectorStoreService.delete_document_vectors

    def delete_document_asset(self, document_id: str) -> DocumentAssetDeleteResult:
        """按 document_id 全链路硬删除文件资产。"""

        document = self.knowledge_store.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

        collection_name = str(document.get("collection_name") or "agent")
        batches = list(self.training_repository.list_batches_by_document_id(document_id))
        batch_ids = [str(batch.get("batch_id")) for batch in batches if batch.get("batch_id")]
        collections_to_clean = {collection_name}
        if batch_ids:
            collections_to_clean.update(TRAINING_COLLECTION_NAMES)

        try:
            for target_collection in sorted(collections_to_clean):
                self.delete_document_vectors(document_id, target_collection)

            for batch_id in batch_ids:
                for target_collection in TRAINING_COLLECTION_NAMES:
                    self.vector_service_factory(target_collection).delete_by_metadata("batch_id", batch_id)

            deleted_minio_object = self.file_storage.delete_object(
                bucket_name=document.get("bucket_name"),
                object_name=document.get("object_name"),
            )
            deleted_training_batches = self.training_repository.delete_batches_by_document_id(document_id)
            deleted_document = self.knowledge_store.delete_document(document_id)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "[知识资产] 文件资产删除失败 文档编号=%s 错误=%s",
                document_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"文件资产删除失败：{exc}") from exc

        logger.info(
            "[知识资产] 文件资产已删除 文档编号=%s 批次数=%s MinIO=%s",
            document_id,
            deleted_training_batches,
            deleted_minio_object,
        )
        return DocumentAssetDeleteResult(
            document_id=document_id,
            deleted_document=deleted_document,
            deleted_training_batches=deleted_training_batches,
            deleted_batch_ids=batch_ids,
            deleted_minio_object=deleted_minio_object,
        )
