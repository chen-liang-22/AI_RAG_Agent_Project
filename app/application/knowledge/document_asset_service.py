"""统一文件资产服务。

该服务使用外观模式，把文件删除涉及的 MySQL、Qdrant、MinIO 操作集中到一个入口。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.file_storage_service import FileStorageService, get_file_storage_service
from app.infrastructure.vector_store_service import VectorStoreService
from app.application.training_support.repository import TrainingRepository
from core.utils.knowledge_asset_constants import TRAINING_COLLECTION_NAMES
from core.utils.redis_client import RedisClient, get_redis_client
from core.utils.logger_handler import logger


@dataclass(slots=True)
class DocumentAssetDeleteResult:
    """文件资产删除结果。"""

    document_id: str
    deleted_document: bool
    deleted_training_batches: int
    deleted_batch_ids: list[str] = field(default_factory=list)
    deleted_minio_object: bool = False
    status: str = "deleted"
    resource_results: dict[str, dict] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


class DocumentAssetService:
    """统一文件资产服务。

    路由和业务服务只调用这个外观服务，不直接拼接 MySQL、Qdrant、MinIO 的删除步骤。
    """

    def __init__(
            self,
            *,
            knowledge_store=None,
            document_repository: DocumentRepository | None = None,
            training_repository: TrainingRepository | None = None,
            file_storage: FileStorageService | None = None,
            vector_service_factory: Callable[[str], VectorStoreService] | None = None,
            delete_document_vectors: Callable[[str, str | None], None] | None = None,
            redis_state_cleaner: Callable[[str], int] | None = None,
    ):
        """初始化文件资产删除外观服务。

        删除文件会同时影响 documents 表、训练资料批次、MinIO 对象和 Qdrant 向量点，
        所以这里把相关依赖集中注入，避免删除流程散落在多个路由里。
        """

        self.knowledge_store = knowledge_store
        self.document_repository = document_repository or DocumentRepository(store=knowledge_store)
        self.training_repository = training_repository or TrainingRepository()
        self.file_storage = file_storage or get_file_storage_service()
        self.vector_service_factory = vector_service_factory or (
            lambda collection_name: VectorStoreService(collection_name=collection_name)
        )
        self.delete_document_vectors = delete_document_vectors or VectorStoreService.delete_document_vectors
        self.redis_state_cleaner = redis_state_cleaner or self.default_redis_state_cleaner

    def delete_document_asset(self, document_id: str) -> DocumentAssetDeleteResult:
        """按 document_id 全链路硬删除文件资产。"""

        document = self.document_repository.get_document(document_id)
        if document is None:
            logger.info("[知识资产] 文件资产重复删除跳过 文档编号=%s", document_id)
            return DocumentAssetDeleteResult(
                document_id=document_id,
                deleted_document=False,
                deleted_training_batches=0,
                status="not_found",
                resource_results={
                    "mysql": {
                        "status": "not_found",
                        "deleted_document": False,
                        "deleted_training_batches": 0,
                    }
                },
            )

        collection_name = str(document.get("collection_name") or "agent")
        batches = list(self.training_repository.list_batches_by_document_id(document_id))
        batch_ids = [str(batch.get("batch_id")) for batch in batches if batch.get("batch_id")]
        collections_to_clean = {collection_name}
        if batch_ids:
            collections_to_clean.update(TRAINING_COLLECTION_NAMES)

        resource_results: dict[str, dict] = {}
        errors: list[dict[str, str]] = []
        qdrant_deleted = self.delete_qdrant_resources(document_id, collections_to_clean, batch_ids, errors)
        resource_results["qdrant"] = {
            "status": "deleted" if qdrant_deleted else "delete_failed",
            "collections": sorted(collections_to_clean),
            "batch_ids": batch_ids,
        }
        if not qdrant_deleted:
            self.mark_delete_failed(document_id, errors)
            return self.failed_result(document_id, batch_ids, resource_results, errors)

        deleted_minio_object = self.delete_minio_resource(document, document_id, errors)
        resource_results["minio"] = {
            "status": "deleted" if deleted_minio_object else "delete_failed",
            "bucket_name": document.get("bucket_name"),
            "object_name": document.get("object_name"),
            "deleted": deleted_minio_object,
        }
        if not deleted_minio_object:
            self.mark_delete_failed(document_id, errors)
            return self.failed_result(document_id, batch_ids, resource_results, errors)

        redis_deleted_count = self.delete_redis_state(document_id, errors)
        resource_results["redis"] = {
            "status": "deleted" if redis_deleted_count >= 0 else "delete_failed",
            "deleted_count": max(0, redis_deleted_count),
        }
        if redis_deleted_count < 0:
            self.mark_delete_failed(document_id, errors)
            return self.failed_result(document_id, batch_ids, resource_results, errors)

        try:
            deleted_training_batches = self.training_repository.delete_batches_by_document_id(document_id)
            deleted_document = self.document_repository.delete_document(document_id)
            resource_results["mysql"] = {
                "status": "deleted" if deleted_document else "not_found",
                "deleted_document": deleted_document,
                "deleted_training_batches": deleted_training_batches,
            }
        except Exception as exc:
            errors.append({"resource": "mysql", "message": str(exc)})
            logger.error("[知识资产] MySQL记录删除失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
            return self.failed_result(document_id, batch_ids, resource_results, errors)

        logger.info(
            "[知识资产] 文件资产已删除 文档编号=%s 批次数=%s MinIO=%s Redis删除数=%s",
            document_id,
            deleted_training_batches,
            deleted_minio_object,
            redis_deleted_count,
        )
        return DocumentAssetDeleteResult(
            document_id=document_id,
            deleted_document=deleted_document,
            deleted_training_batches=deleted_training_batches,
            deleted_batch_ids=batch_ids,
            deleted_minio_object=deleted_minio_object,
            status="deleted",
            resource_results=resource_results,
            errors=errors,
        )

    def delete_qdrant_resources(
            self,
            document_id: str,
            collections_to_clean: set[str],
            batch_ids: list[str],
            errors: list[dict[str, str]],
    ) -> bool:
        """删除 Qdrant 正式库、训练库和临时库向量点。"""

        try:
            for target_collection in sorted(collections_to_clean):
                self.delete_document_vectors(document_id, target_collection)

            for batch_id in batch_ids:
                for target_collection in TRAINING_COLLECTION_NAMES:
                    self.vector_service_factory(target_collection).delete_by_metadata("batch_id", batch_id)
            return True
        except Exception as exc:
            errors.append({"resource": "qdrant", "message": str(exc)})
            logger.error("[知识资产] Qdrant向量删除失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
            return False

    def delete_minio_resource(self, document: dict, document_id: str, errors: list[dict[str, str]]) -> bool:
        """删除 MinIO 原文件对象。"""

        try:
            return self.file_storage.delete_object(
                bucket_name=document.get("bucket_name"),
                object_name=document.get("object_name"),
            )
        except Exception as exc:
            errors.append({"resource": "minio", "message": str(exc)})
            logger.error("[知识资产] MinIO对象删除失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
            return False

    def delete_redis_state(self, document_id: str, errors: list[dict[str, str]]) -> int:
        """删除文件相关 Redis 临时状态，返回删除 key 数量。"""

        try:
            return int(self.redis_state_cleaner(document_id))
        except Exception as exc:
            errors.append({"resource": "redis", "message": str(exc)})
            logger.error("[知识资产] Redis状态删除失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
            return -1

    def mark_delete_failed(self, document_id: str, errors: list[dict[str, str]]) -> None:
        """外部资源删除失败时，把 MySQL 文档状态标记为 delete_failed。"""

        error_message = "; ".join(f"{item['resource']}:{item['message']}" for item in errors)
        try:
            self.document_repository.update_document_status(
                document_id,
                "delete_failed",
                error_message=error_message[:1000],
            )
        except Exception as exc:
            errors.append({"resource": "mysql_status", "message": str(exc)})
            logger.error("[知识资产] 删除失败状态更新失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)

    @staticmethod
    def failed_result(
            document_id: str,
            batch_ids: list[str],
            resource_results: dict[str, dict],
            errors: list[dict[str, str]],
    ) -> DocumentAssetDeleteResult:
        """构造删除失败但可补偿的返回结果。"""

        resource_results.setdefault("mysql", {
            "status": "retained",
            "deleted_document": False,
            "deleted_training_batches": 0,
        })
        return DocumentAssetDeleteResult(
            document_id=document_id,
            deleted_document=False,
            deleted_training_batches=0,
            deleted_batch_ids=batch_ids,
            deleted_minio_object=False,
            status="delete_failed",
            resource_results=resource_results,
            errors=errors,
        )

    @staticmethod
    def default_redis_state_cleaner(document_id: str) -> int:
        """删除当前文档可能关联的 Redis 临时状态。"""

        redis_client: RedisClient = get_redis_client()
        keys = [
            redis_client.build_key("document_delete", document_id),
            redis_client.build_key("document_index", document_id),
            redis_client.build_key("knowledge_preview", document_id),
        ]
        return int(redis_client.delete(*keys))
