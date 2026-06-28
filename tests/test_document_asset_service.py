"""统一文件资产删除服务测试。"""

from app.application.knowledge.document_asset_service import DocumentAssetService


class FakeDocumentRepository:
    """记录文件资产删除和状态补偿动作。"""

    def __init__(self, document=None):
        self.document = document
        self.deleted_document_id = None
        self.status_updates = []

    def get_document(self, document_id: str):
        """返回待删除文件。"""

        if self.document and self.document["document_id"] == document_id:
            return self.document
        return None

    def update_document_status(self, document_id: str, status: str, **kwargs):
        """记录删除失败时的补偿状态。"""

        self.status_updates.append((document_id, status, kwargs))

    def delete_document(self, document_id: str):
        """记录 MySQL 文件记录是否被物理删除。"""

        self.deleted_document_id = document_id
        self.document = None
        return True


class FakeTrainingRepository:
    """记录训练资料批次删除动作。"""

    def __init__(self, batches=None):
        self.batches = batches or []
        self.deleted_document_id = None

    def list_batches_by_document_id(self, document_id: str):
        """返回文档关联的训练资料批次。"""

        return self.batches

    def delete_batches_by_document_id(self, document_id: str):
        """记录批次删除动作。"""

        self.deleted_document_id = document_id
        deleted_count = len(self.batches)
        self.batches = []
        return deleted_count


class FakeFileStorage:
    """记录 MinIO 删除动作。"""

    def __init__(self, *, should_fail=False):
        self.should_fail = should_fail
        self.deleted = []

    def delete_object(self, *, bucket_name: str | None, object_name: str | None):
        """删除对象，必要时模拟失败。"""

        if self.should_fail:
            raise RuntimeError("MinIO 删除失败")
        self.deleted.append((bucket_name, object_name))
        return True


class FakeRedisCleaner:
    """记录 Redis 临时状态清理动作。"""

    def __init__(self):
        self.document_ids = []

    def __call__(self, document_id: str) -> int:
        self.document_ids.append(document_id)
        return 2


class FakeVectorCollection:
    """记录按 batch_id 删除向量动作。"""

    def __init__(self, collection_name: str, calls: list):
        self.collection_name = collection_name
        self.calls = calls

    def delete_by_metadata(self, key: str, value: str):
        self.calls.append((self.collection_name, key, value))


def _document():
    """构造文件资产记录。"""

    return {
        "document_id": "doc_1",
        "collection_name": "agent",
        "bucket_name": "pub",
        "object_name": "documents/doc_1/a.txt",
    }


def _batch():
    """构造训练资料批次记录。"""

    return {"batch_id": "batch_1", "document_id": "doc_1"}


def test_delete_document_asset_returns_each_resource_result():
    """删除成功时应返回 Qdrant、MinIO、Redis、MySQL 每类资源结果。"""

    vector_calls = []
    redis_cleaner = FakeRedisCleaner()
    service = DocumentAssetService(
        document_repository=FakeDocumentRepository(_document()),
        training_repository=FakeTrainingRepository([_batch()]),
        file_storage=FakeFileStorage(),
        vector_service_factory=lambda collection_name: FakeVectorCollection(collection_name, vector_calls),
        delete_document_vectors=lambda document_id, collection_name: vector_calls.append((collection_name, "document_id", document_id)),
        redis_state_cleaner=redis_cleaner,
    )

    result = service.delete_document_asset("doc_1")

    assert result.status == "deleted"
    assert result.resource_results["qdrant"]["status"] == "deleted"
    assert result.resource_results["minio"]["status"] == "deleted"
    assert result.resource_results["redis"]["deleted_count"] == 2
    assert result.resource_results["mysql"]["deleted_document"] is True
    assert redis_cleaner.document_ids == ["doc_1"]
    assert ("agent", "document_id", "doc_1") in vector_calls
    assert ("sales_training_cases", "batch_id", "batch_1") in vector_calls


def test_delete_document_asset_marks_delete_failed_when_qdrant_fails():
    """Qdrant 删除失败时应保留 MySQL 记录并标记 delete_failed。"""

    document_repository = FakeDocumentRepository(_document())
    training_repository = FakeTrainingRepository([_batch()])

    def fail_delete_vectors(document_id: str, collection_name: str | None):
        raise RuntimeError("Qdrant 删除失败")

    service = DocumentAssetService(
        document_repository=document_repository,
        training_repository=training_repository,
        file_storage=FakeFileStorage(),
        vector_service_factory=lambda collection_name: FakeVectorCollection(collection_name, []),
        delete_document_vectors=fail_delete_vectors,
        redis_state_cleaner=FakeRedisCleaner(),
    )

    result = service.delete_document_asset("doc_1")

    assert result.status == "delete_failed"
    assert result.deleted_document is False
    assert result.deleted_training_batches == 0
    assert result.errors
    assert document_repository.deleted_document_id is None
    assert training_repository.deleted_document_id is None
    assert document_repository.status_updates[0][1] == "delete_failed"


def test_delete_document_asset_missing_document_is_idempotent():
    """重复删除不存在的文件时不应抛 500。"""

    service = DocumentAssetService(
        document_repository=FakeDocumentRepository(None),
        training_repository=FakeTrainingRepository(),
        file_storage=FakeFileStorage(),
        vector_service_factory=lambda collection_name: FakeVectorCollection(collection_name, []),
        redis_state_cleaner=FakeRedisCleaner(),
    )

    result = service.delete_document_asset("missing")

    assert result.status == "not_found"
    assert result.document_id == "missing"
    assert result.resource_results["mysql"]["status"] == "not_found"
