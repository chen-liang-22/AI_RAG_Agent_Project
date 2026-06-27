from utils.knowledge_asset_constants import TRAINING_COLLECTION_NAMES


class FakeDocumentStore:
    def __init__(self):
        self.documents = {
            "doc_general": {
                "document_id": "doc_general",
                "filename": "general.txt",
                "collection_name": "agent",
                "bucket_name": "pub",
                "object_name": "documents/doc_general/general.txt",
            },
            "doc_training": {
                "document_id": "doc_training",
                "filename": "training.txt",
                "collection_name": "sales_training_cases",
                "bucket_name": "pub",
                "object_name": "documents/doc_training/training.txt",
            },
        }
        self.deleted_document_ids = []

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def delete_document(self, document_id):
        self.deleted_document_ids.append(document_id)
        self.documents.pop(document_id, None)
        return True

    def list_documents(self, *, include_training=False):
        rows = list(self.documents.values())
        if include_training:
            return rows
        return [
            row
            for row in rows
            if row.get("collection_name") not in TRAINING_COLLECTION_NAMES
        ]


def test_fake_store_documents_filter_matches_training_collection_rule():
    store = FakeDocumentStore()

    rows = store.list_documents()

    assert [row["document_id"] for row in rows] == ["doc_general"]


def test_document_repository_exposes_hard_delete_and_filtered_list_methods():
    from app_v2.infrastructure.repositories.document_repository import DocumentRepository

    assert hasattr(DocumentRepository, "delete_document")
    parameters = __import__("inspect").signature(DocumentRepository.list_documents).parameters
    assert "include_training" in parameters


def test_training_repository_exposes_hard_delete_helpers():
    from training.repository import TrainingRepository

    assert hasattr(TrainingRepository, "list_batches_by_document_id")
    assert hasattr(TrainingRepository, "delete_batch")
    assert hasattr(TrainingRepository, "delete_batches_by_document_id")


class FakeTrainingRepository:
    def __init__(self):
        self.batches_by_document_id = {
            "doc_training": [
                {
                    "batch_id": "batch_1",
                    "document_id": "doc_training",
                }
            ]
        }
        self.deleted_document_ids = []
        self.deleted_batch_ids = []

    def list_batches_by_document_id(self, document_id):
        return self.batches_by_document_id.get(document_id, [])

    def delete_batches_by_document_id(self, document_id):
        self.deleted_document_ids.append(document_id)
        return len(self.batches_by_document_id.pop(document_id, []))

    def delete_batch(self, batch_id):
        self.deleted_batch_ids.append(batch_id)
        return True


class FakeVectorService:
    document_vector_deletes = []

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.deleted_metadata = []

    def delete_by_metadata(self, field_name, field_value):
        self.deleted_metadata.append((field_name, field_value))

    @staticmethod
    def delete_document_vectors(document_id, collection_name=None):
        FakeVectorService.document_vector_deletes.append((document_id, collection_name))


class FakeFileStorage:
    def __init__(self):
        self.deleted_objects = []

    def delete_object(self, *, bucket_name=None, object_name=None):
        self.deleted_objects.append((bucket_name, object_name))
        return True


def test_document_asset_service_deletes_document_training_batches_vectors_and_minio():
    from app_v2.application.knowledge.document_asset_service import DocumentAssetService

    document_store = FakeDocumentStore()
    training_repository = FakeTrainingRepository()
    storage = FakeFileStorage()
    published_vector = FakeVectorService("sales_training_cases")
    staging_vector = FakeVectorService("sales_training_cases_staging")
    general_vector = FakeVectorService("agent")
    FakeVectorService.document_vector_deletes = []

    service = DocumentAssetService(
        knowledge_store=None,
        document_repository=document_store,
        training_repository=training_repository,
        file_storage=storage,
        vector_service_factory=lambda collection_name: {
            "agent": general_vector,
            "sales_training_cases": published_vector,
            "sales_training_cases_staging": staging_vector,
        }[collection_name],
        delete_document_vectors=FakeVectorService.delete_document_vectors,
    )

    result = service.delete_document_asset("doc_training")

    assert result.document_id == "doc_training"
    assert result.deleted_batch_ids == ["batch_1"]
    assert result.deleted_document is True
    assert result.deleted_training_batches == 1
    assert FakeVectorService.document_vector_deletes == [
        ("doc_training", "sales_training_cases"),
        ("doc_training", "sales_training_cases_staging"),
    ]
    assert published_vector.deleted_metadata == [("batch_id", "batch_1")]
    assert staging_vector.deleted_metadata == [("batch_id", "batch_1")]
    assert storage.deleted_objects == [("pub", "documents/doc_training/training.txt")]
    assert document_store.deleted_document_ids == ["doc_training"]
    assert training_repository.deleted_document_ids == ["doc_training"]

class ExplodingLegacyStoreForAssetDelete:
    """删除服务如果继续访问旧 store，本替身会让测试失败。"""

    def get_document(self, document_id):
        raise AssertionError("文档资产删除不应该继续通过旧存储查询 documents")

    def delete_document(self, document_id):
        raise AssertionError("文档资产删除不应该继续通过旧存储删除 documents")


class FakeDocumentRepositoryForAssetDelete(FakeDocumentStore):
    """测试用 V2 文档仓储，复用 FakeDocumentStore 的 documents 行为。"""


def test_document_asset_service_deletes_document_through_document_repository():
    from app_v2.application.knowledge.document_asset_service import DocumentAssetService

    document_repository = FakeDocumentRepositoryForAssetDelete()
    training_repository = FakeTrainingRepository()
    storage = FakeFileStorage()
    published_vector = FakeVectorService("sales_training_cases")
    staging_vector = FakeVectorService("sales_training_cases_staging")
    FakeVectorService.document_vector_deletes = []

    service = DocumentAssetService(
        knowledge_store=ExplodingLegacyStoreForAssetDelete(),
        document_repository=document_repository,
        training_repository=training_repository,
        file_storage=storage,
        vector_service_factory=lambda collection_name: {
            "sales_training_cases": published_vector,
            "sales_training_cases_staging": staging_vector,
        }[collection_name],
        delete_document_vectors=FakeVectorService.delete_document_vectors,
    )

    result = service.delete_document_asset("doc_training")

    assert result.deleted_document is True
    assert document_repository.deleted_document_ids == ["doc_training"]
    assert training_repository.deleted_document_ids == ["doc_training"]
