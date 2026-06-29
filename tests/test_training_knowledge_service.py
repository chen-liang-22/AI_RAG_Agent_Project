"""销售训练资料服务测试。"""

import json
from datetime import datetime

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.application.training.training_knowledge_service import TrainingKnowledgeService


def _batch(**overrides):
    """构造训练资料批次行，字段形态贴近 repository 返回结果。"""

    base = {
        "batch_id": "batch_1",
        "document_id": "doc_1",
        "source_type": "lms_case",
        "source_file": "legacy.docx",
        "file_path": None,
        "file_md5": None,
        "document_filename": "case.docx",
        "document_file_path": "minio://pub/documents/case.docx",
        "document_file_md5": "md5_1",
        "document_bucket_name": "pub",
        "document_object_name": "documents/doc_1/case.docx",
        "document_public_url": "http://localhost:9000/pub/documents/doc_1/case.docx",
        "version_group_id": "vg_1",
        "version_no": 2,
        "previous_batch_id": "batch_0",
        "is_current": True,
        "profile_type": None,
        "task_type": None,
        "industry": None,
        "difficulty": None,
        "visibility_default": "visible",
        "status": "published",
        "chunk_count": 2,
        "point_count": 2,
        "error_message": None,
        "quality_report_json": json.dumps({"score": 90}, ensure_ascii=False),
        "created_by": "tester",
        "created_at": datetime(2026, 1, 1, 10, 0, 0),
        "updated_at": datetime(2026, 1, 1, 11, 0, 0),
    }
    base.update(overrides)
    return base


class FakeTrainingRepository:
    """训练资料测试仓储。"""

    def __init__(self):
        self.deleted_batch_id = None
        self.list_page = None
        self.list_page_size = None
        self.created_batches = []
        self.status_updates = []

    def list_batches(self, *, page: int, page_size: int):
        """记录分页参数并返回固定批次。"""

        self.list_page = page
        self.list_page_size = page_size
        return [_batch()], 1

    def get_batch(self, batch_id: str):
        """返回指定批次。"""

        if batch_id == "missing":
            return None
        if batch_id == "legacy":
            return _batch(batch_id="legacy", document_id="", status="published")
        return _batch(batch_id=batch_id)

    def delete_batch(self, batch_id: str):
        """记录删除的历史批次。"""

        self.deleted_batch_id = batch_id
        return True

    def get_published_batch_by_md5(self, file_md5: str):
        """测试默认没有重复文件。"""

        return None

    def get_existing_batch_by_md5(self, file_md5: str):
        """测试默认没有任意未删除重复文件。"""

        return None

    def get_latest_batch_for_version(self, *, source_type: str, source_file: str):
        """测试默认按新版本组创建。"""

        return None

    def create_batch(self, **values):
        """记录创建的批次。"""

        batch = _batch(
            batch_id=values["batch_id"],
            document_id=values["document_id"],
            source_type=values["source_type"],
            source_file=values["source_file"],
            status=values["status"],
            chunk_count=0,
            point_count=0,
            quality_report_json=None,
        )
        self.created_batches.append(batch)
        return batch

    def update_batch_status(self, batch_id: str, **values):
        """记录状态更新。"""

        self.status_updates.append((batch_id, values))

    def list_batches_in_version_group(self, version_group_id: str):
        """返回同一版本组的批次。"""

        return [_batch(batch_id="batch_1", version_group_id=version_group_id)]


class FakeVectorService:
    """记录向量库删除动作。"""

    def __init__(self):
        self.deleted = []
        self.listed_metadata = []
        self.added_documents = []

    def delete_by_metadata(self, key: str, value: str):
        """记录按元数据删除的参数。"""

        self.deleted.append((key, value))

    def list_documents_by_metadata(self, key: str, value: str):
        """返回 Qdrant Document 形式的切片。"""

        self.listed_metadata.append((key, value))
        return [
            Document(
                page_content="客户案例正文",
                metadata={
                    "chunk_id": "chunk_1",
                    "batch_id": value,
                    "case_part": "case_profile",
                    "visibility": "visible",
                    "source_file": "case.docx",
                },
            )
        ]

    @property
    def vector_store(self):
        """模拟 LangChain 向量库写入接口。"""

        return self

    def add_documents(self, documents):
        """记录写入的切片。"""

        self.added_documents.extend(documents)


class FakeDocumentRepository:
    """测试用文档仓储。"""

    def __init__(self):
        self.created_documents = []
        self.status_updates = []

    def get_document(self, document_id: str):
        """测试批次已经带有联表字段，不需要额外查文档。"""

        for document in self.created_documents:
            if document["document_id"] == document_id:
                return document
        return None

    def create_document(self, **values):
        """记录文档资产。"""

        document = {
            **values,
            "version": 1,
            "chunk_count": 0,
            "created_at": datetime(2026, 1, 1, 10, 0, 0),
            "updated_at": datetime(2026, 1, 1, 10, 0, 0),
            "error_message": None,
        }
        self.created_documents.append(document)
        return document

    def update_document_status(self, document_id: str, status: str, **values):
        """记录文件状态更新。"""

        self.status_updates.append((document_id, status, values))


class FakeAssetResult:
    """统一文件资产删除结果。"""

    document_id = "doc_1"
    status = "deleted"
    resource_results = {"qdrant": {"status": "deleted"}, "mysql": {"status": "deleted"}}
    errors = []


class FakeAssetService:
    """记录训练资料删除是否调用统一资产服务。"""

    def __init__(self):
        self.deleted_document_id = None

    def delete_document_asset(self, document_id: str):
        self.deleted_document_id = document_id
        return FakeAssetResult()


def _service(repository=None, vector_service=None, staging_vector_service=None, asset_service=None) -> TrainingKnowledgeService:
    """构造训练资料服务。"""

    return TrainingKnowledgeService(
        repository=repository or FakeTrainingRepository(),
        vector_service=vector_service or FakeVectorService(),
        staging_vector_service=staging_vector_service or FakeVectorService(),
        document_repository=FakeDocumentRepository(),
        asset_service=asset_service,
    )


class FakeStoredFile:
    """模拟 MinIO 保存结果。"""

    file_md5 = "md5_new"
    file_path = "minio://pub/training/doc_1/case.txt"
    file_size = 12
    bucket_name = "pub"
    object_name = "training/doc_1/case.txt"
    public_url = "http://localhost:9000/pub/training/doc_1/case.txt"


class FakeStorageService:
    """模拟文件存储服务，记录重复上传时是否删除新对象。"""

    def __init__(self):
        self.deleted_objects = []

    def save_upload_file(self, **kwargs):
        """返回固定 MD5 的已保存文件。"""

        return FakeStoredFile()

    def delete_object(self, *, bucket_name: str, object_name: str):
        """记录被删除的 MinIO 对象。"""

        self.deleted_objects.append((bucket_name, object_name))


class FakeUploadFile:
    """模拟上传文件。"""

    filename = "case.txt"


class FakeTaskService:
    """记录是否创建训练入库任务。"""

    def __init__(self, latest_task=None):
        self.created = []
        self.latest_task = latest_task

    def create_training_ingest_task(self, **values):
        self.created.append(values)
        return {
            "task_id": "task_1",
            "task_status": "queued",
            "current_step": "queued",
            "progress": 5,
        }

    def task_snapshot(self, task):
        return {
            "task_id": task["task_id"],
            "task_status": task["status"],
            "status": task["status"],
            "current_step": task["current_step"],
            "progress": task["progress"],
        }

    @property
    def task_repository(self):
        return self

    def get_latest_task_for_batch(self, batch_id: str):
        return self.latest_task


def test_list_batches_normalizes_page_and_page_size():
    """资料列表需要限制分页参数并转换响应结构。"""

    repository = FakeTrainingRepository()
    response = _service(repository=repository).list_batches(page=0, page_size=999)

    assert repository.list_page == 1
    assert repository.list_page_size == 50
    assert response.total == 1
    assert response.items[0].batch_id == "batch_1"
    assert response.items[0].source_file == "case.docx"
    assert response.items[0].quality_report == {"score": 90}
    assert response.items[0].created_at == "2026-01-01 10:00:00"


def test_delete_legacy_batch_cleans_both_vector_collections_and_repository():
    """没有 document_id 的历史批次必须清理正式库、临时库和批次记录。"""

    repository = FakeTrainingRepository()
    vector_service = FakeVectorService()
    staging_vector_service = FakeVectorService()

    response = _service(
        repository=repository,
        vector_service=vector_service,
        staging_vector_service=staging_vector_service,
    ).delete_batch("legacy")

    assert response.status == "deleted"
    assert response.batch_id == "legacy"
    assert vector_service.deleted == [("batch_id", "legacy")]
    assert staging_vector_service.deleted == [("batch_id", "legacy")]
    assert repository.deleted_batch_id == "legacy"


def test_delete_batch_returns_full_asset_resource_results():
    """有关联 document_id 的训练资料删除应透传统一资产删除结果。"""

    asset_service = FakeAssetService()
    response = _service(asset_service=asset_service).delete_batch("batch_1")

    assert response.status == "deleted"
    assert response.batch_id == "batch_1"
    assert response.document_id == "doc_1"
    assert response.resource_results["qdrant"]["status"] == "deleted"
    assert response.errors == []
    assert asset_service.deleted_document_id == "doc_1"


def test_delete_batch_rejects_running_ingest_task():
    """运行中的异步入库任务不能被删除，避免后台线程继续写 Qdrant。"""

    asset_service = FakeAssetService()
    task_service = FakeTaskService({
        "task_id": "task_running",
        "status": "running",
        "current_step": "chunking",
        "progress": 45,
    })
    service = TrainingKnowledgeService(
        repository=FakeTrainingRepository(),
        vector_service=FakeVectorService(),
        staging_vector_service=FakeVectorService(),
        document_repository=FakeDocumentRepository(),
        asset_service=asset_service,
        ingest_task_service=task_service,
    )

    with pytest.raises(HTTPException) as exc_info:
        service.delete_batch("batch_1")

    assert exc_info.value.status_code == 409
    assert "正在入库处理中" in exc_info.value.detail
    assert asset_service.deleted_document_id is None


def test_delete_batch_rejects_queued_ingest_task():
    """排队中的异步入库任务同样不能被删除，避免后台线程稍后处理已删除批次。"""

    asset_service = FakeAssetService()
    task_service = FakeTaskService({
        "task_id": "task_queued",
        "status": "queued",
        "current_step": "queued",
        "progress": 5,
    })
    service = TrainingKnowledgeService(
        repository=FakeTrainingRepository(),
        vector_service=FakeVectorService(),
        staging_vector_service=FakeVectorService(),
        document_repository=FakeDocumentRepository(),
        asset_service=asset_service,
        ingest_task_service=task_service,
    )

    with pytest.raises(HTTPException) as exc_info:
        service.delete_batch("batch_1")

    assert exc_info.value.status_code == 409
    assert asset_service.deleted_document_id is None


def test_list_versions_returns_version_group_batches():
    """版本列表需要按当前批次所在 version_group_id 查询。"""

    response = _service().list_batch_versions("batch_1")

    assert response.version_group_id == "vg_1"
    assert len(response.items) == 1
    assert response.items[0].version_no == 2


def test_list_chunks_maps_metadata_json():
    """切片列表需要把 metadata_json 转成前端可用对象。"""

    vector_service = FakeVectorService()
    response = _service(vector_service=vector_service).list_chunks("batch_1")

    assert response.batch_id == "batch_1"
    assert response.chunks[0].chunk_id == "chunk_1"
    assert response.chunks[0].case_part == "case_profile"
    assert response.chunks[0].metadata["source_file"] == "case.docx"
    assert vector_service.listed_metadata == [("batch_id", "batch_1")]


def test_upload_training_knowledge_returns_task_without_writing_staging(monkeypatch):
    """销售资料上传应快速创建任务，不同步写临时向量库。"""

    repository = FakeTrainingRepository()
    document_repository = FakeDocumentRepository()
    staging_vector_service = FakeVectorService()
    task_service = FakeTaskService()
    service = TrainingKnowledgeService(
        repository=repository,
        vector_service=FakeVectorService(),
        staging_vector_service=staging_vector_service,
        document_repository=document_repository,
        ingest_task_service=task_service,
    )
    monkeypatch.setattr(
        "app.application.training.training_knowledge_service.get_file_storage_service",
        lambda: type("Storage", (), {"save_upload_file": lambda self, **kwargs: FakeStoredFile()})(),
    )

    response = service.upload_knowledge(
        file=FakeUploadFile(),
        source_type="lms_case",
        created_by="tester",
        model_mode="fast",
    )

    assert response.status == "parsing"
    assert response.task_id == "task_1"
    assert response.task_status == "queued"
    assert response.progress == 5
    assert repository.created_batches[0]["status"] == "parsing"
    assert staging_vector_service.added_documents == []
    assert task_service.created[0]["batch_id"] == repository.created_batches[0]["batch_id"]


def test_upload_training_knowledge_reuses_unpublished_duplicate_batch(monkeypatch):
    """未发布训练资料命中相同 MD5 时，不应继续创建新的上传批次。"""

    class RepositoryWithUnpublishedDuplicate(FakeTrainingRepository):
        """返回一个未发布但未删除的重复批次。"""

        def get_existing_batch_by_md5(self, file_md5: str):
            """按 MD5 返回待发布批次。"""

            if file_md5 == "md5_new":
                return _batch(
                    batch_id="pending_batch",
                    document_id="pending_doc",
                    status="pending_review",
                    document_file_md5="md5_new",
                    chunk_count=3,
                    point_count=0,
                )
            return None

    repository = RepositoryWithUnpublishedDuplicate()
    storage_service = FakeStorageService()
    task_service = FakeTaskService()
    service = TrainingKnowledgeService(
        repository=repository,
        vector_service=FakeVectorService(),
        staging_vector_service=FakeVectorService(),
        document_repository=FakeDocumentRepository(),
        ingest_task_service=task_service,
    )
    monkeypatch.setattr(
        "app.application.training.training_knowledge_service.get_file_storage_service",
        lambda: storage_service,
    )

    response = service.upload_knowledge(
        file=FakeUploadFile(),
        source_type="lms_case",
        created_by="tester",
        model_mode="fast",
    )

    assert response.status == "duplicated"
    assert response.batch_id == "pending_batch"
    assert response.document_id == "pending_doc"
    assert response.duplicate_of == "pending_batch"
    assert response.chunk_count == 3
    assert repository.created_batches == []
    assert task_service.created == []
    assert storage_service.deleted_objects == [("pub", "training/doc_1/case.txt")]
