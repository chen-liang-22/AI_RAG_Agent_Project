"""销售训练资料服务测试。"""

import json
from datetime import datetime

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

    def list_batches_in_version_group(self, version_group_id: str):
        """返回同一版本组的批次。"""

        return [_batch(batch_id="batch_1", version_group_id=version_group_id)]


class FakeVectorService:
    """记录向量库删除动作。"""

    def __init__(self):
        self.deleted = []
        self.listed_metadata = []

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


class FakeDocumentRepository:
    """测试用文档仓储。"""

    def get_document(self, document_id: str):
        """测试批次已经带有联表字段，不需要额外查文档。"""

        return None


def _service(repository=None, vector_service=None, staging_vector_service=None) -> TrainingKnowledgeService:
    """构造训练资料服务。"""

    return TrainingKnowledgeService(
        repository=repository or FakeTrainingRepository(),
        vector_service=vector_service or FakeVectorService(),
        staging_vector_service=staging_vector_service or FakeVectorService(),
        document_repository=FakeDocumentRepository(),
    )


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
