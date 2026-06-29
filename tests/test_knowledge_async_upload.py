"""知识库异步确认入库测试。"""

from datetime import datetime

from api.schemas import KnowledgeUploadConfirmRequest
from app.application.knowledge_service import KnowledgeApplicationService


class FakePreviewFile:
    """模拟上传预览文件。"""

    filename = "faq.txt"
    file_md5 = "md5_1"


class FakeStoredFile:
    """模拟临时文件转正式文件结果。"""

    file_path = "minio://pub/documents/doc_1/faq.txt"
    file_md5 = "md5_1"
    file_size = 100
    bucket_name = "pub"
    object_name = "documents/doc_1/faq.txt"
    public_url = "http://localhost:9000/pub/documents/doc_1/faq.txt"


class FakeDocumentRepository:
    """知识库测试文件仓储。"""

    def __init__(self):
        self.created = None

    def find_active_document_by_md5(self, file_md5: str, *, collection_name: str | None = None):
        return None

    def create_document(self, **values):
        self.created = {
            **values,
            "version": 1,
            "chunk_count": 0,
            "created_at": datetime(2026, 1, 1, 10, 0, 0),
            "updated_at": datetime(2026, 1, 1, 10, 0, 0),
            "error_message": None,
        }
        return self.created


class FakeDictionaryRepository:
    """字典仓储测试替身。"""

    def list_items(self, dictionary_code: str):
        names = {
            "document_structure": ("text", "文本型"),
            "split_strategy": ("recursive", "递归切分"),
        }
        item_code, item_name = names[dictionary_code]
        return [{
            "dictionary_item_id": f"dict_{item_code}",
            "dictionary_code": dictionary_code,
            "dictionary_name": dictionary_code,
            "item_code": item_code,
            "item_name": item_name,
            "parent_item_id": None,
            "item_level": 1,
            "sort_order": 1,
            "enabled": 1,
            "description": "",
            "metadata_json": "{}",
        }]

    def normalize_code(self, dictionary_code: str, item_code: str):
        return item_code


class FakeTaskService:
    """记录知识库入库任务创建。"""

    def __init__(self):
        self.created = []

    def create_document_ingest_task(self, **values):
        self.created.append(values)
        return {
            "task_id": "task_doc_1",
            "task_status": "queued",
            "current_step": "queued",
            "progress": 5,
        }


def test_confirm_upload_creates_async_task_without_sync_index(monkeypatch):
    """确认入库应创建后台任务并快速返回。"""

    document_repository = FakeDocumentRepository()
    task_service = FakeTaskService()
    monkeypatch.setattr("app.application.knowledge_service._get_preview_file", lambda upload_id: FakePreviewFile())
    monkeypatch.setattr("app.application.knowledge_service._promote_preview_file", lambda upload_id, document_id: FakeStoredFile())

    def fail_if_sync_index(*args, **kwargs):
        raise AssertionError("不应该同步执行入库")

    monkeypatch.setattr("app.application.knowledge_service._index_document", fail_if_sync_index)
    service = KnowledgeApplicationService(
        document_repository=document_repository,
        dictionary_repository=FakeDictionaryRepository(),
        ingest_task_service=task_service,
    )

    response = service.confirm_upload(KnowledgeUploadConfirmRequest(
        upload_id="tmp_1",
        document_type="text",
        split_strategy="recursive",
        collection_name="agent",
    ))

    assert response.status == "queued"
    assert response.task_id == "task_doc_1"
    assert response.task_status == "queued"
    assert response.progress == 5
    assert document_repository.created["status"] == "uploaded"
    assert task_service.created[0]["document_id"] == document_repository.created["document_id"]
