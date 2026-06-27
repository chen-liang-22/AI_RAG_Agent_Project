"""V2 知识资产应用服务测试。"""

from app_v2.application.knowledge_service import KnowledgeApplicationService


class ExplodingKnowledgeStore:
    """如果 V2 知识服务继续调用旧 KnowledgeStore，测试应该立刻失败。"""

    def list_dictionary_items(self, *, dictionary_code=None):
        raise AssertionError("KnowledgeApplicationService 文件列表字典快照不应该继续走 KnowledgeStore")

    def normalize_dictionary_code(self, dictionary_code: str, item_code: str | None = None):
        return item_code or "default"

    def list_documents(self, *args, **kwargs):
        raise AssertionError("KnowledgeApplicationService 查询列表不应该继续调用旧 store.list_documents")

    def get_document(self, *args, **kwargs):
        raise AssertionError("KnowledgeApplicationService 查询详情不应该继续调用旧 store.get_document")

    def find_active_document_by_md5(self, *args, **kwargs):
        raise AssertionError("KnowledgeApplicationService 上传去重不应该继续调用旧 store.find_active_document_by_md5")


class FakeDocumentRepository:
    """测试用 V2 文档仓储，记录服务层是否真的通过 repository 查询文档。"""

    def __init__(self):
        self.list_include_training: bool | None = None
        self.requested_document_ids: list[str] = []
        self.document = {
            "document_id": "doc_repo",
            "filename": "机器人知识.txt",
            "file_path": "minio://knowledge/doc_repo.txt",
            "storage_type": "minio",
            "bucket_name": "knowledge",
            "object_name": "doc_repo.txt",
            "public_url": None,
            "file_type": "txt",
            "file_md5": "md5_repo",
            "file_size": 128,
            "status": "indexed",
            "version": 1,
            "chunk_count": 2,
            "collection_name": "agent",
            "document_type": "text",
            "split_strategy": "recursive",
            "created_at": "2026-06-26 10:00:00",
            "updated_at": "2026-06-26 10:01:00",
            "error_message": None,
        }

    def list_documents(self, *, include_training: bool = False):
        self.list_include_training = include_training
        return [self.document]

    def get_document(self, document_id: str):
        self.requested_document_ids.append(document_id)
        if document_id == "doc_repo":
            return self.document
        return None


class FakeDictionaryRepository:
    """测试用 V2 字典仓储，给文档响应归一化提供启用字典项。"""

    def __init__(self):
        self.calls: list[str | None] = []

    def list_items(self, dictionary_code: str | None = None) -> list[dict]:
        """模拟 dictionary_items 表按字典编码查询。"""

        self.calls.append(dictionary_code)
        if dictionary_code == "document_structure":
            return [
                {"item_code": "text", "enabled": 1},
                {"item_code": "qa", "enabled": 1},
            ]
        if dictionary_code == "split_strategy":
            return [
                {"item_code": "recursive", "enabled": 1},
                {"item_code": "qa_pair", "enabled": 1},
            ]
        return []


def test_knowledge_service_reads_documents_through_repository(monkeypatch):
    """知识资产应用服务的列表和详情读取应该走 V2 仓储。"""

    import api.services.common_services as common_services

    monkeypatch.setattr(
        common_services,
        "_get_knowledge_store",
        lambda: (_ for _ in ()).throw(AssertionError("文档响应不应该自己创建旧 KnowledgeStore")),
    )
    repository = FakeDocumentRepository()
    dictionary_repository = FakeDictionaryRepository()
    service = KnowledgeApplicationService(
        store=ExplodingKnowledgeStore(),
        document_repository=repository,
        dictionary_repository=dictionary_repository,
    )

    files = service.list_files(include_training=True)
    detail = service.get_file("doc_repo")

    assert repository.list_include_training is True
    assert repository.requested_document_ids == ["doc_repo"]
    assert dictionary_repository.calls[:2] == ["document_structure", "split_strategy"]
    assert [item.document_id for item in files] == ["doc_repo"]
    assert detail.document_id == "doc_repo"


def test_knowledge_service_reindex_all_reads_documents_through_repository(monkeypatch):
    """全量重建索引也应该从 V2 文档仓储读取文档列表。"""

    repository = FakeDocumentRepository()
    indexed_documents: list[str] = []

    class FakeVectorAdapter:
        vector_service = object()

        @staticmethod
        def recreate_collection(collection_name):
            assert collection_name == "agent"
            return FakeVectorAdapter()

    def fake_index_document(store, document, **kwargs):
        indexed_documents.append(document["document_id"])
        assert kwargs["collection_name"] == "agent"
        # 模拟索引服务更新后的文档记录，reindex_all 只关心 chunk_count。
        return {**document, "chunk_count": 9}

    import app_v2.application.knowledge_service as knowledge_service_module

    monkeypatch.setattr(knowledge_service_module, "VectorStoreAdapter", FakeVectorAdapter)
    monkeypatch.setattr(knowledge_service_module, "_index_document", fake_index_document)

    service = KnowledgeApplicationService(
        store=ExplodingKnowledgeStore(),
        document_repository=repository,
    )

    result = service.reindex_all()

    assert repository.list_include_training is False
    assert indexed_documents == ["doc_repo"]
    assert result.total == 1
    assert result.succeeded == 1
    assert result.failed == 0


def test_knowledge_service_confirm_upload_creates_and_indexes_through_repository(monkeypatch):
    """确认入库应该通过 V2 文档仓储创建文档，并把仓储传给索引服务更新状态。"""

    import app_v2.application.knowledge_service as knowledge_service_module
    from api.schemas import KnowledgeUploadConfirmRequest

    class PreviewFile:
        filename = "确认入库.txt"
        file_md5 = "md5_confirm"
        file_size = 456

    class StoredFile:
        file_path = "minio://knowledge/doc_confirm/确认入库.txt"
        file_md5 = "md5_confirm"
        file_size = 456
        bucket_name = "knowledge"
        object_name = "doc_confirm/确认入库.txt"
        public_url = None

    class WriteDocumentRepository(FakeDocumentRepository):
        def __init__(self):
            super().__init__()
            self.created_values = None

        def find_active_document_by_md5(self, file_md5: str, *, collection_name: str | None = None):
            assert file_md5 == "md5_confirm"
            assert collection_name == "agent"
            return None

        def create_document(self, **values):
            self.created_values = values
            self.document = {
                **self.document,
                **values,
                "version": 1,
                "chunk_count": 0,
                "created_at": "2026-06-26 10:00:00",
                "updated_at": "2026-06-26 10:00:00",
                "error_message": None,
            }
            return self.document

    repository = WriteDocumentRepository()
    indexed_store_objects: list[object] = []

    def fake_index_document(store, document, **kwargs):
        indexed_store_objects.append(store)
        assert str(document["document_id"]).isdigit()
        return {**document, "status": "indexed", "chunk_count": 4}

    monkeypatch.setattr(knowledge_service_module, "_get_preview_file", lambda upload_id: PreviewFile())
    monkeypatch.setattr(knowledge_service_module, "_promote_preview_file", lambda upload_id, document_id: StoredFile())
    monkeypatch.setattr(knowledge_service_module, "_delete_preview_file", lambda upload_id: None)
    monkeypatch.setattr(knowledge_service_module, "_index_document", fake_index_document)

    service = KnowledgeApplicationService(
        store=ExplodingKnowledgeStore(),
        document_repository=repository,
    )
    request = KnowledgeUploadConfirmRequest(
        upload_id="tmp_confirm",
        collection_name="agent",
        document_type="text",
        split_strategy="recursive",
    )

    result = service.confirm_upload(request)

    assert repository.created_values is not None
    assert repository.created_values["filename"] == "确认入库.txt"
    assert repository.created_values["file_md5"] == "md5_confirm"
    assert indexed_store_objects == [repository]
    assert result.status == "indexed"
    assert result.document.chunk_count == 4
