"""销售训练核心外观资料服务委托测试。"""

from app.application.training import sales_training_core


class FakeRepository:
    """核心外观测试仓储，占位避免初始化真实 MySQL 仓储。"""


class FakeDocumentRepository:
    """核心外观测试文档仓储，占位避免初始化真实文件台账。"""


class FakeVectorService:
    """核心外观测试向量服务，占位避免连接真实 Qdrant。"""

    def __init__(self, *, collection_name: str):
        self.collection_name = collection_name


class FakeKnowledgeService:
    """记录核心外观是否把资料管理入口委托给新服务。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def list_batches(self, *, page: int, page_size: int):
        self.calls.append(("list_batches", page, page_size))
        return "list-result"

    def preview_batch(self, batch_id: str, *, max_chars: int):
        self.calls.append(("preview_batch", batch_id, max_chars))
        return "preview-result"

    def delete_batch(self, batch_id: str):
        self.calls.append(("delete_batch", batch_id))
        return "delete-result"

    def list_batch_versions(self, batch_id: str):
        self.calls.append(("list_batch_versions", batch_id))
        return "versions-result"

    def list_chunks(self, batch_id: str):
        self.calls.append(("list_chunks", batch_id))
        return "chunks-result"


def _patch_core_dependencies(monkeypatch):
    """替换核心外观的重依赖，专注验证委托边界。"""

    monkeypatch.setattr(sales_training_core, "VectorStoreService", FakeVectorService)
    monkeypatch.setattr(sales_training_core, "TrainingRepository", FakeRepository)
    monkeypatch.setattr(sales_training_core, "DocumentRepository", lambda store=None: FakeDocumentRepository())
    monkeypatch.setattr(sales_training_core, "TrainingKnowledgeService", FakeKnowledgeService)


def test_core_delegates_knowledge_management_methods(monkeypatch):
    """资料列表、预览、删除、版本和切片入口都应委托给 TrainingKnowledgeService。"""

    _patch_core_dependencies(monkeypatch)

    core_service = sales_training_core.V2SalesTrainingCoreService()

    assert core_service.list_batches(page=2, page_size=20) == "list-result"
    assert core_service.preview_batch("batch_1", max_chars=1000) == "preview-result"
    assert core_service.delete_batch("batch_1") == "delete-result"
    assert core_service.list_batch_versions("batch_1") == "versions-result"
    assert core_service.list_chunks("batch_1") == "chunks-result"
    assert core_service.knowledge_service.calls == [
        ("list_batches", 2, 20),
        ("preview_batch", "batch_1", 1000),
        ("delete_batch", "batch_1"),
        ("list_batch_versions", "batch_1"),
        ("list_chunks", "batch_1"),
    ]
