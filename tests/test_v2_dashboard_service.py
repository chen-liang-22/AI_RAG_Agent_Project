"""V2 首页驾驶舱服务测试。"""

from app_v2.application.dashboard_service import DashboardApplicationService


class FakeDictionaryRepository:
    """测试用字典仓储，同时提供健康状态和文档归一化字典。"""

    def __init__(self):
        self.list_calls: list[str | None] = []

    def normalize_code(self, dictionary_code: str, value: str | None = None) -> str:
        return value or "ok"

    def list_items(self, dictionary_code: str | None = None) -> list[dict]:
        self.list_calls.append(dictionary_code)
        if dictionary_code == "document_structure":
            return [{"item_code": "text", "enabled": 1}]
        if dictionary_code == "split_strategy":
            return [{"item_code": "recursive", "enabled": 1}]
        return []


class ExplodingStore:
    """如果首页概览继续用旧 store 构建字典快照，测试应该失败。"""

    def list_dictionary_items(self, *args, **kwargs):
        raise AssertionError("首页知识库概览不应该继续通过旧存储查询字典")


class FakeDocumentRepository:
    def __init__(self):
        self.store = ExplodingStore()

    def list_documents(self, *, include_training: bool = False):
        assert include_training is True
        return [
            {
                "document_id": "doc_home",
                "filename": "首页知识.txt",
                "file_path": "minio://knowledge/doc_home.txt",
                "storage_type": "minio",
                "bucket_name": "knowledge",
                "object_name": "doc_home.txt",
                "public_url": None,
                "file_type": "txt",
                "file_md5": "md5_home",
                "file_size": 10,
                "status": "indexed",
                "version": 1,
                "chunk_count": 1,
                "collection_name": "agent",
                "document_type": "text",
                "split_strategy": "recursive",
                "created_at": "2026-06-26 10:00:00",
                "updated_at": "2026-06-26 10:01:00",
                "error_message": None,
            }
        ]


class FakeConversationRepository:
    def list_conversations(self, *, page: int, page_size: int, keyword: str | None = None):
        return [], 0


class FakeTrainingRepository:
    def list_batches(self, *, page: int, page_size: int, status: str | None = None, keyword: str | None = None):
        return [], 0

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None):
        return [], 0

    def list_sessions(self, *, page: int, page_size: int, trainee_id: str | None = None):
        return [], 0


class DashboardServiceForTest(DashboardApplicationService):
    """跳过真实仓储构造，方便单测只验证首页聚合逻辑。"""

    def __init__(self):
        self.dictionary_repository = FakeDictionaryRepository()
        self.document_repository = FakeDocumentRepository()
        self.conversation_repository = FakeConversationRepository()
        self.training_repository = FakeTrainingRepository()

    def health(self):
        from api.schemas import HealthResponse

        return HealthResponse(
            status="ok",
            qdrant="ok",
            redis="ok",
            collection_name="agent",
            collections=["agent"],
            collection_points={"agent": 1},
        )


def test_dashboard_overview_uses_v2_dictionary_repository_for_documents():
    """首页知识库概览的文档响应应该使用 V2 字典仓储快照。"""

    service = DashboardServiceForTest()
    overview = service.overview()

    assert [item.document_id for item in overview.knowledge_files] == ["doc_home"]
    assert service.dictionary_repository.list_calls == ["document_structure", "split_strategy"]
