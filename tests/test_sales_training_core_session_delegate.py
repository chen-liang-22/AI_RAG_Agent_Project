"""销售训练核心外观会话基础服务委托测试。"""

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
    """资料服务占位，本测试不验证资料流程。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePlanService:
    """方案服务占位，本测试不验证方案流程。"""

    def __init__(self, *, repository):
        self.repository = repository


class FakeTrainingSessionService:
    """记录核心外观是否把完整会话入口委托给训练会话聚合服务。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def start_session(self, request):
        self.calls.append(("start_session", request))
        return "start-result"

    def list_sessions(self, *, page: int, page_size: int, trainee_id: str | None = None):
        self.calls.append(("list_sessions", page, page_size, trainee_id))
        return "list-result"

    def get_session_detail(self, session_id: str):
        self.calls.append(("get_session_detail", session_id))
        return "detail-result"

    def submit_turn(self, session_id: str, request):
        self.calls.append(("submit_turn", session_id, request))
        return "submit-result"

    def stream_turn(self, session_id: str, request):
        self.calls.append(("stream_turn", session_id, request))
        return iter(["event: done\ndata: {}\n\n"])

    def final_score(self, session_id: str, model_mode: str | None = None):
        self.calls.append(("final_score", session_id, model_mode))
        return "score-result"


def _patch_core_dependencies(monkeypatch):
    """替换核心外观的重依赖，专注验证委托边界。"""

    monkeypatch.setattr(sales_training_core, "VectorStoreService", FakeVectorService)
    monkeypatch.setattr(sales_training_core, "TrainingRepository", FakeRepository)
    monkeypatch.setattr(sales_training_core, "DocumentRepository", lambda store=None: FakeDocumentRepository())
    monkeypatch.setattr(sales_training_core, "TrainingKnowledgeService", FakeKnowledgeService)
    monkeypatch.setattr(sales_training_core, "TrainingPlanDomainService", FakePlanService)
    monkeypatch.setattr(sales_training_core, "TrainingSessionService", FakeTrainingSessionService)


def test_core_delegates_session_methods(monkeypatch):
    """训练会话入口应统一委托给 TrainingSessionService。"""

    _patch_core_dependencies(monkeypatch)
    start_request = object()
    turn_request = object()

    core_service = sales_training_core.V2SalesTrainingCoreService()

    assert core_service.start_session(start_request) == "start-result"
    assert core_service.list_sessions(page=0, page_size=999, trainee_id="stu_1") == "list-result"
    assert core_service.get_session_detail("session_1") == "detail-result"
    assert core_service.submit_turn("session_1", turn_request) == "submit-result"
    assert list(core_service.stream_turn("session_1", turn_request)) == ["event: done\ndata: {}\n\n"]
    assert core_service.final_score("session_1", model_mode="fast") == "score-result"
    assert core_service.training_session_service.calls == [
        ("start_session", start_request),
        ("list_sessions", 0, 999, "stu_1"),
        ("get_session_detail", "session_1"),
        ("submit_turn", "session_1", turn_request),
        ("stream_turn", "session_1", turn_request),
        ("final_score", "session_1", "fast"),
    ]
