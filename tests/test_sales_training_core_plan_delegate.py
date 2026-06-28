"""销售训练核心外观方案服务委托测试。"""

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
    """记录核心外观是否把训练方案入口委托给方案服务。"""

    def __init__(self, *, repository):
        self.repository = repository
        self.calls = []

    def create_plan(self, request):
        self.calls.append(("create_plan", request))
        return "create-result"

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None):
        self.calls.append(("list_plans", page, page_size, keyword))
        return "list-result"

    def get_plan_detail(self, plan_id: str):
        self.calls.append(("get_plan_detail", plan_id))
        return "detail-result"

    def delete_plan(self, plan_id: str):
        self.calls.append(("delete_plan", plan_id))
        return "delete-result"

    def update_plan(self, plan_id: str, request):
        self.calls.append(("update_plan", plan_id, request))
        return "update-result"


class FakeRoleApplicationService:
    """记录核心外观是否把角色生成入口委托给角色应用服务。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def generate_supplement_questions(self, request):
        self.calls.append(("generate_supplement_questions", request))
        return "questions-result"

    def polish_scenario(self, request):
        self.calls.append(("polish_scenario", request))
        return "polish-result"

    def generate_role(self, request):
        self.calls.append(("generate_role", request))
        return "role-result"


def _patch_core_dependencies(monkeypatch):
    """替换核心外观的重依赖，专注验证委托边界。"""

    monkeypatch.setattr(sales_training_core, "VectorStoreService", FakeVectorService)
    monkeypatch.setattr(sales_training_core, "TrainingRepository", FakeRepository)
    monkeypatch.setattr(sales_training_core, "DocumentRepository", lambda store=None: FakeDocumentRepository())
    monkeypatch.setattr(sales_training_core, "TrainingKnowledgeService", FakeKnowledgeService)
    monkeypatch.setattr(sales_training_core, "TrainingPlanDomainService", FakePlanService)
    monkeypatch.setattr(sales_training_core, "TrainingRoleApplicationService", FakeRoleApplicationService)


def test_core_delegates_plan_methods(monkeypatch):
    """训练方案 CRUD 入口都应委托给 TrainingPlanDomainService。"""

    _patch_core_dependencies(monkeypatch)
    create_request = object()
    update_request = object()

    core_service = sales_training_core.V2SalesTrainingCoreService()

    assert core_service.create_plan(create_request) == "create-result"
    assert core_service.list_plans(page=0, page_size=999, keyword="客户") == "list-result"
    assert core_service.get_plan_detail("plan_1") == "detail-result"
    assert core_service.delete_plan("plan_1") == "delete-result"
    assert core_service.update_plan("plan_1", update_request) == "update-result"
    assert core_service.plan_service.calls == [
        ("create_plan", create_request),
        ("list_plans", 0, 999, "客户"),
        ("get_plan_detail", "plan_1"),
        ("delete_plan", "plan_1"),
        ("update_plan", "plan_1", update_request),
    ]


def test_core_delegates_role_methods(monkeypatch):
    """补充问题、场景润色和角色生成入口应委托给 TrainingRoleApplicationService。"""

    _patch_core_dependencies(monkeypatch)
    request = object()

    core_service = sales_training_core.V2SalesTrainingCoreService()

    assert core_service.generate_supplement_questions(request) == "questions-result"
    assert core_service.polish_scenario(request) == "polish-result"
    assert core_service.generate_role(request) == "role-result"
    assert core_service.role_application_service.calls == [
        ("generate_supplement_questions", request),
        ("polish_scenario", request),
        ("generate_role", request),
    ]
