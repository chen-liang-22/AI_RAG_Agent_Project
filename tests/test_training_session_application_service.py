"""销售训练会话应用服务测试。"""

from app.application.training.session_service import TrainingSessionApplicationService
from app.application.training_support.schemas import TrainingSessionStartRequest


class FakeCoreService:
    """记录应用服务是否把会话创建请求委托给核心服务。"""

    def __init__(self):
        self.start_request = None

    def start_session(self, request):
        self.start_request = request
        return "session-started"


def test_start_session_does_not_require_plan_id():
    """开始训练请求没有 plan_id 字段时，应用服务日志不能打断真实接口。"""

    core_service = FakeCoreService()
    service = TrainingSessionApplicationService(core_service=core_service)
    request = TrainingSessionStartRequest(
        profile_id="profile_1",
        setting_id="setting_1",
        trainee_id="trainee_1",
        response_mode="stream",
        model_mode="low",
    )

    result = service.start_session(request)

    assert result == "session-started"
    assert core_service.start_request is request
