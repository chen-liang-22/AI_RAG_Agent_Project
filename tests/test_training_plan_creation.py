"""销售训练方案创建测试。"""

from datetime import datetime
import json

from app.application.training.training_plan_domain_service import TrainingPlanDomainService
from app.application.training_support.schemas import TrainingPlanCreateRequest, TrainingPlanUpdateRequest


class FakeTrainingPlanRepository:
    """记录训练方案创建入参的仓储替身。"""

    def __init__(self):
        """初始化仓储调用记录。"""

        self.created_values = None
        self.updated_values = None
        self.plan_row = None

    def create_plan(self, **values):
        """记录创建参数并返回模拟数据库行。"""

        self.created_values = values
        self.plan_row = {
            "plan_id": "plan_empty",
            "plan_name": values["plan_name"],
            "trainee_id": "",
            "trainee_name": "",
            "profile_type": values.get("profile_type") or "",
            "trainee_json": "{}",
            "selected_fields_json": "{}",
            "scenario_description": values.get("scenario_description") or "",
            "extra_details": values.get("extra_details") or "",
            "model_mode": values.get("model_mode"),
            "active_profile_id": None,
            "active_setting_id": None,
            "role_status": "pending",
            "goal_status": "pending",
            "score_status": "pending",
            "created_at": datetime(2026, 7, 1, 10, 0, 0),
            "updated_at": datetime(2026, 7, 1, 10, 0, 0),
        }
        return self.plan_row

    @staticmethod
    def _json(value):
        """按真实仓储格式序列化 JSON 字段。"""

        return json.dumps(value, ensure_ascii=False)

    def get_plan(self, plan_id):
        """返回当前测试方案。"""

        if self.plan_row and self.plan_row["plan_id"] == plan_id:
            return self.plan_row
        return None

    def update_plan(self, plan_id, **values):
        """记录修改参数并返回合并后的模拟数据库行。"""

        self.updated_values = values
        self.plan_row = {
            **self.plan_row,
            **values,
            "updated_at": datetime(2026, 7, 1, 10, 5, 0),
        }
        return self.plan_row

    def get_role_profile(self, profile_id):
        """空方案不会关联角色画像。"""

        return None

    def get_goal_setting(self, setting_id):
        """空方案不会关联训练目标。"""

        return None


def test_create_plan_allows_name_only_without_default_profile_payload():
    """只输入训练名称创建方案时，不应偷偷写入默认学员或客户画像。"""

    repository = FakeTrainingPlanRepository()
    service = TrainingPlanDomainService(repository=repository)

    response = service.create_plan(TrainingPlanCreateRequest(plan_name="汪汪队训练4"))

    assert repository.created_values == {
        "plan_name": "汪汪队训练4",
        "trainee": {},
        "profile_type": "",
        "selected_fields": {},
        "scenario_description": "",
        "extra_details": "",
        "model_mode": None,
    }
    assert response.plan.plan_id == "plan_empty"
    assert response.trainee == {}
    assert response.selected_fields == {}
    assert response.scenario_description == ""


def test_update_empty_plan_keeps_pending_status_before_role_generated():
    """空方案补齐画像时，未生成过的角色、阶段和评分仍应保持待生成。"""

    repository = FakeTrainingPlanRepository()
    service = TrainingPlanDomainService(repository=repository)
    service.create_plan(TrainingPlanCreateRequest(plan_name="汪汪队训练4"))

    service.update_plan(
        "plan_empty",
        TrainingPlanUpdateRequest(
            trainee={
                "trainee_id": "trainee-001",
                "trainee_name": "销售学员",
                "position_role": "overseas_bd",
                "experience_level": "junior",
                "task_goal": "goal_junior",
                "weakness_tags": ["price_negotiation"],
                "student_portrait_other": "",
            },
            profile_type="overseas_bd",
            selected_fields={"客户类型": "C端"},
            scenario_description="客户担心成本和交付风险。",
            extra_details="",
            model_mode="high",
        ),
    )

    assert repository.updated_values["role_status"] == "pending"
    assert repository.updated_values["goal_status"] == "pending"
    assert repository.updated_values["score_status"] == "pending"
