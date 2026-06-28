"""销售训练角色生成服务测试。"""

from app.application.training.training_role_service import TrainingRoleService
from app.application.training_support.schemas import (
    RoleGenerateRequest,
    ScenarioPolishRequest,
    TraineeProfileRequest,
)


def _role_request() -> RoleGenerateRequest:
    """构造稳定的角色生成请求，避免每个测试重复铺 DTO 字段。"""

    return RoleGenerateRequest(
        plan_id="plan_1",
        trainee=TraineeProfileRequest(
            trainee_id="trainee_1",
            trainee_name="张三",
            weakness_tags=["价格异议", "需求挖掘"],
        ),
        profile_type="overseas_bd",
        selected_fields={
            "行业": "制造业",
            "客户类型": "C端客户",
            "合作阶段": "初次沟通",
            "价格敏感度": "高",
        },
        scenario_description="客户想了解销售陪练系统能否提升新人转化率。",
        extra_details="客户担心上线周期长，团队不愿意配合。",
        model_mode="fast",
    )


def test_build_role_query_collects_profile_scenario_and_trainee_context():
    """角色检索 query 必须包含画像、场景、补充信息和学员短板。"""

    query = TrainingRoleService.build_role_query(_role_request())

    assert "overseas_bd" in query
    assert "销售陪练系统" in query
    assert "上线周期长" in query
    assert "价格异议 需求挖掘" in query
    assert '"行业": "制造业"' in query
    assert '"客户类型": "C端客户"' in query


def test_normalize_supplement_questions_fills_missing_text_and_options():
    """LLM 输出不完整时，补充题仍要补齐题干和四个选项。"""

    raw_questions = [
        {
            "question": "",
            "options": [
                {"text": "先看案例"},
                "再比较价格",
            ],
        }
    ]

    questions = TrainingRoleService.normalize_supplement_questions(raw_questions, _role_request())

    assert len(questions) == 1
    assert questions[0].question_id == "q1"
    assert questions[0].question_no == 1
    assert questions[0].question
    assert len(questions[0].options) == 4
    assert [option.option_code for option in questions[0].options] == ["A", "B", "C", "D"]
    assert questions[0].options[0].option_text == "先看案例"
    assert questions[0].options[1].option_text == "再比较价格"


def test_fallback_polished_scenario_keeps_selected_fields_and_extra_details():
    """场景润色兜底不能丢掉用户已经输入的画像字段和补充要求。"""

    request = ScenarioPolishRequest(
        profile_type="overseas_bd",
        selected_fields={"客户类型": "海外BD", "服务内容": "AI销售陪练"},
        scenario_description="客户希望提高新人销售能力。",
        extra_details="强调客户预算有限。",
    )

    polished = TrainingRoleService.fallback_polished_scenario(request)

    assert "overseas_bd" in polished
    assert "客户类型：海外BD" in polished
    assert "服务内容：AI销售陪练" in polished
    assert "客户希望提高新人销售能力" in polished
    assert "预算有限" in polished


def test_fallback_role_returns_frontend_required_sections():
    """角色生成兜底结果必须包含可见画像、隐藏画像、扮演画像和确认卡片。"""

    fallback = TrainingRoleService.fallback_role(
        _role_request(),
        evidence=[{"content": "同类制造业客户关注新人转化率和上线周期。"}],
    )

    assert set(fallback) == {"visible_profile", "hidden_profile", "role_profile", "role_confirm_card"}
    assert fallback["visible_profile"]["角色名称"] == "C端客户客户"
    assert fallback["visible_profile"]["身份"] == "制造业｜业务负责人"
    assert "初次沟通" in fallback["visible_profile"]["角色摘要"]
    assert "上线周期长" in fallback["role_profile"]["业务痛点"][0]
    assert "价格异议、需求挖掘" in fallback["role_profile"]["挑战策略"][0]
    assert fallback["hidden_profile"]["真实顾虑"]
    assert fallback["role_confirm_card"]["角色名称"] == "C端客户客户"
