"""销售训练目标生成服务测试。"""

import json

from app.application.training.training_goal_setting_service import TrainingGoalSettingService


def _profile() -> dict:
    """构造数据库角色画像行，字段形态贴近真实 repository 返回结果。"""

    return {
        "role_profile_json": json.dumps(
            {
                "业务痛点": ["新人转化率低", "跟进过程缺少标准话术"],
                "challenge_strategy": ["持续追问 ROI", "要求给出落地路径"],
            },
            ensure_ascii=False,
        ),
        "hidden_profile_json": json.dumps(
            {
                "真实顾虑": ["担心上线周期长", "担心团队不配合"],
            },
            ensure_ascii=False,
        ),
        "retrieved_evidence_json": json.dumps(
            [{"content": "制造业客户案例"}, {"content": "新人转化训练案例"}],
            ensure_ascii=False,
        ),
        "scenario_description": "客户希望通过 AI 销售陪练提升新人转化率。",
    }


def test_goal_prompt_renders_profile_and_hidden_profile_json():
    """目标生成 prompt 必须包含角色画像和隐藏画像。"""

    prompt = TrainingGoalSettingService.goal_prompt(_profile())

    assert "新人转化率低" in prompt
    assert "跟进过程缺少标准话术" in prompt
    assert "担心上线周期长" in prompt
    assert "担心团队不配合" in prompt


def test_fallback_goal_uses_profile_complexity_and_default_scoring_rules():
    """目标生成兜底应根据画像复杂度估算轮数，并带上完整评分规则。"""

    fallback = TrainingGoalSettingService.fallback_goal(_profile())

    assert fallback["training_purpose"] == "需求挖掘"
    assert fallback["round_limit"] > 6
    assert fallback["stages"][0]["stage_name"] == "开放式需求挖掘"
    assert fallback["stages"][0]["stage_no"] == 1
    assert fallback["scoring_rules"]["total_score"] == 100
    assert fallback["scoring_rules"]["general_score"] == 40
    assert fallback["scoring_rules"]["stage_score"] == 60
    assert len(fallback["scoring_rules"]["stage_dimensions"]) == 3
