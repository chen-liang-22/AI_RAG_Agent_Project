"""销售训练目标生成纯逻辑服务。

这个模块承接“开放式训练目标和评分设置生成”里的本地逻辑：
- 渲染训练目标生成提示词；
- 在 LLM 不可用时生成兜底训练阶段、动态轮数和默认评分规则。

它不访问数据库、不调用模型，核心服务只需要负责读取角色画像、调用模型和保存结果。
"""

from __future__ import annotations

import json
from typing import Any

from app.application.training.training_score_service import TrainingScoreService
from core.utils.prompt_manager import prompt_manager


class TrainingGoalSettingService:
    """销售训练目标生成纯逻辑服务。"""

    @classmethod
    def goal_prompt(cls, profile: dict[str, Any]) -> str:
        """构造开放式训练目标和评分规则生成提示词。"""

        role_profile = cls.load_json(profile.get("role_profile_json"), {})
        hidden_profile = cls.load_json(profile.get("hidden_profile_json"), {})
        return prompt_manager.render(
            "training.goal_setting.user",
            role_profile_json=json.dumps(role_profile, ensure_ascii=False, indent=2),
            hidden_profile_json=json.dumps(hidden_profile, ensure_ascii=False, indent=2),
        )

    @classmethod
    def fallback_goal(cls, profile: dict[str, Any]) -> dict:
        """训练目标生成失败时的本地兜底结果。"""

        role_profile = cls.load_json(profile.get("role_profile_json"), {})
        hidden_profile = cls.load_json(profile.get("hidden_profile_json"), {})
        evidence = cls.load_json(profile.get("retrieved_evidence_json"), [])
        scenario_text = str(profile.get("scenario_description") or "")
        role_complexity = len(role_profile.get("business_pain_points") or []) + len(
            role_profile.get("challenge_strategy") or []
        )
        concern_complexity = len(hidden_profile.get("real_concerns") or [])
        # LLM 正常时会直接给出 round_limit；兜底时按场景复杂度估算，避免一期训练轮数退化成固定值。
        estimated_round_limit = 6 + min(
            8,
            role_complexity + concern_complexity + len(evidence) // 2 + len(scenario_text) // 120,
        )
        stages = [
            {
                "stage_no": 1,
                "stage_name": "开放式需求挖掘",
                "core_goal": "通过自然沟通获取客户痛点、顾虑和下一步意向。",
                "success_conditions": ["客户说出至少一个具体痛点", "客户愿意继续了解方案"],
                "failure_conditions": ["客户明确拒绝继续沟通", "学员连续多轮没有回应客户关切"],
            }
        ]
        return {
            "training_purpose": "需求挖掘",
            "round_limit": estimated_round_limit,
            "stages": stages,
            "scoring_rules": TrainingScoreService.default_scoring_rules(stages=stages, profile=profile),
        }

    @staticmethod
    def load_json(value: Any, default: Any) -> Any:
        """读取 JSON 字段，兼容数据库字符串和已解析对象。"""

        if value is None or value == "":
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
