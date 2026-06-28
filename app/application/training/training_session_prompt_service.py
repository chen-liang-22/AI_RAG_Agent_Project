"""销售训练会话 Prompt 纯逻辑服务。

这个模块承接训练会话中的提示词和兜底话术：
- AI 客户开场白提示词；
- 每轮 AI 客户回复提示词；
- 最终评分报告提示词；
- 会话文本拼接和模型失败兜底话术。

它不访问数据库、不调用模型，调用方需要把已查询到的 profile、setting、turns
和 evidence 显式传进来。
"""

from __future__ import annotations

import json
from typing import Any

from app.application.training.training_score_service import TrainingScoreService
from core.utils.prompt_manager import prompt_manager


class TrainingSessionPromptService:
    """销售训练会话 Prompt 纯逻辑服务。"""

    @classmethod
    def opening_prompt(cls, profile: dict[str, Any], setting: dict[str, Any]) -> str:
        """构造 AI 客户开场白提示词。"""

        role_profile = cls.load_json(profile.get("role_profile_json"), {})
        hidden_profile = cls.load_json(profile.get("hidden_profile_json"), {})
        stages = cls.load_json(setting.get("stages_json"), [])
        return prompt_manager.render(
            "training.opening_message.user",
            role_profile_json=json.dumps(role_profile, ensure_ascii=False, indent=2),
            hidden_profile_json=json.dumps(hidden_profile, ensure_ascii=False, indent=2),
            stages_json=json.dumps(stages, ensure_ascii=False, indent=2),
        )

    @classmethod
    def customer_prompt(
            cls,
            profile: dict[str, Any],
            setting: dict[str, Any],
            *,
            turns: list[dict[str, Any]],
            trainee_message: str,
            evidence: list[dict[str, Any]],
    ) -> str:
        """构造每轮 AI 客户回复提示词。"""

        role_profile = cls.load_json(profile.get("role_profile_json"), {})
        hidden_profile = cls.load_json(profile.get("hidden_profile_json"), {})
        stages = cls.load_json(setting.get("stages_json"), [])
        return prompt_manager.render(
            "training.customer_reply.user",
            role_profile_json=json.dumps(role_profile, ensure_ascii=False, indent=2),
            hidden_profile_json=json.dumps(hidden_profile, ensure_ascii=False, indent=2),
            stages_json=json.dumps(stages, ensure_ascii=False, indent=2),
            recent_turns_json=json.dumps(
                [{"role": item["role"], "content": item["content"]} for item in turns[-10:]],
                ensure_ascii=False,
                indent=2,
            ),
            trainee_message=trainee_message,
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        )

    @classmethod
    def score_prompt(
            cls,
            profile: dict[str, Any],
            setting: dict[str, Any],
            *,
            turns: list[dict[str, Any]],
            evidence: list[dict[str, Any]],
    ) -> str:
        """构造最终评分报告提示词。"""

        scoring_rules = cls.load_json(
            setting.get("scoring_rules_json"),
            TrainingScoreService.default_scoring_rules(),
        )
        return prompt_manager.render(
            "training.score_report.user",
            role_profile_json=profile.get("role_profile_json"),
            stages_json=setting.get("stages_json"),
            scoring_rules_json=json.dumps(scoring_rules, ensure_ascii=False, indent=2),
            conversation_json=json.dumps(
                [{"round_no": item["round_no"], "role": item["role"], "content": item["content"]} for item in turns],
                ensure_ascii=False,
                indent=2,
            ),
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def conversation_text(turns: list[dict[str, Any]]) -> str:
        """把训练对话轮次拼成评分和证据检索使用的纯文本。"""

        return "\n".join(f"{item['role']}：{item['content']}" for item in turns)

    @staticmethod
    def fallback_customer_reply(evidence: list[dict[str, Any]]) -> str:
        """AI 客户回复失败时的兜底话术。"""

        if evidence:
            return "你说的方向我能理解，不过我更关心实际效果和投入风险。你能结合类似客户案例，具体说说为什么这个方案适合我吗？"
        return "我先听听你的思路，但我比较关注投入产出和落地风险，你别只讲概念。"

    @classmethod
    def fallback_opening_message(cls, profile: dict[str, Any]) -> str:
        """AI 客户开场白失败时的兜底话术。"""

        role_profile = cls.load_json(profile.get("role_profile_json"), {})
        position = role_profile.get("position") or role_profile.get("职位") or "业务负责人"
        pain_points = role_profile.get("business_pain_points") or role_profile.get("业务痛点") or ["投入产出和落地风险"]
        first_pain = str(pain_points[0]) if pain_points else "投入产出和落地风险"
        return f"我是这边的{position}。你可以先简单讲讲方案，不过我更关心{first_pain}，如果只是概念性的介绍，可能很难推动内部继续评估。"

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
