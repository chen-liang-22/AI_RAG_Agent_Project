"""销售训练评分规则服务。

这个模块承接“评分规则归一化、兜底评分、等级换算”等纯逻辑。
它不直接访问数据库，也不调用大模型，方便后续单独测试评分规则。
"""

from __future__ import annotations

import json
from typing import Any


class TrainingScoreService:
    """销售训练评分纯逻辑服务。"""

    @classmethod
    def normalize_scoring_rules(
            cls,
            raw_rules: Any,
            stages: list[dict[str, Any]],
            profile: dict[str, Any],
    ) -> dict[str, Any]:
        """归一化评分规则，保证总分始终是 100。

        通用能力固定 40 分，阶段能力固定 60 分；如果模型输出的阶段评分维度不完整，
        就回退到后端默认三维度评分规则。
        """

        rules = raw_rules if isinstance(raw_rules, dict) else {}
        default_rules = cls.default_scoring_rules(stages=stages, profile=profile)
        stage_dimensions = rules.get("stage_dimensions")
        if not isinstance(stage_dimensions, list) or not stage_dimensions:
            stage_dimensions = default_rules["stage_dimensions"]
        else:
            valid_dimensions = [item for item in stage_dimensions if isinstance(item, dict)]
            has_enough_dimensions = len(valid_dimensions) >= 3
            has_enough_points = all(
                isinstance(item.get("points"), list) and len(item.get("points") or []) >= 3
                for item in valid_dimensions
            )
            if not has_enough_dimensions or not has_enough_points:
                stage_dimensions = default_rules["stage_dimensions"]
        normalized_stage_dimensions = cls.normalize_dimension_scores(stage_dimensions, total_score=60)
        return {
            "total_score": 100,
            "general_score": 40,
            "stage_score": 60,
            "general_dimensions": default_rules["general_dimensions"],
            "stage_dimensions": normalized_stage_dimensions,
            "review_mode": "ai_auto",
            "formula": "总分 = 通用能力得分 + 阶段能力得分 - 扣分；一期暂不启用违规词扣分",
        }

    @classmethod
    def default_scoring_rules(
            cls,
            *,
            stages: list[dict[str, Any]] | None = None,
            profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """返回默认评分规则。

        通用能力严格固定 40 分；阶段能力 60 分在模型失败时按开放式训练兜底拆分。
        """

        role_profile = cls.load_json((profile or {}).get("role_profile_json"), {}) if profile else {}
        stage = (stages or [{}])[0] if stages else {}
        core_goal = str(stage.get("core_goal") or "围绕客户痛点推进有效沟通")
        customer_focus = role_profile.get("业务痛点") or role_profile.get("business_pain_points") or []
        focus_text = "、".join(str(item) for item in customer_focus[:2]) if isinstance(customer_focus, list) else str(customer_focus)
        return {
            "total_score": 100,
            "general_score": 40,
            "stage_score": 60,
            "general_dimensions": [
                {
                    "dimension_name": "内容质量",
                    "score": 20,
                    "points": [
                        {"point_name": "信息准确性", "score": 10, "description": "回答不编造事实，能基于已知客户信息和训练知识表达。"},
                        {"point_name": "需求理解与回应", "score": 5, "description": "能承接客户问题，不答非所问。"},
                        {"point_name": "价值传递", "score": 5, "description": "能把方案价值和客户痛点连接起来。"},
                    ],
                },
                {
                    "dimension_name": "语言表达",
                    "score": 10,
                    "points": [
                        {"point_name": "流利度", "score": 4, "description": "表达自然顺畅。"},
                        {"point_name": "专业术语使用", "score": 3, "description": "术语准确，不过度堆砌。"},
                        {"point_name": "逻辑清晰度", "score": 3, "description": "先回应问题，再给理由和下一步。"},
                    ],
                },
                {
                    "dimension_name": "互动与态度",
                    "score": 10,
                    "points": [
                        {"point_name": "倾听与承接", "score": 4, "description": "能接住客户情绪和顾虑。"},
                        {"point_name": "礼貌与亲和力", "score": 3, "description": "沟通态度专业、尊重客户。"},
                        {"point_name": "主动引导", "score": 3, "description": "能用问题推进下一步沟通。"},
                    ],
                },
            ],
            "stage_dimensions": [
                {
                    "dimension_name": "需求挖掘与痛点确认",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "背景追问", "score": 7, "description": "能围绕行业、规模、现有流程等背景连续追问，而不是直接讲方案。"},
                        {"point_name": "痛点定位", "score": 7, "description": f"能识别并复述客户真实痛点，重点关注：{focus_text or '投入产出、风险和落地成本'}。"},
                        {"point_name": "需求确认", "score": 6, "description": "能向客户确认优先级、影响范围和是否愿意继续沟通。"},
                    ],
                },
                {
                    "dimension_name": "价值呈现与证据支撑",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "价值匹配", "score": 7, "description": "能把方案价值与客户已经表达的痛点建立清晰连接。"},
                        {"point_name": "证据提供", "score": 7, "description": "能引用案例、数据、流程或知识库事实支撑表达，避免空泛承诺。"},
                        {"point_name": "风险降低", "score": 6, "description": "能解释落地方式、验证路径或试点方式，降低客户决策顾虑。"},
                    ],
                },
                {
                    "dimension_name": "异议处理与推进动作",
                    "score": 20,
                    "core_goal": core_goal,
                    "points": [
                        {"point_name": "异议承接", "score": 7, "description": "面对价格、风险、交付等异议时先承接再回应，不回避客户质疑。"},
                        {"point_name": "针对回应", "score": 7, "description": "能根据客户具体异议给出对应解释，不使用模板化套话。"},
                        {"point_name": "下一步推进", "score": 6, "description": "能争取客户继续沟通、试点、提供资料或约定下一次联系。"},
                    ],
                },
            ],
            "review_mode": "ai_auto",
            "formula": "总分 = 通用能力得分 + 阶段能力得分 - 扣分；一期暂不启用违规词扣分",
        }

    @staticmethod
    def normalize_dimension_scores(dimensions: list[Any], *, total_score: int) -> list[dict[str, Any]]:
        """按总分归一化评分维度。"""

        normalized: list[dict[str, Any]] = []
        source_dimensions = [item for item in dimensions if isinstance(item, dict)]
        if not source_dimensions:
            return []
        raw_total = sum(max(0, int(item.get("score") or 0)) for item in source_dimensions) or total_score
        allocated = 0
        for index, item in enumerate(source_dimensions):
            score = int(round(max(0, int(item.get("score") or 0)) * total_score / raw_total))
            if index == len(source_dimensions) - 1:
                score = total_score - allocated
            allocated += score
            points = item.get("points") if isinstance(item.get("points"), list) else []
            normalized.append({
                "dimension_name": str(item.get("dimension_name") or item.get("stage_name") or "阶段评分"),
                "score": max(0, score),
                "core_goal": item.get("core_goal") or "",
                "points": points,
            })
        return normalized

    @staticmethod
    def fallback_score(turns: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict:
        """评分模型失败时的兜底评分。"""

        trainee_turns = [item for item in turns if item["role"] == "trainee"]
        base_score = 72 + min(10, len(trainee_turns) * 2)
        return {
            "total_score": min(88, base_score),
            "general_score": 32,
            "stage_score": max(0, min(56, base_score - 32)),
            "penalty_score": 0,
            "hit_points": ["完成了基本沟通", "能围绕客户问题继续回应"],
            "missing_points": ["需要更多追问客户真实顾虑", "需要引用更具体案例"],
            "wrong_points": [],
            "evidence_refs": [{"type": "dialogue", "round_no": item["round_no"]} for item in trainee_turns[:3]],
            "improvement_advice": "下一次训练重点加强需求挖掘和案例化表达。",
            "reference_script": "可以先确认客户当前卡点，再用同类客户案例降低风险感。",
            "next_training_plan": ["需求挖掘专项", "异议处理专项"],
        }

    @staticmethod
    def score_level(score: int) -> str:
        """把最终得分转换成中文等级。"""

        if score > 90:
            return "优秀"
        if score > 80:
            return "良好"
        if score >= 75:
            return "及格"
        if score >= 60:
            return "待观察"
        return "不及格"

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
