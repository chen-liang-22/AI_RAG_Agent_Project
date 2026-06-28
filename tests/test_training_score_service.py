"""销售训练评分服务测试。"""

import json

from app.application.training.training_score_service import TrainingScoreService


def test_score_level_maps_score_to_chinese_level():
    """评分等级边界必须稳定，避免前端展示结果漂移。"""

    assert TrainingScoreService.score_level(91) == "优秀"
    assert TrainingScoreService.score_level(81) == "良好"
    assert TrainingScoreService.score_level(75) == "及格"
    assert TrainingScoreService.score_level(60) == "待观察"
    assert TrainingScoreService.score_level(59) == "不及格"


def test_normalize_scoring_rules_falls_back_when_stage_dimensions_are_too_thin():
    """LLM 给出的阶段评分维度太少时，必须回退到后端三维度规则。"""

    profile = {
        "role_profile_json": json.dumps(
            {"业务痛点": ["报价效率低", "客户跟进慢"]},
            ensure_ascii=False,
        )
    }
    stages = [{"stage_name": "需求挖掘", "core_goal": "确认客户真实痛点"}]
    raw_rules = {
        "stage_dimensions": [
            {"dimension_name": "单一维度", "score": 60, "points": [{"point_name": "单点", "score": 60}]}
        ]
    }

    rules = TrainingScoreService.normalize_scoring_rules(raw_rules, stages, profile)

    assert rules["total_score"] == 100
    assert rules["general_score"] == 40
    assert rules["stage_score"] == 60
    assert len(rules["general_dimensions"]) == 3
    assert len(rules["stage_dimensions"]) == 3
    assert sum(item["score"] for item in rules["stage_dimensions"]) == 60
    assert "报价效率低" in rules["stage_dimensions"][0]["points"][1]["description"]


def test_fallback_score_uses_trainee_turns_and_caps_score():
    """兜底评分只统计学员轮次，并把总分限制在安全范围内。"""

    turns = [
        {"role": "customer", "round_no": 1, "content": "你们有什么价值？"},
        {"role": "trainee", "round_no": 1, "content": "我先了解下您的场景。"},
        {"role": "trainee", "round_no": 2, "content": "我们可以先做试点。"},
        {"role": "trainee", "round_no": 3, "content": "我给您发案例。"},
        {"role": "trainee", "round_no": 4, "content": "再约一次沟通。"},
        {"role": "trainee", "round_no": 5, "content": "确认下一步。"},
        {"role": "trainee", "round_no": 6, "content": "补充 ROI。"},
    ]

    fallback = TrainingScoreService.fallback_score(turns, evidence=[])

    assert fallback["total_score"] == 82
    assert fallback["general_score"] == 32
    assert fallback["stage_score"] == 50
    assert fallback["penalty_score"] == 0
    assert fallback["evidence_refs"] == [
        {"type": "dialogue", "round_no": 1},
        {"type": "dialogue", "round_no": 2},
        {"type": "dialogue", "round_no": 3},
    ]
