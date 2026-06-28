"""销售训练会话 Prompt 服务测试。"""

import json

from app.application.training.training_session_prompt_service import TrainingSessionPromptService


def _profile() -> dict:
    """构造角色画像数据库行。"""

    return {
        "role_profile_json": json.dumps(
            {
                "职位": "采购负责人",
                "business_pain_points": ["新人转化率低"],
                "角色简介": "谨慎型制造业客户",
            },
            ensure_ascii=False,
        ),
        "hidden_profile_json": json.dumps({"真实顾虑": ["担心上线周期长"]}, ensure_ascii=False),
    }


def _setting() -> dict:
    """构造训练设置数据库行。"""

    return {
        "stages_json": json.dumps(
            [{"stage_no": 1, "stage_name": "需求挖掘", "core_goal": "确认真实顾虑"}],
            ensure_ascii=False,
        ),
        "scoring_rules_json": json.dumps({"total_score": 100, "stage_dimensions": []}, ensure_ascii=False),
    }


def test_opening_prompt_contains_role_hidden_profile_and_stage():
    """开场白 prompt 必须包含角色、隐藏顾虑和训练阶段。"""

    prompt = TrainingSessionPromptService.opening_prompt(_profile(), _setting())

    assert "采购负责人" in prompt
    assert "担心上线周期长" in prompt
    assert "需求挖掘" in prompt


def test_customer_prompt_contains_recent_turns_message_and_evidence():
    """客户回复 prompt 必须包含最近对话、学员本轮输入和检索证据。"""

    prompt = TrainingSessionPromptService.customer_prompt(
        _profile(),
        _setting(),
        turns=[
            {"role": "customer", "content": "你先介绍一下。"},
            {"role": "trainee", "content": "我想先了解您的目标。"},
        ],
        trainee_message="我们可以先做试点。",
        evidence=[{"chunk_id": "chunk_1", "content": "制造业客户试点案例"}],
    )

    assert "你先介绍一下" in prompt
    assert "我想先了解您的目标" in prompt
    assert "我们可以先做试点" in prompt
    assert "制造业客户试点案例" in prompt


def test_score_prompt_contains_dialogue_scoring_rules_and_evidence():
    """评分 prompt 必须包含评分规则、完整对话和评分证据。"""

    prompt = TrainingSessionPromptService.score_prompt(
        _profile(),
        _setting(),
        turns=[
            {"round_no": 1, "role": "trainee", "content": "请问当前最大的销售卡点是什么？"},
            {"round_no": 1, "role": "customer", "content": "新人转化不稳定。"},
        ],
        evidence=[{"content": "评分案例证据"}],
    )

    assert "total_score" in prompt
    assert "请问当前最大的销售卡点是什么" in prompt
    assert "新人转化不稳定" in prompt
    assert "评分案例证据" in prompt


def test_fallback_messages_use_profile_and_evidence():
    """兜底话术要优先使用角色信息和证据状态。"""

    opening = TrainingSessionPromptService.fallback_opening_message(_profile())
    reply_with_evidence = TrainingSessionPromptService.fallback_customer_reply([{"content": "案例"}])
    reply_without_evidence = TrainingSessionPromptService.fallback_customer_reply([])
    conversation_text = TrainingSessionPromptService.conversation_text([
        {"role": "trainee", "content": "您好"},
        {"role": "customer", "content": "你好"},
    ])

    assert "采购负责人" in opening
    assert "新人转化率低" in opening
    assert "类似客户案例" in reply_with_evidence
    assert "投入产出" in reply_without_evidence
    assert conversation_text == "trainee：您好\ncustomer：你好"
