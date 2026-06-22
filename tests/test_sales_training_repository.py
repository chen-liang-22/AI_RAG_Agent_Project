from training.repository import TrainingRepository


def test_training_repository_persists_history_and_score(tmp_path):
    """训练仓储应能保存会话、开场白、学员轮次和评分，供前端复盘使用。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    role = repository.save_role_profile(
        trainee_id="trainee-1",
        profile_type="overseas_bd",
        visible_profile={"role": "采购负责人"},
        hidden_profile={"real_concerns": ["风险"]},
        role_profile={"position": "采购负责人"},
        role_confirm_card={"role_name": "谨慎客户"},
        selected_fields={},
        scenario_description="客户关注成本和风险",
        extra_details="",
        retrieved_evidence=[],
        status="confirmed",
    )
    setting = repository.save_goal_setting(
        profile_id=role["profile_id"],
        trainee_id="trainee-1",
        training_mode="open",
        training_purpose="需求挖掘",
        round_limit=6,
        stages=[
            {
                "stage_no": 1,
                "stage_name": "开放式",
                "core_goal": "挖掘客户顾虑",
                "success_conditions": ["客户愿意继续沟通"],
                "failure_conditions": ["客户拒绝沟通"],
            }
        ],
        status="confirmed",
    )
    session = repository.create_session(
        profile_id=role["profile_id"],
        setting_id=setting["setting_id"],
        trainee_id="trainee-1",
        training_mode="open",
        response_mode="stream",
        round_limit=6,
        status="active",
    )

    repository.add_turn(
        session_id=session["session_id"],
        role="customer",
        content="你先说说方案价值。",
        round_no=0,
        stage_no=1,
        response_mode="stream",
    )
    repository.add_turn(
        session_id=session["session_id"],
        role="trainee",
        content="我想先了解您的核心顾虑。",
        round_no=1,
        stage_no=1,
        response_mode="stream",
    )
    score = repository.save_score(
        session_id=session["session_id"],
        general_score=32,
        stage_score=44,
        penalty_score=0,
        final_score=76,
        level="及格",
        is_passed=True,
        detail={"hit_points": ["能主动提问"]},
        review_status="confirmed",
    )
    repository.update_session_status(
        session["session_id"],
        status="completed",
        total_score=76,
        level="及格",
        report={"hit_points": ["能主动提问"]},
    )

    sessions, total = repository.list_sessions(page=1, page_size=10, trainee_id="trainee-1")
    turns = repository.list_turns(session["session_id"])
    latest_score = repository.get_latest_score_by_session(session["session_id"])

    assert total == 1
    assert sessions[0]["answered_count"] == 1
    assert sessions[0]["total_score"] == 76
    assert [turn["role"] for turn in turns] == ["customer", "trainee"]
    assert latest_score["score_id"] == score["score_id"]
