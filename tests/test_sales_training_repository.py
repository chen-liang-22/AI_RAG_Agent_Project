from training.repository import TrainingRepository
from training.schemas import TrainingPlanCreateRequest, TrainingPlanUpdateRequest
from training.services.sales_training_service import SalesTrainingService


def test_training_repository_allows_duplicate_plan_names(tmp_path):
    """训练名称允许重复，训练记录通过 plan_id 区分。"""

    repository = TrainingRepository()
    base_payload = {
        "plan_name": "海外 BD 异议处理",
        "trainee": {
            "trainee_id": "trainee-1",
            "trainee_name": "销售学员",
            "position_role": "overseas_bd",
            "experience_level": "junior",
            "task_goal": "goal_junior",
            "weakness_tags": [],
            "student_portrait_other": "",
        },
        "profile_type": "overseas_bd",
        "selected_fields": {},
        "scenario_description": "客户担心交付风险。",
        "extra_details": "",
        "model_mode": "high",
    }

    first_plan = repository.create_plan(**base_payload)
    second_plan = repository.create_plan(**base_payload)
    plans, total = repository.list_plans(page=1, page_size=10, keyword="海外 BD")

    assert first_plan["plan_id"] != second_plan["plan_id"]
    assert total == 2
    assert {plan["plan_name"] for plan in plans} == {"海外 BD 异议处理"}


def test_training_repository_persists_history_and_score(tmp_path):
    """训练仓储应能保存会话、开场白、学员轮次和评分，供前端复盘使用。"""

    repository = TrainingRepository()
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


def test_training_plan_snapshot_is_independent_and_only_real_changes_mark_stale(tmp_path):
    """训练方案允许同名独立保存；只有画像或场景真实变化时才标记后续内容需重新生成。"""

    service = SalesTrainingService(repository=TrainingRepository())
    base_trainee = {
        "trainee_id": "trainee-1",
        "trainee_name": "张三",
        "position_role": "overseas_bd",
        "experience_level": "junior",
        "task_goal": "goal_junior",
        "weakness_tags": ["price_negotiation"],
        "student_portrait_other": "新员工",
    }
    base_fields = {
        "画像类型": "海外BD画像",
        "客户类型": "B端客户",
        "客户分类": "中等意向",
    }
    first_plan = service.create_plan(TrainingPlanCreateRequest(
        plan_name="同名训练",
        trainee=base_trainee,
        profile_type="overseas_bd",
        selected_fields=base_fields,
        scenario_description="客户关注成本和风险",
        extra_details="",
        model_mode="high",
    ))
    second_plan = service.create_plan(TrainingPlanCreateRequest(
        plan_name="同名训练",
        trainee={**base_trainee, "trainee_id": "trainee-2", "trainee_name": "李四"},
        profile_type="overseas_bd",
        selected_fields=base_fields,
        scenario_description="客户关注成本和风险",
        extra_details="",
        model_mode="high",
    ))
    assert first_plan.plan.plan_id != second_plan.plan.plan_id

    role = service.repository.save_role_profile(
        trainee_id="trainee-1",
        plan_id=first_plan.plan.plan_id,
        profile_type="overseas_bd",
        visible_profile={"role": "采购负责人"},
        hidden_profile={"real_concerns": ["风险"]},
        role_profile={"position": "采购负责人"},
        role_confirm_card={"role_name": "谨慎客户"},
        selected_fields=base_fields,
        scenario_description="客户关注成本和风险",
        extra_details="",
        retrieved_evidence=[],
        status="confirmed",
    )
    service.repository.attach_role_to_plan(first_plan.plan.plan_id, role["profile_id"])
    setting = service.repository.save_goal_setting(
        profile_id=role["profile_id"],
        plan_id=first_plan.plan.plan_id,
        trainee_id="trainee-1",
        training_mode="open",
        training_purpose="需求挖掘",
        round_limit=6,
        stages=[{
            "stage_no": 1,
            "stage_name": "开放式",
            "core_goal": "挖掘客户顾虑",
            "success_conditions": ["客户愿意继续沟通"],
            "failure_conditions": ["客户拒绝沟通"],
        }],
        scoring_rules={"general_dimensions": [], "stage_dimensions": []},
        status="confirmed",
    )
    service.repository.attach_goal_to_plan(first_plan.plan.plan_id, setting["setting_id"])

    unchanged = service.update_plan(first_plan.plan.plan_id, TrainingPlanUpdateRequest(
        plan_name="同名训练修改名称",
        trainee=base_trainee,
        profile_type="overseas_bd",
        selected_fields=base_fields,
        scenario_description="客户关注成本和风险",
        extra_details="",
        model_mode="medium",
    ))
    assert unchanged.plan.active_profile_id == role["profile_id"]
    assert unchanged.plan.active_setting_id == setting["setting_id"]
    assert unchanged.plan.role_status == "generated"
    assert unchanged.plan.goal_status == "generated"
    assert unchanged.plan.score_status == "generated"

    changed = service.update_plan(first_plan.plan.plan_id, TrainingPlanUpdateRequest(
        selected_fields={**base_fields, "客户分类": "高意向"},
    ))
    assert changed.plan.active_profile_id is None
    assert changed.plan.active_setting_id is None
    assert changed.plan.role_status == "stale"
    assert changed.plan.goal_status == "stale"
    assert changed.plan.score_status == "stale"
