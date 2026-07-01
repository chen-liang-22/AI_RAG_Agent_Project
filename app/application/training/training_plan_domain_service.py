"""销售训练方案领域服务。"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException

from app.application.training.goal_stage_adapter import GoalStageAdapter
from app.application.training.training_score_service import TrainingScoreService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import (
    GoalSettingResponse,
    GoalStage,
    TrainingPlanCreateRequest,
    TrainingPlanDeleteResponse,
    TrainingPlanDetailResponse,
    TrainingPlanListResponse,
    TrainingPlanSummaryResponse,
    TrainingPlanUpdateRequest,
)
from core.utils.database_connection import DatabaseErrorTypes
from core.utils.logger_handler import logger


class TrainingPlanDomainService:
    """销售训练方案领域服务。

    这里使用外观模式，把训练方案 CRUD、状态联动和响应转换收拢成稳定接口。
    """

    def __init__(self, *, repository: TrainingRepository):
        """初始化训练方案领域服务。"""

        self.repository = repository

    def create_plan(self, request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
        """创建训练方案。"""

        trainee = request.trainee.model_dump() if request.trainee is not None else {}
        plan = self.repository.create_plan(
            plan_name=request.plan_name.strip(),
            trainee=trainee,
            profile_type=request.profile_type,
            selected_fields=request.selected_fields,
            scenario_description=request.scenario_description.strip(),
            extra_details=request.extra_details.strip(),
            model_mode=request.model_mode,
        )
        logger.info("[销售训练] 训练方案创建完成 方案编号=%s 名称=%s", plan["plan_id"], plan["plan_name"])
        return self.plan_detail_response(plan)

    def list_plans(self, *, page: int = 1, page_size: int = 10, keyword: str | None = None) -> TrainingPlanListResponse:
        """分页查询训练方案列表。"""

        safe_page = max(1, page)
        safe_page_size = max(1, min(50, page_size))
        rows, total = self.repository.list_plans(page=safe_page, page_size=safe_page_size, keyword=keyword)
        return TrainingPlanListResponse(
            items=[self.plan_summary(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def get_plan_detail(self, plan_id: str) -> TrainingPlanDetailResponse:
        """查询训练方案完整详情。"""

        return self.plan_detail_response(self.require_plan(plan_id))

    def delete_plan(self, plan_id: str) -> TrainingPlanDeleteResponse:
        """删除训练方案。"""

        plan = self.require_plan(plan_id)
        deleted = self.repository.delete_plan(plan_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        logger.info("[销售训练] 训练方案已删除 方案编号=%s 名称=%s", plan_id, plan.get("plan_name"))
        return TrainingPlanDeleteResponse(status="deleted", plan_id=plan_id)

    def update_plan(self, plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
        """修改训练方案，并维护角色、阶段、评分的联动失效状态。"""

        plan = self.require_plan(plan_id)
        updates: dict[str, Any] = {}
        role_input_changed = False
        role_content_changed = False
        goal_changed = False

        if request.plan_name is not None and request.plan_name.strip() != plan["plan_name"]:
            updates["plan_name"] = request.plan_name.strip()
        if request.trainee is not None:
            trainee_data = request.trainee.model_dump()
            if self.json_changed(self.load_json(plan.get("trainee_json"), {}), trainee_data):
                updates["trainee_json"] = self.repository._json(trainee_data)
                updates["trainee_id"] = trainee_data["trainee_id"]
                updates["trainee_name"] = trainee_data.get("trainee_name") or "销售学员"
                role_input_changed = True
        if request.profile_type is not None and request.profile_type != plan["profile_type"]:
            updates["profile_type"] = request.profile_type
            role_input_changed = True
        if request.selected_fields is not None:
            if self.json_changed(self.load_json(plan.get("selected_fields_json"), {}), request.selected_fields):
                updates["selected_fields_json"] = self.repository._json(request.selected_fields)
                role_input_changed = True
        if request.scenario_description is not None:
            scenario_description = request.scenario_description.strip()
            if scenario_description != (plan.get("scenario_description") or ""):
                updates["scenario_description"] = scenario_description
                role_input_changed = True
        if request.extra_details is not None:
            extra_details = request.extra_details.strip()
            if extra_details != (plan.get("extra_details") or ""):
                updates["extra_details"] = extra_details
                role_input_changed = True
        if request.model_mode is not None:
            updates["model_mode"] = request.model_mode

        if role_input_changed:
            updates.update({
                "active_profile_id": None,
                "active_setting_id": None,
                "role_status": "stale" if plan.get("active_profile_id") else "pending",
                "goal_status": "stale" if plan.get("active_setting_id") else "pending",
                "score_status": "stale" if plan.get("active_setting_id") else "pending",
            })

        active_profile_id = updates.get("active_profile_id", plan.get("active_profile_id"))
        if active_profile_id and any(value is not None for value in (
                request.role_confirm_card,
                request.visible_profile,
                request.hidden_profile,
                request.role_profile,
        )):
            self.repository.update_role_profile(
                active_profile_id,
                visible_profile=request.visible_profile,
                hidden_profile=request.hidden_profile,
                role_profile=request.role_profile,
                role_confirm_card=request.role_confirm_card,
            )
            role_content_changed = request.hidden_profile is not None or request.role_profile is not None
            if role_content_changed and not role_input_changed:
                updates.update({
                    "active_setting_id": None,
                    "goal_status": "stale",
                    "score_status": "stale",
                })

        active_setting_id = updates.get("active_setting_id", plan.get("active_setting_id"))
        if active_setting_id and (
                request.training_purpose is not None
                or request.round_limit is not None
                or request.stages is not None
        ):
            self.repository.update_goal_setting(
                active_setting_id,
                training_purpose=request.training_purpose.strip() if request.training_purpose is not None else None,
                round_limit=request.round_limit,
                stages=[item.model_dump() for item in request.stages] if request.stages is not None else None,
            )
            goal_changed = True
        if active_setting_id and request.scoring_rules is not None:
            self.repository.update_goal_setting(active_setting_id, scoring_rules=request.scoring_rules)
        if goal_changed and not role_input_changed and not role_content_changed:
            updates["score_status"] = "stale"

        try:
            updated = self.repository.update_plan(plan_id, **updates) if updates else self.require_plan(plan_id)
        except DatabaseErrorTypes as exc:
            logger.error("[销售训练] 训练方案保存数据库异常 方案编号=%s 错误=%s", plan_id, exc, exc_info=True)
            raise
        logger.info(
            "[销售训练] 训练方案已修改 方案编号=%s 角色输入变化=%s 角色内容变化=%s 阶段变化=%s",
            plan_id,
            role_input_changed,
            role_content_changed,
            goal_changed,
        )
        return self.plan_detail_response(updated)

    def require_plan(self, plan_id: str) -> dict[str, Any]:
        """查询训练方案，不存在时直接抛出 404。"""

        plan = self.repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        return plan

    @staticmethod
    def plan_summary(row: dict[str, Any]) -> TrainingPlanSummaryResponse:
        """把训练方案数据库行转换成列表摘要。"""

        return TrainingPlanSummaryResponse(
            plan_id=row["plan_id"],
            plan_name=row["plan_name"],
            trainee_id=row["trainee_id"],
            trainee_name=row["trainee_name"],
            profile_type=row["profile_type"],
            model_mode=row.get("model_mode"),
            role_status=row["role_status"],
            goal_status=row["goal_status"],
            score_status=row["score_status"],
            active_profile_id=row.get("active_profile_id"),
            active_setting_id=row.get("active_setting_id"),
            created_at=TrainingPlanDomainService.format_response_time(row["created_at"]),
            updated_at=TrainingPlanDomainService.format_response_time(row["updated_at"]),
        )

    def plan_detail_response(self, row: dict[str, Any]) -> TrainingPlanDetailResponse:
        """把训练方案数据库行转换成完整详情。"""

        role_row = self.repository.get_role_profile(row["active_profile_id"]) if row.get("active_profile_id") else None
        setting_row = self.repository.get_goal_setting(row["active_setting_id"]) if row.get("active_setting_id") else None
        visible_profile = self.load_json(role_row.get("visible_profile_json"), {}) if role_row else {}
        hidden_profile = self.load_json(role_row.get("hidden_profile_json"), {}) if role_row else {}
        role_profile = self.load_json(role_row.get("role_profile_json"), {}) if role_row else {}
        role_confirm_card = self.load_json(role_row.get("role_confirm_card_json"), {}) if role_row else {}
        retrieved_cases = self.load_json(role_row.get("retrieved_evidence_json"), []) if role_row else []
        return TrainingPlanDetailResponse(
            plan=self.plan_summary(row),
            trainee=self.load_json(row.get("trainee_json"), {}),
            selected_fields=self.load_json(row.get("selected_fields_json"), {}),
            scenario_description=row.get("scenario_description") or "",
            extra_details=row.get("extra_details") or "",
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            retrieved_cases=retrieved_cases,
            goal_setting=self.goal_response(setting_row) if setting_row else None,
        )

    def goal_response(self, row: dict[str, Any]) -> GoalSettingResponse:
        """把数据库训练设置行转换成 Pydantic 响应。"""

        stages = [
            GoalStage(**item)
            for item in GoalStageAdapter.normalize_stages(self.load_json(row.get("stages_json"), []))
        ]
        return GoalSettingResponse(
            setting_id=row["setting_id"],
            profile_id=row["profile_id"],
            training_mode=row["training_mode"],
            training_purpose=row["training_purpose"],
            round_limit=int(row["round_limit"]),
            stages=stages,
            scoring_rules=self.load_json(row.get("scoring_rules_json"), TrainingScoreService.default_scoring_rules()),
            status=row["status"],
        )

    @staticmethod
    def load_json(value: Any, default: Any) -> Any:
        """安全读取 JSON 字段。"""

        if not value:
            return default
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def json_changed(old_value: Any, new_value: Any) -> bool:
        """比较两个 JSON 结构是否真正变化。"""

        return json.dumps(old_value, ensure_ascii=False, sort_keys=True) != json.dumps(
            new_value,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def format_response_time(value: object) -> str | None:
        """把数据库时间字段统一转换成接口响应字符串。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds", sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return str(value)
