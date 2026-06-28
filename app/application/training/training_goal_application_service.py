"""销售训练目标生成应用服务。"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.training.goal_stage_adapter import GoalStageAdapter
from app.application.training.training_goal_setting_service import TrainingGoalSettingService
from app.application.training.training_score_service import TrainingScoreService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import GoalSettingResponse, GoalStage
from core.model.factory import get_chat_model
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


class TrainingGoalApplicationService:
    """销售训练目标生成应用服务。

    这里使用外观模式，把训练阶段、动态轮数和评分规则生成流程收拢成稳定接口。
    """

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            goal_setting_service: TrainingGoalSettingService,
    ):
        """初始化目标生成应用服务。"""

        self.repository = repository
        self.goal_setting_service = goal_setting_service

    def generate_goal_setting(
            self,
            *,
            profile_id: str,
            trainee_id: str,
            training_mode: str,
            plan_id: str | None = None,
            model_mode: str | None = None,
    ) -> GoalSettingResponse:
        """生成一期开放式训练设置。"""

        if training_mode != "open":
            raise HTTPException(status_code=400, detail="流程式训练二期支持，一期只支持开放式")

        if plan_id:
            self.require_plan(plan_id)
        profile = self.require_role_profile(profile_id)
        prompt = self.goal_setting_service.goal_prompt(profile)
        logger.info(
            "[销售训练][训练设置] 开始生成 方案编号=%s 角色编号=%s 学员=%s 模式=%s 模型档位=%s",
            plan_id or "-",
            profile_id,
            trainee_id,
            training_mode,
            model_mode or "默认",
        )
        fallback = TrainingGoalSettingService.fallback_goal(profile)
        result = self.invoke_json(
            prompt,
            model_mode=model_mode,
            fallback=fallback,
            task_name="训练阶段和评分规则生成",
        )
        round_limit = self.normalize_round_limit(result.get("round_limit"))
        stages = GoalStageAdapter.normalize_stages(
            result.get("stages"),
            fallback_stages=fallback["stages"],
        )
        scoring_rules = TrainingScoreService.normalize_scoring_rules(result.get("scoring_rules"), stages[:1], profile)

        saved = self.repository.save_goal_setting(
            profile_id=profile_id,
            trainee_id=trainee_id,
            training_mode="open",
            training_purpose=str(result.get("training_purpose") or "开放式销售训练")[:20],
            round_limit=round_limit,
            stages=stages[:1],
            scoring_rules=scoring_rules,
            plan_id=plan_id,
            status="confirmed",
        )
        if plan_id:
            self.require_plan(plan_id)
            self.repository.attach_goal_to_plan(plan_id, saved["setting_id"])
        logger.info(
            "[销售训练][训练设置] 生成完成 设置编号=%s 轮数=%s 阶段数量=%s 评分维度=%s",
            saved["setting_id"],
            round_limit,
            len(stages[:1]),
            len(scoring_rules.get("dimensions") or []),
        )
        return self.goal_response(saved)

    def require_role_profile(self, profile_id: str) -> dict[str, Any]:
        """查询 AI 角色画像，不存在时直接抛出 404。"""

        profile = self.repository.get_role_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="AI 陪练角色不存在")
        return profile

    def require_plan(self, plan_id: str) -> dict[str, Any]:
        """查询训练方案，不存在时直接抛出 404。"""

        plan = self.repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        return plan

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

    def invoke_json(self, prompt: str, *, model_mode: str | None, fallback: dict, task_name: str) -> dict:
        """调用 LLM 并解析 JSON，失败时使用可解释兜底。"""

        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI调用开始] 任务=%s 模型档位=%s 提示词长度=%s 兜底字段=%s",
            task_name,
            model_mode or "默认",
            len(prompt),
            self.dict_key_text(fallback),
        )
        try:
            response = get_chat_model(model_mode).invoke(
                self.messages(prompt_manager.get("training.json_only_system"), prompt)
            )
            text = self.content_text(response.content)
            parsed = self.parse_json_object(text)
            logger.info(
                "[销售训练][AI调用完成] 任务=%s 模型档位=%s 返回长度=%s JSON字段=%s 耗时秒=%s",
                task_name,
                model_mode or "默认",
                len(text),
                self.dict_key_text(parsed),
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
            return parsed
        except Exception as exc:
            logger.warning(
                "[销售训练] LLM JSON生成失败，使用兜底结构 任务=%s 模型档位=%s 提示词长度=%s 耗时秒=%s 错误=%s",
                task_name,
                model_mode or "默认",
                len(prompt),
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return fallback

    @staticmethod
    def normalize_round_limit(value: Any) -> int:
        """把模型返回的轮数限制在合理范围。"""

        try:
            round_limit = int(value)
        except (TypeError, ValueError):
            round_limit = 8
        return max(5, min(100, round_limit))

    @staticmethod
    def messages(system: str, human: str) -> list:
        """构造 LangChain 聊天消息。"""

        return [SystemMessage(content=system), HumanMessage(content=human)]

    @staticmethod
    def content_text(content: Any) -> str:
        """把不同模型返回格式统一转成字符串。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
        return str(content)

    @staticmethod
    def parse_json_object(text: str) -> dict:
        """从模型输出文本中提取 JSON 对象。"""

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("模型没有输出 JSON 对象")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return parsed

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
    def dict_key_text(value: Any) -> str:
        """只打印字典字段名，不打印完整内容，避免日志泄露隐藏画像细节。"""

        if not isinstance(value, dict) or not value:
            return "-"
        return "、".join(str(key) for key in value.keys())
