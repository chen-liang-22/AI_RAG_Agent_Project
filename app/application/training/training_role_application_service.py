"""销售训练角色生成应用服务。"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.training.training_query_service import TrainingQueryService
from app.application.training.training_role_service import TrainingRoleService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import (
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestionGenerateResponse,
)
from core.model.factory import get_chat_model
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


class TrainingRoleApplicationService:
    """销售训练角色生成应用服务。

    这里使用外观模式，把补充问答、场景润色和 AI 客户角色生成收拢到角色应用服务中。
    """

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            query_service: TrainingQueryService,
            role_service: TrainingRoleService,
    ):
        """初始化角色生成应用服务。"""

        self.repository = repository
        self.query_service = query_service
        self.role_service = role_service

    def generate_supplement_questions(self, request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
        """生成 AI 陪练角色前的补充问答题。"""

        query = self.role_service.build_role_query(request)
        logger.info(
            "[销售训练][补充问答题] 开始生成 方案编号=%s 学员=%s 模型档位=%s 查询预览=%s",
            request.plan_id or "-",
            request.trainee.trainee_id,
            request.model_mode or "默认",
            self.short_text(query),
        )
        evidence = self.search_training_evidence(query, visibility=("visible", "hidden"), k=4)
        prompt = self.role_service.supplement_questions_prompt(request, evidence)
        fallback = {"questions": self.role_service.fallback_supplement_questions(request)}
        result = self.invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=fallback,
            task_name="补充问答题生成",
        )
        questions = self.role_service.normalize_supplement_questions(result.get("questions"), request)
        logger.info(
            "[销售训练][补充问答题] 生成完成 题目数=%s 学员=%s 证据数量=%s",
            len(questions),
            request.trainee.trainee_id,
            len(evidence),
        )
        return SupplementQuestionGenerateResponse(questions=questions)

    def polish_scenario(self, request: ScenarioPolishRequest) -> ScenarioPolishResponse:
        """根据客户画像字段润色训练场景描述。"""

        prompt = self.role_service.scenario_polish_prompt(request)
        fallback = {"polished_scenario": TrainingRoleService.fallback_polished_scenario(request)}
        logger.info(
            "[销售训练][场景润色] 开始润色 画像类型=%s 模型档位=%s 原始长度=%s 选择字段数=%s",
            request.profile_type,
            request.model_mode or "默认",
            len(request.scenario_description),
            len(request.selected_fields or {}),
        )
        result = self.invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=fallback,
            task_name="场景描述润色",
        )
        polished_scenario = str(result.get("polished_scenario") or "").strip()
        if not polished_scenario:
            polished_scenario = TrainingRoleService.fallback_polished_scenario(request)
        logger.info(
            "[销售训练] 场景描述AI润色完成 画像类型=%s 原始长度=%s 润色后长度=%s",
            request.profile_type,
            len(request.scenario_description),
            len(polished_scenario),
        )
        return ScenarioPolishResponse(
            polished_scenario=polished_scenario,
            original_scenario=request.scenario_description,
        )

    def generate_role(self, request: RoleGenerateRequest) -> RoleGenerateResponse:
        """生成 AI 陪练角色。"""

        if request.plan_id:
            self.require_plan(request.plan_id)
        query = self.role_service.build_role_query(request)
        logger.info(
            "[销售训练][角色生成] 开始生成 方案编号=%s 学员=%s 画像类型=%s 模型档位=%s 选择字段数=%s 场景长度=%s",
            request.plan_id or "-",
            request.trainee.trainee_id,
            request.profile_type,
            request.model_mode or "默认",
            len(request.selected_fields or {}),
            len(request.scenario_description or ""),
        )
        # 角色生成阶段允许使用 visible 和 hidden 知识；hidden 只给 AI 客户使用，不直接暴露给学员。
        evidence = self.search_training_evidence(query, visibility=("visible", "hidden"), k=6)
        logger.info(
            "[销售训练][角色生成] 证据召回完成 方案编号=%s 证据数量=%s 命中切片=%s",
            request.plan_id or "-",
            len(evidence),
            self.join_values(item.get("chunk_id") for item in evidence),
        )
        prompt = self.role_service.role_prompt(request, evidence)
        result = self.invoke_json(
            prompt,
            model_mode=request.model_mode,
            fallback=self.role_service.fallback_role(request, evidence),
            task_name="AI客户角色生成",
        )

        visible_profile = result.get("visible_profile") or {}
        hidden_profile = result.get("hidden_profile") or {}
        role_profile = result.get("role_profile") or {}
        role_confirm_card = result.get("role_confirm_card") or visible_profile

        saved = self.repository.save_role_profile(
            plan_id=request.plan_id,
            trainee_id=request.trainee.trainee_id,
            profile_type=request.profile_type,
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            selected_fields=request.selected_fields,
            scenario_description=request.scenario_description,
            extra_details=request.extra_details,
            retrieved_evidence=evidence,
            status="confirmed",
        )
        if request.plan_id:
            self.repository.attach_role_to_plan(request.plan_id, saved["profile_id"])
        logger.info(
            "[销售训练][角色生成] 生成完成 角色编号=%s 学员=%s 证据数量=%s 角色字段=%s",
            saved["profile_id"],
            request.trainee.trainee_id,
            len(evidence),
            self.dict_key_text(role_profile),
        )
        return RoleGenerateResponse(
            profile_id=saved["profile_id"],
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            hidden_summary="已生成隐藏心理画像，学员不可见",
            retrieved_cases=evidence,
            knowledge_facts=[item["content"][:160] for item in evidence],
        )

    def require_plan(self, plan_id: str) -> dict[str, Any]:
        """查询训练方案，不存在时直接抛出 404。"""

        plan = self.repository.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="训练方案不存在")
        return plan

    def search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict[str, Any]]:
        """检索训练证据库，并过滤学员不可直接看到的内容。"""

        return self.query_service.search_training_evidence(query, visibility=visibility, k=k)

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
    def short_text(value: Any, limit: int = 120) -> str:
        """把长文本压缩成日志预览，避免 PyCharm 控制台被完整提示词刷屏。"""

        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text or "-"
        return f"{text[:limit]}..."

    @staticmethod
    def join_values(values: Any, limit: int = 6) -> str:
        """把列表、元组或生成器压成一行日志文本，方便查看命中的来源。"""

        if values is None:
            return "-"
        if isinstance(values, (str, int, float)):
            return str(values)
        result: list[str] = []
        for value in values:
            if value is None or value == "":
                continue
            text = str(value)
            if text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return "、".join(result) if result else "-"

    @staticmethod
    def dict_key_text(value: Any) -> str:
        """只打印字典字段名，不打印完整内容，避免日志泄露隐藏画像细节。"""

        if not isinstance(value, dict) or not value:
            return "-"
        return "、".join(str(key) for key in value.keys())
