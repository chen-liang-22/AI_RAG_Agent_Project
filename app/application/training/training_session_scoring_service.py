"""销售训练会话评分服务。"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.training.training_query_service import TrainingQueryService
from app.application.training.training_score_service import TrainingScoreService
from app.application.training.training_session_prompt_service import TrainingSessionPromptService
from app.application.training_support.repository import TrainingRepository
from app.application.training_support.schemas import TrainingScoreResponse
from core.model.factory import get_chat_model
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


class TrainingSessionScoringService:
    """销售训练会话最终评分服务。

    这里使用外观模式，把会话校验、评分证据召回、评分模型调用和评分落库收拢成稳定接口。
    """

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            query_service: TrainingQueryService,
            session_prompt_service: TrainingSessionPromptService,
    ):
        """初始化会话评分服务。"""

        self.repository = repository
        self.query_service = query_service
        self.session_prompt_service = session_prompt_service

    def final_score(self, session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
        """结束训练并生成评分报告。"""

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")

        existing_score = self.repository.get_latest_score_by_session(session_id)
        if existing_score and session["status"] == "completed":
            logger.info("[销售训练] 训练评分已存在，直接返回 会话编号=%s", session_id)
            return self.score_response(existing_score)
        if session["status"] not in {"active", "scoring"}:
            raise HTTPException(status_code=400, detail=f"当前训练状态不允许评分：{session['status']}")

        turns = self.repository.list_turns(session_id)
        if not any(turn["role"] == "trainee" for turn in turns):
            raise HTTPException(status_code=400, detail="没有学员回复，不能评分")

        setting = self.require_goal_setting(session["setting_id"])
        profile = self.require_role_profile(session["profile_id"])
        conversation_text = self.conversation_text(turns)
        logger.info(
            "[销售训练][评分] 开始生成评分 会话编号=%s 模型档位=%s 对话轮次=%s 对话长度=%s",
            session_id,
            model_mode or "默认",
            len(turns),
            len(conversation_text),
        )
        evidence = self.search_training_evidence(conversation_text, visibility=("visible", "scoring_only"), k=6)
        logger.info(
            "[销售训练][评分] 评分证据召回完成 会话编号=%s 证据数量=%s 命中切片=%s",
            session_id,
            len(evidence),
            self.join_values(item.get("chunk_id") for item in evidence),
        )
        result = self.invoke_json(
            self.score_prompt(profile, setting, turns, evidence),
            model_mode=model_mode,
            fallback=TrainingScoreService.fallback_score(turns, evidence),
            task_name="训练评分报告生成",
        )

        general_score = int(max(0, min(40, result.get("general_score") or 32)))
        stage_score = int(max(0, min(60, result.get("stage_score") or 43)))
        penalty_score = int(max(0, min(20, result.get("penalty_score") or 0)))
        # 最终得分以后端公式为准，避免模型直接返回的 total_score 破坏评分规则。
        final_score = int(max(0, min(100, general_score + stage_score - penalty_score)))
        level = TrainingScoreService.score_level(final_score)
        report = {
            "hit_points": result.get("hit_points") or [],
            "missing_points": result.get("missing_points") or [],
            "wrong_points": result.get("wrong_points") or [],
            "evidence_refs": result.get("evidence_refs") or [],
            "improvement_advice": result.get("improvement_advice") or "",
            "reference_script": result.get("reference_script") or "",
            "next_training_plan": result.get("next_training_plan") or [],
            "scoring_rules": self.load_json(setting.get("scoring_rules_json"), TrainingScoreService.default_scoring_rules()),
        }
        score = self.repository.save_score(
            session_id=session_id,
            general_score=general_score,
            stage_score=stage_score,
            penalty_score=penalty_score,
            final_score=final_score,
            level=level,
            is_passed=final_score >= 75,
            detail=report,
            review_status="confirmed",
        )
        self.repository.update_session_status(session_id, status="completed", total_score=final_score, level=level, report=report)
        logger.info("[销售训练] 训练评分完成 会话编号=%s 得分=%s 等级=%s", session_id, final_score, level)
        return self.score_response(score)

    def require_role_profile(self, profile_id: str) -> dict[str, Any]:
        """查询 AI 角色画像，不存在时直接抛出 404。"""

        profile = self.repository.get_role_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="AI 陪练角色不存在")
        return profile

    def require_goal_setting(self, setting_id: str) -> dict[str, Any]:
        """查询训练目标设置，不存在时直接抛出 404。"""

        setting = self.repository.get_goal_setting(setting_id)
        if not setting:
            raise HTTPException(status_code=404, detail="训练设置不存在")
        return setting

    def search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict[str, Any]]:
        """检索训练证据库，并过滤学员不可直接看到的内容。"""

        return self.query_service.search_training_evidence(query, visibility=visibility, k=k)

    def score_prompt(
            self,
            profile: dict[str, Any],
            setting: dict[str, Any],
            turns: list[dict[str, Any]],
            evidence: list[dict[str, Any]],
    ) -> str:
        """构造最终评分报告提示词。"""

        return self.session_prompt_service.score_prompt(profile, setting, turns=turns, evidence=evidence)

    @staticmethod
    def conversation_text(turns: list[dict[str, Any]]) -> str:
        """把训练对话轮次拼成评分和证据检索使用的纯文本。"""

        return TrainingSessionPromptService.conversation_text(turns)

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

    def score_response(self, row: dict[str, Any]) -> TrainingScoreResponse:
        """把数据库评分行转换成评分响应。"""

        return TrainingScoreResponse(
            score_id=row["score_id"],
            session_id=row["session_id"],
            total_score=int(row["final_score"]),
            level=row["level"],
            is_passed=bool(row["is_passed"]),
            general_score=int(row["general_score"]),
            stage_score=int(row["stage_score"]),
            penalty_score=int(row["penalty_score"]),
            report=self.load_json(row.get("detail_json"), {}),
        )

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
