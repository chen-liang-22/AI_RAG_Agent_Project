"""销售训练会话基础服务。"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.training.training_score_service import TrainingScoreService
from app.application.training.training_session_prompt_service import TrainingSessionPromptService
from app.application.training_support.repository import TrainingRepository, utc_now_text
from app.application.training_support.schemas import (
    TrainingScoreResponse,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingSessionSummaryResponse,
    TrainingTurnRecordResponse,
)
from core.model.factory import get_chat_model
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


class TrainingSessionBasicService:
    """销售训练会话基础服务。

    这里使用外观模式，把会话创建、会话列表和复盘详情收拢成稳定接口。
    对话提交和流式回复由会话对话服务处理，最终评分后续再拆。
    """

    def __init__(self, *, repository: TrainingRepository, session_prompt_service: TrainingSessionPromptService):
        """初始化会话基础服务。"""

        self.repository = repository
        self.session_prompt_service = session_prompt_service

    def start_session(self, request: TrainingSessionStartRequest) -> TrainingSessionResponse:
        """开始一次开放式训练，并保存 AI 客户开场白。"""

        setting = self.require_goal_setting(request.setting_id)
        if setting["profile_id"] != request.profile_id:
            raise HTTPException(status_code=400, detail="训练设置和陪练角色不匹配")
        response_mode = self.normalize_response_mode(request.response_mode)
        logger.info(
            "[销售训练] 创建训练会话开始 角色编号=%s 设置编号=%s 学员编号=%s 回复模式=%s 轮数上限=%s",
            request.profile_id,
            request.setting_id,
            request.trainee_id,
            response_mode,
            setting["round_limit"],
        )
        session = self.repository.create_session(
            profile_id=request.profile_id,
            setting_id=request.setting_id,
            trainee_id=request.trainee_id,
            training_mode="open",
            response_mode=response_mode,
            round_limit=int(setting["round_limit"]),
            status="active",
        )
        opening_message = self.generate_opening_message(session, model_mode=request.model_mode)
        self.repository.add_turn(
            session_id=session["session_id"],
            role="customer",
            content=opening_message,
            round_no=0,
            response_mode=response_mode,
            stage_no=1,
            started_at=session["started_at"],
            submitted_at=utc_now_text(),
            metadata={"turn_type": "opening"},
        )
        logger.info("[销售训练] 训练会话开始 会话编号=%s 回复模式=%s 已生成开场白", session["session_id"], response_mode)
        return self.session_response(session, opening_message=opening_message)

    def list_sessions(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            trainee_id: str | None = None,
    ) -> TrainingSessionListResponse:
        """分页查询训练历史。"""

        safe_page = max(1, page)
        safe_page_size = max(1, min(50, page_size))
        rows, total = self.repository.list_sessions(page=safe_page, page_size=safe_page_size, trainee_id=trainee_id)
        return TrainingSessionListResponse(
            items=[self.session_summary(row) for row in rows],
            total=total,
            page=safe_page,
            page_size=safe_page_size,
        )

    def get_session_detail(self, session_id: str) -> TrainingSessionDetailResponse:
        """查询训练复盘详情。"""

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")

        turns = self.repository.list_turns(session_id)
        profile = self.require_role_profile(session["profile_id"])
        setting = self.require_goal_setting(session["setting_id"])
        score = self.repository.get_latest_score_by_session(session_id)
        visible_profile = self.load_json(profile.get("visible_profile_json"), {})
        hidden_profile = self.load_json(profile.get("hidden_profile_json"), {})
        role_profile = self.load_json(profile.get("role_profile_json"), {})
        role_confirm_card = self.load_json(profile.get("role_confirm_card_json"), {})
        retrieved_evidence = self.load_json(profile.get("retrieved_evidence_json"), [])
        summary = self.session_summary({**session, "answered_count": sum(1 for item in turns if item["role"] == "trainee")})
        return TrainingSessionDetailResponse(
            session=summary,
            turns=[self.turn_record(item) for item in turns],
            visible_profile=visible_profile,
            hidden_profile=hidden_profile,
            role_profile=role_profile,
            role_confirm_card=role_confirm_card,
            goal_setting={
                "setting_id": setting["setting_id"],
                "training_purpose": setting["training_purpose"],
                "round_limit": int(setting["round_limit"]),
                "stages": self.load_json(setting.get("stages_json"), []),
                "scoring_rules": self.load_json(setting.get("scoring_rules_json"), TrainingScoreService.default_scoring_rules()),
            },
            knowledge_facts=[item["content"][:160] for item in retrieved_evidence if item.get("content")],
            score=self.score_response(score) if score else None,
        )

    def generate_opening_message(self, session: dict[str, Any], *, model_mode: str | None) -> str:
        """生成 AI 客户开场白，让训练会话一开始就像真实客户在场。"""

        prompt = self.opening_prompt(session)
        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI客户开场白] 调用开始 会话编号=%s 模型档位=%s 提示词长度=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
        )
        try:
            response = get_chat_model(model_mode).invoke(
                self.messages(prompt_manager.get("training.ai_customer_system"), prompt)
            )
            text = self.content_text(response.content).strip()
            if text:
                logger.info(
                    "[销售训练][AI客户开场白] 调用完成 会话编号=%s 回复长度=%s 耗时秒=%s 回复预览=%s",
                    session["session_id"],
                    len(text),
                    round(max(0.0, time.perf_counter() - start_perf), 3),
                    self.short_text(text),
                )
                return text
            logger.warning("[销售训练][AI客户开场白] 模型返回为空，使用兜底开场白 会话编号=%s", session["session_id"])
            return self.fallback_opening_message(session)
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户开场白生成失败，使用兜底开场白 会话编号=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return self.fallback_opening_message(session)

    def opening_prompt(self, session: dict[str, Any]) -> str:
        """构造 AI 客户开场白提示词。"""

        profile = self.require_role_profile(session["profile_id"])
        setting = self.require_goal_setting(session["setting_id"])
        return self.session_prompt_service.opening_prompt(profile, setting)

    def fallback_opening_message(self, session: dict[str, Any]) -> str:
        """AI 客户开场白失败时的兜底话术。"""

        profile = self.require_role_profile(session["profile_id"])
        return self.session_prompt_service.fallback_opening_message(profile)

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

    @staticmethod
    def normalize_response_mode(response_mode: str | None) -> str:
        """统一响应模式枚举。"""

        return "blocking" if response_mode in {"blocking", "once"} else "stream"

    @staticmethod
    def session_response(row: dict[str, Any], opening_message: str | None = None) -> TrainingSessionResponse:
        """把数据库训练会话行转换成接口响应对象。"""

        return TrainingSessionResponse(
            session_id=row["session_id"],
            profile_id=row["profile_id"],
            setting_id=row["setting_id"],
            trainee_id=row["trainee_id"],
            training_mode=row["training_mode"],
            response_mode=row["response_mode"],
            current_stage_no=int(row["current_stage_no"]),
            status=row["status"],
            round_limit=int(row["round_limit"]),
            opening_message=opening_message,
        )

    @staticmethod
    def session_summary(row: dict[str, Any]) -> TrainingSessionSummaryResponse:
        """把数据库会话行转换成前端历史摘要。"""

        return TrainingSessionSummaryResponse(
            session_id=row["session_id"],
            trainee_id=row["trainee_id"],
            training_mode=row["training_mode"],
            response_mode=row["response_mode"],
            status=row["status"],
            round_limit=int(row["round_limit"]),
            answered_count=int(row.get("answered_count") or 0),
            total_score=row.get("total_score"),
            level=row.get("level"),
            started_at=TrainingSessionBasicService.format_response_time(row["started_at"]),
            ended_at=TrainingSessionBasicService.format_response_time(row.get("ended_at")),
            updated_at=TrainingSessionBasicService.format_response_time(row["updated_at"]),
        )

    def turn_record(self, row: dict[str, Any]) -> TrainingTurnRecordResponse:
        """把数据库轮次行转换成复盘消息。"""

        return TrainingTurnRecordResponse(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            round_no=int(row["round_no"]),
            stage_no=int(row["stage_no"]),
            response_mode=row.get("response_mode"),
            response_seconds=row.get("response_seconds"),
            retrieved_chunk_ids=self.load_json(row.get("retrieved_chunk_ids_json"), []),
            stage_decision=self.load_json(row.get("stage_decision_json"), {}),
            coach_analysis=self.load_json(row.get("coach_analysis_json"), {}),
            created_at=self.format_response_time(row["created_at"]),
        )

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
    def short_text(value: Any, limit: int = 120) -> str:
        """把长文本压缩成日志预览。"""

        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text or "-"
        return f"{text[:limit]}..."

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
    def format_response_time(value: object) -> str | None:
        """把数据库时间字段统一转换成接口响应字符串。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds", sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return str(value)
