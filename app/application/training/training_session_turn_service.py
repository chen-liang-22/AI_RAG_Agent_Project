"""销售训练会话对话服务。"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.training.training_query_service import TrainingQueryService
from app.application.training.training_session_prompt_service import TrainingSessionPromptService
from app.application.training_support.repository import TrainingRepository, utc_now_text
from app.application.training_support.schemas import TrainingTurnRequest, TrainingTurnResponse
from core.model.factory import get_chat_model
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


class TrainingSessionTurnService:
    """销售训练会话对话服务。

    这里使用外观模式，把一次性回复、SSE 流式回复、证据召回和轮次收尾收拢成稳定接口。
    """

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            query_service: TrainingQueryService,
            session_prompt_service: TrainingSessionPromptService,
    ):
        """初始化会话对话服务。"""

        self.repository = repository
        self.query_service = query_service
        self.session_prompt_service = session_prompt_service

    def submit_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """提交学员回复并一次性返回 AI 客户回复。"""

        return self.handle_turn(session_id, request)

    def stream_turn(self, session_id: str, request: TrainingTurnRequest) -> Iterator[str]:
        """提交学员回复并返回 SSE 流。"""

        try:
            start_perf = time.perf_counter()
            session = self.require_session(session_id)
            started_at = utc_now_text()
            round_no = self.repository.next_round_no(session_id)
            logger.info(
                "[销售训练][流式轮次] 开始处理 会话编号=%s 轮次=%s 模型档位=%s 学员输入长度=%s 输入预览=%s",
                session_id,
                round_no,
                request.model_mode or "默认",
                len(request.message or ""),
                self.short_text(request.message),
            )
            self.repository.add_turn(
                session_id=session_id,
                role="trainee",
                content=request.message,
                round_no=round_no,
                response_mode="stream",
                started_at=started_at,
                submitted_at=started_at,
            )
            evidence = self.turn_evidence(session, request.message)
            logger.info(
                "[销售训练][流式轮次] 本轮检索完成 会话编号=%s 轮次=%s 证据数量=%s 命中切片=%s",
                session_id,
                round_no,
                len(evidence),
                self.join_values(item.get("chunk_id") for item in evidence),
            )
            yield self.sse("retrieval_done", {"retrieved_chunk_ids": [item["chunk_id"] for item in evidence], "evidence": evidence})

            chunks: list[str] = []
            for chunk in self.stream_customer_reply(session, request.message, evidence, model_mode=request.model_mode):
                chunks.append(chunk)
                yield self.sse("customer_delta", {"content": chunk})

            customer_reply = "".join(chunks).strip() or self.fallback_customer_reply(evidence)
            response = self.finish_customer_turn(
                session=session,
                round_no=round_no,
                customer_reply=customer_reply,
                response_mode="stream",
                evidence=evidence,
                started_at=started_at,
                start_perf=start_perf,
            )
            yield self.sse(
                "stage_decision",
                {"stage_status": response.stage_status, "session_status": response.session_status},
            )
            yield self.sse("turn_done", response.model_dump())
            logger.info(
                "[销售训练][流式轮次] 处理完成 会话编号=%s 轮次=%s 回复长度=%s 总耗时秒=%s",
                session_id,
                round_no,
                len(customer_reply),
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
        except Exception as exc:
            logger.error("[销售训练] 流式训练轮次失败 会话编号=%s 错误=%s", session_id, exc, exc_info=True)
            yield self.sse("error", {"error": str(exc)})

    def handle_turn(self, session_id: str, request: TrainingTurnRequest) -> TrainingTurnResponse:
        """处理一次训练轮次，一次性接口完整生成 AI 客户回复后再返回 JSON。"""

        session = self.require_session(session_id)
        start_perf = time.perf_counter()
        started_at = utc_now_text()
        round_no = self.repository.next_round_no(session_id)
        response_mode = self.normalize_response_mode(request.response_mode)
        logger.info(
            "[销售训练][一次性轮次] 开始处理 会话编号=%s 轮次=%s 回复模式=%s 模型档位=%s 学员输入长度=%s 输入预览=%s",
            session_id,
            round_no,
            response_mode,
            request.model_mode or "默认",
            len(request.message or ""),
            self.short_text(request.message),
        )
        self.repository.add_turn(
            session_id=session_id,
            role="trainee",
            content=request.message,
            round_no=round_no,
            response_mode=response_mode,
            started_at=started_at,
            submitted_at=started_at,
        )
        evidence = self.turn_evidence(session, request.message)
        logger.info(
            "[销售训练][一次性轮次] 本轮检索完成 会话编号=%s 轮次=%s 证据数量=%s 命中切片=%s",
            session_id,
            round_no,
            len(evidence),
            self.join_values(item.get("chunk_id") for item in evidence),
        )
        customer_reply = self.generate_customer_reply(session, request.message, evidence, model_mode=request.model_mode)
        return self.finish_customer_turn(
            session=session,
            round_no=round_no,
            customer_reply=customer_reply,
            response_mode=response_mode,
            evidence=evidence,
            started_at=started_at,
            start_perf=start_perf,
        )

    def finish_customer_turn(
            self,
            *,
            session: dict[str, Any],
            round_no: int,
            customer_reply: str,
            response_mode: str,
            evidence: list[dict[str, Any]],
            started_at: str,
            start_perf: float,
    ) -> TrainingTurnResponse:
        """保存 AI 客户回复并更新本轮状态。"""

        session_status = "active"
        stage_status = "active"
        if round_no >= int(session["round_limit"]):
            session_status = "scoring"
            stage_status = "round_limit_reached"
            self.repository.update_session_status(session["session_id"], status=session_status)

        now = utc_now_text()
        response_seconds = round(max(0.0, time.perf_counter() - start_perf), 3)
        coach_analysis = self.build_turn_coach_analysis(session, round_no, evidence)
        self.repository.add_turn(
            session_id=session["session_id"],
            role="customer",
            content=customer_reply,
            round_no=round_no,
            response_mode=response_mode,
            stage_no=1,
            started_at=started_at,
            submitted_at=now,
            response_seconds=response_seconds,
            retrieved_chunk_ids=[item["chunk_id"] for item in evidence],
            retrieved_evidence=evidence,
            stage_decision={"stage_status": stage_status, "session_status": session_status},
            coach_analysis=coach_analysis,
        )
        logger.info(
            "[销售训练] 训练轮次完成 会话编号=%s 轮次=%s 状态=%s 回复模式=%s 回复长度=%s 证据数量=%s 耗时秒=%s",
            session["session_id"],
            round_no,
            session_status,
            response_mode,
            len(customer_reply or ""),
            len(evidence),
            response_seconds,
        )
        return TrainingTurnResponse(
            customer_reply=customer_reply,
            current_stage_no=1,
            stage_status=stage_status,
            session_status=session_status,
            retrieved_chunk_ids=[item["chunk_id"] for item in evidence],
            coach_analysis=coach_analysis,
            response_seconds=response_seconds,
        )

    def turn_evidence(self, session: dict[str, Any], message: str) -> list[dict[str, Any]]:
        """为某一轮学员回复检索训练证据。"""

        profile = self.require_role_profile(session["profile_id"])
        query = f"{message}\n{profile.get('scenario_description') or ''}"
        logger.info(
            "[销售训练][本轮证据] 构造检索文本 会话编号=%s 角色编号=%s 学员输入预览=%s 检索文本长度=%s",
            session["session_id"],
            session["profile_id"],
            self.short_text(message),
            len(query),
        )
        return self.search_training_evidence(query, visibility=("visible", "hidden"), k=5)

    def search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict[str, Any]]:
        """检索训练证据库，并过滤学员不可直接看到的内容。"""

        return self.query_service.search_training_evidence(query, visibility=visibility, k=k)

    def generate_customer_reply(
            self,
            session: dict[str, Any],
            trainee_message: str,
            evidence: list[dict[str, Any]],
            *,
            model_mode: str | None,
    ) -> str:
        """一次性生成 AI 客户回复。"""

        model = get_chat_model(model_mode)
        prompt = self.customer_prompt(session, trainee_message, evidence)
        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][AI客户回复] 一次性调用开始 会话编号=%s 模型档位=%s 提示词长度=%s 证据数量=%s 学员输入预览=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
            len(evidence),
            self.short_text(trainee_message),
        )
        try:
            response = model.invoke(self.messages(prompt_manager.get("training.ai_customer_system"), prompt))
            text = self.content_text(response.content).strip()
            if text:
                logger.info(
                    "[销售训练][AI客户回复] 一次性调用完成 会话编号=%s 回复长度=%s 耗时秒=%s 回复预览=%s",
                    session["session_id"],
                    len(text),
                    round(max(0.0, time.perf_counter() - start_perf), 3),
                    self.short_text(text),
                )
                return text
            logger.warning("[销售训练][AI客户回复] 模型返回为空，使用兜底回复 会话编号=%s", session["session_id"])
            return self.fallback_customer_reply(evidence)
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户回复生成失败，使用兜底回复 会话编号=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            return self.fallback_customer_reply(evidence)

    def stream_customer_reply(
            self,
            session: dict[str, Any],
            trainee_message: str,
            evidence: list[dict[str, Any]],
            *,
            model_mode: str | None,
    ) -> Iterator[str]:
        """流式生成 AI 客户回复。"""

        model = get_chat_model(model_mode)
        prompt = self.customer_prompt(session, trainee_message, evidence)
        start_perf = time.perf_counter()
        chunk_count = 0
        char_count = 0
        logger.info(
            "[销售训练][AI客户回复] 流式调用开始 会话编号=%s 模型档位=%s 提示词长度=%s 证据数量=%s 学员输入预览=%s",
            session["session_id"],
            model_mode or "默认",
            len(prompt),
            len(evidence),
            self.short_text(trainee_message),
        )
        try:
            for chunk in model.stream(self.messages(prompt_manager.get("training.ai_customer_system"), prompt)):
                text = self.content_text(chunk.content)
                if text:
                    chunk_count += 1
                    char_count += len(text)
                    yield text
            logger.info(
                "[销售训练][AI客户回复] 流式调用完成 会话编号=%s 分片数量=%s 回复累计长度=%s 耗时秒=%s",
                session["session_id"],
                chunk_count,
                char_count,
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
        except Exception as exc:
            logger.warning(
                "[销售训练] AI客户流式生成失败，使用兜底回复 会话编号=%s 已返回分片=%s 耗时秒=%s 错误=%s",
                session["session_id"],
                chunk_count,
                round(max(0.0, time.perf_counter() - start_perf), 3),
                exc,
            )
            yield self.fallback_customer_reply(evidence)

    def customer_prompt(self, session: dict[str, Any], trainee_message: str, evidence: list[dict[str, Any]]) -> str:
        """构造每轮 AI 客户回复提示词。"""

        profile = self.require_role_profile(session["profile_id"])
        setting = self.require_goal_setting(session["setting_id"])
        turns = self.repository.list_turns(session["session_id"])[-10:]
        return self.session_prompt_service.customer_prompt(
            profile,
            setting,
            turns=turns,
            trainee_message=trainee_message,
            evidence=evidence,
        )

    def require_session(self, session_id: str) -> dict[str, Any]:
        """查询可继续对话的训练会话。"""

        session = self.repository.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="训练会话不存在")
        if session["status"] not in {"active", "scoring"}:
            raise HTTPException(status_code=400, detail=f"当前训练状态不允许继续对话：{session['status']}")
        return session

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

    def build_turn_coach_analysis(
            self,
            session: dict[str, Any],
            round_no: int,
            evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """生成每轮即时教练分析。"""

        turns = self.repository.list_turns(session["session_id"])
        trainee_turn = next((item for item in reversed(turns) if item["role"] == "trainee" and int(item["round_no"]) == round_no), None)
        trainee_text = str(trainee_turn.get("content") or "") if trainee_turn else ""
        has_question = "?" in trainee_text or "？" in trainee_text
        has_case = any(keyword in trainee_text for keyword in ("案例", "客户", "数据", "证据", "效果", "ROI", "试点"))
        has_next_step = any(keyword in trainee_text for keyword in ("下一步", "约", "试", "确认", "发您", "安排", "继续"))
        strengths: list[str] = []
        suggestions: list[str] = []
        if has_question:
            strengths.append("本轮有提问动作，能推动客户继续释放信息。")
        else:
            suggestions.append("建议先追问客户当前最核心的顾虑，避免直接进入方案介绍。")
        if has_case:
            strengths.append("本轮尝试使用证据或案例降低客户不确定感。")
        else:
            suggestions.append("可以补一句同类客户案例、数据或试点路径，让表达更可信。")
        if has_next_step:
            strengths.append("本轮有下一步推进意识。")
        else:
            suggestions.append("结尾建议给出轻量下一步，例如约 15 分钟确认需求或先做小范围验证。")
        if not strengths:
            strengths.append("表达已完成基础回应，但还需要增强销售推进动作。")
        return {
            "round_no": round_no,
            "summary": "本轮建议优先补强需求挖掘、证据化表达和下一步推进。",
            "strengths": strengths,
            "suggestions": suggestions,
            "retrieval_hint": f"本轮命中 {len(evidence)} 条训练知识，可结合命中切片补充案例化表达。",
            "next_reply_hint": "先承接客户顾虑，再追问影响范围，最后用案例或试点降低风险。",
        }

    @staticmethod
    def normalize_response_mode(response_mode: str | None) -> str:
        """统一响应模式枚举。"""

        return "blocking" if response_mode in {"blocking", "once"} else "stream"

    @staticmethod
    def sse(event: str, payload: dict) -> str:
        """把事件名和数据包装成 SSE 协议文本。"""

        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

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
    def fallback_customer_reply(evidence: list[dict[str, Any]]) -> str:
        """AI 客户回复失败时的兜底话术。"""

        return TrainingSessionPromptService.fallback_customer_reply(evidence)
