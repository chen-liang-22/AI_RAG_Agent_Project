"""聊天应用服务。

这里使用外观模式封装会话、RAG 直答、Agent 工具链和 SSE 流式输出。
路由层只负责 HTTP 协议，不再直接保存会话或判断聊天链路。
"""

import json
import time
from datetime import date, datetime
from collections.abc import Iterator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDeleteResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessageResponse,
    ConversationSummaryResponse,
    DebugRetrieveRequest,
)
from app_v2.application.chat_generation_service import (
    _get_agent,
    _get_knowledge_answer_service,
    _prepare_chat_conversation,
    _save_chat_exchange,
    _should_use_direct_rag,
    _stream_agent,
    _stream_direct_rag,
)
from app_v2.infrastructure.repositories.conversation_repository import ConversationRepository
from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from model.factory import get_chat_model_name_for_mode, normalize_chat_model_mode
from utils.logger_handler import logger
from utils.qdrant_options import normalize_qdrant_collection_name


class ChatApplicationService:
    """聊天外观服务。"""

    def __init__(
            self,
            store=None,
            conversation_repository: ConversationRepository | None = None,
            dictionary_repository: DictionaryRepository | None = None,
    ):
        # store 参数保留给旧测试占位；真实状态码归一化走 V2 字典仓储。
        self.store = store
        self.conversation_repository = conversation_repository or ConversationRepository()
        self.dictionary_repository = dictionary_repository or DictionaryRepository()

    def list_conversations(self, *, page: int, page_size: int, user_id: str | None = None, keyword: str | None = None) -> ConversationListResponse:
        """分页查询聊天记录列表。"""

        logger.info(
            "[V2聊天] 查询聊天记录列表 页码=%s 每页数量=%s 用户编号=%s 关键词=%s",
            page,
            page_size,
            user_id,
            keyword,
        )
        conversations, total = self.conversation_repository.list_conversations(page=page, page_size=page_size, user_id=user_id, keyword=keyword)
        return ConversationListResponse(
            items=[self._conversation_summary(row) for row in conversations],
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_conversation_detail(self, conversation_id: str) -> ConversationDetailResponse:
        """查询单个聊天记录详情。"""

        logger.info("[V2聊天] 查询聊天记录详情 会话编号=%s", conversation_id)
        conversation = self.conversation_repository.get_conversation(conversation_id)
        if conversation is None or conversation.get("status") == self._conversation_status("deleted"):
            raise HTTPException(status_code=404, detail="会话不存在")
        messages = self.conversation_repository.list_conversation_messages(conversation_id)
        return ConversationDetailResponse(
            conversation=self._conversation_summary(conversation),
            messages=[self._conversation_message(row) for row in messages],
        )

    def delete_conversation(self, conversation_id: str) -> ConversationDeleteResponse:
        """删除聊天记录。"""

        logger.info("[V2聊天] 删除聊天记录 会话编号=%s", conversation_id)
        deleted = self.conversation_repository.delete_conversation(conversation_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在或已删除")
        return ConversationDeleteResponse(status=self._conversation_status("deleted"), conversation_id=conversation_id)

    def chat_once(self, request: ChatRequest) -> ChatResponse:
        """一次性聊天接口业务流程。"""

        request_start_time = time.perf_counter()
        conversation_id, history = _prepare_chat_conversation(request)
        selected_model_mode = normalize_chat_model_mode(request.model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        selected_collection_name = normalize_qdrant_collection_name(request.collection_name)
        logger.info(
            "[V2聊天] 非流式请求准备完成 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 历史消息数=%s",
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
            self._elapsed_ms(request_start_time),
            len(history),
        )

        use_direct_rag, route_reason = _should_use_direct_rag(request.message)
        if use_direct_rag:
            logger.info(
                "[V2聊天] 非流式路由=知识直答 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
                route_reason,
                request.user_id,
                conversation_id,
                request.message,
            )
            answer = _get_knowledge_answer_service().answer(
                request.message,
                history=history,
                model_mode=request.model_mode,
                collection_name=selected_collection_name,
            )
            total_ms = self._elapsed_ms(request_start_time)
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=request.message,
                answer=answer,
                model_name=selected_model_name,
                metadata={
                    "mode": "direct_rag_once",
                    "model_mode": selected_model_mode,
                    "model_name": selected_model_name,
                    "collection_name": selected_collection_name,
                    "route_reason": route_reason,
                    "first_token_ms": total_ms,
                    "total_ms": total_ms,
                },
            )
            return ChatResponse(answer=answer, conversation_id=conversation_id, first_token_ms=total_ms, total_ms=total_ms)

        logger.info(
            "[V2聊天] 非流式路由=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
            route_reason,
            request.user_id,
            conversation_id,
            request.message,
        )
        answer = _get_agent().execute(request.message, user_id=request.user_id, conversation_id=conversation_id, history=history)
        total_ms = self._elapsed_ms(request_start_time)
        _save_chat_exchange(
            conversation_id=conversation_id,
            message=request.message,
            answer=answer,
            model_name=selected_model_name,
            metadata={
                "mode": "agent_once",
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
                "route_reason": route_reason,
                "first_token_ms": total_ms,
                "total_ms": total_ms,
            },
        )
        return ChatResponse(answer=answer, conversation_id=conversation_id, first_token_ms=total_ms, total_ms=total_ms)

    def chat_stream(self, request: ChatRequest) -> StreamingResponse:
        """流式聊天接口业务流程。"""

        request_start_time = time.perf_counter()
        conversation_id, history = _prepare_chat_conversation(request)
        selected_model_mode = normalize_chat_model_mode(request.model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        selected_collection_name = normalize_qdrant_collection_name(request.collection_name)
        logger.info(
            "[V2聊天] 流式请求准备完成 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 历史消息数=%s",
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
            self._elapsed_ms(request_start_time),
            len(history),
        )

        use_direct_rag, route_reason = _should_use_direct_rag(request.message)
        if use_direct_rag:
            logger.info(
                "[V2聊天] 流式路由=知识直答 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
                route_reason,
                request.user_id,
                conversation_id,
                request.message,
            )
            stream = _stream_direct_rag(
                request.message,
                user_id=request.user_id,
                conversation_id=conversation_id,
                history=history,
                model_mode=selected_model_mode,
                collection_name=selected_collection_name,
            )
        else:
            logger.info(
                "[V2聊天] 流式路由=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
                route_reason,
                request.user_id,
                conversation_id,
                request.message,
            )
            _get_agent()
            stream = _stream_agent(request.message, user_id=request.user_id, conversation_id=conversation_id, history=history)
        return self._streaming_response(stream)

    def debug_retrieve(self, request: DebugRetrieveRequest) -> dict:
        """调试 RAG 检索链路。"""

        logger.info("[V2聊天] 检索调试请求 问题=%s", request.query)
        try:
            from rag.services.rag_service import RagSummarizeService

            return RagSummarizeService().debug_retrieve(request.query)
        except Exception as exc:
            logger.error("[V2聊天] 检索调试失败 错误=%s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"RAG retrieve debug failed: {exc}") from exc

    def _conversation_status(self, item_code: str) -> str:
        """从字典读取会话状态码。"""

        if self.store is not None and hasattr(self.store, "normalize_dictionary_code"):
            return self.store.normalize_dictionary_code("conversation_status", item_code)
        try:
            return self.dictionary_repository.normalize_code("conversation_status", item_code)
        except ValueError:
            return item_code

    @classmethod
    def _conversation_summary(cls, row: dict) -> ConversationSummaryResponse:
        """把数据库会话行转换成前端列表响应。"""

        return ConversationSummaryResponse(
            conversation_id=row["conversation_id"],
            user_id=row.get("user_id"),
            title=row.get("title"),
            status=row["status"],
            message_count=int(row.get("message_count") or 0),
            created_at=cls._datetime_to_text(row["created_at"]),
            updated_at=cls._datetime_to_text(row["updated_at"]),
            last_message_at=cls._datetime_to_text(row.get("last_message_at")),
        )

    @classmethod
    def _conversation_message(cls, row: dict) -> ConversationMessageResponse:
        """把数据库消息行转换成前端详情响应。"""

        metadata = cls._read_message_metadata(row.get("metadata_json"))
        return ConversationMessageResponse(
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            sequence_no=int(row["sequence_no"]),
            role=row["role"],
            content=row["content"],
            content_type=row["content_type"],
            model_name=row.get("model_name"),
            token_count=row.get("token_count"),
            first_token_ms=cls._optional_float(metadata.get("first_token_ms")),
            total_ms=cls._optional_float(metadata.get("total_ms")),
            created_at=cls._datetime_to_text(row["created_at"]),
        )

    @staticmethod
    def _datetime_to_text(value: object) -> str | None:
        """把数据库日期时间转换成接口字符串。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds", sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _read_message_metadata(metadata_json: str | None) -> dict:
        """读取消息 metadata_json。"""

        if not metadata_json:
            return {}
        try:
            parsed = json.loads(metadata_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _optional_float(value: object) -> float | None:
        """把 metadata 中的耗时安全转换成 float。"""

        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """计算耗时毫秒。"""

        return (time.perf_counter() - start_time) * 1000

    @staticmethod
    def _streaming_response(stream: Iterator[str]) -> StreamingResponse:
        """包装统一 SSE 响应头。"""

        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
