"""聊天生成应用服务。

这个模块负责智能客服聊天的生成链路：
- 创建或读取 conversations 会话；
- 根据配置决定走直连 RAG 还是旧 Agent 工具链；
- 支持一次性回答和 SSE 流式回答；
- 把用户问题和助手回答保存到聊天记录表。
"""

import json
import threading
import time
from collections.abc import Iterator

from fastapi import HTTPException

from api.schemas import ChatRequest
from app.infrastructure.repositories.conversation_repository import ConversationRepository
from core.model.factory import get_chat_model_name_for_mode, normalize_chat_model_mode
from core.utils.config_handler import rag_conf
from core.utils.logger_handler import logger
from core.utils.qdrant_options import normalize_qdrant_collection_name


_agent = None  # 全局 Agent 单例；第一次聊天请求进来时才真正初始化
_agent_lock = threading.Lock()  # 初始化锁；防止多个请求同时初始化多个 Agent
_knowledge_answer_service = None  # 知识库直答服务单例；普通知识问答不再进入 Agent 工具链
_knowledge_answer_lock = threading.Lock()  # 知识直答服务初始化锁


def _get_agent():
    """获取全局 Agent 单例。"""

    global _agent

    if _agent is not None:
        return _agent

    with _agent_lock:
        if _agent is None:
            try:
                from core.agent.react_agent import ReactAgent

                _agent = ReactAgent()
            except (ImportError, RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=500, detail=f"Agent 初始化失败：{exc}") from exc

    return _agent


def _get_knowledge_answer_service():
    """获取知识库直答服务单例。"""

    global _knowledge_answer_service

    if _knowledge_answer_service is not None:
        return _knowledge_answer_service

    with _knowledge_answer_lock:
        if _knowledge_answer_service is None:
            try:
                from core.rag.services.knowledge_answer_service import KnowledgeAnswerService

                _knowledge_answer_service = KnowledgeAnswerService()
            except (ImportError, RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=500, detail=f"知识库直答服务初始化失败：{exc}") from exc

    return _knowledge_answer_service


def _get_chat_route_mode() -> str:
    """读取聊天路由模式；不配置时默认直连 RAG。"""

    mode = str(
        rag_conf.get("chat_route_mode")
        or rag_conf.get("knowledge_chat_mode")
        or "direct_rag"
    ).strip().lower()
    return mode.replace("-", "_")


def _should_use_direct_rag(message: str) -> tuple[bool, str]:
    """判断当前问题是否走知识库直答。"""

    mode = _get_chat_route_mode()
    if mode in {"agent", "react_agent", "legacy_agent"}:
        return False, "配置 chat_route_mode=agent，使用 Agent 工具链"

    return True, f"配置 chat_route_mode={mode}，使用直连 RAG"


def _build_conversation_title(message: str) -> str:
    """用首条用户问题生成一个简短会话标题。"""

    return message.strip().replace("\n", " ")[:40] or "新会话"


def _prepare_chat_conversation(request: ChatRequest) -> tuple[str, list[dict]]:
    """创建或读取会话，并返回 conversation_id 和最近历史。"""

    repository = ConversationRepository()
    conversation = repository.ensure_conversation(
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        title=_build_conversation_title(request.message),
    )
    conversation_id = conversation["conversation_id"]
    history = repository.list_recent_messages(conversation_id, limit=20)
    return conversation_id, history


def _save_chat_exchange(
        *,
        conversation_id: str,
        message: str,
        answer: str,
        model_name: str | None = None,
        metadata: dict | None = None,
) -> None:
    """把一轮用户问题和助手回答保存到业务数据库会话历史。"""

    repository = ConversationRepository()
    repository.save_chat_exchange(
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=answer,
        model_name=model_name,
        metadata=metadata,
    )


def _stream_agent(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
        trace_id: str | None = None,
) -> Iterator[str]:
    """把 Agent token 流转换成浏览器能识别的 SSE 文本流。"""

    try:
        stream_start_time = time.perf_counter()
        first_token_ms: float | None = None
        selected_model_mode = normalize_chat_model_mode(None)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        logger.info(
            "[接口] Agent流式输出开始 追踪编号=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
            trace_id,
            user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
            message,
        )
        meta_payload = json.dumps(
            {
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
            },
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        chunks: list[str] = []
        for chunk in _get_agent().execute_stream(
                message,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history or [],
        ):
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(stream_start_time)
                metric_payload = json.dumps({"first_token_ms": first_token_ms}, ensure_ascii=False)
                yield f"event: metric\ndata: {metric_payload}\n\n"
            chunks.append(chunk)
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"event: chunk\ndata: {payload}\n\n"

        answer = "".join(chunks).strip()
        total_ms = _elapsed_ms(stream_start_time)
        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                model_name=selected_model_name,
                metadata={
                    "mode": "stream",
                    "trace_id": trace_id,
                    "model_mode": selected_model_mode,
                    "model_name": selected_model_name,
                    "first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                },
            )

        logger.info(
            "[接口] Agent流式输出完成 追踪编号=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s",
            trace_id,
            user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
        )
        done_payload = json.dumps(
            {
                "done": True,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
            },
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"
    except (HTTPException, RuntimeError, ValueError) as exc:
        logger.error("[接口] Agent流式输出失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


def _stream_direct_rag(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
        model_mode: str | None = None,
        collection_name: str | None = None,
        trace_id: str | None = None,
) -> Iterator[str]:
    """把知识库直答 token 流转换成 SSE 文本流。"""

    try:
        stream_start_time = time.perf_counter()
        first_token_ms: float | None = None
        selected_model_mode = normalize_chat_model_mode(model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        selected_collection_name = normalize_qdrant_collection_name(collection_name)
        logger.info(
            "[聊天路由] 流式接口进入知识库直答 追踪编号=%s 用户编号=%s 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 问题=%s",
            trace_id,
            user_id,
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
            message,
        )

        meta_payload = json.dumps(
            {
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "mode": "direct_rag",
                "collection_name": selected_collection_name,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
            },
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        chunks: list[str] = []
        for chunk in _get_knowledge_answer_service().stream_answer(
            message,
            history=history or [],
            model_mode=selected_model_mode,
            collection_name=selected_collection_name,
            trace_id=trace_id,
        ):
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(stream_start_time)
                metric_payload = json.dumps({"first_token_ms": first_token_ms}, ensure_ascii=False)
                yield f"event: metric\ndata: {metric_payload}\n\n"
            chunks.append(chunk)
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"event: chunk\ndata: {payload}\n\n"

        answer = "".join(chunks).strip()
        total_ms = _elapsed_ms(stream_start_time)

        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                model_name=selected_model_name,
                metadata={
                    "mode": "direct_rag_stream",
                    "trace_id": trace_id,
                    "model_mode": selected_model_mode,
                    "model_name": selected_model_name,
                    "collection_name": selected_collection_name,
                    "first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                },
            )

        logger.info(
            "[聊天路由] 知识直答流式完成 追踪编号=%s 用户编号=%s 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s",
            trace_id,
            user_id,
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
        )

        done_payload = json.dumps(
            {
                "done": True,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "collection_name": selected_collection_name,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
            },
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"
    except (HTTPException, RuntimeError, ValueError) as exc:
        logger.error("[聊天路由] 知识直答流式失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


def _elapsed_ms(start_time: float) -> float:
    """计算从 start_time 到当前时间的毫秒耗时。"""

    return (time.perf_counter() - start_time) * 1000
