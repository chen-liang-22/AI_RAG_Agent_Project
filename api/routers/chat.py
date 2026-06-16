import json
import time

from fastapi import APIRouter, HTTPException, Query
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
from api.services import (
    _get_agent,
    _get_knowledge_answer_service,
    _get_knowledge_store,
    _prepare_chat_conversation,
    _save_chat_exchange,
    _should_use_direct_rag,
    _stream_agent,
    _stream_direct_rag,
)
from model.factory import get_chat_model_name_for_mode, normalize_chat_model_mode
from utils.logger_handler import logger
from utils.qdrant_options import normalize_qdrant_collection_name

router = APIRouter()


def _conversation_status(item_code: str) -> str:
    """从会话状态字典读取状态码，避免聊天路由里散落状态常量。"""

    return _get_knowledge_store().normalize_dictionary_code("conversation_status", item_code)


def _conversation_summary(row: dict) -> ConversationSummaryResponse:
    """把 SQLite 会话行转换成前端列表响应。"""

    return ConversationSummaryResponse(
        conversation_id=row["conversation_id"],
        user_id=row.get("user_id"),
        title=row.get("title"),
        status=row["status"],
        message_count=int(row.get("message_count") or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_message_at=row.get("last_message_at"),
    )


def _conversation_message(row: dict) -> ConversationMessageResponse:
    """把 SQLite 消息行转换成前端详情响应。"""

    metadata = _read_message_metadata(row.get("metadata_json"))
    return ConversationMessageResponse(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        sequence_no=int(row["sequence_no"]),
        role=row["role"],
        content=row["content"],
        content_type=row["content_type"],
        model_name=row.get("model_name"),
        token_count=row.get("token_count"),
        first_token_ms=_optional_float(metadata.get("first_token_ms")),
        total_ms=_optional_float(metadata.get("total_ms")),
        created_at=row["created_at"],
    )


def _read_message_metadata(metadata_json: str | None) -> dict:
    """读取消息 metadata_json，旧数据为空或格式异常时按空字典处理。"""

    if not metadata_json:
        return {}
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: object) -> float | None:
    """把 metadata 中的耗时安全转换成 float。"""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=50),
        user_id: str | None = None,
        keyword: str | None = Query(default=None, max_length=100),
) -> ConversationListResponse:
    """分页查询聊天记录列表。"""

    logger.info(
        "[接口] 查询聊天记录列表 页码=%s 每页数量=%s 用户编号=%s 关键词=%s",
        page,
        page_size,
        user_id,
        keyword,
    )
    conversations, total = _get_knowledge_store().list_conversations(
        page=page,
        page_size=page_size,
        user_id=user_id,
        keyword=keyword,
    )
    return ConversationListResponse(
        items=[_conversation_summary(row) for row in conversations],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_detail(conversation_id: str) -> ConversationDetailResponse:
    """查询单个聊天记录详情。"""

    logger.info("[接口] 查询聊天记录详情 会话编号=%s", conversation_id)
    store = _get_knowledge_store()
    conversation = store.get_conversation(conversation_id)
    if conversation is None or conversation.get("status") == _conversation_status("deleted"):
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = store.list_conversation_messages(conversation_id)
    return ConversationDetailResponse(
        conversation=_conversation_summary(conversation),
        messages=[_conversation_message(row) for row in messages],
    )


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(conversation_id: str) -> ConversationDeleteResponse:
    """删除聊天记录。"""

    logger.info("[接口] 删除聊天记录 会话编号=%s", conversation_id)
    deleted = _get_knowledge_store().delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在或已删除")

    return ConversationDeleteResponse(status=_conversation_status("deleted"), conversation_id=conversation_id)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """一次性聊天接口。

    调用流程：
    1. 前端发送 POST `/chat`，请求体包含 message 和 user_id。
    2. 后端调用 `ReactAgent.execute()`。
    3. Agent 内部可能会调用 RAG、天气、用户数据等工具。
    4. 后端等待 Agent 全部执行完，只取最后一条 AIMessage 作为最终回答。
    5. 返回 JSON：`{"answer": "...完整回答..."}`

    这个接口适合“不关心首 token 速度，只想一次拿到完整结果”的场景。
    """

    request_start_time = time.perf_counter()
    conversation_id, history = _prepare_chat_conversation(request)
    selected_model_mode = normalize_chat_model_mode(request.model_mode)
    selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
    selected_collection_name = normalize_qdrant_collection_name(request.collection_name)
    logger.info(
        "[性能] 聊天请求准备完成 模式=非流式 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 历史消息数=%s",
        conversation_id,
        selected_collection_name,
        selected_model_mode,
        selected_model_name,
        _elapsed_ms(request_start_time),
        len(history),
    )
    logger.info(
        f"[接口] 非流式聊天请求 用户编号={request.user_id} "
        f"会话编号={conversation_id} 模型模式={selected_model_mode} 模型名称={selected_model_name} 问题={request.message}"
    )  # 标记本次调用的是一次性接口

    use_direct_rag, route_reason = _should_use_direct_rag(request.message)
    if use_direct_rag:
        logger.info(
            "[聊天路由] 非流式接口路由结果=知识直答 原因=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
            route_reason,
            request.user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
            request.message,
        )
        answer = _get_knowledge_answer_service().answer(
            request.message,
            history=history,
            model_mode=request.model_mode,
            collection_name=selected_collection_name,
        )
        total_ms = _elapsed_ms(request_start_time)
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
        logger.info(
            "[性能] 聊天请求完成 模式=非流式 路由=知识直答 会话编号=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f",
            conversation_id,
            selected_model_mode,
            selected_model_name,
            total_ms,
        )
        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            first_token_ms=total_ms,
            total_ms=total_ms,
        )

    logger.info(
        "[聊天路由] 非流式接口路由结果=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
        route_reason,
        request.user_id,
        conversation_id,
        selected_model_mode,
        selected_model_name,
        request.message,
    )
    answer = _get_agent().execute(
        request.message,
        user_id=request.user_id,
        conversation_id=conversation_id,
        history=history,
    )  # 阻塞等待 Agent 完整生成最终回答
    total_ms = _elapsed_ms(request_start_time)
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
    logger.info(
        "[性能] 聊天请求完成 模式=非流式 路由=Agent工具链 会话编号=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f",
        conversation_id,
        selected_model_mode,
        selected_model_name,
        total_ms,
    )
    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        first_token_ms=total_ms,
        total_ms=total_ms,
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口。

    调用流程：
    1. 前端发送 POST `/chat/stream`，请求体同样包含 message 和 user_id。
    2. 后端返回 `StreamingResponse`，浏览器不会等待完整回答结束。
    3. Agent 每生成一段 AI token，`_stream_agent()` 就包装成一个 SSE chunk 事件。
    4. 前端通过 `ReadableStream.getReader()` 持续读取 chunk，并实时追加到页面。

    为什么这里仍然用 POST：
    - 请求体里有 message、user_id，未来还可能有历史上下文、模式配置等结构化参数。
    - 原生 EventSource 只能 GET，不方便携带复杂 JSON 请求体。
    - fetch + ReadableStream 可以同时保留 POST 语义和 SSE 文本流格式。

    响应头说明：
    - Cache-Control: no-cache, no-transform
      禁止缓存和内容转换，避免代理层把流式响应攒完整再发。
    - Connection: keep-alive
      明确告诉客户端这是一个持续连接。
    - X-Accel-Buffering: no
      Nginx 场景下关闭响应缓冲；本地开发也保留，便于以后部署。
    """

    # 记录请求进入时间，用来统计“从接口收到请求到返回流”的准备耗时。
    request_start_time = time.perf_counter()

    # 确保本次聊天有 conversation_id，并从 SQLite 读取最近的历史消息。
    # 如果前端没有传 conversation_id，这里会创建一个新的会话。
    conversation_id, history = _prepare_chat_conversation(request)
    selected_model_mode = normalize_chat_model_mode(request.model_mode)
    selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
    selected_collection_name = normalize_qdrant_collection_name(request.collection_name)

    # 打印接口准备耗时和历史消息数量，方便判断慢点是否出现在会话读取阶段。
    logger.info(
        "[性能] 聊天请求准备完成 模式=流式 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 历史消息数=%s",
        conversation_id,
        selected_collection_name,
        selected_model_mode,
        selected_model_name,
        _elapsed_ms(request_start_time),
        len(history),
    )
    logger.info(
        f"[接口] 流式聊天请求 用户编号={request.user_id} "
        f"会话编号={conversation_id} 模型模式={selected_model_mode} 模型名称={selected_model_name} 问题={request.message}"
    )  # 标记本次调用的是流式接口

    # 根据配置决定走哪条聊天链路：
    # - direct_rag：普通知识问答直接走 RAG 检索 + 最终回答模型。
    # - agent：走 ReAct Agent，让模型自己决定是否调用工具。
    use_direct_rag, route_reason = _should_use_direct_rag(request.message)

    # 当前项目默认走 direct_rag，这条链路更短，首 token 更快。
    if use_direct_rag:
        logger.info(
            "[聊天路由] 流式接口路由结果=知识直答 原因=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
            route_reason,
            request.user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
            request.message,
        )

        # StreamingResponse 接收一个可迭代对象。
        # _stream_direct_rag() 会不断 yield SSE 文本块，FastAPI 会边生成边发给前端。
        return StreamingResponse(
            _stream_direct_rag(
                request.message,  # 用户本轮问题。
                user_id=request.user_id,  # 用户编号，后续工具或画像逻辑可能会用到。
                conversation_id=conversation_id,  # 当前会话编号，用于保存聊天历史。
                history=history,  # 最近历史消息，会传给 RAG 最终回答模型做上下文参考。
                model_mode=selected_model_mode,  # 当前回答模型档位。
                collection_name=selected_collection_name,  # 当前聊天检索的 Qdrant collection。
            ),
            media_type="text/event-stream",  # 告诉浏览器这是 SSE 文本流，不是普通 JSON。
            headers={
                # 禁止浏览器或代理缓存响应，避免流式内容被攒到最后才显示。
                "Cache-Control": "no-cache, no-transform",
                # 保持 HTTP 连接不断开，方便服务端持续推送 chunk。
                "Connection": "keep-alive",
                # Nginx 反向代理场景下关闭缓冲，部署时很有用。
                "X-Accel-Buffering": "no",
            },
        )

    # 如果配置为 agent 模式，会走到这里。
    # Agent 链路更灵活，但通常比 direct_rag 慢，因为模型需要判断工具调用。
    logger.info(
        "[聊天路由] 流式接口路由结果=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
        route_reason,
        request.user_id,
        conversation_id,
        selected_model_mode,
        selected_model_name,
        request.message,
    )

    # 提前初始化 Agent。
    # 这样如果 Agent 初始化失败，可以在 StreamingResponse 开始前直接返回 HTTP 500。
    _get_agent()

    # 返回 Agent 的 SSE 流。
    # _stream_agent() 内部会调用 ReactAgent.execute_stream() 并把 token 包装成 SSE 事件。
    return StreamingResponse(
        _stream_agent(
            request.message,  # 用户本轮问题。
            user_id=request.user_id,  # 用户编号，传给 Agent 工具链。
            conversation_id=conversation_id,  # 当前会话编号，用于保存聊天历史。
            history=history,  # 最近历史消息，给 Agent 保持上下文。
        ),
        media_type="text/event-stream",  # 告诉浏览器这是 SSE 文本流。
        headers={
            "Cache-Control": "no-cache, no-transform",  # 不缓存、不转换内容，减少代理缓冲风险。
            "Connection": "keep-alive",  # 保持连接不断开，适合长响应。
            "X-Accel-Buffering": "no",  # Nginx 代理时关闭响应缓冲。
        },
    )


def _elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000

@router.post("/debug/retrieve")
def debug_retrieve(request: DebugRetrieveRequest) -> dict:
    """调试 RAG 检索链路。

    这个接口不会生成最终回答，只展示检索阶段的信息：
    - intent_analyze 识别到的意图
    - query_rewrite 生成的子查询
    - metadata filter
    - rerank 后的候选资料和分数

    开发阶段可以用它判断“为什么这个问题召回了这些资料”。
    生产环境如果不想暴露调试信息，可以后续加开关或权限。
    """

    logger.info(f"[接口] 检索调试请求 问题={request.query}")

    try:
        from rag.rag_service import RagSummarizeService  # 延迟导入，避免普通接口启动时就初始化向量库

        rag_service = RagSummarizeService()
        return rag_service.debug_retrieve(request.query)
    except Exception as exc:
        logger.error(f"[接口] 检索调试失败 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG retrieve debug failed: {exc}") from exc
