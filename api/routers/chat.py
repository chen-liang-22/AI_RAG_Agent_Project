import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, DebugRetrieveRequest
from api.services import (
    _get_agent,
    _get_knowledge_answer_service,
    _prepare_chat_conversation,
    _save_chat_exchange,
    _should_use_direct_rag,
    _stream_agent,
    _stream_direct_rag,
)
from utils.logger_handler import logger

router = APIRouter()


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
    logger.info(
        "[性能] 聊天请求准备完成 模式=非流式 会话编号=%s 耗时毫秒=%.2f 历史消息数=%s",
        conversation_id,
        _elapsed_ms(request_start_time),
        len(history),
    )
    logger.info(
        f"[接口] 非流式聊天请求 用户编号={request.user_id} "
        f"会话编号={conversation_id} 问题={request.message}"
    )  # 标记本次调用的是一次性接口

    use_direct_rag, route_reason = _should_use_direct_rag(request.message)
    if use_direct_rag:
        logger.info(
            "[聊天路由] 非流式接口路由结果=知识直答 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
            route_reason,
            request.user_id,
            conversation_id,
            request.message,
        )
        answer = _get_knowledge_answer_service().answer(request.message, history=history)
        _save_chat_exchange(
            conversation_id=conversation_id,
            message=request.message,
            answer=answer,
            metadata={"mode": "direct_rag_once", "route_reason": route_reason},
        )
        logger.info(
            "[性能] 聊天请求完成 模式=非流式 路由=知识直答 会话编号=%s 耗时毫秒=%.2f",
            conversation_id,
            _elapsed_ms(request_start_time),
        )
        return ChatResponse(answer=answer, conversation_id=conversation_id)

    logger.info(
        "[聊天路由] 非流式接口路由结果=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
        route_reason,
        request.user_id,
        conversation_id,
        request.message,
    )
    answer = _get_agent().execute(
        request.message,
        user_id=request.user_id,
        conversation_id=conversation_id,
        history=history,
    )  # 阻塞等待 Agent 完整生成最终回答
    _save_chat_exchange(
        conversation_id=conversation_id,
        message=request.message,
        answer=answer,
        metadata={"mode": "agent_once", "route_reason": route_reason},
    )
    logger.info(
        "[性能] 聊天请求完成 模式=非流式 路由=Agent工具链 会话编号=%s 耗时毫秒=%.2f",
        conversation_id,
        _elapsed_ms(request_start_time),
    )
    return ChatResponse(answer=answer, conversation_id=conversation_id)


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

    request_start_time = time.perf_counter()
    conversation_id, history = _prepare_chat_conversation(request)
    logger.info(
        "[性能] 聊天请求准备完成 模式=流式 会话编号=%s 耗时毫秒=%.2f 历史消息数=%s",
        conversation_id,
        _elapsed_ms(request_start_time),
        len(history),
    )
    logger.info(
        f"[接口] 流式聊天请求 用户编号={request.user_id} "
        f"会话编号={conversation_id} 问题={request.message}"
    )  # 标记本次调用的是流式接口

    use_direct_rag, route_reason = _should_use_direct_rag(request.message)
    if use_direct_rag:
        logger.info(
            "[聊天路由] 流式接口路由结果=知识直答 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
            route_reason,
            request.user_id,
            conversation_id,
            request.message,
        )
        return StreamingResponse(
            _stream_direct_rag(
                request.message,
                user_id=request.user_id,
                conversation_id=conversation_id,
                history=history,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    logger.info(
        "[聊天路由] 流式接口路由结果=Agent工具链 原因=%s 用户编号=%s 会话编号=%s 问题=%s",
        route_reason,
        request.user_id,
        conversation_id,
        request.message,
    )
    _get_agent()  # 提前初始化 Agent；初始化失败时可以在返回流之前直接报错
    return StreamingResponse(
        _stream_agent(
            request.message,
            user_id=request.user_id,
            conversation_id=conversation_id,
            history=history,
        ),  # 把生成器交给 FastAPI 持续输出
        media_type="text/event-stream",  # 告诉浏览器这是 SSE 文本流
        headers={
            "Cache-Control": "no-cache, no-transform",  # 不缓存、不转换内容，减少代理缓冲风险
            "Connection": "keep-alive",  # 保持连接不断开，适合长响应
            "X-Accel-Buffering": "no",  # Nginx 代理时关闭响应缓冲
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
