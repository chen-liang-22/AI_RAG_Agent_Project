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

    # 创建会话仓储对象，用于访问业务数据库中的会话和消息记录。
    repository = ConversationRepository()
    # 根据请求中的会话编号查找已有会话；没有会话编号时创建新会话。
    conversation = repository.ensure_conversation(
        # 传入前端携带的会话编号，用于继续已有对话。
        conversation_id=request.conversation_id,
        # 传入当前用户编号，确保会话归属到正确用户。
        user_id=request.user_id,
        # 使用用户问题生成默认标题，便于新会话在列表中展示。
        title=_build_conversation_title(request.message),
    )
    # 从会话记录中取出最终可用的会话编号，兼容新建和已有会话两种场景。
    conversation_id = conversation["conversation_id"]
    # 查询当前会话最近 20 条消息，作为本轮聊天的上下文历史。
    history = repository.list_recent_messages(conversation_id, limit=20)
    # 返回会话编号和历史消息，供上层聊天流程继续生成回答。
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
        # 用户当前输入的问题文本，是 Agent 执行任务的核心输入。
        message: str,
        # 当前请求用户编号，可为空，用于日志记录和下游业务归属。
        user_id: str | None = None,
        # 当前会话编号，可为空，用于关联上下文和保存聊天历史。
        conversation_id: str | None = None,
        # 最近历史消息列表，可为空，用于让 Agent 结合上下文生成回答。
        history: list[dict] | None = None,
        # 本次请求追踪编号，可为空，用于串联接口、Agent 和保存链路日志。
        trace_id: str | None = None,
) -> Iterator[str]:
    """把 Agent token 流转换成浏览器能识别的 SSE 文本流。"""

    # 捕获 Agent 流式执行中的业务异常，并统一转换为 SSE error 事件。
    try:
        # 记录流式输出开始时间，用于计算首 token 延迟和总耗时。
        stream_start_time = time.perf_counter()
        # 初始化首 token 耗时，未收到任何 token 前保持为空。
        first_token_ms: float | None = None
        # 使用默认聊天模型模式，Agent 流当前不从请求参数单独选择模型模式。
        selected_model_mode = normalize_chat_model_mode(None)
        # 根据模型模式获取实际模型名称，用于日志、前端元信息和历史保存。
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        # 记录 Agent 流式输出开始日志，便于排查请求入口和模型选择。
        logger.info(
            # 日志模板记录追踪编号、用户编号、会话编号、模型信息和原始问题。
            "[接口] Agent流式输出开始 追踪编号=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
            # 写入本次请求追踪编号。
            trace_id,
            # 写入当前用户编号。
            user_id,
            # 写入当前会话编号。
            conversation_id,
            # 写入本次选择的模型模式。
            selected_model_mode,
            # 写入本次实际使用的模型名称。
            selected_model_name,
            # 写入用户原始问题。
            message,
        )
        # 构造 SSE meta 事件的数据体，先把基础上下文发给前端。
        meta_payload = json.dumps(
            # meta 事件包含本次流式响应的追踪、会话和模型信息。
            {
                # 返回追踪编号，方便前端和日志系统对应同一次请求。
                "trace_id": trace_id,
                # 返回会话编号，方便前端绑定当前聊天窗口。
                "conversation_id": conversation_id,
                # 返回模型模式，方便前端展示或调试。
                "model_mode": selected_model_mode,
                # 返回模型名称，方便前端展示实际生成模型。
                "model_name": selected_model_name,
            },
            # 保留中文字符不转义，方便浏览器和日志直接查看。
            ensure_ascii=False,
        )
        # 发送 meta 事件，告诉前端本次流式会话的基础信息。
        yield f"event: meta\ndata: {meta_payload}\n\n"

        # 收集 Agent 分片内容，结束后拼成完整回答并写入会话历史。
        chunks: list[str] = []
        # 调用 Agent 流式执行接口，逐个接收生成出来的文本分片。
        for chunk in _get_agent().execute_stream(
                # 传入用户问题，作为 Agent 当前任务输入。
                message,
                # 传入用户编号，供 Agent 工具或下游链路使用。
                user_id=user_id,
                # 传入会话编号，供 Agent 工具或下游链路关联上下文。
                conversation_id=conversation_id,
                # 传入历史消息；如果为空则使用空列表，避免下游处理 None。
                history=history or [],
        ):
            # 首次收到分片时，计算首 token 延迟并只发送一次 metric 事件。
            if first_token_ms is None:
                # 计算从流式开始到首个分片返回的毫秒耗时。
                first_token_ms = _elapsed_ms(stream_start_time)
                # 构造首 token 指标数据，供前端或监控侧展示响应速度。
                metric_payload = json.dumps({"first_token_ms": first_token_ms}, ensure_ascii=False)
                # 发送 metric 事件，通知前端首 token 延迟。
                yield f"event: metric\ndata: {metric_payload}\n\n"
            # 将当前文本分片追加到列表，便于最后合并完整回答。
            chunks.append(chunk)
            # 构造 chunk 事件数据体，只包含当前增量内容。
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            # 发送 chunk 事件，把 Agent 生成的增量文本推给浏览器。
            yield f"event: chunk\ndata: {payload}\n\n"

        # 将所有文本分片合并为完整回答，并去掉首尾空白。
        answer = "".join(chunks).strip()
        # 计算从流式开始到全部输出完成的总耗时。
        total_ms = _elapsed_ms(stream_start_time)
        # 只有存在会话编号且回答非空时，才把本轮问答写入历史记录。
        if conversation_id and answer:
            # 保存用户问题、Agent 回答和本次生成元数据到业务数据库。
            _save_chat_exchange(
                # 传入会话编号，指定保存到哪一个会话下。
                conversation_id=conversation_id,
                # 传入用户原始问题，作为本轮用户消息。
                message=message,
                # 传入完整回答，作为本轮助手消息。
                answer=answer,
                # 传入实际模型名称，便于后续审计和展示。
                model_name=selected_model_name,
                # 保存本次流式生成的运行元数据。
                metadata={
                    # 标记本轮保存来源为流式 Agent 模式。
                    "mode": "stream",
                    # 保存追踪编号，方便历史消息关联接口日志。
                    "trace_id": trace_id,
                    # 保存模型模式，方便后续分析不同模式效果。
                    "model_mode": selected_model_mode,
                    # 保存模型名称，方便后续分析不同模型效果。
                    "model_name": selected_model_name,
                    # 保存首 token 耗时，方便性能统计。
                    "first_token_ms": first_token_ms,
                    # 保存总耗时，方便性能统计。
                    "total_ms": total_ms,
                },
            )

        # 记录 Agent 流式输出完成日志，说明本次流式生成正常结束。
        logger.info(
            # 日志模板记录追踪编号、用户编号、会话编号和模型信息。
            "[接口] Agent流式输出完成 追踪编号=%s 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s",
            # 写入本次请求追踪编号。
            trace_id,
            # 写入当前用户编号。
            user_id,
            # 写入当前会话编号。
            conversation_id,
            # 写入本次选择的模型模式。
            selected_model_mode,
            # 写入本次实际使用的模型名称。
            selected_model_name,
        )
        # 构造 SSE done 事件的数据体，通知前端流式输出已经完成。
        done_payload = json.dumps(
            # done 事件包含结束状态、追踪信息、模型信息和耗时指标。
            {
                # 标记流式响应已经完成。
                "done": True,
                # 返回追踪编号，方便前端定位本次请求。
                "trace_id": trace_id,
                # 返回会话编号，方便前端更新会话状态。
                "conversation_id": conversation_id,
                # 返回模型模式，方便前端展示或调试。
                "model_mode": selected_model_mode,
                # 返回模型名称，方便前端展示实际生成模型。
                "model_name": selected_model_name,
                # 返回首 token 耗时，方便前端展示响应速度。
                "first_token_ms": first_token_ms,
                # 返回总耗时，方便前端展示完整响应耗时。
                "total_ms": total_ms,
            },
            # 保留中文字符不转义，保证 SSE 数据可直接阅读。
            ensure_ascii=False,
        )
        # 发送 done 事件，通知浏览器本次 Agent 流式响应结束。
        yield f"event: done\ndata: {done_payload}\n\n"
    # 捕获接口异常、运行时异常和参数异常，避免流式连接直接中断无响应。
    except (HTTPException, RuntimeError, ValueError) as exc:
        # 记录 Agent 流式输出失败日志，并打印异常堆栈方便排查。
        logger.error("[接口] Agent流式输出失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)
        # 构造 SSE error 事件的数据体，把异常信息返回给前端。
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        # 发送 error 事件，通知浏览器本次流式响应失败。
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
