"""兼容旧导入路径的聊天服务入口。

真实实现已迁移到 ``api.services.chat_services``，保留本模块是为了让历史路由和测试平滑过渡。
"""

from api.services.chat_services import (
    _build_conversation_title,
    _elapsed_ms,
    _get_agent,
    _get_chat_route_mode,
    _get_knowledge_answer_service,
    _prepare_chat_conversation,
    _save_chat_exchange,
    _should_use_direct_rag,
    _stream_agent,
    _stream_direct_rag,
)

__all__ = [
    "_build_conversation_title",
    "_elapsed_ms",
    "_get_agent",
    "_get_chat_route_mode",
    "_get_knowledge_answer_service",
    "_prepare_chat_conversation",
    "_save_chat_exchange",
    "_should_use_direct_rag",
    "_stream_agent",
    "_stream_direct_rag",
]

