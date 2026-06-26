"""V2 智能客服接口。"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDeleteResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    DebugRetrieveRequest,
)
from app_v2.application.chat_service import ChatApplicationService

router = APIRouter(tags=["V2 智能客服"])


def _service() -> ChatApplicationService:
    """创建聊天应用服务。"""

    return ChatApplicationService()


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    user_id: str | None = None,
    keyword: str | None = Query(default=None, max_length=100),
) -> ConversationListResponse:
    """分页查询聊天记录列表。"""

    return _service().list_conversations(page=page, page_size=page_size, user_id=user_id, keyword=keyword)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_detail(conversation_id: str) -> ConversationDetailResponse:
    """查询单个聊天记录详情。"""

    return _service().get_conversation_detail(conversation_id)


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(conversation_id: str) -> ConversationDeleteResponse:
    """删除聊天记录。"""

    return _service().delete_conversation(conversation_id)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """一次性聊天接口。"""

    return _service().chat_once(request)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口。"""

    return _service().chat_stream(request)


@router.post("/debug/retrieve")
def debug_retrieve(request: DebugRetrieveRequest) -> dict:
    """调试 RAG 检索链路。"""

    return _service().debug_retrieve(request)
