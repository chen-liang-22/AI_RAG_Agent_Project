"""V2 聊天应用服务测试。"""

from datetime import datetime

from app_v2.application.chat_service import ChatApplicationService


class FakeConversationStore:
    """测试用状态字典数据源，避免服务测试依赖真实数据库。"""

    def __init__(self):
        self.deleted_id: str | None = None
        self.conversation = {
            "conversation_id": "conv_1",
            "user_id": "user_1",
            "title": "测试会话",
            "status": "active",
            "message_count": 2,
            "created_at": datetime(2026, 1, 1, 10, 0, 0),
            "updated_at": datetime(2026, 1, 1, 10, 1, 0),
            "last_message_at": datetime(2026, 1, 1, 10, 1, 0),
        }
        self.messages = [
            {
                "message_id": "msg_1",
                "conversation_id": "conv_1",
                "sequence_no": 1,
                "role": "user",
                "content": "你好",
                "content_type": "text",
                "model_name": None,
                "token_count": None,
                "metadata_json": None,
                "created_at": datetime(2026, 1, 1, 10, 0, 0),
            },
            {
                "message_id": "msg_2",
                "conversation_id": "conv_1",
                "sequence_no": 2,
                "role": "assistant",
                "content": "你好，有什么可以帮你？",
                "content_type": "text",
                "model_name": "qwen3-max",
                "token_count": None,
                "metadata_json": '{"first_token_ms":12.5,"total_ms":88.1}',
                "created_at": datetime(2026, 1, 1, 10, 1, 0),
            },
        ]

    def list_conversations(self, *, page: int, page_size: int, user_id=None, keyword=None):
        assert page == 1
        assert page_size == 6
        return [self.conversation], 1

    def get_conversation(self, conversation_id: str):
        return self.conversation if conversation_id == "conv_1" else None

    def list_conversation_messages(self, conversation_id: str):
        assert conversation_id == "conv_1"
        return self.messages

    def delete_conversation(self, conversation_id: str):
        self.deleted_id = conversation_id
        return conversation_id == "conv_1"

    def normalize_dictionary_code(self, dictionary_code: str, item_code: str | None = None):
        assert dictionary_code == "conversation_status"
        return item_code or "active"


class FakeConversationRepository(FakeConversationStore):
    """测试用 V2 会话仓储，表达聊天记录查询已经走 repository。"""


def test_chat_service_lists_conversations_and_loads_detail():
    """聊天服务应把数据库行转换成前端响应对象。"""

    store = FakeConversationStore()
    repository = FakeConversationRepository()
    service = ChatApplicationService(store=store, conversation_repository=repository)

    page = service.list_conversations(page=1, page_size=6)
    detail = service.get_conversation_detail("conv_1")

    assert page.total == 1
    assert page.items[0].conversation_id == "conv_1"
    assert page.items[0].created_at == "2026-01-01 10:00:00"
    assert detail.messages[1].first_token_ms == 12.5
    assert detail.messages[1].total_ms == 88.1


def test_chat_service_deletes_conversation_with_dictionary_status():
    """删除会话时返回字典规范化后的 deleted 状态。"""

    store = FakeConversationStore()
    repository = FakeConversationRepository()
    service = ChatApplicationService(store=store, conversation_repository=repository)

    result = service.delete_conversation("conv_1")

    assert result.status == "deleted"
    assert result.conversation_id == "conv_1"
    assert repository.deleted_id == "conv_1"
