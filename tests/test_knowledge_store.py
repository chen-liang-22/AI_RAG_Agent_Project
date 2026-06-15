from rag.knowledge_store import KnowledgeStore


def test_conversation_exchange_is_persisted_in_sequence(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    conversation = store.ensure_conversation(
        conversation_id="conv_test",
        user_id="user_1",
        title="测试会话",
        metadata={"source": "unit-test"},
    )

    store.save_chat_exchange(
        conversation_id=conversation["conversation_id"],
        user_message="你好",
        assistant_message="你好，有什么可以帮你？",
        metadata={"mode": "test"},
    )

    messages = store.list_recent_messages("conv_test", limit=10)
    refreshed = store.get_conversation("conv_test")

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["sequence_no"] for message in messages] == [1, 2]
    assert messages[1]["metadata_json"] == '{"mode": "test"}'
    assert refreshed is not None
    assert refreshed["message_count"] == 2


def test_conversations_can_be_listed_and_loaded_with_messages(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    store.ensure_conversation(conversation_id="conv_a", user_id="user_1", title="第一段会话")
    store.ensure_conversation(conversation_id="conv_b", user_id="user_2", title="第二段会话")
    store.save_chat_exchange(
        conversation_id="conv_a",
        user_message="扫地机器人迷路怎么办？",
        assistant_message="可以先重新建图。",
    )

    conversations, total = store.list_conversations(page=1, page_size=10)
    messages = store.list_conversation_messages("conv_a")

    assert total == 2
    assert {conversation["conversation_id"] for conversation in conversations} == {"conv_a", "conv_b"}
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_conversations_can_be_filtered_by_keyword(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    store.ensure_conversation(conversation_id="conv_robot", user_id="1001", title="扫拖预约设置")
    store.ensure_conversation(conversation_id="conv_report", user_id="1002", title="使用报告查询")

    title_matches, title_total = store.list_conversations(page=1, page_size=10, keyword="预约")
    user_matches, user_total = store.list_conversations(page=1, page_size=10, keyword="1002")
    id_matches, id_total = store.list_conversations(page=1, page_size=10, keyword="robot")

    assert title_total == 1
    assert title_matches[0]["conversation_id"] == "conv_robot"
    assert user_total == 1
    assert user_matches[0]["conversation_id"] == "conv_report"
    assert id_total == 1
    assert id_matches[0]["conversation_id"] == "conv_robot"
