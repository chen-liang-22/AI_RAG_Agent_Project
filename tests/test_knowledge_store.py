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
