import uuid

from rag import knowledge_store
from rag.knowledge_store import KnowledgeStore


def _unique_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class FakeDictionaryRedisClient:
    """测试用 Redis 替身，只记录 JSON 缓存读写删除行为。"""

    def __init__(self):
        self.values: dict[str, object] = {}
        self.set_calls: list[tuple[str, object, int | None]] = []
        self.deleted_keys: list[str] = []

    def build_key(self, *parts: object) -> str:
        """按项目 Redis key 规则拼接测试 key。"""

        return ":".join(["test", *[str(part) for part in parts]])

    def get_json(self, key: str, default=None):
        """模拟读取 JSON 缓存。"""

        return self.values.get(key, default)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        """模拟写入 JSON 缓存。"""

        self.values[key] = value
        self.set_calls.append((key, value, ttl_seconds))
        return True

    def delete(self, *keys: str) -> int:
        """模拟删除 Redis key。"""

        self.deleted_keys.extend(keys)
        count = 0
        for key in keys:
            if key in self.values:
                count += 1
                del self.values[key]
        return count


def test_conversation_exchange_is_persisted_in_sequence(tmp_path):
    store = KnowledgeStore()
    conversation_id = _unique_id("conv_test")
    conversation = store.ensure_conversation(
        conversation_id=conversation_id,
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

    messages = store.list_recent_messages(conversation_id, limit=10)
    refreshed = store.get_conversation(conversation_id)

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["sequence_no"] for message in messages] == [1, 2]
    assert messages[1]["metadata_json"] == '{"mode": "test"}'
    assert refreshed is not None
    assert refreshed["message_count"] == 2


def test_conversations_can_be_listed_and_loaded_with_messages(tmp_path):
    store = KnowledgeStore()
    user_id = _unique_id("user_list")
    conversation_a_id = _unique_id("conv_a")
    conversation_b_id = _unique_id("conv_b")
    store.ensure_conversation(conversation_id=conversation_a_id, user_id=user_id, title="第一段会话")
    store.ensure_conversation(conversation_id=conversation_b_id, user_id=user_id, title="第二段会话")
    store.save_chat_exchange(
        conversation_id=conversation_a_id,
        user_message="扫地机器人迷路怎么办？",
        assistant_message="可以先重新建图。",
    )

    conversations, total = store.list_conversations(page=1, page_size=10, user_id=user_id)
    messages = store.list_conversation_messages(conversation_a_id)

    assert total == 2
    assert {conversation["conversation_id"] for conversation in conversations} == {conversation_a_id, conversation_b_id}
    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_conversations_can_be_filtered_by_keyword(tmp_path):
    store = KnowledgeStore()
    title_token = _unique_id("title")
    user_token = _unique_id("user")
    id_token = _unique_id("robot")
    robot_conversation_id = f"conv_{id_token}"
    report_conversation_id = _unique_id("conv_report")
    store.ensure_conversation(conversation_id=robot_conversation_id, user_id="1001", title=f"扫拖预约设置 {title_token}")
    store.ensure_conversation(conversation_id=report_conversation_id, user_id=user_token, title="使用报告查询")

    title_matches, title_total = store.list_conversations(page=1, page_size=10, keyword=title_token)
    user_matches, user_total = store.list_conversations(page=1, page_size=10, keyword=user_token)
    id_matches, id_total = store.list_conversations(page=1, page_size=10, keyword=id_token)

    assert title_total == 1
    assert title_matches[0]["conversation_id"] == robot_conversation_id
    assert user_total == 1
    assert user_matches[0]["conversation_id"] == report_conversation_id
    assert id_total == 1
    assert id_matches[0]["conversation_id"] == robot_conversation_id


def test_dictionary_items_use_redis_cache(monkeypatch, tmp_path):
    """验证字典列表第一次查库并写 Redis，后续相同查询直接命中 Redis。"""

    fake_redis = FakeDictionaryRedisClient()
    monkeypatch.setattr(knowledge_store, "get_redis_client", lambda: fake_redis)
    store = KnowledgeStore()
    fake_redis.values.clear()
    fake_redis.set_calls.clear()

    first_rows = store.list_dictionary_items("document_structure")
    cached_key = fake_redis.build_key("dictionary", "items", "document_structure")

    assert first_rows
    assert cached_key in fake_redis.values

    def fail_database_query(dictionary_code=None):
        """如果第二次没有命中 Redis，这个替身会让测试失败。"""

        raise AssertionError("字典缓存命中时不应该再次查询数据库")

    monkeypatch.setattr(store, "_list_dictionary_items_from_db", fail_database_query)
    second_rows = store.list_dictionary_items("document_structure")

    assert second_rows == first_rows


def test_upsert_dictionary_item_refreshes_redis_cache(monkeypatch, tmp_path):
    """验证新增或更新字典项后，会刷新全部字典和当前字典的 Redis 缓存。"""

    fake_redis = FakeDictionaryRedisClient()
    monkeypatch.setattr(knowledge_store, "get_redis_client", lambda: fake_redis)
    store = KnowledgeStore()
    fake_redis.values.clear()
    fake_redis.set_calls.clear()

    store.upsert_dictionary_item(
        dictionary_code="demo_dictionary",
        dictionary_name="演示字典",
        item_code="demo_item",
        item_name="演示项",
        sort_order=1,
        enabled=True,
        description="用于验证字典缓存刷新",
    )

    all_key = fake_redis.build_key("dictionary", "items", "all")
    code_key = fake_redis.build_key("dictionary", "items", "demo_dictionary")
    assert all_key in fake_redis.values
    assert code_key in fake_redis.values
    assert any(row["item_code"] == "demo_item" for row in fake_redis.values[all_key])
    assert fake_redis.values[code_key][0]["item_code"] == "demo_item"


def test_exam_session_questions_and_answers_are_persisted(tmp_path):
    store = KnowledgeStore()
    session_id = _unique_id("exam_unit")
    user_id = _unique_id("user_exam")
    session = store.create_exam_session(
        session_id=session_id,
        user_id=user_id,
        title="Java 基础测评",
        collection_name="agent",
        document_id="doc_java",
        filename="Java面试题.pdf",
        section_path="Java 基础",
        round_count=2,
        question_types=["true_false", "short_answer"],
        model_mode="low",
        metadata={"seed": 7},
    )

    first_question = store.add_exam_question(
        session_id=session["session_id"],
        round_no=1,
        source_question_id="qa_001",
        source_document_id="doc_java",
        source_filename="Java面试题.pdf",
        source_page=3,
        section_path="Java 基础",
        question_type="true_false",
        prompt="判断正误：Java 支持面向对象。",
        options=["正确", "错误"],
        correct_answer="正确",
        reference_answer="Java 支持面向对象。",
        max_score=50,
    )
    store.add_exam_question(
        session_id=session["session_id"],
        round_no=2,
        source_question_id="qa_002",
        source_document_id="doc_java",
        source_filename="Java面试题.pdf",
        source_page=4,
        section_path="Java 基础",
        question_type="short_answer",
        prompt="简述 JVM 的作用。",
        options=[],
        correct_answer="运行 Java 字节码",
        reference_answer="JVM 负责加载并运行 Java 字节码。",
        max_score=50,
    )

    answered = store.answer_exam_question(
        session_id=session["session_id"],
        exam_question_id=first_question["exam_question_id"],
        user_answer="正确",
        is_correct=True,
        score=50,
        analysis={
            "correct_answer": "正确",
            "hit_points": ["Java 支持面向对象"],
            "missing_points": [],
            "wrong_points": [],
            "comment": "回答正确。",
        },
    )
    histories, total = store.list_exam_sessions(page=1, page_size=10, user_id=user_id)
    questions = store.list_exam_questions(session["session_id"])
    refreshed_session = store.get_exam_session(session["session_id"])

    assert answered["status"] == "answered"
    assert answered["user_answer"] == "正确"
    assert total == 1
    assert histories[0]["session_id"] == session_id
    assert refreshed_session is not None
    assert refreshed_session["answered_count"] == 1
    assert refreshed_session["total_score"] == 50
    assert refreshed_session["status"] == "active"
    assert [question["round_no"] for question in questions] == [1, 2]
