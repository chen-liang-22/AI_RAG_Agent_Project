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


def test_exam_session_questions_and_answers_are_persisted(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    session = store.create_exam_session(
        session_id="exam_unit",
        user_id="user_exam",
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
    histories, total = store.list_exam_sessions(page=1, page_size=10, user_id="user_exam")
    questions = store.list_exam_questions(session["session_id"])
    refreshed_session = store.get_exam_session(session["session_id"])

    assert answered["status"] == "answered"
    assert answered["user_answer"] == "正确"
    assert total == 1
    assert histories[0]["session_id"] == "exam_unit"
    assert refreshed_session is not None
    assert refreshed_session["answered_count"] == 1
    assert refreshed_session["total_score"] == 50
    assert refreshed_session["status"] == "active"
    assert [question["round_no"] for question in questions] == [1, 2]
