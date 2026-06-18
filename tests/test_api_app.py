import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.routers.knowledge as knowledge_router
import api.routers.exam as exam_router
from api.main import app
from utils.path_tool import get_abs_path


def test_openapi_exposes_core_routes():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/chat" in paths
    assert "/chat/stream" in paths
    assert "/conversations" in paths
    assert "/conversations/{conversation_id}" in paths
    assert "/debug/retrieve" in paths
    assert "/knowledge/upload/preview" in paths
    assert "/knowledge/upload/recommend" in paths
    assert "/knowledge/upload/confirm" in paths
    assert "/knowledge/files" in paths
    assert "/knowledge/files/{document_id}" in paths
    assert "/knowledge/files/{document_id}/preview" in paths
    assert "/knowledge/files/reindex-all" in paths
    assert "/knowledge/files/{document_id}/reindex" in paths
    assert "/knowledge/reload" in paths
    assert "/dictionaries" in paths
    assert "/dictionaries/items" in paths
    assert "/exam/sections" in paths
    assert "/exam/sessions" in paths
    assert "/exam/sessions/{session_id}/answer" in paths
    assert "/exam/sessions/{session_id}" in paths
    assert "/exam/generate" not in paths
    assert "/exam/grade" not in paths


def test_dictionaries_return_document_structure_items():
    client = TestClient(app)

    response = client.get("/dictionaries?dictionary_code=document_structure")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["dictionary_code"] == "document_structure"
    assert {item["item_code"] for item in data[0]["items"]} == {"qa", "numbered", "text"}


def test_preview_knowledge_file_reads_text_from_registered_document(monkeypatch):
    client = TestClient(app)
    data_path = get_abs_path("data")
    filename = next(name for name in os.listdir(data_path) if name.endswith(".txt"))
    file_path = os.path.join(data_path, filename)

    class FakeKnowledgeStore:
        def get_document(self, document_id: str):
            assert document_id == "doc_test"
            return {
                "document_id": "doc_test",
                "filename": filename,
                "file_path": file_path,
                "file_type": "txt",
                "file_md5": "md5_for_test",
                "file_size": 1024,
                "status": "indexed",
                "version": 1,
                "chunk_count": 3,
                "collection_name": "agent",
                "document_type": "text",
                "split_strategy": "recursive",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "error_message": None,
            }

    monkeypatch.setattr(knowledge_router, "_get_knowledge_store", lambda: FakeKnowledgeStore())

    response = client.get("/knowledge/files/doc_test/preview?max_chars=1000")

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_id"] == "doc_test"
    assert data["preview_type"] == "text"
    assert data["content"].strip()
    assert data["page_count"] is None


def test_exam_sections_only_expose_first_level_directory(monkeypatch):
    client = TestClient(app)

    def fake_scroll_candidate_questions(*, collection_name: str | None, document_id: str | None = None, section_path: str | None = None):
        assert collection_name == "doProblems"
        assert document_id == "doc_java"
        assert section_path is None
        return [
            {"metadata": {"section_path": "10、Spring 面试题 / 1、Spring Bean 生命周期"}},
            {"metadata": {"section_path": "2、Java 集合面试题 / 1、ArrayList 和 LinkedList 的区别"}},
            {"metadata": {"section_path": "Dubbo 面试题 / 24、Dubbo 必须依赖的包有哪些？"}},
            {"metadata": {"section_path": "Dubbo 面试题 / 25、Dubbo 支持哪些协议？"}},
            {"metadata": {"section_path": "1、Java 基础面试题 / 1、JVM 的作用"}},
        ]

    monkeypatch.setattr(exam_router, "_scroll_candidate_questions", fake_scroll_candidate_questions)

    response = client.get("/exam/sections?collection_name=doProblems&document_id=doc_java")

    assert response.status_code == 200
    assert response.json()["sections"] == [
        {"section_path": "1、Java 基础面试题", "question_count": 1},
        {"section_path": "2、Java 集合面试题", "question_count": 1},
        {"section_path": "10、Spring 面试题", "question_count": 1},
        {"section_path": "Dubbo 面试题", "question_count": 2},
    ]


def test_exam_question_generation_and_grading_use_model_polishing(monkeypatch):
    model_calls: list[str] = []

    class FakeModel:
        def invoke(self, messages):
            text = messages[-1].content
            model_calls.append(text)
            if "请生成一题正式考试题" in text:
                return SimpleNamespace(
                    content=(
                        '{"prompt":"关于 JVM 的作用，下列说法正确的是哪一项？",'
                        '"options":["负责运行 Java 字节码","负责管理浏览器缓存","负责生成 SQL 索引","负责压缩图片"],'
                        '"correct_answer":"负责运行 Java 字节码"}'
                    )
                )
            return SimpleNamespace(
                content=(
                    '{"score":100,"is_correct":true,"correct_answer":"JVM 的核心作用是加载并执行 Java 字节码，'
                    '同时提供运行时环境。","hit_points":["运行 Java 字节码"],'
                    '"missing_points":[],"wrong_points":[],"comment":"回答准确，抓住了 JVM 的核心职责。"}'
                )
            )

    item = {
        "metadata": {
            "question": "JVM 的作用是什么？",
            "question_id": "qa_jvm",
            "document_id": "doc_java",
            "source_file": "Java面试题.pdf",
            "section_path": "1、Java 基础面试题 / 1、JVM 的作用",
        },
        "content": "问题：JVM 的作用是什么？\n答案：JVM 负责加载并运行 Java 字节码。",
    }
    monkeypatch.setattr(exam_router, "get_chat_model", lambda model_mode=None: FakeModel())

    question = exam_router._build_conversation_question(
        item=item,
        candidates=[item],
        question_type="single_choice",
        random_generator=__import__("random").Random(1),
        max_score=100,
        model_mode="low",
    )
    analysis = exam_router._model_answer_analysis(
        {
            "exam_question_id": "exam_q_test",
            "question_type": question["question_type"],
            "prompt": question["prompt"],
            "correct_answer_json": __import__("json").dumps(question["correct_answer"], ensure_ascii=False),
            "reference_answer": question["reference_answer"],
            "max_score": question["max_score"],
        },
        "负责运行 Java 字节码",
        "low",
    )

    assert question["prompt"] == "关于 JVM 的作用，下列说法正确的是哪一项？"
    assert question["options"] == ["负责运行 Java 字节码", "负责管理浏览器缓存", "负责生成 SQL 索引", "负责压缩图片"]
    assert question["correct_answer"] == "负责运行 Java 字节码"
    assert analysis.score == 100
    assert analysis.correct_answer == "JVM 的核心作用是加载并执行 Java 字节码，同时提供运行时环境。"
    assert "正式考试题" in model_calls[0]
    assert "阅卷老师" in model_calls[1] or "用户答案" in model_calls[1]
