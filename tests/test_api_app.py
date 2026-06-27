import os
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app_v2.application.knowledge_service as v2_knowledge_service
import app_v2.application.exam_service as exam_router
from app_v2.infrastructure.repositories.exam_repository import ExamRepository
from api.main import app
from utils.path_tool import get_abs_path


def _unique_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def test_openapi_exposes_core_routes():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v2/health" in paths
    assert "/api/v2/auth/login" in paths
    assert "/api/v2/auth/me" in paths
    assert "/api/v2/auth/logout" in paths
    assert "/api/v2/chat" in paths
    assert "/api/v2/chat/stream" in paths
    assert "/api/v2/conversations" in paths
    assert "/api/v2/conversations/{conversation_id}" in paths
    assert "/api/v2/debug/retrieve" in paths
    assert "/api/v2/knowledge/upload/preview" in paths
    assert "/api/v2/knowledge/upload/recommend" in paths
    assert "/api/v2/knowledge/upload/confirm" in paths
    assert "/api/v2/knowledge/files" in paths
    assert "/api/v2/knowledge/files/{document_id}" in paths
    assert "/api/v2/knowledge/files/{document_id}/preview" in paths
    assert "/api/v2/knowledge/files/reindex-all" in paths
    assert "/api/v2/knowledge/files/{document_id}/reindex" in paths
    assert "/api/v2/knowledge/reload" in paths
    assert "/api/v2/dictionaries" in paths
    assert "/api/v2/dictionaries/items" in paths
    assert "/api/v2/exam/sections" in paths
    assert "/api/v2/exam/sessions" in paths
    assert "/api/v2/exam/sessions/{session_id}/answer" in paths
    assert "/api/v2/exam/sessions/{session_id}" in paths
    assert "/api/v2/training/knowledge/upload" in paths
    assert "/api/v2/training/knowledge/batches" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/preview" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/publish" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/rollback" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/reparse" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/versions" in paths
    assert "/api/v2/training/knowledge/batches/{batch_id}/chunks" in paths
    assert "/api/v2/training/profile-dictionaries" in paths
    assert "/api/v2/training/plans" in paths
    assert "/api/v2/training/plans/{plan_id}" in paths
    assert "delete" in paths["/api/v2/training/plans/{plan_id}"]
    assert "/api/v2/training/profiles/generate" in paths
    assert "/api/v2/training/sessions" in paths
    assert "/api/v2/training/sessions/{session_id}" in paths
    assert "/api/v2/training/sessions/{session_id}/turns" in paths
    assert "/api/v2/training/sessions/{session_id}/final-score" in paths
    assert "/internal/jobs/minio/cleanup-preview-uploads" in paths
    assert "/exam/generate" not in paths
    assert "/exam/grade" not in paths


def test_dictionaries_return_document_structure_items():
    client = TestClient(app)

    response = client.get("/api/v2/dictionaries?dictionary_code=document_structure")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["dictionary_code"] == "document_structure"
    assert {item["item_code"] for item in data[0]["items"]} == {"qa", "numbered", "text"}


def test_training_profile_dictionaries_follow_current_portrait_spec():
    client = TestClient(app)

    response = client.get("/api/v2/training/profile-dictionaries")

    assert response.status_code == 200
    data = response.json()
    groups = {group["dictionary_code"]: group for group in data}
    assert set(groups) == {
        "student_portrait",
        "wzf_customer_manager",
        "wm_ai_service",
        "overseas_bd",
        "training_source_type",
        "training_case_part",
        "training_chunk_usage",
        "training_batch_status",
    }

    student_fields = {item["item_code"]: item for item in groups["student_portrait"]["items"]}
    position_role_options = {child["item_code"] for child in student_fields["position_role"]["children"]}
    assert position_role_options == {"wzf_customer_manager", "wm_ai_service", "overseas_bd"}
    assert {child["item_code"] for child in student_fields["task_goal"]["children"]} == {
        "goal_junior",
        "goal_intermediate",
        "goal_senior",
    }
    assert student_fields["student_portrait_other"]["metadata"]["input_type"] == "text"
    status_items = {item["item_code"]: item for item in groups["training_batch_status"]["items"]}
    assert set(status_items) == {
        "parsing",
        "pending_review",
            "embedding",
            "published",
            "archived",
            "parsing_failed",
            "publish_failed",
            "deleted",
            "duplicated",
    }
    assert status_items["published"]["item_name"] == "已发布"

    overseas_fields = {item["item_code"]: item for item in groups["overseas_bd"]["items"]}
    assert "overseas_bd_high_intention_cooperation_stage" in overseas_fields
    assert {child["item_code"] for child in overseas_fields["overseas_bd_customer_type"]["children"]} == {
        "overseas_bd_customer_type_c_end",
        "overseas_bd_customer_type_b_end",
        "overseas_bd_customer_type_g_end",
    }

    case_part_names = {item["item_code"]: item["item_name"] for item in groups["training_case_part"]["items"]}
    assert case_part_names["case_profile"] == "客户背景"
    assert case_part_names["hidden_psychology"] == "客户顾虑"

    usage_names = {item["item_code"]: item["item_name"] for item in groups["training_chunk_usage"]["items"]}
    assert usage_names["visible"] == "通用知识"
    assert usage_names["scoring_only"] == "评分专用"


def test_preview_knowledge_file_reads_text_from_registered_document(monkeypatch):
    client = TestClient(app)
    data_path = get_abs_path("data")
    filename = next(name for name in os.listdir(data_path) if name.endswith(".txt"))

    class FakeDictionaryRepository:
        """测试用 V2 字典仓储，提供文档响应归一化所需字典项。"""

        def list_items(self, dictionary_code=None):
            if dictionary_code == "document_structure":
                return [{"item_code": "text", "enabled": 1}]
            if dictionary_code == "split_strategy":
                return [{"item_code": "recursive", "enabled": 1}]
            return []

    class FakeDocumentRepository:
        """测试用 V2 文档仓储，模拟 documents 表按编号查询。"""

        def __init__(self, *args, **kwargs):
            pass

        def get_document(self, document_id: str):
            assert document_id == "doc_test"
            return {
                "document_id": "doc_test",
                "filename": filename,
                "file_path": f"minio://pub/documents/doc_test/{filename}",
                "storage_type": "minio",
                "bucket_name": "pub",
                "object_name": f"documents/doc_test/{filename}",
                "public_url": f"http://127.0.0.1:9000/pub/documents/doc_test/{filename}",
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

    class FakeFileStorage:
        """测试用文件存储服务，把 MinIO 下载动作映射到 data 目录样例文件。"""

        def downloaded_temp_file(self, **kwargs):
            class TempFileContext:
                def __enter__(self):
                    return os.path.join(data_path, filename)

                def __exit__(self, exc_type, exc, traceback):
                    return False

            return TempFileContext()

    monkeypatch.setattr(v2_knowledge_service, "DocumentRepository", FakeDocumentRepository)
    monkeypatch.setattr(v2_knowledge_service, "DictionaryRepository", FakeDictionaryRepository)
    monkeypatch.setattr(v2_knowledge_service, "FileStorageAdapter", lambda: FakeFileStorage())

    response = client.get("/api/v2/knowledge/files/doc_test/preview?max_chars=1000")

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_id"] == "doc_test"
    assert data["preview_type"] == "text"
    assert data["content"].strip()
    assert data["page_count"] is None


def test_knowledge_files_excludes_training_collections_by_default(monkeypatch):
    client = TestClient(app)

    class FakeDictionaryRepository:
        """测试用 V2 字典仓储，提供文档响应归一化所需字典项。"""

        def list_items(self, dictionary_code=None):
            if dictionary_code == "document_structure":
                return [{"item_code": "text", "enabled": 1}]
            if dictionary_code == "split_strategy":
                return [{"item_code": "recursive", "enabled": 1}]
            return []

    class FakeDocumentRepository:
        """测试用 V2 文档仓储，模拟 documents 表默认排除训练资料。"""

        def __init__(self, *args, **kwargs):
            pass

        def list_documents(self, *, include_training=False):
            assert include_training is False
            return [
                {
                    "document_id": "doc_general",
                    "filename": "general.txt",
                    "file_path": "minio://pub/documents/doc_general/general.txt",
                    "storage_type": "minio",
                    "bucket_name": "pub",
                    "object_name": "documents/doc_general/general.txt",
                    "public_url": None,
                    "file_type": "txt",
                    "file_md5": "md5_general",
                    "file_size": 10,
                    "status": "indexed",
                    "version": 1,
                    "chunk_count": 1,
                    "collection_name": "agent",
                    "document_type": "text",
                    "split_strategy": "recursive",
                    "created_at": "2026-01-01 00:00:00",
                    "updated_at": "2026-01-01 00:00:00",
                    "error_message": None,
                }
            ]

    monkeypatch.setattr(v2_knowledge_service, "DocumentRepository", FakeDocumentRepository)
    monkeypatch.setattr(v2_knowledge_service, "DictionaryRepository", FakeDictionaryRepository)

    response = client.get("/api/v2/knowledge/files")

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()] == ["doc_general"]


def test_delete_knowledge_file_uses_document_asset_service(monkeypatch):
    client = TestClient(app)
    deleted_document_ids = []

    class FakeDeleteResult:
        document_id = "doc_general"

    class FakeAssetService:
        def delete_document_asset(self, document_id):
            deleted_document_ids.append(document_id)
            return FakeDeleteResult()

    monkeypatch.setattr(v2_knowledge_service, "DocumentAssetService", lambda **kwargs: FakeAssetService())

    response = client.delete("/api/v2/knowledge/files/doc_general")

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc_general"
    assert deleted_document_ids == ["doc_general"]


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

    response = client.get("/api/v2/exam/sections?collection_name=doProblems&document_id=doc_java")

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
        "A",
        "low",
    )

    assert question["prompt"] == "关于 JVM 的作用，下列说法正确的是哪一项？"
    assert question["options"] == ["A. 负责运行 Java 字节码", "B. 负责管理浏览器缓存", "C. 负责生成 SQL 索引", "D. 负责压缩图片"]
    assert question["correct_answer"] == "A"
    assert analysis.score == 100
    assert analysis.correct_answer == "JVM 的核心作用是加载并执行 Java 字节码，同时提供运行时环境。"
    assert "正式考试题" in model_calls[0]
    assert "阅卷老师" in model_calls[1] or "用户答案" in model_calls[1]


def test_exam_question_rows_can_generate_first_question_before_remaining(monkeypatch, tmp_path):
    store = ExamRepository()
    session = store.create_exam_session(
        session_id=_unique_id("exam_fast_start"),
        user_id="user_exam",
        title="快速开始测评",
        collection_name="agent",
        document_id=None,
        filename=None,
        section_path=None,
        round_count=3,
        question_types=["short_answer"],
        model_mode="low",
        metadata={"seed": 11},
    )
    candidates = [
        {
            "metadata": {
                "question": f"第 {index} 题是什么？",
                "question_id": f"qa_{index}",
                "document_id": "doc_java",
                "source_file": "Java面试题.pdf",
            },
            "content": f"问题：第 {index} 题是什么？\n答案：第 {index} 题答案。",
        }
        for index in range(1, 4)
    ]

    class FakeModel:
        def invoke(self, messages):
            text = messages[-1].content
            question_match = __import__("re").search(r"原始问题：(.*?)\n\n参考答案", text, flags=__import__("re").DOTALL)
            question_text = question_match.group(1).strip() if question_match else "题目"
            return SimpleNamespace(
                content=(
                    '{"prompt":"正式题：'
                    f'{question_text}",'
                    '"options":[],'
                    '"correct_answer":"标准答案"}'
                )
            )

    monkeypatch.setattr(exam_router, "_store", lambda: store)
    monkeypatch.setattr(exam_router, "get_chat_model", lambda model_mode=None: FakeModel())

    exam_router._build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=candidates[:1],
        candidates=candidates,
        question_types=["short_answer"],
        model_mode="low",
        seed=11,
        max_score=33.3333,
        start_round=1,
    )
    first_batch = store.list_exam_questions(session["session_id"])

    exam_router._build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=candidates,
        candidates=candidates,
        question_types=["short_answer"],
        model_mode="low",
        seed=11,
        max_score=33.3333,
        start_round=2,
    )
    all_questions = store.list_exam_questions(session["session_id"])

    assert [item["round_no"] for item in first_batch] == [1]
    assert [item["round_no"] for item in all_questions] == [1, 2, 3]


def test_exam_first_question_fast_mode_skips_model(monkeypatch, tmp_path):
    store = ExamRepository()
    session = store.create_exam_session(
        session_id=_unique_id("exam_fast_rule_start"),
        user_id="user_exam",
        title="快速规则首题",
        collection_name="agent",
        round_count=1,
        question_types=["short_answer"],
        model_mode="low",
        metadata={"seed": 21},
    )
    candidates = [
        {
            "metadata": {
                "question": "快速首题是什么？",
                "question_id": "qa_fast_1",
                "document_id": "doc_java",
                "source_file": "Java面试题.pdf",
            },
            "content": "问题：快速首题是什么？\n答案：快速首题答案。",
        }
    ]

    class ForbiddenModel:
        def invoke(self, messages):
            raise AssertionError("快速首题不应该同步调用模型")

    monkeypatch.setattr(exam_router, "_store", lambda: store)
    monkeypatch.setattr(exam_router, "get_chat_model", lambda model_mode=None: ForbiddenModel())

    exam_router._build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=candidates,
        candidates=candidates,
        question_types=["short_answer"],
        model_mode="low",
        seed=21,
        max_score=100,
        start_round=1,
        prefer_model=False,
    )
    questions = store.list_exam_questions(session["session_id"])

    assert len(questions) == 1
    assert questions[0]["prompt"] == "快速首题是什么？"


def test_choice_answer_value_is_normalized_to_label(tmp_path):
    store = ExamRepository()
    session = store.create_exam_session(
        session_id=_unique_id("exam_choice_label"),
        user_id="user_exam",
        title="选择题测评",
        collection_name="agent",
        round_count=1,
        question_types=["single_choice"],
        metadata={"seed": 3},
    )
    question = store.add_exam_question(
        session_id=session["session_id"],
        round_no=1,
        source_question_id="qa_001",
        source_document_id="doc_java",
        source_filename="Java面试题.pdf",
        source_page=1,
        section_path="Java 基础",
        question_type="single_choice",
        prompt="JVM 的作用是什么？",
        options=["A. 运行 Java 字节码", "B. 管理浏览器缓存"],
        correct_answer="A",
        reference_answer="JVM 负责运行 Java 字节码。",
        max_score=100,
    )

    user_answer = exam_router._normalize_answer_value_for_question(question, "A. 运行 Java 字节码")

    assert user_answer == "A"


def test_single_choice_generated_label_answer_maps_to_real_option():
    raw_result = {
        "prompt": "关于 Spring 依赖注入的作用，下列说法正确的是哪一项？",
        "options": [
            "A. 降低对象之间的耦合度",
            "B. 直接替代数据库事务",
            "C. 自动压缩静态资源",
            "D. 强制所有类继承同一父类",
        ],
        "correct_answer": "A",
    }

    prompt, options, correct_answer = exam_router._validate_generated_question(raw_result, "single_choice")
    display_options, display_answer = exam_router._prepare_objective_question_for_display(
        "single_choice",
        options,
        correct_answer,
    )

    assert prompt == "关于 Spring 依赖注入的作用，下列说法正确的是哪一项？"
    assert options == ["降低对象之间的耦合度", "直接替代数据库事务", "自动压缩静态资源", "强制所有类继承同一父类"]
    assert correct_answer == "降低对象之间的耦合度"
    assert display_options[0] == "A. 降低对象之间的耦合度"
    assert "A. A" not in display_options
    assert display_answer == "A"


def test_generated_choice_rejects_label_only_options():
    raw_result = {
        "prompt": "关于 Spring 依赖注入的作用，下列说法正确的是哪一项？",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "A",
    }

    try:
        exam_router._validate_generated_question(raw_result, "single_choice")
    except ValueError as exc:
        assert "单选题选项或答案不完整" in str(exc)
    else:
        raise AssertionError("纯选项编号不能作为有效选择题选项")


def test_multiple_choice_allows_all_options_when_single_question_needs_it():
    raw_result = {
        "prompt": "以下哪些属于 JVM 的职责？",
        "options": ["加载字节码", "执行字节码", "提供运行时环境", "管理浏览器缓存"],
        "correct_answer": ["加载字节码", "执行字节码", "提供运行时环境", "管理浏览器缓存"],
    }

    prompt, options, correct_answer = exam_router._validate_generated_question(raw_result, "multiple_choice")

    assert prompt == "以下哪些属于 JVM 的职责？"
    assert len(options) == 4
    assert len(correct_answer) == 4


def test_true_false_generation_passes_target_answer_to_model(monkeypatch):
    model_calls: list[str] = []

    class FakeModel:
        def invoke(self, messages):
            text = messages[-1].content
            model_calls.append(text)
            return SimpleNamespace(
                content='{"prompt":"判断正误：JVM 只负责编译 Java 源码。","options":["正确","错误"],"correct_answer":"错误"}'
            )

    monkeypatch.setattr(exam_router, "get_chat_model", lambda model_mode=None: FakeModel())

    prompt, options, correct_answer = exam_router._generate_question_with_model(
        question="JVM 的作用是什么？",
        reference_answer="JVM 负责加载并运行 Java 字节码。",
        question_type="true_false",
        model_mode="low",
        target_answer="错误",
    )

    assert prompt == "判断正误：JVM 只负责编译 Java 源码。"
    assert options == ["正确", "错误"]
    assert correct_answer == "错误"
    assert "答案为“错误”" in model_calls[0]


def test_exam_start_request_requires_title():
    from pydantic import ValidationError

    from api.exam_schemas import ExamStartRequest

    try:
        ExamStartRequest(title="", round_count=1)
    except ValidationError as exc:
        assert "title" in str(exc)
    else:
        raise AssertionError("开始测评请求必须要求测评名称")


def test_multiple_choice_distribution_breaks_repeated_all_select(monkeypatch, tmp_path):
    store = ExamRepository()
    session = store.create_exam_session(
        session_id=_unique_id("exam_multi_distribution"),
        user_id="user_exam",
        title="多选分布测评",
        collection_name="agent",
        round_count=2,
        question_types=["multiple_choice"],
        metadata={"seed": 5},
    )
    candidates = [
        {
            "metadata": {
                "question": f"第 {index} 道多选题？",
                "question_id": f"qa_multi_{index}",
                "document_id": "doc_java",
                "source_file": "Java面试题.pdf",
            },
            "content": f"问题：第 {index} 道多选题？\n答案：要点一；要点二；要点三；要点四。",
        }
        for index in range(1, 3)
    ]

    class FakeModel:
        def invoke(self, messages):
            return SimpleNamespace(
                content=(
                    '{"prompt":"以下说法哪些正确？",'
                    '"options":["要点一","要点二","要点三","要点四"],'
                    '"correct_answer":["要点一","要点二","要点三","要点四"]}'
                )
            )

    monkeypatch.setattr(exam_router, "_store", lambda: store)
    monkeypatch.setattr(exam_router, "get_chat_model", lambda model_mode=None: FakeModel())

    exam_router._build_exam_question_rows(
        session_id=session["session_id"],
        selected_items=candidates,
        candidates=candidates,
        question_types=["multiple_choice"],
        model_mode="low",
        seed=5,
        max_score=50,
        start_round=1,
    )
    questions = store.list_exam_questions(session["session_id"])
    first_correct = __import__("json").loads(questions[0]["correct_answer_json"])
    second_correct = __import__("json").loads(questions[1]["correct_answer_json"])

    assert set(first_correct) == {"A", "B", "C", "D"}
    assert set(second_correct) != {"A", "B", "C", "D"}
