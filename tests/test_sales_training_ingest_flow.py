from io import BytesIO

from fastapi import UploadFile

from training.quality import TrainingIngestQualityEvaluator
from training.repository import TrainingRepository
from training.services.sales_training_service import SalesTrainingService
from training.strategies.knowledge_ingest_strategy import TrainingChunk


class FakeVectorStore:
    """测试用向量库，记录写入文档但不调用真实 embedding。"""

    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        """模拟 Qdrant 写入。"""

        self.documents.extend(documents)


class FakeVectorService:
    """测试用向量服务。"""

    def __init__(self):
        self.vector_store = FakeVectorStore()

    def search_documents(self, query, *, k=None, filters=None):
        """模拟按 batch_id 过滤后的向量检索。"""

        batch_ids = (filters or {}).get("batch_id") or []
        documents = []
        for document in self.vector_store.documents:
            if batch_ids and document.metadata.get("batch_id") not in batch_ids:
                continue
            document.metadata["_vector_score"] = 0.99
            documents.append(document)
        return documents[: k or 3]


def test_training_ingest_quality_detects_weak_split():
    """质量评估应能识别内容过度集中、核心片段缺失的弱切片。"""

    chunks = [
        TrainingChunk(
            chunk_id="chunk_1",
            text="客户背景" * 500,
            case_part="case_profile",
            visibility="visible",
            metadata={"case_index": 1},
        )
    ]

    report = TrainingIngestQualityEvaluator().evaluate(chunks)

    assert report.passed is False
    assert report.level in {"review", "poor"}
    assert "task_requirement" in report.metrics["missing_required_parts"]


def test_training_upload_waits_for_manual_publish(tmp_path):
    """上传阶段只生成预览切片，确认发布后才写入向量库。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    fake_vector_service = FakeVectorService()
    service.vector_service = fake_vector_service
    content = "\n".join([
        "一、客户案例",
        "企业：某外贸公司",
        "任务要求",
        "请完成需求挖掘。",
        "匹配答案",
        "可以先询问客户当前获客渠道。",
        "命中点",
        "确认客户预算和决策链。",
    ])
    upload_file = UploadFile(filename="case.txt", file=BytesIO(content.encode("utf-8")))

    upload_result = service.upload_knowledge(
        file=upload_file,
        source_type="lms_case",
        created_by="tester",
    )

    assert upload_result.status == "pending_review"
    assert upload_result.point_count == 0
    assert upload_result.quality_report["score"] > 0
    assert fake_vector_service.vector_store.documents == []
    batch = repository.get_batch(upload_result.batch_id)
    assert batch["status"] == "pending_review"
    assert len(repository.list_chunks(upload_result.batch_id)) > 0

    publish_result = service.publish_batch(upload_result.batch_id)

    assert publish_result.status == "published"
    assert publish_result.point_count == upload_result.chunk_count
    assert len(fake_vector_service.vector_store.documents) == upload_result.chunk_count
    published_batch = repository.get_batch(upload_result.batch_id)
    assert published_batch["status"] == "published"


def test_training_upload_uses_llm_fallback_when_quality_is_low(tmp_path, monkeypatch):
    """规则切分质量低时，应采用质量更高的 LLM 兜底切片。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    service.vector_service = FakeVectorService()

    class FakeFallbackSplitter:
        """测试用 LLM 兜底切分器。"""

        def should_trigger(self, quality_report):
            return True

        def split(self, *, source_text, batch_id, source_file, source_type, visibility_default):
            return [
                TrainingChunk(
                    chunk_id=f"{batch_id}_001_case_profile",
                    text="一、客户案例\n企业：某外贸公司",
                    case_part="case_profile",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "一、客户案例", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_001_task_requirement",
                    text="一、客户案例\n请完成需求挖掘。",
                    case_part="task_requirement",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "一、客户案例", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_001_standard_answer",
                    text="一、客户案例\n可以先询问客户当前获客渠道。",
                    case_part="standard_answer",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "一、客户案例", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_001_scoring_rubric",
                    text="一、客户案例\n命中客户预算和决策链。",
                    case_part="scoring_rubric",
                    visibility="scoring_only",
                    metadata={"case_index": 1, "case_title": "一、客户案例", "splitter": "llm_fallback"},
                ),
            ]

    monkeypatch.setattr("training.services.sales_training_service.TrainingLlmFallbackSplitter", FakeFallbackSplitter)
    upload_file = UploadFile(filename="weak.txt", file=BytesIO("这是一段没有结构的普通资料。".encode("utf-8")))

    upload_result = service.upload_knowledge(
        file=upload_file,
        source_type="lms_case",
        created_by="tester",
    )

    assert upload_result.status == "pending_review"
    assert upload_result.quality_report["selected_splitter"] == "llm_fallback"
    assert upload_result.quality_report["llm_fallback_used"] is True
    assert len(repository.list_chunks(upload_result.batch_id)) == 4


def test_training_manual_reparse_uses_llm_fallback(tmp_path, monkeypatch):
    """人工重新切分时，应主动采用 LLM 兜底结果并回到待确认状态。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    service.vector_service = FakeVectorService()

    class FakeFallbackSplitter:
        """测试用人工 LLM 重切器。"""

        def __init__(self):
            # 服务层会把 enabled 打开；这里保留 config 用于兼容真实对象接口。
            self.config = {"enabled": False}

        def should_trigger(self, quality_report):
            # 上传阶段不自动触发，保证本用例只验证人工重切入口。
            return False

        def split(self, *, source_text, batch_id, source_file, source_type, visibility_default):
            return [
                TrainingChunk(
                    chunk_id=f"{batch_id}_llm_case_profile",
                    text="LLM 客户背景\n企业：重新整理后的客户画像。",
                    case_part="case_profile",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "LLM 客户背景", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_llm_task_requirement",
                    text="LLM 任务要求\n请围绕客户预算、风险和决策链完成追问。",
                    case_part="task_requirement",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "LLM 客户背景", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_llm_standard_answer",
                    text="LLM 参考话术\n建议先确认预算范围，再用案例降低客户风险感。",
                    case_part="standard_answer",
                    visibility="visible",
                    metadata={"case_index": 1, "case_title": "LLM 客户背景", "splitter": "llm_fallback"},
                ),
                TrainingChunk(
                    chunk_id=f"{batch_id}_llm_scoring_rubric",
                    text="LLM 评分依据\n命中预算、风险、决策链三个关键点。",
                    case_part="scoring_rubric",
                    visibility="scoring_only",
                    metadata={"case_index": 1, "case_title": "LLM 客户背景", "splitter": "llm_fallback"},
                ),
            ]

    monkeypatch.setattr("training.services.sales_training_service.TrainingLlmFallbackSplitter", FakeFallbackSplitter)
    content = "\n".join([
        "一、客户案例",
        "企业：规则切分版本",
        "任务要求",
        "完成需求挖掘。",
        "匹配答案",
        "询问客户预算。",
    ])
    upload_result = service.upload_knowledge(
        file=UploadFile(filename="manual_reparse.txt", file=BytesIO(content.encode("utf-8"))),
        source_type="lms_case",
        created_by="tester",
    )

    reparse_result = service.reparse_batch(upload_result.batch_id, use_llm_fallback=True)

    assert reparse_result.status == "pending_review"
    assert reparse_result.quality_report["selected_splitter"] == "llm_fallback"
    assert reparse_result.quality_report["manual_reparse"] is True
    assert reparse_result.point_count == 0
    saved_chunks = repository.list_chunks(upload_result.batch_id)
    assert len(saved_chunks) == 4
    assert all("LLM" in row["chunk_text"] for row in saved_chunks)
    saved_batch = repository.get_batch(upload_result.batch_id)
    assert saved_batch["status"] == "pending_review"


def test_training_preview_uses_saved_chunks(tmp_path):
    """预览接口应展示已保存切片，避免重新解析导致预览和发布不一致。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    content = "\n".join([
        "一、客户案例",
        "企业：某外贸公司",
        "任务要求",
        "请完成需求挖掘。",
        "匹配答案",
        "可以先询问客户当前获客渠道。",
    ])
    upload_file = UploadFile(filename="preview.txt", file=BytesIO(content.encode("utf-8")))
    upload_result = service.upload_knowledge(file=upload_file, source_type="lms_case", created_by="tester")

    preview = service.preview_batch(upload_result.batch_id)

    assert preview.preview_type == "saved_chunks"
    assert "case_profile" in preview.content
    assert "standard_answer" in preview.content


def test_training_publish_writes_validation_report(tmp_path):
    """发布完成后，应把抽样检索验证结果写回质量报告。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    service.vector_service = FakeVectorService()
    content = "\n".join([
        "一、客户案例",
        "企业：某外贸公司",
        "任务要求",
        "请完成需求挖掘。",
        "匹配答案",
        "可以先询问客户当前获客渠道。",
        "命中点",
        "确认客户预算和决策链。",
    ])
    upload_file = UploadFile(filename="publish.txt", file=BytesIO(content.encode("utf-8")))
    upload_result = service.upload_knowledge(file=upload_file, source_type="lms_case", created_by="tester")

    publish_result = service.publish_batch(upload_result.batch_id)

    assert publish_result.quality_report["publish_validation"]["passed"] is True
    saved_report = service._load_json(repository.get_batch(upload_result.batch_id)["quality_report_json"], {})
    assert saved_report["publish_validation"]["hit_count"] > 0


def test_training_publish_archives_previous_version_and_rollback(tmp_path, monkeypatch):
    """同名资料发布新版本后旧版本应归档，并支持回滚为当前版本。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    service.vector_service = FakeVectorService()
    deleted_batch_ids = []

    monkeypatch.setattr(
        "training.services.sales_training_service.VectorStoreService.delete_vectors_by_metadata",
        lambda key, value, collection_name=None: deleted_batch_ids.append(value),
    )

    first_content = "\n".join([
        "一、客户案例",
        "企业：第一版公司",
        "任务要求",
        "完成需求挖掘。",
        "匹配答案",
        "询问客户预算。",
        "命中点",
        "确认决策链。",
    ])
    second_content = first_content.replace("第一版公司", "第二版公司")

    first_upload = service.upload_knowledge(
        file=UploadFile(filename="version.txt", file=BytesIO(first_content.encode("utf-8"))),
        source_type="lms_case",
        created_by="tester",
    )
    first_publish = service.publish_batch(first_upload.batch_id)
    second_upload = service.upload_knowledge(
        file=UploadFile(filename="version.txt", file=BytesIO(second_content.encode("utf-8"))),
        source_type="lms_case",
        created_by="tester",
    )
    second_publish = service.publish_batch(second_upload.batch_id)

    first_batch = repository.get_batch(first_publish.batch_id)
    second_batch = repository.get_batch(second_publish.batch_id)
    assert first_batch["status"] == "archived"
    assert first_batch["is_current"] == 0
    assert second_batch["status"] == "published"
    assert second_batch["is_current"] == 1
    assert second_batch["version_no"] == 2
    assert first_publish.batch_id in deleted_batch_ids

    rollback_result = service.rollback_batch(first_publish.batch_id)

    rolled_back_batch = repository.get_batch(first_publish.batch_id)
    archived_second_batch = repository.get_batch(second_publish.batch_id)
    assert rollback_result.version_no == 1
    assert rolled_back_batch["status"] == "published"
    assert rolled_back_batch["is_current"] == 1
    assert archived_second_batch["status"] == "archived"
    assert archived_second_batch["is_current"] == 0


def test_training_list_batch_versions_returns_version_chain(tmp_path, monkeypatch):
    """版本链接口应返回同一版本组内的全部未删除版本。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    service.vector_service = FakeVectorService()
    monkeypatch.setattr(
        "training.services.sales_training_service.VectorStoreService.delete_vectors_by_metadata",
        lambda key, value, collection_name=None: None,
    )

    first_content = "\n".join([
        "一、客户案例",
        "企业：第一版客户",
        "任务要求",
        "完成需求挖掘。",
        "匹配答案",
        "询问客户预算。",
        "命中点",
        "确认决策链。",
    ])
    second_content = first_content.replace("第一版客户", "第二版客户")
    first_upload = service.upload_knowledge(
        file=UploadFile(filename="chain.txt", file=BytesIO(first_content.encode("utf-8"))),
        source_type="lms_case",
        created_by="tester",
    )
    service.publish_batch(first_upload.batch_id)
    second_upload = service.upload_knowledge(
        file=UploadFile(filename="chain.txt", file=BytesIO(second_content.encode("utf-8"))),
        source_type="lms_case",
        created_by="tester",
    )

    versions = service.list_batch_versions(first_upload.batch_id)

    assert versions.version_group_id == first_upload.batch_id
    assert [item.version_no for item in versions.items] == [2, 1]
    assert versions.items[0].batch_id == second_upload.batch_id
    assert versions.items[0].status == "pending_review"
    assert versions.items[1].status == "published"
