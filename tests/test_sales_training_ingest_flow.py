from io import BytesIO

from fastapi import UploadFile

from training.quality import TrainingIngestQualityEvaluator
from training.repository import TrainingRepository
from training.services.sales_training_service import SalesTrainingService
from training.strategies.knowledge_ingest_strategy import TrainingChunk


class FakeVectorStore:
    """测试用向量库，记录写入文档但不调用真实 embedding。"""

    def __init__(self, owner):
        self.owner = owner

    def add_documents(self, documents):
        """模拟 Qdrant 写入。"""

        self.owner.documents.extend(documents)


class FakeVectorService:
    """测试用向量服务。"""

    def __init__(self, collection_name="fake_collection"):
        self.collection_name = collection_name
        self.documents = []
        self.vector_store = FakeVectorStore(self)

    def delete_by_metadata(self, field_name, field_value):
        """模拟按 metadata 删除向量点。"""

        self.documents = [
            document
            for document in self.documents
            if document.metadata.get(field_name) != field_value
        ]

    def list_documents_by_metadata(self, field_name, field_value, *, limit=1000):
        """模拟按 metadata 读取向量点。"""

        return [
            document
            for document in self.documents
            if document.metadata.get(field_name) == field_value
        ][:limit]

    def copy_points_by_metadata_to(self, target_service, field_name, field_value, *, metadata_updates=None, limit=5000):
        """模拟从临时库复制向量点到正式库。"""

        copied = []
        for document in self.list_documents_by_metadata(field_name, field_value, limit=limit):
            metadata = dict(document.metadata)
            metadata.update(metadata_updates or {})
            copied.append(type(document)(page_content=document.page_content, metadata=metadata))
        target_service.documents.extend(copied)
        return len(copied)

    def update_metadata_by_metadata(self, field_name, field_value, *, metadata_updates, limit=5000):
        """模拟原地更新 metadata。"""

        updated_count = 0
        for document in self.documents:
            if document.metadata.get(field_name) != field_value:
                continue
            document.metadata.update(metadata_updates)
            updated_count += 1
            if updated_count >= limit:
                break
        return updated_count

    def search_documents(self, query, *, k=None, filters=None):
        """模拟按 batch_id 过滤后的向量检索。"""

        batch_ids = (filters or {}).get("batch_id") or []
        documents = []
        for document in self.documents:
            if batch_ids and document.metadata.get("batch_id") not in batch_ids:
                continue
            document.metadata["_vector_score"] = 0.99
            documents.append(document)
        return documents[: k or 3]


def attach_fake_vector_services(service):
    """给服务挂上测试用正式库和临时库。"""

    published_service = FakeVectorService("sales_training_cases")
    staging_service = FakeVectorService("sales_training_cases_staging")
    service.vector_service = published_service
    service.staging_vector_service = staging_service
    return published_service, staging_service


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
    """上传阶段只写临时向量库，确认发布后才进入正式向量库。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    fake_vector_service, fake_staging_service = attach_fake_vector_services(service)
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
    assert upload_result.point_count == upload_result.chunk_count
    assert upload_result.quality_report["score"] > 0
    assert fake_vector_service.documents == []
    assert len(fake_staging_service.documents) == upload_result.chunk_count
    batch = repository.get_batch(upload_result.batch_id)
    assert batch["status"] == "pending_review"
    with repository.connect() as conn:
        chunk_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'training_knowledge_chunks'"
        ).fetchone()
    assert chunk_table is None

    publish_result = service.publish_batch(upload_result.batch_id)

    assert publish_result.status == "published"
    assert publish_result.point_count == upload_result.chunk_count
    assert len(fake_vector_service.documents) == upload_result.chunk_count
    assert fake_staging_service.documents == []
    published_batch = repository.get_batch(upload_result.batch_id)
    assert published_batch["status"] == "published"


def test_training_upload_uses_llm_fallback_when_quality_is_low(tmp_path, monkeypatch):
    """规则切分质量低时，应采用质量更高的 LLM 兜底切片。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    fake_vector_service, fake_staging_service = attach_fake_vector_services(service)
    captured_model_modes = []

    class FakeFallbackSplitter:
        """测试用 LLM 兜底切分器。"""

        def should_trigger(self, quality_report):
            return True

        def split(self, *, source_text, batch_id, source_file, source_type, visibility_default, model_mode=None):
            captured_model_modes.append(model_mode)
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
        model_mode="medium",
    )

    assert upload_result.status == "pending_review"
    assert upload_result.quality_report["selected_splitter"] == "llm_fallback"
    assert upload_result.quality_report["llm_fallback_used"] is True
    assert len(fake_staging_service.documents) == 4
    assert fake_vector_service.documents == []
    assert captured_model_modes == ["medium"]


def test_training_manual_reparse_uses_llm_fallback(tmp_path, monkeypatch):
    """人工重新切分时，应主动采用 LLM 兜底结果并回到待确认状态。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    fake_vector_service, fake_staging_service = attach_fake_vector_services(service)
    captured_model_modes = []

    class FakeFallbackSplitter:
        """测试用人工 LLM 重切器。"""

        def __init__(self):
            # 服务层会把 enabled 打开；这里保留 config 用于兼容真实对象接口。
            self.config = {"enabled": False}

        def should_trigger(self, quality_report):
            # 上传阶段不自动触发，保证本用例只验证人工重切入口。
            return False

        def split(self, *, source_text, batch_id, source_file, source_type, visibility_default, model_mode=None):
            captured_model_modes.append(model_mode)
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

    reparse_result = service.reparse_batch(upload_result.batch_id, use_llm_fallback=True, model_mode="low")

    assert reparse_result.status == "pending_review"
    assert reparse_result.quality_report["selected_splitter"] == "llm_fallback"
    assert reparse_result.quality_report["manual_reparse"] is True
    assert reparse_result.point_count == 4
    saved_chunks = service.list_chunks(upload_result.batch_id).chunks
    assert len(saved_chunks) == 4
    assert all("LLM" in row.chunk_text for row in saved_chunks)
    assert fake_vector_service.documents == []
    assert len(fake_staging_service.documents) == 4
    saved_batch = repository.get_batch(upload_result.batch_id)
    assert saved_batch["status"] == "pending_review"
    assert captured_model_modes == ["low"]


def test_training_preview_uses_saved_chunks(tmp_path):
    """预览接口应展示已保存切片，避免重新解析导致预览和发布不一致。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    attach_fake_vector_services(service)
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
    attach_fake_vector_services(service)
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


def test_training_publish_archives_previous_version_and_rollback(tmp_path):
    """同名资料发布新版本后旧版本应归档，并支持回滚为当前版本。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    fake_vector_service, fake_staging_service = attach_fake_vector_services(service)

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
    first_documents = fake_vector_service.list_documents_by_metadata("batch_id", first_publish.batch_id)
    assert first_documents
    assert all(document.metadata["status"] == "archived" for document in first_documents)
    assert fake_staging_service.documents == []

    rollback_result = service.rollback_batch(first_publish.batch_id)

    rolled_back_batch = repository.get_batch(first_publish.batch_id)
    archived_second_batch = repository.get_batch(second_publish.batch_id)
    rolled_back_documents = fake_vector_service.list_documents_by_metadata("batch_id", first_publish.batch_id)
    second_documents = fake_vector_service.list_documents_by_metadata("batch_id", second_publish.batch_id)
    assert rollback_result.version_no == 1
    assert rolled_back_batch["status"] == "published"
    assert rolled_back_batch["is_current"] == 1
    assert archived_second_batch["status"] == "archived"
    assert archived_second_batch["is_current"] == 0
    assert all(document.metadata["status"] == "published" for document in rolled_back_documents)
    assert all(document.metadata["status"] == "archived" for document in second_documents)


def test_training_list_batch_versions_returns_version_chain(tmp_path):
    """版本链接口应返回同一版本组内的全部未删除版本。"""

    repository = TrainingRepository(str(tmp_path / "training.db"))
    service = SalesTrainingService(repository=repository)
    attach_fake_vector_services(service)

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
