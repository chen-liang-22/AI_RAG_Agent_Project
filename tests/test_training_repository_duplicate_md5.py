"""销售训练资料 MD5 去重仓储测试。"""

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.application.training_support import repository as repository_module
from app.application.training_support.repository import TrainingRepository
from app.domain.entities import BaseOrmModel, DocumentEntity, TrainingKnowledgeBatchEntity


@contextmanager
def _session_context(factory):
    """模拟项目里的 ORM Session 上下文。"""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _build_repository_session_factory():
    """创建训练资料仓储测试用 SQLite SessionFactory。"""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    BaseOrmModel.metadata.create_all(
        engine,
        tables=[
            DocumentEntity.__table__,
            TrainingKnowledgeBatchEntity.__table__,
        ],
    )
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _document(document_id: str, *, file_md5: str) -> DocumentEntity:
    """构造测试文档实体。"""

    now = datetime(2026, 6, 29, 10, 0, 0)
    return DocumentEntity(
        document_id=document_id,
        filename=f"{document_id}.txt",
        file_path=f"minio://pub/training/{document_id}.txt",
        storage_type="minio",
        bucket_name="pub",
        object_name=f"training/{document_id}.txt",
        public_url=f"http://localhost:9000/pub/training/{document_id}.txt",
        file_type="txt",
        file_md5=file_md5,
        file_size=12,
        status="uploaded",
        version=1,
        chunk_count=0,
        collection_name="sales_training_cases",
        document_type="text",
        split_strategy="recursive",
        created_at=now,
        updated_at=now,
        error_message=None,
    )


def _batch(batch_id: str, document_id: str, *, status: str) -> TrainingKnowledgeBatchEntity:
    """构造测试训练资料批次实体。"""

    now = datetime(2026, 6, 29, 10, 0, 0)
    return TrainingKnowledgeBatchEntity(
        batch_id=batch_id,
        document_id=document_id,
        source_type="lms_case",
        source_file=f"{batch_id}.txt",
        file_path=None,
        file_md5=None,
        version_group_id=batch_id,
        version_no=1,
        previous_batch_id=None,
        is_current=0,
        profile_type=None,
        task_type=None,
        industry=None,
        difficulty=None,
        visibility_default="visible",
        status=status,
        chunk_count=0,
        point_count=0,
        error_message=None,
        quality_report_json=None,
        created_by="tester",
        created_at=now,
        updated_at=now,
    )


def test_training_repository_finds_unpublished_duplicate_by_md5(monkeypatch):
    """MD5 去重应命中未删除的未发布批次，而不是只查 published。"""

    factory = _build_repository_session_factory()

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(repository_module, "orm_session_context", fake_orm_session_context)
    repository = TrainingRepository()
    with _session_context(factory) as session:
        session.add(_document("doc_pending", file_md5="md5_same"))
        session.add(_batch("batch_pending", "doc_pending", status="pending_review"))
        session.add(_document("doc_deleted", file_md5="md5_deleted"))
        session.add(_batch("batch_deleted", "doc_deleted", status="deleted"))

    duplicate = repository.get_existing_batch_by_md5("md5_same")
    deleted_duplicate = repository.get_existing_batch_by_md5("md5_deleted")

    assert duplicate is not None
    assert duplicate.batch_id == "batch_pending"
    assert duplicate.status == "pending_review"
    assert duplicate.document_file_md5 == "md5_same"
    assert deleted_duplicate is None
