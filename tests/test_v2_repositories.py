"""V2 字典仓储测试。"""

from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app_v2.shared.pagination import escape_like_keyword, normalize_page


class FakeStore:
    """测试用字典数据源，模拟旧字典方法。"""

    def __init__(self):
        self.list_calls: list[str | None] = []
        self.upsert_values: dict | None = None
        self.deleted_item_id: str | None = None

    def list_dictionary_items(self, dictionary_code: str | None = None):
        self.list_calls.append(dictionary_code)
        return [
            {
                "dictionary_item_id": "item_1",
                "dictionary_code": dictionary_code or "all",
                "dictionary_name": "测试字典",
                "item_code": "enabled",
                "item_name": "启用",
                "item_level": 1,
                "sort_order": 1,
                "enabled": 1,
            }
        ]

    def upsert_dictionary_item(self, **values):
        self.upsert_values = values
        return {**values, "dictionary_item_id": "item_saved", "item_level": 1, "metadata_json": None}

    def delete_dictionary_item(self, dictionary_item_id: str):
        self.deleted_item_id = dictionary_item_id
        return True

    def normalize_dictionary_code(self, dictionary_code: str, value: str | None = None):
        return value or "default"


class FakeRedisClient:
    """测试用 Redis 替身，只记录 JSON 缓存读写删除行为。"""

    def __init__(self):
        self.values: dict[str, object] = {}
        self.deleted_keys: list[str] = []

    def build_key(self, *parts: object) -> str:
        return ":".join(["test", *[str(part) for part in parts]])

    def get_json(self, key: str, default=None):
        return self.values.get(key, default)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        self.values[key] = value
        return True

    def delete(self, *keys: str) -> int:
        self.deleted_keys.extend(keys)
        for key in keys:
            self.values.pop(key, None)
        return len(keys)


def test_dictionary_repository_reads_and_writes_cache(monkeypatch):
    """第一次查字典走 ORM，第二次相同查询应直接命中 Redis 缓存。"""

    factory = _build_sqlite_session_factory()
    _seed_dictionary_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(dictionary_repository_module, "orm_session_context", fake_orm_session_context)
    redis_client = FakeRedisClient()
    repository = DictionaryRepository(store=ExplodingDictionaryStore(), redis_client=redis_client)

    first_rows = repository.list_items("model_mode")
    second_rows = repository.list_items("model_mode")

    assert first_rows == second_rows
    assert [row["item_code"] for row in first_rows] == ["default", "disabled", "recommendation"]
    assert redis_client.build_key("v2", "dictionary", "items", "model_mode") in redis_client.values


def test_dictionary_repository_clears_cache_after_write_and_delete(monkeypatch):
    """字典写入和删除后必须清理缓存，避免页面继续展示旧字典项。"""

    factory = _build_sqlite_session_factory()
    _seed_dictionary_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(dictionary_repository_module, "orm_session_context", fake_orm_session_context)
    redis_client = FakeRedisClient()
    repository = DictionaryRepository(store=ExplodingDictionaryStore(), redis_client=redis_client)
    repository.list_items("model_mode")

    saved = repository.upsert_item(
        dictionary_code="model_mode",
        dictionary_name="模型档位",
        item_code="ok",
        item_name="正常",
    )
    deleted = repository.delete_item("dict_disabled")

    expected_key = redis_client.build_key("v2", "dictionary", "items", "model_mode")
    all_key = redis_client.build_key("v2", "dictionary", "items", "all")
    assert expected_key in redis_client.deleted_keys
    assert all_key in redis_client.deleted_keys
    assert saved["item_code"] == "ok"
    assert deleted is True


def test_pagination_normalizes_page_and_escapes_like_keyword():
    """分页参数要收敛到安全范围，LIKE 关键字要转义通配符。"""

    page = normalize_page(page=0, page_size=999, max_page_size=50)

    assert page.page == 1
    assert page.page_size == 50
    assert page.offset == 0
    assert escape_like_keyword(r"50%_done\path") == r"50\%\_done\\path"


from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app_v2.infrastructure.repositories import conversation_repository as conversation_repository_module
from app_v2.infrastructure.repositories import dictionary_repository as dictionary_repository_module
from app_v2.infrastructure.repositories import document_repository as document_repository_module
from app_v2.infrastructure.repositories.conversation_repository import ConversationRepository
from app_v2.infrastructure.repositories.document_repository import DocumentRepository
from domain.entities import BaseOrmModel, ConversationEntity, ConversationMessageEntity, DictionaryItemEntity, DocumentEntity


def _build_sqlite_session_factory():
    """创建测试用 SQLite SessionFactory，避免单元测试依赖本地 MySQL。"""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    BaseOrmModel.metadata.create_all(
        engine,
        tables=[
            ConversationEntity.__table__,
            ConversationMessageEntity.__table__,
            DocumentEntity.__table__,
            DictionaryItemEntity.__table__,
        ],
    )
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def _session_context(factory):
    """模拟项目里的 orm_session_context，正常提交、异常回滚。"""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _seed_conversation_rows(factory):
    """准备会话和消息测试数据。"""

    now = datetime(2026, 6, 26, 9, 30, 0)
    with _session_context(factory) as session:
        session.add_all([
            ConversationEntity(
                conversation_id="conv_active_1",
                user_id="1001",
                title="地图保存问题",
                status="active",
                message_count=2,
                summary=None,
                metadata_json=None,
                created_at=now,
                updated_at=now,
                last_message_at=now,
            ),
            ConversationEntity(
                conversation_id="conv_active_2",
                user_id="1002",
                title="清扫记录删除",
                status="active",
                message_count=1,
                summary=None,
                metadata_json=None,
                created_at=now,
                updated_at=now,
                last_message_at=now,
            ),
            ConversationEntity(
                conversation_id="conv_deleted",
                user_id="1003",
                title="已删除会话",
                status="deleted",
                message_count=1,
                summary=None,
                metadata_json=None,
                created_at=now,
                updated_at=now,
                last_message_at=now,
            ),
        ])
        session.add_all([
            ConversationMessageEntity(
                message_id="msg_1",
                conversation_id="conv_active_1",
                sequence_no=1,
                role="user",
                content="APP 中拖扫地图如何保存？",
                content_type="text",
                model_name=None,
                token_count=None,
                metadata_json=None,
                created_at=now,
            ),
            ConversationMessageEntity(
                message_id="msg_2",
                conversation_id="conv_active_1",
                sequence_no=2,
                role="assistant",
                content="建图完成后会自动保存。",
                content_type="text",
                model_name="qwen3-max",
                token_count=12,
                metadata_json='{"first_token_ms": 10, "total_ms": 20}',
                created_at=now,
            ),
        ])


def test_conversation_repository_uses_orm_for_list_detail_and_delete(monkeypatch):
    """V2 会话仓储应该直接使用 ORM，不再委托旧存储。"""

    factory = _build_sqlite_session_factory()
    _seed_conversation_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(conversation_repository_module, "orm_session_context", fake_orm_session_context)
    repository = ConversationRepository()

    conversations, total = repository.list_conversations(page=1, page_size=10, keyword="地图")
    detail = repository.get_conversation("conv_active_1")
    messages = repository.list_conversation_messages("conv_active_1")
    deleted = repository.delete_conversation("conv_active_1")
    after_delete = repository.get_conversation("conv_active_1")
    deleted_messages = repository.list_conversation_messages("conv_active_1")

    assert total == 1
    assert [row.conversation_id for row in conversations] == ["conv_active_1"]
    assert detail is not None
    assert detail.title == "地图保存问题"
    assert [message.sequence_no for message in messages] == [1, 2]
    assert deleted is True
    assert after_delete is not None
    assert after_delete.status == "deleted"
    assert after_delete.message_count == 0
    assert deleted_messages == []

class ExplodingDocumentStore:
    """如果 V2 文档仓储仍然调用旧存储，本替身会让测试立刻失败。"""

    def list_documents(self, *args, **kwargs):
        raise AssertionError("V2 文档仓储不应该继续委托旧存储查询列表")

    def get_document(self, *args, **kwargs):
        raise AssertionError("V2 文档仓储不应该继续委托旧存储查询详情")

    def find_active_document_by_md5(self, *args, **kwargs):
        raise AssertionError("V2 文档仓储不应该继续委托旧存储做去重查询")


def _seed_document_rows(factory):
    """准备文档资产测试数据，覆盖普通知识库、训练知识库和已删除文档。"""

    now = datetime(2026, 6, 26, 10, 0, 0)
    with _session_context(factory) as session:
        session.add_all([
            DocumentEntity(
                document_id="doc_common",
                filename="扫拖一体机器人100问.txt",
                file_path="minio://knowledge/doc_common.txt",
                storage_type="minio",
                bucket_name="knowledge",
                object_name="doc_common.txt",
                public_url=None,
                file_type="txt",
                file_md5="md5_same",
                file_size=1200,
                status="indexed",
                version=1,
                chunk_count=8,
                collection_name="agent",
                document_type="qa",
                split_strategy="qa_pair",
                created_at=now,
                updated_at=now,
                error_message=None,
            ),
            DocumentEntity(
                document_id="doc_training",
                filename="销售陪练案例.docx",
                file_path="minio://knowledge/doc_training.docx",
                storage_type="minio",
                bucket_name="knowledge",
                object_name="doc_training.docx",
                public_url=None,
                file_type="docx",
                file_md5="md5_training",
                file_size=2400,
                status="indexed",
                version=1,
                chunk_count=16,
                collection_name="sales_training_cases",
                document_type="scenario",
                split_strategy="paragraph",
                created_at=now,
                updated_at=now,
                error_message=None,
            ),
            DocumentEntity(
                document_id="doc_deleted",
                filename="已删除资料.txt",
                file_path="minio://knowledge/doc_deleted.txt",
                storage_type="minio",
                bucket_name="knowledge",
                object_name="doc_deleted.txt",
                public_url=None,
                file_type="txt",
                file_md5="md5_same",
                file_size=800,
                status="deleted",
                version=1,
                chunk_count=0,
                collection_name="agent",
                document_type="text",
                split_strategy="recursive",
                created_at=now,
                updated_at=now,
                error_message=None,
            ),
        ])


def test_document_repository_uses_orm_for_list_detail_and_duplicate_lookup(monkeypatch):
    """V2 文档仓储应该直接查询 documents 表，避免继续把读路径压在旧存储上。"""

    factory = _build_sqlite_session_factory()
    _seed_document_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(document_repository_module, "orm_session_context", fake_orm_session_context)
    repository = DocumentRepository(store=ExplodingDocumentStore())

    normal_documents = repository.list_documents(include_training=False)
    all_documents = repository.list_documents(include_training=True)
    detail = repository.get_document("doc_common")
    duplicate = repository.find_active_document_by_md5("md5_same", collection_name="agent")

    assert [row.document_id for row in normal_documents] == ["doc_common"]
    assert [row.document_id for row in all_documents] == ["doc_common", "doc_training"]
    assert detail is not None
    assert detail.filename == "扫拖一体机器人100问.txt"
    assert duplicate is not None
    assert duplicate.document_id == "doc_common"

def test_document_repository_uses_orm_for_create_update_and_delete(monkeypatch):
    """V2 文档仓储应该直接负责 documents 表的创建、状态更新和删除。"""

    factory = _build_sqlite_session_factory()

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(document_repository_module, "orm_session_context", fake_orm_session_context)
    repository = DocumentRepository(store=ExplodingDocumentStore())

    created = repository.create_document(
        document_id="doc_write",
        filename="写入测试.txt",
        file_path="minio://knowledge/doc_write.txt",
        file_type="txt",
        file_md5="md5_write",
        file_size=300,
        storage_type="minio",
        bucket_name="knowledge",
        object_name="doc_write.txt",
        public_url=None,
        status="uploaded",
        collection_name="agent",
        document_type="text",
        split_strategy="recursive",
    )
    repository.update_document_status(
        "doc_write",
        "indexed",
        chunk_count=5,
        increment_version=True,
        collection_name="agent_v2",
        document_type="qa",
        split_strategy="qa_pair",
    )
    updated = repository.get_document("doc_write")
    deleted = repository.delete_document("doc_write")
    missing_after_delete = repository.get_document("doc_write")

    assert created.document_id == "doc_write"
    assert created.version == 1
    assert updated is not None
    assert updated.status == "indexed"
    assert updated.chunk_count == 5
    assert updated.version == 2
    assert updated.collection_name == "agent_v2"
    assert updated.document_type == "qa"
    assert updated.split_strategy == "qa_pair"
    assert deleted is True
    assert missing_after_delete is None

class ExplodingDictionaryStore:
    """如果 V2 字典仓储仍然调用旧存储，本替身会让测试失败。"""

    def list_dictionary_items(self, *args, **kwargs):
        raise AssertionError("V2 字典仓储不应该继续委托旧存储查询字典")

    def upsert_dictionary_item(self, *args, **kwargs):
        raise AssertionError("V2 字典仓储不应该继续委托旧存储保存字典")

    def delete_dictionary_item(self, *args, **kwargs):
        raise AssertionError("V2 字典仓储不应该继续委托旧存储删除字典")

    def normalize_dictionary_code(self, *args, **kwargs):
        raise AssertionError("V2 字典仓储不应该继续委托旧存储归一化字典编码")


def _seed_dictionary_rows(factory):
    """准备字典测试数据，覆盖排序、启用禁用、父子层级和 metadata 查询。"""

    now = datetime(2026, 6, 26, 11, 0, 0)
    with _session_context(factory) as session:
        session.add_all([
            DictionaryItemEntity(
                dictionary_item_id="dict_parent",
                dictionary_code="model_mode",
                dictionary_name="模型档位",
                item_code="default",
                item_name="默认档位",
                parent_item_id=None,
                item_level=1,
                sort_order=1,
                enabled=1,
                description="默认使用",
                metadata_json='{"default": true}',
                created_at=now,
                updated_at=now,
            ),
            DictionaryItemEntity(
                dictionary_item_id="dict_recommend",
                dictionary_code="model_mode",
                dictionary_name="模型档位",
                item_code="recommendation",
                item_name="推荐档位",
                parent_item_id="dict_parent",
                item_level=2,
                sort_order=2,
                enabled=1,
                description=None,
                metadata_json='{"recommendation": true}',
                created_at=now,
                updated_at=now,
            ),
            DictionaryItemEntity(
                dictionary_item_id="dict_disabled",
                dictionary_code="model_mode",
                dictionary_name="模型档位",
                item_code="disabled",
                item_name="禁用档位",
                parent_item_id=None,
                item_level=1,
                sort_order=3,
                enabled=0,
                description=None,
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
        ])


def test_dictionary_repository_uses_orm_for_list_normalize_metadata_upsert_and_delete(monkeypatch):
    """V2 字典仓储应该直接访问 dictionary_items 表，并继续维护 Redis 缓存。"""

    factory = _build_sqlite_session_factory()
    _seed_dictionary_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(dictionary_repository_module, "orm_session_context", fake_orm_session_context)
    redis_client = FakeRedisClient()
    repository = DictionaryRepository(store=ExplodingDictionaryStore(), redis_client=redis_client)

    first_rows = repository.list_items("model_mode")
    second_rows = repository.list_items("model_mode")
    normalized_blank = repository.normalize_code("model_mode", None)
    normalized_invalid = repository.normalize_code("model_mode", "missing")
    normalized_recommend = repository.normalize_code("model_mode", "recommendation")
    metadata_code = repository.get_code_by_metadata("model_mode", "recommendation", True)
    saved = repository.upsert_item(
        dictionary_code="model_mode",
        dictionary_name="模型档位",
        item_code="premium",
        item_name="高阶档位",
        parent_item_code="default",
        sort_order=4,
        enabled=True,
        description="新增档位",
        metadata={"premium": True},
    )
    deleted = repository.delete_item("dict_disabled")
    after_delete_codes = [row["item_code"] for row in repository.list_items("model_mode")]

    assert [row["item_code"] for row in first_rows] == ["default", "disabled", "recommendation"]
    assert first_rows == second_rows
    assert redis_client.build_key("v2", "dictionary", "items", "model_mode") in redis_client.values
    assert normalized_blank == "default"
    assert normalized_invalid == "default"
    assert normalized_recommend == "recommendation"
    assert metadata_code == "recommendation"
    assert saved["dictionary_item_id"] == "dict_model_mode_premium"
    assert saved["parent_item_id"] == "dict_parent"
    assert saved["item_level"] == 2
    assert deleted is True
    assert "disabled" not in after_delete_codes
    assert redis_client.build_key("v2", "dictionary", "items", "model_mode") in redis_client.deleted_keys
