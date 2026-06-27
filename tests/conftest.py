"""pytest 全局隔离配置。

项目运行时使用 MySQL、MinIO、Qdrant 等真实外部组件；单元测试默认不能写这些真实资源。
本文件在 pytest 启动时统一切到内存 SQLite，并替换销售训练上传链路的文件存储服务。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from app_v2.infrastructure.repositories.default_dictionaries import DEFAULT_DICTIONARY_ITEMS
from domain.entities import BaseOrmModel
from infrastructure import orm_session
from infrastructure.file_storage_service import FileStorageService, StoredFileInfo
from training.repository import utc_now
from utils.file_handler import get_file_md5_hex


os.environ.setdefault("AI_RAG_TESTING", "1")
os.environ.setdefault("REDIS_ENABLED", "false")


def _build_sqlite_session_factory():
    """创建测试专用 SQLite SessionFactory。"""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    BaseOrmModel.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _seed_default_dictionaries(factory) -> None:
    """向测试库写入模型档位等默认字典，支撑导入期配置读取。"""

    from domain.entities import DictionaryItemEntity

    now = utc_now()
    with factory() as session:
        for dictionary in DEFAULT_DICTIONARY_ITEMS:
            dictionary_code = dictionary["dictionary_code"]
            dictionary_name = dictionary["dictionary_name"]
            item_id_by_code: dict[str, str] = {}
            for item in dictionary["items"]:
                item_code, item_name, parent_code, sort_order, description = item[:5]
                metadata = item[5] if len(item) > 5 else None
                parent_item_id = item_id_by_code.get(parent_code or "")
                dictionary_item_id = f"test_dict_{dictionary_code}_{item_code}".replace("-", "_")
                session.add(DictionaryItemEntity(
                    dictionary_item_id=dictionary_item_id,
                    dictionary_code=dictionary_code,
                    dictionary_name=dictionary_name,
                    item_code=item_code,
                    item_name=item_name,
                    parent_item_id=parent_item_id,
                    item_level=1 if parent_item_id is None else 2,
                    sort_order=int(sort_order),
                    enabled=1,
                    description=description,
                    metadata_json=__import__("json").dumps(metadata, ensure_ascii=False) if metadata else None,
                    created_at=now,
                    updated_at=now,
                ))
                item_id_by_code[str(item_code)] = dictionary_item_id
        session.commit()


def _install_test_session_factory():
    """安装测试 SessionFactory，供测试模块导入阶段使用。"""

    factory = _build_sqlite_session_factory()
    _seed_default_dictionaries(factory)
    orm_session.set_session_factory_override(factory)
    return factory


_INITIAL_FACTORY = _install_test_session_factory()


class FakeFileStorageService:
    """测试用文件存储服务，模拟 MinIO 但只写系统临时目录。"""

    def __init__(self) -> None:
        """初始化临时对象目录和删除记录。"""

        self.root_dir = Path(tempfile.mkdtemp(prefix="ai_rag_test_storage_"))
        self.deleted_objects: list[tuple[str | None, str | None]] = []

    def save_upload_file(self, *, file, filename: str, prefix: str, owner_id: str) -> StoredFileInfo:
        """把上传流保存到测试临时目录，并返回与 MinIO 一致的文件信息。"""

        safe_filename = Path(filename).name
        object_name = f"{prefix.strip('/')}/{owner_id.strip('/')}/{safe_filename}"
        target_path = self.root_dir / object_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file.file.seek(0)
        with target_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
        file.file.seek(0)

        file_md5 = get_file_md5_hex(str(target_path))
        return StoredFileInfo(
            filename=safe_filename,
            file_type=target_path.suffix.lower().lstrip("."),
            file_md5=file_md5,
            file_size=target_path.stat().st_size,
            bucket_name="test-bucket",
            object_name=object_name,
            public_url=f"http://test.local/test-bucket/{object_name}",
            file_path=FileStorageService.build_storage_uri("test-bucket", object_name),
        )

    def delete_object(self, *, object_name: str | None, bucket_name: str | None = None) -> bool:
        """删除测试临时目录里的对象。"""

        self.deleted_objects.append((bucket_name, object_name))
        if not object_name:
            return False
        target_path = self.root_dir / object_name
        if target_path.exists():
            target_path.unlink()
            return True
        return False

    @contextmanager
    def downloaded_temp_file(
            self,
            *,
            bucket_name: str | None,
            object_name: str,
            filename: str,
    ) -> Iterator[str]:
        """把测试对象复制到临时文件，模拟 MinIO 下载上下文。"""

        source_path = self.root_dir / object_name
        if not source_path.is_file():
            raise FileNotFoundError(f"测试文件对象不存在：{object_name}")
        temp_dir = Path(tempfile.mkdtemp(prefix="ai_rag_test_download_"))
        temp_path = temp_dir / Path(filename).name
        try:
            shutil.copyfile(source_path, temp_path)
            yield str(temp_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def cleanup(self) -> None:
        """清理测试临时对象目录。"""

        shutil.rmtree(self.root_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_orm_session_and_storage(monkeypatch):
    """每个测试使用独立内存 SQLite 和 fake 文件存储，防止污染真实业务数据。"""

    factory = _build_sqlite_session_factory()
    _seed_default_dictionaries(factory)
    orm_session.set_session_factory_override(factory)

    fake_storage = FakeFileStorageService()
    import app_v2.application.knowledge.document_asset_service as document_asset_service
    import app_v2.application.knowledge.upload_preview_service as upload_preview_service
    import app_v2.application.training.sales_training_core as sales_training_core

    monkeypatch.setattr(sales_training_core, "get_file_storage_service", lambda: fake_storage)
    monkeypatch.setattr(upload_preview_service, "get_file_storage_service", lambda: fake_storage)
    monkeypatch.setattr(document_asset_service, "get_file_storage_service", lambda: fake_storage)

    yield

    fake_storage.cleanup()
    orm_session.clear_session_factory_override()
    orm_session.reset_orm_engines()
