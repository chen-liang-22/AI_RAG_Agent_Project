from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

import app_v2.application.knowledge.upload_preview_service as upload_services
from infrastructure.file_storage_service import StoredFileInfo


class FakePreviewStore:
    """测试用预览状态存储，模拟 Redis 元数据读写。"""

    def __init__(self):
        self.values = {}
        self.deleted = []

    def save(self, upload_id, stored_file):
        self.values[upload_id] = stored_file
        return stored_file

    def get(self, upload_id):
        stored_file = self.values.get(upload_id)
        if stored_file is None:
            return None

        class Metadata:
            def to_stored_file_info(self):
                return stored_file

        return Metadata()

    def delete(self, upload_id):
        self.deleted.append(upload_id)
        return upload_id in self.values


class FakeFileStorage:
    """测试用文件存储服务，模拟 MinIO 上传和删除。"""

    def __init__(self, stored_file):
        self.stored_file = stored_file
        self.deleted = []

    def save_upload_file(self, **kwargs):
        return self.stored_file

    def delete_object(self, *, object_name, bucket_name=None):
        self.deleted.append((bucket_name, object_name))
        return True


def _stored_file(file_size=123):
    return StoredFileInfo(
        filename="demo.txt",
        file_type="txt",
        file_md5="md5",
        file_size=file_size,
        bucket_name="pub",
        object_name="previews/tmp_001/demo.txt",
        public_url="http://127.0.0.1:9000/pub/previews/tmp_001/demo.txt",
        file_path="minio://pub/previews/tmp_001/demo.txt",
    )


def test_save_and_get_preview_file_uses_redis_store(monkeypatch):
    preview_store = FakePreviewStore()
    file_storage = FakeFileStorage(_stored_file())
    monkeypatch.setattr(upload_services, "get_preview_upload_store", lambda: preview_store)
    monkeypatch.setattr(upload_services, "get_file_storage_service", lambda: file_storage)
    monkeypatch.setattr(
        upload_services,
        "load_upload_preview_config",
        lambda: type("Config", (), {"max_file_size_bytes": 1024})(),
    )

    upload_file = UploadFile(filename="demo.txt", file=BytesIO(b"hello"))
    saved = upload_services._save_preview_file(upload_file, "demo.txt", "tmp_001")
    loaded = upload_services._get_preview_file("tmp_001")

    assert saved.object_name == "previews/tmp_001/demo.txt"
    assert loaded.object_name == "previews/tmp_001/demo.txt"
    assert preview_store.values["tmp_001"].file_md5 == "md5"


def test_get_preview_file_returns_expired_message_when_missing(monkeypatch):
    preview_store = FakePreviewStore()
    monkeypatch.setattr(upload_services, "get_preview_upload_store", lambda: preview_store)

    with pytest.raises(HTTPException) as exc_info:
        upload_services._get_preview_file("tmp_missing")

    assert exc_info.value.status_code == 404
    assert "已过期或不存在" in exc_info.value.detail


def test_save_preview_file_deletes_minio_object_when_too_large(monkeypatch):
    preview_store = FakePreviewStore()
    file_storage = FakeFileStorage(_stored_file(file_size=2048))
    monkeypatch.setattr(upload_services, "get_preview_upload_store", lambda: preview_store)
    monkeypatch.setattr(upload_services, "get_file_storage_service", lambda: file_storage)
    monkeypatch.setattr(
        upload_services,
        "load_upload_preview_config",
        lambda: type("Config", (), {"max_file_size_bytes": 1024})(),
    )

    upload_file = UploadFile(filename="demo.txt", file=BytesIO(b"x" * 2048))

    with pytest.raises(HTTPException) as exc_info:
        upload_services._save_preview_file(upload_file, "demo.txt", "tmp_001")

    assert exc_info.value.status_code == 400
    assert "文件过大" in exc_info.value.detail
    assert file_storage.deleted == [("pub", "previews/tmp_001/demo.txt")]
    assert preview_store.values == {}
