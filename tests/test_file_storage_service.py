from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi import UploadFile

import infrastructure.file_storage_service as storage_module
from infrastructure.file_storage_service import FileStorageService
from utils.minio_client import MinioObjectInfo


class FakeStorageClient:
    """测试用文件存储客户端，模拟 MinIO 上传、复制、下载和删除。"""

    def __init__(self):
        self.deleted: list[tuple[str | None, str]] = []

    def upload_file(self, file_path: str, object_name: str | None = None):
        """模拟上传本地临时文件到 MinIO。"""

        return MinioObjectInfo(
            bucket_name="pub",
            object_name=object_name or "files/demo.txt",
            public_url=f"http://127.0.0.1:9000/pub/{object_name}",
            file_size=Path(file_path).stat().st_size,
            content_type="text/plain",
        )

    def copy_object(self, source_object_name: str, target_object_name: str, **kwargs):
        """模拟 MinIO 内部复制对象。"""

        return MinioObjectInfo(
            bucket_name="pub",
            object_name=target_object_name,
            public_url=f"http://127.0.0.1:9000/pub/{target_object_name}",
            file_size=12,
            content_type="text/plain",
        )

    def delete_object(self, object_name: str, bucket_name: str | None = None):
        """模拟删除 MinIO 对象。"""

        self.deleted.append((bucket_name, object_name))
        return True

    def download_file(self, object_name: str, target_path: str, bucket_name: str | None = None):
        """模拟下载 MinIO 对象到临时文件。"""

        Path(target_path).write_text("hello minio", encoding="utf-8")
        return target_path

    def ensure_bucket(self):
        """模拟 MinIO 桶可用检查。"""

        return "pub"


def test_file_storage_upload_copy_download_and_delete(monkeypatch):
    """统一文件存储服务应把上传、复制、下载、删除都收敛到 MinIO。"""

    fake_client = FakeStorageClient()
    monkeypatch.setattr(storage_module, "get_minio_client", lambda: fake_client)
    service = FileStorageService()
    upload_file = UploadFile(filename="demo.txt", file=BytesIO(b"hello minio"))

    stored = service.save_upload_file(
        file=upload_file,
        filename="demo.txt",
        prefix="previews",
        owner_id="tmp_001",
    )
    copied = service.copy_object(source=stored, prefix="documents", owner_id="doc_001")
    with service.downloaded_temp_file(
            bucket_name=copied.bucket_name,
            object_name=copied.object_name,
            filename=copied.filename,
    ) as temp_path:
        assert Path(temp_path).read_text(encoding="utf-8") == "hello minio"
    deleted = service.delete_object(bucket_name=copied.bucket_name, object_name=copied.object_name)

    assert stored.file_md5 == "1738ebfeeab21fef70b0622d63af59d3"
    assert stored.object_name == "previews/tmp_001/demo.txt"
    assert stored.file_path == "minio://pub/previews/tmp_001/demo.txt"
    assert copied.object_name == "documents/doc_001/demo.txt"
    assert deleted is True
    assert fake_client.deleted == [("pub", "documents/doc_001/demo.txt")]


def test_file_storage_save_local_file_uses_minio_facade(monkeypatch, tmp_path):
    """本地初始化文件或历史迁移文件也应通过统一文件存储服务进入 MinIO。"""

    fake_client = FakeStorageClient()
    monkeypatch.setattr(storage_module, "get_minio_client", lambda: fake_client)
    service = FileStorageService()

    source_file = tmp_path / "内置知识.txt"
    source_file.write_text("hello local file", encoding="utf-8")

    stored = service.save_local_file(
        file_path=str(source_file),
        filename=source_file.name,
        prefix="documents",
        owner_id="doc_local",
    )

    assert service.ensure_bucket_ready() == "pub"
    assert stored.file_md5 == "ef91f5bb45bfdad80cc73904db85a63b"
    assert stored.object_name == "documents/doc_local/内置知识.txt"
    assert stored.file_path == "minio://pub/documents/doc_local/内置知识.txt"


def test_file_storage_lists_objects_through_minio_facade(monkeypatch):
    """文件存储外观应提供 MinIO 前缀扫描能力，供清理任务复用。"""

    fake_client = FakeStorageClient()
    fake_client.list_objects = lambda prefix, bucket_name=None: [
        SimpleNamespace(
            bucket_name=bucket_name or "pub",
            object_name=f"{prefix}tmp_001/demo.txt",
            last_modified=None,
            size=12,
        )
    ]
    monkeypatch.setattr(storage_module, "get_minio_client", lambda: fake_client)

    objects = FileStorageService().list_objects(prefix="previews/")

    assert objects[0].object_name == "previews/tmp_001/demo.txt"
