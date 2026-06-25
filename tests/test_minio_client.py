from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

import pytest

from pathlib import Path

from utils import minio_client
from utils.minio_client import MinioObjectInfo, MinioStorageClient
from utils.minio_client import get_minio_client, reset_minio_client


class FakeUploadResult:
    """测试用上传结果对象。"""

    def __init__(self, etag: str = "etag-001"):
        self.etag = etag


class FakeListedObject:
    """测试用 MinIO 对象列表项。"""

    def __init__(self, object_name, last_modified, size):
        self.object_name = object_name
        self.last_modified = last_modified
        self.size = size


class FakeMinioClient:
    """测试用 MinIO 客户端替身。"""

    def __init__(self, bucket_exists: bool = True):
        self.bucket_exists_flag = bucket_exists
        self.make_bucket_calls: list[str] = []
        self.fput_calls: list[dict[str, str]] = []
        self.put_calls: list[dict[str, object]] = []
        self.remove_calls: list[tuple[str, str]] = []
        self.stat_calls: list[tuple[str, str]] = []
        self.policy_calls: list[tuple[str, str]] = []
        self.copy_calls: list[dict[str, object]] = []
        self.download_calls: list[tuple[str, str, str]] = []

    def bucket_exists(self, bucket_name: str):
        """模拟检查桶是否存在。"""

        return self.bucket_exists_flag

    def make_bucket(self, bucket_name: str):
        """模拟创建桶。"""

        self.make_bucket_calls.append(bucket_name)
        self.bucket_exists_flag = True

    def fput_object(self, bucket_name: str, object_name: str, file_path: str, content_type: str):
        """模拟上传本地文件。"""

        self.fput_calls.append(
            {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "file_path": file_path,
                "content_type": content_type,
            }
        )
        return FakeUploadResult()

    def put_object(self, bucket_name: str, object_name: str, data, length: int, content_type: str):
        """模拟上传文件流。"""

        self.put_calls.append(
            {
                "bucket_name": bucket_name,
                "object_name": object_name,
                "data": data,
                "length": length,
                "content_type": content_type,
            }
        )
        return FakeUploadResult("etag-002")

    def remove_object(self, bucket_name: str, object_name: str):
        """模拟删除对象。"""

        self.remove_calls.append((bucket_name, object_name))

    def stat_object(self, bucket_name: str, object_name: str):
        """模拟查询对象元数据。"""

        self.stat_calls.append((bucket_name, object_name))
        if object_name == "missing.txt":
            raise RuntimeError("not found")
        return SimpleNamespace(etag="etag-stat", size=9, content_type="text/plain")

    def set_bucket_policy(self, bucket_name: str, policy: str):
        """模拟设置桶策略。"""

        self.policy_calls.append((bucket_name, policy))

    def copy_object(self, bucket_name: str, object_name: str, source):
        """模拟复制对象。"""

        self.copy_calls.append({"bucket_name": bucket_name, "object_name": object_name, "source": source})
        return FakeUploadResult("etag-copy")

    def fget_object(self, bucket_name: str, object_name: str, file_path: str):
        """模拟下载对象到本地文件。"""

        Path(file_path).write_text("downloaded", encoding="utf-8")
        self.download_calls.append((bucket_name, object_name, file_path))


def test_minio_upload_file_delete_and_public_url(monkeypatch, tmp_path):
    """验证本地文件上传、删除和公开地址拼接。"""

    fake_sdk = SimpleNamespace(Minio=lambda **kwargs: FakeMinioClient(bucket_exists=True))
    monkeypatch.setattr(minio_client, "minio", fake_sdk)
    monkeypatch.setattr(minio_client, "S3Error", RuntimeError)

    file_path = tmp_path / "demo.txt"
    file_path.write_text("hello minio", encoding="utf-8")

    client = MinioStorageClient(
        {
            "enabled": True,
            "endpoint": "127.0.0.1:9000",
            "access_key": "admin",
            "secret_key": "1234qwer",
            "secure": False,
            "bucket_name": "pub",
            "public_base_url": "http://127.0.0.1:9000",
            "auto_create_bucket": False,
            "public_read": True,
        }
    )

    result = client.upload_file(str(file_path), object_name="docs/demo.txt")
    assert isinstance(result, MinioObjectInfo)
    assert result.bucket_name == "pub"
    assert result.object_name == "docs/demo.txt"
    assert result.public_url == "http://127.0.0.1:9000/pub/docs/demo.txt"
    assert result.file_size == len("hello minio".encode("utf-8"))
    assert result.content_type == "text/plain"
    assert len(client.client().policy_calls) == 1

    assert client.object_exists("docs/demo.txt") is True
    assert client.object_exists("missing.txt") is False
    assert client.delete_object("docs/demo.txt") is True

    copied = client.copy_object("docs/demo.txt", "docs/copied.txt")
    downloaded_path = tmp_path / "downloaded.txt"
    client.download_file("docs/copied.txt", str(downloaded_path))
    assert copied.object_name == "docs/copied.txt"
    assert downloaded_path.read_text(encoding="utf-8") == "downloaded"


def test_minio_list_objects_returns_clean_summaries(monkeypatch):
    """按前缀列出对象时应返回业务可读的轻量摘要。"""

    listed_at = datetime(2026, 6, 25, tzinfo=timezone.utc)
    fake_client = FakeMinioClient(bucket_exists=True)

    def fake_list_objects(bucket_name, prefix="", recursive=True):
        assert bucket_name == "pub"
        assert prefix == "previews/"
        assert recursive is True
        return [
            FakeListedObject("previews/tmp_001/demo.txt", listed_at, 12),
            FakeListedObject("previews/tmp_002/demo.pdf", listed_at, 34),
        ]

    fake_client.list_objects = fake_list_objects
    fake_sdk = SimpleNamespace(Minio=lambda **kwargs: fake_client)
    monkeypatch.setattr(minio_client, "minio", fake_sdk)
    monkeypatch.setattr(minio_client, "S3Error", RuntimeError)

    client = MinioStorageClient(
        {
            "enabled": True,
            "endpoint": "127.0.0.1:9000",
            "access_key": "admin",
            "secret_key": "1234qwer",
            "secure": False,
            "bucket_name": "pub",
            "auto_create_bucket": False,
        }
    )

    objects = client.list_objects(prefix="previews/")

    assert [item.object_name for item in objects] == [
        "previews/tmp_001/demo.txt",
        "previews/tmp_002/demo.pdf",
    ]
    assert objects[0].bucket_name == "pub"
    assert objects[0].last_modified == listed_at
    assert objects[0].size == 12


def test_minio_auto_create_bucket(monkeypatch, tmp_path):
    """验证自动创建桶的行为。"""

    fake_sdk = SimpleNamespace(Minio=lambda **kwargs: FakeMinioClient(bucket_exists=False))
    monkeypatch.setattr(minio_client, "minio", fake_sdk)
    monkeypatch.setattr(minio_client, "S3Error", RuntimeError)

    file_path = tmp_path / "stream.bin"
    file_path.write_bytes(b"abc123")

    client = MinioStorageClient(
        {
            "enabled": True,
            "endpoint": "127.0.0.1:9000",
            "access_key": "admin",
            "secret_key": "1234qwer",
            "secure": False,
            "bucket_name": "pub",
            "public_base_url": "http://127.0.0.1:9000",
            "auto_create_bucket": True,
        }
    )

    uploaded = client.upload_stream(
        stream=BytesIO(b"abc123"),
        source_name="stream.bin",
        file_size=6,
        object_name="files/stream.bin",
    )

    assert uploaded.object_name == "files/stream.bin"
    assert uploaded.public_url == "http://127.0.0.1:9000/pub/files/stream.bin"


def test_minio_missing_bucket_raises(monkeypatch):
    """验证桶不存在且不允许自动创建时会直接报错。"""

    fake_sdk = SimpleNamespace(Minio=lambda **kwargs: FakeMinioClient(bucket_exists=False))
    monkeypatch.setattr(minio_client, "minio", fake_sdk)
    monkeypatch.setattr(minio_client, "S3Error", RuntimeError)

    client = MinioStorageClient(
        {
            "enabled": True,
            "endpoint": "127.0.0.1:9000",
            "access_key": "admin",
            "secret_key": "1234qwer",
            "secure": False,
            "bucket_name": "pub",
            "public_base_url": "http://127.0.0.1:9000",
            "auto_create_bucket": False,
        }
    )

    with pytest.raises(RuntimeError, match="MinIO 存储桶不存在"):
        client.ensure_bucket()



@pytest.mark.skipif(os.getenv("RUN_REAL_MINIO_TESTS") != "1", reason="真实 MinIO 手动验证用例默认跳过")
def test_real_minio_upload_file():
    """真实上传文件到 MinIO，用于本地手动验证。"""

    reset_minio_client()

    file_path = Path("uploads/manual_minio_test.txt")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("hello real minio", encoding="utf-8")

    client = get_minio_client()
    info = client.upload_file(
        file_path=str(file_path),
        object_name="files/manual_minio_test.txt",
    )

    assert info.object_name == "files/manual_minio_test.txt"
    assert client.object_exists(info.object_name) is True
    print(info.public_url)
