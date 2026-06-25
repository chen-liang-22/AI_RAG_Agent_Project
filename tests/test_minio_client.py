from __future__ import annotations

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
