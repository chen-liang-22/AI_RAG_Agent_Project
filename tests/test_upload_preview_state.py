from datetime import datetime, timedelta, timezone

from app_v2.application.knowledge.upload_preview_state import (
    PreviewUploadMetadata,
    PreviewUploadStore,
    load_upload_preview_config,
)
from infrastructure.file_storage_service import StoredFileInfo


class FakeRedisClient:
    """测试用 Redis 客户端，记录 JSON 写入内容和 TTL。"""

    def __init__(self):
        self.values = {}
        self.deleted = []

    def build_key(self, *parts):
        return "test:" + ":".join(str(part) for part in parts)

    def set_json(self, key, value, ttl_seconds=None):
        self.values[key] = {"value": value, "ttl_seconds": ttl_seconds}
        return True

    def get_json(self, key, default=None):
        return self.values.get(key, {}).get("value", default)

    def delete(self, *keys):
        count = 0
        for key in keys:
            self.deleted.append(key)
            if key in self.values:
                count += 1
                del self.values[key]
        return count


def test_preview_upload_store_saves_metadata_without_file_body():
    redis_client = FakeRedisClient()
    store = PreviewUploadStore(redis_client=redis_client, ttl_seconds=86400)
    stored_file = StoredFileInfo(
        filename="demo.pdf",
        file_type="pdf",
        file_md5="md5",
        file_size=123,
        bucket_name="pub",
        object_name="previews/tmp_001/demo.pdf",
        public_url="http://127.0.0.1:9000/pub/previews/tmp_001/demo.pdf",
        file_path="minio://pub/previews/tmp_001/demo.pdf",
    )

    metadata = store.save("tmp_001", stored_file)

    key = "test:upload_preview:tmp_001"
    assert redis_client.values[key]["ttl_seconds"] == 86400
    assert redis_client.values[key]["value"]["object_name"] == "previews/tmp_001/demo.pdf"
    assert "sample_text" not in redis_client.values[key]["value"]
    assert "content" not in redis_client.values[key]["value"]
    assert metadata.to_stored_file_info() == stored_file


def test_preview_upload_store_loads_and_deletes_metadata():
    redis_client = FakeRedisClient()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=86400)
    key = "test:upload_preview:tmp_002"
    redis_client.values[key] = {
        "value": {
            "upload_id": "tmp_002",
            "filename": "demo.txt",
            "file_type": "txt",
            "file_md5": "md5",
            "file_size": 10,
            "bucket_name": "pub",
            "object_name": "previews/tmp_002/demo.txt",
            "public_url": "http://127.0.0.1:9000/pub/previews/tmp_002/demo.txt",
            "file_path": "minio://pub/previews/tmp_002/demo.txt",
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        },
        "ttl_seconds": 86400,
    }
    store = PreviewUploadStore(redis_client=redis_client, ttl_seconds=86400)

    metadata = store.get("tmp_002")
    deleted = store.delete("tmp_002")

    assert isinstance(metadata, PreviewUploadMetadata)
    assert metadata.object_name == "previews/tmp_002/demo.txt"
    assert deleted is True
    assert key in redis_client.deleted


def test_load_upload_preview_config_has_safe_defaults():
    config = load_upload_preview_config({})

    assert config.ttl_seconds == 86400
    assert config.max_file_size_bytes == 52428800
    assert config.sample_text_chars == 5000
    assert config.recommendation_sample_chars == 10000
