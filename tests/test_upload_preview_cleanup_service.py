from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app_v2.application.knowledge.upload_preview_cleanup_service import PreviewUploadCleanupService


class FakeFileStorage:
    """测试用文件存储外观，记录被删除的 MinIO 对象。"""

    def __init__(self, objects):
        self.objects = objects
        self.deleted = []

    def list_objects(self, *, prefix, bucket_name=None):
        assert prefix == "previews/"
        return self.objects

    def delete_object(self, *, object_name, bucket_name=None):
        self.deleted.append((bucket_name, object_name))
        return True


def test_cleanup_deletes_only_expired_preview_objects():
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(hours=25)
    fresh_time = now - timedelta(hours=1)
    objects = [
        SimpleNamespace(bucket_name="pub", object_name="previews/tmp_old/demo.txt", last_modified=old_time, size=10),
        SimpleNamespace(bucket_name="pub", object_name="previews/tmp_fresh/demo.txt", last_modified=fresh_time, size=10),
        SimpleNamespace(bucket_name="pub", object_name="documents/doc_001/demo.txt", last_modified=old_time, size=10),
    ]
    storage = FakeFileStorage(objects)

    result = PreviewUploadCleanupService(
        file_storage_service=storage,
        ttl_seconds=86400,
        now_provider=lambda: now,
    ).cleanup_expired_previews()

    assert result.scanned_count == 3
    assert result.deleted_count == 1
    assert result.skipped_count == 2
    assert result.failed_count == 0
    assert storage.deleted == [("pub", "previews/tmp_old/demo.txt")]


def test_cleanup_counts_delete_failures():
    now = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    old_time = now - timedelta(hours=25)

    class FailingStorage(FakeFileStorage):
        def delete_object(self, *, object_name, bucket_name=None):
            raise RuntimeError("delete failed")

    storage = FailingStorage([
        SimpleNamespace(bucket_name="pub", object_name="previews/tmp_old/demo.txt", last_modified=old_time, size=10)
    ])

    result = PreviewUploadCleanupService(
        file_storage_service=storage,
        ttl_seconds=86400,
        now_provider=lambda: now,
    ).cleanup_expired_previews()

    assert result.deleted_count == 0
    assert result.failed_count == 1
