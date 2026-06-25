from fastapi.testclient import TestClient

import api.routers.internal_jobs as internal_jobs
from api.main import app


class FakeRedisLock:
    """测试用 Redis 锁上下文。"""

    def __init__(self, acquired):
        self.acquired = acquired

    def __enter__(self):
        return self.acquired

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeRedisClient:
    """测试用 Redis 客户端，只模拟分布式锁。"""

    def __init__(self, acquired=True):
        self.acquired = acquired

    def lock(self, lock_name, resource_id, timeout_seconds=None, blocking_timeout_seconds=0):
        return FakeRedisLock(self.acquired)


class FakeCleanupService:
    """测试用清理服务，返回固定清理统计。"""

    def cleanup_expired_previews(self):
        return type(
            "Result",
            (),
            {
                "scanned_count": 2,
                "deleted_count": 1,
                "skipped_count": 1,
                "failed_count": 0,
                "errors": [],
                "locked": True,
                "skipped_reason": None,
            },
        )()


def test_cleanup_preview_uploads_requires_internal_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_JOB_TOKEN", "secret")
    client = TestClient(app)

    response = client.post("/internal/jobs/minio/cleanup-preview-uploads")

    assert response.status_code == 403


def test_cleanup_preview_uploads_runs_with_valid_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_JOB_TOKEN", "secret")
    monkeypatch.setattr(internal_jobs, "get_redis_client", lambda: FakeRedisClient(acquired=True))
    monkeypatch.setattr(internal_jobs, "PreviewUploadCleanupService", lambda: FakeCleanupService())
    client = TestClient(app)

    response = client.post(
        "/internal/jobs/minio/cleanup-preview-uploads",
        headers={"X-INTERNAL-JOB-TOKEN": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["deleted_count"] == 1


def test_cleanup_preview_uploads_skips_when_lock_not_acquired(monkeypatch):
    monkeypatch.setenv("INTERNAL_JOB_TOKEN", "secret")
    monkeypatch.setattr(internal_jobs, "get_redis_client", lambda: FakeRedisClient(acquired=False))
    client = TestClient(app)

    response = client.post(
        "/internal/jobs/minio/cleanup-preview-uploads",
        headers={"X-INTERNAL-JOB-TOKEN": "secret"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert response.json()["locked"] is False
