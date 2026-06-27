"""上传预览临时对象清理服务。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app_v2.application.knowledge.upload_preview_state import PREVIEW_OBJECT_PREFIX, load_upload_preview_config
from app_v2.infrastructure.file_storage_service import FileStorageService, get_file_storage_service
from core.utils.logger_handler import logger


@dataclass(frozen=True)
class PreviewCleanupResult:
    """MinIO 预览临时对象清理结果。"""

    scanned_count: int = 0
    deleted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    errors: list[str] = field(default_factory=list)
    locked: bool = True
    skipped_reason: str | None = None


class PreviewUploadCleanupService:
    """清理 MinIO previews 前缀下的过期临时上传对象。"""

    def __init__(
            self,
            *,
            file_storage_service: FileStorageService | None = None,
            ttl_seconds: int | None = None,
            now_provider: Callable[[], datetime] | None = None,
    ):
        self.file_storage_service = file_storage_service or get_file_storage_service()
        self.ttl_seconds = ttl_seconds or load_upload_preview_config().ttl_seconds
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def cleanup_expired_previews(self) -> PreviewCleanupResult:
        """扫描并删除超过 TTL 的 previews 临时对象。"""

        scanned_count = 0
        deleted_count = 0
        skipped_count = 0
        failed_count = 0
        errors: list[str] = []
        now = self._as_aware_utc(self.now_provider())

        for item in self.file_storage_service.list_objects(prefix=PREVIEW_OBJECT_PREFIX):
            scanned_count += 1
            object_name = str(getattr(item, "object_name", "") or "")
            bucket_name = getattr(item, "bucket_name", None)
            last_modified = self._as_aware_utc(getattr(item, "last_modified", None))

            if not object_name.startswith(PREVIEW_OBJECT_PREFIX):
                skipped_count += 1
                continue
            if last_modified is None or (now - last_modified).total_seconds() < self.ttl_seconds:
                skipped_count += 1
                continue

            try:
                self.file_storage_service.delete_object(bucket_name=bucket_name, object_name=object_name)
                deleted_count += 1
                logger.info("[V2知识库] 过期预览对象已清理 桶名=%s 对象名=%s", bucket_name, object_name)
            except Exception as exc:
                failed_count += 1
                message = f"{bucket_name}/{object_name}: {exc}"
                errors.append(message)
                logger.warning("[V2知识库] 清理过期预览对象失败 桶名=%s 对象名=%s 错误=%s", bucket_name, object_name, exc)

        return PreviewCleanupResult(
            scanned_count=scanned_count,
            deleted_count=deleted_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            errors=errors,
        )

    @staticmethod
    def _as_aware_utc(value):
        """把 MinIO 返回的时间统一转换为 UTC aware datetime。"""

        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
