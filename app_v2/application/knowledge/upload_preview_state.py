"""知识库上传预览状态存储。

本模块只保存 upload_id 到 MinIO 临时对象的短期元数据，不保存文件正文或预览文本。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app_v2.infrastructure.file_storage_service import StoredFileInfo
from core.utils.config_handler import qdrant_conf
from core.utils.redis_client import RedisClient, get_redis_client


DEFAULT_PREVIEW_TTL_SECONDS = 86400
DEFAULT_PREVIEW_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
DEFAULT_PREVIEW_SAMPLE_TEXT_CHARS = 5000
DEFAULT_RECOMMENDATION_SAMPLE_CHARS = 10000
PREVIEW_OBJECT_PREFIX = "previews/"


@dataclass(frozen=True)
class UploadPreviewConfig:
    """上传预览生命周期配置。"""

    ttl_seconds: int
    max_file_size_bytes: int
    sample_text_chars: int
    recommendation_sample_chars: int


@dataclass(frozen=True)
class PreviewUploadMetadata:
    """Redis 中保存的上传预览元数据。"""

    upload_id: str
    filename: str
    file_type: str
    file_md5: str
    file_size: int
    bucket_name: str
    object_name: str
    public_url: str
    file_path: str
    created_at: str
    expires_at: str

    @classmethod
    def from_stored_file(
            cls,
            *,
            upload_id: str,
            stored_file: StoredFileInfo,
            ttl_seconds: int,
            now: datetime | None = None,
    ) -> "PreviewUploadMetadata":
        """从 MinIO 上传结果生成可写入 Redis 的元数据。"""

        created_at = now or datetime.now(timezone.utc)
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        return cls(
            upload_id=upload_id,
            filename=stored_file.filename,
            file_type=stored_file.file_type,
            file_md5=stored_file.file_md5,
            file_size=stored_file.file_size,
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
            public_url=stored_file.public_url,
            file_path=stored_file.file_path,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreviewUploadMetadata":
        """从 Redis JSON 载荷恢复预览元数据。"""

        return cls(
            upload_id=str(value["upload_id"]),
            filename=str(value["filename"]),
            file_type=str(value["file_type"]),
            file_md5=str(value["file_md5"]),
            file_size=int(value["file_size"]),
            bucket_name=str(value["bucket_name"]),
            object_name=str(value["object_name"]),
            public_url=str(value.get("public_url") or ""),
            file_path=str(value["file_path"]),
            created_at=str(value["created_at"]),
            expires_at=str(value["expires_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为 Redis JSON 载荷。"""

        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_md5": self.file_md5,
            "file_size": self.file_size,
            "bucket_name": self.bucket_name,
            "object_name": self.object_name,
            "public_url": self.public_url,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    def to_stored_file_info(self) -> StoredFileInfo:
        """转换回业务层通用的 MinIO 文件信息。"""

        return StoredFileInfo(
            filename=self.filename,
            file_type=self.file_type,
            file_md5=self.file_md5,
            file_size=self.file_size,
            bucket_name=self.bucket_name,
            object_name=self.object_name,
            public_url=self.public_url,
            file_path=self.file_path,
        )


def _positive_int(value: Any, default: int) -> int:
    """读取正整数配置，非法值回退到默认值。"""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_upload_preview_config(raw_config: dict[str, Any] | None = None) -> UploadPreviewConfig:
    """读取上传预览配置，并提供生产安全默认值。"""

    source = raw_config if raw_config is not None else qdrant_conf
    section = source.get("upload_preview") if isinstance(source, dict) else {}
    section = section if isinstance(section, dict) else {}
    return UploadPreviewConfig(
        ttl_seconds=_positive_int(section.get("ttl_seconds"), DEFAULT_PREVIEW_TTL_SECONDS),
        max_file_size_bytes=_positive_int(
            section.get("max_file_size_bytes"),
            DEFAULT_PREVIEW_MAX_FILE_SIZE_BYTES,
        ),
        sample_text_chars=_positive_int(section.get("sample_text_chars"), DEFAULT_PREVIEW_SAMPLE_TEXT_CHARS),
        recommendation_sample_chars=_positive_int(
            section.get("recommendation_sample_chars"),
            DEFAULT_RECOMMENDATION_SAMPLE_CHARS,
        ),
    )


class PreviewUploadStore:
    """上传预览 Redis 存储外观。"""

    def __init__(self, *, redis_client: RedisClient | None = None, ttl_seconds: int | None = None):
        self.redis_client = redis_client or get_redis_client()
        self.ttl_seconds = ttl_seconds or load_upload_preview_config().ttl_seconds

    def save(self, upload_id: str, stored_file: StoredFileInfo) -> PreviewUploadMetadata:
        """保存 upload_id 对应的 MinIO 临时对象元数据。"""

        metadata = PreviewUploadMetadata.from_stored_file(
            upload_id=upload_id,
            stored_file=stored_file,
            ttl_seconds=self.ttl_seconds,
        )
        saved = self.redis_client.set_json(self.key(upload_id), metadata.to_dict(), ttl_seconds=self.ttl_seconds)
        if not saved:
            raise RuntimeError("Redis 预览上传元数据写入失败")
        return metadata

    def get(self, upload_id: str) -> PreviewUploadMetadata | None:
        """读取 upload_id 对应的预览元数据。"""

        value = self.redis_client.get_json(self.key(upload_id), default=None)
        if not isinstance(value, dict):
            return None
        return PreviewUploadMetadata.from_dict(value)

    def delete(self, upload_id: str) -> bool:
        """删除 upload_id 对应的预览元数据。"""

        return self.redis_client.delete(self.key(upload_id)) > 0

    def key(self, upload_id: str) -> str:
        """生成 Redis key。"""

        return self.redis_client.build_key("upload_preview", upload_id.strip())


def get_preview_upload_store() -> PreviewUploadStore:
    """创建上传预览状态存储。"""

    return PreviewUploadStore()
