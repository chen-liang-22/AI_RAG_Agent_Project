"""MinIO 公共文件存储工具。

这个模块只负责把“本地文件 -> MinIO 对象”以及“MinIO 对象删除”封装起来，
业务层以后直接调用这里即可，不要把 MinIO SDK 调用散落到各个 service 里。
"""

from __future__ import annotations

import mimetypes
import os
import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import yaml

from utils.config_handler import load_env_file
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

try:
    import minio
    from minio.commonconfig import CopySource
    from minio.error import S3Error
except ImportError:  # pragma: no cover - 没装依赖时通过明确异常提示。
    minio = None
    CopySource = None  # type: ignore[assignment]
    S3Error = RuntimeError  # type: ignore[assignment]


DEFAULT_MINIO_CONFIG: dict[str, Any] = {
    "enabled": False,
    "endpoint": "127.0.0.1:9000",
    "access_key": "admin",
    "secret_key": "",
    "secure": False,
    "bucket_name": "pub",
    "public_base_url": "",
    "region": None,
    "auto_create_bucket": False,
    "public_read": False,
}

_minio_client: "MinioStorageClient | None" = None


@dataclass(frozen=True)
class MinioObjectInfo:
    """MinIO 对象信息。

    上传成功后返回这个结构，方便业务层拿到对象名、公开地址和尺寸。
    """

    bucket_name: str  # 存储桶名称
    object_name: str  # 对象路径
    public_url: str  # 公共访问地址
    etag: str | None = None  # 上传返回的 ETag
    file_size: int | None = None  # 文件大小，单位字节
    content_type: str | None = None  # MIME 类型


@dataclass(frozen=True)
class MinioObjectSummary:
    """MinIO 对象列表摘要，用于清理任务判断对象年龄。"""

    bucket_name: str
    object_name: str
    last_modified: Any
    size: int | None = None


def load_minio_config(config_path: str = get_abs_path("config/minio.yml")) -> dict[str, Any]:
    """读取 MinIO 配置，并允许环境变量覆盖关键连接参数。"""

    load_env_file()
    config = dict(DEFAULT_MINIO_CONFIG)
    path = Path(config_path)
    if path.exists():
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw_config, dict):
            config.update(raw_config)

    password_env = str(config.get("secret_key_env") or "MINIO_ROOT_PASSWORD")
    config.update({
        "enabled": _env_bool("MINIO_ENABLED", bool(config.get("enabled"))),
        "endpoint": os.getenv("MINIO_ENDPOINT", str(config.get("endpoint") or "127.0.0.1:9000")),
        "access_key": os.getenv("MINIO_ACCESS_KEY", str(config.get("access_key") or "admin")),
        "secret_key": os.getenv(password_env, os.getenv("MINIO_SECRET_KEY", str(config.get("secret_key") or ""))),
        "secure": _env_bool("MINIO_SECURE", bool(config.get("secure"))),
        "bucket_name": os.getenv("MINIO_BUCKET", str(config.get("bucket_name") or "pub")),
        "public_base_url": os.getenv("MINIO_PUBLIC_BASE_URL", str(config.get("public_base_url") or "")),
        "region": os.getenv("MINIO_REGION", config.get("region")),
        "auto_create_bucket": _env_bool(
            "MINIO_AUTO_CREATE_BUCKET",
            bool(config.get("auto_create_bucket")),
        ),
        "public_read": _env_bool("MINIO_PUBLIC_READ", bool(config.get("public_read"))),
    })
    return config


def get_minio_client() -> "MinioStorageClient":
    """获取进程级 MinIO 工具单例。"""

    global _minio_client
    if _minio_client is None:
        _minio_client = MinioStorageClient(load_minio_config())
    return _minio_client


def reset_minio_client() -> None:
    """重置 MinIO 工具单例，主要用于测试切换配置后重新加载。"""

    global _minio_client
    _minio_client = None


class MinioStorageClient:
    """MinIO 公共文件存储封装。

    这个类默认面向公开桶 pub：
    - 上传：把本地文件或文件流写入 MinIO；
    - 删除：按 object_name 删除对象；
    - 访问：拼出公开 URL，供前端直链或回显使用。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化 MinIO 存储客户端，并缓存配置。"""

        self.config = dict(DEFAULT_MINIO_CONFIG)
        self.config.update(config or {})
        self._client = None
        self._public_read_buckets: set[str] = set()

    @property
    def enabled(self) -> bool:
        """判断配置层是否启用 MinIO。"""

        return bool(self.config.get("enabled")) and minio is not None

    def client(self):
        """懒加载 MinIO SDK 客户端。"""

        if minio is None:
            raise RuntimeError("缺少 MinIO 依赖，请先执行 pip install -r requirements.txt")
        if not bool(self.config.get("enabled")):
            raise RuntimeError("MinIO 未启用，请检查 config/minio.yml 或 MINIO_ENABLED")
        if self._client is None:
            self._client = minio.Minio(
                endpoint=str(self.config.get("endpoint") or "127.0.0.1:9000"),
                access_key=str(self.config.get("access_key") or "admin"),
                secret_key=str(self.config.get("secret_key") or ""),
                secure=bool(self.config.get("secure")),
                region=str(self.config.get("region") or "") or None,
            )
        return self._client

    def ensure_bucket(self, bucket_name: str | None = None) -> str:
        """确保桶存在；如果配置了自动创建且桶不存在，则自动创建。"""

        final_bucket = self._bucket_name(bucket_name)
        client = self.client()
        try:
            exists = bool(client.bucket_exists(final_bucket))
        except S3Error as exc:
            raise RuntimeError(f"检查 MinIO 存储桶失败：{exc}") from exc

        if not exists and bool(self.config.get("auto_create_bucket")):
            try:
                client.make_bucket(final_bucket)
                logger.info("[MinIO] 自动创建存储桶 完成 桶名=%s", final_bucket)
                exists = True
            except S3Error as exc:
                raise RuntimeError(f"创建 MinIO 存储桶失败：{exc}") from exc
        if not exists:
            raise RuntimeError(f"MinIO 存储桶不存在：{final_bucket}，请先创建或开启自动创建")
        if bool(self.config.get("public_read")):
            self._ensure_public_read_policy(final_bucket)
        return final_bucket

    def _ensure_public_read_policy(self, bucket_name: str) -> None:
        """确保桶拥有公开读取对象的策略。

        这里只开放 GetObject，方便浏览器直接访问 public_url；
        不开放写入权限，上传和删除仍然必须走后端密钥。
        """

        if bucket_name in self._public_read_buckets:
            return

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        try:
            self.client().set_bucket_policy(bucket_name, json.dumps(policy, ensure_ascii=False))
        except S3Error as exc:
            raise RuntimeError(f"设置 MinIO 存储桶公开读取策略失败：{exc}") from exc

        self._public_read_buckets.add(bucket_name)
        logger.info("[MinIO] 存储桶公开读取策略已同步 桶名=%s", bucket_name)

    def upload_file(
            self,
            file_path: str,
            object_name: str | None = None,
            bucket_name: str | None = None,
            content_type: str | None = None,
    ) -> MinioObjectInfo:
        """上传本地文件到 MinIO。"""

        local_path = Path(file_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"待上传文件不存在：{file_path}")

        final_bucket = self.ensure_bucket(bucket_name)
        final_object_name = self.build_object_name(local_path.name, object_name)
        final_content_type = content_type or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"

        try:
            result = self.client().fput_object(
                bucket_name=final_bucket,
                object_name=final_object_name,
                file_path=str(local_path),
                content_type=final_content_type,
            )
        except S3Error as exc:
            raise RuntimeError(f"上传文件到 MinIO 失败：{exc}") from exc

        public_url = self.get_public_url(final_object_name, final_bucket)
        logger.info(
            "[MinIO] 文件上传完成 桶名=%s 对象名=%s 文件=%s 公共地址=%s",
            final_bucket,
            final_object_name,
            local_path.name,
            public_url,
        )
        return MinioObjectInfo(
            bucket_name=final_bucket,
            object_name=final_object_name,
            public_url=public_url,
            etag=getattr(result, "etag", None),
            file_size=local_path.stat().st_size,
            content_type=final_content_type,
        )

    def upload_stream(
            self,
            stream: BinaryIO,
            source_name: str,
            file_size: int,
            object_name: str | None = None,
            bucket_name: str | None = None,
            content_type: str | None = None,
    ) -> MinioObjectInfo:
        """上传文件流到 MinIO。"""

        if file_size < 0:
            raise ValueError("file_size 不能小于 0")

        final_bucket = self.ensure_bucket(bucket_name)
        final_object_name = self.build_object_name(source_name, object_name)
        final_content_type = content_type or mimetypes.guess_type(source_name)[0] or "application/octet-stream"

        try:
            result = self.client().put_object(
                bucket_name=final_bucket,
                object_name=final_object_name,
                data=stream,
                length=file_size,
                content_type=final_content_type,
            )
        except S3Error as exc:
            raise RuntimeError(f"上传文件流到 MinIO 失败：{exc}") from exc

        public_url = self.get_public_url(final_object_name, final_bucket)
        logger.info(
            "[MinIO] 文件流上传完成 桶名=%s 对象名=%s 公共地址=%s",
            final_bucket,
            final_object_name,
            public_url,
        )
        return MinioObjectInfo(
            bucket_name=final_bucket,
            object_name=final_object_name,
            public_url=public_url,
            etag=getattr(result, "etag", None),
            file_size=file_size,
            content_type=final_content_type,
        )

    def delete_object(self, object_name: str, bucket_name: str | None = None) -> bool:
        """从 MinIO 删除对象。"""

        final_bucket = self._bucket_name(bucket_name)
        clean_object_name = self._clean_object_name(object_name)
        try:
            self.client().remove_object(final_bucket, clean_object_name)
        except S3Error as exc:
            raise RuntimeError(f"删除 MinIO 对象失败：{exc}") from exc

        logger.info("[MinIO] 对象删除完成 桶名=%s 对象名=%s", final_bucket, clean_object_name)
        return True

    def copy_object(
            self,
            source_object_name: str,
            target_object_name: str,
            *,
            source_bucket_name: str | None = None,
            target_bucket_name: str | None = None,
    ) -> MinioObjectInfo:
        """在 MinIO 内部复制对象，常用于把预览临时文件转成正式文件。"""

        if CopySource is None:
            raise RuntimeError("缺少 MinIO CopySource 依赖，请检查 minio 包是否安装完整")

        source_bucket = self._bucket_name(source_bucket_name)
        target_bucket = self.ensure_bucket(target_bucket_name)
        clean_source = self._clean_object_name(source_object_name)
        clean_target = self._clean_object_name(target_object_name)

        try:
            result = self.client().copy_object(
                bucket_name=target_bucket,
                object_name=clean_target,
                source=CopySource(source_bucket, clean_source),
            )
            stat = self.client().stat_object(target_bucket, clean_target)
        except S3Error as exc:
            raise RuntimeError(f"复制 MinIO 对象失败：{exc}") from exc

        public_url = self.get_public_url(clean_target, target_bucket)
        logger.info(
            "[MinIO] 对象复制完成 源=%s/%s 目标=%s/%s",
            source_bucket,
            clean_source,
            target_bucket,
            clean_target,
        )
        return MinioObjectInfo(
            bucket_name=target_bucket,
            object_name=clean_target,
            public_url=public_url,
            etag=getattr(result, "etag", None),
            file_size=getattr(stat, "size", None),
            content_type=getattr(stat, "content_type", None),
        )

    def download_file(self, object_name: str, target_path: str, bucket_name: str | None = None) -> str:
        """把 MinIO 对象下载到指定本地临时路径，返回下载后的路径。"""

        final_bucket = self._bucket_name(bucket_name)
        clean_object_name = self._clean_object_name(object_name)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client().fget_object(final_bucket, clean_object_name, str(target))
        except S3Error as exc:
            raise RuntimeError(f"下载 MinIO 对象失败：{exc}") from exc

        logger.info("[MinIO] 对象下载完成 桶名=%s 对象名=%s 本地路径=%s", final_bucket, clean_object_name, target)
        return str(target)

    def list_objects(self, prefix: str, bucket_name: str | None = None) -> list[MinioObjectSummary]:
        """按前缀列出 MinIO 对象摘要。"""

        final_bucket = self.ensure_bucket(bucket_name)
        clean_prefix = prefix.strip().lstrip("/")
        try:
            objects = self.client().list_objects(final_bucket, prefix=clean_prefix, recursive=True)
            return [
                MinioObjectSummary(
                    bucket_name=final_bucket,
                    object_name=str(item.object_name),
                    last_modified=getattr(item, "last_modified", None),
                    size=getattr(item, "size", None),
                )
                for item in objects
            ]
        except S3Error as exc:
            raise RuntimeError(f"列出 MinIO 对象失败：{exc}") from exc

    def object_exists(self, object_name: str, bucket_name: str | None = None) -> bool:
        """判断对象是否存在。"""

        final_bucket = self._bucket_name(bucket_name)
        clean_object_name = self._clean_object_name(object_name)
        try:
            if not bool(self.config.get("enabled")):
                return False
            self.client().stat_object(final_bucket, clean_object_name)
            return True
        except S3Error:
            return False

    def get_public_url(self, object_name: str, bucket_name: str | None = None) -> str:
        """拼接对象的公共访问地址。"""

        final_bucket = self._bucket_name(bucket_name)
        clean_object_name = self._clean_object_name(object_name)
        base_url = str(self.config.get("public_base_url") or "").strip()
        if not base_url:
            scheme = "https" if bool(self.config.get("secure")) else "http"
            base_url = f"{scheme}://{str(self.config.get('endpoint') or '127.0.0.1:9000').strip()}"

        safe_object_name = "/".join(quote(part) for part in clean_object_name.split("/"))
        return f"{base_url.rstrip('/')}/{final_bucket}/{safe_object_name}"

    @staticmethod
    def build_object_name(source_name: str, object_name: str | None = None) -> str:
        """生成默认对象名。"""

        if object_name and object_name.strip():
            return object_name.strip().lstrip("/")

        clean_source_name = Path(source_name).name
        suffix = Path(clean_source_name).suffix
        return f"files/{uuid.uuid4().hex}{suffix}"

    def _bucket_name(self, bucket_name: str | None = None) -> str:
        """返回最终使用的桶名。"""

        final_bucket = (bucket_name or self.config.get("bucket_name") or "pub").strip()
        if not final_bucket:
            raise ValueError("MinIO 桶名不能为空")
        return final_bucket

    @staticmethod
    def _clean_object_name(object_name: str) -> str:
        """清理对象名，避免路径前导斜杠影响 URL 和 SDK 调用。"""

        clean_name = object_name.strip().lstrip("/")
        if not clean_name:
            raise ValueError("MinIO 对象名不能为空")
        return clean_name


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
