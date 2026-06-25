"""统一文件存储服务。

本模块把业务文件持久化统一收敛到 MinIO。
本地磁盘只作为解析 PDF、DOCX、TXT 时的临时下载缓存，不再保存长期业务文件。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from fastapi import HTTPException, UploadFile

from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger
from utils.minio_client import MinioObjectInfo, get_minio_client


@dataclass(frozen=True)
class StoredFileInfo:
    """业务文件在 MinIO 中的存储信息。"""

    filename: str  # 原始文件名
    file_type: str  # 文件扩展类型
    file_md5: str  # 文件内容 MD5
    file_size: int  # 文件大小，单位字节
    bucket_name: str  # MinIO 桶名
    object_name: str  # MinIO 对象路径
    public_url: str  # 公共访问地址
    file_path: str  # 存储 URI，格式为 minio://bucket/object_name


class FileStorageService:
    """MinIO 文件存储外观服务。

    业务层只依赖这个服务，不直接依赖 MinIO SDK。
    这样以后即使对象路径规则或临时下载目录变化，也不会污染上传、预览、重建等业务流程。
    """

    def __init__(self):
        """初始化文件存储服务，并复用进程级 MinIO 客户端。"""

        self.client = get_minio_client()

    @staticmethod
    def build_storage_uri(bucket_name: str, object_name: str) -> str:
        """生成统一的 MinIO 存储 URI。"""

        clean_bucket = bucket_name.strip()
        clean_object = object_name.strip().lstrip("/")
        return f"minio://{clean_bucket}/{clean_object}"

    @staticmethod
    def _object_name(prefix: str, owner_id: str, filename: str) -> str:
        """按业务前缀生成对象路径。"""

        safe_prefix = prefix.strip().strip("/")
        safe_owner_id = owner_id.strip().strip("/")
        safe_filename = Path(filename).name
        return f"{safe_prefix}/{safe_owner_id}/{safe_filename}"

    @staticmethod
    def _write_upload_to_temp(file: UploadFile, filename: str) -> tuple[str, int, str]:
        """把上传流写入临时文件，用于计算 MD5 和调用 MinIO 上传。"""

        suffix = Path(filename).suffix
        temp_handle = tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, delete=False)
        temp_path = temp_handle.name
        file_size = 0
        try:
            with temp_handle:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    temp_handle.write(chunk)
        except OSError:
            Path(temp_path).unlink(missing_ok=True)
            raise

        if file_size <= 0:
            Path(temp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="上传文件不能为空")

        file_md5 = get_file_md5_hex(temp_path)
        if not file_md5:
            Path(temp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="上传文件 MD5 计算失败")
        return temp_path, file_size, file_md5

    def save_upload_file(self, *, file: UploadFile, filename: str, prefix: str, owner_id: str) -> StoredFileInfo:
        """把 FastAPI 上传文件流保存到 MinIO。"""

        temp_path, file_size, file_md5 = self._write_upload_to_temp(file, filename)
        object_name = self._object_name(prefix, owner_id, filename)
        try:
            uploaded = self.client.upload_file(temp_path, object_name=object_name)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return self._to_stored_file_info(
            filename=filename,
            file_md5=file_md5,
            file_size=file_size,
            uploaded=uploaded,
        )

    def copy_object(self, *, source: StoredFileInfo, prefix: str, owner_id: str) -> StoredFileInfo:
        """把一个 MinIO 对象复制到新的业务位置。"""

        target_object_name = self._object_name(prefix, owner_id, source.filename)
        copied = self.client.copy_object(
            source_object_name=source.object_name,
            target_object_name=target_object_name,
            source_bucket_name=source.bucket_name,
        )
        return self._to_stored_file_info(
            filename=source.filename,
            file_md5=source.file_md5,
            file_size=int(copied.file_size or source.file_size),
            uploaded=copied,
        )

    def delete_object(self, *, object_name: str | None, bucket_name: str | None = None) -> bool:
        """删除 MinIO 对象；对象名为空时直接跳过。"""

        clean_object_name = str(object_name or "").strip()
        if not clean_object_name:
            logger.warning("[文件存储] 跳过空对象名删除")
            return False
        return self.client.delete_object(clean_object_name, bucket_name=bucket_name)

    @contextmanager
    def downloaded_temp_file(self, *, bucket_name: str | None, object_name: str, filename: str) -> Iterator[str]:
        """下载 MinIO 对象到临时文件，并在使用后清理临时目录。"""

        temp_dir = tempfile.mkdtemp(prefix="minio_read_")
        temp_path = os.path.join(temp_dir, Path(filename).name)
        try:
            yield self.client.download_file(object_name, temp_path, bucket_name=bucket_name)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _to_stored_file_info(
            *,
            filename: str,
            file_md5: str,
            file_size: int,
            uploaded: MinioObjectInfo,
    ) -> StoredFileInfo:
        """把 MinIO 上传结果转换成业务存储信息。"""

        file_type = Path(filename).suffix.lower().lstrip(".")
        return StoredFileInfo(
            filename=filename,
            file_type=file_type,
            file_md5=file_md5,
            file_size=file_size,
            bucket_name=uploaded.bucket_name,
            object_name=uploaded.object_name,
            public_url=uploaded.public_url,
            file_path=FileStorageService.build_storage_uri(uploaded.bucket_name, uploaded.object_name),
        )


def get_file_storage_service() -> FileStorageService:
    """创建文件存储服务实例。"""

    return FileStorageService()
