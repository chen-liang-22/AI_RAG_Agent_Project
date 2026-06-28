"""文件存储适配器。

V2 业务层只认识这个适配器，不直接依赖 MinIO SDK 或旧文件存储服务。
"""

from contextlib import AbstractContextManager
from typing import Any

from fastapi import UploadFile

from app.infrastructure.file_storage_service import FileStorageService, StoredFileInfo, get_file_storage_service


class FileStorageAdapter:
    """MinIO 文件存储适配器。

    这里使用适配器模式，把现有 `FileStorageService` 包装成 V2 内部接口。
    """

    def __init__(self, storage_service: FileStorageService | None = None):
        """初始化文件存储适配器。

        默认包装统一的 MinIO 文件服务；测试时可传入假的 storage_service。
        """

        self.storage_service = storage_service or get_file_storage_service()

    def save_upload_file(self, *, file: UploadFile, filename: str, prefix: str, owner_id: str) -> StoredFileInfo:
        """保存上传文件到 MinIO。"""

        return self.storage_service.save_upload_file(file=file, filename=filename, prefix=prefix, owner_id=owner_id)

    def save_local_file(self, *, file_path: str, filename: str | None, prefix: str, owner_id: str) -> StoredFileInfo:
        """保存服务端本地文件到 MinIO。"""

        return self.storage_service.save_local_file(file_path=file_path, filename=filename, prefix=prefix, owner_id=owner_id)

    def copy_object(self, *, source: StoredFileInfo, prefix: str, owner_id: str) -> StoredFileInfo:
        """复制 MinIO 对象到正式业务路径。"""

        return self.storage_service.copy_object(source=source, prefix=prefix, owner_id=owner_id)

    def delete_object(self, *, object_name: str | None, bucket_name: str | None = None) -> bool:
        """删除 MinIO 对象。"""

        return self.storage_service.delete_object(object_name=object_name, bucket_name=bucket_name)

    def downloaded_temp_file(self, *, bucket_name: str | None, object_name: str, filename: str) -> AbstractContextManager[str]:
        """下载 MinIO 对象为临时文件。"""

        return self.storage_service.downloaded_temp_file(bucket_name=bucket_name, object_name=object_name, filename=filename)
