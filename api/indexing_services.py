"""兼容旧导入路径的索引服务入口。

真实实现已迁移到 ``api.services.indexing_services``，保留本模块是为了让历史引用平滑过渡。
"""

from api.services.indexing_services import (
    _data_manifest_entry,
    _index_document,
    _load_data_manifest,
    _sync_data_files_to_documents,
)

__all__ = [
    "_data_manifest_entry",
    "_index_document",
    "_load_data_manifest",
    "_sync_data_files_to_documents",
]

