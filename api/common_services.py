"""兼容旧导入路径的 API 通用服务入口。

真实实现已迁移到 ``api.services.common_services``，保留本模块是为了让历史引用平滑过渡。
"""

from api.services.common_services import (
    DictionaryCodeSnapshot,
    _document_to_response,
    _format_response_time,
    _get_knowledge_store,
    _normalize_document_structure_type,
    _normalize_split_strategy,
    build_document_dictionary_snapshot,
)

__all__ = [
    "DictionaryCodeSnapshot",
    "_document_to_response",
    "_format_response_time",
    "_get_knowledge_store",
    "_normalize_document_structure_type",
    "_normalize_split_strategy",
    "build_document_dictionary_snapshot",
]

