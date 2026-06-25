"""API 服务编排层。

这个包集中放置路由层可复用的聊天、上传、索引和通用业务编排逻辑。
"""

from api.services.chat_services import (
    _build_conversation_title,
    _elapsed_ms,
    _get_agent,
    _get_chat_route_mode,
    _get_knowledge_answer_service,
    _prepare_chat_conversation,
    _save_chat_exchange,
    _should_use_direct_rag,
    _stream_agent,
    _stream_direct_rag,
)
from api.services.common_services import (
    DictionaryCodeSnapshot,
    _document_to_response,
    _format_response_time,
    _get_knowledge_store,
    _normalize_document_structure_type,
    _normalize_split_strategy,
    build_document_dictionary_snapshot,
)
from api.services.indexing_services import (
    _data_manifest_entry,
    _index_document,
    _load_data_manifest,
    _sync_data_files_to_documents,
)
from api.services.upload_services import (
    _analyze_structure_text,
    _build_structure_sample,
    _delete_preview_file,
    _dictionary_options_text,
    _get_preview_file,
    _get_recommendation_model_mode,
    _normalize_recommendation,
    _parse_model_json,
    _recommend_upload_split_strategy,
    _sanitize_upload_filename,
    _save_preview_file,
    _slice_text_window,
    _validate_file_type,
    get_preview_upload_store,
    load_upload_preview_config,
)

__all__ = [
    "DictionaryCodeSnapshot",
    "_analyze_structure_text",
    "_build_conversation_title",
    "_build_structure_sample",
    "_data_manifest_entry",
    "_delete_preview_file",
    "_dictionary_options_text",
    "_document_to_response",
    "_elapsed_ms",
    "_format_response_time",
    "_get_agent",
    "_get_chat_route_mode",
    "_get_knowledge_answer_service",
    "_get_knowledge_store",
    "_get_preview_file",
    "_get_recommendation_model_mode",
    "_index_document",
    "_load_data_manifest",
    "_normalize_document_structure_type",
    "_normalize_recommendation",
    "_normalize_split_strategy",
    "_parse_model_json",
    "_prepare_chat_conversation",
    "_recommend_upload_split_strategy",
    "_sanitize_upload_filename",
    "_save_chat_exchange",
    "_save_preview_file",
    "_should_use_direct_rag",
    "_slice_text_window",
    "_stream_agent",
    "_stream_direct_rag",
    "_sync_data_files_to_documents",
    "_validate_file_type",
    "build_document_dictionary_snapshot",
    "get_preview_upload_store",
    "load_upload_preview_config",
]

