"""旧服务入口兼容层。

真实业务逻辑已经拆到 api.common_services、api.upload_services、
api.indexing_services 和 api.chat_services。
保留这个文件是为了兼容历史 import，后续新增代码应直接引用对应模块。
"""

from api.chat_services import (
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
from api.common_services import (
    _document_to_response,
    _get_knowledge_store,
    _normalize_document_structure_type,
    _normalize_split_strategy,
)
from api.indexing_services import (
    _data_manifest_entry,
    _index_document,
    _load_data_manifest,
    _sync_data_files_to_documents,
)
from api.upload_services import (
    _analyze_structure_text,
    _build_structure_sample,
    _dictionary_options_text,
    _get_preview_file,
    _get_recommendation_model_mode,
    _move_upload_file,
    _normalize_recommendation,
    _parse_model_json,
    _recommend_upload_split_strategy,
    _remove_created_upload_dir,
    _sanitize_upload_filename,
    _save_preview_file,
    _slice_text_window,
    _validate_file_type,
)
