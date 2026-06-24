"""兼容旧导入路径的上传服务入口。

真实实现已迁移到 ``api.services.upload_services``，保留本模块是为了让历史引用平滑过渡。
"""

from api.services.upload_services import (
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

__all__ = [
    "_analyze_structure_text",
    "_build_structure_sample",
    "_dictionary_options_text",
    "_get_preview_file",
    "_get_recommendation_model_mode",
    "_move_upload_file",
    "_normalize_recommendation",
    "_parse_model_json",
    "_recommend_upload_split_strategy",
    "_remove_created_upload_dir",
    "_sanitize_upload_filename",
    "_save_preview_file",
    "_slice_text_window",
    "_validate_file_type",
]

