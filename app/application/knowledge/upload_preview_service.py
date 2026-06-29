"""知识库上传预览服务。

这个模块负责“确认入库前”的临时流程：
1. 校验上传文件名和类型；
2. 把文件保存到 MinIO previews 前缀；
3. 把临时对象元数据写入 Redis；
4. 抽样文件内容，让模型推荐文档类型和切分策略；
5. 用户确认后把临时对象复制为正式 documents 对象。

注意：这里不直接写 documents 表，也不写 Qdrant；正式入库由 knowledge_service + indexing_service 完成。
"""

import json
import os
import re

from fastapi import HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from app.application.knowledge.upload_preview_state import (
    PREVIEW_OBJECT_PREFIX,
    get_preview_upload_store,
    load_upload_preview_config,
)
from app.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app.shared.document_response import normalize_document_structure_type, normalize_split_strategy
from app.infrastructure.file_storage_service import StoredFileInfo, get_file_storage_service
from core.model.factory import get_chat_model, get_chat_model_name_for_mode
from core.utils.config_handler import qdrant_conf
from core.utils.logger_handler import logger
from core.utils.prompt_manager import prompt_manager


def _sanitize_upload_filename(filename: str | None) -> str:
    """清理上传文件名，避免用户传入带目录的路径。"""

    raw_filename = (filename or "").strip()
    safe_filename = raw_filename.replace("\\", "/").split("/")[-1].strip()
    safe_filename = safe_filename.replace("\x00", "")

    if not safe_filename:
        raise HTTPException(status_code=400, detail="上传文件名不能为空")

    return safe_filename


def _validate_file_type(filename: str) -> str:
    """校验上传文件后缀是否在配置允许范围内。"""

    file_type = os.path.splitext(filename)[1].lower().lstrip(".")
    allowed_types = {item.lower().lstrip(".") for item in qdrant_conf["allow_knowledge_file_type"]}

    if file_type not in allowed_types:
        allowed_text = ", ".join(sorted(allowed_types))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{file_type}，仅支持：{allowed_text}")

    return file_type


def _save_preview_file(file: UploadFile, filename: str, upload_id: str) -> StoredFileInfo:
    """把上传文件保存到 MinIO 预览对象。"""

    storage_service = get_file_storage_service()
    logger.info("[知识库] 上传预览文件保存开始 上传编号=%s 文件名=%s", upload_id, filename)
    stored_file = storage_service.save_upload_file(
        file=file,
        filename=filename,
        prefix="previews",
        owner_id=upload_id,
    )
    preview_config = load_upload_preview_config()
    if stored_file.file_size > preview_config.max_file_size_bytes:
        storage_service.delete_object(
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
        )
        raise HTTPException(
            status_code=400,
            detail=f"文件过大，最大支持 {preview_config.max_file_size_bytes} 字节",
        )

    try:
        get_preview_upload_store().save(upload_id, stored_file)
    except Exception as exc:
        storage_service.delete_object(
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
        )
        logger.error("[知识库] Redis 预览上传元数据写入失败 上传编号=%s 错误=%s", upload_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="预览上传状态保存失败，请稍后重试") from exc
    logger.info(
        "[知识库] 上传预览文件保存完成 上传编号=%s 桶名=%s 对象名=%s 文件大小=%s MD5=%s",
        upload_id,
        stored_file.bucket_name,
        stored_file.object_name,
        stored_file.file_size,
        stored_file.file_md5,
    )
    return stored_file


def _get_preview_file(upload_id: str) -> StoredFileInfo:
    """根据上传编号找到 MinIO 临时预览对象。"""

    clean_upload_id = upload_id.strip()
    metadata = get_preview_upload_store().get(clean_upload_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail=f"临时上传已过期或不存在：{clean_upload_id}")

    stored_file = metadata.to_stored_file_info()
    if not stored_file.object_name.startswith(PREVIEW_OBJECT_PREFIX):
        raise HTTPException(status_code=400, detail="临时上传对象路径非法")
    logger.info("[知识库] 读取预览上传对象 上传编号=%s 对象名=%s", clean_upload_id, stored_file.object_name)
    return stored_file



def _delete_preview_file(upload_id: str) -> None:
    """删除 MinIO 临时预览对象。"""

    clean_upload_id = upload_id.strip()
    store = get_preview_upload_store()
    metadata = store.get(clean_upload_id)
    store.delete(clean_upload_id)
    if metadata is None:
        logger.info("[知识库] 删除预览上传对象跳过 原因=元数据不存在 上传编号=%s", clean_upload_id)
        return

    stored_file = metadata.to_stored_file_info()
    if not stored_file.object_name.startswith(PREVIEW_OBJECT_PREFIX):
        logger.warning("[知识库] 跳过非法 MinIO 临时对象删除 上传编号=%s 对象名=%s", upload_id, stored_file.object_name)
        return
    try:
        get_file_storage_service().delete_object(
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
        )
        logger.info("[知识库] 删除预览上传对象完成 上传编号=%s 对象名=%s", upload_id, stored_file.object_name)
    except Exception as exc:
        logger.warning("[知识库] 删除 MinIO 临时预览对象失败 上传编号=%s 错误=%s", upload_id, exc)
    return



def _promote_preview_file(upload_id: str, document_id: str) -> StoredFileInfo:
    """把 MinIO 预览对象复制为正式知识库对象，并删除临时对象。"""

    stored_file = _get_preview_file(upload_id)
    logger.info(
        "[知识库] 预览对象转正式对象开始 上传编号=%s 文档编号=%s 源对象=%s",
        upload_id,
        document_id,
        stored_file.object_name,
    )
    final_file = get_file_storage_service().copy_object(
        source=stored_file,
        prefix="documents",
        owner_id=document_id,
    )
    _delete_preview_file(upload_id)
    logger.info(
        "[知识库] 预览对象转正式对象完成 上传编号=%s 文档编号=%s 正式对象=%s",
        upload_id,
        document_id,
        final_file.object_name,
    )
    return final_file


def _slice_text_window(text: str, start: int, length: int) -> str:
    """从文本中截取一个窗口，并清理首尾空白。"""

    if not text or length <= 0:
        return ""
    safe_start = max(0, min(start, len(text)))
    return text[safe_start:safe_start + length].strip()


def _build_structure_sample(full_text: str, *, max_chars: int = 10000) -> str:
    """从全文中抽取开头、中间和结尾样本，供模型判断切分方式。"""

    clean_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()
    if len(clean_text) <= max_chars:
        return clean_text

    head_chars = min(3800, max_chars // 3 + 500)
    middle_chars = min(3600, max_chars // 3 + 300)
    tail_chars = max(1200, max_chars - head_chars - middle_chars)
    middle_start = max(0, len(clean_text) // 2 - middle_chars // 2)
    tail_start = max(0, len(clean_text) - tail_chars)

    return "\n\n".join(
        part
        for part in [
            "【开头样本】\n" + _slice_text_window(clean_text, 0, head_chars),
            "【中间样本】\n" + _slice_text_window(clean_text, middle_start, middle_chars),
            "【结尾样本】\n" + _slice_text_window(clean_text, tail_start, tail_chars),
        ]
        if part.strip()
    )[:max_chars]


def _analyze_structure_text(text: str) -> dict:
    """统计样本文本中的结构特征，辅助模型判断文档切分策略。"""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered_lines = [line for line in lines if re.match(r"^\d+[.、)]\s*", line)]
    qa_lines = [line for line in lines if re.match(r"^(Q|A|问|答)[:：]", line, re.IGNORECASE)]
    heading_lines = [
        line
        for line in lines
        if re.match(r"^(#{1,6}\s+|第[一二三四五六七八九十\d]+[章节]|[一二三四五六七八九十]+[、.])", line)
    ]
    return {
        "line_count": len(lines),
        "numbered_line_count": len(numbered_lines),
        "qa_marker_count": len(qa_lines),
        "heading_line_count": len(heading_lines),
    }


def _parse_model_json(content: object) -> dict:
    """从模型返回内容中解析 JSON 对象。"""

    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match:
            text = object_match.group(0)

    return json.loads(text)


def _normalize_recommendation(value: dict) -> dict:
    """根据字典表校验并归一化模型推荐结果。"""

    dictionary_repository = DictionaryRepository()
    document_type_codes = set(dictionary_repository.list_enabled_codes("document_structure"))
    split_strategy_codes = set(dictionary_repository.list_enabled_codes("split_strategy"))
    raw_split_strategy = str(value.get("split_strategy") or "").strip().lower()
    if raw_split_strategy in split_strategy_codes:
        split_strategy = raw_split_strategy
    else:
        split_strategy = normalize_split_strategy("recursive")
    raw_document_type = str(value.get("document_type") or "").strip().lower()
    if split_strategy == "llm_semantic":
        document_type = normalize_document_structure_type("text", split_strategy)
    elif raw_document_type in document_type_codes:
        document_type = normalize_document_structure_type(raw_document_type, split_strategy)
    else:
        document_type = normalize_document_structure_type(None, split_strategy)

    try:
        confidence = float(value.get("confidence") or 0.6)
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(confidence, 1.0))

    raw_reasons = value.get("reasons") or []
    if isinstance(raw_reasons, str):
        reasons = [raw_reasons]
    elif isinstance(raw_reasons, list):
        reasons = [str(reason).strip() for reason in raw_reasons if str(reason).strip()]
    else:
        reasons = []
    if raw_split_strategy and raw_split_strategy not in split_strategy_codes:
        reasons.insert(0, "模型返回了非支持切分策略，已回退到递归切分")
    if raw_document_type not in document_type_codes:
        reasons.insert(0, "模型返回了非支持结构类型，已按切分策略回退到合法文档类型")

    return {
        "document_type": document_type,
        "split_strategy": split_strategy,
        "confidence": confidence,
        "reasons": reasons[:5] or ["模型根据文档结构样本给出推荐"],
    }
def _outline_from_documents(documents: list) -> list[dict]:
    """从文件 Document 元数据中读取 PDF 书签目录。"""

    for document in documents:
        outline = document.metadata.get("_pdf_outline")
        if isinstance(outline, list):
            return outline
    return []


def _deterministic_recommendation(filename: str, sample_text: str, documents: list) -> dict | None:
    """先用可解释规则推荐结构类型，避免模型把明显 QA 文件误判成普通文本。"""

    from app.infrastructure.vector_store_service import VectorStoreService

    parser = VectorStoreService().document_parser
    detection = parser.detect_document_type(filename, sample_text, outline=_outline_from_documents(documents))
    if detection.split_strategy == "recursive" and detection.document_type == "text":
        return None
    logger.info(
        "[知识库] 规则推荐切分方式命中 文件名=%s 文档类型=%s 切分策略=%s 置信度=%s 原因=%s",
        filename,
        detection.document_type,
        detection.split_strategy,
        detection.confidence,
        detection.reasons,
    )
    return {
        "document_type": detection.document_type,
        "split_strategy": detection.split_strategy,
        "confidence": detection.confidence,
        "reasons": detection.reasons,
        "sample_chars": len(sample_text),
        "model_name": "",
    }


def _dictionary_options_text(dictionary_code: str) -> str:
    """把启用字典项拼成给模型看的枚举说明，避免 prompt 里散落固定可选值。"""

    rows = DictionaryRepository().list_items(dictionary_code=dictionary_code)
    options = [
        f"{row['item_code']}（{row['item_name']}）"
        for row in rows
        if int(row.get("enabled") or 0) == 1
    ]
    return "、".join(options)


def _get_recommendation_model_mode() -> str:
    """从模型档位字典中读取用于切分推荐的小模型档位。"""

    dictionary_repository = DictionaryRepository()
    return (
        dictionary_repository.get_code_by_metadata("model_mode", "recommendation", True)
        or dictionary_repository.normalize_code("model_mode", None)
    )


def _recommend_upload_split_strategy(upload_id: str) -> dict:
    """读取临时上传文件样本，并调用低延迟模型推荐文档类型和切分策略。"""

    stored_file = _get_preview_file(upload_id)
    filename = stored_file.filename
    file_type = _validate_file_type(filename)
    from app.infrastructure.vector_store_service import VectorStoreService

    with get_file_storage_service().downloaded_temp_file(
            bucket_name=stored_file.bucket_name,
            object_name=stored_file.object_name,
            filename=stored_file.filename,
    ) as file_path:
        documents = VectorStoreService().get_file_documents(file_path)
    full_text = "\n\n".join(document.page_content for document in documents)
    preview_config = load_upload_preview_config()
    sample_text = _build_structure_sample(full_text, max_chars=preview_config.recommendation_sample_chars)
    if not sample_text:
        raise HTTPException(status_code=400, detail="文件没有可用于模型推荐的文本内容")

    deterministic_recommendation = _deterministic_recommendation(filename, sample_text, documents)
    if deterministic_recommendation is not None:
        return deterministic_recommendation

    structure = _analyze_structure_text(sample_text)
    selected_model_mode = _get_recommendation_model_mode()
    selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
    document_type_options = _dictionary_options_text("document_structure")
    split_strategy_options = _dictionary_options_text("split_strategy")
    logger.info(
        "[知识库] 模型推荐切分方式开始 上传编号=%s 文件名=%s 模型名称=%s 样本字符数=%s 结构统计=%s",
        upload_id,
        filename,
        selected_model_name,
        len(sample_text),
        structure,
    )

    model = get_chat_model(selected_model_mode)
    response = model.invoke(
        [
            SystemMessage(content=prompt_manager.get("knowledge_upload.split_recommendation.system")),
            HumanMessage(
                content=prompt_manager.render(
                    "knowledge_upload.split_recommendation.user",
                    document_type_options=document_type_options,
                    split_strategy_options=split_strategy_options,
                    filename=filename,
                    file_type=file_type,
                    structure_json=json.dumps(structure, ensure_ascii=False),
                    sample_text=sample_text,
                )
            ),
        ]
    )
    recommendation = _normalize_recommendation(_parse_model_json(response.content))
    logger.info(
        "[知识库] 模型推荐切分方式完成 上传编号=%s 文件名=%s 推荐=%s",
        upload_id,
        filename,
        recommendation,
    )
    return {
        **recommendation,
        "sample_chars": len(sample_text),
        "model_name": selected_model_name,
    }


def _recommend_upload_split_strategy_or_fallback(upload_id: str) -> dict:
    """上传预览阶段尽力调用模型推荐，失败时回退到普通递归切分。"""

    try:
        return _recommend_upload_split_strategy(upload_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[知识库] 模型推荐切分方式失败，预览回退递归切分 上传编号=%s 错误=%s", upload_id, exc)
        return {
            "document_type": "text",
            "split_strategy": "recursive",
            "confidence": 0.5,
            "reasons": ["模型推荐失败，已回退到普通文本递归切分"],
            "sample_chars": 0,
            "model_name": "",
        }
