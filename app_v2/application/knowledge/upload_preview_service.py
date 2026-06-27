import json
import os
import re

from fastapi import HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from app_v2.application.knowledge.upload_preview_state import (
    PREVIEW_OBJECT_PREFIX,
    get_preview_upload_store,
    load_upload_preview_config,
)
from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app_v2.shared.document_response import normalize_document_structure_type, normalize_split_strategy
from infrastructure.file_storage_service import StoredFileInfo, get_file_storage_service
from model.factory import get_chat_model, get_chat_model_name_for_mode
from utils.config_handler import qdrant_conf
from utils.logger_handler import logger


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
        "[知识库] 上传预览文件已保存到 MinIO 上传编号=%s 对象名=%s",
        upload_id,
        stored_file.object_name,
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
    return stored_file



def _delete_preview_file(upload_id: str) -> None:
    """删除 MinIO 临时预览对象。"""

    clean_upload_id = upload_id.strip()
    store = get_preview_upload_store()
    metadata = store.get(clean_upload_id)
    store.delete(clean_upload_id)
    if metadata is None:
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
    except Exception as exc:
        logger.warning("[知识库] 删除 MinIO 临时预览对象失败 上传编号=%s 错误=%s", upload_id, exc)
    return



def _promote_preview_file(upload_id: str, document_id: str) -> StoredFileInfo:
    """把 MinIO 预览对象复制为正式知识库对象，并删除临时对象。"""

    stored_file = _get_preview_file(upload_id)
    final_file = get_file_storage_service().copy_object(
        source=stored_file,
        prefix="documents",
        owner_id=document_id,
    )
    _delete_preview_file(upload_id)
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
    split_strategy = normalize_split_strategy(value.get("split_strategy"))
    raw_document_type = str(value.get("document_type") or "").strip().lower()
    if raw_document_type in document_type_codes:
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
    if raw_document_type not in document_type_codes:
        reasons.insert(0, "模型返回了非支持结构类型，已按切分策略回退到合法文档类型")

    return {
        "document_type": document_type,
        "split_strategy": split_strategy,
        "confidence": confidence,
        "reasons": reasons[:5] or ["模型根据文档结构样本给出推荐"],
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
    from infrastructure.vector_store_service import VectorStoreService

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
            SystemMessage(
                content=(
                    "你是知识库文档切分策略推荐器。只根据文档结构推荐，不总结正文。"
                    "必须只返回 JSON，不要返回 Markdown。"
                )
            ),
            HumanMessage(
                content=(
                    "请从以下枚举中选择：\n"
                    f"document_type: {document_type_options}\n"
                    f"split_strategy: {split_strategy_options}\n\n"
                    "判断原则：\n"
                    "- 编号问答型：document_type=qa，split_strategy=numbered_qa。\n"
                    "- PDF目录问答型：document_type=qa，split_strategy=outline_qa。"
                    "只有目录或样本清楚呈现“章节 -> 问题”时才使用，不要把普通目录 PDF 误判成这种策略。\n"
                    "- 编号条目型：document_type=numbered，split_strategy=numbered_segments。\n"
                    "- 普通文本型：document_type=text，split_strategy=recursive。\n\n"
                    f"文件名：{filename}\n文件类型：{file_type}\n结构统计：{json.dumps(structure, ensure_ascii=False)}\n\n"
                    f"文档结构样本：\n{sample_text}\n\n"
                    "返回 JSON 格式："
                    '{"document_type":"text","split_strategy":"recursive","confidence":0.75,"reasons":["原因1","原因2"]}'
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
