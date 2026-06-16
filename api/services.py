import json
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Iterator

import yaml
from fastapi import HTTPException, UploadFile

from api.schemas import ChatRequest, KnowledgeFileResponse
from langchain_core.messages import HumanMessage, SystemMessage

from model.factory import get_chat_model, get_chat_model_name_for_mode, normalize_chat_model_mode
from rag.knowledge_store import KnowledgeStore
from utils.config_handler import qdrant_conf, rag_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.qdrant_options import get_qdrant_collection_name, normalize_qdrant_collection_name


_agent = None  # 全局 Agent 单例；第一次聊天请求进来时才真正初始化
_agent_lock = threading.Lock()  # 初始化锁；防止多个请求同时初始化多个 Agent
_knowledge_answer_service = None  # 知识库直答服务单例；普通知识问答不再进入 Agent 工具链
_knowledge_answer_lock = threading.Lock()  # 知识直答服务初始化锁

def _get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据仓库。

    KnowledgeStore 内部使用 SQLite。
    每次请求创建一个轻量对象即可，真正的数据库连接只在执行 SQL 时短暂打开。
    """

    return KnowledgeStore()


def _document_to_response(document: dict) -> KnowledgeFileResponse:
    """把 SQLite 文档记录转换成 FastAPI 响应模型。

    SQLite 取出来的数字字段有时可能是字符串或兼容类型，这里统一转成 int，
    这样前端拿到的数据类型更稳定。
    """

    return KnowledgeFileResponse(
        document_id=document["document_id"],
        filename=document["filename"],
        file_path=document["file_path"],
        file_type=document["file_type"],
        file_md5=document["file_md5"],
        file_size=int(document["file_size"]),
        status=document["status"],
        version=int(document["version"]),
        chunk_count=int(document["chunk_count"]),
        collection_name=document.get("collection_name") or get_qdrant_collection_name(),
        document_type=_normalize_document_structure_type(
            document.get("document_type"),
            document.get("split_strategy"),
        ),
        split_strategy=_normalize_split_strategy(document.get("split_strategy")),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        error_message=document.get("error_message"),
    )


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


def _normalize_split_strategy(split_strategy: str | None = None) -> str:
    """从字典表归一化切分策略。"""

    return _get_knowledge_store().normalize_dictionary_code("split_strategy", split_strategy)


def _normalize_document_structure_type(
        document_type: str | None = None,
        split_strategy: str | None = None,
) -> str:
    """从字典表归一化文档结构类型。"""

    store = _get_knowledge_store()
    enabled_codes = set(store.list_enabled_dictionary_codes("document_structure"))
    default_code = store.normalize_dictionary_code("document_structure", None)
    value = str(document_type or "").strip().lower()
    normalized_split_strategy = _normalize_split_strategy(split_strategy)
    if normalized_split_strategy in {"numbered_qa", "outline_qa"} and "qa" in enabled_codes:
        return "qa"
    if normalized_split_strategy == "numbered_segments" and "numbered" in enabled_codes:
        return "numbered"
    if value in enabled_codes:
        return value
    if not value:
        return default_code
    supported_text = "、".join(sorted(enabled_codes))
    raise HTTPException(status_code=400, detail=f"文档结构类型只支持：{supported_text}")


def _remove_created_upload_dir(upload_dir: str) -> None:
    """删除本次上传刚创建的临时目录。"""

    uploads_root = os.path.abspath(get_abs_path("uploads"))
    target_dir = os.path.abspath(upload_dir)

    if os.path.commonpath([uploads_root, target_dir]) != uploads_root:
        logger.warning("[知识库] 跳过不安全的上传目录清理 目录=%s", target_dir)
        return

    shutil.rmtree(target_dir, ignore_errors=True)


def _move_upload_file(source_path: str, document_id: str, filename: str) -> str:
    """把临时上传文件移动到正式 uploads/{document_id}/ 目录。"""

    upload_dir = os.path.join(get_abs_path("uploads"), document_id)
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.join(upload_dir, filename)
    shutil.move(source_path, target_path)
    return target_path


def _save_preview_file(file: UploadFile, filename: str, upload_id: str) -> tuple[str, int]:
    """把上传文件保存到临时 preview 目录。"""

    preview_dir = os.path.join(get_abs_path("uploads"), "_preview", upload_id)
    os.makedirs(preview_dir, exist_ok=True)
    file_path = os.path.join(preview_dir, filename)
    file_size = 0

    try:
        with open(file_path, "wb") as target:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                target.write(chunk)
    except OSError:
        _remove_created_upload_dir(preview_dir)
        raise

    if file_size <= 0:
        _remove_created_upload_dir(preview_dir)
        raise HTTPException(status_code=400, detail="上传文件不能为空")

    return file_path, file_size


def _get_preview_file(upload_id: str) -> tuple[str, str]:
    """根据上传编号找到临时预览文件，并返回预览目录和文件路径。"""

    clean_upload_id = upload_id.strip()
    preview_root = os.path.abspath(os.path.join(get_abs_path("uploads"), "_preview"))
    preview_dir = os.path.abspath(os.path.join(preview_root, clean_upload_id))

    if os.path.commonpath([preview_root, preview_dir]) != preview_root or not os.path.isdir(preview_dir):
        raise HTTPException(status_code=404, detail=f"临时上传不存在：{clean_upload_id}")

    files = [name for name in os.listdir(preview_dir) if os.path.isfile(os.path.join(preview_dir, name))]
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="临时上传文件状态异常")

    filename = _sanitize_upload_filename(files[0])
    return preview_dir, os.path.join(preview_dir, filename)


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

    store = _get_knowledge_store()
    document_type_codes = set(store.list_enabled_dictionary_codes("document_structure"))
    split_strategy = _normalize_split_strategy(value.get("split_strategy"))
    raw_document_type = str(value.get("document_type") or "").strip().lower()
    if raw_document_type in document_type_codes:
        document_type = _normalize_document_structure_type(raw_document_type, split_strategy)
    else:
        document_type = _normalize_document_structure_type(None, split_strategy)

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

    rows = _get_knowledge_store().list_dictionary_items(dictionary_code=dictionary_code)
    options = [
        f"{row['item_code']}（{row['item_name']}）"
        for row in rows
        if int(row.get("enabled") or 0) == 1
    ]
    return "、".join(options)


def _get_recommendation_model_mode() -> str:
    """从模型档位字典中读取用于切分推荐的小模型档位。"""

    # 获取 SQLite 字典表访问对象，用它读取 model_mode 字典配置。
    store = _get_knowledge_store()
    # 优先查找 metadata_json 中标记 recommendation=true 的模型档位。
    # 如果字典里没有配置推荐档位，则回退到 model_mode 字典的默认编码。
    return (
        store.get_dictionary_code_by_metadata("model_mode", "default", True)
        or store.normalize_dictionary_code("model_mode", None)
    )


def _recommend_upload_split_strategy(upload_id: str) -> dict:
    """读取临时上传文件样本，并调用低延迟模型推荐文档类型和切分策略。"""

    _, file_path = _get_preview_file(upload_id)
    filename = os.path.basename(file_path)
    file_type = _validate_file_type(filename)
    from rag.vector_store import VectorStoreService

    documents = VectorStoreService().get_file_documents(file_path)
    full_text = "\n\n".join(document.page_content for document in documents)
    sample_text = _build_structure_sample(full_text, max_chars=10000)
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


def _index_document(
        store: KnowledgeStore,
        document: dict,
        *,
        increment_version: bool = False,
        vector_store=None,
        document_type: str | None = None,
        split_strategy: str | None = None,
        collection_name: str | None = None,
) -> dict:
    """把 documents 表中的文件解析、分片、向量化并写入 Qdrant。

    这个函数把“修改 SQLite 状态”和“写 Qdrant”封装在一起，上传和重建索引都复用它。

    流程：
    1. documents.status 改成 indexing。
    2. VectorStoreService 读取原始文件并切分成知识单元。
    3. 按 document_id 删除旧 Qdrant points。
    4. 写入新的 Qdrant points。
    5. documents.status 改成 indexed。
    """

    document_id = document["document_id"]
    final_collection_name = normalize_qdrant_collection_name(collection_name or document.get("collection_name"))
    final_split_strategy = _normalize_split_strategy(split_strategy or document.get("split_strategy"))
    final_document_type = _normalize_document_structure_type(
        document_type or document.get("document_type"),
        final_split_strategy,
    )
    store.update_document_status(
        document_id,
        "indexing",
        error_message=None,
        increment_version=increment_version,
        collection_name=final_collection_name,
        document_type=final_document_type,
        split_strategy=final_split_strategy,
    )

    indexing_document = store.get_document(document_id)
    if indexing_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        if vector_store is None:
            from rag.vector_store import VectorStoreService  # 延迟导入，只有真正入库时才加载向量库和 embedding 模型

            vector_store = VectorStoreService(collection_name=final_collection_name)

        chunk_count = vector_store.index_file(
            indexing_document,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
        store.update_document_status(
            document_id,
            "indexed",
            chunk_count=chunk_count,
            error_message=None,
            collection_name=final_collection_name,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[知识库] 文件入库失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
        store.update_document_status(
            document_id,
            "failed",
            error_message=str(exc),
            collection_name=final_collection_name,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
        raise HTTPException(status_code=500, detail=f"知识库入库失败：{exc}") from exc

    indexed_document = store.get_document(document_id)
    if indexed_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return indexed_document


def _load_data_manifest() -> dict:
    manifest_path = get_abs_path(os.path.join(qdrant_conf["data_path"], "knowledge_manifest.yml"))
    if not os.path.exists(manifest_path):
        return {"defaults": {}, "files": {}}

    with open(manifest_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return {
        "defaults": data.get("defaults") or {},
        "files": data.get("files") or {},
    }


def _data_manifest_entry(filename: str, manifest: dict) -> dict:
    defaults = manifest.get("defaults") or {}
    files = manifest.get("files") or {}
    entry = files.get(filename) or {}
    return {
        "collection_name": normalize_qdrant_collection_name(entry.get("collection_name") or defaults.get("collection_name")),
        "document_type": _normalize_document_structure_type(
            entry.get("document_type") or defaults.get("document_type"),
            entry.get("split_strategy") or defaults.get("split_strategy"),
        ),
        "split_strategy": _normalize_split_strategy(entry.get("split_strategy") or defaults.get("split_strategy")),
    }


def _sync_data_files_to_documents(store: KnowledgeStore) -> list[dict]:
    """把 data/ 目录中的内置知识文件同步到 documents 表。

    用户手动上传的文件保存在 uploads/ 目录，项目自带的示例或基础知识文件通常放在 data/ 目录。
    reload 时先把 data/ 文件注册成 documents 记录，再统一调用 _index_document 入库。

    同一份内容按 MD5 去重：如果 active documents 中已有相同 MD5，就复用已有记录。
    """

    data_path = get_abs_path(qdrant_conf["data_path"])
    allowed_types = tuple(qdrant_conf["allow_knowledge_file_type"])
    file_paths = listdir_with_allowed_type(data_path, allowed_types)
    manifest = _load_data_manifest()
    documents: list[dict] = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        manifest_entry = _data_manifest_entry(filename, manifest)
        file_type = os.path.splitext(filename)[1].lower().lstrip(".")
        file_md5 = get_file_md5_hex(file_path)

        if not file_md5:
            logger.warning("[知识库] 跳过无法计算 MD5 的内置知识文件 路径=%s", file_path)
            continue

        existing_document = store.find_active_document_by_md5(
            file_md5,
            collection_name=manifest_entry["collection_name"],
        )
        if existing_document is not None:
            store.update_document_status(
                existing_document["document_id"],
                existing_document["status"],
                collection_name=manifest_entry["collection_name"],
                document_type=manifest_entry["document_type"],
                split_strategy=manifest_entry["split_strategy"],
            )
            existing_document = store.get_document(existing_document["document_id"]) or existing_document
            documents.append(existing_document)
            continue

        document_id = f"doc_{uuid.uuid4().hex}"
        documents.append(
            store.create_document(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                file_md5=file_md5,
                file_size=os.path.getsize(file_path),
                status="uploaded",
                collection_name=manifest_entry["collection_name"],
                document_type=manifest_entry["document_type"],
                split_strategy=manifest_entry["split_strategy"],
            )
        )

    return documents


def _get_agent():
    """获取全局 Agent 单例。"""

    global _agent

    if _agent is not None:
        return _agent

    with _agent_lock:
        if _agent is None:
            try:
                from agent.react_agent import ReactAgent  # 延迟导入，避免应用启动时就加载模型和向量库

                _agent = ReactAgent()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Agent 初始化失败：{exc}") from exc

    return _agent


def _get_knowledge_answer_service():
    """获取知识库直答服务单例。"""

    global _knowledge_answer_service

    if _knowledge_answer_service is not None:
        return _knowledge_answer_service

    with _knowledge_answer_lock:
        if _knowledge_answer_service is None:
            try:
                from rag.knowledge_answer_service import KnowledgeAnswerService

                _knowledge_answer_service = KnowledgeAnswerService()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"知识库直答服务初始化失败：{exc}") from exc

    return _knowledge_answer_service
def _get_chat_route_mode() -> str:
    """读取聊天路由模式；不配置时默认直连 RAG。"""

    mode = str(
        rag_conf.get("chat_route_mode")
        or rag_conf.get("knowledge_chat_mode")
        or "direct_rag"
    ).strip().lower()
    return mode.replace("-", "_")


def _should_use_direct_rag(message: str) -> tuple[bool, str]:
    """判断当前问题是否走知识库直答。"""

    mode = _get_chat_route_mode()
    if mode in {"agent", "react_agent", "legacy_agent"}:
        return False, "配置 chat_route_mode=agent，使用 Agent 工具链"

    return True, f"配置 chat_route_mode={mode}，使用直连 RAG"


def _build_conversation_title(message: str) -> str:
    """用首条用户问题生成一个简短会话标题。"""

    return message.strip().replace("\n", " ")[:40] or "新会话"


def _prepare_chat_conversation(request: ChatRequest) -> tuple[str, list[dict]]:
    """创建或读取会话，并返回 conversation_id 和最近历史。"""

    store = _get_knowledge_store()
    conversation = store.ensure_conversation(
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        title=_build_conversation_title(request.message),
    )
    conversation_id = conversation["conversation_id"]
    history = store.list_recent_messages(conversation_id, limit=20)
    return conversation_id, history


def _save_chat_exchange(
        *,
        conversation_id: str,
        message: str,
        answer: str,
        model_name: str | None = None,
        metadata: dict | None = None,
) -> None:
    """把一轮用户问题和助手回答保存到 SQLite 会话历史。"""

    store = _get_knowledge_store()
    store.save_chat_exchange(
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=answer,
        model_name=model_name,
        metadata=metadata,
    )

def _stream_agent(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
) -> Iterator[str]:
    """把 Agent token 流转换成浏览器能识别的 SSE 文本流。"""

    try:
        stream_start_time = time.perf_counter()
        first_token_ms: float | None = None
        selected_model_mode = normalize_chat_model_mode(None)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        logger.info(
            "[接口] Agent流式输出开始 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s 问题=%s",
            user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
            message,
        )
        meta_payload = json.dumps(
            {
                "conversation_id": conversation_id,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
            },
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        chunks: list[str] = []
        for chunk in _get_agent().execute_stream(
                message,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history or [],
        ):
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(stream_start_time)
                metric_payload = json.dumps({"first_token_ms": first_token_ms}, ensure_ascii=False)
                yield f"event: metric\ndata: {metric_payload}\n\n"
            chunks.append(chunk)
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"event: chunk\ndata: {payload}\n\n"

        answer = "".join(chunks).strip()
        total_ms = _elapsed_ms(stream_start_time)
        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                model_name=selected_model_name,
                metadata={
                    "mode": "stream",
                    "model_mode": selected_model_mode,
                    "model_name": selected_model_name,
                    "first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                },
            )

        logger.info(
            "[接口] Agent流式输出完成 用户编号=%s 会话编号=%s 模型模式=%s 模型名称=%s",
            user_id,
            conversation_id,
            selected_model_mode,
            selected_model_name,
        )
        done_payload = json.dumps(
            {
                "done": True,
                "conversation_id": conversation_id,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
            },
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"
    except Exception as exc:
        logger.error("[接口] Agent流式输出失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


def _stream_direct_rag(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
        model_mode: str | None = None,
        collection_name: str | None = None,
) -> Iterator[str]:
    """把知识库直答 token 流转换成 SSE 文本流。"""

    try:
        stream_start_time = time.perf_counter()
        first_token_ms: float | None = None
        selected_model_mode = normalize_chat_model_mode(model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        selected_collection_name = normalize_qdrant_collection_name(collection_name)
        logger.info(
            "[聊天路由] 流式接口进入知识库直答 用户编号=%s 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s 问题=%s",
            user_id,
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
            message,
        )

        meta_payload = json.dumps(
            {
                "conversation_id": conversation_id,
                "mode": "direct_rag",
                "collection_name": selected_collection_name,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
            },
            ensure_ascii=False,
        )
        yield f"event: meta\ndata: {meta_payload}\n\n"

        chunks: list[str] = []
        for chunk in _get_knowledge_answer_service().stream_answer(
            message,
            history=history or [],
            model_mode=selected_model_mode,
            collection_name=selected_collection_name,
        ):
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(stream_start_time)
                metric_payload = json.dumps({"first_token_ms": first_token_ms}, ensure_ascii=False)
                yield f"event: metric\ndata: {metric_payload}\n\n"
            chunks.append(chunk)
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"event: chunk\ndata: {payload}\n\n"

        answer = "".join(chunks).strip()
        total_ms = _elapsed_ms(stream_start_time)

        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                model_name=selected_model_name,
                metadata={
                    "mode": "direct_rag_stream",
                    "model_mode": selected_model_mode,
                    "model_name": selected_model_name,
                    "collection_name": selected_collection_name,
                    "first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                },
            )

        logger.info(
            "[聊天路由] 知识直答流式完成 用户编号=%s 会话编号=%s Collection=%s 模型模式=%s 模型名称=%s",
            user_id,
            conversation_id,
            selected_collection_name,
            selected_model_mode,
            selected_model_name,
        )

        done_payload = json.dumps(
            {
                "done": True,
                "conversation_id": conversation_id,
                "collection_name": selected_collection_name,
                "model_mode": selected_model_mode,
                "model_name": selected_model_name,
                "first_token_ms": first_token_ms,
                "total_ms": total_ms,
            },
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"
    except Exception as exc:
        logger.error("[聊天路由] 知识直答流式失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"
def _elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000
