import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterator

from fastapi import HTTPException, UploadFile

from api.schemas import ChatRequest, KnowledgeFileResponse
from rag.knowledge_store import KnowledgeStore
from utils.config_handler import qdrant_conf, rag_conf
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


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
    """把 SQLite 字典记录转换成 FastAPI 响应模型。

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
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        error_message=document.get("error_message"),
    )


def _sanitize_upload_filename(filename: str | None) -> str:
    """清理上传文件名，避免用户传入带目录的路径。

    浏览器正常只会上送文件名，比如 `guide.pdf`。
    但接口调用方也可能传 `C:\\tmp\\guide.pdf` 或 `../../guide.pdf`。
    后端只保留最后的文件名部分，避免把文件写到 uploads/ 之外。
    """

    raw_filename = (filename or "").strip()
    safe_filename = raw_filename.replace("\\", "/").split("/")[-1].strip()
    safe_filename = safe_filename.replace("\x00", "")

    if not safe_filename:
        raise HTTPException(status_code=400, detail="上传文件名不能为空")

    return safe_filename


def _validate_file_type(filename: str) -> str:
    """校验上传文件后缀是否在配置允许范围内。

    允许类型来自 config/qdrant.yml：
    allow_knowledge_file_type: ["txt", "pdf"]
    """

    file_type = os.path.splitext(filename)[1].lower().lstrip(".")
    allowed_types = {item.lower().lstrip(".") for item in qdrant_conf["allow_knowledge_file_type"]}

    if file_type not in allowed_types:
        allowed_text = ", ".join(sorted(allowed_types))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{file_type}，仅支持：{allowed_text}")

    return file_type


def _remove_created_upload_dir(upload_dir: str) -> None:
    """删除本次上传刚创建的目录。

    只在重复上传或保存失败时使用。
    删除前会校验目标目录必须位于项目 uploads/ 目录内，避免误删其它路径。
    """

    uploads_root = os.path.abspath(get_abs_path("uploads"))
    target_dir = os.path.abspath(upload_dir)

    if os.path.commonpath([uploads_root, target_dir]) != uploads_root:
        logger.warning(f"[知识库] 跳过不安全的上传目录清理 目录={target_dir}")
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
    except Exception:
        _remove_created_upload_dir(preview_dir)
        raise

    if file_size <= 0:
        _remove_created_upload_dir(preview_dir)
        raise HTTPException(status_code=400, detail="上传文件不能为空")

    return file_path, file_size


def _index_document(
        store: KnowledgeStore,
        document: dict,
        *,
        increment_version: bool = False,
        vector_store=None,
        document_type: str | None = None,
        split_strategy: str | None = None,
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
    store.update_document_status(
        document_id,
        "indexing",
        error_message=None,
        increment_version=increment_version,
    )

    indexing_document = store.get_document(document_id)
    if indexing_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        if vector_store is None:
            from rag.vector_store import VectorStoreService  # 延迟导入，只有真正入库时才加载向量库和 embedding 模型

            vector_store = VectorStoreService()

        chunk_count = vector_store.index_file(
            indexing_document,
            document_type=document_type,
            split_strategy=split_strategy,
        )
        store.update_document_status(document_id, "indexed", chunk_count=chunk_count, error_message=None)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[知识库] 文件入库失败 文档编号={document_id} 错误={exc}", exc_info=True)
        store.update_document_status(document_id, "failed", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"知识库入库失败：{exc}") from exc

    indexed_document = store.get_document(document_id)
    if indexed_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return indexed_document


def _sync_data_files_to_documents(store: KnowledgeStore) -> list[dict]:
    """把 data/ 目录中的内置知识文件同步到 documents 表。

    用户手动上传的文件会保存在 uploads/ 目录，而项目自带的示例/基础知识文件
    通常放在 data/ 目录。为了让这两类文件都能进入 documents 管理，
    reload 时先把 data/ 文件注册成 documents 记录，再统一走 _index_document。

    同一个文件内容按 MD5 去重：如果 active documents 中已有相同 MD5，就复用
    已有记录，不重复创建。
    """

    data_path = get_abs_path(qdrant_conf["data_path"])
    allowed_types = tuple(qdrant_conf["allow_knowledge_file_type"])
    file_paths = listdir_with_allowed_type(data_path, allowed_types)
    documents: list[dict] = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        file_type = os.path.splitext(filename)[1].lower().lstrip(".")
        file_md5 = get_file_md5_hex(file_path)

        if not file_md5:
            logger.warning(f"[知识库] 跳过无法计算MD5的内置知识文件 路径={file_path}")
            continue

        existing_document = store.find_active_document_by_md5(file_md5)
        if existing_document is not None:
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
            )
        )

    return documents


def _get_agent():
    """获取全局 Agent 单例。

    Agent 初始化会加载模型、工具、提示词和 RAG 检索链路，成本比较高。
    因此这里使用懒加载：
    - 第一次真正收到聊天请求时才创建 ReactAgent。
    - 后续请求复用同一个 Agent 实例，避免每次请求都重新初始化。

    `_agent_lock` 用于防止并发首个请求同时初始化多个 Agent。
    """

    global _agent  # 函数内部要给模块级变量 _agent 赋值，因此需要 global

    if _agent is not None:
        return _agent  # 已初始化过就直接复用，避免重复加载模型和工具

    with _agent_lock:
        # 进入锁之后再判断一次，防止两个请求排队时后一个重复初始化。
        if _agent is None:
            try:
                from agent.react_agent import ReactAgent  # 延迟导入，避免应用启动时就加载模型和向量库

                _agent = ReactAgent()  # 创建真正的 ReAct Agent 实例
            except Exception as exc:
                # Agent 初始化失败时，返回 500，让前端能看到明确错误。
                raise HTTPException(status_code=500, detail=f"Agent initialization failed: {exc}") from exc

    return _agent  # 返回初始化好的 Agent


def _get_knowledge_answer_service():
    """获取知识库直答服务单例。

    普通知识问答走这条直线路径：
    用户问题 -> Query Planner -> Qdrant -> 最终回答模型。
    这条路径不会绑定 Agent 工具，所以不会触发“模型判断是否调用 rag_summarize”。
    """

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
    """判断当前问题是否走知识库直答。

    这里不再按关键词猜意图：
    - 不配置 chat_route_mode：默认全部走直连 RAG。
    - chat_route_mode: agent：恢复之前的 Agent 工具链。
    """

    mode = _get_chat_route_mode()
    if mode in {"agent", "react_agent", "legacy_agent"}:
        return False, "配置 chat_route_mode=agent，使用之前的 Agent 工具链"

    return True, f"配置 chat_route_mode={mode}，使用直连 RAG"


def _build_conversation_title(message: str) -> str:
    """用首条用户问题生成一个简短会话标题。"""

    return message.strip().replace("\n", " ")[:40] or "新会话"


def _prepare_chat_conversation(request: ChatRequest) -> tuple[str, list[dict]]:
    """创建或读取会话，并返回 conversation_id 和最近历史。"""

    # KnowledgeStore 是 SQLite 元数据仓库，负责 documents 和 conversations 等业务表。
    store = _get_knowledge_store()

    # 如果 request.conversation_id 已存在，则复用旧会话。
    # 如果前端没有传 conversation_id，则创建新会话，并用当前问题生成一个短标题。
    conversation = store.ensure_conversation(
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        title=_build_conversation_title(request.message),
    )

    # 统一取出最终使用的 conversation_id；新建会话时这个值由后端生成。
    conversation_id = conversation["conversation_id"]

    # 读取最近 20 条历史消息，传给后续模型做上下文。
    # 这里不会读取整段会话，避免 prompt 越来越长。
    history = store.list_recent_messages(conversation_id, limit=20)

    # 返回给路由层：conversation_id 用于响应和保存历史，history 用于 RAG/Agent 上下文。
    return conversation_id, history


def _save_chat_exchange(
        *,
        conversation_id: str,
        message: str,
        answer: str,
        metadata: dict | None = None,
) -> None:
    """把一轮用户问题和助手回答保存到 SQLite 会话历史。"""

    store = _get_knowledge_store()
    store.save_chat_exchange(
        conversation_id=conversation_id,
        user_message=message,
        assistant_message=answer,
        metadata=metadata,
    )


def _stream_agent(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
) -> Iterator[str]:
    """把 Agent 的 token 流转换成浏览器能识别的 SSE 文本流。

    SSE(Server-Sent Events) 的基础格式是：

        event: 事件名
        data: JSON 字符串

    每个事件之间必须用一个空行分隔，也就是末尾的 `\n\n`。
    前端会按空行切分事件，再解析 `data:` 后面的 JSON。

    本项目约定了三类事件：
    - chunk：正常回答片段，格式为 `{"content": "..."}`
    - done：回答结束，格式为 `{"done": true}`
    - error：生成失败，格式为 `{"error": "..."}`

    注意：这里拿到的是 `ReactAgent.execute_stream()` 已经过滤后的内容，
    只包含最终 AI 回答 token，不包含 RAG 工具返回的参考资料或元数据。
    """

    try:  # 流式生成期间任何异常都会转成 SSE error 事件
        logger.info(
            f"[接口] Agent流式输出开始 用户编号={user_id} "
            f"会话编号={conversation_id} 问题={message}"
        )
        meta_payload = json.dumps({"conversation_id": conversation_id}, ensure_ascii=False)
        yield f"event: meta\ndata: {meta_payload}\n\n"

        chunks: list[str] = []
        for chunk in _get_agent().execute_stream(
                message,
                user_id=user_id,
                conversation_id=conversation_id,
                history=history or [],
        ):
            # ensure_ascii=False 可以让中文直接以 UTF-8 传给前端，
            # 避免浏览器端看到 \u4f60\u597d 这种转义内容。
            chunks.append(chunk)
            payload = json.dumps({"content": chunk}, ensure_ascii=False)  # 把文本片段包装成 JSON
            yield f"event: chunk\ndata: {payload}\n\n"  # yield 一次，浏览器就有机会收到一个 SSE chunk

        answer = "".join(chunks).strip()
        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                metadata={"mode": "stream"},
            )

        logger.info(f"[接口] Agent流式输出完成 用户编号={user_id} 会话编号={conversation_id}")
        done_payload = json.dumps({"done": True, "conversation_id": conversation_id}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"  # 告诉前端流式回答已经结束
    except Exception as exc:
        logger.error(f"[接口] Agent流式输出失败 用户编号={user_id} 错误={exc}", exc_info=True)
        # 流式响应已经开始后，HTTP 状态码通常不能再改成 500。
        # 因此这里用 SSE error 事件把错误发给前端，让前端显示错误消息。
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)  # 把异常信息包装成 JSON
        yield f"event: error\ndata: {payload}\n\n"  # 用 SSE error 事件通知前端


def _stream_direct_rag(
        message: str,
        user_id: str | None = None,
        conversation_id: str | None = None,
        history: list[dict] | None = None,
) -> Iterator[str]:
    """把知识库直答 token 流转换成 SSE 文本流。

    这个生成器会被 FastAPI 的 StreamingResponse 消费。
    只要这里 yield 一段文本，前端就有机会立刻收到一段内容。
    """

    try:
        # 记录当前流式请求进入 direct_rag 链路。
        logger.info(
            "[聊天路由] 流式接口走知识直答 用户编号=%s 会话编号=%s 问题=%s",
            user_id,
            conversation_id,
            message,
        )

        # 先发一个 meta 事件给前端。
        # 前端可以在收到正文前就拿到 conversation_id，并知道当前模式是 direct_rag。
        meta_payload = json.dumps(
            {"conversation_id": conversation_id, "mode": "direct_rag"},
            ensure_ascii=False,
        )

        # SSE 格式要求：
        # event: 事件名
        # data: JSON 字符串
        # 空行表示一个事件结束。
        yield f"event: meta\ndata: {meta_payload}\n\n"

        # 用列表缓存所有 chunk，流式结束后拼成完整回答并保存到 SQLite。
        chunks: list[str] = []

        # 调用知识库直答服务：
        # 内部流程是 RAG 检索上下文 -> 调用最终回答模型 stream() -> 逐段返回文本。
        for chunk in _get_knowledge_answer_service().stream_answer(message, history=history or []):
            # 保存当前分片，后面用于拼接完整回答。
            chunks.append(chunk)

            # 把文本分片包装成 JSON，ensure_ascii=False 保证中文直接输出。
            payload = json.dumps({"content": chunk}, ensure_ascii=False)

            # 发给前端一个 chunk 事件。
            # 前端收到后会把 content 追加到聊天气泡里。
            yield f"event: chunk\ndata: {payload}\n\n"

        # 模型流结束后，把所有分片拼成最终回答。
        answer = "".join(chunks).strip()

        # 如果有会话编号且回答非空，就把“用户问题 + 助手回答”保存到 SQLite。
        if conversation_id and answer:
            _save_chat_exchange(
                conversation_id=conversation_id,
                message=message,
                answer=answer,
                metadata={"mode": "direct_rag_stream"},
            )

        # 记录后端流式生成结束。
        logger.info("[聊天路由] 知识直答流式完成 用户编号=%s 会话编号=%s", user_id, conversation_id)

        # 通知前端本次流式回答结束。
        done_payload = json.dumps({"done": True, "conversation_id": conversation_id}, ensure_ascii=False)
        yield f"event: done\ndata: {done_payload}\n\n"
    except Exception as exc:
        # 流式响应开始后，HTTP 状态码通常已经发给浏览器了。
        # 所以异常不能再改成 500，只能通过 SSE error 事件告诉前端。
        logger.error("[聊天路由] 知识直答流式失败 用户编号=%s 错误=%s", user_id, exc, exc_info=True)

        # 把错误消息包装成 JSON 并推给前端。
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"
