import json  # 用于把 Python 字典序列化成 SSE data 里的 JSON 字符串
import os  # 用于拼接上传文件路径、判断文件后缀和创建目录
import shutil  # 用于清理重复上传或失败上传时产生的临时目录
import threading  # 用于给 Agent 单例初始化加锁，避免并发请求重复创建 Agent
import uuid  # 用于生成 document_id，保证每个上传文件都有唯一 ID
from collections.abc import Iterator  # 标注 _stream_agent 返回的是一个可迭代的字符串流

from fastapi import FastAPI, File, HTTPException, UploadFile  # FastAPI 创建应用；File/UploadFile 用于文件上传；HTTPException 用于抛出 HTTP 错误
from fastapi.responses import StreamingResponse  # StreamingResponse 用于持续向浏览器推送 SSE 文本流
from pydantic import BaseModel, Field  # Pydantic 用于定义请求体和响应体结构
from qdrant_client import QdrantClient  # 健康检查时用于连接 Qdrant 并读取 collection 列表

from rag.knowledge_store import KnowledgeStore  # SQLite 元数据存储，保存文件记录和知识单元记录
from utils.config_handler import qdrant_conf  # 读取知识库允许文件类型、切分配置等
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type  # 计算文件 MD5；扫描 data 目录允许入库的文件
from utils.logger_handler import logger  # 项目统一日志对象
from utils.path_tool import get_abs_path  # 把 uploads/storage 等相对路径转成项目绝对路径
from utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name  # 读取 Qdrant 配置


app = FastAPI(
    title="AI RAG Agent API",  # Swagger/OpenAPI 页面显示的服务名称
    description="Deployable API service for the RAG customer-service agent.",  # API 文档描述
    version="1.0.0",  # API 版本号
)

_agent = None  # 全局 Agent 单例；第一次聊天请求进来时才真正初始化
_agent_lock = threading.Lock()  # 初始化锁；防止多个请求同时初始化多个 Agent


class ChatRequest(BaseModel):
    """前端聊天请求体。

    一次性接口 `/chat` 和流式接口 `/chat/stream` 共用同一个请求结构：
    - message：用户本次输入的问题。
    - user_id：当前会话的用户 ID，用于工具调用时读取用户画像或外部数据。

    两个接口使用相同请求体，是为了保证前端只需要切换接口地址，
    不需要为两种输出模式维护两套参数结构。
    """

    message: str = Field(..., min_length=1)  # 用户输入的问题；不能为空字符串
    user_id: str | None = None  # 当前会话用户 ID；可为空，工具层会兜底随机用户


class ChatResponse(BaseModel):
    """一次性接口 `/chat` 的响应体。

    一次性返回时，后端会等待 Agent 完整执行结束，然后只返回最终回答。
    这里不会返回中间 RAG 参考资料、工具调用结果或推理过程。
    """

    answer: str  # 一次性接口返回的完整最终回答


class DebugRetrieveRequest(BaseModel):
    """RAG 检索调试请求体。"""

    query: str = Field(..., min_length=1)  # 要调试的用户问题


class HealthResponse(BaseModel):
    """健康检查响应体，用于前端侧边栏展示服务状态。"""

    status: str  # 整体状态；ok 表示全部可用，degraded 表示部分依赖不可用
    qdrant: str  # Qdrant 状态；ok 或 unavailable
    collection_name: str  # 当前项目使用的 Qdrant collection 名称
    collections: list[str] = Field(default_factory=list)  # Qdrant 中已有的 collection 列表


class KnowledgeFileResponse(BaseModel):
    """知识库文件响应体。

    这个模型对应 SQLite 里的 documents 表。
    前端、Swagger、Postman 看到的文件列表和文件详情都会按这个结构返回。
    """

    document_id: str  # 文件唯一 ID；删除和重建索引都靠它定位文件
    filename: str  # 用户上传时的原始文件名
    file_path: str  # 文件在服务端 uploads/ 目录下的保存路径
    file_type: str  # 文件类型，例如 txt/pdf
    file_md5: str  # 文件内容 MD5；用于判断重复上传
    file_size: int  # 文件大小，单位字节
    status: str  # uploaded/indexing/indexed/failed/deleted
    version: int  # 文件索引版本；每次 reindex 会递增
    chunk_count: int  # 当前版本写入 Qdrant 的知识单元数量
    created_at: str  # 文件记录创建时间
    updated_at: str  # 文件记录最后更新时间
    error_message: str | None = None  # 入库失败时保存错误原因


class KnowledgeUploadResponse(BaseModel):
    """上传知识库文件的响应体。

    status 可能是：
    - indexed：新文件已经成功解析并写入 Qdrant。
    - duplicate：相同 MD5 的文件已经存在，本次没有重复入库。
    """

    status: str  # indexed 或 duplicate
    message: str  # 面向调用方的简短说明
    document: KnowledgeFileResponse  # 成功入库或已存在的文件记录


class KnowledgeUploadPreviewResponse(BaseModel):
    """上传预览响应体。

    预览阶段只保存临时文件并识别文档类型，不写 documents，也不写 Qdrant。
    """

    upload_id: str  # 临时上传 ID，确认入库时使用
    filename: str  # 原始文件名
    file_type: str  # 文件类型
    file_size: int  # 文件大小
    file_md5: str  # 文件 MD5
    duplicate: bool = False  # 是否与已有 active 文件重复
    duplicate_document: KnowledgeFileResponse | None = None  # 重复时对应的已有文件
    detected_type: str  # 系统识别的文档类型
    split_strategy: str  # 系统建议的切分策略
    confidence: float  # 识别置信度
    reasons: list[str] = Field(default_factory=list)  # 识别原因
    llm_used: bool = False  # 是否使用了 LLM 兜底
    sample_text: str = ""  # 抽样文本，给前端预览


class KnowledgeUploadConfirmRequest(BaseModel):
    """上传确认请求体。"""

    upload_id: str = Field(..., min_length=1)  # 预览阶段返回的 upload_id
    document_type: str = Field(..., min_length=1)  # 用户确认后的文档类型
    split_strategy: str = Field(..., min_length=1)  # 用户确认后的切分策略


class KnowledgeDeleteResponse(BaseModel):
    """删除知识库文件的响应体。"""

    status: str  # 固定返回 deleted
    document_id: str  # 被删除的文件 ID


class KnowledgeReindexResult(BaseModel):
    """单个文件重建索引结果。"""

    document_id: str  # 文件 ID
    filename: str  # 文件名
    status: str  # indexed 或 failed
    message: str | None = None  # 失败原因或成功说明


class KnowledgeBulkReindexResponse(BaseModel):
    """全部重建索引响应体。"""

    status: str  # ok 或 partial_failed
    total: int  # 参与重建的文件数
    succeeded: int  # 成功数量
    failed: int  # 失败数量
    results: list[KnowledgeReindexResult]  # 每个文件的结果


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
        logger.warning(f"[knowledge] skip unsafe upload cleanup: {target_dir}")
        return

    shutil.rmtree(target_dir, ignore_errors=True)


def _move_upload_file(source_path: str, document_id: str, filename: str) -> str:
    """把临时上传文件移动到正式 uploads/{document_id}/ 目录。"""

    upload_dir = os.path.join(get_abs_path("uploads"), document_id)
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.join(upload_dir, filename)
    shutil.move(source_path, target_path)
    return target_path


def _save_upload_file(file: UploadFile, filename: str, document_id: str) -> tuple[str, int]:
    """把上传文件保存到 uploads/{document_id}/{filename}。

    返回：
    - file_path：服务端保存路径
    - file_size：写入的字节数
    """

    upload_dir = os.path.join(get_abs_path("uploads"), document_id)
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)
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
        _remove_created_upload_dir(upload_dir)
        raise

    if file_size <= 0:
        _remove_created_upload_dir(upload_dir)
        raise HTTPException(status_code=400, detail="上传文件不能为空")

    return file_path, file_size


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
    5. knowledge_units 表替换成新的知识单元。
    6. documents.status 改成 indexed。
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

        chunk_count, segments, faq_items = vector_store.index_file(
            indexing_document,
            document_type=document_type,
            split_strategy=split_strategy,
        )
        store.replace_segments_and_faqs(document_id, segments, faq_items)
        store.update_document_status(document_id, "indexed", chunk_count=chunk_count, error_message=None)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[knowledge] index failed document_id={document_id}: {exc}", exc_info=True)
        store.update_document_status(document_id, "failed", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"知识库入库失败：{exc}") from exc

    indexed_document = store.get_document(document_id)
    if indexed_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return indexed_document


def _sync_data_files_to_documents(store: KnowledgeStore) -> list[dict]:
    """把 data/ 目录中的内置知识文件同步到 documents 表。

    用户手动上传的文件会保存在 uploads/ 目录，而项目自带的示例/基础知识文件
    通常放在 data/ 目录。为了让这两类文件都能生成 document_segments/faq_items，
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
            logger.warning(f"[knowledge] skip data file without md5: {file_path}")
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


def _stream_agent(message: str, user_id: str | None = None) -> Iterator[str]:
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
        logger.info(f"[api] /chat/stream start user_id={user_id} message={message}")
        for chunk in _get_agent().execute_stream(message, user_id=user_id):
            # ensure_ascii=False 可以让中文直接以 UTF-8 传给前端，
            # 避免浏览器端看到 \u4f60\u597d 这种转义内容。
            payload = json.dumps({"content": chunk}, ensure_ascii=False)  # 把文本片段包装成 JSON
            yield f"event: chunk\ndata: {payload}\n\n"  # yield 一次，浏览器就有机会收到一个 SSE chunk

        logger.info(f"[api] /chat/stream done user_id={user_id}")
        yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"  # 告诉前端流式回答已经结束
    except Exception as exc:
        logger.error(f"[api] /chat/stream failed user_id={user_id}: {exc}", exc_info=True)
        # 流式响应已经开始后，HTTP 状态码通常不能再改成 500。
        # 因此这里用 SSE error 事件把错误发给前端，让前端显示错误消息。
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)  # 把异常信息包装成 JSON
        yield f"event: error\ndata: {payload}\n\n"  # 用 SSE error 事件通知前端


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info("[api] /health")  # 记录健康检查访问日志
    collection_name = get_qdrant_collection_name()  # 读取当前配置中的 collection 名称

    try:
        client = QdrantClient(**get_qdrant_client_options())  # 按配置连接 Qdrant
        collections = [collection.name for collection in client.get_collections().collections]  # 获取全部 collection 名称
        qdrant_status = "ok"  # 能正常连接和读取 collection，说明 Qdrant 可用
    except Exception:
        collections = []  # Qdrant 不可用时返回空列表，避免健康检查接口直接报错
        qdrant_status = "unavailable"  # 标记 Qdrant 不可用

    status = "ok" if qdrant_status == "ok" else "degraded"  # 依赖不可用时整体状态降级
    return HealthResponse(
        status=status,  # 返回整体状态
        qdrant=qdrant_status,  # 返回 Qdrant 状态
        collection_name=collection_name,  # 返回当前 collection 名称
        collections=collections,  # 返回 Qdrant collection 列表
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """一次性聊天接口。

    调用流程：
    1. 前端发送 POST `/chat`，请求体包含 message 和 user_id。
    2. 后端调用 `ReactAgent.execute()`。
    3. Agent 内部可能会调用 RAG、天气、用户数据等工具。
    4. 后端等待 Agent 全部执行完，只取最后一条 AIMessage 作为最终回答。
    5. 返回 JSON：`{"answer": "...完整回答..."}`

    这个接口适合“不关心首 token 速度，只想一次拿到完整结果”的场景。
    """

    logger.info(f"[api] /chat user_id={request.user_id} message={request.message}")  # 标记本次调用的是一次性接口
    answer = _get_agent().execute(request.message, user_id=request.user_id)  # 阻塞等待 Agent 完整生成最终回答
    return ChatResponse(answer=answer)  # 按 response_model 返回 {"answer": "..."}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式聊天接口。

    调用流程：
    1. 前端发送 POST `/chat/stream`，请求体同样包含 message 和 user_id。
    2. 后端返回 `StreamingResponse`，浏览器不会等待完整回答结束。
    3. Agent 每生成一段 AI token，`_stream_agent()` 就包装成一个 SSE chunk 事件。
    4. 前端通过 `ReadableStream.getReader()` 持续读取 chunk，并实时追加到页面。

    为什么这里仍然用 POST：
    - 请求体里有 message、user_id，未来还可能有历史上下文、模式配置等结构化参数。
    - 原生 EventSource 只能 GET，不方便携带复杂 JSON 请求体。
    - fetch + ReadableStream 可以同时保留 POST 语义和 SSE 文本流格式。

    响应头说明：
    - Cache-Control: no-cache, no-transform
      禁止缓存和内容转换，避免代理层把流式响应攒完整再发。
    - Connection: keep-alive
      明确告诉客户端这是一个持续连接。
    - X-Accel-Buffering: no
      Nginx 场景下关闭响应缓冲；本地开发也保留，便于以后部署。
    """

    logger.info(f"[api] /chat/stream requested user_id={request.user_id} message={request.message}")  # 标记本次调用的是流式接口
    _get_agent()  # 提前初始化 Agent；初始化失败时可以在返回流之前直接报错
    return StreamingResponse(
        _stream_agent(request.message, user_id=request.user_id),  # 把生成器交给 FastAPI 持续输出
        media_type="text/event-stream",  # 告诉浏览器这是 SSE 文本流
        headers={
            "Cache-Control": "no-cache, no-transform",  # 不缓存、不转换内容，减少代理缓冲风险
            "Connection": "keep-alive",  # 保持连接不断开，适合长响应
            "X-Accel-Buffering": "no",  # Nginx 代理时关闭响应缓冲
        },
    )


@app.post("/knowledge/upload", response_model=KnowledgeUploadResponse)
def upload_knowledge_file(file: UploadFile = File(...)) -> KnowledgeUploadResponse:
    """上传知识库文件，并立即写入 SQLite 和 Qdrant。

    这个接口是设计文档第一阶段的核心入口。

    调用方通过 multipart/form-data 上传文件：
    - key: file
    - value: .txt 或 .pdf 文件

    后端处理流程：
    1. 清理文件名，避免目录穿越。
    2. 校验文件类型，只允许 qdrant.yml 中配置的后缀。
    3. 保存原始文件到 uploads/{document_id}/。
    4. 计算文件 MD5，判断是否已经有相同内容的 active 文件。
    5. 如果重复，删除本次刚保存的文件，返回已有文件记录。
    6. 如果不重复，写入 documents 表。
    7. 解析文件、切分知识单元、写入 Qdrant。
    8. 写入 knowledge_units 表，并更新 documents.status。
    """

    filename = _sanitize_upload_filename(file.filename)
    file_type = _validate_file_type(filename)
    document_id = f"doc_{uuid.uuid4().hex}"

    logger.info(f"[knowledge] upload start filename={filename} document_id={document_id}")

    try:
        file_path, file_size = _save_upload_file(file, filename, document_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[knowledge] upload save failed filename={filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc

    upload_dir = os.path.dirname(file_path)
    file_md5 = get_file_md5_hex(file_path)
    if not file_md5:
        _remove_created_upload_dir(upload_dir)
        raise HTTPException(status_code=500, detail="上传文件 MD5 计算失败")

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(file_md5)
    if duplicate_document is not None:
        _remove_created_upload_dir(upload_dir)
        logger.info(
            "[knowledge] duplicate upload "
            f"filename={filename} existing_document_id={duplicate_document['document_id']}"
        )
        return KnowledgeUploadResponse(
            status="duplicate",
            message="相同内容的文件已经存在，本次没有重复入库。",
            document=_document_to_response(duplicate_document),
        )

    document = store.create_document(
        document_id=document_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_md5=file_md5,
        file_size=file_size,
        status="uploaded",
    )

    try:
        from rag.vector_store import VectorStoreService

        vector_store = VectorStoreService()
        preview = vector_store.preview_file(filename=filename, file_path=file_path)
        indexed_document = _index_document(
            store,
            document,
            document_type=preview["document_type"],
            split_strategy=preview["split_strategy"],
            vector_store=vector_store,
        )
    except HTTPException:
        raise
    logger.info(f"[knowledge] upload indexed document_id={document_id}")

    return KnowledgeUploadResponse(
        status="indexed",
        message="文件已上传并写入知识库。",
        document=_document_to_response(indexed_document),
    )


@app.post("/knowledge/upload/preview", response_model=KnowledgeUploadPreviewResponse)
def preview_knowledge_file(file: UploadFile = File(...)) -> KnowledgeUploadPreviewResponse:
    """上传文件并返回识别结果，等待用户确认后再正式入库。"""

    filename = _sanitize_upload_filename(file.filename)
    file_type = _validate_file_type(filename)
    upload_id = f"tmp_{uuid.uuid4().hex}"

    logger.info(f"[knowledge] upload preview filename={filename} upload_id={upload_id}")

    try:
        file_path, file_size = _save_preview_file(file, filename, upload_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[knowledge] preview save failed filename={filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc

    file_md5 = get_file_md5_hex(file_path)
    if not file_md5:
        _remove_created_upload_dir(os.path.dirname(file_path))
        raise HTTPException(status_code=500, detail="上传文件 MD5 计算失败")

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(file_md5)
    if duplicate_document is not None:
        _remove_created_upload_dir(os.path.dirname(file_path))
        return KnowledgeUploadPreviewResponse(
            upload_id=upload_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_md5=file_md5,
            duplicate=True,
            duplicate_document=_document_to_response(duplicate_document),
            detected_type="duplicate",
            split_strategy="duplicate",
            confidence=1.0,
            reasons=["相同内容的文件已经存在"],
            llm_used=False,
            sample_text="",
        )

    try:
        from rag.vector_store import VectorStoreService

        preview = VectorStoreService().preview_file(filename=filename, file_path=file_path)
    except Exception as exc:
        _remove_created_upload_dir(os.path.dirname(file_path))
        logger.error(f"[knowledge] preview parse failed filename={filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件预解析失败：{exc}") from exc

    return KnowledgeUploadPreviewResponse(
        upload_id=upload_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        file_md5=file_md5,
        duplicate=duplicate_document is not None,
        duplicate_document=_document_to_response(duplicate_document) if duplicate_document else None,
        detected_type=preview["document_type"],
        split_strategy=preview["split_strategy"],
        confidence=preview["confidence"],
        reasons=preview["reasons"],
        llm_used=preview["llm_used"],
        sample_text=preview["sample_text"],
    )


@app.post("/knowledge/upload/confirm", response_model=KnowledgeUploadResponse)
def confirm_knowledge_file(request: KnowledgeUploadConfirmRequest) -> KnowledgeUploadResponse:
    """确认预览结果，并正式写入 SQLite 和 Qdrant。"""

    upload_id = request.upload_id.strip()
    preview_dir = os.path.abspath(os.path.join(get_abs_path("uploads"), "_preview", upload_id))
    preview_root = os.path.abspath(os.path.join(get_abs_path("uploads"), "_preview"))

    if os.path.commonpath([preview_root, preview_dir]) != preview_root or not os.path.isdir(preview_dir):
        raise HTTPException(status_code=404, detail=f"临时上传不存在：{upload_id}")

    files = [name for name in os.listdir(preview_dir) if os.path.isfile(os.path.join(preview_dir, name))]
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="临时上传文件状态异常")

    filename = _sanitize_upload_filename(files[0])
    file_type = _validate_file_type(filename)
    preview_path = os.path.join(preview_dir, filename)
    file_md5 = get_file_md5_hex(preview_path)
    file_size = os.path.getsize(preview_path)

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(file_md5)
    if duplicate_document is not None:
        _remove_created_upload_dir(preview_dir)
        return KnowledgeUploadResponse(
            status="duplicate",
            message="相同内容的文件已经存在，本次没有重复入库。",
            document=_document_to_response(duplicate_document),
        )

    document_id = f"doc_{uuid.uuid4().hex}"
    try:
        file_path = _move_upload_file(preview_path, document_id, filename)
        _remove_created_upload_dir(preview_dir)
    except Exception as exc:
        logger.error(f"[knowledge] confirm move failed upload_id={upload_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"临时文件转正式文件失败：{exc}") from exc

    document = store.create_document(
        document_id=document_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_md5=file_md5,
        file_size=file_size,
        status="uploaded",
    )

    indexed_document = _index_document(
        store,
        document,
        document_type=request.document_type,
        split_strategy=request.split_strategy,
    )

    return KnowledgeUploadResponse(
        status="indexed",
        message="文件已按确认配置写入知识库。",
        document=_document_to_response(indexed_document),
    )


@app.get("/knowledge/files", response_model=list[KnowledgeFileResponse])
def list_knowledge_files() -> list[KnowledgeFileResponse]:
    """查询知识库文件列表。

    只返回 status != deleted 的文件。
    这个接口后续可以直接给前端知识库管理页面使用。
    """

    logger.info("[knowledge] list files")
    store = _get_knowledge_store()
    return [_document_to_response(document) for document in store.list_documents()]


@app.get("/knowledge/files/{document_id}", response_model=KnowledgeFileResponse)
def get_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """查询单个知识库文件详情。"""

    logger.info(f"[knowledge] get file document_id={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return _document_to_response(document)


@app.delete("/knowledge/files/{document_id}", response_model=KnowledgeDeleteResponse)
def delete_knowledge_file(document_id: str) -> KnowledgeDeleteResponse:
    """按 document_id 删除知识库文件。

    这里的“删除”是知识库层面的删除：
    - Qdrant 中该 document_id 的 points 会被删除。
    - SQLite knowledge_units 中该文件的知识单元会被删除。
    - SQLite documents 中该文件会标记为 deleted。

    原始上传文件暂时保留在 uploads/ 目录中，方便排查和审计。
    如果后续希望物理删除原始文件，可以在这个接口里追加文件删除逻辑。
    """

    logger.info(f"[knowledge] delete file document_id={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        from rag.vector_store import VectorStoreService  # 延迟导入，只在删除 Qdrant points 时加载

        VectorStoreService.delete_document_vectors(document_id)
        store.delete_units(document_id)
        store.mark_document_deleted(document_id)
    except Exception as exc:
        logger.error(f"[knowledge] delete failed document_id={document_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"知识库文件删除失败：{exc}") from exc

    return KnowledgeDeleteResponse(status="deleted", document_id=document_id)


@app.post("/knowledge/files/reindex-all", response_model=KnowledgeBulkReindexResponse)
def reindex_all_knowledge_files() -> KnowledgeBulkReindexResponse:
    """清空 Qdrant collection，并重建所有 active 知识库文件的索引。

    这个接口用于把旧的粗糙 chunk 数据迁移到新的结构化知识单元。
    注意：
    - 会先删除并重建当前 Qdrant collection，清理无 document_id 的旧 points。
    - 会重新读取 uploads/ 中的原始文件。
    - 会重新调用 embedding。
    - 文件多时会比较慢，也会消耗模型调用额度。
    """

    logger.info("[knowledge] reindex all files")
    store = _get_knowledge_store()
    documents = store.list_documents()
    results: list[KnowledgeReindexResult] = []
    succeeded = 0
    failed = 0

    try:
        from rag.vector_store import VectorStoreService

        vector_store = VectorStoreService.recreate_collection_service()
    except Exception as exc:
        logger.error(f"[knowledge] recreate collection failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Qdrant collection 重建失败：{exc}") from exc

    for document in documents:
        document_id = document["document_id"]
        filename = document["filename"]

        try:
            store.delete_units(document_id)
            indexed_document = _index_document(
                store,
                document,
                increment_version=True,
                vector_store=vector_store,
            )
            succeeded += 1
            results.append(
                KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status="indexed",
                    message=f"chunk_count={indexed_document['chunk_count']}",
                )
            )
        except Exception as exc:
            failed += 1
            message = exc.detail if isinstance(exc, HTTPException) else str(exc)
            results.append(
                KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status="failed",
                    message=str(message),
                )
            )

    status = "ok" if failed == 0 else "partial_failed"
    return KnowledgeBulkReindexResponse(
        status=status,
        total=len(documents),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.post("/knowledge/files/{document_id}/reindex", response_model=KnowledgeFileResponse)
def reindex_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """重新解析并索引某个知识库文件。

    使用场景：
    - 调整了 chunk_size/chunk_overlap 后，希望重新切分。
    - 后续升级了结构化切分规则，希望重新生成 knowledge_units。
    - Qdrant 中某个文件的向量异常，需要按 document_id 重建。

    reindex 会递增 documents.version。
    新写入 Qdrant 的 payload 里也会带上新的 version。
    """

    logger.info(f"[knowledge] reindex file document_id={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    indexed_document = _index_document(store, document, increment_version=True)
    return _document_to_response(indexed_document)


@app.post("/knowledge/reload")
def reload_knowledge() -> dict:
    """扫描 data/ 目录并按新结构重建知识库。

    旧版 reload 只把 data/ 文件切块写进 Qdrant，不会写 documents、
    document_segments、faq_items。现在这个接口会把 data/ 文件也同步到
    documents 表，然后复用 reindex-all 的新流程，保证结构化 FAQ 查询可用。
    """

    logger.info("[api] /knowledge/reload")  # 记录知识库重载请求
    store = _get_knowledge_store()

    try:
        from rag.vector_store import VectorStoreService  # 延迟导入，只有重载知识库时才加载向量库服务

        _sync_data_files_to_documents(store)
        documents = store.list_documents()
        vector_store = VectorStoreService.recreate_collection_service()

        results: list[dict] = []
        succeeded = 0
        failed = 0
        for document in documents:
            try:
                store.delete_units(document["document_id"])
                indexed_document = _index_document(
                    store,
                    document,
                    increment_version=True,
                    vector_store=vector_store,
                )
                succeeded += 1
                results.append(
                    {
                        "document_id": indexed_document["document_id"],
                        "filename": indexed_document["filename"],
                        "status": "indexed",
                        "chunk_count": indexed_document["chunk_count"],
                    }
                )
            except Exception as exc:
                failed += 1
                message = exc.detail if isinstance(exc, HTTPException) else str(exc)
                results.append(
                    {
                        "document_id": document["document_id"],
                        "filename": document["filename"],
                        "status": "failed",
                        "message": str(message),
                    }
                )
    except Exception as exc:
        # 知识库加载失败时返回 500，前端会弹出错误提示。
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc

    return {
        "status": "ok" if failed == 0 else "partial_failed",
        "collection_name": get_qdrant_collection_name(),
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }


@app.post("/debug/retrieve")
def debug_retrieve(request: DebugRetrieveRequest) -> dict:
    """调试 RAG 检索链路。

    这个接口不会生成最终回答，只展示检索阶段的信息：
    - intent_analyze 识别到的意图
    - query_rewrite 生成的子查询
    - metadata filter
    - rerank 后的候选资料和分数

    开发阶段可以用它判断“为什么这个问题召回了这些资料”。
    生产环境如果不想暴露调试信息，可以后续加开关或权限。
    """

    logger.info(f"[api] /debug/retrieve query={request.query}")

    try:
        from rag.rag_service import RagSummarizeService  # 延迟导入，避免普通接口启动时就初始化向量库

        rag_service = RagSummarizeService()
        return rag_service.debug_retrieve(request.query)
    except Exception as exc:
        logger.error(f"[api] /debug/retrieve failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG retrieve debug failed: {exc}") from exc
