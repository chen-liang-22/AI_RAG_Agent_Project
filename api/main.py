import json  # 用于把 Python 字典序列化成 SSE data 里的 JSON 字符串
import threading  # 用于给 Agent 单例初始化加锁，避免并发请求重复创建 Agent
from collections.abc import Iterator  # 标注 _stream_agent 返回的是一个可迭代的字符串流

from fastapi import FastAPI, HTTPException  # FastAPI 创建应用；HTTPException 用于抛出 HTTP 错误
from fastapi.responses import StreamingResponse  # StreamingResponse 用于持续向浏览器推送 SSE 文本流
from pydantic import BaseModel, Field  # Pydantic 用于定义请求体和响应体结构
from qdrant_client import QdrantClient  # 健康检查时用于连接 Qdrant 并读取 collection 列表

from utils.logger_handler import logger  # 项目统一日志对象
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


class HealthResponse(BaseModel):
    """健康检查响应体，用于前端侧边栏展示服务状态。"""

    status: str  # 整体状态；ok 表示全部可用，degraded 表示部分依赖不可用
    qdrant: str  # Qdrant 状态；ok 或 unavailable
    collection_name: str  # 当前项目使用的 Qdrant collection 名称
    collections: list[str] = []  # Qdrant 中已有的 collection 列表


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


@app.post("/knowledge/reload")
def reload_knowledge() -> dict:
    logger.info("[api] /knowledge/reload")  # 记录知识库重载请求
    try:
        from rag.vector_store import VectorStoreService  # 延迟导入，只有重载知识库时才加载向量库服务

        vector_store = VectorStoreService()  # 创建向量库服务
        vector_store.load_document()  # 重新读取本地知识库文件并写入 Qdrant
    except Exception as exc:
        # 知识库加载失败时返回 500，前端会弹出错误提示。
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc

    return {"status": "ok", "collection_name": get_qdrant_collection_name()}  # 返回重载成功和 collection 名称
