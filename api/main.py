"""FastAPI 应用入口。

这里只负责创建应用、执行启动预热，并挂载各业务路由：
- health：健康检查；
- chat：智能客服聊天；
- knowledge：知识库文件管理；
- dictionaries：字典配置；
- exam：对话式考试。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import chat, dictionaries, exam, health, knowledge
from api.warmup import run_startup_warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时执行预热，关闭时交还 FastAPI 默认生命周期。"""

    run_startup_warmup()
    yield


app = FastAPI(
    title="AI RAG Agent API",
    description="Deployable API service for the RAG customer-service agent.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(dictionaries.router)
app.include_router(exam.router)
