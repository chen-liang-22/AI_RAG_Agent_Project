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

from api.routers import auth, chat, dictionaries, exam, health, internal_jobs, knowledge
from api.warmup import run_startup_warmup
from training.api.router import router as training_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时执行预热，关闭时交还 FastAPI 默认生命周期。"""

    run_startup_warmup()
    yield


app = FastAPI(
    title="知习台 API",
    description="知习台后端服务，提供知识库问答、资料管理、销售陪练和掌握度测评能力。",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(dictionaries.router)
app.include_router(exam.router)
app.include_router(training_router)
app.include_router(internal_jobs.router)
