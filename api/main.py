"""FastAPI 应用入口。

V2 重构后，这里只负责创建应用、启动预热和挂载路由。
业务流程放到 `app` 的应用服务层，避免入口文件继续变成大杂烩。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.warmup import run_startup_warmup
from app.api.routes import internal_jobs
from app.api.router import router as v2_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子。

    FastAPI 启动时只执行必要预热；不要在这里偷偷修改表结构。
    """

    run_startup_warmup()
    yield


app = FastAPI(
    title="知识台 V2 API",
    description="AI RAG Agent V2 分层架构接口，只暴露新的 /api/v2 协议。",
    version="2.0.0",
    lifespan=lifespan,
)

# V2 业务接口统一挂载到 /api/v2，前端也只调用这套协议。
app.include_router(v2_router)

# 内部定时任务保留原 URL，但实现已经迁入 V2 路由模块。
app.include_router(internal_jobs.router)
