from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import chat, dictionaries, exam, health, knowledge
from api.warmup import run_startup_warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
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
