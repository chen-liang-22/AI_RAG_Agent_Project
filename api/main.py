from fastapi import FastAPI

from api.routers import chat, health, knowledge

app = FastAPI(
    title="AI RAG Agent API",
    description="Deployable API service for the RAG customer-service agent.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
