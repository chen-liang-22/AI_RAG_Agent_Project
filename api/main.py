import json
import threading
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name


app = FastAPI(
    title="AI RAG Agent API",
    description="Deployable API service for the RAG customer-service agent.",
    version="1.0.0",
)

_agent = None
_agent_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection_name: str
    collections: list[str] = []


def _get_agent():
    global _agent

    if _agent is not None:
        return _agent

    with _agent_lock:
        if _agent is None:
            try:
                from agent.react_agent import ReactAgent

                _agent = ReactAgent()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Agent initialization failed: {exc}") from exc

    return _agent


def _stream_agent(message: str) -> Iterator[str]:
    try:
        for chunk in _get_agent().execute_stream(message):
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"event: chunk\ndata: {payload}\n\n"

        yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"
    except Exception as exc:
        payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield f"event: error\ndata: {payload}\n\n"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    collection_name = get_qdrant_collection_name()

    try:
        client = QdrantClient(**get_qdrant_client_options())
        collections = [collection.name for collection in client.get_collections().collections]
        qdrant_status = "ok"
    except Exception:
        collections = []
        qdrant_status = "unavailable"

    status = "ok" if qdrant_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant_status,
        collection_name=collection_name,
        collections=collections,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    chunks = list(_get_agent().execute_stream(request.message))
    answer = chunks[-1].strip() if chunks else ""
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    _get_agent()
    return StreamingResponse(_stream_agent(request.message), media_type="text/event-stream")


@app.post("/knowledge/reload")
def reload_knowledge() -> dict:
    try:
        from rag.vector_store import VectorStoreService

        vector_store = VectorStoreService()
        vector_store.load_document()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc

    return {"status": "ok", "collection_name": get_qdrant_collection_name()}
