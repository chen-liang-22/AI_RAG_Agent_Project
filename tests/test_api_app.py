import os

from fastapi.testclient import TestClient

import api.routers.knowledge as knowledge_router
from api.main import app
from utils.path_tool import get_abs_path


def test_openapi_exposes_core_routes():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/chat" in paths
    assert "/chat/stream" in paths
    assert "/conversations" in paths
    assert "/conversations/{conversation_id}" in paths
    assert "/debug/retrieve" in paths
    assert "/knowledge/upload/preview" in paths
    assert "/knowledge/upload/recommend" in paths
    assert "/knowledge/upload/confirm" in paths
    assert "/knowledge/files" in paths
    assert "/knowledge/files/{document_id}" in paths
    assert "/knowledge/files/{document_id}/preview" in paths
    assert "/knowledge/files/reindex-all" in paths
    assert "/knowledge/files/{document_id}/reindex" in paths
    assert "/knowledge/reload" in paths
    assert "/dictionaries" in paths


def test_dictionaries_return_document_structure_items():
    client = TestClient(app)

    response = client.get("/dictionaries?dictionary_code=document_structure")

    assert response.status_code == 200
    data = response.json()
    assert data[0]["dictionary_code"] == "document_structure"
    assert {item["item_code"] for item in data[0]["items"]} == {"qa", "numbered", "text"}


def test_preview_knowledge_file_reads_text_from_registered_document(monkeypatch):
    client = TestClient(app)
    data_path = get_abs_path("data")
    filename = next(name for name in os.listdir(data_path) if name.endswith(".txt"))
    file_path = os.path.join(data_path, filename)

    class FakeKnowledgeStore:
        def get_document(self, document_id: str):
            assert document_id == "doc_test"
            return {
                "document_id": "doc_test",
                "filename": filename,
                "file_path": file_path,
                "file_type": "txt",
                "file_md5": "md5_for_test",
                "file_size": 1024,
                "status": "indexed",
                "version": 1,
                "chunk_count": 3,
                "collection_name": "agent",
                "document_type": "text",
                "split_strategy": "recursive",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "error_message": None,
            }

    monkeypatch.setattr(knowledge_router, "_get_knowledge_store", lambda: FakeKnowledgeStore())

    response = client.get("/knowledge/files/doc_test/preview?max_chars=1000")

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_id"] == "doc_test"
    assert data["preview_type"] == "text"
    assert data["content"].strip()
    assert data["page_count"] is None
