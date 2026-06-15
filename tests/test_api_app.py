from fastapi.testclient import TestClient

from api.main import app


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
    assert "/knowledge/upload/confirm" in paths
    assert "/knowledge/files" in paths
    assert "/knowledge/files/{document_id}" in paths
    assert "/knowledge/files/reindex-all" in paths
    assert "/knowledge/files/{document_id}/reindex" in paths
    assert "/knowledge/reload" in paths
