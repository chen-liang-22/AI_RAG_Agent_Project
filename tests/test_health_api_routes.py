"""健康检查 API 路由测试。"""

from fastapi.testclient import TestClient

from api.main import app


def test_openapi_exposes_health_dependencies_route():
    """OpenAPI 应暴露依赖健康明细接口。"""

    client = TestClient(app)
    schema = client.get("/openapi.json").json()

    assert "/api/v2/health/dependencies" in schema["paths"]
