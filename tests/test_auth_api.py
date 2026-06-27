from datetime import datetime, timedelta

from fastapi.testclient import TestClient

import app_v2.api.routes.auth as auth_router
from api.main import app
from app_v2.application.auth_service import (
    AuthService,
    RefreshSession,
    RefreshSessionStore,
    create_password_hash,
    create_refresh_token,
    hash_refresh_token,
    verify_password,
)
from domain.entities import SystemUserEntity


def test_password_hash_verifies_original_password():
    """密码哈希必须能验证原密码，同时拒绝错误密码。"""

    password_hash = create_password_hash("1234qwer")

    assert verify_password("1234qwer", password_hash)
    assert not verify_password("wrong-password", password_hash)
    assert "1234qwer" not in password_hash


def test_access_token_roundtrip_uses_jwt(monkeypatch):
    """access_token 使用 JWT，解析后应能拿到登录用户载荷。"""

    monkeypatch.setenv("AUTH_TOKEN_SECRET", "unit-test-secret")
    service = AuthService()

    token = service.create_access_token(
        {
            "user_id": "user_001",
            "username": "admin",
            "display_name": "系统管理员",
            "role": "admin",
        }
    )
    payload = service.parse_access_token(token)

    assert token.count(".") == 2
    assert payload is not None
    assert payload.user_id == "user_001"
    assert payload.username == "admin"
    assert payload.display_name == "系统管理员"
    assert payload.role == "admin"


def test_expired_access_token_is_rejected(monkeypatch):
    """过期 access_token 不能继续访问当前用户接口。"""

    monkeypatch.setenv("AUTH_TOKEN_SECRET", "unit-test-secret")
    service = AuthService()

    token = service.create_access_token(
        {
            "user_id": "user_001",
            "username": "admin",
            "display_name": "系统管理员",
            "role": "admin",
        },
        expires_at=datetime.utcnow() - timedelta(seconds=1),
    )

    assert service.parse_access_token(token) is None


def test_refresh_token_hash_never_equals_plain_token():
    """后端只能保存 refresh_token 哈希，不能保存明文。"""

    refresh_token = create_refresh_token()
    token_hash = hash_refresh_token(refresh_token)

    assert token_hash != refresh_token
    assert len(token_hash) == 64
    assert hash_refresh_token(refresh_token) == token_hash


def test_openapi_exposes_auth_routes():
    """登录与续签接口需要出现在 OpenAPI 中，方便前端和接口文档发现。"""

    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v2/auth/login" in paths
    assert "/api/v2/auth/refresh" in paths
    assert "/api/v2/auth/me" in paths
    assert "/api/v2/auth/logout" in paths


class FakeAuthRepository:
    """测试用用户仓储，避免接口测试依赖真实 MySQL。"""

    def __init__(self):
        self.user = SystemUserEntity(
            user_id="user_001",
            username="admin",
            display_name="系统管理员",
            password_hash=create_password_hash("1234qwer"),
            role="admin",
            status="active",
            last_login_at=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def ensure_user_table(self) -> None:
        return None

    def get_by_username(self, username: str):
        return self.user if username == self.user.username else None

    def get_by_user_id(self, user_id: str):
        return self.user if user_id == self.user.user_id else None

    def create_user(self, **kwargs):
        return self.user

    def mark_login_success(self, user_id: str) -> None:
        return None


class FakeRedisClient:
    """测试用 Redis 包装，模拟 refresh session 的 TTL 存取。"""

    def __init__(self):
        self.values: dict[str, dict] = {}

    def is_available(self):
        return True

    def build_key(self, *parts: object) -> str:
        return ":".join(str(part) for part in parts)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        self.values[key] = value
        return True

    def get_json(self, key: str, default=None):
        return self.values.get(key, default)

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.values:
                deleted += 1
                del self.values[key]
        return deleted


def test_login_refresh_and_logout_manage_refresh_cookie(monkeypatch):
    """登录写入 refresh Cookie，续签可换新 access token，退出后旧 refresh 失效。"""

    monkeypatch.setenv("AUTH_TOKEN_SECRET", "unit-test-secret")
    monkeypatch.setenv("AUTH_ENABLE_DEFAULT_ADMIN", "false")
    fake_redis = FakeRedisClient()
    service = AuthService(
        repository=FakeAuthRepository(),
        refresh_store=RefreshSessionStore(fake_redis),
    )
    monkeypatch.setattr(auth_router, "_auth_service", service)
    client = TestClient(app)

    login_response = client.post("/api/v2/auth/login", json={"username": "admin", "password": "1234qwer"})

    assert login_response.status_code == 200
    login_data = login_response.json()
    assert login_data["user"]["username"] == "admin"
    assert login_data["access_token"].count(".") == 2
    assert service.refresh_cookie_name in client.cookies
    assert len(fake_redis.values) == 1

    refresh_response = client.post("/api/v2/auth/refresh")

    assert refresh_response.status_code == 200
    assert refresh_response.json()["access_token"].count(".") == 2

    logout_response = client.post(
        "/api/v2/auth/logout",
        headers={"Authorization": f"Bearer {login_data['access_token']}"},
    )

    assert logout_response.status_code == 200
    assert fake_redis.values == {}
    assert client.post("/api/v2/auth/refresh").status_code == 401
