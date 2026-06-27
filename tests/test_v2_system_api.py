"""V2 系统管理接口测试。"""

from datetime import datetime

from fastapi.testclient import TestClient

import app_v2.api.routes.auth as auth_router
import app_v2.api.routes.system as system_router
from api.main import app
from app_v2.application.auth_service import AuthService, RefreshSessionStore, create_password_hash
from app_v2.domain.system_schemas import SystemUserListResponse
from domain.entities import SystemUserEntity


class FakeAuthRepository:
    """测试用用户仓储。"""

    def __init__(self, role: str = "admin"):
        self.user = SystemUserEntity(
            user_id="user_001",
            username="admin",
            display_name="系统管理员",
            password_hash=create_password_hash("1234qwer"),
            role=role,
            status="active",
            last_login_at=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def ensure_user_table(self) -> None:
        """测试中不创建真实用户表。"""

        return None

    def get_by_username(self, username: str):
        """按账号返回测试用户。"""

        return self.user if username == self.user.username else None

    def get_by_user_id(self, user_id: str):
        """按用户 ID 返回测试用户。"""

        return self.user if user_id == self.user.user_id else None

    def create_user(self, **kwargs):
        """默认管理员测试中不需要真实创建。"""

        return self.user

    def mark_login_success(self, user_id: str) -> None:
        """测试中不记录登录时间。"""

        return None


class FakeRedisClient:
    """测试用 Redis。"""

    def __init__(self):
        self.values = {}

    def is_available(self):
        """模拟 Redis 可用。"""

        return True

    def build_key(self, *parts: object) -> str:
        """生成测试 Redis 键。"""

        return ":".join(str(part) for part in parts)

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        """写入测试 JSON。"""

        self.values[key] = value
        return True

    def get_json(self, key: str, default=None):
        """读取测试 JSON。"""

        return self.values.get(key, default)

    def delete(self, *keys: str) -> int:
        """删除测试 Redis 键。"""

        for key in keys:
            self.values.pop(key, None)
        return len(keys)


def _client_with_role(monkeypatch, role: str):
    """创建指定角色登录后的 TestClient。"""

    monkeypatch.setenv("AUTH_TOKEN_SECRET", "unit-test-secret")
    monkeypatch.setenv("AUTH_ENABLE_DEFAULT_ADMIN", "false")
    service = AuthService(
        repository=FakeAuthRepository(role=role),
        refresh_store=RefreshSessionStore(FakeRedisClient()),
    )
    monkeypatch.setattr(auth_router, "_auth_service", service)
    client = TestClient(app)
    login_response = client.post("/api/v2/auth/login", json={"username": "admin", "password": "1234qwer"})
    token = login_response.json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def test_openapi_exposes_system_routes():
    """系统管理接口需要出现在 OpenAPI 中。"""

    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v2/system/menus/me" in paths
    assert "/api/v2/system/users" in paths
    assert "/api/v2/system/roles" in paths
    assert "/api/v2/system/users/{user_id}" in paths
    assert "/api/v2/system/users/{user_id}/password" in paths
    assert "/api/v2/system/users/{user_id}/delete" in paths
    assert "/api/v2/system/users/{user_id}/enable" in paths
    assert "/api/v2/system/roles/{role_id}" in paths
    assert "/api/v2/system/roles/{role_id}/menus" in paths
    assert "/api/v2/system/roles/{role_id}/delete" in paths
    assert "/api/v2/system/roles/{role_id}/enable" in paths
    assert "/api/v2/system/menus/{menu_id}" in paths
    assert "/api/v2/system/menus/{menu_id}/delete" in paths
    assert "/api/v2/system/menus/{menu_id}/enable" in paths


class FakeSystemService:
    """测试用系统服务。"""

    def __init__(self, allowed: bool):
        self.allowed = allowed

    def require_menu_access(self, user, menu_codes: set[str]) -> None:
        """模拟菜单级接口权限。"""

        if not self.allowed:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="无权限访问菜单")

    def list_users(self, *, page: int, page_size: int, keyword: str | None, role: str | None, status: str | None):
        """返回测试用户分页。"""

        return SystemUserListResponse(items=[], total=0, page=page, page_size=page_size)


def test_system_routes_allow_non_admin_when_role_has_menu(monkeypatch):
    """非 admin 角色只要有对应菜单权限，也应能调用该页面接口。"""

    client, headers = _client_with_role(monkeypatch, "manager")
    monkeypatch.setattr(system_router, "_service", lambda: FakeSystemService(allowed=True))

    response = client.get("/api/v2/system/users", headers=headers)

    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_system_routes_reject_when_role_has_no_menu(monkeypatch):
    """没有对应菜单权限时，系统接口应返回 403。"""

    client, headers = _client_with_role(monkeypatch, "user")
    monkeypatch.setattr(system_router, "_service", lambda: FakeSystemService(allowed=False))

    response = client.get("/api/v2/system/users", headers=headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "无权限访问菜单"
