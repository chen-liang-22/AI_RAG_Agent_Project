"""V2 系统管理服务测试。"""

from datetime import datetime

import pytest
from fastapi import HTTPException

from app_v2.application.auth_service import verify_password
from app_v2.application.system_service import SystemApplicationService
from app_v2.domain.system_schemas import (
    SystemMenuCreateRequest,
    SystemMenuResponse,
    SystemMenuUpdateRequest,
    SystemRoleCreateRequest,
    SystemRoleMenuUpdateRequest,
    SystemRoleResponse,
    SystemUserCreateRequest,
)
from domain.entities import SystemMenuEntity, SystemRoleEntity, SystemUserEntity


class FakeMenuRepository:
    """测试用菜单仓储。"""

    def list_visible_menus_for_role(self, role_code: str):
        """返回测试菜单数据。"""

        now = datetime(2026, 6, 27, 10, 0, 0)
        return [
            SystemMenuEntity(
                menu_id=2033000000000010001,
                parent_menu_id=None,
                menu_code="home",
                menu_name="首页",
                menu_type="page",
                page_key="home",
                route_path="/home",
                component_key="HomePage",
                icon="LayoutDashboard",
                permission_code="dashboard:view",
                sort_order=10,
                visible=1,
                status="active",
                metadata_json='{"sub_label": "系统驾驶舱"}',
                created_at=now,
                updated_at=now,
            ),
            SystemMenuEntity(
                menu_id=2033000000000010005,
                parent_menu_id=None,
                menu_code="system",
                menu_name="系统管理",
                menu_type="directory",
                page_key=None,
                route_path=None,
                component_key=None,
                icon="Settings",
                permission_code="system:view",
                sort_order=90,
                visible=1,
                status="active",
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
            SystemMenuEntity(
                menu_id=2033000000000010007,
                parent_menu_id=2033000000000010005,
                menu_code="roleManagement",
                menu_name="角色管理",
                menu_type="page",
                page_key="roleManagement",
                route_path="/system/roles",
                component_key="RoleManagementPage",
                icon="ShieldCheck",
                permission_code="system:role:manage",
                sort_order=20,
                visible=1,
                status="active",
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
        ]


class FakeMenuAccessRepository:
    """测试菜单级权限仓储。"""

    def list_visible_menus_for_role(self, role_code: str):
        """按角色返回不同菜单，用于验证菜单级接口权限。"""

        if role_code != "manager":
            return []
        now = datetime(2026, 6, 27, 10, 0, 0)
        return [
            SystemMenuEntity(
                menu_id=2033000000000010006,
                parent_menu_id=2033000000000010005,
                menu_code="userManagement",
                menu_name="用户管理",
                menu_type="page",
                page_key="userManagement",
                route_path="/system/users",
                component_key="UserManagementPage",
                icon="Users",
                permission_code="system:user:manage",
                sort_order=10,
                visible=1,
                status="active",
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
        ]


def test_system_response_models_transport_snowflake_ids_as_strings():
    """系统响应模型应把雪花 ID 作为字符串传给前端。"""

    menu = SystemMenuResponse(
        menu_id="2033000000000010001",
        parent_menu_id=None,
        menu_code="home",
        menu_name="首页",
        menu_type="page",
        page_key="home",
        route_path="/home",
        component_key="HomePage",
        icon="LayoutDashboard",
        permission_code="dashboard:view",
        sort_order=10,
        visible=True,
        status="active",
    )
    role = SystemRoleResponse(
        role_id="2033000000000000001",
        role_code="admin",
        role_name="管理员",
        status="active",
        sort_order=10,
        built_in=True,
        description="系统内置管理员",
        created_at="2026-06-27 10:00:00",
        updated_at="2026-06-27 10:00:00",
    )

    assert menu.menu_id == "2033000000000010001"
    assert role.role_id == "2033000000000000001"


def test_system_service_builds_menu_tree_and_stringifies_ids():
    """系统服务应把平铺菜单组装成树，并把雪花 ID 转成字符串。"""

    service = SystemApplicationService(repository=FakeMenuRepository())

    tree = service.get_current_user_menus(role_code="admin")

    assert [menu.menu_code for menu in tree] == ["home", "system"]
    assert tree[0].menu_id == "2033000000000010001"
    assert tree[1].children[0].menu_id == "2033000000000010007"
    assert tree[1].children[0].parent_menu_id == "2033000000000010005"
    assert tree[0].metadata == {"sub_label": "系统驾驶舱"}


def test_system_service_requires_menu_access_instead_of_admin_role():
    """系统接口权限应按角色菜单授权判断，而不是只允许 admin。"""

    service = SystemApplicationService(repository=FakeMenuAccessRepository())
    manager = SystemUserEntity(
        user_id="manager_001",
        username="manager",
        display_name="菜单管理员",
        password_hash="hash",
        role="manager",
        status="active",
        last_login_at=None,
        created_at=datetime(2026, 6, 27, 10, 0, 0),
        updated_at=datetime(2026, 6, 27, 10, 0, 0),
    )
    normal_user = SystemUserEntity(
        user_id="user_001",
        username="normal",
        display_name="普通用户",
        password_hash="hash",
        role="user",
        status="active",
        last_login_at=None,
        created_at=datetime(2026, 6, 27, 10, 0, 0),
        updated_at=datetime(2026, 6, 27, 10, 0, 0),
    )

    service.require_menu_access(manager, {"userManagement"})
    with pytest.raises(HTTPException) as permission_exc:
        service.require_menu_access(normal_user, {"userManagement"})

    assert permission_exc.value.status_code == 403
    assert permission_exc.value.detail == "无权限访问菜单"


class FakeSystemRepository:
    """测试用系统仓储。"""

    def __init__(self):
        now = datetime(2026, 6, 27, 10, 0, 0)
        self.roles = {
            2033000000000000001: SystemRoleEntity(
                role_id=2033000000000000001,
                role_code="admin",
                role_name="管理员",
                status="active",
                sort_order=10,
                built_in=1,
                description="系统内置管理员",
                created_at=now,
                updated_at=now,
            ),
            2033000000000000002: SystemRoleEntity(
                role_id=2033000000000000002,
                role_code="user",
                role_name="普通用户",
                status="active",
                sort_order=20,
                built_in=1,
                description="普通用户",
                created_at=now,
                updated_at=now,
            ),
        }
        self.users = {}
        self.role_menu_ids = {}
        self.menus = {
            2033000000000010001: SystemMenuEntity(
                menu_id=2033000000000010001,
                parent_menu_id=None,
                menu_code="home",
                menu_name="首页",
                menu_type="page",
                page_key="home",
                route_path="/home",
                component_key="HomePage",
                icon="LayoutDashboard",
                permission_code="dashboard:view",
                sort_order=10,
                visible=1,
                status="active",
                metadata_json='{"sub_label": "系统驾驶舱"}',
                created_at=now,
                updated_at=now,
            ),
            2033000000000010005: SystemMenuEntity(
                menu_id=2033000000000010005,
                parent_menu_id=None,
                menu_code="system",
                menu_name="系统管理",
                menu_type="directory",
                page_key=None,
                route_path=None,
                component_key=None,
                icon="Settings",
                permission_code="system:view",
                sort_order=90,
                visible=1,
                status="active",
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
            2033000000000010007: SystemMenuEntity(
                menu_id=2033000000000010007,
                parent_menu_id=2033000000000010005,
                menu_code="roleManagement",
                menu_name="角色管理",
                menu_type="page",
                page_key="roleManagement",
                route_path="/system/roles",
                component_key="RoleManagementPage",
                icon="ShieldCheck",
                permission_code="system:role:manage",
                sort_order=20,
                visible=1,
                status="active",
                metadata_json=None,
                created_at=now,
                updated_at=now,
            ),
        }

    def get_role_by_code(self, role_code: str):
        """按角色编码查询角色。"""

        return next((role for role in self.roles.values() if role.role_code == role_code), None)

    def get_role_by_id(self, role_id: int):
        """按角色 ID 查询角色。"""

        return self.roles.get(role_id)

    def create_role(self, **kwargs):
        """创建测试角色。"""

        now = datetime(2026, 6, 27, 10, 0, 0)
        role = SystemRoleEntity(
            role_id=kwargs["role_id"],
            role_code=kwargs["role_code"],
            role_name=kwargs["role_name"],
            status=kwargs["status"],
            sort_order=kwargs["sort_order"],
            built_in=0,
            description=kwargs["description"],
            created_at=now,
            updated_at=now,
        )
        self.roles[role.role_id] = role
        return role

    def get_menu_by_id(self, menu_id: int):
        """按菜单 ID 查询测试菜单。"""

        return self.menus.get(menu_id)

    def get_menu_by_code(self, menu_code: str):
        """按菜单编码查询测试菜单。"""

        return next((menu for menu in self.menus.values() if menu.menu_code == menu_code), None)

    def create_menu(self, **kwargs):
        """创建测试菜单。"""

        now = datetime(2026, 6, 27, 10, 0, 0)
        menu = SystemMenuEntity(
            menu_id=kwargs["menu_id"],
            parent_menu_id=kwargs["parent_menu_id"],
            menu_code=kwargs["menu_code"],
            menu_name=kwargs["menu_name"],
            menu_type=kwargs["menu_type"],
            page_key=kwargs["page_key"],
            route_path=kwargs["route_path"],
            component_key=kwargs["component_key"],
            icon=kwargs["icon"],
            permission_code=kwargs["permission_code"],
            sort_order=kwargs["sort_order"],
            visible=1 if kwargs["visible"] else 0,
            status=kwargs["status"],
            metadata_json=kwargs["metadata_json"],
            created_at=now,
            updated_at=now,
        )
        self.menus[menu.menu_id] = menu
        return menu

    def update_menu(self, menu_id: int, **kwargs):
        """修改测试菜单。"""

        menu = self.menus.get(menu_id)
        if menu is None:
            return None
        for field_name, field_value in kwargs.items():
            if field_value is not None:
                setattr(menu, field_name, field_value)
        return menu

    def count_child_menus(self, parent_menu_id: int):
        """统计测试子菜单数量。"""

        return sum(1 for menu in self.menus.values() if menu.parent_menu_id == parent_menu_id)

    def count_users_by_role(self, role_code: str):
        """统计测试角色下全部用户数量。"""

        return sum(1 for user in self.users.values() if user.role == role_code)

    def delete_menu(self, menu_id: int):
        """删除测试菜单。"""

        return self.menus.pop(menu_id, None) is not None

    def list_all_active_menus(self):
        """返回可授权菜单。"""

        return [
            menu
            for menu in self.menus.values()
            if menu.status == "active"
        ]

    def list_all_menus(self):
        """返回全部测试菜单。"""

        return list(self.menus.values())

    def replace_role_menus(self, *, role_id: int, menu_ids: list[int], relation_ids: list[int]):
        """覆盖测试角色菜单。"""

        self.role_menu_ids[role_id] = menu_ids

    def list_role_menu_ids(self, role_id: int):
        """查询测试角色菜单。"""

        return self.role_menu_ids.get(role_id, [])

    def get_user_by_username(self, username: str):
        """按账号查询测试用户。"""

        return next((user for user in self.users.values() if user.username == username), None)

    def create_user(self, **kwargs):
        """创建测试用户。"""

        now = datetime(2026, 6, 27, 10, 0, 0)
        user = SystemUserEntity(
            user_id=kwargs["user_id"],
            username=kwargs["username"],
            display_name=kwargs["display_name"],
            password_hash=kwargs["password_hash"],
            role=kwargs["role"],
            status=kwargs["status"],
            last_login_at=None,
            created_at=now,
            updated_at=now,
        )
        self.users[user.user_id] = user
        return user

    def get_user(self, user_id: str):
        """按用户 ID 查询测试用户。"""

        return self.users.get(user_id)

    def delete_user(self, user_id: str):
        """删除测试用户。"""

        return self.users.pop(user_id, None) is not None

    def delete_role(self, role_id: int):
        """删除测试角色。"""

        deleted = self.roles.pop(role_id, None)
        self.role_menu_ids.pop(role_id, None)
        return deleted is not None

    def update_user_status(self, user_id: str, status: str):
        """修改测试用户状态。"""

        user = self.users.get(user_id)
        if user is None:
            return None
        user.status = status
        return user


def test_system_service_creates_role_and_user_with_snowflake_ids():
    """创建角色和用户时应使用雪花 ID，并把 ID 作为字符串返回。"""

    repository = FakeSystemRepository()
    service = SystemApplicationService(
        repository=repository,
        id_generator=lambda: 2033000000000090001,
    )

    role = service.create_role(SystemRoleCreateRequest(
        role_code="trainer",
        role_name="训练管理员",
        status="active",
        sort_order=30,
        description="负责训练资料",
        menu_ids=["2033000000000010001"],
    ))
    user = service.create_user(SystemUserCreateRequest(
        username="trainer01",
        display_name="训练一号",
        password="123456",
        role="trainer",
        status="active",
    ))

    assert role.role_id == "2033000000000090001"
    assert repository.role_menu_ids[2033000000000090001] == [2033000000000010001]
    assert user.user_id == "2033000000000090001"
    assert verify_password("123456", repository.users[user.user_id].password_hash)


def test_system_service_creates_updates_disables_and_enables_menu():
    """系统服务应支持菜单新增、修改、禁用和启用。"""

    repository = FakeSystemRepository()
    service = SystemApplicationService(
        repository=repository,
        id_generator=lambda: 2033000000000090008,
    )

    created = service.create_menu(SystemMenuCreateRequest(
        parent_menu_id="2033000000000010005",
        menu_code="menuManagement",
        menu_name="菜单管理",
        menu_type="page",
        page_key="menuManagement",
        route_path="/system/menus",
        component_key="MenuManagementPage",
        icon="Menu",
        permission_code="system:menu:manage",
        sort_order=30,
        visible=True,
        status="active",
        metadata={"sub_label": "菜单配置"},
    ))
    updated = service.update_menu(created.menu_id, SystemMenuUpdateRequest(
        menu_name="菜单配置",
        sort_order=35,
        visible=False,
        metadata={"sub_label": "配置入口"},
    ))
    disabled = service.disable_menu(created.menu_id)
    enabled = service.enable_menu(created.menu_id)

    assert created.menu_id == "2033000000000090008"
    assert created.parent_menu_id == "2033000000000010005"
    assert created.metadata == {"sub_label": "菜单配置"}
    assert updated.menu_name == "菜单配置"
    assert updated.visible is False
    assert updated.metadata == {"sub_label": "配置入口"}
    assert disabled == {"status": "disabled", "menu_id": "2033000000000090008"}
    assert enabled == {"status": "active", "menu_id": "2033000000000090008"}


def test_system_service_rejects_duplicate_menu_code_and_parent_disable():
    """系统服务应拒绝重复菜单编码，并阻止直接禁用仍有子菜单的目录。"""

    repository = FakeSystemRepository()
    service = SystemApplicationService(repository=repository)

    with pytest.raises(HTTPException) as duplicate_exc:
        service.create_menu(SystemMenuCreateRequest(
            menu_code="roleManagement",
            menu_name="重复角色菜单",
            menu_type="page",
            status="active",
        ))
    with pytest.raises(HTTPException) as disable_exc:
        service.disable_menu("2033000000000010005")

    assert duplicate_exc.value.status_code == 400
    assert duplicate_exc.value.detail == "菜单编码已存在"
    assert disable_exc.value.status_code == 400
    assert disable_exc.value.detail == "该菜单仍有子菜单，不能禁用"


def test_system_service_deletes_user_role_and_menu_with_guards():
    """系统服务应支持物理删除，并保护当前用户、内置角色和有子菜单的菜单。"""

    repository = FakeSystemRepository()
    service = SystemApplicationService(
        repository=repository,
        id_generator=lambda: 2033000000000090009,
    )
    custom_role = service.create_role(SystemRoleCreateRequest(
        role_code="deleter",
        role_name="可删除角色",
        status="active",
        sort_order=40,
        description="测试删除",
        menu_ids=[],
    ))
    custom_user = service.create_user(SystemUserCreateRequest(
        username="delete_user",
        display_name="待删除用户",
        password="123456",
        role="deleter",
        status="active",
    ))
    custom_menu = service.create_menu(SystemMenuCreateRequest(
        menu_code="deleteMenu",
        menu_name="待删除菜单",
        menu_type="page",
        status="active",
    ))

    user_deleted = service.delete_user(custom_user.user_id, current_user_id="other_user")
    role_deleted = service.delete_role(custom_role.role_id)
    menu_deleted = service.delete_menu(custom_menu.menu_id)

    assert user_deleted == {"status": "deleted", "user_id": custom_user.user_id}
    assert role_deleted == {"status": "deleted", "role_id": custom_role.role_id}
    assert menu_deleted == {"status": "deleted", "menu_id": custom_menu.menu_id}
    assert repository.get_user(custom_user.user_id) is None
    assert repository.get_role_by_id(int(custom_role.role_id)) is None
    assert repository.get_menu_by_id(int(custom_menu.menu_id)) is None


def test_system_service_rejects_risky_physical_delete():
    """系统服务应拒绝危险的物理删除操作。"""

    repository = FakeSystemRepository()
    service = SystemApplicationService(repository=repository)

    with pytest.raises(HTTPException) as self_delete_exc:
        service.delete_user("current_user", current_user_id="current_user")
    with pytest.raises(HTTPException) as built_in_role_exc:
        service.delete_role("2033000000000000001")
    with pytest.raises(HTTPException) as parent_menu_exc:
        service.delete_menu("2033000000000010005")

    assert self_delete_exc.value.status_code == 400
    assert self_delete_exc.value.detail == "不能删除当前登录用户"
    assert built_in_role_exc.value.status_code == 400
    assert built_in_role_exc.value.detail == "内置角色不允许删除"
    assert parent_menu_exc.value.status_code == 400
    assert parent_menu_exc.value.detail == "核心系统菜单不允许删除"


def test_system_service_rejects_non_admin_and_self_disable():
    """系统管理接口必须拒绝非管理员，并阻止禁用当前用户。"""

    service = SystemApplicationService(repository=FakeSystemRepository())
    normal_user = SystemUserEntity(
        user_id="user_001",
        username="normal",
        display_name="普通用户",
        password_hash="hash",
        role="user",
        status="active",
        last_login_at=None,
        created_at=datetime(2026, 6, 27, 10, 0, 0),
        updated_at=datetime(2026, 6, 27, 10, 0, 0),
    )

    with pytest.raises(HTTPException) as permission_exc:
        service.require_admin(normal_user)
    with pytest.raises(HTTPException) as disable_exc:
        service.disable_user("user_001", current_user_id="user_001")

    assert permission_exc.value.status_code == 403
    assert permission_exc.value.detail == "无权限访问系统管理"
    assert disable_exc.value.status_code == 400
    assert disable_exc.value.detail == "不能禁用当前登录用户"
