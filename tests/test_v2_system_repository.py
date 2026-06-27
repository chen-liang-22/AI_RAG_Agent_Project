"""V2 系统管理仓储测试。"""

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app_v2.application.auth_service import create_password_hash
from app_v2.infrastructure.repositories import system_repository as system_repository_module
from app_v2.infrastructure.repositories.system_repository import SystemRepository
from domain.entities import (
    BaseOrmModel,
    SystemMenuEntity,
    SystemRoleEntity,
    SystemRoleMenuEntity,
    SystemUserEntity,
)


@contextmanager
def _session_context(factory):
    """模拟项目里的 ORM Session 上下文。"""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _build_system_session_factory():
    """创建系统管理仓储测试用 SQLite SessionFactory。"""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    BaseOrmModel.metadata.create_all(
        engine,
        tables=[
            SystemRoleEntity.__table__,
            SystemMenuEntity.__table__,
            SystemRoleMenuEntity.__table__,
            SystemUserEntity.__table__,
        ],
    )
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _seed_system_rows(factory):
    """准备系统角色、菜单和用户测试数据。"""

    now = datetime(2026, 6, 27, 10, 0, 0)
    with _session_context(factory) as session:
        session.add_all([
            SystemRoleEntity(
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
            SystemRoleEntity(
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
        ])
        session.add_all([
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
        ])
        session.add_all([
            SystemRoleMenuEntity(
                role_menu_id=2033000000000020001,
                role_id=2033000000000000001,
                menu_id=2033000000000010001,
                created_at=now,
            ),
            SystemRoleMenuEntity(
                role_menu_id=2033000000000020002,
                role_id=2033000000000000001,
                menu_id=2033000000000010005,
                created_at=now,
            ),
            SystemRoleMenuEntity(
                role_menu_id=2033000000000020003,
                role_id=2033000000000000001,
                menu_id=2033000000000010007,
                created_at=now,
            ),
            SystemRoleMenuEntity(
                role_menu_id=2033000000000021001,
                role_id=2033000000000000002,
                menu_id=2033000000000010001,
                created_at=now,
            ),
        ])


def test_system_entities_create_sqlite_tables():
    """系统管理新增实体应能创建测试表，并使用整数雪花 ID 字段。"""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    BaseOrmModel.metadata.create_all(
        engine,
        tables=[
            SystemRoleEntity.__table__,
            SystemMenuEntity.__table__,
            SystemRoleMenuEntity.__table__,
            SystemUserEntity.__table__,
        ],
    )

    now = datetime(2026, 6, 27, 10, 0, 0)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    with factory() as session:
        role = SystemRoleEntity(
            role_id=2033000000000000001,
            role_code="admin",
            role_name="管理员",
            status="active",
            sort_order=10,
            built_in=1,
            description="系统内置管理员",
            created_at=now,
            updated_at=now,
        )
        session.add(role)
        session.commit()

    with factory() as session:
        saved = session.get(SystemRoleEntity, 2033000000000000001)
        assert saved is not None
        assert saved.role_code == "admin"


def test_system_repository_lists_role_menus_by_role_code(monkeypatch):
    """仓储应按角色编码查到启用且可见的菜单。"""

    factory = _build_system_session_factory()
    _seed_system_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(system_repository_module, "orm_session_context", fake_orm_session_context)
    repository = SystemRepository()

    admin_menus = repository.list_visible_menus_for_role("admin")
    user_menus = repository.list_visible_menus_for_role("user")

    assert [menu.menu_code for menu in admin_menus] == ["home", "system", "roleManagement"]
    assert [menu.menu_code for menu in user_menus] == ["home"]


def test_system_repository_creates_role_overwrites_menus_and_updates_user(monkeypatch):
    """仓储应支持角色创建、菜单关系覆盖、用户分页和用户状态更新。"""

    factory = _build_system_session_factory()
    _seed_system_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(system_repository_module, "orm_session_context", fake_orm_session_context)
    repository = SystemRepository()

    created_role = repository.create_role(
        role_id=2033000000000000003,
        role_code="trainer",
        role_name="训练管理员",
        status="active",
        sort_order=30,
        description="负责训练资料",
    )
    repository.replace_role_menus(
        role_id=created_role.role_id,
        menu_ids=[2033000000000010001],
        relation_ids=[2033000000000022001],
    )
    checked_menu_ids = repository.list_role_menu_ids(created_role.role_id)
    created_user = repository.create_user(
        user_id="2033000000000030001",
        username="trainer01",
        display_name="训练一号",
        password_hash=create_password_hash("123456"),
        role="trainer",
        status="active",
    )
    users, total = repository.list_users(page=1, page_size=10, keyword="trainer", role="trainer", status="active")
    repository.update_user_status(created_user.user_id, "disabled")
    disabled_user = repository.get_user(created_user.user_id)

    assert created_role.role_id == 2033000000000000003
    assert checked_menu_ids == [2033000000000010001]
    assert total == 1
    assert users[0].username == "trainer01"
    assert disabled_user is not None
    assert disabled_user.status == "disabled"


def test_system_repository_creates_updates_and_lists_all_menus(monkeypatch):
    """仓储应支持菜单新增、修改状态，并能查询包含禁用项的完整菜单。"""

    factory = _build_system_session_factory()
    _seed_system_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(system_repository_module, "orm_session_context", fake_orm_session_context)
    repository = SystemRepository()

    created = repository.create_menu(
        menu_id=2033000000000010008,
        parent_menu_id=2033000000000010005,
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
        metadata_json='{"sub_label": "菜单配置"}',
    )
    updated = repository.update_menu(created.menu_id, menu_name="菜单配置", visible=0, status="disabled")
    all_menus = repository.list_all_menus()
    active_menus = repository.list_all_active_menus()

    assert repository.get_menu_by_code("menuManagement").menu_id == 2033000000000010008
    assert repository.get_menu_by_id(2033000000000010008).menu_name == "菜单配置"
    assert updated is not None
    assert updated.status == "disabled"
    assert [menu.menu_code for menu in all_menus] == ["home", "system", "roleManagement", "menuManagement"]
    assert "menuManagement" not in [menu.menu_code for menu in active_menus]
    assert repository.count_child_menus(2033000000000010005) == 2


def test_system_repository_deletes_user_role_and_menu(monkeypatch):
    """仓储应支持物理删除用户、角色和菜单，并清理角色菜单关系。"""

    factory = _build_system_session_factory()
    _seed_system_rows(factory)

    @contextmanager
    def fake_orm_session_context():
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(system_repository_module, "orm_session_context", fake_orm_session_context)
    repository = SystemRepository()

    created_role = repository.create_role(
        role_id=2033000000000000003,
        role_code="temporary",
        role_name="临时角色",
        status="active",
        sort_order=30,
        description="用于删除",
    )
    repository.replace_role_menus(
        role_id=created_role.role_id,
        menu_ids=[2033000000000010001],
        relation_ids=[2033000000000022001],
    )
    created_user = repository.create_user(
        user_id="2033000000000030002",
        username="delete_user",
        display_name="待删除用户",
        password_hash=create_password_hash("123456"),
        role="temporary",
        status="active",
    )
    created_menu = repository.create_menu(
        menu_id=2033000000000010008,
        parent_menu_id=None,
        menu_code="deleteMenu",
        menu_name="待删除菜单",
        menu_type="page",
        page_key="deleteMenu",
        route_path="/delete-menu",
        component_key="DeleteMenuPage",
        icon="Menu",
        permission_code="system:delete-menu",
        sort_order=99,
        visible=True,
        status="active",
        metadata_json=None,
    )

    user_deleted = repository.delete_user(created_user.user_id)
    role_deleted = repository.delete_role(created_role.role_id)
    menu_deleted = repository.delete_menu(created_menu.menu_id)

    assert user_deleted is True
    assert role_deleted is True
    assert menu_deleted is True
    assert repository.get_user(created_user.user_id) is None
    assert repository.get_role_by_id(created_role.role_id) is None
    assert repository.get_menu_by_id(created_menu.menu_id) is None
    assert repository.list_role_menu_ids(created_role.role_id) == []
