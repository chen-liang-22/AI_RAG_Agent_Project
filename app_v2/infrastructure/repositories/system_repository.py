"""系统管理仓储。"""

from __future__ import annotations

from sqlalchemy import case, func, or_, select

from app_v2.shared.pagination import escape_like_keyword, normalize_page
from app_v2.domain.entities import SystemMenuEntity, SystemRoleEntity, SystemRoleMenuEntity, SystemUserEntity
from app_v2.infrastructure.orm_session import orm_session_context
from app_v2.application.training_support.repository import utc_now

_UNSET = object()


class SystemRepository:
    """封装系统菜单、角色和用户的数据访问。"""

    def list_visible_menus_for_role(self, role_code: str) -> list[SystemMenuEntity]:
        """按角色编码查询当前可见菜单。"""

        statement = (
            select(SystemMenuEntity)
            .join(SystemRoleMenuEntity, SystemRoleMenuEntity.menu_id == SystemMenuEntity.menu_id)
            .join(SystemRoleEntity, SystemRoleEntity.role_id == SystemRoleMenuEntity.role_id)
            .where(
                SystemRoleEntity.role_code == role_code,
                SystemRoleEntity.status == "active",
                SystemMenuEntity.status == "active",
                SystemMenuEntity.visible == 1,
            )
            .order_by(
                case((SystemMenuEntity.parent_menu_id.is_(None), 0), else_=1).asc(),
                SystemMenuEntity.sort_order.asc(),
                SystemMenuEntity.created_at.asc(),
            )
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def list_all_active_menus(self) -> list[SystemMenuEntity]:
        """查询全部启用菜单，用于角色授权树。"""

        statement = (
            select(SystemMenuEntity)
            .where(SystemMenuEntity.status == "active")
            .order_by(
                case((SystemMenuEntity.parent_menu_id.is_(None), 0), else_=1).asc(),
                SystemMenuEntity.sort_order.asc(),
                SystemMenuEntity.created_at.asc(),
            )
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def list_all_menus(self) -> list[SystemMenuEntity]:
        """查询全部菜单，包含禁用和隐藏项，用于菜单管理。"""

        statement = (
            select(SystemMenuEntity)
            .order_by(
                case((SystemMenuEntity.parent_menu_id.is_(None), 0), else_=1).asc(),
                SystemMenuEntity.sort_order.asc(),
                SystemMenuEntity.created_at.asc(),
            )
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def get_menu_by_id(self, menu_id: int) -> SystemMenuEntity | None:
        """按菜单 ID 查询菜单。"""

        with orm_session_context() as session:
            return session.get(SystemMenuEntity, menu_id)

    def get_menu_by_code(self, menu_code: str) -> SystemMenuEntity | None:
        """按菜单编码查询菜单。"""

        statement = select(SystemMenuEntity).where(SystemMenuEntity.menu_code == menu_code)
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def create_menu(
        self,
        *,
        menu_id: int,
        parent_menu_id: int | None,
        menu_code: str,
        menu_name: str,
        menu_type: str,
        page_key: str | None,
        route_path: str | None,
        component_key: str | None,
        icon: str | None,
        permission_code: str | None,
        sort_order: int,
        visible: bool,
        status: str,
        metadata_json: str | None,
    ) -> SystemMenuEntity:
        """创建系统菜单。"""

        now = utc_now()
        menu = SystemMenuEntity(
            menu_id=menu_id,
            parent_menu_id=parent_menu_id,
            menu_code=menu_code,
            menu_name=menu_name,
            menu_type=menu_type,
            page_key=page_key,
            route_path=route_path,
            component_key=component_key,
            icon=icon,
            permission_code=permission_code,
            sort_order=sort_order,
            visible=1 if visible else 0,
            status=status,
            metadata_json=metadata_json,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(menu)
        created = self.get_menu_by_id(menu_id)
        if created is None:
            raise RuntimeError(f"系统菜单创建失败：{menu_code}")
        return created

    def update_menu(
        self,
        menu_id: int,
        *,
        parent_menu_id: int | None | object = _UNSET,
        menu_code: str | object = _UNSET,
        menu_name: str | object = _UNSET,
        menu_type: str | object = _UNSET,
        page_key: str | None | object = _UNSET,
        route_path: str | None | object = _UNSET,
        component_key: str | None | object = _UNSET,
        icon: str | None | object = _UNSET,
        permission_code: str | None | object = _UNSET,
        sort_order: int | object = _UNSET,
        visible: int | object = _UNSET,
        status: str | object = _UNSET,
        metadata_json: str | None | object = _UNSET,
    ) -> SystemMenuEntity | None:
        """修改系统菜单。"""

        with orm_session_context() as session:
            menu = session.get(SystemMenuEntity, menu_id)
            if menu is None:
                return None
            if parent_menu_id is not _UNSET:
                menu.parent_menu_id = parent_menu_id
            if menu_code is not _UNSET:
                menu.menu_code = menu_code
            if menu_name is not _UNSET:
                menu.menu_name = menu_name
            if menu_type is not _UNSET:
                menu.menu_type = menu_type
            if page_key is not _UNSET:
                menu.page_key = page_key
            if route_path is not _UNSET:
                menu.route_path = route_path
            if component_key is not _UNSET:
                menu.component_key = component_key
            if icon is not _UNSET:
                menu.icon = icon
            if permission_code is not _UNSET:
                menu.permission_code = permission_code
            if sort_order is not _UNSET:
                menu.sort_order = sort_order
            if visible is not _UNSET:
                menu.visible = visible
            if status is not _UNSET:
                menu.status = status
            if metadata_json is not _UNSET:
                menu.metadata_json = metadata_json
            menu.updated_at = utc_now()
            return menu

    def count_child_menus(self, parent_menu_id: int) -> int:
        """统计某菜单下的直接子菜单数量。"""

        statement = (
            select(func.count())
            .select_from(SystemMenuEntity)
            .where(SystemMenuEntity.parent_menu_id == parent_menu_id)
        )
        with orm_session_context() as session:
            return int(session.scalar(statement) or 0)

    def delete_menu(self, menu_id: int) -> bool:
        """物理删除系统菜单。"""

        with orm_session_context() as session:
            menu = session.get(SystemMenuEntity, menu_id)
            if menu is None:
                return False
            existing = session.scalars(
                select(SystemRoleMenuEntity).where(SystemRoleMenuEntity.menu_id == menu_id)
            ).all()
            for row in existing:
                session.delete(row)
            session.delete(menu)
            return True

    def get_role_by_id(self, role_id: int) -> SystemRoleEntity | None:
        """按角色 ID 查询角色。"""

        with orm_session_context() as session:
            return session.get(SystemRoleEntity, role_id)

    def get_role_by_code(self, role_code: str) -> SystemRoleEntity | None:
        """按角色编码查询角色。"""

        statement = select(SystemRoleEntity).where(SystemRoleEntity.role_code == role_code)
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def list_roles(
        self,
        *,
        page: int,
        page_size: int,
        keyword: str | None = None,
        status: str | None = None,
    ) -> tuple[list[SystemRoleEntity], int]:
        """分页查询角色。"""

        page_request = normalize_page(page, page_size, max_page_size=50)
        conditions = []
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{escape_like_keyword(clean_keyword)}%"
            conditions.append(or_(
                SystemRoleEntity.role_code.like(like_keyword, escape="\\"),
                SystemRoleEntity.role_name.like(like_keyword, escape="\\"),
            ))
        if status:
            conditions.append(SystemRoleEntity.status == status)
        count_statement = select(func.count()).select_from(SystemRoleEntity).where(*conditions)
        list_statement = (
            select(SystemRoleEntity)
            .where(*conditions)
            .order_by(SystemRoleEntity.sort_order.asc(), SystemRoleEntity.created_at.asc())
            .limit(page_request.page_size)
            .offset(page_request.offset)
        )
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = list(session.scalars(list_statement).all())
        return rows, total

    def list_role_options(self) -> list[SystemRoleEntity]:
        """查询启用角色选项。"""

        statement = (
            select(SystemRoleEntity)
            .where(SystemRoleEntity.status == "active")
            .order_by(SystemRoleEntity.sort_order.asc(), SystemRoleEntity.created_at.asc())
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def create_role(
        self,
        *,
        role_id: int,
        role_code: str,
        role_name: str,
        status: str,
        sort_order: int,
        description: str | None,
    ) -> SystemRoleEntity:
        """创建角色。"""

        now = utc_now()
        role = SystemRoleEntity(
            role_id=role_id,
            role_code=role_code,
            role_name=role_name,
            status=status,
            sort_order=sort_order,
            built_in=0,
            description=description,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(role)
        created = self.get_role_by_id(role_id)
        if created is None:
            raise RuntimeError(f"系统角色创建失败：{role_code}")
        return created

    def update_role(
        self,
        role_id: int,
        *,
        role_name: str | None = None,
        status: str | None = None,
        sort_order: int | None = None,
        description: str | None = None,
    ) -> SystemRoleEntity | None:
        """修改角色基础信息。"""

        with orm_session_context() as session:
            role = session.get(SystemRoleEntity, role_id)
            if role is None:
                return None
            if role_name is not None:
                role.role_name = role_name
            if status is not None:
                role.status = status
            if sort_order is not None:
                role.sort_order = sort_order
            if description is not None:
                role.description = description
            role.updated_at = utc_now()
            return role

    def list_role_menu_ids(self, role_id: int) -> list[int]:
        """查询角色已授权菜单 ID。"""

        statement = (
            select(SystemRoleMenuEntity.menu_id)
            .where(SystemRoleMenuEntity.role_id == role_id)
            .order_by(SystemRoleMenuEntity.menu_id.asc())
        )
        with orm_session_context() as session:
            return [int(menu_id) for menu_id in session.scalars(statement).all()]

    def replace_role_menus(self, *, role_id: int, menu_ids: list[int], relation_ids: list[int]) -> None:
        """整体覆盖角色菜单关系。"""

        if len(menu_ids) != len(relation_ids):
            raise ValueError("菜单 ID 和关系 ID 数量不一致")
        now = utc_now()
        with orm_session_context() as session:
            existing = session.scalars(
                select(SystemRoleMenuEntity).where(SystemRoleMenuEntity.role_id == role_id)
            ).all()
            for row in existing:
                session.delete(row)
            for relation_id, menu_id in zip(relation_ids, menu_ids):
                session.add(SystemRoleMenuEntity(
                    role_menu_id=relation_id,
                    role_id=role_id,
                    menu_id=menu_id,
                    created_at=now,
                ))

    def delete_role(self, role_id: int) -> bool:
        """物理删除角色，并清理角色菜单关系。"""

        with orm_session_context() as session:
            role = session.get(SystemRoleEntity, role_id)
            if role is None:
                return False
            existing = session.scalars(
                select(SystemRoleMenuEntity).where(SystemRoleMenuEntity.role_id == role_id)
            ).all()
            for row in existing:
                session.delete(row)
            session.delete(role)
            return True

    def count_active_users_by_role(self, role_code: str) -> int:
        """统计某角色下仍启用的用户数量。"""

        statement = (
            select(func.count())
            .select_from(SystemUserEntity)
            .where(SystemUserEntity.role == role_code, SystemUserEntity.status == "active")
        )
        with orm_session_context() as session:
            return int(session.scalar(statement) or 0)

    def count_users_by_role(self, role_code: str) -> int:
        """统计某角色下全部用户数量。"""

        statement = (
            select(func.count())
            .select_from(SystemUserEntity)
            .where(SystemUserEntity.role == role_code)
        )
        with orm_session_context() as session:
            return int(session.scalar(statement) or 0)

    def list_users(
        self,
        *,
        page: int,
        page_size: int,
        keyword: str | None = None,
        role: str | None = None,
        status: str | None = None,
    ) -> tuple[list[SystemUserEntity], int]:
        """分页查询系统用户。"""

        page_request = normalize_page(page, page_size, max_page_size=50)
        conditions = []
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{escape_like_keyword(clean_keyword)}%"
            conditions.append(or_(
                SystemUserEntity.username.like(like_keyword, escape="\\"),
                SystemUserEntity.display_name.like(like_keyword, escape="\\"),
            ))
        if role:
            conditions.append(SystemUserEntity.role == role)
        if status:
            conditions.append(SystemUserEntity.status == status)
        count_statement = select(func.count()).select_from(SystemUserEntity).where(*conditions)
        list_statement = (
            select(SystemUserEntity)
            .where(*conditions)
            .order_by(SystemUserEntity.created_at.desc())
            .limit(page_request.page_size)
            .offset(page_request.offset)
        )
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = list(session.scalars(list_statement).all())
        return rows, total

    def get_user(self, user_id: str) -> SystemUserEntity | None:
        """按用户 ID 查询用户。"""

        with orm_session_context() as session:
            return session.get(SystemUserEntity, user_id)

    def get_user_by_username(self, username: str) -> SystemUserEntity | None:
        """按账号查询用户。"""

        statement = select(SystemUserEntity).where(SystemUserEntity.username == username)
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def create_user(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        status: str,
    ) -> SystemUserEntity:
        """创建系统用户。"""

        now = utc_now()
        user = SystemUserEntity(
            user_id=user_id,
            username=username,
            display_name=display_name,
            password_hash=password_hash,
            role=role,
            status=status,
            last_login_at=None,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(user)
        created = self.get_user(user_id)
        if created is None:
            raise RuntimeError(f"系统用户创建失败：{username}")
        return created

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        password_hash: str | None = None,
        role: str | None = None,
        status: str | None = None,
    ) -> SystemUserEntity | None:
        """修改系统用户。"""

        with orm_session_context() as session:
            user = session.get(SystemUserEntity, user_id)
            if user is None:
                return None
            if display_name is not None:
                user.display_name = display_name
            if password_hash is not None:
                user.password_hash = password_hash
            if role is not None:
                user.role = role
            if status is not None:
                user.status = status
            user.updated_at = utc_now()
            return user

    def update_user_status(self, user_id: str, status: str) -> SystemUserEntity | None:
        """修改用户状态。"""

        return self.update_user(user_id, status=status)

    def delete_user(self, user_id: str) -> bool:
        """物理删除系统用户。"""

        with orm_session_context() as session:
            user = session.get(SystemUserEntity, user_id)
            if user is None:
                return False
            session.delete(user)
            return True
