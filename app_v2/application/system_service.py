"""V2 系统管理应用服务。"""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from app_v2.application.auth_service import create_password_hash
from app_v2.domain.system_schemas import (
    SystemMenuCreateRequest,
    SystemMenuResponse,
    SystemMenuUpdateRequest,
    SystemRoleCreateRequest,
    SystemRoleListResponse,
    SystemRoleMenuResponse,
    SystemRoleMenuUpdateRequest,
    SystemRoleOptionResponse,
    SystemRoleResponse,
    SystemRoleUpdateRequest,
    SystemUserCreateRequest,
    SystemUserListResponse,
    SystemUserResponse,
    SystemUserUpdateRequest,
)
from app_v2.infrastructure.repositories.system_repository import SystemRepository
from app_v2.domain.entities import SystemMenuEntity, SystemUserEntity
from app_v2.infrastructure.id_generator import new_id


class SystemApplicationService:
    """系统管理业务外观。"""

    _VALID_MENU_TYPES = {"directory", "page"}

    def __init__(self, repository: SystemRepository | None = None, id_generator=None):
        self.repository = repository or SystemRepository()
        self.id_generator = id_generator or self._default_id_generator

    def get_current_user_menus(self, *, role_code: str) -> list[SystemMenuResponse]:
        """查询当前角色可见菜单树。"""

        menus = self.repository.list_visible_menus_for_role(role_code)
        return self._build_menu_tree(menus)

    def get_all_menus(self, *, include_disabled: bool = False) -> list[SystemMenuResponse]:
        """查询菜单树，可按场景包含禁用菜单。"""

        menus = self.repository.list_all_menus() if include_disabled else self.repository.list_all_active_menus()
        return self._build_menu_tree(menus)

    def require_admin(self, user: SystemUserEntity) -> None:
        """校验当前用户是否为管理员。"""

        if user.role != "admin":
            raise HTTPException(status_code=403, detail="无权限访问系统管理")

    def require_menu_access(self, user: SystemUserEntity, menu_codes: set[str]) -> None:
        """校验当前用户角色是否拥有指定菜单之一。"""

        if user.role == "admin":
            return
        menus = self.repository.list_visible_menus_for_role(user.role)
        allowed_codes = {menu.menu_code for menu in menus}
        if allowed_codes.isdisjoint(menu_codes):
            raise HTTPException(status_code=403, detail="无权限访问菜单")

    def _default_id_generator(self) -> int:
        """生成新增系统数据主键。

        复用项目已有雪花算法主键生成器，数据库 BIGINT 字段写入整数。
        """

        return int(new_id())

    def create_role(self, request: SystemRoleCreateRequest) -> SystemRoleResponse:
        """创建系统角色，并可同步菜单授权。"""

        self._validate_status(request.status)
        role_code = request.role_code.strip()
        if self.repository.get_role_by_code(role_code) is not None:
            raise HTTPException(status_code=400, detail="角色编码已存在")
        role_id = self.id_generator()
        role = self.repository.create_role(
            role_id=role_id,
            role_code=role_code,
            role_name=request.role_name.strip(),
            status=request.status,
            sort_order=request.sort_order,
            description=request.description,
        )
        if request.menu_ids:
            self.save_role_menus(str(role_id), SystemRoleMenuUpdateRequest(menu_ids=request.menu_ids))
        return self._role_to_response(role)

    def create_menu(self, request: SystemMenuCreateRequest) -> SystemMenuResponse:
        """创建系统菜单。"""

        self._validate_status(request.status)
        self._validate_menu_type(request.menu_type)
        menu_code = request.menu_code.strip()
        if self.repository.get_menu_by_code(menu_code) is not None:
            raise HTTPException(status_code=400, detail="菜单编码已存在")
        parent_menu_id = self._parse_optional_parent_menu_id(request.parent_menu_id)
        self._validate_parent_menu(parent_menu_id)
        metadata_json = self._dump_metadata(request.metadata)
        menu_id = self.id_generator()
        menu = self.repository.create_menu(
            menu_id=menu_id,
            parent_menu_id=parent_menu_id,
            menu_code=menu_code,
            menu_name=request.menu_name.strip(),
            menu_type=request.menu_type,
            page_key=self._clean_optional_text(request.page_key),
            route_path=self._clean_optional_text(request.route_path),
            component_key=self._clean_optional_text(request.component_key),
            icon=self._clean_optional_text(request.icon),
            permission_code=self._clean_optional_text(request.permission_code),
            sort_order=request.sort_order,
            visible=request.visible,
            status=request.status,
            metadata_json=metadata_json,
        )
        return self._menu_to_response(menu)

    def update_menu(self, menu_id: str, request: SystemMenuUpdateRequest) -> SystemMenuResponse:
        """修改系统菜单。"""

        int_menu_id = self._parse_snowflake_id(menu_id, "菜单 ID 格式不正确")
        menu = self.repository.get_menu_by_id(int_menu_id)
        if menu is None:
            raise HTTPException(status_code=404, detail="菜单不存在")
        if request.status is not None:
            self._validate_status(request.status)
        if request.menu_type is not None:
            self._validate_menu_type(request.menu_type)
        next_menu_code = request.menu_code.strip() if request.menu_code is not None else None
        if next_menu_code and next_menu_code != menu.menu_code:
            existing = self.repository.get_menu_by_code(next_menu_code)
            if existing is not None and existing.menu_id != int_menu_id:
                raise HTTPException(status_code=400, detail="菜单编码已存在")
        self._validate_menu_can_be_disabled(int_menu_id, request.status)
        update_fields = self._build_menu_update_fields(request, int_menu_id, next_menu_code)
        updated = self.repository.update_menu(int_menu_id, **update_fields)
        if updated is None:
            raise HTTPException(status_code=404, detail="菜单不存在")
        return self._menu_to_response(updated)

    def disable_menu(self, menu_id: str) -> dict[str, str]:
        """禁用菜单，不物理删除。"""

        int_menu_id = self._parse_snowflake_id(menu_id, "菜单 ID 格式不正确")
        menu = self.repository.get_menu_by_id(int_menu_id)
        if menu is None:
            raise HTTPException(status_code=404, detail="菜单不存在")
        self._validate_menu_can_be_disabled(int_menu_id, "disabled")
        self.repository.update_menu(int_menu_id, status="disabled")
        return {"status": "disabled", "menu_id": str(int_menu_id)}

    def delete_menu(self, menu_id: str) -> dict[str, str]:
        """物理删除菜单。"""

        int_menu_id = self._parse_snowflake_id(menu_id, "菜单 ID 格式不正确")
        menu = self.repository.get_menu_by_id(int_menu_id)
        if menu is None:
            raise HTTPException(status_code=404, detail="菜单不存在")
        if menu.menu_code in {"system", "userManagement", "roleManagement", "menuManagement"}:
            raise HTTPException(status_code=400, detail="核心系统菜单不允许删除")
        if self.repository.count_child_menus(int_menu_id) > 0:
            raise HTTPException(status_code=400, detail="该菜单仍有子菜单，不能删除")
        if not self.repository.delete_menu(int_menu_id):
            raise HTTPException(status_code=404, detail="菜单不存在")
        return {"status": "deleted", "menu_id": str(int_menu_id)}

    def enable_menu(self, menu_id: str) -> dict[str, str]:
        """启用菜单。"""

        int_menu_id = self._parse_snowflake_id(menu_id, "菜单 ID 格式不正确")
        menu = self.repository.update_menu(int_menu_id, status="active")
        if menu is None:
            raise HTTPException(status_code=404, detail="菜单不存在")
        return {"status": "active", "menu_id": str(int_menu_id)}

    def create_user(self, request: SystemUserCreateRequest) -> SystemUserResponse:
        """创建系统用户，user_id 使用字符串化雪花 ID。"""

        self._validate_status(request.status)
        username = request.username.strip()
        if self.repository.get_user_by_username(username) is not None:
            raise HTTPException(status_code=400, detail="登录账号已存在")
        role = self.repository.get_role_by_code(request.role)
        if role is None or role.status != "active":
            raise HTTPException(status_code=400, detail="角色不存在或已停用")
        user_id = str(self.id_generator())
        user = self.repository.create_user(
            user_id=user_id,
            username=username,
            display_name=request.display_name.strip(),
            password_hash=create_password_hash(request.password),
            role=request.role,
            status=request.status,
        )
        return self._user_to_response(user)

    def save_role_menus(self, role_id: str, request: SystemRoleMenuUpdateRequest) -> SystemRoleMenuResponse:
        """整体保存角色菜单权限。"""

        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.get_role_by_id(int_role_id)
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        menu_ids = [self._parse_snowflake_id(menu_id, "菜单 ID 格式不正确") for menu_id in request.menu_ids]
        relation_ids = [self.id_generator() for _ in menu_ids]
        self.repository.replace_role_menus(role_id=int_role_id, menu_ids=menu_ids, relation_ids=relation_ids)
        return self.get_role_menus(role_id)

    def get_role_menus(self, role_id: str) -> SystemRoleMenuResponse:
        """查询角色菜单权限。"""

        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.get_role_by_id(int_role_id)
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        checked_menu_ids = [str(menu_id) for menu_id in self.repository.list_role_menu_ids(int_role_id)]
        return SystemRoleMenuResponse(
            role=self._role_to_response(role),
            checked_menu_ids=checked_menu_ids,
            menu_tree=self._build_menu_tree(self.repository.list_all_active_menus()),
        )

    def list_roles(
        self,
        *,
        page: int,
        page_size: int,
        keyword: str | None,
        status: str | None,
    ) -> SystemRoleListResponse:
        """分页查询角色。"""

        rows, total = self.repository.list_roles(page=page, page_size=page_size, keyword=keyword, status=status)
        return SystemRoleListResponse(
            items=[self._role_to_response(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    def list_role_options(self) -> list[SystemRoleOptionResponse]:
        """查询启用角色选项。"""

        return [
            SystemRoleOptionResponse(role_code=role.role_code, role_name=role.role_name)
            for role in self.repository.list_role_options()
        ]

    def update_role(self, role_id: str, request: SystemRoleUpdateRequest) -> SystemRoleResponse:
        """修改系统角色。"""

        if request.status is not None:
            self._validate_status(request.status)
        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.update_role(
            int_role_id,
            role_name=request.role_name.strip() if request.role_name else None,
            status=request.status,
            sort_order=request.sort_order,
            description=request.description,
        )
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        return self._role_to_response(role)

    def disable_role(self, role_id: str) -> dict[str, str]:
        """禁用角色，不物理删除。"""

        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.get_role_by_id(int_role_id)
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        if role.built_in:
            raise HTTPException(status_code=400, detail="内置角色不允许删除")
        if self.repository.count_active_users_by_role(role.role_code) > 0:
            raise HTTPException(status_code=400, detail="该角色仍有启用用户，不能禁用")
        self.repository.update_role(int_role_id, status="disabled")
        return {"status": "disabled", "role_id": str(int_role_id)}

    def delete_role(self, role_id: str) -> dict[str, str]:
        """物理删除角色。"""

        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.get_role_by_id(int_role_id)
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        if role.built_in:
            raise HTTPException(status_code=400, detail="内置角色不允许删除")
        if self.repository.count_users_by_role(role.role_code) > 0:
            raise HTTPException(status_code=400, detail="该角色仍有用户，不能删除")
        if not self.repository.delete_role(int_role_id):
            raise HTTPException(status_code=404, detail="角色不存在")
        return {"status": "deleted", "role_id": str(int_role_id)}

    def enable_role(self, role_id: str) -> dict[str, str]:
        """启用角色。"""

        int_role_id = self._parse_snowflake_id(role_id, "角色 ID 格式不正确")
        role = self.repository.update_role(int_role_id, status="active")
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在")
        return {"status": "active", "role_id": str(int_role_id)}

    def list_users(
        self,
        *,
        page: int,
        page_size: int,
        keyword: str | None,
        role: str | None,
        status: str | None,
    ) -> SystemUserListResponse:
        """分页查询用户。"""

        rows, total = self.repository.list_users(page=page, page_size=page_size, keyword=keyword, role=role, status=status)
        return SystemUserListResponse(
            items=[self._user_to_response(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_user(
        self,
        user_id: str,
        request: SystemUserUpdateRequest,
        *,
        current_user_id: str,
    ) -> SystemUserResponse:
        """修改系统用户。"""

        if request.status is not None:
            self._validate_status(request.status)
        if request.status == "disabled" and user_id == current_user_id:
            raise HTTPException(status_code=400, detail="不能禁用当前登录用户")
        if request.role is not None:
            role = self.repository.get_role_by_code(request.role)
            if role is None or role.status != "active":
                raise HTTPException(status_code=400, detail="角色不存在或已停用")
        password_hash = create_password_hash(request.password) if request.password else None
        user = self.repository.update_user(
            user_id,
            display_name=request.display_name.strip() if request.display_name else None,
            password_hash=password_hash,
            role=request.role,
            status=request.status,
        )
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        return self._user_to_response(user)

    def reset_user_password(self, user_id: str, password: str) -> dict[str, str]:
        """重置用户密码。"""

        user = self.repository.update_user(user_id, password_hash=create_password_hash(password))
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "ok", "user_id": user_id}

    def disable_user(self, user_id: str, *, current_user_id: str) -> dict[str, str]:
        """禁用用户，不物理删除。"""

        if user_id == current_user_id:
            raise HTTPException(status_code=400, detail="不能禁用当前登录用户")
        user = self.repository.update_user_status(user_id, "disabled")
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "disabled", "user_id": user_id}

    def delete_user(self, user_id: str, *, current_user_id: str) -> dict[str, str]:
        """物理删除用户。"""

        if user_id == current_user_id:
            raise HTTPException(status_code=400, detail="不能删除当前登录用户")
        if self.repository.get_user(user_id) is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        if not self.repository.delete_user(user_id):
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "deleted", "user_id": user_id}

    def enable_user(self, user_id: str) -> dict[str, str]:
        """启用用户。"""

        user = self.repository.update_user_status(user_id, "active")
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"status": "active", "user_id": user_id}

    @classmethod
    def _build_menu_tree(cls, menus: list[SystemMenuEntity]) -> list[SystemMenuResponse]:
        """把数据库平铺菜单组装成树形响应。"""

        responses = [cls._menu_to_response(menu) for menu in menus]
        by_id = {menu.menu_id: menu for menu in responses}
        roots: list[SystemMenuResponse] = []
        for menu in responses:
            if menu.parent_menu_id and menu.parent_menu_id in by_id:
                by_id[menu.parent_menu_id].children.append(menu)
            else:
                roots.append(menu)
        return roots

    @classmethod
    def _menu_to_response(cls, menu: SystemMenuEntity) -> SystemMenuResponse:
        """把菜单实体转换成接口响应。"""

        return SystemMenuResponse(
            menu_id=str(menu.menu_id),
            parent_menu_id=str(menu.parent_menu_id) if menu.parent_menu_id is not None else None,
            menu_code=menu.menu_code,
            menu_name=menu.menu_name,
            menu_type=menu.menu_type,
            page_key=menu.page_key,
            route_path=menu.route_path,
            component_key=menu.component_key,
            icon=menu.icon,
            permission_code=menu.permission_code,
            sort_order=int(menu.sort_order),
            visible=bool(menu.visible),
            status=menu.status,
            metadata=cls._parse_metadata(menu.metadata_json),
        )

    @staticmethod
    def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
        """安全解析菜单扩展 JSON。"""

        if not metadata_json:
            return {}
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, ValueError):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _dump_metadata(metadata: dict | None) -> str | None:
        """把菜单扩展配置序列化为 JSON 字符串。"""

        if not metadata:
            return None
        return json.dumps(metadata, ensure_ascii=False)

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        """清理可空文本字段，空字符串按空值保存。"""

        if value is None:
            return None
        clean_value = value.strip()
        return clean_value or None

    @staticmethod
    def _validate_status(status: str) -> None:
        """校验启用状态枚举。"""

        if status not in {"active", "disabled"}:
            raise HTTPException(status_code=400, detail="状态只能是 active 或 disabled")

    @classmethod
    def _validate_menu_type(cls, menu_type: str) -> None:
        """校验菜单类型枚举。"""

        if menu_type not in cls._VALID_MENU_TYPES:
            raise HTTPException(status_code=400, detail="菜单类型只能是 directory 或 page")

    def _parse_optional_parent_menu_id(self, parent_menu_id: str | None) -> int | None:
        """解析可空父级菜单 ID。"""

        if parent_menu_id is None or parent_menu_id == "":
            return None
        return self._parse_snowflake_id(parent_menu_id, "父级菜单 ID 格式不正确")

    def _validate_parent_menu(self, parent_menu_id: int | None, *, current_menu_id: int | None = None) -> None:
        """校验父级菜单存在且不指向自身。"""

        if parent_menu_id is None:
            return
        if current_menu_id is not None and parent_menu_id == current_menu_id:
            raise HTTPException(status_code=400, detail="父级菜单不能选择自身")
        parent = self.repository.get_menu_by_id(parent_menu_id)
        if parent is None:
            raise HTTPException(status_code=400, detail="父级菜单不存在")

    def _validate_menu_can_be_disabled(self, menu_id: int, status: str | None) -> None:
        """校验菜单是否允许被禁用。"""

        if status != "disabled":
            return
        if self.repository.count_child_menus(menu_id) > 0:
            raise HTTPException(status_code=400, detail="该菜单仍有子菜单，不能禁用")

    def _resolve_update_parent_menu_id(self, request: SystemMenuUpdateRequest, menu_id: int) -> int | None:
        """解析修改菜单时的父级菜单 ID。"""

        parent_menu_id = self._parse_optional_parent_menu_id(request.parent_menu_id)
        self._validate_parent_menu(parent_menu_id, current_menu_id=menu_id)
        return parent_menu_id

    def _build_menu_update_fields(
        self,
        request: SystemMenuUpdateRequest,
        menu_id: int,
        next_menu_code: str | None,
    ) -> dict[str, object]:
        """根据实际传入字段构造菜单更新参数。"""

        update_fields: dict[str, object] = {}
        if "parent_menu_id" in request.model_fields_set:
            update_fields["parent_menu_id"] = self._resolve_update_parent_menu_id(request, menu_id)
        if next_menu_code is not None:
            update_fields["menu_code"] = next_menu_code
        if request.menu_name is not None:
            update_fields["menu_name"] = request.menu_name.strip()
        if request.menu_type is not None:
            update_fields["menu_type"] = request.menu_type
        if "page_key" in request.model_fields_set:
            update_fields["page_key"] = self._clean_optional_text(request.page_key)
        if "route_path" in request.model_fields_set:
            update_fields["route_path"] = self._clean_optional_text(request.route_path)
        if "component_key" in request.model_fields_set:
            update_fields["component_key"] = self._clean_optional_text(request.component_key)
        if "icon" in request.model_fields_set:
            update_fields["icon"] = self._clean_optional_text(request.icon)
        if "permission_code" in request.model_fields_set:
            update_fields["permission_code"] = self._clean_optional_text(request.permission_code)
        if request.sort_order is not None:
            update_fields["sort_order"] = request.sort_order
        if request.visible is not None:
            update_fields["visible"] = 1 if request.visible else 0
        if request.status is not None:
            update_fields["status"] = request.status
        if request.metadata is not None:
            update_fields["metadata_json"] = self._dump_metadata(request.metadata)
        return update_fields

    @staticmethod
    def _parse_snowflake_id(value: str, error_message: str) -> int:
        """把接口字符串雪花 ID 转成数据库整数。"""

        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=error_message) from exc

    @staticmethod
    def _role_to_response(role) -> SystemRoleResponse:
        """把角色实体转换成响应。"""

        return SystemRoleResponse(
            role_id=str(role.role_id),
            role_code=role.role_code,
            role_name=role.role_name,
            status=role.status,
            sort_order=int(role.sort_order),
            built_in=bool(role.built_in),
            description=role.description,
            created_at=str(role.created_at),
            updated_at=str(role.updated_at),
        )

    @staticmethod
    def _user_to_response(user) -> SystemUserResponse:
        """把用户实体转换成响应，排除 password_hash。"""

        return SystemUserResponse(
            user_id=str(user.user_id),
            username=user.username,
            display_name=user.display_name,
            role=user.role,
            status=user.status,
            last_login_at=str(user.last_login_at) if user.last_login_at else None,
            created_at=str(user.created_at),
            updated_at=str(user.updated_at),
        )
