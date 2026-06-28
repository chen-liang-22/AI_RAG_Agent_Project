"""V2 系统管理接口。"""

from fastapi import APIRouter, Depends, Query

from app.api.routes.auth import get_current_user
from app.application.system_service import SystemApplicationService
from app.domain.system_schemas import (
    StatusChangeResponse,
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
    SystemUserPasswordResetRequest,
    SystemUserResponse,
    SystemUserUpdateRequest,
)
from app.domain.entities import SystemUserEntity

router = APIRouter(prefix="/system", tags=["V2 系统管理"])


def _service() -> SystemApplicationService:
    """创建系统管理应用服务。"""

    return SystemApplicationService()


@router.get("/menus/me", response_model=list[SystemMenuResponse])
def list_current_user_menus(user: SystemUserEntity = Depends(get_current_user)) -> list[SystemMenuResponse]:
    """查询当前登录用户可见菜单。"""

    return _service().get_current_user_menus(role_code=user.role)


@router.get("/menus", response_model=list[SystemMenuResponse])
def list_all_menus(
    include_disabled: bool = Query(default=False),
    user: SystemUserEntity = Depends(get_current_user),
) -> list[SystemMenuResponse]:
    """查询全部菜单树，用于角色授权和菜单管理。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return service.get_all_menus(include_disabled=include_disabled)


@router.post("/menus", response_model=SystemMenuResponse)
def create_menu(
    request: SystemMenuCreateRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemMenuResponse:
    """创建系统菜单。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return service.create_menu(request)


@router.put("/menus/{menu_id}", response_model=SystemMenuResponse)
def update_menu(
    menu_id: str,
    request: SystemMenuUpdateRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemMenuResponse:
    """修改系统菜单。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return service.update_menu(menu_id, request)


@router.delete("/menus/{menu_id}", response_model=StatusChangeResponse)
def disable_menu(menu_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """禁用系统菜单。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return StatusChangeResponse(**service.disable_menu(menu_id))


@router.delete("/menus/{menu_id}/delete", response_model=StatusChangeResponse)
def delete_menu(menu_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """物理删除系统菜单。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return StatusChangeResponse(**service.delete_menu(menu_id))


@router.post("/menus/{menu_id}/enable", response_model=StatusChangeResponse)
def enable_menu(menu_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """启用系统菜单。"""

    service = _service()
    service.require_menu_access(user, {"menuManagement"})
    return StatusChangeResponse(**service.enable_menu(menu_id))


@router.get("/roles", response_model=SystemRoleListResponse)
def list_roles(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    keyword: str | None = Query(default=None),
    status: str | None = Query(default=None),
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemRoleListResponse:
    """分页查询角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return service.list_roles(page=page, page_size=page_size, keyword=keyword, status=status)


@router.get("/roles/options", response_model=list[SystemRoleOptionResponse])
def list_role_options(user: SystemUserEntity = Depends(get_current_user)) -> list[SystemRoleOptionResponse]:
    """查询启用角色选项。"""

    service = _service()
    service.require_menu_access(user, {"userManagement", "roleManagement"})
    return service.list_role_options()


@router.post("/roles", response_model=SystemRoleResponse)
def create_role(request: SystemRoleCreateRequest, user: SystemUserEntity = Depends(get_current_user)) -> SystemRoleResponse:
    """创建系统角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return service.create_role(request)


@router.put("/roles/{role_id}", response_model=SystemRoleResponse)
def update_role(
    role_id: str,
    request: SystemRoleUpdateRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemRoleResponse:
    """修改系统角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return service.update_role(role_id, request)


@router.get("/roles/{role_id}/menus", response_model=SystemRoleMenuResponse)
def get_role_menus(role_id: str, user: SystemUserEntity = Depends(get_current_user)) -> SystemRoleMenuResponse:
    """查询角色菜单权限。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return service.get_role_menus(role_id)


@router.put("/roles/{role_id}/menus", response_model=SystemRoleMenuResponse)
def update_role_menus(
    role_id: str,
    request: SystemRoleMenuUpdateRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemRoleMenuResponse:
    """保存角色菜单权限。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return service.save_role_menus(role_id, request)


@router.delete("/roles/{role_id}", response_model=StatusChangeResponse)
def disable_role(role_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """禁用角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return StatusChangeResponse(**service.disable_role(role_id))


@router.delete("/roles/{role_id}/delete", response_model=StatusChangeResponse)
def delete_role(role_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """物理删除角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return StatusChangeResponse(**service.delete_role(role_id))


@router.post("/roles/{role_id}/enable", response_model=StatusChangeResponse)
def enable_role(role_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """启用角色。"""

    service = _service()
    service.require_menu_access(user, {"roleManagement"})
    return StatusChangeResponse(**service.enable_role(role_id))


@router.get("/users", response_model=SystemUserListResponse)
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    keyword: str | None = Query(default=None),
    role: str | None = Query(default=None),
    status: str | None = Query(default=None),
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemUserListResponse:
    """分页查询系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return service.list_users(page=page, page_size=page_size, keyword=keyword, role=role, status=status)


@router.post("/users", response_model=SystemUserResponse)
def create_user(request: SystemUserCreateRequest, user: SystemUserEntity = Depends(get_current_user)) -> SystemUserResponse:
    """创建系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return service.create_user(request)


@router.put("/users/{user_id}", response_model=SystemUserResponse)
def update_user(
    user_id: str,
    request: SystemUserUpdateRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> SystemUserResponse:
    """修改系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return service.update_user(user_id, request, current_user_id=user.user_id)


@router.post("/users/{user_id}/password")
def reset_user_password(
    user_id: str,
    request: SystemUserPasswordResetRequest,
    user: SystemUserEntity = Depends(get_current_user),
) -> dict[str, str]:
    """重置系统用户密码。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return service.reset_user_password(user_id, request.password)


@router.delete("/users/{user_id}", response_model=StatusChangeResponse)
def disable_user(user_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """禁用系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return StatusChangeResponse(**service.disable_user(user_id, current_user_id=user.user_id))


@router.delete("/users/{user_id}/delete", response_model=StatusChangeResponse)
def delete_user(user_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """物理删除系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return StatusChangeResponse(**service.delete_user(user_id, current_user_id=user.user_id))


@router.post("/users/{user_id}/enable", response_model=StatusChangeResponse)
def enable_user(user_id: str, user: SystemUserEntity = Depends(get_current_user)) -> StatusChangeResponse:
    """启用系统用户。"""

    service = _service()
    service.require_menu_access(user, {"userManagement"})
    return StatusChangeResponse(**service.enable_user(user_id))
