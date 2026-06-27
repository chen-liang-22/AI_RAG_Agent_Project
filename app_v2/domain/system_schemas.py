"""V2 系统管理请求和响应模型。"""

from pydantic import BaseModel, Field


class SystemMenuResponse(BaseModel):
    """系统菜单响应，雪花 ID 按字符串返回给前端。"""

    menu_id: str
    parent_menu_id: str | None = None
    menu_code: str
    menu_name: str
    menu_type: str
    page_key: str | None = None
    route_path: str | None = None
    component_key: str | None = None
    icon: str | None = None
    permission_code: str | None = None
    sort_order: int
    visible: bool
    status: str
    metadata: dict = Field(default_factory=dict)
    children: list["SystemMenuResponse"] = Field(default_factory=list)


class SystemMenuCreateRequest(BaseModel):
    """创建系统菜单请求。"""

    parent_menu_id: str | None = Field(None, description="父级菜单雪花 ID 字符串")
    menu_code: str = Field(..., min_length=1, max_length=128)
    menu_name: str = Field(..., min_length=1, max_length=128)
    menu_type: str = Field("page", max_length=32)
    page_key: str | None = Field(None, max_length=128)
    route_path: str | None = Field(None, max_length=255)
    component_key: str | None = Field(None, max_length=128)
    icon: str | None = Field(None, max_length=64)
    permission_code: str | None = Field(None, max_length=128)
    sort_order: int = 0
    visible: bool = True
    status: str = Field("active", max_length=32)
    metadata: dict = Field(default_factory=dict)


class SystemMenuUpdateRequest(BaseModel):
    """修改系统菜单请求。"""

    parent_menu_id: str | None = Field(None, description="父级菜单雪花 ID 字符串")
    menu_code: str | None = Field(None, min_length=1, max_length=128)
    menu_name: str | None = Field(None, min_length=1, max_length=128)
    menu_type: str | None = Field(None, max_length=32)
    page_key: str | None = Field(None, max_length=128)
    route_path: str | None = Field(None, max_length=255)
    component_key: str | None = Field(None, max_length=128)
    icon: str | None = Field(None, max_length=64)
    permission_code: str | None = Field(None, max_length=128)
    sort_order: int | None = None
    visible: bool | None = None
    status: str | None = Field(None, max_length=32)
    metadata: dict | None = None


class SystemUserResponse(BaseModel):
    """系统用户响应，不包含密码哈希。"""

    user_id: str
    username: str
    display_name: str
    role: str
    status: str
    last_login_at: str | None = None
    created_at: str
    updated_at: str


class SystemUserListResponse(BaseModel):
    """系统用户分页响应。"""

    items: list[SystemUserResponse]
    total: int
    page: int
    page_size: int


class SystemUserCreateRequest(BaseModel):
    """创建系统用户请求。"""

    username: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field("user", max_length=64)
    status: str = Field("active", max_length=32)


class SystemUserUpdateRequest(BaseModel):
    """修改系统用户请求。"""

    display_name: str | None = Field(None, min_length=1, max_length=128)
    password: str | None = Field(None, min_length=6, max_length=128)
    role: str | None = Field(None, max_length=64)
    status: str | None = Field(None, max_length=32)


class SystemUserPasswordResetRequest(BaseModel):
    """重置系统用户密码请求。"""

    password: str = Field(..., min_length=6, max_length=128)


class SystemRoleResponse(BaseModel):
    """系统角色响应，雪花 ID 按字符串返回给前端。"""

    role_id: str
    role_code: str
    role_name: str
    status: str
    sort_order: int
    built_in: bool
    description: str | None = None
    created_at: str
    updated_at: str


class SystemRoleListResponse(BaseModel):
    """系统角色分页响应。"""

    items: list[SystemRoleResponse]
    total: int
    page: int
    page_size: int


class SystemRoleOptionResponse(BaseModel):
    """系统角色下拉选项。"""

    role_code: str
    role_name: str


class SystemRoleCreateRequest(BaseModel):
    """创建系统角色请求。"""

    role_code: str = Field(..., min_length=1, max_length=64)
    role_name: str = Field(..., min_length=1, max_length=128)
    status: str = Field("active", max_length=32)
    sort_order: int = 0
    description: str | None = Field(None, max_length=255)
    menu_ids: list[str] = Field(default_factory=list, description="菜单雪花 ID 字符串列表")


class SystemRoleUpdateRequest(BaseModel):
    """修改系统角色请求。"""

    role_name: str | None = Field(None, min_length=1, max_length=128)
    status: str | None = Field(None, max_length=32)
    sort_order: int | None = None
    description: str | None = Field(None, max_length=255)


class SystemRoleMenuResponse(BaseModel):
    """角色菜单权限响应。"""

    role: SystemRoleResponse
    checked_menu_ids: list[str] = Field(default_factory=list)
    menu_tree: list[SystemMenuResponse] = Field(default_factory=list)


class SystemRoleMenuUpdateRequest(BaseModel):
    """保存角色菜单权限请求。"""

    menu_ids: list[str] = Field(default_factory=list, description="菜单雪花 ID 字符串列表")


class StatusChangeResponse(BaseModel):
    """启用或禁用响应。"""

    status: str
    role_id: str | None = None
    user_id: str | None = None
    menu_id: str | None = None
