"""认证接口请求和响应模型。

这里只定义 FastAPI/Pydantic DTO，不写认证逻辑。
字段名属于前后端协议，修改前需要同步前端和接口文档。
"""

from pydantic import BaseModel, Field


class CurrentUserResponse(BaseModel):
    """当前登录用户信息。"""

    user_id: str  # 用户唯一编号
    username: str  # 登录账号
    display_name: str  # 页面展示名称
    role: str  # 用户角色，第一期只保留 admin/user
    status: str  # 用户状态，active 才允许登录


class LoginRequest(BaseModel):
    """登录请求体。"""

    username: str = Field(..., min_length=1, max_length=128)  # 登录账号
    password: str = Field(..., min_length=1, max_length=128)  # 登录密码，只在请求内短暂使用


class LoginResponse(BaseModel):
    """登录成功响应体。"""

    access_token: str  # 后续请求放到 Authorization: Bearer 中
    token_type: str = "bearer"  # 固定使用 bearer，方便前端统一拼接请求头
    expires_in: int  # token 有效秒数
    user: CurrentUserResponse  # 当前登录用户信息


class RefreshResponse(LoginResponse):
    """续签成功响应体，结构与登录成功保持一致。"""


class LogoutResponse(BaseModel):
    """退出登录响应体。"""

    status: str  # 固定返回 ok，前端收到后清理本地 token
