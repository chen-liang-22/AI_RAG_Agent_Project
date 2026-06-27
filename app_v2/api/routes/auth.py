"""V2 登录认证接口。"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth_schemas import CurrentUserResponse, LoginRequest, LoginResponse, LogoutResponse, RefreshResponse
from app_v2.application.auth_service import AuthService
from domain.entities import SystemUserEntity
from utils.logger_handler import logger

router = APIRouter(prefix="/auth", tags=["V2 认证"])
bearer_scheme = HTTPBearer(auto_error=False)
_auth_service: AuthService | None = None


def _get_auth_service() -> AuthService:
    """获取认证服务单例。

    这里用简单单例是为了复用 Redis refresh session 配置，避免每次请求重复初始化。
    """

    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


def _current_user_response(user: SystemUserEntity) -> CurrentUserResponse:
    """把用户实体转换成响应对象，避免把 password_hash 返回给前端。"""

    return CurrentUserResponse(user_id=user.user_id, username=user.username, display_name=user.display_name, role=user.role, status=user.status)


def _set_refresh_cookie(response: Response, auth_service: AuthService, refresh_token: str) -> None:
    """把 refresh_token 写入 V2 专用 HttpOnly Cookie。"""

    response.set_cookie(
        key=auth_service.refresh_cookie_name,
        value=refresh_token,
        max_age=auth_service.refresh_token_expire_seconds,
        httponly=True,
        secure=auth_service.refresh_cookie_secure,
        samesite=auth_service.refresh_cookie_samesite,
        path="/api/v2/auth",
    )


def _clear_refresh_cookie(response: Response, auth_service: AuthService) -> None:
    """清理 V2 refresh_token Cookie。"""

    response.delete_cookie(
        key=auth_service.refresh_cookie_name,
        path="/api/v2/auth",
        secure=auth_service.refresh_cookie_secure,
        samesite=auth_service.refresh_cookie_samesite,
    )


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> SystemUserEntity:
    """根据 Bearer JWT 解析当前登录用户。"""

    if credentials is None or credentials.scheme.lower() != "bearer":
        logger.warning("[V2认证] 当前请求缺少 Bearer access_token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    user = _get_auth_service().get_active_user_by_token(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态已失效，请重新登录")
    return user


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, http_request: Request, response: Response) -> LoginResponse:
    """账号密码登录。"""

    auth_service = _get_auth_service()
    try:
        access_token, refresh_token, expires_in, user = auth_service.login(
            request.username,
            request.password,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    _set_refresh_cookie(response, auth_service, refresh_token)
    return LoginResponse(access_token=access_token, expires_in=expires_in, user=_current_user_response(user))


@router.post("/refresh", response_model=RefreshResponse)
def refresh(request: Request, response: Response) -> RefreshResponse:
    """使用 HttpOnly Cookie 中的 refresh_token 换取新的 access_token。"""

    auth_service = _get_auth_service()
    refresh_token = request.cookies.get(auth_service.refresh_cookie_name, "")
    try:
        access_token, expires_in, user = auth_service.refresh_access_token(refresh_token)
    except ValueError as exc:
        _clear_refresh_cookie(response, auth_service)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return RefreshResponse(access_token=access_token, expires_in=expires_in, user=_current_user_response(user))


@router.get("/me", response_model=CurrentUserResponse)
def current_user(user: SystemUserEntity = Depends(get_current_user)) -> CurrentUserResponse:
    """返回当前登录用户信息。"""

    logger.info("[V2认证] 查询当前用户 用户ID=%s 用户名=%s", user.user_id, user.username)
    return _current_user_response(user)


@router.post("/logout", response_model=LogoutResponse)
def logout(request: Request, response: Response, user: SystemUserEntity = Depends(get_current_user)) -> LogoutResponse:
    """退出登录。"""

    auth_service = _get_auth_service()
    auth_service.logout(request.cookies.get(auth_service.refresh_cookie_name))
    _clear_refresh_cookie(response, auth_service)
    logger.info("[V2认证] 用户退出登录 用户ID=%s 用户名=%s", user.user_id, user.username)
    return LogoutResponse(status="ok")
