from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from sqlalchemy import select

from app_v2.domain.entities import SystemUserEntity
from app_v2.infrastructure.id_generator import new_id
from app_v2.infrastructure.orm_session import get_orm_engine, orm_session_context
from core.utils.config_handler import load_env_file
from core.utils.logger_handler import logger
from core.utils.redis_client import RedisClient, get_redis_client


PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 200_000
PASSWORD_SALT_BYTES = 16
ACCESS_TOKEN_TYPE = "access"
ACTIVE_USER_STATUS = "active"
DEFAULT_ACCESS_TOKEN_EXPIRE_SECONDS = 30 * 60
DEFAULT_REFRESH_TOKEN_EXPIRE_SECONDS = 7 * 24 * 60 * 60
DEFAULT_AUTH_SECRET = "ai-rag-agent-local-dev-secret"
DEFAULT_REFRESH_COOKIE_NAME = "ai_rag_refresh_token"

_secret_warning_logged = False


@dataclass(frozen=True)
class AuthTokenPayload:
    """解析后的 access_token 载荷。"""

    user_id: str
    username: str
    display_name: str
    role: str
    exp: int


@dataclass(frozen=True)
class RefreshSession:
    """Redis 中保存的 refresh_token 会话信息。"""

    user_id: str
    username: str
    role: str
    created_at: str
    last_used_at: str
    ip_address: str | None = None
    user_agent: str | None = None


def create_password_hash(password: str) -> str:
    """生成不可逆密码哈希，避免数据库保存明文密码。"""

    salt = os.urandom(PASSWORD_SALT_BYTES)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return "$".join(
        [
            PASSWORD_HASH_ALGORITHM,
            str(PASSWORD_HASH_ITERATIONS),
            _base64_encode(salt),
            _base64_encode(derived_key),
        ]
    )


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码是否匹配数据库中的 PBKDF2 哈希。"""

    try:
        algorithm, iterations_text, salt_text, stored_hash_text = password_hash.split("$", 3)
        if algorithm != PASSWORD_HASH_ALGORITHM:
            return False
        salt = _base64_decode(salt_text)
        stored_hash = _base64_decode(stored_hash_text)
        candidate_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations_text),
        )
        return hmac.compare_digest(candidate_hash, stored_hash)
    except Exception:
        # 哈希格式异常统一按失败处理，避免把内部细节暴露给调用方。
        return False


def create_refresh_token() -> str:
    """生成强随机 refresh_token 明文，只返回给浏览器 Cookie。"""

    return secrets.token_urlsafe(48)


def hash_refresh_token(refresh_token: str) -> str:
    """计算 refresh_token 哈希，后端 Redis 只保存这个值。"""

    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()


def _base64_encode(raw_data: bytes) -> str:
    """把二进制数据转成 URL 安全文本，便于写入密码哈希。"""

    return base64.urlsafe_b64encode(raw_data).decode("ascii").rstrip("=")


def _base64_decode(text: str) -> bytes:
    """还原去掉填充符的 URL 安全 Base64 文本。"""

    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(f"{text}{padding}".encode("ascii"))


def _utc_now() -> datetime:
    """返回去掉微秒的 UTC 时间，便于写入 MySQL DATETIME 字段。"""

    return datetime.utcnow().replace(microsecond=0)


def _utc_now_aware() -> datetime:
    """返回带 UTC 时区的当前时间，专门用于 JWT 时间戳计算。"""

    return datetime.now(timezone.utc).replace(microsecond=0)


def _datetime_to_epoch_seconds(value: datetime) -> int:
    """把 datetime 转成 UTC 秒级时间戳。

    Python 的 naive datetime.timestamp() 会按本地时区解释。
    认证 token 必须按 UTC 计算，否则在东八区会被错误判定为已过期。
    """

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def _utc_now_text() -> str:
    """返回 Redis 会话展示用时间文本。"""

    return _utc_now().isoformat(timespec="seconds", sep=" ")


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，配置错误时使用默认值。"""

    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("[认证] 环境变量不是有效整数 名称=%s 值=%s，已使用默认值=%s", name, value, default)
        return default


class AuthRepository:
    """系统用户仓储，集中负责 system_users 表读写。"""

    def ensure_user_table(self) -> None:
        """确保登录用户表存在，降低旧环境首次启动成本。"""

        engine = get_orm_engine()
        SystemUserEntity.__table__.create(bind=engine, checkfirst=True)

    def get_by_username(self, username: str) -> SystemUserEntity | None:
        """按登录账号查询用户。"""

        statement = select(SystemUserEntity).where(SystemUserEntity.username == username)
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def get_by_user_id(self, user_id: str) -> SystemUserEntity | None:
        """按用户编号查询用户。"""

        with orm_session_context() as session:
            return session.get(SystemUserEntity, user_id)

    def create_user(
            self,
            *,
            username: str,
            password: str,
            display_name: str,
            role: str,
            status: str = ACTIVE_USER_STATUS,
    ) -> SystemUserEntity:
        """创建系统用户。"""

        now = _utc_now()
        user = SystemUserEntity(
            user_id=new_id(),
            username=username,
            display_name=display_name,
            password_hash=create_password_hash(password),
            role=role,
            status=status,
            last_login_at=None,
            created_at=now,
            updated_at=now,
        )
        with orm_session_context() as session:
            session.add(user)
        created = self.get_by_username(username)
        if created is None:
            raise RuntimeError(f"系统用户创建失败：{username}")
        return created

    def mark_login_success(self, user_id: str) -> None:
        """记录最后登录时间，用于排查账号是否正常使用。"""

        now = _utc_now()
        with orm_session_context() as session:
            user = session.get(SystemUserEntity, user_id)
            if user is None:
                return
            user.last_login_at = now
            user.updated_at = now


class RefreshSessionStore:
    """Refresh 会话存储，使用 Redis TTL 管理登录续签状态。"""

    def __init__(self, redis_client: RedisClient | None = None):
        self.redis_client = redis_client or get_redis_client()

    def save_session(
            self,
            *,
            refresh_token_hash: str,
            session: RefreshSession,
            ttl_seconds: int,
    ) -> bool:
        """保存 refresh 会话，返回是否写入成功。"""

        key = self._session_key(refresh_token_hash)
        payload = {
            "user_id": session.user_id,
            "username": session.username,
            "role": session.role,
            "created_at": session.created_at,
            "last_used_at": session.last_used_at,
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
        }
        saved = self.redis_client.set_json(key, payload, ttl_seconds=ttl_seconds)
        if saved:
            logger.info("[认证] refresh 会话已写入 Redis 用户编号=%s Redis键=%s", session.user_id, key)
        else:
            logger.warning("[认证] refresh 会话写入 Redis 失败 用户编号=%s Redis键=%s", session.user_id, key)
        return saved

    def get_session(self, refresh_token_hash: str) -> RefreshSession | None:
        """读取 refresh 会话，不存在表示已过期、已退出或 Redis 不可用。"""

        value = self.redis_client.get_json(self._session_key(refresh_token_hash), default=None)
        if not isinstance(value, dict):
            return None
        try:
            return RefreshSession(
                user_id=str(value["user_id"]),
                username=str(value.get("username") or ""),
                role=str(value.get("role") or "user"),
                created_at=str(value.get("created_at") or ""),
                last_used_at=str(value.get("last_used_at") or ""),
                ip_address=value.get("ip_address"),
                user_agent=value.get("user_agent"),
            )
        except KeyError:
            logger.warning("[认证] Redis refresh 会话格式异常")
            return None

    def touch_session(self, refresh_token_hash: str, ttl_seconds: int) -> bool:
        """更新 refresh 会话最后使用时间，并维持 7 天 TTL。"""

        session = self.get_session(refresh_token_hash)
        if session is None:
            return False
        return self.save_session(
            refresh_token_hash=refresh_token_hash,
            session=RefreshSession(
                user_id=session.user_id,
                username=session.username,
                role=session.role,
                created_at=session.created_at,
                last_used_at=_utc_now_text(),
                ip_address=session.ip_address,
                user_agent=session.user_agent,
            ),
            ttl_seconds=ttl_seconds,
        )

    def delete_session(self, refresh_token_hash: str) -> None:
        """删除 refresh 会话，用于退出登录和吊销。"""

        key = self._session_key(refresh_token_hash)
        deleted = self.redis_client.delete(key)
        logger.info("[认证] refresh 会话删除结果 Redis键=%s 删除数量=%s", key, deleted)

    def is_available(self) -> bool:
        """判断 Redis 会话服务是否可用。"""

        return self.redis_client.is_available()

    def _session_key(self, refresh_token_hash: str) -> str:
        """生成带项目前缀的 Redis 会话 key。"""

        return self.redis_client.build_key("auth", "refresh", refresh_token_hash)


class AuthService:
    """认证服务外观，统一封装登录、续签、退出和 token 处理。"""

    def __init__(
            self,
            repository: AuthRepository | None = None,
            refresh_store: RefreshSessionStore | None = None,
    ):
        self.repository = repository or AuthRepository()
        self.refresh_store = refresh_store or RefreshSessionStore()

    @property
    def access_token_expire_seconds(self) -> int:
        """读取 access_token 有效期，默认 30 分钟。"""

        return max(60, _env_int("AUTH_ACCESS_TOKEN_EXPIRE_SECONDS", DEFAULT_ACCESS_TOKEN_EXPIRE_SECONDS))

    @property
    def refresh_token_expire_seconds(self) -> int:
        """读取 refresh_token 有效期，默认 7 天。"""

        return max(60, _env_int("AUTH_REFRESH_TOKEN_EXPIRE_SECONDS", DEFAULT_REFRESH_TOKEN_EXPIRE_SECONDS))

    @property
    def refresh_cookie_name(self) -> str:
        """读取 refresh_token Cookie 名称。"""

        return os.getenv("AUTH_REFRESH_COOKIE_NAME", DEFAULT_REFRESH_COOKIE_NAME).strip() or DEFAULT_REFRESH_COOKIE_NAME

    @property
    def refresh_cookie_secure(self) -> bool:
        """读取 Cookie Secure 配置，本地开发默认关闭。"""

        return _env_bool("AUTH_REFRESH_COOKIE_SECURE", False)

    @property
    def refresh_cookie_samesite(self) -> str:
        """读取 Cookie SameSite 配置。"""

        value = os.getenv("AUTH_REFRESH_COOKIE_SAMESITE", "lax").strip().lower()
        return value if value in {"lax", "strict", "none"} else "lax"

    def login(
            self,
            username: str,
            password: str,
            *,
            ip_address: str | None = None,
            user_agent: str | None = None,
    ) -> tuple[str, str, int, SystemUserEntity]:
        """执行用户名密码登录，成功时返回 access_token、refresh_token、有效期和用户。"""

        self.prepare_auth_storage()
        self._ensure_refresh_store_available()
        clean_username = username.strip()
        logger.info("[认证] 收到登录请求 用户名=%s", clean_username)

        user = self.repository.get_by_username(clean_username)
        if user is None or user.status != ACTIVE_USER_STATUS:
            logger.warning("[认证] 登录失败 用户不存在或未启用 用户名=%s", clean_username)
            raise ValueError("用户名或密码错误")
        if not verify_password(password, user.password_hash):
            logger.warning("[认证] 登录失败 密码错误 用户名=%s", clean_username)
            raise ValueError("用户名或密码错误")

        self.repository.mark_login_success(user.user_id)
        access_token = self.create_access_token(self.user_to_payload(user))
        refresh_token = create_refresh_token()
        refresh_token_hash = hash_refresh_token(refresh_token)
        now_text = _utc_now_text()
        saved = self.refresh_store.save_session(
            refresh_token_hash=refresh_token_hash,
            session=RefreshSession(
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                created_at=now_text,
                last_used_at=now_text,
                ip_address=ip_address,
                user_agent=user_agent,
            ),
            ttl_seconds=self.refresh_token_expire_seconds,
        )
        if not saved:
            raise RuntimeError("登录会话服务暂不可用")

        logger.info("[认证] 登录成功 用户编号=%s 用户名=%s 角色=%s", user.user_id, user.username, user.role)
        return access_token, refresh_token, self.access_token_expire_seconds, user

    def refresh_access_token(self, refresh_token: str) -> tuple[str, int, SystemUserEntity]:
        """使用 refresh_token 续签新的 access_token。"""

        self.prepare_auth_storage()
        self._ensure_refresh_store_available()
        if not refresh_token:
            raise ValueError("登录状态已失效，请重新登录")

        refresh_token_hash = hash_refresh_token(refresh_token)
        session = self.refresh_store.get_session(refresh_token_hash)
        if session is None:
            logger.warning("[认证] refresh_token 无效或已过期")
            raise ValueError("登录状态已失效，请重新登录")

        user = self.repository.get_by_user_id(session.user_id)
        if user is None or user.status != ACTIVE_USER_STATUS:
            logger.warning("[认证] refresh_token 对应用户不存在或未启用 用户编号=%s", session.user_id)
            self.refresh_store.delete_session(refresh_token_hash)
            raise ValueError("登录状态已失效，请重新登录")

        self.refresh_store.touch_session(refresh_token_hash, self.refresh_token_expire_seconds)
        access_token = self.create_access_token(self.user_to_payload(user))
        logger.info("[认证] access_token 续签成功 用户编号=%s 用户名=%s", user.user_id, user.username)
        return access_token, self.access_token_expire_seconds, user

    def logout(self, refresh_token: str | None) -> None:
        """退出登录，删除 Redis refresh 会话。"""

        if not refresh_token:
            return
        self.refresh_store.delete_session(hash_refresh_token(refresh_token))

    def prepare_auth_storage(self) -> None:
        """初始化认证表和本地默认管理员。"""

        load_env_file()
        self.repository.ensure_user_table()
        self.ensure_default_admin()

    def ensure_default_admin(self) -> None:
        """本地开发时自动准备默认管理员账号。"""

        if not _env_bool("AUTH_ENABLE_DEFAULT_ADMIN", True):
            logger.info("[认证] 默认管理员初始化已关闭")
            return

        username = os.getenv("AUTH_DEFAULT_USERNAME", "admin").strip()
        password = os.getenv("AUTH_DEFAULT_PASSWORD", "1234qwer").strip()
        display_name = os.getenv("AUTH_DEFAULT_DISPLAY_NAME", "系统管理员").strip()
        role = os.getenv("AUTH_DEFAULT_ROLE", "admin").strip() or "admin"
        if not username or not password:
            logger.info("[认证] 未配置默认管理员账号或密码，跳过默认管理员初始化")
            return

        existing_user = self.repository.get_by_username(username)
        if existing_user is not None:
            return

        self.repository.create_user(
            username=username,
            password=password,
            display_name=display_name or username,
            role=role,
        )
        logger.warning("[认证] 已创建本地默认管理员 用户名=%s，请在生产环境修改默认密码", username)

    def create_access_token(
            self,
            payload: dict[str, Any],
            *,
            expires_at: datetime | None = None,
    ) -> str:
        """签发 JWT access_token。"""

        expire_time = expires_at or (_utc_now_aware() + timedelta(seconds=self.access_token_expire_seconds))
        token_payload = {
            "sub": payload["user_id"],
            "username": payload["username"],
            "display_name": payload.get("display_name") or payload["username"],
            "role": payload.get("role") or "user",
            "type": ACCESS_TOKEN_TYPE,
            "iat": _datetime_to_epoch_seconds(_utc_now_aware()),
            "exp": _datetime_to_epoch_seconds(expire_time),
        }
        return jwt.encode(token_payload, self._token_secret(), algorithm="HS256")

    def parse_access_token(self, token: str) -> AuthTokenPayload | None:
        """解析并校验 JWT access_token。"""

        try:
            payload = jwt.decode(token, self._token_secret(), algorithms=["HS256"])
            if payload.get("type") != ACCESS_TOKEN_TYPE:
                logger.warning("[认证] token 类型不正确 类型=%s", payload.get("type"))
                return None
            return AuthTokenPayload(
                user_id=str(payload["sub"]),
                username=str(payload["username"]),
                display_name=str(payload.get("display_name") or payload["username"]),
                role=str(payload.get("role") or "user"),
                exp=int(payload["exp"]),
            )
        except ExpiredSignatureError:
            logger.info("[认证] access_token 已过期")
            return None
        except (InvalidTokenError, KeyError, TypeError, ValueError) as exc:
            logger.warning("[认证] access_token 解析失败 错误=%s", exc)
            return None

    def get_active_user_by_token(self, token: str) -> SystemUserEntity | None:
        """根据 access_token 查询当前仍启用的数据库用户。"""

        self.prepare_auth_storage()
        payload = self.parse_access_token(token)
        if payload is None:
            return None
        user = self.repository.get_by_user_id(payload.user_id)
        if user is None or user.status != ACTIVE_USER_STATUS:
            logger.warning("[认证] access_token 对应用户不存在或未启用 用户编号=%s", payload.user_id)
            return None
        return user

    @staticmethod
    def user_to_payload(user: SystemUserEntity) -> dict[str, str]:
        """把用户实体转换成 access_token 需要的安全字段。"""

        return {
            "user_id": user.user_id,
            "username": user.username,
            "display_name": user.display_name,
            "role": user.role,
        }

    def _ensure_refresh_store_available(self) -> None:
        """登录和续签必须依赖 Redis，避免无法吊销 refresh_token。"""

        if not self.refresh_store.is_available():
            logger.error("[认证] Redis 会话服务不可用")
            raise RuntimeError("登录会话服务暂不可用")

    @staticmethod
    def _token_secret() -> str:
        """读取 JWT 签名密钥，开发环境缺省值只用于本地调试。"""

        global _secret_warning_logged

        secret = os.getenv("AUTH_TOKEN_SECRET", "").strip()
        if secret:
            return secret
        if not _secret_warning_logged:
            logger.warning("[认证] 未配置 AUTH_TOKEN_SECRET，当前使用本地开发默认签名密钥")
            _secret_warning_logged = True
        return DEFAULT_AUTH_SECRET
