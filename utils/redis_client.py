"""Redis 基础工具。

这一层只负责连接、缓存、任务状态和轻量锁，不承载具体业务逻辑。
业务代码不要直接依赖 redis-py，统一通过这里调用，方便后续做降级、替换或测试替身。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from utils.config_handler import load_env_file
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

try:
    import redis
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - 未安装依赖时通过 is_available 降级。
    redis = None
    Redis = Any  # type: ignore[assignment]
    RedisError = RuntimeError  # type: ignore[assignment]


DEFAULT_REDIS_CONFIG: dict[str, Any] = {
    "enabled": False,
    "host": "127.0.0.1",
    "port": 6379,
    "db": 0,
    "password_env": "REDIS_PASSWORD",
    "password": "",
    "socket_timeout_seconds": 2,
    "socket_connect_timeout_seconds": 2,
    "key_prefix": "ai_rag_agent",
    "default_ttl_seconds": 1800,
    "lock_timeout_seconds": 30,
}

_redis_client: "RedisClient | None" = None


def load_redis_config(config_path: str = get_abs_path("config/redis.yml")) -> dict[str, Any]:
    """读取 Redis 配置，并允许环境变量覆盖连接信息。"""

    load_env_file()
    config = dict(DEFAULT_REDIS_CONFIG)
    path = Path(config_path)
    if path.exists():
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw_config, dict):
            config.update(raw_config)

    password_env = str(config.get("password_env") or "REDIS_PASSWORD")
    config.update({
        "enabled": _env_bool("REDIS_ENABLED", bool(config.get("enabled"))),
        "host": os.getenv("REDIS_HOST", str(config.get("host") or "127.0.0.1")),
        "port": _env_int("REDIS_PORT", int(config.get("port") or 6379)),
        "db": _env_int("REDIS_DB", int(config.get("db") or 0)),
        "password": os.getenv(password_env, os.getenv("REDIS_PASSWORD", str(config.get("password") or ""))),
        "socket_timeout_seconds": _env_float(
            "REDIS_SOCKET_TIMEOUT",
            float(config.get("socket_timeout_seconds") or 2),
        ),
        "socket_connect_timeout_seconds": _env_float(
            "REDIS_SOCKET_CONNECT_TIMEOUT",
            float(config.get("socket_connect_timeout_seconds") or 2),
        ),
        "key_prefix": os.getenv("REDIS_KEY_PREFIX", str(config.get("key_prefix") or "ai_rag_agent")),
        "default_ttl_seconds": _env_int(
            "REDIS_DEFAULT_TTL_SECONDS",
            int(config.get("default_ttl_seconds") or 1800),
        ),
        "lock_timeout_seconds": _env_int(
            "REDIS_LOCK_TIMEOUT_SECONDS",
            int(config.get("lock_timeout_seconds") or 30),
        ),
    })
    return config


def get_redis_client() -> "RedisClient":
    """获取进程级 Redis 工具单例。"""

    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient(load_redis_config())
    return _redis_client


def reset_redis_client() -> None:
    """重置 Redis 工具单例，主要用于测试切换配置后重新加载。"""

    global _redis_client
    _redis_client = None


class RedisClient:
    """Redis 客户端包装，提供项目内常用的缓存、任务状态和锁能力。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(DEFAULT_REDIS_CONFIG)
        self.config.update(config or {})
        self._client: Redis | None = None

    @property
    def enabled(self) -> bool:
        """判断配置层是否启用 Redis。"""

        return bool(self.config.get("enabled")) and redis is not None

    def is_available(self) -> bool:
        """检查 Redis 是否可用，不向外抛连接异常。"""

        if not self.enabled:
            return False
        try:
            return bool(self.client().ping())
        except RedisError as exc:
            logger.warning("[Redis] 连接不可用 错误=%s", exc)
            return False

    def client(self) -> Redis:
        """懒加载 redis-py 客户端，避免服务启动时硬依赖 Redis 在线。"""

        if redis is None:
            raise RuntimeError("缺少 Redis 依赖，请先执行 pip install -r requirements.txt")
        if not bool(self.config.get("enabled")):
            raise RuntimeError("Redis 未启用，请检查 config/redis.yml 或 REDIS_ENABLED")
        if self._client is None:
            self._client = redis.Redis(
                host=str(self.config.get("host") or "127.0.0.1"),
                port=int(self.config.get("port") or 6379),
                db=int(self.config.get("db") or 0),
                password=str(self.config.get("password") or "") or None,
                decode_responses=True,
                socket_timeout=float(self.config.get("socket_timeout_seconds") or 2),
                socket_connect_timeout=float(self.config.get("socket_connect_timeout_seconds") or 2),
            )
        return self._client

    def build_key(self, *parts: object) -> str:
        """按项目统一前缀拼接 Redis key，避免不同模块 key 冲突。"""

        prefix = str(self.config.get("key_prefix") or "ai_rag_agent").strip(":")
        clean_parts = [str(part).strip(":") for part in parts if str(part).strip(":")]
        return ":".join([prefix, *clean_parts])

    def get_json(self, key: str, default: Any = None) -> Any:
        """读取 JSON 缓存，解析失败或 Redis 不可用时返回默认值。"""

        if not self.enabled:
            return default
        try:
            value = self.client().get(key)
            return json.loads(value) if value else default
        except (RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("[Redis] 读取缓存失败 Key=%s 错误=%s", key, exc)
            return default

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """写入 JSON 缓存，返回是否写入成功。"""

        if not self.enabled:
            return False
        try:
            ttl = self._normalize_ttl(ttl_seconds)
            payload = json.dumps(value, ensure_ascii=False)
            return bool(self.client().set(name=key, value=payload, ex=ttl))
        except (RedisError, TypeError, ValueError) as exc:
            logger.warning("[Redis] 写入缓存失败 Key=%s 错误=%s", key, exc)
            return False

    def delete(self, *keys: str) -> int:
        """删除一个或多个 Redis key，Redis 不可用时返回 0。"""

        if not self.enabled or not keys:
            return 0
        try:
            return int(self.client().delete(*keys))
        except RedisError as exc:
            logger.warning("[Redis] 删除缓存失败 Keys=%s 错误=%s", keys, exc)
            return 0

    def set_task_status(
            self,
            task_type: str,
            task_id: str,
            status: str,
            payload: dict[str, Any] | None = None,
            ttl_seconds: int | None = None,
    ) -> bool:
        """保存长任务状态，供上传、发布、重切、后台生成等流程复用。"""

        task_payload = {
            "task_type": task_type,
            "task_id": task_id,
            "status": status,
            **(payload or {}),
        }
        return self.set_json(self.task_key(task_type, task_id), task_payload, ttl_seconds=ttl_seconds)

    def get_task_status(self, task_type: str, task_id: str) -> dict[str, Any] | None:
        """读取长任务状态；不存在或不可用时返回 None。"""

        value = self.get_json(self.task_key(task_type, task_id), default=None)
        return value if isinstance(value, dict) else None

    def task_key(self, task_type: str, task_id: str) -> str:
        """生成长任务状态 key。"""

        return self.build_key("task", task_type, task_id)

    def lock_key(self, lock_name: str, resource_id: str) -> str:
        """生成分布式锁 key。"""

        return self.build_key("lock", lock_name, resource_id)

    @contextmanager
    def lock(
            self,
            lock_name: str,
            resource_id: str,
            timeout_seconds: int | None = None,
            blocking_timeout_seconds: int | None = 0,
    ) -> Iterator[bool]:
        """获取 Redis 分布式锁，yield 表示是否成功拿到锁。"""

        if not self.enabled:
            yield False
            return

        lock = self.client().lock(
            name=self.lock_key(lock_name, resource_id),
            timeout=timeout_seconds or int(self.config.get("lock_timeout_seconds") or 30),
            blocking_timeout=blocking_timeout_seconds,
        )
        acquired = False
        try:
            acquired = bool(lock.acquire(blocking=blocking_timeout_seconds is not None))
            yield acquired
        except RedisError as exc:
            logger.warning("[Redis] 获取锁失败 锁名称=%s 资源编号=%s 错误=%s", lock_name, resource_id, exc)
            yield False
        finally:
            if acquired:
                try:
                    lock.release()
                except RedisError as exc:
                    logger.warning("[Redis] 释放锁失败 锁名称=%s 资源编号=%s 错误=%s", lock_name, resource_id, exc)

    def _normalize_ttl(self, ttl_seconds: int | None) -> int:
        """统一缓存过期时间，避免传入非正数导致 Redis 行为不符合预期。"""

        ttl = ttl_seconds if ttl_seconds is not None else int(self.config.get("default_ttl_seconds") or 1800)
        return max(1, int(ttl))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)
