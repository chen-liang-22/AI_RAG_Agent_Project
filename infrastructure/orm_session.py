"""SQLAlchemy ORM 会话适配层。

这一层的作用类似 Java 项目里的数据源 + SqlSessionFactory：
- 从现有 config/database.yml 和 .env 读取数据库配置；
- 只为 MySQL 创建 SQLAlchemy Engine；
- 提供统一的 Session 上下文，正常提交、异常回滚。

仓储层统一使用这里的 ORM Session，风格接近 MyBatis-Plus 的
Entity + Mapper/Repository 操作。
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker

from utils.database_connection import load_database_config


_engine_cache: dict[tuple[Any, ...], Engine] = {}
_session_factory_cache: dict[tuple[Any, ...], sessionmaker[Session]] = {}
_engine_lock = Lock()
_session_factory_override: sessionmaker[Session] | None = None


def _mysql_engine_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    """生成 MySQL Engine 缓存签名。

    配置发生变化时签名也会变化，新的请求会自动创建新的 Engine。
    """

    return (
        "mysql",
        config.get("host"),
        int(config.get("port") or 3306),
        config.get("database"),
        config.get("user"),
        config.get("password"),
        config.get("charset"),
        int(config.get("pool_size") or 5),
        int(config.get("max_overflow") or 10),
        float(config.get("pool_timeout_seconds") or 5),
        int(config.get("pool_recycle_seconds") or 1800),
    )


def _create_mysql_engine(config: dict[str, Any]) -> Engine:
    """创建 MySQL SQLAlchemy Engine。"""

    url = URL.create(
        drivername="mysql+pymysql",
        username=str(config.get("user") or ""),
        password=str(config.get("password") or ""),
        host=str(config.get("host") or "127.0.0.1"),
        port=int(config.get("port") or 3306),
        database=str(config.get("database") or ""),
        query={"charset": str(config.get("charset") or "utf8mb4")},
    )
    return create_engine(
        url,
        pool_size=int(config.get("pool_size") or 5),
        max_overflow=int(config.get("max_overflow") or 10),
        pool_timeout=float(config.get("pool_timeout_seconds") or 5),
        pool_recycle=int(config.get("pool_recycle_seconds") or 1800),
        pool_pre_ping=True,
        future=True,
    )


def _is_truthy_env(name: str) -> bool:
    """判断环境变量是否表示开启状态。"""

    import os

    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _guard_real_database_in_tests(config: dict[str, Any]) -> None:
    """阻止 pytest 默认连接正式业务库。"""

    if not _is_truthy_env("AI_RAG_TESTING"):
        return
    if _is_truthy_env("RUN_REAL_MYSQL_TESTS"):
        return
    database_name = str(config.get("database") or "").strip().lower()
    if database_name == "ai_rag_agent":
        raise RuntimeError(
            "测试环境禁止连接正式 MySQL 数据库 ai_rag_agent；"
            "请使用测试 SessionFactory 覆盖，或显式设置 RUN_REAL_MYSQL_TESTS=1"
        )


def get_orm_engine() -> Engine:
    """获取 MySQL ORM Engine。"""

    config = load_database_config()
    if config["type"] != "mysql":
        raise RuntimeError("ORM层只支持 MySQL，请把 config/database.yml 的 type 配置为 mysql")
    _guard_real_database_in_tests(config["mysql"])
    signature = _mysql_engine_signature(config["mysql"])

    with _engine_lock:
        engine = _engine_cache.get(signature)
        if engine is None:
            engine = _create_mysql_engine(config["mysql"])
            _engine_cache[signature] = engine
        return engine


def get_session_factory() -> sessionmaker[Session]:
    """获取 Session 工厂，类似 MyBatis 的 SqlSessionFactory。"""

    if _session_factory_override is not None:
        return _session_factory_override
    engine = get_orm_engine()
    # Engine 已经按完整数据库配置缓存；这里直接用对象身份做 key，
    # 避免 SQLAlchemy URL 隐藏密码后导致不同数据源误共用 Session 工厂。
    signature = ("session_factory", id(engine))
    with _engine_lock:
        factory = _session_factory_cache.get(signature)
        if factory is None:
            factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
            _session_factory_cache[signature] = factory
        return factory


@contextmanager
def orm_session_context() -> Iterator[Session]:
    """打开 ORM Session，自动提交或回滚事务。"""

    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_orm_engines() -> None:
    """清理 ORM Engine 缓存，主要给测试或配置切换后使用。"""

    with _engine_lock:
        for engine in _engine_cache.values():
            engine.dispose()
        _engine_cache.clear()
        _session_factory_cache.clear()


def set_session_factory_override(factory: sessionmaker[Session]) -> None:
    """设置测试专用 Session 工厂，避免单元测试连接真实 MySQL。"""

    global _session_factory_override
    with _engine_lock:
        _session_factory_override = factory


def clear_session_factory_override() -> None:
    """清理测试专用 Session 工厂覆盖。"""

    global _session_factory_override
    with _engine_lock:
        _session_factory_override = None
