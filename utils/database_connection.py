"""业务数据库配置读取。

项目业务数据已经统一使用 MySQL + SQLAlchemy ORM。
这个模块只保留配置解析和数据库异常别名，真实连接和事务由
`infrastructure.orm_session` 统一管理。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.exc import DatabaseError, IntegrityError

from utils.path_tool import get_abs_path


IntegrityErrorTypes = (IntegrityError,)
"""数据库唯一键、外键等完整性错误类型。"""

DatabaseErrorTypes = (DatabaseError,)
"""SQLAlchemy 数据库错误类型。"""


def _load_env_file(env_path: str = get_abs_path(".env")) -> None:
    """轻量读取 .env，避免数据库密码只写在 .env 时无法被读取。"""

    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_database_config() -> dict[str, Any]:
    """读取 MySQL 配置，并允许环境变量覆盖关键连接信息。"""

    _load_env_file()
    config_path = Path(get_abs_path("config/database.yml"))
    config: dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    mysql_config = dict(config.get("mysql") or {})
    password_env = str(mysql_config.get("password_env") or "MYSQL_PASSWORD")
    mysql_config.update({
        "host": os.getenv("MYSQL_HOST", mysql_config.get("host", "127.0.0.1")),
        "port": int(os.getenv("MYSQL_PORT", mysql_config.get("port", 3306))),
        "database": os.getenv("MYSQL_DATABASE", mysql_config.get("database", "ai_rag_agent")),
        "user": os.getenv("MYSQL_USER", mysql_config.get("user", "root")),
        "password": os.getenv(password_env, os.getenv("MYSQL_PASSWORD", mysql_config.get("password", ""))),
        "charset": os.getenv("MYSQL_CHARSET", mysql_config.get("charset", "utf8mb4")),
        "pool_size": int(os.getenv("MYSQL_POOL_SIZE", mysql_config.get("pool_size", 5))),
        "max_overflow": int(os.getenv("MYSQL_MAX_OVERFLOW", mysql_config.get("max_overflow", 10))),
        "pool_timeout_seconds": float(
            os.getenv("MYSQL_POOL_TIMEOUT_SECONDS", mysql_config.get("pool_timeout_seconds", 5))
        ),
        "pool_recycle_seconds": int(
            os.getenv("MYSQL_POOL_RECYCLE_SECONDS", mysql_config.get("pool_recycle_seconds", 1800))
        ),
    })

    return {
        "type": "mysql",
        "mysql": mysql_config,
    }
