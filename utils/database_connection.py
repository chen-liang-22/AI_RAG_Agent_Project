"""业务数据库连接兼容层。

运行时统一连接 MySQL；单元测试显式传入 db_path 时仍可使用临时 SQLite。
这一层让仓储里的 `?` 占位符 SQL 可以在 MySQL 下继续工作，降低一次性迁移风险。
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from utils.path_tool import get_abs_path

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:  # pragma: no cover - 仅在未安装 MySQL 驱动时触发。
    pymysql = None
    DictCursor = None


MYSQL_CREATE_SKIP_MESSAGE = "MySQL 模式下跳过本地兼容库自动建表，请先执行 docs/mysql初始化建表和基础数据.sql"
IntegrityErrorTypes = (sqlite3.IntegrityError,)
DatabaseErrorTypes = (sqlite3.DatabaseError,)
if pymysql is not None:
    IntegrityErrorTypes = (sqlite3.IntegrityError, pymysql.err.IntegrityError)
    DatabaseErrorTypes = (sqlite3.DatabaseError, pymysql.err.DatabaseError)


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
    """读取数据库配置，并允许环境变量覆盖关键连接信息。"""

    _load_env_file()
    config_path = Path(get_abs_path("config/database.yml"))
    config: dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    configured_database_type = os.getenv("DATABASE_TYPE") or str(config.get("type") or "mysql")
    database_type = configured_database_type.lower()
    if database_type == "sqlite" and os.getenv("AI_RAG_ALLOW_TEST_SQLITE") != "1":
        # SQLite 只允许测试显式打开，避免本地运行时误把业务数据写回临时库。
        database_type = "mysql"
    mysql_config = dict(config.get("mysql") or {})
    password_env = str(mysql_config.get("password_env") or "MYSQL_PASSWORD")
    mysql_config.update({
        "host": os.getenv("MYSQL_HOST", mysql_config.get("host", "127.0.0.1")),
        "port": int(os.getenv("MYSQL_PORT", mysql_config.get("port", 3306))),
        "database": os.getenv("MYSQL_DATABASE", mysql_config.get("database", "ai_rag_agent")),
        "user": os.getenv("MYSQL_USER", mysql_config.get("user", "root")),
        "password": os.getenv(password_env, os.getenv("MYSQL_PASSWORD", mysql_config.get("password", ""))),
        "charset": os.getenv("MYSQL_CHARSET", mysql_config.get("charset", "utf8mb4")),
    })

    sqlite_config = dict(config.get("sqlite") or {})
    default_sqlite_path = os.path.join(tempfile.gettempdir(), "ai_rag_agent_test_knowledge.db")
    sqlite_config["path"] = os.getenv("SQLITE_DB_PATH", sqlite_config.get("path") or default_sqlite_path)
    return {
        "type": database_type,
        "mysql": mysql_config,
        "sqlite": sqlite_config,
    }


def is_mysql_runtime(db_path: str | None = None) -> bool:
    """判断当前仓储是否应使用 MySQL。

    只要显式传入 db_path，就认为调用方需要临时兼容库，主要用于单元测试。
    """

    if db_path:
        return False
    return load_database_config()["type"] == "mysql"


def _convert_sqlite_placeholders(sql: str) -> str:
    """把通用的 ? 占位符转换成 PyMySQL 使用的 %s。

    项目里的 SQL 占位符都在 SQL 字符串层，用户值通过参数传入。
    这里跳过单双引号内的问号，避免误改文本字面量。
    """

    chars: list[str] = []
    in_single_quote = False
    in_double_quote = False
    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'" and not in_double_quote:
            chars.append(char)
            if in_single_quote and index + 1 < len(sql) and sql[index + 1] == "'":
                chars.append(sql[index + 1])
                index += 2
                continue
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            chars.append(char)
            in_double_quote = not in_double_quote
        elif char == "?" and not in_single_quote and not in_double_quote:
            chars.append("%s")
        else:
            chars.append(char)
        index += 1
    return "".join(chars)


class MySqlConnection:
    """仿 DB-API Connection 的 MySQL 连接包装。"""

    def __init__(self, config: dict[str, Any]):
        if pymysql is None or DictCursor is None:
            raise RuntimeError("缺少 MySQL 驱动 PyMySQL，请先执行 pip install -r requirements.txt")
        self._conn = pymysql.connect(
            host=str(config["host"]),
            port=int(config["port"]),
            user=str(config["user"]),
            password=str(config.get("password") or ""),
            database=str(config["database"]),
            charset=str(config.get("charset") or "utf8mb4"),
            cursorclass=DictCursor,
            autocommit=False,
        )

    def execute(self, sql: str, parameters: Any = None):
        """执行 SQL，并返回 PyMySQL DictCursor。"""

        cursor = self._conn.cursor()
        mysql_sql = _convert_sqlite_placeholders(sql)
        cursor.execute(mysql_sql, parameters or ())
        return cursor

    def commit(self) -> None:
        """提交事务。"""

        self._conn.commit()

    def rollback(self) -> None:
        """回滚事务。"""

        self._conn.rollback()

    def close(self) -> None:
        """关闭连接。"""

        self._conn.close()


def open_database_connection(db_path: str | None = None):
    """打开业务数据库连接。"""

    if db_path:
        sqlite_conn = sqlite3.connect(db_path)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_conn.execute("PRAGMA foreign_keys = ON")
        return sqlite_conn

    config = load_database_config()
    if config["type"] == "sqlite":
        sqlite_path = get_abs_path(str(config["sqlite"]["path"]))
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_conn.execute("PRAGMA foreign_keys = ON")
        return sqlite_conn

    return MySqlConnection(config["mysql"])


@contextmanager
def database_context(db_path: str | None = None) -> Iterator[Any]:
    """统一事务上下文，正常提交、异常回滚、最后关闭连接。"""

    conn = open_database_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
