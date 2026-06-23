"""把当前 SQLite 业务数据导出为 MySQL INSERT 脚本。

这个脚本用于一次性迁移现有业务数据。它不会导出已经删除的旧切片表，
也不会导出 Qdrant 向量数据；向量库仍按 Qdrant 自己的 collection 保存。
"""

from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_tool import get_abs_path


SOURCE_DB_PATH = Path(get_abs_path("storage/knowledge.db"))
OUTPUT_PATH = Path("docs/mysql业务数据导入.sql")

# 按外键依赖顺序导出，父表在前，子表在后。
TABLE_ORDER = [
    "documents",
    "conversations",
    "conversation_messages",
    "dictionary_items",
    "exam_sessions",
    "exam_questions",
    "training_knowledge_batches",
    "training_plans",
    "training_role_profiles",
    "training_goal_settings",
    "sales_training_sessions",
    "sales_training_turns",
    "sales_training_scores",
]

DATETIME_COLUMNS = {
    "created_at",
    "updated_at",
    "last_message_at",
    "completed_at",
    "answered_at",
    "started_at",
    "ended_at",
    "submitted_at",
}


def sql_quote(value: Any, column_name: str | None = None) -> str:
    """把 SQLite 字段值转换成 MySQL 字面量。"""

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if column_name in DATETIME_COLUMNS and "T" in text:
        text = text.replace("T", " ", 1)
    text = text.replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """判断 SQLite 表是否存在。"""

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    """读取 SQLite 表字段顺序。"""

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row["name"]) for row in rows]


def render_table_data(conn: sqlite3.Connection, table_name: str) -> tuple[str, int]:
    """生成单表 INSERT 语句。"""

    if not table_exists(conn, table_name):
        return f"-- 跳过不存在的表：{table_name}\n", 0
    columns = table_columns(conn, table_name)
    if not columns:
        return f"-- 跳过无字段表：{table_name}\n", 0

    quoted_columns = ", ".join(f"`{column}`" for column in columns)
    rows = conn.execute(f"SELECT {quoted_columns} FROM `{table_name}`").fetchall()
    lines = [f"-- 表数据：{table_name}，共 {len(rows)} 条"]
    if not rows:
        lines.append("")
        return "\n".join(lines), 0

    for row in rows:
        values = ", ".join(sql_quote(row[column], column) for column in columns)
        update_columns = [column for column in columns if column != columns[0]]
        update_sql = ", ".join(f"`{column}` = VALUES(`{column}`)" for column in update_columns)
        lines.append(
            f"INSERT INTO `{table_name}` ({quoted_columns}) VALUES ({values}) "
            f"ON DUPLICATE KEY UPDATE {update_sql};"
        )
    lines.append("")
    return "\n".join(lines), len(rows)


def main() -> None:
    """导出当前 SQLite 业务数据。"""

    if not SOURCE_DB_PATH.exists():
        raise FileNotFoundError(f"SQLite 数据库不存在：{SOURCE_DB_PATH}")

    conn = sqlite3.connect(SOURCE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        total = 0
        sections = [
            "-- AI_RAG_Agent_Project SQLite -> MySQL 业务数据导入脚本。",
            "-- 使用前请先执行 docs/mysql初始化建表和基础数据.sql。",
            "-- 本脚本只导出现有 SQLite 业务表数据，不包含 Qdrant 向量数据。",
            "-- 注意：本脚本会先清空下列业务表，再导入 SQLite 当前数据，适合一次性迁移。",
            "",
            "USE ai_rag_agent;",
            "SET NAMES utf8mb4;",
            "SET FOREIGN_KEY_CHECKS = 0;",
            "START TRANSACTION;",
            "",
            "-- 按外键依赖反向清空表，避免默认字典主键和 SQLite 原始主键混用导致层级错乱。",
        ]
        for table_name in reversed(TABLE_ORDER):
            if table_exists(conn, table_name):
                sections.append(f"DELETE FROM `{table_name}`;")
        sections.append("")
        for table_name in TABLE_ORDER:
            section, count = render_table_data(conn, table_name)
            sections.append(section)
            total += count
        sections.extend([
            "COMMIT;",
            "SET FOREIGN_KEY_CHECKS = 1;",
            f"-- 导出完成，总记录数：{total}",
            "",
        ])
        OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    finally:
        conn.close()

    print(f"已导出：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
