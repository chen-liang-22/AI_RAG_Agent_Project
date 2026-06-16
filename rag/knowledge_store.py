import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from utils.path_tool import get_abs_path


DEFAULT_DICTIONARY_ITEMS = [
    {
        "dictionary_code": "document_structure",
        "dictionary_name": "文档结构类型",
        "items": [
            ("text", "普通文本型", None, 1, "没有稳定结构的普通文本", {"default": True}),
            ("qa", "问答型", None, 2, "按问题和答案组织的文档结构"),
            ("numbered", "编号条目型", None, 3, "按编号条目组织的文档结构"),
        ],
    },
    {
        "dictionary_code": "split_strategy",
        "dictionary_name": "切分策略",
        "items": [
            ("recursive", "递归通用切分", None, 1, "按分隔符和长度递归切分", {"default": True}),
            ("numbered_qa", "编号问答切分", None, 2, "把编号问答切成 QA 片段"),
            ("outline_qa", "目录问答切分", None, 3, "把 PDF 书签目录中的章节和问题切成 QA 片段"),
            ("numbered_segments", "编号条目切分", None, 4, "把编号条目切成普通片段"),
        ],
    },
    {
        "dictionary_code": "model_mode",
        "dictionary_name": "回答模型档位",
        "items": [
            ("high", "高", None, 1, "高质量模型档位", {"quality": "high", "default": True}),
            ("medium", "中", None, 2, "平衡模型档位", {"quality": "medium"}),
            ("low", "低", None, 3, "低延迟模型档位", {"quality": "low", "recommendation": True}),
        ],
    },
    {
        "dictionary_code": "output_mode",
        "dictionary_name": "输出模式",
        "items": [
            ("stream", "流式", None, 1, "边生成边返回", {"mode_kind": "stream", "default": True}),
            ("once", "一次性", None, 2, "生成完成后一次性返回", {"mode_kind": "once"}),
        ],
    },
    {
        "dictionary_code": "document_status",
        "dictionary_name": "知识库文件状态",
        "items": [
            ("uploaded", "已上传", None, 1, "文件已保存但未完成入库", {"tag_type": "info"}),
            ("indexing", "入库中", None, 2, "正在解析、切分和写入向量库", {"tag_type": "warning"}),
            ("indexed", "已索引", None, 3, "已完成向量索引", {"tag_type": "success"}),
            ("failed", "入库失败", None, 4, "入库过程失败", {"tag_type": "danger"}),
            ("deleted", "已删除", None, 5, "文件已标记删除", {"tag_type": "info"}),
        ],
    },
    {
        "dictionary_code": "conversation_status",
        "dictionary_name": "会话状态",
        "items": [
            ("active", "正常", None, 1, "可继续使用的会话"),
            ("deleted", "已删除", None, 2, "已删除的会话"),
        ],
    },
    {
        "dictionary_code": "message_role",
        "dictionary_name": "消息角色",
        "items": [
            ("user", "用户", None, 1, "用户消息"),
            ("assistant", "助手", None, 2, "助手消息"),
            ("system", "系统", None, 3, "系统消息"),
        ],
    },
    {
        "dictionary_code": "content_type",
        "dictionary_name": "内容类型",
        "items": [
            ("text", "文本", None, 1, "普通文本内容"),
            ("qa", "问答片段", None, 2, "问答型知识片段"),
            ("segment", "普通片段", None, 3, "普通知识片段"),
        ],
    },
    {
        "dictionary_code": "service_status",
        "dictionary_name": "服务状态",
        "items": [
            ("ok", "正常", None, 1, "服务可用"),
            ("degraded", "降级", None, 2, "部分依赖不可用"),
            ("unavailable", "不可用", None, 3, "服务或依赖不可用"),
        ],
    },
    {
        "dictionary_code": "knowledge_result_status",
        "dictionary_name": "知识库操作结果",
        "items": [
            ("indexed", "已索引", None, 1, "文件入库成功", {"result_kind": "indexed"}),
            ("duplicate", "重复", None, 2, "存在相同内容文件", {"result_kind": "duplicate"}),
            ("failed", "失败", None, 3, "操作失败", {"result_kind": "failed"}),
            ("ok", "成功", None, 4, "批量操作全部成功", {"result_kind": "ok"}),
            ("partial_failed", "部分失败", None, 5, "批量操作存在失败项", {"result_kind": "partial_failed"}),
        ],
    },
    {
        "dictionary_code": "preview_type",
        "dictionary_name": "预览类型",
        "items": [
            ("text", "TXT 文本", None, 1, "TXT 文件预览"),
            ("pdf_text", "PDF 文本", None, 2, "PDF 提取文本预览"),
            ("unsupported", "不支持", None, 3, "暂不支持预览"),
        ],
    },
]


def utc_now_text() -> str:
    """返回统一格式的 UTC 时间字符串。"""

    return datetime.utcnow().isoformat(timespec="seconds")


class KnowledgeStore:
    """SQLite 业务元数据存储。

    最终设计里，SQLite 只保存业务状态：
    - documents：文件管理、索引状态、版本、chunk_count。
    - conversations / conversation_messages：会话历史。

    知识正文、FAQ、向量和可检索 payload 都以 Qdrant 为准。
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_abs_path("storage/knowledge.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """打开 SQLite 连接，并在 SQL 执行结束后自动提交和关闭。"""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_md5 TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    collection_name TEXT NOT NULL DEFAULT 'agent',
                    document_type TEXT NOT NULL DEFAULT 'text',
                    split_strategy TEXT NOT NULL DEFAULT 'recursive',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    summary TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT 'text',
                    model_name TEXT,
                    token_count INTEGER,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
                    UNIQUE(conversation_id, sequence_no)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_file_md5
                ON documents(file_md5)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dictionary_groups (
                    dictionary_code TEXT PRIMARY KEY,
                    dictionary_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dictionary_items (
                    dictionary_item_id TEXT PRIMARY KEY,
                    dictionary_code TEXT NOT NULL,
                    dictionary_name TEXT NOT NULL,
                    item_code TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    parent_item_id TEXT,
                    item_level INTEGER NOT NULL DEFAULT 1,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    description TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dictionary_code, item_code)
                )
                """
            )
            self._ensure_document_columns(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_collection
                ON documents(collection_name)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_status
                ON conversations(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
                ON conversation_messages(conversation_id, sequence_no)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON conversation_messages(conversation_id, created_at)
                """
            )

            # 旧知识表不再作为知识答案来源，启动时清理，避免继续双写或误查。
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dictionary_items_code_parent
                ON dictionary_items(dictionary_code, parent_item_id, sort_order)
                """
            )
            self.seed_default_dictionaries(conn)
            self._sync_dictionary_groups_from_items(conn)
            conn.execute("DROP TABLE IF EXISTS qa_items")
            conn.execute("DROP TABLE IF EXISTS document_segments")
            conn.execute("DROP TABLE IF EXISTS knowledge_units")

    def seed_default_dictionaries(self, conn: sqlite3.Connection) -> None:
        """初始化系统默认字典项，已有字典项只更新展示信息。"""

        now = utc_now_text()
        for dictionary in DEFAULT_DICTIONARY_ITEMS:
            dictionary_code = dictionary["dictionary_code"]
            dictionary_name = dictionary["dictionary_name"]
            conn.execute(
                """
                INSERT INTO dictionary_groups (
                    dictionary_code, dictionary_name, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dictionary_code) DO UPDATE SET
                    dictionary_name = excluded.dictionary_name,
                    updated_at = excluded.updated_at
                """,
                (dictionary_code, dictionary_name, now, now),
            )
            item_id_by_code: dict[str, str] = {}
            for item in dictionary["items"]:
                item_code, item_name, parent_code, sort_order, description = item[:5]
                metadata = item[5] if len(item) > 5 else None
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                existing = conn.execute(
                    """
                    SELECT dictionary_item_id
                    FROM dictionary_items
                    WHERE dictionary_code = ? AND item_code = ?
                    """,
                    (dictionary_code, item_code),
                ).fetchone()
                parent_item_id = item_id_by_code.get(parent_code or "")
                item_level = 1 if parent_item_id is None else 2
                if existing:
                    dictionary_item_id = existing["dictionary_item_id"]
                    conn.execute(
                        """
                        UPDATE dictionary_items
                        SET dictionary_name = ?, item_name = ?, parent_item_id = ?,
                            item_level = ?, sort_order = ?, description = ?, metadata_json = ?, updated_at = ?
                        WHERE dictionary_item_id = ?
                        """,
                        (
                            dictionary_name,
                            item_name,
                            parent_item_id,
                            item_level,
                            sort_order,
                            description,
                            metadata_json,
                            now,
                            dictionary_item_id,
                        ),
                    )
                else:
                    dictionary_item_id = f"dict_{uuid.uuid4().hex}"
                    conn.execute(
                        """
                        INSERT INTO dictionary_items (
                            dictionary_item_id, dictionary_code, dictionary_name,
                            item_code, item_name, parent_item_id, item_level,
                            sort_order, enabled, description, metadata_json,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            dictionary_item_id,
                            dictionary_code,
                            dictionary_name,
                            item_code,
                            item_name,
                            parent_item_id,
                            item_level,
                            sort_order,
                            description,
                            metadata_json,
                            now,
                            now,
                        ),
                    )
                item_id_by_code[item_code] = dictionary_item_id

    def _sync_dictionary_groups_from_items(self, conn: sqlite3.Connection) -> None:
        """把旧字典项表里已有的分组同步到父级字典表。"""

        now = utc_now_text()
        rows = conn.execute(
            """
            SELECT dictionary_code, MIN(dictionary_name) AS dictionary_name
            FROM dictionary_items
            GROUP BY dictionary_code
            """
        ).fetchall()
        for row in rows:
            conn.execute(
                """
                INSERT INTO dictionary_groups (
                    dictionary_code, dictionary_name, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(dictionary_code) DO NOTHING
                """,
                (row["dictionary_code"], row["dictionary_name"] or row["dictionary_code"], now, now),
            )

    def list_dictionary_items(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """查询字典项列表，支持按字典编码过滤。"""

        with self.connect() as conn:
            if dictionary_code:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM dictionary_items
                    WHERE dictionary_code = ?
                    ORDER BY dictionary_code, item_level, sort_order, item_code
                    """,
                    (dictionary_code,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM dictionary_items
                    ORDER BY dictionary_code, item_level, sort_order, item_code
                    """
                ).fetchall()
        return [dict(row) for row in rows]

    def list_dictionary_groups(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """查询父级字典列表。"""

        with self.connect() as conn:
            if dictionary_code:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM dictionary_groups
                    WHERE dictionary_code = ?
                    ORDER BY dictionary_code
                    """,
                    (dictionary_code,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM dictionary_groups
                    ORDER BY dictionary_code
                    """
                ).fetchall()
        return [dict(row) for row in rows]

    def create_dictionary_group(self, dictionary_code: str, dictionary_name: str) -> dict[str, Any]:
        """新增父级字典。"""

        now = utc_now_text()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO dictionary_groups (
                    dictionary_code, dictionary_name, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (dictionary_code, dictionary_name, now, now),
            )
            row = conn.execute(
                "SELECT * FROM dictionary_groups WHERE dictionary_code = ?",
                (dictionary_code,),
            ).fetchone()
        return dict(row)

    def update_dictionary_group(self, dictionary_code: str, dictionary_name: str) -> dict[str, Any] | None:
        """修改父级字典名称，并同步到字典项冗余字段。"""

        now = utc_now_text()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE dictionary_groups
                SET dictionary_name = ?, updated_at = ?
                WHERE dictionary_code = ?
                """,
                (dictionary_name, now, dictionary_code),
            )
            if cursor.rowcount == 0:
                return None
            conn.execute(
                """
                UPDATE dictionary_items
                SET dictionary_name = ?, updated_at = ?
                WHERE dictionary_code = ?
                """,
                (dictionary_name, now, dictionary_code),
            )
            row = conn.execute(
                "SELECT * FROM dictionary_groups WHERE dictionary_code = ?",
                (dictionary_code,),
            ).fetchone()
        return dict(row)

    def get_dictionary_item(self, dictionary_item_id: str) -> dict[str, Any] | None:
        """按 ID 查询单个字典项。"""

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM dictionary_items
                WHERE dictionary_item_id = ?
                """,
                (dictionary_item_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_dictionary_item(
            self,
            *,
            dictionary_code: str,
            dictionary_name: str,
            item_code: str,
            item_name: str,
            parent_item_id: str | None = None,
            sort_order: int = 0,
            enabled: bool = True,
            description: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """新增字典项，并根据父级自动计算层级。"""

        dictionary_item_id = f"dict_{uuid.uuid4().hex}"
        now = utc_now_text()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        with self.connect() as conn:
            group = conn.execute(
                "SELECT dictionary_name FROM dictionary_groups WHERE dictionary_code = ?",
                (dictionary_code,),
            ).fetchone()
            if group is None:
                raise ValueError("父级字典不存在，请先新增父级字典")
            dictionary_name = str(group["dictionary_name"])
            parent_level = self._get_dictionary_parent_level(conn, dictionary_code, parent_item_id)
            conn.execute(
                """
                INSERT INTO dictionary_items (
                    dictionary_item_id, dictionary_code, dictionary_name,
                    item_code, item_name, parent_item_id, item_level,
                    sort_order, enabled, description, metadata_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dictionary_item_id,
                    dictionary_code,
                    dictionary_name,
                    item_code,
                    item_name,
                    parent_item_id,
                    parent_level + 1,
                    sort_order,
                    1 if enabled else 0,
                    description,
                    metadata_json,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM dictionary_items WHERE dictionary_item_id = ?",
                (dictionary_item_id,),
            ).fetchone()
        return dict(row)

    def update_dictionary_item(
            self,
            dictionary_item_id: str,
            *,
            dictionary_name: str | None = None,
            item_code: str | None = None,
            item_name: str | None = None,
            parent_item_id: str | None = None,
            sort_order: int | None = None,
            enabled: bool | None = None,
            description: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """修改字典项。"""

        now = utc_now_text()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata is not None else None
        with self.connect() as conn:
            current = conn.execute(
                "SELECT * FROM dictionary_items WHERE dictionary_item_id = ?",
                (dictionary_item_id,),
            ).fetchone()
            if current is None:
                return None

            current_dict = dict(current)
            target_parent_item_id = parent_item_id
            if target_parent_item_id == dictionary_item_id:
                raise ValueError("字典项不能选择自己作为父级")
            parent_level = self._get_dictionary_parent_level(
                conn,
                current_dict["dictionary_code"],
                target_parent_item_id,
            )
            conn.execute(
                """
                UPDATE dictionary_items
                SET dictionary_name = ?, item_code = ?, item_name = ?, parent_item_id = ?,
                    item_level = ?, sort_order = ?, enabled = ?, description = ?,
                    metadata_json = ?, updated_at = ?
                WHERE dictionary_item_id = ?
                """,
                (
                    dictionary_name if dictionary_name is not None else current_dict["dictionary_name"],
                    item_code if item_code is not None else current_dict["item_code"],
                    item_name if item_name is not None else current_dict["item_name"],
                    target_parent_item_id,
                    parent_level + 1,
                    sort_order if sort_order is not None else current_dict["sort_order"],
                    1 if (enabled if enabled is not None else bool(current_dict["enabled"])) else 0,
                    description,
                    metadata_json if metadata is not None else current_dict.get("metadata_json"),
                    now,
                    dictionary_item_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM dictionary_items WHERE dictionary_item_id = ?",
                (dictionary_item_id,),
            ).fetchone()
        return dict(row)

    def set_dictionary_item_enabled(self, dictionary_item_id: str, enabled: bool) -> dict[str, Any] | None:
        """启用或禁用字典项。"""

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE dictionary_items
                SET enabled = ?, updated_at = ?
                WHERE dictionary_item_id = ?
                """,
                (1 if enabled else 0, utc_now_text(), dictionary_item_id),
            )
            row = conn.execute(
                "SELECT * FROM dictionary_items WHERE dictionary_item_id = ?",
                (dictionary_item_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_dictionary_item(self, dictionary_item_id: str) -> bool:
        """删除没有子级的字典项。"""

        with self.connect() as conn:
            child_count = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM dictionary_items
                WHERE parent_item_id = ?
                """,
                (dictionary_item_id,),
            ).fetchone()["total"]
            if int(child_count) > 0:
                raise ValueError("当前字典项存在子级，请先删除子级")
            cursor = conn.execute(
                "DELETE FROM dictionary_items WHERE dictionary_item_id = ?",
                (dictionary_item_id,),
            )
        return cursor.rowcount > 0

    def delete_dictionary_group(self, dictionary_code: str) -> int:
        """删除某个父级字典分组下的全部字典项。"""

        with self.connect() as conn:
            group_cursor = conn.execute(
                "DELETE FROM dictionary_groups WHERE dictionary_code = ?",
                (dictionary_code,),
            )
            item_cursor = conn.execute(
                "DELETE FROM dictionary_items WHERE dictionary_code = ?",
                (dictionary_code,),
            )
        return int(group_cursor.rowcount) + int(item_cursor.rowcount)

    def _get_dictionary_parent_level(
            self,
            conn: sqlite3.Connection,
            dictionary_code: str,
            parent_item_id: str | None,
    ) -> int:
        """查询父级层级；没有父级时返回 0。"""

        if parent_item_id is None:
            return 0
        parent = conn.execute(
            """
            SELECT item_level
            FROM dictionary_items
            WHERE dictionary_item_id = ? AND dictionary_code = ?
            """,
            (parent_item_id, dictionary_code),
        ).fetchone()
        if parent is None:
            raise ValueError("父级字典项不存在或不属于当前字典")
        return int(parent["item_level"])

    def list_enabled_dictionary_codes(self, dictionary_code: str) -> list[str]:
        """查询某个字典下已启用的字典项编码。"""

        rows = self.list_dictionary_items(dictionary_code=dictionary_code)
        # 遍历该字典下的所有字典项，只保留 enabled=1 的项，并返回它们的 item_code 列表。
        return [str(row["item_code"]) for row in rows if int(row.get("enabled") or 0) == 1]

    def get_default_dictionary_code(self, dictionary_code: str) -> str:
        """查询某个字典的默认编码，默认取启用且排序最靠前的字典项。"""

        # list_enabled_dictionary_codes 返回的是已按 sort_order 排好的启用编码列表。
        codes = self.list_enabled_dictionary_codes(dictionary_code)
        if not codes:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        # codes[0] 表示取列表中的第一项，也就是默认字典项。
        return codes[0]

    def get_dictionary_code_by_metadata(self, dictionary_code: str, metadata_key: str, metadata_value: Any) -> str | None:
        """按字典项 metadata 查询编码，用于把默认项、推荐项等业务含义放到字典表维护。"""

        rows = self.list_dictionary_items(dictionary_code=dictionary_code)
        for row in rows:
            if int(row.get("enabled") or 0) != 1:
                continue
            raw_metadata = row.get("metadata_json")
            if not raw_metadata:
                continue
            try:
                metadata = json.loads(str(raw_metadata))
            except json.JSONDecodeError:
                continue
            if metadata.get(metadata_key) == metadata_value:
                return str(row["item_code"])
        return None

    def normalize_dictionary_code(self, dictionary_code: str, value: str | None = None) -> str:
        """按字典表归一化编码；非法或空值时返回该字典的默认编码。"""

        # 查询当前字典中所有启用编码，并转成 set，方便后续快速判断传入值是否合法。
        enabled_codes = set(self.list_enabled_dictionary_codes(dictionary_code))
        # 获取当前字典的默认编码；通常是启用且排序最靠前的字典项。
        default_code = self.get_default_dictionary_code(dictionary_code)
        # 优先使用传入 value；如果 value 为空，则使用默认编码。
        # 然后统一转字符串、去掉前后空格、转成小写，避免大小写或空格导致匹配失败。
        normalized_value = str(value or default_code).strip().lower()
        # 如果规范化后的值存在于启用编码集合中，说明传入值合法，直接返回该值。
        if normalized_value in enabled_codes:
            return normalized_value
        # 如果传入值为空或不合法，则兜底返回默认编码。
        return default_code

    @staticmethod
    def _ensure_document_columns(conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        migrations = {
            "collection_name": "ALTER TABLE documents ADD COLUMN collection_name TEXT NOT NULL DEFAULT 'agent'",
            "document_type": "ALTER TABLE documents ADD COLUMN document_type TEXT NOT NULL DEFAULT 'text'",
            "split_strategy": "ALTER TABLE documents ADD COLUMN split_strategy TEXT NOT NULL DEFAULT 'recursive'",
        }
        for column_name, statement in migrations.items():
            if column_name not in columns:
                conn.execute(statement)

    @staticmethod
    def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def create_document(
            self,
            *,
            document_id: str,
            filename: str,
            file_path: str,
            file_type: str,
            file_md5: str,
            file_size: int,
            status: str = "uploaded",
            collection_name: str = "agent",
            document_type: str = "text",
            split_strategy: str = "recursive",
    ) -> dict[str, Any]:
        now = utc_now_text()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_path, file_type, file_md5,
                    file_size, status, version, chunk_count, collection_name,
                    document_type, split_strategy, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    file_path,
                    file_type,
                    file_md5,
                    file_size,
                    status,
                    collection_name,
                    document_type,
                    split_strategy,
                    now,
                    now,
                ),
            )

        document = self.get_document(document_id)
        if document is None:
            raise RuntimeError(f"Document {document_id} was not created")
        return document

    def find_active_document_by_md5(
            self,
            file_md5: str,
            collection_name: str | None = None,
    ) -> dict[str, Any] | None:
        """按文件 MD5 查找未删除文档；传入 collection 时只在该 collection 内去重。"""

        if collection_name:
            with self.connect() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE file_md5 = ? AND collection_name = ? AND status != 'deleted'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (file_md5, collection_name),
                ).fetchone()
            return self.row_to_dict(row)

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE file_md5 = ? AND status != 'deleted'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (file_md5,),
            ).fetchone()
        return self.row_to_dict(row)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self.row_to_dict(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                WHERE status != 'deleted'
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def update_document_status(
            self,
            document_id: str,
            status: str,
            *,
            chunk_count: int | None = None,
            error_message: str | None = None,
            increment_version: bool = False,
            collection_name: str | None = None,
            document_type: str | None = None,
            split_strategy: str | None = None,
    ) -> None:
        document = self.get_document(document_id)
        if document is None:
            raise ValueError(f"Document {document_id} does not exist")

        version = int(document["version"]) + 1 if increment_version else int(document["version"])
        final_chunk_count = int(document["chunk_count"]) if chunk_count is None else chunk_count
        final_collection_name = collection_name or document.get("collection_name") or "agent"
        final_document_type = document_type or document.get("document_type") or "text"
        final_split_strategy = split_strategy or document.get("split_strategy") or "recursive"

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?, version = ?,
                    collection_name = ?, document_type = ?, split_strategy = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (
                    status,
                    final_chunk_count,
                    error_message,
                    version,
                    final_collection_name,
                    final_document_type,
                    final_split_strategy,
                    utc_now_text(),
                    document_id,
                ),
            )

    def mark_document_deleted(self, document_id: str) -> None:
        self.update_document_status(document_id, "deleted")

    def ensure_conversation(
            self,
            *,
            conversation_id: str | None = None,
            user_id: str | None = None,
            title: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_conversation_id = (conversation_id or "").strip()
        if clean_conversation_id:
            existing = self.get_conversation(clean_conversation_id)
            if existing is not None and existing.get("status") != "deleted":
                return existing
            if existing is not None and existing.get("status") == "deleted":
                clean_conversation_id = f"conv_{uuid.uuid4().hex}"
        else:
            clean_conversation_id = f"conv_{uuid.uuid4().hex}"

        now = utc_now_text()
        clean_title = (title or "").strip()[:80] or None
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, user_id, title, status, message_count, summary,
                    metadata_json, created_at, updated_at, last_message_at
                )
                VALUES (?, ?, ?, 'active', 0, NULL, ?, ?, ?, NULL)
                """,
                (clean_conversation_id, user_id, clean_title, metadata_json, now, now),
            )

        conversation = self.get_conversation(clean_conversation_id)
        if conversation is None:
            raise RuntimeError(f"Conversation {clean_conversation_id} was not created")
        return conversation

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return self.row_to_dict(row)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除聊天记录。

        会话主表保留一条 deleted 状态记录，避免历史 ID 误复用；
        消息明细直接删除，避免已删除会话的问答正文继续留在数据库里。
        """

        now = utc_now_text()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None or row["status"] == "deleted":
                return False

            conn.execute(
                "DELETE FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                """
                UPDATE conversations
                SET status = 'deleted',
                    message_count = 0,
                    updated_at = ?,
                    last_message_at = ?
                WHERE conversation_id = ?
                """,
                (now, now, conversation_id),
            )

        return True

    def list_conversations(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            user_id: str | None = None,
            keyword: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询会话列表。"""

        final_page = max(1, int(page))
        final_page_size = max(1, min(int(page_size), 50))
        offset = (final_page - 1) * final_page_size
        conditions = ["status != 'deleted'"]
        params: list[Any] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{self._escape_like_keyword(clean_keyword)}%"
            conditions.append(
                "(title LIKE ? ESCAPE '\\' OR user_id LIKE ? ESCAPE '\\' OR conversation_id LIKE ? ESCAPE '\\')"
            )
            params.extend([like_keyword, like_keyword, like_keyword])

        where_sql = " AND ".join(conditions)
        with self.connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM conversations WHERE {where_sql}",
                params,
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT *
                FROM conversations
                WHERE {where_sql}
                ORDER BY COALESCE(last_message_at, updated_at, created_at) DESC
                LIMIT ? OFFSET ?
                """,
                [*params, final_page_size, offset],
            ).fetchall()

        return [dict(row) for row in rows], int(total_row["total"] if total_row else 0)

    @staticmethod
    def _escape_like_keyword(keyword: str) -> str:
        """转义 LIKE 通配符，避免用户输入被当成模式语法。"""

        return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def list_recent_messages(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        final_limit = max(1, min(int(limit), 100))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY sequence_no DESC
                LIMIT ?
                """,
                (conversation_id, final_limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """查询某个会话的全部消息。"""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY sequence_no ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
            self,
            *,
            conversation_id: str,
            role: str,
            content: str,
            content_type: str = "text",
            model_name: str | None = None,
            token_count: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.get_conversation(conversation_id) is None:
            self.ensure_conversation(conversation_id=conversation_id)

        now = utc_now_text()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        message_id = f"msg_{uuid.uuid4().hex}"

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence_no
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            sequence_no = int(row["next_sequence_no"])
            conn.execute(
                """
                INSERT INTO conversation_messages (
                    message_id, conversation_id, sequence_no, role, content, content_type,
                    model_name, token_count, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    sequence_no,
                    role,
                    content,
                    content_type,
                    model_name,
                    token_count,
                    metadata_json,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE conversations
                SET message_count = message_count + 1,
                    updated_at = ?,
                    last_message_at = ?
                WHERE conversation_id = ?
                """,
                (now, now, conversation_id),
            )

        return self.get_message(message_id) or {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "sequence_no": sequence_no,
            "role": role,
            "content": content,
            "content_type": content_type,
            "model_name": model_name,
            "token_count": token_count,
            "metadata_json": metadata_json,
            "created_at": now,
        }

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return self.row_to_dict(row)

    def save_chat_exchange(
            self,
            *,
            conversation_id: str,
            user_message: str,
            assistant_message: str,
            model_name: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        self.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
        )
        self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            model_name=model_name,
            metadata=metadata,
        )

    # 下面这些方法是旧知识表接口的兼容壳。知识检索和知识答案不再访问 SQLite。
    def replace_units(self, document_id: str, units: list[dict[str, Any]]) -> None:
        return None

    def delete_units(self, document_id: str) -> None:
        return None

    def replace_segments_and_qas(
            self,
            document_id: str,
            segments: list[dict[str, Any]],
            qa_items: list[dict[str, Any]],
    ) -> None:
        return None

    def search_segments_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict[str, Any]]:
        return []

    def find_qa_document(self, document_hint: str | None = None) -> dict[str, Any] | None:
        return None

    def search_qa_items_by_question(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return []

    def search_units_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict[str, Any]]:
        return []
