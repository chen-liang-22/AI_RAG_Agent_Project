import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from utils.path_tool import get_abs_path


def utc_now_text() -> str:
    """返回统一格式的时间字符串。"""

    return datetime.utcnow().isoformat(timespec="seconds")


class KnowledgeStore:
    """知识库元数据存储。

    SQLite 只保存文件和知识单元的结构化信息。
    Qdrant 只保存向量索引和检索 payload。
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_abs_path("storage/knowledge.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """打开 SQLite 连接，并在 SQL 执行结束后自动提交和关闭。

        注意：sqlite3.Connection 自己也能放进 with 语句里，
        但它的 with 只负责 commit/rollback，不负责 close。
        Windows 上如果连接不关闭，数据库文件会一直被进程占用。
        """

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_units (
                    unit_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    title TEXT,
                    question TEXT,
                    answer TEXT,
                    content TEXT NOT NULL,
                    category TEXT,
                    tags TEXT,
                    source_page INTEGER,
                    unit_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_segments (
                    segment_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    page_no INTEGER,
                    heading_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_items (
                    faq_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    segment_id TEXT,
                    question_no INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT,
                    category TEXT,
                    tags_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id),
                    FOREIGN KEY(segment_id) REFERENCES document_segments(segment_id)
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
                CREATE INDEX IF NOT EXISTS idx_knowledge_units_document_id
                ON knowledge_units(document_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segments_document_id
                ON document_segments(document_id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_segments_document_index
                ON document_segments(document_id, segment_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_faq_document_id
                ON faq_items(document_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_faq_question_no
                ON faq_items(document_id, question_no)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_faq_category
                ON faq_items(category)
                """
            )

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
    ) -> dict[str, Any]:
        now = utc_now_text()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_path, file_type, file_md5,
                    file_size, status, version, chunk_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
                """,
                (document_id, filename, file_path, file_type, file_md5, file_size, status, now, now),
            )

        document = self.get_document(document_id)
        if document is None:
            raise RuntimeError(f"Document {document_id} was not created")
        return document

    def find_active_document_by_md5(self, file_md5: str) -> dict[str, Any] | None:
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
    ) -> None:
        document = self.get_document(document_id)
        if document is None:
            raise ValueError(f"Document {document_id} does not exist")

        version = int(document["version"]) + 1 if increment_version else int(document["version"])
        final_chunk_count = int(document["chunk_count"]) if chunk_count is None else chunk_count

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?, version = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (status, final_chunk_count, error_message, version, utc_now_text(), document_id),
            )

    def mark_document_deleted(self, document_id: str) -> None:
        self.update_document_status(document_id, "deleted")

    def replace_units(self, document_id: str, units: list[dict[str, Any]]) -> None:
        now = utc_now_text()
        with self.connect() as conn:
            conn.execute("DELETE FROM knowledge_units WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO knowledge_units (
                    unit_id, document_id, unit_type, title, question, answer, content,
                    category, tags, source_page, unit_index, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        unit["unit_id"],
                        document_id,
                        unit.get("unit_type", "general"),
                        unit.get("title"),
                        unit.get("question"),
                        unit.get("answer"),
                        unit["content"],
                        unit.get("category"),
                        json.dumps(unit.get("tags", []), ensure_ascii=False),
                        unit.get("source_page"),
                        unit.get("unit_index", index),
                        now,
                    )
                    for index, unit in enumerate(units)
                ],
            )

    def delete_units(self, document_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM knowledge_units WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM faq_items WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_segments WHERE document_id = ?", (document_id,))

    def replace_segments_and_faqs(
            self,
            document_id: str,
            segments: list[dict[str, Any]],
            faq_items: list[dict[str, Any]],
    ) -> None:
        """替换某个文件对应的通用片段和 FAQ 问答。

        新设计里：
        - document_segments 是所有文件的通用原文片段表。
        - faq_items 只保存 FAQ/100问 这类结构化问答。

        为了兼容旧调试逻辑，这里不再主动写 knowledge_units。
        旧表会保留，但新入库流程以这两张新表为准。
        """

        now = utc_now_text()
        with self.connect() as conn:
            conn.execute("DELETE FROM faq_items WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_segments WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO document_segments (
                    segment_id, document_id, segment_index, content, content_hash,
                    page_no, heading_path, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        segment["segment_id"],
                        document_id,
                        segment["segment_index"],
                        segment["content"],
                        segment["content_hash"],
                        segment.get("page_no"),
                        segment.get("heading_path"),
                        json.dumps(segment.get("metadata", {}), ensure_ascii=False),
                        now,
                    )
                    for segment in segments
                ],
            )
            conn.executemany(
                """
                INSERT INTO faq_items (
                    faq_id, document_id, segment_id, question_no, question, answer,
                    category, tags_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["faq_id"],
                        document_id,
                        item.get("segment_id"),
                        item.get("question_no"),
                        item["question"],
                        item.get("answer"),
                        item.get("category"),
                        json.dumps(item.get("tags", []), ensure_ascii=False),
                        json.dumps(item.get("metadata", {}), ensure_ascii=False),
                        now,
                    )
                    for item in faq_items
                ],
            )

    def search_segments_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict[str, Any]]:
        """按关键词从新 document_segments/faq_items 做补充召回。"""

        clean_keywords = self._clean_keywords(keywords)
        if not clean_keywords:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        for keyword in clean_keywords[:8]:
            like_value = f"%{keyword}%"
            clauses.extend(
                [
                    "ds.content LIKE ?",
                    "ds.heading_path LIKE ?",
                    "fi.question LIKE ?",
                    "fi.answer LIKE ?",
                    "fi.category LIKE ?",
                ]
            )
            params.extend([like_value, like_value, like_value, like_value, like_value])

        sql = f"""
            SELECT
                ds.*,
                d.filename,
                d.file_path,
                d.file_type,
                d.file_md5,
                d.version,
                fi.faq_id,
                fi.question_no,
                fi.question,
                fi.answer,
                fi.category AS faq_category,
                fi.tags_json
            FROM document_segments ds
            JOIN documents d ON d.document_id = ds.document_id
            LEFT JOIN faq_items fi ON fi.segment_id = ds.segment_id
            WHERE d.status != 'deleted'
              AND ({' OR '.join(clauses)})
            ORDER BY d.updated_at DESC, ds.segment_index ASC
            LIMIT ?
        """
        params.append(limit)

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    def find_faq_document(self, document_hint: str | None = None) -> dict[str, Any] | None:
        """查找最匹配的 FAQ 文档。

        document_hint 可以是文件名，也可以是用户问题里带着的文档描述，
        例如“扫拖一体机器人100问”。如果没有传入 hint，就返回最近更新的
        active FAQ 文档，便于处理“第95问是什么”这类省略文档名的问题。
        """

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.*,
                    COUNT(fi.faq_id) AS faq_count
                FROM documents d
                JOIN faq_items fi ON fi.document_id = d.document_id
                WHERE d.status != 'deleted'
                GROUP BY d.document_id
                ORDER BY d.updated_at DESC
                """
            ).fetchall()

        documents = [dict(row) for row in rows]
        if not documents:
            return None

        clean_hint = (document_hint or "").strip()
        if not clean_hint:
            return documents[0]

        scored_documents = [
            (self._document_match_score(clean_hint, document["filename"]), document)
            for document in documents
        ]
        scored_documents.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        return scored_documents[0][1]

    def get_faq_item_by_number(
            self,
            question_no: int,
            document_hint: str | None = None,
    ) -> dict[str, Any] | None:
        """按问题编号从 faq_items 精确查询某一问。"""

        matched_document = self.find_faq_document(document_hint)
        params: list[Any] = [question_no]
        document_clause = ""

        if matched_document is not None:
            document_clause = "AND fi.document_id = ?"
            params.append(matched_document["document_id"])

        sql = f"""
            SELECT
                fi.*,
                d.filename,
                d.file_path,
                d.file_type,
                d.file_md5,
                d.version,
                ds.content AS segment_content,
                ds.page_no,
                ds.heading_path
            FROM faq_items fi
            JOIN documents d ON d.document_id = fi.document_id
            LEFT JOIN document_segments ds ON ds.segment_id = fi.segment_id
            WHERE d.status != 'deleted'
              AND fi.question_no = ?
              {document_clause}
            ORDER BY d.updated_at DESC
            LIMIT 1
        """

        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()

        if row is not None:
            return dict(row)

        # 如果根据 hint 选中的文档里没有这个编号，再放宽到所有 FAQ 文档兜底查一次。
        if matched_document is not None:
            with self.connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        fi.*,
                        d.filename,
                        d.file_path,
                        d.file_type,
                        d.file_md5,
                        d.version,
                        ds.content AS segment_content,
                        ds.page_no,
                        ds.heading_path
                    FROM faq_items fi
                    JOIN documents d ON d.document_id = fi.document_id
                    LEFT JOIN document_segments ds ON ds.segment_id = fi.segment_id
                    WHERE d.status != 'deleted'
                      AND fi.question_no = ?
                    ORDER BY d.updated_at DESC
                    LIMIT 1
                    """,
                    (question_no,),
                ).fetchone()

            if row is not None:
                return dict(row)

        return None

    def search_faq_items_by_question(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """按问题标题从 faq_items 中查找候选问答。"""

        clean_query = query.strip()
        if not clean_query:
            return []

        like_value = f"%{clean_query}%"
        final_limit = max(1, min(int(limit), 20))

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    fi.*,
                    d.filename,
                    d.file_path,
                    d.file_type,
                    d.file_md5,
                    d.version,
                    ds.content AS segment_content,
                    ds.page_no,
                    ds.heading_path
                FROM faq_items fi
                JOIN documents d ON d.document_id = fi.document_id
                LEFT JOIN document_segments ds ON ds.segment_id = fi.segment_id
                WHERE d.status != 'deleted'
                  AND fi.question LIKE ?
                ORDER BY d.updated_at DESC, fi.question_no
                LIMIT ?
                """,
                (like_value, final_limit),
            ).fetchall()

        if rows:
            return [dict(row) for row in rows]

        normalized_query = self._normalize_match_text(clean_query)
        if len(normalized_query) < 6:
            return []

        # SQLite 不方便直接做中文标点归一化，这里取少量 FAQ 到 Python 中二次过滤。
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    fi.*,
                    d.filename,
                    d.file_path,
                    d.file_type,
                    d.file_md5,
                    d.version,
                    ds.content AS segment_content,
                    ds.page_no,
                    ds.heading_path
                FROM faq_items fi
                JOIN documents d ON d.document_id = fi.document_id
                LEFT JOIN document_segments ds ON ds.segment_id = fi.segment_id
                WHERE d.status != 'deleted'
                ORDER BY d.updated_at DESC, fi.question_no
                LIMIT 500
                """
            ).fetchall()

        candidates = []
        for row in rows:
            item = dict(row)
            question = self._normalize_match_text(item.get("question") or "")
            if not question:
                continue

            if normalized_query in question or question in normalized_query:
                score = 1.0
            else:
                common_chars = set(normalized_query) & set(question)
                score = len(common_chars) / max(len(set(question)), 1)

            if score >= 0.65:
                candidates.append((score, item))

        candidates.sort(key=lambda value: value[0], reverse=True)
        return [item for _, item in candidates[:final_limit]]

    def list_faq_questions(
            self,
            document_hint: str | None = None,
            limit: int = 120,
    ) -> dict[str, Any]:
        """列出某份 FAQ 文档中的问题编号和问题标题。"""

        matched_document = self.find_faq_document(document_hint)
        if matched_document is None:
            return {"document": None, "total": 0, "items": []}

        final_limit = max(1, min(int(limit), 300))
        with self.connect() as conn:
            total = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM faq_items
                WHERE document_id = ?
                """,
                (matched_document["document_id"],),
            ).fetchone()["total"]
            rows = conn.execute(
                """
                SELECT
                    faq_id,
                    document_id,
                    segment_id,
                    question_no,
                    question,
                    category
                FROM faq_items
                WHERE document_id = ?
                ORDER BY COALESCE(question_no, 999999), created_at
                LIMIT ?
                """,
                (matched_document["document_id"], final_limit),
            ).fetchall()

        return {
            "document": matched_document,
            "total": int(total),
            "items": [dict(row) for row in rows],
        }

    def search_units_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict[str, Any]]:
        """按关键词从 SQLite knowledge_units 做补充召回。

        这不是严格 BM25，只是第一版轻量关键词召回。
        它的价值是补齐向量召回的盲区：
        - 用户问题里有型号、品牌、故障词时，精确关键词可能比向量更稳。
        - Qdrant 还没完成重建索引时，SQLite 中的结构化单元仍可参与召回。

        返回结果会带上 documents 表中的 filename/version 等文件级信息，
        方便后续组装成 LangChain Document.metadata。
        """

        clean_keywords = self._clean_keywords(keywords)

        if not clean_keywords:
            return []

        clauses: list[str] = []
        params: list[Any] = []

        for keyword in clean_keywords[:8]:
            like_value = f"%{keyword}%"
            clauses.extend(
                [
                    "ku.title LIKE ?",
                    "ku.question LIKE ?",
                    "ku.answer LIKE ?",
                    "ku.content LIKE ?",
                    "ku.category LIKE ?",
                ]
            )
            params.extend([like_value, like_value, like_value, like_value, like_value])

        sql = f"""
            SELECT
                ku.*,
                d.filename,
                d.file_path,
                d.file_type,
                d.file_md5,
                d.version
            FROM knowledge_units ku
            JOIN documents d ON d.document_id = ku.document_id
            WHERE d.status != 'deleted'
              AND ({' OR '.join(clauses)})
            ORDER BY d.updated_at DESC, ku.unit_index ASC
            LIMIT ?
        """
        params.append(limit)

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]

    @staticmethod
    def _clean_keywords(keywords: list[str]) -> list[str]:
        clean_keywords = []
        for keyword in keywords:
            keyword = keyword.strip()
            if not keyword or keyword in clean_keywords:
                continue
            clean_keywords.append(keyword)
        return clean_keywords

    @staticmethod
    def _document_match_score(document_hint: str, filename: str) -> float:
        """给“用户提到的文档描述”和文件名做轻量匹配打分。"""

        hint = KnowledgeStore._normalize_match_text(document_hint)
        name = KnowledgeStore._normalize_match_text(os.path.splitext(filename)[0])

        if not hint or not name:
            return 0.0

        if name in hint or hint in name:
            return 100.0

        ignore_chars = set("的是了和与及第问题什么哪些全部都有一下一个这个那个")
        hint_chars = {char for char in hint if char not in ignore_chars}
        name_chars = {char for char in name if char not in ignore_chars}

        if not hint_chars or not name_chars:
            return 0.0

        common_chars = hint_chars & name_chars
        return len(common_chars) / max(len(name_chars), 1)

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        ignore_chars = set(" \t\r\n，。！？?：:；;、,.()（）【】[]“”\"'")
        return "".join(char.lower() for char in value.strip() if char not in ignore_chars)
