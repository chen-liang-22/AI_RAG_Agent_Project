import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from utils.path_tool import get_abs_path


def utc_now_text() -> str:
    """返回统一格式的 UTC 时间字符串。

    Python 的 datetime.utcnow() 返回 UTC 时间。
    timespec="seconds" 表示只保留到秒，避免数据库里出现太长的小数秒。
    """

    return datetime.utcnow().isoformat(timespec="seconds")


class TrainingRepository:
    """销售训练 SQLite 仓储。

    训练域表较多，独立仓储能避免把 KnowledgeStore 继续膨胀。
    """

    def __init__(self, db_path: str | None = None):
        # db_path 允许测试时传入临时数据库；线上默认复用 storage/knowledge.db。
        self.db_path = db_path or get_abs_path("storage/knowledge.db")
        # 对象创建时确保表存在，类似 Java 构造器里初始化 DAO 依赖。
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """打开 SQLite 连接，自动提交或回滚。

        @contextmanager 让这个函数可以写成：

            with self.connect() as conn:
                ...

        这和 Java 的 try-with-resources 很像：
        - 正常执行：yield 后面的 conn.commit() 会提交事务；
        - 出现异常：except 里 rollback；
        - 最后都会 close 连接。
        """

        conn = sqlite3.connect(self.db_path)
        # row_factory 设置后，查询结果可以用 row["字段名"] 读取，比 tuple 下标更清晰。
        conn.row_factory = sqlite3.Row
        # SQLite 默认不强制外键，这里显式开启，避免脏数据。
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
        """初始化一期需要的训练表。

        SQLite 的 CREATE TABLE IF NOT EXISTS 是幂等操作，重复执行不会清空旧数据。
        这里没有引入 Alembic 迁移工具，是因为一期表结构还比较轻。
        """

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_knowledge_batches (
                    batch_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    file_path TEXT,
                    file_md5 TEXT,
                    version_group_id TEXT,
                    version_no INTEGER NOT NULL DEFAULT 1,
                    previous_batch_id TEXT,
                    is_current INTEGER NOT NULL DEFAULT 0,
                    profile_type TEXT,
                    task_type TEXT,
                    industry TEXT,
                    difficulty TEXT,
                    visibility_default TEXT,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    point_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    quality_report_json TEXT,
                    created_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # training_knowledge_batches：一次文件上传对应一条批次记录。
            # 它主要保存文件级元数据和处理状态，不保存大段正文。
            self._ensure_column(conn, "training_knowledge_batches", "file_path", "TEXT")
            self._ensure_column(conn, "training_knowledge_batches", "quality_report_json", "TEXT")
            self._ensure_column(conn, "training_knowledge_batches", "version_group_id", "TEXT")
            self._ensure_column(conn, "training_knowledge_batches", "version_no", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "training_knowledge_batches", "previous_batch_id", "TEXT")
            self._ensure_column(conn, "training_knowledge_batches", "is_current", "INTEGER NOT NULL DEFAULT 0")
            self._migrate_training_batch_versions(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    qdrant_point_id TEXT,
                    chunk_text TEXT NOT NULL,
                    source_type TEXT,
                    profile_type TEXT,
                    task_type TEXT,
                    industry TEXT,
                    difficulty TEXT,
                    case_part TEXT,
                    visibility TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(batch_id) REFERENCES training_knowledge_batches(batch_id)
                )
                """
            )
            # training_knowledge_chunks：保存训练知识切片明细。
            # Qdrant 负责向量检索；SQLite 负责可追溯的结构化记录。
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_role_profiles (
                    profile_id TEXT PRIMARY KEY,
                    trainee_id TEXT NOT NULL,
                    plan_id TEXT,
                    profile_type TEXT NOT NULL,
                    visible_profile_json TEXT NOT NULL,
                    hidden_profile_json TEXT NOT NULL,
                    role_profile_json TEXT NOT NULL,
                    role_confirm_card_json TEXT NOT NULL,
                    selected_fields_json TEXT,
                    scenario_description TEXT,
                    extra_details TEXT,
                    retrieved_evidence_json TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # training_role_profiles：保存 AI 客户画像。
            # visible/hidden/role/confirm_card 分开存，方便控制哪些内容给学员看。
            self._ensure_column(conn, "training_role_profiles", "plan_id", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_goal_settings (
                    setting_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    plan_id TEXT,
                    trainee_id TEXT NOT NULL,
                    training_mode TEXT NOT NULL,
                    training_purpose TEXT NOT NULL,
                    round_limit INTEGER NOT NULL,
                    stages_json TEXT NOT NULL,
                    scoring_rules_json TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # training_goal_settings：保存开放式训练目标、动态轮数和阶段条件。
            self._ensure_column(conn, "training_goal_settings", "plan_id", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_plans (
                    plan_id TEXT PRIMARY KEY,
                    plan_name TEXT NOT NULL,
                    trainee_id TEXT NOT NULL,
                    trainee_name TEXT NOT NULL,
                    profile_type TEXT NOT NULL,
                    trainee_json TEXT NOT NULL,
                    selected_fields_json TEXT NOT NULL,
                    scenario_description TEXT NOT NULL,
                    extra_details TEXT,
                    model_mode TEXT,
                    active_profile_id TEXT,
                    active_setting_id TEXT,
                    role_status TEXT NOT NULL,
                    goal_status TEXT NOT NULL,
                    score_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # training_plans：训练方案主表，负责把“训练名称、角色、阶段、评分规则”串起来。
            self._migrate_training_plans_allow_duplicate_names(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sales_training_sessions (
                    session_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    setting_id TEXT NOT NULL,
                    trainee_id TEXT NOT NULL,
                    training_mode TEXT NOT NULL,
                    response_mode TEXT NOT NULL,
                    current_stage_no INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    round_limit INTEGER NOT NULL,
                    total_score INTEGER,
                    level TEXT,
                    report_json TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # sales_training_sessions：一场训练会话的主表。
            # 对话轮次和评分分开存，避免主表被大文本撑大。
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sales_training_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    round_no INTEGER NOT NULL,
                    stage_no INTEGER NOT NULL DEFAULT 1,
                    response_mode TEXT,
                    started_at TEXT,
                    submitted_at TEXT,
                    response_seconds REAL,
                    retrieved_chunk_ids_json TEXT,
                    retrieved_evidence_json TEXT,
                    stage_decision_json TEXT,
                    coach_analysis_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sales_training_sessions(session_id)
                )
                """
            )
            self._ensure_column(conn, "training_goal_settings", "scoring_rules_json", "TEXT")
            self._ensure_column(conn, "sales_training_turns", "coach_analysis_json", "TEXT")
            # sales_training_turns：保存每一轮学员/AI 客户对话。
            # round_no=0 约定为 AI 客户开场白。
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sales_training_scores (
                    score_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    general_score INTEGER NOT NULL,
                    stage_score INTEGER NOT NULL,
                    penalty_score INTEGER NOT NULL,
                    final_score INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    is_passed INTEGER NOT NULL,
                    detail_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sales_training_sessions(session_id)
                )
                """
            )

    @staticmethod
    def _json(data: Any) -> str:
        # ensure_ascii=False 保证中文直接存中文，而不是 \u4e2d 这种转义。
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> None:
        """给旧 SQLite 表补字段。

        SQLite 没有 ADD COLUMN IF NOT EXISTS，所以先用 PRAGMA table_info 读取已有字段。
        table_name 和 column_name 都是代码里的固定值，不接收外部输入。
        """

        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    @staticmethod
    def _migrate_training_plans_allow_duplicate_names(conn: sqlite3.Connection) -> None:
        """把旧训练方案表迁移为允许同名训练。

        SQLite 不能直接删除 UNIQUE 约束，所以检测到旧表仍然把 plan_name 作为唯一字段时，
        需要建临时表、复制数据、替换旧表。每场训练仍然由 plan_id 保证唯一。
        """

        unique_indexes = conn.execute("PRAGMA index_list(training_plans)").fetchall()
        has_plan_name_unique_index = False
        for index_row in unique_indexes:
            if int(index_row["unique"] or 0) != 1:
                continue
            index_columns = conn.execute(f"PRAGMA index_info({index_row['name']})").fetchall()
            column_names = [str(column["name"]) for column in index_columns]
            if column_names == ["plan_name"]:
                has_plan_name_unique_index = True
                break
        if not has_plan_name_unique_index:
            return

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_plans_new (
                plan_id TEXT PRIMARY KEY,
                plan_name TEXT NOT NULL,
                trainee_id TEXT NOT NULL,
                trainee_name TEXT NOT NULL,
                profile_type TEXT NOT NULL,
                trainee_json TEXT NOT NULL,
                selected_fields_json TEXT NOT NULL,
                scenario_description TEXT NOT NULL,
                extra_details TEXT,
                model_mode TEXT,
                active_profile_id TEXT,
                active_setting_id TEXT,
                role_status TEXT NOT NULL,
                goal_status TEXT NOT NULL,
                score_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO training_plans_new (
                plan_id, plan_name, trainee_id, trainee_name, profile_type,
                trainee_json, selected_fields_json, scenario_description, extra_details,
                model_mode, active_profile_id, active_setting_id,
                role_status, goal_status, score_status, created_at, updated_at
            )
            SELECT
                plan_id, plan_name, trainee_id, trainee_name, profile_type,
                trainee_json, selected_fields_json, scenario_description, extra_details,
                model_mode, active_profile_id, active_setting_id,
                role_status, goal_status, score_status, created_at, updated_at
            FROM training_plans
            """
        )
        conn.execute("DROP TABLE training_plans")
        conn.execute("ALTER TABLE training_plans_new RENAME TO training_plans")

    @staticmethod
    def _migrate_training_batch_versions(conn: sqlite3.Connection) -> None:
        """给历史训练资料批次补齐版本字段。

        老数据没有 version_group_id 和 is_current。
        这里把每个已发布批次先视为一个独立版本组的当前版本，
        后续同名文件再次发布时再进入真正的版本链。
        """

        conn.execute(
            """
            UPDATE training_knowledge_batches
            SET version_group_id = batch_id
            WHERE version_group_id IS NULL OR version_group_id = ''
            """
        )
        conn.execute(
            """
            UPDATE training_knowledge_batches
            SET version_no = 1
            WHERE version_no IS NULL OR version_no <= 0
            """
        )
        conn.execute(
            """
            UPDATE training_knowledge_batches
            SET is_current = 1
            WHERE status = 'published' AND COALESCE(is_current, 0) = 0
            """
        )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        # sqlite3.Row 不是普通 dict，转成 dict 后业务层更好处理。
        return dict(row) if row is not None else None

    def create_batch(self, **values: Any) -> dict[str, Any]:
        """创建训练知识上传批次。

        **values 是 Python 的关键字参数收集写法，类似 Java 里传一个 Map。
        这里用它是为了让调用方只传自己关心的字段，仓储内部统一补默认值。
        """

        now = utc_now_text()
        batch_id = values.get("batch_id") or f"batch_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_knowledge_batches (
                    batch_id, source_type, source_file, file_path, file_md5,
                    version_group_id, version_no, previous_batch_id, is_current,
                    profile_type, task_type, industry, difficulty, visibility_default, status,
                    error_message, quality_report_json, created_by, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    # SQLite 使用 ? 作为参数占位符，避免手动拼 SQL 导致注入风险。
                    batch_id,
                    values.get("source_type"),
                    values.get("source_file"),
                    values.get("file_path"),
                    values.get("file_md5"),
                    values.get("version_group_id") or batch_id,
                    int(values.get("version_no") or 1),
                    values.get("previous_batch_id"),
                    int(bool(values.get("is_current"))),
                    values.get("profile_type"),
                    values.get("task_type"),
                    values.get("industry"),
                    values.get("difficulty"),
                    values.get("visibility_default"),
                    values.get("status", "uploaded"),
                    values.get("error_message"),
                    self._json(values.get("quality_report")) if values.get("quality_report") is not None else None,
                    values.get("created_by"),
                    now,
                    now,
                ),
            )
        return self.get_batch(batch_id) or {}

    def update_batch_status(
            self,
            batch_id: str,
            *,
            status: str,
            chunk_count: int | None = None,
            point_count: int | None = None,
            error_message: str | None = None,
            quality_report: dict[str, Any] | None = None,
            is_current: bool | None = None,
    ) -> None:
        """更新训练知识上传批次状态和统计信息。

        COALESCE(?, chunk_count) 的意思是：
        - 如果传入的新值不是 None，就用新值；
        - 如果传入 None，就保留原字段值。
        """

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE training_knowledge_batches
                SET status = ?,
                    chunk_count = COALESCE(?, chunk_count),
                    point_count = COALESCE(?, point_count),
                    error_message = ?,
                    quality_report_json = COALESCE(?, quality_report_json),
                    is_current = COALESCE(?, is_current),
                    updated_at = ?
                WHERE batch_id = ?
                """,
                (
                    status,
                    chunk_count,
                    point_count,
                    error_message,
                    self._json(quality_report) if quality_report is not None else None,
                    int(is_current) if is_current is not None else None,
                    utc_now_text(),
                    batch_id,
                ),
            )

    def get_latest_batch_for_version(self, *, source_type: str, source_file: str) -> dict[str, Any] | None:
        """按资料类型和文件名查询最新版本批次。"""

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM training_knowledge_batches
                WHERE source_type = ? AND source_file = ? AND status != 'deleted'
                ORDER BY version_no DESC, updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (source_type, source_file),
            ).fetchone()
        return self._row(row)

    def list_current_published_batch_ids(self) -> list[str]:
        """查询当前参与训练检索的已发布批次编号。"""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT batch_id
                FROM training_knowledge_batches
                WHERE status = 'published' AND COALESCE(is_current, 0) = 1
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
        return [str(row["batch_id"]) for row in rows]

    def list_published_batches_in_version_group(
            self,
            version_group_id: str,
            *,
            exclude_batch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """查询同一版本组内已发布或已归档的批次。"""

        params: list[Any] = [version_group_id]
        exclude_sql = ""
        if exclude_batch_id:
            exclude_sql = "AND batch_id != ?"
            params.append(exclude_batch_id)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM training_knowledge_batches
                WHERE version_group_id = ?
                  AND status IN ('published', 'archived')
                  {exclude_sql}
                ORDER BY version_no DESC, updated_at DESC
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_batches_in_version_group(self, version_group_id: str) -> list[dict[str, Any]]:
        """查询同一版本组内的全部未删除批次。"""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM training_knowledge_batches
                WHERE version_group_id = ?
                  AND status != 'deleted'
                ORDER BY version_no DESC, updated_at DESC, created_at DESC
                """,
                (version_group_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def archive_other_versions(self, *, version_group_id: str, current_batch_id: str) -> None:
        """把同版本组内非当前版本标记为归档。"""

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE training_knowledge_batches
                SET status = 'archived',
                    is_current = 0,
                    updated_at = ?
                WHERE version_group_id = ?
                  AND batch_id != ?
                  AND status = 'published'
                """,
                (utc_now_text(), version_group_id, current_batch_id),
            )

    def get_published_batch_by_md5(self, file_md5: str) -> dict[str, Any] | None:
        """按文件 MD5 查询已经成功入库的训练资料。

        只复用 published 批次，避免解析失败或半入库数据被误当成可用资料。
        """

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM training_knowledge_batches
                WHERE file_md5 = ? AND status = 'published'
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (file_md5,),
            ).fetchone()
        return self._row(row)

    def list_batches(self, *, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        """分页查询训练资料上传批次。"""

        offset = (page - 1) * page_size
        with self.connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS total FROM training_knowledge_batches WHERE status != 'deleted'"
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM training_knowledge_batches
                WHERE status != 'deleted'
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (page_size, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total_row["total"] or 0)

    def mark_batch_deleted(self, batch_id: str) -> bool:
        """软删除训练资料批次。

        这里不物理删除原文件，方便后续审计、预览问题排查或恢复。
        返回 False 表示批次不存在或已经删除。
        """

        with self.connect() as conn:
            row = conn.execute(
                "SELECT batch_id FROM training_knowledge_batches WHERE batch_id = ? AND status != 'deleted'",
                (batch_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE training_knowledge_batches
                SET status = 'deleted',
                    updated_at = ?
                WHERE batch_id = ?
                """,
                (utc_now_text(), batch_id),
            )
        return True

    def add_chunk(self, **values: Any) -> dict[str, Any]:
        """保存一个训练知识切片明细。

        注意：正文会同时写入 Qdrant 和 SQLite。
        Qdrant 用于相似度检索；SQLite 用于前端预览、排查和追踪。
        """

        chunk_id = values.get("chunk_id") or f"chunk_{uuid.uuid4().hex}"
        metadata = values.get("metadata") or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_knowledge_chunks (
                    chunk_id, batch_id, qdrant_point_id, chunk_text, source_type,
                    profile_type, task_type, industry, difficulty, case_part,
                    visibility, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    values["batch_id"],
                    values.get("qdrant_point_id"),
                    values["chunk_text"],
                    values.get("source_type"),
                    values.get("profile_type"),
                    values.get("task_type"),
                    values.get("industry"),
                    values.get("difficulty"),
                    values.get("case_part"),
                    values.get("visibility"),
                    self._json(metadata),
                    utc_now_text(),
                ),
            )
        return self.get_chunk(chunk_id) or {}

    def replace_chunks(self, batch_id: str, chunks: list[dict[str, Any]]) -> None:
        """替换某个批次的切片明细。

        预览阶段会先保存切片；确认发布时如果重新解析，也可以用这个方法覆盖旧切片。
        """

        with self.connect() as conn:
            conn.execute("DELETE FROM training_knowledge_chunks WHERE batch_id = ?", (batch_id,))
            for values in chunks:
                metadata = values.get("metadata") or {}
                conn.execute(
                    """
                    INSERT INTO training_knowledge_chunks (
                        chunk_id, batch_id, qdrant_point_id, chunk_text, source_type,
                        profile_type, task_type, industry, difficulty, case_part,
                        visibility, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        values.get("chunk_id"),
                        batch_id,
                        values.get("qdrant_point_id"),
                        values.get("chunk_text"),
                        values.get("source_type"),
                        values.get("profile_type"),
                        values.get("task_type"),
                        values.get("industry"),
                        values.get("difficulty"),
                        values.get("case_part"),
                        values.get("visibility"),
                        self._json(metadata),
                        utc_now_text(),
                    ),
                )

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM training_knowledge_batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return self._row(row)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM training_knowledge_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row(row)

    def list_chunks(self, batch_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM training_knowledge_chunks
                WHERE batch_id = ?
                ORDER BY created_at, chunk_id
                """,
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_plan(self, **values: Any) -> dict[str, Any]:
        """创建训练方案。

        训练方案是角色、训练阶段、评分规则的上层聚合。
        plan_id 保证唯一，plan_name 允许重复，方便同一主题多次训练。
        """

        now = utc_now_text()
        plan_id = values.get("plan_id") or f"plan_{uuid.uuid4().hex}"
        trainee = values["trainee"]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_plans (
                    plan_id, plan_name, trainee_id, trainee_name, profile_type,
                    trainee_json, selected_fields_json, scenario_description, extra_details,
                    model_mode, active_profile_id, active_setting_id,
                    role_status, goal_status, score_status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    values["plan_name"],
                    trainee["trainee_id"],
                    trainee.get("trainee_name") or "销售学员",
                    values["profile_type"],
                    self._json(trainee),
                    self._json(values.get("selected_fields") or {}),
                    values["scenario_description"],
                    values.get("extra_details") or "",
                    values.get("model_mode"),
                    "pending",
                    "pending",
                    "pending",
                    now,
                    now,
                ),
            )
        return self.get_plan(plan_id) or {}

    def list_plans(self, *, page: int, page_size: int, keyword: str | None = None) -> tuple[list[dict[str, Any]], int]:
        """分页查询训练方案列表。"""

        offset = (page - 1) * page_size
        params: list[Any] = []
        where_sql = ""
        if keyword and keyword.strip():
            where_sql = "WHERE plan_name LIKE ?"
            params.append(f"%{keyword.strip()}%")
        with self.connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM training_plans {where_sql}",
                params,
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT *
                FROM training_plans
                {where_sql}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()
        return [dict(row) for row in rows], int(total_row["total"] or 0)

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        """按 ID 查询训练方案。"""

        with self.connect() as conn:
            row = conn.execute("SELECT * FROM training_plans WHERE plan_id = ?", (plan_id,)).fetchone()
        return self._row(row)

    def update_plan(self, plan_id: str, **values: Any) -> dict[str, Any]:
        """更新训练方案基础信息和状态。

        这里使用动态 SQL，但字段白名单来自代码固定集合，不接收外部字段名。
        """

        allowed_columns = {
            "plan_name",
            "trainee_id",
            "trainee_name",
            "profile_type",
            "trainee_json",
            "selected_fields_json",
            "scenario_description",
            "extra_details",
            "model_mode",
            "active_profile_id",
            "active_setting_id",
            "role_status",
            "goal_status",
            "score_status",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            if key not in allowed_columns:
                continue
            assignments.append(f"{key} = ?")
            params.append(value)
        assignments.append("updated_at = ?")
        params.append(utc_now_text())
        params.append(plan_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE training_plans SET {', '.join(assignments)} WHERE plan_id = ?",
                params,
            )
        return self.get_plan(plan_id) or {}

    def attach_role_to_plan(self, plan_id: str, profile_id: str) -> dict[str, Any]:
        """把生成好的 AI 角色关联到训练方案，并标记阶段/评分需要重新生成。"""

        return self.update_plan(
            plan_id,
            active_profile_id=profile_id,
            active_setting_id=None,
            role_status="generated",
            goal_status="stale",
            score_status="stale",
        )

    def attach_goal_to_plan(self, plan_id: str, setting_id: str) -> dict[str, Any]:
        """把生成好的训练阶段关联到训练方案，并标记评分规则已随阶段生成。"""

        return self.update_plan(
            plan_id,
            active_setting_id=setting_id,
            goal_status="generated",
            score_status="generated",
        )

    def save_role_profile(self, **values: Any) -> dict[str, Any]:
        """保存一次 AI 陪练角色。"""

        now = utc_now_text()
        profile_id = values.get("profile_id") or f"profile_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_role_profiles (
                    profile_id, trainee_id, plan_id, profile_type, visible_profile_json,
                    hidden_profile_json, role_profile_json, role_confirm_card_json,
                    selected_fields_json, scenario_description, extra_details,
                    retrieved_evidence_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    values["trainee_id"],
                    values.get("plan_id"),
                    values["profile_type"],
                    self._json(values["visible_profile"]),
                    self._json(values["hidden_profile"]),
                    self._json(values["role_profile"]),
                    self._json(values["role_confirm_card"]),
                    self._json(values.get("selected_fields") or {}),
                    values.get("scenario_description"),
                    values.get("extra_details"),
                    self._json(values.get("retrieved_evidence") or []),
                    values.get("status", "confirmed"),
                    now,
                    now,
                ),
            )
        return self.get_role_profile(profile_id) or {}

    def get_role_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM training_role_profiles WHERE profile_id = ?", (profile_id,)).fetchone()
        return self._row(row)

    def update_role_profile(
            self,
            profile_id: str,
            *,
            visible_profile: dict | None = None,
            hidden_profile: dict | None = None,
            role_profile: dict | None = None,
            role_confirm_card: dict | None = None,
    ) -> dict[str, Any]:
        """人工修改 AI 客户角色的某些 JSON 字段。"""

        updates: dict[str, str] = {}
        if visible_profile is not None:
            updates["visible_profile_json"] = self._json(visible_profile)
        if hidden_profile is not None:
            updates["hidden_profile_json"] = self._json(hidden_profile)
        if role_profile is not None:
            updates["role_profile_json"] = self._json(role_profile)
        if role_confirm_card is not None:
            updates["role_confirm_card_json"] = self._json(role_confirm_card)
        if not updates:
            return self.get_role_profile(profile_id) or {}
        assignments = [f"{column} = ?" for column in updates]
        params = [*updates.values(), utc_now_text(), profile_id]
        with self.connect() as conn:
            conn.execute(
                f"UPDATE training_role_profiles SET {', '.join(assignments)}, updated_at = ? WHERE profile_id = ?",
                params,
            )
        return self.get_role_profile(profile_id) or {}

    def save_goal_setting(self, **values: Any) -> dict[str, Any]:
        """保存开放式训练设置。"""

        now = utc_now_text()
        setting_id = values.get("setting_id") or f"setting_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO training_goal_settings (
                    setting_id, profile_id, plan_id, trainee_id, training_mode, training_purpose,
                    round_limit, stages_json, scoring_rules_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    setting_id,
                    values["profile_id"],
                    values.get("plan_id"),
                    values["trainee_id"],
                    values.get("training_mode", "open"),
                    values["training_purpose"],
                    int(values["round_limit"]),
                    self._json(values["stages"]),
                    self._json(values.get("scoring_rules") or {}),
                    values.get("status", "confirmed"),
                    now,
                    now,
                ),
            )
        return self.get_goal_setting(setting_id) or {}

    def get_goal_setting(self, setting_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM training_goal_settings WHERE setting_id = ?", (setting_id,)).fetchone()
        return self._row(row)

    def update_goal_setting(
            self,
            setting_id: str,
            *,
            training_purpose: str | None = None,
            round_limit: int | None = None,
            stages: list[dict[str, Any]] | None = None,
            scoring_rules: dict | None = None,
    ) -> dict[str, Any]:
        """人工修改训练宗旨、轮数、阶段或评分规则。"""

        updates: dict[str, Any] = {}
        if training_purpose is not None:
            updates["training_purpose"] = training_purpose
        if round_limit is not None:
            updates["round_limit"] = int(round_limit)
        if stages is not None:
            updates["stages_json"] = self._json(stages)
        if scoring_rules is not None:
            updates["scoring_rules_json"] = self._json(scoring_rules)
        if not updates:
            return self.get_goal_setting(setting_id) or {}
        assignments = [f"{column} = ?" for column in updates]
        params = [*updates.values(), utc_now_text(), setting_id]
        with self.connect() as conn:
            conn.execute(
                f"UPDATE training_goal_settings SET {', '.join(assignments)}, updated_at = ? WHERE setting_id = ?",
                params,
            )
        return self.get_goal_setting(setting_id) or {}

    def create_session(self, **values: Any) -> dict[str, Any]:
        """创建一次开放式训练会话。"""

        now = utc_now_text()
        session_id = values.get("session_id") or f"session_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sales_training_sessions (
                    session_id, profile_id, setting_id, trainee_id, training_mode,
                    response_mode, current_stage_no, status, round_limit,
                    started_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    values["profile_id"],
                    values["setting_id"],
                    values["trainee_id"],
                    values.get("training_mode", "open"),
                    values.get("response_mode", "stream"),
                    values.get("status", "active"),
                    int(values["round_limit"]),
                    now,
                    now,
                    now,
                ),
            )
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sales_training_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return self._row(row)

    def list_sessions(
            self,
            *,
            page: int,
            page_size: int,
            trainee_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询训练会话历史，并统计学员已回答轮数。

        返回值是 tuple[list[dict], int]：
        - 第一个元素是当前页数据；
        - 第二个元素是符合条件的总数。

        Python 的 tuple 类似 Java 里简单返回 Pair，不过这里用类型注解明确结构。
        """

        offset = (page - 1) * page_size
        filters: list[str] = []
        params: list[Any] = []
        if trainee_id:
            filters.append("s.trainee_id = ?")
            params.append(trainee_id)
        # where_sql 只由固定片段拼接，不拼接用户输入；用户输入仍然走 ? 参数。
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self.connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM sales_training_sessions s {where_sql}",
                params,
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT s.*,
                       COALESCE(t.answered_count, 0) AS answered_count
                FROM sales_training_sessions s
                LEFT JOIN (
                    SELECT session_id, COUNT(*) AS answered_count
                    FROM sales_training_turns
                    WHERE role = 'trainee'
                    GROUP BY session_id
                ) t ON t.session_id = s.session_id
                {where_sql}
                ORDER BY s.updated_at DESC, s.created_at DESC
                LIMIT ? OFFSET ?
                """,
                # [*params, page_size, offset] 是 Python 列表解包写法。
                # 等价于 Java 中把已有参数列表复制后再追加两个分页参数。
                [*params, page_size, offset],
            ).fetchall()
        return [dict(row) for row in rows], int(total_row["total"] or 0)

    def update_session_status(
            self,
            session_id: str,
            *,
            status: str,
            total_score: int | None = None,
            level: str | None = None,
            report: dict | None = None,
    ) -> None:
        """更新训练会话状态，评分完成时同步写报告摘要。"""

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sales_training_sessions
                SET status = ?, total_score = COALESCE(?, total_score),
                    level = COALESCE(?, level), report_json = COALESCE(?, report_json),
                    ended_at = CASE WHEN ? IN ('completed', 'failed') THEN COALESCE(ended_at, ?) ELSE ended_at END,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    status,
                    total_score,
                    level,
                    self._json(report) if report is not None else None,
                    status,
                    utc_now_text(),
                    utc_now_text(),
                    session_id,
                ),
            )

    def add_turn(self, **values: Any) -> dict[str, Any]:
        """保存训练对话轮次。

        role 字段当前会出现：
        - customer：AI 客户；
        - trainee：学员；
        - system：系统消息。

        训练复盘时按 round_no + created_at 排序，还原真实对话顺序。
        """

        turn_id = values.get("turn_id") or f"turn_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sales_training_turns (
                    turn_id, session_id, role, content, round_no, stage_no,
                    response_mode, started_at, submitted_at, response_seconds,
                    retrieved_chunk_ids_json, retrieved_evidence_json,
                    stage_decision_json, coach_analysis_json, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    values["session_id"],
                    values["role"],
                    values["content"],
                    int(values["round_no"]),
                    int(values.get("stage_no", 1)),
                    values.get("response_mode"),
                    values.get("started_at"),
                    values.get("submitted_at"),
                    values.get("response_seconds"),
                    self._json(values.get("retrieved_chunk_ids") or []),
                    self._json(values.get("retrieved_evidence") or []),
                    self._json(values.get("stage_decision") or {}),
                    self._json(values.get("coach_analysis") or {}),
                    self._json(values.get("metadata") or {}),
                    utc_now_text(),
                ),
            )
        return self.get_turn(turn_id) or {}

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sales_training_turns WHERE turn_id = ?", (turn_id,)).fetchone()
        return self._row(row)

    def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM sales_training_turns
                WHERE session_id = ?
                ORDER BY round_no, created_at
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def next_round_no(self, session_id: str) -> int:
        """计算下一轮学员回复轮次。

        只统计 role='trainee'，因为 AI 客户开场白 round_no=0，
        AI 客户回复和学员回复共享同一个 round_no。
        """

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(round_no), 0) AS max_round
                FROM sales_training_turns
                WHERE session_id = ? AND role = 'trainee'
                """,
                (session_id,),
            ).fetchone()
        return int(row["max_round"] or 0) + 1

    def save_score(self, **values: Any) -> dict[str, Any]:
        """保存训练评分结果。

        一期先保留多评分记录能力，方便后续“AI 自动评分 + 人工复核”。
        服务层会做幂等控制，避免已完成会话重复生成多份评分。
        """

        now = utc_now_text()
        score_id = values.get("score_id") or f"score_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sales_training_scores (
                    score_id, session_id, general_score, stage_score, penalty_score,
                    final_score, level, is_passed, detail_json, review_status,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score_id,
                    values["session_id"],
                    int(values["general_score"]),
                    int(values["stage_score"]),
                    int(values.get("penalty_score", 0)),
                    int(values["final_score"]),
                    values["level"],
                    1 if values.get("is_passed") else 0,
                    self._json(values.get("detail") or {}),
                    values.get("review_status", "confirmed"),
                    now,
                    now,
                ),
            )
        return self.get_score(score_id) or {}

    def get_score(self, score_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sales_training_scores WHERE score_id = ?", (score_id,)).fetchone()
        return self._row(row)

    def get_latest_score_by_session(self, session_id: str) -> dict[str, Any] | None:
        """查询某个训练会话最新的一份评分报告。

        如果未来允许人工复核产生新版本评分，这里始终取 updated_at 最新的一份。
        """

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sales_training_scores
                WHERE session_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._row(row)
