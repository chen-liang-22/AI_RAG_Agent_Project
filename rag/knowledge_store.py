import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, inspect, or_, select, text
from sqlalchemy.orm import Session

from domain.entities import (
    ConversationEntity,
    ConversationMessageEntity,
    DictionaryItemEntity,
    DocumentEntity,
    ExamQuestionEntity,
    ExamSessionEntity,
)
from infrastructure.orm_session import orm_session_context
from rag.profile_dictionaries import PROFILE_DICTIONARY_ITEMS
from utils.knowledge_asset_constants import TRAINING_COLLECTION_NAMES
from utils.logger_handler import logger
from utils.redis_client import get_redis_client


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
    {
        "dictionary_code": "training_source_type",
        "dictionary_name": "销售训练资料来源类型",
        "items": [
            (
                "lms_case",
                "LMS 销售训练案例",
                None,
                1,
                "按客户案例、任务要求、标准答案、隐藏心理、评分标准做专门结构化拆分",
                {
                    "default": True,
                    "implemented": True,
                    "strategy": "LmsCaseIngestStrategy",
                    "collection_name": "sales_training_cases",
                },
            ),
        ],
    },
    {
        "dictionary_code": "training_case_part",
        "dictionary_name": "销售训练切片类型",
        "items": [
            ("case_profile", "客户背景", None, 1, "客户案例、公司背景、合作阶段等信息", {"tag_type": "info"}),
            ("task_requirement", "训练任务", None, 2, "本次训练要求、沟通目标和操作约束", {"tag_type": "primary"}),
            ("standard_answer", "参考话术", None, 3, "标准答案、优秀话术和建议表达", {"tag_type": "success"}),
            ("hidden_psychology", "客户顾虑", None, 4, "客户真实顾虑、隐性心理和潜在异议", {"tag_type": "warning"}),
            ("scoring_rubric", "评分依据", None, 5, "评分标准、命中点和扣分点", {"tag_type": "danger"}),
            ("product_fact", "产品事实", None, 6, "产品功能、参数、服务和交付事实", {"tag_type": "info"}),
            ("faq", "常见问答", None, 7, "常见客户问题和标准答复", {"tag_type": "info"}),
            ("competitor", "竞品信息", None, 8, "竞品对比、优势劣势和替代方案", {"tag_type": "warning"}),
            ("success_case", "成功案例", None, 9, "成交案例、客户证言和效果证明", {"tag_type": "success"}),
            ("glossary", "术语说明", None, 10, "行业术语、缩写和业务概念", {"tag_type": "info"}),
        ],
    },
    {
        "dictionary_code": "training_chunk_usage",
        "dictionary_name": "训练切片模型用途",
        "items": [
            ("visible", "通用知识", None, 1, "角色生成、对话训练和评分都可参考", {"tag_type": "info"}),
            ("hidden", "客户内部顾虑", None, 2, "主要用于 AI 客户扮演和追问，不作为权限展示", {"tag_type": "warning"}),
            ("scoring_only", "评分专用", None, 3, "主要用于训练结束评分，不作为权限展示", {"tag_type": "danger"}),
        ],
    },
    {
        "dictionary_code": "training_batch_status",
        "dictionary_name": "销售训练资料批次状态",
        "items": [
            ("parsing", "解析中", None, 1, "文件已保存，正在解析和切片", {"tag_type": "warning"}),
            ("pending_review", "待确认", None, 2, "切片和质量评估已完成，等待人工确认发布", {"tag_type": "warning"}),
            ("embedding", "发布中", None, 3, "正在从临时向量库发布到正式向量库", {"tag_type": "warning"}),
            ("published", "已发布", None, 4, "训练资料已完成向量入库，可以参与训练检索", {"tag_type": "success"}),
            ("archived", "历史版本", None, 5, "同一资料的新版本已发布，该版本不再参与训练检索，可按需回滚", {"tag_type": "info"}),
            ("parsing_failed", "解析失败", None, 6, "文件解析或临时向量库写入失败，需要查看错误信息", {"tag_type": "danger"}),
            ("publish_failed", "发布失败", None, 7, "从临时向量库发布到正式向量库失败，可重试发布", {"tag_type": "danger"}),
            ("deleted", "已删除", None, 8, "批次已软删除，列表不再展示，相关向量点会被删除", {"tag_type": "info"}),
            ("duplicated", "重复复用", None, 9, "上传响应状态，表示文件 MD5 命中已发布批次并复用历史数据", {"tag_type": "info"}),
        ],
    },
] + PROFILE_DICTIONARY_ITEMS

_MYSQL_DEFAULT_DICTIONARY_SYNCED = False
DICTIONARY_CACHE_TTL_SECONDS = 3600
DEPRECATED_DICTIONARY_CODES = {"collection_domain_keyword", "sales_customer_profile_template"}


def utc_now_text() -> str:
    """返回统一格式的 UTC 时间字符串。"""

    return datetime.utcnow().isoformat(timespec="seconds", sep=" ")


def utc_now() -> datetime:
    """返回去掉微秒的 UTC 时间，便于写入 MySQL DATETIME 字段。"""

    return datetime.utcnow().replace(microsecond=0)


class KnowledgeStore:
    """业务元数据仓储。

    关系型数据库只保存文件、字典、会话和考试这些业务状态。
    知识正文、分片正文、向量和可检索 payload 仍以 Qdrant 为准。
    """

    def __init__(self):
        self.init_db()

    def init_db(self) -> None:
        """同步默认字典数据。

        表结构由 MySQL 初始化脚本维护；这里只做业务默认字典 upsert。
        """

        global _MYSQL_DEFAULT_DICTIONARY_SYNCED

        if _MYSQL_DEFAULT_DICTIONARY_SYNCED:
            return
        with orm_session_context() as session:
            self.ensure_document_storage_columns(session)
            self.seed_default_dictionaries(session)
        _MYSQL_DEFAULT_DICTIONARY_SYNCED = True
        self.refresh_all_dictionary_cache()

    @staticmethod
    def ensure_document_storage_columns(session: Session) -> None:
        """确保 documents 表具备 MinIO 存储字段，便于旧库平滑启动。"""

        bind = session.get_bind()
        inspector = inspect(bind)
        columns = {column["name"] for column in inspector.get_columns("documents")}
        ddl_statements: list[str] = []
        if "storage_type" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN storage_type VARCHAR(32) NOT NULL DEFAULT 'minio' "
                "COMMENT '文件存储类型：minio 表示对象存储' AFTER file_path"
            )
        if "bucket_name" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN bucket_name VARCHAR(128) NULL COMMENT 'MinIO 桶名' AFTER storage_type"
            )
        if "object_name" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN object_name VARCHAR(1024) NULL COMMENT 'MinIO 对象路径' AFTER bucket_name"
            )
        if "public_url" not in columns:
            ddl_statements.append(
                "ALTER TABLE documents ADD COLUMN public_url VARCHAR(2048) NULL COMMENT 'MinIO 公共访问地址' AFTER object_name"
            )
        for ddl_statement in ddl_statements:
            session.execute(text(ddl_statement))
        if ddl_statements:
            logger.info("[知识库] documents 表 MinIO 存储字段已自动补齐 字段数量=%s", len(ddl_statements))

        indexes = {index["name"] for index in inspector.get_indexes("documents")}
        if "idx_documents_storage_object" not in indexes:
            session.execute(text(
                "CREATE INDEX idx_documents_storage_object "
                "ON documents(storage_type, bucket_name, object_name(255))"
            ))
            logger.info("[知识库] documents 表 MinIO 存储索引已自动补齐 索引名=idx_documents_storage_object")

    def seed_default_dictionaries(self, session: Session) -> None:
        """初始化系统默认字典项，已有字典项只更新展示信息。"""

        deprecated_items = session.scalars(
            select(DictionaryItemEntity).where(
                DictionaryItemEntity.dictionary_code.in_(DEPRECATED_DICTIONARY_CODES)
            )
        ).all()
        for item in deprecated_items:
            session.delete(item)

        now = utc_now()
        for dictionary in DEFAULT_DICTIONARY_ITEMS:
            dictionary_code = dictionary["dictionary_code"]
            dictionary_name = dictionary["dictionary_name"]
            item_id_by_code: dict[str, str] = {}
            for item in dictionary["items"]:
                item_code, item_name, parent_code, sort_order, description = item[:5]
                metadata = item[5] if len(item) > 5 else None
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                existing = session.scalars(
                    select(DictionaryItemEntity).where(
                        DictionaryItemEntity.dictionary_code == dictionary_code,
                        DictionaryItemEntity.item_code == item_code,
                    )
                ).first()
                parent_item_id = item_id_by_code.get(parent_code or "")
                item_level = 1 if parent_item_id is None else 2
                if existing:
                    existing.dictionary_name = dictionary_name
                    existing.item_name = item_name
                    existing.parent_item_id = parent_item_id
                    existing.item_level = item_level
                    existing.sort_order = int(sort_order)
                    existing.enabled = 1
                    existing.description = description
                    existing.metadata_json = metadata_json
                    existing.updated_at = now
                    dictionary_item_id = existing.dictionary_item_id
                else:
                    dictionary_item_id = f"dict_{uuid.uuid4().hex}"
                    session.add(
                        DictionaryItemEntity(
                            dictionary_item_id=dictionary_item_id,
                            dictionary_code=dictionary_code,
                            dictionary_name=dictionary_name,
                            item_code=item_code,
                            item_name=item_name,
                            parent_item_id=parent_item_id,
                            item_level=item_level,
                            sort_order=int(sort_order),
                            enabled=1,
                            description=description,
                            metadata_json=metadata_json,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                item_id_by_code[item_code] = dictionary_item_id

    def list_dictionary_items(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """查询字典项列表，支持按字典编码过滤。"""

        clean_dictionary_code = dictionary_code.strip() if dictionary_code else None
        cached_rows = self._load_dictionary_items_cache(clean_dictionary_code)
        if cached_rows is not None:
            return cached_rows

        rows = self._list_dictionary_items_from_db(clean_dictionary_code)
        self._write_dictionary_items_cache(clean_dictionary_code, rows)
        return rows

    def _list_dictionary_items_from_db(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """直接从数据库查询字典项，供缓存未命中或刷新缓存时使用。"""

        statement = select(DictionaryItemEntity)
        if dictionary_code:
            statement = statement.where(DictionaryItemEntity.dictionary_code == dictionary_code)
        statement = statement.order_by(
            DictionaryItemEntity.dictionary_code.asc(),
            DictionaryItemEntity.item_level.asc(),
            DictionaryItemEntity.sort_order.asc(),
            DictionaryItemEntity.item_code.asc(),
        )
        with orm_session_context() as session:
            rows = session.scalars(statement).all()
        return [self._serialize_dictionary_row(row.to_dict()) for row in rows]

    def _dictionary_items_cache_key(self, dictionary_code: str | None = None) -> str:
        """生成字典项列表缓存 key，空编码表示全部字典。"""

        cache_code = dictionary_code or "all"
        return get_redis_client().build_key("dictionary", "items", cache_code)

    def _load_dictionary_items_cache(self, dictionary_code: str | None = None) -> list[dict[str, Any]] | None:
        """从 Redis 读取字典项缓存，Redis 不可用或内容异常时返回 None。"""

        cache_key = self._dictionary_items_cache_key(dictionary_code)
        cached_rows = get_redis_client().get_json(cache_key, default=None)
        if isinstance(cached_rows, list) and all(isinstance(row, dict) for row in cached_rows):
            logger.debug("[字典表] 命中Redis缓存 字典编码=%s 数量=%s", dictionary_code or "全部", len(cached_rows))
            return cached_rows
        return None

    def _write_dictionary_items_cache(
            self,
            dictionary_code: str | None,
            rows: list[dict[str, Any]],
    ) -> None:
        """把字典项列表写入 Redis，写入失败时只记录日志，不影响主流程。"""

        cache_key = self._dictionary_items_cache_key(dictionary_code)
        if get_redis_client().set_json(cache_key, rows, ttl_seconds=DICTIONARY_CACHE_TTL_SECONDS):
            logger.debug("[字典表] 写入Redis缓存 字典编码=%s 数量=%s", dictionary_code or "全部", len(rows))

    def refresh_dictionary_cache(self, dictionary_code: str) -> None:
        """刷新某个字典和全部字典缓存，用于新增或修改字典项后保持 Redis 最新。"""

        clean_dictionary_code = dictionary_code.strip()
        all_rows = self._list_dictionary_items_from_db(None)
        self._write_dictionary_items_cache(None, all_rows)
        code_rows = [row for row in all_rows if row.get("dictionary_code") == clean_dictionary_code]
        self._write_dictionary_items_cache(clean_dictionary_code, code_rows)

    def refresh_all_dictionary_cache(self) -> None:
        """刷新全部字典缓存，并按字典编码同步写入各自的 Redis key。"""

        all_rows = self._list_dictionary_items_from_db(None)
        redis_client = get_redis_client()
        stale_keys = [self._dictionary_items_cache_key(code) for code in DEPRECATED_DICTIONARY_CODES]
        redis_client.delete(*stale_keys)
        self._write_dictionary_items_cache(None, all_rows)
        rows_by_dictionary: dict[str, list[dict[str, Any]]] = {}
        for row in all_rows:
            rows_by_dictionary.setdefault(str(row["dictionary_code"]), []).append(row)
        for dictionary_code, rows in rows_by_dictionary.items():
            self._write_dictionary_items_cache(dictionary_code, rows)

    @staticmethod
    def _serialize_dictionary_row(row: dict[str, Any]) -> dict[str, Any]:
        """把数据库字典行转换为 Redis JSON 可保存的基础类型。"""

        serialized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                serialized_row[key] = value.isoformat(timespec="seconds", sep=" ")
            else:
                serialized_row[key] = value
        return serialized_row

    def get_dictionary_item_by_code(self, dictionary_code: str, item_code: str) -> dict[str, Any] | None:
        """按字典编码和字典项编码查询单个字典项。"""

        clean_item_code = item_code.strip()
        for row in self.list_dictionary_items(dictionary_code=dictionary_code):
            if str(row.get("item_code") or "") == clean_item_code:
                return row
        return None

    def upsert_dictionary_item(
            self,
            *,
            dictionary_code: str,
            dictionary_name: str,
            item_code: str,
            item_name: str,
            parent_item_id: str | None = None,
            parent_item_code: str | None = None,
            sort_order: int = 0,
            enabled: bool = True,
            description: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """新增或更新字典项。"""

        clean_dictionary_code = dictionary_code.strip()
        clean_item_code = item_code.strip()
        clean_dictionary_name = dictionary_name.strip()
        clean_item_name = item_name.strip()
        if not clean_dictionary_code or not clean_item_code or not clean_dictionary_name or not clean_item_name:
            raise ValueError("字典编码、字典名称、字典项编码和字典项名称不能为空")

        with orm_session_context() as session:
            final_parent_item_id = self._resolve_dictionary_parent_id(
                session,
                clean_dictionary_code,
                parent_item_id,
                parent_item_code,
            )
            item_level = self._resolve_dictionary_item_level(session, final_parent_item_id)
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
            now = utc_now()
            existing = session.scalars(
                select(DictionaryItemEntity).where(
                    DictionaryItemEntity.dictionary_code == clean_dictionary_code,
                    DictionaryItemEntity.item_code == clean_item_code,
                )
            ).first()
            if existing:
                dictionary_item_id = existing.dictionary_item_id
                existing.dictionary_name = clean_dictionary_name
                existing.item_name = clean_item_name
                existing.parent_item_id = final_parent_item_id
                existing.item_level = item_level
                existing.sort_order = int(sort_order)
                existing.enabled = 1 if enabled else 0
                existing.description = description
                existing.metadata_json = metadata_json
                existing.updated_at = now
            else:
                dictionary_item_id = f"dict_{uuid.uuid4().hex}"
                session.add(
                    DictionaryItemEntity(
                        dictionary_item_id=dictionary_item_id,
                        dictionary_code=clean_dictionary_code,
                        dictionary_name=clean_dictionary_name,
                        item_code=clean_item_code,
                        item_name=clean_item_name,
                        parent_item_id=final_parent_item_id,
                        item_level=item_level,
                        sort_order=int(sort_order),
                        enabled=1 if enabled else 0,
                        description=description,
                        metadata_json=metadata_json,
                        created_at=now,
                        updated_at=now,
                    )
                )

        self.refresh_dictionary_cache(clean_dictionary_code)
        item = self.get_dictionary_item_by_code(clean_dictionary_code, clean_item_code)
        if item is None:
            raise RuntimeError(f"字典项保存失败：{dictionary_item_id}")
        return item

    @staticmethod
    def _resolve_dictionary_parent_id(
            session: Session,
            dictionary_code: str,
            parent_item_id: str | None,
            parent_item_code: str | None,
    ) -> str | None:
        """解析字典父级 ID，支持直接传 ID 或传父级编码。"""

        if parent_item_id:
            row = session.get(DictionaryItemEntity, parent_item_id)
            if row is None or row.dictionary_code != dictionary_code:
                raise ValueError(f"父级字典项不存在或不属于当前字典：{parent_item_id}")
            return row.dictionary_item_id
        if not parent_item_code:
            return None

        row = session.scalars(
            select(DictionaryItemEntity).where(
                DictionaryItemEntity.dictionary_code == dictionary_code,
                DictionaryItemEntity.item_code == parent_item_code,
            )
        ).first()
        if row is None:
            raise ValueError(f"父级字典项不存在：{parent_item_code}")
        return row.dictionary_item_id

    @staticmethod
    def _resolve_dictionary_item_level(session: Session, parent_item_id: str | None) -> int:
        """根据父级字典项计算当前字典项层级。"""

        if not parent_item_id:
            return 1
        row = session.get(DictionaryItemEntity, parent_item_id)
        if row is None:
            raise ValueError(f"父级字典项不存在：{parent_item_id}")
        return int(row.item_level) + 1

    def list_enabled_dictionary_codes(self, dictionary_code: str) -> list[str]:
        """查询某个字典下已启用的字典项编码。"""

        rows = self.list_dictionary_items(dictionary_code=dictionary_code)
        return [str(row["item_code"]) for row in rows if int(row.get("enabled") or 0) == 1]

    def get_default_dictionary_code(self, dictionary_code: str) -> str:
        """查询某个字典的默认编码，默认取启用且排序最靠前的字典项。"""

        codes = self.list_enabled_dictionary_codes(dictionary_code)
        if not codes:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        return codes[0]

    def get_dictionary_code_by_metadata(self, dictionary_code: str, metadata_key: str, metadata_value: Any) -> str | None:
        """按字典项 metadata 查询编码，用于把默认项、推荐项等业务含义放到字典表维护。"""

        rows = self.list_dictionary_items(dictionary_code=dictionary_code)
        for row in rows:
            if int(row.get("enabled") or 0) != 1:
                continue
            metadata = self.parse_metadata(row.get("metadata_json"))
            if metadata.get(metadata_key) == metadata_value:
                return str(row["item_code"])
        return None

    @staticmethod
    def parse_metadata(metadata_json: str | None) -> dict[str, Any]:
        """安全解析字典项 metadata_json。"""

        if not metadata_json:
            return {}
        try:
            metadata = json.loads(str(metadata_json))
        except (json.JSONDecodeError, TypeError):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def normalize_dictionary_code(self, dictionary_code: str, value: str | None = None) -> str:
        """按字典表归一化编码；非法或空值时返回该字典的默认编码。"""

        rows = self.list_dictionary_items(dictionary_code=dictionary_code)
        enabled_codes = [str(row["item_code"]) for row in rows if int(row.get("enabled") or 0) == 1]
        if not enabled_codes:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        default_code = enabled_codes[0]
        normalized_value = str(value or default_code).strip().lower()
        if normalized_value in enabled_codes:
            return normalized_value
        return default_code

    def create_document(
            self,
            *,
            document_id: str,
            filename: str,
            file_path: str,
            file_type: str,
            file_md5: str,
            file_size: int,
            storage_type: str = "minio",
            bucket_name: str | None = None,
            object_name: str | None = None,
            public_url: str | None = None,
            status: str = "uploaded",
            collection_name: str = "agent",
            document_type: str = "text",
            split_strategy: str = "recursive",
    ) -> DocumentEntity:
        """创建知识库文档元数据。"""

        now = utc_now()
        document = DocumentEntity(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            storage_type=storage_type,
            bucket_name=bucket_name,
            object_name=object_name,
            public_url=public_url,
            file_type=file_type,
            file_md5=file_md5,
            file_size=int(file_size),
            status=status,
            version=1,
            chunk_count=0,
            collection_name=collection_name,
            document_type=document_type,
            split_strategy=split_strategy,
            created_at=now,
            updated_at=now,
            error_message=None,
        )
        with orm_session_context() as session:
            session.add(document)
        created = self.get_document(document_id)
        if created is None:
            raise RuntimeError(f"Document {document_id} was not created")
        return created

    def find_active_document_by_md5(
            self,
            file_md5: str,
            collection_name: str | None = None,
    ) -> DocumentEntity | None:
        """按文件 MD5 查找未删除文档；传入 collection 时只在该 collection 内去重。"""

        conditions = [
            DocumentEntity.file_md5 == file_md5,
            DocumentEntity.status != "deleted",
        ]
        if collection_name:
            conditions.append(DocumentEntity.collection_name == collection_name)
        statement = (
            select(DocumentEntity)
            .where(*conditions)
            .order_by(DocumentEntity.created_at.desc())
            .limit(1)
        )
        with orm_session_context() as session:
            return session.scalars(statement).first()

    def get_document(self, document_id: str) -> DocumentEntity | None:
        """按 ID 查询文档。"""

        with orm_session_context() as session:
            return session.get(DocumentEntity, document_id)

    def list_documents(self, *, include_training: bool = False) -> list[DocumentEntity]:
        """查询全部未删除文档。"""

        conditions = [DocumentEntity.status != "deleted"]
        if not include_training:
            conditions.append(DocumentEntity.collection_name.not_in(TRAINING_COLLECTION_NAMES))
        statement = (
            select(DocumentEntity)
            .where(*conditions)
            .order_by(DocumentEntity.created_at.desc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

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
        """更新文档索引状态。"""

        with orm_session_context() as session:
            document = session.get(DocumentEntity, document_id)
            if document is None:
                raise ValueError(f"Document {document_id} does not exist")
            document.status = status
            document.chunk_count = int(document.chunk_count if chunk_count is None else chunk_count)
            document.error_message = error_message
            if increment_version:
                document.version = int(document.version) + 1
            document.collection_name = collection_name or document.collection_name or "agent"
            document.document_type = document_type or document.document_type or "text"
            document.split_strategy = split_strategy or document.split_strategy or "recursive"
            document.updated_at = utc_now()

    def mark_document_deleted(self, document_id: str) -> None:
        """把文档标记为删除。"""

        self.update_document_status(document_id, "deleted")

    def delete_document(self, document_id: str) -> bool:
        """从 documents 表物理删除文件资产记录。"""

        with orm_session_context() as session:
            document = session.get(DocumentEntity, document_id)
            if document is None:
                return False
            session.delete(document)
            return True

    def ensure_conversation(
            self,
            *,
            conversation_id: str | None = None,
            user_id: str | None = None,
            title: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ConversationEntity:
        """确保会话存在，已删除会话不会复用原 ID。"""

        clean_conversation_id = (conversation_id or "").strip()
        if clean_conversation_id:
            existing = self.get_conversation(clean_conversation_id)
            if existing is not None and existing.get("status") != "deleted":
                return existing
            if existing is not None and existing.get("status") == "deleted":
                clean_conversation_id = f"conv_{uuid.uuid4().hex}"
        else:
            clean_conversation_id = f"conv_{uuid.uuid4().hex}"

        now = utc_now()
        clean_title = (title or "").strip()[:80] or None
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        conversation = ConversationEntity(
            conversation_id=clean_conversation_id,
            user_id=user_id,
            title=clean_title,
            status="active",
            message_count=0,
            summary=None,
            metadata_json=metadata_json,
            created_at=now,
            updated_at=now,
            last_message_at=None,
        )
        with orm_session_context() as session:
            session.add(conversation)
        created = self.get_conversation(clean_conversation_id)
        if created is None:
            raise RuntimeError(f"Conversation {clean_conversation_id} was not created")
        return created

    def get_conversation(self, conversation_id: str) -> ConversationEntity | None:
        """按 ID 查询会话。"""

        with orm_session_context() as session:
            return session.get(ConversationEntity, conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除聊天记录。

        会话主表保留 deleted 状态；消息明细直接删除，避免正文继续留存。
        """

        now = utc_now()
        with orm_session_context() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None or conversation.status == "deleted":
                return False
            messages = session.scalars(
                select(ConversationMessageEntity).where(
                    ConversationMessageEntity.conversation_id == conversation_id
                )
            ).all()
            for message in messages:
                session.delete(message)
            conversation.status = "deleted"
            conversation.message_count = 0
            conversation.updated_at = now
            conversation.last_message_at = now
            return True

    def list_conversations(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            user_id: str | None = None,
            keyword: str | None = None,
    ) -> tuple[list[ConversationEntity], int]:
        """分页查询会话列表。"""

        final_page = max(1, int(page))
        final_page_size = max(1, min(int(page_size), 50))
        offset = (final_page - 1) * final_page_size
        conditions = [ConversationEntity.status != "deleted"]
        if user_id:
            conditions.append(ConversationEntity.user_id == user_id)
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{self._escape_like_keyword(clean_keyword)}%"
            conditions.append(
                or_(
                    ConversationEntity.title.like(like_keyword, escape="\\"),
                    ConversationEntity.user_id.like(like_keyword, escape="\\"),
                    ConversationEntity.conversation_id.like(like_keyword, escape="\\"),
                )
            )

        count_statement = select(func.count()).select_from(ConversationEntity).where(*conditions)
        list_statement = (
            select(ConversationEntity)
            .where(*conditions)
            .order_by(
                func.coalesce(
                    ConversationEntity.last_message_at,
                    ConversationEntity.updated_at,
                    ConversationEntity.created_at,
                ).desc()
            )
            .limit(final_page_size)
            .offset(offset)
        )
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.scalars(list_statement).all()
        return rows, total

    @staticmethod
    def _escape_like_keyword(keyword: str) -> str:
        """转义 LIKE 通配符，避免用户输入被当成模式语法。"""

        return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def list_recent_messages(self, conversation_id: str, limit: int = 20) -> list[ConversationMessageEntity]:
        """查询最近若干条会话消息，并按正序返回。"""

        final_limit = max(1, min(int(limit), 100))
        statement = (
            select(ConversationMessageEntity)
            .where(ConversationMessageEntity.conversation_id == conversation_id)
            .order_by(ConversationMessageEntity.sequence_no.desc())
            .limit(final_limit)
        )
        with orm_session_context() as session:
            rows = session.scalars(statement).all()
        return list(reversed(rows))

    def list_conversation_messages(self, conversation_id: str) -> list[ConversationMessageEntity]:
        """查询某个会话的全部消息。"""

        statement = (
            select(ConversationMessageEntity)
            .where(ConversationMessageEntity.conversation_id == conversation_id)
            .order_by(ConversationMessageEntity.sequence_no.asc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

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
    ) -> ConversationMessageEntity:
        """新增一条会话消息并刷新会话统计。"""

        if self.get_conversation(conversation_id) is None:
            self.ensure_conversation(conversation_id=conversation_id)

        now = utc_now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        message_id = f"msg_{uuid.uuid4().hex}"
        with orm_session_context() as session:
            next_sequence_no = int(
                session.scalar(
                    select(func.coalesce(func.max(ConversationMessageEntity.sequence_no), 0) + 1).where(
                        ConversationMessageEntity.conversation_id == conversation_id
                    )
                )
                or 1
            )
            message = ConversationMessageEntity(
                message_id=message_id,
                conversation_id=conversation_id,
                sequence_no=next_sequence_no,
                role=role,
                content=content,
                content_type=content_type,
                model_name=model_name,
                token_count=token_count,
                metadata_json=metadata_json,
                created_at=now,
            )
            session.add(message)
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is not None:
                conversation.message_count = int(conversation.message_count or 0) + 1
                conversation.updated_at = now
                conversation.last_message_at = now

        created = self.get_message(message_id)
        if created is None:
            raise RuntimeError(f"消息保存失败：{message_id}")
        return created

    def get_message(self, message_id: str) -> ConversationMessageEntity | None:
        """按 ID 查询消息。"""

        with orm_session_context() as session:
            return session.get(ConversationMessageEntity, message_id)

    def save_chat_exchange(
            self,
            *,
            conversation_id: str,
            user_message: str,
            assistant_message: str,
            model_name: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        """保存一问一答两条聊天消息。"""

        self.add_message(conversation_id=conversation_id, role="user", content=user_message)
        self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            model_name=model_name,
            metadata=metadata,
        )

    def create_exam_session(
            self,
            *,
            session_id: str | None = None,
            user_id: str | None = None,
            title: str | None = None,
            collection_name: str,
            document_id: str | None = None,
            filename: str | None = None,
            section_path: str | None = None,
            round_count: int,
            question_types: list[str],
            model_mode: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ExamSessionEntity:
        """创建一场对话式考试会话。"""

        now = utc_now()
        clean_session_id = (session_id or "").strip() or f"exam_{uuid.uuid4().hex}"
        exam_session = ExamSessionEntity(
            session_id=clean_session_id,
            user_id=user_id,
            title=(title or "").strip()[:120] or "知识掌握度测评",
            collection_name=collection_name,
            document_id=document_id,
            filename=filename,
            section_path=section_path,
            round_count=int(round_count),
            question_types_json=json.dumps(question_types, ensure_ascii=False),
            status="active",
            current_round=1,
            answered_count=0,
            total_score=0,
            max_score=100,
            model_mode=model_mode,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False) if metadata else None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        with orm_session_context() as session:
            session.add(exam_session)
        created = self.get_exam_session(clean_session_id)
        if created is None:
            raise RuntimeError(f"考试会话创建失败：{clean_session_id}")
        return created

    def add_exam_question(
            self,
            *,
            session_id: str,
            round_no: int,
            source_question_id: str | None,
            source_document_id: str | None,
            source_filename: str | None,
            source_page: int | None,
            section_path: str | None,
            question_type: str,
            prompt: str,
            options: list[str] | None,
            correct_answer: Any,
            reference_answer: str,
            max_score: float,
    ) -> ExamQuestionEntity:
        """保存考试会话中的单轮题目。"""

        now = utc_now()
        exam_question_id = f"exam_q_{uuid.uuid4().hex}"
        question = ExamQuestionEntity(
            exam_question_id=exam_question_id,
            session_id=session_id,
            round_no=int(round_no),
            source_question_id=source_question_id,
            source_document_id=source_document_id,
            source_filename=source_filename,
            source_page=source_page,
            section_path=section_path,
            question_type=question_type,
            prompt=prompt,
            options_json=json.dumps(options or [], ensure_ascii=False),
            correct_answer_json=json.dumps(correct_answer, ensure_ascii=False),
            reference_answer=reference_answer,
            user_answer=None,
            is_correct=None,
            score=None,
            max_score=float(max_score),
            analysis_json=None,
            status="pending",
            created_at=now,
            answered_at=None,
        )
        with orm_session_context() as session:
            session.add(question)
        created = self.get_exam_question(exam_question_id=exam_question_id)
        if created is None:
            raise RuntimeError(f"考试题目保存失败：{exam_question_id}")
        return created

    def get_exam_session(self, session_id: str) -> ExamSessionEntity | None:
        """按考试会话编号查询考试会话。"""

        with orm_session_context() as session:
            return session.get(ExamSessionEntity, session_id)

    def get_exam_question(
            self,
            *,
            exam_question_id: str | None = None,
            session_id: str | None = None,
            round_no: int | None = None,
    ) -> ExamQuestionEntity | None:
        """查询单道考试题目，支持按题目编号或会话轮次定位。"""

        with orm_session_context() as session:
            if exam_question_id:
                return session.get(ExamQuestionEntity, exam_question_id)
            if session_id and round_no is not None:
                statement = select(ExamQuestionEntity).where(
                    ExamQuestionEntity.session_id == session_id,
                    ExamQuestionEntity.round_no == int(round_no),
                )
                return session.scalars(statement).first()
            return None

    def list_exam_questions(self, session_id: str) -> list[ExamQuestionEntity]:
        """查询某场考试的全部题目。"""

        statement = (
            select(ExamQuestionEntity)
            .where(ExamQuestionEntity.session_id == session_id)
            .order_by(ExamQuestionEntity.round_no.asc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

    def answer_exam_question(
            self,
            *,
            session_id: str,
            exam_question_id: str,
            user_answer: str,
            is_correct: bool,
            score: float,
            analysis: dict[str, Any],
    ) -> ExamQuestionEntity:
        """保存用户单轮作答和分析结果，并刷新考试会话分数。"""

        now = utc_now()
        with orm_session_context() as session:
            question = session.get(ExamQuestionEntity, exam_question_id)
            if question is None or question.session_id != session_id:
                raise RuntimeError(f"考试题目不存在：{exam_question_id}")
            question.user_answer = user_answer
            question.is_correct = 1 if is_correct else 0
            question.score = float(score)
            question.analysis_json = json.dumps(analysis, ensure_ascii=False)
            question.status = "answered"
            question.answered_at = now

            answered_count = int(
                session.scalar(
                    select(func.count()).select_from(ExamQuestionEntity).where(
                        ExamQuestionEntity.session_id == session_id,
                        ExamQuestionEntity.status == "answered",
                    )
                )
                or 0
            )
            total_score = float(
                session.scalar(
                    select(func.coalesce(func.sum(ExamQuestionEntity.score), 0)).where(
                        ExamQuestionEntity.session_id == session_id,
                        ExamQuestionEntity.status == "answered",
                    )
                )
                or 0
            )
            exam_session = session.get(ExamSessionEntity, session_id)
            if exam_session is not None:
                round_count = int(exam_session.round_count or 0)
                completed = answered_count >= round_count > 0
                exam_session.answered_count = answered_count
                exam_session.total_score = total_score
                exam_session.current_round = min(answered_count + 1, round_count)
                exam_session.status = "completed" if completed else "active"
                exam_session.updated_at = now
                exam_session.completed_at = now if completed else None

        answered = self.get_exam_question(exam_question_id=exam_question_id)
        if answered is None:
            raise RuntimeError(f"考试题目不存在：{exam_question_id}")
        return answered

    def list_exam_sessions(
            self,
            *,
            page: int = 1,
            page_size: int = 10,
            user_id: str | None = None,
            keyword: str | None = None,
    ) -> tuple[list[ExamSessionEntity], int]:
        """分页查询考试会话记录。"""

        final_page = max(1, int(page))
        final_page_size = max(1, min(int(page_size), 50))
        offset = (final_page - 1) * final_page_size
        conditions = []
        if user_id:
            conditions.append(ExamSessionEntity.user_id == user_id)
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{self._escape_like_keyword(clean_keyword)}%"
            conditions.append(
                or_(
                    ExamSessionEntity.title.like(like_keyword, escape="\\"),
                    ExamSessionEntity.filename.like(like_keyword, escape="\\"),
                    ExamSessionEntity.section_path.like(like_keyword, escape="\\"),
                )
            )

        count_statement = select(func.count()).select_from(ExamSessionEntity)
        list_statement = (
            select(ExamSessionEntity)
            .order_by(ExamSessionEntity.updated_at.desc(), ExamSessionEntity.created_at.desc())
            .limit(final_page_size)
            .offset(offset)
        )
        if conditions:
            count_statement = count_statement.where(*conditions)
            list_statement = list_statement.where(*conditions)
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.scalars(list_statement).all()
        return rows, total
