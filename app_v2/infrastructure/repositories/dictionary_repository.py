"""字典仓储。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from domain.entities import DictionaryItemEntity
from infrastructure.orm_session import orm_session_context
from training.repository import utc_now
from utils.redis_client import RedisClient, get_redis_client


class DictionaryRepository:
    """封装 dictionary_items 表访问。

    这里使用仓储模式，把字典 SQL 从旧 KnowledgeStore 拆出来。应用服务只关心“查、存、删、归一化”，
    不需要知道底层是 MySQL 还是 Redis 缓存。
    """

    _cache_ttl_seconds = 600

    def __init__(self, store: Any | None = None, redis_client: RedisClient | None = None):
        # store 参数只保留给旧测试和过渡调用占位，真实数据访问已经改为 ORM。
        self.store = store
        self.redis_client = redis_client or get_redis_client()

    def list_items(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """查询字典项列表，支持按字典编码过滤并使用 Redis 缓存。"""

        clean_dictionary_code = dictionary_code.strip() if dictionary_code else None
        cache_key = self._items_cache_key(clean_dictionary_code)
        cached_rows = self.redis_client.get_json(cache_key, default=None)
        if isinstance(cached_rows, list):
            return cached_rows

        rows = self._list_items_from_db(clean_dictionary_code)
        self.redis_client.set_json(cache_key, rows, ttl_seconds=self._cache_ttl_seconds)
        return rows

    def upsert_item(self, **values: Any) -> dict[str, Any]:
        """新增或修改字典项，并清理相关缓存。"""

        dictionary_code = str(values.get("dictionary_code") or "").strip()
        item_code = str(values.get("item_code") or "").strip()
        dictionary_name = str(values.get("dictionary_name") or "").strip()
        item_name = str(values.get("item_name") or "").strip()
        if not dictionary_code or not item_code or not dictionary_name or not item_name:
            raise ValueError("字典编码、字典名称、字典项编码和字典项名称不能为空")

        with orm_session_context() as session:
            parent_item_id = self._resolve_parent_item_id(
                session,
                dictionary_code,
                values.get("parent_item_id"),
                values.get("parent_item_code"),
            )
            item_level = self._resolve_item_level(session, parent_item_id)
            metadata = values.get("metadata")
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
            now = utc_now()
            existing = session.scalars(
                select(DictionaryItemEntity).where(
                    DictionaryItemEntity.dictionary_code == dictionary_code,
                    DictionaryItemEntity.item_code == item_code,
                )
            ).first()
            if existing is None:
                dictionary_item_id = self._new_dictionary_item_id(dictionary_code, item_code)
                session.add(DictionaryItemEntity(
                    dictionary_item_id=dictionary_item_id,
                    dictionary_code=dictionary_code,
                    dictionary_name=dictionary_name,
                    item_code=item_code,
                    item_name=item_name,
                    parent_item_id=parent_item_id,
                    item_level=item_level,
                    sort_order=int(values.get("sort_order") or 0),
                    enabled=1 if bool(values.get("enabled", True)) else 0,
                    description=values.get("description"),
                    metadata_json=metadata_json,
                    created_at=now,
                    updated_at=now,
                ))
            else:
                dictionary_item_id = existing.dictionary_item_id
                existing.dictionary_name = dictionary_name
                existing.item_name = item_name
                existing.parent_item_id = parent_item_id
                existing.item_level = item_level
                existing.sort_order = int(values.get("sort_order") or 0)
                existing.enabled = 1 if bool(values.get("enabled", True)) else 0
                existing.description = values.get("description")
                existing.metadata_json = metadata_json
                existing.updated_at = now

        self.clear_cache(dictionary_code)
        item = self._get_item_by_code(dictionary_code, item_code)
        if item is None:
            raise RuntimeError(f"字典项保存失败：{dictionary_item_id}")
        return item

    def delete_item(self, dictionary_item_id: str) -> bool:
        """按字典项 ID 物理删除字典项，并清理缓存。"""

        deleted_dictionary_code: str | None = None
        with orm_session_context() as session:
            item = session.get(DictionaryItemEntity, dictionary_item_id)
            if item is None:
                return False
            deleted_dictionary_code = item.dictionary_code
            session.delete(item)
        self.clear_cache(deleted_dictionary_code)
        return True

    def normalize_code(self, dictionary_code: str, value: str | None = None) -> str:
        """把外部传入的字典编码归一化为启用项。"""

        enabled_codes = self.list_enabled_codes(dictionary_code)
        if not enabled_codes:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        default_code = enabled_codes[0]
        normalized_value = str(value or default_code).strip().lower()
        if normalized_value in enabled_codes:
            return normalized_value
        return default_code

    def list_enabled_codes(self, dictionary_code: str) -> list[str]:
        """查询某个字典下已启用的字典项编码。"""

        rows = self.list_items(dictionary_code=dictionary_code)
        return [str(row["item_code"]) for row in rows if int(row.get("enabled") or 0) == 1]

    def get_default_code(self, dictionary_code: str) -> str:
        """查询某个字典的默认编码，默认取启用且排序最靠前的字典项。"""

        codes = self.list_enabled_codes(dictionary_code)
        if not codes:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        return codes[0]

    def get_code_by_metadata(self, dictionary_code: str, metadata_key: str, metadata_value: Any) -> str | None:
        """按字典项 metadata 查询编码，用于把默认项、推荐项等业务含义放到字典表维护。"""

        for row in self.list_items(dictionary_code=dictionary_code):
            if int(row.get("enabled") or 0) != 1:
                continue
            metadata = self.parse_metadata(row.get("metadata_json"))
            if metadata.get(metadata_key) == metadata_value:
                return str(row["item_code"])
        return None

    def clear_cache(self, dictionary_code: str | None = None) -> None:
        """清理字典缓存。

        指定字典编码时，同时清理“全部字典”缓存，避免父级列表继续读到旧数据。
        """

        keys = [self._items_cache_key(None)]
        if dictionary_code:
            keys.append(self._items_cache_key(str(dictionary_code)))
        self.redis_client.delete(*keys)

    def _list_items_from_db(self, dictionary_code: str | None = None) -> list[dict[str, Any]]:
        """直接从数据库查询字典项，供缓存未命中时使用。"""

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
        return [self._serialize_row(row.to_dict()) for row in rows]

    def _get_item_by_code(self, dictionary_code: str, item_code: str) -> dict[str, Any] | None:
        """按字典编码和字典项编码查询单个字典项。"""

        statement = select(DictionaryItemEntity).where(
            DictionaryItemEntity.dictionary_code == dictionary_code,
            DictionaryItemEntity.item_code == item_code,
        )
        with orm_session_context() as session:
            row = session.scalars(statement).first()
        return self._serialize_row(row.to_dict()) if row else None

    @staticmethod
    def _resolve_parent_item_id(session, dictionary_code: str, parent_item_id: str | None, parent_item_code: str | None) -> str | None:
        """解析父级字典项，支持直接传 ID 或传父级编码。"""

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
    def _resolve_item_level(session, parent_item_id: str | None) -> int:
        """根据父级字典项计算当前字典项层级。"""

        if not parent_item_id:
            return 1
        row = session.get(DictionaryItemEntity, parent_item_id)
        if row is None:
            raise ValueError(f"父级字典项不存在：{parent_item_id}")
        return int(row.item_level) + 1

    @staticmethod
    def _new_dictionary_item_id(dictionary_code: str, item_code: str) -> str:
        """生成稳定可读的字典项 ID，便于排查数据。"""

        safe_dictionary_code = "".join(ch if ch.isalnum() else "_" for ch in dictionary_code).strip("_")
        safe_item_code = "".join(ch if ch.isalnum() else "_" for ch in item_code).strip("_")
        return f"dict_{safe_dictionary_code}_{safe_item_code}"

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

    @staticmethod
    def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
        """统一字典项输出形态，避免上层关心 ORM 对象。"""

        return {
            **row,
            "enabled": int(row.get("enabled") or 0),
            "item_level": int(row.get("item_level") or 1),
            "sort_order": int(row.get("sort_order") or 0),
        }

    def _items_cache_key(self, dictionary_code: str | None) -> str:
        """生成字典列表缓存 key。"""

        return self.redis_client.build_key("v2", "dictionary", "items", dictionary_code or "all")