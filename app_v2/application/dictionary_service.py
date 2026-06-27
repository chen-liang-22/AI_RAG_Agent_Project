"""字典应用服务。"""

import json
from typing import Any

from fastapi import HTTPException

from api.schemas import DictionaryGroupResponse, DictionaryItemResponse
from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from core.utils.logger_handler import logger


class DictionaryApplicationService:
    """字典业务外观。

    路由层不直接接触 `KnowledgeStore`，后续字典缓存、分页和权限都放在这里扩展。
    """

    def __init__(self, repository: DictionaryRepository | None = None):
        self.repository = repository or DictionaryRepository()

    def list_groups(self, dictionary_code: str | None = None) -> list[DictionaryGroupResponse]:
        """查询字典分组并组织成父子树。"""

        rows = self.repository.list_items(dictionary_code=dictionary_code)
        groups = self._build_dictionary_groups(rows)
        logger.info("[V2字典] 查询字典分组 字典编码=%s 分组数量=%s", dictionary_code or "全部", len(groups))
        return groups

    def upsert_item(self, request) -> DictionaryItemResponse:
        """新增或修改字典项。"""

        try:
            item = self.repository.upsert_item(
                dictionary_code=request.dictionary_code,
                dictionary_name=request.dictionary_name,
                item_code=request.item_code,
                item_name=request.item_name,
                parent_item_id=request.parent_item_id,
                parent_item_code=request.parent_item_code,
                sort_order=request.sort_order,
                enabled=request.enabled,
                description=request.description,
                metadata=request.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("[V2字典] 保存字典项 字典编码=%s 字典项=%s", request.dictionary_code, request.item_code)
        return self._item_to_response(item)

    def delete_item(self, dictionary_item_id: str) -> dict[str, str]:
        """删除字典项。"""

        deleted = self.repository.delete_item(dictionary_item_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="字典项不存在或当前仓储不支持删除")
        logger.info("[V2字典] 删除字典项 字典项ID=%s", dictionary_item_id)
        return {"status": "deleted", "dictionary_item_id": dictionary_item_id}

    @classmethod
    def _build_dictionary_groups(cls, items: list[dict[str, Any]]) -> list[DictionaryGroupResponse]:
        """按 dictionary_code 归组。"""

        groups: dict[str, list[dict[str, Any]]] = {}
        for row in items:
            groups.setdefault(row["dictionary_code"], []).append(row)

        result: list[DictionaryGroupResponse] = []
        for code, group_rows in groups.items():
            dictionary_name = group_rows[0]["dictionary_name"] if group_rows else code
            result.append(DictionaryGroupResponse(
                dictionary_code=code,
                dictionary_name=dictionary_name,
                items=cls._build_dictionary_tree(group_rows),
            ))
        return result

    @classmethod
    def _build_dictionary_tree(cls, items: list[dict[str, Any]]) -> list[DictionaryItemResponse]:
        """把平铺字典项组装成父子树。"""

        children_by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for item in items:
            children_by_parent.setdefault(item.get("parent_item_id"), []).append(item)

        def build_children(parent_item_id: str | None) -> list[DictionaryItemResponse]:
            """递归组装某个父级下面的子字典项。"""

            children = []
            for child in children_by_parent.get(parent_item_id, []):
                children.append(cls._item_to_response(child, build_children(child["dictionary_item_id"])))
            return children

        return build_children(None)

    @staticmethod
    def _item_to_response(item: dict[str, Any], children: list[DictionaryItemResponse] | None = None) -> DictionaryItemResponse:
        """把数据库行转换成字典响应对象。"""

        metadata = {}
        metadata_json = item.get("metadata_json")
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, ValueError):
                metadata = {}

        return DictionaryItemResponse(
            dictionary_item_id=item["dictionary_item_id"],
            dictionary_code=item["dictionary_code"],
            dictionary_name=item["dictionary_name"],
            item_code=item["item_code"],
            item_name=item["item_name"],
            parent_item_id=item.get("parent_item_id"),
            item_level=int(item["item_level"]),
            sort_order=int(item["sort_order"]),
            enabled=bool(item["enabled"]),
            description=item.get("description"),
            metadata=metadata,
            children=children or [],
        )
