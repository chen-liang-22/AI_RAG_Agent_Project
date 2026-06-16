from fastapi import APIRouter, Query

from api.schemas import DictionaryGroupResponse, DictionaryItemResponse
from rag.knowledge_store import KnowledgeStore
from utils.logger_handler import logger

router = APIRouter()


def _get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据存储实例。"""

    return KnowledgeStore()


def _item_to_response(item: dict, children: list[DictionaryItemResponse] | None = None) -> DictionaryItemResponse:
    """把 SQLite 字典行转换为接口响应对象。"""

    metadata_json = item.get("metadata_json")
    metadata = {}
    if metadata_json:
        try:
            import json

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


def _build_dictionary_tree(items: list[dict]) -> list[DictionaryItemResponse]:
    """把平铺字典项组装成树形结构。"""

    children_by_parent: dict[str | None, list[dict]] = {}
    for item in items:
        children_by_parent.setdefault(item.get("parent_item_id"), []).append(item)

    def build_children(parent_item_id: str | None) -> list[DictionaryItemResponse]:
        """递归构建某个父节点下的子字典项。"""

        children = []
        for child in children_by_parent.get(parent_item_id, []):
            child_items = build_children(child["dictionary_item_id"])
            children.append(_item_to_response(child, child_items))
        return children

    return build_children(None)


@router.get("/dictionaries", response_model=list[DictionaryGroupResponse])
def list_dictionaries(
        dictionary_code: str | None = Query(default=None, description="字典编码；为空时返回全部字典"),
) -> list[DictionaryGroupResponse]:
    """查询系统字典，返回按字典编码分组后的多层级结构。"""

    store = _get_knowledge_store()
    rows = store.list_dictionary_items(dictionary_code=dictionary_code)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["dictionary_code"], []).append(row)

    result = []
    for code, group_rows in groups.items():
        dictionary_name = group_rows[0]["dictionary_name"] if group_rows else code
        result.append(
            DictionaryGroupResponse(
                dictionary_code=code,
                dictionary_name=dictionary_name,
                items=_build_dictionary_tree(group_rows),
            )
        )

    logger.info("[字典表] 查询字典完成 字典编码=%s 字典数量=%s", dictionary_code or "全部", len(result))
    return result
