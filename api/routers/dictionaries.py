from fastapi import APIRouter, HTTPException, Query

from api.schemas import DictionaryGroupResponse, DictionaryItemResponse, DictionaryItemUpsertRequest
from rag.knowledge_store import KnowledgeStore
from utils.logger_handler import logger

router = APIRouter()


def _get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据存储实例。"""

    return KnowledgeStore()


def _item_to_response(item: dict, children: list[DictionaryItemResponse] | None = None) -> DictionaryItemResponse:
    """把数据库字典行转换为接口响应对象。"""

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


def build_dictionary_groups(items: list[dict]) -> list[DictionaryGroupResponse]:
    """把数据库字典行按字典编码分组，并组装成树形响应。

    这个函数是字典接口的门面函数，训练模块也会复用它，避免两边各写一套树构建逻辑。
    """

    groups: dict[str, list[dict]] = {}
    for row in items:
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
    return result


@router.get("/dictionaries", response_model=list[DictionaryGroupResponse])
def list_dictionaries(
        dictionary_code: str | None = Query(default=None, description="字典编码；为空时返回全部字典"),
) -> list[DictionaryGroupResponse]:
    """查询系统字典，返回按字典编码分组后的多层级结构。"""

    store = _get_knowledge_store()
    rows = store.list_dictionary_items(dictionary_code=dictionary_code)
    result = build_dictionary_groups(rows)

    logger.info("[字典表] 查询字典完成 字典编码=%s 字典数量=%s", dictionary_code or "全部", len(result))
    return result


@router.post("/dictionaries/items", response_model=DictionaryItemResponse)
def upsert_dictionary_item(request: DictionaryItemUpsertRequest) -> DictionaryItemResponse:
    """新增或更新字典项。"""

    store = _get_knowledge_store()
    try:
        item = store.upsert_dictionary_item(
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

    logger.info(
        "[字典表] 保存字典项完成 字典编码=%s 字典项编码=%s 启用=%s",
        request.dictionary_code,
        request.item_code,
        request.enabled,
    )
    return _item_to_response(item)
