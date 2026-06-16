import sqlite3

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    DictionaryGroupCreateRequest,
    DictionaryGroupResponse,
    DictionaryGroupUpdateRequest,
    DictionaryItemCreateRequest,
    DictionaryItemResponse,
    DictionaryItemUpdateRequest,
)
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
    group_rows = store.list_dictionary_groups(dictionary_code=dictionary_code)
    rows = store.list_dictionary_items(dictionary_code=dictionary_code)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["dictionary_code"], []).append(row)

    result = []
    for group_row in group_rows:
        code = group_row["dictionary_code"]
        item_rows = groups.get(code, [])
        result.append(
            DictionaryGroupResponse(
                dictionary_code=code,
                dictionary_name=group_row["dictionary_name"],
                items=_build_dictionary_tree(item_rows),
            )
        )

    logger.info("[字典表] 查询字典完成 字典编码=%s 字典数量=%s", dictionary_code or "全部", len(result))
    return result


@router.post("/dictionaries", response_model=DictionaryGroupResponse)
def create_dictionary_group(request: DictionaryGroupCreateRequest) -> DictionaryGroupResponse:
    """新增父级字典。"""

    store = _get_knowledge_store()
    try:
        row = store.create_dictionary_group(
            request.dictionary_code.strip(),
            request.dictionary_name.strip(),
        )
    except sqlite3.IntegrityError as exc:
        logger.info("[字典表] 新增父级字典失败 原因=编码重复 字典编码=%s", request.dictionary_code)
        raise HTTPException(status_code=409, detail="父级字典编码已存在") from exc

    logger.info("[字典表] 新增父级字典完成 字典编码=%s", row["dictionary_code"])
    return DictionaryGroupResponse(
        dictionary_code=row["dictionary_code"],
        dictionary_name=row["dictionary_name"],
        items=[],
    )


@router.put("/dictionaries/{dictionary_code}", response_model=DictionaryGroupResponse)
def update_dictionary_group(
        dictionary_code: str,
        request: DictionaryGroupUpdateRequest,
) -> DictionaryGroupResponse:
    """修改父级字典。"""

    store = _get_knowledge_store()
    row = store.update_dictionary_group(dictionary_code, request.dictionary_name.strip())
    if row is None:
        raise HTTPException(status_code=404, detail="父级字典不存在")

    logger.info("[字典表] 修改父级字典完成 字典编码=%s", dictionary_code)
    groups = list_dictionaries(dictionary_code=dictionary_code)
    return groups[0] if groups else DictionaryGroupResponse(
        dictionary_code=row["dictionary_code"],
        dictionary_name=row["dictionary_name"],
        items=[],
    )


@router.post("/dictionaries/items", response_model=DictionaryItemResponse)
def create_dictionary_item(request: DictionaryItemCreateRequest) -> DictionaryItemResponse:
    """新增字典项。"""

    store = _get_knowledge_store()
    try:
        row = store.create_dictionary_item(
            dictionary_code=request.dictionary_code.strip(),
            dictionary_name=request.dictionary_name.strip(),
            item_code=request.item_code.strip(),
            item_name=request.item_name.strip(),
            parent_item_id=request.parent_item_id,
            sort_order=request.sort_order,
            enabled=request.enabled,
            description=request.description,
            metadata=request.metadata,
        )
    except sqlite3.IntegrityError as exc:
        logger.info("[字典表] 新增字典项失败 原因=编码重复 字典编码=%s 项编码=%s", request.dictionary_code, request.item_code)
        raise HTTPException(status_code=409, detail="同一字典下的字典项编码已存在") from exc
    except ValueError as exc:
        logger.info("[字典表] 新增字典项失败 原因=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[字典表] 新增字典项完成 字典编码=%s 项编码=%s", row["dictionary_code"], row["item_code"])
    return _item_to_response(row)


@router.put("/dictionaries/items/{dictionary_item_id}", response_model=DictionaryItemResponse)
def update_dictionary_item(
        dictionary_item_id: str,
        request: DictionaryItemUpdateRequest,
) -> DictionaryItemResponse:
    """修改字典项。"""

    store = _get_knowledge_store()
    try:
        row = store.update_dictionary_item(
            dictionary_item_id,
            dictionary_name=request.dictionary_name.strip() if request.dictionary_name else None,
            item_code=request.item_code.strip() if request.item_code else None,
            item_name=request.item_name.strip() if request.item_name else None,
            parent_item_id=request.parent_item_id,
            sort_order=request.sort_order,
            enabled=request.enabled,
            description=request.description,
            metadata=request.metadata,
        )
    except sqlite3.IntegrityError as exc:
        logger.info("[字典表] 修改字典项失败 原因=编码重复 字典项ID=%s", dictionary_item_id)
        raise HTTPException(status_code=409, detail="同一字典下的字典项编码已存在") from exc
    except ValueError as exc:
        logger.info("[字典表] 修改字典项失败 原因=%s 字典项ID=%s", exc, dictionary_item_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if row is None:
        raise HTTPException(status_code=404, detail="字典项不存在")

    logger.info("[字典表] 修改字典项完成 字典项ID=%s", dictionary_item_id)
    return _item_to_response(row)


@router.patch("/dictionaries/items/{dictionary_item_id}/enabled", response_model=DictionaryItemResponse)
def set_dictionary_item_enabled(
        dictionary_item_id: str,
        enabled: bool = Query(..., description="是否启用"),
) -> DictionaryItemResponse:
    """启用或禁用字典项。"""

    store = _get_knowledge_store()
    row = store.set_dictionary_item_enabled(dictionary_item_id, enabled)
    if row is None:
        raise HTTPException(status_code=404, detail="字典项不存在")

    logger.info("[字典表] 更新启用状态完成 字典项ID=%s 启用=%s", dictionary_item_id, enabled)
    return _item_to_response(row)


@router.delete("/dictionaries/items/{dictionary_item_id}")
def delete_dictionary_item(dictionary_item_id: str) -> dict:
    """删除字典项。"""

    store = _get_knowledge_store()
    try:
        deleted = store.delete_dictionary_item(dictionary_item_id)
    except ValueError as exc:
        logger.info("[字典表] 删除字典项失败 原因=%s 字典项ID=%s", exc, dictionary_item_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="字典项不存在")

    logger.info("[字典表] 删除字典项完成 字典项ID=%s", dictionary_item_id)
    return {"status": "deleted", "dictionary_item_id": dictionary_item_id}


@router.delete("/dictionaries/{dictionary_code}")
def delete_dictionary_group(dictionary_code: str) -> dict:
    """删除父级字典分组及其全部字典项。"""

    store = _get_knowledge_store()
    deleted_count = store.delete_dictionary_group(dictionary_code)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="父级字典不存在")

    logger.info("[字典表] 删除父级字典完成 字典编码=%s 删除数量=%s", dictionary_code, deleted_count)
    return {
        "status": "deleted",
        "dictionary_code": dictionary_code,
        "deleted_count": deleted_count,
    }
