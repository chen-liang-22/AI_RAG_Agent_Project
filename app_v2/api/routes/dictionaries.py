"""V2 字典接口。"""

from fastapi import APIRouter, Query

from api.schemas import DictionaryGroupResponse, DictionaryItemResponse, DictionaryItemUpsertRequest
from app_v2.application.dictionary_service import DictionaryApplicationService

router = APIRouter(prefix="/dictionaries", tags=["V2 字典"])


@router.get("", response_model=list[DictionaryGroupResponse])
def list_dictionaries(dictionary_code: str | None = Query(default=None, description="可选，按字典编码过滤")) -> list[DictionaryGroupResponse]:
    """查询字典分组。"""

    return DictionaryApplicationService().list_groups(dictionary_code=dictionary_code)


@router.post("/items", response_model=DictionaryItemResponse)
def upsert_dictionary_item(request: DictionaryItemUpsertRequest) -> DictionaryItemResponse:
    """新增或修改字典项。"""

    return DictionaryApplicationService().upsert_item(request)


@router.delete("/items/{dictionary_item_id}")
def delete_dictionary_item(dictionary_item_id: str) -> dict[str, str]:
    """删除字典项。"""

    return DictionaryApplicationService().delete_item(dictionary_item_id)
