"""V2 数据启动初始化入口。"""

from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app_v2.infrastructure.repositories.document_repository import DocumentRepository
from app_v2.infrastructure.orm_session import orm_session_context


_BOOTSTRAPPED = False


def bootstrap_v2_metadata() -> None:
    """初始化 V2 元数据基础项。"""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with orm_session_context() as session:
        DocumentRepository.ensure_storage_columns(session)
    DictionaryRepository().seed_default_items()
    _BOOTSTRAPPED = True
