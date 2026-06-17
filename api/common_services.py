from fastapi import HTTPException

from api.schemas import KnowledgeFileResponse
from rag.knowledge_store import KnowledgeStore
from utils.qdrant_options import get_qdrant_collection_name


def _get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据仓库。

    KnowledgeStore 内部使用 SQLite。
    每次请求创建一个轻量对象即可，真正的数据库连接只在执行 SQL 时短暂打开。
    """

    return KnowledgeStore()


def _normalize_split_strategy(split_strategy: str | None = None) -> str:
    """从字典表归一化切分策略。"""

    return _get_knowledge_store().normalize_dictionary_code("split_strategy", split_strategy)


def _normalize_document_structure_type(
        document_type: str | None = None,
        split_strategy: str | None = None,
) -> str:
    """从字典表归一化文档结构类型。"""

    store = _get_knowledge_store()
    enabled_codes = set(store.list_enabled_dictionary_codes("document_structure"))
    default_code = store.normalize_dictionary_code("document_structure", None)
    value = str(document_type or "").strip().lower()
    normalized_split_strategy = _normalize_split_strategy(split_strategy)
    if normalized_split_strategy in {"numbered_qa", "outline_qa"} and "qa" in enabled_codes:
        return "qa"
    if normalized_split_strategy == "numbered_segments" and "numbered" in enabled_codes:
        return "numbered"
    if value in enabled_codes:
        return value
    if not value:
        return default_code
    supported_text = "、".join(sorted(enabled_codes))
    raise HTTPException(status_code=400, detail=f"文档结构类型只支持：{supported_text}")


def _document_to_response(document: dict) -> KnowledgeFileResponse:
    """把 SQLite 文档记录转换成 FastAPI 响应模型。

    SQLite 取出来的数字字段有时可能是字符串或兼容类型，这里统一转成 int，
    这样前端拿到的数据类型更稳定。
    """

    return KnowledgeFileResponse(
        document_id=document["document_id"],
        filename=document["filename"],
        file_path=document["file_path"],
        file_type=document["file_type"],
        file_md5=document["file_md5"],
        file_size=int(document["file_size"]),
        status=document["status"],
        version=int(document["version"]),
        chunk_count=int(document["chunk_count"]),
        collection_name=document.get("collection_name") or get_qdrant_collection_name(),
        document_type=_normalize_document_structure_type(
            document.get("document_type"),
            document.get("split_strategy"),
        ),
        split_strategy=_normalize_split_strategy(document.get("split_strategy")),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        error_message=document.get("error_message"),
    )
