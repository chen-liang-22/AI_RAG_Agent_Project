"""V2 文档响应转换工具。"""

from dataclasses import dataclass
from datetime import date, datetime

from fastapi import HTTPException

from api.schemas import KnowledgeFileResponse
from core.utils.qdrant_options import get_qdrant_collection_name


@dataclass(frozen=True)
class DictionaryCodeSnapshot:
    """一次性加载字典编码，避免列表接口为每条记录反复访问 MySQL。"""

    enabled_codes_by_dictionary: dict[str, set[str]]
    default_code_by_dictionary: dict[str, str]

    def normalize(self, dictionary_code: str, value: str | None = None) -> str:
        """使用内存快照归一化字典编码。"""

        enabled_codes = self.enabled_codes_by_dictionary.get(dictionary_code) or set()
        default_code = self.default_code_by_dictionary.get(dictionary_code)
        if not default_code:
            raise ValueError(f"字典没有可用项：{dictionary_code}")
        normalized_value = str(value or default_code).strip().lower()
        if normalized_value in enabled_codes:
            return normalized_value
        return default_code

    def enabled_codes(self, dictionary_code: str) -> set[str]:
        """返回某个字典的启用编码集合。"""

        return set(self.enabled_codes_by_dictionary.get(dictionary_code) or set())


def normalize_split_strategy(
        split_strategy: str | None = None,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> str:
    """从 V2 字典快照归一化切分策略。"""

    if dictionary_snapshot is None:
        return str(split_strategy or "recursive").strip().lower()
    return dictionary_snapshot.normalize("split_strategy", split_strategy)


def normalize_document_structure_type(
        document_type: str | None = None,
        split_strategy: str | None = None,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> str:
    """从 V2 字典快照归一化文档结构类型。"""

    if dictionary_snapshot is None:
        normalized_split_strategy = str(split_strategy or "").strip().lower()
        if normalized_split_strategy == "llm_semantic":
            return "text"
        return str(document_type or "text").strip().lower()
    enabled_codes = dictionary_snapshot.enabled_codes("document_structure")
    default_code = dictionary_snapshot.normalize("document_structure", None)
    value = str(document_type or "").strip().lower()
    normalized_split_strategy = normalize_split_strategy(split_strategy, dictionary_snapshot)
    if normalized_split_strategy == "llm_semantic" and "text" in enabled_codes:
        return "text"
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


def format_response_time(value: object) -> str:
    """把数据库时间字段统一转换成响应字符串。"""

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def document_to_response(
        document: dict,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> KnowledgeFileResponse:
    """把数据库文档记录转换成 FastAPI 响应模型。"""

    return KnowledgeFileResponse(
        document_id=document["document_id"],
        filename=document["filename"],
        file_path=document["file_path"],
        storage_type=document.get("storage_type") or "minio",
        bucket_name=document.get("bucket_name"),
        object_name=document.get("object_name"),
        public_url=document.get("public_url"),
        file_type=document["file_type"],
        file_md5=document["file_md5"],
        file_size=int(document["file_size"]),
        status=document["status"],
        version=int(document["version"]),
        chunk_count=int(document["chunk_count"]),
        collection_name=document.get("collection_name") or get_qdrant_collection_name(),
        document_type=normalize_document_structure_type(
            document.get("document_type"),
            document.get("split_strategy"),
            dictionary_snapshot,
        ),
        split_strategy=normalize_split_strategy(document.get("split_strategy"), dictionary_snapshot),
        created_at=format_response_time(document["created_at"]),
        updated_at=format_response_time(document["updated_at"]),
        error_message=document.get("error_message"),
    )
