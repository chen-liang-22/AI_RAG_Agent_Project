from datetime import date, datetime
from dataclasses import dataclass

from fastapi import HTTPException

from api.schemas import KnowledgeFileResponse
from rag.knowledge_store import KnowledgeStore
from utils.qdrant_options import get_qdrant_collection_name


def _get_knowledge_store() -> KnowledgeStore:
    """创建知识库元数据仓库。

    每次请求创建一个轻量对象即可，真正的数据库连接只在执行 SQL 时短暂打开。
    """

    return KnowledgeStore()


@dataclass(frozen=True)
class DictionaryCodeSnapshot:
    """一次性加载字典编码，避免列表接口为每条记录反复访问 MySQL。"""

    enabled_codes_by_dictionary: dict[str, set[str]]
    default_code_by_dictionary: dict[str, str]

    @classmethod
    def load(cls, store: KnowledgeStore, dictionary_codes: set[str]) -> "DictionaryCodeSnapshot":
        """按字典编码批量读取启用项和默认项。"""

        enabled_codes_by_dictionary: dict[str, set[str]] = {}
        default_code_by_dictionary: dict[str, str] = {}
        for dictionary_code in dictionary_codes:
            rows = store.list_dictionary_items(dictionary_code=dictionary_code)
            enabled_rows = [row for row in rows if int(row.get("enabled") or 0) == 1]
            enabled_codes = {str(row["item_code"]) for row in enabled_rows}
            enabled_codes_by_dictionary[dictionary_code] = enabled_codes
            if enabled_rows:
                default_code_by_dictionary[dictionary_code] = str(enabled_rows[0]["item_code"])
        return cls(
            enabled_codes_by_dictionary=enabled_codes_by_dictionary,
            default_code_by_dictionary=default_code_by_dictionary,
        )

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


def _normalize_split_strategy(
        split_strategy: str | None = None,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> str:
    """从字典表归一化切分策略。"""

    if dictionary_snapshot is not None:
        return dictionary_snapshot.normalize("split_strategy", split_strategy)
    return _get_knowledge_store().normalize_dictionary_code("split_strategy", split_strategy)


def _normalize_document_structure_type(
        document_type: str | None = None,
        split_strategy: str | None = None,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> str:
    """从字典表归一化文档结构类型。"""

    if dictionary_snapshot is None:
        store = _get_knowledge_store()
        dictionary_snapshot = DictionaryCodeSnapshot.load(store, {"document_structure", "split_strategy"})
    enabled_codes = dictionary_snapshot.enabled_codes("document_structure")
    default_code = dictionary_snapshot.normalize("document_structure", None)
    value = str(document_type or "").strip().lower()
    normalized_split_strategy = _normalize_split_strategy(split_strategy, dictionary_snapshot)
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


def _format_response_time(value: object) -> str:
    """把数据库时间字段统一转换成响应字符串。"""

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def build_document_dictionary_snapshot(store: KnowledgeStore | None = None) -> DictionaryCodeSnapshot:
    """构建文档列表响应所需的字典快照。"""

    return DictionaryCodeSnapshot.load(
        store or _get_knowledge_store(),
        {"document_structure", "split_strategy"},
    )


def _document_to_response(
        document: dict,
        dictionary_snapshot: DictionaryCodeSnapshot | None = None,
) -> KnowledgeFileResponse:
    """把数据库文档记录转换成 FastAPI 响应模型。

    数据库取出来的数字字段有时可能是字符串或兼容类型，这里统一转成 int，
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
            dictionary_snapshot,
        ),
        split_strategy=_normalize_split_strategy(document.get("split_strategy"), dictionary_snapshot),
        created_at=_format_response_time(document["created_at"]),
        updated_at=_format_response_time(document["updated_at"]),
        error_message=document.get("error_message"),
    )
