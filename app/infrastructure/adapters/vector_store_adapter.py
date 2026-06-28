"""向量库适配器。"""

from typing import Any

from app.infrastructure.vector_store_service import VectorStoreService


class VectorStoreAdapter:
    """Qdrant 向量库适配器。

    这里使用适配器模式，让应用服务不直接依赖 Qdrant/LangChain 的具体类。
    """

    def __init__(self, collection_name: str | None = None, vector_service: VectorStoreService | None = None):
        """初始化向量库适配器。

        collection_name 指定要访问的 Qdrant collection；vector_service 可注入用于测试。
        """

        self.vector_service = vector_service or VectorStoreService(collection_name=collection_name)

    @property
    def collection_name(self) -> str:
        """当前适配器操作的 collection 名称。"""

        return self.vector_service.collection_name

    def preview_file(self, *, filename: str, file_path: str, sample_limit: int) -> dict[str, Any]:
        """预览文件结构和样本文本。"""

        return self.vector_service.preview_file(filename=filename, file_path=file_path, sample_limit=sample_limit)

    def index_file(self, document: dict, *, document_type: str | None = None, split_strategy: str | None = None) -> int:
        """把文件写入向量库。"""

        return self.vector_service.index_file(document, document_type=document_type, split_strategy=split_strategy)

    def delete_document_vectors(self, document_id: str, collection_name: str | None = None) -> None:
        """按 document_id 删除向量。"""

        self.vector_service.delete_document_vectors(document_id, collection_name=collection_name)

    @classmethod
    def recreate_collection(cls, collection_name: str | None = None) -> "VectorStoreAdapter":
        """重建 collection 并返回适配器。"""

        return cls(vector_service=VectorStoreService.recreate_collection_service(collection_name))
