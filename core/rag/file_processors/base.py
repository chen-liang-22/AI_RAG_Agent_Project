"""文件处理器抽象层。

文件处理器只负责“读文件 -> LangChain Document”。
后续的切片、Embedding、Qdrant 写入都在 DocumentParser 和 VectorStoreService 中完成。
"""

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseFileProcessor(ABC):
    """知识库文件处理器抽象基类。

    文件处理器只负责把原始文件读取成统一的 LangChain Document，
    不负责切片、embedding 或写入 Qdrant。
    """

    supported_file_types: tuple[str, ...] = ()

    def support_file_type(self, file_type: str) -> bool:
        """判断当前处理器是否支持指定文件类型。"""

        normalized_type = self.normalize_file_type(file_type)
        return normalized_type in self.supported_file_types

    @abstractmethod
    def load_documents(self, file_path: str) -> list[Document]:
        """读取原始文件，并转换为 LangChain Document 列表。"""

    def load_preview_text(self, file_path: str, max_chars: int = 5000) -> str:
        """读取预览文本，默认复用完整文件读取逻辑。"""

        documents = self.load_documents(file_path)
        return "\n\n".join(document.page_content for document in documents)[:max_chars]

    @staticmethod
    def normalize_file_type(file_type: str) -> str:
        """归一化文件类型，兼容带点后缀和大小写。"""

        return (file_type or "").strip().lower().lstrip(".")

    @classmethod
    def file_type_from_path(cls, file_path: str) -> str:
        """从文件路径中提取归一化后的文件类型。"""

        return cls.normalize_file_type(Path(file_path).suffix)
