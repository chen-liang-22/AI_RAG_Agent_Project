"""TXT 文件处理器。

TXT 没有结构化目录，这里只负责读取纯文本。
后续是否按递归、编号、LLM 语义切片，由 DocumentParser 决定。
"""

from langchain_core.documents import Document

from core.rag.file_processors.base import BaseFileProcessor
from core.utils.file_handler import txt_loader


class TxtFileProcessor(BaseFileProcessor):
    """TXT 知识库文件处理器。"""

    supported_file_types = ("txt",)

    def load_documents(self, file_path: str) -> list[Document]:
        """读取 TXT 文件，并转换为 LangChain Document 列表。"""

        return txt_loader(file_path)
