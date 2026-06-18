from typing import Any

from langchain_core.documents import Document

from rag.file_processors.base import BaseFileProcessor
from utils.file_handler import pdf_loader
from utils.logger_handler import logger


class PdfFileProcessor(BaseFileProcessor):
    """PDF 知识库文件处理器。"""

    supported_file_types = ("pdf",)

    def load_documents(self, file_path: str) -> list[Document]:
        """读取 PDF 文件，并把 PDF 书签目录挂到首个 Document metadata。"""

        documents = pdf_loader(file_path)
        outline = self.read_pdf_outline(file_path)
        if documents and outline:
            documents[0].metadata["_pdf_outline"] = outline
        return documents

    @staticmethod
    def read_pdf_outline(file_path: str) -> list[dict[str, Any]]:
        """读取 PDF 书签目录，返回 level/title/page 结构。"""

        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            outline_items: list[dict[str, Any]] = []

            def walk(items, level: int = 0) -> None:
                """递归遍历 PDF 书签树，并记录层级和页码。"""

                for item in items:
                    if isinstance(item, list):
                        walk(item, level + 1)
                        continue

                    title = str(getattr(item, "title", item)).strip()
                    if not title:
                        continue
                    try:
                        page_no = reader.get_destination_page_number(item) + 1
                    except (KeyError, ValueError, TypeError, AttributeError):
                        page_no = None

                    outline_items.append(
                        {
                            "level": level,
                            "title": title,
                            "page": page_no,
                        }
                    )

            walk(reader.outline)
            return outline_items
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("[知识库] PDF书签读取失败 文件=%s 错误=%s", file_path, exc)
            return []
