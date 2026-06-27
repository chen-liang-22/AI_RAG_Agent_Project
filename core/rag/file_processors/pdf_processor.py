"""PDF 文件处理器。

本模块负责读取 PDF 正文，并尽量读取 PDF 书签目录。
书签目录会挂到首个 Document 的 metadata["_pdf_outline"]，
后续 DocumentParser 可以用它做“目录问答”切片。
"""

from typing import Any

from langchain_core.documents import Document

from core.rag.file_processors.base import BaseFileProcessor
from core.utils.file_handler import pdf_loader
from core.utils.logger_handler import logger


class PdfFileProcessor(BaseFileProcessor):
    """PDF 知识库文件处理器。"""

    supported_file_types = ("pdf",)

    def load_documents(self, file_path: str) -> list[Document]:
        """读取 PDF 文件，并把 PDF 书签目录挂到首个 Document metadata。"""

        documents = pdf_loader(file_path)
        outline = self.read_pdf_outline(file_path)
        outline_by_page = self._outline_by_page(outline)
        for index, document in enumerate(documents, start=1):
            page_no = int(document.metadata.get("page") or index - 1) + 1
            document.metadata["page_no"] = page_no
            document.metadata["structured_blocks"] = [
                {
                    "block_index": index,
                    "block_type": "page",
                    "page_no": page_no,
                    "outline_title": outline_by_page.get(page_no),
                    "text": document.page_content,
                }
            ]
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

    @staticmethod
    def _outline_by_page(outline: list[dict[str, Any]]) -> dict[int, str]:
        """把 PDF 书签目录转换成页码到标题的映射。"""

        result: dict[int, str] = {}
        for item in outline:
            page = item.get("page")
            title = str(item.get("title") or "").strip()
            if isinstance(page, int) and title and page not in result:
                result[page] = title
        return result
