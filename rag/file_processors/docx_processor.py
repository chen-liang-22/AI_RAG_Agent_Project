import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from langchain_core.documents import Document

from rag.file_processors.base import BaseFileProcessor


class DocxFileProcessor(BaseFileProcessor):
    """DOCX 文件处理器。

    训练案例第一批资料是 LMS 的 Word 文档。这里用标准库读取 docx，
    避免为了一个读取动作强制引入新的运行时依赖。
    """

    supported_file_types = ("docx",)

    def load_documents(self, file_path: str) -> list[Document]:
        """读取 DOCX 段落，并尽量保留标题样式信息。

        返回 LangChain Document 列表：
        - page_content：所有段落拼成的大文本；
        - metadata["paragraphs"]：保留每个段落，方便 LMS 策略按段落切案例。
        """

        paragraphs = self._read_paragraphs(file_path)
        if not paragraphs:
            return []

        content = "\n".join(item["text"] for item in paragraphs if item["text"].strip())
        return [
            Document(
                page_content=content,
                metadata={
                    "source": str(Path(file_path).name),
                    "paragraphs": paragraphs,
                },
            )
        ]

    @staticmethod
    def _read_paragraphs(file_path: str) -> list[dict[str, str]]:
        """从 word/document.xml 中读取段落文本和段落样式。

        docx 本质上是一个 zip 包，正文 XML 通常在 word/document.xml。
        这里用 Python 标准库 zipfile + ElementTree 解析，
        避免新增 python-docx 依赖。
        """

        with zipfile.ZipFile(file_path) as docx_file:
            document_xml = docx_file.read("word/document.xml")

        # Word XML 使用命名空间，ElementTree 查询节点时必须带 namespace。
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        root = ElementTree.fromstring(document_xml)
        paragraphs: list[dict[str, str]] = []

        for paragraph in root.findall(".//w:p", namespace):
            style_value = ""
            # w:pPr 是段落属性；w:pStyle 是段落样式，例如标题样式。
            paragraph_properties = paragraph.find("w:pPr", namespace)
            if paragraph_properties is not None:
                style = paragraph_properties.find("w:pStyle", namespace)
                if style is not None:
                    style_value = style.attrib.get(f"{{{namespace['w']}}}val", "")

            # 一个段落可能被 Word 拆成多个 w:t 文本节点，这里合并回完整段落。
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            # 把连续空白压成一个空格，减少 Word 内部格式带来的噪声。
            clean_text = re.sub(r"\s+", " ", "".join(texts)).strip()
            if clean_text:
                paragraphs.append({"style": style_value, "text": clean_text})

        return paragraphs
