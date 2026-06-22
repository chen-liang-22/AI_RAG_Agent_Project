import re
import zipfile
from pathlib import Path
from typing import Any
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

        # 读取 DOCX 原始段落列表，每个段落会保留 text 正文和 style 样式，方便后续按标题/段落切片。
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
                    "structured_blocks": paragraphs,
                },
            )
        ]

    @staticmethod
    def _read_paragraphs(file_path: str) -> list[dict[str, Any]]:
        """从 word/document.xml 中读取段落文本和段落样式。

        docx 本质上是一个 zip 包，正文 XML 通常在 word/document.xml。
        这里用 Python 标准库 zipfile + ElementTree 解析，
        避免新增 python-docx 依赖。
        """

        # 打开 docx 文件；docx 本质是 zip 压缩包，with 会在读取后自动关闭文件。
        with zipfile.ZipFile(file_path) as docx_file:
            # 从压缩包中读取 Word 正文 XML，后续所有段落都从这个 XML 里解析。
            document_xml = docx_file.read("word/document.xml")

        # Word XML 使用命名空间，ElementTree 查询节点时必须带 namespace。
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        # 把 XML 字节内容解析成 ElementTree 根节点，方便按节点路径遍历。
        root = ElementTree.fromstring(document_xml)
        # 准备保存段落结果；每个元素包含段落样式 style 和段落文本 text。
        paragraphs: list[dict[str, Any]] = []

        # 遍历所有 w:p 段落节点；.// 表示不管层级深度，找到全部段落。
        for paragraph_index, paragraph in enumerate(root.findall(".//w:p", namespace), start=1):
            # 先给段落样式一个默认空值；普通正文段落通常没有标题样式。
            style_value = ""
            # w:pPr 是段落属性；w:pStyle 是段落样式，例如标题样式。
            paragraph_properties = paragraph.find("w:pPr", namespace)
            # 只有存在段落属性时，才继续读取其中的段落样式。
            if paragraph_properties is not None:
                # 从段落属性里读取 w:pStyle 节点，例如 Heading1、Heading2。
                style = paragraph_properties.find("w:pStyle", namespace)
                # 只有样式节点存在时，才读取具体样式值。
                if style is not None:
                    # Word XML 属性也带命名空间，这里读取 w:val 对应的样式编码。
                    style_value = style.attrib.get(f"{{{namespace['w']}}}val", "")

            # 一个段落可能被 Word 拆成多个 w:t 文本节点，这里合并回完整段落。
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            # 清理多余空白，减少 Word 排版产生的噪声。
            clean_text = re.sub(r"\s+", " ", "".join(texts)).strip()
            # 空段落没有检索价值，只有清理后仍有文本时才保存。
            if clean_text:
                # heading_level 用于第三阶段结构化 block 抽取，普通正文为 None。
                heading_level = DocxFileProcessor._heading_level_from_style(style_value)
                # 保存当前段落的样式和文本，后续切分器可根据样式识别标题结构。
                paragraphs.append(
                    {
                        "block_index": paragraph_index,
                        "block_type": "heading" if heading_level is not None else "paragraph",
                        "heading_level": heading_level,
                        "style": style_value,
                        "text": clean_text,
                    }
                )

        # 返回按 Word 文档顺序解析出的段落列表。
        return paragraphs

    @staticmethod
    def _heading_level_from_style(style_value: str) -> int | None:
        """从 Word 样式名中识别标题级别。"""

        clean_style = str(style_value or "").strip()
        match = re.search(r"(?:Heading|Title|标题)\s*([1-6])", clean_style, flags=re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))
