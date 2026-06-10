import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ParsedKnowledgeUnit:
    """解析后的知识单元。

    这是写入 Qdrant 和 SQLite 前的中间结构。
    它比普通 chunk 多了 question/answer/title/category 等字段。
    """

    title: str
    content: str
    category: str
    unit_type: str
    question: str | None = None
    answer: str | None = None
    source_page: int | None = None


class KnowledgeUnitParser:
    """规则版知识单元解析器。

    它解决设计文档里“不要只做粗糙固定长度分片”的第一步：
    - FAQ 文档按“问题 + 答案”切。
    - 故障文档按“故障现象/检测/修复”切。
    - 维护/选购指南按编号条目切。
    - 如果无法识别结构，则回退到 RecursiveCharacterTextSplitter。

    这仍然是规则版结构化。
    后续如果要更强，可以在这里接 LLM 抽取或专门的文档解析器。
    """

    numbered_item_pattern = re.compile(r"^\s*(\d+)[.、]\s*(.+)$", re.MULTILINE)
    heading_pattern = re.compile(r"^\s*#{2,6}\s*(.+)$", re.MULTILINE)

    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        self.splitter = splitter

    def build_index_documents(
            self,
            *,
            document_id: str,
            filename: str,
            file_md5: str,
            version: int,
            documents: list[Document],
            unit_type: str,
            category: str,
    ) -> tuple[list[Document], list[dict]]:
        """把原始 Document 解析成 Qdrant Document 和 SQLite knowledge_units。"""

        parsed_units = self.parse_documents(
            documents=documents,
            filename=filename,
            unit_type=unit_type,
            category=category,
        )

        if not parsed_units:
            return self._fallback_split_documents(
                document_id=document_id,
                filename=filename,
                file_md5=file_md5,
                version=version,
                documents=documents,
                unit_type=unit_type,
                category=category,
            )

        index_documents: list[Document] = []
        units: list[dict] = []

        for index, unit in enumerate(parsed_units):
            unit_id = f"{document_id}_unit_{index:04d}"
            content = unit.content.strip()

            if not content:
                continue

            metadata = {
                "document_id": document_id,
                "unit_id": unit_id,
                "chunk_id": unit_id,
                "unit_type": unit.unit_type,
                "title": unit.title,
                "question": unit.question,
                "category": unit.category,
                "source_file": filename,
                "file_md5": file_md5,
                "version": version,
                "chunk_index": index,
                "source_page": unit.source_page,
            }

            index_documents.append(Document(page_content=content, metadata=metadata))
            units.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit.unit_type,
                    "title": unit.title,
                    "question": unit.question,
                    "answer": unit.answer,
                    "content": content,
                    "category": unit.category,
                    "tags": self._build_tags(unit),
                    "source_page": unit.source_page,
                    "unit_index": index,
                }
            )

        return index_documents, units

    def parse_documents(
            self,
            *,
            documents: list[Document],
            filename: str,
            unit_type: str,
            category: str,
    ) -> list[ParsedKnowledgeUnit]:
        """按页或按文档解析结构化知识单元。"""

        parsed_units: list[ParsedKnowledgeUnit] = []

        for document in documents:
            text = document.page_content.strip()
            if not text:
                continue

            source_page = document.metadata.get("page")
            parsed_units.extend(
                self._parse_text(
                    text=text,
                    filename=filename,
                    unit_type=unit_type,
                    category=category,
                    source_page=source_page,
                )
            )

        return parsed_units

    def _parse_text(
            self,
            *,
            text: str,
            filename: str,
            unit_type: str,
            category: str,
            source_page: int | None,
    ) -> list[ParsedKnowledgeUnit]:
        """解析单段文本。"""

        numbered_blocks = self._split_numbered_blocks(text, default_category=category)
        if not numbered_blocks:
            return []

        parsed_units: list[ParsedKnowledgeUnit] = []

        for block_category, block in numbered_blocks:
            unit = self._parse_block(
                block=block,
                filename=filename,
                unit_type=unit_type,
                category=block_category or category,
                source_page=source_page,
            )
            if unit:
                parsed_units.append(unit)

        return parsed_units

    def _split_numbered_blocks(self, text: str, default_category: str) -> list[tuple[str, str]]:
        """按章节标题和编号条目切分文本。

        支持这样的格式：

            ### 拖扫功能融合类
            1. **问题？**
            - 答案
            2. 下一个问题

        也支持：

            1. 故障现象：...；检测：...；修复：...
        """

        blocks: list[tuple[str, str]] = []
        current_category = default_category
        current_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            heading_match = self.heading_pattern.match(stripped)
            numbered_match = self.numbered_item_pattern.match(stripped)

            if heading_match:
                if current_lines:
                    blocks.append((current_category, "\n".join(current_lines).strip()))
                    current_lines = []
                current_category = self._clean_markdown(heading_match.group(1)) or default_category
                continue

            if numbered_match:
                if current_lines:
                    blocks.append((current_category, "\n".join(current_lines).strip()))
                current_lines = [stripped]
                continue

            if current_lines:
                current_lines.append(line)

        if current_lines:
            blocks.append((current_category, "\n".join(current_lines).strip()))

        return [(block_category, block) for block_category, block in blocks if block]

    def _parse_block(
            self,
            *,
            block: str,
            filename: str,
            unit_type: str,
            category: str,
            source_page: int | None,
    ) -> ParsedKnowledgeUnit | None:
        """把一个编号块解析成知识单元。"""

        block_without_number = self.numbered_item_pattern.sub(r"\2", block, count=1).strip()
        block_without_number = self._clean_markdown(block_without_number)

        if not block_without_number:
            return None

        if unit_type == "faq":
            return self._parse_faq_block(block_without_number, category, source_page)

        if unit_type == "troubleshooting" or "故障现象：" in block_without_number:
            return self._parse_troubleshooting_block(block_without_number, category, source_page)

        title = self._extract_title(block_without_number, fallback=filename)
        return ParsedKnowledgeUnit(
            title=title,
            content=block_without_number,
            category=category,
            unit_type=unit_type,
            source_page=source_page,
        )

    def _parse_faq_block(
            self,
            block: str,
            category: str,
            source_page: int | None,
    ) -> ParsedKnowledgeUnit:
        """解析 FAQ 问答块。"""

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        question = self._clean_markdown(lines[0]) if lines else block[:60]
        answer_lines = []

        for line in lines[1:]:
            answer_lines.append(line.removeprefix("-").strip())

        answer = "\n".join(answer_lines).strip() or None
        content = f"问题：{question}"
        if answer:
            content += f"\n答案：{answer}"

        return ParsedKnowledgeUnit(
            title=question,
            question=question,
            answer=answer,
            content=content,
            category=category,
            unit_type="faq",
            source_page=source_page,
        )

    def _parse_troubleshooting_block(
            self,
            block: str,
            category: str,
            source_page: int | None,
    ) -> ParsedKnowledgeUnit:
        """解析故障排查块。"""

        problem = self._extract_between(block, "故障现象：", ["；", "\n"]) or self._extract_title(block, "故障排查")
        detection = self._extract_between(block, "检测：", ["；", "\n"])
        solution = self._extract_between(block, "修复：", ["；", "\n"])

        content_parts = [f"故障现象：{problem}"]
        if detection:
            content_parts.append(f"检测：{detection}")
        if solution:
            content_parts.append(f"修复：{solution}")

        return ParsedKnowledgeUnit(
            title=problem,
            question=f"{problem}怎么办？",
            answer=solution,
            content="\n".join(content_parts),
            category=category,
            unit_type="troubleshooting",
            source_page=source_page,
        )

    def _fallback_split_documents(
            self,
            *,
            document_id: str,
            filename: str,
            file_md5: str,
            version: int,
            documents: list[Document],
            unit_type: str,
            category: str,
    ) -> tuple[list[Document], list[dict]]:
        """无法结构化解析时，回退到旧的固定长度分片。"""

        split_documents = self.splitter.split_documents(documents)
        index_documents: list[Document] = []
        units: list[dict] = []

        for index, doc in enumerate(split_documents):
            content = doc.page_content.strip()
            if not content:
                continue

            unit_id = f"{document_id}_unit_{index:04d}"
            source_page = doc.metadata.get("page")
            title = self._extract_title(content, fallback=filename)
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "document_id": document_id,
                    "unit_id": unit_id,
                    "chunk_id": unit_id,
                    "unit_type": unit_type,
                    "title": title,
                    "question": None,
                    "category": category,
                    "source_file": filename,
                    "file_md5": file_md5,
                    "version": version,
                    "chunk_index": index,
                    "source_page": source_page,
                }
            )

            index_documents.append(Document(page_content=content, metadata=metadata))
            units.append(
                {
                    "unit_id": unit_id,
                    "unit_type": unit_type,
                    "title": title,
                    "question": None,
                    "answer": None,
                    "content": content,
                    "category": category,
                    "tags": [category, unit_type],
                    "source_page": source_page,
                    "unit_index": index,
                }
            )

        return index_documents, units

    @staticmethod
    def _extract_between(text: str, start: str, end_marks: list[str]) -> str | None:
        """提取 start 后到任意结束符之前的文本。"""

        if start not in text:
            return None

        value = text.split(start, 1)[1]
        end_positions = [value.find(mark) for mark in end_marks if value.find(mark) >= 0]
        if end_positions:
            value = value[:min(end_positions)]

        return value.strip() or None

    @staticmethod
    def _extract_title(text: str, fallback: str) -> str:
        """从内容里提取短标题。"""

        first_line = text.splitlines()[0].strip() if text.splitlines() else text
        first_line = KnowledgeUnitParser._clean_markdown(first_line)
        for mark in ["；", "。", ".", "，", ","]:
            if mark in first_line:
                first_line = first_line.split(mark, 1)[0]
                break

        return first_line[:80] or fallback

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """去掉常见 Markdown 标记。"""

        text = text.strip()
        text = text.replace("**", "")
        text = text.replace("__", "")
        return text.strip()

    @staticmethod
    def _build_tags(unit: ParsedKnowledgeUnit) -> list[str]:
        """生成知识单元标签。"""

        tags = [unit.category, unit.unit_type]
        if unit.question:
            tags.append("question")
        if unit.answer:
            tags.append("answer")

        result: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not tag or tag in seen:
                continue
            seen.add(tag)
            result.append(tag)

        return result
