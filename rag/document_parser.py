import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentTypeDetection:
    """文档类型识别结果。

    这个结果会返回给前端做“上传确认”。
    第一版用规则识别，后续可以在 confidence 较低时接 LLM 兜底。
    """

    document_type: str
    split_strategy: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class DocumentSegment:
    """通用文档片段。

    所有文档都会先拆成 segment。
    Qdrant 向量也主要对应这些 segment。
    """

    segment_id: str
    segment_index: int
    content: str
    content_hash: str
    page_no: int | None = None
    heading_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaqItem:
    """FAQ/100问 文档里抽取出的结构化问答。"""

    faq_id: str
    segment_id: str
    question_no: int | None
    question: str
    answer: str | None
    category: str | None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """通用文档解析器。

    - 规则识别文档类型。
    - 所有文档都会生成 document_segments。
    - FAQ 文档额外生成 faq_items。
    """

    numbered_question_pattern = re.compile(r"^\s*(\d+)[.、]\s*(?:\*\*)?.*?[？?].*$", re.MULTILINE)
    numbered_item_pattern = re.compile(r"^\s*(\d+)[.、]\s*(.+)$", re.MULTILINE)
    heading_pattern = re.compile(r"^\s*#{2,6}\s*(.+)$", re.MULTILINE)
    qa_mark_pattern = re.compile(r"(问[:：]|答[:：]|Q[:：]|A[:：])", re.IGNORECASE)

    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        self.splitter = splitter

    def detect_document_type(self, filename: str, sample_text: str) -> DocumentTypeDetection:
        """用规则识别文档类型。

        LLM 兜底接口先预留字段，不在这一版默认调用，避免上传时引入额外模型成本。
        """

        lower_filename = filename.lower()
        text = sample_text or ""
        score = 0.0
        reasons: list[str] = []

        if any(word in lower_filename for word in ["faq", "q&a", "问答", "常见问题", "100问"]):
            score += 0.4
            reasons.append("文件名包含 FAQ/问答/100问 等特征")

        numbered_questions = self.numbered_question_pattern.findall(text)
        if len(numbered_questions) >= 5:
            score += 0.35
            reasons.append(f"检测到 {len(numbered_questions)} 个编号问题")

        qa_marks = self.qa_mark_pattern.findall(text)
        if len(qa_marks) >= 4:
            score += 0.2
            reasons.append("检测到问答标记")

        if "故障现象" in text and ("检测" in text or "修复" in text):
            if score < 0.7:
                return DocumentTypeDetection(
                    document_type="troubleshooting",
                    split_strategy="numbered_segments",
                    confidence=0.82,
                    reasons=["检测到故障现象/检测/修复结构"],
                )

        if any(word in lower_filename for word in ["维护", "保养"]):
            return DocumentTypeDetection(
                document_type="maintenance",
                split_strategy="numbered_segments",
                confidence=0.78,
                reasons=["文件名包含维护/保养"],
            )

        if any(word in lower_filename for word in ["选购", "指南"]):
            return DocumentTypeDetection(
                document_type="guide",
                split_strategy="numbered_segments",
                confidence=0.78,
                reasons=["文件名包含选购/指南"],
            )

        if score >= 0.7:
            return DocumentTypeDetection(
                document_type="faq",
                split_strategy="numbered_qa",
                confidence=round(min(score, 0.98), 2),
                reasons=reasons,
            )

        return DocumentTypeDetection(
            document_type="general",
            split_strategy="recursive",
            confidence=round(score, 2),
            reasons=reasons or ["未检测到稳定结构，按普通文档处理"],
        )

    def build_segments_and_faqs(
            self,
            *,
            document_id: str,
            documents: list[Document],
            document_type: str,
            split_strategy: str,
    ) -> tuple[list[DocumentSegment], list[FaqItem]]:
        """按确认后的类型和切分策略生成 segment/faq。"""

        if document_type == "faq" and split_strategy == "numbered_qa":
            return self._build_faq_segments(document_id, documents)

        if split_strategy == "numbered_segments":
            segments = self._build_numbered_segments(document_id, documents, document_type=document_type)
            if segments:
                return segments, []

        return self._build_recursive_segments(document_id, documents, document_type=document_type), []

    def _build_faq_segments(
            self,
            document_id: str,
            documents: list[Document],
    ) -> tuple[list[DocumentSegment], list[FaqItem]]:
        segments: list[DocumentSegment] = []
        faq_items: list[FaqItem] = []

        for document in documents:
            blocks = self._split_numbered_blocks(document.page_content)
            for category, question_no, block in blocks:
                segment_index = len(segments)
                segment_id = f"{document_id}_seg_{segment_index:04d}"
                question, answer = self._parse_question_answer(block)
                content = self._format_faq_content(question, answer)
                page_no = document.metadata.get("page")

                segment = DocumentSegment(
                    segment_id=segment_id,
                    segment_index=segment_index,
                    content=content,
                    content_hash=self._hash_text(content),
                    page_no=page_no,
                    heading_path=category,
                    metadata={
                        "document_type": "faq",
                        "split_strategy": "numbered_qa",
                        "question_no": question_no,
                    },
                )
                segments.append(segment)

                faq_items.append(
                    FaqItem(
                        faq_id=f"{document_id}_faq_{segment_index:04d}",
                        segment_id=segment_id,
                        question_no=question_no,
                        question=question,
                        answer=answer,
                        category=category,
                        tags=["faq", "question", "answer"],
                        metadata={"source_page": page_no},
                    )
                )

        if segments:
            return segments, faq_items

        return self._build_recursive_segments(document_id, documents, document_type="faq"), []

    def _build_numbered_segments(
            self,
            document_id: str,
            documents: list[Document],
            *,
            document_type: str,
    ) -> list[DocumentSegment]:
        segments: list[DocumentSegment] = []

        for document in documents:
            blocks = self._split_numbered_blocks(document.page_content)
            for category, item_no, block in blocks:
                content = self._clean_markdown(block)
                if not content:
                    continue
                segment_index = len(segments)
                segments.append(
                    DocumentSegment(
                        segment_id=f"{document_id}_seg_{segment_index:04d}",
                        segment_index=segment_index,
                        content=content,
                        content_hash=self._hash_text(content),
                        page_no=document.metadata.get("page"),
                        heading_path=category,
                        metadata={
                            "document_type": document_type,
                            "split_strategy": "numbered_segments",
                            "item_no": item_no,
                        },
                    )
                )

        return segments

    def _build_recursive_segments(
            self,
            document_id: str,
            documents: list[Document],
            *,
            document_type: str,
    ) -> list[DocumentSegment]:
        split_documents = self.splitter.split_documents(documents)
        segments: list[DocumentSegment] = []

        for index, document in enumerate(split_documents):
            content = document.page_content.strip()
            if not content:
                continue

            segments.append(
                DocumentSegment(
                    segment_id=f"{document_id}_seg_{len(segments):04d}",
                    segment_index=len(segments),
                    content=content,
                    content_hash=self._hash_text(content),
                    page_no=document.metadata.get("page"),
                    heading_path=None,
                    metadata={
                        "document_type": document_type,
                        "split_strategy": "recursive",
                        "source": document.metadata.get("source"),
                        "chunk_index": index,
                    },
                )
            )

        return segments

    def _split_numbered_blocks(self, text: str) -> list[tuple[str | None, int | None, str]]:
        blocks: list[tuple[str | None, int | None, str]] = []
        current_category: str | None = None
        current_number: int | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            heading_match = self.heading_pattern.match(stripped)
            numbered_match = self.numbered_item_pattern.match(stripped)

            if heading_match:
                if current_lines:
                    blocks.append((current_category, current_number, "\n".join(current_lines).strip()))
                    current_lines = []
                    current_number = None
                current_category = self._clean_markdown(heading_match.group(1))
                continue

            if numbered_match:
                if current_lines:
                    blocks.append((current_category, current_number, "\n".join(current_lines).strip()))
                current_number = int(numbered_match.group(1))
                current_lines = [stripped]
                continue

            if current_lines:
                current_lines.append(line)

        if current_lines:
            blocks.append((current_category, current_number, "\n".join(current_lines).strip()))

        return [(category, number, block) for category, number, block in blocks if block]

    def _parse_question_answer(self, block: str) -> tuple[str, str | None]:
        block_without_number = self.numbered_item_pattern.sub(r"\2", block, count=1).strip()
        block_without_number = self._clean_markdown(block_without_number)
        lines = [line.strip() for line in block_without_number.splitlines() if line.strip()]

        if not lines:
            return block_without_number[:80], None

        question = self._clean_markdown(lines[0])
        answer_lines = []
        for line in lines[1:]:
            cleaned = line.removeprefix("-").strip()
            cleaned = re.sub(r"^(答[:：]|A[:：])\s*", "", cleaned, flags=re.IGNORECASE)
            if cleaned:
                answer_lines.append(cleaned)

        answer = "\n".join(answer_lines).strip() or None
        return question, answer

    @staticmethod
    def _format_faq_content(question: str, answer: str | None) -> str:
        if answer:
            return f"问题：{question}\n答案：{answer}"
        return f"问题：{question}"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_markdown(text: str) -> str:
        text = text.strip()
        text = text.replace("**", "")
        text = text.replace("__", "")
        return text.strip()

    @staticmethod
    def dumps_metadata(value: dict[str, Any] | list[str] | None) -> str:
        return json.dumps(value or {}, ensure_ascii=False)
