import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentTypeDetection:
    """文档结构识别结果。"""

    document_type: str
    split_strategy: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class DocumentSegment:
    """准备写入向量库的通用文档片段。"""

    segment_id: str
    segment_index: int
    content: str
    content_hash: str
    page_no: int | None = None
    heading_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QaItem:
    """从问答型文档里抽取出的结构化问答。"""

    qa_id: str
    segment_id: str
    question_no: int | None
    question: str
    answer: str | None
    category: str | None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """通用文档解析器，只识别文档结构，不绑定具体业务分类。"""

    numbered_question_pattern = re.compile(r"^\s*(\d+)[.、]\s*(?:\*\*)?.*?[？?].*$", re.MULTILINE)
    numbered_item_pattern = re.compile(r"^\s*(\d+)[.、]\s*(.+)$", re.MULTILINE)
    heading_pattern = re.compile(r"^\s*#{2,6}\s*(.+)$", re.MULTILINE)
    qa_mark_pattern = re.compile(r"(问[:：]|答[:：]|Q[:：]|A[:：])", re.IGNORECASE)
    outline_question_keywords = (
        "什么",
        "为什么",
        "怎么",
        "如何",
        "是否",
        "能不能",
        "有哪些",
        "区别",
        "比较",
        "说说",
        "介绍",
        "原理",
        "步骤",
        "作用",
    )

    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        """初始化文档解析器。"""

        self.splitter = splitter

    def detect_document_type(
            self,
            filename: str,
            sample_text: str,
            outline: list[dict[str, Any]] | None = None,
    ) -> DocumentTypeDetection:
        """用轻量规则给出默认结构建议，模型推荐由上传推荐接口单独完成。"""

        lower_filename = filename.lower()
        text = sample_text or ""
        reasons: list[str] = []

        if self.is_outline_qa(outline or [], sample_text=text):
            return DocumentTypeDetection(
                document_type="qa",
                split_strategy="outline_qa",
                confidence=0.9,
                reasons=["检测到 PDF 书签呈现“章节 -> 问题”的问答结构"],
            )

        numbered_questions = self.numbered_question_pattern.findall(text)
        numbered_items = self.numbered_item_pattern.findall(text)
        qa_marks = self.qa_mark_pattern.findall(text)

        qa_score = 0.0
        if any(word in lower_filename for word in ["问答", "常见问题", "faq", "100问"]):
            qa_score += 0.4
            reasons.append("文件名包含 FAQ、问答或 100问 等结构提示")
        if len(numbered_questions) >= 5:
            qa_score += 0.35
            reasons.append(f"检测到 {len(numbered_questions)} 个编号问题")
        if len(qa_marks) >= 4:
            qa_score += 0.2
            reasons.append("检测到多个 Q/A 或问/答标记")

        if qa_score >= 0.7:
            return DocumentTypeDetection(
                document_type="qa",
                split_strategy="numbered_qa",
                confidence=round(min(qa_score, 0.98), 2),
                reasons=reasons,
            )

        if len(numbered_items) >= 3:
            return DocumentTypeDetection(
                document_type="numbered",
                split_strategy="numbered_segments",
                confidence=0.72,
                reasons=[f"检测到 {len(numbered_items)} 个编号条目"],
            )

        return DocumentTypeDetection(
            document_type="text",
            split_strategy="recursive",
            confidence=round(min(qa_score, 0.5), 2),
            reasons=reasons or ["未检测到稳定结构，按普通文本递归切分"],
        )

    def build_segments_and_qas(
            self,
            *,
            document_id: str,
            documents: list[Document],
            document_type: str,
            split_strategy: str,
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """按用户确认后的结构类型和切分策略生成 segment/qa。"""

        normalized_type = self._normalize_document_type(document_type, split_strategy)
        if normalized_type == "qa" and split_strategy == "outline_qa":
            segments, qa_items = self._build_outline_qa_segments(document_id, documents)
            if segments:
                return segments, qa_items
            return self._build_qa_segments(document_id, documents)

        if normalized_type == "qa" and split_strategy == "numbered_qa":
            return self._build_qa_segments(document_id, documents)

        if normalized_type == "numbered" and split_strategy == "numbered_segments":
            segments = self._build_numbered_segments(document_id, documents, document_type=normalized_type)
            if segments:
                return segments, []

        return self._build_recursive_segments(document_id, documents, document_type=normalized_type), []

    def build_segments_and_faqs(
            self,
            *,
            document_id: str,
            documents: list[Document],
            document_type: str,
            split_strategy: str,
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """兼容旧命名，内部统一走 build_segments_and_qas。"""

        return self.build_segments_and_qas(
            document_id=document_id,
            documents=documents,
            document_type=document_type,
            split_strategy=split_strategy,
        )

    def _build_qa_segments(
            self,
            document_id: str,
            documents: list[Document],
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """把编号问答文档切成 QA segment。"""

        segments: list[DocumentSegment] = []
        qa_items: list[QaItem] = []

        for document in documents:
            blocks = self._split_numbered_blocks(document.page_content)
            for category, question_no, block in blocks:
                segment_index = len(segments)
                segment_id = f"{document_id}_seg_{segment_index:04d}"
                question, answer = self._parse_question_answer(block)
                content = self._format_qa_content(question, answer)
                page_no = document.metadata.get("page")

                segments.append(
                    DocumentSegment(
                        segment_id=segment_id,
                        segment_index=segment_index,
                        content=content,
                        content_hash=self._hash_text(content),
                        page_no=page_no,
                        heading_path=category,
                        metadata={
                            "document_type": "qa",
                            "split_strategy": "numbered_qa",
                            "question_no": question_no,
                        },
                    )
                )

                qa_items.append(
                    QaItem(
                        qa_id=f"{document_id}_qa_{segment_index:04d}",
                        segment_id=segment_id,
                        question_no=question_no,
                        question=question,
                        answer=answer,
                        category=category,
                        tags=["qa", "question", "answer"],
                        metadata={"source_page": page_no},
                    )
                )

        if segments:
            return segments, qa_items

        return self._build_recursive_segments(document_id, documents, document_type="qa"), []

    def _build_outline_qa_segments(
            self,
            document_id: str,
            documents: list[Document],
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """把“一级书签=章节，二级书签=题目”的 PDF 切成 QA segment。"""

        outline = self._get_pdf_outline(documents)
        if not self.is_outline_qa(outline, sample_text=self._sample_document_text(documents)):
            return [], []

        full_text, page_offsets = self._join_documents_with_offsets(documents)
        outline_questions = self._outline_question_items(outline)
        if not full_text or not outline_questions:
            return [], []

        located_questions = self._locate_outline_questions(outline_questions, full_text, page_offsets)
        if not located_questions:
            return [], []

        segments: list[DocumentSegment] = []
        qa_items: list[QaItem] = []

        for item_index, item in enumerate(located_questions):
            next_start = (
                located_questions[item_index + 1]["start"]
                if item_index + 1 < len(located_questions)
                else len(full_text)
            )
            answer = full_text[item["end"]:next_start].strip()
            answer = self._remove_leading_repeated_title(answer, item["raw_title"])
            question = self._clean_question_title(item["raw_title"])
            question_no = self._extract_question_no(item["raw_title"])
            category = item["category"]
            source_page = item.get("page")
            section_path = f"{category} / {item['raw_title']}" if category else item["raw_title"]
            question_id = f"{document_id}_outline_q_{item_index:04d}"
            content_parts = self._split_outline_qa_content(question, answer)

            for part_index, content in enumerate(content_parts, start=1):
                segment_index = len(segments)
                segment_id = f"{document_id}_seg_{segment_index:04d}"
                qa_id = f"{document_id}_qa_{segment_index:04d}"
                metadata = {
                    "document_type": "qa",
                    "split_strategy": "outline_qa",
                    "question_no": question_no,
                    "question_id": question_id,
                    "section_title": category,
                    "section_path": section_path,
                    "structure_source": "pdf_outline",
                    "part_index": part_index,
                    "part_count": len(content_parts),
                }
                segments.append(
                    DocumentSegment(
                        segment_id=segment_id,
                        segment_index=segment_index,
                        content=content,
                        content_hash=self._hash_text(content),
                        page_no=source_page,
                        heading_path=category,
                        metadata=metadata,
                    )
                )
                qa_items.append(
                    QaItem(
                        qa_id=qa_id,
                        segment_id=segment_id,
                        question_no=question_no,
                        question=question,
                        answer=answer if part_index == 1 else None,
                        category=category,
                        tags=["qa", "question", "answer", "outline"],
                        metadata={
                            "source_page": source_page,
                            "question_id": question_id,
                            "section_path": section_path,
                            "part_index": part_index,
                            "part_count": len(content_parts),
                        },
                    )
                )

        return segments, qa_items

    def _build_numbered_segments(
            self,
            document_id: str,
            documents: list[Document],
            *,
            document_type: str,
    ) -> list[DocumentSegment]:
        """把编号条目型文档按编号切成 segment。"""

        segments: list[DocumentSegment] = []

        for document in documents:
            blocks = self._split_numbered_blocks(document.page_content)
            for category, item_no, block in blocks:
                content = self._clean_markdown(block)
                if not content:
                    continue
                segments.append(
                    DocumentSegment(
                        segment_id=f"{document_id}_seg_{len(segments):04d}",
                        segment_index=len(segments),
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
        """把普通文本用递归切分器切成 segment。"""

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
        """按 Markdown 标题和编号行切出连续条目。"""

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
        """从单个编号块里拆出问题和答案。"""

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

    @classmethod
    def is_outline_qa(cls, outline: list[dict[str, Any]], *, sample_text: str = "") -> bool:
        """判断 PDF outline 是否是“章节 -> 问题”的两层问答结构。"""

        if not outline:
            return False

        level0_count = sum(1 for item in outline if int(item.get("level") or 0) == 0)
        level1_items = [item for item in outline if int(item.get("level") or 0) == 1]
        if level0_count < 2 or len(level1_items) < max(6, level0_count * 3):
            return False

        question_like_count = sum(1 for item in level1_items if cls._is_question_like_title(str(item.get("title") or "")))
        question_ratio = question_like_count / len(level1_items)
        if question_ratio < 0.5:
            return False

        if sample_text:
            sample_hits = 0
            for item in level1_items[: min(20, len(level1_items))]:
                title = str(item.get("title") or "").strip()
                clean_title = cls._clean_question_title(title)
                if title and (title in sample_text or clean_title in sample_text):
                    sample_hits += 1
            return sample_hits >= 1

        return True

    @staticmethod
    def _get_pdf_outline(documents: list[Document]) -> list[dict[str, Any]]:
        """从 Document metadata 中取 PDF outline。"""

        for document in documents:
            outline = document.metadata.get("_pdf_outline")
            if isinstance(outline, list):
                return outline
        return []

    @staticmethod
    def _sample_document_text(documents: list[Document], *, max_chars: int = 8000) -> str:
        """抽取文档前几页文本，用于校验 outline 标题是否能在正文中找到。"""

        parts: list[str] = []
        total = 0
        for document in documents[:8]:
            text = document.page_content.strip()
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]

    @classmethod
    def _outline_question_items(cls, outline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """取出二级问题书签，并补齐所属一级章节。"""

        result: list[dict[str, Any]] = []
        current_category: str | None = None

        for item in outline:
            level = int(item.get("level") or 0)
            title = str(item.get("title") or "").strip()
            if not title:
                continue

            if level == 0:
                current_category = cls._clean_markdown(title)
                continue

            if level == 1 and current_category and cls._is_question_like_title(title):
                result.append(
                    {
                        "category": current_category,
                        "raw_title": cls._clean_markdown(title),
                        "page": item.get("page"),
                    }
                )

        return result

    @staticmethod
    def _join_documents_with_offsets(documents: list[Document]) -> tuple[str, dict[int, int]]:
        """拼接所有页面文本，同时记录页码到全局字符偏移。"""

        parts: list[str] = []
        page_offsets: dict[int, int] = {}
        cursor = 0

        for index, document in enumerate(documents, start=1):
            page_no = document.metadata.get("page")
            if page_no is None:
                normalized_page = index
            else:
                normalized_page = int(page_no) + 1

            separator = "\n\n" if parts else ""
            cursor += len(separator)
            page_offsets[normalized_page] = cursor
            text = document.page_content or ""
            parts.append(separator + text)
            cursor += len(text)

        return "".join(parts), page_offsets

    @classmethod
    def _locate_outline_questions(
            cls,
            outline_questions: list[dict[str, Any]],
            full_text: str,
            page_offsets: dict[int, int],
    ) -> list[dict[str, Any]]:
        """按 outline 顺序在正文里定位题目标题。"""

        located: list[dict[str, Any]] = []
        search_start = 0

        for item in outline_questions:
            raw_title = item["raw_title"]
            page = item.get("page")
            page_start = page_offsets.get(int(page), search_start) if page else search_start
            start = cls._find_title_position(full_text, raw_title, max(search_start, page_start - 300))
            if start < 0:
                start = cls._find_title_position(full_text, cls._clean_question_title(raw_title), max(search_start, page_start - 300))
            if start < 0:
                continue

            end = start + len(raw_title)
            clean_title = cls._clean_question_title(raw_title)
            if full_text[start:start + len(clean_title)] == clean_title:
                end = start + len(clean_title)

            located.append({**item, "start": start, "end": end})
            search_start = max(end, start + 1)

        return located

    @staticmethod
    def _find_title_position(full_text: str, title: str, start: int) -> int:
        """在全文中从指定位置开始查找标题。"""

        clean_title = title.strip()
        if not clean_title:
            return -1

        position = full_text.find(clean_title, max(0, start))
        if position >= 0:
            return position

        compact_text = re.sub(r"\s+", "", full_text)
        compact_title = re.sub(r"\s+", "", clean_title)
        compact_position = compact_text.find(compact_title, max(0, start))
        if compact_position < 0:
            return -1

        non_space_count = 0
        for index, char in enumerate(full_text):
            if char.isspace():
                continue
            if non_space_count == compact_position:
                return index
            non_space_count += 1
        return -1

    @classmethod
    def _is_question_like_title(cls, title: str) -> bool:
        """判断书签标题是否像一道题目。"""

        clean_title = cls._clean_markdown(title)
        if not clean_title:
            return False

        if re.match(r"^\s*\d+[.、]\s*\S+", clean_title):
            return True
        if any(mark in clean_title for mark in ("？", "?")):
            return True
        return any(keyword in clean_title for keyword in cls.outline_question_keywords)

    @staticmethod
    def _extract_question_no(title: str) -> int | None:
        """从标题开头提取题号。"""

        match = re.match(r"^\s*(\d+)[.、]\s*", title)
        return int(match.group(1)) if match else None

    @staticmethod
    def _clean_question_title(title: str) -> str:
        """去掉题目前置编号，得到纯问题文本。"""

        clean_title = DocumentParser._clean_markdown(title)
        return re.sub(r"^\s*\d+[.、]\s*", "", clean_title).strip()

    @staticmethod
    def _remove_leading_repeated_title(answer: str, raw_title: str) -> str:
        """去掉答案开头重复出现的题目标题。"""

        clean_answer = answer.strip()
        for candidate in (raw_title, DocumentParser._clean_question_title(raw_title)):
            candidate = candidate.strip()
            if candidate and clean_answer.startswith(candidate):
                clean_answer = clean_answer[len(candidate):].strip()
        return clean_answer

    def _split_outline_qa_content(self, question: str, answer: str | None, *, max_chars: int = 1500) -> list[str]:
        """把 outline QA 格式化成文本，超长答案按段落二次切片。"""

        answer_text = (answer or "").strip()
        if not answer_text:
            return [self._format_qa_content(question, None)]

        if len(question) + len(answer_text) <= max_chars:
            return [self._format_qa_content(question, answer_text)]

        chunks: list[str] = []
        current_lines: list[str] = []
        current_length = 0
        for line in answer_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if current_lines and current_length + len(line) > max_chars:
                chunks.append(self._format_qa_content(question, "\n".join(current_lines)))
                current_lines = []
                current_length = 0
            current_lines.append(line)
            current_length += len(line)

        if current_lines:
            chunks.append(self._format_qa_content(question, "\n".join(current_lines)))

        return chunks or [self._format_qa_content(question, answer_text[:max_chars])]

    @staticmethod
    def _normalize_document_type(document_type: str, split_strategy: str) -> str:
        """统一文档结构类型，只允许 qa/numbered/text。"""

        value = (document_type or "").strip().lower()
        if split_strategy in {"numbered_qa", "outline_qa"}:
            return "qa"
        if split_strategy == "numbered_segments":
            return "numbered"
        if value in {"qa", "numbered", "text"}:
            return value
        return "text"

    @staticmethod
    def _format_qa_content(question: str, answer: str | None) -> str:
        """把问答内容格式化成可检索文本。"""

        if answer:
            return f"问题：{question}\n答案：{answer}"
        return f"问题：{question}"

    @staticmethod
    def _hash_text(text: str) -> str:
        """计算文本内容哈希。"""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """清理少量 Markdown 强调标记。"""

        text = text.strip()
        text = text.replace("**", "")
        text = text.replace("__", "")
        return text.strip()

    @staticmethod
    def dumps_metadata(value: dict[str, Any] | list[str] | None) -> str:
        """把 metadata 序列化为 JSON 字符串。"""

        return json.dumps(value or {}, ensure_ascii=False)
