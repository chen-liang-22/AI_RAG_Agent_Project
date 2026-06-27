"""通用文档解析器。

这个模块把 PDF/TXT/DOCX 读取后的 Document 转成两类结构：
1. DocumentSegment：真正写入 Qdrant 的文本片段；
2. QaItem：从问答资料中额外抽出的结构化问题和答案。

它只负责“文档结构解析和切片编排”，不负责文件读取、Embedding、Qdrant 写入。
具体切分算法通过 split_strategies 包里的策略类实现。
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.rag.llm_semantic_splitter import LlmSemanticSplitter
from core.rag.split_strategies.base import SplitContext
from core.rag.split_strategies.factory import SplitStrategyFactory
from core.utils.config_handler import rag_conf
from core.utils.logger_handler import logger


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


@dataclass(frozen=True)
class _TitleMatch:
    """PDF 目录标题在原文中的定位结果。"""

    start: int
    end: int
    method: str


@dataclass(frozen=True)
class DocumentParseRules:
    """文档解析规则。

    这些规则来自 `config/app.yml` 的 rag.document_parse_rules。
    规则对象负责“编译和使用正则”，DocumentParser 只关心解析流程，避免把资料格式写死在业务代码里。
    """

    numbered_item_pattern: re.Pattern[str]
    heading_pattern: re.Pattern[str]
    number_prefix_pattern: re.Pattern[str]
    number_prefix_only_pattern: re.Pattern[str]
    answer_prefix_pattern: re.Pattern[str]
    invalid_answer_pattern: re.Pattern[str]
    question_marks: tuple[str, ...]

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "DocumentParseRules":
        """从配置创建规则对象。

        这里故意不在 Python 里写默认正则。
        原因：编号、标题、答案前缀属于“资料格式约定”，后续资料格式变化时应该改 YAML，
        不应该改业务代码；配置缺失时直接报错，能尽早发现部署配置不完整。
        """

        if not isinstance(config, dict):
            raise ValueError("document_parse_rules 配置缺失或格式错误，请在 config/app.yml 的 rag 节点中配置文档解析规则")

        question_marks = config.get("question_marks")
        if not isinstance(question_marks, list) or not question_marks:
            raise ValueError("document_parse_rules.question_marks 必须配置为非空列表")

        return cls(
            numbered_item_pattern=cls._compile(
                config.get("numbered_item_pattern"),
                flags=re.MULTILINE,
                name="numbered_item_pattern",
            ),
            heading_pattern=cls._compile(
                config.get("heading_pattern"),
                flags=re.MULTILINE,
                name="heading_pattern",
            ),
            number_prefix_pattern=cls._compile(
                config.get("number_prefix_pattern"),
                name="number_prefix_pattern",
            ),
            number_prefix_only_pattern=cls._compile(
                config.get("number_prefix_only_pattern"),
                name="number_prefix_only_pattern",
            ),
            answer_prefix_pattern=cls._compile(
                config.get("answer_prefix_pattern"),
                flags=re.IGNORECASE,
                name="answer_prefix_pattern",
            ),
            invalid_answer_pattern=cls._compile(
                config.get("invalid_answer_pattern"),
                flags=re.IGNORECASE,
                name="invalid_answer_pattern",
            ),
            question_marks=tuple(str(mark) for mark in question_marks if str(mark)),
        )

    @staticmethod
    def _compile(pattern: object, *, flags: int = 0, name: str) -> re.Pattern[str]:
        """编译配置正则；配置缺失或写错时直接报中文错误，避免静默使用隐藏规则。"""

        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"document_parse_rules.{name} 必须配置为非空正则字符串")

        raw_pattern = pattern.strip()
        try:
            return re.compile(raw_pattern, flags)
        except re.error as exc:
            raise ValueError(f"document_parse_rules.{name} 正则配置无效：{exc}") from exc

    def remove_number_prefix(self, text: str) -> str:
        """移除文本开头的编号前缀。"""

        return self.number_prefix_pattern.sub("", text, count=1).strip()

    def extract_number(self, text: str) -> int | None:
        """从文本开头提取编号。"""

        match = self.number_prefix_pattern.match(text)
        return int(match.group(1)) if match else None

    def remove_answer_prefix(self, text: str) -> str:
        """移除答案行开头的“答：/A：/答案：”等前缀。"""

        return self.answer_prefix_pattern.sub("", text, count=1).strip()

    def is_invalid_answer(self, text: str) -> bool:
        """判断清洗后的答案是否只是无效占位标签。"""

        compact_text = re.sub(r"\s+", "", text or "")
        if not compact_text:
            return True
        return self.invalid_answer_pattern.fullmatch(compact_text) is not None

    def is_number_prefix_only(self, text: str) -> bool:
        """判断一段文本是否只是编号前缀。"""

        return self.number_prefix_only_pattern.fullmatch(text) is not None

    def has_question_mark(self, text: str) -> bool:
        """判断文本中是否包含配置的问题标点。"""

        return any(mark in text for mark in self.question_marks)


class DocumentParser:
    """通用文档解析器，只识别文档结构，不绑定具体业务分类。"""

    default_rules = DocumentParseRules.from_config((rag_conf or {}).get("document_parse_rules"))

    def __init__(
            self,
            splitter: RecursiveCharacterTextSplitter,
            semantic_splitter: object | None = None,
            parse_rules: DocumentParseRules | None = None,
    ):
        """初始化文档解析器。"""

        self.splitter = splitter
        self.semantic_splitter = semantic_splitter or LlmSemanticSplitter()
        self.parse_rules = parse_rules or self.default_rules

    def detect_document_type(
            self,
            filename: str,
            sample_text: str,
            outline: list[dict[str, Any]] | None = None,
    ) -> DocumentTypeDetection:
        """用轻量规则给出默认结构建议，模型推荐由上传推荐接口单独完成。"""

        return DocumentTypeDetection(
            document_type="text",
            split_strategy="recursive",
            confidence=0.5,
            reasons=["未使用模型推荐时默认按普通文本递归切分"],
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
        strategy = SplitStrategyFactory.get_strategy(split_strategy)
        logger.info(
            "[文档解析] 开始切片 文档编号=%s 文档类型=%s 切分策略=%s 原始页数=%s",
            document_id,
            normalized_type,
            split_strategy,
            len(documents),
        )
        segments, qa_items = strategy.split(
            SplitContext(
                document_id=document_id,
                documents=documents,
                document_type=normalized_type,
                split_strategy=split_strategy,
                parser=self,
            )
        )
        logger.info(
            "[文档解析] 切片完成 文档编号=%s 文档类型=%s 切分策略=%s 片段数=%s 问答数=%s",
            document_id,
            normalized_type,
            split_strategy,
            len(segments),
            len(qa_items),
        )
        return segments, qa_items

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

        logger.info("[文档解析] 编号问答切片开始 文档编号=%s 页数=%s", document_id, len(documents))
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
            logger.info("[文档解析] 编号问答切片完成 文档编号=%s 片段数=%s", document_id, len(segments))
            return segments, qa_items

        logger.warning("[文档解析] 编号问答未识别到有效条目，回退递归切分 文档编号=%s", document_id)
        return self._build_recursive_segments(document_id, documents, document_type="qa"), []

    def _build_outline_qa_segments(
            self,
            document_id: str,
            documents: list[Document],
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """把“一级书签=章节，二级书签=题目”的 PDF 切成 QA segment。"""

        outline = self._get_pdf_outline(documents)
        if not self.is_outline_qa(outline, sample_text=self._sample_document_text(documents)):
            logger.info("[文档解析] PDF目录不是问答结构，跳过目录问答切片 文档编号=%s 目录项=%s", document_id, len(outline))
            return [], []

        full_text, page_offsets = self._join_documents_with_offsets(documents)
        outline_questions = self._outline_question_items(outline)
        if not full_text or not outline_questions:
            logger.warning("[文档解析] PDF目录问答缺少正文或题目 文档编号=%s 目录题数=%s", document_id, len(outline_questions))
            return [], []

        located_questions = self._locate_outline_questions(outline_questions, full_text, page_offsets)
        if not located_questions:
            logger.warning("[文档解析] PDF目录问答定位失败 目录题数=%s", len(outline_questions))
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
            answer = self._clean_outline_answer(answer)
            if self._is_invalid_outline_answer(answer):
                logger.warning(
                    "[文档解析] PDF目录问答跳过异常答案 标题=%s 页码=%s 匹配方式=%s 答案预览=%s",
                    item["raw_title"],
                    item.get("page"),
                    item.get("match_method"),
                    answer[:80],
                )
                continue

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
                    "outline_match_method": item.get("match_method"),
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
                            "outline_match_method": item.get("match_method"),
                            "part_index": part_index,
                            "part_count": len(content_parts),
                        },
                    )
                )

        logger.info("[文档解析] PDF目录问答切片完成 文档编号=%s 片段数=%s 问答数=%s", document_id, len(segments), len(qa_items))
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

    def _build_llm_semantic_segments(
            self,
            document_id: str,
            documents: list[Document],
            *,
            document_type: str,
    ) -> tuple[list[DocumentSegment], list[QaItem]]:
        """按 LLM 返回的原文 span 生成语义切片。"""

        full_text = LlmSemanticSplitter.join_documents(documents)
        try:
            plans = self.semantic_splitter.split(documents)
        except Exception as exc:
            logger.warning("[文档解析] LLM语义切片失败，回退递归切分 文档编号=%s 错误=%s", document_id, exc)
            return self._build_recursive_segments(document_id, documents, document_type=document_type), []

        segments: list[DocumentSegment] = []
        qa_items: list[QaItem] = []
        used_ranges: list[tuple[int, int]] = []

        for plan in plans:
            raw_start = self._plan_value(plan, "start")
            raw_end = self._plan_value(plan, "end")
            try:
                start = int(raw_start)
                end = int(raw_end)
            except (TypeError, ValueError):
                continue
            if not self._is_valid_semantic_range(start, end, len(full_text), used_ranges):
                continue

            content = full_text[start:end].strip()
            if not content:
                continue

            segment_index = len(segments)
            segment_id = f"{document_id}_seg_{segment_index:04d}"
            content_type = str(self._plan_value(plan, "content_type") or "segment").strip().lower()
            question = str(self._plan_value(plan, "question") or "").strip()
            category = str(self._plan_value(plan, "category") or "").strip() or None
            semantic_title = str(self._plan_value(plan, "title") or "").strip() or None
            metadata = {
                "document_type": document_type,
                "split_strategy": "llm_semantic",
                "semantic_title": semantic_title,
                "semantic_reason": str(self._plan_value(plan, "reason") or "").strip() or None,
                "source_start": start,
                "source_end": end,
                "structure_source": "llm_semantic",
            }
            segments.append(
                DocumentSegment(
                    segment_id=segment_id,
                    segment_index=segment_index,
                    content=content,
                    content_hash=self._hash_text(content),
                    page_no=None,
                    heading_path=category,
                    metadata=metadata,
                )
            )

            if content_type == "qa":
                qa_items.append(
                    QaItem(
                        qa_id=f"{document_id}_qa_{segment_index:04d}",
                        segment_id=segment_id,
                        question_no=None,
                        question=question or semantic_title or content[:80],
                        answer=content,
                        category=category,
                        tags=["qa", "llm_semantic"],
                        metadata={
                            "source_start": start,
                            "source_end": end,
                            "structure_source": "llm_semantic",
                        },
                    )
                )
            used_ranges.append((start, end))

        if not segments:
            logger.warning("[文档解析] LLM语义切片无有效范围，回退递归切分 文档编号=%s", document_id)
            return self._build_recursive_segments(document_id, documents, document_type=document_type), []

        return segments, qa_items

    def _split_numbered_blocks(self, text: str) -> list[tuple[str | None, int | None, str]]:
        """按 Markdown 标题和编号行切出连续条目。"""

        blocks: list[tuple[str | None, int | None, str]] = []
        current_category: str | None = None
        current_number: int | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            heading_match = self.parse_rules.heading_pattern.match(stripped)
            numbered_match = self.parse_rules.numbered_item_pattern.match(stripped)

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

        block_without_number = self.parse_rules.numbered_item_pattern.sub(r"\2", block, count=1).strip()
        block_without_number = self._clean_markdown(block_without_number)
        lines = [line.strip() for line in block_without_number.splitlines() if line.strip()]

        if not lines:
            return block_without_number[:80], None

        question = self._clean_markdown(lines[0])
        answer_lines = []
        for line in lines[1:]:
            cleaned = line.removeprefix("-").strip()
            cleaned = self.parse_rules.remove_answer_prefix(cleaned)
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
                raw_match = cls._find_title_match(sample_text, title, 0)
                clean_match = cls._find_title_match(sample_text, clean_title, 0, expand_prefix=True)
                if title and (raw_match or clean_match):
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
            search_from, search_to = cls._outline_title_search_range(
                page,
                page_offsets,
                len(full_text),
                search_start,
            )
            match = cls._find_outline_title_match(full_text, raw_title, search_from, search_to)
            if match is None and search_to < len(full_text):
                match = cls._find_outline_title_match(full_text, raw_title, search_from, len(full_text))
            if match is None:
                logger.warning(
                    "[文档解析] PDF目录标题定位失败 标题=%s 页码=%s 搜索范围=%s-%s",
                    raw_title,
                    page,
                    search_from,
                    search_to,
                )
                continue

            located.append(
                {
                    **item,
                    "start": match.start,
                    "end": match.end,
                    "match_method": match.method,
                }
            )
            search_start = max(match.end, match.start + 1)

        return located

    @classmethod
    def _find_title_position(cls, full_text: str, title: str, start: int) -> int:
        """在全文中从指定位置开始查找标题。"""

        match = cls._find_title_match(full_text, title, start)
        return match.start if match else -1

    @classmethod
    def _find_outline_title_match(
            cls,
            full_text: str,
            raw_title: str,
            start: int,
            end: int | None = None,
    ) -> _TitleMatch | None:
        """按原始标题和清洗标题两种方式定位目录题目。"""

        raw_match = cls._find_title_match(full_text, raw_title, start, end)
        if raw_match:
            return raw_match

        clean_title = cls._clean_question_title(raw_title)
        return cls._find_title_match(full_text, clean_title, start, end, expand_prefix=True)

    @classmethod
    def _find_title_match(
            cls,
            full_text: str,
            title: str,
            start: int,
            end: int | None = None,
            *,
            expand_prefix: bool = False,
    ) -> _TitleMatch | None:
        """在原文中查找标题，并返回原文坐标中的标题起止位置。"""

        clean_title = title.strip()
        if not clean_title:
            return None

        search_start = max(0, start)
        search_end = min(len(full_text), len(full_text) if end is None else end)
        if search_start >= search_end:
            return None

        position = full_text.find(clean_title, search_start, search_end)
        if position >= 0:
            match_start = cls._expand_match_start_to_number_prefix(full_text, position, search_start)
            method = "exact_with_prefix" if expand_prefix and match_start != position else "exact"
            return _TitleMatch(match_start, position + len(clean_title), method)

        compact_text, compact_offsets = cls._build_compact_text_offsets(full_text, search_start, search_end)
        compact_title = re.sub(r"\s+", "", clean_title)
        if not compact_title:
            return None

        compact_position = compact_text.find(compact_title)
        if compact_position < 0:
            return None

        original_start = compact_offsets[compact_position]
        original_end = compact_offsets[compact_position + len(compact_title) - 1] + 1
        match_start = cls._expand_match_start_to_number_prefix(full_text, original_start, search_start)
        method = "compact_with_prefix" if expand_prefix and match_start != original_start else "compact"
        return _TitleMatch(match_start, original_end, method)

    @staticmethod
    def _outline_title_search_range(
            page: Any,
            page_offsets: dict[int, int],
            full_text_length: int,
            search_start: int,
    ) -> tuple[int, int]:
        """根据 PDF 书签页码给标题定位限定搜索范围，避免误匹配目录页。"""

        try:
            page_no = int(page) if page else None
        except (TypeError, ValueError):
            page_no = None

        if page_no is None or page_no not in page_offsets:
            return max(0, search_start), full_text_length

        page_start = page_offsets[page_no]
        next_page_start = full_text_length
        for candidate_page, candidate_offset in sorted(page_offsets.items()):
            if candidate_page > page_no:
                next_page_start = candidate_offset
                break

        search_from = max(search_start, page_start)
        search_to = min(full_text_length, next_page_start + 800)
        return search_from, search_to

    @staticmethod
    def _build_compact_text_offsets(text: str, start: int, end: int) -> tuple[str, list[int]]:
        """生成去空白文本，并记录每个压缩字符对应的原文位置。"""

        compact_chars: list[str] = []
        offsets: list[int] = []
        for index in range(start, end):
            char = text[index]
            if char.isspace():
                continue
            compact_chars.append(char)
            offsets.append(index)
        return "".join(compact_chars), offsets

    @staticmethod
    def _expand_match_start_to_number_prefix(full_text: str, match_start: int, min_start: int) -> int:
        """清洗标题命中时，把同一行前面的题号前缀一起纳入标题范围。"""

        line_start = full_text.rfind("\n", max(0, min_start), match_start) + 1
        prefix = full_text[line_start:match_start]
        if DocumentParser.default_rules.is_number_prefix_only(prefix):
            return line_start
        return match_start

    @classmethod
    def _is_question_like_title(cls, title: str) -> bool:
        """判断书签标题是否像一道题目。"""

        clean_title = cls._clean_markdown(title)
        if not clean_title:
            return False

        if cls.default_rules.number_prefix_pattern.match(clean_title):
            return True
        if cls.default_rules.has_question_mark(clean_title):
            return True
        return False

    @staticmethod
    def _plan_value(plan: object, key: str) -> object:
        """兼容 dataclass 和 dict 形式的 LLM 切片计划。"""

        if isinstance(plan, dict):
            return plan.get(key)
        return getattr(plan, key, None)

    @staticmethod
    def _is_valid_semantic_range(
            start: int,
            end: int,
            full_text_length: int,
            used_ranges: list[tuple[int, int]],
    ) -> bool:
        """校验 LLM 返回的原文范围。"""

        if start < 0 or end <= start or end > full_text_length:
            return False
        for used_start, used_end in used_ranges:
            overlap_start = max(start, used_start)
            overlap_end = min(end, used_end)
            if overlap_end <= overlap_start:
                continue
            overlap_size = overlap_end - overlap_start
            current_size = end - start
            if overlap_size / current_size > 0.8:
                return False
        return True

    @staticmethod
    def _extract_question_no(title: str) -> int | None:
        """从标题开头提取题号。"""

        return DocumentParser.default_rules.extract_number(title)

    @staticmethod
    def _clean_question_title(title: str) -> str:
        """去掉题目前置编号，得到纯问题文本。"""

        clean_title = DocumentParser._clean_markdown(title)
        return DocumentParser.default_rules.remove_number_prefix(clean_title)

    @staticmethod
    def _remove_leading_repeated_title(answer: str, raw_title: str) -> str:
        """去掉答案开头重复出现的题目标题。"""

        clean_answer = answer.strip()
        for candidate in (raw_title, DocumentParser._clean_question_title(raw_title)):
            candidate = candidate.strip()
            clean_answer = DocumentParser._remove_leading_text_variant(clean_answer, candidate)
        return clean_answer

    @staticmethod
    def _remove_leading_text_variant(text: str, candidate: str) -> str:
        """按原样或去空白后的写法移除答案开头重复文本。"""

        clean_text = text.strip()
        clean_candidate = candidate.strip()
        if not clean_text or not clean_candidate:
            return clean_text

        if clean_text.startswith(clean_candidate):
            return clean_text[len(clean_candidate):].strip()

        compact_text = re.sub(r"\s+", "", clean_text)
        compact_candidate = re.sub(r"\s+", "", clean_candidate)
        if not compact_candidate or not compact_text.startswith(compact_candidate):
            return clean_text

        matched_count = 0
        for index, char in enumerate(clean_text):
            if char.isspace():
                continue
            matched_count += 1
            if matched_count == len(compact_candidate):
                return clean_text[index + 1:].strip()

        return clean_text

    @staticmethod
    def _clean_outline_answer(answer: str) -> str:
        """清理目录问答答案开头常见的答案标签。"""

        clean_answer = answer.strip()
        return DocumentParser.default_rules.remove_answer_prefix(clean_answer)

    @staticmethod
    def _is_invalid_outline_answer(answer: str) -> bool:
        """判断目录问答切出来的答案是否明显无效。"""

        return DocumentParser.default_rules.is_invalid_answer(answer)

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
