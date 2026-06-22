import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from rag.file_processors import FileProcessorFactory


@dataclass
class TrainingChunk:
    """训练知识切片。

    @dataclass 类似 Java 的 Lombok @Data：
    Python 会自动生成 __init__、__repr__ 等方法。
    这个对象只在入库流程中临时使用，不直接作为 API 响应。
    """

    chunk_id: str  # 切片唯一 ID。
    text: str  # 切片正文，会进入 embedding。
    case_part: str  # 业务片段类型。
    visibility: str  # 可见范围：visible / hidden / scoring_only。
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据，default_factory 避免共享可变 dict。


class KnowledgeIngestStrategy(ABC):
    """训练知识入库策略接口。

    ABC + @abstractmethod 类似 Java 的 interface / abstract class。
    不同资料类型的切片规则不同，但都要实现 parse_chunks。
    """

    @abstractmethod
    def parse_chunks(self, file_path: str, context: dict[str, Any]) -> list[TrainingChunk]:
        """把文件解析成训练知识切片。

        file_path：本地已保存的上传文件路径。
        context：上传批次、来源类型、行业、难度等上下文信息。
        """


class LmsCaseIngestStrategy(KnowledgeIngestStrategy):
    """LMS 场景任务案例入库策略。

    策略模式适合这里：LMS、FAQ、竞品资料的切片规则不同，
    一期先实现 LMS，后续扩展不需要改上传主流程。
    """

    # 编译正则后复用，避免每次切片都重新解析正则。
    # 该规则用于识别“一、客户案例”这类中文编号标题。
    CASE_TITLE_PATTERN = re.compile(r"^[一二三四五六七八九十]+[、.．]\s*.+")
    # PART_MARKERS 用关键词把案例内容分到不同业务片段。
    # 元组第一项是内部类型，第二项是一组可匹配的中文标题关键词。
    PART_MARKERS = [
        ("task_requirement", ("任务要求",)),
        ("standard_answer", ("匹配答案", "话术案例", "话术原文", "谈单话术", "参考答案")),
        ("hidden_psychology", ("隐性心理", "底层顾虑", "底层需求", "客户底层心理", "核心显性卡点", "隐性痛点")),
        ("scoring_rubric", ("命中点", "扣分点", "评分标准", "能力维度")),
        ("case_profile", ("客户案例", "企业", "老板身份", "合作行业", "客户阶段")),
    ]

    def parse_chunks(self, file_path: str, context: dict[str, Any]) -> list[TrainingChunk]:
        """按 LMS 任务案例结构切片。

        这一步只负责“结构化拆分”，不直接调用 embedding。
        向量写入由 SalesTrainingService 统一处理。
        """

        documents = FileProcessorFactory.load_documents(file_path)
        # FileProcessorFactory 返回 LangChain Document；这里再抽出段落列表。
        paragraphs = self._paragraphs_from_documents(documents)
        cases = self._split_cases(paragraphs)
        chunks: list[TrainingChunk] = []

        # enumerate(..., start=1) 会从 1 开始计数，更符合业务展示习惯。
        for case_index, case in enumerate(cases, start=1):
            case_title = case["title"] or f"训练案例 {case_index}"
            parts = self._split_parts(case["lines"])
            for case_part, lines in parts.items():
                text = "\n".join(line for line in lines if line.strip()).strip()
                if not text:
                    continue
                visibility = self._visibility_for_part(case_part)
                # :03d 表示数字补足 3 位，例如 1 -> 001，方便排序和排查。
                chunk_id = f"{context['batch_id']}_{case_index:03d}_{case_part}"
                chunks.append(
                    TrainingChunk(
                        chunk_id=chunk_id,
                        text=f"{case_title}\n{text}",
                        case_part=case_part,
                        visibility=visibility,
                        metadata={
                            "case_title": case_title,
                            "case_index": case_index,
                            "source_file": context.get("source_file"),
                        },
                    )
                )

        if chunks:
            return chunks

        # 兜底：如果文档结构完全不符合预期，至少保证能入库并被检索。
        fallback_text = "\n".join(paragraphs).strip()
        return [
            TrainingChunk(
                chunk_id=f"{context['batch_id']}_001_case_profile",
                text=fallback_text,
                case_part="case_profile",
                visibility=context.get("visibility_default") or "visible",
                metadata={"case_title": "未识别结构的 LMS 文档", "source_file": context.get("source_file")},
            )
        ] if fallback_text else []

    @staticmethod
    def _paragraphs_from_documents(documents: list[Document]) -> list[str]:
        """优先使用 DOCX 处理器保留的段落，否则按换行拆分。

        DOCX 处理器会把原始段落放到 metadata["paragraphs"]。
        PDF/TXT 没有这个结构时，就用 page_content.splitlines() 兜底。
        """

        paragraphs: list[str] = []
        for document in documents:
            raw_paragraphs = document.metadata.get("paragraphs")
            if isinstance(raw_paragraphs, list):
                paragraphs.extend(str(item.get("text") or "").strip() for item in raw_paragraphs if isinstance(item, dict))
            else:
                paragraphs.extend(line.strip() for line in document.page_content.splitlines())
        return [text for text in paragraphs if text]

    def _split_cases(self, paragraphs: list[str]) -> list[dict[str, Any]]:
        """按“一、二、三”这类标题拆成训练案例。

        返回结构示例：
            [{"title": "一、xxx", "lines": ["..."]}]
        """

        cases: list[dict[str, Any]] = []
        current_title = ""
        current_lines: list[str] = []

        for paragraph in paragraphs:
            if self.CASE_TITLE_PATTERN.match(paragraph):
                if current_lines:
                    cases.append({"title": current_title, "lines": current_lines})
                current_title = paragraph
                current_lines = []
                continue
            current_lines.append(paragraph)

        if current_lines:
            cases.append({"title": current_title, "lines": current_lines})
        return cases

    def _split_parts(self, lines: list[str]) -> dict[str, list[str]]:
        """把单个案例拆成业务段落。

        current_part 表示“当前正在收集哪个片段”。
        遇到新的标题关键词后，后续行会进入新的片段。
        """

        parts: dict[str, list[str]] = {
            "case_profile": [],
            "task_requirement": [],
            "standard_answer": [],
            "hidden_psychology": [],
            "scoring_rubric": [],
        }
        current_part = "case_profile"

        for line in lines:
            matched_part = self._detect_part(line)
            if matched_part:
                current_part = matched_part
                continue
            parts[current_part].append(line)

        return parts

    def _detect_part(self, line: str) -> str | None:
        """根据标题关键词判断当前段落属于哪类业务片段。

        返回 None 表示这一行不是标题，而是正文。
        """

        clean_line = line.strip()
        for part, markers in self.PART_MARKERS:
            if any(marker in clean_line for marker in markers):
                return part
        return None

    @staticmethod
    def _visibility_for_part(case_part: str) -> str:
        """根据片段类型决定可见范围。

        hidden_psychology 只给 AI 客户使用；
        scoring_rubric 只给评分模型使用；
        其他内容默认学员也可见。
        """

        if case_part == "hidden_psychology":
            return "hidden"
        if case_part == "scoring_rubric":
            return "scoring_only"
        return "visible"


class GenericTrainingIngestStrategy(KnowledgeIngestStrategy):
    """通用训练资料入库策略，供产品资料、FAQ 等一期兜底使用。"""

    def parse_chunks(self, file_path: str, context: dict[str, Any]) -> list[TrainingChunk]:
        """按整篇文档生成一个通用切片。

        这不是最精细的切法，但能保证未知结构的文档也能进入训练库。
        后续如果有新的资料类型，再新增专门 Strategy。
        """

        documents = FileProcessorFactory.load_documents(file_path)
        text = "\n\n".join(document.page_content for document in documents).strip()
        if not text:
            return []
        return [
            TrainingChunk(
                chunk_id=f"{context['batch_id']}_001_{context.get('source_type') or 'generic'}",
                text=text,
                case_part=str(context.get("case_part") or context.get("source_type") or "product_fact"),
                visibility=str(context.get("visibility_default") or "visible"),
                metadata={"source_file": context.get("source_file")},
            )
        ]
