"""训练资料入库切片策略。

这里使用策略模式：
- LMS 场景案例按案例标题、任务要求、标准话术、隐藏心理、评分规则切片；
- 通用资料按普通文本兜底切片。

切片规则配置放在 config/training_ingest.yml，避免业务关键词散落在代码里。
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import yaml
from langchain_core.documents import Document

from core.rag.file_processors import FileProcessorFactory
from core.utils.logger_handler import logger
from core.utils.path_tool import get_abs_path


TRAINING_INGEST_CONFIG_PATH = get_abs_path("config/training_ingest.yml")


def _load_training_ingest_config() -> dict[str, Any]:
    """读取销售训练入库配置。

    配置文件是 LMS 切片规则的唯一来源，读取失败时直接抛出明确错误。
    """

    try:
        with open(TRAINING_INGEST_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except OSError as exc:
        logger.error("[销售训练] 读取入库配置文件失败 配置路径=%s 错误=%s", TRAINING_INGEST_CONFIG_PATH, exc, exc_info=True)
        raise RuntimeError(f"销售训练入库配置文件读取失败：{TRAINING_INGEST_CONFIG_PATH}") from exc
    except yaml.YAMLError as exc:
        logger.error("[销售训练] 解析入库配置文件失败 配置路径=%s 错误=%s", TRAINING_INGEST_CONFIG_PATH, exc, exc_info=True)
        raise RuntimeError(f"销售训练入库配置文件解析失败：{TRAINING_INGEST_CONFIG_PATH}") from exc
    if not isinstance(data, dict):
        logger.error("[销售训练] 入库配置文件根节点必须是字典 配置路径=%s", TRAINING_INGEST_CONFIG_PATH)
        raise ValueError(f"销售训练入库配置文件根节点必须是字典：{TRAINING_INGEST_CONFIG_PATH}")
    return data


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

    REQUIRED_PARTS = ("case_profile", "task_requirement", "standard_answer", "hidden_psychology", "scoring_rubric")
    ALLOWED_VISIBILITY = ("visible", "hidden", "scoring_only")

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化 LMS 切片策略，并从配置文件读取标题关键词。

        关键词放在 config/training_ingest.yml，避免业务词散落在代码里。
        """

        ingest_config = config or _load_training_ingest_config()
        lms_config = self._require_lms_config(ingest_config)
        pattern = self._require_case_title_pattern(lms_config)
        try:
            self.case_title_pattern = re.compile(pattern)
        except re.error as exc:
            logger.error("[销售训练] LMS案例标题正则配置错误 配置项=lms_case.case_title_pattern 错误=%s", exc, exc_info=True)
            raise ValueError("销售训练入库配置错误：lms_case.case_title_pattern 不是合法正则") from exc
        self.part_markers = self._normalize_part_markers(lms_config.get("part_markers"))
        self.part_visibility = self._normalize_part_visibility(lms_config.get("part_visibility"))

    def parse_chunks(self, file_path: str, context: dict[str, Any]) -> list[TrainingChunk]:
        """按 LMS 任务案例结构切片。

        这一步只负责“结构化拆分”，不直接调用 embedding。
        向量写入由 SalesTrainingService 统一处理。
        """

        # 第一步：先用通用文件处理器读取 txt/pdf/docx，输出 LangChain Document。
        # 这里不关心具体文件格式，格式差异交给 FileProcessorFactory。
        documents = FileProcessorFactory.load_documents(file_path)
        # FileProcessorFactory 返回 LangChain Document；这里再抽出段落列表。
        paragraphs = self._paragraphs_from_documents(documents)
        # 第二步：按中文编号标题拆出多个客户案例。
        # 例如“一、客户画像A”“二、客户画像B”会变成两个 case。
        cases = self._split_cases(paragraphs)
        chunks: list[TrainingChunk] = []

        # enumerate(..., start=1) 会从 1 开始计数，更符合业务展示习惯。
        for case_index, case in enumerate(cases, start=1):
            case_title = case["title"] or f"训练案例 {case_index}"
            # 第三步：在单个案例内部，按任务要求、话术、隐性心理、评分标准等标题拆段。
            parts = self._split_parts(case["lines"])
            for case_part, lines in parts.items():
                text = self._text_from_blocks(lines).strip()
                if not text:
                    continue
                # 第四步：不同业务片段分配不同 visibility，后续生成角色、对话、评分会按用途过滤。
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
                            **self._block_metadata(lines),
                        },
                    )
                )

        if chunks:
            return chunks

        # 兜底：如果文档结构完全不符合预期，至少保证能入库并被检索。
        # 这种兜底切片效果不如结构化 LMS 文档，文档里会提醒用户检查源文件结构。
        fallback_text = self._text_from_blocks(paragraphs).strip()
        return [
            TrainingChunk(
                chunk_id=f"{context['batch_id']}_001_case_profile",
                text=fallback_text,
                case_part="case_profile",
                visibility=context.get("visibility_default") or "visible",
                metadata={
                    "case_title": "未识别结构的 LMS 文档",
                    "source_file": context.get("source_file"),
                    **self._block_metadata(paragraphs),
                },
            )
        ] if fallback_text else []

    @staticmethod
    def _paragraphs_from_documents(documents: list[Document]) -> list[dict[str, Any]]:
        """优先使用 DOCX 处理器保留的段落，否则按换行拆分。

        DOCX 处理器会把原始段落放到 metadata["paragraphs"]。
        PDF/TXT 没有这个结构时，就用 page_content.splitlines() 兜底。
        """

        paragraphs: list[dict[str, Any]] = []
        for document in documents:
            raw_paragraphs = document.metadata.get("structured_blocks") or document.metadata.get("paragraphs")
            if isinstance(raw_paragraphs, list):
                for item in raw_paragraphs:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or "").strip()
                    if text:
                        paragraphs.append({**item, "text": text})
            else:
                page_no = document.metadata.get("page_no") or document.metadata.get("page")
                for index, line in enumerate(document.page_content.splitlines(), start=1):
                    text = line.strip()
                    if text:
                        paragraphs.append(
                            {
                                "block_index": index,
                                "block_type": "paragraph",
                                "page_no": page_no,
                                "text": text,
                            }
                        )
        return paragraphs

    def _split_cases(self, paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按“一、二、三”这类标题拆成训练案例。

        返回结构示例：
            [{"title": "一、xxx", "lines": ["..."]}]
        """

        cases: list[dict[str, Any]] = []
        current_title = ""
        current_lines: list[dict[str, Any]] = []

        for paragraph in paragraphs:
            text = str(paragraph.get("text") or "").strip()
            if self.case_title_pattern.match(text):
                if current_lines:
                    cases.append({"title": current_title, "lines": current_lines})
                current_title = text
                current_lines = []
                continue
            current_lines.append(paragraph)

        if current_lines:
            cases.append({"title": current_title, "lines": current_lines})
        return cases

    def _split_parts(self, lines: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """把单个案例拆成业务段落。

        current_part 表示“当前正在收集哪个片段”。
        遇到新的标题关键词后，后续行会进入新的片段。
        """

        parts: dict[str, list[dict[str, Any]]] = {
            "case_profile": [],
            "task_requirement": [],
            "standard_answer": [],
            "hidden_psychology": [],
            "scoring_rubric": [],
        }
        current_part = "case_profile"

        for line in lines:
            text = str(line.get("text") or "").strip()
            matched_part = self._detect_part(text)
            if matched_part:
                current_part = matched_part
                # “任务要求”这种纯标题只负责切换片段；“企业：某外贸公司”这种带内容的标签行要保留下来。
                if not self._is_part_heading_only(text, matched_part):
                    parts[current_part].append(line)
                continue
            parts[current_part].append(line)

        return parts

    def _detect_part(self, line: str) -> str | None:
        """根据标题关键词判断当前段落属于哪类业务片段。

        返回 None 表示这一行不是标题，而是正文。
        """

        clean_line = line.strip()
        for part, markers in self.part_markers:
            if any(marker in clean_line for marker in markers):
                return part
        return None

    def _is_part_heading_only(self, line: str, case_part: str) -> bool:
        """判断命中的片段标记是不是纯标题行。

        纯标题行示例：任务要求、匹配答案、命中点。
        带内容行示例：企业：某外贸公司、客户阶段：已初步沟通。
        带内容行必须保留到正文里，否则客户画像字段会在切分时丢失。
        """

        clean_line = line.strip()
        for part, markers in self.part_markers:
            if part != case_part:
                continue
            for marker in markers:
                if marker not in clean_line:
                    continue
                remainder = clean_line.replace(marker, "", 1).strip()
                remainder = re.sub(r"^[：:、.．\-\s]+", "", remainder).strip()
                return not bool(remainder)
        return False

    def _visibility_for_part(self, case_part: str) -> str:
        """根据片段类型决定可见范围。

        hidden_psychology 只给 AI 客户使用；
        scoring_rubric 只给评分模型使用；
        其他内容默认学员也可见。
        """

        return self.part_visibility.get(case_part, "visible")

    @staticmethod
    def _text_from_blocks(blocks: list[dict[str, Any]]) -> str:
        """把结构化 block 合并成切片正文。"""

        return "\n".join(str(block.get("text") or "").strip() for block in blocks if str(block.get("text") or "").strip())

    @staticmethod
    def _block_metadata(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        """从切片 block 中提取可追踪的结构化来源信息。"""

        if not blocks:
            return {}
        block_indexes = [block.get("block_index") for block in blocks if block.get("block_index") is not None]
        page_numbers = sorted({
            int(block.get("page_no"))
            for block in blocks
            if isinstance(block.get("page_no"), int)
        })
        heading_levels = sorted({
            int(block.get("heading_level"))
            for block in blocks
            if isinstance(block.get("heading_level"), int)
        })
        outline_titles = []
        for block in blocks:
            title = str(block.get("outline_title") or "").strip()
            if title and title not in outline_titles:
                outline_titles.append(title)

        metadata: dict[str, Any] = {"structure_source": "structured_blocks"}
        if block_indexes:
            metadata["start_block_index"] = min(block_indexes)
            metadata["end_block_index"] = max(block_indexes)
        if page_numbers:
            metadata["page_numbers"] = page_numbers
        if heading_levels:
            metadata["heading_levels"] = heading_levels
        if outline_titles:
            metadata["outline_titles"] = outline_titles[:5]
        return metadata

    @classmethod
    def _raise_config_error(cls, message: str) -> None:
        """记录 LMS 入库配置错误并中断策略初始化。"""

        logger.error("[销售训练] LMS入库配置错误 %s", message)
        raise ValueError(f"销售训练入库配置错误：{message}")

    @classmethod
    def _require_lms_config(cls, ingest_config: dict[str, Any]) -> dict[str, Any]:
        """读取并校验 LMS 案例入库配置节点。"""

        lms_config = ingest_config.get("lms_case")
        if not isinstance(lms_config, dict):
            cls._raise_config_error("缺少 lms_case 配置节点，或该节点不是字典")
        return lms_config

    @classmethod
    def _require_case_title_pattern(cls, lms_config: dict[str, Any]) -> str:
        """读取并校验 LMS 案例标题识别正则。"""

        pattern = lms_config.get("case_title_pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            cls._raise_config_error("缺少 lms_case.case_title_pattern 配置，或配置值为空")
        return pattern.strip()

    @classmethod
    def _normalize_part_markers(cls, raw_markers: Any) -> list[tuple[str, tuple[str, ...]]]:
        """把配置里的关键词字典转换成策略内部使用的有序元组列表。"""

        if not isinstance(raw_markers, dict):
            cls._raise_config_error("缺少 lms_case.part_markers 配置，或该节点不是字典")
        normalized: list[tuple[str, tuple[str, ...]]] = []
        for case_part in cls.REQUIRED_PARTS:
            markers = raw_markers.get(case_part)
            if not isinstance(markers, list):
                cls._raise_config_error(f"lms_case.part_markers.{case_part} 必须配置为列表")
            clean_markers = tuple(str(marker).strip() for marker in markers if str(marker).strip())
            if not clean_markers:
                cls._raise_config_error(f"lms_case.part_markers.{case_part} 至少要配置一个非空关键词")
            if clean_markers:
                normalized.append((case_part, clean_markers))
        return normalized

    @classmethod
    def _normalize_part_visibility(cls, raw_visibility: Any) -> dict[str, str]:
        """读取不同切片类型的默认可见性配置。"""

        if not isinstance(raw_visibility, dict):
            cls._raise_config_error("缺少 lms_case.part_visibility 配置，或该节点不是字典")
        visibility: dict[str, str] = {}
        for case_part in cls.REQUIRED_PARTS:
            clean_value = str(raw_visibility.get(case_part) or "").strip()
            if not clean_value:
                cls._raise_config_error(f"缺少 lms_case.part_visibility.{case_part} 配置，或配置值为空")
            if clean_value not in cls.ALLOWED_VISIBILITY:
                cls._raise_config_error(
                    f"lms_case.part_visibility.{case_part} 只能是 visible、hidden 或 scoring_only"
                )
            visibility[case_part] = clean_value
        return visibility


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
