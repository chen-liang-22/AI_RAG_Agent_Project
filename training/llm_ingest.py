import json
import re
from typing import Any

import yaml

from model.factory import get_chat_model
from training.strategies.knowledge_ingest_strategy import TrainingChunk
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


TRAINING_INGEST_CONFIG_PATH = get_abs_path("config/training_ingest.yml")


class TrainingLlmFallbackSplitter:
    """训练资料 LLM 兜底切分器。

    这个类只在规则切分质量较低时使用，不参与常规上传路径。
    它的职责是把一段原始资料抽取成统一的 TrainingChunk 列表，
    后续仍交给质量评估器再次打分，避免模型输出直接入库。
    """

    DEFAULT_CONFIG = {
        "enabled": False,
        "trigger_score_below": 60,
        "model_mode": None,
        "max_source_chars": 12000,
        "max_cases": 20,
        "max_chunks": 80,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """初始化 LLM 兜底切分器，并读取可调参数。"""

        merged_config = dict(self.DEFAULT_CONFIG)
        merged_config.update(config or self._load_config())
        self.config = merged_config

    def should_trigger(self, quality_report: dict[str, Any]) -> bool:
        """根据质量报告判断是否需要触发 LLM 兜底切分。"""

        if not bool(self.config.get("enabled")):
            return False
        score = int(quality_report.get("score") or 0)
        return score < int(self.config.get("trigger_score_below") or 0)

    def split(
            self,
            *,
            source_text: str,
            batch_id: str,
            source_file: str,
            source_type: str,
            visibility_default: str,
    ) -> list[TrainingChunk]:
        """调用 LLM 抽取训练切片。"""

        clean_text = source_text.strip()
        if not clean_text:
            return []

        max_source_chars = int(self.config.get("max_source_chars") or self.DEFAULT_CONFIG["max_source_chars"])
        prompt = self._build_prompt(
            source_text=clean_text[:max_source_chars],
            source_file=source_file,
            source_type=source_type,
        )
        model_mode = self.config.get("model_mode")
        logger.info(
            "[销售训练][资料切分] LLM兜底切分开始 批次编号=%s 文件名=%s 输入字符数=%s 模型档位=%s",
            batch_id,
            source_file,
            min(len(clean_text), max_source_chars),
            model_mode or "默认",
        )

        try:
            response = get_chat_model(model_mode).invoke(self._messages("请只输出 JSON。", prompt))
            text = self._content_text(response.content)
            payload = self._parse_json_object(text)
            chunks = self._chunks_from_payload(
                payload,
                batch_id=batch_id,
                source_file=source_file,
                source_type=source_type,
                visibility_default=visibility_default,
            )
            logger.info(
                "[销售训练][资料切分] LLM兜底切分完成 批次编号=%s 文件名=%s 切片数量=%s",
                batch_id,
                source_file,
                len(chunks),
            )
            return chunks
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError, TimeoutError, OSError) as exc:
            logger.warning(
                "[销售训练][资料切分] LLM兜底切分失败 批次编号=%s 文件名=%s 错误=%s",
                batch_id,
                source_file,
                exc,
            )
            return []

    def _build_prompt(self, *, source_text: str, source_file: str, source_type: str) -> str:
        """构造 LLM 切分提示词。"""

        max_cases = int(self.config.get("max_cases") or self.DEFAULT_CONFIG["max_cases"])
        max_chunks = int(self.config.get("max_chunks") or self.DEFAULT_CONFIG["max_chunks"])
        return f"""
你是销售训练资料入库助理。请把原始资料抽取成适合销售陪练检索的结构化切片。

要求：
1. 只根据原文抽取，不要补写原文没有的事实。
2. 保留原文里的客户背景、任务要求、参考话术、客户隐性心理、评分标准。
3. 每个案例最多输出 5 类片段：case_profile、task_requirement、standard_answer、hidden_psychology、scoring_rubric。
4. case_profile/task_requirement/standard_answer 默认可见；hidden_psychology 默认 hidden；scoring_rubric 默认 scoring_only。
5. 最多输出 {max_cases} 个案例、{max_chunks} 个切片。
6. 只输出 JSON，不要解释。

JSON 格式：
{{
  "cases": [
    {{
      "case_title": "案例标题",
      "case_index": 1,
      "parts": [
        {{
          "case_part": "case_profile",
          "visibility": "visible",
          "text": "切片正文"
        }}
      ]
    }}
  ]
}}

文件名：{source_file}
资料类型：{source_type}
原文：
{source_text}
""".strip()

    def _chunks_from_payload(
            self,
            payload: dict[str, Any],
            *,
            batch_id: str,
            source_file: str,
            source_type: str,
            visibility_default: str,
    ) -> list[TrainingChunk]:
        """把模型 JSON 转换成 TrainingChunk 列表。"""

        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("模型输出缺少 cases 数组")

        max_cases = int(self.config.get("max_cases") or self.DEFAULT_CONFIG["max_cases"])
        max_chunks = int(self.config.get("max_chunks") or self.DEFAULT_CONFIG["max_chunks"])
        chunks: list[TrainingChunk] = []

        for case_index, case in enumerate(cases[:max_cases], start=1):
            if not isinstance(case, dict):
                continue
            raw_case_index = case.get("case_index") or case_index
            normalized_case_index = self._safe_int(raw_case_index, case_index)
            case_title = str(case.get("case_title") or f"训练案例 {normalized_case_index}").strip()
            parts = case.get("parts")
            if not isinstance(parts, list):
                continue

            used_parts: dict[str, int] = {}
            for part in parts:
                if not isinstance(part, dict):
                    continue
                case_part = self._normalize_case_part(part.get("case_part"))
                text = str(part.get("text") or "").strip()
                if not text:
                    continue
                used_parts[case_part] = used_parts.get(case_part, 0) + 1
                part_suffix = case_part if used_parts[case_part] == 1 else f"{case_part}_{used_parts[case_part]}"
                chunk_id = f"{batch_id}_{normalized_case_index:03d}_{part_suffix}"
                chunks.append(
                    TrainingChunk(
                        chunk_id=chunk_id,
                        text=f"{case_title}\n{text}",
                        case_part=case_part,
                        visibility=self._normalize_visibility(part.get("visibility"), case_part, visibility_default),
                        metadata={
                            "case_title": case_title,
                            "case_index": normalized_case_index,
                            "source_file": source_file,
                            "source_type": source_type,
                            "splitter": "llm_fallback",
                        },
                    )
                )
                if len(chunks) >= max_chunks:
                    return chunks

        return chunks

    @staticmethod
    def _normalize_case_part(value: Any) -> str:
        """把模型返回的片段类型收敛到系统支持的枚举。"""

        allowed_parts = {
            "case_profile",
            "task_requirement",
            "standard_answer",
            "hidden_psychology",
            "scoring_rubric",
        }
        case_part = str(value or "").strip()
        return case_part if case_part in allowed_parts else "case_profile"

    @staticmethod
    def _normalize_visibility(value: Any, case_part: str, visibility_default: str) -> str:
        """根据片段类型兜底可见性。"""

        visibility = str(value or "").strip()
        if visibility in {"visible", "hidden", "scoring_only"}:
            return visibility
        if case_part == "hidden_psychology":
            return "hidden"
        if case_part == "scoring_rubric":
            return "scoring_only"
        return visibility_default or "visible"

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """把模型返回的序号转换成整数。"""

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _messages(system: str, human: str) -> list:
        """构造 LangChain 聊天消息。"""

        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=human)]

    @staticmethod
    def _content_text(content: Any) -> str:
        """把不同模型返回格式统一转换成字符串。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
        return str(content)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        """从模型输出里提取 JSON 对象。"""

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("模型没有输出 JSON 对象")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return payload

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """从配置文件读取 LLM 兜底切分参数。"""

        try:
            with open(TRAINING_INGEST_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError):
            return {}
        fallback_config = data.get("llm_fallback") if isinstance(data, dict) else {}
        return fallback_config if isinstance(fallback_config, dict) else {}
