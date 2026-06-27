import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from utils.logger_handler import logger


@dataclass(frozen=True)
class SemanticChunkPlan:
    """LLM 返回的语义切片计划。

    LLM 只负责给出原文范围和语义元数据，真正入库文本由后端按 start/end 从原文截取。
    """

    start: int
    end: int
    title: str | None = None
    content_type: str = "segment"
    question: str | None = None
    category: str | None = None
    reason: str | None = None


class LlmSemanticSplitter:
    """基于 LLM 的语义切片计划生成器。"""

    def __init__(self, *, model_mode: str | None = None, window_chars: int = 12000):
        self.model_mode = model_mode
        self.window_chars = window_chars

    def split(self, documents: list[Document]) -> list[SemanticChunkPlan]:
        """按全文生成语义切片计划。"""

        full_text = self.join_documents(documents)
        if not full_text.strip():
            return []

        from model.factory import get_chat_model, get_chat_model_name_for_mode

        model_name = get_chat_model_name_for_mode(self.model_mode)
        logger.info("[文档解析] LLM语义切片开始 模型名称=%s 文本字符数=%s", model_name, len(full_text))
        model = get_chat_model(self.model_mode)
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是知识库语义切片规划器。只能基于用户给出的原文做切片计划，"
                        "不要改写、总结或补充原文。必须只返回 JSON。"
                    )
                ),
                HumanMessage(
                    content=(
                        "请为下面原文生成语义切片计划。要求：\n"
                        "1. 只返回 JSON，不要 Markdown。\n"
                        "2. 每个 chunk 必须给出 start/end 字符偏移，偏移基于原文字符串。\n"
                        "3. content_type 只能是 segment 或 qa。\n"
                        "4. 如果是 qa，可给出 question/category；answer 不需要返回，后端会按原文截取。\n"
                        "5. 不要返回切片正文 content。\n"
                        "6. 覆盖核心内容，避免大量重复。\n\n"
                        "返回格式："
                        '{"chunks":[{"title":"主题","start":0,"end":120,"content_type":"segment","reason":"原因"}]}\n\n'
                        f"原文：\n{full_text[:self.window_chars]}"
                    )
                ),
            ]
        )
        chunks = self._parse_chunks(response.content)
        logger.info("[文档解析] LLM语义切片完成 模型名称=%s 切片计划数=%s", model_name, len(chunks))
        return chunks

    @staticmethod
    def join_documents(documents: list[Document]) -> str:
        """用稳定分隔符拼接文档，供 LLM 和后端 span 校验共用。"""

        return "\n\n".join(document.page_content or "" for document in documents)

    @classmethod
    def _parse_chunks(cls, content: object) -> list[SemanticChunkPlan]:
        """解析模型 JSON 输出。"""

        payload = cls._parse_model_json(content)
        raw_chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(raw_chunks, list):
            return []

        chunks: list[SemanticChunkPlan] = []
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict):
                continue
            try:
                start = int(raw_chunk.get("start"))
                end = int(raw_chunk.get("end"))
            except (TypeError, ValueError):
                continue
            chunks.append(
                SemanticChunkPlan(
                    start=start,
                    end=end,
                    title=str(raw_chunk.get("title") or "").strip() or None,
                    content_type=cls._normalize_content_type(raw_chunk.get("content_type")),
                    question=str(raw_chunk.get("question") or "").strip() or None,
                    category=str(raw_chunk.get("category") or "").strip() or None,
                    reason=str(raw_chunk.get("reason") or "").strip() or None,
                )
            )
        return chunks

    @staticmethod
    def _normalize_content_type(value: Any) -> str:
        """归一化 LLM 返回的切片类型。"""

        normalized = str(value or "segment").strip().lower()
        return "qa" if normalized == "qa" else "segment"

    @staticmethod
    def _parse_model_json(content: object) -> dict:
        """从模型返回中解析 JSON 对象。"""

        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if object_match:
                text = object_match.group(0)
        return json.loads(text)
