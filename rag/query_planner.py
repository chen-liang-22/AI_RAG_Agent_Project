import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.factory import chat_model
from rag.query_pipeline import RuleBasedIntentAnalyzer
from utils.logger_handler import logger


class QueryPlannerService:
    """LLM Query Planner。

    输入用户问题，输出适合分别检索 Qdrant 的 search_query 列表。
    模型不可用或 JSON 解析失败时，使用规则拆分兜底。
    """

    def __init__(self, *, max_queries: int = 6):
        self.max_queries = max_queries
        self.fallback_analyzer = RuleBasedIntentAnalyzer()

    def plan(
            self,
            query: str,
            *,
            history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(
            "[query planner] start query=%s history_count=%s",
            clean_query,
            len(history or []),
        )
        explicit_queries = self._split_explicit_questions(clean_query)
        if len(explicit_queries) >= 2:
            planned_queries = self._normalize_queries(explicit_queries, original_query=clean_query)
            logger.info("[query planner] explicit_split_queries=%s", planned_queries)
            return planned_queries

        try:
            queries = self._plan_with_llm(clean_query, history=history or [])
            if queries:
                logger.info("[query planner] llm_queries=%s", queries)
                return queries
        except Exception as exc:
            logger.warning(f"[query planner] llm planner failed, fallback to rules: {exc}")

        fallback_queries = self._fallback_queries(clean_query)
        logger.info("[query planner] fallback_queries=%s", fallback_queries)
        return fallback_queries

    def _plan_with_llm(self, query: str, *, history: list[dict[str, Any]]) -> list[str]:
        history_text = self._format_history(history)
        response = chat_model.invoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=self._user_prompt(query, history_text)),
            ]
        )
        content = self._message_content_to_text(response.content)
        logger.info("[query planner] raw_output=%s", content[:1200])
        data = self._parse_json_object(content)
        raw_queries = data.get("queries") if isinstance(data, dict) else None
        if not isinstance(raw_queries, list):
            return []

        return self._normalize_queries([str(item) for item in raw_queries], original_query=query)

    def _fallback_queries(self, query: str) -> list[str]:
        parts = self._split_explicit_questions(query)
        if len(parts) <= 1:
            analysis = self.fallback_analyzer.analyze(query)
            parts = analysis.sub_queries
        return self._normalize_queries(parts, original_query=query)

    def _normalize_queries(self, queries: list[str], *, original_query: str) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        source_queries = [original_query, *queries] if len(queries) <= 1 else queries

        for value in source_queries:
            clean_value = self._clean_query(value)
            if not clean_value:
                continue
            key = self._query_key(clean_value)
            if key in seen:
                continue
            seen.add(key)
            result.append(clean_value)
            if len(result) >= self.max_queries:
                break

        return result

    @staticmethod
    def _clean_query(value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n，。！？?；;、")
        return value[:160]

    @classmethod
    def _split_explicit_questions(cls, query: str) -> list[str]:
        return [
            cls._clean_query(part)
            for part in re.split(r"[？?；;\n\r]+", query)
            if cls._clean_query(part)
        ]

    @staticmethod
    def _query_key(value: str) -> str:
        return re.sub(r"[\s，。！？?；;、,.]+", "", value.lower())

    @staticmethod
    def _format_history(history: list[dict[str, Any]], limit: int = 6) -> str:
        recent_history = history[-limit:]
        lines = []
        for message in recent_history:
            role = message.get("role") or "unknown"
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content[:300]}")
        return "\n".join(lines) or "无"

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        clean_content = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", clean_content, flags=re.S)
        if fenced_match:
            clean_content = fenced_match.group(1)
        else:
            object_match = re.search(r"\{.*\}", clean_content, flags=re.S)
            if object_match:
                clean_content = object_match.group(0)

        return json.loads(clean_content)

    @staticmethod
    def _message_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是 RAG 检索 query planner。"
            "你的任务是把用户问题拆成适合向量检索的 search_query。"
            "只输出 JSON，不要输出解释。"
            "JSON 格式必须是：{\"queries\":[\"...\"]}。"
            "如果用户一次问多个问题，要拆成多个 query。"
            "如果用户只有一个问题，输出 1 个 query。"
            "不要编造知识答案，不要输出分类字段。"
        )

    @staticmethod
    def _user_prompt(query: str, history_text: str) -> str:
        return (
            f"最近会话历史：\n{history_text}\n\n"
            f"用户当前问题：\n{query}\n\n"
            "请输出 JSON。"
        )
