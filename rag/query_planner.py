import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from model.factory import chat_model
from rag.query_pipeline import RuleBasedIntentAnalyzer
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class QueryPlannerModelError(RuntimeError):
    """Raised when the LLM planner call fails."""


class QueryPlannerParseError(ValueError):
    """Raised when the LLM planner output is not valid JSON."""


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
            trace_id: str | None = None,
    ) -> list[str]:
        start_time = time.perf_counter()
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(
            "[查询规划] 开始 追踪编号=%s 原问题=%s 历史消息数=%s",
            trace_id,
            clean_query,
            len(history or []),
        )
        explicit_queries = self._split_explicit_questions(clean_query)
        if len(explicit_queries) >= 2:
            planned_queries = self._normalize_queries(explicit_queries, original_query=clean_query)
            logger.info(
                "[性能] 查询规划完成 追踪编号=%s 模式=显式切分 耗时毫秒=%.2f 检索问题数=%s",
                trace_id,
                self._elapsed_ms(start_time),
                len(planned_queries),
            )
            logger.info("[查询规划] 显式切分结果=%s", planned_queries)
            self._log_split_questions("explicit", planned_queries)
            return planned_queries

        planner_mode = str(rag_conf.get("query_planner_mode") or "llm").strip().lower()
        if planner_mode in {"rule", "rules", "off", "disabled"}:
            fallback_queries = self._fallback_queries(clean_query)
            logger.info(
                "[性能] 查询规划完成 追踪编号=%s 模式=规则 耗时毫秒=%.2f 检索问题数=%s",
                trace_id,
                self._elapsed_ms(start_time),
                len(fallback_queries),
            )
            logger.info("[查询规划] 配置为规则模式，结果=%s", fallback_queries)
            self._log_split_questions("fallback", fallback_queries)
            return fallback_queries

        try:
            queries = self._plan_with_llm(clean_query, history=history or [], trace_id=trace_id)
            if queries:
                logger.info(
                    "[性能] 查询规划完成 追踪编号=%s 模式=模型 耗时毫秒=%.2f 检索问题数=%s",
                    trace_id,
                    self._elapsed_ms(start_time),
                    len(queries),
                )
                logger.info("[查询规划] 模型拆分结果=%s", queries)
                self._log_split_questions("llm", queries)
                return queries
        except (QueryPlannerModelError, QueryPlannerParseError) as exc:
            logger.warning("[查询规划] 模型拆分失败，改用规则兜底：%s", exc)

        fallback_queries = self._fallback_queries(clean_query)
        logger.info(
            "[性能] 查询规划完成 追踪编号=%s 模式=兜底 耗时毫秒=%.2f 检索问题数=%s",
            trace_id,
            self._elapsed_ms(start_time),
            len(fallback_queries),
        )
        logger.info("[查询规划] 规则兜底结果=%s", fallback_queries)
        self._log_split_questions("fallback", fallback_queries)
        return fallback_queries

    def plan_initial(self, query: str, *, trace_id: str | None = None) -> list[str]:
        """adaptive 模式的首轮查询：只保留原问题，不调用模型或规则扩展。"""

        clean_query = self._clean_query(query)
        if not clean_query:
            return []
        logger.info("[查询规划] adaptive首轮使用原问题 追踪编号=%s 检索问题=%s", trace_id, clean_query)
        return [clean_query]

    def plan_with_config(
            self,
            query: str,
            *,
            history: list[dict[str, Any]] | None = None,
            trace_id: str | None = None,
    ) -> list[str]:
        """按配置执行完整查询规划，用于 adaptive 首轮召回质量不足后的补救。"""

        return self.plan(query, history=history, trace_id=trace_id)

    def merge_queries(self, primary_queries: list[str], secondary_queries: list[str]) -> list[str]:
        """合并两批检索问题，保留顺序并去重。"""

        return self._normalize_queries([*primary_queries, *secondary_queries], original_query="")

    def _plan_with_llm(
            self,
            query: str,
            *,
            history: list[dict[str, Any]],
            trace_id: str | None = None,
    ) -> list[str]:
        history_text = self._format_history(history)
        start_time = time.perf_counter()
        try:
            response = chat_model.invoke(
                [
                    SystemMessage(content=self._system_prompt()),
                    HumanMessage(content=self._user_prompt(query, history_text)),
                ]
            )
        except (ConnectionError, TimeoutError, RuntimeError, ValueError) as exc:
            raise QueryPlannerModelError(str(exc)) from exc

        logger.info(
            "[性能] 查询规划模型调用完成 追踪编号=%s 耗时毫秒=%.2f",
            trace_id,
            self._elapsed_ms(start_time),
        )
        content = self._message_content_to_text(response.content)
        logger.info("[查询规划] 模型原始输出=%s", content[:1200])
        try:
            data = self._parse_json_object(content)
        except json.JSONDecodeError as exc:
            raise QueryPlannerParseError(f"invalid planner JSON: {content[:200]}") from exc

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
    def _log_split_questions(source: str, queries: list[str]) -> None:
        source_text = {
            "explicit": "显式切分",
            "llm": "模型拆分",
            "fallback": "规则兜底",
        }.get(source, source)
        total = len(queries)
        for index, query in enumerate(queries, start=1):
            logger.info(
                "[查询规划] 拆分问题 来源=%s 序号=%s/%s 问题=%s",
                source_text,
                index,
                total,
                query,
            )

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

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
