import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.model.factory import chat_model
from core.utils.config_handler import rag_conf
from core.utils.logger_handler import logger


class QueryPlannerModelError(RuntimeError):
    """查询规划模型调用失败。"""


class QueryPlannerParseError(ValueError):
    """查询规划模型输出不是合法 JSON。"""


class QueryPlannerService:
    """LLM Query Planner。

    输入用户问题，输出适合分别检索 Qdrant 的 search_query 列表。
    模型不可用或 JSON 解析失败时，只使用原问题或显式多问句切分兜底。
    """

    def __init__(self, *, max_queries: int = 6):
        self.max_queries = max_queries

    def plan(
            self,
            query: str,
            *,
            history: list[dict[str, Any]] | None = None,
            trace_id: str | None = None,
    ) -> list[str]:
        """按当前配置生成检索问题列表。

        返回值不是最终回答，而是给 Qdrant 向量检索使用的 search_query 列表。
        这个方法只在需要完整查询规划时调用：
        - `llm` 模式：每次都尝试调用模型拆分/改写。
        - `adaptive` 模式：首轮召回质量不足后，才调用这里做补救。
        - `off/rule` 等非模型模式：不做硬编码规则拆分，只用原问题兜底。
        """

        # 记录开始时间，用于后面打印查询规划耗时。
        start_time = time.perf_counter()

        # 去掉用户问题前后的空白，避免空字符串进入 LLM 或向量检索。
        clean_query = query.strip()
        if not clean_query:
            return []

        # 打印入口日志，trace_id 用来把一次聊天里的检索、规划、模型调用串起来看。
        logger.info(
            "[查询规划] 开始 追踪编号=%s 原问题=%s 历史消息数=%s",
            trace_id,
            clean_query,
            len(history or []),
        )

        # 第一优先级：如果用户自己已经用问号、分号、换行写了多个问题，直接按显式分隔符切。
        # 这个分支不调用 LLM，所以速度最快，也不会引入模型改写的不确定性。
        explicit_queries = self._split_explicit_questions(clean_query)
        if len(explicit_queries) >= 2:
            # 清洗、去重、截断，最多保留 self.max_queries 个检索问题。
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

        # 读取配置里的 Query Planner 模式。
        # adaptive 模式不会直接在这里首轮调用模型，而是在 rag_service 里先查一次再决定是否进入这里。
        planner_mode = str(rag_conf.get("query_planner_mode") or "llm").strip().lower()

        # 非模型模式：保留兼容配置名 rule/rules，但这里不会再做规则关键词硬拆分。
        # 也就是说不会根据“宠物、地毯、充电”等词硬编码生成子问题，只返回原问题。
        if planner_mode in {"rule", "rules", "off", "disabled"}:
            fallback_queries = self._fallback_queries(clean_query)
            logger.info(
                "[性能] 查询规划完成 追踪编号=%s 模式=原问题兜底 耗时毫秒=%.2f 检索问题数=%s",
                trace_id,
                self._elapsed_ms(start_time),
                len(fallback_queries),
            )
            logger.info("[查询规划] 配置为非模型模式，不做规则硬拆分，结果=%s", fallback_queries)
            self._log_split_questions("fallback", fallback_queries)
            return fallback_queries

        # 模型模式：调用 LLM Query Planner，让模型把复杂问题改写成多个适合向量检索的 search_query。
        # 这里的模型只负责“拆检索问题”，不生成最终答案。
        try:
            queries = self._plan_with_llm(clean_query, history=history or [], trace_id=trace_id)
            if queries:
                # 模型返回后仍会经过清洗、去重、截断，所以这里拿到的是可直接检索的列表。
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
            # 只捕获模型调用失败和 JSON 解析失败。
            # 失败时不再使用规则硬匹配，避免自由输入场景下被固定词表带偏。
            logger.warning("[查询规划] 模型拆分失败，改用原问题兜底：%s", exc)

        # 最终兜底：模型没有返回有效 queries，或者模型调用/解析失败。
        # 此时只返回原问题，确保 RAG 链路仍然能继续往下走。
        fallback_queries = self._fallback_queries(clean_query)
        logger.info(
            "[性能] 查询规划完成 追踪编号=%s 模式=原问题兜底 耗时毫秒=%.2f 检索问题数=%s",
            trace_id,
            self._elapsed_ms(start_time),
            len(fallback_queries),
        )
        logger.info("[查询规划] 原问题兜底结果=%s", fallback_queries)
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
        history_limit = int(rag_conf.get("query_planner_history_limit", 20) or 20)
        history_chars = int(rag_conf.get("query_planner_history_chars", 300) or 300)
        history_text = self._format_history(history, limit=history_limit, chars_per_message=history_chars)
        logger.info(
            "[查询规划] 准备调用模型 追踪编号=%s 历史消息上限=%s 单条历史字符上限=%s 实际历史消息数=%s",
            trace_id,
            history_limit,
            history_chars,
            min(len(history), history_limit),
        )
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
            parts = [query]
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
            "fallback": "原问题兜底",
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
    def _format_history(history: list[dict[str, Any]], limit: int = 20, chars_per_message: int = 300) -> str:
        """格式化最近历史消息，帮助 Query Planner 理解追问上下文。"""

        recent_history = history[-limit:]
        lines = []
        for message in recent_history:
            role = message.get("role") or "unknown"
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content[:chars_per_message]}")
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
