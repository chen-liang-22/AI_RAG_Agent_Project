"""知识库直答服务。

这条链路专门处理普通产品知识问答：
用户问题 -> Query Planner -> Qdrant 检索/精排 -> 大模型最终回答。

它不绑定 Agent 工具，也不会让模型先判断“要不要调用 rag_summarize”，
因此适合扫拖机器人说明、APP 操作、保养、故障处理等知识类问题。
"""

import time
import uuid
from collections.abc import Iterator

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from core.model.factory import get_chat_model, get_chat_model_name_for_mode, normalize_chat_model_mode
from core.rag.services.rag_service import RagSummarizeService
from core.utils.logger_handler import logger


class KnowledgeAnswerService:
    """基于 RAG 上下文直接生成最终客服回答。"""

    def __init__(self):
        self.rag = RagSummarizeService()

    def answer(
            self,
            query: str,
            *,
            history: list[dict] | None = None,
            model_mode: str | None = None,
            collection_name: str | None = None,
    ) -> str:
        """一次性生成知识库问答结果。"""

        trace_id = self._new_trace_id()
        total_start_time = time.perf_counter()
        selected_model = get_chat_model(model_mode)
        selected_model_mode = normalize_chat_model_mode(model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        context = self._retrieve_context(
            query,
            history=history,
            trace_id=trace_id,
            collection_name=collection_name,
        )
        logger.info(
            "[知识直答] 非流式回答开始 追踪编号=%s 问题=%s 参考资料字符数=%s 模型模式=%s 模型名称=%s",
            trace_id,
            query,
            len(context),
            selected_model_mode,
            selected_model_name,
        )
        llm_start_time = time.perf_counter()
        response = selected_model.invoke(self._build_messages(query, context, history=history))
        logger.info(
            "[性能] 最终模型调用完成 追踪编号=%s 模式=非流式 模型模式=%s 模型名称=%s 耗时毫秒=%.2f",
            trace_id,
            selected_model_mode,
            selected_model_name,
            self._elapsed_ms(llm_start_time),
        )
        answer = self._message_content_to_text(response.content).strip()
        logger.info(
            "[性能] 知识直答总耗时 追踪编号=%s 模式=非流式 耗时毫秒=%.2f 回答字符数=%s",
            trace_id,
            self._elapsed_ms(total_start_time),
            len(answer),
        )
        logger.info("[知识直答] 非流式回答完成 追踪编号=%s 问题=%s 回答字符数=%s", trace_id, query, len(answer))
        return answer

    def stream_answer(
            self,
            query: str,
            *,
            history: list[dict] | None = None,
            model_mode: str | None = None,
            collection_name: str | None = None,
    ) -> Iterator[str]:
        """流式生成知识库问答结果。"""

        trace_id = self._new_trace_id()
        total_start_time = time.perf_counter()
        selected_model = get_chat_model(model_mode)
        selected_model_mode = normalize_chat_model_mode(model_mode)
        selected_model_name = get_chat_model_name_for_mode(selected_model_mode)
        context = self._retrieve_context(
            query,
            history=history,
            trace_id=trace_id,
            collection_name=collection_name,
        )
        logger.info(
            "[知识直答] 流式回答开始 追踪编号=%s 问题=%s 参考资料字符数=%s 模型模式=%s 模型名称=%s",
            trace_id,
            query,
            len(context),
            selected_model_mode,
            selected_model_name,
        )
        total_chars = 0
        chunk_count = 0
        llm_start_time = time.perf_counter()
        first_chunk_time: float | None = None
        for chunk in selected_model.stream(self._build_messages(query, context, history=history)):
            content = self._message_content_to_text(chunk.content)
            if not content:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                logger.info(
                    "[性能] 首个回答分片到达 追踪编号=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 上下文字符数=%s",
                    trace_id,
                    selected_model_mode,
                    selected_model_name,
                    self._elapsed_ms(llm_start_time),
                    len(context),
                )
            total_chars += len(content)
            chunk_count += 1
            yield content
        logger.info(
            "[性能] 流式模型输出完成 追踪编号=%s 模型模式=%s 模型名称=%s 耗时毫秒=%.2f 分片数=%s 回答字符数=%s",
            trace_id,
            selected_model_mode,
            selected_model_name,
            self._elapsed_ms(llm_start_time),
            chunk_count,
            total_chars,
        )
        logger.info(
            "[性能] 知识直答总耗时 追踪编号=%s 模式=流式 耗时毫秒=%.2f",
            trace_id,
            self._elapsed_ms(total_start_time),
        )
        logger.info("[知识直答] 流式回答完成 追踪编号=%s 问题=%s 回答字符数=%s", trace_id, query, total_chars)

    def _retrieve_context(
            self,
            query: str,
            *,
            history: list[dict] | None = None,
            trace_id: str | None = None,
            collection_name: str | None = None,
    ) -> str:
        """统一走普通多意图 RAG 检索，不再做 FAQ 编号/列表特殊分支。"""

        start_time = time.perf_counter()
        context = self.rag.rag_summarize(
            query,
            history=history,
            trace_id=trace_id,
            collection_name=collection_name,
        )
        logger.info(
            "[性能] 知识直答上下文准备完成 追踪编号=%s 耗时毫秒=%.2f 上下文字符数=%s",
            trace_id,
            self._elapsed_ms(start_time),
            len(context),
        )
        logger.info("[知识直答] RAG 检索完成 追踪编号=%s 问题=%s 参考资料字符数=%s", trace_id, query, len(context))
        return context

    def _build_messages(
            self,
            query: str,
            context: str,
            *,
            history: list[dict] | None = None,
    ) -> list[AnyMessage]:
        return [
            SystemMessage(content=self._system_prompt()),
            *self._history_to_messages(history or []),
            HumanMessage(content=self._user_message(query, context)),
        ]

    @staticmethod
    def _history_to_messages(history: list[dict]) -> list[AnyMessage]:
        """把数据库会话历史转换成 LangChain 消息。"""

        messages: list[AnyMessage] = []
        for item in history[-12:]:
            role = item.get("role")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        return messages

    @staticmethod
    def _message_content_to_text(content) -> str:
        """兼容不同模型返回的字符串或结构化 content。"""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
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
            "你是扫地/扫拖机器人客服,叫阿良。"
            "优先根据参考资料回答。"
            "如果问题属于基础常识，可以直接简洁回答。"
            "不要编造具体品牌型号、价格、参数、APP路径、售后政策或故障代码。"
            "先给结论，再给步骤。"
            "不要输出参考资料原文、编号和元数据。"
            "资料不足且不是基础常识时，直接说明缺少依据。"
        )

    @staticmethod
    def _user_message(query: str, context: str) -> str:
        return (
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context}\n\n"
            "请生成最终回答。"
        )

    @staticmethod
    def _new_trace_id() -> str:
        return f"chat_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
