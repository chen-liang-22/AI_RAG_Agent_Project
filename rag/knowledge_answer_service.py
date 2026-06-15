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

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger


class KnowledgeAnswerService:
    """基于 RAG 上下文直接生成最终客服回答。"""

    def __init__(self):
        self.rag = RagSummarizeService()

    def answer(self, query: str, *, history: list[dict] | None = None) -> str:
        """一次性生成知识库问答结果。"""

        trace_id = self._new_trace_id()
        total_start_time = time.perf_counter()
        context = self._retrieve_context(query, history=history, trace_id=trace_id)
        logger.info(
            "[知识直答] 非流式回答开始 追踪编号=%s 问题=%s 参考资料字符数=%s",
            trace_id,
            query,
            len(context),
        )
        llm_start_time = time.perf_counter()
        response = chat_model.invoke(self._build_messages(query, context, history=history))
        logger.info(
            "[性能] 最终模型调用完成 追踪编号=%s 模式=非流式 耗时毫秒=%.2f",
            trace_id,
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

    def stream_answer(self, query: str, *, history: list[dict] | None = None) -> Iterator[str]:
        """流式生成知识库问答结果。"""

        trace_id = self._new_trace_id()
        total_start_time = time.perf_counter()
        context = self._retrieve_context(query, history=history, trace_id=trace_id)
        logger.info(
            "[知识直答] 流式回答开始 追踪编号=%s 问题=%s 参考资料字符数=%s",
            trace_id,
            query,
            len(context),
        )
        total_chars = 0
        chunk_count = 0
        llm_start_time = time.perf_counter()
        first_chunk_time: float | None = None
        for chunk in chat_model.stream(self._build_messages(query, context, history=history)):
            content = self._message_content_to_text(chunk.content)
            if not content:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                logger.info(
                    "[性能] 首个回答分片到达 追踪编号=%s 耗时毫秒=%.2f 上下文字符数=%s",
                    trace_id,
                    self._elapsed_ms(llm_start_time),
                    len(context),
                )
            total_chars += len(content)
            chunk_count += 1
            yield content
        logger.info(
            "[性能] 流式模型输出完成 追踪编号=%s 耗时毫秒=%.2f 分片数=%s 回答字符数=%s",
            trace_id,
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
    ) -> str:
        """统一走普通多意图 RAG 检索，不再做 FAQ 编号/列表特殊分支。"""

        start_time = time.perf_counter()
        context = self.rag.rag_summarize(query, history=history, trace_id=trace_id)
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
        """把 SQLite 会话历史转换成 LangChain 消息。"""

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
            "你是一名资深的中文客服专家，熟悉扫地机器人、扫拖一体机器人、基站和配套 APP 的选购、使用、维护与故障处理。"
            "回答时要像专业人士直接给用户建议：先给结论，再按用户关心点分条说明，必要时给出可执行的操作步骤或选购优先级。"
            "用户一句话里可能包含多个诉求，请按用户原问题的顺序逐个覆盖，不要漏答，也不要把多个问题混成一句空泛建议。"
            "如果信息不足以确定结论，要直接说明不确定的点，并给出最稳妥的判断方式；不要编造不存在的功能、路径、价格或型号。"
            "对于大众都知道、几乎不依赖具体型号资料的基础常识问题，可以直接作答，例如没电了不能继续工作、尘盒满了需要清理、断网后远程控制可能不可用。"
            "不要把经验性建议、复杂维护方案、选购判断或故障诊断都当成常识；这些问题仍要基于已有信息谨慎回答。"
            "如果问题涉及具体品牌型号、APP路径、售后政策、价格、参数或故障代码，而现有信息不足，就不要当成常识编造答案。"
            "如果已有信息足够回答，就不要泛泛建议用户再去查说明书或联系客服。"
            "表达要自然、简洁、专业，避免套话；可以使用小标题或列表，让用户一眼能看懂怎么选、怎么做。"
            "不要输出内部检索流程、工具名、向量库名，也不要说“根据参考资料”“资料显示”这类暴露内部机制的话。"
            "不要输出任何资料编号、来源编号或引用标记，例如“【参考资料1】”“参考资料2”“来源文件”等。"
            "除非用户明确询问来源，否则不要机械展示来源文件、知识类型、分类等字段。"
        )

    @staticmethod
    def _user_message(query: str, context: str) -> str:
        return (
            f"用户问题：{query}\n\n"
            f"可用信息：\n{context}\n\n"
            "请结合可用信息生成最终回答。回答时不要暴露“可用信息”、资料编号、来源编号或内部检索痕迹。"
        )

    @staticmethod
    def _new_trace_id() -> str:
        return f"chat_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
