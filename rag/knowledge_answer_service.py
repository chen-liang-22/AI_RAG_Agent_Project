"""知识库直答服务。

这条链路专门处理普通产品知识问答：
用户问题 -> Query Planner -> Qdrant 检索/精排 -> 大模型最终回答。

它不绑定 Agent 工具，也不会让模型先判断“要不要调用 rag_summarize”，
因此适合扫拖机器人说明、APP 操作、保养、故障处理等知识类问题。
"""

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

        context = self._retrieve_context(query, history=history)
        logger.info(
            "[知识直答] 非流式回答开始 问题=%s 参考资料字符数=%s",
            query,
            len(context),
        )
        response = chat_model.invoke(self._build_messages(query, context, history=history))
        answer = self._message_content_to_text(response.content).strip()
        logger.info("[知识直答] 非流式回答完成 问题=%s 回答字符数=%s", query, len(answer))
        return answer

    def stream_answer(self, query: str, *, history: list[dict] | None = None) -> Iterator[str]:
        """流式生成知识库问答结果。"""

        context = self._retrieve_context(query, history=history)
        logger.info(
            "[知识直答] 流式回答开始 问题=%s 参考资料字符数=%s",
            query,
            len(context),
        )
        total_chars = 0
        for chunk in chat_model.stream(self._build_messages(query, context, history=history)):
            content = self._message_content_to_text(chunk.content)
            if not content:
                continue
            total_chars += len(content)
            yield content
        logger.info("[知识直答] 流式回答完成 问题=%s 回答字符数=%s", query, total_chars)

    def _retrieve_context(self, query: str, *, history: list[dict] | None = None) -> str:
        """统一走普通多意图 RAG 检索，不再做 FAQ 编号/列表特殊分支。"""

        context = self.rag.rag_summarize(query, history=history)
        logger.info("[知识直答] RAG 检索完成 问题=%s 参考资料字符数=%s", query, len(context))
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
            "你是扫地机器人、扫拖一体机器人和配套 APP 的中文客服。"
            "你必须只根据参考资料回答用户问题，不要编造参考资料没有的功能、路径或数据。"
            "用户一句话里可能包含多个问题，请按用户原问题的顺序逐个覆盖，不要漏答。"
            "如果某个子问题在参考资料中没有明确说明，请直接说明“参考资料中未找到该问题的明确说明”。"
            "如果参考资料已经能回答，就不要泛泛建议用户再去查说明书或联系客服。"
            "回答要自然、简洁、专业；可以使用小标题或列表，但不要输出内部检索流程、工具名、向量库名。"
            "除非用户明确询问来源，否则不要机械展示来源文件、知识类型、分类等字段。"
        )

    @staticmethod
    def _user_message(query: str, context: str) -> str:
        return (
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context}\n\n"
            "请根据参考资料生成最终回答。"
        )
