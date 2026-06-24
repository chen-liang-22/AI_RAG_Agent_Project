from typing import Annotated, TypedDict  # TypedDict 定义 LangGraph 状态；Annotated 绑定消息合并规则

from langchain_core.messages import (  # LangChain 标准消息类型
    AIMessage,  # 完整 AI 消息；一次性 invoke 的最终结果通常是它
    AIMessageChunk,  # 流式 AI 消息片段；SSE 只应该把它推给前端
    AnyMessage,  # 任意 LangChain 消息类型
    HumanMessage,  # 用户消息类型
    SystemMessage,  # 系统提示词消息类型
    ToolMessage,  # 工具返回消息类型；用于判断是否触发报告模式
)
from langgraph.graph import END, StateGraph  # END 表示图结束；StateGraph 用于显式搭建工作流
from langgraph.graph.message import add_messages  # 消息 reducer，负责把新消息追加进 state["messages"]
from langgraph.prebuilt import ToolNode, tools_condition  # ToolNode 执行工具；tools_condition 判断是否需要走工具节点

from agent.tools.agent_tools import (  # 项目原有工具，继续复用
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_user_id,
    get_user_location,
    get_weather,
    rag_summarize,
)
from model.factory import chat_model  # 项目统一模型实例，已开启 streaming=True
from utils.logger_handler import logger  # 项目统一日志
from utils.prompt_loader import load_report_prompts, load_system_prompts  # 普通客服提示词和报告提示词


class AgentState(TypedDict, total=False):
    """LangGraph 的状态结构。

    messages：当前会话消息列表。`add_messages` 会让每个节点返回的新消息追加到历史里。
    report：是否进入报告生成场景。调用 fill_context_for_report 工具后会被置为 True。
    """

    messages: Annotated[list[AnyMessage], add_messages]  # 图里的对话消息历史
    report: bool  # 是否启用报告提示词


class ReactAgent:
    """基于 LangGraph 显式状态图实现的 ReAct Agent。

    对外仍然保留两个方法：
    - execute()：一次性返回完整回答，供 `/chat` 使用。
    - execute_stream()：逐 chunk 返回最终回答，供 `/chat/stream` 使用。

    因为方法名和返回格式没变，FastAPI 和前端不需要跟着改。
    """

    def __init__(self):
        # 工具列表仍然复用原项目工具；LangGraph 的 ToolNode 会负责真正执行它们。
        self.tools = [
            rag_summarize,
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            fetch_external_data,
            fill_context_for_report,
        ]

        # bind_tools 把工具 schema 绑定到模型，让模型可以在回答过程中发起 tool_calls。
        self.model = chat_model.bind_tools(self.tools)

        # 编译后的 LangGraph 图对象。
        self.graph = self._build_graph()

    def _build_graph(self):
        """搭建 LangGraph 状态图。

        图结构：

            agent
              │
              ├── 有 tool_calls ──> tools ──> update_report_state ──> agent
              │
              └── 无 tool_calls ──> END

        agent 节点负责调用大模型。
        tools 节点负责执行 RAG、天气、用户数据等工具。
        update_report_state 节点负责把“报告生成场景”写进 state。
        """

        graph = StateGraph(AgentState)  # 创建一个以 AgentState 为状态结构的图
        graph.add_node("agent", self._call_model)  # agent 节点：调用模型生成 AIMessage
        graph.add_node("tools", ToolNode(self.tools))  # tools 节点：执行模型提出的工具调用
        graph.add_node("update_report_state", self._update_report_state)  # 工具后处理节点：更新 report 状态

        graph.set_entry_point("agent")  # 每次请求都从 agent 节点开始
        graph.add_conditional_edges(  # agent 节点执行后，根据最后一条 AIMessage 判断是否需要工具
            "agent",
            tools_condition,  # 如果有 tool_calls 返回 "tools"，否则返回 "__end__"
            {
                "tools": "tools",  # 有工具调用就进入 tools 节点
                "__end__": END,  # 没有工具调用就结束图执行
            },
        )
        graph.add_edge("tools", "update_report_state")  # 工具执行完后先更新业务状态
        graph.add_edge("update_report_state", "agent")  # 更新完状态后回到模型，让模型生成最终回答

        return graph.compile()  # 编译图；会话历史由 MySQL 在每次请求前注入

    @staticmethod
    def _message_content_to_text(content) -> str:
        """把 LangChain 消息内容统一转换成字符串。"""

        if isinstance(content, str):  # 普通文本直接返回
            return content

        if isinstance(content, list):  # 兼容结构化消息内容
            parts = []  # 保存每个文本块
            for item in content:  # 逐项处理内容块
                if isinstance(item, str):  # 字符串块
                    parts.append(item)  # 直接加入
                elif isinstance(item, dict):  # 字典块
                    parts.append(str(item.get("text") or item.get("content") or ""))  # 取 text/content 字段
            return "".join(parts)  # 拼接成完整文本

        return str(content or "")  # 兜底转换

    @classmethod
    def _input_state(cls, query: str, history: list[dict] | None = None) -> AgentState:
        """把用户输入转换成 LangGraph 的初始 state。

        这里会把 MySQL 中最近的历史消息和当前用户新消息一起传入。
        这样服务重启后也能恢复会话上下文。
        """

        return {
            "messages": [*cls._history_to_messages(history or []), HumanMessage(content=query)],  # 历史 + 当前轮用户消息
            "report": False,  # 每一轮先从普通客服模式开始；工具可在本轮把它切成报告模式
        }

    @staticmethod
    def _history_to_messages(history: list[dict]) -> list[AnyMessage]:
        """把 MySQL 消息记录转换成 LangChain 消息。"""

        messages: list[AnyMessage] = []
        for item in history:
            role = item.get("role")
            content = str(item.get("content") or "")
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
    def _run_config(user_id: str | None, conversation_id: str | None = None) -> dict:
        """生成 LangGraph 运行配置。

        thread_id 主要用于 LangGraph 内部 tracing。
        持久化历史以 MySQL conversation_id 为准。
        """

        return {"configurable": {"thread_id": conversation_id or user_id or "anonymous"}}

    def _call_model(self, state: AgentState) -> dict:
        """LangGraph 的 agent 节点：调用模型生成下一条 AIMessage。

        如果 state["report"] 为 True，就使用报告生成提示词；
        否则使用普通客服系统提示词。
        """

        messages = state.get("messages", [])  # 取出当前图状态里的完整消息历史
        report = state.get("report", False)  # 判断是否处于报告生成场景
        prompt = load_report_prompts() if report else load_system_prompts()  # 根据状态选择系统提示词

        logger.info(f"[LangGraph代理] 调用模型 消息数={len(messages)} 是否报告模式={report}")  # 记录模型调用日志
        response = self.model.invoke([SystemMessage(content=prompt), *messages])  # 系统提示词 + 历史消息一起发给模型

        return {"messages": [response]}  # 返回新 AIMessage，由 add_messages 合并进 state["messages"]

    @staticmethod
    def _update_report_state(state: AgentState) -> dict:
        """工具执行后的状态修正节点。

        原 LangChain create_agent 版本里，middleware 会在 fill_context_for_report 工具执行后
        把 runtime.context["report"] 设置为 True。

        改成显式 LangGraph 后，我们在这里检查刚执行完的 ToolMessage：
        - 如果工具名是 fill_context_for_report，说明进入报告生成场景。
        - 返回 {"report": True}，后续 agent 节点会切换到报告提示词。
        """

        for message in reversed(state.get("messages", [])):  # 从最新消息往前看
            if isinstance(message, ToolMessage):  # 只关心工具返回消息
                if message.name == "fill_context_for_report":  # 该工具用于触发报告场景
                    logger.info("[LangGraph代理] 检测到报告工具调用，切换为报告模式")  # 记录状态切换
                    return {"report": True}  # 更新图状态
                continue  # 其他工具消息继续往前检查

            break  # 遇到非 ToolMessage，说明最新一批工具消息已经检查完

        return {}  # 没有报告工具调用，不修改状态

    def execute(
            self,
            query: str,
            user_id: str | None = None,
            conversation_id: str | None = None,
            history: list[dict] | None = None,
    ) -> str:
        """一次性执行 LangGraph，并返回最终回答。"""

        result = self.graph.invoke(  # invoke 会等整张图运行结束
            self._input_state(query, history=history),  # 历史 + 当前轮输入
            config=self._run_config(user_id, conversation_id),  # thread_id 配置，用于 tracing
            context={"user_id": user_id},  # 运行时上下文，ToolRuntime 会把它注入 get_user_id 工具
        )
        latest_message = result["messages"][-1]  # 图结束时最后一条消息就是最终 AI 回答
        return self._message_content_to_text(latest_message.content).strip()  # 返回纯文本

    def execute_stream(
            self,
            query: str,
            user_id: str | None = None,
            conversation_id: str | None = None,
            history: list[dict] | None = None,
    ):
        """流式执行 LangGraph，并逐段产出最终 AI 回答。"""

        for message, metadata in self.graph.stream(  # stream 会在图运行过程中持续产出消息片段
                self._input_state(query, history=history),  # 历史 + 当前轮输入
                config=self._run_config(user_id, conversation_id),  # thread_id 配置
                context={"user_id": user_id},  # 工具运行上下文
                stream_mode="messages",  # 只订阅消息流，适合转发模型 token
        ):
            if metadata.get("langgraph_node") != "agent":  # 只转发 agent 节点产生的模型消息
                continue

            if not isinstance(message, AIMessageChunk):  # 只转发真实 token chunk，避免工具消息或完整消息混入
                continue

            content = self._message_content_to_text(message.content)  # 转成文本
            if content:  # 空 chunk 不输出
                yield content  # 交给 FastAPI SSE 接口继续往前端推

    def _answer_from_retrieved_context(
            self,
            query: str,
            context: str,
            *,
            history: list[dict] | None = None,
    ) -> str:
        """把 Qdrant 检索结果交给模型，生成更自然的最终回答。"""

        response = chat_model.invoke(
            [
                SystemMessage(content=self._retrieved_context_prompt()),
                *self._history_to_messages(history or []),
                HumanMessage(content=self._retrieved_context_user_message(query, context)),
            ]
        )
        return self._message_content_to_text(response.content).strip()

    def _stream_answer_from_retrieved_context(
            self,
            query: str,
            context: str,
            *,
            history: list[dict] | None = None,
    ):
        """把 Qdrant 检索结果交给模型，并流式生成自然回答。"""

        for chunk in chat_model.stream(
                [
                    SystemMessage(content=self._retrieved_context_prompt()),
                    *self._history_to_messages(history or []),
                    HumanMessage(content=self._retrieved_context_user_message(query, context)),
                ]
        ):
            content = self._message_content_to_text(chunk.content)
            if content:
                yield content

    @staticmethod
    def _retrieved_context_prompt() -> str:
        return (
            "你是扫地机器人/扫拖一体机器人客服。"
            "请只根据参考资料回答用户问题，不要编造。"
            "可以结合会话历史理解用户省略的上下文。"
            "回答要像真实客服对话一样自然、简洁、专业。"
            "不要机械复述字段名，比如“来源文件、分类”，除非用户明确问来源。"
            "如果用户问第几问，说明该问题是什么，并直接给出答案。"
            "如果参考资料是问题清单，保留编号和问题标题，不要擅自删减。"
        )

    @staticmethod
    def _retrieved_context_user_message(query: str, context: str) -> str:
        return (
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context}\n\n"
            "请基于参考资料给出最终回答。"
        )


if __name__ == '__main__':  # 允许直接运行本文件做手动测试
    agent = ReactAgent()  # 创建 LangGraph Agent

    for chunk in agent.execute_stream("给我生成我的使用报告", user_id="1001"):  # 测试流式输出
        print(chunk, end="", flush=True)  # 模拟前端逐 chunk 拼接
