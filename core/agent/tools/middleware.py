"""Agent 工具链中间件。

这些中间件只服务旧 Agent 模式：
- monitor_tool：记录工具调用入参和异常；
- log_before_model：记录模型调用前消息数量；
- report_prompt_switch：报告模式动态切换提示词。
"""

from typing import Callable
from core.utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from core.utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    """记录 Agent 工具调用过程，并在报告工具触发时切换上下文状态。"""

    logger.info(f"[工具监控] 执行工具：{request.tool_call['name']}")
    logger.info(f"[工具监控] 传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[工具监控] 工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    """模型调用前打印消息数量，便于排查 Agent 上下文是否过长。"""

    logger.info(f"[模型调用前] 即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[模型调用前] {type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    """根据运行时上下文选择普通聊天提示词或报告提示词。"""

    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
