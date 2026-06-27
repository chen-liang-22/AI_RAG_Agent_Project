"""提示词文件加载工具。

Agent 主提示词、RAG 工具提示词和报告提示词都通过 config/prompts.yml 定位。
这里集中读取文件，避免各个工具分散拼路径。
"""

from core.utils.config_handler import prompts_conf
from core.utils.path_tool import get_abs_path
from core.utils.logger_handler import logger


def load_system_prompts():
    """读取 Agent 主系统提示词。"""

    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[提示词加载] 在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[提示词加载] 解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    """读取 RAG 工具提示词。"""

    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[提示词加载] 在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[提示词加载] 解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    """读取报告生成提示词。"""

    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[提示词加载] 在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[提示词加载] 解析报告生成提示词出错，{str(e)}")
        raise e


if __name__ == '__main__':
    print(load_report_prompts())

