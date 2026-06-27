"""
配置文件加载工具。

本模块在进程启动时读取 .env、config/rag.yml、config/qdrant.yml、config/agent.yml。
业务代码通过 rag_conf / qdrant_conf / agent_conf 读取配置，避免各处重复打开 YAML。
"""
import os

import yaml

from core.utils.path_tool import get_abs_path


def load_env_file(env_path: str = get_abs_path(".env"), encoding: str = "utf-8") -> None:
    """读取项目根目录 .env，并写入当前进程环境变量。

    这个项目的模型 SDK 会从环境变量里读取 DASHSCOPE_API_KEY。
    如果只在 PyCharm 里配置了解析器，有时命令行启动不会自动加载 .env。

    这里做一个轻量兜底：
    - 只支持常见的 KEY=VALUE 格式。
    - 空行和 # 开头的注释会跳过。
    - 如果系统环境变量里已经有同名 key，不覆盖系统已有值。
    - 不打印任何 value，避免泄露密钥。
    """

    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding=encoding) as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    """读取 RAG 和模型相关配置。"""

    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_qdrant_config(config_path: str = get_abs_path("config/qdrant.yml"), encoding: str = "utf-8"):
    """读取 Qdrant 向量库配置。"""

    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    """读取 Agent 工具链配置。"""

    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


load_env_file()
rag_conf = load_rag_config()
qdrant_conf = load_qdrant_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
