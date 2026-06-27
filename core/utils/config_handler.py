"""项目配置加载门面。

本模块负责读取 .env 以及 config/app.yml、config/storage.yml、config/training.yml。
业务代码优先通过这里导出的配置对象读取参数，避免各处重复打开 YAML 文件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.utils.path_tool import get_abs_path


CONFIG_DIR = Path(get_abs_path("config"))


def load_env_file(env_path: str = get_abs_path(".env"), encoding: str = "utf-8") -> None:
    """读取项目根目录 .env，并写入当前进程环境变量。

    这里只做轻量 KEY=VALUE 解析，且不会覆盖系统已经存在的环境变量。
    原因是部署环境中的真实环境变量优先级应该高于本地 .env 文件。
    """

    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding=encoding) as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_yaml_config(config_path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """读取 YAML 配置文件，根节点必须是字典。

    统一入口能把“文件不存在、YAML 格式错误、根节点类型错误”集中处理，
    调用方不需要重复写 try/except。
    """

    path = Path(config_path)
    try:
        data = yaml.safe_load(path.read_text(encoding=encoding)) or {}
    except OSError as exc:
        raise RuntimeError(f"配置文件读取失败：{path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"配置文件解析失败：{path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是字典：{path}")
    return data


def load_app_config(config_path: str | Path = CONFIG_DIR / "app.yml") -> dict[str, Any]:
    """读取应用级配置。"""

    return load_yaml_config(config_path)


def load_storage_config(config_path: str | Path = CONFIG_DIR / "storage.yml") -> dict[str, Any]:
    """读取存储与中间件配置。"""

    return load_yaml_config(config_path)


def load_training_config(config_path: str | Path = CONFIG_DIR / "training.yml") -> dict[str, Any]:
    """读取销售训练配置。"""

    return load_yaml_config(config_path)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """从配置字典中安全读取一个二级配置段。"""

    value = config.get(name)
    return value if isinstance(value, dict) else {}


load_env_file()

app_conf = load_app_config()
storage_conf = load_storage_config()
training_conf = load_training_config()

# 兼容旧业务代码的配置变量名。
rag_conf = _section(app_conf, "rag")
agent_conf = _section(app_conf, "agent")
id_generator_conf = _section(app_conf, "id_generator")
knowledge_manifest_conf = _section(app_conf, "knowledge_manifest")

database_conf = _section(storage_conf, "database")
redis_conf = _section(storage_conf, "redis")
minio_conf = _section(storage_conf, "minio")
qdrant_conf = _section(storage_conf, "qdrant")


if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
