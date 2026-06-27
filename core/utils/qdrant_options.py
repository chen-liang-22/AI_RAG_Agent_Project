"""Qdrant 配置解析工具。

这里统一处理 Qdrant 的 collection、连接参数、距离算法。
优先读取环境变量，未配置时再读取 config/storage.yml，方便本地和部署环境复用同一套代码。
"""

import os

from qdrant_client import models

from core.utils.config_handler import qdrant_conf


def normalize_qdrant_collection_name(collection_name: str | None = None) -> str:
    """归一化 collection 名称，空值时使用默认 collection。"""

    name = (collection_name or get_qdrant_collection_name()).strip()
    if not name:
        name = str(qdrant_conf["collection_name"]).strip()
    return name


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值，未配置时返回默认值。"""

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    """从环境变量读取整数值，未配置时返回默认值。"""

    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def get_qdrant_collection_name() -> str:
    """读取默认 Qdrant collection 名称。"""

    return os.getenv("QDRANT_COLLECTION_NAME") or qdrant_conf["collection_name"]


def get_qdrant_client_options() -> dict:
    """生成 QdrantClient 初始化参数。"""

    url = os.getenv("QDRANT_URL") or qdrant_conf.get("url")
    prefer_grpc = _env_bool("QDRANT_PREFER_GRPC", qdrant_conf.get("prefer_grpc", False))
    grpc_port = _env_int("QDRANT_GRPC_PORT", qdrant_conf.get("grpc_port", 6334))
    api_key = os.getenv("QDRANT_API_KEY") or qdrant_conf.get("api_key")
    timeout = _env_int("QDRANT_TIMEOUT", qdrant_conf.get("timeout"))

    if url:
        options = {
            "url": url,
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "api_key": api_key,
            "timeout": timeout,
        }
    else:
        options = {
            "host": os.getenv("QDRANT_HOST") or qdrant_conf.get("host", "localhost"),
            "port": _env_int("QDRANT_PORT", qdrant_conf.get("port", 6333)),
            "grpc_port": grpc_port,
            "prefer_grpc": prefer_grpc,
            "api_key": api_key,
            "timeout": timeout,
        }

    return {key: value for key, value in options.items() if value is not None}


def get_qdrant_distance() -> models.Distance:
    """读取向量距离算法，配置非法时默认使用 COSINE。"""

    distance_name = (os.getenv("QDRANT_DISTANCE") or qdrant_conf.get("distance", "COSINE")).upper()
    return getattr(models.Distance, distance_name, models.Distance.COSINE)
