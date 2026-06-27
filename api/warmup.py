"""FastAPI 启动预热逻辑。

这个模块只负责在应用启动阶段提前触发 RAG 链路的关键依赖：
- 初始化知识直答服务和向量库服务对象。
- 检查 Qdrant collection 是否可访问。
- 调用一次 embedding 模型，提前完成鉴权、连接和模型侧冷启动。
- 可选调用一次最终回答模型，进一步降低第一次真实聊天的首字等待。

预热不是业务必需流程，所以默认采用“失败不中断启动”的策略。
是否中断由 `warmup_fail_fast` 或环境变量 `WARMUP_FAIL_FAST` 控制。
"""

import os
import time
from dataclasses import dataclass

from core.utils.config_handler import qdrant_conf, rag_conf
from core.utils.logger_handler import logger
from core.utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name


@dataclass(frozen=True)
class WarmupSettings:
    """启动预热配置快照。

    使用 dataclass 的好处是把多个开关集中成一个只读对象，后续传递时不需要反复读取配置。
    `frozen=True` 表示加载完成后不再修改，避免运行过程中被误改。
    """

    # 总开关：关闭后整个预热流程直接跳过。
    enabled: bool

    # 是否检查 Qdrant 连接和 collection 存在性。
    qdrant: bool

    # 是否调用一次 embedding 模型，提前建立模型连接。
    embedding: bool

    # 是否调用一次最终回答模型。该步骤会消耗少量模型额度，因此默认关闭。
    chat_model: bool

    # 预热失败时是否中断应用启动。生产环境如果要求依赖必须可用，可以打开。
    fail_fast: bool


def _config_bool(name: str, default: bool) -> bool:
    """读取布尔配置，环境变量优先，其次读取 `config/app.yml` 的 rag 配置。

    约定：
    - 环境变量名使用大写，如 `WARMUP_QDRANT`。
    - YAML 配置名使用小写，如 `warmup_qdrant`。
    - 只有 `1/true/yes/on` 会被识别为 True，其它非空值都会视为 False。
    """

    # 环境变量优先，方便不同部署环境在不改配置文件的情况下临时开关预热项。
    value = os.getenv(name)
    if value is None or value == "":
        # 配置文件里使用小写 key，所以这里统一转成 lower。
        raw_value = rag_conf.get(name.lower(), default)
    else:
        raw_value = value

    # YAML 已经解析成 bool 时直接返回，避免字符串转换影响原始语义。
    if isinstance(raw_value, bool):
        return raw_value

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _load_settings() -> WarmupSettings:
    """从环境变量和配置文件中加载启动预热配置。"""

    return WarmupSettings(
        enabled=_config_bool("STARTUP_WARMUP_ENABLED", True),
        qdrant=_config_bool("WARMUP_QDRANT", True),
        embedding=_config_bool("WARMUP_EMBEDDING", True),
        chat_model=_config_bool("WARMUP_CHAT_MODEL", False),
        fail_fast=_config_bool("WARMUP_FAIL_FAST", False),
    )


def run_startup_warmup() -> None:
    """在 FastAPI 启动阶段预热 RAG 关键依赖，降低第一次聊天的冷启动耗时。"""

    settings = _load_settings()

    # 总开关关闭时直接返回，不初始化任何模型或外部服务。
    if not settings.enabled:
        logger.info("[启动预热] 已跳过 原因=配置关闭")
        return

    total_start_time = time.perf_counter()
    logger.info(
        "[启动预热] 开始 Qdrant=%s Embedding=%s 回答模型=%s 失败中断=%s",
        settings.qdrant,
        settings.embedding,
        settings.chat_model,
        settings.fail_fast,
    )

    # 先初始化服务对象，因为后续 Qdrant/Embedding 预热都会间接依赖这些单例。
    _run_step("服务对象初始化", _warmup_services, fail_fast=settings.fail_fast)

    # Qdrant 是向量检索的核心外部依赖，预热时只检查连接和 collection，不执行检索。
    if settings.qdrant:
        _run_step("Qdrant连接", _warmup_qdrant, fail_fast=settings.fail_fast)
    else:
        logger.info("[启动预热] Qdrant连接跳过 原因=配置关闭")

    # embedding 预热会真实调用一次向量模型，能提前发现 API Key、网络或模型名配置问题。
    if settings.embedding:
        _run_step("Embedding模型", _warmup_embedding, fail_fast=settings.fail_fast)
    else:
        logger.info("[启动预热] Embedding模型跳过 原因=配置关闭")

    # 回答模型预热会消耗少量 token/额度，所以默认关闭，只在需要更低首字延迟时打开。
    if settings.chat_model:
        _run_step("回答模型", _warmup_chat_model, fail_fast=settings.fail_fast)
    else:
        logger.info("[启动预热] 回答模型跳过 原因=配置关闭")

    logger.info("[启动预热] 完成 总耗时毫秒=%.2f", _elapsed_ms(total_start_time))


def _run_step(name: str, callback, *, fail_fast: bool) -> None:
    """执行单个预热步骤，并统一记录耗时和异常。

    `fail_fast=False` 时，某个预热步骤失败不会影响应用启动；
    `fail_fast=True` 时，异常会继续向上抛出，让 FastAPI 启动失败。
    """

    start_time = time.perf_counter()
    try:
        callback()
    except Exception as exc:
        logger.error("[启动预热] %s失败 错误=%s", name, exc, exc_info=True)
        if fail_fast:
            raise
        return

    logger.info("[启动预热] %s完成 耗时毫秒=%.2f", name, _elapsed_ms(start_time))


def _warmup_services() -> None:
    """初始化知识直答服务和向量库服务对象。

    `bootstrap_v2_metadata()` 会显式初始化默认字典和历史存储字段。
    `_get_knowledge_answer_service()` 内部会创建 `KnowledgeAnswerService` 单例，
    `service.rag._get_vector_store()` 会继续初始化 `VectorStoreService`，提前完成 QdrantVectorStore 构造。
    """

    # 延迟导入可以避免模块加载时就触发模型和向量库初始化，只有真正预热时才执行。
    from app_v2.application.chat_generation_service import _get_knowledge_answer_service
    from app_v2.infrastructure.repositories.bootstrap import bootstrap_v2_metadata

    bootstrap_v2_metadata()
    service = _get_knowledge_answer_service()
    service.rag._get_vector_store()


def _warmup_qdrant() -> None:
    """检查 Qdrant collection 是否可访问。

    这里不写入数据，也不执行向量检索，只做轻量的 collection 存在性检查。
    """

    from qdrant_client import QdrantClient

    collection_name = get_qdrant_collection_name()
    client = QdrantClient(**get_qdrant_client_options())
    exists = client.collection_exists(collection_name)
    logger.info("[启动预热] Qdrant collection检查 collection=%s 存在=%s", collection_name, exists)


def _warmup_embedding() -> None:
    """调用一次 embedding 模型，提前完成向量模型冷启动。"""

    from core.model.factory import embed_model

    # 使用极短中文文本，既能验证模型可用性，又尽量减少调用成本。
    vector = embed_model.embed_query("预热")
    logger.info("[启动预热] Embedding模型调用完成 模型=%s 向量维度=%s", rag_conf["embedding_model_name"], len(vector))


def _warmup_chat_model() -> None:
    """可选预热最终回答模型。

    该步骤会真实调用聊天模型并等待首个流式分片返回，所以默认不启用。
    开启后可以进一步降低第一次真实聊天时的首字耗时。
    """

    from langchain_core.messages import HumanMessage

    from core.model.factory import chat_model

    chunk_count = 0

    # 只等待第一个有效分片即可，预热目标是建立连接和触发模型侧初始化，不需要完整回答。
    for chunk in chat_model.stream([HumanMessage(content="请只回复：ok")]):
        content = getattr(chunk, "content", "")
        if content:
            chunk_count += 1
            break
    logger.info("[启动预热] 回答模型首个分片完成 模型=%s 分片数=%s", rag_conf["chat_model_name"], chunk_count)


def _elapsed_ms(start_time: float) -> float:
    """把 `time.perf_counter()` 的开始时间转换为毫秒耗时。"""

    return (time.perf_counter() - start_time) * 1000
