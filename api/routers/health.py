from fastapi import APIRouter
from qdrant_client import QdrantClient

from api.schemas import HealthResponse
from rag.knowledge_store import KnowledgeStore
from utils.logger_handler import logger
from utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name
from utils.redis_client import get_redis_client

router = APIRouter()


def _service_status(item_code: str) -> str:
    """从服务状态字典读取状态码，避免健康检查里写散落状态值。"""

    return KnowledgeStore().normalize_dictionary_code("service_status", item_code)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info("[接口] 健康检查请求")  # 记录健康检查访问日志
    collection_name = get_qdrant_collection_name()  # 读取当前配置中的 collection 名称

    try:
        client = QdrantClient(**get_qdrant_client_options())  # 按配置连接 Qdrant
        collections = [collection.name for collection in client.get_collections().collections]  # 获取全部 collection 名称
        collection_points = {
            # count(exact=True) 读取 Qdrant 当前 collection 的真实向量点数量。
            # 首页需要这个数字区分“普通文件列表为空”和“向量库确实为空”。
            collection_name: int(client.count(collection_name=collection_name, exact=True).count)
            for collection_name in collections
        }
        qdrant_status = _service_status("ok")  # 能正常连接和读取 collection，说明 Qdrant 可用
    except Exception:
        collections = []  # Qdrant 不可用时返回空列表，避免健康检查接口直接报错
        collection_points = {}  # Qdrant 不可用时无法统计点数，返回空字典
        qdrant_status = _service_status("unavailable")  # 标记 Qdrant 不可用

    # Redis 是缓存和任务状态加速层，不可用时主业务仍可降级运行。
    redis_status = _service_status("ok") if get_redis_client().is_available() else _service_status("unavailable")
    status = (
        _service_status("ok")
        if qdrant_status == _service_status("ok") and redis_status == _service_status("ok")
        else _service_status("degraded")
    )
    return HealthResponse(
        status=status,  # 返回整体状态
        qdrant=qdrant_status,  # 返回 Qdrant 状态
        redis=redis_status,  # 返回 Redis 状态
        collection_name=collection_name,  # 返回当前 collection 名称
        collections=collections,  # 返回 Qdrant collection 列表
        collection_points=collection_points,  # 返回每个 collection 的真实向量点数量
    )
