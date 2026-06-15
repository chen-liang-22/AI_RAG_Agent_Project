from fastapi import APIRouter
from qdrant_client import QdrantClient

from api.schemas import HealthResponse
from utils.logger_handler import logger
from utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info("[接口] 健康检查请求")  # 记录健康检查访问日志
    collection_name = get_qdrant_collection_name()  # 读取当前配置中的 collection 名称

    try:
        client = QdrantClient(**get_qdrant_client_options())  # 按配置连接 Qdrant
        collections = [collection.name for collection in client.get_collections().collections]  # 获取全部 collection 名称
        qdrant_status = "ok"  # 能正常连接和读取 collection，说明 Qdrant 可用
    except Exception:
        collections = []  # Qdrant 不可用时返回空列表，避免健康检查接口直接报错
        qdrant_status = "unavailable"  # 标记 Qdrant 不可用

    status = "ok" if qdrant_status == "ok" else "degraded"  # 依赖不可用时整体状态降级
    return HealthResponse(
        status=status,  # 返回整体状态
        qdrant=qdrant_status,  # 返回 Qdrant 状态
        collection_name=collection_name,  # 返回当前 collection 名称
        collections=collections,  # 返回 Qdrant collection 列表
    )
