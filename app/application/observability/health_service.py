"""依赖健康检查应用服务。

第八阶段可观测性增强先把 MySQL、Redis、MinIO、Qdrant、模型配置的状态收敛到统一入口。
路由层只调用 HealthDependencyService，不直接拼接各类 SDK 检查逻辑。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import text

from api.schemas import DependencyHealthItem, HealthDependenciesResponse
from app.infrastructure.orm_session import orm_session_context
from core.utils.config_handler import rag_conf
from core.utils.logger_handler import logger
from core.utils.minio_client import get_minio_client
from core.utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name
from core.utils.redis_client import get_redis_client


@dataclass(slots=True)
class DependencyHealthCheckResult:
    """单个依赖检查结果。"""

    name: str
    status: str
    latency_ms: float | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class DependencyHealthCheck(ABC):
    """依赖健康检查策略接口。"""

    @abstractmethod
    def check(self) -> DependencyHealthCheckResult:
        """执行依赖健康检查。"""


class HealthDependencyService:
    """依赖健康检查外观服务。"""

    def __init__(self, checkers: list[DependencyHealthCheck] | None = None):
        """初始化依赖健康检查服务。"""

        self.checkers = checkers or [
            MySqlHealthCheck(),
            RedisHealthCheck(),
            MinioHealthCheck(),
            QdrantHealthCheck(),
            ChatModelConfigHealthCheck(),
            EmbeddingModelConfigHealthCheck(),
        ]

    def check_dependencies(self) -> HealthDependenciesResponse:
        """返回所有关键依赖的健康状态明细。"""

        results = [checker.check() for checker in self.checkers]
        summary = self._summary(results)
        status = "ok" if summary.get("unavailable", 0) == 0 else "degraded"
        logger.info(
            "[健康检查] 依赖健康检查完成 整体状态=%s 总数=%s 正常=%s 不可用=%s",
            status,
            summary.get("total", 0),
            summary.get("ok", 0),
            summary.get("unavailable", 0),
        )
        return HealthDependenciesResponse(
            status=status,
            summary=summary,
            dependencies=[self._to_response_item(item) for item in results],
        )

    @staticmethod
    def _summary(results: list[DependencyHealthCheckResult]) -> dict[str, int]:
        """统计依赖状态数量。"""

        summary = {"total": len(results), "ok": 0, "unavailable": 0}
        for item in results:
            if item.status == "ok":
                summary["ok"] += 1
            else:
                summary["unavailable"] += 1
        return summary

    @staticmethod
    def _to_response_item(result: DependencyHealthCheckResult) -> DependencyHealthItem:
        """把内部检查结果转换成 API 响应项。"""

        return DependencyHealthItem(
            name=result.name,
            status=result.status,
            latency_ms=result.latency_ms,
            message=result.message,
            details=result.details,
        )


class MySqlHealthCheck(DependencyHealthCheck):
    """MySQL 健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """执行 SELECT 1 验证数据库连接。"""

        start_time = time.perf_counter()
        try:
            with orm_session_context() as session:
                session.execute(text("SELECT 1"))
                bind = session.get_bind()
                database_url = getattr(bind, "url", None)
                database_name = getattr(database_url, "database", None) if database_url is not None else None
            return DependencyHealthCheckResult(
                name="mysql",
                status="ok",
                latency_ms=_elapsed_ms(start_time),
                message="MySQL 连接正常",
                details={"database": database_name or ""},
            )
        except Exception as exc:
            logger.warning("[健康检查] MySQL 不可用 错误=%s", exc)
            return DependencyHealthCheckResult(
                name="mysql",
                status="unavailable",
                latency_ms=_elapsed_ms(start_time),
                message=f"MySQL 不可用：{exc}",
            )


class RedisHealthCheck(DependencyHealthCheck):
    """Redis 健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """通过 Redis PING 检查可用性。"""

        start_time = time.perf_counter()
        redis_client = get_redis_client()
        available = redis_client.is_available()
        return DependencyHealthCheckResult(
            name="redis",
            status="ok" if available else "unavailable",
            latency_ms=_elapsed_ms(start_time),
            message="Redis 连接正常" if available else "Redis 未启用或不可用",
            details={
                "enabled": redis_client.enabled,
                "host": str(redis_client.config.get("host") or ""),
                "port": int(redis_client.config.get("port") or 0),
                "db": int(redis_client.config.get("db") or 0),
            },
        )


class MinioHealthCheck(DependencyHealthCheck):
    """MinIO 健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """检查默认 bucket 是否可访问。"""

        start_time = time.perf_counter()
        client = get_minio_client()
        try:
            bucket_name = client.ensure_bucket()
            return DependencyHealthCheckResult(
                name="minio",
                status="ok",
                latency_ms=_elapsed_ms(start_time),
                message="MinIO 存储桶可用",
                details={
                    "enabled": client.enabled,
                    "endpoint": str(client.config.get("endpoint") or ""),
                    "bucket_name": bucket_name,
                    "secure": bool(client.config.get("secure")),
                },
            )
        except Exception as exc:
            logger.warning("[健康检查] MinIO 不可用 错误=%s", exc)
            return DependencyHealthCheckResult(
                name="minio",
                status="unavailable",
                latency_ms=_elapsed_ms(start_time),
                message=f"MinIO 不可用：{exc}",
                details={
                    "enabled": client.enabled,
                    "endpoint": str(client.config.get("endpoint") or ""),
                    "bucket_name": str(client.config.get("bucket_name") or ""),
                    "secure": bool(client.config.get("secure")),
                },
            )


class QdrantHealthCheck(DependencyHealthCheck):
    """Qdrant 健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """检查 Qdrant collection 列表和默认 collection 点数。"""

        start_time = time.perf_counter()
        collection_name = get_qdrant_collection_name()
        try:
            client = QdrantClient(**get_qdrant_client_options())
            collections = [collection.name for collection in client.get_collections().collections]
            point_count = int(client.count(collection_name=collection_name, exact=False).count) if collection_name in collections else 0
            return DependencyHealthCheckResult(
                name="qdrant",
                status="ok",
                latency_ms=_elapsed_ms(start_time),
                message="Qdrant 连接正常",
                details={
                    "collection_name": collection_name,
                    "collections": collections,
                    "default_collection_points": point_count,
                },
            )
        except Exception as exc:
            logger.warning("[健康检查] Qdrant 不可用 错误=%s", exc)
            return DependencyHealthCheckResult(
                name="qdrant",
                status="unavailable",
                latency_ms=_elapsed_ms(start_time),
                message=f"Qdrant 不可用：{exc}",
                details={"collection_name": collection_name},
            )


class ChatModelConfigHealthCheck(DependencyHealthCheck):
    """聊天模型配置健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """检查聊天模型配置是否具备模型名称。"""

        start_time = time.perf_counter()
        provider = str(rag_conf.get("chat_provider") or "").strip()
        model_name = str(rag_conf.get("chat_model_name") or "").strip()
        ok = bool(provider and model_name)
        return DependencyHealthCheckResult(
            name="chat_model",
            status="ok" if ok else "unavailable",
            latency_ms=_elapsed_ms(start_time),
            message="聊天模型配置正常" if ok else "聊天模型配置缺少 provider 或 model_name",
            details={"provider": provider, "model_name": model_name},
        )


class EmbeddingModelConfigHealthCheck(DependencyHealthCheck):
    """Embedding 模型配置健康检查。"""

    def check(self) -> DependencyHealthCheckResult:
        """检查向量模型名称是否配置。"""

        start_time = time.perf_counter()
        model_name = str(rag_conf.get("embedding_model_name") or "").strip()
        return DependencyHealthCheckResult(
            name="embedding_model",
            status="ok" if model_name else "unavailable",
            latency_ms=_elapsed_ms(start_time),
            message="Embedding 模型配置正常" if model_name else "Embedding 模型名称未配置",
            details={"model_name": model_name},
        )


def _elapsed_ms(start_time: float) -> float:
    """计算检查耗时毫秒。"""

    return (time.perf_counter() - start_time) * 1000
