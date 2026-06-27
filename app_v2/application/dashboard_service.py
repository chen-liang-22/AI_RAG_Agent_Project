"""首页驾驶舱应用服务。"""

import time
from datetime import date, datetime
from typing import Any

from qdrant_client import QdrantClient

from api.schemas import ConversationSummaryResponse, HealthResponse
from app_v2.domain.constants import HOME_PAGE_SIZE
from app_v2.domain.schemas import DashboardOverviewResponse
from app_v2.infrastructure.repositories.conversation_repository import ConversationRepository
from app_v2.infrastructure.repositories.document_repository import DocumentRepository
from app_v2.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app_v2.infrastructure.repositories.training_repository import V2TrainingRepository
from app_v2.application.training.sales_training_core import V2SalesTrainingCoreService
from app_v2.shared.document_response import DictionaryCodeSnapshot, document_to_response
from core.utils.logger_handler import logger
from core.utils.qdrant_options import get_qdrant_client_options, get_qdrant_collection_name
from core.utils.redis_client import get_redis_client


class DashboardApplicationService:
    """首页驾驶舱外观服务。

    路由层只调用这个服务；服务内部再协调字典、文档、会话、训练和健康检查。
    这里使用外观模式，是为了让首页聚合逻辑集中在一个入口，而不是散在多个路由里。
    """

    _health_cache: tuple[float, HealthResponse] | None = None
    _health_cache_seconds = 10

    def __init__(self):
        """初始化首页驾驶舱依赖的仓储。

        首页只做聚合查询，所以这里组合字典、文件、聊天和训练仓储，不直接写业务数据。
        """

        self.dictionary_repository = DictionaryRepository()
        self.document_repository = DocumentRepository()
        self.conversation_repository = ConversationRepository()
        self.training_repository = V2TrainingRepository()

    def health(self) -> HealthResponse:
        """查询系统健康状态。

        Qdrant 检查相对重，所以这里做一个 10 秒短缓存，避免首页刷新时重复打后端依赖。
        """

        now = time.monotonic()
        if self._health_cache and now - self._health_cache[0] <= self._health_cache_seconds:
            return self._health_cache[1]

        collection_name = get_qdrant_collection_name()
        ok = self.dictionary_repository.normalize_code("service_status", "ok")
        degraded = self.dictionary_repository.normalize_code("service_status", "degraded")
        unavailable = self.dictionary_repository.normalize_code("service_status", "unavailable")

        try:
            client = QdrantClient(**get_qdrant_client_options())
            collections = [collection.name for collection in client.get_collections().collections]
            collection_points = {item: int(client.count(collection_name=item, exact=False).count) for item in collections}
            qdrant_status = ok
        except Exception as exc:
            logger.warning("[V2首页] Qdrant 健康检查失败 原因=%s", exc)
            collections = []
            collection_points = {}
            qdrant_status = unavailable

        redis_status = ok if get_redis_client().is_available() else unavailable
        status = ok if qdrant_status == ok and redis_status == ok else degraded
        response = HealthResponse(
            status=status,
            qdrant=qdrant_status,
            redis=redis_status,
            collection_name=collection_name,
            collections=collections,
            collection_points=collection_points,
        )
        self._health_cache = (now, response)
        return response

    def overview(self) -> DashboardOverviewResponse:
        """聚合首页驾驶舱数据。"""

        logger.info("[V2首页] 开始聚合首页驾驶舱数据")
        health = self.health()
        dictionary_snapshot = self._document_dictionary_snapshot()
        documents = self.document_repository.list_documents(include_training=True)
        knowledge_files = [document_to_response(row, dictionary_snapshot) for row in documents[:HOME_PAGE_SIZE]]

        conversations, conversation_total = self.conversation_repository.list_conversations(page=1, page_size=HOME_PAGE_SIZE)
        recent_conversations = [self._conversation_summary(row) for row in conversations]

        batches, batch_total = self.training_repository.list_batches(page=1, page_size=HOME_PAGE_SIZE)
        plans, plan_total = self.training_repository.list_plans(page=1, page_size=HOME_PAGE_SIZE)
        sessions, session_total = self.training_repository.list_sessions(page=1, page_size=HOME_PAGE_SIZE)

        response = DashboardOverviewResponse(
            health=health,
            knowledge_files=knowledge_files,
            training_batches=[V2SalesTrainingCoreService._batch_response(row) for row in batches],
            training_plans=[V2SalesTrainingCoreService._plan_summary(row) for row in plans],
            training_sessions=[V2SalesTrainingCoreService._session_summary(row) for row in sessions],
            recent_conversations=recent_conversations,
            metrics={
                "document_total": len(documents),
                "conversation_total": int(conversation_total),
                "training_batch_total": int(batch_total),
                "training_plan_total": int(plan_total),
                "training_session_total": int(session_total),
            },
        )
        logger.info("[V2首页] 首页驾驶舱数据聚合完成 指标=%s", response.metrics)
        return response

    def _document_dictionary_snapshot(self) -> DictionaryCodeSnapshot:
        """从 V2 字典仓储构建首页知识库概览使用的字典快照。"""

        enabled_codes_by_dictionary: dict[str, set[str]] = {}
        default_code_by_dictionary: dict[str, str] = {}
        for dictionary_code in ("document_structure", "split_strategy"):
            rows = self.dictionary_repository.list_items(dictionary_code=dictionary_code)
            enabled_rows = [row for row in rows if int(row.get("enabled") or 0) == 1]
            enabled_codes_by_dictionary[dictionary_code] = {str(row["item_code"]) for row in enabled_rows}
            if enabled_rows:
                default_code_by_dictionary[dictionary_code] = str(enabled_rows[0]["item_code"])
        return DictionaryCodeSnapshot(
            enabled_codes_by_dictionary=enabled_codes_by_dictionary,
            default_code_by_dictionary=default_code_by_dictionary,
        )
    @staticmethod
    def _datetime_to_text(value: object) -> str | None:
        """把数据库时间转换成前端可直接展示的文本。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat(timespec="seconds", sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @classmethod
    def _conversation_summary(cls, row: dict[str, Any]) -> ConversationSummaryResponse:
        """把会话行数据转换成响应对象。"""

        return ConversationSummaryResponse(
            conversation_id=row["conversation_id"],
            user_id=row.get("user_id"),
            title=row.get("title"),
            status=row["status"],
            message_count=int(row.get("message_count") or 0),
            created_at=cls._datetime_to_text(row["created_at"]) or "",
            updated_at=cls._datetime_to_text(row["updated_at"]) or "",
            last_message_at=cls._datetime_to_text(row.get("last_message_at")),
        )
