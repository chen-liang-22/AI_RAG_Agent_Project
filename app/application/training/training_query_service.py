"""销售训练资料检索服务。

这个模块只负责把“训练问题 -> 训练证据”这段链路收拢起来：
- 查询当前已发布的训练资料批次；
- 调用 Qdrant 向量检索；
- 根据 visibility 过滤可用证据；
- 把 LangChain Document 转成销售训练 LLM 可直接使用的 evidence。

拆出这个服务后，核心编排类不再直接关心训练召回细节，后续要调 top_k、
过滤规则、日志字段或召回评估时，可以优先改这里。
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document

from app.application.training_support.repository import TrainingRepository
from app.infrastructure.vector_store_service import VectorStoreService
from core.utils.logger_handler import logger


class TrainingQueryService:
    """销售训练向量检索服务。

    这里使用服务对象封装检索算法，核心外观服务只需要传入 query 和可见性。
    这不是新的接口协议，只是内部职责拆分。
    """

    def __init__(
            self,
            *,
            repository: TrainingRepository,
            vector_service: VectorStoreService,
            collection_name: str,
    ):
        """初始化训练检索服务。

        repository 用来读取当前发布版本批次，vector_service 用来访问 Qdrant。
        collection_name 只用于日志，避免排查多 collection 时看不清来源。
        """

        self.repository = repository
        self.vector_service = vector_service
        self.collection_name = collection_name

    def search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict[str, Any]]:
        """检索训练证据库，并按可见性过滤证据。

        当前训练知识只允许命中“当前发布版本”的批次，避免旧版本资料继续影响 AI 客户。
        visibility 用于区分显性资料、隐藏画像资料和只用于评分的资料。
        """

        start_perf = time.perf_counter()
        logger.info(
            "[销售训练][训练证据检索] 开始检索 collection=%s 可见性=%s 返回数量=%s 查询长度=%s 查询预览=%s",
            self.collection_name,
            self._join_values(visibility),
            k,
            len(query or ""),
            self._short_text(query),
        )
        current_batch_ids = self.repository.list_current_published_batch_ids()
        if not current_batch_ids:
            logger.info(
                "[销售训练][训练证据检索] 没有当前发布版本，跳过向量检索 collection=%s 耗时秒=%s",
                self.collection_name,
                round(max(0.0, time.perf_counter() - start_perf), 3),
            )
            return []

        documents = self.vector_service.search_documents(query, k=k, filters={"batch_id": current_batch_ids})
        evidence = self.documents_to_evidence(documents, visibility=visibility)
        logger.info(
            "[销售训练][训练证据检索] 检索完成 collection=%s 原始文档数=%s 过滤后证据数=%s 来源文件=%s 案例部分=%s 耗时秒=%s",
            self.collection_name,
            len(documents),
            len(evidence),
            self._join_values(item.get("source_file") for item in evidence),
            self._join_values(item.get("case_part") for item in evidence),
            round(max(0.0, time.perf_counter() - start_perf), 3),
        )
        return evidence

    @staticmethod
    def documents_to_evidence(documents: list[Document], *, visibility: tuple[str, ...]) -> list[dict[str, Any]]:
        """把向量库返回的 Document 转成训练证据结构。

        单条证据正文限制在 800 字以内，避免把过长片段直接塞进 LLM 上下文。
        """

        evidence: list[dict[str, Any]] = []
        for document in documents:
            metadata = dict(document.metadata)
            item_visibility = str(metadata.get("visibility") or "visible")
            if item_visibility not in visibility:
                continue
            evidence.append(
                {
                    "chunk_id": str(metadata.get("chunk_id") or metadata.get("batch_id") or ""),
                    "case_part": str(metadata.get("case_part") or ""),
                    "visibility": item_visibility,
                    "score": metadata.get("_vector_score"),
                    "content": document.page_content[:800],
                    "source_file": metadata.get("source_file"),
                }
            )
        return evidence

    @staticmethod
    def _short_text(value: Any, limit: int = 120) -> str:
        """把长文本压缩成日志预览，避免控制台输出完整提示词。"""

        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text or "-"
        return f"{text[:limit]}..."

    @staticmethod
    def _join_values(values: Any, limit: int = 6) -> str:
        """把列表、元组或生成器压成一行日志文本。"""

        if values is None:
            return "-"
        if isinstance(values, (str, int, float)):
            return str(values)
        result: list[str] = []
        for value in values:
            if value is None or value == "":
                continue
            text = str(value)
            if text not in result:
                result.append(text)
            if len(result) >= limit:
                break
        return "、".join(result) if result else "-"
