"""异步入库任务处理器。"""

from __future__ import annotations

from typing import Any

from app.application.knowledge.indexing_service import _index_document
from app.application.training.sales_training_core import V2SalesTrainingCoreService
from app.infrastructure.repositories.document_repository import DocumentRepository


def process_document_ingest_task(task: dict[str, Any], reporter) -> None:
    """处理普通知识库文件入库任务。"""

    document_id = str(task.get("document_id") or "")
    metadata = task.get("metadata") or {}
    document_repository = DocumentRepository()
    document = document_repository.get_document(document_id)
    if document is None:
        raise ValueError(f"文件不存在：{document_id}")

    reporter("parsing", 25)
    _index_document(
        document_repository,
        document,
        document_type=metadata.get("document_type"),
        split_strategy=metadata.get("split_strategy"),
        collection_name=metadata.get("collection_name"),
    )
    reporter("indexing", 90)


def process_training_ingest_task(task: dict[str, Any], reporter) -> None:
    """处理销售训练资料入库任务。"""

    V2SalesTrainingCoreService().knowledge_service.process_training_ingest_task(
        task=task,
        reporter=reporter,
        force_llm=False,
    )


def process_training_reparse_task(task: dict[str, Any], reporter) -> None:
    """处理销售训练资料重新切分任务。"""

    V2SalesTrainingCoreService().knowledge_service.process_training_ingest_task(
        task=task,
        reporter=reporter,
        force_llm=bool((task.get("metadata") or {}).get("use_llm_fallback", True)),
    )


def build_ingest_task_processors():
    """构建任务处理器映射。"""

    return {
        "document_ingest": process_document_ingest_task,
        "training_ingest": process_training_ingest_task,
        "training_reparse": process_training_reparse_task,
    }

