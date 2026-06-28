"""知识库索引应用函数。

这个模块承接 KnowledgeApplicationService 的入库动作：
- 把 documents 表中文件解析成文本片段；
- 调用向量库服务写入 Qdrant；
- 回写 documents 表状态、chunk_count、结构类型和切分策略。

函数名前面的下划线表示“模块内部函数”，目前为了兼容重构前调用方式暂时保留。
"""

import os

from fastapi import HTTPException

from app.shared.document_response import normalize_document_structure_type, normalize_split_strategy
from app.infrastructure.file_storage_service import get_file_storage_service
from app.infrastructure.id_generator import new_id
from core.utils.config_handler import knowledge_manifest_conf, qdrant_conf
from core.utils.file_handler import get_file_md5_hex, listdir_with_allowed_type
from core.utils.logger_handler import logger
from core.utils.path_tool import get_abs_path
from core.utils.qdrant_options import normalize_qdrant_collection_name


def _index_document(
        store,
        document: dict,
        *,
        increment_version: bool = False,
        vector_store=None,
        document_type: str | None = None,
        split_strategy: str | None = None,
        collection_name: str | None = None,
) -> dict:
    """把 documents 表中的文件解析、分片、向量化并写入 Qdrant。

    这个函数把“修改业务数据库状态”和“写 Qdrant”封装在一起，上传和重建索引都复用它。
    """

    document_id = document["document_id"]
    final_collection_name = normalize_qdrant_collection_name(collection_name or document.get("collection_name"))
    final_split_strategy = normalize_split_strategy(split_strategy or document.get("split_strategy"))
    final_document_type = normalize_document_structure_type(
        document_type or document.get("document_type"),
        final_split_strategy,
    )
    logger.info(
        "[知识库] 文件入库开始 文档编号=%s 文件名=%s Collection=%s 文档类型=%s 切分策略=%s 是否递增版本=%s",
        document_id,
        document.get("filename"),
        final_collection_name,
        final_document_type,
        final_split_strategy,
        increment_version,
    )
    store.update_document_status(
        document_id,
        "indexing",
        error_message=None,
        increment_version=increment_version,
        collection_name=final_collection_name,
        document_type=final_document_type,
        split_strategy=final_split_strategy,
    )

    indexing_document = store.get_document(document_id)
    if indexing_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        if vector_store is None:
            from app.infrastructure.vector_store_service import VectorStoreService

            vector_store = VectorStoreService(collection_name=final_collection_name)

        chunk_count = vector_store.index_file(
            indexing_document,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
        logger.info(
            "[知识库] 向量写入完成 文档编号=%s Collection=%s 分片数量=%s",
            document_id,
            final_collection_name,
            chunk_count,
        )
        store.update_document_status(
            document_id,
            "indexed",
            chunk_count=chunk_count,
            error_message=None,
            collection_name=final_collection_name,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("[知识库] 文件入库失败 文档编号=%s 错误=%s", document_id, exc, exc_info=True)
        store.update_document_status(
            document_id,
            "failed",
            error_message=str(exc),
            collection_name=final_collection_name,
            document_type=final_document_type,
            split_strategy=final_split_strategy,
        )
        raise HTTPException(status_code=500, detail=f"知识库入库失败：{exc}") from exc

    indexed_document = store.get_document(document_id)
    if indexed_document is None:
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    logger.info("[知识库] 文件入库完成 文档编号=%s 状态=%s", document_id, indexed_document["status"])
    return indexed_document


def _load_data_manifest() -> dict:
    """读取内置知识文件清单，缺失时返回空配置。"""

    return {
        "defaults": knowledge_manifest_conf.get("defaults") or {},
        "files": knowledge_manifest_conf.get("files") or {},
    }


def _data_manifest_entry(filename: str, manifest: dict) -> dict:
    """合并 manifest 默认项和单文件项，得到 data 文件的入库配置。"""

    defaults = manifest.get("defaults") or {}
    files = manifest.get("files") or {}
    entry = files.get(filename) or {}
    return {
        "collection_name": normalize_qdrant_collection_name(entry.get("collection_name") or defaults.get("collection_name")),
        "document_type": normalize_document_structure_type(
            entry.get("document_type") or defaults.get("document_type"),
            entry.get("split_strategy") or defaults.get("split_strategy"),
        ),
        "split_strategy": normalize_split_strategy(entry.get("split_strategy") or defaults.get("split_strategy")),
    }


def _sync_data_files_to_documents(store) -> list[dict]:
    """把 data/ 目录中的内置知识文件同步到 documents 表。"""

    data_path = get_abs_path(qdrant_conf["data_path"])
    allowed_types = tuple(qdrant_conf["allow_knowledge_file_type"])
    file_paths = listdir_with_allowed_type(data_path, allowed_types)
    manifest = _load_data_manifest()
    documents: list[dict] = []
    logger.info("[知识库] 内置 data 文件同步开始 路径=%s 文件数=%s", data_path, len(file_paths))

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        manifest_entry = _data_manifest_entry(filename, manifest)
        file_type = os.path.splitext(filename)[1].lower().lstrip(".")
        file_md5 = get_file_md5_hex(file_path)

        if not file_md5:
            logger.warning("[知识库] 跳过无法计算 MD5 的内置知识文件 路径=%s", file_path)
            continue

        existing_document = store.find_active_document_by_md5(
            file_md5,
            collection_name=manifest_entry["collection_name"],
        )
        if existing_document is not None:
            logger.info(
                "[知识库] 内置文件已存在，刷新入库配置 文件名=%s 文档编号=%s Collection=%s",
                filename,
                existing_document["document_id"],
                manifest_entry["collection_name"],
            )
            store.update_document_status(
                existing_document["document_id"],
                existing_document["status"],
                collection_name=manifest_entry["collection_name"],
                document_type=manifest_entry["document_type"],
                split_strategy=manifest_entry["split_strategy"],
            )
            existing_document = store.get_document(existing_document["document_id"]) or existing_document
            documents.append(existing_document)
            continue

        document_id = new_id()
        stored_file = get_file_storage_service().save_local_file(
            file_path=file_path,
            filename=filename,
            prefix="documents",
            owner_id=document_id,
        )
        documents.append(
            store.create_document(
                document_id=document_id,
                filename=filename,
                file_path=stored_file.file_path,
                file_type=file_type,
                file_md5=file_md5,
                file_size=os.path.getsize(file_path),
                storage_type="minio",
                bucket_name=stored_file.bucket_name,
                object_name=stored_file.object_name,
                public_url=stored_file.public_url,
                status="uploaded",
                collection_name=manifest_entry["collection_name"],
                document_type=manifest_entry["document_type"],
                split_strategy=manifest_entry["split_strategy"],
            )
        )
        logger.info(
            "[知识库] 内置文件已同步到 MinIO 文件名=%s 文档编号=%s Collection=%s",
            filename,
            document_id,
            manifest_entry["collection_name"],
        )

    logger.info("[知识库] 内置 data 文件同步完成 文件数=%s", len(documents))
    return documents
