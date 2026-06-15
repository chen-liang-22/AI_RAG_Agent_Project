import os
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    KnowledgeBulkReindexResponse,
    KnowledgeDeleteResponse,
    KnowledgeFileResponse,
    KnowledgeReindexResult,
    KnowledgeUploadConfirmRequest,
    KnowledgeUploadPreviewResponse,
    KnowledgeUploadResponse,
)
from api.services import (
    _document_to_response,
    _get_knowledge_store,
    _index_document,
    _move_upload_file,
    _remove_created_upload_dir,
    _sanitize_upload_filename,
    _save_preview_file,
    _sync_data_files_to_documents,
    _validate_file_type,
)
from utils.file_handler import get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.qdrant_options import get_qdrant_collection_name

router = APIRouter()


@router.post("/knowledge/upload/preview", response_model=KnowledgeUploadPreviewResponse)
def preview_knowledge_file(file: UploadFile = File(...)) -> KnowledgeUploadPreviewResponse:
    """上传文件并返回识别结果，等待用户确认后再正式入库。"""

    filename = _sanitize_upload_filename(file.filename)
    file_type = _validate_file_type(filename)
    upload_id = f"tmp_{uuid.uuid4().hex}"

    logger.info(f"[知识库] 上传预览开始 文件名={filename} 上传编号={upload_id}")

    try:
        file_path, file_size = _save_preview_file(file, filename, upload_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[知识库] 上传预览保存失败 文件名={filename} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc

    file_md5 = get_file_md5_hex(file_path)
    if not file_md5:
        _remove_created_upload_dir(os.path.dirname(file_path))
        raise HTTPException(status_code=500, detail="上传文件 MD5 计算失败")

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(file_md5)
    if duplicate_document is not None:
        _remove_created_upload_dir(os.path.dirname(file_path))
        return KnowledgeUploadPreviewResponse(
            upload_id=upload_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_md5=file_md5,
            duplicate=True,
            duplicate_document=_document_to_response(duplicate_document),
            detected_type="duplicate",
            split_strategy="duplicate",
            confidence=1.0,
            reasons=["相同内容的文件已经存在"],
            llm_used=False,
            sample_text="",
        )

    try:
        from rag.vector_store import VectorStoreService

        preview = VectorStoreService().preview_file(filename=filename, file_path=file_path)
    except Exception as exc:
        _remove_created_upload_dir(os.path.dirname(file_path))
        logger.error(f"[知识库] 上传预览解析失败 文件名={filename} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件预解析失败：{exc}") from exc

    return KnowledgeUploadPreviewResponse(
        upload_id=upload_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        file_md5=file_md5,
        duplicate=duplicate_document is not None,
        duplicate_document=_document_to_response(duplicate_document) if duplicate_document else None,
        detected_type=preview["document_type"],
        split_strategy=preview["split_strategy"],
        confidence=preview["confidence"],
        reasons=preview["reasons"],
        llm_used=preview["llm_used"],
        sample_text=preview["sample_text"],
    )


@router.post("/knowledge/upload/confirm", response_model=KnowledgeUploadResponse)
def confirm_knowledge_file(request: KnowledgeUploadConfirmRequest) -> KnowledgeUploadResponse:
    """确认预览结果，并正式写入 SQLite 和 Qdrant。"""

    upload_id = request.upload_id.strip()
    preview_dir = os.path.abspath(os.path.join(get_abs_path("uploads"), "_preview", upload_id))
    preview_root = os.path.abspath(os.path.join(get_abs_path("uploads"), "_preview"))

    if os.path.commonpath([preview_root, preview_dir]) != preview_root or not os.path.isdir(preview_dir):
        raise HTTPException(status_code=404, detail=f"临时上传不存在：{upload_id}")

    files = [name for name in os.listdir(preview_dir) if os.path.isfile(os.path.join(preview_dir, name))]
    if len(files) != 1:
        raise HTTPException(status_code=400, detail="临时上传文件状态异常")

    filename = _sanitize_upload_filename(files[0])
    file_type = _validate_file_type(filename)
    preview_path = os.path.join(preview_dir, filename)
    file_md5 = get_file_md5_hex(preview_path)
    file_size = os.path.getsize(preview_path)

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(file_md5)
    if duplicate_document is not None:
        _remove_created_upload_dir(preview_dir)
        return KnowledgeUploadResponse(
            status="duplicate",
            message="相同内容的文件已经存在，本次没有重复入库。",
            document=_document_to_response(duplicate_document),
        )

    document_id = f"doc_{uuid.uuid4().hex}"
    try:
        file_path = _move_upload_file(preview_path, document_id, filename)
        _remove_created_upload_dir(preview_dir)
    except Exception as exc:
        logger.error(f"[知识库] 上传确认移动文件失败 上传编号={upload_id} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"临时文件转正式文件失败：{exc}") from exc

    document = store.create_document(
        document_id=document_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_md5=file_md5,
        file_size=file_size,
        status="uploaded",
    )

    indexed_document = _index_document(
        store,
        document,
        document_type=request.document_type,
        split_strategy=request.split_strategy,
    )

    return KnowledgeUploadResponse(
        status="indexed",
        message="文件已按确认配置写入知识库。",
        document=_document_to_response(indexed_document),
    )


@router.get("/knowledge/files", response_model=list[KnowledgeFileResponse])
def list_knowledge_files() -> list[KnowledgeFileResponse]:
    """查询知识库文件列表。

    只返回 status != deleted 的文件。
    这个接口后续可以直接给前端知识库管理页面使用。
    """

    logger.info("[知识库] 查询文件列表")
    store = _get_knowledge_store()
    return [_document_to_response(document) for document in store.list_documents()]


@router.get("/knowledge/files/{document_id}", response_model=KnowledgeFileResponse)
def get_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """查询单个知识库文件详情。"""

    logger.info(f"[知识库] 查询文件详情 文档编号={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return _document_to_response(document)


@router.delete("/knowledge/files/{document_id}", response_model=KnowledgeDeleteResponse)
def delete_knowledge_file(document_id: str) -> KnowledgeDeleteResponse:
    """按 document_id 删除知识库文件。

    这里的“删除”是知识库层面的删除：
    - Qdrant 中该 document_id 的 points 会被删除。
    - SQLite documents 中该文件会标记为 deleted。

    原始上传文件暂时保留在 uploads/ 目录中，方便排查和审计。
    如果后续希望物理删除原始文件，可以在这个接口里追加文件删除逻辑。
    """

    logger.info(f"[知识库] 删除文件 文档编号={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        from rag.vector_store import VectorStoreService  # 延迟导入，只在删除 Qdrant points 时加载

        VectorStoreService.delete_document_vectors(document_id)
        store.mark_document_deleted(document_id)
    except Exception as exc:
        logger.error(f"[知识库] 删除文件失败 文档编号={document_id} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"知识库文件删除失败：{exc}") from exc

    return KnowledgeDeleteResponse(status="deleted", document_id=document_id)


@router.post("/knowledge/files/reindex-all", response_model=KnowledgeBulkReindexResponse)
def reindex_all_knowledge_files() -> KnowledgeBulkReindexResponse:
    """清空 Qdrant collection，并重建所有 active 知识库文件的索引。

    这个接口用于把旧的粗糙 chunk 数据迁移到新的结构化知识单元。
    注意：
    - 会先删除并重建当前 Qdrant collection，清理无 document_id 的旧 points。
    - 会重新读取 uploads/ 中的原始文件。
    - 会重新调用 embedding。
    - 文件多时会比较慢，也会消耗模型调用额度。
    """

    logger.info("[知识库] 全量重建索引开始")
    store = _get_knowledge_store()
    documents = store.list_documents()
    results: list[KnowledgeReindexResult] = []
    succeeded = 0
    failed = 0

    try:
        from rag.vector_store import VectorStoreService

        vector_store = VectorStoreService.recreate_collection_service()
    except Exception as exc:
        logger.error(f"[知识库] 重建向量集合失败 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Qdrant collection 重建失败：{exc}") from exc

    for document in documents:
        document_id = document["document_id"]
        filename = document["filename"]

        try:
            indexed_document = _index_document(
                store,
                document,
                increment_version=True,
                vector_store=vector_store,
            )
            succeeded += 1
            results.append(
                KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status="indexed",
                    message=f"chunk_count={indexed_document['chunk_count']}",
                )
            )
        except Exception as exc:
            failed += 1
            message = exc.detail if isinstance(exc, HTTPException) else str(exc)
            results.append(
                KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status="failed",
                    message=str(message),
                )
            )

    status = "ok" if failed == 0 else "partial_failed"
    return KnowledgeBulkReindexResponse(
        status=status,
        total=len(documents),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.post("/knowledge/files/{document_id}/reindex", response_model=KnowledgeFileResponse)
def reindex_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """重新解析并索引某个知识库文件。

    使用场景：
    - 调整了 chunk_size/chunk_overlap 后，希望重新切分。
    - 后续升级了结构化切分规则，希望重新生成 Qdrant payload。
    - Qdrant 中某个文件的向量异常，需要按 document_id 重建。

    reindex 会递增 documents.version。
    新写入 Qdrant 的 payload 里也会带上新的 version。
    """

    logger.info(f"[知识库] 重建单个文件索引 文档编号={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    indexed_document = _index_document(store, document, increment_version=True)
    return _document_to_response(indexed_document)


@router.post("/knowledge/reload")
def reload_knowledge() -> dict:
    """扫描 data/ 目录并按新结构重建知识库。

    旧版 reload 只把 data/ 文件切块写进 Qdrant，不会写 documents、
    旧知识表。现在这个接口会把 data/ 文件也同步到
    documents 表，然后复用 reindex-all 的新流程，保证结构化 FAQ 查询可用。
    """

    logger.info("[接口] 知识库重载请求")  # 记录知识库重载请求
    store = _get_knowledge_store()

    try:
        from rag.vector_store import VectorStoreService  # 延迟导入，只有重载知识库时才加载向量库服务

        _sync_data_files_to_documents(store)
        documents = store.list_documents()
        vector_store = VectorStoreService.recreate_collection_service()

        results: list[dict] = []
        succeeded = 0
        failed = 0
        for document in documents:
            try:
                indexed_document = _index_document(
                    store,
                    document,
                    increment_version=True,
                    vector_store=vector_store,
                )
                succeeded += 1
                results.append(
                    {
                        "document_id": indexed_document["document_id"],
                        "filename": indexed_document["filename"],
                        "status": "indexed",
                        "chunk_count": indexed_document["chunk_count"],
                    }
                )
            except Exception as exc:
                failed += 1
                message = exc.detail if isinstance(exc, HTTPException) else str(exc)
                results.append(
                    {
                        "document_id": document["document_id"],
                        "filename": document["filename"],
                        "status": "failed",
                        "message": str(message),
                    }
                )
    except Exception as exc:
        # 知识库加载失败时返回 500，前端会弹出错误提示。
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc

    return {
        "status": "ok" if failed == 0 else "partial_failed",
        "collection_name": get_qdrant_collection_name(),
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
