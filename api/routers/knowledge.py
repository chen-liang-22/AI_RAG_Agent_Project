import json
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.schemas import (
    KnowledgeBulkReindexResponse,
    KnowledgeDeleteResponse,
    KnowledgeFileResponse,
    KnowledgeFilePreviewResponse,
    KnowledgeReindexResult,
    KnowledgeUploadConfirmRequest,
    KnowledgeUploadPreviewResponse,
    KnowledgeUploadRecommendRequest,
    KnowledgeUploadRecommendResponse,
    KnowledgeUploadResponse,
)
from api.services.common_services import _document_to_response, _get_knowledge_store, build_document_dictionary_snapshot
from api.services.document_asset_service import DocumentAssetService
from api.services.indexing_services import _index_document, _sync_data_files_to_documents
from api.services.upload_services import (
    _delete_preview_file,
    _get_preview_file,
    _promote_preview_file,
    _recommend_upload_split_strategy,
    _sanitize_upload_filename,
    _save_preview_file,
    _validate_file_type,
    load_upload_preview_config,
)
from infrastructure.file_storage_service import get_file_storage_service
from rag.file_processors import FileProcessorFactory
from utils.file_handler import pdf_loader
from utils.logger_handler import logger
from utils.qdrant_options import get_qdrant_collection_name
from utils.qdrant_options import normalize_qdrant_collection_name

router = APIRouter()

DEFAULT_PREVIEW_CHAR_LIMIT = 20000
MAX_PREVIEW_CHAR_LIMIT = 100000


def _dictionary_status(dictionary_code: str, item_code: str) -> str:
    """从字典表读取协议状态码，避免接口里直接散落未校验的状态值。"""

    store = _get_knowledge_store()
    if hasattr(store, "normalize_dictionary_code"):
        return store.normalize_dictionary_code(dictionary_code, item_code)
    return item_code


def _read_text_file_preview(file_path: str, max_chars: int) -> tuple[str, bool]:
    """读取 TXT 文件预览内容。

    TXT 预览不走 LangChain loader，直接按 UTF-8 读取。
    errors="replace" 可以避免少量异常字符导致整个预览失败。
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        content = file.read(max_chars + 1)

    truncated = len(content) > max_chars
    return content[:max_chars], truncated


def _append_preview_page(
        parts: list[str],
        total_chars: int,
        page_title: str,
        page_content: str,
        max_chars: int,
) -> tuple[int, bool]:
    """把一页 PDF 文本追加到预览内容中，超出 max_chars 时截断。"""

    page_text = f"{page_title}\n{page_content.strip()}"
    if not page_content.strip():
        return total_chars, False

    candidate = f"\n\n{page_text}" if parts else page_text
    remaining_chars = max_chars - total_chars
    if len(candidate) > remaining_chars:
        parts.append(candidate[:remaining_chars])
        return max_chars, True

    parts.append(candidate)
    return total_chars + len(candidate), False


def _read_pdf_file_preview(file_path: str, max_chars: int) -> tuple[str, bool, int]:
    """读取 PDF 文件的文本预览。

    PDF 在浏览器端直接预览需要额外的文件流接口和鉴权设计。
    当前接口先返回抽取后的文本，适合知识库管理里快速核对文件内容。
    """

    documents = pdf_loader(file_path)
    parts: list[str] = []
    total_chars = 0
    truncated = False

    for page_index, document in enumerate(documents, start=1):
        page_no = int(document.metadata.get("page", page_index - 1)) + 1
        total_chars, truncated = _append_preview_page(
            parts=parts,
            total_chars=total_chars,
            page_title=f"第 {page_no} 页",
            page_content=document.page_content,
            max_chars=max_chars,
        )
        if truncated:
            break

    return "".join(parts), truncated, len(documents)


def _read_document_file_preview(file_path: str, max_chars: int) -> tuple[str, bool]:
    """读取 DOCX 等文档类文件的文本预览。"""

    documents = FileProcessorFactory.load_documents(file_path)
    content = "\n\n".join(document.page_content.strip() for document in documents if document.page_content.strip())
    truncated = len(content) > max_chars
    return content[:max_chars], truncated


def _read_knowledge_file_preview(document: dict, max_chars: int) -> dict:
    """从 MinIO 下载原文件并按文件类型读取预览文本。"""

    file_type = str(document["file_type"]).lower().lstrip(".")
    object_name = str(document.get("object_name") or "").strip()
    if not object_name:
        raise HTTPException(status_code=400, detail="文件缺少 MinIO 对象路径，请先完成历史文件迁移")

    with get_file_storage_service().downloaded_temp_file(
            bucket_name=document.get("bucket_name"),
            object_name=object_name,
            filename=document["filename"],
    ) as file_path:
        if file_type == "txt":
            content, truncated = _read_text_file_preview(file_path, max_chars)
            return {
                "preview_type": "text",
                "content": content,
                "truncated": truncated,
                "page_count": None,
            }

        if file_type == "pdf":
            content, truncated, page_count = _read_pdf_file_preview(file_path, max_chars)
            return {
                "preview_type": "pdf_text",
                "content": content,
                "truncated": truncated,
                "page_count": page_count,
            }

        if file_type == "docx":
            content, truncated = _read_document_file_preview(file_path, max_chars)
            return {
                "preview_type": "document_text",
                "content": content,
                "truncated": truncated,
                "page_count": None,
            }

    raise HTTPException(status_code=400, detail=f"当前文件类型不支持预览：{file_type}")


@router.post("/knowledge/upload/preview", response_model=KnowledgeUploadPreviewResponse)
def preview_knowledge_file(file: UploadFile = File(...)) -> KnowledgeUploadPreviewResponse:
    """上传文件并返回识别结果，等待用户确认后再正式入库。"""

    filename = _sanitize_upload_filename(file.filename)
    file_type = _validate_file_type(filename)
    upload_id = f"tmp_{uuid.uuid4().hex}"

    logger.info(f"[知识库] 上传预览开始 文件名={filename} 上传编号={upload_id}")

    try:
        stored_file = _save_preview_file(file, filename, upload_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[知识库] 上传预览保存失败 文件名={filename} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc

    store = _get_knowledge_store()
    duplicate_document = store.find_active_document_by_md5(stored_file.file_md5)

    try:
        from infrastructure.vector_store_service import VectorStoreService

        with get_file_storage_service().downloaded_temp_file(
                bucket_name=stored_file.bucket_name,
                object_name=stored_file.object_name,
                filename=stored_file.filename,
        ) as file_path:
            preview_config = load_upload_preview_config()
            preview = VectorStoreService().preview_file(
                filename=filename,
                file_path=file_path,
                sample_limit=preview_config.sample_text_chars,
            )
    except Exception as exc:
        _delete_preview_file(upload_id)
        logger.error(f"[知识库] 上传预览解析失败 文件名={filename} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件预解析失败：{exc}") from exc

    return KnowledgeUploadPreviewResponse(
        upload_id=upload_id,
        filename=filename,
        file_type=file_type,
        file_size=stored_file.file_size,
        file_md5=stored_file.file_md5,
        duplicate=duplicate_document is not None,
        duplicate_document=_document_to_response(duplicate_document) if duplicate_document else None,
        detected_type=preview["document_type"],
        split_strategy=preview["split_strategy"],
        confidence=preview["confidence"],
        reasons=preview["reasons"],
        llm_used=preview["llm_used"],
        sample_text=preview["sample_text"],
    )


@router.post("/knowledge/upload/recommend", response_model=KnowledgeUploadRecommendResponse)
def recommend_knowledge_upload(request: KnowledgeUploadRecommendRequest) -> KnowledgeUploadRecommendResponse:
    """对临时上传文件调用模型推荐文档类型和切分策略。"""

    try:
        recommendation = _recommend_upload_split_strategy(request.upload_id)
    except HTTPException:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.error(f"[知识库] 模型推荐切分方式失败 上传编号={request.upload_id} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型推荐失败：{exc}") from exc

    return KnowledgeUploadRecommendResponse(**recommendation)


@router.post("/knowledge/upload/confirm", response_model=KnowledgeUploadResponse)
def confirm_knowledge_file(request: KnowledgeUploadConfirmRequest) -> KnowledgeUploadResponse:
    """确认预览结果，并正式写入 MySQL 业务表和 Qdrant。"""

    upload_id = request.upload_id.strip()
    preview_file = _get_preview_file(upload_id)
    filename = _sanitize_upload_filename(preview_file.filename)
    file_type = _validate_file_type(filename)

    store = _get_knowledge_store()
    collection_name = normalize_qdrant_collection_name(request.collection_name)
    duplicate_document = store.find_active_document_by_md5(preview_file.file_md5, collection_name=collection_name)
    if duplicate_document is not None:
        _delete_preview_file(upload_id)
        return KnowledgeUploadResponse(
            status=_dictionary_status("knowledge_result_status", "duplicate"),
            message="相同内容的文件已经存在，本次没有重复入库。",
            document=_document_to_response(duplicate_document),
        )

    document_id = f"doc_{uuid.uuid4().hex}"
    try:
        stored_file = _promote_preview_file(upload_id, document_id)
    except Exception as exc:
        logger.error(f"[知识库] 上传确认转正式对象失败 上传编号={upload_id} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"临时文件转正式文件失败：{exc}") from exc

    document = store.create_document(
        document_id=document_id,
        filename=filename,
        file_path=stored_file.file_path,
        file_type=file_type,
        file_md5=stored_file.file_md5,
        file_size=stored_file.file_size,
        storage_type="minio",
        bucket_name=stored_file.bucket_name,
        object_name=stored_file.object_name,
        public_url=stored_file.public_url,
        status=_dictionary_status("document_status", "uploaded"),
        collection_name=collection_name,
        document_type=request.document_type,
        split_strategy=request.split_strategy,
    )

    indexed_document = _index_document(
        store,
        document,
        document_type=request.document_type,
        split_strategy=request.split_strategy,
        collection_name=collection_name,
    )

    return KnowledgeUploadResponse(
        status=_dictionary_status("knowledge_result_status", "indexed"),
        message="文件已按确认配置写入知识库。",
        document=_document_to_response(indexed_document),
    )


@router.get("/knowledge/files", response_model=list[KnowledgeFileResponse])
def list_knowledge_files(
        include_training: bool = Query(False, description="是否包含销售训练集合，仅排查数据时使用"),
) -> list[KnowledgeFileResponse]:
    """查询知识库文件列表。

    只返回 status != deleted 的文件。
    这个接口后续可以直接给前端知识库管理页面使用。
    """

    logger.info("[知识库] 查询文件列表 包含训练资料=%s", include_training)
    store = _get_knowledge_store()
    dictionary_snapshot = build_document_dictionary_snapshot(store)
    return [
        _document_to_response(document, dictionary_snapshot)
        for document in store.list_documents(include_training=include_training)
    ]


@router.get("/knowledge/files/{document_id}", response_model=KnowledgeFileResponse)
def get_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """查询单个知识库文件详情。"""

    logger.info(f"[知识库] 查询文件详情 文档编号={document_id}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == _dictionary_status("document_status", "deleted"):
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    return _document_to_response(document)


@router.get("/knowledge/files/{document_id}/preview", response_model=KnowledgeFilePreviewResponse)
def preview_indexed_knowledge_file(
        document_id: str,
        max_chars: int = Query(
            DEFAULT_PREVIEW_CHAR_LIMIT,
            ge=1000,
            le=MAX_PREVIEW_CHAR_LIMIT,
            description="最多返回的预览字符数，避免大文件一次性返回过多内容。",
        ),
) -> KnowledgeFilePreviewResponse:
    """预览已入库知识库文件的原始文本内容。

    这个接口服务于知识库管理页面：
    - 它读取 documents 表中的 MinIO 对象路径。
    - 它不访问 Qdrant，不触发 embedding，不改变索引。
    """

    logger.info(f"[知识库] 预览文件 文档编号={document_id} 最大字符数={max_chars}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == _dictionary_status("document_status", "deleted"):
        raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")

    try:
        preview = _read_knowledge_file_preview(document, max_chars)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[知识库] 预览文件失败 文档编号={document_id} 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文件预览失败：{exc}") from exc

    return KnowledgeFilePreviewResponse(
        document=_document_to_response(document),
        preview_type=preview["preview_type"],
        content=preview["content"],
        truncated=preview["truncated"],
        page_count=preview["page_count"],
    )


@router.delete("/knowledge/files/{document_id}", response_model=KnowledgeDeleteResponse)
def delete_knowledge_file(document_id: str) -> KnowledgeDeleteResponse:
    """? document_id ?????????????"""

    logger.info("[???] ???? ????=%s", document_id)
    result = DocumentAssetService().delete_document_asset(document_id)
    return KnowledgeDeleteResponse(status="deleted", document_id=result.document_id)


@router.post("/knowledge/files/reindex-all", response_model=KnowledgeBulkReindexResponse)
def reindex_all_knowledge_files() -> KnowledgeBulkReindexResponse:
    """清空 Qdrant collection，并重建所有 active 知识库文件的索引。

    这个接口用于把旧的粗糙 chunk 数据迁移到新的结构化知识单元。
    注意：
    - 会先删除并重建当前 Qdrant collection，清理无 document_id 的旧 points。
    - 会从 MinIO 临时下载原始文件并重新解析。
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
        from infrastructure.vector_store_service import VectorStoreService

        vector_stores: dict[str, VectorStoreService] = {}
    except Exception as exc:
        logger.error(f"[知识库] 重建向量集合失败 错误={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Qdrant collection 重建失败：{exc}") from exc

    for document in documents:
        document_id = document["document_id"]
        filename = document["filename"]

        try:
            collection_name = normalize_qdrant_collection_name(document.get("collection_name"))
            if collection_name not in vector_stores:
                vector_stores[collection_name] = VectorStoreService.recreate_collection_service(collection_name)

            indexed_document = _index_document(
                store,
                document,
                increment_version=True,
                vector_store=vector_stores[collection_name],
                collection_name=collection_name,
            )
            succeeded += 1
            results.append(
                KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status=_dictionary_status("knowledge_result_status", "indexed"),
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
                    status=_dictionary_status("knowledge_result_status", "failed"),
                    message=str(message),
                )
            )

    status = _dictionary_status("knowledge_result_status", "ok" if failed == 0 else "partial_failed")
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

    if document is None or document["status"] == _dictionary_status("document_status", "deleted"):
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
        from infrastructure.vector_store_service import VectorStoreService  # 延迟导入，只有重载知识库时才加载向量库服务

        _sync_data_files_to_documents(store)
        documents = store.list_documents()
        vector_stores: dict[str, VectorStoreService] = {}

        results: list[dict] = []
        succeeded = 0
        failed = 0
        for document in documents:
            try:
                collection_name = normalize_qdrant_collection_name(document.get("collection_name"))
                if collection_name not in vector_stores:
                    vector_stores[collection_name] = VectorStoreService.recreate_collection_service(collection_name)

                indexed_document = _index_document(
                    store,
                    document,
                    increment_version=True,
                    vector_store=vector_stores[collection_name],
                    collection_name=collection_name,
                )
                succeeded += 1
                results.append(
                    {
                        "document_id": indexed_document["document_id"],
                        "filename": indexed_document["filename"],
                        "status": _dictionary_status("knowledge_result_status", "indexed"),
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
                        "status": _dictionary_status("knowledge_result_status", "failed"),
                        "message": str(message),
                    }
                )
    except Exception as exc:
        # 知识库加载失败时返回 500，前端会弹出错误提示。
        raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc

    return {
        "status": _dictionary_status("knowledge_result_status", "ok" if failed == 0 else "partial_failed"),
        "collection_name": get_qdrant_collection_name(),
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
