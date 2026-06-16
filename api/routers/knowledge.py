import json
import os
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
from api.services import (
    _document_to_response,
    _get_preview_file,
    _get_knowledge_store,
    _index_document,
    _move_upload_file,
    _recommend_upload_split_strategy,
    _remove_created_upload_dir,
    _sanitize_upload_filename,
    _save_preview_file,
    _sync_data_files_to_documents,
    _validate_file_type,
)
from utils.file_handler import get_file_md5_hex, pdf_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.qdrant_options import get_qdrant_collection_name
from utils.qdrant_options import normalize_qdrant_collection_name

router = APIRouter()

DEFAULT_PREVIEW_CHAR_LIMIT = 20000
MAX_PREVIEW_CHAR_LIMIT = 100000


def _is_path_inside(child_path: str, parent_path: str) -> bool:
    """判断 child_path 是否位于 parent_path 下。

    文件预览会读取 documents.file_path 指向的本地文件。
    这里先做路径白名单校验，避免数据库中异常路径导致接口读取项目外文件。
    """

    try:
        return os.path.commonpath([parent_path, child_path]) == parent_path
    except ValueError:
        return False


def _validate_preview_file_path(file_path: str) -> str:
    """校验知识库文件是否允许被预览，并返回绝对路径。"""

    target_path = os.path.abspath(file_path)
    uploads_root = os.path.abspath(get_abs_path("uploads"))
    data_root = os.path.abspath(get_abs_path("data"))
    preview_root = os.path.abspath(os.path.join(uploads_root, "_preview"))

    allowed_roots = (uploads_root, data_root)
    if not any(_is_path_inside(target_path, root) for root in allowed_roots):
        logger.warning(f"[知识库] 拒绝预览非知识库目录文件 路径={target_path}")
        raise HTTPException(status_code=403, detail="该文件路径不允许预览")

    if _is_path_inside(target_path, preview_root):
        logger.warning(f"[知识库] 拒绝预览临时上传目录文件 路径={target_path}")
        raise HTTPException(status_code=403, detail="临时上传文件不允许通过正式预览接口访问")

    if not os.path.isfile(target_path):
        raise HTTPException(status_code=404, detail="原始文件不存在，可能已被移动或删除")

    return target_path


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


def _read_knowledge_file_preview(document: dict, max_chars: int) -> dict:
    """按文件类型读取预览文本。"""

    file_path = _validate_preview_file_path(document["file_path"])
    file_type = str(document["file_type"]).lower().lstrip(".")

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

    raise HTTPException(status_code=400, detail=f"当前文件类型不支持预览：{file_type}")


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
    """确认预览结果，并正式写入 SQLite 和 Qdrant。"""

    upload_id = request.upload_id.strip()
    preview_dir, preview_path = _get_preview_file(upload_id)
    filename = _sanitize_upload_filename(os.path.basename(preview_path))
    file_type = _validate_file_type(filename)
    file_md5 = get_file_md5_hex(preview_path)
    file_size = os.path.getsize(preview_path)

    store = _get_knowledge_store()
    collection_name = normalize_qdrant_collection_name(request.collection_name)
    duplicate_document = store.find_active_document_by_md5(file_md5, collection_name=collection_name)
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
    - 它读取 documents 表中的 file_path。
    - 它只允许访问 uploads/ 和 data/ 下的知识库文件。
    - 它不访问 Qdrant，不触发 embedding，不改变索引。
    """

    logger.info(f"[知识库] 预览文件 文档编号={document_id} 最大字符数={max_chars}")
    store = _get_knowledge_store()
    document = store.get_document(document_id)

    if document is None or document["status"] == "deleted":
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

        VectorStoreService.delete_document_vectors(document_id, collection_name=document.get("collection_name"))
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
