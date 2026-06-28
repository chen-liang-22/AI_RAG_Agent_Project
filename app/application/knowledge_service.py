"""知识资产应用服务。

这个服务使用外观模式，把上传预览、确认入库、文件预览、删除、重建索引收敛到一个入口。
路由层只负责 HTTP 参数和响应模型，不再直接编排 MinIO、Qdrant、MySQL。
"""

import json
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from api.schemas import (
    KnowledgeBulkReindexResponse,
    KnowledgeDeleteResponse,
    KnowledgeFilePreviewResponse,
    KnowledgeFileResponse,
    KnowledgeReindexResult,
    KnowledgeUploadConfirmRequest,
    KnowledgeUploadPreviewResponse,
    KnowledgeUploadRecommendRequest,
    KnowledgeUploadRecommendResponse,
    KnowledgeUploadResponse,
)
from app.application.knowledge.upload_preview_state import load_upload_preview_config
from app.application.knowledge.document_asset_service import DocumentAssetService
from app.application.knowledge.indexing_service import _index_document, _sync_data_files_to_documents
from app.application.knowledge.upload_preview_service import (
    _delete_preview_file,
    _get_preview_file,
    _promote_preview_file,
    _recommend_upload_split_strategy_or_fallback,
    _recommend_upload_split_strategy,
    _sanitize_upload_filename,
    _save_preview_file,
    _validate_file_type,
)
from app.infrastructure.adapters.file_storage_adapter import FileStorageAdapter
from app.infrastructure.adapters.vector_store_adapter import VectorStoreAdapter
from app.infrastructure.file_storage_service import get_file_storage_service
from app.infrastructure.repositories.dictionary_repository import DictionaryRepository
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.shared.document_response import DictionaryCodeSnapshot, document_to_response
from app.infrastructure.id_generator import new_id
from core.rag.file_processors import FileProcessorFactory
from core.utils.logger_handler import logger
from core.utils.qdrant_options import get_qdrant_collection_name, normalize_qdrant_collection_name

DEFAULT_PREVIEW_CHAR_LIMIT = 20000
MAX_PREVIEW_CHAR_LIMIT = 100000


class KnowledgeApplicationService:
    """知识资产外观服务。"""

    def __init__(
        self,
        *,
        store=None,
        file_storage: FileStorageAdapter | None = None,
        vector_adapter_factory=None,
        document_repository: DocumentRepository | None = None,
        dictionary_repository: DictionaryRepository | None = None,
    ):
        """初始化知识库应用服务。

        这里把 MinIO、Qdrant、documents 表和字典仓储组合起来。
        file_storage/vector_adapter_factory 支持注入，是为了测试和以后替换具体基础设施。
        """

        self.file_storage = file_storage or FileStorageAdapter()
        self.vector_adapter_factory = vector_adapter_factory or (lambda collection_name=None: VectorStoreAdapter(collection_name))
        # 文档查询和写入走 V2 仓储；store 参数只为旧测试注入保留，不再作为默认依赖。
        self.document_repository = document_repository or DocumentRepository(store=store)
        # 文档列表响应需要用字典做编码归一化；这里改走 V2 字典仓储，逐步移除旧 KnowledgeStore。
        self.dictionary_repository = dictionary_repository or DictionaryRepository()
        self._cached_document_dictionary_snapshot: DictionaryCodeSnapshot | None = None

    def preview_upload(self, file: UploadFile) -> KnowledgeUploadPreviewResponse:
        """上传文件到 MinIO 预览区，并返回结构识别结果。"""

        filename = _sanitize_upload_filename(file.filename)
        file_type = _validate_file_type(filename)
        upload_id = f"tmp_{uuid.uuid4().hex}"
        logger.info("[V2知识资产] 上传预览开始 文件名=%s 上传编号=%s", filename, upload_id)

        try:
            stored_file = _save_preview_file(file, filename, upload_id)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("[V2知识资产] 上传预览保存失败 文件名=%s 错误=%s", filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"上传文件保存失败：{exc}") from exc

        duplicate_document = self.document_repository.find_active_document_by_md5(stored_file.file_md5)
        try:
            with self.file_storage.downloaded_temp_file(
                bucket_name=stored_file.bucket_name,
                object_name=stored_file.object_name,
                filename=stored_file.filename,
            ) as file_path:
                preview_config = load_upload_preview_config()
                preview = self.vector_adapter_factory(None).preview_file(
                    filename=filename,
                    file_path=file_path,
                    sample_limit=preview_config.sample_text_chars,
                )
            recommendation = _recommend_upload_split_strategy_or_fallback(upload_id)
            preview.update({
                "document_type": recommendation["document_type"],
                "split_strategy": recommendation["split_strategy"],
                "confidence": recommendation["confidence"],
                "reasons": recommendation["reasons"],
                "llm_used": bool(recommendation.get("model_name")),
            })
        except Exception as exc:
            _delete_preview_file(upload_id)
            logger.error("[V2知识资产] 上传预览解析失败 文件名=%s 错误=%s", filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"文件预解析失败：{exc}") from exc

        return KnowledgeUploadPreviewResponse(
            upload_id=upload_id,
            filename=filename,
            file_type=file_type,
            file_size=stored_file.file_size,
            file_md5=stored_file.file_md5,
            duplicate=duplicate_document is not None,
            duplicate_document=document_to_response(duplicate_document, self._document_dictionary_snapshot()) if duplicate_document else None,
            detected_type=preview["document_type"],
            split_strategy=preview["split_strategy"],
            confidence=preview["confidence"],
            reasons=preview["reasons"],
            llm_used=preview["llm_used"],
            sample_text=preview["sample_text"],
        )

    def recommend_upload(self, request: KnowledgeUploadRecommendRequest) -> KnowledgeUploadRecommendResponse:
        """调用模型推荐上传文件的文档类型和切分策略。"""

        try:
            recommendation = _recommend_upload_split_strategy(request.upload_id)
        except HTTPException:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.error("[V2知识资产] 模型推荐切分方式失败 上传编号=%s 错误=%s", request.upload_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"模型推荐失败：{exc}") from exc
        return KnowledgeUploadRecommendResponse(**recommendation)

    def confirm_upload(self, request: KnowledgeUploadConfirmRequest) -> KnowledgeUploadResponse:
        """确认预览结果，并正式写入 documents 和 Qdrant。"""

        upload_id = request.upload_id.strip()
        preview_file = _get_preview_file(upload_id)
        filename = _sanitize_upload_filename(preview_file.filename)
        file_type = _validate_file_type(filename)
        collection_name = normalize_qdrant_collection_name(request.collection_name)
        logger.info(
            "[V2知识资产] 上传确认开始 上传编号=%s 文件名=%s Collection=%s 文档类型=%s 切分策略=%s",
            upload_id,
            filename,
            collection_name,
            request.document_type,
            request.split_strategy,
        )

        duplicate_document = self.document_repository.find_active_document_by_md5(preview_file.file_md5, collection_name=collection_name)
        if duplicate_document is not None:
            logger.info(
                "[V2知识资产] 上传确认命中重复文件 上传编号=%s 已有文档编号=%s Collection=%s",
                upload_id,
                duplicate_document["document_id"],
                collection_name,
            )
            _delete_preview_file(upload_id)
            return KnowledgeUploadResponse(
                status=self._dictionary_status("knowledge_result_status", "duplicate"),
                message="相同内容的文件已经存在，本次没有重复入库。",
                document=document_to_response(duplicate_document, self._document_dictionary_snapshot()),
            )

        document_id = new_id()
        try:
            stored_file = _promote_preview_file(upload_id, document_id)
        except Exception as exc:
            logger.error("[V2知识资产] 上传确认转正式对象失败 上传编号=%s 错误=%s", upload_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"临时文件转正式文件失败：{exc}") from exc

        document = self.document_repository.create_document(
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
            status=self._dictionary_status("document_status", "uploaded"),
            collection_name=collection_name,
            document_type=request.document_type,
            split_strategy=request.split_strategy,
        )
        logger.info("[V2知识资产] 文档资产记录创建完成 文档编号=%s 文件名=%s", document_id, filename)
        indexed_document = _index_document(
            self.document_repository,
            document,
            document_type=request.document_type,
            split_strategy=request.split_strategy,
            collection_name=collection_name,
        )
        return KnowledgeUploadResponse(
            status=self._dictionary_status("knowledge_result_status", "indexed"),
            message="文件已按确认配置写入知识库。",
            document=document_to_response(indexed_document, self._document_dictionary_snapshot()),
        )

    def list_files(self, *, include_training: bool = False) -> list[KnowledgeFileResponse]:
        """查询知识资产文件列表。"""

        logger.info("[V2知识资产] 查询文件列表 包含训练资料=%s", include_training)
        dictionary_snapshot = self._document_dictionary_snapshot()
        return [
            document_to_response(document, dictionary_snapshot)
            for document in self.document_repository.list_documents(include_training=include_training)
        ]

    def get_file(self, document_id: str) -> KnowledgeFileResponse:
        """查询单个知识资产文件。"""

        logger.info("[V2知识资产] 查询文件详情 文档编号=%s", document_id)
        document = self._active_document_or_404(document_id)
        return document_to_response(document, self._document_dictionary_snapshot())

    def preview_file(self, document_id: str, max_chars: int) -> KnowledgeFilePreviewResponse:
        """返回已入库文件预览数据。

        文本类文件由后端解码后返回 content，避免浏览器直接打开 MinIO TXT 时乱码；
        其他浏览器可直接展示的类型返回 file_url，前端在弹窗内嵌入预览。
        """

        logger.info("[V2知识资产] 预览文件开始 文档编号=%s 最大字符数=%s", document_id, max_chars)
        document = self._active_document_or_404(document_id)
        file_url = self._document_file_url(document)
        preview = self._build_file_preview(document, max_chars=max_chars)
        logger.info(
            "[V2知识资产] 预览文件完成 文档编号=%s 预览类型=%s 字符集=%s 是否截断=%s",
            document_id,
            preview["preview_type"],
            preview.get("charset") or "无",
            preview["truncated"],
        )
        return KnowledgeFilePreviewResponse(
            document=document_to_response(document, self._document_dictionary_snapshot()),
            preview_type=preview["preview_type"],
            content=preview["content"],
            truncated=preview["truncated"],
            page_count=preview.get("page_count"),
            file_url=file_url,
            charset=preview.get("charset"),
        )

    def delete_file(self, document_id: str) -> KnowledgeDeleteResponse:
        """按 document_id 删除文件资产。"""

        logger.info("[V2知识资产] 删除文件资产 文档编号=%s", document_id)
        result = DocumentAssetService(document_repository=self.document_repository).delete_document_asset(document_id)
        return KnowledgeDeleteResponse(status="deleted", document_id=result.document_id)

    def reindex_file(self, document_id: str) -> KnowledgeFileResponse:
        """重新解析并索引单个文件。"""

        logger.info("[V2知识资产] 重建单个文件索引 文档编号=%s", document_id)
        document = self._active_document_or_404(document_id)
        indexed_document = _index_document(self.document_repository, document, increment_version=True)
        return document_to_response(indexed_document, self._document_dictionary_snapshot())

    def reindex_all(self) -> KnowledgeBulkReindexResponse:
        """重建所有文件索引。"""

        logger.info("[V2知识资产] 全量重建索引开始")
        documents = self.document_repository.list_documents()
        results: list[KnowledgeReindexResult] = []
        succeeded = 0
        failed = 0
        vector_adapters: dict[str, VectorStoreAdapter] = {}

        for document in documents:
            document_id = document["document_id"]
            filename = document["filename"]
            try:
                collection_name = normalize_qdrant_collection_name(document.get("collection_name"))
                if collection_name not in vector_adapters:
                    vector_adapters[collection_name] = VectorStoreAdapter.recreate_collection(collection_name)
                logger.info("[V2知识资产] 全量重建单文件开始 文档编号=%s 文件名=%s Collection=%s", document_id, filename, collection_name)
                indexed_document = _index_document(
                    self.document_repository,
                    document,
                    increment_version=True,
                    vector_store=vector_adapters[collection_name].vector_service,
                    collection_name=collection_name,
                )
                succeeded += 1
                logger.info(
                    "[V2知识资产] 全量重建单文件完成 文档编号=%s 文件名=%s 分片数量=%s",
                    document_id,
                    filename,
                    indexed_document["chunk_count"],
                )
                results.append(KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status=self._dictionary_status("knowledge_result_status", "indexed"),
                    message=f"chunk_count={indexed_document['chunk_count']}",
                ))
            except Exception as exc:
                failed += 1
                message = exc.detail if isinstance(exc, HTTPException) else str(exc)
                logger.error("[V2知识资产] 全量重建单文件失败 文档编号=%s 文件名=%s 错误=%s", document_id, filename, message, exc_info=True)
                results.append(KnowledgeReindexResult(
                    document_id=document_id,
                    filename=filename,
                    status=self._dictionary_status("knowledge_result_status", "failed"),
                    message=str(message),
                ))

        status = self._dictionary_status("knowledge_result_status", "ok" if failed == 0 else "partial_failed")
        logger.info("[V2知识资产] 全量重建索引完成 总数=%s 成功=%s 失败=%s", len(documents), succeeded, failed)
        return KnowledgeBulkReindexResponse(total=len(documents), succeeded=succeeded, failed=failed, results=results, status=status)

    def reload_from_data_dir(self) -> dict:
        """扫描 data/ 目录并重建索引。"""

        logger.info("[V2知识资产] 知识库重载请求")
        try:
            _sync_data_files_to_documents(self.document_repository)
            response = self.reindex_all()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Knowledge reload failed: {exc}") from exc
        return {
            "status": response.status,
            "collection_name": get_qdrant_collection_name(),
            "total": response.total,
            "succeeded": response.succeeded,
            "failed": response.failed,
            "results": [item.model_dump() for item in response.results],
        }

    def _document_dictionary_snapshot(self) -> DictionaryCodeSnapshot:
        """从 V2 字典仓储构建文档响应需要的字典快照。"""

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

    def _dictionary_status(self, dictionary_code: str, item_code: str) -> str:
        """从字典表读取协议状态码。"""

        if not hasattr(self.dictionary_repository, "normalize_code"):
            return item_code
        try:
            return self.dictionary_repository.normalize_code(dictionary_code, item_code)
        except ValueError:
            return item_code

    def _active_document_or_404(self, document_id: str) -> dict:
        """读取未删除的文件，不存在时抛 404。"""

        document = self.document_repository.get_document(document_id)
        if document is None or document["status"] == self._dictionary_status("document_status", "deleted"):
            raise HTTPException(status_code=404, detail=f"文件不存在：{document_id}")
        return document

    @staticmethod
    def _is_text_like_file(file_type: str) -> bool:
        """判断文件是否适合在后端解析为文本后展示。"""

        return file_type.lower().lstrip(".") in {"txt", "md", "csv", "json", "log", "docx", "pdf"}

    @staticmethod
    def _looks_mojibake(text: str) -> bool:
        """粗略判断文本是否存在明显乱码。"""

        if not text:
            return False
        bad_chars = text.count("\ufffd") + text.count("�")
        suspicious_chars = sum(1 for char in text if "\ue000" <= char <= "\uf8ff")
        return (bad_chars + suspicious_chars) / max(len(text), 1) > 0.01

    @classmethod
    def _decode_text_bytes(cls, raw_data: bytes) -> tuple[str, str]:
        """按常见中文编码解码 TXT，避免浏览器直开 MinIO 时乱码。"""

        for charset in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                text = raw_data.decode(charset)
            except UnicodeDecodeError:
                continue
            if not cls._looks_mojibake(text):
                return text, charset
        return raw_data.decode("utf-8", errors="replace"), "utf-8-replace"

    def _document_file_url(self, document: dict) -> str:
        """读取或补全 MinIO HTTP 文件地址。"""

        file_url = str(document.get("public_url") or "").strip()
        if file_url:
            return file_url

        object_name = str(document.get("object_name") or "").strip()
        if not object_name:
            raise HTTPException(status_code=400, detail="文件缺少 MinIO 对象路径，请先完成历史文件迁移")
        return get_file_storage_service().client.get_public_url(
            object_name,
            bucket_name=document.get("bucket_name"),
        )

    def _build_file_preview(self, document: dict, *, max_chars: int) -> dict:
        """根据文件类型生成弹窗预览数据。"""

        file_type = str(document.get("file_type") or Path(str(document.get("filename") or "")).suffix).lower().lstrip(".")
        if not self._is_text_like_file(file_type):
            return {"preview_type": "file_url", "content": "", "truncated": False, "page_count": None, "charset": None}

        object_name = str(document.get("object_name") or "").strip()
        if not object_name:
            raise HTTPException(status_code=400, detail="文件缺少 MinIO 对象路径，请先完成历史文件迁移")

        with self.file_storage.downloaded_temp_file(
            bucket_name=document.get("bucket_name"),
            object_name=object_name,
            filename=str(document.get("filename") or "preview_file"),
        ) as file_path:
            if file_type in {"txt", "md", "csv", "json", "log"}:
                raw_data = Path(file_path).read_bytes()
                content, charset = self._decode_text_bytes(raw_data)
            else:
                documents = FileProcessorFactory.load_documents(file_path)
                content = "\n\n".join(item.page_content.strip() for item in documents if item.page_content.strip())
                charset = "document-parser"

        safe_max_chars = max(1000, min(MAX_PREVIEW_CHAR_LIMIT, max_chars))
        truncated = len(content) > safe_max_chars
        return {
            "preview_type": "text",
            "content": content[:safe_max_chars],
            "truncated": truncated,
            "page_count": None,
            "charset": charset,
        }
