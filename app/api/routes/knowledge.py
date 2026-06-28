"""V2 知识资产接口。"""

from fastapi import APIRouter, File, Query, UploadFile

from api.schemas import (
    KnowledgeBulkReindexResponse,
    KnowledgeDeleteResponse,
    KnowledgeFilePreviewResponse,
    KnowledgeFileResponse,
    KnowledgeUploadConfirmRequest,
    KnowledgeUploadPreviewResponse,
    KnowledgeUploadRecommendRequest,
    KnowledgeUploadRecommendResponse,
    KnowledgeUploadResponse,
)
from app.application.knowledge_service import (
    DEFAULT_PREVIEW_CHAR_LIMIT,
    MAX_PREVIEW_CHAR_LIMIT,
    KnowledgeApplicationService,
)

router = APIRouter(prefix="/knowledge", tags=["V2 知识资产"])


def _service() -> KnowledgeApplicationService:
    """创建知识资产应用服务。"""

    return KnowledgeApplicationService()


@router.post("/upload/preview", response_model=KnowledgeUploadPreviewResponse)
def preview_knowledge_file(file: UploadFile = File(...)) -> KnowledgeUploadPreviewResponse:
    """上传文件并返回识别结果，确认后才正式入库。"""

    return _service().preview_upload(file)


@router.post("/upload/recommend", response_model=KnowledgeUploadRecommendResponse)
def recommend_knowledge_upload(request: KnowledgeUploadRecommendRequest) -> KnowledgeUploadRecommendResponse:
    """对临时上传文件调用模型推荐文档类型和切分策略。"""

    return _service().recommend_upload(request)


@router.post("/upload/confirm", response_model=KnowledgeUploadResponse)
def confirm_knowledge_file(request: KnowledgeUploadConfirmRequest) -> KnowledgeUploadResponse:
    """确认预览结果，并正式写入 MySQL 和 Qdrant。"""

    return _service().confirm_upload(request)


@router.get("/files", response_model=list[KnowledgeFileResponse])
def list_knowledge_files(include_training: bool = Query(False, description="是否包含销售训练集合，仅排查数据时使用")) -> list[KnowledgeFileResponse]:
    """查询知识资产文件列表。"""

    return _service().list_files(include_training=include_training)


@router.get("/files/{document_id}", response_model=KnowledgeFileResponse)
def get_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """查询单个知识资产文件详情。"""

    return _service().get_file(document_id)


@router.get("/files/{document_id}/preview", response_model=KnowledgeFilePreviewResponse)
def preview_indexed_knowledge_file(
    document_id: str,
    max_chars: int = Query(
        DEFAULT_PREVIEW_CHAR_LIMIT,
        ge=1000,
        le=MAX_PREVIEW_CHAR_LIMIT,
        description="最多返回的预览字符数，避免大文件一次性返回过多内容。",
    ),
) -> KnowledgeFilePreviewResponse:
    """预览已入库知识资产文件的原始文本内容。"""

    return _service().preview_file(document_id, max_chars)


@router.delete("/files/{document_id}", response_model=KnowledgeDeleteResponse)
def delete_knowledge_file(document_id: str) -> KnowledgeDeleteResponse:
    """删除知识资产文件。"""

    return _service().delete_file(document_id)


@router.post("/files/reindex-all", response_model=KnowledgeBulkReindexResponse)
def reindex_all_knowledge_files() -> KnowledgeBulkReindexResponse:
    """全量重建知识资产索引。"""

    return _service().reindex_all()


@router.post("/files/{document_id}/reindex", response_model=KnowledgeFileResponse)
def reindex_knowledge_file(document_id: str) -> KnowledgeFileResponse:
    """重建单个知识资产文件索引。"""

    return _service().reindex_file(document_id)


@router.post("/reload")
def reload_knowledge() -> dict:
    """扫描 data/ 目录并按 V2 知识资产流程重建索引。"""

    return _service().reload_from_data_dir()
