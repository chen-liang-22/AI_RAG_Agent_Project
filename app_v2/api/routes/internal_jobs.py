"""V2 内部任务接口。"""

import os

from fastapi import APIRouter, Header, HTTPException

from app_v2.application.knowledge.upload_preview_cleanup_service import PreviewUploadCleanupService
from utils.config_handler import load_env_file
from utils.logger_handler import logger
from utils.redis_client import get_redis_client


router = APIRouter(prefix="/internal/jobs", tags=["V2 内部任务"])


def _internal_job_token() -> str:
    """读取内部任务调用 Token。"""

    load_env_file()
    return os.getenv("INTERNAL_JOB_TOKEN", "").strip()


def _verify_internal_job_token(token: str | None) -> None:
    """校验内部任务调用 Token。"""

    expected_token = _internal_job_token()
    if not expected_token:
        logger.warning("[V2内部任务] INTERNAL_JOB_TOKEN 未配置")
        raise HTTPException(status_code=503, detail="内部任务 Token 未配置")
    if token != expected_token:
        logger.warning("[V2内部任务] 内部任务 Token 校验失败")
        raise HTTPException(status_code=403, detail="无权调用内部任务")


@router.post("/minio/cleanup-preview-uploads")
def cleanup_preview_uploads(x_internal_job_token: str | None = Header(default=None)) -> dict:
    """供 XXL-JOB 调用，清理 MinIO previews 前缀下的过期临时上传对象。"""

    _verify_internal_job_token(x_internal_job_token)
    redis_client = get_redis_client()
    with redis_client.lock(
            "upload_preview_cleanup",
            "minio_previews",
            timeout_seconds=1800,
            blocking_timeout_seconds=0,
    ) as acquired:
        if not acquired:
            logger.info("[V2内部任务] 清理预览上传对象跳过 原因=未获取到分布式锁")
            return {
                "status": "skipped",
                "locked": False,
                "skipped_reason": "未获取到分布式锁",
                "scanned_count": 0,
                "deleted_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "errors": [],
            }

        result = PreviewUploadCleanupService().cleanup_expired_previews()
        return {
            "status": "ok" if result.failed_count == 0 else "partial_failed",
            "locked": True,
            "skipped_reason": None,
            "scanned_count": result.scanned_count,
            "deleted_count": result.deleted_count,
            "skipped_count": result.skipped_count,
            "failed_count": result.failed_count,
            "errors": result.errors,
        }
