import os
import shutil
from pathlib import Path

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _is_path_inside(child_path: Path, parent_path: Path) -> bool:
    """判断目标路径是否位于允许的父目录下，防止误删项目外文件。"""

    try:
        child_path.resolve().relative_to(parent_path.resolve())
        return True
    except ValueError:
        return False


def _remove_path(target_path: Path) -> bool:
    """删除单个文件或目录，返回是否实际执行了删除。"""

    if target_path.is_dir():
        shutil.rmtree(target_path)
        return True
    if target_path.is_file():
        target_path.unlink()
        return True
    return False


def delete_upload_path(file_path: str | None, *, document_id: str | None = None) -> bool:
    """删除 documents.file_path 对应的 uploads 原始文件目录。

    删除知识库记录时，数据库里保存的是具体文件路径，例如
    uploads/doc_xxx/a.pdf。为了避免目录里留下空壳，这里优先删除
    uploads/{document_id} 整个目录；没有 document_id 时再删除单个文件。
    """

    if not file_path:
        logger.info("[上传文件] 文件路径为空，跳过物理删除 文档编号=%s", document_id)
        return False

    uploads_root = Path(get_abs_path("uploads")).resolve()
    target_file = Path(file_path).resolve()

    if not _is_path_inside(target_file, uploads_root):
        logger.warning("[上传文件] 跳过非 uploads 目录文件删除 路径=%s 文档编号=%s", target_file, document_id)
        return False

    if document_id:
        target_path = uploads_root / document_id
    else:
        target_path = target_file

    target_path = target_path.resolve()
    if not _is_path_inside(target_path, uploads_root):
        logger.warning("[上传文件] 跳过不安全的上传路径删除 路径=%s 文档编号=%s", target_path, document_id)
        return False

    try:
        deleted = _remove_path(target_path)
    except OSError as exc:
        logger.warning("[上传文件] 物理删除失败 路径=%s 文档编号=%s 错误=%s", target_path, document_id, exc)
        return False

    if deleted:
        logger.info("[上传文件] 已物理删除 路径=%s 文档编号=%s", target_path, document_id)
    else:
        logger.info("[上传文件] 路径不存在，跳过物理删除 路径=%s 文档编号=%s", target_path, document_id)
    return deleted
