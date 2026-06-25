"""清理 uploads 目录中已经没有数据库引用的孤儿文件。

运行方式：
    python scripts/cleanup_orphan_uploads.py --dry-run
    python scripts/cleanup_orphan_uploads.py --delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from domain.entities import DocumentEntity, TrainingKnowledgeBatchEntity
from infrastructure.orm_session import orm_session_context
from utils.path_tool import get_abs_path


def _is_path_inside(child_path: Path, parent_path: Path) -> bool:
    """判断 child_path 是否位于 parent_path 下，避免清理项目外文件。"""

    try:
        child_path.resolve().relative_to(parent_path.resolve())
        return True
    except ValueError:
        return False


def _top_upload_dir(path_value: str | None, uploads_root: Path) -> Path | None:
    """把文件路径归一到 uploads 下第一层目录。"""

    if not path_value:
        return None
    target_path = Path(str(path_value)).resolve()
    if not _is_path_inside(target_path, uploads_root):
        return None
    relative_path = target_path.relative_to(uploads_root)
    if not relative_path.parts:
        return None
    return uploads_root / relative_path.parts[0]


def collect_referenced_upload_dirs() -> set[Path]:
    """收集数据库当前仍引用的 uploads 顶层目录。"""

    uploads_root = Path(get_abs_path("uploads")).resolve()
    referenced_dirs: set[Path] = set()

    with orm_session_context() as session:
        documents = session.scalars(select(DocumentEntity)).all()
        for document in documents:
            if document.status == "deleted":
                continue
            top_dir = _top_upload_dir(document.file_path, uploads_root)
            if top_dir is not None:
                referenced_dirs.add(top_dir.resolve())

        batches = session.scalars(select(TrainingKnowledgeBatchEntity)).all()
        for batch in batches:
            if batch.status == "deleted":
                continue
            top_dir = _top_upload_dir(batch.file_path, uploads_root)
            if top_dir is not None:
                referenced_dirs.add(top_dir.resolve())

            fallback_file = uploads_root / batch.batch_id / batch.source_file
            if fallback_file.exists():
                referenced_dirs.add((uploads_root / batch.batch_id).resolve())

    return referenced_dirs


def collect_orphan_upload_paths() -> list[Path]:
    """列出 uploads 下没有数据库引用的顶层文件或目录。"""

    uploads_root = Path(get_abs_path("uploads")).resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)
    referenced_dirs = collect_referenced_upload_dirs()

    orphan_paths: list[Path] = []
    for child_path in uploads_root.iterdir():
        resolved_child = child_path.resolve()
        if resolved_child not in referenced_dirs:
            orphan_paths.append(resolved_child)
    return sorted(orphan_paths, key=lambda item: str(item).lower())


def delete_paths(paths: list[Path]) -> int:
    """删除文件或目录，返回成功删除数量。"""

    uploads_root = Path(get_abs_path("uploads")).resolve()
    deleted_count = 0
    for path in paths:
        if not _is_path_inside(path, uploads_root):
            print(f"跳过不安全路径：{path}")
            continue
        if path.is_dir():
            shutil.rmtree(path)
            deleted_count += 1
            print(f"已删除目录：{path}")
        elif path.is_file():
            path.unlink()
            deleted_count += 1
            print(f"已删除文件：{path}")
        else:
            print(f"路径不存在，跳过：{path}")
    return deleted_count


def main() -> None:
    """脚本入口。"""

    parser = argparse.ArgumentParser(description="清理 uploads 中没有数据库引用的孤儿文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印待删除路径，不执行删除")
    parser.add_argument("--delete", action="store_true", help="执行真实删除")
    args = parser.parse_args()

    if args.dry_run == args.delete:
        raise SystemExit("请二选一：--dry-run 或 --delete")

    orphan_paths = collect_orphan_upload_paths()
    print(f"待清理数量：{len(orphan_paths)}")
    for path in orphan_paths:
        print(path)

    if args.delete:
        deleted_count = delete_paths(orphan_paths)
        print(f"实际删除数量：{deleted_count}")


if __name__ == "__main__":
    main()
