"""为整个工程提供统一的绝对路径。"""

import os
from pathlib import Path


def get_project_root() -> str:
    """获取工程根目录。

    `core.utils` 被迁移后，工具文件不再直接位于项目根目录下一层。
    因此这里不再依赖固定层级，而是从当前文件逐级向上查找项目标记文件。
    """

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "config").is_dir() and (parent / "requirements.txt").is_file():
            return str(parent)
    return str(current_path.parents[2])


def get_abs_path(relative_path: str) -> str:
    """把项目内相对路径转换成绝对路径。"""

    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))
