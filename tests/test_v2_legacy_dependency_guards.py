"""V2 旧代码物理退场守卫测试。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATHS = (
    "api/services",
    "api/routers",
    "training/services/sales_training_service.py",
    "rag/knowledge_store.py",
    "app_v2/infrastructure/repositories/knowledge_store_provider.py",
)

FORBIDDEN_IMPORT_SNIPPETS = (
    "from api.services",
    "import api.services",
    "from api.routers",
    "import api.routers",
    "from training.services.sales_training_service",
    "import training.services.sales_training_service",
    "from rag.knowledge_store",
    "import rag.knowledge_store",
    "KnowledgeStore(",
)

SCAN_ROOTS = (
    "api",
    "app_v2",
    "model",
    "rag",
    "scripts",
    "tests",
    "training",
)


def _python_files(root: Path) -> list[Path]:
    """列出需要检查的 Python 文件，跳过缓存目录。"""

    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_legacy_v1_business_files_are_removed():
    """旧业务入口文件必须物理删除，避免后续继续误接旧路径。"""

    offenders = [
        path
        for relative_path in FORBIDDEN_PATHS
        if (path := PROJECT_ROOT / relative_path).exists()
    ]

    assert offenders == []


def test_python_code_does_not_import_removed_legacy_modules():
    """生产和测试代码不能继续导入已删除的旧命名空间。"""

    offenders: list[str] = []
    for root_name in SCAN_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in _python_files(root):
            if path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_IMPORT_SNIPPETS:
                if forbidden in text:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {forbidden}")

    assert offenders == []
