"""测试环境配置。

生产和本地运行默认使用 MySQL；单元测试统一切回系统临时目录里的兼容库，
避免测试依赖开发机 MySQL 密码和服务状态，也避免在项目 storage 下生成本地库文件。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("DATABASE_TYPE", "sqlite")
os.environ.setdefault("AI_RAG_ALLOW_TEST_SQLITE", "1")
test_db_path = Path(tempfile.gettempdir()) / "ai_rag_agent_pytest_knowledge.db"
if test_db_path.exists():
    test_db_path.unlink()
os.environ.setdefault("SQLITE_DB_PATH", str(test_db_path))
