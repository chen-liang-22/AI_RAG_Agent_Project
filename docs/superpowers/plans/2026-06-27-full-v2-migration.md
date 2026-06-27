# 全量 V2 化三周迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 3 周节奏内把后端生产链路迁到 `app_v2`，让销售训练、知识上传索引、聊天 RAG、考试和旧依赖退场都有可测试的完成标准。

**Architecture:** 路由层只保留 HTTP 入参和响应模型，真实业务进入 `app_v2/application`。MySQL 访问统一走 `app_v2/infrastructure/repositories`，MinIO、Qdrant、LLM、Redis 访问统一走 adapter 或小型基础设施服务；旧服务先被测试封锁，再按业务域替换。

**Tech Stack:** Python 3.12, FastAPI, Pydantic, SQLAlchemy ORM, Pytest, MinIO, Qdrant, Redis, Vue/Vite frontend contract tests.

---

## Scope Check

`05-全量V2化三周迁移设计.md` 覆盖多个子系统。执行时不要一次性开全量改动，按本文 5 个任务顺序推进：

1. 先加旧依赖守卫测试。
2. 第 1 周迁销售训练。
3. 第 2 周迁知识上传、文件资产和索引。
4. 第 3 周前半迁聊天 RAG 与考试。
5. 第 3 周后半做旧依赖退场、回归和文档同步。

每个任务结束都要能独立验证。每个任务提交一次，提交前确认没有回退用户已有改动。

## File Structure

### Create

- `tests/test_v2_legacy_dependency_guards.py`  
  用静态扫描和导入守卫锁住 `app_v2` 不能继续依赖旧业务服务。

- `tests/test_v2_training_no_legacy_service.py`  
  用 monkeypatch 让 `SalesTrainingService` 爆炸，验证 V2 training 服务不再委托旧大类。

- `app_v2/application/training/response_mappers.py`  
  承接旧 `SalesTrainingService` 的 `_plan_summary`、`_plan_detail_response`、`_batch_response`、`_session_response`、`_score_response` 等响应转换。

- `app_v2/application/training/llm_json_service.py`  
  承接旧 `_invoke_json`、`_parse_json_object` 和 fallback JSON 处理。

- `app_v2/application/training/evidence_service.py`  
  承接训练资料检索、训练会话证据收集和 training collection 查询。

- `app_v2/application/knowledge/upload_preview_service.py`  
  承接上传预览对象保存、读取、提升、删除和文件名校验。

- `app_v2/application/knowledge/indexing_service.py`  
  承接文件解析、切片、Qdrant 写入和 documents 状态更新。

- `app_v2/application/knowledge/document_asset_service.py`  
  承接文件资产删除协调，统一处理 documents、MinIO、Qdrant。

- `app_v2/application/rag_service.py`  
  承接聊天检索、重排、上下文拼装。

- `app_v2/application/chat_generation_service.py`  
  承接一次性回答、流式回答、直连 RAG、Agent 策略选择和消息保存。

- `app_v2/application/chat_strategies.py`  
  放置 `DirectRagStrategy`、`AgentStrategy`、`FallbackModelStrategy`。

- `app_v2/application/exam_service.py`  
  承接考试 session、题目生成、答题、评分。

- `app_v2/infrastructure/repositories/exam_repository.py`  
  承接考试相关 MySQL 读写。

### Modify

- `app_v2/application/training/service_provider.py`  
  删除 `get_sales_training_service()` 对旧大类的默认创建。保留仓储、LLM、向量检索等 V2 provider。

- `app_v2/application/training/material_service.py`  
  从外观委托改为真实资料服务。

- `app_v2/application/training/profile_service.py`  
  保留 V2 字典仓储；画像生成、场景润色、补充问题迁到 V2。

- `app_v2/application/training/goal_service.py`  
  目标生成迁到 V2。

- `app_v2/application/training/plan_service.py`  
  方案增删改查迁到 V2 repository 和 mapper。

- `app_v2/application/training/session_service.py`  
  会话开始、轮次提交、流式响应、会话详情迁到 V2。

- `app_v2/application/training/scoring_service.py`  
  最终评分迁到 V2。

- `app_v2/application/knowledge_service.py`  
  移除 `api.services.upload_services`、`api.services.indexing_services`、`api.services.document_asset_service` 导入。

- `app_v2/application/chat_service.py`  
  移除 `api.services.chat_services` 导入，改用 `ChatGenerationService`。

- `app_v2/api/routes/exam.py`  
  移除 `from api.routers.exam import router`，改为 V2 路由函数。

- `docs/V2大爆炸架构与页面治理重构/03-V2大爆炸架构与页面治理执行记录.md`  
  每个阶段补充实际命令和结果。

- `docs/V2大爆炸架构与页面治理重构/04-V2重构整改复盘与尾巴清单.md`  
  每个阶段删掉已完成尾巴，补充新发现尾巴。

### Test

- `tests/test_v2_legacy_dependency_guards.py`
- `tests/test_v2_training_no_legacy_service.py`
- `tests/test_v2_training_profile_service.py`
- `tests/test_sales_training_ingest_flow.py`
- `tests/test_sales_training_repository.py`
- `tests/test_v2_knowledge_service.py`
- `tests/test_document_asset_service.py`
- `tests/test_v2_chat_service.py`
- `tests/test_rag_pipeline.py`
- `tests/test_query_planner.py`
- `tests/test_api_app.py`
- `tests/test_auth_api.py`

---

### Task 1: Add Legacy Dependency Guard Tests

**Files:**
- Create: `tests/test_v2_legacy_dependency_guards.py`
- Test: `tests/test_v2_legacy_dependency_guards.py`

- [ ] **Step 1: Write the failing guard tests**

Create `tests/test_v2_legacy_dependency_guards.py`:

```python
"""V2 旧依赖退场守卫测试。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_V2_ROOT = PROJECT_ROOT / "app_v2"

FORBIDDEN_APP_V2_IMPORTS = (
    "from api.services.chat_services",
    "from api.services.upload_services",
    "from api.services.indexing_services",
    "from api.services.document_asset_service",
    "from api.routers.exam import router",
    "from training.services.sales_training_service import SalesTrainingService",
)


def _python_files(root: Path) -> list[Path]:
    """列出需要检查的 V2 Python 文件，跳过缓存目录。"""

    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_app_v2_does_not_import_legacy_business_services():
    """V2 应用层不能继续导入旧业务服务。"""

    offenders: list[str] = []
    for path in _python_files(APP_V2_ROOT):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_APP_V2_IMPORTS:
            if forbidden in text:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {forbidden}")

    assert offenders == []


def test_exam_route_is_not_legacy_router_alias():
    """V2 考试路由不能只是挂载旧路由。"""

    exam_route = APP_V2_ROOT / "api" / "routes" / "exam.py"
    text = exam_route.read_text(encoding="utf-8")

    assert "from api.routers.exam import router" not in text
    assert "APIRouter" in text
    assert "ExamApplicationService" in text
```

- [ ] **Step 2: Run the guard tests to verify current failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_legacy_dependency_guards.py -q
```

Expected: FAIL. Current failure should mention at least `app_v2/application/knowledge_service.py`, `app_v2/application/chat_service.py`, `app_v2/application/training/service_provider.py`, or `app_v2/api/routes/exam.py`.

- [ ] **Step 3: Commit the failing tests**

Run:

```powershell
git add tests/test_v2_legacy_dependency_guards.py
git commit -m "test: guard v2 legacy dependencies"
```

Expected: Commit succeeds. If commit is not desired in the current local workflow, record the staged file list and skip only the commit command.

---

### Task 2: Migrate Sales Training Core Services

**Files:**
- Create: `tests/test_v2_training_no_legacy_service.py`
- Create: `app_v2/application/training/response_mappers.py`
- Create: `app_v2/application/training/llm_json_service.py`
- Create: `app_v2/application/training/evidence_service.py`
- Modify: `app_v2/application/training/service_provider.py`
- Modify: `app_v2/application/training/material_service.py`
- Modify: `app_v2/application/training/profile_service.py`
- Modify: `app_v2/application/training/goal_service.py`
- Modify: `app_v2/application/training/plan_service.py`
- Modify: `app_v2/application/training/session_service.py`
- Modify: `app_v2/application/training/scoring_service.py`
- Test: `tests/test_v2_training_no_legacy_service.py`
- Test: `tests/test_v2_training_profile_service.py`
- Test: `tests/test_sales_training_ingest_flow.py`
- Test: `tests/test_sales_training_repository.py`

- [ ] **Step 1: Write a failing test that makes the old training service unusable**

Create `tests/test_v2_training_no_legacy_service.py`:

```python
"""V2 销售训练服务不能继续委托旧 SalesTrainingService。"""

import pytest

from app_v2.application.training import service_provider
from app_v2.application.training.goal_service import TrainingGoalApplicationService
from app_v2.application.training.material_service import TrainingMaterialApplicationService
from app_v2.application.training.plan_service import TrainingPlanApplicationService
from app_v2.application.training.profile_service import TrainingProfileApplicationService
from app_v2.application.training.scoring_service import TrainingScoringApplicationService
from app_v2.application.training.session_service import TrainingSessionApplicationService


class ExplodingSalesTrainingService:
    """旧大类替身；V2 核心路径调用它就说明迁移失败。"""

    def __getattr__(self, name):
        raise AssertionError(f"V2 销售训练不应该继续调用旧 SalesTrainingService.{name}")


@pytest.fixture(autouse=True)
def block_legacy_sales_training_service(monkeypatch):
    """禁止 service_provider 创建旧销售训练大类。"""

    monkeypatch.setattr(
        service_provider,
        "get_sales_training_service",
        lambda: ExplodingSalesTrainingService(),
        raising=False,
    )


def test_v2_training_services_can_be_constructed_without_legacy_service():
    """V2 服务构造阶段不应该需要旧大类。"""

    services = [
        TrainingMaterialApplicationService(),
        TrainingProfileApplicationService(),
        TrainingGoalApplicationService(),
        TrainingPlanApplicationService(),
        TrainingSessionApplicationService(),
        TrainingScoringApplicationService(),
    ]

    for service in services:
        assert service.__class__.__name__.startswith("Training")
        assert getattr(service, "service", None) is None
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_training_no_legacy_service.py -q
```

Expected: FAIL because current V2 training services set `self.service = service or get_sales_training_service()`.

- [ ] **Step 3: Create V2 training helper modules**

Create `app_v2/application/training/response_mappers.py`:

```python
"""销售训练响应转换工具。"""

import json
from typing import Any


def load_json_dict(value: Any) -> dict[str, Any]:
    """把数据库 JSON 字符串转换成字典，异常时返回空字典。"""

    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def load_json_list(value: Any) -> list[Any]:
    """把数据库 JSON 字符串转换成列表，异常时返回空列表。"""

    if isinstance(value, list):
        return value
    if not value:
        return []
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return loaded if isinstance(loaded, list) else []
    return []
```

Create `app_v2/application/training/llm_json_service.py`:

```python
"""销售训练 LLM JSON 调用服务。"""

import json
from typing import Any

from utils.logger_handler import logger


class TrainingLlmJsonService:
    """封装销售训练中需要 JSON 输出的 LLM 调用。"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def parse_json_object(self, text: str) -> dict[str, Any]:
        """从模型输出中解析 JSON 对象。"""

        clean_text = (text or "").strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.strip("`")
            if clean_text.lower().startswith("json"):
                clean_text = clean_text[4:].strip()
        try:
            value = json.loads(clean_text)
        except json.JSONDecodeError as exc:
            logger.error("[V2销售训练-模型] JSON解析失败 错误=%s", exc, exc_info=True)
            raise ValueError(f"模型返回内容不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("模型返回 JSON 必须是对象")
        return value
```

Create `app_v2/application/training/evidence_service.py`:

```python
"""销售训练证据检索服务。"""


class TrainingEvidenceService:
    """封装训练资料向量检索，避免会话和画像服务直接操作旧大类。"""

    def __init__(self, vector_adapter_factory=None):
        self.vector_adapter_factory = vector_adapter_factory

    def search_training_evidence(self, query: str, *, visibility: tuple[str, ...], k: int) -> list[dict]:
        """检索训练资料证据。"""

        if not query.strip():
            return []
        if self.vector_adapter_factory is None:
            return []
        adapter = self.vector_adapter_factory()
        return adapter.search(query=query, visibility=visibility, k=k)
```

- [ ] **Step 4: Remove default legacy service construction**

Modify each V2 training service constructor so it accepts focused dependencies and does not default to `get_sales_training_service()`.

Use this pattern in `material_service.py`, then apply the same dependency style to the other five services:

```python
def __init__(self, *, repository=None, file_storage=None, vector_adapter_factory=None):
    self.repository = repository
    self.file_storage = file_storage
    self.vector_adapter_factory = vector_adapter_factory
    self.service = None
```

In `service_provider.py`, remove these imports:

```python
from rag.knowledge_store import KnowledgeStore
from training.services.sales_training_service import SalesTrainingService
```

Delete `get_sales_training_service()` after all V2 services stop importing it.

- [ ] **Step 5: Move old sales training methods by domain**

Move behavior from `training/services/sales_training_service.py` into the V2 files below. Preserve public response models from `training.schemas`.

| Old method range | New target |
| --- | --- |
| `upload_knowledge`, `list_batches`, `delete_batch`, `publish_batch`, `rollback_batch`, `reparse_batch`, `list_batch_versions`, `list_chunks` | `app_v2/application/training/material_service.py` |
| `_write_staging_chunks`, `_publish_staging_vectors`, `_list_batch_chunk_rows`, `_list_staging_chunk_rows`, `_list_published_chunk_rows`, `_documents_to_chunk_rows`, `_chunk_sort_key`, `_next_training_batch_version`, `_archive_previous_training_versions`, `_mark_batch_vectors_current`, `_saved_chunk_preview`, `_batch_file_info`, `_batch_response`, `_get_active_batch`, `_download_batch_file` | `app_v2/application/training/material_service.py` and `response_mappers.py` |
| `create_plan`, `list_plans`, `get_plan_detail`, `delete_plan`, `update_plan` | `app_v2/application/training/plan_service.py` |
| `_plan_summary`, `_plan_detail_response` | `app_v2/application/training/response_mappers.py` |
| `generate_supplement_questions`, `generate_role`, `polish_scenario` | `app_v2/application/training/profile_service.py` |
| `generate_goal_setting`, `_goal_prompt`, `_fallback_goal`, `_goal_response` | `app_v2/application/training/goal_service.py` |
| `start_session`, `list_sessions`, `get_session_detail`, `submit_turn`, `stream_turn`, `_generate_opening_message`, `_customer_prompt`, `_conversation_text`, `_fallback_customer_reply`, `_fallback_opening_message` | `app_v2/application/training/session_service.py` |
| `final_score`, `_fallback_score`, `_normalize_dimension_scores`, `_score_response` | `app_v2/application/training/scoring_service.py` |
| `_invoke_json`, `_parse_json_object` | `app_v2/application/training/llm_json_service.py` |
| `_turn_evidence`, `_search_training_evidence` | `app_v2/application/training/evidence_service.py` |

When moving a method, add a Chinese docstring that states the new responsibility. Preserve existing Chinese logs and change log tags to `V2销售训练-资料`、`V2销售训练-画像`、`V2销售训练-目标`、`V2销售训练-会话`、`V2销售训练-评分`.

- [ ] **Step 6: Run focused training tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_training_no_legacy_service.py tests\test_v2_training_profile_service.py tests\test_sales_training_ingest_flow.py tests\test_sales_training_repository.py -q
```

Expected: PASS.

- [ ] **Step 7: Run V2 API route smoke tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api_app.py::test_openapi_exposes_core_routes -q
```

Expected: PASS and `/api/v2/training/*` routes remain exposed.

- [ ] **Step 8: Commit sales training migration**

Run:

```powershell
git add app_v2/application/training tests/test_v2_training_no_legacy_service.py
git commit -m "refactor: migrate sales training core to v2 services"
```

Expected: Commit succeeds, or the changed files remain unstaged if the current workflow is intentionally no-commit.

---

### Task 3: Migrate Knowledge Upload, Asset Delete, and Indexing

**Files:**
- Create: `app_v2/application/knowledge/upload_preview_service.py`
- Create: `app_v2/application/knowledge/indexing_service.py`
- Create: `app_v2/application/knowledge/document_asset_service.py`
- Modify: `app_v2/application/knowledge_service.py`
- Modify: `tests/test_v2_knowledge_service.py`
- Modify: `tests/test_document_asset_service.py`

- [ ] **Step 1: Add failing assertions for old upload and indexing imports**

Extend `tests/test_v2_legacy_dependency_guards.py` if Task 1 did not already fail on these imports:

```python
def test_knowledge_service_does_not_import_legacy_upload_or_indexing_services():
    """V2 知识资产不能继续委托旧上传和索引服务。"""

    service_file = APP_V2_ROOT / "application" / "knowledge_service.py"
    text = service_file.read_text(encoding="utf-8")

    assert "from api.services.upload_services" not in text
    assert "from api.services.indexing_services" not in text
    assert "from api.services.document_asset_service" not in text
```

- [ ] **Step 2: Run the guard test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_legacy_dependency_guards.py::test_knowledge_service_does_not_import_legacy_upload_or_indexing_services -q
```

Expected: FAIL while `knowledge_service.py` still imports legacy upload, indexing, or document asset services.

- [ ] **Step 3: Create V2 upload preview service**

Create `app_v2/application/knowledge/upload_preview_service.py`:

```python
"""V2 知识上传预览服务。"""

from fastapi import UploadFile


class KnowledgeUploadPreviewService:
    """管理上传预览对象的保存、读取、提升和删除。"""

    def __init__(self, file_storage):
        self.file_storage = file_storage

    def sanitize_filename(self, filename: str | None) -> str:
        """清理上传文件名，避免空文件名进入后续链路。"""

        clean_name = (filename or "").strip()
        if not clean_name:
            raise ValueError("上传文件名不能为空")
        return clean_name

    def validate_file_type(self, filename: str) -> str:
        """校验知识库允许上传的文件类型。"""

        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in {"pdf", "docx", "txt", "md"}:
            raise ValueError(f"不支持的文件类型：{suffix}")
        return suffix

    def save_preview_file(self, file: UploadFile, filename: str, upload_id: str):
        """保存上传预览文件。"""

        return self.file_storage.save_preview_file(file=file, filename=filename, upload_id=upload_id)

    def get_preview_file(self, upload_id: str):
        """读取上传预览对象。"""

        return self.file_storage.get_preview_file(upload_id)

    def promote_preview_file(self, upload_id: str, document_id: str):
        """把预览对象提升为正式文件对象。"""

        return self.file_storage.promote_preview_file(upload_id=upload_id, document_id=document_id)

    def delete_preview_file(self, upload_id: str) -> None:
        """删除上传预览对象。"""

        self.file_storage.delete_preview_file(upload_id)
```

If `FileStorageAdapter` does not yet expose the four preview methods, add them there by moving the corresponding logic from `api/services/upload_services.py`.

- [ ] **Step 4: Create V2 indexing service**

Create `app_v2/application/knowledge/indexing_service.py`:

```python
"""V2 知识索引服务。"""


class KnowledgeIndexingService:
    """统一执行文件解析、切片、向量写入和 documents 状态更新。"""

    def __init__(self, *, document_repository, file_storage, vector_adapter_factory):
        self.document_repository = document_repository
        self.file_storage = file_storage
        self.vector_adapter_factory = vector_adapter_factory

    def index_document(
            self,
            document: dict,
            *,
            document_type: str | None = None,
            split_strategy: str | None = None,
            collection_name: str | None = None,
            increment_version: bool = False,
    ) -> dict:
        """索引单个文档并返回更新后的 documents 行。"""

        adapter = self.vector_adapter_factory(collection_name or document.get("collection_name"))
        return adapter.index_document(
            document=document,
            document_type=document_type or document.get("document_type"),
            split_strategy=split_strategy or document.get("split_strategy"),
            increment_version=increment_version,
            document_repository=self.document_repository,
            file_storage=self.file_storage,
        )
```

If `VectorStoreAdapter` does not expose `index_document()`, move `_index_document` from `api/services/indexing_services.py` into this service and keep the public signature above.

- [ ] **Step 5: Create V2 document asset service**

Create `app_v2/application/knowledge/document_asset_service.py`:

```python
"""V2 文件资产删除服务。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentAssetDeleteResult:
    """文件资产删除结果。"""

    document_id: str


class DocumentAssetApplicationService:
    """协调 documents、MinIO 和 Qdrant 的删除。"""

    def __init__(self, *, document_repository, file_storage, vector_adapter_factory):
        self.document_repository = document_repository
        self.file_storage = file_storage
        self.vector_adapter_factory = vector_adapter_factory

    def delete_document_asset(self, document_id: str) -> DocumentAssetDeleteResult:
        """删除一个文件资产。"""

        document = self.document_repository.get_document(document_id)
        if document is None:
            return DocumentAssetDeleteResult(document_id=document_id)
        bucket_name = document.get("bucket_name")
        object_name = document.get("object_name")
        collection_name = document.get("collection_name")
        if bucket_name and object_name:
            self.file_storage.delete_object(bucket_name=bucket_name, object_name=object_name)
        if collection_name:
            self.vector_adapter_factory(collection_name).delete_document(document_id)
        self.document_repository.delete_document(document_id)
        return DocumentAssetDeleteResult(document_id=document_id)
```

- [ ] **Step 6: Wire `KnowledgeApplicationService` to V2 services**

Modify `app_v2/application/knowledge_service.py`:

```python
from app_v2.application.knowledge.document_asset_service import DocumentAssetApplicationService
from app_v2.application.knowledge.indexing_service import KnowledgeIndexingService
from app_v2.application.knowledge.upload_preview_service import KnowledgeUploadPreviewService
```

Remove imports from:

```python
from api.services.document_asset_service import DocumentAssetService
from api.services.indexing_services import _index_document, _sync_data_files_to_documents
from api.services.upload_services import ...
```

Construct the three V2 services in `__init__` and replace calls to old helpers with methods on those services.

- [ ] **Step 7: Run knowledge tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_knowledge_service.py tests\test_document_asset_service.py tests\test_upload_preview_state.py tests\test_upload_services_preview_state.py -q
```

Expected: PASS.

- [ ] **Step 8: Run MinIO and factory tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_minio_client.py tests\test_processor_and_strategy_factories.py -q
```

Expected: PASS. If MinIO integration checks need a real server, keep fake-backed unit tests separate from real-object verification and document the difference in `03-V2大爆炸架构与页面治理执行记录.md`.

- [ ] **Step 9: Commit knowledge migration**

Run:

```powershell
git add app_v2/application/knowledge app_v2/application/knowledge_service.py tests/test_v2_legacy_dependency_guards.py tests/test_v2_knowledge_service.py tests/test_document_asset_service.py
git commit -m "refactor: migrate knowledge upload and indexing to v2"
```

---

### Task 4: Migrate Chat RAG and Exam Entry

**Files:**
- Create: `app_v2/application/rag_service.py`
- Create: `app_v2/application/chat_generation_service.py`
- Create: `app_v2/application/chat_strategies.py`
- Create: `app_v2/application/exam_service.py`
- Create: `app_v2/infrastructure/repositories/exam_repository.py`
- Modify: `app_v2/application/chat_service.py`
- Modify: `app_v2/api/routes/exam.py`
- Modify: `tests/test_v2_chat_service.py`
- Modify: `tests/test_api_app.py`

- [ ] **Step 1: Add failing chat and exam guard assertions**

Extend `tests/test_v2_legacy_dependency_guards.py`:

```python
def test_chat_service_does_not_import_legacy_chat_services():
    """V2 聊天服务不能继续委托旧聊天服务。"""

    service_file = APP_V2_ROOT / "application" / "chat_service.py"
    text = service_file.read_text(encoding="utf-8")

    assert "from api.services.chat_services" not in text
    assert "ChatGenerationService" in text
```

- [ ] **Step 2: Run the chat and exam guard tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_legacy_dependency_guards.py::test_chat_service_does_not_import_legacy_chat_services tests\test_v2_legacy_dependency_guards.py::test_exam_route_is_not_legacy_router_alias -q
```

Expected: FAIL while chat and exam still use old services.

- [ ] **Step 3: Create chat strategy interfaces**

Create `app_v2/application/chat_strategies.py`:

```python
"""V2 聊天生成策略。"""


class DirectRagStrategy:
    """直连 RAG 策略。"""

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def supports(self, request) -> bool:
        """判断是否适合直连 RAG。"""

        return True

    def generate(self, request):
        """生成一次性回答。"""

        return self.rag_service.answer(request)

    def stream(self, request):
        """生成流式回答。"""

        yield from self.rag_service.stream_answer(request)


class AgentStrategy:
    """Agent 工具链策略。"""

    def __init__(self, agent_service):
        self.agent_service = agent_service

    def supports(self, request) -> bool:
        """判断是否需要 Agent 工具链。"""

        return bool(getattr(request, "use_agent", False))

    def generate(self, request):
        """生成 Agent 回答。"""

        return self.agent_service.answer(request)

    def stream(self, request):
        """流式生成 Agent 回答。"""

        yield from self.agent_service.stream_answer(request)


class FallbackModelStrategy:
    """兜底模型策略。"""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def supports(self, request) -> bool:
        """兜底策略总是可用。"""

        return True

    def generate(self, request):
        """生成兜底回答。"""

        return self.llm_service.answer(request)

    def stream(self, request):
        """流式生成兜底回答。"""

        yield from self.llm_service.stream_answer(request)
```

- [ ] **Step 4: Create chat generation service**

Create `app_v2/application/chat_generation_service.py`:

```python
"""V2 聊天生成应用服务。"""


class ChatGenerationService:
    """统一一次性回答、流式回答和消息保存。"""

    def __init__(self, *, conversation_repository, strategies):
        self.conversation_repository = conversation_repository
        self.strategies = list(strategies)

    def _select_strategy(self, request):
        """选择第一个支持当前请求的生成策略。"""

        for strategy in self.strategies:
            if strategy.supports(request):
                return strategy
        raise RuntimeError("没有可用的聊天生成策略")

    def answer(self, request):
        """生成一次性聊天回答并保存消息。"""

        strategy = self._select_strategy(request)
        response = strategy.generate(request)
        self.conversation_repository.save_chat_exchange(
            conversation_id=getattr(request, "conversation_id", None),
            user_message=getattr(request, "message", ""),
            assistant_message=getattr(response, "answer", ""),
            metadata=getattr(response, "metadata", {}),
        )
        return response

    def stream_answer(self, request):
        """生成流式聊天回答。"""

        strategy = self._select_strategy(request)
        yield from strategy.stream(request)
```

- [ ] **Step 5: Create RAG service facade**

Create `app_v2/application/rag_service.py`:

```python
"""V2 RAG 应用服务。"""


class RagApplicationService:
    """封装检索、重排和上下文拼装。"""

    def __init__(self, *, vector_adapter_factory, reranker=None, llm_service=None):
        self.vector_adapter_factory = vector_adapter_factory
        self.reranker = reranker
        self.llm_service = llm_service

    def answer(self, request):
        """返回一次性 RAG 回答。"""

        collection_name = getattr(request, "collection_name", None)
        adapter = self.vector_adapter_factory(collection_name)
        return adapter.answer(request)

    def stream_answer(self, request):
        """返回流式 RAG 回答。"""

        collection_name = getattr(request, "collection_name", None)
        adapter = self.vector_adapter_factory(collection_name)
        yield from adapter.stream_answer(request)
```

- [ ] **Step 6: Wire chat service to generation service**

Modify `app_v2/application/chat_service.py` so chat answer and stream methods call `ChatGenerationService`. Remove all imports from `api.services.chat_services`.

Required import:

```python
from app_v2.application.chat_generation_service import ChatGenerationService
```

- [ ] **Step 7: Create V2 exam service and repository**

Create `app_v2/infrastructure/repositories/exam_repository.py`:

```python
"""V2 考试仓储。"""


class ExamRepository:
    """管理考试 session、题目、答案和评分记录。"""

    def create_session(self, values: dict) -> dict:
        """创建考试会话。"""

        raise NotImplementedError("ExamRepository.create_session 需要迁移旧考试持久化逻辑")
```

Create `app_v2/application/exam_service.py`:

```python
"""V2 问答考试应用服务。"""


class ExamApplicationService:
    """承接考试题源选择、生成、答题和评分。"""

    def __init__(self, *, exam_repository=None, vector_adapter_factory=None):
        self.exam_repository = exam_repository
        self.vector_adapter_factory = vector_adapter_factory

    def list_sections(self, *, collection_name: str | None = None, document_id: str | None = None):
        """查询可出题章节。"""

        adapter = self.vector_adapter_factory(collection_name) if self.vector_adapter_factory else None
        return adapter.list_exam_sections(document_id=document_id) if adapter else []
```

Move the real exam logic from `api/routers/exam.py` into these two files before enabling routes. Replace `NotImplementedError` with ORM implementation during the same task.

- [ ] **Step 8: Replace `app_v2/api/routes/exam.py`**

Modify `app_v2/api/routes/exam.py`:

```python
"""V2 问答考试接口。"""

from fastapi import APIRouter, Query

from app_v2.application.exam_service import ExamApplicationService

router = APIRouter(prefix="/exam", tags=["V2 问答考试"])


def _exam_service() -> ExamApplicationService:
    """创建考试应用服务。"""

    return ExamApplicationService()


@router.get("/sections")
def list_exam_sections(
    collection_name: str | None = Query(None),
    document_id: str | None = Query(None),
):
    """查询可生成试题的章节。"""

    return _exam_service().list_sections(collection_name=collection_name, document_id=document_id)
```

After this compiles, migrate remaining old exam routes one by one from `api/routers/exam.py` while keeping existing `/api/v2/exam/*` OpenAPI paths stable.

- [ ] **Step 9: Run chat, RAG, exam route tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_chat_service.py tests\test_rag_pipeline.py tests\test_query_planner.py tests\test_api_app.py::test_openapi_exposes_core_routes -q
```

Expected: PASS.

- [ ] **Step 10: Commit chat and exam migration**

Run:

```powershell
git add app_v2/application/rag_service.py app_v2/application/chat_generation_service.py app_v2/application/chat_strategies.py app_v2/application/chat_service.py app_v2/application/exam_service.py app_v2/infrastructure/repositories/exam_repository.py app_v2/api/routes/exam.py tests/test_v2_legacy_dependency_guards.py tests/test_v2_chat_service.py tests/test_api_app.py
git commit -m "refactor: migrate chat rag and exam entry to v2"
```

---

### Task 5: Legacy Exit, Regression, and Documentation

**Files:**
- Modify: `docs/V2大爆炸架构与页面治理重构/03-V2大爆炸架构与页面治理执行记录.md`
- Modify: `docs/V2大爆炸架构与页面治理重构/04-V2重构整改复盘与尾巴清单.md`
- Modify: `docs/V2大爆炸架构与页面治理重构/05-全量V2化三周迁移设计.md`

- [ ] **Step 1: Run backend legacy scan**

Run:

```powershell
rg -n "KnowledgeStore|_get_knowledge_store|get_knowledge_store|from api\\.services|from api\\.routers|from training\\.services" app_v2 api training rag
```

Expected: No unexplained production dependency from `app_v2` to old services. Test files and historical docs may still mention old paths.

- [ ] **Step 2: Run backend target tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_v2_chat_service.py tests\test_v2_repositories.py tests\test_v2_knowledge_service.py tests\test_v2_training_profile_service.py tests\test_v2_dashboard_service.py tests\test_document_asset_service.py tests\test_api_app.py tests\test_auth_api.py -q
```

Expected: PASS.

- [ ] **Step 3: Run backend full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: PASS. Existing deprecation warnings are acceptable only if failures are zero.

- [ ] **Step 4: Run frontend contract checks**

In `D:\PycharmProjects\AI_RAG_Agent_Frontend`, run:

```powershell
node scripts/api-url-contract.test.mjs
node scripts/app-shell-contract.test.mjs
node scripts/feature-pages-contract.test.mjs
node scripts/async-pages-contract.test.mjs
npm run build
```

Expected: All contract scripts pass and `npm run build` succeeds.

- [ ] **Step 5: Update execution record**

Append this block to `docs/V2大爆炸架构与页面治理重构/03-V2大爆炸架构与页面治理执行记录.md` after Step 2, Step 3, and Step 4 have been run. Copy the exact pass counts and warning summaries from the terminal output into the four verification bullets before saving the file:

```markdown
## 2026-06-27 全量 V2 化执行记录

- 销售训练核心服务已从旧 `SalesTrainingService` 迁移到 V2 training application services。
- 知识上传、文件资产删除和索引链路已从旧 `api.services.*` 迁移到 V2 knowledge services。
- 聊天生成链路已从旧 `api.services.chat_services` 迁移到 V2 chat generation service。
- 考试入口已从旧 `api.routers.exam` 迁移到 V2 exam service。
- 旧依赖扫描结果已确认，无未解释的 `app_v2` 到旧业务服务生产调用点。

### 验证

- `.\.venv\Scripts\python.exe -m pytest tests\test_v2_chat_service.py tests\test_v2_repositories.py tests\test_v2_knowledge_service.py tests\test_v2_training_profile_service.py tests\test_v2_dashboard_service.py tests\test_document_asset_service.py tests\test_api_app.py tests\test_auth_api.py -q`：记录终端输出的通过数量和 warning 摘要。
- `.\.venv\Scripts\python.exe -m pytest -q`：记录终端输出的通过数量和 warning 摘要。
- `node scripts/api-url-contract.test.mjs`、`node scripts/app-shell-contract.test.mjs`、`node scripts/feature-pages-contract.test.mjs`、`node scripts/async-pages-contract.test.mjs`：记录四个脚本的通过结果。
- `npm run build`：记录构建成功输出和仍存在的 warning 摘要。
```

- [ ] **Step 6: Update tail list**

Edit `docs/V2大爆炸架构与页面治理重构/04-V2重构整改复盘与尾巴清单.md`:

```markdown
## 10. 全量 V2 化收口状态

### 已迁移

- 销售训练核心业务：已迁移到 `app_v2/application/training/*`。
- 知识上传和索引：已迁移到 `app_v2/application/knowledge/*`。
- 聊天生成：已迁移到 `app_v2/application/chat_generation_service.py` 和策略类。
- 考试入口：已迁移到 `app_v2/application/exam_service.py`。

### 冻结旧实现

- 如果仍保留旧文件，只允许作为历史对照或测试替身，不允许新增业务逻辑。

### 删除准入

- 删除旧文件前必须确认没有生产导入、没有前端引用、OpenAPI 不再暴露旧路径、目标测试和全量测试通过。
```

- [ ] **Step 7: Run documentation placeholder scan**

Run:

```powershell
$patterns = @("T" + "BD", "TO" + "DO", "待" + "补", "占" + "位")
rg -n ($patterns -join "|") docs/V2大爆炸架构与页面治理重构
```

Expected: No matches in the V2 design and execution documents.

- [ ] **Step 8: Final commit**

Run:

```powershell
git add app_v2 api training rag tests docs
git status --short
git commit -m "refactor: complete v2 migration"
```

Expected: Commit succeeds after all tests pass. If frontend files changed in the paired Vue repo, commit them in that repo separately.

---

## Final Verification Matrix

| Verification | Command | Expected |
| --- | --- | --- |
| Legacy guard | `.\.venv\Scripts\python.exe -m pytest tests\test_v2_legacy_dependency_guards.py -q` | PASS |
| V2 target suite | `.\.venv\Scripts\python.exe -m pytest tests\test_v2_chat_service.py tests\test_v2_repositories.py tests\test_v2_knowledge_service.py tests\test_v2_training_profile_service.py tests\test_v2_dashboard_service.py tests\test_document_asset_service.py tests\test_api_app.py tests\test_auth_api.py -q` | PASS |
| Backend full suite | `.\.venv\Scripts\python.exe -m pytest -q` | PASS |
| Legacy scan | `rg -n "KnowledgeStore|_get_knowledge_store|get_knowledge_store|from api\\.services|from api\\.routers|from training\\.services" app_v2 api training rag` | No unexplained V2 production dependency |
| Frontend API contract | `node scripts/api-url-contract.test.mjs` | PASS |
| Frontend shell contract | `node scripts/app-shell-contract.test.mjs` | PASS |
| Frontend feature contract | `node scripts/feature-pages-contract.test.mjs` | PASS |
| Frontend async contract | `node scripts/async-pages-contract.test.mjs` | PASS |
| Frontend build | `npm run build` | PASS |

## Self-Review Notes

- Spec coverage: The plan covers sales training, knowledge upload/indexing, chat RAG, exam entry, legacy exit, regression, and documentation updates from `05-全量V2化三周迁移设计.md`.
- Documentation scan: Task 5 includes a scan over the V2 design and execution documents before the final commit.
- Type consistency: Service names match current `app_v2` naming, and newly introduced names are used consistently across tasks.
