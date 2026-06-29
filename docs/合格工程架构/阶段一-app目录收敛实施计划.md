# App 目录收敛实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将后端业务目录从 `app_v2` 收敛为稳定目录名 `app`，并保持现有接口行为不变。

**Architecture:** 本阶段只做目录命名和 import 收敛，不拆业务 Service，不改变 API 路径，不改变数据库结构。`api/` 继续作为 FastAPI 启动入口，`app/` 承接原 `app_v2/` 的路由层、应用服务层、领域层、基础设施层和 shared 工具。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy、LangChain、Qdrant、MinIO、Redis、MySQL、pytest、PowerShell。

## Global Constraints

- 日志输出继续使用中文。
- API 路径仍保持 `/api/v2`，不因为目录改名而改接口协议。
- 数据库字段名、SSE 事件名、响应字段名不改变。
- 本阶段不修改业务逻辑，只做目录和引用迁移。
- 当前工作区已有未提交改动，执行前必须先确认是否暂存、提交或继续叠加。
- 所有手工代码编辑使用 `apply_patch`；批量机械替换可以使用脚本命令。

---

## File Structure

本阶段最终结构：

```text
api/
app/
  api/
  application/
  domain/
  infrastructure/
  shared/
core/
config/
docs/
tests/
```

需要修改的文件类型：

| 类型 | 处理方式 |
|---|---|
| Python import | `app_v2` 全部替换为 `app` |
| Python 注释/文档字符串 | 当前运行路径描述同步为 `app` |
| Markdown 历史文档 | 历史记录可保留 `app_v2`，本阶段不强制改旧复盘文档 |
| 新设计文档 | 继续使用 `app` 作为目标目录名 |
| Docker/启动命令 | 如引用 `app_v2`，改为 `app` |

## Task 1: 工作区保护和迁移前基线

**Files:**
- Inspect: `git status`
- Inspect: `app_v2/`
- Inspect: `api/main.py`

**Interfaces:**
- Consumes: 当前 git 工作区。
- Produces: 明确的迁移前状态，避免覆盖用户未提交改动。

- [ ] **Step 1: 查看当前工作区状态**

Run:

```powershell
git status --short
```

Expected:

```text
能清楚看到当前已有改动。
```

- [ ] **Step 2: 如果存在无关改动，先和用户确认处理方式**

当前已知可能存在：

```text
app_v2/application/exam_service.py
app_v2/application/training/sales_training_core.py
tests/test_config_handler.py
tests/test_document_parser_llm_semantic.py
tests/test_prompt_manager.py
docs/合格工程架构/
docs/整体分析/
```

处理原则：

```text
不回滚用户改动。
不删除已有改动。
如果继续叠加迁移，后续 git diff 会同时包含旧改动和目录迁移。
```

- [ ] **Step 3: 记录迁移前 app_v2 import 数量**

Run:

```powershell
rg -n "app_v2" api app_v2 core tests -g "*.py"
```

Expected:

```text
列出所有 Python 代码中的 app_v2 引用。
```

## Task 2: 重命名业务目录

**Files:**
- Move: `app_v2/` -> `app/`

**Interfaces:**
- Consumes: 原 `app_v2` 包。
- Produces: 新 `app` 包，内部文件结构保持不变。

- [ ] **Step 1: 确认 app 目录不存在**

Run:

```powershell
Test-Path app
```

Expected:

```text
False
```

- [ ] **Step 2: 执行目录重命名**

Run:

```powershell
Rename-Item -Path app_v2 -NewName app
```

Expected:

```text
命令无输出，项目根目录出现 app/，app_v2/ 不再存在。
```

- [ ] **Step 3: 确认 Python 包初始化文件仍存在**

Run:

```powershell
Get-ChildItem app -Recurse -Filter __init__.py | Measure-Object | Select-Object -ExpandProperty Count
```

Expected:

```text
14
```

## Task 3: 批量替换 Python import

**Files:**
- Modify: `api/**/*.py`
- Modify: `app/**/*.py`
- Modify: `core/**/*.py`
- Modify: `tests/**/*.py`

**Interfaces:**
- Consumes: 所有 `app_v2` Python import。
- Produces: 所有 Python 代码改为 `app` import。

- [ ] **Step 1: 执行 Python 文件内机械替换**

Run:

```powershell
$files = Get-ChildItem -Path api,app,core,tests -Recurse -File -Include *.py
foreach ($file in $files) {
    $text = Get-Content -Raw -Encoding UTF8 $file.FullName
    $updated = $text -replace 'app_v2', 'app'
    if ($updated -ne $text) {
        Set-Content -Encoding UTF8 -NoNewline -Path $file.FullName -Value $updated
    }
}
```

Expected:

```text
命令无输出。
```

- [ ] **Step 2: 扫描 Python 代码中是否还有 app_v2**

Run:

```powershell
rg -n "app_v2" api app core tests -g "*.py"
```

Expected:

```text
无输出。
```

- [ ] **Step 3: 检查启动入口**

Open:

```text
api/main.py
```

Expected import:

```python
from app.api.router import router as v2_router
```

Expected docstring contains:

```text
业务流程放到 `app` 的应用服务层
```

## Task 4: 修正文档和配置中的当前路径说明

**Files:**
- Modify: `api/schemas.py`
- Modify if needed: `Dockerfile`
- Modify if needed: `docker-compose.yml`
- Modify: `docs/合格工程架构/*.md`
- Do not mass rewrite historical docs under `docs/V2大爆炸架构与页面治理重构/` unless they describe current state.

**Interfaces:**
- Consumes: 当前路径说明。
- Produces: 新架构文档和当前代码注释统一使用 `app`。

- [ ] **Step 1: 扫描非历史区域 app_v2 文本**

Run:

```powershell
rg -n "app_v2" api app core tests Dockerfile docker-compose.yml docs\合格工程架构 -g "*"
```

Expected:

```text
只允许 docs/合格工程架构 中描述“原 app_v2”历史来源时出现 app_v2。
代码、Dockerfile、docker-compose.yml 不应出现 app_v2。
```

- [ ] **Step 2: 如 `api/schemas.py` 仍写 app_v2，改为 app**

Expected content:

```python
"""API Schema 定义。

业务逻辑在 app/application 中，schema 文件只负责参数校验和 OpenAPI 文档生成。
"""
```

- [ ] **Step 3: 保持 `/api/v2` 路由前缀不变**

Open:

```text
app/api/router.py
```

Expected:

```python
router = APIRouter(prefix="/api/v2")
```

如果当前文件不是这行精确写法，但实际仍挂载 `/api/v2`，不要为格式强行改代码。

## Task 5: 编译和导入验证

**Files:**
- Verify: `api/`
- Verify: `app/`
- Verify: `core/`
- Verify: `tests/`

**Interfaces:**
- Consumes: 已替换后的 import。
- Produces: 可导入、可编译的 Python 代码。

- [ ] **Step 1: 运行 compileall**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q api app core tests
```

Expected:

```text
退出码 0，无输出。
```

- [ ] **Step 2: 验证 FastAPI app 可以导入**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from api.main import app; print(app.title)"
```

Expected output contains:

```text
知识台 V2 API
```

- [ ] **Step 3: 验证核心 app 包可以导入**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import app; from app.api.router import router; print(router.prefix)"
```

Expected output:

```text
/api/v2
```

## Task 6: 测试验证

**Files:**
- Test: `tests/`

**Interfaces:**
- Consumes: 重命名后的项目结构。
- Produces: 测试结果和失败清单。

- [ ] **Step 1: 先运行配置和 prompt 相关测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_config_handler.py tests\test_prompt_manager.py -q
```

Expected:

```text
全部通过；如果测试文件当前被删除或未恢复，需要先和用户确认是否恢复。
```

- [ ] **Step 2: 运行全量测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected:

```text
全部通过，或输出明确失败原因。
```

- [ ] **Step 3: 如果全量测试受外部依赖影响，至少保留 compileall 和核心导入验证结果**

外部依赖包括：

```text
MySQL
Redis
MinIO
Qdrant
真实 LLM API Key
```

## Task 7: Git 差异检查

**Files:**
- Inspect: git diff

**Interfaces:**
- Consumes: 本阶段所有文件变更。
- Produces: 可评审的目录迁移 diff。

- [ ] **Step 1: 查看重命名识别情况**

Run:

```powershell
git status --short
```

Expected:

```text
大量 app_v2/* -> app/* rename，另有 import 修改。
```

- [ ] **Step 2: 确认 Python 代码不再引用 app_v2**

Run:

```powershell
rg -n "app_v2" api app core tests -g "*.py"
```

Expected:

```text
无输出。
```

- [ ] **Step 3: 查看 diff 统计**

Run:

```powershell
git diff --stat
```

Expected:

```text
主要是 app_v2 到 app 的 rename 和 import 替换。
```

## Task 8: 提交建议

**Files:**
- Commit: directory rename and import replacement.

**Interfaces:**
- Consumes: 验证通过后的迁移变更。
- Produces: 一个可回滚的阶段一提交。

- [ ] **Step 1: 暂存目录迁移相关文件**

Run:

```powershell
git add api app core tests docs\合格工程架构
git add -u app_v2
```

Expected:

```text
目录重命名和 import 修改已暂存。
```

- [ ] **Step 2: 提交**

Run:

```powershell
git commit -m "重构：收敛后端业务目录为app"
```

Expected:

```text
生成一个 commit。
```

## Self-Review

Spec coverage:

- 目录收敛：Task 2、Task 3。
- API 路径保持不变：Task 4。
- 编译和导入验证：Task 5。
- 测试验证：Task 6。
- 风险控制和工作区保护：Task 1、Task 7。

Placeholder scan:

- 本计划不包含 TBD、TODO、implement later。

Type consistency:

- 新包名统一为 `app`。
- `/api/v2` 作为接口路径保持不变。

## Execution Options

Plan complete and saved to `docs/合格工程架构/阶段一-app目录收敛实施计划.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
