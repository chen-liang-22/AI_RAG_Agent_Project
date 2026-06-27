# AI_RAG_Agent_Project

Deployable FastAPI service for a LangChain ReAct agent with Qdrant-backed RAG.

## Services

- `api.main:app`: FastAPI backend.
- `qdrant`: Vector database.
- `../AI_RAG_Agent_Frontend`: Vue 3 frontend.

## Backend Layout

```text
api/                 # FastAPI 启动入口，保留 api.main:app
app_v2/
  api/               # 路由层
  application/       # 业务服务层
  domain/            # ORM 实体、请求响应 schema、常量
  infrastructure/    # MySQL、MinIO、Qdrant、Redis、ID 生成等基础设施
  shared/            # 分页、响应转换等共享能力
core/
  agent/             # ReAct Agent 与工具
  model/             # LLM / Embedding 工厂
  rag/               # RAG 检索、切分、精排、答案生成
  utils/             # 配置、日志、路径、文件工具
config/              # YAML 配置
docs/                # 项目文档
scripts/             # 维护脚本
tests/               # 后端测试
```

`api` 只作为启动门面保留，真实业务入口统一在 `app_v2`，可复用的 RAG、模型和工具能力统一在 `core`。

## Environment

Copy `.env.example` to `.env` and fill in your DashScope key.

```powershell
Copy-Item .env.example .env
```

Required variable:

```text
DASHSCOPE_API_KEY=your_dashscope_api_key
```

## Local Run

Recommended Python version:

```text
Python 3.12
```

Create and activate a Python 3.12 virtual environment:

```powershell
& "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv312
.\.venv312\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The API service depends on `fastapi` and `uvicorn[standard]`. If FastAPI is missing, install project
dependencies again:

```powershell
pip install fastapi "uvicorn[standard]"
```

Install test dependencies when running the local test suite:

```powershell
pip install -r requirements-dev.txt
```

Start Qdrant:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Start the API:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Load knowledge documents into Qdrant:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/knowledge/reload
```

RAG query planning is configured in `config/app.yml` under the `rag` section. The recommended default is `query_planner_mode: adaptive`:
the backend first retrieves with the original question, evaluates recall quality, and only calls the LLM Query Planner
when the first recall is weak. Rule-based keyword intent matching is not used in the main chain.

Chat:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/chat -ContentType "application/json" -Body '{"message":"小户型适合哪些扫地机器人？"}'
```

## Docker Compose

Make sure Docker Desktop is running, then start both the API and Qdrant:

```powershell
docker compose up --build
```

API docs:

```text
http://localhost:8000/docs
```

Vue frontend:

```text
http://localhost:8080
```

## MySQL Schema

运行时默认数据库是 MySQL，连接配置在：

```text
config/storage.yml
```

新环境初始化文件统一放在：

```text
docs/初始化文件
```

MySQL 初始化只需要执行这一份 SQL：

```powershell
mysql -u root -p < docs/初始化文件/mysql初始化建表和基础数据.sql
```

这份 SQL 只负责创建表、索引、外键和系统默认字典，不包含 Qdrant 向量数据。知识库和销售训练资料需要通过系统上传，由项目写入 MinIO、MySQL 和 Qdrant。

重新生成 SQL：

```powershell
# 初始化 SQL 当前保存在 docs/初始化文件/，如需重新生成请按项目脚本说明执行
```

## Vue Frontend

The frontend project is created next to this backend project:

```text
C:\Users\Administrator\WebstormProjects\AI_RAG_Agent_Frontend
```

Local frontend development:

```powershell
cd C:\Users\Administrator\WebstormProjects\AI_RAG_Agent_Frontend
npm install
npm run dev
```

## Tests

```powershell
.\.venv312\Scripts\python.exe -m pytest
```

Vite serves the frontend at:

```text
http://127.0.0.1:5173
```
