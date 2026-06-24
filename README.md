# AI_RAG_Agent_Project

Deployable FastAPI service for a LangChain ReAct agent with Qdrant-backed RAG.

## Services

- `api.main:app`: FastAPI backend.
- `app.py`: Streamlit demo UI.
- `qdrant`: Vector database.
- `../AI_RAG_Agent_Frontend`: Vue 3 frontend.

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

RAG query planning is configured in `config/rag.yml`. The recommended default is `query_planner_mode: adaptive`:
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

运行时默认业务数据库是 MySQL，连接配置在：

```text
config/database.yml
```

MySQL 初始化建表和基础字典数据在：

```text
docs/mysql初始化建表和基础数据.sql
```

当前 MySQL 业务数据导入脚本在：

```text
docs/mysql纯业务数据导入.sql
```

推荐执行顺序：

```powershell
mysql -u root -p < docs/mysql初始化建表和基础数据.sql
mysql -u root -p < docs/mysql纯业务数据导入.sql
```

这两个脚本都不包含 Qdrant 向量数据。业务数据导入脚本会先清空目标业务表，再导入 MySQL 业务数据。
项目运行时业务数据以 MySQL 为准。

重新生成 SQL：

```powershell
.\.venv312\Scripts\python.exe scripts\generate_mysql_init_sql.py
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
