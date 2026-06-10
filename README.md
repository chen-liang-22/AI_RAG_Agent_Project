# AI_RAG_Agent_Project

Deployable FastAPI service for a LangChain ReAct agent with Qdrant-backed RAG.

## Services

- `api.main:app`: FastAPI backend.
- `app.py`: Streamlit demo UI.
- `qdrant`: Vector database.

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

Install dependencies:

```powershell
pip install -r requirements.txt
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
