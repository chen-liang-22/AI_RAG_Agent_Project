# Docker 本地中间件启动命令

本文档记录本项目在 Win11 Docker Desktop 本地开发时使用的中间件容器命令。

当前项目用到的容器：

| 容器名 | 镜像 | 用途 | 端口 |
| --- | --- | --- | --- |
| `qdrant` | `qdrant/qdrant` | 向量数据库 | `6333` HTTP，`6334` gRPC |
| `mysql8` | `mysql:8.0` | 业务 MySQL 数据库 | `3306` |
| `redis7` | `redis:7.2` | 缓存、上传预览状态、分布式锁 | `6379` |
| `minio` | `minio/minio:RELEASE.2023-07-11T21-29-34Z` | 文件对象存储 | `9000` API，`9001` 控制台 |

统一密码：`1234qwer`。

## 当前 Docker Desktop 已启动四件套

下面是 2026-06-29 按 Docker Desktop 截图和 `docker ps` 确认的当前本机运行态。截图里的 `lms-admin`
不是运行态，不纳入本项目本地中间件四件套。

| 容器名 | 镜像 | 当前端口映射 | 账号 | 密码 | 状态 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `minio` | `quay.io/minio/minio:RELEASE.2025-04-22T22-12-26Z` | `9000-9001 -> 9000-9001/tcp` | `admin` | `1234qwer` | 运行中 | 文件对象存储，`9000` 为 API，`9001` 为控制台 |
| `redis7` | `redis:7-alpine` | `6379 -> 6379/tcp` | 无账号 | `1234qwer` | 运行中 | 缓存、上传预览状态、分布式锁 |
| `mysql8` | `mysql:8.4` | `3306 -> 3306/tcp` | `root` | `1234qwer` | 运行中 | 业务 MySQL 数据库，当前容器初始化库为 `rag_db` |
| `qdrant-local` | `qdrant/qdrant` | `6333-6334 -> 6333-6334/tcp` | 无 | 无 | 运行中 | 向量数据库，`6333` 为 HTTP，`6334` 为 gRPC |

当前连接信息：

```text
MinIO 控制台：http://localhost:9001
MinIO API：http://localhost:9000
MySQL：127.0.0.1:3306，root / 1234qwer，数据库 rag_db
Redis：127.0.0.1:6379，密码 1234qwer
Qdrant：http://localhost:6333，无账号密码
```

按当前本机容器名启动已存在容器：

```powershell
docker start minio redis7 mysql8 qdrant-local
```

按当前本机容器名停止容器：

```powershell
docker stop minio redis7 mysql8 qdrant-local
```

## 一键创建本地网络

如果各容器需要通过容器名互相访问，建议先创建一个网络。

```powershell
docker network create ai-rag-net
```

如果提示网络已存在，可以忽略。

## 启动 Qdrant

```powershell
docker run -d `
  --name qdrant `
  --network ai-rag-net `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

访问地址：

```text
http://localhost:6333
```

项目本地配置通常使用：

```text
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=agent
QDRANT_PREFER_GRPC=false
QDRANT_GRPC_PORT=6334
```

## 启动 MySQL 8

```powershell
docker run -d `
  --name mysql8 `
  --network ai-rag-net `
  -p 3306:3306 `
  -e MYSQL_ROOT_PASSWORD=1234qwer `
  -e MYSQL_DATABASE=ai_rag_agent `
  -e TZ=Asia/Shanghai `
  -v mysql8_data:/var/lib/mysql `
  mysql:8.0
```

项目业务库：

```text
ai_rag_agent
```

项目本地配置通常使用：

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=ai_rag_agent
MYSQL_USER=root
MYSQL_PASSWORD=1234qwer
```

初始化业务表和基础数据：

```powershell
docker exec -i mysql8 mysql -uroot -p1234qwer < docs/初始化文件/mysql初始化建表和基础数据.sql
```

进入 MySQL：

```powershell
docker exec -it mysql8 mysql -uroot -p1234qwer
```

## 启动 Redis 7

```powershell
docker run -d `
  --name redis7 `
  --network ai-rag-net `
  -p 6379:6379 `
  -e TZ=Asia/Shanghai `
  -v redis7_data:/data `
  redis:7.2 redis-server --appendonly yes --requirepass 1234qwer
```

项目本地配置通常使用：

```text
REDIS_ENABLED=true
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=1234qwer
REDIS_KEY_PREFIX=ai_rag_agent
```

检查 Redis：

```powershell
docker exec -it redis7 redis-cli -a 1234qwer ping
```

正常返回：

```text
PONG
```

## 启动 MinIO

```powershell
docker run -d `
  --name minio `
  --network ai-rag-net `
  -p 9000:9000 `
  -p 9001:9001 `
  -e MINIO_ROOT_USER=admin `
  -e MINIO_ROOT_PASSWORD=1234qwer `
  -v minio_data_2023_console:/data `
  minio/minio:RELEASE.2023-07-11T21-29-34Z server /data --console-address ":9001"
```

访问地址：

```text
MinIO API：http://localhost:9000
MinIO 控制台：http://localhost:9001
用户名：admin
密码：1234qwer
```

项目本地配置通常使用：

```text
MINIO_ENABLED=true
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=admin
MINIO_ROOT_PASSWORD=1234qwer
MINIO_BUCKET=pub
MINIO_PUBLIC_BASE_URL=http://127.0.0.1:9000
MINIO_SECURE=false
```

本项目默认桶名是 `pub`。如果桶不存在，需要在 MinIO 控制台手动创建 `pub`，或用 MinIO Client 创建。

使用 MinIO Client 创建 `pub` 桶：

```powershell
docker run --rm `
  --network ai-rag-net `
  minio/mc sh -c "mc alias set local http://minio:9000 admin 1234qwer && mc mb --ignore-existing local/pub && mc anonymous set download local/pub"
```

## 异步入库说明

当前项目已放弃 XXL-JOB，文件入库改为 FastAPI 内部异步任务。

本地开发只需要启动 MySQL、Redis、MinIO 和 Qdrant。

## 查看当前容器

```powershell
docker ps
```

只看本项目中间件：

```powershell
docker ps --filter "name=qdrant"
docker ps --filter "name=mysql8"
docker ps --filter "name=redis7"
docker ps --filter "name=minio"
```

## 查看日志

```powershell
docker logs -f qdrant
docker logs -f mysql8
docker logs -f redis7
docker logs -f minio
```

## 停止容器

```powershell
docker stop qdrant mysql8 redis7 minio
```

## 启动已存在容器

容器已经创建过，只是停止了，用下面命令启动即可。

```powershell
docker start qdrant mysql8 redis7 minio
```

## 删除容器但保留数据卷

只删除容器，不删除数据。

```powershell
docker rm -f qdrant mysql8 redis7 minio
```

## 删除容器和数据卷

这个操作会清空本地数据，包括 MySQL 数据、Redis 数据、Qdrant 向量、MinIO 文件。

```powershell
docker rm -f qdrant mysql8 redis7 minio

docker volume rm qdrant_storage
docker volume rm mysql8_data
docker volume rm redis7_data
docker volume rm minio_data_2023_console
```

## 项目启动前检查

```powershell
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"
```

检查 Qdrant：

```powershell
curl http://localhost:6333/collections
```

检查 MySQL：

```powershell
docker exec -it mysql8 mysql -uroot -p1234qwer -e "SHOW DATABASES;"
```

检查 Redis：

```powershell
docker exec -it redis7 redis-cli -a 1234qwer ping
```

检查 MinIO：

```powershell
curl http://localhost:9000/minio/health/live
```

检查 FastAPI：

```powershell
curl http://localhost:8000/api/v2/health
```

## 当前项目后端本地启动命令

中间件启动后，在项目根目录执行：

```powershell
cd D:\PycharmProjects\AI_RAG_Agent_Project
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

前端本地启动：

```powershell
cd D:\PycharmProjects\AI_RAG_Agent_Frontend
npm run dev
```
