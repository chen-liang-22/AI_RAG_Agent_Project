# Docker 本地中间件启动命令

本文档记录本项目在 Win11 Docker Desktop 本地开发时使用的中间件容器命令。

当前项目用到的容器：

| 容器名 | 镜像 | 用途 | 端口 |
| --- | --- | --- | --- |
| `qdrant` | `qdrant/qdrant` | 向量数据库 | `6333` HTTP，`6334` gRPC |
| `mysql8` | `mysql:8.0` | 业务 MySQL 数据库 | `3306` |
| `redis7` | `redis:7.2` | 缓存、上传预览状态、分布式锁 | `6379` |
| `minio` | `minio/minio:RELEASE.2023-07-11T21-29-34Z` | 文件对象存储 | `9000` API，`9001` 控制台 |
| `xxl-job-admin` | `xuxueli/xxl-job-admin:2.4.1` | 定时任务调度中心 | `8080` |

统一密码：`1234qwer`。

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

## 启动 XXL-JOB Admin

XXL-JOB Admin 需要 MySQL 中存在 `xxl_job` 数据库和对应表。

先创建数据库：

```powershell
docker exec -it mysql8 mysql -uroot -p1234qwer -e "CREATE DATABASE IF NOT EXISTS xxl_job DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

启动容器：

```powershell
docker run -d `
  --name xxl-job-admin `
  --network ai-rag-net `
  -p 8080:8080 `
  -e TZ=Asia/Shanghai `
  -e PARAMS="--spring.datasource.url=jdbc:mysql://host.docker.internal:3306/xxl_job?Unicode=true&characterEncoding=UTF-8&serverTimezone=Asia/Shanghai --spring.datasource.username=root --spring.datasource.password=1234qwer" `
  xuxueli/xxl-job-admin:2.4.1
```

访问地址：

```text
http://localhost:8080/xxl-job-admin
```

XXL-JOB 默认账号通常是：

```text
用户名：admin
密码：123456
```

如果使用 Docker 网络内的 MySQL 容器名访问，可以把 JDBC 地址改成：

```text
jdbc:mysql://mysql8:3306/xxl_job?Unicode=true&characterEncoding=UTF-8&serverTimezone=Asia/Shanghai
```

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
docker ps --filter "name=xxl-job-admin"
```

## 查看日志

```powershell
docker logs -f qdrant
docker logs -f mysql8
docker logs -f redis7
docker logs -f minio
docker logs -f xxl-job-admin
```

## 停止容器

```powershell
docker stop qdrant mysql8 redis7 minio xxl-job-admin
```

## 启动已存在容器

容器已经创建过，只是停止了，用下面命令启动即可。

```powershell
docker start qdrant mysql8 redis7 minio xxl-job-admin
```

## 删除容器但保留数据卷

只删除容器，不删除数据。

```powershell
docker rm -f qdrant mysql8 redis7 minio xxl-job-admin
```

## 删除容器和数据卷

这个操作会清空本地数据，包括 MySQL 数据、Redis 数据、Qdrant 向量、MinIO 文件。

```powershell
docker rm -f qdrant mysql8 redis7 minio xxl-job-admin

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

检查 XXL-JOB：

```powershell
curl http://localhost:8080/xxl-job-admin
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
