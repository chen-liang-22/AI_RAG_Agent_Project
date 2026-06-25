# 上传预览生产化设计

## 背景

当前知识库上传分为预览和确认两个阶段。预览阶段会把文件保存到 MinIO `previews/{upload_id}/{filename}`，并把 `upload_id` 对应的 `StoredFileInfo` 放在 Python 进程内字典 `_PREVIEW_UPLOADS`。这个做法在单进程开发环境可用，但在生产环境有两个问题：

- 服务重启、多 worker 或水平扩容后，确认接口可能找不到 `upload_id`。
- 用户只预览不确认时，MinIO `previews/` 临时对象会长期保留。

本设计将预览上传状态持久化到 Redis，并通过 XXL-JOB 调度 MinIO 临时对象清理。

## 设计模式

- 外观模式：继续通过 `FileStorageService` 统一操作 MinIO。业务层和任务接口不直接散落 MinIO SDK 调用。
- 单例模式：继续复用现有 `get_redis_client()` 和 `get_minio_client()` 的进程级客户端。
- 此场景不新增复杂设计模式，避免为了清理任务引入额外抽象。

## 目标

1. 预览上传后，`upload_id` 元数据不依赖进程内存。
2. 确认入库在服务重启或多 worker 场景下仍可通过 `upload_id` 找到 MinIO 临时对象。
3. 临时预览对象默认保留 24 小时，过期后需要重新上传。
4. 通过 XXL-JOB 调用内部接口清理 MinIO `previews/` 过期对象。
5. 预览阶段保留现有返回文本截断，并增加文件大小上限，避免超大文件拖垮解析流程。

## Redis 预览元数据

Redis 只保存临时上传登记信息，不保存文件正文，也不保存 `sample_text`。文件本体仍只保存在 MinIO。

推荐 key：

```text
ai_rag_agent:upload_preview:{upload_id}
```

推荐 TTL：

```text
86400 秒
```

推荐 value：

```json
{
  "upload_id": "tmp_xxx",
  "filename": "demo.pdf",
  "file_type": "pdf",
  "file_md5": "文件内容 MD5",
  "file_size": 123456,
  "bucket_name": "pub",
  "object_name": "previews/tmp_xxx/demo.pdf",
  "public_url": "http://127.0.0.1:9000/pub/previews/tmp_xxx/demo.pdf",
  "file_path": "minio://pub/previews/tmp_xxx/demo.pdf",
  "created_at": "2026-06-25T10:00:00+08:00",
  "expires_at": "2026-06-26T10:00:00+08:00"
}
```

确认成功、确认发现重复、预览解析失败时，删除 Redis key 和对应 MinIO preview 对象。Redis 不可用时不静默降级为生产可用状态，应返回明确错误，避免用户拿到无法确认的 `upload_id`。

## XXL-JOB 清理接口

新增内部接口：

```text
POST /internal/jobs/minio/cleanup-preview-uploads
```

接口职责：

1. 校验内部调用 Token，例如请求头 `X-INTERNAL-JOB-TOKEN`。
2. 使用 Redis 分布式锁防止多个 worker 同时清理。
3. 扫描 MinIO 默认桶下 `previews/` 前缀对象。
4. 根据对象 `last_modified` 判断是否超过 TTL。
5. 只删除 `object_name` 以 `previews/` 开头且已过期的对象。
6. 返回扫描数、删除数、跳过数、失败数和是否拿到锁。

XXL-JOB 建议每小时调用一次该接口。任务失败时由 XXL-JOB 重试或告警，后端接口保持幂等。

## 安全边界

- 清理逻辑必须硬编码或配置限定 `previews/` 前缀，禁止删除 `documents/`、`training/` 或其他对象。
- 内部接口必须校验 Token，Token 从环境变量读取，不写死在代码中。
- 删除失败要记录中文日志，包含上传编号、桶名、对象名和错误原因。
- 拿不到 Redis 锁时返回 skipped，不视为业务失败。

## 预览文件大小和字数限制

现有预览响应 `sample_text` 由 `VectorStoreService.preview_file()` 截断到 5000 字符。模型推荐接口会重新构造结构样本，最多 10000 字符。这两个限制解决的是响应大小和模型输入大小。

还需要增加文件大小上限，解决解析前的资源风险。建议新增配置：

```yaml
upload_preview:
  ttl_seconds: 86400
  max_file_size_bytes: 52428800
  sample_text_chars: 5000
  recommendation_sample_chars: 10000
```

默认最大文件大小为 50MB。超过上限时，预览上传直接返回 400，提示文件过大。这个限制只作用于知识库预览上传；销售训练资料如需限制，应单独评估。

## 流程

### 预览上传

1. 校验文件名和类型。
2. 保存文件到 MinIO `previews/{upload_id}/{filename}`。
3. 计算 MD5 和文件大小。
4. 文件大小超过上限则删除 preview 对象并返回 400。
5. 将预览元数据写入 Redis，设置 TTL。
6. 下载 MinIO 临时对象到本地临时目录，解析并返回 `sample_text`。
7. 如果解析失败，删除 Redis key 和 MinIO preview 对象。

### 确认入库

1. 从 Redis 根据 `upload_id` 读取预览元数据。
2. 如果 Redis 不存在，返回 404，提示临时上传已过期或不存在。
3. 校验 `object_name` 仍是 `previews/` 前缀。
4. 查重。
5. 不重复时复制 MinIO 对象到 `documents/{document_id}/{filename}`。
6. 删除 Redis key 和 MinIO preview 对象。
7. 创建 `documents` 记录并继续索引。

### 清理任务

1. XXL-JOB 调用内部清理接口。
2. 接口拿 Redis 锁。
3. 扫描 MinIO `previews/`。
4. 删除超过 TTL 的对象。
5. 返回清理统计。

## 测试计划

- 预览上传成功后，Redis 存在 `upload_id` 元数据，且不包含文件正文。
- 清空进程内 `_PREVIEW_UPLOADS` 后，确认接口仍能通过 Redis 元数据完成入库。
- Redis key 不存在时，确认接口返回临时上传已过期或不存在。
- 确认成功后，Redis key 被删除，MinIO preview 对象被删除。
- 清理接口缺少 Token 时返回 401 或 403。
- 清理接口只删除过期 `previews/` 对象，不删除未过期对象。
- 清理接口不删除 `documents/` 和 `training/` 对象。
- 拿不到 Redis 锁时清理接口返回 skipped。
- 超过 `max_file_size_bytes` 的预览上传返回 400，并清理已上传的 preview 对象。

## 非目标

- 不把文件正文、预览文本或切片结果写入 Redis。
- 不改变正式文件路径 `documents/{document_id}/{filename}`。
- 不改变销售训练资料路径 `training/{document_id}/{filename}`。
- 不引入常驻后台循环替代 XXL-JOB。
