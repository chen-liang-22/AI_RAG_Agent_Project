# MinIO 唯一文件存储改造设计

## 背景

当前知识库上传、预览、重建索引和销售训练资料上传都把原始文件长期保存在本地 `uploads/` 目录。这样会带来两个问题：

1. 后端多实例部署时，本地文件不共享，任意实例都可能读不到文件。
2. 文件删除、迁移、备份和公开预览分散在本地目录和数据库字段里，后续维护成本高。

本次改造目标是把 MinIO 作为唯一持久文件存储。本地磁盘只作为解析 PDF、DOCX、TXT 时的临时缓存，不再作为业务文件来源。

## 改造范围

- 普通知识库上传预览：上传文件直接写入 MinIO 临时对象。
- 普通知识库确认入库：把临时对象复制为正式对象，`documents` 记录 MinIO 对象信息。
- 普通知识库文件预览：从 MinIO 下载临时文件后抽取文本。
- 普通知识库删除：删除 Qdrant points、软删 `documents`、删除 MinIO 对象。
- 普通知识库重建索引：从 MinIO 下载临时文件后重新解析。
- 销售训练资料上传：上传文件直接写入 MinIO 正式对象，`documents` 统一记录文件信息。
- 销售训练重新切分：从 MinIO 下载临时文件后解析。
- 历史文件迁移：一次性把 `documents.file_path` 指向的旧本地文件上传到 MinIO，并回填对象字段。

## 存储路径规范

| 场景 | MinIO object_name |
| --- | --- |
| 知识库临时预览 | `previews/{upload_id}/{filename}` |
| 知识库正式文件 | `documents/{document_id}/{filename}` |
| 销售训练正式文件 | `training/{document_id}/{filename}` |
| 脚本迁移旧文件 | 优先按当前文件归属写入 `documents/{document_id}/{filename}` |

## documents 字段设计

保留 `file_path` 兼容接口字段，但它不再表示本地路径，而是保存 `minio://bucket/object_name` 形式的存储 URI。

新增字段：

| 字段 | 作用 |
| --- | --- |
| `storage_type` | 文件存储类型。本次统一为 `minio` |
| `bucket_name` | MinIO 桶名，例如 `pub` |
| `object_name` | MinIO 对象路径 |
| `public_url` | 可公开访问的文件 URL |

## 运行时原则

1. 业务代码不直接读 `uploads/`。
2. 业务代码不直接调用 MinIO SDK，统一走 `FileStorageService`。
3. 解析库如果必须接收本地路径，则由 `FileStorageService.download_to_temp_file()` 下载临时文件。
4. 临时文件只放在系统临时目录或 `uploads/_tmp_minio/`，用完即删。
5. 删除文件时必须同步删除 MinIO 对象，避免对象存储泄漏。

## 迁移策略

1. 执行表结构迁移 SQL，给 `documents` 补齐 MinIO 字段。
2. 确认 MinIO `pub` 桶存在，并配置公开读。
3. 运行迁移脚本，把历史 `documents.file_path` 指向的本地文件上传到 MinIO。
4. 脚本回填：
   - `storage_type='minio'`
   - `bucket_name='pub'`
   - `object_name='documents/{document_id}/{filename}'`
   - `public_url='http://127.0.0.1:9000/pub/documents/{document_id}/{filename}'`
   - `file_path='minio://pub/documents/{document_id}/{filename}'`
5. 迁移完成后，应用代码不再按本地路径读取旧文件。

## 回滚策略

如果 MinIO 不可用，可以回滚到改造前代码版本。数据库新增字段不会影响旧代码读取 `file_path`，但迁移后 `file_path` 已变成 `minio://...`，如需旧代码读取本地文件，需要从备份恢复迁移前的 `file_path`。
