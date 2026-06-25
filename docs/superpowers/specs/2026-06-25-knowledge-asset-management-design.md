# 知识资产管理重新设计方案

## 1. 背景

当前项目里有两类看起来相似、但业务语义不同的数据：

- 通用知识库文件：用于普通 RAG 问答、考试、知识库预览等场景。
- 销售训练资料：用于销售陪练、AI 客户生成、训练阶段、评分设置等场景。

两类数据现在都和文件上传、MinIO、Qdrant、MySQL 有关系，但页面和接口边界不够清晰，导致用户容易看到重复入口、混合列表、测试脏数据，以及删除不彻底的问题。

本次重新设计允许不兼容老数据，目标是重新建立清晰边界：一个上传文件就是一个知识资产，文件资产必须能被统一追踪、统一预览、统一删除。

## 2. 设计目标

1. `documents` 作为唯一文件资产主表，所有上传文件都必须有一条 `documents` 记录。
2. 销售训练资料如果需要额外业务状态，使用扩展表关联 `documents.document_id`，不重复承担文件台账职责。
3. 通用知识库和销售训练资料在页面、接口、数据范围上彻底分开。
4. 删除操作必须全链路删除：MySQL、Qdrant、MinIO 都要同步清理。
5. 页面展示不再混乱：首页知识库只管通用知识库，销售陪练只管销售训练资料。
6. 老脏数据可以不迁移；必要时用一次性清理脚本删除。

## 3. 数据模型设计

### 3.1 documents 文件资产主表

`documents` 是所有上传文件的主表。无论文件来自通用知识库、销售训练资料、考试资料，还是后续新增业务，都必须先写入该表。

核心字段：

| 字段 | 含义 |
| --- | --- |
| `document_id` | 文件资产唯一编号，系统内所有文件相关业务都以它为主键关联。 |
| `filename` | 用户上传时的原始文件名。 |
| `file_type` | 文件类型，例如 `txt`、`pdf`、`docx`、`csv`。 |
| `file_size` | 文件大小，单位字节。 |
| `file_md5` | 文件 MD5，用于去重和追踪同一文件。 |
| `collection_name` | 写入的 Qdrant 集合名称。 |
| `document_type` | 文档内容类型，例如普通文本、问答资料、训练案例。 |
| `split_strategy` | 分片策略，例如递归分片、编号问答分片、LMS 案例分片。 |
| `chunk_count` | 当前文件写入向量库的切片数量。 |
| `status` | 文件状态，例如 `uploaded`、`indexing`、`indexed`、`failed`。新删除流程不再依赖 `deleted` 状态，删除后记录不存在。 |
| `version` | 文件重建索引版本号。 |
| `storage_type` | 文件存储方式，当前为 `minio`。 |
| `bucket_name` | MinIO 桶名。 |
| `object_name` | MinIO 对象路径。 |
| `public_url` | 可选的公共访问地址。 |
| `error_message` | 入库或重建失败时的错误信息。 |
| `created_at` | 创建时间。 |
| `updated_at` | 更新时间。 |

`documents` 只表达“文件资产本身”，不直接表达销售训练的发布、回滚、评分、角色生成等业务流程。删除文件资产时，该表记录需要物理删除。

### 3.2 training_knowledge_batches 销售训练资料扩展表

`training_knowledge_batches` 只保存销售训练资料特有的业务信息。它必须通过 `document_id` 关联 `documents`。

核心字段：

| 字段 | 含义 |
| --- | --- |
| `batch_id` | 销售训练资料批次编号。 |
| `document_id` | 关联 `documents.document_id`。 |
| `source_type` | 训练资料来源类型，例如 `lms_case`、`product_doc`、`faq`。 |
| `status` | 训练资料业务状态，例如 `parsing`、`parsed`、`published`、`failed`。新删除流程不再依赖 `deleted` 状态，删除后记录不存在。 |
| `version_group_id` | 同一份训练资料多版本所属的版本组。 |
| `version_no` | 当前批次在版本组内的版本号。 |
| `is_active` | 是否为当前生效版本。 |
| `quality_report_json` | 解析质量、切片质量、发布校验等信息。 |
| `created_by` | 创建人或来源标识。 |
| `created_at` | 创建时间。 |
| `updated_at` | 更新时间。 |

训练资料列表展示时，通过 `training_knowledge_batches` join `documents` 获取文件名、文件大小、文件类型、MinIO 对象路径等文件基础信息。删除销售训练资料时，该表关联记录需要物理删除。

### 3.3 Qdrant 向量数据

Qdrant 不作为文件台账，只保存检索用向量点。

每个向量点的 metadata 必须包含：

| 字段 | 含义 |
| --- | --- |
| `document_id` | 所属文件资产编号，删除文件时必须用它删除全部向量点。 |
| `collection_name` | 所属集合。 |
| `filename` | 来源文件名，便于日志和调试。 |
| `chunk_index` | 文件内切片序号。 |
| `document_type` | 文档类型。 |
| `split_strategy` | 分片策略。 |
| `batch_id` | 训练资料向量必须携带，通用知识库可以为空。 |
| `case_part` | LMS 案例等训练资料的结构化片段类型。 |
| `visibility` | 训练资料向量用途，例如草稿、已发布、检索可见。 |

删除文件时，必须按 `document_id` 删除所有集合里的相关向量点。销售训练资料还要按 `batch_id` 删除 staging 和 published 集合中的向量点，避免残留。

### 3.4 MinIO 对象

MinIO 保存原始上传文件。`documents.bucket_name` 和 `documents.object_name` 是唯一可信来源。

对象路径：

```text
documents/{document_id}/{filename}
```

删除文件时必须删除该 MinIO 对象。若对象不存在，删除流程不应该中断，但要记录中文日志。

## 4. 页面边界设计

### 4.1 首页知识库管理

首页知识库管理只展示通用知识库文件。

数据来源：

- `GET /knowledge/files`
- 只返回非销售训练集合。

默认排除集合：

- `sales_training_cases`
- `sales_training_cases_staging`

功能范围：

- 上传通用知识库文件。
- 预览原文件。
- 重建索引。
- 删除文件资产。
- 按 collection 查看。

不再展示：

- 销售训练资料上传入口。
- 销售训练批次。
- 发布、回滚、重切、版本链。
- 训练切片质量报告。

### 4.2 销售陪练资料管理

销售陪练资料管理只展示销售训练资料。

数据来源：

- `GET /training/knowledge/batches`
- 查询 `training_knowledge_batches`，并关联 `documents` 获取文件基础信息。

功能范围：

- 上传销售训练资料。
- 解析预览。
- 发布到正式训练向量集合。
- LLM 重切。
- 查看版本链。
- 回滚生效版本。
- 删除训练资料资产。

销售训练资料不再作为普通知识库文件展示在首页知识库弹窗里。

## 5. 接口边界设计

### 5.1 通用知识库接口

`GET /knowledge/files`

默认只返回通用知识库文件。

接口参数：

| 参数 | 含义 |
| --- | --- |
| `include_training` | 是否包含销售训练集合，默认 `false`。仅排查数据时使用。 |

默认行为：

```text
collection_name NOT IN ('sales_training_cases', 'sales_training_cases_staging')
```

### 5.2 销售训练资料接口

`GET /training/knowledge/batches`

只返回销售训练批次。返回值中可以包含关联文件字段，例如：

| 字段 | 含义 |
| --- | --- |
| `batch_id` | 训练资料批次编号。 |
| `document_id` | 文件资产编号。 |
| `source_file` | 来源文件名，来自 `documents.filename`。 |
| `file_type` | 文件类型，来自 `documents.file_type`。 |
| `file_size` | 文件大小，来自 `documents.file_size`。 |
| `status` | 训练资料业务状态。 |
| `source_type` | 资料来源类型。 |
| `chunk_count` | 切片数量。 |
| `created_at` | 创建时间。 |
| `updated_at` | 更新时间。 |

### 5.3 统一删除接口

通用知识库删除和销售训练资料删除都必须复用统一删除逻辑。

内部服务：

```text
DocumentAssetService.delete_document_asset(document_id)
```

如果前端从训练批次删除，可以先通过 `batch_id` 找到 `document_id`，再调用统一删除服务。

## 6. 删除流程设计

删除必须是强一致的业务动作，不能只删一张表。

流程：

1. 根据 `document_id` 查询 `documents`。
2. 查询是否存在关联的 `training_knowledge_batches`。
3. 从 Qdrant 删除所有 metadata.document_id 等于当前文件的向量点。
4. 如果有关联训练批次，再从训练 staging 和 published 集合中按 `batch_id` 删除向量点。
5. 删除 MinIO 对象：`bucket_name + object_name`。
6. 物理删除 `training_knowledge_batches` 记录。
7. 物理删除 `documents` 记录。
8. 输出中文日志，记录每一步结果。

本次删除策略使用硬删除：

- MySQL 中 `documents` 记录物理删除。
- MySQL 中关联的 `training_knowledge_batches` 记录物理删除。
- Qdrant 中关联向量点物理删除。
- MinIO 中原文件对象物理删除。

如果删除中途失败，接口必须返回错误，并通过中文日志说明失败环节。后续可以新增独立审计日志表记录删除历史，但不能用 `documents` 或 `training_knowledge_batches` 的残留记录代替删除。

## 7. 老数据处理

本次设计允许不兼容老数据。

处理方式：

1. 页面默认不展示训练集合中的普通 documents 记录。
2. 老库里如果已有 `status = deleted` 的历史记录，页面默认不展示。
3. 提供一次性清理脚本，删除明显测试数据，例如：
   - `preview.txt`
   - `version.txt`
   - `weak.txt`
   - `chain_` 开头文件
   - `created_by = tester` 的训练批次
4. 清理脚本必须同时清理：
   - `documents`
   - `training_knowledge_batches`
   - Qdrant 向量点
   - MinIO 对象

## 8. 设计模式

### 8.1 外观模式

新增或整理 `DocumentAssetService`，对外提供统一文件资产操作。

选择原因：

删除一个文件涉及 MySQL、Qdrant、MinIO、训练批次等多个子系统。页面和路由不应该理解这些细节，只需要调用“删除文件资产”。

### 8.2 仓储模式

继续保留并明确：

- `KnowledgeStore` 负责 `documents`。
- `TrainingRepository` 负责 `training_knowledge_batches`。

选择原因：

数据库访问逻辑集中在仓储层，服务层负责业务编排，路由层只处理请求响应。

### 8.3 策略模式

上传和解析阶段继续按资料类型选择策略：

- 通用文本策略。
- FAQ 策略。
- LMS 案例策略。
- 后续可扩展竞品资料、成功案例、术语表等策略。

选择原因：

不同文档的解析、分片、metadata 写入不同，不应该堆在一个巨大函数里判断。

## 9. 实施范围

本次设计确认后，实施以下改动：

1. 后端增加或整理统一文件资产删除服务。
2. `/knowledge/files` 默认排除销售训练集合。
3. 首页知识库弹窗移除销售训练资料 Tab。
4. 销售训练资料保留在销售陪练页面。
5. 销售训练删除逻辑改为通过统一文件资产删除服务清理全链路数据。
6. 补充测试，验证删除时 MySQL、Qdrant、MinIO 都会被调用。

## 10. 非目标

本次不做以下事情：

1. 不迁移所有历史脏数据。
2. 不重新设计 RAG 检索链路。
3. 不修改聊天、考试、AI 客户生成的核心业务流程。
4. 不把销售训练所有字段强行塞进 `documents`。

## 11. 验收标准

1. 首页知识库管理只展示通用知识库文件。
2. 销售训练资料只在销售陪练资料管理中展示。
3. 删除通用知识库文件后：
   - 页面不再展示。
   - `documents` 中该记录不存在。
   - Qdrant 中该 `document_id` 的向量点被删除。
   - MinIO 原文件被删除。
4. 删除销售训练资料后：
   - 销售训练列表不再展示。
   - `training_knowledge_batches` 中该记录不存在。
   - `documents` 中该记录不存在。
   - Qdrant 中该 `document_id` 和相关 `batch_id` 的向量点被删除。
   - MinIO 原文件被删除。
5. 删除过程失败时，接口返回明确错误，日志能定位失败环节。
