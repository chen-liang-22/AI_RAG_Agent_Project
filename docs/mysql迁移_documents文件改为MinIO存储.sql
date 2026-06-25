-- 迁移名称：documents 文件存储改为 MinIO。
-- 适用场景：已有 MySQL 库需要从本地 uploads 文件路径迁移到 MinIO 对象存储。
-- 执行顺序：
-- 1. 先执行本 SQL 补齐字段；
-- 2. 再运行 scripts/migrate_local_files_to_minio.py 上传历史文件并回填对象字段。

USE ai_rag_agent;

SET @schema_name = DATABASE();

SET @has_storage_type = (
  SELECT COUNT(1)
  FROM information_schema.columns
  WHERE table_schema = @schema_name
    AND table_name = 'documents'
    AND column_name = 'storage_type'
);
SET @ddl = IF(
  @has_storage_type = 0,
  'ALTER TABLE documents ADD COLUMN storage_type VARCHAR(32) NOT NULL DEFAULT ''minio'' COMMENT ''文件存储类型：minio 表示对象存储'' AFTER file_path',
  'SELECT ''documents.storage_type 已存在，跳过字段迁移'' AS message'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_bucket_name = (
  SELECT COUNT(1)
  FROM information_schema.columns
  WHERE table_schema = @schema_name
    AND table_name = 'documents'
    AND column_name = 'bucket_name'
);
SET @ddl = IF(
  @has_bucket_name = 0,
  'ALTER TABLE documents ADD COLUMN bucket_name VARCHAR(128) NULL COMMENT ''MinIO 桶名'' AFTER storage_type',
  'SELECT ''documents.bucket_name 已存在，跳过字段迁移'' AS message'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_object_name = (
  SELECT COUNT(1)
  FROM information_schema.columns
  WHERE table_schema = @schema_name
    AND table_name = 'documents'
    AND column_name = 'object_name'
);
SET @ddl = IF(
  @has_object_name = 0,
  'ALTER TABLE documents ADD COLUMN object_name VARCHAR(1024) NULL COMMENT ''MinIO 对象路径'' AFTER bucket_name',
  'SELECT ''documents.object_name 已存在，跳过字段迁移'' AS message'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_public_url = (
  SELECT COUNT(1)
  FROM information_schema.columns
  WHERE table_schema = @schema_name
    AND table_name = 'documents'
    AND column_name = 'public_url'
);
SET @ddl = IF(
  @has_public_url = 0,
  'ALTER TABLE documents ADD COLUMN public_url VARCHAR(2048) NULL COMMENT ''MinIO 公共访问地址'' AFTER object_name',
  'SELECT ''documents.public_url 已存在，跳过字段迁移'' AS message'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @has_object_index = (
  SELECT COUNT(1)
  FROM information_schema.statistics
  WHERE table_schema = @schema_name
    AND table_name = 'documents'
    AND index_name = 'idx_documents_storage_object'
);
SET @ddl = IF(
  @has_object_index = 0,
  'CREATE INDEX idx_documents_storage_object ON documents(storage_type, bucket_name, object_name(255))',
  'SELECT ''idx_documents_storage_object 已存在，跳过索引迁移'' AS message'
);
PREPARE stmt FROM @ddl;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
