-- 销售训练资料批次关联 documents 文件台账迁移脚本。
-- 适用场景：已经初始化过 MySQL 表结构的环境，需要补齐 training_knowledge_batches.document_id。
-- 说明：新上传会自动写 documents，并把 batch.document_id 指向对应文件。
-- 历史批次如果没有 document_id，会继续使用 source_file/file_path/file_md5 兼容读取。

USE ai_rag_agent;

SET @column_exists := (
  SELECT COUNT(*)
  FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'training_knowledge_batches'
    AND column_name = 'document_id'
);
SET @sql := IF(
  @column_exists = 0,
  'ALTER TABLE training_knowledge_batches ADD COLUMN document_id VARCHAR(64) NULL COMMENT ''关联 documents.document_id，文件基础信息统一保存在 documents 表'' AFTER batch_id',
  'SELECT ''training_knowledge_batches.document_id 已存在，跳过字段迁移'' AS message'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @index_exists := (
  SELECT COUNT(*)
  FROM information_schema.statistics
  WHERE table_schema = DATABASE()
    AND table_name = 'training_knowledge_batches'
    AND index_name = 'idx_training_batches_document'
);
SET @sql := IF(
  @index_exists = 0,
  'CREATE INDEX idx_training_batches_document ON training_knowledge_batches(document_id)',
  'SELECT ''idx_training_batches_document 已存在，跳过索引迁移'' AS message'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_exists := (
  SELECT COUNT(*)
  FROM information_schema.referential_constraints
  WHERE constraint_schema = DATABASE()
    AND table_name = 'training_knowledge_batches'
    AND constraint_name = 'fk_training_batches_document'
);
SET @sql := IF(
  @fk_exists = 0,
  'ALTER TABLE training_knowledge_batches ADD CONSTRAINT fk_training_batches_document FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE SET NULL',
  'SELECT ''fk_training_batches_document 已存在，跳过外键迁移'' AS message'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
