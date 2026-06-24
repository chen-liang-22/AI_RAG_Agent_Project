"""生成 MySQL 初始化 SQL。

这个脚本只生成建表和系统默认字典数据，便于统一维护 MySQL 初始化脚本。
字典种子直接复用 rag.knowledge_store.DEFAULT_DICTIONARY_ITEMS，避免手写 SQL 和代码口径不一致。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_store import DEFAULT_DICTIONARY_ITEMS


OUTPUT_PATH = Path("docs/mysql初始化建表和基础数据.sql")


def sql_quote(value: Any) -> str:
    """把 Python 值转换成 MySQL 字符串字面量。"""

    if value is None:
        return "NULL"
    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def dictionary_item_id(dictionary_code: str, item_code: str) -> str:
    """生成稳定的字典项 ID，避免每次初始化产生不同主键。"""

    digest = hashlib.md5(f"{dictionary_code}:{item_code}".encode("utf-8")).hexdigest()
    return f"dict_{digest}"


def render_dictionary_seed_sql() -> str:
    """生成系统默认字典的 INSERT 语句。"""

    lines: list[str] = [
        "-- 清理已废弃字典，避免旧库迁移后继续展示过期配置。",
        "DELETE FROM dictionary_items",
        "WHERE dictionary_code IN ('collection_domain_keyword', 'sales_customer_profile_template');",
        "",
        "-- 初始化系统默认字典；重复执行时会更新展示名称、排序、描述和元数据。",
    ]
    for dictionary in DEFAULT_DICTIONARY_ITEMS:
        dictionary_code = str(dictionary["dictionary_code"])
        dictionary_name = str(dictionary["dictionary_name"])
        item_id_by_code: dict[str, str] = {}
        lines.append("")
        lines.append(f"-- 字典：{dictionary_name}（{dictionary_code}）")
        for item in dictionary["items"]:
            item_code, item_name, parent_code, sort_order, description = item[:5]
            metadata = item[5] if len(item) > 5 else None
            metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) if metadata else None
            parent_item_id = item_id_by_code.get(parent_code or "")
            item_level = 1 if parent_item_id is None else 2
            item_id = dictionary_item_id(dictionary_code, str(item_code))
            values = [
                sql_quote(item_id),
                sql_quote(dictionary_code),
                sql_quote(dictionary_name),
                sql_quote(item_code),
                sql_quote(item_name),
                sql_quote(parent_item_id),
                str(item_level),
                str(int(sort_order)),
                "1",
                sql_quote(description),
                sql_quote(metadata_json),
            ]
            lines.append(
                "INSERT INTO dictionary_items ("
                "dictionary_item_id, dictionary_code, dictionary_name, item_code, item_name, "
                "parent_item_id, item_level, sort_order, enabled, description, metadata_json, created_at, updated_at"
                f") VALUES ({', '.join(values)}, NOW(), NOW()) "
                "ON DUPLICATE KEY UPDATE "
                "dictionary_name = VALUES(dictionary_name), "
                "item_name = VALUES(item_name), "
                "parent_item_id = VALUES(parent_item_id), "
                "item_level = VALUES(item_level), "
                "sort_order = VALUES(sort_order), "
                "enabled = VALUES(enabled), "
                "description = VALUES(description), "
                "metadata_json = VALUES(metadata_json), "
                "updated_at = NOW();"
            )
            item_id_by_code[str(item_code)] = item_id
    return "\n".join(lines)


DDL_SQL = """-- AI_RAG_Agent_Project MySQL 初始化脚本。
-- 适用范围：当前项目业务元数据表，不包含 Qdrant 向量数据。
-- 说明：训练知识切片正文不进入 MySQL，待审核切片在临时 Qdrant，发布后进入正式 Qdrant。

CREATE DATABASE IF NOT EXISTS ai_rag_agent
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE ai_rag_agent;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

CREATE TABLE IF NOT EXISTS documents (
  document_id VARCHAR(64) NOT NULL COMMENT '文档唯一编号',
  filename VARCHAR(255) NOT NULL COMMENT '原始文件名',
  file_path VARCHAR(1024) NOT NULL COMMENT '服务端保存路径',
  file_type VARCHAR(32) NOT NULL COMMENT '文件类型，例如 txt、pdf、docx',
  file_md5 CHAR(32) NOT NULL COMMENT '文件 MD5，用于去重',
  file_size BIGINT NOT NULL COMMENT '文件大小，单位字节',
  status VARCHAR(32) NOT NULL COMMENT '文档状态，例如 uploaded、indexing、indexed、failed、deleted',
  version INT NOT NULL DEFAULT 1 COMMENT '文档版本号',
  chunk_count INT NOT NULL DEFAULT 0 COMMENT '写入向量库的切片数量',
  collection_name VARCHAR(128) NOT NULL DEFAULT 'agent' COMMENT 'Qdrant collection 名称',
  document_type VARCHAR(64) NOT NULL DEFAULT 'text' COMMENT '文档结构类型，来自 document_structure 字典',
  split_strategy VARCHAR(64) NOT NULL DEFAULT 'recursive' COMMENT '切分策略，来自 split_strategy 字典',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  error_message TEXT NULL COMMENT '失败原因',
  PRIMARY KEY (document_id),
  KEY idx_documents_file_md5 (file_md5),
  KEY idx_documents_collection (collection_name),
  KEY idx_documents_status_updated (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='普通知识库文件元数据表';

CREATE TABLE IF NOT EXISTS conversations (
  conversation_id VARCHAR(64) NOT NULL COMMENT '会话唯一编号',
  user_id VARCHAR(128) NULL COMMENT '用户编号，当前可为空',
  title VARCHAR(255) NULL COMMENT '会话标题',
  status VARCHAR(32) NOT NULL DEFAULT 'active' COMMENT '会话状态，例如 active、deleted',
  message_count INT NOT NULL DEFAULT 0 COMMENT '消息数量',
  summary TEXT NULL COMMENT '会话摘要',
  metadata_json JSON NULL COMMENT '会话扩展元数据',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  last_message_at DATETIME NULL COMMENT '最后消息时间',
  PRIMARY KEY (conversation_id),
  KEY idx_conversations_user_updated (user_id, updated_at),
  KEY idx_conversations_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='聊天会话表';

CREATE TABLE IF NOT EXISTS conversation_messages (
  message_id VARCHAR(64) NOT NULL COMMENT '消息唯一编号',
  conversation_id VARCHAR(64) NOT NULL COMMENT '所属会话编号',
  sequence_no INT NOT NULL COMMENT '会话内消息序号',
  role VARCHAR(32) NOT NULL COMMENT '消息角色，例如 user、assistant、system',
  content LONGTEXT NOT NULL COMMENT '消息正文',
  content_type VARCHAR(32) NOT NULL DEFAULT 'text' COMMENT '内容类型',
  model_name VARCHAR(128) NULL COMMENT '生成该消息使用的模型名',
  token_count INT NULL COMMENT '消息 token 数量',
  metadata_json JSON NULL COMMENT '消息扩展元数据',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  PRIMARY KEY (message_id),
  UNIQUE KEY uk_messages_conversation_sequence (conversation_id, sequence_no),
  KEY idx_messages_conversation_created (conversation_id, created_at),
  CONSTRAINT fk_messages_conversation
    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='聊天消息表';

CREATE TABLE IF NOT EXISTS dictionary_items (
  dictionary_item_id VARCHAR(64) NOT NULL COMMENT '字典项唯一编号',
  dictionary_code VARCHAR(128) NOT NULL COMMENT '字典编码，例如 model_mode',
  dictionary_name VARCHAR(128) NOT NULL COMMENT '字典名称',
  item_code VARCHAR(128) NOT NULL COMMENT '字典项编码',
  item_name VARCHAR(255) NOT NULL COMMENT '字典项展示名称',
  parent_item_id VARCHAR(64) NULL COMMENT '父级字典项编号，用于多层级字典',
  item_level INT NOT NULL DEFAULT 1 COMMENT '层级，一级为 1，二级为 2',
  sort_order INT NOT NULL DEFAULT 0 COMMENT '排序值，越小越靠前',
  enabled TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否启用，1 启用，0 禁用',
  description TEXT NULL COMMENT '字典项说明',
  metadata_json JSON NULL COMMENT '字典项扩展元数据',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (dictionary_item_id),
  UNIQUE KEY uk_dictionary_code_item (dictionary_code, item_code),
  KEY idx_dictionary_items_code_parent (dictionary_code, parent_item_id, sort_order),
  CONSTRAINT fk_dictionary_parent
    FOREIGN KEY (parent_item_id) REFERENCES dictionary_items (dictionary_item_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统字典表';

CREATE TABLE IF NOT EXISTS exam_sessions (
  session_id VARCHAR(64) NOT NULL COMMENT '测评会话编号',
  user_id VARCHAR(128) NULL COMMENT '用户编号，当前可为空',
  title VARCHAR(255) NULL COMMENT '测评标题',
  collection_name VARCHAR(128) NOT NULL COMMENT '测评来源 Qdrant collection',
  document_id VARCHAR(64) NULL COMMENT '来源文档编号',
  filename VARCHAR(255) NULL COMMENT '来源文件名',
  section_path VARCHAR(1024) NULL COMMENT '来源章节路径',
  round_count INT NOT NULL COMMENT '测评轮数',
  question_types_json JSON NULL COMMENT '题型配置',
  status VARCHAR(32) NOT NULL DEFAULT 'active' COMMENT '测评状态',
  current_round INT NOT NULL DEFAULT 1 COMMENT '当前轮次',
  answered_count INT NOT NULL DEFAULT 0 COMMENT '已答题数量',
  total_score DECIMAL(10,2) NOT NULL DEFAULT 0 COMMENT '当前总得分',
  max_score DECIMAL(10,2) NOT NULL DEFAULT 100 COMMENT '最高分',
  model_mode VARCHAR(32) NULL COMMENT '生成题目使用的模型档位',
  metadata_json JSON NULL COMMENT '测评扩展元数据，例如随机种子',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  completed_at DATETIME NULL COMMENT '完成时间',
  PRIMARY KEY (session_id),
  KEY idx_exam_sessions_updated (updated_at),
  KEY idx_exam_sessions_user_status (user_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识掌握度测评会话表';

CREATE TABLE IF NOT EXISTS exam_questions (
  exam_question_id VARCHAR(64) NOT NULL COMMENT '测评题目编号',
  session_id VARCHAR(64) NOT NULL COMMENT '所属测评会话编号',
  round_no INT NOT NULL COMMENT '轮次编号',
  source_question_id VARCHAR(128) NULL COMMENT '来源题目编号',
  source_document_id VARCHAR(64) NULL COMMENT '来源文档编号',
  source_filename VARCHAR(255) NULL COMMENT '来源文件名',
  source_page INT NULL COMMENT '来源页码',
  section_path VARCHAR(1024) NULL COMMENT '来源章节路径',
  question_type VARCHAR(32) NOT NULL COMMENT '题型，例如 single_choice、multi_choice、judge、short_answer',
  prompt LONGTEXT NOT NULL COMMENT '题干',
  options_json JSON NULL COMMENT '选择题选项',
  correct_answer_json JSON NULL COMMENT '正确答案',
  reference_answer LONGTEXT NULL COMMENT '参考答案',
  user_answer LONGTEXT NULL COMMENT '用户答案',
  is_correct TINYINT(1) NULL COMMENT '是否答对，1 对，0 错',
  score DECIMAL(10,2) NULL COMMENT '本题得分',
  max_score DECIMAL(10,2) NOT NULL COMMENT '本题满分',
  analysis_json JSON NULL COMMENT '判题分析',
  status VARCHAR(32) NOT NULL DEFAULT 'pending' COMMENT '题目状态，例如 pending、answered',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  answered_at DATETIME NULL COMMENT '答题时间',
  PRIMARY KEY (exam_question_id),
  UNIQUE KEY uk_exam_questions_session_round (session_id, round_no),
  KEY idx_exam_questions_session_round (session_id, round_no),
  CONSTRAINT fk_exam_questions_session
    FOREIGN KEY (session_id) REFERENCES exam_sessions (session_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识掌握度测评题目表';

CREATE TABLE IF NOT EXISTS training_knowledge_batches (
  batch_id VARCHAR(64) NOT NULL COMMENT '训练资料上传批次编号',
  source_type VARCHAR(64) NOT NULL COMMENT '资料来源类型，例如 lms_case',
  source_file VARCHAR(255) NOT NULL COMMENT '原始文件名',
  file_path VARCHAR(1024) NULL COMMENT '服务端保存路径',
  file_md5 CHAR(32) NULL COMMENT '文件 MD5',
  version_group_id VARCHAR(64) NULL COMMENT '版本组编号，同一资料多版本共享',
  version_no INT NOT NULL DEFAULT 1 COMMENT '版本号，从 1 递增',
  previous_batch_id VARCHAR(64) NULL COMMENT '上一版本批次编号',
  is_current TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否当前参与训练检索',
  profile_type VARCHAR(64) NULL COMMENT '历史兼容字段，新上传通常为空',
  task_type VARCHAR(64) NULL COMMENT '历史兼容字段，新上传通常为空',
  industry VARCHAR(128) NULL COMMENT '历史兼容字段，新上传通常为空',
  difficulty VARCHAR(64) NULL COMMENT '历史兼容字段，新上传通常为空',
  visibility_default VARCHAR(32) NULL COMMENT '默认可见性，例如 visible',
  status VARCHAR(32) NOT NULL COMMENT '批次状态，例如 pending_review、published、archived',
  chunk_count INT NOT NULL DEFAULT 0 COMMENT '切片数量',
  point_count INT NOT NULL DEFAULT 0 COMMENT '当前阶段向量点数量',
  error_message TEXT NULL COMMENT '失败原因',
  quality_report_json JSON NULL COMMENT '切片质量报告和发布验证结果',
  created_by VARCHAR(128) NULL COMMENT '上传人',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (batch_id),
  KEY idx_training_batches_md5_status (file_md5, status),
  KEY idx_training_batches_version_group (version_group_id, version_no),
  KEY idx_training_batches_current_status (status, is_current, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练资料上传批次表';

CREATE TABLE IF NOT EXISTS training_plans (
  plan_id VARCHAR(64) NOT NULL COMMENT '训练方案编号',
  plan_name VARCHAR(255) NOT NULL COMMENT '训练名称，允许重复',
  trainee_id VARCHAR(128) NOT NULL COMMENT '学员编号',
  trainee_name VARCHAR(128) NOT NULL COMMENT '学员姓名',
  profile_type VARCHAR(64) NOT NULL COMMENT '画像类型',
  trainee_json JSON NOT NULL COMMENT '学员画像快照',
  selected_fields_json JSON NOT NULL COMMENT '客户画像字段选择快照',
  scenario_description TEXT NOT NULL COMMENT '训练场景描述',
  extra_details TEXT NULL COMMENT '补充细节',
  model_mode VARCHAR(32) NULL COMMENT '模型档位',
  active_profile_id VARCHAR(64) NULL COMMENT '当前 AI 客户角色编号',
  active_setting_id VARCHAR(64) NULL COMMENT '当前训练阶段设置编号',
  role_status VARCHAR(32) NOT NULL COMMENT '角色生成状态，例如 pending、generated、stale',
  goal_status VARCHAR(32) NOT NULL COMMENT '训练阶段状态，例如 pending、generated、stale',
  score_status VARCHAR(32) NOT NULL COMMENT '评分设置状态，例如 pending、generated、stale',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (plan_id),
  KEY idx_training_plans_trainee_updated (trainee_id, updated_at),
  KEY idx_training_plans_name (plan_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练方案表';

CREATE TABLE IF NOT EXISTS training_role_profiles (
  profile_id VARCHAR(64) NOT NULL COMMENT 'AI 客户角色编号',
  trainee_id VARCHAR(128) NOT NULL COMMENT '学员编号',
  plan_id VARCHAR(64) NULL COMMENT '所属训练方案编号',
  profile_type VARCHAR(64) NOT NULL COMMENT '画像类型',
  visible_profile_json JSON NOT NULL COMMENT '学员可见客户画像',
  hidden_profile_json JSON NOT NULL COMMENT 'AI 内部使用隐藏画像',
  role_profile_json JSON NOT NULL COMMENT 'AI 陪练角色完整画像',
  role_confirm_card_json JSON NOT NULL COMMENT '前端确认卡片数据',
  selected_fields_json JSON NULL COMMENT '生成角色时的字段选择',
  scenario_description TEXT NULL COMMENT '场景描述',
  extra_details TEXT NULL COMMENT '补充细节',
  retrieved_evidence_json JSON NULL COMMENT '生成角色引用的检索证据',
  status VARCHAR(32) NOT NULL COMMENT '角色状态，例如 confirmed',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (profile_id),
  KEY idx_role_profiles_plan (plan_id),
  KEY idx_role_profiles_trainee (trainee_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练 AI 客户角色表';

CREATE TABLE IF NOT EXISTS training_goal_settings (
  setting_id VARCHAR(64) NOT NULL COMMENT '训练阶段设置编号',
  profile_id VARCHAR(64) NOT NULL COMMENT 'AI 客户角色编号',
  plan_id VARCHAR(64) NULL COMMENT '所属训练方案编号',
  trainee_id VARCHAR(128) NOT NULL COMMENT '学员编号',
  training_mode VARCHAR(32) NOT NULL COMMENT '训练模式，例如 open',
  training_purpose TEXT NOT NULL COMMENT '训练目标',
  round_limit INT NOT NULL COMMENT '训练轮数上限',
  stages_json JSON NOT NULL COMMENT '训练阶段配置',
  scoring_rules_json JSON NULL COMMENT '评分规则配置',
  status VARCHAR(32) NOT NULL COMMENT '设置状态，例如 confirmed',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (setting_id),
  KEY idx_goal_settings_plan (plan_id),
  KEY idx_goal_settings_profile (profile_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练阶段和评分设置表';

CREATE TABLE IF NOT EXISTS sales_training_sessions (
  session_id VARCHAR(64) NOT NULL COMMENT '训练会话编号',
  profile_id VARCHAR(64) NOT NULL COMMENT 'AI 客户角色编号',
  setting_id VARCHAR(64) NOT NULL COMMENT '训练阶段设置编号',
  trainee_id VARCHAR(128) NOT NULL COMMENT '学员编号',
  training_mode VARCHAR(32) NOT NULL COMMENT '训练模式',
  response_mode VARCHAR(32) NOT NULL COMMENT '响应模式，例如 stream',
  current_stage_no INT NOT NULL DEFAULT 1 COMMENT '当前训练阶段编号',
  status VARCHAR(32) NOT NULL COMMENT '会话状态，例如 active、completed、failed',
  round_limit INT NOT NULL COMMENT '轮数上限',
  total_score INT NULL COMMENT '最终得分',
  level VARCHAR(64) NULL COMMENT '评分等级',
  report_json JSON NULL COMMENT '训练复盘报告',
  started_at DATETIME NOT NULL COMMENT '开始时间',
  ended_at DATETIME NULL COMMENT '结束时间',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (session_id),
  KEY idx_training_sessions_trainee_status (trainee_id, status),
  KEY idx_training_sessions_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练会话表';

CREATE TABLE IF NOT EXISTS sales_training_turns (
  turn_id VARCHAR(64) NOT NULL COMMENT '训练对话轮次编号',
  session_id VARCHAR(64) NOT NULL COMMENT '所属训练会话编号',
  role VARCHAR(32) NOT NULL COMMENT '发言角色，例如 customer、trainee、system',
  content LONGTEXT NOT NULL COMMENT '发言内容',
  round_no INT NOT NULL COMMENT '训练轮次编号',
  stage_no INT NOT NULL DEFAULT 1 COMMENT '训练阶段编号',
  response_mode VARCHAR(32) NULL COMMENT '响应模式',
  started_at DATETIME NULL COMMENT '本轮开始时间',
  submitted_at DATETIME NULL COMMENT '学员提交时间',
  response_seconds DECIMAL(10,3) NULL COMMENT '响应耗时秒数',
  retrieved_chunk_ids_json JSON NULL COMMENT '本轮检索命中的切片编号列表',
  retrieved_evidence_json JSON NULL COMMENT '本轮检索证据',
  stage_decision_json JSON NULL COMMENT '阶段推进判断结果',
  coach_analysis_json JSON NULL COMMENT '教练分析结果',
  metadata_json JSON NULL COMMENT '轮次扩展元数据',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  PRIMARY KEY (turn_id),
  KEY idx_training_turns_session_round (session_id, round_no),
  KEY idx_training_turns_session_created (session_id, created_at),
  CONSTRAINT fk_training_turns_session
    FOREIGN KEY (session_id) REFERENCES sales_training_sessions (session_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练对话轮次表';

CREATE TABLE IF NOT EXISTS sales_training_scores (
  score_id VARCHAR(64) NOT NULL COMMENT '训练评分编号',
  session_id VARCHAR(64) NOT NULL COMMENT '所属训练会话编号',
  general_score INT NOT NULL COMMENT '通用能力得分',
  stage_score INT NOT NULL COMMENT '阶段目标得分',
  penalty_score INT NOT NULL COMMENT '扣分',
  final_score INT NOT NULL COMMENT '最终得分',
  level VARCHAR(64) NOT NULL COMMENT '评分等级',
  is_passed TINYINT(1) NOT NULL COMMENT '是否通过，1 通过，0 未通过',
  detail_json JSON NOT NULL COMMENT '评分明细',
  review_status VARCHAR(32) NOT NULL COMMENT '复核状态，例如 confirmed',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (score_id),
  KEY idx_training_scores_session_updated (session_id, updated_at),
  CONSTRAINT fk_training_scores_session
    FOREIGN KEY (session_id) REFERENCES sales_training_sessions (session_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='销售训练评分表';

SET FOREIGN_KEY_CHECKS = 1;
"""


def main() -> None:
    """生成 MySQL 初始化 SQL 文件。"""

    sql = DDL_SQL.rstrip() + "\n\n" + render_dictionary_seed_sql().rstrip() + "\n"
    OUTPUT_PATH.write_text(sql, encoding="utf-8")
    print(f"已生成：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
