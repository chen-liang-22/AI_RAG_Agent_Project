"""关系型数据库表实体类型。

项目当前使用手写 SQL 和 dict 行数据，不引入 ORM。
这些 TypedDict 只描述数据库行结构，方便 PyCharm 类型提示和后续维护字段名。
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class DocumentEntity(TypedDict):
    """documents 表实体，记录知识库文件元数据和索引状态。"""

    document_id: str  # 文件唯一编号
    filename: str  # 原始文件名
    file_path: str  # 本地存储路径
    file_type: str  # 文件扩展类型
    file_md5: str  # 文件内容 MD5，用于去重
    file_size: int  # 文件大小，单位字节
    status: str  # 文件状态，例如 uploaded/indexing/indexed/failed/deleted
    version: int  # 文件索引版本号
    chunk_count: int  # 已写入向量库的切片数量
    collection_name: str  # Qdrant collection 名称
    document_type: str  # 文档结构类型，例如 text/qa/numbered
    split_strategy: str  # 切分策略，例如 recursive/outline_qa
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    error_message: str | None  # 失败原因


class ConversationEntity(TypedDict):
    """conversations 表实体，记录一次聊天会话摘要。"""

    conversation_id: str  # 会话唯一编号
    user_id: str | None  # 用户编号
    title: str | None  # 会话标题
    status: str  # 会话状态，例如 active/deleted
    message_count: int  # 消息数量
    summary: str | None  # 会话摘要
    metadata_json: str | None  # 扩展元数据 JSON
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    last_message_at: str | None  # 最后一条消息时间


class ConversationMessageEntity(TypedDict):
    """conversation_messages 表实体，记录会话中的单条消息。"""

    message_id: str  # 消息唯一编号
    conversation_id: str  # 所属会话编号
    sequence_no: int  # 会话内顺序号
    role: str  # 消息角色，例如 user/assistant/system
    content: str  # 消息正文
    content_type: str  # 内容类型，默认 text
    model_name: str | None  # 生成该消息的模型名
    token_count: int | None  # token 数量
    metadata_json: str | None  # 扩展元数据 JSON
    created_at: str  # 创建时间


class DictionaryItemEntity(TypedDict):
    """dictionary_items 表实体，记录系统字典项。"""

    dictionary_item_id: str  # 字典项唯一编号
    dictionary_code: str  # 字典编码
    dictionary_name: str  # 字典名称
    item_code: str  # 字典项编码
    item_name: str  # 字典项展示名
    parent_item_id: str | None  # 父级字典项编号
    item_level: int  # 字典层级
    sort_order: int  # 排序值
    enabled: int  # 是否启用，1 启用，0 禁用
    description: str | None  # 字典项说明
    metadata_json: str | None  # 扩展配置 JSON
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class ExamSessionEntity(TypedDict):
    """exam_sessions 表实体，记录一次对话式考试会话。"""

    session_id: str  # 考试会话编号
    user_id: str | None  # 用户编号
    title: str  # 考试标题
    collection_name: str  # 题源向量库 collection
    document_id: str | None  # 限定题源文件编号
    filename: str | None  # 限定题源文件名
    section_path: str | None  # 限定一级目录
    round_count: int  # 题目轮数
    question_types_json: str  # 题型列表 JSON
    status: str  # 考试状态，例如 active/completed
    current_round: int  # 当前轮次
    answered_count: int  # 已回答数量
    total_score: float  # 当前总分
    max_score: float  # 满分
    model_mode: str | None  # 模型档位
    metadata_json: str | None  # 扩展元数据 JSON
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    completed_at: str | None  # 完成时间


class ExamQuestionEntity(TypedDict):
    """exam_questions 表实体，记录考试中的单道题和作答结果。"""

    exam_question_id: str  # 考试题目编号
    session_id: str  # 所属考试会话编号
    round_no: int  # 轮次
    source_question_id: str | None  # 来源题目编号
    source_document_id: str | None  # 来源文档编号
    source_filename: str | None  # 来源文件名
    source_page: int | None  # 来源页码
    section_path: str | None  # 来源目录
    question_type: str  # 题型，例如 single_choice/multiple_choice/true_false
    prompt: str  # 题干
    options_json: str | None  # 选项 JSON
    correct_answer_json: str | None  # 标准答案 JSON
    reference_answer: str  # 参考答案
    user_answer: str | None  # 用户答案
    is_correct: int | None  # 是否正确，1 正确，0 错误
    score: float | None  # 得分
    max_score: float  # 本题满分
    analysis_json: str | None  # 评分分析 JSON
    status: str  # 题目状态，例如 pending/answered
    created_at: str  # 创建时间
    answered_at: str | None  # 回答时间


class TrainingKnowledgeBatchEntity(TypedDict):
    """training_knowledge_batches 表实体，记录销售训练资料上传批次。"""

    batch_id: str  # 上传批次编号
    source_type: str  # 资料来源类型
    source_file: str  # 原始文件名
    file_path: str | None  # 本地文件路径
    file_md5: str | None  # 文件 MD5
    version_group_id: str | None  # 版本组编号
    version_no: int  # 版本号
    previous_batch_id: str | None  # 上一个批次编号
    is_current: int  # 是否当前版本
    profile_type: str | None  # 画像类型，兼容旧字段
    task_type: str | None  # 任务类型，兼容旧字段
    industry: str | None  # 行业，兼容旧字段
    difficulty: str | None  # 难度，兼容旧字段
    visibility_default: str | None  # 默认可见性
    status: str  # 批次状态
    chunk_count: int  # 切片数量
    point_count: int  # 向量点数量
    error_message: str | None  # 失败原因
    quality_report_json: str | None  # 质量评估报告 JSON
    created_by: str | None  # 创建人
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class TrainingPlanEntity(TypedDict):
    """training_plans 表实体，记录销售训练方案。"""

    plan_id: str  # 训练方案编号
    plan_name: str  # 训练方案名称
    trainee_id: str  # 学员编号
    trainee_name: str  # 学员名称
    profile_type: str  # 画像类型
    trainee_json: str  # 学员画像 JSON
    selected_fields_json: str  # 画像字段选择 JSON
    scenario_description: str  # 场景描述
    extra_details: str | None  # 补充说明
    model_mode: str | None  # 模型档位
    active_profile_id: str | None  # 当前客户画像编号
    active_setting_id: str | None  # 当前训练目标编号
    role_status: str  # 角色生成状态
    goal_status: str  # 目标生成状态
    score_status: str  # 评分规则状态
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class TrainingRoleProfileEntity(TypedDict):
    """training_role_profiles 表实体，记录 AI 客户画像。"""

    profile_id: str  # 画像编号
    trainee_id: str  # 学员编号
    plan_id: str | None  # 所属训练方案编号
    profile_type: str  # 画像类型
    visible_profile_json: str  # 学员可见画像 JSON
    hidden_profile_json: str  # 隐藏画像 JSON
    role_profile_json: str  # AI 扮演画像 JSON
    role_confirm_card_json: str  # 画像确认卡片 JSON
    selected_fields_json: str | None  # 画像字段选择 JSON
    scenario_description: str | None  # 场景描述
    extra_details: str | None  # 补充说明
    retrieved_evidence_json: str | None  # 生成画像时召回证据 JSON
    status: str  # 画像状态
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class TrainingGoalSettingEntity(TypedDict):
    """training_goal_settings 表实体，记录训练目标和阶段设置。"""

    setting_id: str  # 目标设置编号
    profile_id: str  # 关联画像编号
    plan_id: str | None  # 所属训练方案编号
    trainee_id: str  # 学员编号
    training_mode: str  # 训练模式
    training_purpose: str  # 训练目的
    round_limit: int  # 轮数上限
    stages_json: str  # 阶段设置 JSON
    scoring_rules_json: str | None  # 评分规则 JSON
    status: str  # 设置状态
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


class SalesTrainingSessionEntity(TypedDict):
    """sales_training_sessions 表实体，记录一场销售训练会话。"""

    session_id: str  # 训练会话编号
    profile_id: str  # 客户画像编号
    setting_id: str  # 训练目标编号
    trainee_id: str  # 学员编号
    training_mode: str  # 训练模式
    response_mode: str  # 响应模式
    current_stage_no: int  # 当前阶段编号
    status: str  # 会话状态
    round_limit: int  # 轮数上限
    total_score: int | None  # 总分
    level: str | None  # 评级
    report_json: str | None  # 训练报告 JSON
    started_at: str  # 开始时间
    ended_at: str | None  # 结束时间
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    answered_count: NotRequired[int]  # 列表查询时额外返回的已回答轮数


class SalesTrainingTurnEntity(TypedDict):
    """sales_training_turns 表实体，记录销售训练单轮对话。"""

    turn_id: str  # 对话轮次编号
    session_id: str  # 所属训练会话编号
    role: str  # 角色，例如 customer/trainee/system
    content: str  # 对话内容
    round_no: int  # 轮次
    stage_no: int  # 阶段编号
    response_mode: str | None  # 响应模式
    started_at: str | None  # 开始时间
    submitted_at: str | None  # 提交时间
    response_seconds: float | None  # 响应耗时
    retrieved_chunk_ids_json: str | None  # 召回切片编号 JSON
    retrieved_evidence_json: str | None  # 召回证据 JSON
    stage_decision_json: str | None  # 阶段判断 JSON
    coach_analysis_json: str | None  # 教练分析 JSON
    metadata_json: str | None  # 扩展元数据 JSON
    created_at: str  # 创建时间


class SalesTrainingScoreEntity(TypedDict):
    """sales_training_scores 表实体，记录销售训练评分结果。"""

    score_id: str  # 评分编号
    session_id: str  # 所属训练会话编号
    general_score: int  # 通用能力分
    stage_score: int  # 阶段表现分
    penalty_score: int  # 扣分
    final_score: int  # 最终分
    level: str  # 评级
    is_passed: int  # 是否通过，1 通过，0 未通过
    detail_json: str  # 评分明细 JSON
    review_status: str  # 复核状态
    created_at: str  # 创建时间
    updated_at: str  # 更新时间


KnowledgeStoreRow = (
    DocumentEntity
    | ConversationEntity
    | ConversationMessageEntity
    | DictionaryItemEntity
    | ExamSessionEntity
    | ExamQuestionEntity
)
"""知识库仓储可能返回的表实体联合类型。"""

TrainingRepositoryRow = (
    TrainingKnowledgeBatchEntity
    | TrainingPlanEntity
    | TrainingRoleProfileEntity
    | TrainingGoalSettingEntity
    | SalesTrainingSessionEntity
    | SalesTrainingTurnEntity
    | SalesTrainingScoreEntity
)
"""销售训练仓储可能返回的表实体联合类型。"""

EntityRow = KnowledgeStoreRow | TrainingRepositoryRow
"""项目关系型表实体联合类型。"""
