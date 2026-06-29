"""关系型数据库 ORM 实体。

这些类是真正的 SQLAlchemy ORM Model，作用类似 Java MyBatis-Plus 的实体类：
- 类名对应业务实体；
- `__tablename__` 对应数据库表；
- `mapped_column` 对应表字段；
- `to_dict()` 用于把 ORM 对象转换成接口和 service 层仍在使用的字典视图。

注意：这里定义的是数据库实体，不再把普通类型提示当作实体类使用。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseOrmModel(DeclarativeBase):
    """ORM 实体基类。"""


def format_orm_value(value: Any) -> Any:
    """把 ORM 字段值转换成接口层更容易处理的基础类型。"""

    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds", sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


class DictMixin:
    """提供统一的 ORM 对象转 dict 能力。"""

    def _field_names(self) -> list[str]:
        """返回表字段名和仓储层临时挂载的只读展示字段。"""

        column_names = [column.name for column in self.__table__.columns]
        extra_names = [
            key
            for key in vars(self)
            if not key.startswith("_") and key not in column_names
        ]
        return [*column_names, *extra_names]

    def __getitem__(self, key: str) -> Any:
        """兼容旧 service 中的 row["field"] 读取方式。"""

        return format_orm_value(getattr(self, key))

    def get(self, key: str, default: Any = None) -> Any:
        """兼容旧 service 中的 row.get("field") 读取方式。"""

        if not hasattr(self, key):
            return default
        value = getattr(self, key)
        return format_orm_value(value)

    def items(self):
        """兼容少量需要遍历字段的旧代码。"""

        return self.to_dict().items()

    def keys(self):
        """兼容 `{**entity}` 这类 Mapping 解包读取方式。"""

        return self.to_dict().keys()

    def values(self):
        """兼容少量需要按字典值遍历的调用。"""

        return self.to_dict().values()

    def __iter__(self):
        """让实体可以按字典 key 迭代。"""

        return iter(self.keys())

    def to_dict(self) -> dict[str, Any]:
        """按表字段导出字典，保持现有 service 层读取方式稳定。"""

        return {
            field_name: format_orm_value(getattr(self, field_name))
            for field_name in self._field_names()
        }


class DocumentEntity(BaseOrmModel, DictMixin):
    """documents 表实体，记录知识库文件元数据和索引状态。"""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="文件唯一编号")
    filename: Mapped[str] = mapped_column(String(255), nullable=False, comment="原始文件名")
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False, comment="MinIO 存储 URI，格式为 minio://桶名/对象路径")
    storage_type: Mapped[str] = mapped_column(String(32), nullable=False, default="minio", comment="文件存储类型")
    bucket_name: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="MinIO 桶名")
    object_name: Mapped[str | None] = mapped_column(String(1024), nullable=True, comment="MinIO 对象路径")
    public_url: Mapped[str | None] = mapped_column(String(2048), nullable=True, comment="MinIO 公共访问地址")
    file_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="文件扩展类型")
    file_md5: Mapped[str] = mapped_column(String(64), nullable=False, comment="文件内容 MD5")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小，单位字节")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="文件状态")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="文件索引版本号")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已写入向量库的切片数量")
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False, default="agent", comment="Qdrant collection 名称")
    document_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text", comment="文档结构类型")
    split_strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="recursive", comment="切分策略")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败原因")


class ConversationEntity(BaseOrmModel, DictMixin):
    """conversations 表实体，记录一次聊天会话摘要。"""

    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="会话唯一编号")
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="用户编号")
    title: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="会话标题")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="会话状态")
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="消息数量")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment="会话摘要")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展元数据 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")
    last_message_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="最后一条消息时间")


class ConversationMessageEntity(BaseOrmModel, DictMixin):
    """conversation_messages 表实体，记录会话中的单条消息。"""

    __tablename__ = "conversation_messages"

    message_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="消息唯一编号")
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属会话编号")
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="会话内顺序号")
    role: Mapped[str] = mapped_column(String(32), nullable=False, comment="消息角色")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息正文")
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text", comment="内容类型")
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="模型名")
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="token 数量")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展元数据 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")


class DictionaryItemEntity(BaseOrmModel, DictMixin):
    """dictionary_items 表实体，记录系统字典项。"""

    __tablename__ = "dictionary_items"

    dictionary_item_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="字典项唯一编号")
    dictionary_code: Mapped[str] = mapped_column(String(128), nullable=False, comment="字典编码")
    dictionary_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="字典名称")
    item_code: Mapped[str] = mapped_column(String(128), nullable=False, comment="字典项编码")
    item_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="字典项展示名")
    parent_item_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="父级字典项编号")
    item_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="字典层级")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序值")
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="是否启用")
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="字典项说明")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展配置 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class IngestTaskEntity(BaseOrmModel, DictMixin):
    """ingest_tasks 表实体，记录文件异步入库任务。"""

    __tablename__ = "ingest_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="入库任务编号")
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="任务类型")
    business_scene: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="业务场景")
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="关联 documents.document_id")
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="关联训练资料批次编号")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="任务状态")
    current_step: Mapped[str] = mapped_column(String(64), nullable=False, comment="当前处理步骤")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=5, comment="处理进度，0到100")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已尝试次数")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3, comment="最大尝试次数")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败原因")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="任务扩展参数 JSON")
    started_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="开始处理时间")
    finished_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="处理完成时间")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SystemUserEntity(BaseOrmModel, DictMixin):
    """system_users 表实体，记录系统登录用户。"""

    __tablename__ = "system_users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="用户唯一编号")
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, comment="登录账号")
    display_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="展示名称")
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False, comment="密码哈希")
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="admin", comment="用户角色")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="用户状态")
    last_login_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="最后登录时间")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SystemRoleEntity(BaseOrmModel, DictMixin):
    """system_roles 表实体，记录系统角色。"""

    __tablename__ = "system_roles"

    role_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="角色唯一编号，雪花算法生成")
    role_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="角色编码")
    role_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="角色名称")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="角色状态")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序号")
    built_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="是否内置角色")
    description: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="角色说明")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SystemMenuEntity(BaseOrmModel, DictMixin):
    """system_menus 表实体，记录左侧菜单和页面入口。"""

    __tablename__ = "system_menus"

    menu_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="菜单唯一编号，雪花算法生成")
    parent_menu_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="父级菜单编号")
    menu_code: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, comment="菜单编码")
    menu_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="菜单展示名称")
    menu_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="菜单类型")
    page_key: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="前端页面键")
    route_path: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="路由路径")
    component_key: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="组件键")
    icon: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="菜单图标")
    permission_code: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="权限编码")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序号")
    visible: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="是否展示")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="菜单状态")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展配置 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SystemRoleMenuEntity(BaseOrmModel, DictMixin):
    """system_role_menus 表实体，记录角色可见菜单。"""

    __tablename__ = "system_role_menus"

    role_menu_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="角色菜单关系编号，雪花算法生成")
    role_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="角色编号")
    menu_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="菜单编号")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")


class ExamSessionEntity(BaseOrmModel, DictMixin):
    """exam_sessions 表实体，记录一次对话式考试会话。"""

    __tablename__ = "exam_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="考试会话编号")
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="用户编号")
    title: Mapped[str] = mapped_column(String(255), nullable=False, comment="考试标题")
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="题源向量库 collection")
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="限定题源文件编号")
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="限定题源文件名")
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="限定一级目录")
    round_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="题目轮数")
    question_types_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="题型列表 JSON")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="考试状态")
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="当前轮次")
    answered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="已回答数量")
    total_score: Mapped[float] = mapped_column(Float, nullable=False, default=0, comment="当前总分")
    max_score: Mapped[float] = mapped_column(Float, nullable=False, default=100, comment="满分")
    model_mode: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="模型档位")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展元数据 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")
    completed_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="完成时间")


class ExamQuestionEntity(BaseOrmModel, DictMixin):
    """exam_questions 表实体，记录考试中的单道题和作答结果。"""

    __tablename__ = "exam_questions"

    exam_question_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="考试题目编号")
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属考试会话编号")
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮次")
    source_question_id: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="来源题目编号")
    source_document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="来源文档编号")
    source_filename: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="来源文件名")
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="来源页码")
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="来源目录")
    question_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="题型")
    prompt: Mapped[str] = mapped_column(Text, nullable=False, comment="题干")
    options_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="选项 JSON")
    correct_answer_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="标准答案 JSON")
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="参考答案")
    user_answer: Mapped[str | None] = mapped_column(Text, nullable=True, comment="用户答案")
    is_correct: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="是否正确")
    score: Mapped[float | None] = mapped_column(Float, nullable=True, comment="得分")
    max_score: Mapped[float] = mapped_column(Float, nullable=False, comment="本题满分")
    analysis_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="评分分析 JSON")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", comment="题目状态")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    answered_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="回答时间")


class TrainingKnowledgeBatchEntity(BaseOrmModel, DictMixin):
    """training_knowledge_batches 表实体，记录销售训练资料上传批次。"""

    __tablename__ = "training_knowledge_batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="上传批次编号")
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="关联 documents.document_id")
    source_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="资料来源类型")
    source_file: Mapped[str] = mapped_column(String(255), nullable=False, comment="原始文件名")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True, comment="历史兼容文件路径")
    file_md5: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="历史兼容文件 MD5")
    version_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="版本组编号")
    version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="版本号")
    previous_batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="上一个批次编号")
    is_current: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="是否当前版本")
    profile_type: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="画像类型")
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="任务类型")
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="行业")
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="难度")
    visibility_default: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="默认可见性")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="批次状态")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="切片数量")
    point_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="向量点数量")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="失败原因")
    quality_report_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="质量评估报告 JSON")
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="创建人")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class TrainingPlanEntity(BaseOrmModel, DictMixin):
    """training_plans 表实体，记录销售训练方案。"""

    __tablename__ = "training_plans"

    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="训练方案编号")
    plan_name: Mapped[str] = mapped_column(String(255), nullable=False, comment="训练方案名称")
    trainee_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="学员编号")
    trainee_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="学员名称")
    profile_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="画像类型")
    trainee_json: Mapped[str] = mapped_column(Text, nullable=False, comment="学员画像 JSON")
    selected_fields_json: Mapped[str] = mapped_column(Text, nullable=False, comment="画像字段选择 JSON")
    scenario_description: Mapped[str] = mapped_column(Text, nullable=False, comment="场景描述")
    extra_details: Mapped[str | None] = mapped_column(Text, nullable=True, comment="补充说明")
    model_mode: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="模型档位")
    active_profile_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="当前客户画像编号")
    active_setting_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="当前训练目标编号")
    role_status: Mapped[str] = mapped_column(String(32), nullable=False, comment="角色生成状态")
    goal_status: Mapped[str] = mapped_column(String(32), nullable=False, comment="目标生成状态")
    score_status: Mapped[str] = mapped_column(String(32), nullable=False, comment="评分规则状态")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class TrainingRoleProfileEntity(BaseOrmModel, DictMixin):
    """training_role_profiles 表实体，记录 AI 客户画像。"""

    __tablename__ = "training_role_profiles"

    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="画像编号")
    trainee_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="学员编号")
    plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="所属训练方案编号")
    profile_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="画像类型")
    visible_profile_json: Mapped[str] = mapped_column(Text, nullable=False, comment="学员可见画像 JSON")
    hidden_profile_json: Mapped[str] = mapped_column(Text, nullable=False, comment="隐藏画像 JSON")
    role_profile_json: Mapped[str] = mapped_column(Text, nullable=False, comment="AI 扮演画像 JSON")
    role_confirm_card_json: Mapped[str] = mapped_column(Text, nullable=False, comment="画像确认卡片 JSON")
    selected_fields_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="画像字段选择 JSON")
    scenario_description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="场景描述")
    extra_details: Mapped[str | None] = mapped_column(Text, nullable=True, comment="补充说明")
    retrieved_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="生成画像时召回证据 JSON")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="画像状态")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class TrainingGoalSettingEntity(BaseOrmModel, DictMixin):
    """training_goal_settings 表实体，记录训练目标和阶段设置。"""

    __tablename__ = "training_goal_settings"

    setting_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="目标设置编号")
    profile_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="关联画像编号")
    plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="所属训练方案编号")
    trainee_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="学员编号")
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False, comment="训练模式")
    training_purpose: Mapped[str] = mapped_column(Text, nullable=False, comment="训练目的")
    round_limit: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮数上限")
    stages_json: Mapped[str] = mapped_column(Text, nullable=False, comment="阶段设置 JSON")
    scoring_rules_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="评分规则 JSON")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="设置状态")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SalesTrainingSessionEntity(BaseOrmModel, DictMixin):
    """sales_training_sessions 表实体，记录一场销售训练会话。"""

    __tablename__ = "sales_training_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="训练会话编号")
    profile_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="客户画像编号")
    setting_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="训练目标编号")
    trainee_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="学员编号")
    training_mode: Mapped[str] = mapped_column(String(64), nullable=False, comment="训练模式")
    response_mode: Mapped[str] = mapped_column(String(32), nullable=False, comment="响应模式")
    current_stage_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="当前阶段编号")
    status: Mapped[str] = mapped_column(String(32), nullable=False, comment="会话状态")
    round_limit: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮数上限")
    total_score: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="总分")
    level: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="评级")
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="训练报告 JSON")
    started_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="开始时间")
    ended_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="结束时间")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


class SalesTrainingTurnEntity(BaseOrmModel, DictMixin):
    """sales_training_turns 表实体，记录销售训练单轮对话。"""

    __tablename__ = "sales_training_turns"

    turn_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="对话轮次编号")
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属训练会话编号")
    role: Mapped[str] = mapped_column(String(32), nullable=False, comment="角色")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="对话内容")
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="轮次")
    stage_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="阶段编号")
    response_mode: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="响应模式")
    started_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="开始时间")
    submitted_at: Mapped[datetime | str | None] = mapped_column(DateTime, nullable=True, comment="提交时间")
    response_seconds: Mapped[float | None] = mapped_column(Float, nullable=True, comment="响应耗时")
    retrieved_chunk_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="召回切片编号 JSON")
    retrieved_evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="召回证据 JSON")
    stage_decision_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="阶段判断 JSON")
    coach_analysis_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="教练分析 JSON")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展元数据 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")


class SalesTrainingScoreEntity(BaseOrmModel, DictMixin):
    """sales_training_scores 表实体，记录销售训练评分结果。"""

    __tablename__ = "sales_training_scores"

    score_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="评分编号")
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="所属训练会话编号")
    general_score: Mapped[int] = mapped_column(Integer, nullable=False, comment="通用能力分")
    stage_score: Mapped[int] = mapped_column(Integer, nullable=False, comment="阶段表现分")
    penalty_score: Mapped[int] = mapped_column(Integer, nullable=False, comment="扣分")
    final_score: Mapped[int] = mapped_column(Integer, nullable=False, comment="最终分")
    level: Mapped[str] = mapped_column(String(64), nullable=False, comment="评级")
    is_passed: Mapped[int] = mapped_column(Integer, nullable=False, comment="是否通过")
    detail_json: Mapped[str] = mapped_column(Text, nullable=False, comment="评分明细 JSON")
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, comment="复核状态")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")


KnowledgeStoreRow = (
    DocumentEntity
    | ConversationEntity
    | ConversationMessageEntity
    | DictionaryItemEntity
    | SystemUserEntity
    | ExamSessionEntity
    | ExamQuestionEntity
)
"""知识库仓储可能返回的 ORM 实体联合类型。"""

TrainingRepositoryRow = (
    TrainingKnowledgeBatchEntity
    | TrainingPlanEntity
    | TrainingRoleProfileEntity
    | TrainingGoalSettingEntity
    | SalesTrainingSessionEntity
    | SalesTrainingTurnEntity
    | SalesTrainingScoreEntity
)
"""销售训练仓储可能返回的 ORM 实体联合类型。"""

EntityRow = KnowledgeStoreRow | TrainingRepositoryRow | SystemUserEntity | SystemRoleEntity | SystemMenuEntity | SystemRoleMenuEntity
"""项目关系型 ORM 实体联合类型。"""
