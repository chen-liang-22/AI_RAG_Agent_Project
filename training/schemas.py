from pydantic import BaseModel, Field

# 这里的类都是 Pydantic 模型，作用类似 Java 里的 DTO / VO：
# 1. FastAPI 会根据这些模型自动校验请求参数；
# 2. FastAPI 会根据这些模型生成 OpenAPI 文档；
# 3. 返回响应时，模型会把 Python 对象序列化成 JSON。
# 注意：字段名是前后端接口协议，不要随意改成中文。


class TrainingKnowledgeUploadResponse(BaseModel):
    """训练知识上传预览响应。"""

    batch_id: str  # 上传批次 ID，用于后续查询本次文件拆出的切片。
    status: str  # 批次状态，例如 pending_review / duplicated。
    chunk_count: int  # 本次上传解析出的切片数量。
    point_count: int  # 写入 Qdrant 的向量点数量；预览阶段通常为 0。
    source_file: str | None = None  # 原始文件名，方便前端展示“上传了哪个资料”。
    duplicate_of: str | None = None  # 如果命中 MD5 去重，这里返回已存在的批次 ID。
    quality_report: dict = Field(default_factory=dict)  # 切片质量报告，供前端确认发布前查看。
    # Python 的 list 是可变对象，不能直接写 failed_chunks: list[str] = []。
    # default_factory=list 类似“每次 new 一个新 ArrayList”，避免多个响应对象共享同一个列表。
    failed_chunks: list[str] = Field(default_factory=list)


class TrainingKnowledgeBatchResponse(BaseModel):
    """训练资料上传批次摘要。"""

    batch_id: str  # 批次 ID，用于点击后查询切片。
    source_type: str  # 来源类型，例如 lms_case。
    source_file: str  # 上传文件名。
    file_path: str | None = None  # 原始文件在服务端的保存路径，用于预览和排查。
    file_md5: str | None = None  # 文件 MD5，用于判断重复上传。
    version_group_id: str | None = None  # 版本组 ID，同一个文件多次发布会归到同一版本组。
    version_no: int = 1  # 版本号，从 1 开始递增。
    previous_batch_id: str | None = None  # 上一个版本的批次 ID。
    is_current: bool = False  # 是否为当前参与训练检索的版本。
    profile_type: str | None = None  # 适用客户画像类型。
    task_type: str | None = None  # 训练任务类型。
    industry: str | None = None  # 行业标签。
    difficulty: str | None = None  # 难度标签。
    visibility_default: str | None = None  # 默认可见范围。
    status: str  # published / parsing_failed 等。
    chunk_count: int  # 切片数量。
    point_count: int  # 向量点数量。
    error_message: str | None = None  # 入库失败原因。
    quality_report: dict = Field(default_factory=dict)  # 最近一次切片质量报告。
    created_by: str | None = None  # 上传人。
    created_at: str  # 创建时间。
    updated_at: str  # 更新时间。


class TrainingKnowledgeBatchListResponse(BaseModel):
    """训练资料批次分页响应。"""

    items: list[TrainingKnowledgeBatchResponse]
    total: int
    page: int
    page_size: int


class TrainingKnowledgePreviewResponse(BaseModel):
    """训练资料原文件预览响应。"""

    batch: TrainingKnowledgeBatchResponse  # 被预览的训练资料批次。
    preview_type: str  # text / document_text，表示当前采用的预览方式。
    content: str  # 从原文件抽取出的预览文本。
    truncated: bool  # 是否因为内容过长被截断。


class TrainingKnowledgeDeleteResponse(BaseModel):
    """训练资料删除响应。"""

    status: str  # 固定返回 deleted。
    batch_id: str  # 被删除的训练资料批次 ID。


class TrainingKnowledgePublishResponse(BaseModel):
    """训练资料确认发布响应。"""

    batch_id: str  # 被发布的批次 ID。
    status: str  # 发布后的状态，通常为 published。
    chunk_count: int  # 发布使用的切片数量。
    point_count: int  # 写入 Qdrant 的向量点数量。
    quality_report: dict = Field(default_factory=dict)  # 发布时参考的质量报告。


class TrainingKnowledgeRollbackResponse(BaseModel):
    """训练资料版本回滚响应。"""

    batch_id: str  # 回滚后恢复为当前版本的批次 ID。
    status: str  # 回滚后的状态，通常为 published。
    version_group_id: str  # 版本组 ID。
    version_no: int  # 回滚到的版本号。
    chunk_count: int  # 重新写入的切片数量。
    point_count: int  # 重新写入 Qdrant 的向量点数量。
    quality_report: dict = Field(default_factory=dict)  # 回滚版本的质量报告。


class TrainingKnowledgeReparseResponse(BaseModel):
    """训练资料重新切分响应。"""

    batch_id: str  # 被重新切分的批次 ID。
    status: str  # 重新切分后的状态，通常为 pending_review。
    chunk_count: int  # 重新生成的切片数量。
    point_count: int  # 预览阶段固定为 0。
    source_file: str | None = None  # 原始文件名。
    quality_report: dict = Field(default_factory=dict)  # 重新切分后的质量报告。


class TrainingKnowledgeVersionListResponse(BaseModel):
    """训练资料版本链响应。"""

    version_group_id: str  # 版本组 ID。
    items: list[TrainingKnowledgeBatchResponse]  # 同版本组内的批次列表，按版本号倒序。


class TrainingKnowledgeChunkResponse(BaseModel):
    """训练知识切片响应。"""

    chunk_id: str  # 切片唯一 ID。
    batch_id: str  # 所属上传批次。
    case_part: str  # 切片业务类型，例如 case_profile / hidden_psychology / scoring_rubric。
    visibility: str  # 可见范围：visible / hidden / scoring_only。
    chunk_text: str  # 切片正文，会写入向量库用于检索。
    metadata: dict = Field(default_factory=dict)  # 额外元数据，例如案例标题、案例序号。


class TrainingKnowledgeChunkListResponse(BaseModel):
    """某次上传批次的切片列表。"""

    batch_id: str
    chunks: list[TrainingKnowledgeChunkResponse]


class TraineeProfileRequest(BaseModel):
    """一期学员画像输入。"""

    # Field(..., min_length=1) 表示这个字段必填，并且字符串长度至少为 1。
    trainee_id: str = Field(..., min_length=1)
    trainee_name: str = "销售学员"  # 学员展示名。
    position_role: str = "overseas_bd"  # 岗位/角色标签。
    experience_level: str = "junior"  # 经验等级，影响 AI 客户追问强度。
    task_goal: str = "goal_junior"  # 任务目标档位：goal_junior / goal_intermediate / goal_senior。
    weakness_tags: list[str] = Field(default_factory=list)  # 学员短板标签，例如价格异议、需求挖掘。
    student_portrait_other: str = ""  # 学员画像补充说明，对应 student_portrait_other 输入型字典项。


class RoleGenerateRequest(BaseModel):
    """生成 AI 陪练角色请求。"""

    plan_id: str | None = None  # 可选：关联训练方案；为空时保持旧的临时生成流程。
    trainee: TraineeProfileRequest  # 嵌套模型，类似 Java DTO 里包含另一个 DTO。
    profile_type: str = "overseas_bd"  # 客户画像类型。
    selected_fields: dict = Field(default_factory=dict)  # 前端选择的客户画像字段，先用 dict 保持扩展性。
    scenario_description: str = Field(..., min_length=1)  # 场景描述必填，是生成角色的重要上下文。
    extra_details: str = ""  # 补充细节，可为空字符串。
    model_mode: str | None = None  # Python 3.10 写法：str | None 等价于 Optional[str]。


class ScenarioPolishRequest(BaseModel):
    """AI 润色训练场景描述请求。"""

    profile_type: str = "overseas_bd"  # 当前选择的客户画像类型。
    selected_fields: dict = Field(default_factory=dict)  # 当前客户画像字段，用来控制润色方向。
    scenario_description: str = Field(..., min_length=1)  # 原始场景描述，按钮点击时从前端文本框传入。
    extra_details: str = ""  # 补充细节，会被模型融合进更自然的场景表达。
    model_mode: str | None = None  # 沿用销售陪练页选择的模型档位。


class ScenarioPolishResponse(BaseModel):
    """AI 润色训练场景描述响应。"""

    polished_scenario: str  # 润色后的场景描述，前端会回填到场景描述输入框。
    original_scenario: str  # 原始场景描述，方便前端或日志排查润色前后差异。


class SupplementQuestionOption(BaseModel):
    """补充问答选项。"""

    option_code: str  # 选项编码，前端展示为 A/B/C/D。
    option_text: str  # 选项文案。


class SupplementQuestion(BaseModel):
    """生成角色前的补充问答题。"""

    question_id: str  # 题目编号，前端回传时用于定位。
    question_no: int  # 展示序号，从 1 开始。
    question: str  # 题干。
    options: list[SupplementQuestionOption]  # 标准选项，通常 4 个。
    allow_other: bool = True  # 是否允许填写“其他”。
    dimension: str = ""  # 题目关注维度，例如价格、痛点、性格、风险。


class SupplementQuestionGenerateResponse(BaseModel):
    """补充问答题生成响应。"""

    questions: list[SupplementQuestion]


class RoleGenerateResponse(BaseModel):
    """生成 AI 陪练角色响应。"""

    profile_id: str  # AI 客户画像 ID。
    visible_profile: dict  # 可展示给管理员/教练看的客户画像摘要。
    hidden_profile: dict = Field(default_factory=dict)  # 教练配置页可查看；训练对话中不会向学员原文暴露。
    role_profile: dict  # AI 客户实际扮演时使用的角色设定。
    role_confirm_card: dict  # 前端确认卡片，适合直接渲染。
    hidden_summary: str  # 隐藏画像摘要提示，不暴露具体隐藏心理。
    retrieved_cases: list[dict] = Field(default_factory=list)  # 生成角色时召回的案例证据。
    knowledge_facts: list[str] = Field(default_factory=list)  # 提炼给前端展示的关键事实。


class GoalSettingGenerateRequest(BaseModel):
    """开放式训练设置生成请求。"""

    plan_id: str | None = None  # 可选：关联训练方案，生成后会更新方案的训练阶段状态。
    trainee_id: str  # 学员 ID。
    training_mode: str = "open"  # 一期只支持 open，流程式二期再扩展。
    model_mode: str | None = None  # 使用哪个模型档位生成目标。


class GoalStage(BaseModel):
    """训练阶段。一期开放式固定只有一个阶段。"""

    stage_no: int = 1  # 阶段序号，一期固定 1。
    stage_name: str  # 阶段名称。
    core_goal: str  # 本次训练核心目标。
    success_conditions: list[str]  # 达成条件。
    failure_conditions: list[str]  # 失败条件。


class GoalSettingResponse(BaseModel):
    """开放式训练设置响应。"""

    setting_id: str  # 训练设置 ID。
    profile_id: str  # 对应的 AI 客户画像 ID。
    training_mode: str  # 训练方式，一期为 open。
    training_purpose: str  # 训练宗旨，限制较短，适合标题展示。
    round_limit: int  # LLM 动态给出的训练轮数，后端会限制在 5-100。
    stages: list[GoalStage]  # 阶段配置，一期只有一个元素。
    scoring_rules: dict = Field(default_factory=dict)  # 评分设置：固定通用 40 分 + LLM 阶段 60 分。
    status: str  # 设置状态，例如 confirmed。


class TrainingPlanCreateRequest(BaseModel):
    """创建训练方案请求。"""

    plan_name: str = Field(..., min_length=1, max_length=80)  # 训练名称，允许同名，使用 plan_id 区分记录。
    trainee: TraineeProfileRequest  # 学员画像快照。
    profile_type: str = "overseas_bd"  # 客户画像类型。
    selected_fields: dict = Field(default_factory=dict)  # 客户画像字段快照。
    scenario_description: str = Field(..., min_length=1)  # 训练场景描述。
    extra_details: str = ""  # 补充细节。
    model_mode: str | None = None  # 默认模型档位。


class TrainingPlanUpdateRequest(BaseModel):
    """修改训练方案请求。"""

    plan_name: str | None = Field(None, min_length=1, max_length=80)  # 训练名称，允许同名。
    trainee: TraineeProfileRequest | None = None  # 修改学员画像会影响角色和后续阶段。
    profile_type: str | None = None  # 修改客户画像类型会影响角色和后续阶段。
    selected_fields: dict | None = None  # 修改客户画像字段会影响角色和后续阶段。
    scenario_description: str | None = None  # 修改场景会影响角色和后续阶段。
    extra_details: str | None = None  # 修改补充细节会影响角色和后续阶段。
    model_mode: str | None = None  # 只修改模型档位不强制重新生成。
    role_confirm_card: dict | None = None  # 只人工微调确认卡，不影响阶段。
    visible_profile: dict | None = None  # 人工微调可见画像，不影响阶段。
    hidden_profile: dict | None = None  # 修改隐藏画像会影响阶段和评分。
    role_profile: dict | None = None  # 修改扮演画像会影响阶段和评分。
    training_purpose: str | None = None  # 人工修改训练宗旨，会影响后续训练会话展示。
    round_limit: int | None = Field(None, ge=1, le=100)  # 人工修改训练轮数，限制在 1-100。
    stages: list[GoalStage] | None = None  # 修改训练阶段会影响评分设置。
    scoring_rules: dict | None = None  # 只修改评分规则不影响前置步骤。


class TrainingPlanSummaryResponse(BaseModel):
    """训练方案列表摘要。"""

    plan_id: str
    plan_name: str
    trainee_id: str
    trainee_name: str
    profile_type: str
    model_mode: str | None = None
    role_status: str
    goal_status: str
    score_status: str
    active_profile_id: str | None = None
    active_setting_id: str | None = None
    created_at: str
    updated_at: str


class TrainingPlanListResponse(BaseModel):
    """训练方案分页响应。"""

    items: list[TrainingPlanSummaryResponse]
    total: int
    page: int
    page_size: int


class TrainingPlanDetailResponse(BaseModel):
    """训练方案详情响应。"""

    plan: TrainingPlanSummaryResponse
    trainee: dict = Field(default_factory=dict)
    selected_fields: dict = Field(default_factory=dict)
    scenario_description: str = ""
    extra_details: str = ""
    visible_profile: dict = Field(default_factory=dict)
    hidden_profile: dict = Field(default_factory=dict)
    role_profile: dict = Field(default_factory=dict)
    role_confirm_card: dict = Field(default_factory=dict)
    retrieved_cases: list[dict] = Field(default_factory=list)
    goal_setting: GoalSettingResponse | None = None


class TrainingSessionStartRequest(BaseModel):
    """开始训练会话请求。"""

    profile_id: str  # 使用哪个 AI 客户画像。
    setting_id: str  # 使用哪个训练目标设置。
    trainee_id: str  # 学员 ID。
    response_mode: str = "stream"  # stream=流式，blocking=一次性。
    model_mode: str | None = None  # 开场白使用的模型档位。


class TrainingSessionResponse(BaseModel):
    """训练会话响应。"""

    session_id: str  # 训练会话 ID，后续对话和评分都依赖它。
    profile_id: str  # AI 客户画像 ID。
    setting_id: str  # 训练设置 ID。
    trainee_id: str  # 学员 ID。
    training_mode: str  # 一期固定 open。
    response_mode: str  # 当前会话默认回复模式。
    current_stage_no: int  # 当前阶段，一期固定 1。
    status: str  # active / scoring / completed。
    round_limit: int  # 本场训练最大轮数。
    opening_message: str | None = None  # AI 客户开场白，创建会话时返回给前端首屏展示。


class TrainingTurnRequest(BaseModel):
    """提交学员回复请求。"""

    message: str = Field(..., min_length=1)  # 学员本轮回复。
    response_mode: str = "stream"  # 本轮使用流式还是一次性。
    model_mode: str | None = None  # 本轮 AI 客户回复使用的模型档位。


class TrainingTurnResponse(BaseModel):
    """一次性训练对话响应。"""

    customer_reply: str  # AI 客户本轮回复。
    current_stage_no: int  # 当前阶段。
    stage_status: str  # 阶段状态，例如 active / round_limit_reached。
    session_status: str  # 会话状态，例如 active / scoring。
    retrieved_chunk_ids: list[str] = Field(default_factory=list)  # 本轮检索命中的训练知识切片 ID。
    coach_analysis: dict = Field(default_factory=dict)  # 给学员看的本轮销售教练分析。
    response_seconds: float | None = None  # 后端生成本轮回复耗时，单位秒。


class TrainingScoreResponse(BaseModel):
    """训练评分报告响应。"""

    score_id: str
    session_id: str
    total_score: int
    level: str
    is_passed: bool
    general_score: int
    stage_score: int
    penalty_score: int
    report: dict


class TrainingTurnRecordResponse(BaseModel):
    """训练会话中的单条对话记录。"""

    turn_id: str
    session_id: str
    role: str
    content: str
    round_no: int
    stage_no: int
    response_mode: str | None = None
    response_seconds: float | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    stage_decision: dict = Field(default_factory=dict)
    coach_analysis: dict = Field(default_factory=dict)  # 历史复盘时回显本轮教练分析。
    created_at: str


class TrainingSessionSummaryResponse(BaseModel):
    """训练会话历史摘要。"""

    session_id: str
    trainee_id: str
    training_mode: str
    response_mode: str
    status: str
    round_limit: int
    answered_count: int
    total_score: int | None = None
    level: str | None = None
    started_at: str
    ended_at: str | None = None
    updated_at: str


class TrainingSessionListResponse(BaseModel):
    """训练会话历史分页响应。"""

    items: list[TrainingSessionSummaryResponse]
    total: int
    page: int
    page_size: int


class TrainingSessionDetailResponse(BaseModel):
    """训练会话复盘详情。"""

    session: TrainingSessionSummaryResponse
    turns: list[TrainingTurnRecordResponse]
    visible_profile: dict = Field(default_factory=dict)
    hidden_profile: dict = Field(default_factory=dict)
    role_profile: dict = Field(default_factory=dict)
    role_confirm_card: dict = Field(default_factory=dict)
    goal_setting: dict = Field(default_factory=dict)
    knowledge_facts: list[str] = Field(default_factory=list)
    score: TrainingScoreResponse | None = None
