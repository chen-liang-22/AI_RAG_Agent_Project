from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """前端聊天请求体。

    一次性接口 `/chat` 和流式接口 `/chat/stream` 共用同一个请求结构：
    - message：用户本次输入的问题。
    - user_id：当前会话的用户 ID，用于工具调用时读取用户画像或外部数据。

    两个接口使用相同请求体，是为了保证前端只需要切换接口地址，
    不需要为两种输出模式维护两套参数结构。
    """

    message: str = Field(..., min_length=1)  # 用户输入的问题；不能为空字符串
    user_id: str | None = None  # 当前会话用户 ID；可为空，工具层会兜底随机用户
    conversation_id: str | None = None  # 当前会话 ID；为空时后端创建新会话
    model_mode: str | None = None  # 回答模型档位，具体可选值和默认值来自 model_mode 字典
    collection_name: str | None = None  # 本次聊天要检索的 Qdrant collection；为空时使用默认 collection


class ChatResponse(BaseModel):
    """一次性接口 `/chat` 的响应体。

    一次性返回时，后端会等待 Agent 完整执行结束，然后只返回最终回答。
    这里不会返回中间 RAG 参考资料、工具调用结果或推理过程。
    """

    answer: str  # 一次性接口返回的完整最终回答
    conversation_id: str  # 当前会话 ID；前端后续请求需要继续携带
    first_token_ms: float | None = None  # 首字/首片耗时；一次性接口没有真实首字，默认等于 total_ms
    total_ms: float | None = None  # 本次请求总耗时


class ConversationSummaryResponse(BaseModel):
    """会话列表中的单个会话摘要。"""

    conversation_id: str  # 会话唯一 ID
    user_id: str | None = None  # 会话所属用户 ID
    title: str | None = None  # 会话标题，通常来自首条用户问题
    status: str  # 会话状态
    message_count: int  # 会话消息数量
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    last_message_at: str | None = None  # 最后一条消息时间


class ConversationListResponse(BaseModel):
    """会话分页列表响应体。"""

    items: list[ConversationSummaryResponse]  # 当前页会话摘要
    total: int  # 总会话数
    page: int  # 当前页码
    page_size: int  # 每页数量


class ConversationMessageResponse(BaseModel):
    """会话详情中的单条消息。"""

    message_id: str  # 消息唯一 ID
    conversation_id: str  # 所属会话 ID
    sequence_no: int  # 会话内顺序号
    role: str  # user/assistant/system
    content: str  # 消息正文
    content_type: str  # 消息类型，默认 text
    model_name: str | None = None  # 助手消息使用的模型名
    token_count: int | None = None  # token 数量，当前可为空
    first_token_ms: float | None = None  # 助手首字/首片耗时；旧数据没有该字段时为空
    total_ms: float | None = None  # 助手完整回答耗时；旧数据没有该字段时为空
    created_at: str  # 创建时间


class ConversationDetailResponse(BaseModel):
    """会话详情响应体。"""

    conversation: ConversationSummaryResponse  # 会话摘要
    messages: list[ConversationMessageResponse]  # 会话全部消息


class ConversationDeleteResponse(BaseModel):
    """删除聊天会话响应体。"""

    status: str  # 固定返回 deleted
    conversation_id: str  # 被删除的会话 ID


class DebugRetrieveRequest(BaseModel):
    """RAG 检索调试请求体。"""

    query: str = Field(..., min_length=1)  # 要调试的用户问题


class HealthResponse(BaseModel):
    """健康检查响应体，用于前端侧边栏展示服务状态。"""

    status: str  # 整体状态；ok 表示全部可用，degraded 表示部分依赖不可用
    qdrant: str  # Qdrant 状态；ok 或 unavailable
    collection_name: str  # 当前项目使用的 Qdrant collection 名称
    collections: list[str] = Field(default_factory=list)  # Qdrant 中已有的 collection 列表


class DictionaryItemResponse(BaseModel):
    """字典项响应体，支持通过 children 表示多层级字典。"""

    dictionary_item_id: str  # 字典项唯一 ID
    dictionary_code: str  # 字典编码，例如 document_structure
    dictionary_name: str  # 字典名称，例如 文档结构类型
    item_code: str  # 字典项编码，例如 qa
    item_name: str  # 字典项名称，例如 问答型
    parent_item_id: str | None = None  # 父级字典项 ID；为空表示一级项
    item_level: int  # 层级，从 1 开始
    sort_order: int  # 同级排序号
    enabled: bool  # 是否启用
    description: str | None = None  # 字典项说明
    metadata: dict = Field(default_factory=dict)  # 前端展示或业务扩展元数据
    children: list["DictionaryItemResponse"] = Field(default_factory=list)  # 子级字典项


class DictionaryGroupResponse(BaseModel):
    """单个字典分组响应体。"""

    dictionary_code: str  # 字典编码
    dictionary_name: str  # 字典名称
    items: list[DictionaryItemResponse] = Field(default_factory=list)  # 当前字典的树形字典项


class KnowledgeFileResponse(BaseModel):
    """知识库文件响应体。

    这个模型对应 SQLite 里的 documents 表。
    前端、Swagger、Postman 看到的文件列表和文件详情都会按这个结构返回。
    """

    document_id: str  # 文件唯一 ID；删除和重建索引都靠它定位文件
    filename: str  # 用户上传时的原始文件名
    file_path: str  # 文件在服务端 uploads/ 目录下的保存路径
    file_type: str  # 文件类型，例如 txt/pdf
    file_md5: str  # 文件内容 MD5；用于判断重复上传
    file_size: int  # 文件大小，单位字节
    status: str  # uploaded/indexing/indexed/failed/deleted
    version: int  # 文件索引版本；每次 reindex 会递增
    chunk_count: int  # 当前版本写入 Qdrant 的知识单元数量
    collection_name: str = "agent"  # 文件写入的 Qdrant collection
    document_type: str = "text"  # 文档结构类型：qa/numbered/text
    split_strategy: str = "recursive"  # 文件入库时使用的切分策略：numbered_qa/outline_qa/numbered_segments/recursive
    created_at: str  # 文件记录创建时间
    updated_at: str  # 文件记录最后更新时间
    error_message: str | None = None  # 入库失败时保存错误原因


class KnowledgeFilePreviewResponse(BaseModel):
    """已入库知识库文件的预览响应体。

    这个模型用于知识库管理页面查看原始文件内容。
    它只读取 documents.file_path 指向的服务端原文件，不参与 Qdrant 检索，也不改变索引数据。
    """

    document: KnowledgeFileResponse  # 被预览的文件元数据
    preview_type: str  # text/pdf_text/unsupported，表示本次预览内容的来源类型
    content: str  # 预览文本内容；大文件会按 max_chars 截断
    truncated: bool  # 是否因为超过 max_chars 被截断
    page_count: int | None = None  # PDF 页数；TXT 文件为空


class KnowledgeUploadResponse(BaseModel):
    """上传知识库文件的响应体。

    status 可能是：
    - indexed：新文件已经成功解析并写入 Qdrant。
    - duplicate：相同 MD5 的文件已经存在，本次没有重复入库。
    """

    status: str  # indexed 或 duplicate
    message: str  # 面向调用方的简短说明
    document: KnowledgeFileResponse  # 成功入库或已存在的文件记录


class KnowledgeUploadPreviewResponse(BaseModel):
    """上传预览响应体。

    预览阶段只保存临时文件并识别文档类型，不写 documents，也不写 Qdrant。
    """

    upload_id: str  # 临时上传 ID，确认入库时使用
    filename: str  # 原始文件名
    file_type: str  # 文件类型
    file_size: int  # 文件大小
    file_md5: str  # 文件 MD5
    duplicate: bool = False  # 是否与已有 active 文件重复
    duplicate_document: KnowledgeFileResponse | None = None  # 重复时对应的已有文件
    detected_type: str  # 系统识别的文档类型
    split_strategy: str  # 系统建议的切分策略
    confidence: float  # 识别置信度
    reasons: list[str] = Field(default_factory=list)  # 识别原因
    llm_used: bool = False  # 是否使用了 LLM 兜底
    sample_text: str = ""  # 抽样文本，给前端预览


class KnowledgeUploadRecommendRequest(BaseModel):
    """上传文件模型推荐请求体。"""

    upload_id: str = Field(..., min_length=1)  # 预览阶段返回的 upload_id


class KnowledgeUploadRecommendResponse(BaseModel):
    """上传文件模型推荐响应体。"""

    document_type: str  # 模型推荐的文档类型
    split_strategy: str  # 模型推荐的切分策略：numbered_qa/outline_qa/numbered_segments/recursive
    confidence: float  # 模型推荐置信度，范围 0 到 1
    reasons: list[str] = Field(default_factory=list)  # 模型推荐原因
    sample_chars: int  # 实际发送给模型的样本文本字符数
    model_name: str  # 本次用于推荐的模型名称


class KnowledgeUploadConfirmRequest(BaseModel):
    """上传确认请求体。"""

    upload_id: str = Field(..., min_length=1)  # 预览阶段返回的 upload_id
    document_type: str = Field(..., min_length=1)  # 用户确认后的文档结构类型：qa/numbered/text
    split_strategy: str = Field(..., min_length=1)  # 用户确认后的切分策略：numbered_qa/outline_qa/numbered_segments/recursive
    collection_name: str | None = None  # 用户选择或输入的 Qdrant collection；为空时使用默认 collection


class KnowledgeDeleteResponse(BaseModel):
    """删除知识库文件的响应体。"""

    status: str  # 固定返回 deleted
    document_id: str  # 被删除的文件 ID


class KnowledgeReindexResult(BaseModel):
    """单个文件重建索引结果。"""

    document_id: str  # 文件 ID
    filename: str  # 文件名
    status: str  # indexed 或 failed
    message: str | None = None  # 失败原因或成功说明


class KnowledgeBulkReindexResponse(BaseModel):
    """全部重建索引响应体。"""

    status: str  # ok 或 partial_failed
    total: int  # 参与重建的文件数
    succeeded: int  # 成功数量
    failed: int  # 失败数量
    results: list[KnowledgeReindexResult]  # 每个文件的结果
