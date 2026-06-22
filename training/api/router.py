from functools import lru_cache

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse

from api.routers.dictionaries import build_dictionary_groups
from api.schemas import DictionaryGroupResponse
from rag.knowledge_store import KnowledgeStore
from training.schemas import (
    GoalSettingGenerateRequest,
    GoalSettingResponse,
    RoleGenerateRequest,
    RoleGenerateResponse,
    ScenarioPolishRequest,
    ScenarioPolishResponse,
    SupplementQuestionGenerateResponse,
    TrainingPlanCreateRequest,
    TrainingPlanDetailResponse,
    TrainingPlanListResponse,
    TrainingPlanUpdateRequest,
    TrainingSessionDetailResponse,
    TrainingSessionListResponse,
    TrainingKnowledgeBatchListResponse,
    TrainingKnowledgeChunkListResponse,
    TrainingKnowledgeDeleteResponse,
    TrainingKnowledgePublishResponse,
    TrainingKnowledgePreviewResponse,
    TrainingKnowledgeReparseResponse,
    TrainingKnowledgeRollbackResponse,
    TrainingKnowledgeVersionListResponse,
    TrainingKnowledgeUploadResponse,
    TrainingScoreResponse,
    TrainingSessionResponse,
    TrainingSessionStartRequest,
    TrainingTurnRequest,
    TrainingTurnResponse,
)
from training.services.sales_training_service import SalesTrainingService

router = APIRouter(prefix="/training", tags=["sales-training"])

PROFILE_DICTIONARY_CODES = (
    "student_portrait",
    "wzf_customer_manager",
    "wm_ai_service",
    "overseas_bd",
    "training_source_type",
    "training_case_part",
    "training_chunk_usage",
    "training_batch_status",
)


@lru_cache(maxsize=1)
def _service() -> SalesTrainingService:
    """训练服务单例，复用 Qdrant 和 SQLite 连接配置。

    lru_cache(maxsize=1) 会把第一次创建的 SalesTrainingService 缓存起来，
    后续请求复用同一个对象。它在这里的作用类似一个轻量单例。
    """

    return SalesTrainingService()


def _knowledge_store() -> KnowledgeStore:
    """创建知识库元数据存储实例，用于读取系统字典。"""

    return KnowledgeStore()


@router.get("/profile-dictionaries", response_model=list[DictionaryGroupResponse])
def list_profile_dictionaries() -> list[DictionaryGroupResponse]:
    """查询销售训练画像字典。

    返回内容包括：
    - student_portrait：学员画像字段；
    - wzf_customer_manager：外综服客户经理画像字段；
    - wm_ai_service：超级客服画像字段；
    - overseas_bd：海外BD画像字段。
    - training_source_type：训练资料来源类型。
    - training_case_part：训练资料切片类型。
    - training_chunk_usage：训练切片模型用途。
    - training_batch_status：训练资料上传批次状态。

    该接口是训练模块的专用门面，底层仍复用通用 dictionary_items 表。
    """

    store = _knowledge_store()
    rows = []
    for dictionary_code in PROFILE_DICTIONARY_CODES:
        rows.extend(store.list_dictionary_items(dictionary_code=dictionary_code))
    return build_dictionary_groups(rows)


@router.post("/knowledge/upload", response_model=TrainingKnowledgeUploadResponse)
def upload_training_knowledge(
        # 上传的原始训练资料文件。后端会先保存原文件，再解析内容并生成待发布切片预览。
        file: UploadFile = File(...),
        # 资料来源类型。默认 lms_case，决定使用哪一种入库切片策略。
        source_type: str = Form("lms_case"),
        # 资料切分阶段 LLM 兜底使用的模型档位。为空时使用系统默认档位。
        model_mode: str | None = Form(None),
        # 上传人标识。作为审计字段入库，方便追踪这批训练资料由谁上传或维护。
        created_by: str | None = Form(None),
) -> TrainingKnowledgeUploadResponse:
    """上传销售训练知识，并生成待确认发布的预览切片。

    FastAPI 参数来源说明：
    - File(...)：从 multipart/form-data 的文件字段读取上传文件；
    - Form(...)：只接收当前真正参与上传入库的普通表单字段；
    - response_model：把服务层返回值序列化为前端需要的 JSON。

    注意：profile_type、task_type、industry、difficulty 当前不参与训练检索过滤，
    上传时不再接收这些弱标签，避免向量库 payload 中出现没有业务价值的空字段。
    """

    return _service().upload_knowledge(
        file=file,
        source_type=source_type,
        created_by=created_by,
        model_mode=model_mode,
    )


@router.get("/knowledge/batches", response_model=TrainingKnowledgeBatchListResponse)
def list_training_batches(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=50),
) -> TrainingKnowledgeBatchListResponse:
    """分页查询已经上传过的训练资料。"""

    return _service().list_batches(page=page, page_size=page_size)


@router.get("/knowledge/batches/{batch_id}/preview", response_model=TrainingKnowledgePreviewResponse)
def preview_training_batch(
        batch_id: str,
        max_chars: int = Query(30000, ge=500, le=100000),
) -> TrainingKnowledgePreviewResponse:
    """预览训练资料原文件内容。

    这个接口读取的是上传时保存下来的原文件，再解析成可读文本。
    它不会重新写入向量库，只用于前端展示和排查资料是否传对。
    """

    return _service().preview_batch(batch_id, max_chars=max_chars)


@router.delete("/knowledge/batches/{batch_id}", response_model=TrainingKnowledgeDeleteResponse)
def delete_training_batch(batch_id: str) -> TrainingKnowledgeDeleteResponse:
    """删除训练资料批次。

    删除采用软删除：SQLite 批次标记为 deleted，同时删除 Qdrant 中该 batch_id 的向量点。
    原始上传文件暂时保留，方便以后做审计或恢复。
    """

    return _service().delete_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/publish", response_model=TrainingKnowledgePublishResponse)
def publish_training_batch(batch_id: str) -> TrainingKnowledgePublishResponse:
    """人工确认发布训练资料。

    上传接口只负责解析、切片和质量评估；确认发布时才写入 Qdrant。
    """

    return _service().publish_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/rollback", response_model=TrainingKnowledgeRollbackResponse)
def rollback_training_batch(batch_id: str) -> TrainingKnowledgeRollbackResponse:
    """回滚训练资料到指定历史版本。

    只允许对 published / archived 版本执行。
    回滚后，该版本会重新成为当前参与训练检索的版本。
    """

    return _service().rollback_batch(batch_id)


@router.post("/knowledge/batches/{batch_id}/reparse", response_model=TrainingKnowledgeReparseResponse)
def reparse_training_batch(
        batch_id: str,
        use_llm_fallback: bool = Query(True),
        model_mode: str | None = Query(None),
) -> TrainingKnowledgeReparseResponse:
    """重新切分未发布训练资料。

    用于人工预览发现切片质量不好时，主动触发 LLM 兜底切分。
    """

    return _service().reparse_batch(batch_id, use_llm_fallback=use_llm_fallback, model_mode=model_mode)


@router.get("/knowledge/batches/{batch_id}/versions", response_model=TrainingKnowledgeVersionListResponse)
def list_training_batch_versions(batch_id: str) -> TrainingKnowledgeVersionListResponse:
    """查询训练资料版本链。"""

    return _service().list_batch_versions(batch_id)


@router.get("/knowledge/batches/{batch_id}/chunks", response_model=TrainingKnowledgeChunkListResponse)
def list_training_chunks(batch_id: str) -> TrainingKnowledgeChunkListResponse:
    """查询训练知识上传批次的切片。

    batch_id 来自路径参数：/training/knowledge/batches/{batch_id}/chunks。
    """

    return _service().list_chunks(batch_id)


@router.post("/plans", response_model=TrainingPlanDetailResponse)
def create_training_plan(request: TrainingPlanCreateRequest) -> TrainingPlanDetailResponse:
    """创建训练方案，训练名称允许重复，每条记录用 plan_id 区分。"""

    return _service().create_plan(request)


@router.get("/plans", response_model=TrainingPlanListResponse)
def list_training_plans(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=50),
        keyword: str | None = None,
) -> TrainingPlanListResponse:
    """分页查询训练方案列表。"""

    return _service().list_plans(page=page, page_size=page_size, keyword=keyword)


@router.get("/plans/{plan_id}", response_model=TrainingPlanDetailResponse)
def get_training_plan(plan_id: str) -> TrainingPlanDetailResponse:
    """查看训练方案每一步详情。"""

    return _service().get_plan_detail(plan_id)


@router.put("/plans/{plan_id}", response_model=TrainingPlanDetailResponse)
def update_training_plan(plan_id: str, request: TrainingPlanUpdateRequest) -> TrainingPlanDetailResponse:
    """修改训练方案某一步，并按依赖关系标记后续步骤需要重新生成。"""

    return _service().update_plan(plan_id, request)


@router.post("/profiles/generate", response_model=RoleGenerateResponse)
def generate_role_profile(request: RoleGenerateRequest) -> RoleGenerateResponse:
    """生成 AI 陪练角色。"""

    return _service().generate_role(request)


@router.post("/profiles/scenario/polish", response_model=ScenarioPolishResponse)
def polish_training_scenario(request: ScenarioPolishRequest) -> ScenarioPolishResponse:
    """根据客户画像字段润色训练场景描述。"""

    return _service().polish_scenario(request)


@router.post("/profiles/supplement-questions/generate", response_model=SupplementQuestionGenerateResponse)
def generate_role_supplement_questions(request: RoleGenerateRequest) -> SupplementQuestionGenerateResponse:
    """生成 AI 陪练角色前的补充问答题。"""

    return _service().generate_supplement_questions(request)


@router.post("/profiles/{profile_id}/goal-settings/generate", response_model=GoalSettingResponse)
def generate_goal_setting(profile_id: str, request: GoalSettingGenerateRequest) -> GoalSettingResponse:
    """生成一期开放式训练设置。

    profile_id 来自路径参数；request 来自 JSON 请求体。
    FastAPI 会自动把 JSON 反序列化成 GoalSettingGenerateRequest。
    """

    return _service().generate_goal_setting(
        profile_id=profile_id,
        trainee_id=request.trainee_id,
        training_mode=request.training_mode,
        plan_id=request.plan_id,
        model_mode=request.model_mode,
    )


@router.post("/sessions", response_model=TrainingSessionResponse)
def start_training_session(request: TrainingSessionStartRequest) -> TrainingSessionResponse:
    """开始训练会话。"""

    return _service().start_session(request)


@router.get("/sessions", response_model=TrainingSessionListResponse)
def list_training_sessions(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=50),
        trainee_id: str | None = None,
) -> TrainingSessionListResponse:
    """分页查询训练会话历史。

    Query(1, ge=1) 表示 query 参数默认值是 1，并且必须 >= 1。
    page_size 限制最大 50，避免一次请求拉太多历史记录。
    """

    return _service().list_sessions(page=page, page_size=page_size, trainee_id=trainee_id)


@router.get("/sessions/{session_id}", response_model=TrainingSessionDetailResponse)
def get_training_session_detail(session_id: str) -> TrainingSessionDetailResponse:
    """查询训练会话复盘详情。"""

    return _service().get_session_detail(session_id)


@router.post("/sessions/{session_id}/turns", response_model=TrainingTurnResponse)
def submit_training_turn(
        session_id: str,
        request: TrainingTurnRequest,
        stream: bool = Query(False),
):
    """提交学员回复，支持一次性或流式返回。

    同一个接口根据 stream query 参数或 request.response_mode 分支：
    - stream=true：返回 StreamingResponse，浏览器按 SSE 接收；
    - 否则：返回普通 JSON。
    """

    if stream or request.response_mode == "stream":
        # StreamingResponse 接收一个可迭代对象；service.stream_turn 会不断 yield SSE 字符串。
        return StreamingResponse(
            _service().stream_turn(session_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return _service().submit_turn(session_id, request)


@router.post("/sessions/{session_id}/final-score", response_model=TrainingScoreResponse)
def final_score(session_id: str, model_mode: str | None = None) -> TrainingScoreResponse:
    """结束训练并生成评分报告。"""

    return _service().final_score(session_id, model_mode=model_mode)
